//! A node's fill and border form one surface. Paint their layers before applying
//! coverage, so adjacent antialiased regions never leak the backdrop or double opacity.

use crate::{
    clipping::{clip_polygon, rect_is_clipped, rect_without_hole, rounded_inner_rect},
    pipeline::{ui_color_target_state, ui_render_target_format, UiPipelineKey},
    render_pass::TransparentUi,
    stack_z_offsets,
    ui_mesh::{DrawUiMesh, UiMesh},
    RenderUiSystems, UiAntiAlias, UiCameraMap, UiCameraView, QUAD_VERTEX_POSITIONS,
};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::visibility::InheritedVisibility;
use bevy_color::LinearRgba;
use bevy_ecs::prelude::*;
use bevy_math::{ops, Affine2, FloatOrd, Rect, Vec2};
use bevy_render::{
    render_phase::{
        AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, SetItemPipeline,
        ViewSortedRenderPhases,
    },
    render_resource::{
        binding_types::{texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, TemporaryRenderEntity},
    view::{ExtractedView, ViewUniform, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader};
use bevy_sprite::BorderRect;
use bevy_ui::{
    BackgroundColor, BackgroundGradient, BorderColor, BorderGradient, CalculatedClip, ComputedNode,
    ComputedStackIndex, ComputedUiRenderTargetInfo, ComputedUiTargetCamera, UiGlobalTransform,
};
use bytemuck::{Pod, Zeroable};

mod gradient;
use gradient::GradientExtractor;

#[cfg(test)]
mod tests;

pub(crate) struct SurfacePlugin;

impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "paint.wgsl");
        load_shader_library!(app, "gradient.wgsl");
        embedded_asset!(app, "surface.wgsl");
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedSurfaces>()
                .init_gpu_resource::<SurfacePaint>()
                .init_gpu_resource::<UiMesh<SurfaceVertex>>()
                .init_gpu_resource::<SpecializedRenderPipelines<SurfacePipeline>>()
                .add_render_command::<TransparentUi, (SetItemPipeline, DrawUiMesh<SurfaceVertex>)>()
                .add_systems(RenderStartup, init_surface_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_surfaces.in_set(RenderUiSystems::ExtractBackgrounds),
                )
                .add_systems(
                    Render,
                    (
                        queue_surfaces.in_set(RenderSystems::Queue),
                        prepare_surfaces.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

// Records occupy whole RGBA32Uint texels. Integer fetches preserve float bits and
// work on WebGL2 as well as native/WebGPU, without storage buffers or stop limits.
// Geometry only carries a texel address; paint is uploaded once per node/layer.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuSurface {
    size: [f32; 2],
    background_gradients: [u32; 2],
    radius: [f32; 4],
    border: [f32; 4],
    background: [f32; 4],
    border_colors: [[f32; 4]; 4],
    border_gradients: [u32; 2],
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceVertex {
    position: [f32; 3],
    point: [f32; 2],
    surface: u32,
}

struct ExtractedSurface {
    size: Vec2,
    paint: u32,
    transform: Affine2,
    clip: Option<CalculatedClip>,
    camera: Entity,
    entity: MainEntity,
    render_entity: Entity,
    z_order: f32,
    hole: Option<Rect>,
}

#[derive(Resource, Default)]
struct ExtractedSurfaces {
    nodes: Vec<ExtractedSurface>,
    texels: Vec<[u32; 4]>,
}

struct PaintTexture {
    texture: Texture,
    view: TextureView,
}

#[derive(Resource, Default)]
struct SurfacePaint {
    texture: Option<PaintTexture>,
}

fn append_paint<T: Pod>(texels: &mut Vec<[u32; 4]>, records: &[T]) -> u32 {
    let address = texels.len() as u32;
    texels.extend_from_slice(bytemuck::cast_slice(records));
    address
}

impl SurfacePaint {
    fn upload(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        texels: &mut Vec<[u32; 4]>,
    ) -> &TextureView {
        let limit = device.limits().max_texture_dimension_2d;
        let width = 1024.min(limit);
        let rows = texels.len().div_ceil(width as usize) as u32;
        assert!(
            rows <= limit,
            "UI paint data exceeds the device's texture capacity"
        );
        if self
            .texture
            .as_ref()
            .is_none_or(|paint| paint.texture.height() < rows)
        {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("ui_surface_paint"),
                size: Extent3d {
                    width,
                    height: rows.next_power_of_two().min(limit),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            self.texture = Some(PaintTexture { texture, view });
        }
        let paint = self.texture.as_ref().unwrap();
        // Only upload used rows; a one-row table need not upload the unused columns.
        let upload_width = width.min(texels.len() as u32);
        texels.resize((upload_width * rows) as usize, [0; 4]);
        queue.write_texture(
            paint.texture.as_image_copy(),
            bytemuck::cast_slice(texels),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload_width * 16),
                rows_per_image: None,
            },
            Extent3d {
                width: upload_width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        &paint.view
    }
}

// Match the UI's gamma-2.2 working space, once per constant color rather than per pixel.
fn gamma_color(color: LinearRgba) -> [f32; 4] {
    [
        ops::powf(color.red.max(0.), 1. / 2.2),
        ops::powf(color.green.max(0.), 1. / 2.2),
        ops::powf(color.blue.max(0.), 1. / 2.2),
        color.alpha,
    ]
}

fn extract_surfaces(
    mut commands: Commands,
    mut extracted: ResMut<ExtractedSurfaces>,
    nodes: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &ComputedUiTargetCamera,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            AnyOf<(
                &BackgroundColor,
                &BorderColor,
                &BackgroundGradient,
                &BorderGradient,
            )>,
        )>,
    >,
    cameras: Extract<UiCameraMap>,
    mut gradients: Local<GradientExtractor>,
) {
    let mut cameras = cameras.get_mapper();
    let ExtractedSurfaces {
        nodes: surfaces,
        texels,
    } = &mut *extracted;
    for (
        entity,
        node,
        stack,
        camera,
        target,
        transform,
        visibility,
        clip,
        (background, border, background_gradient, border_gradient),
    ) in &nodes
    {
        if !visibility.get()
            || node.is_empty()
            || rect_is_clipped(clip, transform.into(), node.size())
        {
            continue;
        }
        let Some(camera) = cameras.map(camera) else {
            continue;
        };
        let background = background.map_or([0.; 4], |c| gamma_color(c.0.to_linear()));
        let background_gradients = gradients.extract(
            background_gradient.map_or(&[], |g| g.0.as_slice()),
            node.size(),
            target,
            texels,
        );
        let (border_colors, border_gradients) = if node.border() == BorderRect::ZERO {
            ([[0.; 4]; 4], [0; 2])
        } else {
            (
                border.map_or([[0.; 4]; 4], |c| {
                    [c.left, c.top, c.right, c.bottom].map(|color| gamma_color(color.to_linear()))
                }),
                gradients.extract(
                    border_gradient.map_or(&[], |g| g.0.as_slice()),
                    node.size(),
                    target,
                    texels,
                ),
            )
        };
        let has_fill = background[3] != 0. || background_gradients[1] != 0;
        let has_border = border_colors.iter().any(|c| c[3] != 0.) || border_gradients[1] != 0;
        if !has_fill && !has_border {
            continue;
        }
        let border = node.border();
        let data = GpuSurface {
            size: node.size().into(),
            radius: node.border_radius().into(),
            border: [
                border.min_inset.x,
                border.min_inset.y,
                border.max_inset.x,
                border.max_inset.y,
            ],
            background,
            border_colors,
            background_gradients,
            border_gradients,
            padding: [0; 2],
        };
        surfaces.push(ExtractedSurface {
            size: node.size(),
            paint: append_paint(texels, core::slice::from_ref(&data)),
            transform: transform.into(),
            clip: clip.cloned(),
            camera,
            entity: entity.into(),
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            z_order: stack.0 as f32 + stack_z_offsets::BACKGROUND_COLOR,
            hole: if has_fill {
                None
            } else {
                rounded_inner_rect(node.size(), border, node.border_radius())
            },
        });
    }
}

#[derive(Resource)]
struct SurfacePipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

fn init_surface_pipeline(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(SurfacePipeline {
        layout: BindGroupLayoutDescriptor::new(
            "ui_surface_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    (
                        0,
                        uniform_buffer::<ViewUniform>(true).visibility(ShaderStages::VERTEX),
                    ),
                    (1, texture_2d(TextureSampleType::Uint)),
                ),
            ),
        ),
        shader: load_embedded_asset!(assets.as_ref(), "surface.wgsl"),
    });
}

impl SpecializedRenderPipeline for SurfacePipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            vec![]
        };
        let mut target = ui_color_target_state(key.target_format);
        target.blend = Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        RenderPipelineDescriptor {
            label: Some("ui_surface_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![bevy_mesh::VertexBufferLayout::from_vertex_formats(
                    VertexStepMode::Vertex,
                    vec![
                        VertexFormat::Float32x3,
                        VertexFormat::Float32x2,
                        VertexFormat::Uint32,
                    ],
                )],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(target)],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

fn queue_surfaces(
    extracted: Res<ExtractedSurfaces>,
    pipeline: Res<SurfacePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SurfacePipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    ui_views: Query<(&UiCameraView, Option<&UiAntiAlias>), With<ExtractedView>>,
    views: Query<&ExtractedView>,
    cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions
        .read()
        .id::<(SetItemPipeline, DrawUiMesh<SurfaceVertex>)>();
    for (index, surface) in extracted.nodes.iter().enumerate() {
        let Ok((camera, antialias)) = ui_views.get(surface.camera) else {
            continue;
        };
        let Ok(view) = views.get(camera.0) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let pipeline = pipelines.specialize(
            &cache,
            &pipeline,
            UiPipelineKey {
                target_format: ui_render_target_format(view.target_format),
                anti_alias: matches!(antialias, None | Some(UiAntiAlias::On)),
            },
        );
        phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (surface.render_entity, surface.entity),
            sort_key: FloatOrd(surface.z_order),
            index,
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: true,
        });
    }
}

fn prepare_surfaces(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    mut mesh: ResMut<UiMesh<SurfaceVertex>>,
    mut paint: ResMut<SurfacePaint>,
    mut extracted: ResMut<ExtractedSurfaces>,
    view_uniforms: Res<ViewUniforms>,
    pipeline: Res<SurfacePipeline>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
) {
    let ExtractedSurfaces { nodes, texels } = &mut *extracted;
    if let Some(view_binding) = view_uniforms.uniforms.binding()
        && !nodes.is_empty()
    {
        mesh.vertices.clear();
        mesh.indices.clear();
        let paint_view = paint.upload(&device, &queue, texels);
        mesh.view_bind_group = Some(device.create_bind_group(
            "ui_surface_bind_group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((view_binding, paint_view)),
        ));
        let mut batches = Vec::new();
        for phase in phases.values_mut() {
            let mut first = None;
            for item_index in 0..phase.items.len() {
                let item = &phase.items[item_index];
                let Some(surface) = nodes
                    .get(item.index)
                    .filter(|n| n.render_entity == item.entity())
                else {
                    continue;
                };
                let start = mesh.indices.len() as u32;
                for part in rect_without_hole(surface.size, surface.hole) {
                    let corners = QUAD_VERTEX_POSITIONS.map(|p| {
                        let point = part.center() + p * part.size();
                        (surface.transform.transform_point2(point), point)
                    });
                    let vertices = clip_polygon(surface.clip.as_ref(), &corners, Vec2::lerp);
                    mesh.push_triangle_fan(vertices.iter().map(|&(position, point)| {
                        SurfaceVertex {
                            position: position.extend(0.).into(),
                            point: point.into(),
                            surface: surface.paint,
                        }
                    }));
                }
                mesh.finish_batch(
                    &mut phase.items,
                    &mut first,
                    item_index,
                    start,
                    &mut batches,
                );
            }
        }
        mesh.vertices.write_buffer(&device, &queue);
        mesh.indices.write_buffer(&device, &queue);
        commands.try_insert_batch(batches);
    }
    nodes.clear();
    texels.clear();
}
