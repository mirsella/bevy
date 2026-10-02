//! Box shadows rendering

use core::hash::Hash;

use crate::{
    clipping::{rect_without_hole, rounded_inner_rect},
    ui_mesh::{DrawUiMesh, UiMesh},
};
use bevy_app::prelude::*;
use bevy_asset::*;
use bevy_camera::visibility::InheritedVisibility;
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_ecs::prelude::*;
use bevy_math::{vec2, Affine2, FloatOrd, Rect, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_render::sync_world::{MainEntity, TemporaryRenderEntity};
use bevy_render::{
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderDevice, RenderQueue},
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_ui::{
    BackgroundColor, BoxShadow, CalculatedClip, ComputedNode, ComputedStackIndex,
    ComputedUiRenderTargetInfo, ComputedUiTargetCamera, ResolvedBorderRadius, UiGlobalTransform,
    Val,
};
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};

use crate::{
    clipping::clip_polygon,
    pipeline::{ui_color_target_state, ui_render_target_format, MANUAL_SRGB_SHADER_DEF},
    BoxShadowSamples, RenderUiSystems, TransparentUi, UiCameraMap,
};

use super::{stack_z_offsets, UiCameraView, QUAD_VERTEX_POSITIONS};

/// A plugin that enables the rendering of box shadows.
pub struct BoxShadowPlugin;

impl Plugin for BoxShadowPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "box_shadow.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawBoxShadows>()
                .init_resource::<ExtractedBoxShadows>()
                .init_gpu_resource::<UiMesh<BoxShadowVertex>>()
                .init_gpu_resource::<SpecializedRenderPipelines<BoxShadowPipeline>>()
                .add_systems(RenderStartup, init_box_shadow_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_shadows.in_set(RenderUiSystems::ExtractBoxShadows),
                )
                .add_systems(
                    Render,
                    (
                        queue_shadows.in_set(RenderSystems::Queue),
                        prepare_shadows.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BoxShadowVertex {
    position: [f32; 3],
    uvs: [f32; 2],
    vertex_color: [f32; 4],
    size: [f32; 2],
    radius: [f32; 4],
    blur: f32,
    bounds: [f32; 2],
}

#[derive(Resource)]
pub struct BoxShadowPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_box_shadow_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "box_shadow_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    commands.insert_resource(BoxShadowPipeline {
        view_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "box_shadow.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BoxShadowPipelineKey {
    pub target_format: TextureFormat,
    /// Number of samples, a higher value results in better quality shadows.
    pub samples: u32,
}

impl SpecializedRenderPipeline for BoxShadowPipeline {
    type Key = BoxShadowPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // target rect size
                VertexFormat::Float32x2,
                // corner radius values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // blur radius
                VertexFormat::Float32,
                // outer size
                VertexFormat::Float32x2,
            ],
        );
        let shader_defs = vec![
            ShaderDefVal::UInt("SHADOW_SAMPLES".to_string(), key.samples),
            MANUAL_SRGB_SHADER_DEF.into(),
        ];

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ui_color_target_state(key.target_format))],
                ..default()
            }),
            layout: vec![self.view_layout.clone()],
            label: Some("box_shadow_pipeline".into()),
            ..default()
        }
    }
}

/// Description of a shadow to be sorted and queued for rendering
pub struct ExtractedBoxShadow {
    pub stack_index: u32,
    pub transform: Affine2,
    pub bounds: Vec2,
    /// Interior covered by the node's opaque background, in shadow-local coordinates.
    pub opaque_rect: Option<Rect>,
    pub clip: Option<CalculatedClip>,
    pub extracted_camera_entity: Entity,
    pub color: LinearRgba,
    pub radius: ResolvedBorderRadius,
    pub blur_radius: f32,
    pub size: Vec2,
    pub main_entity: MainEntity,
    pub render_entity: Entity,
}

/// List of extracted shadows to be sorted and queued for rendering
#[derive(Resource, Default)]
pub struct ExtractedBoxShadows {
    pub box_shadows: Vec<ExtractedBoxShadow>,
}

pub fn extract_shadows(
    mut commands: Commands,
    mut extracted_box_shadows: ResMut<ExtractedBoxShadows>,
    box_shadow_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            &BoxShadow,
            Option<&BackgroundColor>,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedUiRenderTargetInfo,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let mut mapping = camera_map.get_mapper();

    for (
        entity,
        uinode,
        stack_index,
        transform,
        visibility,
        box_shadow,
        background,
        clip,
        camera,
        target,
    ) in &box_shadow_query
    {
        // Skip if no visible shadows
        if !visibility.get() || box_shadow.is_empty() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = mapping.map(camera) else {
            continue;
        };

        let ui_physical_viewport_size = target.physical_size().as_vec2();
        let scale_factor = target.scale_factor();
        let opaque_rect = opaque_background_rect(uinode, background);

        for drop_shadow in box_shadow.iter() {
            if drop_shadow.color.is_fully_transparent() {
                continue;
            }

            let resolve_val = |val: Val, base| {
                val.resolve(scale_factor, base, ui_physical_viewport_size)
                    .unwrap_or(0.)
            };

            let spread_x = resolve_val(drop_shadow.spread_radius, uinode.size().x);
            let spread_ratio = (spread_x + uinode.size().x) / uinode.size().x;

            let spread = vec2(spread_x, uinode.size().y * spread_ratio - uinode.size().y);

            let blur_radius = resolve_val(drop_shadow.blur_radius, uinode.size().x);
            let offset = vec2(
                resolve_val(drop_shadow.x_offset, uinode.size().x),
                resolve_val(drop_shadow.y_offset, uinode.size().y),
            );

            let shadow_size = uinode.size() + spread;
            if shadow_size.cmple(Vec2::ZERO).any() {
                continue;
            }

            let transform = Affine2::from(transform) * Affine2::from_translation(offset);
            let bounds = shadow_size + 6. * blur_radius;
            if crate::clipping::rect_is_clipped(clip, transform, bounds) {
                continue;
            }

            let radius = ResolvedBorderRadius {
                top_left: uinode.border_radius.top_left * spread_ratio,
                top_right: uinode.border_radius.top_right * spread_ratio,
                bottom_left: uinode.border_radius.bottom_left * spread_ratio,
                bottom_right: uinode.border_radius.bottom_right * spread_ratio,
            };

            extracted_box_shadows.box_shadows.push(ExtractedBoxShadow {
                render_entity: commands.spawn(TemporaryRenderEntity).id(),
                stack_index: stack_index.0,
                transform,
                color: drop_shadow.color.into(),
                bounds,
                opaque_rect: opaque_rect.map(|rect| Rect {
                    min: rect.min - offset,
                    max: rect.max - offset,
                }),
                clip: clip.cloned(),
                extracted_camera_entity,
                radius,
                blur_radius,
                size: shadow_size,
                main_entity: entity.into(),
            });
        }
    }
}

/// The background covers only the inside of the border. Stay clear of rounded corners and
/// the half-pixel antialiasing band in `background_coverage`, including during transforms.
fn opaque_background_rect(
    node: &ComputedNode,
    background: Option<&BackgroundColor>,
) -> Option<Rect> {
    if !background.is_some_and(|background| background.is_fully_opaque()) {
        return None;
    }
    rounded_inner_rect(node.size(), node.border(), node.border_radius())
}

#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of them"
)]
pub fn queue_shadows(
    extracted_box_shadows: Res<ExtractedBoxShadows>,
    box_shadow_pipeline: Res<BoxShadowPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BoxShadowPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(&UiCameraView, Option<&BoxShadowSamples>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawBoxShadows>();
    for (index, extracted_shadow) in extracted_box_shadows.box_shadows.iter().enumerate() {
        let entity = extracted_shadow.render_entity;
        let Ok((default_camera_view, shadow_samples)) =
            render_views.get(extracted_shadow.extracted_camera_entity)
        else {
            continue;
        };

        let Ok(view) = camera_views.get(default_camera_view.0) else {
            continue;
        };

        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let pipeline = pipelines.specialize(
            &pipeline_cache,
            &box_shadow_pipeline,
            BoxShadowPipelineKey {
                target_format: ui_render_target_format(view.target_format),
                samples: shadow_samples.copied().unwrap_or_default().0,
            },
        );

        transparent_phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (entity, extracted_shadow.main_entity),
            sort_key: FloatOrd(extracted_shadow.stack_index as f32 + stack_z_offsets::BOX_SHADOW),

            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            index,
            indexed: true,
        });
    }
}

pub fn prepare_shadows(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<UiMesh<BoxShadowVertex>>,
    mut extracted_shadows: ResMut<ExtractedBoxShadows>,
    view_uniforms: Res<ViewUniforms>,
    box_shadow_pipeline: Res<BoxShadowPipeline>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    mut previous_len: Local<usize>,
) {
    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches = Vec::with_capacity(*previous_len);

        ui_meta.vertices.clear();
        ui_meta.indices.clear();
        ui_meta.view_bind_group = Some(render_device.create_bind_group(
            "box_shadow_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&box_shadow_pipeline.view_layout),
            &BindGroupEntries::single(view_binding),
        ));

        for ui_phase in phases.values_mut() {
            let mut first = None;
            for item_index in 0..ui_phase.items.len() {
                let item = &mut ui_phase.items[item_index];
                let Some(box_shadow) = extracted_shadows
                    .box_shadows
                    .get(item.index)
                    .filter(|n| item.entity() == n.render_entity)
                else {
                    continue;
                };
                let rect_size = box_shadow.bounds;
                let start = ui_meta.indices.len() as u32;

                for rect in rect_without_hole(rect_size, box_shadow.opaque_rect) {
                    let corners = QUAD_VERTEX_POSITIONS.map(|pos| {
                        let local = rect.center() + pos * rect.size();
                        (
                            box_shadow.transform.transform_point2(local),
                            local / rect_size + Vec2::splat(0.5),
                        )
                    });
                    let vertices = clip_polygon(box_shadow.clip.as_ref(), &corners, Vec2::lerp);
                    if vertices.is_empty() {
                        continue;
                    }

                    ui_meta.push_triangle_fan(vertices.iter().map(|vertex| BoxShadowVertex {
                        position: vertex.0.extend(0.).into(),
                        uvs: vertex.1.into(),
                        vertex_color: box_shadow.color.to_f32_array(),
                        size: box_shadow.size.into(),
                        radius: box_shadow.radius.into(),
                        blur: box_shadow.blur_radius,
                        bounds: rect_size.into(),
                    }));
                }

                ui_meta.finish_batch(
                    &mut ui_phase.items,
                    &mut first,
                    item_index,
                    start,
                    &mut batches,
                );
            }
        }
        ui_meta.vertices.write_buffer(&render_device, &render_queue);
        ui_meta.indices.write_buffer(&render_device, &render_queue);
        *previous_len = batches.len();
        commands.try_insert_batch(batches);
    }
    extracted_shadows.box_shadows.clear();
}

pub type DrawBoxShadows = (SetItemPipeline, DrawUiMesh<BoxShadowVertex>);

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::Color;

    #[test]
    fn only_fully_opaque_backgrounds_occlude_shadows() {
        let node = ComputedNode {
            size: vec2(200., 100.),
            ..default()
        };
        assert!(opaque_background_rect(&node, None).is_none());
        for alpha in [0., 0.5, 0.999] {
            assert!(opaque_background_rect(
                &node,
                Some(&BackgroundColor(Color::srgba(1., 1., 1., alpha)))
            )
            .is_none());
        }
        let white = BackgroundColor(Color::WHITE);
        assert_eq!(
            opaque_background_rect(&node, Some(&white)),
            Some(Rect::new(-99., -49., 99., 49.))
        );

        let mut rounded = node;
        rounded.border.min_inset = vec2(20., 2.);
        rounded.border.max_inset = vec2(3., 8.);
        rounded.border_radius = ResolvedBorderRadius {
            top_left: 12.,
            top_right: 9.,
            bottom_left: 15.,
            bottom_right: 4.,
        };
        assert_eq!(
            opaque_background_rect(&rounded, Some(&white)),
            Some(Rect::new(-64., -32., 81., 26.))
        );
        rounded.border_radius.top_left = 50.;
        assert!(opaque_background_rect(&rounded, Some(&white)).is_none());
    }
}
