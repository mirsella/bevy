use core::{hash::Hash, ops::Range};

use crate::clipping::clip_polygon;
use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    prelude::Component,
    system::{
        lifetimeless::{Read, SRes},
        *,
    },
};
use bevy_image::prelude::*;
use bevy_image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy_math::{Affine2, FloatOrd, Rect, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_asset::RenderAssets,
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{sync_world::MainEntity, GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::{SliceScaleMode, SpriteImageMode, TextureSlicer};
use bevy_sprite_render::SpriteAssetEvents;
use bevy_ui::widget::NodeImageMode;
use bevy_ui::{ComputedStackIndex, VisualBox};
use bevy_utils::default;
use binding_types::{sampler, texture_2d};
use bytemuck::{Pod, Zeroable};

pub struct UiTextureSlicerPlugin;

impl Plugin for UiTextureSlicerPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ui_texture_slice.wgsl");
        embedded_asset!(app, "ui_texture_slice_cpu.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiTextureSlices>()
                .init_resource::<ExtractedUiTextureSlices>()
                .init_gpu_resource::<UiTextureSliceMeta>()
                .init_gpu_resource::<UiTextureSliceImageBindGroups>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiTextureSlicePipeline>>()
                .add_systems(RenderStartup, init_ui_texture_slice_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_ui_texture_slices.in_set(RenderUiSystems::ExtractTextureSlice),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_slices.in_set(RenderSystems::Queue),
                        prepare_ui_slices.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiTextureSliceVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub slices: [f32; 4],
    pub border: [f32; 4],
    pub repeat: [f32; 4],
    pub atlas: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CpuSliceVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Component)]
pub struct UiTextureSlicerBatch {
    pub range: Range<u32>,
    pub image: AssetId<Image>,
    cpu_slices: bool,
}

impl UiTextureSlicerBatch {
    fn append(
        items: &mut SortedPhaseItems<TransparentUi>,
        first: &mut Option<usize>,
        batches: &mut Vec<(Entity, Self)>,
        index: usize,
        image: AssetId<Image>,
        range: Range<u32>,
        cpu_slices: bool,
    ) {
        if let Some(previous) = *first
            && items[previous].pipeline == items[index].pipeline
        {
            let batch = &mut batches.last_mut().expect("slice batch has a first item").1;
            if batch.image == image && batch.cpu_slices == cpu_slices {
                batch.range.end = range.end;
                items[previous].batch_range_mut().end = (index - previous + 1) as u32;
                return;
            }
        }
        *first = Some(index);
        *items[index].batch_range_mut() = 0..1;
        batches.push((
            items[index].entity(),
            Self {
                range,
                image,
                cpu_slices,
            },
        ));
    }
}

#[derive(Resource)]
pub struct UiTextureSliceMeta {
    vertices: RawBufferVec<UiTextureSliceVertex>,
    cpu_vertices: RawBufferVec<CpuSliceVertex>,
    indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
}

impl Default for UiTextureSliceMeta {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            cpu_vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            view_bind_group: None,
        }
    }
}

#[derive(Resource, Default)]
pub struct UiTextureSliceImageBindGroups {
    pub values: HashMap<AssetId<Image>, BindGroup>,
}

#[derive(Resource)]
pub struct UiTextureSlicePipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub cpu_shader: Handle<Shader>,
}

pub fn init_ui_texture_slice_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(UiTextureSlicePipeline {
        view_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui_texture_slice.wgsl"),
        cpu_shader: load_embedded_asset!(asset_server.as_ref(), "ui_texture_slice_cpu.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiTextureSlicePipelineKey {
    pub target_format: TextureFormat,
    pub cpu_slices: bool,
}

impl SpecializedRenderPipeline for UiTextureSlicePipeline {
    type Key = UiTextureSlicePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let formats = if key.cpu_slices {
            vec![
                VertexFormat::Float32x3,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
            ]
        } else {
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // normalized texture slicing lines (left, top, right, bottom)
                VertexFormat::Float32x4,
                // normalized target slicing lines (left, top, right, bottom)
                VertexFormat::Float32x4,
                // repeat values (horizontal side, vertical side, horizontal center, vertical center)
                VertexFormat::Float32x4,
                // normalized texture atlas rect (left, top, right, bottom)
                VertexFormat::Float32x4,
            ]
        };
        let vertex_layout =
            VertexBufferLayout::from_vertex_formats(VertexStepMode::Vertex, formats);
        let shader_defs = vec![MANUAL_SRGB_SHADER_DEF.into()];
        let shader = if key.cpu_slices {
            &self.cpu_shader
        } else {
            &self.shader
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs,
                targets: vec![Some(ui_color_target_state(key.target_format))],
                ..default()
            }),
            layout: vec![self.view_layout.clone(), self.image_layout.clone()],
            label: Some("ui_texture_slice_pipeline".into()),
            ..default()
        }
    }
}

pub struct ExtractedUiTextureSlice {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub atlas_rect: Option<Rect>,
    pub image: AssetId<Image>,
    pub clip: Option<CalculatedClip>,
    pub extracted_camera_entity: Entity,
    pub color: LinearRgba,
    pub image_scale_mode: SpriteImageMode,
    pub flip_x: bool,
    pub flip_y: bool,
    pub inverse_scale_factor: f32,
    pub main_entity: MainEntity,
    pub render_entity: Entity,
    cpu_slices: Option<CpuSlices>,
    cpu_slice_sampler: bool,
}

#[derive(Resource, Default)]
pub struct ExtractedUiTextureSlices {
    pub slices: Vec<ExtractedUiTextureSlice>,
}

pub fn extract_ui_texture_slices(
    mut commands: Commands,
    mut extracted_ui_slicers: ResMut<ExtractedUiTextureSlices>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    slicers_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ImageNode,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
    images: Extract<Res<Assets<Image>>>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
) {
    let mut camera_mapper = camera_map.get_mapper();

    for (entity, uinode, stack_index, transform, inherited_visibility, clip, camera, image) in
        &slicers_query
    {
        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };

        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let image_scale_mode = match image.image_mode.clone() {
            NodeImageMode::Sliced(texture_slicer) => SpriteImageMode::Sliced(texture_slicer),
            NodeImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            } => SpriteImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            },
            _ => continue,
        };

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let atlas_rect = match (atlas_rect, image.rect) {
            (None, None) => None,
            (None, Some(image_rect)) => Some(image_rect),
            (Some(atlas_rect), None) => Some(atlas_rect),
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                Some(image_rect)
            }
        };

        extracted_ui_slicers.slices.push(ExtractedUiTextureSlice {
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            stack_index: stack_index.0,
            transform: Affine2::from(*transform) * Affine2::from_translation(visual_box.center()),
            color: image.color.into(),
            rect: Rect {
                min: Vec2::ZERO,
                max: visual_box.size(),
            },
            clip: clip.cloned(),
            image: image.image.id(),
            extracted_camera_entity,
            image_scale_mode,
            atlas_rect,
            flip_x: image.flip_x,
            flip_y: image.flip_y,
            inverse_scale_factor: uinode.inverse_scale_factor,
            main_entity: entity.into(),
            cpu_slices: None,
            cpu_slice_sampler: images.get(image.image.id()).is_some_and(|image| {
                cpu_slice_sampler(match &image.sampler {
                    ImageSampler::Default => &default_sampler.0,
                    ImageSampler::Descriptor(descriptor) => descriptor,
                })
            }),
        });
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of them"
)]
pub fn queue_ui_slices(
    mut extracted_ui_slicers: ResMut<ExtractedUiTextureSlices>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    ui_slicer_pipeline: Res<UiTextureSlicePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiTextureSlicePipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(&UiCameraView, &UiShaderVariants), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawUiTextureSlices>();
    for (index, extracted_slicer) in extracted_ui_slicers.slices.iter_mut().enumerate() {
        let Ok((default_camera_view, variants)) =
            render_views.get(extracted_slicer.extracted_camera_entity)
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

        let key = UiTextureSlicePipelineKey {
            target_format: ui_render_target_format(view.target_format),
            cpu_slices: false,
        };
        let full_pipeline = pipelines.specialize(&pipeline_cache, &ui_slicer_pipeline, key);
        extracted_slicer.cpu_slices = variants
            .cpu_slices
            .then(|| {
                let image = gpu_images.get(extracted_slicer.image)?;
                CpuSlices::eligible(
                    extracted_slicer,
                    image.size_2d().as_vec2(),
                    image.texture_descriptor.mip_level_count,
                )
            })
            .flatten();
        let pipeline = if extracted_slicer.cpu_slices.is_some() {
            let cpu_pipeline = pipelines.specialize(
                &pipeline_cache,
                &ui_slicer_pipeline,
                UiTextureSlicePipelineKey {
                    cpu_slices: true,
                    ..key
                },
            );
            if pipeline_cache.get_render_pipeline(cpu_pipeline).is_some() {
                cpu_pipeline
            } else {
                extracted_slicer.cpu_slices = None;
                full_pipeline
            }
        } else {
            full_pipeline
        };

        transparent_phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (extracted_slicer.render_entity, extracted_slicer.main_entity),
            sort_key: FloatOrd(extracted_slicer.stack_index as f32 + stack_z_offsets::IMAGE),
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            index,
            indexed: true,
        });
    }
}

pub fn prepare_ui_slices(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<UiTextureSliceMeta>,
    mut extracted_slices: ResMut<ExtractedUiTextureSlices>,
    view_uniforms: Res<ViewUniforms>,
    texture_slicer_pipeline: Res<UiTextureSlicePipeline>,
    mut image_bind_groups: ResMut<UiTextureSliceImageBindGroups>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    events: Res<SpriteAssetEvents>,
    mut previous_len: Local<usize>,
) {
    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Added { .. } |
            AssetEvent::Unused { .. } |
            // Images don't have dependencies
            AssetEvent::LoadedWithDependencies { .. } => {}
            AssetEvent::Modified { id } | AssetEvent::Removed { id } => {
                image_bind_groups.values.remove(id);
            }
        };
    }

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches: Vec<(Entity, UiTextureSlicerBatch)> = Vec::with_capacity(*previous_len);

        ui_meta.vertices.clear();
        ui_meta.cpu_vertices.clear();
        ui_meta.indices.clear();
        ui_meta.view_bind_group = Some(render_device.create_bind_group(
            "ui_texture_slice_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&texture_slicer_pipeline.view_layout),
            &BindGroupEntries::single(view_binding),
        ));

        for ui_phase in phases.values_mut() {
            let mut first = None;

            for item_index in 0..ui_phase.items.len() {
                let item = &ui_phase.items[item_index];
                let Some(texture_slices) = extracted_slices
                    .slices
                    .get(item.index)
                    .filter(|n| item.entity() == n.render_entity)
                else {
                    first = None;
                    continue;
                };
                let Some(gpu_image) = gpu_images.get(texture_slices.image) else {
                    first = None;
                    continue;
                };
                let start = ui_meta.indices.len() as u32;
                let color = texture_slices.color.to_f32_array();
                if let Some(cpu_slices) = &texture_slices.cpu_slices {
                    cpu_slices.for_each_polygon(texture_slices, |vertices| {
                        let base = ui_meta.cpu_vertices.len() as u32;
                        for &(position, uv) in vertices {
                            ui_meta.cpu_vertices.push(CpuSliceVertex {
                                position: position.extend(0.).into(),
                                uv: uv.into(),
                                color,
                            });
                        }
                        append_slice_indices(&mut ui_meta.indices, base, vertices.len());
                    });
                } else {
                    let rect_size = texture_slices.rect.size();
                    let positions = QUAD_VERTEX_POSITIONS
                        .map(|pos| texture_slices.transform.transform_point2(pos * rect_size));
                    let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];
                    let ([slices, border, repeat], atlas) =
                        slice_parameters(texture_slices, gpu_image.size_2d().as_vec2());
                    let vertices = clip_polygon(
                        texture_slices.clip.as_ref(),
                        &[
                            (positions[0], uvs[0]),
                            (positions[1], uvs[1]),
                            (positions[2], uvs[2]),
                            (positions[3], uvs[3]),
                        ],
                        Vec2::lerp,
                    );
                    let base = ui_meta.vertices.len() as u32;
                    for vertex in &vertices {
                        ui_meta.vertices.push(UiTextureSliceVertex {
                            position: vertex.0.extend(0.).into(),
                            uv: vertex.1.into(),
                            color,
                            slices,
                            border,
                            repeat,
                            atlas,
                        });
                    }

                    append_slice_indices(&mut ui_meta.indices, base, vertices.len());
                }
                let end = ui_meta.indices.len() as u32;
                if start == end {
                    continue;
                }
                image_bind_groups
                    .values
                    .entry(texture_slices.image)
                    .or_insert_with(|| {
                        render_device.create_bind_group(
                            "ui_texture_slice_image_layout",
                            &pipeline_cache
                                .get_bind_group_layout(&texture_slicer_pipeline.image_layout),
                            &BindGroupEntries::sequential((
                                &gpu_image.texture_view,
                                &gpu_image.sampler,
                            )),
                        )
                    });
                UiTextureSlicerBatch::append(
                    &mut ui_phase.items,
                    &mut first,
                    &mut batches,
                    item_index,
                    texture_slices.image,
                    start..end,
                    texture_slices.cpu_slices.is_some(),
                );
            }
        }
        ui_meta.vertices.write_buffer(&render_device, &render_queue);
        ui_meta
            .cpu_vertices
            .write_buffer(&render_device, &render_queue);
        ui_meta.indices.write_buffer(&render_device, &render_queue);
        *previous_len = batches.len();
        commands.try_insert_batch(batches);
    }
    extracted_slices.slices.clear();
}

pub type DrawUiTextureSlices = (
    SetItemPipeline,
    SetSlicerViewBindGroup<0>,
    SetSlicerTextureBindGroup<1>,
    DrawSlicer,
);

pub struct SetSlicerViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSlicerViewBindGroup<I> {
    type Param = SRes<UiTextureSliceMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
pub struct SetSlicerTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSlicerTextureBindGroup<I> {
    type Param = SRes<UiTextureSliceImageBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<UiTextureSlicerBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiTextureSlicerBatch>,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };

        let Some(bind_group) = image_bind_groups.values.get(&batch.image) else {
            tracing::error!(
                "Skipping UI texture slice draw: missing bind group for image {:?}",
                batch.image
            );
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}
pub struct DrawSlicer;
impl<P: PhaseItem> RenderCommand<P> for DrawSlicer {
    type Param = SRes<UiTextureSliceMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<UiTextureSlicerBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiTextureSlicerBatch>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let vertices = if batch.cpu_slices {
            ui_meta.cpu_vertices.buffer()
        } else {
            ui_meta.vertices.buffer()
        };
        let Some(vertices) = vertices else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));
        // Define how to "connect" the vertices
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}

fn append_slice_indices(indices: &mut RawBufferVec<u32>, base: u32, len: usize) {
    for i in 1..len.saturating_sub(1) as u32 {
        indices.push(base);
        indices.push(base + i);
        indices.push(base + i + 1);
    }
}

fn slice_parameters(
    node: &ExtractedUiTextureSlice,
    texture_size: Vec2,
) -> ([[f32; 4]; 3], [f32; 4]) {
    let (image_size, mut atlas) = if let Some(rect) = node.atlas_rect {
        (
            rect.size(),
            [
                rect.min.x / texture_size.x,
                rect.min.y / texture_size.y,
                rect.max.x / texture_size.x,
                rect.max.y / texture_size.y,
            ],
        )
    } else {
        (texture_size, [0., 0., 1., 1.])
    };
    if node.flip_x {
        atlas.swap(0, 2);
    }
    if node.flip_y {
        atlas.swap(1, 3);
    }
    (
        compute_texture_slices(
            image_size,
            node.rect.size() * node.inverse_scale_factor,
            &node.image_scale_mode,
        ),
        atlas,
    )
}

#[derive(Clone, Copy)]
struct CpuSlices {
    x: [f32; 4],
    y: [f32; 4],
    u: [f32; 4],
    v: [f32; 4],
}

impl CpuSlices {
    fn eligible(
        node: &ExtractedUiTextureSlice,
        texture_size: Vec2,
        mip_count: u32,
    ) -> Option<Self> {
        let geometry = Self::new(node, texture_size)?;
        // FULL's implicit derivatives can cross slice seams; the affine patch's
        // derivatives cannot. Only one mip, linear min=mag, no anisotropy makes
        // those different derivatives immaterial. Keep all other sampling FULL.
        if mip_count != 1 || !node.cpu_slice_sampler {
            return None;
        }
        // At p == the upper seam FULL's fract(1) wraps to the *lower* seam.
        // Do not silently remove that discontinuity. For axis-aligned geometry
        // it is cheap to prove both upper seams avoid pixel centers (with a
        // rounding guard). Rotated/sheared geometry cannot use this proof.
        let matrix = node.transform.matrix2;
        if matrix.x_axis.y != 0. || matrix.y_axis.x != 0. {
            return None;
        }
        for (lines, scale, translation) in [
            (geometry.x, matrix.x_axis.x, node.transform.translation.x),
            (geometry.y, matrix.y_axis.y, node.transform.translation.y),
        ] {
            if lines.windows(2).any(|p| {
                let width = ((p[1] - p[0]) * scale).abs();
                width != 0. && width < 1.
            }) {
                return None;
            }
            let seam = lines[2] * scale + translation - 0.5;
            if lines[2] != lines[3] && (seam - seam.round()).abs() < 0.01 {
                return None;
            }
        }
        Some(geometry)
    }

    fn new(node: &ExtractedUiTextureSlice, texture_size: Vec2) -> Option<Self> {
        // Tiled modes remain reference even when their computed repeat happens
        // to be one. Reject degenerate and invalid input before dividing.
        let SpriteImageMode::Sliced(slicer) = &node.image_scale_mode else {
            return None;
        };
        if !matches!(slicer.center_scale_mode, SliceScaleMode::Stretch)
            || !matches!(slicer.sides_scale_mode, SliceScaleMode::Stretch)
            || !texture_size.is_finite()
            || texture_size.min_element() <= 0.
            || !node.rect.size().is_finite()
            || node.rect.size().min_element() <= 0.
            || !node.inverse_scale_factor.is_finite()
            || node.inverse_scale_factor <= 0.
            || !slicer.max_corner_scale.is_finite()
            || slicer.max_corner_scale <= 0.
            || !node.transform.is_finite()
            || !node.transform.matrix2.determinant().is_finite()
            || node.transform.matrix2.determinant().abs() < 1e-6
        {
            return None;
        }
        if let Some(atlas) = node.atlas_rect
            && (!atlas.min.is_finite()
                || !atlas.max.is_finite()
                || atlas.min.cmplt(Vec2::ZERO).any()
                || atlas.max.cmpgt(texture_size).any()
                || atlas.size().min_element() <= 0.)
        {
            return None;
        }
        if let Some(CalculatedClip::Rects(regions)) = &node.clip
            && regions.iter().any(|region| {
                region.rect.min.is_nan()
                    || region.rect.max.is_nan()
                    || region.rect.min.cmpge(region.rect.max).any()
                    || !region.world_to_clip_local.is_finite()
                    || !region.world_to_clip_local.matrix2.determinant().is_finite()
                    || region.world_to_clip_local.matrix2.determinant().abs() < 1e-6
                    || QUAD_VERTEX_POSITIONS.iter().any(|p| {
                        let world = node.transform.transform_point2(*p * node.rect.size());
                        !region
                            .world_to_clip_local
                            .transform_point2(world)
                            .is_finite()
                    })
            })
        {
            return None;
        }
        if node.color.to_f32_array().iter().any(|v| !v.is_finite()) {
            return None;
        }
        let ([slices, border, repeat], atlas) = slice_parameters(node, texture_size);
        if repeat != [1.; 4] || !valid_slice_lines(slices) || !valid_slice_lines(border) {
            return None;
        }
        let size = node.rect.size();
        let x = [0., border[0], border[2], 1.].map(|p| (p - 0.5) * size.x);
        let y = [0., border[1], border[3], 1.].map(|p| (p - 0.5) * size.y);
        // Limit cancellation/transform roundoff relative to the 0.01px seam
        // guard above; this is deliberately not an arbitrary-coordinate path.
        if x.iter()
            .chain(&y)
            .any(|v| !v.is_finite() || v.abs() > 16384.)
            || x.windows(2).chain(y.windows(2)).any(|p| {
                let width = p[1] - p[0];
                width < 0. || (width > 0. && width < 1e-4)
            })
            || QUAD_VERTEX_POSITIONS.iter().any(|p| {
                let p = node.transform.transform_point2(*p * size);
                !p.is_finite() || p.abs().max_element() > 16384.
            })
        {
            return None;
        }
        Some(Self {
            x,
            y,
            u: [0., slices[0], slices[2], 1.].map(|p| atlas[0] + p * (atlas[2] - atlas[0])),
            v: [0., slices[1], slices[3], 1.].map(|p| atlas[1] + p * (atlas[3] - atlas[1])),
        })
    }

    fn for_each_polygon(
        &self,
        node: &ExtractedUiTextureSlice,
        mut emit: impl FnMut(&[(Vec2, Vec2)]),
    ) {
        // Reuse exactly the same grid coordinates on both sides of every seam.
        // Clip after applying the original transform, interpolating final UVs
        // within each affine patch, not within a resized post-clip rectangle.
        for y in 0..3 {
            for x in 0..3 {
                // Three-slice bars and unsliced axes have empty edge patches.
                // Omit those patches rather than rejecting the complete image.
                if self.x[x + 1] <= self.x[x] || self.y[y + 1] <= self.y[y] {
                    continue;
                }
                let corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)].map(|(x, y)| {
                    (
                        node.transform
                            .transform_point2(Vec2::new(self.x[x], self.y[y])),
                        Vec2::new(self.u[x], self.v[y]),
                    )
                });
                let vertices = clip_polygon(node.clip.as_ref(), &corners, Vec2::lerp);
                if !vertices.is_empty() {
                    emit(&vertices);
                }
            }
        }
    }
}

fn valid_slice_lines(lines: [f32; 4]) -> bool {
    lines.iter().all(|v| v.is_finite())
        && 0. <= lines[0]
        && lines[0] < lines[2]
        && lines[2] <= 1.
        && 0. <= lines[1]
        && lines[1] < lines[3]
        && lines[3] <= 1.
}

fn cpu_slice_sampler(sampler: &ImageSamplerDescriptor) -> bool {
    // Nearest magnifies tiny interpolation rounding into whole texel changes.
    sampler.min_filter == ImageFilterMode::Linear
        && sampler.mag_filter == ImageFilterMode::Linear
        && sampler.anisotropy_clamp == 1
        && sampler.compare.is_none()
}

fn compute_texture_slices(
    image_size: Vec2,
    target_size: Vec2,
    image_scale_mode: &SpriteImageMode,
) -> [[f32; 4]; 3] {
    match image_scale_mode {
        SpriteImageMode::Sliced(TextureSlicer {
            border: border_rect,
            center_scale_mode,
            sides_scale_mode,
            max_corner_scale,
        }) => {
            let min_coeff = (target_size / image_size)
                .min_element()
                .min(*max_corner_scale);

            // calculate the normalized extents of the nine-patched image slices
            let slices = [
                border_rect.min_inset.x / image_size.x,
                border_rect.min_inset.y / image_size.y,
                1. - border_rect.max_inset.x / image_size.x,
                1. - border_rect.max_inset.y / image_size.y,
            ];

            // calculate the normalized extents of the target slices
            let border = [
                (border_rect.min_inset.x / target_size.x) * min_coeff,
                (border_rect.min_inset.y / target_size.y) * min_coeff,
                1. - (border_rect.max_inset.x / target_size.x) * min_coeff,
                1. - (border_rect.max_inset.y / target_size.y) * min_coeff,
            ];

            let image_side_width = image_size.x * (slices[2] - slices[0]);
            let image_side_height = image_size.y * (slices[3] - slices[1]);
            let target_side_width = target_size.x * (border[2] - border[0]);
            let target_side_height = target_size.y * (border[3] - border[1]);

            // compute the number of times to repeat the side and center slices when tiling along each axis
            // if the returned value is `1.` the slice will be stretched to fill the axis.
            let repeat_side_x =
                compute_tiled_subaxis(image_side_width, target_side_width, sides_scale_mode);
            let repeat_side_y =
                compute_tiled_subaxis(image_side_height, target_side_height, sides_scale_mode);
            let repeat_center_x =
                compute_tiled_subaxis(image_side_width, target_side_width, center_scale_mode);
            let repeat_center_y =
                compute_tiled_subaxis(image_side_height, target_side_height, center_scale_mode);

            [
                slices,
                border,
                [
                    repeat_side_x,
                    repeat_side_y,
                    repeat_center_x,
                    repeat_center_y,
                ],
            ]
        }
        SpriteImageMode::Tiled {
            tile_x,
            tile_y,
            stretch_value,
        } => {
            let rx = compute_tiled_axis(*tile_x, image_size.x, target_size.x, *stretch_value);
            let ry = compute_tiled_axis(*tile_y, image_size.y, target_size.y, *stretch_value);
            [[0., 0., 1., 1.], [0., 0., 1., 1.], [1., 1., rx, ry]]
        }
        SpriteImageMode::Auto => {
            unreachable!("Slices can not be computed for SpriteImageMode::Stretch")
        }
        SpriteImageMode::Scale(_) => {
            unreachable!("Slices can not be computed for SpriteImageMode::Scale")
        }
    }
}

fn compute_tiled_axis(tile: bool, image_extent: f32, target_extent: f32, stretch: f32) -> f32 {
    if tile {
        let s = image_extent * stretch;
        target_extent / s
    } else {
        1.
    }
}

fn compute_tiled_subaxis(image_extent: f32, target_extent: f32, mode: &SliceScaleMode) -> f32 {
    match mode {
        SliceScaleMode::Stretch => 1.,
        SliceScaleMode::Tile { stretch_value } => {
            let s = image_extent * *stretch_value;
            target_extent / s
        }
    }
}

#[cfg(test)]
mod tests;
