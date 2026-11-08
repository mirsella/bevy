//! Exact-input, per-surface raster retention. The reference draw is always available.

use super::{ExtractedSurface, GpuSurface, SurfacePipeline, SurfaceVertex};
use crate::{
    render_pass::TransparentUi,
    ui_mesh::{UiIndexedBatch, UiMesh},
    UiCameraView, UiShaderVariants,
};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_color::LinearRgba;
use bevy_core_pipeline::{
    schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
    upscaling::upscaling,
};
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{Read, SRes},
        SystemParam, SystemParamItem,
    },
};
use bevy_math::{Mat4, URect, UVec2, UVec4, Vec2, Vec3};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    render_phase::{AddRenderCommand, RenderCommand, RenderCommandResult, TrackedRenderPass},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    sync_world::MainEntity,
    view::{ExtractedView, Msaa, RetainedViewEntity, ViewTarget, ViewUniformOffset},
    GpuResourceAppExt, RenderApp, RenderStartup,
};
use bevy_shader::Shader;
use core::{
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};

/// Opts a camera into retaining unchanged standard UI surfaces (fills/borders/gradients).
///
/// Full-view rasterization is copied into bounded per-node textures. Replays use
/// the original clipped geometry and nearest-filtered pixel sampling, in transparent order.
/// Both gamma `Rgba8Unorm` and linear `Rgba8UnormSrgb` are supported with matching
/// source, cache and destination formats. HDR, MSAA, offset/rescaled viewports,
/// pending pipelines and the memory limit retain the reference draw.
///
/// This adds a quantization step before blending: it is an opt-in experiment,
/// not an assertion of bit-identical appearance. Custom UI materials are never cached.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UiCachedSurfaces;

/// Render-world telemetry on each opted-in surface's temporary render entity.
#[derive(Component, Clone, Copy, Debug)]
pub struct UiSurfaceCacheStatus {
    pub main_entity: MainEntity,
    /// True only when this frame selected previously completed, unchanged pixels.
    pub reused: bool,
    /// Conservative occupied pixel count, including the one-pixel raster guard.
    pub pixel_count: u32,
    pub cache_state: &'static str,
}

pub(super) struct SurfaceCachePlugin;

impl Plugin for SurfaceCachePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "cached.wgsl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SurfaceCache>()
            .init_gpu_resource::<SpecializedRenderPipelines<CachedPipeline>>()
            .add_render_command::<TransparentUi, DrawCachedSurface>()
            .add_systems(
                RenderStartup,
                init_pipeline.after(super::init_surface_pipeline),
            )
            .add_systems(
                Core2d,
                ui_surface_cache_pass
                    .after(Core2dSystems::PostProcess)
                    .before(crate::ui_pass)
                    .before(upscaling),
            )
            .add_systems(
                Core3d,
                ui_surface_cache_pass
                    .after(Core3dSystems::PostProcess)
                    .before(crate::ui_pass)
                    .before(upscaling),
            );
    }
}

#[derive(Resource)]
struct CachedPipeline {
    source_layout: BindGroupLayoutDescriptor,
    texture_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, source: Res<SurfacePipeline>) {
    commands.insert_resource(CachedPipeline {
        source_layout: source.layout.clone(),
        texture_layout: BindGroupLayoutDescriptor::new(
            "ui_cached_surface_texture",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    uniform_buffer::<UVec4>(false),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
        shader: load_embedded_asset!(assets.as_ref(), "cached.wgsl"),
    });
}

impl SpecializedRenderPipeline for CachedPipeline {
    type Key = (TextureFormat, u32);

    fn specialize(&self, (format, probe): Self::Key) -> RenderPipelineDescriptor {
        assert!(
            supported_format(format),
            "cached UI pipeline requires an explicit supported color space"
        );
        assert!(probe <= 5, "surface cache probe must be in 0..=5");
        let shader_defs = if probe == 0 {
            vec![]
        } else {
            vec![format!("SURFACE_CACHE_PROBE_{probe}").into()]
        };
        RenderPipelineDescriptor {
            label: Some("ui_cached_surface".into()),
            layout: vec![self.source_layout.clone(), self.texture_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
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
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

fn supported_format(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    entity: MainEntity,
    view: RetainedViewEntity,
}

// Retain actual raster inputs, not the upstream layout state: transforms and
// clipping are already baked into vertices; shader options and target format
// belong to the immutable pipeline. Normalize shared-buffer addresses so an
// unrelated surface insertion cannot invalidate later retained pixels.
#[derive(Clone, PartialEq, Debug)]
struct Contents {
    vertices: Vec<u8>,
    indices: Vec<u32>,
    paint: Vec<[u32; 4]>,
    projection: [u32; 16],
    full_size: UVec2,
    rect: URect,
    pipeline: RenderPipelineId,
}

impl Contents {
    fn snapshot_bytes(&self) -> usize {
        self.vertices.len() + self.indices.len() * 4 + self.paint.len() * 16 + 1024
    }

    fn bytes(&self) -> usize {
        // Reserve the eventual pixels even while waiting for unchanged input.
        self.snapshot_bytes() + self.rect.width() as usize * self.rect.height() as usize * 4
    }
}

fn relativize_paint(paint: &mut [[u32; 4]], header: usize, base: u32) {
    let (gradients, header) = paint.split_at_mut(header);
    let surface: &mut GpuSurface = bytemuck::from_bytes_mut(bytemuck::cast_slice_mut(header));
    for [address, count] in [
        &mut surface.background_gradients,
        &mut surface.border_gradients,
    ] {
        if *count == 0 {
            *address = 0;
            continue;
        }
        *address = address
            .checked_sub(base)
            .expect("gradients belong to the surface paint range");
        super::gradient::relativize_stops(&mut gradients[*address as usize..], *count, base);
    }
}

/// Projection is exactly the one used by `prepare_view_uniforms` (jitter rejected).
fn projection(view: &ExtractedView) -> Mat4 {
    view.clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse())
}

fn occupied_pixels(vertices: &[SurfaceVertex], projection: Mat4, size: UVec2) -> Option<URect> {
    if vertices.is_empty() || size.min_element() == 0 || !projection.is_finite() {
        return None;
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for vertex in vertices {
        let p = projection * Vec3::from(vertex.position).extend(1.);
        // Standard UI is orthographic. Reject other raster mappings rather than
        // risk a crop that misses homogeneous clipping or a perspective pole.
        if !p.is_finite() || p.w.to_bits() != 1f32.to_bits() {
            return None;
        }
        let pixel =
            (p.truncate().truncate() * Vec2::new(0.5, -0.5) + Vec2::splat(0.5)) * size.as_vec2();
        min = min.min(pixel);
        max = max.max(pixel);
    }
    let min = (min.floor() - Vec2::ONE)
        .clamp(Vec2::ZERO, size.as_vec2())
        .as_uvec2();
    let max = (max.ceil() + Vec2::ONE)
        .clamp(Vec2::ZERO, size.as_vec2())
        .as_uvec2();
    let rect = URect { min, max };
    (!rect.is_empty()).then_some(rect)
}

struct Raster {
    texture: Texture,
    view: TextureView,
    size: UVec2,
    format: TextureFormat,
}

struct SourceDraw {
    range: Range<u32>,
    pipeline: CachedRenderPipelineId,
}

struct Entry {
    contents: Contents,
    pixels: Option<Pixels>,
    touched: bool,
}

struct Pixels {
    texture: Texture,
    bind_group: BindGroup,
    valid: AtomicBool,
    source: Option<SourceDraw>,
}

// Includes scratch rasters, retained pixels and CPU snapshots, not merely entry count.
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ENTRIES: usize = 512;

#[derive(Resource, Default)]
pub(super) struct SurfaceCache {
    entries: HashMap<CacheKey, Entry>,
    rasters: HashMap<RetainedViewEntity, Raster>,
    scratch: Option<Contents>,
    sampler: Option<Sampler>,
}

impl SurfaceCache {
    fn bytes(&self) -> usize {
        self.entries
            .values()
            .map(|entry| entry.contents.bytes())
            .sum::<usize>()
            + self
                .rasters
                .values()
                .map(|raster| raster.size.x as usize * raster.size.y as usize * 4)
                .sum::<usize>()
            // Reserve a reusable snapshot even when all entries currently own
            // their buffers, so the next hit cannot exceed the cache budget.
            + self.entries.values().map(|entry| entry.contents.snapshot_bytes()).max()
                .unwrap_or(0).max(self.scratch.as_ref().map_or(0, Contents::snapshot_bytes))
    }

    fn finish_frame(&mut self) {
        self.entries.retain(|_, entry| entry.touched);
        debug_assert!(self.bytes() <= MAX_BYTES);
    }
}

fn texture(
    device: &RenderDevice,
    size: UVec2,
    format: TextureFormat,
    usage: TextureUsages,
    label: &'static str,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

#[derive(Component)]
pub(super) struct CachedDraw {
    key: CacheKey,
    pipeline: CachedRenderPipelineId,
    reuse: bool,
}

#[derive(SystemParam)]
pub(super) struct SurfaceCachePreparation<'w, 's> {
    state: ResMut<'w, SurfaceCache>,
    pipeline: Res<'w, CachedPipeline>,
    pipelines: ResMut<'w, SpecializedRenderPipelines<CachedPipeline>>,
    cameras: Query<
        'w,
        's,
        (
            &'static UiCameraView,
            &'static ViewTarget,
            &'static ExtractedCamera,
            Option<&'static Msaa>,
            &'static UiShaderVariants,
        ),
    >,
    views: Query<
        'w,
        's,
        (
            &'static ExtractedView,
            Has<UiCachedSurfaces>,
            Has<TemporalJitter>,
        ),
    >,
}

impl SurfaceCachePreparation<'_, '_> {
    pub(super) fn is_enabled(&self, camera: Entity) -> bool {
        self.cameras
            .get(camera)
            .ok()
            .and_then(|(camera, ..)| self.views.get(camera.0).ok())
            .is_some_and(|(_, enabled, _)| enabled)
    }

    pub(super) fn begin_frame(&mut self, nodes: &[ExtractedSurface]) {
        let mut keys = HashSet::<CacheKey>::default();
        let mut views = HashSet::<RetainedViewEntity>::default();
        for node in nodes {
            if let Ok((camera, ..)) = self.cameras.get(node.camera)
                && let Ok((view, true, _)) = self.views.get(camera.0)
            {
                keys.insert(CacheKey {
                    entity: node.entity,
                    view: view.retained_view_entity,
                });
                views.insert(view.retained_view_entity);
            }
        }
        self.state.entries.retain(|key, entry| {
            entry.touched = false;
            if let Some(pixels) = &mut entry.pixels {
                pixels.source = None;
            }
            keys.contains(key)
        });
        // Keep scratch targets while their views remain active. Dropping them on
        // every all-hit frame caused a full-target allocation at the next miss.
        self.state.rasters.retain(|key, _| views.contains(key));
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Snapshot all already-prepared source draw inputs in one place"
    )]
    pub(super) fn prepare(
        &mut self,
        commands: &mut Commands,
        device: &RenderDevice,
        queue: &RenderQueue,
        pipelines: &PipelineCache,
        surface: &ExtractedSurface,
        source_pipeline: CachedRenderPipelineId,
        vertices: &[SurfaceVertex],
        vertex_base: u32,
        indices: &[u32],
        range: Range<u32>,
        paint: &[[u32; 4]],
    ) -> bool {
        let mut status = UiSurfaceCacheStatus {
            main_entity: surface.entity,
            reused: false,
            pixel_count: 0,
            cache_state: "unsupported",
        };
        let Some(draw) = self.prepare_inner(
            device,
            queue,
            pipelines,
            surface,
            source_pipeline,
            vertices,
            vertex_base,
            indices,
            range,
            paint,
            &mut status,
        ) else {
            commands.entity(surface.render_entity).insert(status);
            return false;
        };
        commands
            .entity(surface.render_entity)
            .insert((draw, status));
        true
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Snapshot all already-prepared source draw inputs in one place"
    )]
    fn prepare_inner(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        pipelines: &PipelineCache,
        surface: &ExtractedSurface,
        source_pipeline: CachedRenderPipelineId,
        vertices: &[SurfaceVertex],
        vertex_base: u32,
        indices: &[u32],
        range: Range<u32>,
        paint: &[[u32; 4]],
        status: &mut UiSurfaceCacheStatus,
    ) -> Option<CachedDraw> {
        let (camera, target, extracted_camera, msaa, variants) =
            self.cameras.get(surface.camera).ok()?;
        let (view, enabled, jitter) = self.views.get(camera.0).ok()?;
        let size = UVec2::new(
            target.main_texture().width(),
            target.main_texture().height(),
        );
        let format = variants.render_target_format(view.target_format);
        if !enabled
            || jitter
            || !supported_format(format)
            || view.target_format == TextureFormat::Rgba16Float
            || msaa.is_some_and(|msaa| *msaa != Msaa::Off)
            || target.main_texture().sample_count() != 1
            || view.viewport != UVec4::new(0, 0, size.x, size.y)
            || extracted_camera.viewport.as_ref().is_some_and(|viewport| {
                viewport.physical_position != UVec2::ZERO
                    || viewport.physical_size != size
                    || viewport.depth != (0.0..1.0)
            })
            || indices.is_empty()
        {
            return None;
        }
        let matrix = projection(view);
        let rect = occupied_pixels(vertices, matrix, size)?;
        status.pixel_count = rect.width() * rect.height();
        status.cache_state = "pipeline_pending";
        let actual_pipeline = pipelines.get_render_pipeline(source_pipeline)?;
        let source_descriptor = pipelines.get_render_pipeline_descriptor(source_pipeline);
        assert_eq!(
            source_descriptor
                .fragment
                .as_ref()
                .and_then(|fragment| fragment.targets.first())
                .and_then(Option::as_ref)
                .map(|target| target.format),
            Some(format),
            "surface cache source and destination color spaces must match"
        );
        let replay_pipeline = self.pipelines.specialize(
            pipelines,
            &self.pipeline,
            (format, variants.surface_cache_probe),
        );
        pipelines.get_render_pipeline(replay_pipeline)?;
        let key = CacheKey {
            entity: surface.entity,
            view: view.retained_view_entity,
        };
        let (mut vertex_bits, mut relative_indices, mut relative_paint) = self
            .state
            .scratch
            .take()
            .map(|old| (old.vertices, old.indices, old.paint))
            .unwrap_or_default();
        vertex_bits.clear();
        for vertex in vertices {
            assert_eq!(
                vertex.surface, surface.paint,
                "surface owns its paint address"
            );
            let mut vertex = *vertex;
            vertex.surface = 0;
            vertex_bits.extend_from_slice(bytemuck::bytes_of(&vertex));
        }
        relative_indices.clear();
        relative_indices.extend(indices.iter().map(|index| {
            index
                .checked_sub(vertex_base)
                .expect("indices refer to this surface's vertices")
        }));
        relative_paint.clear();
        relative_paint.extend_from_slice(paint);
        relativize_paint(
            &mut relative_paint,
            surface.paint as usize - surface.paint_range.start,
            surface.paint_range.start as u32,
        );
        let contents = Contents {
            vertices: vertex_bits,
            indices: relative_indices,
            paint: relative_paint,
            projection: matrix.to_cols_array().map(f32::to_bits),
            full_size: size,
            rect,
            pipeline: actual_pipeline.id(),
        };
        if let Some(entry) = self.state.entries.get_mut(&key)
            && entry.contents == contents
            && let Some(pixels) = &mut entry.pixels
        {
            entry.touched = true;
            let reuse = pixels.valid.load(Ordering::Acquire);
            status.reused = reuse;
            status.cache_state = if reuse { "reused" } else { "building" };
            if !reuse {
                pixels.source = Some(SourceDraw {
                    range,
                    pipeline: source_pipeline,
                });
            }
            self.state.scratch = Some(contents);
            return Some(CachedDraw {
                key,
                pipeline: replay_pipeline,
                reuse,
            });
        }
        let previous = self.state.entries.remove(&key);
        let stable = previous
            .as_ref()
            .is_some_and(|entry| entry.contents == contents);
        self.state.scratch = previous.map(|entry| entry.contents);
        status.cache_state = "memory_limit";
        if self.state.entries.len() >= MAX_ENTRIES
            || self
                .state
                .bytes()
                .checked_add(contents.bytes())?
                .checked_add(contents.snapshot_bytes())?
                > MAX_BYTES
        {
            return None;
        }
        if !stable {
            // Animated surfaces use their reference draw. Only allocate and
            // rasterize after observing the same input on consecutive frames.
            self.state.entries.insert(
                key,
                Entry {
                    contents,
                    pixels: None,
                    touched: true,
                },
            );
            status.cache_state = "warming";
            return None;
        }
        if self
            .state
            .rasters
            .get(&key.view)
            .is_some_and(|r| r.size != size || r.format != format)
        {
            self.state.rasters.remove(&key.view);
        }
        status.cache_state = "memory_limit";
        let raster_bytes = if self.state.rasters.contains_key(&key.view) {
            0
        } else {
            size.x as usize * size.y as usize * 4
        };
        if self
            .state
            .bytes()
            .checked_add(contents.bytes())?
            .checked_add(contents.snapshot_bytes())?
            .checked_add(raster_bytes)?
            > MAX_BYTES
        {
            return None;
        }
        self.state.rasters.entry(key.view).or_insert_with(|| {
            let texture = texture(
                device,
                size,
                format,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                "ui_surface_cache_raster",
            );
            let view = texture.create_view(&TextureViewDescriptor::default());
            Raster {
                texture,
                view,
                size,
                format,
            }
        });
        let texture = texture(
            device,
            rect.size(),
            format,
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            "ui_surface_cache_pixels",
        );
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let mut crop = UniformBuffer::from(UVec4::new(
            rect.min.x,
            rect.min.y,
            rect.width(),
            rect.height(),
        ));
        crop.write_buffer(device, queue);
        let sampler = self.state.sampler.get_or_insert_with(|| {
            device.create_sampler(&SamplerDescriptor {
                label: Some("ui_surface_cache_nearest"),
                ..Default::default()
            })
        });
        let bind_group = device.create_bind_group(
            "ui_surface_cache_pixels",
            &pipelines.get_bind_group_layout(&self.pipeline.texture_layout),
            &BindGroupEntries::sequential((
                &texture_view,
                crop.binding().expect("uploaded cache crop uniform"),
                &*sampler,
            )),
        );
        self.state.entries.insert(
            key,
            Entry {
                contents,
                pixels: Some(Pixels {
                    texture,
                    bind_group,
                    valid: AtomicBool::new(false),
                    source: Some(SourceDraw {
                        range,
                        pipeline: source_pipeline,
                    }),
                }),
                touched: true,
            },
        );
        status.cache_state = "building";
        Some(CachedDraw {
            key,
            pipeline: replay_pipeline,
            reuse: false,
        })
    }

    pub(super) fn finish_frame(&mut self) {
        self.state.finish_frame();
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;

/// Opaque render inputs keep private surface vertex/cache types out of public systems.
#[derive(SystemParam)]
pub struct UiSurfaceCachePass<'w> {
    state: Res<'w, SurfaceCache>,
    mesh: Res<'w, UiMesh<SurfaceVertex>>,
    pipelines: Res<'w, PipelineCache>,
}

/// Render/copy dirty surface sources before rendering the transparent UI phase.
///
/// Registered before the standard UI pass in Core2d/Core3d. An application with
/// its own direct UI pass must explicitly order that pass after this system.
pub fn ui_surface_cache_pass(
    view: ViewQuery<&UiCameraView>,
    views: Query<(&ExtractedView, &ViewUniformOffset)>,
    inputs: UiSurfaceCachePass,
    mut ctx: RenderContext,
) {
    let Ok((view, offset)) = views.get(view.into_inner().0) else {
        return;
    };
    let Some(raster) = inputs.state.rasters.get(&view.retained_view_entity) else {
        return;
    };
    let (Some(binding), Some(vertices), Some(indices)) = (
        &inputs.mesh.view_bind_group,
        inputs.mesh.vertices.buffer(),
        inputs.mesh.indices.buffer(),
    ) else {
        return;
    };
    for (key, entry) in &inputs.state.entries {
        if key.view != view.retained_view_entity {
            continue;
        }
        let Some(pixels) = &entry.pixels else {
            continue;
        };
        let Some(source) = &pixels.source else {
            continue;
        };
        let Some(pipeline) = inputs.pipelines.get_render_pipeline(source.pipeline) else {
            continue;
        };
        if pipeline.id() != entry.contents.pipeline {
            continue;
        }
        {
            let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("ui_surface_cache_source"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &raster.view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::NONE.into()),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, binding, &[offset.offset]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(source.range.clone(), 0, 0..1);
        }
        let rect = entry.contents.rect;
        let mut source = raster.texture.as_image_copy();
        source.origin = Origin3d {
            x: rect.min.x,
            y: rect.min.y,
            z: 0,
        };
        ctx.command_encoder().copy_texture_to_texture(
            source,
            pixels.texture.as_image_copy(),
            Extent3d {
                width: rect.width(),
                height: rect.height(),
                depth_or_array_layers: 1,
            },
        );
        // Only acknowledge a source after its render and copy were encoded. The
        // main pass uses reference shading this frame; replay starts next frame.
        pixels.valid.store(true, Ordering::Release);
    }
}

pub(super) struct DrawCachedSurface;

impl RenderCommand<TransparentUi> for DrawCachedSurface {
    type Param = (
        SRes<SurfaceCache>,
        SRes<UiMesh<SurfaceVertex>>,
        SRes<PipelineCache>,
    );
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = (Read<UiIndexedBatch>, Option<Read<CachedDraw>>);

    fn render<'w>(
        item: &TransparentUi,
        offset: &'w ViewUniformOffset,
        batch: Option<(&'w UiIndexedBatch, Option<&'w CachedDraw>)>,
        (cache, mesh, pipelines): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (cache, mesh, pipelines) = (
            cache.into_inner(),
            mesh.into_inner(),
            pipelines.into_inner(),
        );
        let Some((batch, cached)) = batch else {
            return RenderCommandResult::Skip;
        };
        let (Some(binding), Some(vertices), Some(indices)) = (
            &mesh.view_bind_group,
            mesh.vertices.buffer(),
            mesh.indices.buffer(),
        ) else {
            return RenderCommandResult::Failure("missing surface draw bindings");
        };
        let replay = cached.filter(|draw| draw.reuse).and_then(|draw| {
            let entry = cache.entries.get(&draw.key)?;
            let pixels = entry.pixels.as_ref()?;
            if !pixels.valid.load(Ordering::Acquire) {
                return None;
            }
            Some((
                pipelines.get_render_pipeline(draw.pipeline)?,
                &pixels.bind_group,
            ))
        });
        let pipeline = if let Some((pipeline, _)) = replay {
            pipeline
        } else {
            let Some(pipeline) = pipelines.get_render_pipeline(item.pipeline) else {
                return RenderCommandResult::Skip;
            };
            pipeline
        };
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, binding, &[offset.offset]);
        if let Some((_, pixels)) = replay {
            pass.set_bind_group(1, pixels, &[]);
        }
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
