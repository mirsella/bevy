//! Exact-input, per-surface raster retention. The reference draw is always available.

use super::{ExtractedSurface, GpuSurface, SurfacePipeline, SurfaceVertex};
use crate::{
    pipeline::ui_render_target_format,
    render_pass::TransparentUi,
    ui_mesh::{UiIndexedBatch, UiMesh},
    UiCameraView,
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
use bevy_reflect::{prelude::ReflectDefault, Reflect};
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

/// Opts a camera into drawing its unchanged UI fills, borders and gradients
/// from retained textures instead of shading them again every frame.
///
/// A surface is rasterized into its own texture once its raster inputs have
/// been bit-identical for 8 consecutive frames, at most one surface per camera
/// per frame; until then, and from the frame any input changes, it keeps the
/// reference draw. A replay quantizes the surface to 8 bits before blending,
/// so it can differ from the reference draw: measured, by at most 2/255 per
/// channel.
///
/// The reference draw is also kept on HDR and MSAA targets, with offset or
/// scaled viewports or temporal jitter, while pipelines compile and once the
/// cache's memory limit is reached. Custom UI materials are never cached.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub struct UiCachedSurfaces;

/// How the surface cache treated one surface this frame. Every state but
/// [`Self::Reused`] draws the surface with its reference shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UiSurfaceCacheState {
    /// The camera's target or the surface's geometry cannot be cached.
    Unsupported,
    /// The reference or the replay pipeline has not compiled yet.
    PipelinePending,
    /// Caching the surface would exceed the cache's memory or entry limit.
    MemoryLimit,
    /// The raster inputs changed too recently, or the camera's build budget is spent.
    Warming,
    /// The texture is rasterized this frame and replayed from the next one.
    Building,
    /// Drawn from the retained texture.
    Reused,
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

fn texture_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ui_cached_surface_texture",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                uniform_buffer::<UVec4>(false),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, source: Res<SurfacePipeline>) {
    commands.insert_resource(CachedPipeline {
        source_layout: source.layout.clone(),
        texture_layout: texture_layout(),
        shader: load_embedded_asset!(assets.as_ref(), "cached.wgsl"),
    });
}

impl SpecializedRenderPipeline for CachedPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        assert!(
            supported_format(format),
            "cached UI pipeline requires a supported target format"
        );
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
    format == TextureFormat::Rgba8Unorm
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    entity: MainEntity,
    view: RetainedViewEntity,
}

/// A view whose target pixels coincide with UI coordinates.
struct Target {
    view: RetainedViewEntity,
    size: UVec2,
    format: TextureFormat,
    projection: Mat4,
}

/// One surface's already-prepared reference draw.
struct Source<'a> {
    entity: MainEntity,
    vertices: &'a [SurfaceVertex],
    vertex_base: u32,
    indices: &'a [u32],
    range: Range<u32>,
    /// The surface's gradient records and then its header, which the shared
    /// paint table holds at `paint_base` and `header`.
    paint: &'a [[u32; 4]],
    paint_base: u32,
    header: u32,
    pipeline: CachedRenderPipelineId,
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

    /// Reuses the allocations of `scratch`.
    fn snapshot(
        scratch: Option<Self>,
        target: &Target,
        source: &Source<'_>,
        rect: URect,
        pipeline: RenderPipelineId,
    ) -> Self {
        let (mut vertices, mut indices, mut paint) = scratch
            .map(|old| (old.vertices, old.indices, old.paint))
            .unwrap_or_default();
        vertices.clear();
        for vertex in source.vertices {
            assert_eq!(
                vertex.surface, source.header,
                "surface owns its paint address"
            );
            let mut vertex = *vertex;
            vertex.surface = 0;
            vertices.extend_from_slice(bytemuck::bytes_of(&vertex));
        }
        indices.clear();
        indices.extend(source.indices.iter().map(|index| {
            index
                .checked_sub(source.vertex_base)
                .expect("indices refer to this surface's vertices")
        }));
        paint.clear();
        paint.extend_from_slice(source.paint);
        relativize_paint(
            &mut paint,
            (source.header - source.paint_base) as usize,
            source.paint_base,
        );
        Self {
            vertices,
            indices,
            paint,
            projection: target.projection.to_cols_array().map(f32::to_bits),
            full_size: target.size,
            rect,
            pipeline,
        }
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

impl Raster {
    fn attachment(&self) -> RenderPassColorAttachment<'_> {
        RenderPassColorAttachment {
            view: &self.view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::NONE.into()),
                store: StoreOp::Store,
            },
            depth_slice: None,
        }
    }
}

struct SourceDraw {
    range: Range<u32>,
    pipeline: CachedRenderPipelineId,
}

struct Entry {
    contents: Contents,
    /// Consecutive frames, the latest included, that produced `contents`.
    stable_frames: u32,
    pixels: Option<Pixels>,
    touched: bool,
}

struct Pixels {
    texture: Texture,
    bind_group: BindGroup,
    valid: AtomicBool,
    source: Option<SourceDraw>,
}

impl Pixels {
    /// Copies `rect` out of the view's raster, which the source was just drawn to.
    fn store(&self, raster: &Raster, rect: URect, encoder: &mut CommandEncoder) {
        let mut source = raster.texture.as_image_copy();
        source.origin = Origin3d {
            x: rect.min.x,
            y: rect.min.y,
            z: 0,
        };
        encoder.copy_texture_to_texture(
            source,
            self.texture.as_image_copy(),
            Extent3d {
                width: rect.width(),
                height: rect.height(),
                depth_or_array_layers: 1,
            },
        );
        // Only acknowledge a source after its render and copy were encoded. The
        // main pass uses reference shading this frame; replay starts next frame.
        self.valid.store(true, Ordering::Release);
    }
}

// Includes scratch rasters, retained pixels and CPU snapshots, not merely entry count.
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ENTRIES: usize = 512;
/// Consecutive frames of bit-identical raster inputs before a surface is
/// rasterized. An animation that updates below the display rate holds each
/// input for a few frames; building on every hold would cost a full-target
/// pass and its allocations each time, only to discard them.
const STABLE_FRAMES: u32 = 8;
/// Surfaces of one view that may await their rasterization at a time. Each is
/// a full-target clear and store, which a tile-based GPU cannot absorb once
/// per surface in the frame that opens a screen.
const MAX_BUILDS_PER_VIEW: usize = 1;

#[derive(Resource, Default)]
pub(super) struct SurfaceCache {
    entries: HashMap<CacheKey, Entry>,
    rasters: HashMap<RetainedViewEntity, Raster>,
    /// Entries per view whose pixels are allocated but not rasterized yet.
    building: HashMap<RetainedViewEntity, usize>,
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

    /// Whether an entry for `contents`, its next snapshot and `extra` bytes fit.
    fn fits(&self, contents: &Contents, extra: usize) -> bool {
        self.bytes()
            .checked_add(contents.bytes())
            .and_then(|bytes| bytes.checked_add(contents.snapshot_bytes()))
            .and_then(|bytes| bytes.checked_add(extra))
            .is_some_and(|bytes| bytes <= MAX_BYTES)
    }

    fn begin_frame(&mut self, surfaces: &HashSet<CacheKey>, views: &HashSet<RetainedViewEntity>) {
        self.building.clear();
        self.entries.retain(|key, entry| {
            entry.touched = false;
            if !surfaces.contains(key) {
                return false;
            }
            if let Some(pixels) = &mut entry.pixels {
                pixels.source = None;
                // A build whose pass has not run keeps its place in the budget,
                // so that passes cannot pile up behind a view that skips them.
                if !pixels.valid.load(Ordering::Acquire) {
                    *self.building.entry(key.view).or_default() += 1;
                }
            }
            true
        });
        // Keep scratch targets while their views remain active. Dropping them on
        // every all-hit frame caused a full-target allocation at the next miss.
        self.rasters.retain(|key, _| views.contains(key));
    }

    /// Decides how one surface is drawn this frame, and requests its
    /// rasterization once it has stayed unchanged for long enough.
    fn observe(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        target: &Target,
        source: &Source<'_>,
        pipeline: impl FnOnce() -> Option<RenderPipelineId>,
        layout: impl FnOnce() -> BindGroupLayout,
    ) -> UiSurfaceCacheState {
        if source.indices.is_empty() {
            return UiSurfaceCacheState::Unsupported;
        }
        let Some(rect) = occupied_pixels(source.vertices, target.projection, target.size) else {
            return UiSurfaceCacheState::Unsupported;
        };
        let Some(pipeline) = pipeline() else {
            return UiSurfaceCacheState::PipelinePending;
        };
        let contents = Contents::snapshot(self.scratch.take(), target, source, rect, pipeline);
        self.retain(device, queue, target, source, contents, layout)
    }

    fn retain(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        target: &Target,
        source: &Source<'_>,
        contents: Contents,
        layout: impl FnOnce() -> BindGroupLayout,
    ) -> UiSurfaceCacheState {
        let key = CacheKey {
            entity: source.entity,
            view: target.view,
        };
        let draw = SourceDraw {
            range: source.range.clone(),
            pipeline: source.pipeline,
        };
        if let Some(entry) = self.entries.get_mut(&key)
            && entry.contents == contents
            && let Some(pixels) = &mut entry.pixels
        {
            entry.touched = true;
            self.scratch = Some(contents);
            if pixels.valid.load(Ordering::Acquire) {
                return UiSurfaceCacheState::Reused;
            }
            pixels.source = Some(draw);
            return UiSurfaceCacheState::Building;
        }
        let previous = self.entries.remove(&key);
        // Any change starts over, including one that discards retained pixels.
        let stable_frames = match &previous {
            Some(entry) if entry.contents == contents => entry.stable_frames.saturating_add(1),
            _ => 1,
        };
        self.scratch = previous.map(|entry| entry.contents);
        if self.entries.len() >= MAX_ENTRIES || !self.fits(&contents, 0) {
            return UiSurfaceCacheState::MemoryLimit;
        }
        // Animated surfaces use their reference draw. Only allocate and
        // rasterize an input that has stopped changing, and only within the
        // view's budget: the count is kept, so a surface that waits for the
        // budget builds as soon as the view can.
        if stable_frames < STABLE_FRAMES
            || self.building.get(&key.view).copied().unwrap_or(0) >= MAX_BUILDS_PER_VIEW
        {
            self.entries.insert(
                key,
                Entry {
                    contents,
                    stable_frames,
                    pixels: None,
                    touched: true,
                },
            );
            return UiSurfaceCacheState::Warming;
        }
        if self
            .rasters
            .get(&key.view)
            .is_some_and(|r| r.size != target.size || r.format != target.format)
        {
            self.rasters.remove(&key.view);
        }
        let raster_bytes = if self.rasters.contains_key(&key.view) {
            0
        } else {
            target.size.x as usize * target.size.y as usize * 4
        };
        if !self.fits(&contents, raster_bytes) {
            return UiSurfaceCacheState::MemoryLimit;
        }
        self.rasters.entry(key.view).or_insert_with(|| {
            let texture = texture(
                device,
                target.size,
                target.format,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                "ui_surface_cache_raster",
            );
            let view = texture.create_view(&TextureViewDescriptor::default());
            Raster {
                texture,
                view,
                size: target.size,
                format: target.format,
            }
        });
        let rect = contents.rect;
        let texture = texture(
            device,
            rect.size(),
            target.format,
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
        let sampler = self.sampler.get_or_insert_with(|| {
            device.create_sampler(&SamplerDescriptor {
                label: Some("ui_surface_cache_nearest"),
                ..Default::default()
            })
        });
        let bind_group = device.create_bind_group(
            "ui_surface_cache_pixels",
            &layout(),
            &BindGroupEntries::sequential((
                &texture_view,
                crop.binding().expect("uploaded cache crop uniform"),
                &*sampler,
            )),
        );
        *self.building.entry(key.view).or_default() += 1;
        self.entries.insert(
            key,
            Entry {
                contents,
                stable_frames,
                pixels: Some(Pixels {
                    texture,
                    bind_group,
                    valid: AtomicBool::new(false),
                    source: Some(draw),
                }),
                touched: true,
            },
        );
        UiSurfaceCacheState::Building
    }

    /// The rasterizations of `view` that this frame requested.
    fn pending(
        &self,
        view: RetainedViewEntity,
    ) -> impl Iterator<Item = (&Entry, &Pixels, &SourceDraw)> {
        self.entries.iter().filter_map(move |(key, entry)| {
            if key.view != view {
                return None;
            }
            let pixels = entry.pixels.as_ref()?;
            Some((entry, pixels, pixels.source.as_ref()?))
        })
    }

    /// The pixels of `key`, once their rasterization has been encoded.
    fn pixels(&self, key: &CacheKey) -> Option<&BindGroup> {
        let pixels = self.entries.get(key)?.pixels.as_ref()?;
        pixels
            .valid
            .load(Ordering::Acquire)
            .then_some(&pixels.bind_group)
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
        self.state.begin_frame(&keys, &views);
    }

    fn target(&self, camera: Entity) -> Option<Target> {
        let (camera, target, extracted_camera, msaa) = self.cameras.get(camera).ok()?;
        let (view, enabled, jitter) = self.views.get(camera.0).ok()?;
        let size = UVec2::new(
            target.main_texture().width(),
            target.main_texture().height(),
        );
        let format = ui_render_target_format(view.target_format);
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
        {
            return None;
        }
        Some(Target {
            view: view.retained_view_entity,
            size,
            format,
            projection: projection(view),
        })
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
        let Some(target) = self.target(surface.camera) else {
            return false;
        };
        let mut replay_pipeline = None;
        let state = self.state.observe(
            device,
            queue,
            &target,
            &Source {
                entity: surface.entity,
                vertices,
                vertex_base,
                indices,
                range,
                paint,
                paint_base: surface.paint_range.start as u32,
                header: surface.paint,
                pipeline: source_pipeline,
            },
            || {
                let source = pipelines.get_render_pipeline(source_pipeline)?;
                assert_eq!(
                    pipelines
                        .get_render_pipeline_descriptor(source_pipeline)
                        .fragment
                        .as_ref()
                        .and_then(|fragment| fragment.targets.first())
                        .and_then(Option::as_ref)
                        .map(|target| target.format),
                    Some(target.format),
                    "surface cache source and destination color spaces must match"
                );
                let replay = self
                    .pipelines
                    .specialize(pipelines, &self.pipeline, target.format);
                pipelines.get_render_pipeline(replay)?;
                replay_pipeline = Some(replay);
                Some(source.id())
            },
            || pipelines.get_bind_group_layout(&self.pipeline.texture_layout),
        );
        let reuse = match state {
            UiSurfaceCacheState::Reused => true,
            UiSurfaceCacheState::Building => false,
            _ => return false,
        };
        commands.entity(surface.render_entity).insert((CachedDraw {
            key: CacheKey {
                entity: surface.entity,
                view: target.view,
            },
            pipeline: replay_pipeline
                .expect("a surface with pixels has a compiled replay pipeline"),
            reuse,
        },));
        true
    }

    pub(super) fn finish_frame(&mut self) {
        self.state.finish_frame();
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;

/// Render/copy dirty surface sources before rendering the transparent UI phase.
fn ui_surface_cache_pass(
    view: ViewQuery<&UiCameraView>,
    views: Query<(&ExtractedView, &ViewUniformOffset)>,
    state: Res<SurfaceCache>,
    mesh: Res<UiMesh<SurfaceVertex>>,
    pipelines: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let Ok((view, offset)) = views.get(view.into_inner().0) else {
        return;
    };
    let Some(raster) = state.rasters.get(&view.retained_view_entity) else {
        return;
    };
    let (Some(binding), Some(vertices), Some(indices)) = (
        &mesh.view_bind_group,
        mesh.vertices.buffer(),
        mesh.indices.buffer(),
    ) else {
        return;
    };
    let mut builds = 0;
    for (entry, pixels, source) in state.pending(view.retained_view_entity) {
        let Some(pipeline) = pipelines.get_render_pipeline(source.pipeline) else {
            continue;
        };
        if pipeline.id() != entry.contents.pipeline {
            continue;
        }
        {
            let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("ui_surface_cache_source"),
                color_attachments: &[Some(raster.attachment())],
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
        pixels.store(raster, entry.contents.rect, ctx.command_encoder());
        builds += 1;
    }
    debug_assert!(
        builds <= MAX_BUILDS_PER_VIEW,
        "{builds} full-target surface cache passes were encoded for one view"
    );
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
            Some((
                pipelines.get_render_pipeline(draw.pipeline)?,
                cache.pixels(&draw.key)?,
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
