extern crate alloc;

use super::*;
use crate::{
    clipping::{rect_without_hole, rounded_inner_rect},
    surface::{
        append_paint, gamma_color,
        gradient::GradientExtractor,
        tests::{projection as target_projection, shader_with_defs, Gpu, HEIGHT, WIDTH},
    },
    QUAD_VERTEX_POSITIONS,
};
use alloc::borrow::Cow;
use bevy_color::Color;
use bevy_math::Affine2;
use bevy_sprite::BorderRect;
use bevy_ui::{
    ColorStop, ComputedUiRenderTargetInfo, Gradient, InterpolationColorSpace, LinearGradient,
    ResolvedBorderRadius,
};
use wgpu::util::DeviceExt;
use UiSurfaceCacheState::{Building, MemoryLimit, Reused, Warming};

fn contents() -> Contents {
    Contents {
        vertices: vec![0; 4 * size_of::<SurfaceVertex>()],
        indices: vec![0, 1, 2, 0, 2, 3],
        // Stand in for contiguous stops, gradient records and surface header.
        paint: vec![[0; 4]; 16],
        projection: Mat4::IDENTITY.to_cols_array().map(f32::to_bits),
        full_size: UVec2::new(100, 80),
        rect: URect::new(10, 20, 30, 40),
        pipeline: RenderPipelineId::new(),
    }
}

#[test]
fn every_raster_input_bit_participates_in_the_snapshot() {
    let original = contents();
    macro_rules! invalidates {
        ($field:ident, $value:expr) => {{
            let mut changed = original.clone();
            changed.$field = $value;
            assert_ne!(original, changed, stringify!($field));
        }};
    }
    invalidates!(vertices, vec![1; original.vertices.len()]);
    invalidates!(indices, vec![0, 2, 1, 0, 2, 3]);
    invalidates!(
        projection,
        Mat4::from_scale(Vec3::splat(2.))
            .to_cols_array()
            .map(f32::to_bits)
    );
    invalidates!(full_size, UVec2::new(101, 80));
    invalidates!(rect, URect::new(11, 20, 31, 40));
    invalidates!(pipeline, RenderPipelineId::new());
    // Every bit in every referenced paint/stop/gradient texel participates.
    for texel in 0..original.paint.len() {
        for channel in 0..4 {
            for bit in 0..32 {
                let mut changed = original.clone();
                changed.paint[texel][channel] ^= 1 << bit;
                assert_ne!(original, changed);
            }
        }
    }
    let mut changed = original.clone();
    changed.paint.push([0; 4]);
    assert_ne!(original, changed);
    // Float equality is intentionally bitwise, including signed zero/NaN payloads.
    let mut matrix = original.projection;
    matrix[1] = (-0f32).to_bits();
    invalidates!(projection, matrix);
}

#[test]
fn cache_keys_include_source_and_retained_view_identity() {
    let source = Entity::from_raw_u32(1).unwrap().into();
    let camera = Entity::from_raw_u32(2).unwrap().into();
    let first = CacheKey {
        entity: source,
        view: RetainedViewEntity::new(camera, None, 1),
    };
    assert_ne!(
        first,
        CacheKey {
            entity: Entity::from_raw_u32(3).unwrap().into(),
            ..first
        }
    );
    assert_ne!(
        first,
        CacheKey {
            view: RetainedViewEntity::new(camera, None, 2),
            ..first
        }
    );
    assert_ne!(
        first,
        CacheKey {
            view: RetainedViewEntity::new(Entity::from_raw_u32(4).unwrap().into(), None, 1),
            ..first
        }
    );
}

#[test]
fn paint_relocation_preserves_pixels_but_stop_changes_invalidate() {
    let snapshot = |base: usize, color| {
        let mut paint = vec![[0; 4]; base];
        let mut extractor = GradientExtractor::default();
        let layers = [LinearGradient {
            stops: vec![ColorStop::auto(Color::WHITE), ColorStop::auto(color)],
            ..Default::default()
        }
        .into()];
        let background_gradients = extractor.extract(
            &layers,
            Vec2::splat(100.),
            &ComputedUiRenderTargetInfo::default(),
            &mut paint,
        );
        let border_gradients = extractor.extract(
            &layers,
            Vec2::splat(100.),
            &ComputedUiRenderTargetInfo::default(),
            &mut paint,
        );
        let header = append_paint(
            &mut paint,
            &[GpuSurface {
                background_gradients,
                border_gradients,
                ..Default::default()
            }],
        );
        let mut local = paint[base..].to_vec();
        relativize_paint(&mut local, header as usize - base, base as u32);
        local
    };
    assert_eq!(snapshot(0, Color::BLACK), snapshot(123, Color::BLACK));
    assert_ne!(snapshot(0, Color::BLACK), snapshot(123, Color::WHITE));
}

fn vertex(x: f32, y: f32) -> SurfaceVertex {
    SurfaceVertex {
        position: [x, y, 0.],
        point: [x, y],
        surface: 0,
    }
}

#[test]
fn occupied_pixels_use_full_view_projection_guard_and_target_clamp() {
    let size = UVec2::new(100, 80);
    let projection = Mat4::orthographic_rh(0., 100., 80., 0., -1., 1.);
    let vertices = [
        vertex(20.25, 15.25),
        vertex(30.75, 15.25),
        vertex(30.75, 25.75),
        vertex(20.25, 25.75),
    ];
    assert_eq!(
        occupied_pixels(&vertices, projection, size),
        Some(URect::new(19, 14, 32, 27))
    );
    let translated = Mat4::from_translation(Vec3::new(0.2, -0.25, 0.)) * projection;
    assert_eq!(
        occupied_pixels(&vertices, translated, size),
        Some(URect::new(29, 24, 42, 37))
    );
    assert_eq!(
        occupied_pixels(&[vertex(-50., -50.), vertex(500., 500.)], projection, size),
        Some(URect::from_corners(UVec2::ZERO, size))
    );
    assert!(occupied_pixels(&[], projection, size).is_none());
    assert!(occupied_pixels(&[vertex(f32::NAN, 0.)], projection, size).is_none());
    assert!(occupied_pixels(&vertices, Mat4::perspective_rh(1., 1., 0.1, 100.), size).is_none());
    assert!(occupied_pixels(&vertices, projection, UVec2::ZERO).is_none());
    assert!(occupied_pixels(
        &[vertex(-500., -500.), vertex(-100., -100.)],
        projection,
        size
    )
    .is_none());
}

fn pipeline() -> CachedPipeline {
    CachedPipeline {
        source_layout: BindGroupLayoutDescriptor::new("source", &[]),
        texture_layout: BindGroupLayoutDescriptor::new("cache", &[]),
        shader: Handle::default(),
    }
}

#[test]
fn replay_pipeline_retains_exact_color_space_and_premultiplied_blending() {
    let descriptor = pipeline().specialize(TextureFormat::Rgba8Unorm);
    assert!(descriptor.vertex.shader_defs.is_empty());
    let fragment = descriptor.fragment.unwrap();
    assert!(fragment.shader_defs.is_empty());
    let target = fragment.targets[0].clone().unwrap();
    assert_eq!(target.format, TextureFormat::Rgba8Unorm);
    assert_eq!(target.blend, Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING));
    assert_eq!(descriptor.multisample.count, 1);
    assert!(descriptor.depth_stencil.is_none());
    assert_eq!(descriptor.layout.len(), 2);
    assert!(!supported_format(TextureFormat::Rgba16Float));
    assert!(!supported_format(TextureFormat::Rgba8UnormSrgb));
    assert!(!supported_format(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn replay_stages_validate_and_translate_without_a_gpu() {
    use naga::{back::glsl, valid, ShaderStage};
    use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};

    for (stage, entry_point) in [
        (ShaderStage::Vertex, "vertex"),
        (ShaderStage::Fragment, "fragment"),
    ] {
        let mut composer = Composer::default();
        if let Err(error) = composer.add_composable_module(ComposableModuleDescriptor {
            source: include_str!("../../../bevy_render/src/view/view.wgsl"),
            file_path: "view.wgsl",
            ..Default::default()
        }) {
            panic!("{}", error.emit_to_string(&composer));
        }
        let module = composer
            .make_naga_module(NagaModuleDescriptor {
                source: include_str!("cached.wgsl"),
                file_path: "cached.wgsl",
                ..Default::default()
            })
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&composer)));
        let info =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::empty())
                .validate(&module)
                .unwrap();
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.stage == stage)
            .unwrap();
        // The source layout exposes View only to vertex shaders. The replay
        // must not accidentally require it in the fragment stage.
        if stage == ShaderStage::Fragment {
            let view = module
                .global_variables
                .iter()
                .find(|(_, global)| {
                    global.binding
                        == Some(naga::ResourceBinding {
                            group: 0,
                            binding: 0,
                        })
                })
                .unwrap()
                .0;
            assert!(entry.function.expressions.iter().all(|(_, expression)| {
                !matches!(expression, naga::Expression::GlobalVariable(handle) if *handle == view)
            }));
        }
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let mut output = String::new();
        glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &options,
            &glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry_point.into(),
                multiview: None,
            },
            Default::default(),
        )
        .unwrap()
        .write()
        .expect("cached replay must translate to WebGL2 GLSL ES 3.00");
        naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: stage,
                entry_point: entry_point.into(),
            }),
        )
        .expect("cached replay must translate to Vulkan SPIR-V");
        naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options::default(),
            &naga::back::msl::PipelineOptions::default(),
        )
        .expect("cached replay must translate to Metal");
    }
}

#[test]
fn camera_opt_in_is_copied_only_to_fresh_ui_view_and_can_be_removed() {
    use crate::{extract_ui_camera_view, render_pass::TransparentUi, UiShaderVariants};
    use bevy_camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo};
    use bevy_ecs::system::RunSystemOnce;
    use bevy_render::{
        camera::CameraMainPassTextureFormats, render_phase::ViewSortedRenderPhases,
        sync_world::RenderEntity, MainWorld,
    };
    let mut world = World::new();
    let render_camera = world.spawn_empty().id();
    let mut main = MainWorld::default();
    let camera = main
        .spawn((
            Camera2d,
            Camera {
                computed: ComputedCameraValues {
                    target_info: Some(RenderTargetInfo {
                        physical_size: UVec2::new(100, 80),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
            RenderEntity::from(render_camera),
            UiCachedSurfaces,
            UiShaderVariants::ALL,
        ))
        .id();
    world.insert_resource(main);
    let mut formats = CameraMainPassTextureFormats::default();
    formats
        .0
        .insert(render_camera, TextureFormat::Rgba8UnormSrgb);
    world.insert_resource(formats);
    world.init_resource::<ViewSortedRenderPhases<TransparentUi>>();
    world.run_system_once(extract_ui_camera_view).unwrap();
    let first = world.get::<UiCameraView>(render_camera).unwrap().0;
    assert!(world.get::<UiCachedSurfaces>(first).is_some());
    assert!(world.get::<UiCachedSurfaces>(render_camera).is_none());
    world
        .resource_mut::<MainWorld>()
        .entity_mut(camera)
        .remove::<UiCachedSurfaces>();
    world.run_system_once(extract_ui_camera_view).unwrap();
    let second = world.get::<UiCameraView>(render_camera).unwrap().0;
    assert_ne!(first, second);
    assert!(world.get::<UiCachedSurfaces>(second).is_none());
    assert!(world.get::<UiCachedSurfaces>(render_camera).is_none());
    assert_eq!(
        *world.get::<UiShaderVariants>(second).unwrap(),
        UiShaderVariants::ALL
    );
    world
        .resource_mut::<MainWorld>()
        .entity_mut(camera)
        .remove::<UiShaderVariants>();
    world.run_system_once(extract_ui_camera_view).unwrap();
    let third = world.get::<UiCameraView>(render_camera).unwrap().0;
    for entity in [third, render_camera] {
        assert_eq!(
            *world.get::<UiShaderVariants>(entity).unwrap(),
            UiShaderVariants::default()
        );
    }
}

fn entity(index: u32) -> MainEntity {
    Entity::from_raw_u32(index).unwrap().into()
}

fn view(index: u32) -> RetainedViewEntity {
    RetainedViewEntity::new(entity(index), None, 0)
}

/// An opaque square, whose shade stands in for any raster input.
#[derive(Clone, Copy)]
struct Square {
    entity: u32,
    view: u32,
    shade: f32,
}

/// Frames of the cache as `prepare_surfaces` and `ui_surface_cache_pass` drive
/// it, on `wgpu`'s noop backend.
struct Frames {
    device: RenderDevice,
    queue: RenderQueue,
    layout: BindGroupLayout,
    pipeline: RenderPipelineId,
    cache: SurfaceCache,
    /// Whether the views' cache passes run.
    pass: bool,
}

impl Frames {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = bevy_tasks::block_on(instance.request_adapter(&Default::default()))
            .expect("the noop backend always has an adapter");
        let (device, queue) =
            bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            }))
            .expect("the noop backend always has a device");
        let device = RenderDevice::from(device);
        Self {
            layout: device.create_bind_group_layout(None, &texture_layout().entries),
            device,
            queue: RenderQueue::new(queue),
            pipeline: RenderPipelineId::new(),
            cache: SurfaceCache::default(),
            pass: true,
        }
    }

    /// Draws `squares` in order and returns their states.
    fn frame(&mut self, squares: &[Square]) -> Vec<UiSurfaceCacheState> {
        let key = |square: &Square| CacheKey {
            entity: entity(square.entity),
            view: view(square.view),
        };
        let views: HashSet<_> = squares.iter().map(|square| view(square.view)).collect();
        self.cache
            .begin_frame(&squares.iter().map(key).collect(), &views);
        // Squares share a mesh and a paint table, as surfaces do, so their
        // snapshots must not depend on addresses.
        let mut vertices = Vec::new();
        let mut texels = vec![[0; 4]; 3];
        let mut states = Vec::new();
        for (index, square) in squares.iter().enumerate() {
            let header = append_paint(
                &mut texels,
                &[GpuSurface {
                    size: [16.; 2],
                    background: [square.shade, 0., 0., 1.],
                    ..Default::default()
                }],
            );
            let base = vertices.len();
            vertices.extend([[8., 8.], [24., 8.], [24., 24.], [8., 24.]].map(|[x, y]| {
                SurfaceVertex {
                    position: [x, y, 0.],
                    point: [x, y],
                    surface: header,
                }
            }));
            let start = index as u32 * 6;
            let state = self.cache.observe(
                &self.device,
                &self.queue,
                &Target {
                    view: view(square.view),
                    size: UVec2::splat(64),
                    format: TextureFormat::Rgba8Unorm,
                    projection: Mat4::orthographic_rh(0., 64., 64., 0., -1., 1.),
                },
                &Source {
                    entity: entity(square.entity),
                    vertices: &vertices[base..],
                    vertex_base: base as u32,
                    indices: &[0, 1, 2, 0, 2, 3].map(|i| base as u32 + i),
                    range: start..start + 6,
                    paint: &texels[header as usize..],
                    paint_base: header,
                    header,
                    pipeline: CachedRenderPipelineId::INVALID,
                },
                || Some(self.pipeline),
                || self.layout.clone(),
            );
            states.push(state);
        }
        self.cache.finish_frame();
        if self.pass {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for view in &views {
                let mut builds = 0;
                for (entry, pixels, _) in self.cache.pending(*view) {
                    pixels.store(&self.cache.rasters[view], entry.contents.rect, &mut encoder);
                    builds += 1;
                }
                assert!(builds <= MAX_BUILDS_PER_VIEW);
            }
            self.queue.submit([encoder.finish()]);
        }
        states
    }

    /// The frames in which an unchanged input is not stable enough to build.
    fn warm(&mut self, squares: &[Square]) {
        for _ in 1..STABLE_FRAMES {
            let states = self.frame(squares);
            assert!(states.iter().all(|state| *state == Warming), "{states:?}");
        }
    }
}

fn squares<const N: usize>(entities: [u32; N]) -> [Square; N] {
    entities.map(|entity| Square {
        entity,
        view: 0,
        shade: 0.5,
    })
}

#[test]
fn a_surface_builds_after_the_stability_count_and_is_then_replayed() {
    let mut frames = Frames::new();
    let square = squares([1]);
    frames.warm(&square);
    assert_eq!(frames.frame(&square), [Building]);
    for _ in 0..3 {
        assert_eq!(frames.frame(&square), [Reused]);
    }
}

#[test]
fn a_changed_input_returns_to_the_reference_draw_and_restarts_the_count() {
    let mut frames = Frames::new();
    let mut square = squares([1]);
    // One frame short of a build.
    frames.warm(&square);
    square[0].shade = 0.25;
    frames.warm(&square);
    assert_eq!(frames.frame(&square), [Building]);
    assert_eq!(frames.frame(&square), [Reused]);
    // Retained pixels are dropped with the input they were rasterized from.
    square[0].shade = 0.75;
    frames.warm(&square);
    assert_eq!(frames.frame(&square), [Building]);
    assert_eq!(frames.frame(&square), [Reused]);
}

#[test]
fn an_input_held_for_fewer_frames_than_the_count_is_never_built() {
    for hold in [1, 2, STABLE_FRAMES - 1] {
        let mut frames = Frames::new();
        let mut square = squares([1]);
        for frame in 0..6 * STABLE_FRAMES {
            square[0].shade = (frame / hold % 2) as f32;
            assert_eq!(
                frames.frame(&square),
                [Warming],
                "hold {hold}, frame {frame}"
            );
        }
        assert!(frames.cache.rasters.is_empty());
    }
}

#[test]
fn surfaces_that_stabilize_together_build_one_per_frame() {
    let mut frames = Frames::new();
    let squares = squares([1, 2, 3, 4]);
    frames.warm(&squares);
    assert_eq!(
        frames.frame(&squares),
        [Building, Warming, Warming, Warming]
    );
    assert_eq!(frames.frame(&squares), [Reused, Building, Warming, Warming]);
    assert_eq!(frames.frame(&squares), [Reused, Reused, Building, Warming]);
    assert_eq!(frames.frame(&squares), [Reused, Reused, Reused, Building]);
    assert_eq!(frames.frame(&squares), [Reused; 4]);
}

#[test]
fn a_surface_waiting_for_the_budget_keeps_its_count_unless_its_input_changes() {
    let mut frames = Frames::new();
    let mut squares = squares([1, 2, 3]);
    frames.warm(&squares);
    assert_eq!(frames.frame(&squares), [Building, Warming, Warming]);
    squares[1].shade = 0.25;
    assert_eq!(frames.frame(&squares), [Reused, Warming, Building]);
    for _ in 2..STABLE_FRAMES {
        assert_eq!(frames.frame(&squares), [Reused, Warming, Reused]);
    }
    assert_eq!(frames.frame(&squares), [Reused, Building, Reused]);
    assert_eq!(frames.frame(&squares), [Reused; 3]);
}

#[test]
fn each_view_has_its_own_build_budget() {
    let mut frames = Frames::new();
    let squares = [(1, 0), (2, 0), (1, 1), (2, 1)].map(|(entity, view)| Square {
        entity,
        view,
        shade: 0.5,
    });
    frames.warm(&squares);
    assert_eq!(
        frames.frame(&squares),
        [Building, Warming, Building, Warming]
    );
    assert_eq!(frames.frame(&squares), [Reused, Building, Reused, Building]);
    assert_eq!(frames.frame(&squares), [Reused; 4]);
}

#[test]
fn a_build_whose_pass_did_not_run_holds_the_budget() {
    let mut frames = Frames::new();
    let squares = squares([1, 2]);
    frames.warm(&squares);
    frames.pass = false;
    for _ in 0..3 {
        assert_eq!(frames.frame(&squares), [Building, Warming]);
    }
    frames.pass = true;
    assert_eq!(frames.frame(&squares), [Building, Warming]);
    assert_eq!(frames.frame(&squares), [Reused, Building]);
    assert_eq!(frames.frame(&squares), [Reused; 2]);
}

#[test]
fn surfaces_over_the_entry_limit_keep_their_reference_draw() {
    let mut frames = Frames::new();
    let squares: Vec<_> = (1..=MAX_ENTRIES as u32 + 1)
        .map(|entity| squares([entity])[0])
        .collect();
    let states = frames.frame(&squares);
    assert!(states[..MAX_ENTRIES].iter().all(|state| *state == Warming));
    assert_eq!(states[MAX_ENTRIES], MemoryLimit);
}

/// One surface as `extract_surfaces` describes it.
struct Panel {
    surface: GpuSurface,
    gradient: Option<Gradient>,
    transform: Affine2,
}

struct Draw {
    vertices: Range<usize>,
    indices: Range<u32>,
    paint: Range<usize>,
    header: u32,
}

/// The mesh and paint table that `prepare_surfaces` builds from extracted surfaces.
struct Scene {
    vertices: Vec<SurfaceVertex>,
    indices: Vec<u32>,
    texels: Vec<[u32; 4]>,
    draws: Vec<Draw>,
}

impl Scene {
    fn new(panels: &[Panel]) -> Self {
        let mut scene = Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            texels: vec![[0; 4]; 3],
            draws: Vec::new(),
        };
        let mut extractor = GradientExtractor::default();
        for panel in panels {
            let mut surface = panel.surface;
            let size = Vec2::from(surface.size);
            let paint = scene.texels.len();
            surface.background_gradients = extractor.extract(
                panel.gradient.as_slice(),
                size,
                &ComputedUiRenderTargetInfo::default(),
                &mut scene.texels,
            );
            let header = append_paint(&mut scene.texels, core::slice::from_ref(&surface));
            let [left, top, right, bottom] = surface.border;
            let [top_left, top_right, bottom_right, bottom_left] = surface.radius;
            let has_fill = surface.background[3] != 0. || surface.background_gradients[1] != 0;
            let hole = if has_fill {
                None
            } else {
                rounded_inner_rect(
                    size,
                    BorderRect {
                        min_inset: Vec2::new(left, top),
                        max_inset: Vec2::new(right, bottom),
                    },
                    ResolvedBorderRadius {
                        top_left,
                        top_right,
                        bottom_right,
                        bottom_left,
                    },
                )
            };
            let (vertices, indices) = (scene.vertices.len(), scene.indices.len() as u32);
            for part in rect_without_hole(size, hole) {
                let base = scene.vertices.len() as u32;
                scene.vertices.extend(QUAD_VERTEX_POSITIONS.map(|p| {
                    let point = part.center() + p * part.size();
                    SurfaceVertex {
                        position: panel.transform.transform_point2(point).extend(0.).into(),
                        point: point.into(),
                        surface: header,
                    }
                }));
                scene.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
            }
            scene.draws.push(Draw {
                vertices: vertices..scene.vertices.len(),
                indices: indices..scene.indices.len() as u32,
                paint: paint..scene.texels.len(),
                header,
            });
        }
        scene
    }
}

/// A translucent rounded fill, a hollow border, a rotated gradient and a
/// bordered fill that the target's right edge cuts.
fn panels() -> [Panel; 4] {
    let color = |red, green, blue, alpha| gamma_color(LinearRgba::new(red, green, blue, alpha));
    [
        Panel {
            surface: GpuSurface {
                size: [28., 18.],
                radius: [6.; 4],
                background: color(0.9, 0.3, 0.1, 0.7),
                ..Default::default()
            },
            gradient: None,
            transform: Affine2::from_translation(Vec2::new(18.375, 12.625)),
        },
        Panel {
            surface: GpuSurface {
                size: [30., 20.],
                radius: [5.; 4],
                border: [3.; 4],
                border_colors: [
                    color(1., 1., 1., 1.),
                    color(0.1, 0.8, 0.2, 0.9),
                    color(0.9, 0.9, 0.1, 0.6),
                    color(0.5, 0.1, 0.9, 1.),
                ],
                ..Default::default()
            },
            gradient: None,
            transform: Affine2::from_translation(Vec2::new(59.5, 13.25)),
        },
        Panel {
            surface: GpuSurface {
                size: [30., 16.],
                radius: [3.; 4],
                ..Default::default()
            },
            gradient: Some(
                LinearGradient {
                    angle: LinearGradient::TO_RIGHT,
                    color_space: InterpolationColorSpace::Srgba,
                    stops: vec![
                        ColorStop::auto(Color::srgba(0.9, 0.2, 0.3, 1.)),
                        ColorStop::auto(Color::srgba(0.1, 0.4, 0.9, 0.5)),
                    ],
                }
                .into(),
            ),
            transform: Affine2::from_angle_translation(0.3, Vec2::new(20.25, 34.5)),
        },
        Panel {
            surface: GpuSurface {
                size: [24., 20.],
                radius: [4.; 4],
                border: [2.; 4],
                background: color(0.1, 0.1, 0.1, 0.8),
                border_colors: [color(1., 0.8, 0.2, 1.); 4],
                ..Default::default()
            },
            gradient: None,
            transform: Affine2::from_translation(Vec2::new(74., 36.)),
        },
    ]
}

/// The cache on a real device, with the passes that `ui_surface_cache_pass`
/// and `DrawCachedSurface` encode.
struct Replay {
    gpu: Gpu,
    device: RenderDevice,
    queue: RenderQueue,
    layout: BindGroupLayout,
    view_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    reference: RenderPipelineId,
    cache: SurfaceCache,
}

impl Replay {
    fn new(backends: wgpu::Backends) -> Self {
        let gpu = Gpu::new(backends);
        let device = RenderDevice::from(gpu.device.clone());
        let layout = device.create_bind_group_layout(None, &texture_layout().entries);
        let view_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cached surface regression"),
            source: ShaderSource::Naga(Cow::Owned(shader_with_defs(
                include_str!("cached.wgsl"),
                true,
                &[],
            ))),
        });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&view_layout), Some(layout.value())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cached surface regression"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<SurfaceVertex>() as u64,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3, 1 => Float32x2, 2 => Uint32
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        Self {
            queue: RenderQueue::new(gpu.queue.clone()),
            gpu,
            device,
            layout,
            view_layout,
            pipeline,
            reference: RenderPipelineId::new(),
            cache: SurfaceCache::default(),
        }
    }

    /// Renders `scene` over [`BACKGROUND`], through the cache when `cached`,
    /// and returns the frame with each surface's state.
    fn frame(&mut self, scene: &Scene, cached: bool) -> (Vec<u8>, Vec<UiSurfaceCacheState>) {
        let device = &self.gpu.device;
        let buffer = |contents: &[u8], usage| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents,
                usage,
            })
        };
        let vertices = buffer(bytemuck::cast_slice(&scene.vertices), BufferUsages::VERTEX);
        let indices = buffer(bytemuck::cast_slice(&scene.indices), BufferUsages::INDEX);
        let reference = self.gpu.bindings(&scene.texels);
        let view = buffer(
            bytemuck::cast_slice(&target_projection().to_cols_array()),
            BufferUsages::UNIFORM,
        );
        let replay = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.view_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view.as_entire_binding(),
            }],
        });
        let key = |draw: usize| CacheKey {
            entity: entity(draw as u32 + 1),
            view: self::view(0),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut states = Vec::new();
        if cached {
            let target = Target {
                view: self::view(0),
                size: UVec2::new(WIDTH, HEIGHT),
                format: TextureFormat::Rgba8Unorm,
                projection: target_projection(),
            };
            self.cache.begin_frame(
                &(0..scene.draws.len()).map(key).collect(),
                &HashSet::from_iter([target.view]),
            );
            for (index, draw) in scene.draws.iter().enumerate() {
                let state = self.cache.observe(
                    &self.device,
                    &self.queue,
                    &target,
                    &Source {
                        entity: key(index).entity,
                        vertices: &scene.vertices[draw.vertices.clone()],
                        vertex_base: draw.vertices.start as u32,
                        indices: &scene.indices
                            [draw.indices.start as usize..draw.indices.end as usize],
                        range: draw.indices.clone(),
                        paint: &scene.texels[draw.paint.clone()],
                        paint_base: draw.paint.start as u32,
                        header: draw.header,
                        pipeline: CachedRenderPipelineId::INVALID,
                    },
                    || Some(self.reference),
                    || self.layout.clone(),
                );
                states.push(state);
            }
            self.cache.finish_frame();
            for (entry, pixels, source) in self.cache.pending(target.view) {
                let raster = &self.cache.rasters[&target.view];
                {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(raster.attachment())],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.gpu.pipeline);
                    pass.set_bind_group(0, &reference, &[]);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(source.range.clone(), 0, 0..1);
                }
                pixels.store(raster, entry.contents.rect, &mut encoder);
            }
        }
        let target = self.gpu.target();
        let target_view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: f64::from(BACKGROUND[0]) / 255.,
                            g: f64::from(BACKGROUND[1]) / 255.,
                            b: f64::from(BACKGROUND[2]) / 255.,
                            a: f64::from(BACKGROUND[3]) / 255.,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
            for (index, draw) in scene.draws.iter().enumerate() {
                if states.get(index) == Some(&Reused) {
                    let pixels: &wgpu::BindGroup = self
                        .cache
                        .pixels(&key(index))
                        .expect("a reused surface has rasterized pixels");
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &replay, &[]);
                    pass.set_bind_group(1, pixels, &[]);
                } else {
                    pass.set_pipeline(&self.gpu.pipeline);
                    pass.set_bind_group(0, &reference, &[]);
                }
                pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
        (self.gpu.read(encoder, &target), states)
    }
}

/// Opaque, so that every translucent pixel of a surface is blended.
const BACKGROUND: [u8; 4] = [51, 102, 153, 255];
/// A replay blends 8-bit colour and alpha where the reference draw blends
/// unquantized ones.
const REPLAY_TOLERANCE: u8 = 2;

/// Returns the largest difference of any channel.
fn assert_replay_close(actual: &[u8], expected: &[u8], label: &str) -> u8 {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.abs_diff(b) <= REPLAY_TOLERANCE,
            "{label}: pixel ({}, {}), channel {}, got {a}, expected {b}",
            i as u32 / 4 % WIDTH,
            i as u32 / 4 / WIDTH,
            i % 4
        );
        worst = worst.max(a.abs_diff(b));
    }
    worst
}

#[expect(
    clippy::print_stderr,
    reason = "Record the replay difference that the opt-in GPU regression measured."
)]
fn check_cached_replay(backends: wgpu::Backends) {
    let mut replay = Replay::new(backends);
    let mut panels = panels();
    let scene = Scene::new(&panels);
    let (reference, _) = replay.frame(&scene, false);
    let painted = reference
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| {
            pixel
                .iter()
                .zip(BACKGROUND)
                .any(|(a, b)| a.abs_diff(b) > 16)
        })
        .count();
    assert!(painted > 1200, "the scene covers only {painted} pixels");

    let mut worst = 0;
    let mut states = Vec::new();
    for frame in 0..STABLE_FRAMES as usize + panels.len() {
        let (pixels, frame_states) = replay.frame(&scene, true);
        worst = worst.max(assert_replay_close(
            &pixels,
            &reference,
            &format!("frame {frame}, {frame_states:?}"),
        ));
        states = frame_states;
    }
    assert_eq!(states, [Reused; 4]);

    // A moved surface is shaded again; a stale texture would stay behind.
    panels[0].transform.translation.x += 3.;
    let scene = Scene::new(&panels);
    let (reference, _) = replay.frame(&scene, false);
    let (pixels, states) = replay.frame(&scene, true);
    assert_eq!(states, [Warming, Reused, Reused, Reused]);
    worst = worst.max(assert_replay_close(&pixels, &reference, "moved surface"));
    eprintln!("UI surface cache replay differs from its reference by at most {worst}/255");
}

#[test]
#[ignore = "Requires a hardware Vulkan adapter"]
fn gpu_cached_replay_matches_reference_vulkan() {
    check_cached_replay(wgpu::Backends::VULKAN);
}

#[test]
#[ignore = "Requires a hardware OpenGL adapter"]
fn gpu_cached_replay_matches_reference_gl() {
    check_cached_replay(wgpu::Backends::GL);
}
