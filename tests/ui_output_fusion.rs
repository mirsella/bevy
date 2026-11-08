//! Hardware-rendered regression coverage for gamma-faithful UI output fusion.
//!
//! Requires a hardware Vulkan/GLES adapter. Run with:
//! `cargo test --no-default-features -F std,async_executor,multi_threaded,bevy_ui_render,bevy_sprite_render,bevy_core_pipeline,bevy_log,bevy_render/gles --test ui_output_fusion -- --include-ignored --test-threads=1`
use bevy::{
    app::{AppExit, PluginsState},
    camera::{
        visibility::RenderLayers, CameraMainTextureUsages, CameraOutputMode, Hdr, RenderTarget,
        Viewport,
    },
    core_pipeline::{
        schedule::{Core2d, Core2dSystems},
        tonemapping::Tonemapping,
    },
    diagnostic::{DiagnosticPath, DiagnosticsStore},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        diagnostic::{RecordDiagnostics, RenderDiagnosticsPlugin},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
            TextureFormat, TextureUsages,
        },
        renderer::{
            FlushCommands, RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph,
            RenderGraphSystems, ViewQuery,
        },
        settings::{Backends, RenderCreation, WgpuSettings},
        view::{
            window::screenshot::{Screenshot, ScreenshotCaptured},
            ExtractedView, ViewTarget,
        },
        Render, RenderApp, RenderPlugin, RenderSystems,
    },
    time::TimeUpdateStrategy,
    ui_render::{
        output_fusion::{
            UiOutputFusion, UiOutputFusionBarrier, UiOutputFusionPlugin, UiOutputFusionStack,
            UiOutputFusionStatus,
        },
        UiCameraView,
    },
    window::ExitCondition,
};
use std::time::{Duration, Instant};

// A real direct-linear phase makes mixed stacks exercise their ordering, rather
// than silently omitting UI when the gamma intermediate is absent.
fn direct_linear_ui(
    world: &World,
    view: ViewQuery<(&UiCameraView, &ViewTarget)>,
    views: Query<(&ExtractedView, &UiShaderVariants)>,
    phases: Res<bevy::render::render_phase::ViewSortedRenderPhases<bevy::ui_render::TransparentUi>>,
    mut context: RenderContext,
) {
    let (camera, target) = view.into_inner();
    let (view, variants) = views.get(camera.0).unwrap();
    if !variants.linear_color {
        return;
    }
    let phase = phases.get(&view.retained_view_entity).unwrap();
    assert!(target.main_texture_format().is_srgb());
    let mut pass = context.begin_tracked_render_pass(RenderPassDescriptor {
        color_attachments: &[Some(target.get_unsampled_color_attachment())],
        ..default()
    });
    phase.render(&mut pass, world, camera.0).unwrap();
}

#[derive(Resource, Default)]
struct Capture(Option<Image>);

#[derive(Resource, Default)]
struct Effect {
    after_ui: bool,
    clear: bool,
    enabled: bool,
    transparent: bool,
    base_only: bool,
}

fn effect(
    view: ViewQuery<(&ViewTarget, &ExtractedCamera)>,
    settings: Res<Effect>,
    mut context: RenderContext,
) {
    let (target, camera) = view.into_inner();
    if !settings.enabled || (settings.base_only && camera.order != 0) {
        return;
    }
    let post = target.post_process_write();
    if settings.clear {
        context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: post.destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(
                            LinearRgba::new(
                                0.03,
                                0.13,
                                0.23,
                                if settings.transparent { 0.35 } else { 1. },
                            )
                            .into(),
                        ),
                        store: StoreOp::Store,
                    },
                })],
                ..default()
            });
    } else {
        context.command_encoder().copy_texture_to_texture(
            post.source_texture.as_image_copy(),
            post.destination_texture.as_image_copy(),
            post.source_texture.size(),
        );
    }
}

#[derive(Resource)]
struct TestScale(f32);

#[derive(Resource, Default)]
struct MainFormat(Option<TextureFormat>);

fn scale_targets(
    scale: Res<TestScale>,
    format: Res<MainFormat>,
    mut cameras: Query<(Entity, &mut ExtractedCamera, &UiCameraView)>,
    mut views: Query<&mut ExtractedView>,
) {
    for (entity, mut camera, ui) in &mut cameras {
        if let Some(format) = format.0 {
            for entity in [entity, ui.0] {
                views.get_mut(entity).unwrap().target_format = format;
            }
        }
        if scale.0 == 1. {
            continue;
        }
        let scaled = (camera.physical_target_size.unwrap().as_vec2() * scale.0)
            .round()
            .as_uvec2();
        camera.physical_target_size = Some(scaled);
        camera.physical_viewport_size = Some(scaled);
        for entity in [entity, ui.0] {
            let mut view = views.get_mut(entity).unwrap();
            view.viewport.z = scaled.x;
            view.viewport.w = scaled.y;
        }
    }
}

fn equivalent(
    app: &mut App,
    cameras: &[Entity],
    output: &Handle<Image>,
    active: usize,
    label: &str,
    tolerance: u8,
) {
    eprintln!("fusion comparison: {label}, expected {active} active views");
    let reference = capture(app, cameras, output, false, 0);
    let fused = capture(app, cameras, output, true, active);
    let delta = reference
        .as_raw()
        .iter()
        .zip(fused.as_raw())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    if let Ok(dir) = std::env::var("UI_FUSION_OUTPUT") {
        std::fs::create_dir_all(&dir).unwrap();
        reference
            .save(format!("{dir}/{label}-reference.png"))
            .unwrap();
        fused.save(format!("{dir}/{label}-fused.png")).unwrap();
    }
    assert_eq!(reference.dimensions(), fused.dimensions());
    assert!(delta <= tolerance, "{label}: max channel delta {delta}");
}

fn frame(app: &mut App) {
    app.update();
    assert!(app.world().resource::<Messages<AppExit>>().is_empty());
    app.sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .poll(bevy::render::render_resource::PollType::wait_indefinitely())
        .unwrap();
}

fn check_frame_timing(app: &App) {
    let device = app.sub_app(RenderApp).world().resource::<RenderDevice>();
    let store = app.world().resource::<DiagnosticsStore>();
    let timing = store
        .get(&DiagnosticPath::new("render/frame/elapsed_gpu"))
        .and_then(|d| d.measurement());
    if !cfg!(target_os = "macos")
        && device
            .features()
            .contains(bevy::render::settings::WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    {
        let timing = timing.expect("completed whole-frame GPU measurement");
        assert!(timing.value > 0. && timing.value.is_finite(), "{timing:?}");
        for span in store.iter().filter_map(|d| {
            d.path()
                .to_string()
                .ends_with("elapsed_gpu")
                .then(|| d.measurement())
                .flatten()
        }) {
            if span.time == timing.time {
                assert!(
                    span.value <= timing.value + 0.001,
                    "inner span {} exceeds frame {}",
                    span.value,
                    timing.value
                );
            }
        }
    } else {
        assert!(
            timing.is_none(),
            "unsupported timestamps must not fabricate frame timing"
        );
    }
}

fn saturate_diagnostics(mut context: RenderContext) {
    let recorder = context
        .diagnostic_recorder()
        .expect("diagnostics plugin installed");
    for _ in 0..150 {
        recorder
            .time_span(context.command_encoder(), "query_pool_stress")
            .end(context.command_encoder());
    }
}

fn capture(
    app: &mut App,
    cameras: &[Entity],
    image: &Handle<Image>,
    fused: bool,
    active: usize,
) -> image::RgbaImage {
    for camera in cameras {
        if fused {
            app.world_mut().entity_mut(*camera).insert(UiOutputFusion);
        } else {
            app.world_mut()
                .entity_mut(*camera)
                .remove::<UiOutputFusion>();
        }
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    for index in 0.. {
        frame(app);
        let status = app
            .sub_app(RenderApp)
            .world()
            .resource::<UiOutputFusionStatus>();
        if index >= 16 && status.active_views == active {
            check_frame_timing(app);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "fusion readiness: {} requested, {} active, {:?}",
            status.requested_views,
            status.active_views,
            status.fallback
        );
    }
    app.world_mut().resource_mut::<Capture>().0 = None;
    app.world_mut()
        .spawn(Screenshot::image(image.clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<Capture>| {
                capture.0 = Some(event.image.clone());
            },
        );
    for _ in 0..120 {
        frame(app);
        if let Some(image) = app.world_mut().resource_mut::<Capture>().0.take() {
            return image.try_into_dynamic().unwrap().to_rgba8();
        }
    }
    panic!("no completed screenshot");
}

fn compare(backend: Backends) {
    let mut app = App::new();
    let plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                backends: Some(backend),
                ..default()
            })),
            ..default()
        })
        .disable::<PipelinedRenderingPlugin>()
        .disable::<bevy::log::LogPlugin>();
    #[cfg(feature = "bevy_winit")]
    let plugins = plugins.disable::<bevy::winit::WinitPlugin>();
    #[cfg(feature = "bevy_audio")]
    let plugins = plugins.disable::<bevy::audio::AudioPlugin>();
    app.add_plugins(plugins)
        .add_plugins(UiOutputFusionPlugin)
        .add_plugins(RenderDiagnosticsPlugin)
        .init_resource::<DiagnosticsStore>()
        .init_resource::<Capture>()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(Instant::now() < deadline);
    }
    app.finish();
    app.cleanup();
    app.sub_app_mut(RenderApp)
        .init_resource::<Effect>()
        .init_resource::<MainFormat>()
        .insert_resource(TestScale(1.))
        .add_systems(
            Core2d,
            direct_linear_ui
                .after(bevy::ui_render::ui_pass)
                .before(bevy::ui_render::srgb_ui_composite_pass),
        )
        .add_systems(Render, scale_targets.in_set(RenderSystems::CreateViews))
        .add_systems(
            Core2d,
            effect
                .run_if(|s: Res<Effect>| !s.after_ui)
                .after(Core2dSystems::PostProcess)
                .before(bevy::ui_render::ui_pass),
        )
        .add_systems(
            Core2d,
            effect
                .run_if(|s: Res<Effect>| s.after_ui)
                .after(bevy::ui_render::srgb_ui_composite_pass)
                .before(bevy::core_pipeline::upscaling::upscaling),
        );
    let adapter = app
        .sub_app(RenderApp)
        .world()
        .resource::<RenderAdapterInfo>();
    println!(
        "output fusion GPU: {} {:?} {}",
        adapter.name, adapter.backend, adapter.driver_info
    );
    let name = adapter.name.to_lowercase();
    assert!(!["llvmpipe", "lavapipe", "swiftshader", "software"]
        .iter()
        .any(|software| name.contains(software)));
    let output = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(Image::new_target_texture(
            321,
            639,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
    let cameras: Vec<_> = (0..3)
        .map(|order| {
            app.world_mut()
                .spawn((
                    Camera2d,
                    Camera {
                        order,
                        output_mode: if order == 2 {
                            CameraOutputMode::default()
                        } else {
                            CameraOutputMode::Skip
                        },
                        clear_color: if order == 0 {
                            ClearColorConfig::Custom(Color::srgb(0.04, 0.12, 0.23))
                        } else {
                            ClearColorConfig::None
                        },
                        ..default()
                    },
                    RenderTarget::Image(output.clone().into()),
                    Msaa::Off,
                    UiOutputFusionStack,
                    CameraMainTextureUsages(
                        CameraMainTextureUsages::default().0
                            | TextureUsages::COPY_SRC
                            | TextureUsages::COPY_DST,
                    ),
                    Tonemapping::None,
                    RenderLayers::layer(order as usize),
                ))
                .id()
        })
        .collect();
    app.world_mut().spawn((
        Sprite::from_color(Color::srgb(0.3, 0.6, 0.2), Vec2::new(220., 410.)),
        Transform::from_xyz(31., 67., 0.),
    ));
    let mut panels = Vec::new();
    for (layer, camera) in cameras.iter().take(2).enumerate() {
        for n in 0..8 {
            panels.push(
                app.world_mut()
                    .spawn((
                        UiTargetCamera(*camera),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(13.25 + n as f32 * 27.),
                            top: px(21.5 + n as f32 * 63. + layer as f32 * 11.),
                            width: px(110.),
                            height: px(108.),
                            border: UiRect::all(px(2.5)),
                            border_radius: BorderRadius::all(px(13.)),
                            ..default()
                        },
                        BorderColor::all(Color::srgba(0.9, 0.3, 0.1, 0.65)),
                        BackgroundColor(Color::srgba(0.23 + n as f32 * 0.08, 0.08, 0.63, 0.4)),
                    ))
                    .id(),
            );
        }
    }
    for format in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb] {
        *app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut(&output)
            .unwrap() = Image::new_target_texture(321, 639, format, None);
        for (case, alpha) in [0., 0.05, 0.25, 0.5, 0.9, 1.].into_iter().enumerate() {
            for entity in &panels {
                app.world_mut()
                    .get_mut::<BackgroundColor>(*entity)
                    .unwrap()
                    .0
                    .set_alpha(alpha);
            }
            equivalent(
                &mut app,
                &cameras,
                &output,
                3,
                &format!("{backend:?}-{format:?}-{case}"),
                2,
            );
        }
    }
    // Earlier UI stays in main when a later camera draws world geometry. Only
    // the terminal UI composite is replaced, including with partial opt-in.
    app.world_mut().spawn((
        UiTargetCamera(cameras[2]),
        Node {
            position_type: PositionType::Absolute,
            left: px(47.),
            top: px(91.),
            width: px(137.),
            height: px(219.),
            ..default()
        },
        ImageNode::solid_color(Color::srgba(0.11, 0.8, 0.43, 0.47)),
    ));
    // Pre-UI effects can make the scene transparent despite an opaque clear.
    app.world_mut()
        .entity_mut(cameras[0])
        .remove::<UiOutputFusionStack>();
    *app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>() = Effect {
        enabled: true,
        clear: true,
        transparent: true,
        base_only: true,
        ..default()
    };
    equivalent(
        &mut app,
        &cameras,
        &output,
        1,
        "transparent-pre-ui-effect",
        3,
    );
    *app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>() = Effect::default();
    app.world_mut()
        .entity_mut(cameras[0])
        .insert(UiOutputFusionStack);
    app.world_mut()
        .entity_mut(cameras[1])
        .insert(UiShaderVariants {
            linear_color: true,
            ..UiShaderVariants::FULL
        });
    let visible = capture(&mut app, &cameras, &output, false, 0);
    for panel in &panels[8..] {
        app.world_mut()
            .entity_mut(*panel)
            .insert(Visibility::Hidden);
    }
    let hidden = capture(&mut app, &cameras, &output, false, 0);
    assert_ne!(
        visible.as_raw(),
        hidden.as_raw(),
        "direct-linear UI must actually draw"
    );
    for panel in &panels[8..] {
        app.world_mut()
            .entity_mut(*panel)
            .insert(Visibility::Inherited);
    }
    equivalent(&mut app, &cameras, &output, 1, "mixed-direct-linear-ui", 3);
    app.world_mut()
        .entity_mut(cameras[1])
        .remove::<UiShaderVariants>();
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .insert(UiRetainedLayer { cache: true });
    }
    equivalent(&mut app, &cameras, &output, 1, "retained-terminal-ui", 3);
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .remove::<UiRetainedLayer>();
    }
    app.world_mut().spawn((
        Sprite::from_color(Color::srgb(0.8, 0.2, 0.5), Vec2::splat(140.)),
        Transform::default(),
        RenderLayers::layer(1),
    ));
    equivalent(&mut app, &cameras, &output, 1, "world-overlay", 3);
    capture(&mut app, &cameras, &output, false, 0);
    equivalent(&mut app, &cameras[2..], &output, 1, "partial-opt-in", 3);
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .remove::<UiOutputFusionStack>();
    }
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>()
        .enabled = true;
    equivalent(&mut app, &cameras, &output, 1, "postprocess-swap", 3);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>()
        .clear = true;
    equivalent(&mut app, &cameras, &output, 1, "pre-ui-effect", 3);
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .insert(UiOutputFusionBarrier);
    }
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>()
        .after_ui = true;
    equivalent(&mut app, &cameras, &output, 0, "post-ui-barrier", 0);
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .remove::<UiOutputFusionBarrier>();
    }
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<Effect>()
        .enabled = false;
    app.world_mut()
        .entity_mut(cameras[0])
        .insert(UiOutputFusionBarrier);
    equivalent(
        &mut app,
        &cameras,
        &output,
        0,
        "earlier-external-reader-barrier",
        0,
    );
    app.world_mut()
        .entity_mut(cameras[0])
        .remove::<UiOutputFusionBarrier>();
    app.world_mut()
        .get_mut::<Camera>(cameras[0])
        .unwrap()
        .clear_color = ClearColorConfig::Custom(Color::srgba(0.04, 0.12, 0.23, 0.35));
    app.world_mut()
        .get_mut::<Camera>(cameras[2])
        .unwrap()
        .output_mode = CameraOutputMode::Write {
        blend_state: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
        clear_color: ClearColorConfig::Custom(Color::srgba(0.2, 0.1, 0.4, 0.3)),
    };
    equivalent(
        &mut app,
        &cameras,
        &output,
        1,
        "transparent-output-blend",
        3,
    );
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<MainFormat>()
        .0 = Some(TextureFormat::Rgb10a2Unorm);
    equivalent(&mut app, &cameras, &output, 1, "rgb10a2-storage-blend", 3);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<MainFormat>()
        .0 = None;
    app.world_mut()
        .get_mut::<Camera>(cameras[0])
        .unwrap()
        .clear_color = ClearColorConfig::Custom(Color::srgb(0.04, 0.12, 0.23));
    app.world_mut()
        .get_mut::<Camera>(cameras[2])
        .unwrap()
        .output_mode = CameraOutputMode::default();
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .insert((Hdr, Tonemapping::Reinhard));
    }
    equivalent(&mut app, &cameras, &output, 1, "hdr-tonemapping", 3);
    for camera in &cameras {
        app.world_mut().entity_mut(*camera).insert(Msaa::Sample4);
    }
    equivalent(&mut app, &cameras, &output, 1, "hdr-msaa", 3);
    for camera in &cameras {
        app.world_mut()
            .entity_mut(*camera)
            .remove::<Hdr>()
            .insert((Msaa::Off, Tonemapping::None));
    }
    for camera in &cameras {
        app.world_mut().get_mut::<Camera>(*camera).unwrap().viewport = Some(Viewport {
            physical_position: UVec2::new(17, 23),
            physical_size: UVec2::new(287, 589),
            ..default()
        });
    }
    equivalent(&mut app, &cameras, &output, 1, "offset-viewport", 3);
    for camera in &cameras {
        app.world_mut().get_mut::<Camera>(*camera).unwrap().viewport = None;
    }
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<TestScale>()
        .0 = 0.61;
    equivalent(&mut app, &cameras, &output, 1, "odd-resolution-scale", 3);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<TestScale>()
        .0 = 1.;
    app.world_mut()
        .entity_mut(cameras[2])
        .remove::<Camera2d>()
        .insert((
            Camera3d::default(),
            bevy::render::camera::CameraRenderGraph::new(bevy::core_pipeline::schedule::Core3d),
        ));
    assert_eq!(
        app.world()
            .get::<bevy::render::camera::CameraRenderGraph>(cameras[2])
            .unwrap()
            .0,
        bevy::ecs::schedule::ScheduleLabel::intern(&bevy::core_pipeline::schedule::Core3d),
    );
    equivalent(&mut app, &cameras, &output, 1, "core3d-terminal", 3);
    // Changing attachment dimensions must rebuild cached bindings and preserve
    // the scene/UI alignment without requiring fusion to be reinstalled.
    *app.world_mut()
        .resource_mut::<Assets<Image>>()
        .get_mut(&output)
        .unwrap() = Image::new_target_texture(355, 685, TextureFormat::Rgba8UnormSrgb, None);
    equivalent(&mut app, &cameras, &output, 1, "resized-output", 3);
    // Separate MSAA storage with no scene clear preserves a different main pair;
    // it cannot safely inherit a preceding camera's deferred UI.
    app.world_mut().entity_mut(cameras[2]).insert(Msaa::Sample4);
    equivalent(&mut app, &cameras, &output, 0, "mixed-msaa-storage", 0);
    app.world_mut().entity_mut(cameras[2]).insert(Msaa::Off);
    for camera in &cameras[1..] {
        app.world_mut()
            .get_mut::<Camera>(*camera)
            .unwrap()
            .is_active = false;
    }
    app.world_mut()
        .get_mut::<Camera>(cameras[0])
        .unwrap()
        .output_mode = CameraOutputMode::default();
    equivalent(&mut app, &cameras, &output, 1, "single-camera", 3);
    for (space, format, label) in [
        (
            bevy::camera::CompositingSpace::Linear,
            TextureFormat::Rgba8Unorm,
            "unorm-storage",
        ),
        (
            bevy::camera::CompositingSpace::Srgb,
            TextureFormat::Rgba8Unorm,
            "srgb-compositing",
        ),
        (
            bevy::camera::CompositingSpace::Oklab,
            TextureFormat::Rgba16Float,
            "oklab-compositing",
        ),
    ] {
        app.world_mut().entity_mut(cameras[0]).insert((space, Hdr));
        app.sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<MainFormat>()
            .0 = Some(format);
        equivalent(&mut app, &cameras, &output, 1, label, 3);
    }
    // Exhaust detailed timestamp slots and flush an early command batch. The
    // reserved frame pair must still cover all subsequent camera commands.
    app.sub_app_mut(RenderApp).add_systems(
        RenderGraph,
        (saturate_diagnostics, |mut commands: FlushCommands| {
            commands.flush()
        })
            .chain()
            .in_set(RenderGraphSystems::Begin),
    );
    for _ in 0..8 {
        frame(&mut app);
    }
    check_frame_timing(&app);
    println!(
        "gamma UI stack, terminal-world, HDR/MSAA, viewport/scaling, Core3d and effect-order comparisons passed"
    );
}

#[test]
#[ignore = "requires a real Vulkan GPU"]
fn gpu_output_fusion_vulkan() {
    compare(Backends::VULKAN);
}

#[test]
#[ignore = "requires a real GLES GPU"]
fn gpu_output_fusion_gl() {
    compare(Backends::GL);
}
