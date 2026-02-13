//! UI rendering benchmark focused on overflow clipping.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    window::{PresentMode, WindowResolution},
    winit::WinitSettings,
};

#[path = "stress_tests/frame_bench_util.rs"]
mod frame_bench_util;

fn main() {
    let args = Args::from_env();

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            present_mode: PresentMode::AutoNoVsync,
            resolution: WindowResolution::new(1920, 1080).with_scale_factor_override(1.0),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(WinitSettings::continuous())
    .insert_resource(args.clone())
    .add_systems(Startup, setup);

    if let Some(config) = args.bench {
        app.add_plugins(frame_bench_util::FrameBenchPlugin { config });
    } else {
        app.add_plugins((
            bevy::diagnostic::FrameTimeDiagnosticsPlugin::default(),
            bevy::diagnostic::LogDiagnosticsPlugin::default(),
        ));
    }

    app.run();
}

#[derive(Resource, Clone)]
struct Args {
    containers_x: u32,
    containers_y: u32,
    images_per_container: u32,
    texture_count: u32,
    clip: bool,
    image_scale: f32,
    bench: Option<frame_bench_util::FrameBenchConfig>,
}

impl Args {
    fn from_env() -> Self {
        let mut containers_x = 40;
        let mut containers_y = 20;
        let mut images_per_container = 1;
        let mut texture_count = 1;
        let mut clip = true;
        let mut image_scale: f32 = 1.5;

        let mut bench = false;
        let mut warmup_frames = 120;
        let mut sample_frames = 300;
        let mut print_each_frame = false;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--containers-x" => {
                    containers_x = it
                        .next()
                        .expect("--containers-x requires a value")
                        .parse()
                        .expect("--containers-x must be a u32");
                }
                "--containers-y" => {
                    containers_y = it
                        .next()
                        .expect("--containers-y requires a value")
                        .parse()
                        .expect("--containers-y must be a u32");
                }
                "--images-per-container" => {
                    images_per_container = it
                        .next()
                        .expect("--images-per-container requires a value")
                        .parse()
                        .expect("--images-per-container must be a u32");
                }
                "--texture-count" => {
                    texture_count = it
                        .next()
                        .expect("--texture-count requires a value")
                        .parse()
                        .expect("--texture-count must be a u32");
                }
                "--no-clip" => clip = false,
                "--image-scale" => {
                    image_scale = it
                        .next()
                        .expect("--image-scale requires a value")
                        .parse()
                        .expect("--image-scale must be a f32");
                }
                "--bench" => bench = true,
                "--bench-warmup" => {
                    warmup_frames = it
                        .next()
                        .expect("--bench-warmup requires a value")
                        .parse()
                        .expect("--bench-warmup must be a u32");
                }
                "--bench-frames" => {
                    sample_frames = it
                        .next()
                        .expect("--bench-frames requires a value")
                        .parse()
                        .expect("--bench-frames must be a u32");
                }
                "--bench-print-each-frame" => print_each_frame = true,
                _ => {}
            }
        }

        let bench = bench.then_some(frame_bench_util::FrameBenchConfig {
            warmup_frames,
            sample_frames,
            print_each_frame,
        });

        Self {
            containers_x: containers_x.max(1),
            containers_y: containers_y.max(1),
            images_per_container: images_per_container.max(1),
            texture_count: texture_count.max(1),
            clip,
            image_scale,
            bench,
        }
    }
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, args: Res<Args>) {
    commands.spawn(Camera2d);

    // Allocate textures in-memory so the benchmark doesn't depend on the AssetServer.
    let handles: Vec<Handle<Image>> = (0..args.texture_count)
        .map(|i| {
            let r = (i & 0xFF) as u8;
            let g = ((i >> 8) & 0xFF) as u8;
            let b = ((i >> 16) & 0xFF) as u8;

            images.add(Image::new_fill(
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[r, g, b, 255],
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            ))
        })
        .collect();

    commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            ..default()
        })
        .with_children(|root| {
            // Layout containers as an absolute-positioned grid.
            let screen_w = 1920.0;
            let screen_h = 1080.0;
            let cell_w = screen_w / args.containers_x as f32;
            let cell_h = screen_h / args.containers_y as f32;

            let image_w = cell_w * args.image_scale;
            let image_h = cell_h * args.image_scale;
            let base_left = (cell_w - image_w) * 0.5;
            let base_top = (cell_h - image_h) * 0.5;

            let overflow = if args.clip {
                Overflow::clip()
            } else {
                Overflow::visible()
            };

            let mut image_index: u32 = 0;
            for y in 0..args.containers_y {
                for x in 0..args.containers_x {
                    let left = x as f32 * cell_w;
                    let top = y as f32 * cell_h;

                    root.spawn(Node {
                        position_type: PositionType::Absolute,
                        left: px(left),
                        top: px(top),
                        width: px(cell_w),
                        height: px(cell_h),
                        overflow,
                        ..default()
                    })
                    .with_children(|container| {
                        for i in 0..args.images_per_container {
                            let handle =
                                handles[(image_index % args.texture_count) as usize].clone();
                            image_index = image_index.wrapping_add(1);

                            // Keep quads small so this is mostly CPU/submission bound.
                            let offset = (i % 4) as f32;
                            container.spawn((
                                ImageNode::new(handle),
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(base_left + offset),
                                    top: px(base_top + offset),
                                    width: px(image_w),
                                    height: px(image_h),
                                    ..default()
                                },
                            ));
                        }
                    });
                }
            }
        });

    if let Some(bench) = &args.bench {
        println!(
            "FRAME_BENCH_CONFIG warmup_frames={} sample_frames={} containers={}x{} images_per_container={} texture_count={} clip={} image_scale={}",
            bench.warmup_frames,
            bench.sample_frames,
            args.containers_x,
            args.containers_y,
            args.images_per_container,
            args.texture_count,
            args.clip,
            args.image_scale,
        );
    }
}
