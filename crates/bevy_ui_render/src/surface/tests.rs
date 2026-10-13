extern crate alloc;

use super::{append_paint, gradient::GradientExtractor, GpuSurface, SurfaceVertex};
use crate::QUAD_VERTEX_POSITIONS;
use alloc::borrow::Cow;
use bevy_color::{Color, LinearRgba};
use bevy_math::{ops, Mat4, Vec2};
use bevy_ui::{
    AngularColorStop, ColorStop, ComputedUiRenderTargetInfo, ConicGradient, Gradient,
    InterpolationColorSpace, LinearGradient, RadialGradient,
};
use core::time::Duration;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[test]
fn outer_color_has_a_defined_order_between_surface_and_outline() {
    use super::{extract_surfaces, ExtractedSurfaces};
    use crate::{
        extract_uinode_outer_colors, extract_uinode_outlines, ExtractedUiItem, ExtractedUiNodes,
        NodeType,
    };
    use bevy_app::{App, HierarchyPropagatePlugin, PostUpdate, Update};
    use bevy_camera::{visibility::InheritedVisibility, Camera};
    use bevy_ecs::{prelude::*, system::RunSystemOnce};
    use bevy_render::{sync_world::RenderEntity, MainWorld};
    use bevy_ui::{
        update::propagate_ui_target_cameras, BackgroundColor, ComputedNode, ComputedUiTargetCamera,
        Node, OuterColor, Outline, ResolvedBorderRadius, UiGlobalTransform, UiScale,
        UiTargetCamera,
    };

    let mut render_world = World::new();
    let mut app = App::new();
    app.init_resource::<UiScale>()
        .add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ))
        .add_systems(Update, propagate_ui_target_cameras);
    let camera = app
        .world_mut()
        .spawn((
            Camera::default(),
            RenderEntity::from(render_world.spawn_empty().id()),
        ))
        .id();
    app.world_mut().spawn((
        Node::default(),
        UiTargetCamera(camera),
        ComputedNode {
            size: Vec2::new(64., 32.),
            border_radius: ResolvedBorderRadius {
                top_left: 8.,
                top_right: 8.,
                bottom_right: 8.,
                bottom_left: 8.,
            },
            outline_width: 2.,
            ..Default::default()
        },
        UiGlobalTransform::default(),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::WHITE),
        OuterColor(Color::BLACK),
        Outline {
            color: Color::WHITE,
            ..Default::default()
        },
    ));
    app.update();
    let mut main_world = MainWorld::default();
    core::mem::swap(&mut *main_world, app.world_mut());
    render_world.insert_resource(main_world);
    render_world.init_resource::<ExtractedSurfaces>();
    render_world.init_resource::<ExtractedUiNodes>();
    render_world.run_system_once(extract_surfaces).unwrap();
    render_world
        .run_system_once(extract_uinode_outer_colors)
        .unwrap();
    render_world
        .run_system_once(extract_uinode_outlines)
        .unwrap();

    let surfaces = &render_world.resource::<ExtractedSurfaces>().nodes;
    let nodes = &render_world.resource::<ExtractedUiNodes>().uinodes;
    assert_eq!(surfaces.len(), 1);
    assert_eq!(nodes.len(), 2);
    let order = |kind| {
        nodes.iter().find(|node| matches!(&node.item, ExtractedUiItem::Node { node_type, .. } if *node_type == kind)).unwrap().z_order
    };
    assert!(surfaces[0].z_order < order(NodeType::Inverted));
    assert!(order(NodeType::Inverted) < order(NodeType::Outline));
}

fn shader(source: &str, antialias: bool) -> wgpu::naga::Module {
    let mut composer = Composer::default();
    for (source, path) in [
        (
            "#define_import_path bevy_render::view\nstruct View { clip_from_world: mat4x4<f32> }",
            "test_view.wgsl",
        ),
        (
            include_str!("../../../bevy_render/src/maths.wgsl"),
            "maths.wgsl",
        ),
        (
            include_str!("../../../bevy_render/src/color_operations.wgsl"),
            "color_operations.wgsl",
        ),
        (include_str!("../ui.wgsl"), "ui.wgsl"),
        (include_str!("../paint.wgsl"), "paint.wgsl"),
        (include_str!("../gradient.wgsl"), "gradient.wgsl"),
    ] {
        if let Err(error) = composer.add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: path,
            ..Default::default()
        }) {
            panic!("{}", error.emit_to_string(&composer));
        }
    }
    composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: "surface.wgsl",
            shader_defs: if antialias {
                [("ANTI_ALIAS".into(), ShaderDefValue::Bool(true))].into()
            } else {
                Default::default()
            },
            ..Default::default()
        })
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&composer)))
}

#[test]
fn ui_shaders_validate_with_and_without_antialiasing() {
    use wgpu::naga::{back::glsl, valid, ShaderStage};
    for (source, antialias) in [include_str!("../surface.wgsl"), include_str!("../ui.wgsl")]
        .into_iter()
        .flat_map(|source| [false, true].map(|aa| (source, aa)))
    {
        let module = shader(source, antialias);
        let info =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::empty())
                .validate(&module)
                .unwrap();
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        for (shader_stage, entry_point) in [
            (ShaderStage::Vertex, "vertex"),
            (ShaderStage::Fragment, "fragment"),
        ] {
            let pipeline = glsl::PipelineOptions {
                shader_stage,
                entry_point: entry_point.into(),
                multiview: None,
            };
            let mut output = String::new();
            glsl::Writer::new(
                &mut output,
                &module,
                &info,
                &options,
                &pipeline,
                Default::default(),
            )
            .unwrap()
            .write()
            .expect("UI surfaces must translate to WebGL2 GLSL ES 3.00");
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
}

impl Gpu {
    #[expect(
        clippy::print_stderr,
        reason = "Record which adapter ran the opt-in GPU regression."
    )]
    fn new(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            bevy_tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU regression requires an adapter");
        eprintln!("UI surface GPU regression: {:?}", adapter.get_info());
        let (device, queue) =
            bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            }))
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface regression"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(shader(
                include_str!("../surface.wgsl"),
                true,
            ))),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surface regression"), layout: None,
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vertex"), compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<SurfaceVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Uint32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fragment"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(),
            multiview_mask: None, cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
        }
    }

    fn render(&self, mut surface: GpuSurface, phase: f32, gradients: &[Gradient]) -> Vec<u8> {
        const WIDTH: u32 = 80;
        const HEIGHT: u32 = 48;
        const ROW: u32 = 512;
        let buffer = |bytes: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage,
                })
        };
        let projection = Mat4::orthographic_rh(0., WIDTH as f32, HEIGHT as f32, 0., -1., 1.);
        let view = buffer(
            bytemuck::cast_slice(&projection.to_cols_array()),
            wgpu::BufferUsages::UNIFORM,
        );
        // Exercise extraction, nonzero addresses and records crossing texture rows.
        let mut texels = vec![[0; 4]; 3];
        let mut extractor = GradientExtractor::default();
        for range in [
            &mut surface.background_gradients,
            &mut surface.border_gradients,
        ] {
            *range = extractor.extract(
                &gradients[range[0] as usize..(range[0] + range[1]) as usize],
                surface.size.into(),
                &ComputedUiRenderTargetInfo::default(),
                &mut texels,
            );
        }
        let address = append_paint(&mut texels, core::slice::from_ref(&surface));
        let paint_size = wgpu::Extent3d {
            width: 7,
            height: texels.len().div_ceil(7) as u32,
            depth_or_array_layers: 1,
        };
        texels.resize((paint_size.width * paint_size.height) as usize, [0; 4]);
        let paint = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface paint regression"),
            size: paint_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            paint.as_image_copy(),
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(paint_size.width * 16),
                rows_per_image: None,
            },
            paint_size,
        );
        let paint_view = paint.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&paint_view),
                },
            ],
        });
        let size = Vec2::from(surface.size);
        let vertices = QUAD_VERTEX_POSITIONS.map(|p| {
            let point = p * size;
            SurfaceVertex {
                position: (point + size * 0.5 + Vec2::splat(8. + phase))
                    .extend(0.)
                    .into(),
                point: point.into(),
                surface: address,
            }
        });
        let vertices = buffer(bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX);
        let indices = buffer(
            bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
            wgpu::BufferUsages::INDEX,
        );
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (ROW * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let target_view = target.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: None,
                },
            },
            target.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range();
        let pixels = mapped
            .as_chunks::<{ ROW as usize }>()
            .0
            .iter()
            .flat_map(|row| row[..WIDTH as usize * 4].iter().copied())
            .collect();
        drop(mapped);
        readback.unmap();
        pixels
    }
}

fn assert_pixels_close(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.abs_diff(b) <= 1,
            "{label}: pixel ({}, {}), channel {}, got {a}, expected {b}",
            i / 4 % 80,
            i / 4 / 80,
            i % 4
        );
    }
}

#[test]
#[ignore = "Requires a GPU adapter; run with --ignored --nocapture"]
fn gpu_surface_coverage_and_paint_vulkan() {
    let gpu = Gpu::new(wgpu::Backends::VULKAN);
    check_surface_coverage_and_paint(&gpu);
    check_gradient_geometry_and_hue(&gpu);
}

#[test]
#[ignore = "Requires an OpenGL adapter; run with --ignored --nocapture"]
fn gpu_surface_coverage_and_paint_gl() {
    let gpu = Gpu::new(wgpu::Backends::GL);
    check_surface_coverage_and_paint(&gpu);
    check_gradient_geometry_and_hue(&gpu);
}

fn check_gradient_geometry_and_hue(gpu: &Gpu) {
    use bevy_color::{Hsla, Hsva, Oklaba, Oklcha};
    use bevy_ui::{RadialGradientShape, Val};
    use core::f32::consts::{FRAC_PI_2, TAU};

    let surface = GpuSurface {
        size: [64., 32.],
        background_gradients: [0, 1],
        ..Default::default()
    };
    let pixel = |pixels: &[u8], x: usize, y: usize, color: Color, label: &str| {
        let color = color.to_linear();
        let expected = [
            ops::powf(color.red.max(0.), 1. / 2.2) * color.alpha,
            ops::powf(color.green.max(0.), 1. / 2.2) * color.alpha,
            ops::powf(color.blue.max(0.), 1. / 2.2) * color.alpha,
            color.alpha,
        ]
        .map(|v| (v.clamp(0., 1.) * 255.).round() as u8);
        assert_pixels_close(
            &pixels[(y * 80 + x) * 4..(y * 80 + x + 1) * 4],
            &expected,
            label,
        );
    };
    let red = Color::linear_rgb(1., 0., 0.);
    let blue = Color::linear_rgb(0., 0., 1.);
    let blend = |t: f32| Color::linear_rgb(1. - t, 0., t);

    // Distinct colors check the circle/ellipse geometry, including the area
    // outside the last stop. A white-to-white gradient cannot catch these errors.
    for (rx, ry) in [(12., 12.), (24., 8.)] {
        let pixels = gpu.render(
            surface,
            0.5,
            &[RadialGradient {
                shape: RadialGradientShape::Ellipse(Val::Px(rx), Val::Px(ry)),
                color_space: InterpolationColorSpace::LinearRgba,
                stops: vec![ColorStop::auto(red), ColorStop::auto(blue)],
                ..Default::default()
            }
            .into()],
        );
        for (x, y) in [(40, 24), (46, 24), (40, 28), (46, 28), (66, 24)] {
            let dx = (x as f32 - 40.) / rx;
            let dy = (y as f32 - 24.) / ry;
            pixel(
                &pixels,
                x,
                y,
                blend(ops::hypot(dx, dy).min(1.)),
                "radial geometry",
            );
        }
    }

    // Cardinal directions make the angular expectation independent of the
    // shader's atan2 expression, and exercise wrapping across the starting ray.
    for start in [0., FRAC_PI_2, -FRAC_PI_2, TAU + FRAC_PI_2] {
        let pixels = gpu.render(
            surface,
            0.5,
            &[ConicGradient {
                start,
                color_space: InterpolationColorSpace::LinearRgba,
                stops: vec![AngularColorStop::auto(red), AngularColorStop::auto(blue)],
                ..Default::default()
            }
            .into()],
        );
        for (x, y, turn) in [(40, 16, 0.), (48, 24, 0.25), (40, 32, 0.5), (32, 24, 0.75)] {
            pixel(
                &pixels,
                x,
                y,
                blend((turn - start / TAU).rem_euclid(1.)),
                "conic rotation",
            );
        }
    }

    // Black/white still carry authored saturation. Going through RGB before
    // interpolation would turn this blue ramp gray or wash out its dark half.
    for space in [InterpolationColorSpace::Hsla, InterpolationColorSpace::Hsva] {
        let color = |level| match space {
            InterpolationColorSpace::Hsla => Color::from(Hsla::new(240., 0.75, level, 1.)),
            _ => Color::from(Hsva::new(240., 0.75, level, 1.)),
        };
        let pixels = gpu.render(
            surface,
            0.5,
            &[LinearGradient {
                angle: LinearGradient::TO_RIGHT,
                color_space: space,
                stops: vec![ColorStop::auto(color(0.)), ColorStop::auto(color(1.))],
            }
            .into()],
        );
        for (x, level) in [(24, 0.25), (40, 0.5), (56, 0.75)] {
            pixel(
                &pixels,
                x,
                24,
                color(level),
                "authored saturation at black/white",
            );
        }
    }

    for space in [
        InterpolationColorSpace::Srgba,
        InterpolationColorSpace::Oklaba,
    ] {
        let color = |t: f32| match space {
            InterpolationColorSpace::Srgba => Color::srgba(0.2 + 0.6 * t, 0.4, 0.8 - 0.6 * t, 0.5),
            _ => Color::from(Oklaba::new(
                0.4 + 0.4 * t,
                0.1 - 0.2 * t,
                0.1 - 0.2 * t,
                0.5,
            )),
        };
        let pixels = gpu.render(
            surface,
            0.5,
            &[LinearGradient {
                angle: LinearGradient::TO_RIGHT,
                color_space: space,
                stops: vec![ColorStop::auto(color(0.)), ColorStop::auto(color(1.))],
            }
            .into()],
        );
        for (x, t) in [(24, 0.25), (40, 0.5), (56, 0.75)] {
            pixel(&pixels, x, 24, color(t), "color-space interpolation");
        }
    }

    for space in [
        InterpolationColorSpace::Hsla,
        InterpolationColorSpace::HslaLong,
        InterpolationColorSpace::Hsva,
        InterpolationColorSpace::HsvaLong,
        InterpolationColorSpace::Oklcha,
        InterpolationColorSpace::OklchaLong,
    ] {
        let color = |hue, alpha| match space {
            InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
                Color::from(Hsla::new(hue, 0.7, 0.5, alpha))
            }
            InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
                Color::from(Hsva::new(hue, 0.7, 0.8, alpha))
            }
            _ => Color::from(Oklcha::new(0.65, 0.1, hue, alpha)),
        };
        let pixels = gpu.render(
            surface,
            0.5,
            &[LinearGradient {
                angle: LinearGradient::TO_RIGHT,
                color_space: space,
                stops: vec![
                    ColorStop::auto(color(350., 0.25)),
                    ColorStop::auto(color(10., 0.75)),
                ],
            }
            .into()],
        );
        let long = matches!(
            space,
            InterpolationColorSpace::HslaLong
                | InterpolationColorSpace::HsvaLong
                | InterpolationColorSpace::OklchaLong
        );
        for (x, t) in [(24, 0.25), (40, 0.5), (56, 0.75)] {
            let hue = (350_f32 + if long { -340. } else { 20. } * t).rem_euclid(360.);
            pixel(
                &pixels,
                x,
                24,
                color(hue, 0.25 + 0.5 * t),
                "hue wrapping and alpha",
            );
        }
    }
}

fn check_surface_coverage_and_paint(gpu: &Gpu) {
    for phase in [0., 0.25, 0.65] {
        for opacity in [1., 0.5, 0.25] {
            let surface = GpuSurface {
                size: [64., 32.],
                radius: [8.; 4],
                background: [1., 1., 1., opacity],
                border_colors: [[1., 1., 1., opacity]; 4],
                ..Default::default()
            };
            let expected = gpu.render(surface, phase, &[]);
            for width in [0.2, 1.5, 4.25] {
                let actual = gpu.render(
                    GpuSurface {
                        border: [width; 4],
                        ..surface
                    },
                    phase,
                    &[],
                );
                assert_pixels_close(
                    &actual,
                    &expected,
                    &format!("phase={phase}, opacity={opacity}, border={width}"),
                );
            }
        }
    }

    // Both regions have a 50%-opaque base plus a 50%-opaque gradient layer.
    // Their shared edge must retain the resulting 75% opacity, including curves.
    let surface = GpuSurface {
        size: [64., 32.],
        radius: [8.; 4],
        border: [1.5; 4],
        background: [1., 1., 1., 0.5],
        border_colors: [[1., 1., 1., 0.5]; 4],
        background_gradients: [0, 1],
        border_gradients: [0, 1],
        ..Default::default()
    };
    let expected = gpu.render(
        GpuSurface {
            background: [1., 1., 1., 0.75],
            border_colors: [[1., 1., 1., 0.75]; 4],
            background_gradients: [0; 2],
            border_gradients: [0; 2],
            ..surface
        },
        0.25,
        &[],
    );
    for kind in 0..3 {
        for space in [
            InterpolationColorSpace::LinearRgba,
            InterpolationColorSpace::Srgba,
            InterpolationColorSpace::Oklaba,
            InterpolationColorSpace::Oklcha,
            InterpolationColorSpace::OklchaLong,
            InterpolationColorSpace::Hsla,
            InterpolationColorSpace::HslaLong,
            InterpolationColorSpace::Hsva,
            InterpolationColorSpace::HsvaLong,
        ] {
            let color = Color::LinearRgba(LinearRgba::new(1., 1., 1., 0.5));
            let stops = vec![ColorStop::auto(color), ColorStop::auto(color)];
            let gradient = match kind {
                0 => Gradient::Linear(LinearGradient {
                    stops,
                    color_space: space,
                    ..Default::default()
                }),
                1 => Gradient::Radial(RadialGradient {
                    stops,
                    color_space: space,
                    ..Default::default()
                }),
                _ => Gradient::Conic(ConicGradient {
                    stops: vec![AngularColorStop::auto(color), AngularColorStop::auto(color)],
                    color_space: space,
                    ..Default::default()
                }),
            };
            assert_pixels_close(
                &gpu.render(surface, 0.25, &[gradient]),
                &expected,
                "layered gradient opacity",
            );
        }
    }

    // A fully transparent border must stay transparent, not acquire the fill's color.
    let pixels = gpu.render(
        GpuSurface {
            border: [4.; 4],
            background: [1.; 4],
            border_colors: [[0.; 4]; 4],
            background_gradients: [0; 2],
            border_gradients: [0; 2],
            ..surface
        },
        0.,
        &[],
    );
    assert_eq!(&pixels[(24 * 80 + 9) * 4..(24 * 80 + 10) * 4], &[0; 4]);
    assert_eq!(&pixels[(24 * 80 + 40) * 4..(24 * 80 + 41) * 4], &[255; 4]);

    let plain = GpuSurface {
        size: [64., 32.],
        background_gradients: [0, 1],
        ..Default::default()
    };
    let red = ColorStop::percent(Color::srgb(1., 0., 0.), 0.);
    let blue = ColorStop::percent(Color::srgb(0., 0., 1.), 100.);
    let gradient = LinearGradient {
        angle: LinearGradient::TO_RIGHT,
        color_space: InterpolationColorSpace::LinearRgba,
        stops: vec![red, blue],
    };
    let pixels = gpu.render(plain, 0., &[gradient.clone().into()]);
    for stops in [
        vec![ColorStop::auto(red.color), ColorStop::auto(blue.color)],
        vec![blue, ColorStop::auto(Color::linear_rgb(0.5, 0., 0.5)), red],
    ] {
        assert_pixels_close(
            &gpu.render(
                plain,
                0.,
                &[LinearGradient {
                    stops,
                    ..gradient.clone()
                }
                .into()],
            ),
            &pixels,
            "implicit and unsorted gradient stops",
        );
    }
    for x in [16, 40, 64] {
        let t = (x as f32 + 0.5 - 8.) / 64.;
        let expected = [
            (ops::powf(1. - t, 1. / 2.2) * 255.).round() as u8,
            0,
            (ops::powf(t, 1. / 2.2) * 255.).round() as u8,
            255,
        ];
        assert_pixels_close(
            &pixels[(24 * 80 + x) * 4..(24 * 80 + x + 1) * 4],
            &expected,
            "linear gradient interpolation",
        );
    }
    let hard_stops = vec![
        red,
        ColorStop::percent(red.color, 50.),
        ColorStop::percent(blue.color, 50.),
        blue,
    ];
    let pixels = gpu.render(
        plain,
        0.,
        &[LinearGradient {
            stops: hard_stops,
            ..gradient
        }
        .into()],
    );
    assert_eq!(
        &pixels[(24 * 80 + 39) * 4..(24 * 80 + 40) * 4],
        &[255, 0, 0, 255]
    );
    assert_eq!(
        &pixels[(24 * 80 + 40) * 4..(24 * 80 + 41) * 4],
        &[0, 0, 255, 255]
    );

    // Skipped layers must not leave stale addresses or alter source-over ordering.
    let layered = GpuSurface {
        background: [0.; 4],
        border_colors: [[0.; 4]; 4],
        background_gradients: [0, 4],
        border_gradients: [0, 4],
        ..surface
    };
    let layers = [
        LinearGradient {
            stops: vec![],
            ..Default::default()
        }
        .into(),
        LinearGradient {
            stops: vec![ColorStop::auto(Color::linear_rgba(1., 0., 0., 0.5))],
            ..Default::default()
        }
        .into(),
        LinearGradient {
            stops: vec![ColorStop::auto(Color::NONE)],
            ..Default::default()
        }
        .into(),
        ConicGradient {
            stops: vec![AngularColorStop::auto(Color::linear_rgba(0., 0., 1., 0.5))],
            ..Default::default()
        }
        .into(),
    ];
    let expected = gpu.render(
        GpuSurface {
            background: [1. / 3., 0., 2. / 3., 0.75],
            border_colors: [[1. / 3., 0., 2. / 3., 0.75]; 4],
            background_gradients: [0; 2],
            border_gradients: [0; 2],
            ..layered
        },
        0.25,
        &[],
    );
    assert_pixels_close(
        &gpu.render(layered, 0.25, &layers),
        &expected,
        "ordered gradient layers",
    );

    let pixels = gpu.render(
        GpuSurface {
            size: [64., 32.],
            border: [2., 3., 4., 5.],
            border_colors: [
                [1., 0., 0., 1.],
                [0., 1., 0., 1.],
                [0., 0., 1., 1.],
                [1.; 4],
            ],
            ..Default::default()
        },
        0.,
        &[],
    );
    for (x, y, color) in [
        (9, 24, [255, 0, 0, 255]),
        (40, 9, [0, 255, 0, 255]),
        (70, 24, [0, 0, 255, 255]),
        (40, 37, [255; 4]),
    ] {
        assert_eq!(&pixels[(y * 80 + x) * 4..(y * 80 + x + 1) * 4], &color);
    }
}
