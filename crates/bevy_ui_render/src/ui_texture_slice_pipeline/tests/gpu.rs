//! Real-hardware offscreen comparisons. The reference slicer has no analytic
//! AA or MSAA specialization: both `UiAntiAlias` modes use these same pipelines.
#![expect(
    unused_qualifications,
    reason = "separate raw wgpu test resources from production wrappers"
)]
extern crate alloc;

use self::alloc::borrow::Cow;
use super::*;
use bevy_math::Mat4;
use core::time::Duration;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;
const ROW: u32 = WIDTH * 4;

fn shader(source: &str) -> naga::Module {
    let mut composer = Composer::default();
    for (source, file_path) in [
        (
            "#define_import_path bevy_render::view\nstruct View { clip_from_world: mat4x4<f32> }",
            "view",
        ),
        (
            include_str!("../../../../bevy_render/src/globals.wgsl"),
            "globals",
        ),
    ] {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path,
                ..default()
            })
            .unwrap();
    }
    composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: "slice",
            shader_defs: [("MANUAL_SRGB".into(), ShaderDefValue::Bool(true))].into(),
            ..default()
        })
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&composer)))
}

#[test]
fn compact_slice_shader_validates() {
    let module = shader(include_str!("../../ui_texture_slice_cpu.wgsl"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(
        module
            .entry_points
            .iter()
            .find(|e| e.name == "fragment")
            .unwrap()
            .function
            .arguments
            .len(),
        1
    );
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    view_layout: wgpu::BindGroupLayout,
    image_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl Gpu {
    #[expect(
        clippy::print_stderr,
        reason = "GPU tests report the actual hardware adapter"
    )]
    fn new(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            bevy_tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("hardware adapter required");
        let info = adapter.get_info();
        eprintln!("CPU slices GPU regression: {info:?}");
        assert_ne!(
            info.device_type,
            wgpu::DeviceType::Cpu,
            "software rendering is not verification"
        );
        assert!(!["llvmpipe", "lavapipe", "swiftshader"]
            .iter()
            .any(|s| info.name.to_lowercase().contains(s)));
        let (device, queue) =
            bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..default()
            }))
            .unwrap();
        // Like production, FULL and CPU use the same explicit layouts, not
        // pipeline-exclusive auto layouts. Exercise dynamic view offsets too.
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&view_layout), Some(&image_layout)],
            immediate_size: 0,
        });
        Self {
            device,
            queue,
            view_layout,
            image_layout,
            pipeline_layout,
        }
    }

    fn pipeline(&self, cpu: bool) -> wgpu::RenderPipeline {
        let source = if cpu {
            include_str!("../../ui_texture_slice_cpu.wgsl")
        } else {
            include_str!("../../ui_texture_slice.wgsl")
        };
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("slices"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(shader(source))),
            });
        let full_attributes = wgpu::vertex_attr_array![0=>Float32x3, 1=>Float32x2, 2=>Float32x4, 3=>Float32x4, 4=>Float32x4, 5=>Float32x4, 6=>Float32x4];
        let cpu_attributes = wgpu::vertex_attr_array![0=>Float32x3, 1=>Float32x2, 2=>Float32x4];
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("slice comparison"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vertex"),
                    compilation_options: default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: if cpu {
                            size_of::<CpuSliceVertex>()
                        } else {
                            size_of::<UiTextureSliceVertex>()
                        } as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: if cpu {
                            &cpu_attributes
                        } else {
                            &full_attributes
                        },
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fragment"),
                    compilation_options: default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: default(),
                depth_stencil: None,
                multisample: default(),
                multiview_mask: None,
                cache: None,
            })
    }

    fn render(
        &self,
        node: &ExtractedUiTextureSlice,
        pipeline: &wgpu::RenderPipeline,
        cpu: bool,
    ) -> Vec<u8> {
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
        let view_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.view_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view.as_entire_binding(),
            }],
        });
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nonuniform sliced texture"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pixels: Vec<u8> = (0..64)
            .flat_map(|y| {
                (0..64).flat_map(move |x| {
                    [
                        (20 + (x * 113 + y * 37) % 215) as u8,
                        (20 + (y * 97 + x * 17) % 215) as u8,
                        220,
                        (60 + (x * 31 + y * 23) % 195) as u8,
                    ]
                })
            })
            .collect();
        self.queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: None,
            },
            texture.size(),
        );
        let texture_view = texture.create_view(&default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..default()
        });
        // Auto-layout groups are pipeline-exclusive. Each comparison creates its
        // own groups; production FULL/CPU pipelines share explicit descriptors.
        let image_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.image_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let mut indices = Vec::new();
        let mut cpu_vertices = Vec::new();
        let mut full_vertices = Vec::new();
        let mut emit = |vertices: &[(Vec2, Vec2)]| {
            let base = if cpu {
                cpu_vertices.len()
            } else {
                full_vertices.len()
            } as u32;
            for &(position, uv) in vertices {
                if cpu {
                    cpu_vertices.push(CpuSliceVertex {
                        position: position.extend(0.).into(),
                        uv: uv.into(),
                        color: node.color.to_f32_array(),
                    });
                } else {
                    let ([slices, border, repeat], atlas) =
                        slice_parameters(node, Vec2::splat(64.));
                    full_vertices.push(UiTextureSliceVertex {
                        position: position.extend(0.).into(),
                        uv: uv.into(),
                        color: node.color.to_f32_array(),
                        slices,
                        border,
                        repeat,
                        atlas,
                    });
                }
            }
            for i in 1..vertices.len().saturating_sub(1) as u32 {
                indices.extend([base, base + i, base + i + 1]);
            }
        };
        if cpu {
            CpuSlices::new(node, Vec2::splat(64.))
                .unwrap()
                .for_each_polygon(node, &mut emit);
        } else {
            let corners = QUAD_VERTEX_POSITIONS.map(|p| {
                (
                    node.transform.transform_point2(p * node.rect.size()),
                    p + Vec2::splat(0.5),
                )
            });
            emit(&clip_polygon(node.clip.as_ref(), &corners, Vec2::lerp));
        }
        let vertices = buffer(
            if cpu {
                bytemuck::cast_slice(&cpu_vertices)
            } else {
                bytemuck::cast_slice(&full_vertices)
            },
            wgpu::BufferUsages::VERTEX,
        );
        let index_buffer = buffer(bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX);
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
        let target_view = target.create_view(&default());
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (ROW * HEIGHT) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.07,
                            g: 0.13,
                            b: 0.19,
                            a: 0.3,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &view_group, &[0]);
            pass.set_bind_group(1, &image_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
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
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            sender.send(result).unwrap();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let pixels = readback.get_mapped_range(..).to_vec();
        readback.unmap();
        pixels
    }
}

/// A node of the comparison matrix.
struct Case {
    node: ExtractedUiTextureSlice,
    size: Vec2,
    phase: f32,
    rotated: bool,
    atlas_flip_clip: bool,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for size in [
        Vec2::new(90., 64.),
        Vec2::new(36., 24.),
        Vec2::new(12., 10.),
    ] {
        for phase in [0., 0.125, 0.5] {
            for rotated in [false, true] {
                for atlas_flip_clip in [false, true] {
                    let mut node = node();
                    node.rect.max = size;
                    node.transform = Affine2::from_angle(if rotated { 0.23 } else { 0. });
                    node.transform.translation = Vec2::new(64. + phase, 48. + phase);
                    if atlas_flip_clip {
                        node.atlas_rect = Some(Rect::new(8., 12., 56., 52.));
                        node.flip_x = true;
                        node.flip_y = true;
                        node.clip = Some(CalculatedClip::Rects(
                            [CalculatedClipRect {
                                rect: Rect::new(-30., -20., 28., 18.),
                                world_to_clip_local: Affine2::from_angle(-0.13)
                                    * Affine2::from_translation(-Vec2::new(64., 48.)),
                            }]
                            .into_iter()
                            .collect(),
                        ));
                    }
                    cases.push(Case {
                        node,
                        size,
                        phase,
                        rotated,
                        atlas_flip_clip,
                    });
                }
            }
        }
    }
    cases
}

#[test]
fn comparison_matrix_reaches_minified_and_clipped_cpu_slices() {
    let cases = cases();
    let mut cpu_cases = 0;
    let mut minified = 0;
    let mut clipped = 0;
    for case in &cases {
        let texture_size = Vec2::splat(64.);
        assert!(CpuSlices::eligible(&case.node, texture_size, 7).is_none());
        let eligible = CpuSlices::eligible(&case.node, texture_size, 1).is_some();
        assert!(!(eligible && case.rotated));
        if eligible {
            cpu_cases += 1;
            minified += usize::from(case.size.x < 64.);
            clipped += usize::from(case.atlas_flip_clip);
        }
    }
    assert_eq!(cpu_cases, 9);
    assert_eq!(cases.len() - cpu_cases, 27);
    assert!(minified > 0 && clipped > 0);
}

#[expect(clippy::print_stderr, reason = "report measured image-error bounds")]
fn compare(backends: wgpu::Backends) {
    let gpu = Gpu::new(backends);
    let full = gpu.pipeline(false);
    let cpu = gpu.pipeline(true);
    let mut overall_max = 0;
    let mut total_error = 0u64;
    let mut samples = 0;
    for Case {
        node,
        size,
        phase,
        atlas_flip_clip,
        ..
    } in cases()
    {
        // Ineligible nodes keep the reference pipeline: there is nothing to compare.
        if CpuSlices::eligible(&node, Vec2::splat(64.), 1).is_none() {
            continue;
        }
        let a = gpu.render(&node, &full, false);
        let b = gpu.render(&node, &cpu, true);
        let max = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        overall_max = overall_max.max(max);
        assert!(
            max <= 1,
            "size={size} phase={phase} atlas={atlas_flip_clip}: max={max}"
        );
        if phase == 0.125 && size == Vec2::new(90., 64.) {
            capture(
                backends,
                &format!("atlas-flip-clip-{atlas_flip_clip}"),
                &a,
                &b,
            );
        }
        total_error += a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| a.abs_diff(b) as u64)
            .sum::<u64>();
        samples += a.len();
    }
    assert!(samples > 0);
    eprintln!(
        "max={overall_max} mean={}",
        total_error as f64 / samples as f64
    );

    // Exercise the newly eligible empty-border axes against the actual reference
    // shader, including its zero-width UV branches and pixel-center seams.
    let mut zero_border_cpu_cases = 0;
    for (min_inset, max_inset) in [
        (Vec2::ZERO, Vec2::ZERO),
        (Vec2::new(6., 0.), Vec2::new(10., 0.)),
        (Vec2::new(0., 4.), Vec2::new(0., 8.)),
    ] {
        for size in [Vec2::new(90., 64.), Vec2::new(36., 24.)] {
            for phase in [0., 0.125, 0.5] {
                for flipped_atlas in [false, true] {
                    let mut node = node();
                    node.rect.max = size;
                    node.transform.translation += Vec2::splat(phase);
                    let SpriteImageMode::Sliced(slicer) = &mut node.image_scale_mode else {
                        unreachable!()
                    };
                    slicer.border = BorderRect {
                        min_inset,
                        max_inset,
                    };
                    if flipped_atlas {
                        node.atlas_rect = Some(Rect::new(8., 12., 56., 52.));
                        node.flip_x = true;
                        node.flip_y = true;
                    }
                    if CpuSlices::eligible(&node, Vec2::splat(64.), 1).is_none() {
                        continue;
                    }
                    zero_border_cpu_cases += 1;
                    let a = gpu.render(&node, &full, false);
                    let b = gpu.render(&node, &cpu, true);
                    let max = a
                        .iter()
                        .zip(&b)
                        .map(|(&a, &b)| a.abs_diff(b))
                        .max()
                        .unwrap();
                    assert!(max <= 1, "empty-border size={size} phase={phase} flipped_atlas={flipped_atlas} border={min_inset}/{max_inset}: max={max}");
                }
            }
        }
    }
    assert!(zero_border_cpu_cases > 0);
    eprintln!("actual empty-border CPU cases={zero_border_cpu_cases}");
}

fn capture(backend: wgpu::Backends, name: &str, original: &[u8], actual: &[u8]) {
    let Some(directory) = std::env::var_os("BEVY_CPU_SLICES_CAPTURE_DIR") else {
        return;
    };
    let directory = std::path::Path::new(&directory).join(format!("{backend:?}"));
    std::fs::create_dir_all(&directory).unwrap();
    let difference: Vec<_> = original
        .iter()
        .zip(actual)
        .map(|(&a, &b)| a.abs_diff(b).saturating_mul(32))
        .collect();
    for (suffix, pixels) in [
        ("original", original),
        ("cpu", actual),
        ("difference-x32", &difference),
    ] {
        let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        for pixel in pixels.as_chunks::<4>().0 {
            ppm.extend_from_slice(&pixel[..3]);
        }
        std::fs::write(directory.join(format!("{name}_{suffix}.ppm")), ppm).unwrap();
    }
}

#[test]
#[ignore = "requires a hardware Vulkan adapter"]
fn gpu_cpu_slices_match_reference_vulkan() {
    compare(wgpu::Backends::VULKAN);
}

#[test]
#[ignore = "requires a hardware GL adapter"]
fn gpu_cpu_slices_match_reference_gl() {
    compare(wgpu::Backends::GL);
}
