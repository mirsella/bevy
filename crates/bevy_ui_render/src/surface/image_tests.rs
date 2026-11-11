//! Pixel comparisons use real adapters only, independently of the game and UI layout.
extern crate alloc;

use super::{shader_with_defs, Gpu};
use crate::{shader_flags, UiVertex, QUAD_VERTEX_POSITIONS};
use alloc::borrow::Cow;
use bevy_math::{Affine2, Mat4, Vec2};
use core::time::Duration;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const WIDTH: u32 = 80;
const HEIGHT: u32 = 48;
const ROW: u32 = 512;

fn pipeline(gpu: &Gpu, aa: bool, variant: u8) -> wgpu::RenderPipeline {
    // Every UI pipeline defines `MANUAL_SRGB`.
    let mut definitions = vec!["MANUAL_SRGB"];
    if variant != 0 {
        definitions.push("IMAGE_RECT");
    }
    if variant == 2 {
        definitions.push("IMAGE_PIXEL_ALIGNED");
    }
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI image equivalence"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(shader_with_defs(
                include_str!("../ui.wgsl"),
                aa,
                &definitions,
            ))),
        });
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("UI image equivalence"), layout: None,
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vertex"), compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<UiVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2,2=>Float32x4,3=>Uint32,
                    4=>Float32x4,5=>Float32x4,6=>Float32x2,7=>Float32x2],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fragment"), compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(), depth_stencil: None, multisample: Default::default(),
        multiview_mask: None, cache: None,
    })
}

fn render(
    gpu: &Gpu,
    pipeline: &wgpu::RenderPipeline,
    size: Vec2,
    phase: f32,
    rotated: bool,
    clipped: bool,
) -> Vec<u8> {
    let buffer = |data: &[u8], usage| {
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            })
    };
    let projection = Mat4::orthographic_rh(0., WIDTH as f32, HEIGHT as f32, 0., -1., 1.);
    let view = buffer(
        bytemuck::cast_slice(&projection.to_cols_array()),
        wgpu::BufferUsages::UNIFORM,
    );
    let view_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: view.as_entire_binding(),
        }],
    });
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nonuniform image with authored mip colours"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 3,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for mip in 0..3 {
        let side = 4 >> mip;
        let mut pixels = Vec::new();
        for y in 0..side {
            for x in 0..side {
                pixels.extend_from_slice(&[
                    40 + (x * 40) as u8,
                    20 + (y * 30) as u8,
                    180 - (mip * 60) as u8,
                    32 + ((x + y + mip) * 30) as u8,
                ]);
            }
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                mip_level: mip,
                ..texture.as_image_copy()
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
    let image_view = texture.create_view(&Default::default());
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let image_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let transform = Affine2::from_angle(if rotated { 0.21 } else { 0. });
    let vertices = QUAD_VERTEX_POSITIONS.map(|corner| {
        let point = corner * size;
        UiVertex {
            position: (transform.transform_point2(point) + Vec2::new(40. + phase, 24. + phase))
                .extend(0.)
                .into(),
            uv: (point / size + Vec2::splat(0.5)).into(),
            color: [0.45, 0.75, 0.2, 0.6],
            flags: shader_flags::TEXTURED,
            radius: [0.; 4],
            border: [0.; 4],
            size: size.into(),
            point: point.into(),
        }
    });
    let vertices = buffer(bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX);
    let indices = buffer(
        bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
        wgpu::BufferUsages::INDEX,
    );
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
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
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(ROW * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let target_view = target.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 0.8,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &view_group, &[]);
        pass.set_bind_group(1, &image_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
        if clipped {
            pass.set_scissor_rect(17, 11, 41, 26);
        }
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
    gpu.queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    gpu.device
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

#[expect(
    clippy::print_stderr,
    reason = "Report the measured image-error bound of the opt-in GPU regression."
)]
fn check(backend: wgpu::Backends) {
    let gpu = Gpu::new(backend);
    let mut maximum = 0;
    let mut differing_channels = 0;
    for aa in [false, true] {
        let reference = pipeline(&gpu, aa, 0);
        for variant in 1..=2 {
            let optimized = pipeline(&gpu, aa, variant);
            for size in [Vec2::new(64., 32.), Vec2::new(2., 2.), Vec2::new(1.5, 2.5)] {
                for phase in [0., 0.125, 0.5, 0.875] {
                    for rotated in [false, true] {
                        for clipped in [false, true] {
                            // Pixel-aligned pipelines are only queued with
                            // an exact physical-pixel edge proof.
                            if variant == 2
                                && (rotated || phase > 0.0 || size.fract() != Vec2::ZERO)
                            {
                                continue;
                            }
                            let expected = render(&gpu, &reference, size, phase, rotated, clipped);
                            let actual = render(&gpu, &optimized, size, phase, rotated, clipped);
                            let difference = actual
                                .iter()
                                .zip(&expected)
                                .map(|(&a, &b)| a.abs_diff(b))
                                .max()
                                .unwrap();
                            maximum = maximum.max(difference);
                            differing_channels +=
                                actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                            // Separate Vulkan programs on RADV can round an image
                            // sample/blend across an 8-bit boundary (observed 8 red
                            // channels of 15,360, max delta 1). No gamma, coverage
                            // or sampling change is allowed: bound every channel.
                            assert!(difference <= 1,"IMAGE_RECT delta={difference}, AA={aa}, size={size:?}, phase={phase}, rotated={rotated}, clipped={clipped}");
                        }
                    }
                }
            }
        }
    }
    eprintln!("IMAGE_RECT equivalence: maximum channel delta={maximum}/255; differing channels across fixtures={differing_channels}");
}

#[test]
#[ignore = "Requires a hardware Vulkan adapter"]
fn gpu_image_rect_matches_reference_vulkan() {
    check(wgpu::Backends::VULKAN);
}

#[test]
#[ignore = "Requires a hardware OpenGL adapter"]
fn gpu_image_rect_matches_reference_gl() {
    check(wgpu::Backends::GL);
}
