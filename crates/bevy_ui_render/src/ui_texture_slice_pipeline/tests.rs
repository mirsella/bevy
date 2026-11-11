use super::*;
use bevy_math::Mat2;
use bevy_sprite::BorderRect;
use bevy_ui::CalculatedClipRect;

mod gpu;

fn node() -> ExtractedUiTextureSlice {
    ExtractedUiTextureSlice {
        stack_index: 0,
        transform: Affine2::from_translation(Vec2::new(64., 48.)),
        rect: Rect::from_corners(Vec2::ZERO, Vec2::new(90., 64.)),
        atlas_rect: None,
        image: AssetId::default(),
        clip: None,
        extracted_camera_entity: Entity::PLACEHOLDER,
        color: LinearRgba::new(0.45, 0.75, 0.2, 0.6),
        image_scale_mode: SpriteImageMode::Sliced(TextureSlicer {
            border: BorderRect {
                min_inset: Vec2::new(6., 4.),
                max_inset: Vec2::new(10., 8.),
            },
            center_scale_mode: SliceScaleMode::Stretch,
            sides_scale_mode: SliceScaleMode::Stretch,
            max_corner_scale: 1.,
        }),
        flip_x: false,
        flip_y: false,
        inverse_scale_factor: 1.,
        main_entity: Entity::PLACEHOLDER.into(),
        render_entity: Entity::PLACEHOLDER,
        cpu_slices: None,
        cpu_slice_sampler: true,
    }
}

#[test]
fn eligibility_rejects_tiling_even_when_repeat_is_one() {
    let mut node = node();
    let size = Vec2::splat(64.);
    assert!(CpuSlices::new(&node, size).is_some());
    for center in [false, true] {
        let mut n = self::node();
        let SpriteImageMode::Sliced(slicer) = &mut n.image_scale_mode else {
            unreachable!()
        };
        if center {
            slicer.center_scale_mode = SliceScaleMode::Tile { stretch_value: 1. };
        } else {
            slicer.sides_scale_mode = SliceScaleMode::Tile { stretch_value: 1. };
        }
        assert!(CpuSlices::new(&n, size).is_none());
    }
    node.image_scale_mode = SpriteImageMode::Tiled {
        tile_x: false,
        tile_y: false,
        stretch_value: 1.,
    };
    assert_eq!(slice_parameters(&node, size).0[2], [1.; 4]);
    assert!(CpuSlices::new(&node, size).is_none());
}

#[test]
fn sampling_and_seam_eligibility_preserve_reference_discontinuities() {
    let size = Vec2::splat(64.);
    assert!(CpuSlices::eligible(&node(), size, 1).is_some());
    assert!(CpuSlices::eligible(&node(), size, 7).is_none());
    for translation in [
        Vec2::new(64.5, 48.),
        Vec2::new(64., 48.5),
        Vec2::new(64.499, 48.),
    ] {
        let mut n = node();
        n.transform.translation = translation;
        assert!(CpuSlices::eligible(&n, size, 1).is_none());
    }
    let mut n = node();
    n.transform.translation += Vec2::splat(0.125);
    assert!(CpuSlices::eligible(&n, size, 1).is_some());
    n.transform.matrix2 = Mat2::from_angle(0.23);
    assert!(CpuSlices::eligible(&n, size, 1).is_none());
    n.transform.matrix2 = Mat2::IDENTITY;
    n.cpu_slice_sampler = false;
    assert!(CpuSlices::eligible(&n, size, 1).is_none());
    assert!(cpu_slice_sampler(&ImageSamplerDescriptor::linear()));
    assert!(!cpu_slice_sampler(&ImageSamplerDescriptor::nearest()));
    assert!(!cpu_slice_sampler(&ImageSamplerDescriptor {
        anisotropy_clamp: 4,
        ..ImageSamplerDescriptor::linear()
    }));
    assert!(!cpu_slice_sampler(&ImageSamplerDescriptor {
        min_filter: ImageFilterMode::Nearest,
        ..ImageSamplerDescriptor::linear()
    }));
    let mut n = node();
    n.transform.matrix2 = Mat2::from_diagonal(Vec2::new(-1., 1.));
    assert!(CpuSlices::eligible(&n, size, 1).is_some());
    n.transform.matrix2 = Mat2::from_diagonal(Vec2::splat(0.1));
    assert!(CpuSlices::eligible(&n, size, 1).is_none());
    n = node();
    n.clip = Some(CalculatedClip::Rects(
        [CalculatedClipRect {
            rect: Rect {
                min: Vec2::splat(f32::NAN),
                max: Vec2::ONE,
            },
            world_to_clip_local: Affine2::IDENTITY,
        }]
        .into_iter()
        .collect(),
    ));
    assert!(CpuSlices::eligible(&n, size, 1).is_none());
}

#[test]
fn eligibility_rejects_invalid_degenerate_and_out_of_range_geometry() {
    let valid = node();
    let size = Vec2::splat(64.);
    for transform in [
        Affine2::from_scale(Vec2::ZERO),
        Affine2::from_scale(Vec2::new(1e-8, 1.)),
        Affine2::from_translation(Vec2::splat(f32::NAN)),
        Affine2::from_translation(Vec2::splat(20_000.)),
    ] {
        let mut n = node();
        n.transform = transform;
        assert!(CpuSlices::new(&n, size).is_none());
    }
    for size in [
        Vec2::ZERO,
        Vec2::new(-1., 2.),
        Vec2::splat(f32::NAN),
        Vec2::splat(f32::INFINITY),
    ] {
        assert!(CpuSlices::new(&valid, size).is_none());
        let mut n = node();
        n.rect.max = size;
        assert!(CpuSlices::new(&n, Vec2::splat(64.)).is_none());
    }
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        let mut n = node();
        n.inverse_scale_factor = scale;
        assert!(CpuSlices::new(&n, size).is_none());
    }
    for border in [
        BorderRect::all(32.),
        BorderRect::all(-1.),
        BorderRect::all(f32::NAN),
    ] {
        let mut n = node();
        let SpriteImageMode::Sliced(slicer) = &mut n.image_scale_mode else {
            unreachable!()
        };
        slicer.border = border;
        assert!(CpuSlices::new(&n, size).is_none());
    }
    for atlas in [
        Rect::new(-1., 0., 16., 16.),
        Rect::new(0., 0., 65., 16.),
        Rect::new(0., 0., 0., 16.),
        Rect {
            min: Vec2::ZERO,
            max: Vec2::splat(f32::NAN),
        },
    ] {
        let mut n = node();
        n.atlas_rect = Some(atlas);
        assert!(CpuSlices::new(&n, size).is_none());
    }
}

#[test]
fn zero_border_axes_emit_only_nonempty_patches() {
    for (border, count) in [
        (BorderRect::ZERO, 1),
        (
            BorderRect {
                min_inset: Vec2::new(6., 0.),
                max_inset: Vec2::new(10., 0.),
            },
            3,
        ),
        (
            BorderRect {
                min_inset: Vec2::new(0., 4.),
                max_inset: Vec2::new(0., 8.),
            },
            3,
        ),
    ] {
        let mut node = node();
        let SpriteImageMode::Sliced(slicer) = &mut node.image_scale_mode else {
            unreachable!()
        };
        slicer.border = border;
        let geometry = CpuSlices::new(&node, Vec2::splat(64.)).unwrap();
        let mut emitted = 0;
        let mut area = 0.;
        geometry.for_each_polygon(&node, |polygon| {
            emitted += 1;
            let patch_area = (polygon[1].0 - polygon[0].0)
                .perp_dot(polygon[3].0 - polygon[0].0)
                .abs();
            assert!(patch_area > 0.);
            area += patch_area;
        });
        assert_eq!(emitted, count);
        assert!((area - node.rect.size().element_product()).abs() < 0.01);
        assert!(CpuSlices::eligible(&node, Vec2::splat(64.), 1).is_some());
    }
}

#[test]
fn patches_preserve_affine_geometry_area_atlas_and_flips() {
    for flip_x in [false, true] {
        for flip_y in [false, true] {
            let mut node = node();
            node.flip_x = flip_x;
            node.flip_y = flip_y;
            node.atlas_rect = Some(Rect::new(8., 12., 56., 52.));
            node.transform.matrix2 =
                Mat2::from_angle(0.23) * Mat2::from_diagonal(Vec2::new(-0.9, 1.1));
            let cpu = CpuSlices::new(&node, Vec2::splat(64.)).unwrap();
            let mut area = 0.;
            let mut count = 0;
            cpu.for_each_polygon(&node, |polygon| {
                count += 1;
                assert_eq!(polygon.len(), 4);
                for &(point, uv) in polygon {
                    assert!(point.is_finite() && uv.is_finite());
                    assert!(uv.cmpge(Vec2::new(8., 12.) / 64.).all());
                    assert!(uv.cmple(Vec2::new(56., 52.) / 64.).all());
                }
                area += (polygon[1].0 - polygon[0].0)
                    .perp_dot(polygon[3].0 - polygon[0].0)
                    .abs();
            });
            assert_eq!(count, 9);
            assert!(
                (area
                    - node.rect.size().element_product()
                        * node.transform.matrix2.determinant().abs())
                .abs()
                    < 0.01
            );
            assert_eq!(cpu.u[0], if flip_x { 56. / 64. } else { 8. / 64. });
            assert_eq!(cpu.v[0], if flip_y { 52. / 64. } else { 12. / 64. });
        }
    }
}

#[test]
fn clipping_interpolates_affine_patch_uvs_and_fully_clipped_emits_nothing() {
    let mut node = node();
    node.transform.matrix2 = Mat2::from_angle(0.23);
    node.clip = Some(CalculatedClip::Rects(
        [CalculatedClipRect {
            rect: Rect::new(-30., -20., 28., 18.),
            world_to_clip_local: Affine2::from_translation(-Vec2::new(64., 48.)),
        }]
        .into_iter()
        .collect(),
    ));
    let cpu = CpuSlices::new(&node, Vec2::splat(64.)).unwrap();
    let ([slices, border, _], _) = slice_parameters(&node, Vec2::splat(64.));
    let map = |p: f32, lo: f32, hi: f32, tl: f32, th: f32| {
        if p < lo {
            p / lo * tl
        } else if p > hi {
            th + (p - hi) / (1. - hi) * (1. - th)
        } else {
            tl + (p - lo) / (hi - lo) * (th - tl)
        }
    };
    let mut count = 0;
    cpu.for_each_polygon(&node, |polygon| {
        count += 1;
        for &(position, uv) in polygon {
            let p = node.transform.inverse().transform_point2(position) / node.rect.size()
                + Vec2::splat(0.5);
            let expected = Vec2::new(
                map(p.x, border[0], border[2], slices[0], slices[2]),
                map(p.y, border[1], border[3], slices[1], slices[3]),
            );
            assert!((uv - expected).abs().max_element() < 1e-5);
        }
    });
    assert!(count > 0);
    node.clip = Some(CalculatedClip::FullyClipped);
    cpu.for_each_polygon(&node, |_| panic!("fully clipped patch was emitted"));
}

#[test]
fn batch_boundaries_isolate_buffers_images_target_formats_and_foreign_items() {
    use bevy_render::render_phase::DrawFunctionId;
    let full = CachedRenderPipelineId::new(0);
    let cpu = CachedRenderPipelineId::new(1);
    let linear_cpu = CachedRenderPipelineId::new(2);
    let mut images = Assets::<Image>::default();
    let image = images.add(Image::default()).id();
    let other = images.add(Image::default()).id();
    let ids = [cpu, cpu, full, cpu, linear_cpu, linear_cpu, cpu, cpu];
    let mut items: SortedPhaseItems<_> = ids
        .into_iter()
        .enumerate()
        .map(|(i, pipeline)| {
            let entity = Entity::from_raw_u32(i as u32).unwrap();
            (
                (entity, entity.into()),
                TransparentUi {
                    sort_key: FloatOrd(i as f32),
                    entity: (entity, entity.into()),
                    pipeline,
                    draw_function: DrawFunctionId(0),
                    batch_range: 0..0,
                    extra_index: PhaseItemExtraIndex::None,
                    index: i,
                    indexed: true,
                },
            )
        })
        .collect();
    let mut batches = Vec::new();
    let mut first = None;
    for (i, &id) in ids.iter().enumerate() {
        if i == 7 {
            first = None;
        } // foreign/skipped item or view boundary
        UiTextureSlicerBatch::append(
            &mut items,
            &mut first,
            &mut batches,
            i,
            if i == 5 { other } else { image },
            i as u32 * 6..(i + 1) as u32 * 6,
            id != full,
        );
    }
    assert_eq!(batches.len(), 7);
    assert_eq!(batches[0].1.range, 0..12);
    assert_eq!(items[0].batch_range, 0..2);
    assert_eq!(
        batches.iter().map(|b| b.1.cpu_slices).collect::<Vec<_>>(),
        [true, false, true, true, true, true, true]
    );
    // Empty/clipped phase items also occupy a slot in a merged batch's span.
    items[2].pipeline = cpu;
    batches.clear();
    first = None;
    UiTextureSlicerBatch::append(&mut items, &mut first, &mut batches, 0, image, 0..6, true);
    UiTextureSlicerBatch::append(&mut items, &mut first, &mut batches, 2, image, 6..12, true);
    assert_eq!(batches.len(), 1);
    assert_eq!(items[0].batch_range, 0..3);
}

#[test]
fn pipeline_keys_share_explicit_bindings_but_isolate_vertex_formats() {
    let mut shaders = Assets::<Shader>::default();
    let pipeline = UiTextureSlicePipeline {
        view_layout: BindGroupLayoutDescriptor::new("view", &[]),
        image_layout: BindGroupLayoutDescriptor::new("image", &[]),
        shader: shaders.add(Shader::from_wgsl("", "full")),
        cpu_shader: shaders.add(Shader::from_wgsl("", "cpu")),
    };
    for cpu_slices in [false, true] {
        let descriptor = pipeline.specialize(UiTextureSlicePipelineKey {
            target_format: TextureFormat::Rgba8Unorm,
            cpu_slices,
        });
        assert_eq!(
            descriptor.layout,
            [pipeline.view_layout.clone(), pipeline.image_layout.clone()]
        );
        assert_eq!(
            descriptor.vertex.shader,
            if cpu_slices {
                pipeline.cpu_shader.clone()
            } else {
                pipeline.shader.clone()
            }
        );
        assert_eq!(
            descriptor.vertex.buffers[0].array_stride,
            if cpu_slices {
                size_of::<CpuSliceVertex>()
            } else {
                size_of::<UiTextureSliceVertex>()
            } as u64
        );
        let fragment = descriptor.fragment.unwrap();
        assert_eq!(descriptor.vertex.shader, fragment.shader);
        assert_eq!(
            fragment.targets[0].as_ref().unwrap().format,
            TextureFormat::Rgba8Unorm
        );
        assert!(fragment
            .shader_defs
            .contains(&MANUAL_SRGB_SHADER_DEF.into()));
    }
}
