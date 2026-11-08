use super::*;

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
    use bevy_color::Color;
    use bevy_ui::{ColorStop, ComputedUiRenderTargetInfo, LinearGradient};
    let snapshot = |base: usize, color| {
        let mut paint = vec![[0; 4]; base];
        let mut extractor = super::super::gradient::GradientExtractor::default();
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
        let header = super::super::append_paint(
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
    let pipeline = pipeline();
    for format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
        let descriptor = pipeline.specialize((format, 0));
        let target = descriptor.fragment.unwrap().targets[0].clone().unwrap();
        assert_eq!(target.format, format);
        assert_eq!(target.blend, Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING));
        assert_eq!(descriptor.multisample.count, 1);
        assert!(descriptor.depth_stencil.is_none());
        assert_eq!(descriptor.layout.len(), 2);
    }
    assert!(!supported_format(TextureFormat::Rgba16Float));
    assert!(!supported_format(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn replay_probe_is_never_a_default_and_preserves_color_space() {
    for variants in [
        UiShaderVariants::FULL,
        UiShaderVariants::SURFACES,
        UiShaderVariants::IMAGES,
        UiShaderVariants::ALL,
        UiShaderVariants::default(),
    ] {
        assert_eq!(variants.surface_cache_probe, 0);
        for probe in 0..=5 {
            for linear_color in [false, true] {
                let variants = UiShaderVariants {
                    surface_cache_probe: probe,
                    linear_color,
                    ..variants
                };
                assert_eq!(
                    variants.render_target_format(TextureFormat::Rgba8UnormSrgb),
                    if linear_color {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    }
                );
            }
        }
    }
}

#[test]
fn replay_probe_keys_are_distinct_and_only_change_fragment_definitions() {
    let pipeline = pipeline();
    let mut keys = HashSet::<(TextureFormat, u32)>::default();
    for format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
        let baseline = pipeline.specialize((format, 0));
        assert!(baseline.vertex.shader_defs.is_empty());
        assert!(baseline.fragment.as_ref().unwrap().shader_defs.is_empty());
        for probe in 0..=5 {
            assert!(keys.insert((format, probe)));
            let mut descriptor = pipeline.specialize((format, probe));
            let defines = &mut descriptor.fragment.as_mut().unwrap().shader_defs;
            assert_eq!(
                *defines,
                if probe == 0 {
                    vec![]
                } else {
                    vec![format!("SURFACE_CACHE_PROBE_{probe}").into()]
                }
            );
            defines.clear();
            // Includes vertex state, explicit source/replay layouts, blend and format.
            assert_eq!(descriptor, baseline);
        }
    }
    assert_eq!(keys.len(), 12);
}

#[test]
#[should_panic(expected = "surface cache probe must be in 0..=5")]
fn invalid_probe_is_rejected_before_ui_extraction() {
    UiShaderVariants {
        surface_cache_probe: 6,
        ..UiShaderVariants::default()
    }
    .render_target_format(TextureFormat::Rgba8UnormSrgb);
}

#[test]
#[should_panic(expected = "surface cache probe must be in 0..=5")]
fn invalid_probe_is_rejected_by_replay_specialization() {
    pipeline().specialize((TextureFormat::Rgba8Unorm, 6));
}

#[test]
fn replay_probe_stages_validate_and_translate_without_a_gpu() {
    use bevy_shader::ShaderDefVal;
    use naga::{back::glsl, valid, ShaderStage};
    use naga_oil::compose::{
        ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
    };

    let pipeline = pipeline();
    for probe in 0..=5 {
        let descriptor = pipeline.specialize((TextureFormat::Rgba8Unorm, probe));
        for (stage, entry_point, defines) in [
            (
                ShaderStage::Vertex,
                "vertex",
                &descriptor.vertex.shader_defs,
            ),
            (
                ShaderStage::Fragment,
                "fragment",
                &descriptor.fragment.as_ref().unwrap().shader_defs,
            ),
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
                    shader_defs: defines
                        .iter()
                        .map(|define| match define {
                            ShaderDefVal::Bool(name, value) => {
                                (name.clone(), ShaderDefValue::Bool(*value))
                            }
                            _ => panic!("cached replay probes require boolean definitions"),
                        })
                        .collect(),
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
            let loads = entry
                .function
                .expressions
                .iter()
                .filter(|(_, expression)| matches!(expression, naga::Expression::ImageLoad { .. }))
                .count();
            assert_eq!(
                loads,
                usize::from(stage == ShaderStage::Fragment && matches!(probe, 4 | 5)),
                "probe {probe}, {stage:?}"
            );
            // The source layout exposes View only to vertex shaders. No probe
            // may accidentally require it in the fragment stage.
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
            .expect("cached replay probes must translate to WebGL2 GLSL ES 3.00");
            naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options::default(),
                Some(&naga::back::spv::PipelineOptions {
                    shader_stage: stage,
                    entry_point: entry_point.into(),
                }),
            )
            .expect("cached replay probes must translate to Vulkan SPIR-V");
            naga::back::msl::write_string(
                &module,
                &info,
                &naga::back::msl::Options::default(),
                &naga::back::msl::PipelineOptions::default(),
            )
            .expect("cached replay probes must translate to Metal");
        }
    }
}

#[test]
fn camera_opt_in_is_copied_only_to_fresh_ui_view_and_can_be_removed() {
    use crate::{extract_ui_camera_view, render_pass::TransparentUi};
    use bevy_camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo};
    use bevy_ecs::system::RunSystemOnce;
    use bevy_render::{
        camera::CameraMainPassTextureFormats, render_phase::ViewSortedRenderPhases,
        sync_world::RenderEntity, MainWorld,
    };
    for linear_color in [false, true] {
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
                UiShaderVariants {
                    linear_color,
                    surface_cache_probe: 5,
                    ..UiShaderVariants::ALL
                },
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
            world.get::<UiShaderVariants>(second).unwrap().linear_color,
            linear_color
        );
        assert_eq!(
            world
                .get::<UiShaderVariants>(second)
                .unwrap()
                .surface_cache_probe,
            5
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
}
