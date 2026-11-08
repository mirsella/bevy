extern crate alloc;

use super::{append_paint, gradient::GradientExtractor, GpuSurface, SurfaceVertex};
use crate::QUAD_VERTEX_POSITIONS;
use alloc::borrow::Cow;
use bevy_color::{Color, ColorToComponents, LinearRgba};
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

#[path = "image_tests.rs"]
mod image_tests;

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

#[test]
fn surface_constants_follow_each_target_camera_and_do_not_reuse_the_previous_mode() {
    use super::{extract_surfaces, gamma_color, ExtractedSurfaces};
    use crate::UiShaderVariants;
    use bevy_app::{App, HierarchyPropagatePlugin, PostUpdate, Update};
    use bevy_camera::{visibility::InheritedVisibility, Camera};
    use bevy_ecs::{prelude::*, system::RunSystemOnce};
    use bevy_render::{sync_world::RenderEntity, MainWorld};
    use bevy_sprite::BorderRect;
    use bevy_ui::{
        update::propagate_ui_target_cameras, BackgroundColor, BorderColor, ComputedNode,
        ComputedUiTargetCamera, Node, UiGlobalTransform, UiScale, UiTargetCamera,
    };

    let mut render_world = World::new();
    let mut app = App::new();
    app.init_resource::<UiScale>()
        .add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ))
        .add_systems(Update, propagate_ui_target_cameras);
    let background = LinearRgba::new(0.125, 0.25, 0.5, 0.375);
    let border = LinearRgba::new(0.0625, 0.125, 0.25, 0.625);
    let mut cameras = Vec::new();
    for linear_color in [false, true] {
        let render_camera = render_world.spawn_empty().id();
        let camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                RenderEntity::from(render_camera),
                UiShaderVariants {
                    linear_color,
                    ..UiShaderVariants::ALL
                },
            ))
            .id();
        cameras.push((camera, render_camera));
        app.world_mut().spawn((
            Node::default(),
            UiTargetCamera(camera),
            ComputedNode {
                size: Vec2::new(64., 32.),
                border: BorderRect::all(2.),
                ..Default::default()
            },
            UiGlobalTransform::default(),
            InheritedVisibility::VISIBLE,
            BackgroundColor(background.into()),
            BorderColor::all(border),
        ));
    }
    app.update();
    let mut main_world = MainWorld::default();
    core::mem::swap(&mut *main_world, app.world_mut());
    render_world.insert_resource(main_world);
    render_world.init_resource::<ExtractedSurfaces>();
    for swap_modes in [false, true] {
        if swap_modes {
            let mut main = render_world.resource_mut::<MainWorld>();
            for (camera, _) in &cameras {
                let mut variants = main.get_mut::<UiShaderVariants>(*camera).unwrap();
                variants.linear_color = !variants.linear_color;
            }
            *render_world.resource_mut::<ExtractedSurfaces>() = ExtractedSurfaces::default();
        }
        render_world.run_system_once(extract_surfaces).unwrap();
        let extracted = render_world.resource::<ExtractedSurfaces>();
        assert_eq!(extracted.nodes.len(), 2);
        for (index, (_, render_camera)) in cameras.iter().enumerate() {
            let linear = (index == 1) != swap_modes;
            let surface = extracted
                .nodes
                .iter()
                .find(|node| node.camera == *render_camera)
                .unwrap();
            let count = size_of::<GpuSurface>() / 16;
            let data: GpuSurface = bytemuck::pod_read_unaligned(bytemuck::cast_slice(
                &extracted.texels[surface.paint as usize..surface.paint as usize + count],
            ));
            assert_eq!(
                data.background,
                if linear {
                    [0.125, 0.25, 0.5, 0.375]
                } else {
                    gamma_color(background)
                }
            );
            assert_eq!(
                data.border_colors,
                [if linear {
                    [0.0625, 0.125, 0.25, 0.625]
                } else {
                    gamma_color(border)
                }; 4]
            );
            assert_eq!(data.border, [2.; 4]);
        }
    }
}

#[test]
fn surface_pipeline_format_keys_isolate_linear_paint_including_full_fallback() {
    use super::{SurfacePipeline, SurfacePipelineKey, SurfaceShader};
    use crate::pipeline::UiPipelineKey;
    use bevy_render::render_resource::{
        BindGroupLayoutDescriptor, BlendState, SpecializedRenderPipeline, TextureFormat,
    };

    let mut shaders = bevy_asset::Assets::<bevy_shader::Shader>::default();
    let pipeline = SurfacePipeline {
        layout: BindGroupLayoutDescriptor::new("test_surface", &[]),
        shader: shaders.add(bevy_shader::Shader::from_wgsl("", "surface.wgsl")),
        fill_shader: shaders.add(bevy_shader::Shader::from_wgsl("", "surface_fill.wgsl")),
    };
    let mut keys = bevy_platform::collections::HashSet::<SurfacePipelineKey>::default();
    for target_format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
        for shader in [
            SurfaceShader::Full,
            SurfaceShader::Solid,
            SurfaceShader::RectFill,
            SurfaceShader::RoundedFill,
        ] {
            let key = SurfacePipelineKey {
                shader,
                ui: UiPipelineKey {
                    target_format,
                    basic_image: false,
                    encoded_image: false,
                    relaxed_precision: false,
                    relaxed_colors: false,
                    anti_alias: true,
                    image_rect: false,
                    rect_math: true,
                    lean_io: true,
                    pixel_aligned: false,
                    implicit_aligned: false,
                },
            };
            assert!(keys.insert(key));
            let descriptor = pipeline.specialize(key);
            assert_eq!(
                descriptor.vertex.shader,
                if shader.is_fill() {
                    pipeline.fill_shader.clone()
                } else {
                    pipeline.shader.clone()
                },
                "lean_io must never select the fill-only shader for Solid"
            );
            let fragment = descriptor.fragment.unwrap();
            assert_eq!(descriptor.vertex.shader_defs, fragment.shader_defs);
            assert_eq!(descriptor.vertex.shader, fragment.shader);
            assert_eq!(
                fragment.shader_defs.contains(&"CHEAP_RECT_DISTANCE".into()),
                shader == SurfaceShader::RectFill
            );
            assert_eq!(
                fragment.shader_defs.contains(&"SURFACE_SOLID".into()),
                shader == SurfaceShader::Solid
            );
            assert_eq!(
                fragment.shader_defs.contains(&"LINEAR_COLOR".into()),
                target_format.is_srgb()
            );
            let target = fragment.targets[0].as_ref().unwrap();
            assert_eq!(target.format, target_format);
            assert_eq!(target.blend, Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING));
        }
    }
    assert_eq!(keys.len(), 8);
}

fn shader_with_defs(source: &str, antialias: bool, defines: &[&str]) -> naga::Module {
    let mut composer = Composer::default();
    for (source, path) in [
        (
            include_str!("../../../bevy_render/src/globals.wgsl"),
            "globals.wgsl",
        ),
        (
            include_str!("../ui_vertex_output.wgsl"),
            "ui_vertex_output.wgsl",
        ),
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
            shader_defs: defines
                .iter()
                .copied()
                .chain(antialias.then_some("ANTI_ALIAS"))
                .map(|name| (name.into(), ShaderDefValue::Bool(true)))
                .chain([("SHADOW_SAMPLES".into(), ShaderDefValue::UInt(4))])
                .collect(),
            ..Default::default()
        })
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&composer)))
}

#[test]
fn ui_shaders_validate_with_and_without_antialiasing() {
    use wgpu::naga::{back::glsl, valid, ShaderStage};
    for (source, defines, antialias) in [
        (include_str!("../box_shadow.wgsl"), &[][..]),
        (include_str!("../box_shadow.wgsl"), &["MANUAL_SRGB"][..]),
        (include_str!("../ui_texture_slice.wgsl"), &[][..]),
        (
            include_str!("../ui_texture_slice.wgsl"),
            &["MANUAL_SRGB"][..],
        ),
        (include_str!("../ui_material.wgsl"), &[][..]),
        (include_str!("../ui_material.wgsl"), &["MANUAL_SRGB"][..]),
        (include_str!("../surface.wgsl"), &[][..]),
        (include_str!("../surface.wgsl"), &["LINEAR_COLOR"][..]),
        (include_str!("../surface.wgsl"), &["SURFACE_SOLID"][..]),
        (
            include_str!("../surface.wgsl"),
            &["LINEAR_COLOR", "SURFACE_SOLID"][..],
        ),
        (include_str!("cached.wgsl"), &[][..]),
        (include_str!("cached.wgsl"), &["LINEAR_COLOR"][..]),
        (
            include_str!("../surface.wgsl"),
            &["LINEAR_COLOR", "SURFACE_RECT_FILL"][..],
        ),
        (
            include_str!("../surface.wgsl"),
            &["LINEAR_COLOR", "SURFACE_ROUNDED_FILL"][..],
        ),
        (include_str!("../surface.wgsl"), &["SURFACE_RECT_FILL"][..]),
        (
            include_str!("../surface.wgsl"),
            &["SURFACE_ROUNDED_FILL"][..],
        ),
        (include_str!("../ui.wgsl"), &[][..]),
        (include_str!("../ui.wgsl"), &["MANUAL_SRGB"][..]),
        (include_str!("../image_basic.wgsl"), &[][..]),
        (include_str!("../image_basic.wgsl"), &["MANUAL_SRGB"][..]),
        (
            include_str!("../image_basic.wgsl"),
            &["IMAGE_BASIC_ENCODED"][..],
        ),
        (include_str!("../ui.wgsl"), &["IMAGE_RECT"][..]),
        (
            include_str!("../ui.wgsl"),
            &["IMAGE_RECT", "CHEAP_RECT_DISTANCE"][..],
        ),
        (
            include_str!("../image_rect.wgsl"),
            &["IMAGE_RECT", "MANUAL_SRGB"][..],
        ),
        (include_str!("../image_rect.wgsl"), &["IMAGE_RECT"][..]),
        (
            include_str!("../image_rect.wgsl"),
            &["IMAGE_RECT", "CHEAP_RECT_DISTANCE"][..],
        ),
        (
            include_str!("../image_rect.wgsl"),
            &["IMAGE_RECT", "MANUAL_SRGB", "CHEAP_RECT_DISTANCE"][..],
        ),
        (
            include_str!("../surface_fill.wgsl"),
            &["SURFACE_RECT_FILL"][..],
        ),
        (
            include_str!("../surface_fill.wgsl"),
            &["LINEAR_COLOR", "SURFACE_RECT_FILL"][..],
        ),
        (
            include_str!("../surface_fill.wgsl"),
            &["LINEAR_COLOR", "SURFACE_ROUNDED_FILL"][..],
        ),
        (
            include_str!("../surface_fill.wgsl"),
            &["SURFACE_RECT_FILL", "CHEAP_RECT_DISTANCE"][..],
        ),
        (
            include_str!("../surface_fill.wgsl"),
            &["SURFACE_ROUNDED_FILL"][..],
        ),
        (
            include_str!("../ui.wgsl"),
            &["IMAGE_RECT", "MANUAL_SRGB"][..],
        ),
    ]
    .into_iter()
    .flat_map(|(source, defines)| [false, true].map(move |aa| (source, defines, aa)))
    {
        let module = shader_with_defs(source, antialias, defines);
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
        naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options::default(),
            &naga::back::msl::PipelineOptions::default(),
        )
        .expect("all UI variants must translate to Metal");
    }
}

#[test]
fn linear_shaders_omit_output_gamma_but_keep_text_group_decoding() {
    fn powers(module: &naga::Module, name: &str) -> usize {
        module
            .functions
            .iter()
            .filter(|(_, function)| {
                function
                    .name
                    .as_deref()
                    .is_some_and(|n| n.starts_with(name))
            })
            .map(|(_, function)| {
                function
                    .expressions
                    .iter()
                    .filter(|(_, expression)| {
                        matches!(
                            expression,
                            naga::Expression::Math {
                                fun: naga::MathFunction::Pow,
                                ..
                            }
                        )
                    })
                    .count()
            })
            .sum::<usize>()
            + module
                .entry_points
                .iter()
                .filter(|entry| entry.name == name)
                .map(|entry| {
                    entry
                        .function
                        .expressions
                        .iter()
                        .filter(|(_, expression)| {
                            matches!(
                                expression,
                                naga::Expression::Math {
                                    fun: naga::MathFunction::Pow,
                                    ..
                                }
                            )
                        })
                        .count()
                })
                .sum::<usize>()
    }
    for (source, function, gamma_defs, linear_defs) in [
        (
            include_str!("../image_basic.wgsl"),
            "manual_srgb",
            &["MANUAL_SRGB"][..],
            &[][..],
        ),
        (
            include_str!("../ui.wgsl"),
            "manual_srgb",
            &["MANUAL_SRGB"][..],
            &[][..],
        ),
        (
            include_str!("../box_shadow.wgsl"),
            "linear_to_gamma_2_2",
            &["MANUAL_SRGB"][..],
            &[][..],
        ),
        (
            include_str!("../ui_texture_slice.wgsl"),
            "linear_to_gamma_2_2",
            &["MANUAL_SRGB"][..],
            &[][..],
        ),
        (
            include_str!("../surface.wgsl"),
            "paint",
            &[][..],
            &["LINEAR_COLOR"][..],
        ),
    ] {
        assert_eq!(
            powers(&shader_with_defs(source, true, gamma_defs), function),
            1
        );
        assert_eq!(
            powers(&shader_with_defs(source, true, linear_defs), function),
            0
        );
    }
    let source = include_str!("../ui.wgsl");
    assert_eq!(powers(&shader_with_defs(source, true, &[]), "fragment"), 1);
    assert_eq!(
        powers(
            &shader_with_defs(source, true, &["MANUAL_SRGB"]),
            "fragment"
        ),
        0
    );
}

#[test]
fn composite_lookup_shader_validates() {
    let source = include_str!("../srgb_ui_composite.wgsl").replace(
        "#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput",
        "struct FullscreenVertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, };",
    );
    for defines in [&[][..], &["UI_COMPOSITE_LOOKUP"][..]] {
        let module = shader_with_defs(&source, false, defines);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("both composite paths must validate");
    }
}

#[test]
fn specialized_surfaces_reject_incompatible_geometry_and_paint() {
    use super::{surface_shader, SurfaceShader};
    let fill = GpuSurface {
        size: [64., 32.],
        background: [0.2, 0.3, 0.6, 0.5],
        ..Default::default()
    };
    assert_eq!(surface_shader(&fill), SurfaceShader::RectFill);
    assert_eq!(
        surface_shader(&GpuSurface {
            radius: [8.; 4],
            ..fill
        }),
        SurfaceShader::RoundedFill
    );
    for incompatible in [
        GpuSurface {
            border: [1., 0., 0., 0.],
            ..fill
        },
        GpuSurface {
            radius: [17., 0., 0., 0.],
            ..fill
        },
        GpuSurface {
            radius: [f32::NAN, 0., 0., 0.],
            ..fill
        },
        GpuSurface {
            radius: [-1., 0., 0., 0.],
            ..fill
        },
        GpuSurface {
            background: [0.; 4],
            ..fill
        },
    ] {
        // Solid keeps the full geometry path, including the inset radius clamp.
        assert_eq!(surface_shader(&incompatible), SurfaceShader::Solid);
    }
}

#[test]
fn solid_surfaces_require_zero_extracted_layers_for_background_and_every_border_side() {
    use super::{surface_shader, SurfaceShader};
    for border in [[0.; 4], [0.2, 1.5, 4.25, 8.]] {
        let solid = GpuSurface {
            size: [64., 32.],
            border,
            radius: [0., 3., 12., 16.],
            background: [0.2, 0.3, 0.6, 0.5],
            // Addresses do not imply layers. Likewise, transparent paint or a
            // zero-width side does not make a nonzero layer count safe to ignore.
            background_gradients: [123, 0],
            border_gradients: [456, 0],
            ..Default::default()
        };
        assert_ne!(surface_shader(&solid), SurfaceShader::Full);
        for count in [1, 4, u32::MAX] {
            for gradients in [
                GpuSurface {
                    background_gradients: [0, count],
                    ..solid
                },
                GpuSurface {
                    border_gradients: [0, count],
                    ..solid
                },
            ] {
                assert_eq!(surface_shader(&gradients), SurfaceShader::Full);
            }
        }
    }
}

#[test]
fn surface_selection_preserves_reference_readiness_and_independent_opt_ins() {
    use super::{select_surface_shader, SurfaceShader};
    use crate::UiShaderVariants;
    for surfaces in [false, true] {
        for solid_surfaces in [false, true] {
            let variants = UiShaderVariants {
                surfaces,
                solid_surfaces,
                ..UiShaderVariants::FULL
            };
            for mask in 0..16 {
                let ready = |shader| mask & (1 << shader as usize) != 0;
                for candidate in SurfaceShader::ALL {
                    let selected = select_surface_shader(candidate, variants, ready);
                    if selected != SurfaceShader::Full {
                        assert!(ready(SurfaceShader::Full) && ready(selected));
                        assert_ne!(candidate, SurfaceShader::Full, "gradient paints stay Full");
                    }
                    assert!(surfaces || !selected.is_fill());
                    assert!(solid_surfaces || selected != SurfaceShader::Solid);
                    if mask == 15 {
                        let expected = if surfaces && candidate.is_fill() {
                            candidate
                        } else if solid_surfaces && candidate != SurfaceShader::Full {
                            SurfaceShader::Solid
                        } else {
                            SurfaceShader::Full
                        };
                        assert_eq!(selected, expected);
                    }
                }
            }
        }
    }
    let variants = UiShaderVariants {
        solid_surfaces: true,
        ..UiShaderVariants::ALL
    };
    // Compiled solid is safe while a fill pipeline is pending, but never vice versa.
    assert_eq!(
        select_surface_shader(SurfaceShader::RoundedFill, variants, |s| s
            != SurfaceShader::RectFill),
        SurfaceShader::Solid
    );
    assert_eq!(
        select_surface_shader(SurfaceShader::Solid, variants, |s| s
            != SurfaceShader::Solid),
        SurfaceShader::Full
    );
}

#[test]
fn common_surface_pipeline_never_absorbs_gradients_into_solid_batches() {
    use super::{common_surface_pipeline, SurfaceShader};
    use bevy_render::render_resource::CachedRenderPipelineId;
    let choices = [0, 1, 2, 3].map(CachedRenderPipelineId::new);
    for a in SurfaceShader::ALL {
        for b in SurfaceShader::ALL {
            assert_eq!(
                common_surface_pipeline(&choices, choices[a as usize], choices[b as usize]),
                choices[(a as usize).min(b as usize)]
            );
        }
    }
    // Default cameras alias the unused Solid entry to Full (no extra compilation).
    let defaults = [choices[0], choices[0], choices[2], choices[3]];
    assert_eq!(
        common_surface_pipeline(&defaults, choices[2], choices[3]),
        choices[2]
    );
    assert_eq!(
        common_surface_pipeline(&defaults, choices[0], choices[3]),
        choices[0]
    );
}

#[test]
fn solid_surface_fragment_has_no_gradient_machinery_or_texture_loads() {
    fn check_function(module: &naga::Module, function: &naga::Function) {
        for (_, expression) in function.expressions.iter() {
            match expression {
                naga::Expression::ImageLoad { .. } | naga::Expression::ImageSample { .. } => {
                    panic!("solid fragment must not load paint or gradients")
                }
                naga::Expression::CallResult(handle) => {
                    check_function(module, &module.functions[*handle]);
                }
                _ => (),
            }
        }
    }
    for aa in [false, true] {
        for defines in [
            &["SURFACE_SOLID"][..],
            &["SURFACE_SOLID", "LINEAR_COLOR"][..],
        ] {
            let module = shader_with_defs(include_str!("../surface.wgsl"), aa, defines);
            assert!(!module.functions.iter().any(|(_, function)| function
                .name
                .as_deref()
                .is_some_and(
                    |name| name.starts_with("sample_gradient") || name.starts_with("paint")
                )));
            let fragment = module
                .entry_points
                .iter()
                .find(|entry| entry.name == "fragment")
                .unwrap();
            check_function(&module, &fragment.function);
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
}

impl Gpu {
    fn new(backends: wgpu::Backends) -> Self {
        Self::with_variant(backends, true, &[])
    }

    fn with_variant(backends: wgpu::Backends, antialias: bool, defines: &[&str]) -> Self {
        Self::with_source(
            backends,
            antialias,
            defines,
            include_str!("../surface.wgsl"),
        )
    }

    #[expect(
        clippy::print_stderr,
        reason = "Record which adapter ran the opt-in GPU regression."
    )]
    fn with_source(
        backends: wgpu::Backends,
        antialias: bool,
        defines: &[&str],
        source: &str,
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            bevy_tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU regression requires an adapter");
        eprintln!("UI surface GPU regression: {:?}", adapter.get_info());
        assert!(
            !matches!(adapter.get_info().device_type, wgpu::DeviceType::Cpu),
            "GPU regression requires hardware rendering, not a software adapter"
        );
        let (device, queue) =
            bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            }))
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface regression"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(shader_with_defs(
                source, antialias, defines,
            ))),
        });
        let format = if defines.contains(&"LINEAR_COLOR") {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
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
                    format,
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
            format,
        }
    }

    fn render(&self, surface: GpuSurface, phase: f32, gradients: &[Gradient]) -> Vec<u8> {
        self.render_geometry(surface, phase, gradients, 0., false)
    }

    fn render_geometry(
        &self,
        mut surface: GpuSurface,
        phase: f32,
        gradients: &[Gradient],
        angle: f32,
        clipped: bool,
    ) -> Vec<u8> {
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
        let corners = QUAD_VERTEX_POSITIONS.map(|p| {
            let point = p * size;
            (
                (bevy_math::Affine2::from_angle(angle).transform_point2(point)
                    + size * 0.5
                    + Vec2::splat(8. + phase)),
                point,
            )
        });
        let clip = clipped.then(|| {
            let mut clip = bevy_ui::CalculatedClip::default();
            clip.push_rect(
                bevy_math::Rect::new(17.125, 11.375, 58.625, 37.125),
                bevy_math::Affine2::from_angle(0.07),
            );
            clip
        });
        // Match the production polygon clipping/attribute interpolation path;
        // scissoring alone would not exercise new vertices and flat paint IO.
        let polygon = crate::clipping::clip_polygon(clip.as_ref(), &corners, Vec2::lerp);
        let vertices: Vec<_> = polygon
            .iter()
            .map(|&(position, point)| SurfaceVertex {
                position: position.extend(0.).into(),
                point: point.into(),
                surface: address,
            })
            .collect();
        let indices: Vec<u16> = (2..vertices.len() as u16)
            .flat_map(|i| [0, i - 1, i])
            .collect();
        let index_count = indices.len() as u32;
        let vertices = buffer(bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX);
        let indices = buffer(bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX);
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
            format: self.format,
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
            if clipped {
                pass.set_scissor_rect(17, 11, 41, 26);
            }
            if index_count != 0 {
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..index_count, 0, 0..1);
            }
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

#[test]
#[ignore = "Requires a hardware Vulkan adapter"]
fn gpu_surface_variants_match_reference_vulkan() {
    check_surface_variants(wgpu::Backends::VULKAN);
}

#[test]
#[ignore = "Requires a hardware OpenGL adapter"]
fn gpu_surface_variants_match_reference_gl() {
    check_surface_variants(wgpu::Backends::GL);
}

#[test]
#[ignore = "Requires a hardware Vulkan adapter"]
fn gpu_solid_surfaces_match_reference_vulkan() {
    check_solid_surface_variants(wgpu::Backends::VULKAN);
}

#[test]
#[ignore = "Requires a hardware OpenGL adapter"]
fn gpu_solid_surfaces_match_reference_gl() {
    check_solid_surface_variants(wgpu::Backends::GL);
}

#[expect(
    clippy::print_stderr,
    reason = "Report measured hardware image differences"
)]
fn check_solid_surface_variants(backend: wgpu::Backends) {
    let mut maximum = 0;
    let mut differing_channels = 0;
    let mut comparisons = 0;
    for aa in [false, true] {
        for linear in [false, true] {
            let mut defines = Vec::new();
            if linear {
                defines.push("LINEAR_COLOR");
            }
            let reference = Gpu::with_variant(backend, aa, &defines);
            defines.push("SURFACE_SOLID");
            let variant = Gpu::with_variant(backend, aa, &defines);
            for (size, radius, border) in [
                ([64., 32.], [0.; 4], [0.; 4]),
                ([64., 32.], [8.; 4], [0.2; 4]),
                ([64., 32.], [0., 3.5, 11.75, 16.], [1.5; 4]),
                ([63.25, 31.75], [7.5, 2., 13.25, 4.], [4.25; 4]),
                ([63.25, 31.75], [0., 16., 2., 9.5], [0., 0.2, 8., 1.5]),
                // Inner-radius clamp and collapsed inner shape stay on the
                // reference SDF path; these are not fill-only specializations.
                ([64., 32.], [18., 24., 40., 2.], [40., 25., 35., 20.]),
                ([2.5, 1.75], [0.5, 1., 0.25, 0.], [0.2, 1.5, 0., 0.75]),
            ] {
                for alpha in [0., 0.375, 1.] {
                    let paint = |rgba: [f32; 4]| {
                        super::surface_color(LinearRgba::from_f32_array(rgba), linear)
                    };
                    let surface = GpuSurface {
                        size,
                        radius,
                        border,
                        background: paint([0.125, 0.45, 0.8, alpha]),
                        border_colors: [
                            paint([0.9, 0.1, 0.25, 0.25]),
                            paint([0.2, 0.8, 0.05, 0.625]),
                            paint([0.1, 0.25, 0.95, 0.]),
                            paint([0.75, 0.5, 0.15, 1.]),
                        ],
                        ..Default::default()
                    };
                    assert_ne!(super::surface_shader(&surface), super::SurfaceShader::Full);
                    for phase in [0., 0.125, 0.5, 0.875] {
                        for angle in [0., 0.21] {
                            for clipped in [false, true] {
                                let expected =
                                    reference.render_geometry(surface, phase, &[], angle, clipped);
                                let actual =
                                    variant.render_geometry(surface, phase, &[], angle, clipped);
                                let difference = actual
                                    .iter()
                                    .zip(&expected)
                                    .map(|(&a, &b)| a.abs_diff(b))
                                    .max()
                                    .unwrap();
                                maximum = maximum.max(difference);
                                differing_channels +=
                                    actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                                comparisons += 1;
                                // Separately compiled hardware programs can round
                                // across an 8-bit boundary; bound every channel.
                                assert!(difference <= 1, "SURFACE_SOLID delta={difference}, AA={aa}, linear={linear}, size={size:?}, radius={radius:?}, border={border:?}, alpha={alpha}, phase={phase}, angle={angle}, clipped={clipped}");
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("SURFACE_SOLID equivalence: {comparisons} image comparisons, maximum channel delta={maximum}/255; differing channels across fixtures={differing_channels}");
}

fn check_surface_variants(backend: wgpu::Backends) {
    for aa in [false, true] {
        let reference = Gpu::with_variant(backend, aa, &[]);
        for (define, radius) in [
            ("SURFACE_RECT_FILL", [0.; 4]),
            ("SURFACE_ROUNDED_FILL", [4., 8., 12., 16.]),
        ] {
            for (source, cheap) in [
                (include_str!("../surface.wgsl"), false),
                (include_str!("../surface.wgsl"), true),
                (include_str!("../surface_fill.wgsl"), false),
                (include_str!("../surface_fill.wgsl"), true),
            ] {
                let mut definitions = vec![define];
                if cheap {
                    definitions.push("CHEAP_RECT_DISTANCE");
                }
                let variant = Gpu::with_source(backend, aa, &definitions, source);
                for phase in [0., 0.125, 0.5, 0.875] {
                    for alpha in [0.1, 0.5, 1.] {
                        let fill = GpuSurface {
                            size: [64., 32.],
                            radius,
                            background: [0.25, 0.5, 0.75, alpha],
                            ..Default::default()
                        };
                        for angle in [0., 0.21] {
                            for clipped in [false, true] {
                                assert_eq!(
                            reference.render_geometry(fill, phase, &[], angle, clipped),
                            variant.render_geometry(fill, phase, &[], angle, clipped),
                            "{define}, AA={aa}, phase={phase}, alpha={alpha}, angle={angle}, clipped={clipped}"
                        );
                            }
                        }
                    }
                }
            }
        }
    }
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
