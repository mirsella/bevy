use super::*;
use crate::{
    box_shadow::{BoxShadowPipeline, BoxShadowPipelineKey},
    ui_texture_slice_pipeline::{UiTextureSlicePipeline, UiTextureSlicePipelineKey},
    UiShaderVariants,
};

fn key(target_format: TextureFormat) -> UiPipelineKey {
    UiPipelineKey {
        target_format,
        basic_image: false,
        encoded_image: false,
        relaxed_precision: true,
        relaxed_colors: true,
        anti_alias: true,
        image_rect: false,
        rect_math: false,
        lean_io: false,
        pixel_aligned: false,
        implicit_aligned: false,
    }
}

fn pipeline() -> UiPipeline {
    UiPipeline {
        view_layout: BindGroupLayoutDescriptor::new("test_view", &[]),
        image_layout: BindGroupLayoutDescriptor::new("test_image", &[]),
        shader: Handle::default(),
        rect_shader: Handle::default(),
        basic_shader: Handle::default(),
    }
}

#[test]
fn linear_color_is_never_a_default_and_preserves_reference_formats() {
    for variants in [
        UiShaderVariants::FULL,
        UiShaderVariants::SURFACES,
        UiShaderVariants::IMAGES,
        UiShaderVariants::ALL,
        UiShaderVariants::default(),
    ] {
        assert!(!variants.linear_color);
        for (input, expected) in [
            (TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgba8Unorm),
            (TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba8Unorm),
            (TextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ] {
            assert_eq!(variants.render_target_format(input), expected);
        }
        for format in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb] {
            assert_eq!(
                UiShaderVariants {
                    linear_color: true,
                    ..variants
                }
                .render_target_format(format),
                format
            );
        }
    }
}

#[test]
fn solid_surfaces_is_opt_in_in_every_preset() {
    for variants in [
        UiShaderVariants::FULL,
        UiShaderVariants::SURFACES,
        UiShaderVariants::IMAGES,
        UiShaderVariants::ALL,
        UiShaderVariants::default(),
    ] {
        assert!(!variants.solid_surfaces);
    }
}

#[test]
#[should_panic(expected = "linear-color UI cannot use encoded-image views")]
fn linear_color_rejects_encoded_images_even_before_basic_pipeline_selection() {
    UiShaderVariants {
        linear_color: true,
        encoded_images: true,
        ..UiShaderVariants::FULL
    }
    .render_target_format(TextureFormat::Rgba8UnormSrgb);
}

#[test]
#[should_panic(expected = "linear-color UI requires an RGBA8 or BGRA8 sRGB main target")]
fn linear_color_rejects_an_incompatible_main_target_instead_of_gamma_fallback() {
    UiShaderVariants {
        linear_color: true,
        ..UiShaderVariants::ALL
    }
    .render_target_format(TextureFormat::Rgba16Float);
}

#[test]
fn full_fallback_and_image_variants_keep_the_same_color_mode() {
    let pipeline = pipeline();
    let mut keys = bevy_platform::collections::HashSet::<UiPipelineKey>::default();
    for format in [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb,
    ] {
        for image in 0..4 {
            let key = UiPipelineKey {
                image_rect: image != 0,
                lean_io: image == 2,
                basic_image: image == 3,
                ..key(format)
            };
            assert!(keys.insert(key));
            let descriptor = pipeline.specialize(key);
            let fragment = descriptor.fragment.unwrap();
            assert_eq!(fragment.targets[0].as_ref().unwrap().format, format);
            let manual = MANUAL_SRGB_SHADER_DEF.into();
            assert_eq!(
                descriptor.vertex.shader_defs.contains(&manual),
                !format.is_srgb()
            );
            assert_eq!(fragment.shader_defs.contains(&manual), !format.is_srgb());
            // Color and interface definitions are shared; precision is fragment-only.
            for precision in [
                bevy_shader::EXPERIMENTAL_RELAXED_FRAGMENT,
                bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT,
            ] {
                assert!(!descriptor.vertex.shader_defs.contains(&precision.into()));
                assert!(fragment.shader_defs.contains(&precision.into()));
            }
        }
    }
    assert_eq!(keys.len(), 12);
}

#[test]
#[should_panic(expected = "linear-color UI cannot use the encoded-image shader")]
fn direct_pipeline_specialization_also_rejects_basic_encoded_linear_color() {
    pipeline().specialize(UiPipelineKey {
        basic_image: true,
        encoded_image: true,
        ..key(TextureFormat::Rgba8UnormSrgb)
    });
}

#[test]
fn shadows_and_sliced_images_share_color_mode_in_both_stages() {
    let shadow = BoxShadowPipeline {
        view_layout: BindGroupLayoutDescriptor::new("test_view", &[]),
        shader: Handle::default(),
    };
    let slice = UiTextureSlicePipeline {
        view_layout: BindGroupLayoutDescriptor::new("test_view", &[]),
        image_layout: BindGroupLayoutDescriptor::new("test_image", &[]),
        shader: Handle::default(),
        cpu_shader: Handle::default(),
    };
    for format in [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb,
    ] {
        for descriptor in [
            shadow.specialize(BoxShadowPipelineKey {
                target_format: format,
                samples: 4,
            }),
            slice.specialize(UiTextureSlicePipelineKey {
                target_format: format,
                cpu_slices: false,
            }),
        ] {
            let fragment = descriptor.fragment.unwrap();
            assert_eq!(descriptor.vertex.shader_defs, fragment.shader_defs);
            assert_eq!(fragment.targets[0].as_ref().unwrap().format, format);
            assert_eq!(
                fragment
                    .shader_defs
                    .contains(&MANUAL_SRGB_SHADER_DEF.into()),
                !format.is_srgb()
            );
        }
    }
}
