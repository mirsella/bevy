use super::*;

fn pipeline() -> UiPipeline {
    UiPipeline {
        view_layout: BindGroupLayoutDescriptor::new("test_view", &[]),
        image_layout: BindGroupLayoutDescriptor::new("test_image", &[]),
        shader: Handle::default(),
    }
}

#[test]
fn full_fallback_and_image_variants_keep_the_same_color_mode() {
    let pipeline = pipeline();
    let mut keys = bevy_platform::collections::HashSet::<UiPipelineKey>::default();
    let shaders: [(UiNodeShader, &[&str]); 3] = [
        (UiNodeShader::Full, &[]),
        (UiNodeShader::ImageRect, &["IMAGE_RECT"]),
        (
            UiNodeShader::AlignedImageRect,
            &["IMAGE_RECT", "IMAGE_PIXEL_ALIGNED"],
        ),
    ];
    for (shader, definitions) in shaders {
        let key = UiPipelineKey {
            target_format: TextureFormat::Rgba8Unorm,
            anti_alias: true,
            relaxed_colors: true,
            shader,
        };
        assert!(keys.insert(key));
        let descriptor = pipeline.specialize(key);
        let fragment = descriptor.fragment.unwrap();
        for name in ["IMAGE_RECT", "IMAGE_PIXEL_ALIGNED"] {
            assert_eq!(
                fragment.shader_defs.contains(&name.into()),
                definitions.contains(&name)
            );
        }
        assert_eq!(
            fragment.targets[0].as_ref().unwrap().format,
            TextureFormat::Rgba8Unorm
        );
        let manual = MANUAL_SRGB_SHADER_DEF.into();
        assert!(descriptor.vertex.shader_defs.contains(&manual));
        assert!(fragment.shader_defs.contains(&manual));
        // Color and interface definitions are shared; precision is fragment-only.
        let precision = bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT.into();
        assert!(!descriptor.vertex.shader_defs.contains(&precision));
        assert!(fragment.shader_defs.contains(&precision));
    }
}
