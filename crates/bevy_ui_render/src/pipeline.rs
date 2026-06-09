use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::ViewUniform,
};
use bevy_shader::Shader;
use bevy_utils::default;

pub(crate) const MANUAL_SRGB_SHADER_DEF: &str = "MANUAL_SRGB";

/// Format used by the UI render pass intermediate texture.
pub(crate) fn ui_render_target_format(hdr: bool) -> TextureFormat {
    if hdr {
        TextureFormat::Rgba16Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

pub(crate) fn ui_color_target_state(hdr: bool) -> ColorTargetState {
    ColorTargetState {
        format: ui_render_target_format(hdr),
        blend: Some(BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    }
}

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayout,
    pub image_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

#[derive(Resource)]
pub struct SrgbUiCompositePipeline {
    pub layout: BindGroupLayout,
    pub sampler: Sampler,
    pub shader: Handle<Shader>,
    pub fullscreen_shader: Handle<Shader>,
}

pub fn init_ui_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let view_layout = render_device.create_bind_group_layout(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = render_device.create_bind_group_layout(
        "ui_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(UiPipeline {
        view_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wgsl"),
    });

    let composite_layout = render_device.create_bind_group_layout(
        "srgb_ui_composite_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let composite_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("srgb_ui_composite_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });

    commands.insert_resource(SrgbUiCompositePipeline {
        layout: composite_layout,
        sampler: composite_sampler,
        shader: load_embedded_asset!(asset_server.as_ref(), "srgb_ui_composite.wgsl"),
        fullscreen_shader: fullscreen_shader.shader(),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub hdr: bool,
    pub anti_alias: bool,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // mode
                VertexFormat::Uint32,
                // border radius
                VertexFormat::Float32x4,
                // border thickness
                VertexFormat::Float32x4,
                // border size
                VertexFormat::Float32x2,
                // position relative to the center
                VertexFormat::Float32x2,
                // shadow color
                VertexFormat::Float32x4,
                // outline color
                VertexFormat::Float32x4,
                // effect params
                VertexFormat::Float32x4,
            ],
        );
        let mut shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            Vec::new()
        };

        shader_defs.push(MANUAL_SRGB_SHADER_DEF.into());

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ui_color_target_state(key.hdr))],
                ..default()
            }),
            layout: vec![self.view_layout.clone(), self.image_layout.clone()],
            label: Some("ui_pipeline".into()),
            ..default()
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct SrgbUiCompositePipelineKey {
    pub format: TextureFormat,
}

impl SpecializedRenderPipeline for SrgbUiCompositePipeline {
    type Key = SrgbUiCompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("srgb_ui_composite_pipeline".into()),
            layout: vec![self.layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.fullscreen_shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    }
}
