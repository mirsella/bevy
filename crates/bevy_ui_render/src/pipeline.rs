use crate::UiViewTarget;
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
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, ViewUniform},
};
use bevy_shader::Shader;
use bevy_utils::default;

pub(crate) const MANUAL_SRGB_SHADER_DEF: &str = "MANUAL_SRGB";

pub(crate) fn ui_render_target_format(target_format: TextureFormat) -> TextureFormat {
    if target_format == TextureFormat::Rgba16Float {
        TextureFormat::Rgba16Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

pub(crate) fn ui_color_target_state(target_format: TextureFormat) -> ColorTargetState {
    ColorTargetState {
        format: target_format,
        blend: Some(BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    }
}

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

#[derive(Resource)]
pub struct SrgbUiCompositePipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

pub fn init_ui_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
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

    let composite_layout = BindGroupLayoutDescriptor::new(
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
        ..default()
    });

    commands.insert_resource(SrgbUiCompositePipeline {
        layout: composite_layout,
        sampler: composite_sampler,
        shader: load_embedded_asset!(asset_server.as_ref(), "srgb_ui_composite.wgsl"),
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
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
                targets: vec![Some(ui_color_target_state(key.target_format))],
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
    format: TextureFormat,
}

impl SpecializedRenderPipeline for SrgbUiCompositePipeline {
    type Key = SrgbUiCompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("srgb_ui_composite_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ui_color_target_state(key.format))],
            }),
            ..default()
        }
    }
}

#[derive(Component)]
pub struct SrgbUiTexture {
    pub(crate) texture: CachedTexture,
}

#[derive(Component)]
pub struct SrgbUiCompositeBindGroup {
    pub(crate) bind_group: BindGroup,
}

#[derive(Component)]
pub struct SrgbUiCompositePipelineId(pub(crate) CachedRenderPipelineId);

pub fn prepare_srgb_ui_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    views: Query<(Entity, &ExtractedView), With<UiViewTarget>>,
) {
    for (entity, view) in &views {
        let size = Extent3d {
            width: view.viewport.z,
            height: view.viewport.w,
            depth_or_array_layers: 1,
        };

        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("srgb_ui_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: ui_render_target_format(view.target_format),
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(SrgbUiTexture { texture });
    }
}

pub fn queue_srgb_ui_composite_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SrgbUiCompositePipeline>>,
    pipeline: Res<SrgbUiCompositePipeline>,
    views: Query<(Entity, &ExtractedView), With<UiViewTarget>>,
) {
    for (entity, view) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SrgbUiCompositePipelineKey {
                format: view.target_format,
            },
        );

        commands
            .entity(entity)
            .insert(SrgbUiCompositePipelineId(pipeline_id));
    }
}

pub fn prepare_srgb_ui_composite_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    composite_pipeline: Res<SrgbUiCompositePipeline>,
    views: Query<(Entity, &SrgbUiTexture)>,
) {
    for (entity, srgb_texture) in &views {
        let bind_group = render_device.create_bind_group(
            "srgb_ui_composite_bind_group",
            &pipeline_cache.get_bind_group_layout(&composite_pipeline.layout),
            &BindGroupEntries::sequential((
                &srgb_texture.texture.default_view,
                &composite_pipeline.sampler,
            )),
        );

        commands
            .entity(entity)
            .insert(SrgbUiCompositeBindGroup { bind_group });
    }
}
