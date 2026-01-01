use crate::{SrgbUiCompositePipeline, SrgbUiCompositePipelineKey};
use bevy_ecs::prelude::*;
use bevy_image::BevyDefault;
use bevy_render::render_resource::{
    BindGroup, BindGroupEntries, CachedRenderPipelineId, Extent3d, FilterMode, PipelineCache,
    SamplerDescriptor, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy_render::renderer::RenderDevice;
use bevy_render::texture::{CachedTexture, TextureCache};
use bevy_render::view::{ExtractedView, Msaa, ViewTarget};

/// Component storing the offscreen sRGB texture for a view.
#[derive(Component)]
pub struct SrgbUiTexture {
    /// The texture to render to. Multisampled when MSAA > 1.
    pub texture: CachedTexture,
    /// The resolve target texture. Only present when MSAA > 1.
    /// This is the non-multisampled texture that will be sampled in the composite pass.
    pub resolve_texture: Option<CachedTexture>,
}

/// Component storing the bind group for the sRGB composite pass.
#[derive(Component)]
pub struct SrgbUiCompositeBindGroup {
    pub bind_group: BindGroup,
}

/// Component storing the pipeline ID for the sRGB composite pass.
#[derive(Component)]
pub struct SrgbUiCompositePipelineId(pub CachedRenderPipelineId);

/// Prepare sRGB UI textures for each view.
pub fn prepare_srgb_ui_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    views: Query<(Entity, &ExtractedView, Option<&Msaa>), With<crate::UiViewTarget>>,
) {
    for (entity, view, msaa) in &views {
        let size = Extent3d {
            width: view.viewport.z,
            height: view.viewport.w,
            depth_or_array_layers: 1,
        };

        let sample_count = msaa.map(|m| m.samples()).unwrap_or(1);

        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("srgb_ui_texture"),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // When MSAA is enabled, we need a separate non-multisampled texture for sampling
        let resolve_texture = if sample_count > 1 {
            Some(texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some("srgb_ui_resolve_texture"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            ))
        } else {
            None
        };

        commands.entity(entity).insert(SrgbUiTexture {
            texture,
            resolve_texture,
        });
    }
}

/// Queue sRGB composite pipelines.
pub fn queue_srgb_ui_composite_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SrgbUiCompositePipeline>>,
    pipeline: Res<SrgbUiCompositePipeline>,
    views: Query<(Entity, &ExtractedView), With<crate::UiViewTarget>>,
) {
    for (entity, view) in &views {
        let format = if view.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SrgbUiCompositePipelineKey { format },
        );

        commands
            .entity(entity)
            .insert(SrgbUiCompositePipelineId(pipeline_id));
    }
}

/// Prepare sRGB composite bind groups.
pub fn prepare_srgb_ui_composite_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    composite_pipeline: Res<crate::SrgbUiCompositePipeline>,
    views: Query<(Entity, &SrgbUiTexture)>,
) {
    for (entity, srgb_texture) in &views {
        // Use the resolve texture for sampling when MSAA is enabled,
        // otherwise use the main texture directly
        let texture_view = srgb_texture
            .resolve_texture
            .as_ref()
            .map(|t| &t.default_view)
            .unwrap_or(&srgb_texture.texture.default_view);

        let bind_group = render_device.create_bind_group(
            "srgb_ui_composite_bind_group",
            &composite_pipeline.layout,
            &BindGroupEntries::sequential((texture_view, &composite_pipeline.sampler)),
        );

        commands
            .entity(entity)
            .insert(SrgbUiCompositeBindGroup { bind_group });
    }
}
