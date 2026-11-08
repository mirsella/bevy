use crate::{UiCompositeLookup, UiViewTarget};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, ViewUniform},
};
use bevy_shader::Shader;
use bevy_utils::default;

#[cfg(test)]
mod tests;

pub(crate) const MANUAL_SRGB_SHADER_DEF: &str = "MANUAL_SRGB";

pub(crate) fn ui_render_target_format(target_format: TextureFormat) -> TextureFormat {
    if target_format == TextureFormat::Rgba16Float {
        TextureFormat::Rgba16Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

// Gamma UI intermediates are never sRGB formats. The opt-in direct linear pass
// uses Rgba8UnormSrgb, so the existing format keys also isolate its color shaders.
pub(crate) fn ui_color_shader_defs(target_format: TextureFormat) -> Vec<bevy_shader::ShaderDefVal> {
    if target_format.is_srgb() {
        Vec::new()
    } else {
        vec![MANUAL_SRGB_SHADER_DEF.into()]
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
    pub rect_shader: Handle<Shader>,
    pub basic_shader: Handle<Shader>,
}

#[derive(Resource)]
pub struct SrgbUiCompositePipeline {
    layout: BindGroupLayoutDescriptor,
    lookup_layout: BindGroupLayoutDescriptor,
    lookup: TextureView,
    sampler: Sampler,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

pub fn init_ui_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    render_queue: Res<RenderQueue>,
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
        rect_shader: load_embedded_asset!(asset_server.as_ref(), "image_rect.wgsl"),
        basic_shader: load_embedded_asset!(asset_server.as_ref(), "image_basic.wgsl"),
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

    let lookup_layout = BindGroupLayoutDescriptor::new(
        "srgb_ui_lookup_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let lookup = render_device.create_texture(&TextureDescriptor {
        label: Some("ui_unpremultiply_gamma_lookup"),
        size: Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let values: Vec<f32> = (0..256)
        .map(|color| bevy_math::ops::powf(color as f32 / 255., 2.2))
        .collect();
    render_queue.write_texture(
        lookup.as_image_copy(),
        bytemuck::cast_slice(&values),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256 * 4),
            rows_per_image: Some(1),
        },
        lookup.size(),
    );

    commands.insert_resource(SrgbUiCompositePipeline {
        layout: composite_layout,
        lookup_layout,
        lookup: lookup.create_view(&TextureViewDescriptor::default()),
        sampler: composite_sampler,
        shader: load_embedded_asset!(asset_server.as_ref(), "srgb_ui_composite.wgsl"),
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub basic_image: bool,
    pub encoded_image: bool,
    pub relaxed_precision: bool,
    pub relaxed_colors: bool,
    pub target_format: TextureFormat,
    pub anti_alias: bool,
    pub image_rect: bool,
    pub rect_math: bool,
    pub lean_io: bool,
    pub pixel_aligned: bool,
    pub implicit_aligned: bool,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        assert!(
            !(key.encoded_image && key.target_format.is_srgb()),
            "linear-color UI cannot use the encoded-image shader"
        );
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
            ],
        );
        let mut shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            Vec::new()
        };
        shader_defs.extend(ui_color_shader_defs(key.target_format));
        if key.image_rect {
            shader_defs.push("IMAGE_RECT".into());
        }
        if key.rect_math {
            shader_defs.push("CHEAP_RECT_DISTANCE".into());
        }
        if key.pixel_aligned {
            shader_defs.push("IMAGE_PIXEL_ALIGNED".into());
        }
        if key.implicit_aligned {
            shader_defs.push("IMAGE_IMPLICIT_SAMPLING".into());
        }
        if key.basic_image {
            shader_defs.push("IMAGE_BASIC".into());
        }
        if key.encoded_image {
            shader_defs.push("IMAGE_BASIC_ENCODED".into());
        }
        let shader = if key.basic_image {
            &self.basic_shader
        } else if key.image_rect && key.lean_io {
            &self.rect_shader
        } else {
            &self.shader
        };

        let vertex_defs = shader_defs.clone();
        if key.relaxed_precision {
            shader_defs.push(bevy_shader::EXPERIMENTAL_RELAXED_FRAGMENT.into());
        }
        if key.relaxed_colors {
            shader_defs.push(bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT.into());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: vertex_defs,
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs,
                // Passthrough modules intentionally have no wgpu reflection;
                // an explicit name is required rather than entry-point inference.
                entry_point: Some("fragment".into()),
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
    lookup: bool,
    relaxed_precision: bool,
    relaxed_colors: bool,
}

impl SpecializedRenderPipeline for SrgbUiCompositePipeline {
    type Key = SrgbUiCompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("srgb_ui_composite_pipeline".into()),
            layout: vec![if key.lookup {
                self.lookup_layout.clone()
            } else {
                self.layout.clone()
            }],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: {
                    let mut defs = Vec::new();
                    if key.lookup {
                        defs.push("UI_COMPOSITE_LOOKUP".into());
                    }
                    if key.relaxed_precision {
                        defs.push(bevy_shader::EXPERIMENTAL_RELAXED_FRAGMENT.into());
                    }
                    if key.relaxed_colors {
                        defs.push(bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT.into());
                    }
                    defs
                },
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
    pub(crate) rect: Option<bevy_math::URect>,
    pub(crate) raster: Option<CachedTexture>,
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
    views: Query<(
        Entity,
        &ExtractedView,
        &crate::UiShaderVariants,
        &UiViewTarget,
    )>,
    cameras: Query<&bevy_render::camera::ExtractedCamera>,
    mut retained: ResMut<crate::retained_layer::RetainedUiLayers>,
) {
    let live: bevy_platform::collections::HashSet<_> = views
        .iter()
        .filter(|(_, _, variants, _)| !variants.linear_color)
        .map(|(_, v, _, _)| v.retained_view_entity)
        .collect();
    let crate::retained_layer::RetainedUiLayers { plans, layers } = &mut *retained;
    layers.retain(|key, _| live.contains(key) && plans.contains_key(key));
    for (entity, view, variants, camera) in &views {
        if variants.linear_color {
            continue;
        }
        if let Some(plan) = plans.get(&view.retained_view_entity) {
            let format = ui_render_target_format(view.target_format);
            let layer = layers.entry(view.retained_view_entity).or_insert_with(|| {
                make_retained_layer(&render_device, plan.rect, plan.full_size, format)
            });
            if layer.rect.size() != plan.rect.size()
                || layer.format != format
                || layer.full_size != plan.full_size
            {
                *layer = make_retained_layer(&render_device, plan.rect, plan.full_size, format);
            }
            if layer.rect != plan.rect {
                layer
                    .valid
                    .store(false, core::sync::atomic::Ordering::Relaxed);
                layer.rect = plan.rect;
            }
            let cache_state = if plan.contents.is_none() {
                "uncacheable"
            } else if !layer.valid.load(core::sync::atomic::Ordering::Relaxed) {
                "invalid"
            } else if layer.contents == plan.contents {
                "reused"
            } else if layer
                .contents
                .as_ref()
                .zip(plan.contents.as_ref())
                .is_some_and(|(a, b)| a.vertices != b.vertices)
            {
                "vertices_changed"
            } else {
                "bindings_or_order_changed"
            };
            layer.reuse = layer.valid.load(core::sync::atomic::Ordering::Relaxed)
                && plan.contents.is_some()
                && layer.contents == plan.contents;
            layer
                .valid
                .store(layer.reuse, core::sync::atomic::Ordering::Relaxed);
            layer.contents.clone_from(&plan.contents);
            commands.entity(entity).insert(SrgbUiTexture {
                texture: layer.texture.clone(),
                rect: Some(plan.rect),
                raster: Some(layer.raster.clone()),
            });
            commands
                .entity(entity)
                .insert(crate::UiRetainedLayerStatus {
                    rect: plan.rect,
                    full_size: bevy_math::UVec2::new(view.viewport.z, view.viewport.w),
                    reused: layer.reuse,
                    cache_state,
                });
            continue;
        }
        // UI uses the camera's offset viewport while drawing. Keep the full
        // attachment so that offset viewports fit and composite in place.
        let size = cameras
            .get(camera.0)
            .expect("extracted UI camera")
            .physical_target_size
            .expect("UI camera target size")
            .to_extents();

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

        commands.entity(entity).insert(SrgbUiTexture {
            texture,
            rect: None,
            raster: None,
        });
    }
}

fn make_retained_layer(
    device: &RenderDevice,
    rect: bevy_math::URect,
    full_size: bevy_math::UVec2,
    format: TextureFormat,
) -> crate::retained_layer::Layer {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("bounded_srgb_ui_texture"),
        size: Extent3d {
            width: rect.width(),
            height: rect.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let default_view = texture.create_view(&TextureViewDescriptor::default());
    let raster = device.create_texture(&TextureDescriptor {
        label: Some("retained_reference_ui_raster"),
        size: Extent3d {
            width: full_size.x,
            height: full_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    crate::retained_layer::Layer {
        texture: CachedTexture {
            texture,
            default_view,
        },
        rect,
        raster: CachedTexture {
            default_view: raster.create_view(&TextureViewDescriptor::default()),
            texture: raster,
        },
        full_size,
        format,
        contents: None,
        valid: core::sync::atomic::AtomicBool::new(false),
        reuse: false,
    }
}

pub fn queue_srgb_ui_composite_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SrgbUiCompositePipeline>>,
    pipeline: Res<SrgbUiCompositePipeline>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            Has<UiCompositeLookup>,
            &crate::UiShaderVariants,
        ),
        With<UiViewTarget>,
    >,
) {
    for (entity, view, lookup, variants) in &views {
        if variants.linear_color {
            continue;
        }
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SrgbUiCompositePipelineKey {
                format: view.target_format,
                relaxed_precision: variants.relaxed_precision,
                relaxed_colors: variants.relaxed_colors,
                lookup: lookup
                    && ui_render_target_format(view.target_format) == TextureFormat::Rgba8Unorm,
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
    views: Query<(Entity, &SrgbUiTexture, Has<UiCompositeLookup>)>,
) {
    for (entity, srgb_texture, lookup) in &views {
        let bind_group =
            if lookup && srgb_texture.texture.texture.format() == TextureFormat::Rgba8Unorm {
                render_device.create_bind_group(
                    "srgb_ui_lookup_bind_group",
                    &pipeline_cache.get_bind_group_layout(&composite_pipeline.lookup_layout),
                    &BindGroupEntries::sequential((
                        &srgb_texture.texture.default_view,
                        &composite_pipeline.sampler,
                        &composite_pipeline.lookup,
                    )),
                )
            } else {
                render_device.create_bind_group(
                    "srgb_ui_composite_bind_group",
                    &pipeline_cache.get_bind_group_layout(&composite_pipeline.layout),
                    &BindGroupEntries::sequential((
                        &srgb_texture.texture.default_view,
                        &composite_pipeline.sampler,
                    )),
                )
            };

        commands
            .entity(entity)
            .insert(SrgbUiCompositeBindGroup { bind_group });
    }
}
