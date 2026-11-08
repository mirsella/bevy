//! Opt-in fusion of gamma UI compositing with final output. HDR, MSAA and
//! postprocessing before UI retain their normal ordering and resolved textures.
//! Earlier camera composites remain materialized when later cameras need them.
//! Custom effects after UI must add [`UiOutputFusionBarrier`]: their inputs cannot
//! be removed without changing the effect. External consumers of main textures
//! must likewise declare a barrier. Output textures retain their write timing.
use crate::{SrgbUiTexture, TransparentUi, UiCameraView, UiShaderVariants};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::{CameraOutputMode, ClearColor, ClearColorConfig, CompositingSpace};
use bevy_color::LinearRgba;
use bevy_core_pipeline::{
    core_2d::{AlphaMask2d, Opaque2d, Transparent2d},
    schedule::{Core2d, Core3d},
    tonemapping::Tonemapping,
    upscaling::view_output_blend,
    FullscreenShader,
};
use bevy_ecs::{prelude::*, query::QueryData, schedule::ScheduleLabel};
use bevy_math::{URect, UVec2, UVec4};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::{ExtractedCamera, SortedCameras},
    diagnostic::RecordDiagnostics,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_phase::{ViewBinnedRenderPhases, ViewSortedRenderPhases},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{ExtractedView, Msaa, SkipViewUpscaling, ViewTarget},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_sprite_render::SrgbTransparent2d;
use bevy_utils::default;

#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct UiOutputFusion;

/// All cameras in a deferred stack must declare this contract: no scene effects
/// or external main-texture readers, and the base scene remains fully opaque.
/// Later cameras must be UI-only (verified from geometry, clear and tonemapping).
/// Ordinary terminal-camera fusion needs no such declaration.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct UiOutputFusionStack;

/// Keep UI materialized across the shared main-texture chain for post-UI effects
/// or external main-texture consumers, including readers through another camera.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct UiOutputFusionBarrier;

#[derive(Resource, Default)]
pub struct UiOutputFusionStatus {
    pub requested_views: usize,
    pub active_views: usize,
    pub fallback: Option<&'static str>,
    /// Requested views retaining the reference composite, with the reason.
    pub fallbacks: Vec<(Entity, &'static str)>,
}

pub struct UiOutputFusionPlugin;
impl Plugin for UiOutputFusionPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "output_fusion.wgsl");
        app.add_plugins((
            ExtractComponentPlugin::<UiOutputFusion>::default(),
            ExtractComponentPlugin::<UiOutputFusionStack>::default(),
            ExtractComponentPlugin::<UiOutputFusionBarrier>::default(),
        ));
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<UiOutputFusionStatus>()
                .init_gpu_resource::<SpecializedRenderPipelines<FusionPipeline>>()
                .add_systems(RenderStartup, init)
                .add_systems(
                    Render,
                    prepare
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(crate::prepare_srgb_ui_composite_bind_groups),
                )
                .add_systems(
                    Core2d,
                    draw.after(crate::srgb_ui_composite_pass)
                        .before(bevy_core_pipeline::upscaling::upscaling),
                )
                .add_systems(
                    Core3d,
                    draw.after(crate::srgb_ui_composite_pass)
                        .before(bevy_core_pipeline::upscaling::upscaling),
                );
        }
    }
}

#[derive(Resource)]
struct FusionPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
}

fn init(
    mut commands: Commands,
    device: Res<RenderDevice>,
    assets: Res<AssetServer>,
    fullscreen: Res<FullscreenShader>,
) {
    commands.insert_resource(FusionPipeline {
        layout: BindGroupLayoutDescriptor::new(
            "UI output fusion",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<UiRect>(false),
                ),
            ),
        ),
        sampler: device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }),
        shader: load_embedded_asset!(assets.as_ref(), "output_fusion.wgsl"),
        fullscreen: fullscreen.clone(),
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    output: TextureFormat,
    scene: TextureFormat,
    space: Option<CompositingSpace>,
    blend: Option<BlendState>,
    base: bool,
}

impl SpecializedRenderPipeline for FusionPipeline {
    type Key = PipelineKey;
    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut defs = Vec::new();
        if key.base {
            defs.push("BASE_SCENE".into());
        }
        match key.space {
            Some(CompositingSpace::Srgb) => defs.push("SRGB_TO_LINEAR".into()),
            Some(CompositingSpace::Oklab) => defs.push("OKLAB_TO_LINEAR".into()),
            _ => {}
        }
        if matches!(
            key.scene,
            TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm
        ) {
            defs.push("UNORM_SCENE".into());
        }
        if key.scene.is_srgb() {
            defs.push("SRGB_SCENE".into());
        }
        if key.scene == TextureFormat::Rgba16Float {
            defs.push("F16_SCENE".into());
        }
        if key.scene == TextureFormat::Rgb10a2Unorm {
            defs.push("RGB10A2_SCENE".into());
        }
        RenderPipelineDescriptor {
            label: Some("gamma-faithful UI output fusion".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.output,
                    blend: key.blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }
    }
}

#[derive(ShaderType)]
struct UiRect {
    bounds: UVec4,
}

#[derive(Clone)]
struct Layer {
    pipeline: CachedRenderPipelineId,
    ui: TextureViewId,
    rect: URect,
    // Both ping-pong bindings are prepared once, then selected after effects run.
    bindings: [(TextureViewId, BindGroup); 2],
}

#[derive(Component, Clone, Default)]
#[doc(hidden)]
pub struct Ready {
    // Earlier Skip cameras omit their composite; only the original writer draws.
    layers: Vec<Layer>,
}

#[derive(QueryData)]
struct FusionView {
    entity: Entity,
    camera: &'static ExtractedCamera,
    target: &'static ViewTarget,
    view: Option<&'static ExtractedView>,
    ui: Option<&'static UiCameraView>,
    requested: Has<UiOutputFusion>,
    stack: Has<UiOutputFusionStack>,
    barrier: Has<UiOutputFusionBarrier>,
    ready: Option<&'static Ready>,
    msaa: Option<&'static Msaa>,
    tonemapping: Option<&'static Tonemapping>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "plan substitutions across all consumers of each main texture"
)]
fn prepare(
    mut commands: Commands,
    sorted: Res<SortedCameras>,
    views: Query<FusionView>,
    ui: Query<(&ExtractedView, Option<&SrgbUiTexture>, &UiShaderVariants)>,
    ui_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    opaque: Res<ViewBinnedRenderPhases<Opaque2d>>,
    mask: Res<ViewBinnedRenderPhases<AlphaMask2d>>,
    transparent: Res<ViewSortedRenderPhases<Transparent2d>>,
    sprites: Res<ViewSortedRenderPhases<SrgbTransparent2d>>,
    clear: Res<ClearColor>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<FusionPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FusionPipeline>>,
    cache: Res<PipelineCache>,
    mut status: ResMut<UiOutputFusionStatus>,
) {
    status.requested_views = 0;
    status.active_views = 0;
    status.fallback = None;
    status.fallbacks.clear();
    let mut chains = HashMap::<_, Vec<_>>::default();
    for camera in &sorted.0 {
        if let Ok(view) = views.get(camera.entity) {
            // Include non-opted-in/custom cameras as consumers, not just eligible views.
            chains
                .entry((
                    view.target.main_texture().id(),
                    view.target.main_texture_other().id(),
                ))
                .or_default()
                .push(view);
        }
    }
    let mut active = HashSet::<Entity>::default();
    for chain in chains.values() {
        let requested: Vec<_> = chain
            .iter()
            .filter(|v| v.requested)
            .map(|v| v.entity)
            .collect();
        status.requested_views += requested.len();
        if requested.is_empty() {
            continue;
        }
        let last = chain.last().unwrap();
        let result = (|| -> Result<Vec<(Entity, Ready)>, &'static str> {
            if matches!(chain[0].camera.clear_color, ClearColorConfig::None) {
                return Err("main texture is preserved across frames");
            }
            if !last.requested
                || chain.iter().any(|view| view.barrier)
                || !matches!(last.camera.output_mode, CameraOutputMode::Write { .. })
            {
                return Err(
                    "later main-texture consumer or post-UI effect requires materialized UI",
                );
            }
            if last.camera.schedule != Core2d.intern() && last.camera.schedule != Core3d.intern() {
                return Err("custom camera schedule requires the reference composite");
            }
            let output = last
                .target
                .out_texture_view_format()
                .ok_or("missing output attachment")?;
            if !matches!(
                last.target.main_texture_format(),
                TextureFormat::Rgba8Unorm
                    | TextureFormat::Bgra8Unorm
                    | TextureFormat::Rgba8UnormSrgb
                    | TextureFormat::Bgra8UnormSrgb
                    | TextureFormat::Rgba16Float
                    | TextureFormat::Rgb10a2Unorm
            ) {
                return Err("main texture storage conversion is unsupported");
            }
            let native = last.camera.physical_target_size == last.camera.physical_viewport_size;
            let base_color = match chain[0].camera.clear_color {
                ClearColorConfig::Default => clear.0,
                ClearColorConfig::Custom(color) => color,
                ClearColorConfig::None => unreachable!(),
            };
            // Cross-camera deferral is explicit. Empty geometry alone cannot rule
            // out a custom postprocessor that reads preceding UI.
            let stacked = chain.len() > 1
                && native
                && LinearRgba::from(base_color).alpha >= 1.
                && matches!(
                    output,
                    TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb
                )
                && chain.iter().enumerate().all(|(i, v)| {
                    v.requested
                        && v.stack
                        && v.ui.and_then(|camera| ui.get(camera.0).ok()).is_none_or(
                            |(_, texture, variants)| {
                                !variants.linear_color
                                    && texture.is_none_or(|texture| texture.rect.is_none())
                            },
                        )
                        && v.camera.schedule == Core2d.intern()
                        && v.camera.viewport.is_none()
                        && !v.camera.hdr
                        && v.msaa == Some(&Msaa::Off)
                        && matches!(
                            v.target.compositing_space,
                            None | Some(CompositingSpace::Linear)
                        )
                        && matches!(
                            v.target.main_texture_format(),
                            TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb
                        )
                        && (i + 1 == chain.len()
                            || matches!(v.camera.output_mode, CameraOutputMode::Skip))
                        && (i == 0
                            || matches!(v.camera.clear_color, ClearColorConfig::None)
                                && v.tonemapping.is_none_or(|tone| *tone == Tonemapping::None)
                                && v.view.is_some_and(|view| {
                                    let key = &view.retained_view_entity;
                                    opaque.get(key).is_none_or(|p| p.is_empty())
                                        && mask.get(key).is_none_or(|p| p.is_empty())
                                        && transparent.get(key).is_none_or(|p| p.items.is_empty())
                                        && sprites.get(key).is_none_or(|p| p.items.is_empty())
                                }))
                })
                && matches!(
                    last.camera.output_mode,
                    CameraOutputMode::Write {
                        blend_state: None,
                        ..
                    }
                );
            let candidates = if stacked {
                chain.as_slice()
            } else {
                &chain[chain.len() - 1..]
            };
            let mut layers = Vec::new();
            for v in candidates {
                let Some((view, texture, variants)) = v.ui.and_then(|camera| ui.get(camera.0).ok())
                else {
                    continue;
                };
                if ui_phases
                    .get(&view.retained_view_entity)
                    .is_none_or(|p| p.items.is_empty())
                {
                    continue;
                }
                if variants.linear_color {
                    return Err("direct linear UI has no gamma composite to fuse");
                }
                let texture = texture.ok_or("gamma UI texture is not ready")?;
                let base = layers.is_empty();
                let id = pipelines.specialize(
                    &cache,
                    &pipeline,
                    PipelineKey {
                        output,
                        scene: last.target.main_texture_format(),
                        space: last.target.compositing_space,
                        blend: if base {
                            view_output_blend(Some(last.camera))
                        } else {
                            Some(BlendState::ALPHA_BLENDING)
                        },
                        base,
                    },
                );
                if cache.get_render_pipeline(id).is_none() {
                    return Err("fusion pipeline compiling");
                }
                let scene = last.target.main_texture();
                let rect = texture.rect.unwrap_or(URect::from_corners(
                    UVec2::ZERO,
                    UVec2::new(scene.width(), scene.height()),
                ));
                let ui_view = &texture.texture.default_view;
                let scene_views = [
                    last.target.main_texture_view(),
                    last.target.main_texture_other_view(),
                ];
                if let Some(old) = last.ready.and_then(|old| {
                    old.layers.iter().find(|old| {
                        old.pipeline == id
                            && old.ui == ui_view.id()
                            && old.rect == rect
                            && scene_views
                                .iter()
                                .all(|view| old.bindings.iter().any(|(id, _)| *id == view.id()))
                    })
                }) {
                    layers.push(old.clone());
                    continue;
                }
                let mut uniform = UniformBuffer::from(UiRect {
                    bounds: UVec4::from((rect.min, rect.size())),
                });
                uniform.write_buffer(&device, &queue);
                let bindings = scene_views.map(|scene| {
                    (
                        scene.id(),
                        device.create_bind_group(
                            "UI output fusion",
                            &cache.get_bind_group_layout(&pipeline.layout),
                            &BindGroupEntries::sequential((
                                ui_view,
                                scene,
                                &pipeline.sampler,
                                uniform.binding().unwrap(),
                            )),
                        ),
                    )
                });
                layers.push(Layer {
                    pipeline: id,
                    ui: ui_view.id(),
                    rect,
                    bindings,
                });
            }
            if layers.is_empty() {
                return Err("no UI composite to fuse");
            }
            let mut prepared: Vec<_> = candidates
                .iter()
                .map(|v| (v.entity, Ready::default()))
                .collect();
            prepared.last_mut().unwrap().1.layers = layers;
            Ok(prepared)
        })();
        match result {
            Ok(prepared) => {
                for (entity, ready) in prepared {
                    active.insert(entity);
                    commands.entity(entity).insert((ready, SkipViewUpscaling));
                }
                for entity in requested.iter().filter(|e| !active.contains(*e)) {
                    status
                        .fallbacks
                        .push((*entity, "earlier UI remains materialized for later cameras"));
                }
            }
            Err(reason) => status
                .fallbacks
                .extend(requested.iter().map(|entity| (*entity, reason))),
        }
    }
    status.active_views = active.len();
    status.fallback = status.fallbacks.first().map(|(_, reason)| *reason);
    for view in &views {
        if view.ready.is_some() && !active.contains(&view.entity) {
            commands
                .entity(view.entity)
                .remove::<(Ready, SkipViewUpscaling)>();
        }
    }
}

fn draw(
    view: ViewQuery<(Entity, &Ready, &ViewTarget, &ExtractedCamera)>,
    cache: Res<PipelineCache>,
    clear: Res<ClearColor>,
    mut ctx: RenderContext,
) {
    let (entity, ready, target, camera) = view.into_inner();
    if ready.layers.is_empty() {
        return;
    }
    let CameraOutputMode::Write { clear_color, .. } = camera.output_mode else {
        unreachable!("only original output writers draw fusion");
    };
    let clear_color = match clear_color {
        ClearColorConfig::Default => Some(clear.0.to_linear()),
        ClearColorConfig::Custom(color) => Some(color.to_linear()),
        ClearColorConfig::None => None,
    };
    let diagnostics = ctx.diagnostic_recorder();
    let span = diagnostics.as_deref().map(|d| {
        d.time_span(
            ctx.command_encoder(),
            format!("ui_output_fusion/camera_{}_view_{entity}", camera.order),
        )
    });
    {
        // Every layer writes the same attachment with the same scissor. Keep the
        // stack in one pass so tile-based GPUs need not store/reload it per layer.
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("gamma-faithful UI output fusion"),
            color_attachments: &[Some(
                target
                    .out_texture_color_attachment(clear_color)
                    .expect("prepared output attachment"),
            )],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Some(viewport) = &camera.viewport {
            pass.set_scissor_rect(
                viewport.physical_position.x,
                viewport.physical_position.y,
                viewport.physical_size.x,
                viewport.physical_size.y,
            );
        }
        let main_view = target.main_texture_view().id();
        for layer in &ready.layers {
            let binding = &layer
                .bindings
                .iter()
                .find(|(id, _)| *id == main_view)
                .expect("main texture belongs to its prepared ping-pong pair")
                .1;
            pass.set_render_pipeline(
                cache
                    .get_render_pipeline(layer.pipeline)
                    .expect("prepared fusion pipeline"),
            );
            pass.set_bind_group(0, binding, &[]);
            pass.draw(0..3, 0..1);
        }
    }
    if let Some(span) = span {
        span.end(ctx.command_encoder());
    }
}
