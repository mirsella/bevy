use super::ExtractedWindows;
use crate::{
    camera::NormalizedRenderTargetExt,
    gpu_readback,
    render_asset::RenderAssets,
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, BufferUsages, CachedPipelineState, PipelineCache,
        SpecializedRenderPipeline, SpecializedRenderPipelines, Texture, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::{GpuImage, ManualTextureViews, OutputColorAttachment},
    view::{prepare_view_attachments, prepare_view_targets, ViewTargetAttachments},
    ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
};
use alloc::{borrow::Cow, sync::Arc};
use bevy_app::{First, Plugin, Update};
use bevy_asset::{
    embedded_asset, load_embedded_asset, AssetServer, Assets, Handle, RenderAssetUsages,
};
use bevy_camera::{ManualTextureViewHandle, NormalizedRenderTarget, RenderTarget};
use bevy_derive::{Deref, DerefMut};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    entity::EntityHashMap, message::message_update_system, prelude::*, system::SystemState,
};
use bevy_image::{Image, TextureFormatPixelInfo, ToExtents};
use bevy_log::{error, info, warn};
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_2d, BindGroupLayoutEntries},
    descriptor::{
        BindGroupLayoutDescriptor, CachedRenderPipelineId, FragmentState, RenderPipelineDescriptor,
        VertexState,
    },
};
use bevy_math::UVec2;
use bevy_platform::collections::HashSet;
use bevy_reflect::Reflect;
use bevy_shader::{Shader, ShaderCacheError};
use bevy_tasks::AsyncComputeTaskPool;
use bevy_utils::default;
use bevy_window::{PrimaryWindow, Window, WindowRef};
use std::{
    path::Path,
    sync::{
        mpsc::{Receiver, Sender},
        Mutex,
    },
};
use wgpu::{CommandEncoder, Extent3d, TextureFormat};

#[derive(EntityEvent, Reflect, Deref, DerefMut, Debug)]
#[reflect(Debug, Event)]
pub struct ScreenshotCaptured {
    pub entity: Entity,
    #[deref]
    pub image: Image,
    /// Main-world [`FrameCount`] at extraction of the frame that was captured, not the
    /// request or readback completion frame. Retries use the successful attempt's frame.
    /// Since `FrameCount` increments in `Last`, a request made in `Update` at count `N`
    /// normally captures count `N.wrapping_add(1)`.
    pub frame_count: u32,
}

/// A component that signals to the renderer to capture a screenshot this frame.
///
/// This component should be spawned on a new entity with an observer that will trigger
/// with [`ScreenshotCaptured`] when the screenshot is ready.
///
/// Screenshots are captured asynchronously and may not be available immediately after the frame
/// that the component is spawned on. The observer should be used to handle the screenshot when it
/// is ready.
/// If the capture pipeline is not ready, capture is deferred to a later frame. Frame-specific
/// metadata (such as privacy masks) must not assume the captured image is from the request frame.
/// Use [`ScreenshotCaptured::frame_count`] to identify the captured extraction frame.
/// A permanent pipeline error is logged and the request is despawned without a capture event.
///
/// Note that the screenshot entity will be despawned after the screenshot is captured and the
/// observer is triggered.
///
/// # Usage
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_render::view::screenshot::{save_to_disk, Screenshot};
///
/// fn take_screenshot(mut commands: Commands) {
///    commands.spawn(Screenshot::primary_window())
///       .observe(save_to_disk("screenshot.png"));
/// }
/// ```
#[derive(Component, Deref, DerefMut, Reflect, Debug)]
#[reflect(Component, Debug)]
pub struct Screenshot(pub RenderTarget);

/// The dimensions of the image returned for a [`Screenshot`].
///
/// Rendering still occurs at the target's native resolution. The completed frame is resized on the
/// GPU before readback, so this does not affect cameras, UI layout, or the presented image.
/// Both dimensions must be nonzero and no larger than the render target.
#[derive(Component, Deref, DerefMut, Reflect, Debug)]
#[reflect(Component, Debug)]
pub struct ScreenshotResolution(pub UVec2);

/// A marker component that indicates that a screenshot is currently being captured.
#[derive(Component, Default)]
pub struct Capturing;

/// A marker component that indicates that a screenshot has been captured, the image is ready, and
/// the screenshot entity can be despawned.
#[derive(Component, Default)]
pub struct Captured;

impl Screenshot {
    /// Capture a screenshot of the provided window entity.
    pub fn window(window: Entity) -> Self {
        Self(RenderTarget::Window(WindowRef::Entity(window)))
    }

    /// Capture a screenshot of the primary window, if one exists.
    pub fn primary_window() -> Self {
        Self(RenderTarget::Window(WindowRef::Primary))
    }

    /// Capture a screenshot of the provided render target image.
    pub fn image(image: Handle<Image>) -> Self {
        Self(RenderTarget::Image(image.into()))
    }

    /// Capture a screenshot of the provided manual texture view.
    pub fn texture_view(texture_view: ManualTextureViewHandle) -> Self {
        Self(RenderTarget::TextureView(texture_view))
    }
}

struct ScreenshotPreparedState {
    texture: Texture,
    resized_texture: Option<(Texture, TextureView)>,
    buffer: Buffer,
    bind_group: BindGroup,
    pipeline_id: CachedRenderPipelineId,
    size: Extent3d,
    frame_count: u32,
}

struct RenderScreenshotTarget {
    target: NormalizedRenderTarget,
    resolution: Option<UVec2>,
    failed: bool,
    frame_count: u32,
}

#[derive(Resource, Deref, DerefMut)]
pub struct CapturedScreenshots(pub Arc<Mutex<Receiver<ScreenshotCaptured>>>);

#[derive(Resource, Deref, DerefMut, Default)]
struct RenderScreenshotTargets(EntityHashMap<RenderScreenshotTarget>);

#[derive(Resource, Deref, DerefMut, Default)]
struct RenderScreenshotsPrepared(EntityHashMap<ScreenshotPreparedState>);

#[derive(Resource, Deref, DerefMut)]
struct RenderScreenshotsSender(Sender<ScreenshotCaptured>);

/// Saves the captured screenshot to disk at the provided path.
pub fn save_to_disk(path: impl AsRef<Path>) -> impl FnMut(On<ScreenshotCaptured>) {
    let path = path.as_ref().to_owned();
    move |screenshot_captured| {
        let img = screenshot_captured.image.clone();
        match img.try_into_dynamic() {
            Ok(dyn_img) => match image::ImageFormat::from_path(&path) {
                Ok(format) => {
                    // discard the alpha channel which stores brightness values when HDR is enabled to make sure
                    // the screenshot looks right
                    let img = dyn_img.to_rgb8();
                    #[cfg(not(target_arch = "wasm32"))]
                    match img.save_with_format(&path, format) {
                        Ok(_) => info!("Screenshot saved to {}", path.display()),
                        Err(e) => error!("Cannot save screenshot, IO error: {e}"),
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        let save_screenshot = || {
                            use image::EncodableLayout;
                            use wasm_bindgen::{JsCast, JsValue};

                            let mut image_buffer = std::io::Cursor::new(Vec::new());
                            img.write_to(&mut image_buffer, format)
                                .map_err(|e| JsValue::from_str(&format!("{e}")))?;

                            let parts = js_sys::Array::of1(
                                &js_sys::Uint8Array::new_from_slice(
                                    image_buffer.into_inner().as_bytes(),
                                )
                                .into(),
                            );
                            let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
                            let url = web_sys::Url::create_object_url_with_blob(&blob)?;
                            let window = web_sys::window().unwrap();
                            let document = window.document().unwrap();
                            let link = document.create_element("a")?;
                            link.set_attribute("href", &url)?;
                            link.set_attribute(
                                "download",
                                path.file_name()
                                    .and_then(|filename| filename.to_str())
                                    .ok_or_else(|| JsValue::from_str("Invalid filename"))?,
                            )?;
                            let html_element = link.dyn_into::<web_sys::HtmlElement>()?;
                            html_element.click();
                            web_sys::Url::revoke_object_url(&url)?;
                            Ok::<(), JsValue>(())
                        };

                        match (save_screenshot)() {
                            Ok(_) => info!("Screenshot saved to {}", path.display()),
                            Err(e) => error!("Cannot save screenshot, error: {e:?}"),
                        };
                    }
                }
                Err(e) => error!("Cannot save screenshot, requested format not recognized: {e}"),
            },
            Err(e) => error!("Cannot save screenshot, screen format cannot be understood: {e}"),
        }
    }
}

fn clear_screenshots(mut commands: Commands, screenshots: Query<Entity, With<Captured>>) {
    for entity in screenshots.iter() {
        commands.entity(entity).despawn();
    }
}

pub fn trigger_screenshots(
    mut commands: Commands,
    captured_screenshots: ResMut<CapturedScreenshots>,
) {
    let captured_screenshots = captured_screenshots.lock().unwrap();
    while let Ok(capture) = captured_screenshots.try_recv() {
        commands.entity(capture.entity).insert(Captured);
        commands.trigger(capture);
    }
}

fn extract_screenshots(
    mut targets: ResMut<RenderScreenshotTargets>,
    prepared: Res<RenderScreenshotsPrepared>,
    mut main_world: ResMut<MainWorld>,
    mut system_state: Local<
        Option<
            SystemState<(
                Commands,
                Query<Entity, With<PrimaryWindow>>,
                Query<(Entity, &'static Window)>,
                Res<Assets<Image>>,
                Res<ManualTextureViews>,
                Query<(Entity, &Screenshot, Option<&ScreenshotResolution>), Without<Capturing>>,
            )>,
        >,
    >,
    mut seen_targets: Local<HashSet<NormalizedRenderTarget>>,
) {
    // Extraction reserves requests before preparation can establish pipeline readiness.
    // Release unprepared requests for retry, but leave successful captures awaiting readback.
    for (entity, request) in targets.drain() {
        if let Ok(mut entity) = main_world.get_entity_mut(entity) {
            if request.failed {
                entity.despawn();
            } else if !prepared.contains_key(&entity.id()) {
                entity.remove::<Capturing>();
            }
        }
    }
    let frame_count = main_world.resource::<FrameCount>().0;
    let system_state = system_state.get_or_insert_with(|| SystemState::new(&mut main_world));
    let (mut commands, primary_window, windows, images, manual_texture_views, screenshots) =
        system_state.get_mut(&mut main_world).unwrap();

    seen_targets.clear();

    let primary_window = primary_window.iter().next();

    for (entity, screenshot, resolution) in screenshots.iter() {
        let resolution = resolution.map(|resolution| resolution.0);
        if resolution.is_some_and(|resolution| resolution.x == 0 || resolution.y == 0) {
            error!("Screenshot resolution must be nonzero, skipping entity {entity}");
            commands.entity(entity).despawn();
            continue;
        }
        let render_target = screenshot.0.clone();
        let Some(render_target) = render_target.normalize(primary_window) else {
            warn!(
                "Unknown render target for screenshot, skipping: {:?}",
                render_target
            );
            continue;
        };
        if matches!(render_target, NormalizedRenderTarget::None { .. }) {
            error!("Cannot capture RenderTarget::None, skipping entity {entity}");
            commands.entity(entity).despawn();
            continue;
        }
        let Ok(info) =
            render_target.get_render_target_info(windows.iter(), &images, &manual_texture_views)
        else {
            // The requested asset or window may become available on a later frame.
            continue;
        };
        if let Some(resolution) = resolution
            && (resolution.x > info.physical_size.x || resolution.y > info.physical_size.y)
        {
            error!(
                "Screenshot resolution {resolution} exceeds render target size {}, skipping entity {entity}",
                info.physical_size
            );
            commands.entity(entity).despawn();
            continue;
        }
        if !seen_targets.insert(render_target.clone()) {
            warn!(
                "Duplicate render target for screenshot, skipping entity {}: {:?}",
                entity, render_target
            );
            // If we don't despawn the entity here, it will be captured again in the next frame
            commands.entity(entity).despawn();
            continue;
        }
        targets.insert(
            entity,
            RenderScreenshotTarget {
                target: render_target,
                resolution,
                failed: false,
                frame_count,
            },
        );
        commands.entity(entity).insert(Capturing);
    }

    system_state.apply(&mut main_world);
}

fn prepare_screenshots(
    mut targets: ResMut<RenderScreenshotTargets>,
    mut prepared: ResMut<RenderScreenshotsPrepared>,
    render_device: Res<RenderDevice>,
    screenshot_pipeline: Res<ScreenshotToScreenPipeline>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>,
    images: Res<RenderAssets<GpuImage>>,
    windows: Res<ExtractedWindows>,
    manual_texture_views: Res<ManualTextureViews>,
    mut view_target_attachments: ResMut<ViewTargetAttachments>,
) {
    prepared.clear();
    for (entity, request) in targets.iter_mut() {
        let size = match &request.target {
            NormalizedRenderTarget::Window(window) => {
                windows.get(&window.entity()).map(|window| Extent3d {
                    width: window.physical_width,
                    height: window.physical_height,
                    ..default()
                })
            }
            NormalizedRenderTarget::Image(image) => images
                .get(&image.handle)
                .map(|image| image.texture_descriptor.size),
            NormalizedRenderTarget::TextureView(texture_view) => manual_texture_views
                .get(texture_view)
                .map(|texture_view| texture_view.size.to_extents()),
            NormalizedRenderTarget::None { .. } => None,
        };
        let Some((size, view_format)) = size.zip(request.target.get_texture_view_format(
            &windows,
            &images,
            &manual_texture_views,
        )) else {
            warn!(target = ?request.target, "Unknown render target for screenshot, skipping");
            continue;
        };
        let pipeline_id = pipelines.specialize(&pipeline_cache, &screenshot_pipeline, view_format);
        pipeline_cache.block_on_render_pipeline(pipeline_id);
        match pipeline_cache.get_render_pipeline_state(pipeline_id) {
            CachedPipelineState::Ok(_) => {}
            CachedPipelineState::Queued
            | CachedPipelineState::Creating(_)
            | CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            ) => continue,
            CachedPipelineState::Err(err) => {
                error!("Screenshot pipeline failed for entity {entity}: {err}");
                request.failed = true;
                continue;
            }
        }
        let output_size = request.resolution.map_or(size, |resolution| Extent3d {
            width: resolution.x,
            height: resolution.y,
            ..default()
        });
        assert!(
            output_size.width <= size.width && output_size.height <= size.height,
            "screenshot resolution must not exceed the render target"
        );
        let texture = render_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot-capture-rendertarget"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: view_format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let resized_texture = (output_size != size).then(|| {
            let texture = render_device.create_texture(&wgpu::TextureDescriptor {
                label: Some("screenshot-resized-rendertarget"),
                size: output_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: view_format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            (texture, view)
        });
        let buffer = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot-transfer-buffer"),
            size: gpu_readback::get_aligned_size(
                output_size,
                view_format.pixel_size().unwrap_or(0) as u32,
            ) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = render_device.create_bind_group(
            "screenshot-to-screen-bind-group",
            &pipeline_cache.get_bind_group_layout(&screenshot_pipeline.bind_group_layout),
            &BindGroupEntries::single(&texture_view),
        );
        prepared.insert(
            *entity,
            ScreenshotPreparedState {
                texture,
                resized_texture,
                buffer,
                bind_group,
                pipeline_id,
                size: output_size,
                frame_count: request.frame_count,
            },
        );
        view_target_attachments.insert(
            request.target.clone(),
            OutputColorAttachment::new(texture_view, view_format),
        );
    }
}

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        embedded_asset!(app, "screenshot.wgsl");

        let (tx, rx) = std::sync::mpsc::channel();
        app.register_type::<Screenshot>()
            .register_type::<ScreenshotResolution>()
            .register_type::<ScreenshotCaptured>()
            .insert_resource(CapturedScreenshots(Arc::new(Mutex::new(rx))))
            .add_systems(
                First,
                clear_screenshots
                    .after(message_update_system)
                    .before(ApplyDeferred),
            )
            .add_systems(Update, trigger_screenshots);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(RenderScreenshotsSender(tx))
            .init_resource::<RenderScreenshotTargets>()
            .init_resource::<RenderScreenshotsPrepared>()
            .init_gpu_resource::<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>()
            .add_systems(RenderStartup, init_screenshot_to_screen_pipeline)
            .add_systems(ExtractSchedule, extract_screenshots.ambiguous_with_all())
            .add_systems(
                Render,
                prepare_screenshots
                    .after(prepare_view_attachments)
                    .before(prepare_view_targets)
                    .in_set(RenderSystems::PrepareViews),
            );
    }
}

#[derive(Resource)]
pub struct ScreenshotToScreenPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_screenshot_to_screen_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "screenshot-to-screen-bgl",
        &BindGroupLayoutEntries::single(
            wgpu::ShaderStages::FRAGMENT,
            texture_2d(wgpu::TextureSampleType::Float { filterable: false }),
        ),
    );

    let shader = load_embedded_asset!(asset_server.as_ref(), "screenshot.wgsl");

    commands.insert_resource(ScreenshotToScreenPipeline {
        bind_group_layout,
        shader,
    });
}

impl SpecializedRenderPipeline for ScreenshotToScreenPipeline {
    type Key = TextureFormat;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(Cow::Borrowed("screenshot-to-screen")),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                ..default()
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            multisample: Default::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(wgpu::ColorTargetState {
                    format: key,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

pub(crate) fn submit_screenshot_commands(world: &World, encoder: &mut CommandEncoder) {
    let targets = world.resource::<RenderScreenshotTargets>();
    let prepared = world.resource::<RenderScreenshotsPrepared>();
    let pipelines = world.resource::<PipelineCache>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let windows = world.resource::<ExtractedWindows>();
    let manual_texture_views = world.resource::<ManualTextureViews>();

    for (entity, request) in targets.iter() {
        let texture_view =
            request
                .target
                .get_texture_view(windows, gpu_images, manual_texture_views);
        render_screenshot(
            encoder,
            prepared,
            pipelines,
            entity,
            texture_view.map(|view| &**view),
        );
    }
}

fn render_screenshot(
    encoder: &mut CommandEncoder,
    prepared: &RenderScreenshotsPrepared,
    pipelines: &PipelineCache,
    entity: &Entity,
    texture_view: Option<&wgpu::TextureView>,
) {
    if let Some(prepared_state) = prepared.get(entity) {
        let pipeline = pipelines
            .get_render_pipeline(prepared_state.pipeline_id)
            .expect("screenshot pipeline should be compiled during preparation");

        if let Some((_, resized_view)) = &prepared_state.resized_texture {
            blit_screenshot(
                encoder,
                pipeline,
                &prepared_state.bind_group,
                resized_view,
                "screenshot_resize_pass",
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }

        let output_texture = prepared_state
            .resized_texture
            .as_ref()
            .map_or(&prepared_state.texture, |(texture, _)| texture);
        encoder.copy_texture_to_buffer(
            output_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &prepared_state.buffer,
                layout: gpu_readback::layout_data(
                    prepared_state.size,
                    prepared_state.texture.format(),
                ),
            },
            prepared_state.size,
        );

        if let Some(texture_view) = texture_view {
            blit_screenshot(
                encoder,
                pipeline,
                &prepared_state.bind_group,
                texture_view,
                "screenshot_to_screen_pass",
                wgpu::LoadOp::Load,
            );
        }
    }
}

fn blit_screenshot<'a>(
    encoder: &mut CommandEncoder,
    pipeline: &'a wgpu::RenderPipeline,
    bind_group: &'a BindGroup,
    target: &'a wgpu::TextureView,
    label: &'static str,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

pub(crate) fn collect_screenshots(world: &mut World) {
    #[cfg(feature = "trace")]
    let _span = bevy_log::info_span!("collect_screenshots").entered();

    let sender = world.resource::<RenderScreenshotsSender>().0.clone();
    let prepared = world.resource::<RenderScreenshotsPrepared>();

    for (entity, prepared) in prepared.iter() {
        let entity = *entity;
        let frame_count = prepared.frame_count;
        let sender = sender.clone();
        let width = prepared.size.width;
        let height = prepared.size.height;
        let texture_format = prepared.texture.format();
        let Ok(pixel_size) = texture_format.pixel_size() else {
            continue;
        };
        let buffer = prepared.buffer.clone();

        let finish = async move {
            let (tx, rx) = async_channel::bounded(1);
            let buffer_slice = buffer.slice(..);
            // The polling for this map call is done every frame when the command queue is submitted.
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                if let Err(err) = result {
                    panic!("{}", err.to_string());
                }
                tx.try_send(()).unwrap();
            });
            rx.recv().await.unwrap();
            let data = buffer_slice.get_mapped_range();
            // Move directly into tightly packed CPU memory instead of copying padding first.
            let row_bytes = width as usize * pixel_size;
            let buffered_row_bytes =
                gpu_readback::align_byte_size(width * pixel_size as u32) as usize;
            let result = if row_bytes == buffered_row_bytes {
                Vec::from(&*data)
            } else {
                let mut result = Vec::with_capacity(row_bytes * height as usize);
                for row in data.chunks_exact(buffered_row_bytes) {
                    result.extend_from_slice(&row[..row_bytes]);
                }
                result
            };
            drop(data);

            if let Err(e) = sender.send(ScreenshotCaptured {
                entity,
                frame_count,
                image: Image::new(
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    result,
                    texture_format,
                    RenderAssetUsages::MAIN_WORLD,
                ),
            }) {
                error!("Failed to send screenshot: {}", e);
            }
        };

        AsyncComputeTaskPool::get().spawn(finish).detach();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        extract_plugin::ExtractPlugin,
        renderer::{RenderAdapter, RenderQueue},
        texture::ManualTextureView,
    };
    use bevy_app::App;
    use bevy_ecs::system::RunSystemOnce;
    use bevy_tasks::{block_on, TaskPool};

    #[derive(Resource, Default)]
    struct Captures(Vec<(Entity, UVec2, u32)>);

    fn setup(resolution: Option<UVec2>, synchronous: bool) -> (App, Entity, Handle<Shader>) {
        AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            flags: default(),
            memory_budget_thresholds: default(),
            display: None,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true },
                ..default()
            },
        });
        let adapter = block_on(instance.request_adapter(&default())).unwrap();
        let (device, queue) = block_on(adapter.request_device(&default())).unwrap();
        let device = RenderDevice::from(device);
        let cache = PipelineCache::new(device.clone(), RenderAdapter::new(adapter), synchronous);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&default());
        let handle = ManualTextureViewHandle(0);
        let mut views = ManualTextureViews::default();
        views.insert(
            handle,
            ManualTextureView::with_default_format(view.clone(), UVec2::splat(4)),
        );
        let mut attachments = ViewTargetAttachments::default();
        attachments.insert(
            NormalizedRenderTarget::TextureView(handle),
            OutputColorAttachment::new(view, TextureFormat::Rgba8UnormSrgb),
        );
        let shader = Handle::<Shader>::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = App::new();
        app.add_plugins((ExtractPlugin::default(), bevy_diagnostic::FrameCountPlugin))
            .init_resource::<Assets<Image>>()
            .init_resource::<Captures>()
            .insert_resource(views.clone())
            .insert_resource(CapturedScreenshots(Arc::new(Mutex::new(rx))));
        let mut entity = app.world_mut().spawn(Screenshot::texture_view(handle));
        entity.observe(
            |capture: On<ScreenshotCaptured>, mut captures: ResMut<Captures>| {
                captures
                    .0
                    .push((capture.entity, capture.image.size(), capture.frame_count));
            },
        );
        if let Some(resolution) = resolution {
            entity.insert(ScreenshotResolution(resolution));
        }
        let entity = entity.id();
        app.sub_app_mut(RenderApp)
            .init_resource::<RenderScreenshotTargets>()
            .init_resource::<RenderScreenshotsPrepared>()
            .init_resource::<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>()
            .init_resource::<RenderAssets<GpuImage>>()
            .init_resource::<ExtractedWindows>()
            .insert_resource(views)
            .insert_resource(attachments)
            .insert_resource(device)
            .insert_resource(RenderQueue::new(queue))
            .insert_resource(cache)
            .insert_resource(RenderScreenshotsSender(tx))
            .insert_resource(ScreenshotToScreenPipeline {
                bind_group_layout: BindGroupLayoutDescriptor::new(
                    "screenshot-to-screen-bgl",
                    &BindGroupLayoutEntries::single(
                        wgpu::ShaderStages::FRAGMENT,
                        texture_2d(wgpu::TextureSampleType::Float { filterable: false }),
                    ),
                ),
                shader: shader.clone(),
            })
            .add_systems(ExtractSchedule, extract_screenshots);
        (app, entity, shader)
    }

    fn extract(app: &mut App) {
        let mut render = app.remove_sub_app(RenderApp).unwrap();
        render.extract(app.world_mut());
        app.insert_sub_app(RenderApp, render);
    }

    #[test]
    fn screenshot_waits_for_shader_and_retries_original_request() {
        for (resolution, synchronous) in [
            (None, true),
            (None, false),
            (Some(UVec2::splat(2)), true),
            (Some(UVec2::splat(2)), false),
        ] {
            let (mut app, entity, shader) = setup(resolution, synchronous);
            let target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
            let original_view = app
                .sub_app(RenderApp)
                .world()
                .resource::<ViewTargetAttachments>()[&target]
                .view
                .id();
            // Simulate inactive startup: main schedules run, but extraction has not happened.
            let creation_frame = app.world().resource::<FrameCount>().0;
            app.main_mut().run_default_schedule();
            app.main_mut().run_default_schedule();
            assert!(!app.world().entity(entity).contains::<Capturing>());
            for _ in 0..3 {
                let extraction_frame = app.world().resource::<FrameCount>().0;
                extract(&mut app);
                assert!(app.world().entity(entity).contains::<Capturing>());
                let world = app.sub_app_mut(RenderApp).world_mut();
                assert!(world
                    .resource::<RenderScreenshotTargets>()
                    .contains_key(&entity));
                assert_eq!(
                    world.resource::<RenderScreenshotTargets>()[&entity].frame_count,
                    extraction_frame
                );
                assert_ne!(extraction_frame, creation_frame.wrapping_add(1));
                world.run_system_once(prepare_screenshots).unwrap();
                assert!(world.resource::<RenderScreenshotsPrepared>().is_empty());
                assert_eq!(
                    world.resource::<ViewTargetAttachments>()[&target].view.id(),
                    original_view
                );
                world.resource_mut::<PipelineCache>().process_queue();
                let mut encoder = world
                    .resource::<RenderDevice>()
                    .create_command_encoder(&default());
                submit_screenshot_commands(world, &mut encoder);
                app.main_mut().run_default_schedule();
            }

            let world = app.sub_app_mut(RenderApp).world_mut();
            world.resource_mut::<PipelineCache>().set_shader(
                shader.id(),
                Shader::from_wgsl(include_str!("screenshot.wgsl"), "screenshot.wgsl"),
            );
            world.resource_mut::<PipelineCache>().process_queue();
            let capture_frame = app.world().resource::<FrameCount>().0;
            extract(&mut app);
            let world = app.sub_app_mut(RenderApp).world_mut();
            world.run_system_once(prepare_screenshots).unwrap();
            assert!(world
                .resource::<RenderScreenshotsPrepared>()
                .contains_key(&entity));
            assert_eq!(
                world.resource::<RenderScreenshotsPrepared>()[&entity].frame_count,
                capture_frame
            );
            assert_ne!(
                world.resource::<ViewTargetAttachments>()[&target].view.id(),
                original_view
            );
            let mut encoder = world
                .resource::<RenderDevice>()
                .create_command_encoder(&default());
            submit_screenshot_commands(world, &mut encoder);
            world.resource::<RenderQueue>().submit([encoder.finish()]);
            collect_screenshots(world);
            // A successful request must not be re-extracted while readback is in flight.
            app.main_mut().run_default_schedule();
            extract(&mut app);
            assert!(app
                .sub_app(RenderApp)
                .world()
                .resource::<RenderScreenshotTargets>()
                .is_empty());
            assert!(app.world().entity(entity).contains::<Capturing>());

            let deadline = std::time::Instant::now() + core::time::Duration::from_secs(5);
            while app.world().resource::<Captures>().0.is_empty() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "screenshot readback timed out"
                );
                app.sub_app(RenderApp)
                    .world()
                    .resource::<RenderDevice>()
                    .poll(wgpu::PollType::Poll)
                    .unwrap();
                AsyncComputeTaskPool::get().with_local_executor(|executor| {
                    executor.try_tick();
                });
                app.world_mut()
                    .run_system_once(trigger_screenshots)
                    .unwrap();
                std::thread::yield_now();
            }
            assert_eq!(
                app.world().resource::<Captures>().0,
                [(entity, resolution.unwrap_or(UVec2::splat(4)), capture_frame)]
            );
            assert_ne!(capture_frame, creation_frame.wrapping_add(1));
            assert_ne!(capture_frame, app.world().resource::<FrameCount>().0);
            app.world_mut().run_system_once(clear_screenshots).unwrap();
            assert!(app.world().get_entity(entity).is_err());
        }
    }

    #[test]
    fn screenshot_permanent_shader_error_cancels_request() {
        let (mut app, entity, shader) = setup(None, true);
        app.sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<PipelineCache>()
            .set_shader(
                shader.id(),
                Shader::from_wgsl("invalid wgsl", "broken.wgsl"),
            );
        extract(&mut app);
        let world = app.sub_app_mut(RenderApp).world_mut();
        world.run_system_once(prepare_screenshots).unwrap();
        assert!(world.resource::<RenderScreenshotsPrepared>().is_empty());
        assert!(world.resource::<RenderScreenshotTargets>()[&entity].failed);
        extract(&mut app);
        assert!(app.world().get_entity(entity).is_err());
        assert!(app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderScreenshotTargets>()
            .is_empty());
        assert!(app.world().resource::<Captures>().0.is_empty());
    }
}
