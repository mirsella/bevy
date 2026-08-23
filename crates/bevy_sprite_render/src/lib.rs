#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

//! Provides 2D sprite rendering functionality.

extern crate alloc;

mod mesh2d;
mod render;
mod sprite_mesh;
#[cfg(feature = "bevy_text")]
mod text2d;
mod texture_slice;
mod tilemap_chunk;

/// The sprite prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{ColorMaterial, MeshMaterial2d, SpriteMaterial};
}

use bevy_shader::{load_shader_library, Shader};
pub use mesh2d::*;
pub use render::*;
pub use sprite_mesh::*;
pub(crate) use texture_slice::*;
pub use tilemap_chunk::*;

use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetEventSystems, Handle};
use bevy_core_pipeline::{
    core_2d::{main_opaque_pass_2d, main_transparent_pass_2d, AlphaMask2d, Opaque2d},
    schedule::{Core2d, Core2dSystems},
};
use bevy_ecs::prelude::*;
use bevy_image::{prelude::*, TextureAtlasPlugin};
use bevy_mesh::Mesh2d;
use bevy_render::{
    batching::sort_binned_render_phase,
    render_phase::AddRenderCommand,
    render_phase::{sort_phase_system, DrawFunctions, ViewSortedRenderPhases},
    render_resource::SpecializedRenderPipelines,
    sync_world::SyncToRenderWorld,
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_sprite::Sprite;

#[cfg(feature = "bevy_text")]
pub use crate::text2d::extract_text2d_sprite;

/// Adds support for 2D sprite rendering.
#[derive(Default)]
pub struct SpriteRenderPlugin;

/// System set for sprite rendering.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum SpriteSystems {
    ExtractSprites,
    ComputeSlices,
}

#[derive(Resource)]
struct SrgbCompositeShader {
    _handle: Handle<Shader>,
}

impl Plugin for SpriteRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "render/sprite_view_bindings.wgsl");

        embedded_asset!(app, "render/sprite.wgsl");
        embedded_asset!(app, "render/srgb_composite.wgsl");
        app.insert_resource(SrgbCompositeShader {
            _handle: load_embedded_asset!(app, "render/srgb_composite.wgsl"),
        });

        if !app.is_plugin_added::<TextureAtlasPlugin>() {
            app.add_plugins(TextureAtlasPlugin);
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<SrgbTransparent2d>>()
                .init_resource::<ViewSortedRenderPhases<SrgbTransparent2d>>()
                .allow_ambiguous_resource::<ViewSortedRenderPhases<SrgbTransparent2d>>();
        }

        app.add_plugins((
            Mesh2dRenderPlugin,
            ColorMaterialPlugin,
            SpriteMeshPlugin,
            TilemapChunkPlugin,
            TilemapChunkMaterialPlugin,
        ))
        .add_systems(
            PostUpdate,
            (
                compute_slices_on_asset_event.before(AssetEventSystems),
                compute_slices_on_sprite_change,
            )
                .in_set(SpriteSystems::ComputeSlices),
        );

        app.register_required_components::<Sprite, SyncToRenderWorld>();

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ImageBindGroups>()
                .init_gpu_resource::<SpecializedRenderPipelines<SpritePipeline>>()
                .init_gpu_resource::<SpecializedRenderPipelines<SrgbCompositePipeline>>()
                .init_resource::<SpriteMeta>()
                .init_resource::<ExtractedSprites>()
                .init_resource::<ExtractedSlices>()
                .init_resource::<SpriteAssetEvents>()
                .init_resource::<SpriteBatches>()
                .add_render_command::<SrgbTransparent2d, DrawSprite>()
                .add_systems(RenderStartup, init_sprite_pipeline)
                .add_systems(RenderStartup, init_srgb_composite_pipeline)
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_sprites.in_set(SpriteSystems::ExtractSprites),
                        extract_sprite_events,
                        extract_srgb_sprite_camera_phases,
                        #[cfg(feature = "bevy_text")]
                        extract_text2d_sprite.after(SpriteSystems::ExtractSprites),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        queue_sprites
                            .in_set(RenderSystems::Queue)
                            .ambiguous_with(queue_material2d_meshes::<ColorMaterial>),
                        queue_srgb_composite_pipelines.in_set(RenderSystems::Queue),
                        prepare_sprite_image_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                        prepare_sprite_view_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                        prepare_srgb_sprite_textures.in_set(RenderSystems::PrepareResources),
                        sort_binned_render_phase::<Opaque2d>.in_set(RenderSystems::PhaseSort),
                        sort_binned_render_phase::<AlphaMask2d>.in_set(RenderSystems::PhaseSort),
                        sort_phase_system::<SrgbTransparent2d>.in_set(RenderSystems::PhaseSort),
                    ),
                )
                .add_systems(
                    Core2d,
                    (srgb_sprite_pass, srgb_composite_pass)
                        .chain()
                        .after(main_opaque_pass_2d)
                        .before(main_transparent_pass_2d)
                        .in_set(Core2dSystems::MainPass),
                );
        };
    }
}
