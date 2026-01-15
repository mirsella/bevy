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
#[cfg(feature = "bevy_text")]
mod text2d;
mod texture_slice;
mod tilemap_chunk;

/// The sprite prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{ColorMaterial, MeshMaterial2d};
}

use bevy_shader::load_shader_library;
pub use mesh2d::*;
pub use render::*;
pub(crate) use texture_slice::*;
pub use tilemap_chunk::*;

use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, AssetEventSystems};
use bevy_core_pipeline::core_2d::{
    graph::{Core2d, Node2d},
    AlphaMask2d, Opaque2d, Transparent2d,
};
use bevy_ecs::prelude::*;
use bevy_image::{prelude::*, TextureAtlasPlugin};
use bevy_mesh::Mesh2d;
use bevy_render::{
    batching::sort_binned_render_phase,
    render_graph::{RenderGraphExt, ViewNodeRunner},
    render_phase::{sort_phase_system, AddRenderCommand, DrawFunctions, ViewSortedRenderPhases},
    render_resource::SpecializedRenderPipelines,
    sync_world::SyncToRenderWorld,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
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

/// Deprecated alias for [`SpriteSystems`].
#[deprecated(since = "0.17.0", note = "Renamed to `SpriteSystems`.")]
pub type SpriteSystem = SpriteSystems;

impl Plugin for SpriteRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "render/sprite_view_bindings.wgsl");

        embedded_asset!(app, "render/sprite.wgsl");
        embedded_asset!(app, "render/srgb_composite.wgsl");

        if !app.is_plugin_added::<TextureAtlasPlugin>() {
            app.add_plugins(TextureAtlasPlugin);
        }

        // Initialize render app resources BEFORE adding sub-plugins that depend on them.
        // ColorMaterialPlugin and others call add_render_command::<SrgbTransparent2d, ...>()
        // which requires DrawFunctions<SrgbTransparent2d> to already exist.
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<SrgbTransparent2d>>()
                .init_resource::<ViewSortedRenderPhases<SrgbTransparent2d>>()
                .init_resource::<ImageBindGroups>()
                .init_resource::<SpecializedRenderPipelines<SpritePipeline>>()
                .init_resource::<SpecializedRenderPipelines<SrgbSpritePipeline>>()
                .init_resource::<SpecializedRenderPipelines<SrgbCompositePipeline>>()
                .init_resource::<SpriteMeta>()
                .init_resource::<ExtractedSprites>()
                .init_resource::<ExtractedSlices>()
                .init_resource::<SpriteAssetEvents>()
                .init_resource::<SpriteBatches>()
                .add_render_command::<Transparent2d, DrawSprite>()
                .add_render_command::<SrgbTransparent2d, DrawSprite>()
                .add_systems(RenderStartup, init_sprite_pipeline)
                .add_systems(
                    RenderStartup,
                    init_srgb_sprite_pipeline.after(init_sprite_pipeline),
                )
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
                        prepare_srgb_composite_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                        sort_binned_render_phase::<Opaque2d>.in_set(RenderSystems::PhaseSort),
                        sort_binned_render_phase::<AlphaMask2d>.in_set(RenderSystems::PhaseSort),
                        sort_phase_system::<SrgbTransparent2d>.in_set(RenderSystems::PhaseSort),
                    ),
                )
                .add_render_graph_node::<ViewNodeRunner<SrgbSpritePassNode>>(Core2d, SrgbSpritePass)
                .add_render_graph_node::<ViewNodeRunner<SrgbCompositePassNode>>(
                    Core2d,
                    SrgbCompositePass,
                )
                .add_render_graph_edge(Core2d, Node2d::MainOpaquePass, SrgbSpritePass)
                .add_render_graph_edge(Core2d, SrgbSpritePass, SrgbCompositePass)
                .add_render_graph_edge(Core2d, SrgbCompositePass, Node2d::MainTransparentPass);
        }

        app.add_plugins((
            Mesh2dRenderPlugin,
            ColorMaterialPlugin,
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
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<SrgbCompositePipeline>();
        }
    }
}
