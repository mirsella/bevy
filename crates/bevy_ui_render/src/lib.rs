#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! Provides rendering functionality for `bevy_ui`.

pub mod box_shadow;
pub mod clipping;
mod composite_bounds;
mod retained_layer;
pub use retained_layer::{UiRetainedLayer, UiRetainedLayerStatus};
mod pipeline;
pub mod render_pass;
mod surface;
pub use surface::cache::{
    ui_surface_cache_pass, UiCachedSurfaces, UiSurfaceCachePass, UiSurfaceCacheStatus,
};
mod text;
pub mod ui_material;
mod ui_material_pipeline;
mod ui_mesh;
pub mod ui_texture_slice_pipeline;

#[cfg(feature = "bevy_ui_debug")]
mod debug_overlay;

use bevy_camera::visibility::InheritedVisibility;
use bevy_camera::{Camera, Camera2d, Camera3d, RenderTarget};
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_render::camera::{extract_cameras, CameraMainPassTextureFormats};
use bevy_shader::{load_shader_library, Shader};
use bevy_sprite_render::text_group::{
    TextGroupCache, TextGroupEffects, TextGroupPlugin, TextGroupStyleQuery,
};
use bevy_sprite_render::SpriteAssetEvents;
use bevy_text::{InheritedTextOpacity, TextOpacity};
use bevy_ui::widget::{
    ImageNode, ImageNodeSize, NodeImageMode, TextOutline, TextScroll, TextShadow, ViewportNode,
};
use bevy_ui::{
    CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiTargetCamera, Display, Node,
    OuterColor, Outline, ResolvedBorderRadius, UiGlobalTransform, VisualBox,
};

use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetEvent, AssetId, Assets, Handle};
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_core_pipeline::schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems};
use bevy_core_pipeline::upscaling::upscaling;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::SystemParam;
use bevy_image::{prelude::*, TRANSPARENT_IMAGE_HANDLE};
use bevy_math::{Affine2, FloatOrd, Mat4, Rect, UVec4, Vec2};
use bevy_render::{
    render_asset::RenderAssets,
    render_phase::{
        sort_phase_system, AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
        SortedPhaseItems, ViewSortedRenderPhases,
    },
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity, TemporaryRenderEntity},
    texture::GpuImage,
    view::{ExtractedView, RetainedViewEntity, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_sprite::BorderRect;
#[cfg(feature = "bevy_ui_debug")]
pub use debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

use surface::SurfacePlugin;

use bevy_platform::collections::{HashMap, HashSet};
use bevy_text::{
    text_effect_outline_extent, text_effect_shadow_offset, ComputedTextBlock, EditableText,
    EditableTextPlaceholderColor, PositionedGlyph, Strikethrough, StrikethroughColor,
    TextBackgroundColor, TextColor, TextCursorStyle, TextEffectLayer, TextEntity, TextFont,
    TextLayoutInfo, TextPaintOrder, Underline, UnderlineColor,
};
use bevy_transform::components::GlobalTransform;
use box_shadow::BoxShadowPlugin;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;

pub use pipeline::*;
pub mod output_fusion;
pub use render_pass::*;
pub use ui_material_pipeline::*;
use ui_texture_slice_pipeline::UiTextureSlicerPlugin;

use crate::clipping::{clip_polygon, rect_without_hole, rounded_inner_rect};
use crate::composite_bounds::{CompositeBounds, UiCompositeBounds};
use crate::text::{calculate_text_scroll_clip, extract_preedit_underlines, extract_text_cursor};
use crate::ui_mesh::UiMesh;

pub mod prelude {
    #[cfg(feature = "bevy_ui_debug")]
    pub use crate::debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

    pub use crate::{
        ui_material::*, ui_material_pipeline::UiMaterialPlugin, BoxShadowSamples, UiAntiAlias,
        UiCachedSurfaces, UiCompositeLookup, UiCompositeScissor, UiRetainedLayer, UiShaderVariants,
    };
}

/// Local Z offsets of "extracted nodes" for a given entity. These exist to allow rendering multiple "extracted nodes"
/// for a given source entity (ex: render both a background color _and_ a custom material for a given node).
///
/// When possible these offsets should be defined in _this_ module to ensure z-index coordination across contexts.
/// When this is _not_ possible, pick a suitably unique index unlikely to clash with other things (ex: `0.1826823` not `0.1`).
///
/// Offsets should be unique for a given node entity to avoid z fighting.
/// These should pretty much _always_ be larger than -0.5 and smaller than 0.5 to avoid clipping into nodes
/// above / below the current node in the stack.
///
/// A z-index of 0.0 is the baseline, which is used as the primary "background color" of the node.
///
/// Note that nodes "stack" on each other, so a negative offset on the node above could clip _into_
/// a positive offset on a node below.
pub mod stack_z_offsets {
    pub const BOX_SHADOW: f32 = -0.1;
    pub const BACKGROUND_COLOR: f32 = 0.0;
    pub const OUTER_COLOR: f32 = 0.005;
    pub const OUTLINE: f32 = 0.01;
    pub const IMAGE: f32 = 0.04;
    pub const MATERIAL: f32 = 0.05;
    pub const TEXT_SELECTION: f32 = 0.055;
    pub const TEXT_OUTLINE: f32 = 0.059;
    pub const TEXT: f32 = 0.06;
    pub const TEXT_STRIKETHROUGH: f32 = 0.07;
    pub const TEXT_CURSOR: f32 = 0.08;
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderUiSystems {
    ExtractCameraViews,
    ExtractBoxShadows,
    ExtractBackgrounds,
    ExtractImages,
    ExtractTextureSlice,
    ExtractOutlines,
    ExtractViewportNodes,
    ExtractTextBackgrounds,
    ExtractTextShadows,
    ExtractText,
    ExtractCursor,
    ExtractDebug,
}

/// Marker for controlling whether UI is rendered with or without anti-aliasing
/// in a camera. By default, UI is always anti-aliased.
///
/// **Note:** This does not affect text anti-aliasing. For that, use the `font_smoothing` property of the [`TextFont`](bevy_text::TextFont) component.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         // This will cause all UI in this camera to be rendered without
///         // anti-aliasing
///         UiAntiAlias::Off,
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Default, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub enum UiAntiAlias {
    /// UI will render with anti-aliasing
    #[default]
    On,
    /// UI will render without anti-aliasing
    Off,
}

/// Opts a camera into restricting the sRGB UI composite draw to emitted geometry.
///
/// Only standard UI draws with finite CPU-generated vertices and a full-target,
/// pixel-matched viewport can use the scissor. Other UI draw functions (including
/// surfaces, shadows, sliced images, and custom materials) retain the full-screen
/// composite. UI texture allocation, clearing, and rendering are unchanged.
#[derive(Component, Clone, Copy, Default, Debug, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct UiCompositeScissor;

/// Keep specialized shader batches separate instead of promoting mixed batches
/// to their common general shader. This trades additional draws for less fragment
/// work and never reorders transparent items. Opt-in pending device measurements.
#[derive(Component, Clone, Copy, Default, Debug, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct UiSeparateBatches;

/// Selects geometry-equivalent shader specializations for a camera's UI.
///
/// `FULL` keeps the reference shaders, for comparisons or driver workarounds.
/// Specialized pipelines retain antialiasing and color conversion, and fall back
/// to the reference pipeline until compilation completes. Missing components use
/// [`Self::FULL`]. The separate `linear_color` experiment changes compositing space
/// for all UI draws, including materials and text.
#[derive(Component, Clone, Copy, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub struct UiShaderVariants {
    /// Appearance-changing experiment: emit linear colors directly into an
    /// `Rgba8UnormSrgb` main target, without a UI intermediate or composite.
    /// Linear alpha blending differs from the reference gamma-space blend.
    /// Requires the direct-UI pass; incompatible with `encoded_images`.
    /// Intermediate-only retention, lookup, and composite scissor options are
    /// unused. Custom material shaders must emit linear colors in this mode.
    pub linear_color: bool,
    /// Appearance-changing comparison: ordinary images use sample/tint and
    /// reference gamma encoding (omitted with `linear_color`), without analytic
    /// shape AA. Never selected by the default constants.
    pub basic_images: bool,
    /// Experimental encoded-view sample/tint shader. Requires direct gamma UI
    /// and opt-in dual-view immutable images; all other images retain reference.
    pub encoded_images: bool,
    /// Opt-in: CPU-tessellate stretched, axis-aligned nine-slice images with
    /// single-mip, linear-filtered textures. Tiling, rotation/shear, tiny or
    /// invalid patches, and upper seams near pixel centers retain FULL to avoid
    /// changing its repeat seam or derivative-dependent filtering. Even eligible
    /// images can have small floating-point interpolation differences.
    pub cpu_slices: bool,
    /// Appearance-changing opt-in: affine nine-patch sampling, including rotated,
    /// subpixel and mipmapped images. Keeps geometry/clipping, but replaces FULL's
    /// wrapped seam and cross-seam derivatives with continuous per-patch UVs.
    pub approximate_cpu_slices: bool,
    pub surfaces: bool,
    /// Opt-in: remove gradient machinery from gradient-free fills and borders,
    /// retaining reference geometry, coverage, side selection and color space.
    /// Independent of `surfaces`. Mixed batches promote to the reference shader
    /// without extra draws; `UiSeparateBatches` trades draws for specialization.
    /// Moves all four border colors to flat vertex outputs (more vertex work/IO).
    pub solid_surfaces: bool,
    /// Diagnostic for [`UiCachedSurfaces`] replay fragments only (never source pixels):
    /// 0 = unchanged replay, 1 = opaque magenta, 2 = crop origin / (1080, 2160),
    /// 3 = texture dimensions / 2048, 4 = center texel, 5 = out-of-bounds red.
    /// Modes 2 and 3 put X/Y in R/G with B = 0 and A = 1. Requires 0..=5.
    /// Mode 2 uses a fixed diagnostic viewport, not the camera's actual size.
    pub surface_cache_probe: u32,
    pub images: bool,
    /// Diagnostic: omit the zero-length term for fragments inside rectangular geometry.
    pub rect_math: bool,
    /// Diagnostic: use compact inter-stage interfaces for specialized fills/images.
    pub lean_io: bool,
    /// Opt-in: omit shape coverage only for proven pixel-aligned, untransformed images.
    pub aligned_images: bool,
    /// Diagnostic: use implicit instead of explicit gradients on aligned images.
    pub implicit_aligned_images: bool,
    /// Opt-in: shade large plain image interiors separately from their AA edge band.
    pub image_interiors: bool,
    /// Opt-in Vulkan fragment precision experiment; requires the renderer's
    /// experimental feature and explicit visual approval. Never a default.
    pub relaxed_precision: bool,
    /// Fragment-only sampled-color/gamma precision experiment. Coverage and
    /// texture coordinates keep ordinary precision; requires Vulkan validation.
    pub relaxed_colors: bool,
}

impl UiShaderVariants {
    pub const FULL: Self = Self {
        linear_color: false,
        basic_images: false,
        encoded_images: false,
        cpu_slices: false,
        approximate_cpu_slices: false,
        surfaces: false,
        solid_surfaces: false,
        surface_cache_probe: 0,
        images: false,
        rect_math: false,
        lean_io: false,
        aligned_images: false,
        implicit_aligned_images: false,
        image_interiors: false,
        relaxed_precision: false,
        relaxed_colors: false,
    };
    pub const SURFACES: Self = Self {
        surfaces: true,
        ..Self::FULL
    };
    pub const IMAGES: Self = Self {
        images: true,
        ..Self::FULL
    };
    pub const ALL: Self = Self {
        surfaces: true,
        images: true,
        ..Self::FULL
    };

    pub(crate) fn render_target_format(self, target_format: TextureFormat) -> TextureFormat {
        assert!(
            self.surface_cache_probe <= 5,
            "surface cache probe must be in 0..=5"
        );
        assert!(
            !(self.linear_color && self.encoded_images),
            "linear-color UI cannot use encoded-image views"
        );
        if self.linear_color {
            assert!(
                matches!(target_format, TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb),
                "linear-color UI requires an RGBA8 or BGRA8 sRGB main target, got {target_format:?}"
            );
            target_format
        } else {
            ui_render_target_format(target_format)
        }
    }
}

impl Default for UiShaderVariants {
    fn default() -> Self {
        Self::FULL
    }
}

/// Number of shadow samples.
/// A larger value will result in higher quality shadows.
/// Default is 4, values higher than ~10 offer diminishing returns.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         BoxShadowSamples(6),
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub struct BoxShadowSamples(pub u32);

impl Default for BoxShadowSamples {
    fn default() -> Self {
        Self(4)
    }
}

#[derive(Default)]
pub struct UiRenderPlugin;

#[derive(Resource)]
struct SrgbUiCompositeShader {
    _handle: Handle<Shader>,
}

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TextGroupPlugin>() {
            app.add_plugins(TextGroupPlugin);
        }
        load_shader_library!(app, "ui.wgsl");
        embedded_asset!(app, "image_rect.wgsl");
        embedded_asset!(app, "image_basic.wgsl");
        embedded_asset!(app, "srgb_ui_composite.wgsl");
        app.insert_resource(SrgbUiCompositeShader {
            _handle: load_embedded_asset!(app, "srgb_ui_composite.wgsl"),
        });

        #[cfg(feature = "bevy_ui_debug")]
        app.init_resource::<GlobalUiDebugOptions>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<UiPipeline>>()
            .init_gpu_resource::<UiPipelineVariants>()
            .init_gpu_resource::<SpecializedRenderPipelines<SrgbUiCompositePipeline>>()
            .init_gpu_resource::<ImageNodeBindGroups>()
            .init_gpu_resource::<UiMesh<UiVertex>>()
            .init_resource::<ExtractedUiNodes>()
            .init_resource::<UiCompositeBounds>()
            .init_gpu_resource::<retained_layer::RetainedUiLayers>()
            .allow_ambiguous_resource::<ExtractedUiNodes>()
            .init_resource::<DrawFunctions<TransparentUi>>()
            .init_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .allow_ambiguous_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .add_render_command::<TransparentUi, DrawUi>()
            .configure_sets(
                ExtractSchedule,
                (
                    RenderUiSystems::ExtractCameraViews,
                    RenderUiSystems::ExtractBoxShadows,
                    RenderUiSystems::ExtractBackgrounds,
                    RenderUiSystems::ExtractImages,
                    RenderUiSystems::ExtractTextureSlice,
                    RenderUiSystems::ExtractOutlines,
                    RenderUiSystems::ExtractTextBackgrounds,
                    RenderUiSystems::ExtractTextShadows,
                    RenderUiSystems::ExtractText,
                    RenderUiSystems::ExtractCursor,
                    RenderUiSystems::ExtractDebug,
                )
                    .chain(),
            )
            .add_systems(RenderStartup, init_ui_pipeline)
            .add_systems(
                ExtractSchedule,
                (
                    extract_ui_camera_view
                        .after(extract_cameras)
                        .in_set(RenderUiSystems::ExtractCameraViews),
                    extract_uinode_outer_colors.in_set(RenderUiSystems::ExtractBackgrounds),
                    extract_uinode_images.in_set(RenderUiSystems::ExtractImages),
                    extract_uinode_outlines.in_set(RenderUiSystems::ExtractOutlines),
                    extract_viewport_nodes.in_set(RenderUiSystems::ExtractViewportNodes),
                    extract_text_decorations.in_set(RenderUiSystems::ExtractTextBackgrounds),
                    extract_text_shadows.in_set(RenderUiSystems::ExtractTextShadows),
                    extract_text_sections.in_set(RenderUiSystems::ExtractText),
                    extract_text_cursor.in_set(RenderUiSystems::ExtractCursor),
                    extract_preedit_underlines.in_set(RenderUiSystems::ExtractCursor),
                    #[cfg(feature = "bevy_ui_debug")]
                    debug_overlay::extract_debug_overlay.in_set(RenderUiSystems::ExtractDebug),
                ),
            )
            .add_systems(
                Render,
                (
                    queue_uinodes.in_set(RenderSystems::Queue),
                    queue_srgb_ui_composite_pipelines.in_set(RenderSystems::Queue),
                    sort_phase_system::<TransparentUi>.in_set(RenderSystems::PhaseSort),
                    (
                        prepare_uinodes,
                        prepare_srgb_ui_textures,
                        prepare_srgb_ui_composite_bind_groups,
                    )
                        .chain()
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core2d,
                (ui_pass, srgb_ui_composite_pass)
                    .chain()
                    .after(Core2dSystems::PostProcess)
                    .before(upscaling),
            )
            .add_systems(
                Core3d,
                (ui_pass, srgb_ui_composite_pass)
                    .chain()
                    .after(Core3dSystems::PostProcess)
                    .before(upscaling),
            );

        app.add_plugins(UiTextureSlicerPlugin);
        app.add_plugins(SurfacePlugin);
        app.add_plugins(BoxShadowPlugin);
    }
}

#[derive(SystemParam)]
pub struct UiCameraMap<'w, 's> {
    mapping: Query<'w, 's, RenderEntity>,
}

impl<'w, 's> UiCameraMap<'w, 's> {
    /// Creates a [`UiCameraMapper`] for performing repeated camera-to-render-entity lookups.
    ///
    /// The last successful mapping is cached to avoid redundant queries.
    pub fn get_mapper(&'w self) -> UiCameraMapper<'w, 's> {
        UiCameraMapper {
            mapping: &self.mapping,
            camera_entity: Entity::PLACEHOLDER,
            render_entity: Entity::PLACEHOLDER,
        }
    }
}

/// Helper for mapping UI target camera entities to their corresponding render entities,
/// with caching to avoid repeated lookups for the same camera.
pub struct UiCameraMapper<'w, 's> {
    mapping: &'w Query<'w, 's, RenderEntity>,
    /// Cached camera entity from the last successful `map` call.
    camera_entity: Entity,
    /// Cached camera entity from the last successful `map` call.
    render_entity: Entity,
}

impl<'w, 's> UiCameraMapper<'w, 's> {
    /// Returns the render entity corresponding to the given [`ComputedUiTargetCamera`]'s camera, or none if no corresponding entity was found.
    pub fn map(&mut self, computed_target: &ComputedUiTargetCamera) -> Option<Entity> {
        let camera_entity = computed_target.get()?;
        if self.camera_entity != camera_entity {
            let new_render_camera_entity = self.mapping.get(camera_entity).ok()?;
            self.render_entity = new_render_camera_entity;
            self.camera_entity = camera_entity;
        }

        Some(self.render_entity)
    }

    /// Returns the cached camera entity from the last successful `map` call.
    pub fn current_camera(&self) -> Entity {
        self.camera_entity
    }
}

pub struct ExtractedUiNode {
    pub z_order: f32,
    pub image: AssetId<Image>,
    pub clip: Option<CalculatedClip>,
    /// Render world entity of the extracted camera corresponding to this node's target camera.
    pub extracted_camera_entity: Entity,
    pub item: ExtractedUiItem,
    pub main_entity: MainEntity,
    pub render_entity: Entity,
    pub transform: Affine2,
}

/// The type of UI node.
/// This is used to determine how to render the UI node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    Rect,
    Viewport,
    Inverted,
    Outline,
}

pub enum ExtractedUiItem {
    Node {
        color: LinearRgba,
        rect: Rect,
        atlas_scaling: Option<Vec2>,
        flip_x: bool,
        flip_y: bool,
        /// Border radius of the UI node.
        /// Ordering: top left, top right, bottom right, bottom left.
        border_radius: ResolvedBorderRadius,
        /// Border thickness of the UI node.
        /// Ordering: left, top, right, bottom.
        border: BorderRect,
        node_type: NodeType,
    },
    /// A contiguous sequence of text glyphs from the same section
    Glyphs {
        /// Indices into [`ExtractedUiNodes::glyphs`]
        range: Range<usize>,
    },
}

pub struct ExtractedGlyph {
    pub color: LinearRgba,
    pub translation: Vec2,
    pub rect: Rect,
    /// Text mask and effect selection; color is the paint for this layer.
    pub flags: u32,
}

#[derive(Resource, Default)]
pub struct ExtractedUiNodes {
    pub uinodes: Vec<ExtractedUiNode>,
    pub glyphs: Vec<ExtractedGlyph>,
}

impl ExtractedUiNodes {
    pub fn clear(&mut self) {
        self.uinodes.clear();
        self.glyphs.clear();
    }
}

pub fn extract_uinode_outer_colors(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &OuterColor,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (entity, node, stack, transform, visibility, clip, camera, color) in &uinode_query {
        if !visibility.get()
            || color.is_fully_transparent()
            || node.is_empty()
            || clipping::rect_is_clipped(clip, transform.into(), node.size())
        {
            continue;
        }
        let Some(camera) = camera_mapper.map(camera) else {
            continue;
        };
        extracted_uinodes.uinodes.push(ExtractedUiNode {
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            z_order: stack.0 as f32 + stack_z_offsets::OUTER_COLOR,
            clip: clip.cloned(),
            image: AssetId::default(),
            extracted_camera_entity: camera,
            transform: transform.into(),
            main_entity: entity.into(),
            item: ExtractedUiItem::Node {
                color: color.0.into(),
                rect: Rect {
                    min: Vec2::ZERO,
                    max: node.size(),
                },
                atlas_scaling: None,
                flip_x: false,
                flip_y: false,
                border: BorderRect::ZERO,
                border_radius: node.border_radius(),
                node_type: NodeType::Inverted,
            },
        });
    }
}

pub fn extract_uinode_images(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ImageNode,
            &ImageNodeSize,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        clip,
        camera,
        image,
        image_size,
    ) in &uinode_query
    {
        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };
        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || image.image_mode.uses_slices()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let size = if matches!(image.image_mode, NodeImageMode::Auto) {
            let source = image_size.size().as_vec2();
            if source.cmple(Vec2::ZERO).any() {
                visual_box.size()
            } else {
                source * (visual_box.size() / source).min_element()
            }
        } else {
            visual_box.size()
        };

        let image_transform =
            Affine2::from(*transform) * Affine2::from_translation(visual_box.center());
        if clipping::rect_is_clipped(clip, image_transform, size) {
            continue;
        }

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let mut rect = match (atlas_rect, image.rect) {
            (None, None) => Rect {
                min: Vec2::ZERO,
                max: size,
            },
            (None, Some(image_rect)) => image_rect,
            (Some(atlas_rect), None) => atlas_rect,
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                image_rect
            }
        };

        let atlas_scaling = if atlas_rect.is_some() || image.rect.is_some() {
            let atlas_scaling = size / rect.size();
            rect.min *= atlas_scaling;
            rect.max *= atlas_scaling;
            Some(atlas_scaling)
        } else {
            None
        };

        extracted_uinodes.uinodes.push(ExtractedUiNode {
            z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            clip: clip.cloned(),
            image: image.image.id(),
            extracted_camera_entity,
            transform: image_transform,
            item: ExtractedUiItem::Node {
                color: image.color.into(),
                rect,
                atlas_scaling,
                flip_x: image.flip_x,
                flip_y: image.flip_y,
                border: BorderRect::ZERO,
                border_radius: uinode.border_radius,
                node_type: NodeType::Rect,
            },
            main_entity: entity.into(),
        });
    }
}

pub fn extract_uinode_outlines(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            Option<&Node>,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &Outline,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let image = AssetId::<Image>::default();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        node,
        computed_node,
        stack_index,
        transform,
        inherited_visibility,
        maybe_clip,
        camera,
        outline,
    ) in &uinode_query
    {
        if !inherited_visibility.get()
            || node.is_some_and(|node| node.display == Display::None)
            || computed_node.outline_width() <= 0.
            || outline.color.is_fully_transparent()
        {
            continue;
        }
        let outline_size = computed_node.outlined_node_size();
        if clipping::rect_is_clipped(maybe_clip, transform.into(), outline_size) {
            continue;
        }
        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        extracted_uinodes.uinodes.push(ExtractedUiNode {
            z_order: stack_index.0 as f32 + stack_z_offsets::OUTLINE,
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            image,
            clip: maybe_clip.cloned(),
            extracted_camera_entity,
            transform: transform.into(),
            item: ExtractedUiItem::Node {
                color: outline.color.into(),
                rect: Rect {
                    max: outline_size,
                    ..Default::default()
                },
                atlas_scaling: None,
                flip_x: false,
                flip_y: false,
                border: BorderRect::all(computed_node.outline_width()),
                border_radius: computed_node.outline_radius(),
                node_type: NodeType::Outline,
            },
            main_entity: entity.into(),
        });
    }
}

/// The UI camera is "moved back" by this many units (plus the [`UI_CAMERA_TRANSFORM_OFFSET`]) and also has a view
/// distance of this many units. This ensures that with a left-handed projection,
/// as ui elements are "stacked on top of each other", they are within the camera's view
/// and have room to grow.
// TODO: Consider computing this value at runtime based on the maximum z-value.
const UI_CAMERA_FAR: f32 = 1000.0;

// This value is subtracted from the far distance for the camera's z-position to ensure nodes at z == 0.0 are rendered
// TODO: Evaluate if we still need this.
const UI_CAMERA_TRANSFORM_OFFSET: f32 = -0.1;

/// The ID of the subview associated with a camera on which UI is to be drawn.
///
/// When UI is present, cameras extract to two views: the main 2D/3D one and a
/// UI one. The main 2D or 3D camera gets subview 0, and the corresponding UI
/// camera gets this subview, 1.
const UI_CAMERA_SUBVIEW: u32 = 1;

/// A render-world component that lives on the main render target view and
/// specifies the corresponding UI view.
///
/// For example, if UI is being rendered to a 3D camera, this component lives on
/// the 3D camera and contains the entity corresponding to the UI view.
#[derive(Component)]
/// Entity id of the temporary render entity with the corresponding extracted UI view.
pub struct UiCameraView(pub Entity);

/// A render-world component that lives on the UI view and specifies the
/// corresponding main render target view.
///
/// For example, if the UI is being rendered to a 3D camera, this component
/// lives on the UI view and contains the entity corresponding to the 3D camera.
///
/// This is the inverse of [`UiCameraView`].
#[derive(Component)]
pub struct UiViewTarget(pub Entity);

/// Experimental LDR composite using an 8-bit gamma lookup for opaque pixels.
///
/// Leaves translucent pixels, HDR and unmarked cameras on the reference shader. Rounding a filtered
/// sample back to 8-bit channels can change pixels; callers must validate output.
#[derive(Component, Clone, Copy, Default)]
pub struct UiCompositeLookup;

/// Excludes a camera from UI extraction, for example a presentation-only blit camera.
#[derive(Component, Clone, Copy, Default)]
pub struct UiRenderingDisabled;

/// Extracts all UI elements associated with a camera into the render world.
pub fn extract_ui_camera_view(
    mut commands: Commands,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    query: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &Camera,
                Option<&UiAntiAlias>,
                Option<&BoxShadowSamples>,
                Option<&UiShaderVariants>,
                Has<UiCompositeScissor>,
                Has<UiSeparateBatches>,
                Option<&UiRetainedLayer>,
                Has<UiCompositeLookup>,
                Has<UiCachedSurfaces>,
                Has<UiRenderingDisabled>,
            ),
            Or<(With<Camera2d>, With<Camera3d>)>,
        >,
    >,
    main_pass_formats: Res<CameraMainPassTextureFormats>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();

    for (
        main_entity,
        render_entity,
        camera,
        ui_anti_alias,
        shadow_samples,
        variants,
        scissor,
        separate,
        retained,
        composite_lookup,
        cached_surfaces,
        ui_disabled,
    ) in &query
    {
        // ignore inactive cameras
        if !camera.is_active || ui_disabled {
            commands
                .get_entity(render_entity)
                .expect("Camera entity wasn't synced.")
                .remove::<(
                    UiCameraView,
                    UiAntiAlias,
                    BoxShadowSamples,
                    UiShaderVariants,
                )>();
            continue;
        }

        if let (Some(physical_viewport_rect), Some(_viewport_size), Some(target_size)) = (
            camera.physical_viewport_rect(),
            camera.physical_viewport_size(),
            camera.physical_target_size(),
        ) && target_size.x != 0
            && target_size.y != 0
        {
            let Some(target_format) = main_pass_formats.get(&render_entity).copied() else {
                commands
                    .get_entity(render_entity)
                    .expect("Camera entity wasn't synced.")
                    .remove::<(
                        UiCameraView,
                        UiAntiAlias,
                        BoxShadowSamples,
                        UiShaderVariants,
                    )>();
                continue;
            };
            let variants = variants.copied().unwrap_or_default();
            // Validate before CPU paint extraction or any pipeline specialization.
            variants.render_target_format(target_format);

            // use a projection matrix with the origin in the top left instead of the bottom left that comes with OrthographicProjection
            let projection_matrix = Mat4::orthographic_rh(
                0.0,
                physical_viewport_rect.width() as f32,
                physical_viewport_rect.height() as f32,
                0.0,
                0.0,
                UI_CAMERA_FAR,
            );
            // We use `UI_CAMERA_SUBVIEW` here so as not to conflict with the
            // main 3D or 2D camera, which will have subview index 0.
            let retained_view_entity =
                RetainedViewEntity::new(main_entity.into(), None, UI_CAMERA_SUBVIEW);
            // Creates the UI view.
            let ui_camera_view = commands
                .spawn((
                    ExtractedView {
                        retained_view_entity,
                        clip_from_view: projection_matrix,
                        world_from_view: GlobalTransform::from_xyz(
                            0.0,
                            0.0,
                            UI_CAMERA_FAR + UI_CAMERA_TRANSFORM_OFFSET,
                        ),
                        clip_from_world: None,
                        target_format,
                        viewport: UVec4::from((
                            physical_viewport_rect.min,
                            physical_viewport_rect.size(),
                        )),
                        color_grading: Default::default(),
                        invert_culling: false,
                    },
                    // Link to the main camera view.
                    UiViewTarget(render_entity),
                    TemporaryRenderEntity,
                    variants,
                ))
                .id();
            if let Some(settings) = retained.filter(|_| !variants.linear_color) {
                commands.entity(ui_camera_view).insert(*settings);
            }
            if composite_lookup && !variants.linear_color {
                commands.entity(ui_camera_view).insert(UiCompositeLookup);
            }

            // This view is freshly spawned each frame: removing the camera marker
            // cannot leave an enabled scissor on a retained render entity.
            if scissor && !variants.linear_color {
                commands.entity(ui_camera_view).insert(UiCompositeScissor);
            }
            if separate {
                commands.entity(ui_camera_view).insert(UiSeparateBatches);
            }
            if cached_surfaces {
                commands.entity(ui_camera_view).insert(UiCachedSurfaces);
            }

            let mut entity_commands = commands
                .get_entity(render_entity)
                .expect("Camera entity wasn't synced.");
            // Link from the main 2D/3D camera view to the UI view.
            entity_commands.insert(UiCameraView(ui_camera_view));
            // Always insert defaults too: removing an override must not leave the
            // previous frame's reference/optimized choice in the render world.
            entity_commands.insert(variants);
            if let Some(ui_anti_alias) = ui_anti_alias {
                entity_commands.insert(*ui_anti_alias);
            }
            if let Some(shadow_samples) = shadow_samples {
                entity_commands.insert(*shadow_samples);
            }
            transparent_render_phases.prepare_for_new_frame(retained_view_entity);

            live_entities.insert(retained_view_entity);
        }
    }

    transparent_render_phases.retain(|entity, _| live_entities.contains(entity));
}

pub fn extract_viewport_nodes(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    camera_query: Extract<Query<(&Camera, &RenderTarget)>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ViewportNode,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        clip,
        camera,
        viewport_node,
    ) in &uinode_query
    {
        // Skip invisible images
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        let Some(camera_entity) = viewport_node.camera else {
            continue;
        };

        let Some(image) = camera_query
            .get(camera_entity)
            .ok()
            .and_then(|(_, render_target)| render_target.as_image())
        else {
            continue;
        };

        extracted_uinodes.uinodes.push(ExtractedUiNode {
            z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
            render_entity: commands.spawn(TemporaryRenderEntity).id(),
            clip: clip.cloned(),
            image: image.id(),
            extracted_camera_entity,
            transform: transform.into(),
            item: ExtractedUiItem::Node {
                color: LinearRgba::WHITE,
                rect: Rect {
                    min: Vec2::ZERO,
                    max: uinode.size,
                },
                atlas_scaling: None,
                flip_x: false,
                flip_y: false,
                border: uinode.border(),
                border_radius: uinode.border_radius(),
                node_type: NodeType::Viewport,
            },
            main_entity: entity.into(),
        });
    }
}

pub fn extract_text_sections(
    mut commands: Commands,
    mut groups: ResMut<TextGroupCache>,
    group_styles: Extract<Query<TextGroupStyleQuery>>,
    images: Extract<Res<Assets<Image>>>,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedTextBlock,
            &TextColor,
            Ref<TextLayoutInfo>,
            (
                Option<&TextOpacity>,
                Option<&InheritedTextOpacity>,
                Option<&TextShadow>,
                Option<&TextOutline>,
            ),
            Option<&TextScroll>,
            Option<&TextCursorStyle>,
            Option<&EditableTextPlaceholderColor>,
            (Has<EditableText>, &TextFont),
        )>,
    >,
    text_styles: Extract<Query<&TextColor>>,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        uinode,
        stack_index,
        global_transform,
        inherited_visibility,
        maybe_clip,
        camera,
        computed_block,
        text_color,
        text_layout_info,
        (opacity, inherited_opacity, maybe_shadow, maybe_outline),
        text_scroll,
        cursor_style,
        placeholder_color,
        (editable, text_font),
    ) in &uinode_query
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min - text_scroll.map_or(Vec2::ZERO, |s| s.0),
            );

        let clip = calculate_text_scroll_clip(text_scroll, maybe_clip, uinode, global_transform);
        let paint_order =
            maybe_outline.map_or(TextPaintOrder::default(), |outline| outline.paint_order);

        let mut color = text_color.0.to_linear();

        let selected_text_color = cursor_style
            .and_then(|cursor_style| cursor_style.selected_text_color)
            .map(|selected_text_color| selected_text_color.to_linear());

        let mut current_section_index = 0;

        let outline = maybe_outline
            .filter(|outline| !outline.color.is_fully_transparent())
            .and_then(|outline| {
                text_effect_outline_extent(
                    outline.stroke_width,
                    text_layout_info.scale_factor,
                    entity,
                    "TextOutline",
                )
                .map(|width| (outline.color.to_linear(), width))
            });
        let shadow = maybe_shadow
            .filter(|shadow| !shadow.color.is_fully_transparent())
            .and_then(|shadow| {
                text_effect_shadow_offset(
                    shadow.offset,
                    text_layout_info.scale_factor,
                    entity,
                    "TextShadow",
                )
                .map(|offset| (shadow.color.to_linear(), offset))
            });
        if let Some(opacity) = opacity {
            let alpha = opacity.get() * inherited_opacity.map_or(1., |inherited| inherited.get());
            if alpha == 0. {
                groups.keep_cached(entity);
                continue;
            }
            let effects = TextGroupEffects {
                shadow,
                outline,
                paint_order,
                selected_color: selected_text_color,
                placeholder_color: placeholder_color.map(|c| c.0.to_linear()),
                tint_color_glyphs: false,
            };
            // EditableText owns its layout and does not populate ComputedTextBlock's spans.
            let root_section = TextEntity {
                entity,
                depth: 0,
                font_smoothing: text_font.font_smoothing,
            };
            let sections = if editable {
                core::slice::from_ref(&root_section)
            } else {
                computed_block.entities()
            };
            for tile in groups.extract(
                entity,
                &images,
                &text_layout_info,
                sections,
                &group_styles,
                effects,
            ) {
                let start = extracted_uinodes.glyphs.len();
                extracted_uinodes.glyphs.push(ExtractedGlyph {
                    color: LinearRgba::WHITE.with_alpha(alpha),
                    translation: tile.bounds.center(),
                    rect: tile.rect,
                    flags: shader_flags::TEXTURED | shader_flags::TEXT_GROUP,
                });
                push_glyph_batch(
                    &mut commands,
                    &mut extracted_uinodes,
                    tile.image,
                    start,
                    stack_index.0 as f32 + stack_z_offsets::TEXT,
                    clip.clone(),
                    extracted_camera_entity,
                    entity.into(),
                    transform,
                );
            }
            continue;
        }
        extract_glyphs(
            &mut commands,
            &mut extracted_uinodes,
            text_layout_info.glyphs.iter(),
            |PositionedGlyph {
                 position,
                 atlas_info,
                 section_index,
                 ..
             }| {
                if current_section_index != *section_index {
                    let Some(section_entity) = computed_block
                        .entities()
                        .get(*section_index)
                        .map(|text_entity| text_entity.entity)
                    else {
                        tracing::warn!(
                        "Skipping UI text glyph for {entity:?}: missing text section {section_index}"
                    );
                        return None;
                    };

                    let Ok(text_color) = text_styles.get(section_entity) else {
                        tracing::warn!(
                        "Skipping UI text glyph for {entity:?}: missing TextColor on section entity {section_entity:?}"
                    );
                        return None;
                    };

                    color = LinearRgba::from(text_color.0);
                    current_section_index = *section_index;
                }

                if let Some(selected_text_color) = selected_text_color
                    && text_layout_info
                        .selection_rects
                        .iter()
                        .any(|selection_rect| {
                            let glyph_rect =
                                Rect::from_center_size(*position, atlas_info.rect.size());
                            selection_rect.contains(glyph_rect.min)
                                && selection_rect.contains(glyph_rect.max)
                        })
                {
                    Some(selected_text_color)
                } else {
                    Some(color)
                }
            },
            shadow,
            outline,
            paint_order,
            stack_index.0 as f32 + stack_z_offsets::TEXT,
            clip.clone(),
            extracted_camera_entity,
            entity.into(),
            transform,
        );

        if !text_layout_info.placeholder_glyphs.is_empty() {
            let placeholder_color = placeholder_color
                .map(|placeholder_color| placeholder_color.0)
                .unwrap_or(text_color.0)
                .to_linear();

            extract_glyphs(
                &mut commands,
                &mut extracted_uinodes,
                text_layout_info.placeholder_glyphs.iter(),
                |_| Some(placeholder_color),
                shadow,
                outline,
                paint_order,
                stack_index.0 as f32 + stack_z_offsets::TEXT,
                clip,
                extracted_camera_entity,
                entity.into(),
                transform,
            );
        }
    }
}

fn extract_glyphs<'a>(
    commands: &mut Commands,
    extracted_uinodes: &mut ExtractedUiNodes,
    glyphs: impl Iterator<Item = &'a PositionedGlyph> + Clone,
    mut color: impl FnMut(&'a PositionedGlyph) -> Option<LinearRgba>,
    shadow: Option<(LinearRgba, Vec2)>,
    outline: Option<(LinearRgba, f32)>,
    paint_order: TextPaintOrder,
    z_order: f32,
    clip: Option<CalculatedClip>,
    extracted_camera_entity: Entity,
    main_entity: MainEntity,
    transform: Affine2,
) {
    // Keep block-wide paint order even when the run crosses atlas textures.
    for layer in paint_order.layers() {
        let (paint, offset, layer_flags) = match layer {
            TextEffectLayer::Shadow => {
                let Some((color, offset)) = shadow else {
                    continue;
                };
                (Some(color), offset, shader_flags::TEXT_EFFECT_SHADOW)
            }
            TextEffectLayer::Outline => {
                let Some((color, _)) = outline else { continue };
                (Some(color), Vec2::ZERO, 0)
            }
            TextEffectLayer::Fill => (None, Vec2::ZERO, 0),
        };
        let mask_flags = shader_flags::TEXT_GLYPH
            | if layer != TextEffectLayer::Fill && outline.is_some() {
                shader_flags::TEXT_EFFECT_OUTLINE
            } else {
                0
            };
        let mut start = extracted_uinodes.glyphs.len();
        let mut texture = None;

        for glyph in glyphs.clone() {
            if layer == TextEffectLayer::Outline && !glyph.atlas_info.is_alpha_mask {
                continue;
            }
            let Some(color) = color(glyph) else {
                continue;
            };

            if let Some(previous_texture) = texture
                && previous_texture != glyph.atlas_info.texture
            {
                push_glyph_batch(
                    commands,
                    extracted_uinodes,
                    previous_texture,
                    start,
                    z_order,
                    clip.clone(),
                    extracted_camera_entity,
                    main_entity,
                    transform,
                );
                start = extracted_uinodes.glyphs.len();
            }

            texture = Some(glyph.atlas_info.texture);

            let atlas_info = &glyph.atlas_info;
            let rect = if atlas_info.is_alpha_mask || shadow.is_some() {
                atlas_info
                    .rect
                    .inflate(layer.glyph_padding(atlas_info.padding))
            } else {
                atlas_info.rect
            };

            extracted_uinodes.glyphs.push(ExtractedGlyph {
                color: paint.unwrap_or(if atlas_info.is_alpha_mask {
                    color
                } else {
                    LinearRgba::WHITE
                }),
                translation: glyph.position + offset,
                rect,
                flags: shader_flags::TEXTURED
                    | layer_flags
                    | if atlas_info.is_alpha_mask {
                        mask_flags
                    } else {
                        0
                    },
            });
        }

        if let Some(texture) = texture {
            push_glyph_batch(
                commands,
                extracted_uinodes,
                texture,
                start,
                z_order,
                clip.clone(),
                extracted_camera_entity,
                main_entity,
                transform,
            );
        }
    }
}

fn push_glyph_batch(
    commands: &mut Commands,
    extracted_uinodes: &mut ExtractedUiNodes,
    texture: AssetId<Image>,
    start: usize,
    z_order: f32,
    clip: Option<CalculatedClip>,
    extracted_camera_entity: Entity,
    main_entity: MainEntity,
    transform: Affine2,
) {
    let end = extracted_uinodes.glyphs.len();
    if start == end {
        return;
    }

    extracted_uinodes.uinodes.push(ExtractedUiNode {
        z_order,
        render_entity: commands.spawn(TemporaryRenderEntity).id(),
        image: texture,
        clip,
        extracted_camera_entity,
        item: ExtractedUiItem::Glyphs { range: start..end },
        main_entity,
        transform,
    });
}

pub fn extract_text_shadows(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &ComputedUiTargetCamera,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &TextLayoutInfo,
                &TextShadow,
                &ComputedTextBlock,
                Option<&TextScroll>,
            ),
            Without<TextOpacity>,
        >,
    >,
    text_decoration_query: Extract<Query<(Has<Strikethrough>, Has<Underline>)>>,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        uinode,
        stack_index,
        global_transform,
        target,
        inherited_visibility,
        maybe_clip,
        text_layout_info,
        shadow,
        computed_block,
        text_scroll,
    ) in &uinode_query
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        if shadow.color.is_fully_transparent() {
            continue;
        }

        let Some(shadow_offset) = text_effect_shadow_offset(
            shadow.offset,
            text_layout_info.scale_factor,
            entity,
            "TextShadow",
        ) else {
            continue;
        };

        let Some(extracted_camera_entity) = camera_mapper.map(target) else {
            continue;
        };

        let node_transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min + shadow_offset - text_scroll.map_or(Vec2::ZERO, |s| s.0),
            );

        let clip = calculate_text_scroll_clip(text_scroll, maybe_clip, uinode, global_transform);

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((has_strikethrough, has_underline)) = text_decoration_query.get(section_entity)
            else {
                continue;
            };

            if has_strikethrough {
                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    clip: clip.clone(),
                    image: AssetId::default(),
                    extracted_camera_entity,
                    transform: node_transform
                        * Affine2::from_translation(run.strikethrough_position()),
                    item: ExtractedUiItem::Node {
                        color: shadow.color.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: run.strikethrough_size(),
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: ResolvedBorderRadius::ZERO,
                        node_type: NodeType::Rect,
                    },
                    main_entity: entity.into(),
                });
            }

            if has_underline {
                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    clip: clip.clone(),
                    image: AssetId::default(),
                    extracted_camera_entity,
                    transform: node_transform * Affine2::from_translation(run.underline_position()),
                    item: ExtractedUiItem::Node {
                        color: shadow.color.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: run.underline_size(),
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: ResolvedBorderRadius::ZERO,
                        node_type: NodeType::Rect,
                    },
                    main_entity: entity.into(),
                });
            }
        }
    }
}

pub fn extract_text_decorations(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &ComputedTextBlock,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
                &TextLayoutInfo,
                Option<&TextScroll>,
            ),
            Without<TextOpacity>,
        >,
    >,
    text_background_colors_query: Extract<
        Query<(
            AnyOf<(&TextBackgroundColor, &Strikethrough, &Underline)>,
            &TextColor,
            Option<&StrikethroughColor>,
            Option<&UnderlineColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        uinode,
        stack_index,
        computed_block,
        global_transform,
        inherited_visibility,
        maybe_clip,
        camera,
        text_layout_info,
        text_scroll,
    ) in &uinode_query
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(global_transform)
            * Affine2::from_translation(
                uinode.content_box().min - text_scroll.map_or(Vec2::ZERO, |s| s.0),
            );

        let clip = calculate_text_scroll_clip(text_scroll, maybe_clip, uinode, global_transform);

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((
                (text_background_color, maybe_strikethrough, maybe_underline),
                text_color,
                maybe_strikethrough_color,
                maybe_underline_color,
            )) = text_background_colors_query.get(section_entity)
            else {
                continue;
            };

            if let Some(text_background_color) = text_background_color {
                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    clip: clip.clone(),
                    image: AssetId::default(),
                    extracted_camera_entity,
                    transform: transform * Affine2::from_translation(run.bounds.center()),
                    item: ExtractedUiItem::Node {
                        color: text_background_color.0.to_linear(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: run.bounds.size(),
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: ResolvedBorderRadius::ZERO,
                        node_type: NodeType::Rect,
                    },
                    main_entity: entity.into(),
                });
            }

            if maybe_strikethrough.is_some() {
                let color = maybe_strikethrough_color
                    .map(|sc| sc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    clip: clip.clone(),
                    image: AssetId::default(),
                    extracted_camera_entity,
                    transform: transform * Affine2::from_translation(run.strikethrough_position()),
                    item: ExtractedUiItem::Node {
                        color,
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: run.strikethrough_size(),
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: ResolvedBorderRadius::ZERO,
                        node_type: NodeType::Rect,
                    },
                    main_entity: entity.into(),
                });
            }

            if maybe_underline.is_some() {
                let color = maybe_underline_color
                    .map(|uc| uc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    clip: clip.clone(),
                    image: AssetId::default(),
                    extracted_camera_entity,
                    transform: transform * Affine2::from_translation(run.underline_position()),
                    item: ExtractedUiItem::Node {
                        color,
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: run.underline_size(),
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: ResolvedBorderRadius::ZERO,
                        node_type: NodeType::Rect,
                    },
                    main_entity: entity.into(),
                });
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UiVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// Shader flags to determine how to render the UI node.
    /// See [`shader_flags`] for possible values.
    pub flags: u32,
    /// Border radius of the UI node.
    /// Ordering: top left, top right, bottom right, bottom left.
    pub radius: [f32; 4],
    /// Border thickness of the UI node.
    /// Ordering: left, top, right, bottom.
    pub border: [f32; 4],
    /// Size of the UI node.
    pub size: [f32; 2],
    /// Position relative to the center of the UI node.
    pub point: [f32; 2],
}

pub(crate) const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    Vec2::new(-0.5, -0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(-0.5, 0.5),
];

#[derive(Component)]
pub struct UiBatch {
    pub range: Range<u32>,
    pub image: AssetId<Image>,
}

/// A coverage-free subdraw inside one transparent item. Its edge draw remains
/// adjacent, so overlapping images retain their original source-over order.
#[derive(Component)]
#[doc(hidden)]
pub struct UiImageInterior {
    pub(crate) range: Range<u32>,
    pub(crate) pipeline: CachedRenderPipelineId,
}

#[derive(Component)]
#[doc(hidden)]
pub struct UiEncodedBatch;

fn has_encoded_view(image: &GpuImage) -> bool {
    !image.texture_descriptor.format.is_srgb()
        && image
            .texture_view_descriptor
            .as_ref()
            .and_then(|view| view.format)
            .is_some_and(|format| format.is_srgb())
        && !image
            .texture_descriptor
            .usage
            .contains(TextureUsages::RENDER_ATTACHMENT)
}

impl UiBatch {
    fn append(
        items: &mut SortedPhaseItems<TransparentUi>,
        first: &mut Option<usize>,
        batches: &mut Vec<(Entity, Self)>,
        item_index: usize,
        image: AssetId<Image>,
        range: Range<u32>,
        reference_pipeline: CachedRenderPipelineId,
    ) {
        if let Some(first) = *first
            && items[first].batch_range.end == item_index as u32
        {
            let batch = &mut batches.last_mut().unwrap().1;
            // Fully clipped nodes need neither a texture change nor a new draw.
            if range.is_empty() {
                items[first].batch_range.end += 1;
                return;
            }
            // Untextured vertices can share any texture binding.
            if batch.image == image
                || batch.image == AssetId::default()
                || image == AssetId::default()
            {
                // A batch may contain ordinary images together with rounded
                // images, outlines or text decorations. The general shader is
                // the common superset; never draw these with IMAGE_RECT merely
                // because the first node qualified for it.
                if items[first].pipeline != items[item_index].pipeline {
                    items[first].pipeline = reference_pipeline;
                }
                if batch.image == AssetId::default() {
                    batch.image = image;
                }
                batch.range.end = range.end;
                items[first].batch_range.end += 1;
                return;
            }
        }
        if !range.is_empty() {
            batches.push((items[item_index].entity(), Self { range, image }));
            items[item_index].batch_range = item_index as u32..item_index as u32 + 1;
            *first = Some(item_index);
        }
    }
}

/// The values here should match the values for the constants in `ui.wgsl`
pub mod shader_flags {
    /// Texture should be ignored
    pub const UNTEXTURED: u32 = 0;
    /// Textured
    pub const TEXTURED: u32 = 1;
    pub const OUTLINE: u32 = 256;
    pub const INVERT: u32 = 4096;
    pub const VIEWPORT_TEXTURE: u32 = 8192;
    pub const TEXT_GLYPH: u32 = 16384;
    pub const TEXT_EFFECT_OUTLINE: u32 = 32768;
    pub const TEXT_EFFECT_SHADOW: u32 = 65536;
    pub const TEXT_GROUP: u32 = 131072;
}

fn node_flags(image: AssetId<Image>, node_type: NodeType) -> u32 {
    let texture = if image == AssetId::default() {
        shader_flags::UNTEXTURED
    } else {
        shader_flags::TEXTURED
    };
    texture
        | match node_type {
            NodeType::Outline => shader_flags::OUTLINE,
            NodeType::Inverted => shader_flags::INVERT,
            NodeType::Viewport => shader_flags::VIEWPORT_TEXTURE,
            _ => 0,
        }
}

fn is_rect_image(node: &ExtractedUiNode) -> bool {
    matches!(&node.item, ExtractedUiItem::Node { border_radius, border, node_type, .. }
        if node_flags(node.image, *node_type) == shader_flags::TEXTURED
            && *border_radius == ResolvedBorderRadius::default() && *border == BorderRect::ZERO)
}

/// The reference rectangular SDF is <= -1 throughout this local rectangle,
/// leaving a guard outside its half-unit AA band. Clipping and affine transforms
/// preserve the original interpolated local points; neither needs pixel alignment.
fn image_interior(node: &ExtractedUiNode) -> Option<Rect> {
    if !is_rect_image(node) || !node.transform.is_finite() {
        return None;
    }
    let ExtractedUiItem::Node { rect, .. } = &node.item else {
        return None;
    };
    let size = rect.size();
    let area = (node.transform.matrix2.determinant() * size.x * size.y).abs();
    if !size.is_finite()
        || size.min_element() < 8.
        || size.max_element() > 16384.
        || !area.is_finite()
        || area < 4096.
        || QUAD_VERTEX_POSITIONS.iter().any(|corner| {
            let p = node.transform.transform_point2(*corner * size);
            !p.is_finite() || p.abs().max_element() > 16384.
        })
    {
        return None;
    }
    let half = 0.5 * size - Vec2::ONE;
    Some(Rect::from_corners(-half, half))
}

fn is_pixel_aligned_image(node: &ExtractedUiNode) -> bool {
    if !is_rect_image(node) || node.transform.matrix2 != bevy_math::Mat2::IDENTITY {
        return false;
    }
    let ExtractedUiItem::Node { rect, .. } = &node.item else {
        return false;
    };
    let size = rect.size();
    if !size.is_finite() || size.min_element() <= 0.0 {
        return false;
    }
    let min = node.transform.translation - 0.5 * size;
    let max = node.transform.translation + 0.5 * size;
    // Keep the proof in a range with ample f32 subpixel precision. A clip can
    // only remove fragments; it cannot expose pixels outside these original edges.
    min.is_finite()
        && max.is_finite()
        && min.abs().max(max.abs()).max_element() <= 16_384.0
        && min.floor() == min
        && max.floor() == max
}

#[derive(Clone, Copy)]
struct PreparedUiPipelines {
    encoded: bool,
    basic: Option<CachedRenderPipelineId>,
    full: CachedRenderPipelineId,
    rect: Option<CachedRenderPipelineId>,
    images: bool,
    aligned: Option<CachedRenderPipelineId>,
    interior: Option<CachedRenderPipelineId>,
}

#[derive(Resource, Default)]
#[doc(hidden)]
pub struct UiPipelineVariants(HashMap<Entity, PreparedUiPipelines>);

pub fn queue_uinodes(
    gpu_images: Res<RenderAssets<GpuImage>>,
    extracted_uinodes: Res<ExtractedUiNodes>,
    ui_pipeline: Res<UiPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<
        (&UiCameraView, Option<&UiAntiAlias>, &UiShaderVariants),
        With<ExtractedView>,
    >,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    mut prepared: ResMut<UiPipelineVariants>,
) {
    prepared.0.clear();
    for node in &extracted_uinodes.uinodes {
        let Ok((camera, antialias, variants)) = render_views.get(node.extracted_camera_entity)
        else {
            continue;
        };
        let entity = camera.0;
        if prepared.0.contains_key(&entity) {
            continue;
        }
        let Ok(view) = camera_views.get(camera.0) else {
            continue;
        };
        let key = UiPipelineKey {
            basic_image: false,
            encoded_image: false,
            relaxed_precision: variants.relaxed_precision,
            relaxed_colors: variants.relaxed_colors,
            target_format: variants.render_target_format(view.target_format),
            anti_alias: matches!(antialias, None | Some(UiAntiAlias::On)),
            image_rect: false,
            rect_math: false,
            lean_io: false,
            pixel_aligned: false,
            implicit_aligned: false,
        };
        let full = pipelines.specialize(&pipeline_cache, &ui_pipeline, key);
        let rect = variants.images.then(|| {
            pipelines.specialize(
                &pipeline_cache,
                &ui_pipeline,
                UiPipelineKey {
                    image_rect: true,
                    rect_math: variants.rect_math,
                    lean_io: variants.lean_io,
                    ..key
                },
            )
        });
        let coverage_free = ((variants.aligned_images || variants.image_interiors)
            && view.viewport.x == 0
            && view.viewport.y == 0)
            .then(|| {
                pipelines.specialize(
                    &pipeline_cache,
                    &ui_pipeline,
                    UiPipelineKey {
                        image_rect: true,
                        pixel_aligned: true,
                        implicit_aligned: variants.implicit_aligned_images,
                        ..key
                    },
                )
            })
            .filter(|id| pipeline_cache.get_render_pipeline(*id).is_some());
        let basic = (variants.basic_images && view.viewport.x == 0 && view.viewport.y == 0)
            .then(|| {
                pipelines.specialize(
                    &pipeline_cache,
                    &ui_pipeline,
                    UiPipelineKey {
                        basic_image: true,
                        encoded_image: variants.encoded_images,
                        image_rect: true,
                        ..key
                    },
                )
            })
            .filter(|id| {
                pipeline_cache.get_render_pipeline(*id).is_some()
                    && pipeline_cache.get_render_pipeline(full).is_some()
            });
        prepared.0.insert(
            entity,
            PreparedUiPipelines {
                encoded: variants.encoded_images,
                basic,
                full,
                rect,
                images: variants.images
                    && pipeline_cache.get_render_pipeline(full).is_some()
                    && rect.is_some_and(|id| pipeline_cache.get_render_pipeline(id).is_some()),
                aligned: coverage_free.filter(|_| variants.aligned_images),
                interior: coverage_free.filter(|_| variants.image_interiors),
            },
        );
    }
    let draw_function = draw_functions.read().id::<DrawUi>();
    let mut current_camera_entity = Entity::PLACEHOLDER;
    let mut current_phase = None;

    for (index, extracted_uinode) in extracted_uinodes.uinodes.iter().enumerate() {
        if current_camera_entity != extracted_uinode.extracted_camera_entity {
            current_phase = render_views
                .get(extracted_uinode.extracted_camera_entity)
                .ok()
                .and_then(|(default_camera_view, _, _)| {
                    camera_views
                        .get(default_camera_view.0)
                        .ok()
                        .and_then(|view| {
                            transparent_render_phases
                                .get_mut(&view.retained_view_entity)
                                .map(|transparent_phase| {
                                    (prepared.0[&default_camera_view.0], transparent_phase)
                                })
                        })
                });
            current_camera_entity = extracted_uinode.extracted_camera_entity;
        }

        let Some((choice, transparent_phase)) = current_phase.as_mut() else {
            continue;
        };

        let pipeline = if choice.images && is_rect_image(extracted_uinode) {
            choice
                .basic
                .filter(|_| {
                    !choice.encoded
                        || gpu_images
                            .get(extracted_uinode.image)
                            .is_some_and(has_encoded_view)
                })
                .or_else(|| {
                    choice
                        .aligned
                        .filter(|_| is_pixel_aligned_image(extracted_uinode))
                })
                .unwrap_or_else(|| choice.rect.expect("image specialization enabled"))
        } else {
            choice.full
        };

        transparent_phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (extracted_uinode.render_entity, extracted_uinode.main_entity),
            sort_key: FloatOrd(extracted_uinode.z_order),
            index,
            // batch_range will be calculated in prepare_uinodes
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: true,
        });
    }
}

#[derive(Resource, Default)]
pub struct ImageNodeBindGroups {
    pub values: HashMap<AssetId<Image>, BindGroup>,
    pub encoded: HashMap<AssetId<Image>, (TextureView, BindGroup)>,
}

pub fn prepare_uinodes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<UiMesh<UiVertex>>,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    view_uniforms: Res<ViewUniforms>,
    ui_pipeline: Res<UiPipeline>,
    mut image_bind_groups: ResMut<ImageNodeBindGroups>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    events: Res<SpriteAssetEvents>,
    mut previous_len: Local<usize>,
    prepared: Res<UiPipelineVariants>,
    render_views: Query<&UiCameraView>,
    (
        scissor_views,
        mut composite_bounds,
        draw_functions,
        separate_views,
        retained_views,
        mut retained_layers,
        targets,
        text_groups,
    ): (
        Query<
            &ExtractedView,
            (
                With<UiViewTarget>,
                Or<(With<UiCompositeScissor>, With<UiRetainedLayer>)>,
            ),
        >,
        ResMut<UiCompositeBounds>,
        Res<DrawFunctions<TransparentUi>>,
        Query<&ExtractedView, With<UiSeparateBatches>>,
        Query<(&ExtractedView, &UiViewTarget, &UiRetainedLayer)>,
        ResMut<retained_layer::RetainedUiLayers>,
        Query<(
            &bevy_render::view::ViewTarget,
            &bevy_render::camera::ExtractedCamera,
        )>,
        Option<Res<TextGroupCache>>,
    ),
) {
    // Clear even without a view binding: missing preparation must fall back, not
    // reuse coverage from an earlier frame.
    composite_bounds.0.clear();
    let changed_images: Vec<_> = events
        .images
        .iter()
        .filter_map(|event| match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            | AssetEvent::Unused { id } => Some(*id),
            _ => None,
        })
        .collect();
    retained_layers.begin_frame(&changed_images);

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Added { .. } |
            // Images don't have dependencies
            AssetEvent::LoadedWithDependencies { .. } => {}
            AssetEvent::Unused { id } | AssetEvent::Modified { id } | AssetEvent::Removed { id } => {
                image_bind_groups.values.remove(id);
                image_bind_groups.encoded.remove(id);
            }
        };
    }

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches: Vec<(Entity, UiBatch)> = Vec::with_capacity(*previous_len);
        let mut encoded_images = HashSet::<AssetId<Image>>::default();

        ui_meta.vertices.clear();
        ui_meta.indices.clear();
        ui_meta.view_bind_group = Some(render_device.create_bind_group(
            "ui_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&ui_pipeline.view_layout),
            &BindGroupEntries::single(view_binding),
        ));

        if !scissor_views.is_empty() {
            let draw_ui = draw_functions.read().id::<DrawUi>();
            for view in &scissor_views {
                if let Some(phase) = phases.get(&view.retained_view_entity) {
                    // Inspect every draw before the extracted-node lookup can
                    // quietly skip foreign items with their own vertex transforms.
                    composite_bounds.0.insert(
                        view.retained_view_entity,
                        CompositeBounds::for_draw_functions(
                            phase.items.values().map(|item| item.draw_function),
                            draw_ui,
                        ),
                    );
                }
            }
        }

        for (retained_view_entity, ui_phase) in &mut phases.0 {
            let phase_vertex_start = ui_meta.vertices.len();
            let phase_index_start = ui_meta.indices.len();
            let mut cacheable = true;
            let mut layer_images = Vec::new();
            let mut interior_pipelines = Vec::new();
            let mut first: Option<usize> = None;
            let separate = separate_views
                .iter()
                .any(|view| &view.retained_view_entity == retained_view_entity);
            let mut bounds = composite_bounds.0.get_mut(retained_view_entity);

            for item_index in 0..ui_phase.items.len() {
                let item = &ui_phase.items[item_index];
                let Some(extracted_uinode) = extracted_uinodes
                    .uinodes
                    .get(item.index)
                    .filter(|n| item.entity() == n.render_entity)
                else {
                    continue;
                };

                let Some(image) = gpu_images.get(extracted_uinode.image) else {
                    // Asset loading and GPU uploads can lag behind UI extraction.
                    // Retry next frame without batching across this skipped item.
                    cacheable = false;
                    continue;
                };
                if retained_views
                    .iter()
                    .any(|(v, _, _)| &v.retained_view_entity == retained_view_entity)
                {
                    let generation = if image.texture.usage().intersects(
                        TextureUsages::RENDER_ATTACHMENT | TextureUsages::STORAGE_BINDING,
                    ) {
                        text_groups.as_ref().and_then(|groups| {
                            groups.retained_image_generation(
                                extracted_uinode.main_entity.id(),
                                extracted_uinode.image,
                            )
                        })
                    } else {
                        Some(0)
                    };
                    cacheable &= generation.is_some();
                    layer_images.push((
                        extracted_uinode.image,
                        if prepared.0[&render_views
                            .get(extracted_uinode.extracted_camera_entity)
                            .expect("queued view")
                            .0]
                            .encoded
                            && prepared.0[&render_views
                                .get(extracted_uinode.extracted_camera_entity)
                                .expect("queued view")
                                .0]
                                .basic
                                == Some(item.pipeline)
                        {
                            match image_bind_groups.encoded.get(&extracted_uinode.image) {
                                Some((view, _)) => view.id(),
                                None => {
                                    cacheable = false;
                                    image.texture_view.id()
                                }
                            }
                        } else {
                            image.texture_view.id()
                        },
                        image.sampler.id(),
                        generation.unwrap_or(0),
                    ));
                }
                let start = ui_meta.indices.len() as u32;
                let vertex_start = ui_meta.vertices.len();
                let mut interior_draw = None;
                match &extracted_uinode.item {
                    ExtractedUiItem::Node {
                        atlas_scaling,
                        flip_x,
                        flip_y,
                        border_radius,
                        border,
                        node_type,
                        rect,
                        color,
                    } => {
                        let flags = node_flags(extracted_uinode.image, *node_type);

                        let rect_size = rect.size();

                        let transform = extracted_uinode.transform;

                        let uvs = if flags & shader_flags::TEXTURED == 0 {
                            [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]
                        } else {
                            let mut uinode_rect = *rect;
                            // Rescale atlases. This is done here because we need texture data that might not be available in Extract.
                            let atlas_extent = atlas_scaling
                                .map(|scaling| image.size_2d().as_vec2() * scaling)
                                .unwrap_or(uinode_rect.max);
                            if *flip_x {
                                core::mem::swap(&mut uinode_rect.max.x, &mut uinode_rect.min.x);
                            }
                            if *flip_y {
                                core::mem::swap(&mut uinode_rect.max.y, &mut uinode_rect.min.y);
                            }
                            [
                                Vec2::new(uinode_rect.min.x, uinode_rect.min.y),
                                Vec2::new(uinode_rect.max.x, uinode_rect.min.y),
                                Vec2::new(uinode_rect.max.x, uinode_rect.max.y),
                                Vec2::new(uinode_rect.min.x, uinode_rect.max.y),
                            ]
                            .map(|pos| pos / atlas_extent)
                        };

                        let color = color.to_f32_array();
                        let choice = prepared.0[&render_views
                            .get(extracted_uinode.extracted_camera_entity)
                            .expect("queued UI camera exists")
                            .0];
                        assert!(
                            Some(item.pipeline) != choice.rect || is_rect_image(extracted_uinode),
                            "IMAGE_RECT pipeline queued for an incompatible UI node"
                        );
                        assert!(
                            choice.aligned != Some(item.pipeline)
                                || is_pixel_aligned_image(extracted_uinode),
                            "pixel-aligned image pipeline queued without an exact edge proof"
                        );

                        // Outlines have no coverage in the interior.
                        let omit_interior = matches!(node_type, NodeType::Outline);
                        assert!(
                            choice.basic != Some(item.pipeline) || is_rect_image(extracted_uinode),
                            "basic image shader queued for incompatible geometry"
                        );
                        let interior = (choice.images
                            && choice.aligned != Some(item.pipeline)
                            && choice.basic != Some(item.pipeline))
                        .then(|| choice.interior.zip(image_interior(extracted_uinode)))
                        .flatten();
                        let hole = if omit_interior {
                            rounded_inner_rect(rect_size, *border, *border_radius)
                        } else if let Some((_, rect)) = interior {
                            Some(rect)
                        } else {
                            None
                        };
                        let mut parts = rect_without_hole(rect_size, hole);
                        let edge_parts = parts.len();
                        if let Some((_, rect)) = interior {
                            parts.push(rect);
                        }
                        for (part_index, part) in parts.into_iter().enumerate() {
                            if part_index == edge_parts
                                && let Some((pipeline, _)) = interior
                            {
                                interior_draw = Some(UiImageInterior {
                                    range: ui_meta.indices.len() as u32..0,
                                    pipeline,
                                });
                            }
                            let corners = [0, 1, 2, 3].map(|i| {
                                let pos = QUAD_VERTEX_POSITIONS[i];
                                let point = part.center() + pos * part.size();
                                let uv = if hole.is_none() {
                                    uvs[i]
                                } else {
                                    uvs[0]
                                        + (point / rect_size + Vec2::splat(0.5)) * (uvs[2] - uvs[0])
                                };
                                (transform.transform_point2(point), (uv, point))
                            });
                            let vertices = clip_polygon(
                                extracted_uinode.clip.as_ref(),
                                &corners,
                                |a, b, t| (a.0.lerp(b.0, t), a.1.lerp(b.1, t)),
                            );
                            ui_meta.push_triangle_fan(vertices.iter().map(
                                |&(position, (uv, point))| UiVertex {
                                    position: position.extend(0.).into(),
                                    uv: uv.into(),
                                    color,
                                    flags,
                                    radius: (*border_radius).into(),
                                    border: [
                                        border.min_inset.x,
                                        border.min_inset.y,
                                        border.max_inset.x,
                                        border.max_inset.y,
                                    ],
                                    size: rect_size.into(),
                                    point: point.into(),
                                },
                            ));
                        }
                    }
                    ExtractedUiItem::Glyphs { range } => {
                        let atlas_extent = image.size_2d().as_vec2();

                        for glyph in &extracted_uinodes.glyphs[range.clone()] {
                            let color = glyph.color.to_f32_array();
                            let glyph_rect = glyph.rect;
                            let rect_size = glyph_rect.size();
                            let flags = glyph.flags;

                            // Specify the corners of the glyph
                            let positions = QUAD_VERTEX_POSITIONS.map(|pos| {
                                extracted_uinode
                                    .transform
                                    .transform_point2(glyph.translation + pos * glyph_rect.size())
                            });

                            let vertices = clip_polygon(
                                extracted_uinode.clip.as_ref(),
                                &[
                                    (
                                        positions[0],
                                        Vec2::new(glyph.rect.min.x, glyph.rect.min.y)
                                            / atlas_extent,
                                    ),
                                    (
                                        positions[1],
                                        Vec2::new(glyph.rect.max.x, glyph.rect.min.y)
                                            / atlas_extent,
                                    ),
                                    (
                                        positions[2],
                                        Vec2::new(glyph.rect.max.x, glyph.rect.max.y)
                                            / atlas_extent,
                                    ),
                                    (
                                        positions[3],
                                        Vec2::new(glyph.rect.min.x, glyph.rect.max.y)
                                            / atlas_extent,
                                    ),
                                ],
                                Vec2::lerp,
                            );
                            ui_meta.push_triangle_fan(vertices.iter().map(|vertex| UiVertex {
                                position: vertex.0.extend(0.).into(),
                                uv: vertex.1.into(),
                                color,
                                flags,
                                radius: [0.0; 4],
                                border: [0.0; 4],
                                size: rect_size.into(),
                                point: [0.0; 2],
                            }));
                        }
                    }
                }
                // Both nodes and glyphs contribute their actual clipped vertices.
                // Disabled views never scan geometry or collect bounds.
                if let Some(bounds) = bounds.as_deref_mut() {
                    bounds.include_vertices(&ui_meta.vertices.values()[vertex_start..]);
                }
                let choices = prepared.0[&render_views
                    .get(extracted_uinode.extracted_camera_entity)
                    .expect("queued UI camera exists")
                    .0];
                let basic_boundary = first.is_some_and(|previous| {
                    (Some(ui_phase.items[previous].pipeline) == choices.basic)
                        != (Some(ui_phase.items[item_index].pipeline) == choices.basic)
                });
                if (separate || basic_boundary)
                    && start != ui_meta.indices.len() as u32
                    && let Some(previous) = first
                    && ui_phase.items[previous].pipeline != ui_phase.items[item_index].pipeline
                {
                    first = None;
                }
                let interior_draw = interior_draw.and_then(|mut interior| {
                    interior.range.end = ui_meta.indices.len() as u32;
                    (!interior.range.is_empty()).then_some(interior)
                });
                if interior_draw.is_some() {
                    first = None;
                }
                let item_entity = item.entity();
                let encoded =
                    choices.encoded && choices.basic == Some(ui_phase.items[item_index].pipeline);
                if encoded {
                    assert!(
                        has_encoded_view(image),
                        "encoded UI batch lacks a compatible alternate view"
                    );
                    encoded_images.insert(extracted_uinode.image);
                    commands.entity(item_entity).insert(UiEncodedBatch);
                    // Even an equal texture must never absorb an untextured or
                    // reference draw and subsequently promote its shader.
                    first = None;
                }
                UiBatch::append(
                    &mut ui_phase.items,
                    &mut first,
                    &mut batches,
                    item_index,
                    extracted_uinode.image,
                    start..ui_meta.indices.len() as u32,
                    prepared.0[&render_views
                        .get(extracted_uinode.extracted_camera_entity)
                        .expect("queued UI camera exists")
                        .0]
                        .full,
                );
                if let Some(interior) = interior_draw {
                    interior_pipelines.push(interior.pipeline);
                    commands.entity(item_entity).insert(interior);
                    first = None;
                }
                if encoded {
                    first = None;
                }
            }
            if let Some((view, target, settings)) = retained_views
                .iter()
                .find(|(v, _, _)| &v.retained_view_entity == retained_view_entity)
                && let Ok((target, camera)) = targets.get(target.0)
                && camera.viewport.is_none()
                && let Some(rect) = retained_layer::crop(
                    bounds.as_deref(),
                    view.viewport,
                    bevy_math::UVec2::new(
                        target.main_texture().width(),
                        target.main_texture().height(),
                    ),
                )
            {
                cacheable &= ui_phase
                    .items
                    .values()
                    .all(|item| pipeline_cache.get_render_pipeline(item.pipeline).is_some());
                let contents = (settings.cache && cacheable).then(|| retained_layer::Contents {
                    vertices: bytemuck::cast_slice(
                        &ui_meta.vertices.values()[phase_vertex_start..],
                    )
                    .to_vec(),
                    indices: ui_meta.indices.values()[phase_index_start..]
                        .iter()
                        .map(|index| *index - phase_vertex_start as u32)
                        .collect(),
                    pipelines: ui_phase
                        .items
                        .values()
                        .map(|item| item.pipeline)
                        .chain(interior_pipelines)
                        .collect(),
                    images: layer_images,
                });
                retained_layers.plans.insert(
                    *retained_view_entity,
                    retained_layer::Plan {
                        rect,
                        full_size: bevy_math::UVec2::new(view.viewport.z, view.viewport.w),
                        contents,
                    },
                );
            }
        }

        for (_, batch) in &batches {
            if let Some(image) = gpu_images
                .get(batch.image)
                .filter(|image| encoded_images.contains(&batch.image) && has_encoded_view(image))
            {
                image_bind_groups
                    .encoded
                    .entry(batch.image)
                    .or_insert_with(|| {
                        let mut descriptor =
                            image.texture_view_descriptor.clone().unwrap_or_default();
                        descriptor.format = Some(image.texture_descriptor.format);
                        let view = image.texture.create_view(&descriptor);
                        let binding = render_device.create_bind_group(
                            "ui_encoded_image",
                            &pipeline_cache.get_bind_group_layout(&ui_pipeline.image_layout),
                            &BindGroupEntries::sequential((&view, &image.sampler)),
                        );
                        (view, binding)
                    });
            }
            image_bind_groups
                .values
                .entry(batch.image)
                .or_insert_with(|| {
                    let image = gpu_images
                        .get(batch.image)
                        .expect("UI batches only contain GPU-ready images");
                    render_device.create_bind_group(
                        "ui_material_bind_group",
                        &pipeline_cache.get_bind_group_layout(&ui_pipeline.image_layout),
                        &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
                    )
                });
        }
        ui_meta.vertices.write_buffer(&render_device, &render_queue);
        ui_meta.indices.write_buffer(&render_device, &render_queue);
        *previous_len = batches.len();
        commands.try_insert_batch(batches);
    }
    extracted_uinodes.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::render_phase::DrawFunctionId;

    #[test]
    fn image_variant_accepts_only_plain_textured_rectangles() {
        let mut images = Assets::<Image>::default();
        let image = images.add(Image::default());
        let mut node = ExtractedUiNode {
            z_order: 0.,
            image: image.id(),
            clip: None,
            extracted_camera_entity: Entity::PLACEHOLDER,
            main_entity: Entity::PLACEHOLDER.into(),
            render_entity: Entity::PLACEHOLDER,
            transform: Affine2::IDENTITY,
            item: ExtractedUiItem::Glyphs { range: 0..0 },
        };
        assert!(!is_rect_image(&node));
        for kind in [
            NodeType::Rect,
            NodeType::Outline,
            NodeType::Inverted,
            NodeType::Viewport,
        ] {
            for radius in [0., 1.] {
                for border in [0., 1.] {
                    node.item = ExtractedUiItem::Node {
                        color: LinearRgba::WHITE,
                        rect: Rect::from_corners(Vec2::ZERO, Vec2::splat(20.)),
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border_radius: ResolvedBorderRadius {
                            top_left: radius,
                            ..Default::default()
                        },
                        border: BorderRect {
                            min_inset: Vec2::new(border, 0.),
                            max_inset: Vec2::ZERO,
                        },
                        node_type: kind,
                    };
                    assert_eq!(
                        is_rect_image(&node),
                        kind == NodeType::Rect && radius == 0. && border == 0.
                    );
                    assert_eq!(is_pixel_aligned_image(&node), is_rect_image(&node));
                    if is_rect_image(&node) {
                        for transform in [
                            Affine2::from_translation(Vec2::splat(0.5)),
                            Affine2::from_translation(Vec2::splat(f32::NAN)),
                            Affine2::from_translation(Vec2::splat(20_000.0)),
                            Affine2::from_scale(Vec2::splat(1.25)),
                            Affine2::from_angle(0.21),
                        ] {
                            node.transform = transform;
                            assert!(!is_pixel_aligned_image(&node));
                        }
                        node.transform = Affine2::IDENTITY;
                    }
                    node.image = AssetId::default();
                    assert!(!is_rect_image(&node));
                    node.image = image.id();
                }
            }
        }
    }

    #[test]
    fn mixed_batches_choose_common_shader_without_extra_draws() {
        let full = CachedRenderPipelineId::new(0);
        let rect = CachedRenderPipelineId::new(1);
        let mut images = Assets::<Image>::default();
        let image = images.add(Image::default());
        let other = images.add(Image::default());
        for (pipelines, empty, change_image, expected) in [
            (vec![rect, rect], false, false, rect),
            (vec![full, rect], false, false, full),
            (vec![rect, full, rect], false, false, full),
            (vec![rect, full], true, false, rect),
            (vec![rect, full], false, true, rect),
        ] {
            let mut items: SortedPhaseItems<_> = pipelines
                .iter()
                .enumerate()
                .map(|(index, &pipeline)| {
                    let entity = Entity::from_raw_u32(index as u32).unwrap();
                    (
                        (entity, entity.into()),
                        TransparentUi {
                            sort_key: FloatOrd(index as f32),
                            entity: (entity, entity.into()),
                            pipeline,
                            draw_function: DrawFunctionId(0),
                            batch_range: 0..0,
                            extra_index: PhaseItemExtraIndex::None,
                            index,
                            indexed: true,
                        },
                    )
                })
                .collect();
            let mut first = None;
            let mut batches = Vec::new();
            for index in 0..pipelines.len() {
                let start = index as u32 * 6;
                let range = start..if empty && index == 1 {
                    start
                } else {
                    start + 6
                };
                UiBatch::append(
                    &mut items,
                    &mut first,
                    &mut batches,
                    index,
                    if change_image && index == 1 {
                        other.id()
                    } else {
                        image.id()
                    },
                    range,
                    full,
                );
            }
            assert_eq!(batches.len(), if change_image { 2 } else { 1 });
            assert_eq!(items[0].pipeline, expected);
            if change_image {
                assert_eq!(items[1].pipeline, full);
            }
        }
    }

    #[test]
    fn image_interior_accepts_fractional_affine_images_but_not_degenerate_or_small_geometry() {
        let mut images = Assets::<Image>::default();
        let image = images.add(Image::default());
        let mut node = ExtractedUiNode {
            z_order: 0.,
            image: image.id(),
            clip: None,
            extracted_camera_entity: Entity::PLACEHOLDER,
            main_entity: Entity::PLACEHOLDER.into(),
            render_entity: Entity::PLACEHOLDER,
            transform: Affine2::IDENTITY,
            item: ExtractedUiItem::Node {
                color: LinearRgba::WHITE,
                rect: Rect::from_corners(Vec2::ZERO, Vec2::splat(128.)),
                atlas_scaling: None,
                flip_x: false,
                flip_y: false,
                border_radius: ResolvedBorderRadius::ZERO,
                border: BorderRect::ZERO,
                node_type: NodeType::Rect,
            },
        };
        for transform in [
            Affine2::IDENTITY,
            Affine2::from_translation(Vec2::splat(0.125)),
            Affine2::from_angle(0.21),
            Affine2::from_scale(Vec2::new(0.75, 2.)),
        ] {
            node.transform = transform;
            let interior =
                image_interior(&node).expect("large affine rectangle has a safe interior");
            assert_eq!(interior, Rect::new(-63., -63., 63., 63.));
        }
        for transform in [
            Affine2::from_scale(Vec2::splat(0.01)),
            Affine2::from_scale(Vec2::ZERO),
            Affine2::from_translation(Vec2::splat(f32::NAN)),
        ] {
            node.transform = transform;
            assert!(image_interior(&node).is_none());
        }
    }

    #[test]
    fn image_batches_preserve_order_and_ignore_clipped_textures() {
        let mut images = Assets::<Image>::default();
        let a = images.add(Image::default());
        let b = images.add(Image::default());
        let white = AssetId::default();
        let mut items: SortedPhaseItems<_> = (0..10)
            .map(|index| {
                let entity = Entity::from_raw_u32(index as u32).unwrap();
                let item = TransparentUi {
                    sort_key: FloatOrd(index as f32),
                    entity: (entity, entity.into()),
                    pipeline: CachedRenderPipelineId::new(0),
                    draw_function: DrawFunctionId(0),
                    batch_range: 0..0,
                    extra_index: PhaseItemExtraIndex::None,
                    index,
                    indexed: true,
                };
                (item.entity, item)
            })
            .collect();
        let mut batches = Vec::new();
        let mut first = None;
        // A rect-image batch may absorb a same-texture rounded node or an
        // untextured decoration. It must then use the common full shader.
        items[1].pipeline = CachedRenderPipelineId::new(1);
        items[6].pipeline = CachedRenderPipelineId::new(1);
        items[8].pipeline = CachedRenderPipelineId::new(1);
        for (index, image, range) in [
            (0, b.id(), 0..0),
            (1, white, 0..6),
            (2, a.id(), 6..12),
            (3, white, 12..18),
            (4, b.id(), 18..18),
            (5, a.id(), 18..24),
            (6, b.id(), 24..30),
            // Item 7 has a pending image or belongs to another preparer.
            (8, b.id(), 30..36),
        ] {
            UiBatch::append(
                &mut items,
                &mut first,
                &mut batches,
                index,
                image,
                range,
                CachedRenderPipelineId::new(0),
            );
        }
        assert_eq!(batches.len(), 3);
        assert_eq!(items[1].pipeline, CachedRenderPipelineId::new(0));
        assert_eq!(items[6].pipeline, CachedRenderPipelineId::new(1));
        // A new view cannot merge with the preceding view's last draw.
        first = None;
        UiBatch::append(
            &mut items,
            &mut first,
            &mut batches,
            9,
            b.id(),
            36..42,
            CachedRenderPipelineId::new(0),
        );
        assert_eq!(
            batches
                .iter()
                .map(|(_, batch)| (batch.image, batch.range.clone()))
                .collect::<Vec<_>>(),
            [
                (a.id(), 0..24),
                (b.id(), 24..30),
                (b.id(), 30..36),
                (b.id(), 36..42)
            ]
        );
        assert_eq!(
            items
                .values()
                .map(|item| item.batch_range.clone())
                .collect::<Vec<_>>(),
            [0..0, 1..6, 0..0, 0..0, 0..0, 0..0, 6..7, 0..0, 8..9, 9..10]
        );
    }
}
