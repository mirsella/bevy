use crate::{
    ComputedUiRenderTargetInfo, ContentSize, Measure, MeasureArgs, Node, NodeMeasure, ResolvedAxis,
    VisualBox,
};
use bevy_asset::{AsAssetId, AssetId, Assets, Handle};
use bevy_color::Color;
use bevy_ecs::prelude::*;
use bevy_image::{prelude::*, TRANSPARENT_IMAGE_HANDLE};
use bevy_math::{Rect, UVec2, Vec2};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_sprite::TextureSlicer;
use taffy::{MaybeMath, ResolveOrZero};

/// A UI Node that renders an image.
#[derive(Component, Clone, Debug, Reflect, FromTemplate)]
#[reflect(Component, Default, Debug, Clone)]
#[require(Node, ImageNodeSize)]
pub struct ImageNode {
    /// The tint color used to draw the image.
    ///
    /// This is multiplied by the color of each pixel in the image.
    /// The field value defaults to solid white, which will pass the image through unmodified.
    pub color: Color,
    /// Handle to the texture.
    ///
    /// This defaults to a [`TRANSPARENT_IMAGE_HANDLE`], which points to a fully transparent 1x1 texture.
    pub image: Handle<Image>,
    /// The (optional) texture atlas used to render the image.
    #[template(built_in)]
    pub texture_atlas: Option<TextureAtlas>,
    /// Whether the image should be flipped along its x-axis.
    pub flip_x: bool,
    /// Whether the image should be flipped along its y-axis.
    pub flip_y: bool,
    /// An optional rectangle representing the region of the image to render, instead of rendering
    /// the full image. This is an easy one-off alternative to using a [`TextureAtlas`].
    ///
    /// When used with a [`TextureAtlas`], the rect
    /// is offset by the atlas's minimal (top-left) corner position.
    pub rect: Option<Rect>,
    /// Controls how the image is altered to fit within the layout and how the layout algorithm determines the space to allocate for the image.
    pub image_mode: NodeImageMode,
    /// Which region of the UI node the image should be drawn within.
    pub visual_box: VisualBox,
}

impl Default for ImageNode {
    /// A transparent 1x1 image with a solid white tint.
    ///
    /// # Warning
    ///
    /// This will be invisible by default.
    /// To set this to a visible image, you need to set the `texture` field to a valid image handle,
    /// or use [`Handle<Image>`]'s default 1x1 solid white texture (as is done in [`ImageNode::solid_color`]).
    fn default() -> Self {
        ImageNode {
            // This should be white because the tint is multiplied with the image,
            // so if you set an actual image with default tint you'd want its original colors
            color: Color::WHITE,
            texture_atlas: None,
            // This texture needs to be transparent by default, to avoid covering the background color
            image: TRANSPARENT_IMAGE_HANDLE,
            flip_x: false,
            flip_y: false,
            rect: None,
            image_mode: NodeImageMode::Auto,
            visual_box: VisualBox::ContentBox,
        }
    }
}

impl ImageNode {
    /// Create a new [`ImageNode`] with the given texture.
    pub fn new(texture: Handle<Image>) -> Self {
        Self {
            image: texture,
            color: Color::WHITE,
            ..Default::default()
        }
    }

    /// Create a solid color [`ImageNode`].
    ///
    /// This is primarily useful for debugging / mocking the extents of your image.
    pub fn solid_color(color: Color) -> Self {
        Self {
            image: Handle::default(),
            color,
            flip_x: false,
            flip_y: false,
            texture_atlas: None,
            rect: None,
            image_mode: NodeImageMode::Auto,
            visual_box: VisualBox::ContentBox,
        }
    }

    /// Create a [`ImageNode`] from an image, with an associated texture atlas
    pub fn from_atlas_image(image: Handle<Image>, atlas: TextureAtlas) -> Self {
        Self {
            image,
            texture_atlas: Some(atlas),
            ..Default::default()
        }
    }

    /// Set the color tint
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Flip the image along its x-axis
    #[must_use]
    pub const fn with_flip_x(mut self) -> Self {
        self.flip_x = true;
        self
    }

    /// Flip the image along its y-axis
    #[must_use]
    pub const fn with_flip_y(mut self) -> Self {
        self.flip_y = true;
        self
    }

    #[must_use]
    pub const fn with_rect(mut self, rect: Rect) -> Self {
        self.rect = Some(rect);
        self
    }

    #[must_use]
    pub const fn with_mode(mut self, mode: NodeImageMode) -> Self {
        self.image_mode = mode;
        self
    }
}

impl From<Handle<Image>> for ImageNode {
    fn from(texture: Handle<Image>) -> Self {
        Self::new(texture)
    }
}

impl AsAssetId for ImageNode {
    type Asset = Image;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.image.id()
    }
}

/// Controls how the image is altered to fit within the layout and how the layout algorithm determines the space in the layout for the image
#[derive(Default, Debug, Clone, PartialEq, Reflect)]
#[reflect(Clone, Default, PartialEq)]
pub enum NodeImageMode {
    /// The image will be sized automatically by taking the size of the source image and applying any layout constraints.
    #[default]
    Auto,
    /// The image will be resized to match the size of the node. The image's original size and aspect ratio will be ignored.
    Stretch,
    /// The texture will be cut in 9 slices, keeping the texture in proportions on resize
    Sliced(TextureSlicer),
    /// The texture will be repeated if stretched beyond `stretched_value`
    Tiled {
        /// Should the image repeat horizontally
        tile_x: bool,
        /// Should the image repeat vertically
        tile_y: bool,
        /// The texture will repeat when the ratio between the *drawing dimensions* of texture and the
        /// *original texture size* are above this value.
        stretch_value: f32,
    },
}

impl NodeImageMode {
    /// Returns true if this mode uses slices internally ([`NodeImageMode::Sliced`] or [`NodeImageMode::Tiled`])
    #[inline]
    pub const fn uses_slices(&self) -> bool {
        matches!(
            self,
            NodeImageMode::Sliced(..) | NodeImageMode::Tiled { .. }
        )
    }
}

/// The size of the image's texture
///
/// This component is updated automatically by [`update_image_content_size_system`]
#[derive(Component, Debug, Copy, Clone, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct ImageNodeSize {
    /// The size of the image's texture
    ///
    /// This field is updated automatically by [`update_image_content_size_system`]
    size: UVec2,
    /// Inputs to the measure last installed by this system. Tint and UV changes do not
    /// affect layout, even though they mark the whole `ImageNode` as changed.
    measure: Option<ImageMeasure>,
}

impl ImageNodeSize {
    /// The size of the image's texture
    #[inline]
    pub const fn size(&self) -> UVec2 {
        self.size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
/// Used to calculate the size of UI image nodes
pub struct ImageMeasure {
    /// The size of the image's texture
    pub size: Vec2,
    /// The region of the UI node containing the image
    pub visual_box: VisualBox,
}

impl Measure for ImageMeasure {
    fn measure(&mut self, measure_args: MeasureArgs) -> Vec2 {
        let mut width = measure_args.resolve_width();
        let mut height = measure_args.resolve_height();

        let calc = |_, _| 0.;
        let padding = measure_args.style.padding.resolve_or_zero(
            taffy::Size {
                width: width.effective,
                height: height.effective,
            },
            calc,
        );
        let border = measure_args.style.border.resolve_or_zero(
            taffy::Size {
                width: width.effective,
                height: height.effective,
            },
            calc,
        );
        let content_inset = Vec2::new(
            padding.left + padding.right + border.left + border.right,
            padding.top + padding.bottom + border.top + border.bottom,
        );

        if measure_args.style.box_sizing == taffy::style::BoxSizing::BorderBox {
            width.min = width.min.map(|min| (min - content_inset.x).max(0.));
            width.preferred = width
                .preferred
                .map(|preferred| (preferred - content_inset.x).max(0.));
            width.max = width.max.map(|max| (max - content_inset.x).max(0.));
            width.effective = width
                .effective
                .map(|effective| (effective - content_inset.x).max(0.));

            height.min = height.min.map(|min| (min - content_inset.y).max(0.));
            height.preferred = height
                .preferred
                .map(|preferred| (preferred - content_inset.y).max(0.));
            height.max = height.max.map(|max| (max - content_inset.y).max(0.));
            height.effective = height
                .effective
                .map(|effective| (effective - content_inset.y).max(0.));
        }

        let inset = match self.visual_box {
            VisualBox::ContentBox => Vec2::ZERO,
            VisualBox::PaddingBox => {
                Vec2::new(padding.left + padding.right, padding.top + padding.bottom)
            }
            VisualBox::BorderBox => Vec2::new(content_inset.x, content_inset.y),
        };

        let width = ResolvedAxis {
            min: width.min.map(|min| min + inset.x),
            preferred: width.preferred.map(|preferred| preferred + inset.x),
            max: width.max.map(|max| max + inset.x),
            effective: width.effective.map(|effective| effective + inset.x),
        };
        let height = ResolvedAxis {
            min: height.min.map(|min| min + inset.y),
            preferred: height.preferred.map(|preferred| preferred + inset.y),
            max: height.max.map(|max| max + inset.y),
            effective: height.effective.map(|effective| effective + inset.y),
        };

        // Use aspect_ratio from style, fall back to inherent aspect ratio
        let aspect_ratio = measure_args
            .style
            .aspect_ratio
            .unwrap_or_else(|| self.size.x / self.size.y);

        // Apply aspect ratio
        // If only one of width or height was determined at this point, then the other is set beyond this point using the aspect ratio.
        let taffy_size = taffy::Size {
            width: width.effective,
            height: height.effective,
        }
        .maybe_apply_aspect_ratio(Some(aspect_ratio));

        (Vec2::new(
            taffy_size
                .width
                .unwrap_or(self.size.x)
                .maybe_clamp(width.min, width.max),
            taffy_size
                .height
                .unwrap_or(self.size.y)
                .maybe_clamp(height.min, height.max),
        ) - inset)
            .max(Vec2::ZERO)
    }
}

type UpdateImageFilter = (With<Node>, Without<crate::prelude::Text>);

/// Updates content size of the node based on the image provided
pub fn update_image_content_size_system(
    textures: Res<Assets<Image>>,
    atlases: Res<Assets<TextureAtlasLayout>>,
    mut query: Query<
        (
            &mut ContentSize,
            &ImageNode,
            &mut ImageNodeSize,
            &ComputedUiRenderTargetInfo,
        ),
        UpdateImageFilter,
    >,
) {
    for (mut content_size, image, mut image_size, computed_target) in &mut query {
        let measure =
            if matches!(image.image_mode, NodeImageMode::Auto)
                && image.image.id() != TRANSPARENT_IMAGE_HANDLE.id()
            {
                let Some(size) = image.rect.map(|rect| rect.size().as_uvec2()).or_else(|| {
                    match &image.texture_atlas {
                        Some(atlas) => atlas.texture_rect(&atlases).map(|t| t.size()),
                        None => textures.get(&image.image).map(Image::size),
                    }
                }) else {
                    // Keep the previous measurement until the new asset is available.
                    continue;
                };
                if size != image_size.size {
                    image_size.size = size;
                }
                Some(ImageMeasure {
                    size: size.as_vec2() * computed_target.scale_factor(),
                    visual_box: image.visual_box,
                })
            } else {
                None
            };
        if image_size.measure != measure || (measure.is_some() && content_size.is_added()) {
            image_size.measure = measure;
            match measure {
                Some(measure) => content_size.set(NodeMeasure::Image(measure)),
                None => content_size.clear(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::{App, Update};

    #[test]
    fn stretch_image_preserves_custom_content_measure() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<TextureAtlasLayout>>()
            .add_systems(Update, update_image_content_size_system);
        let entity = app
            .world_mut()
            .spawn((
                ImageNode {
                    image_mode: NodeImageMode::Stretch,
                    ..ImageNode::solid_color(Color::WHITE)
                },
                ContentSize::fixed_size(Vec2::splat(42.)),
            ))
            .id();
        app.update();
        let content = app.world().get::<ContentSize>(entity).unwrap();
        assert!(
            matches!(&content.measure, Some(NodeMeasure::Fixed(measure)) if measure.size == Vec2::splat(42.))
        );
    }

    #[test]
    fn image_measure_observes_late_assets_resizes_and_atlas_edits() {
        fn image(width: u32, height: u32) -> Image {
            let mut image = Image::default();
            let mut size = image.texture_descriptor.size;
            size.width = width;
            size.height = height;
            image.resize(size);
            image
        }
        fn measure(app: &App, entity: Entity) -> Option<Vec2> {
            match &app.world().get::<ContentSize>(entity).unwrap().measure {
                Some(NodeMeasure::Image(measure)) => Some(measure.size),
                None => None,
                _ => panic!("expected an image measure"),
            }
        }
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<TextureAtlasLayout>>()
            .add_systems(Update, update_image_content_size_system);
        let handle = app.world().resource::<Assets<Image>>().reserve_handle();
        let entity = app.world_mut().spawn(ImageNode::new(handle.clone())).id();
        app.update();
        assert_eq!(measure(&app, entity), None);
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .insert(handle.id(), image(20, 10))
            .unwrap();
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(20., 10.)));
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut(&handle)
            .unwrap()
            .resize(image(40, 30).texture_descriptor.size);
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(40., 30.)));
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .remove(handle.id());
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(40., 30.)));
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .insert(handle.id(), image(60, 50))
            .unwrap();
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(60., 50.)));

        let atlas = app
            .world()
            .resource::<Assets<TextureAtlasLayout>>()
            .reserve_handle();
        app.world_mut()
            .get_mut::<ImageNode>(entity)
            .unwrap()
            .texture_atlas = Some(TextureAtlas {
            layout: atlas.clone(),
            index: 0,
        });
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(60., 50.)));
        let mut layout = TextureAtlasLayout::new_empty(UVec2::splat(128));
        layout.add_texture(bevy_math::URect::new(0, 0, 12, 8));
        app.world_mut()
            .resource_mut::<Assets<TextureAtlasLayout>>()
            .insert(atlas.id(), layout)
            .unwrap();
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(12., 8.)));
        app.world_mut()
            .resource_mut::<Assets<TextureAtlasLayout>>()
            .get_mut(&atlas)
            .unwrap()
            .textures[0] = bevy_math::URect::new(2, 3, 8, 7);
        app.update();
        assert_eq!(measure(&app, entity), Some(Vec2::new(6., 4.)));
    }

    #[test]
    fn image_measure_changes_only_with_layout_inputs() {
        #[derive(Resource, Default)]
        struct Changes(Vec<bool>);
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<TextureAtlasLayout>>()
            .init_resource::<Changes>()
            .add_systems(
                Update,
                (
                    update_image_content_size_system,
                    |mut query: Query<&mut ContentSize>, mut changes: ResMut<Changes>| {
                        let mut content = query.single_mut().unwrap();
                        changes.0.push(content.is_changed());
                        // Layout consumes the pending measure without changing its component tick.
                        content.bypass_change_detection().measure.take();
                    },
                )
                    .chain(),
            );
        let entity = app
            .world_mut()
            .spawn(ImageNode {
                rect: Some(Rect::new(0., 0., 20., 10.)),
                ..ImageNode::solid_color(Color::WHITE)
            })
            .id();
        app.update();
        app.world_mut().get_mut::<ImageNode>(entity).unwrap().color = Color::BLACK;
        app.update();
        app.world_mut()
            .get_mut::<ImageNode>(entity)
            .unwrap()
            .visual_box = VisualBox::BorderBox;
        app.update();
        app.world_mut()
            .get_mut::<ComputedUiRenderTargetInfo>(entity)
            .unwrap()
            .scale_factor = 2.;
        app.update();
        app.world_mut()
            .get_mut::<ImageNode>(entity)
            .unwrap()
            .image_mode = NodeImageMode::Stretch;
        app.update();
        app.world_mut().get_mut::<ImageNode>(entity).unwrap().color = Color::WHITE;
        app.update();
        app.world_mut()
            .get_mut::<ImageNode>(entity)
            .unwrap()
            .image_mode = NodeImageMode::Auto;
        app.update();
        app.world_mut()
            .entity_mut(entity)
            .remove::<ContentSize>()
            .insert(ContentSize::default());
        app.update();
        assert_eq!(
            app.world().resource::<Changes>().0,
            [true, false, true, true, true, false, true, true]
        );
    }
}
