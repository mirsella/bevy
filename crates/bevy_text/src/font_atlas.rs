use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_color::Srgba;
use bevy_ecs::entity::Entity;
use bevy_image::{prelude::*, ImageSampler, ToExtents};
use bevy_math::{UVec2, Vec2};
use bevy_platform::collections::HashMap;
use swash::scale::Scaler;
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

use crate::{FontSmoothing, GlyphAtlasInfo, GlyphAtlasLocation, TextError};

/// A paint layer within a text block.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TextEffectLayer {
    Shadow,
    Outline,
    Fill,
}

impl TextEffectLayer {
    /// Keep one transparent texel for filtering the face; only effect layers
    /// need the full stroke margin. This avoids shading empty face fragments.
    pub fn glyph_padding(self, padding: u32) -> f32 {
        (match self {
            Self::Fill => padding.min(1),
            _ => padding,
        }) as f32
    }
}

/// Returns a validated, physical-pixel offset for a translated text shadow.
#[doc(hidden)]
pub fn text_effect_shadow_offset(
    offset: Vec2,
    scale_factor: f32,
    entity: Entity,
    component: &str,
) -> Option<Vec2> {
    let offset = offset * scale_factor;

    if !offset.is_finite() {
        bevy_log::warn_once!(
            "Skipping {component} on {entity:?}: shadow offset must be finite, got {offset}"
        );
        return None;
    }

    // Shadows are translated geometry, not samples displaced inside the atlas.
    Some(offset)
}

/// Converts a full centered stroke width in logical pixels to its validated
/// outward extent in physical pixels, used by atlas padding and rasterization.
#[doc(hidden)]
pub fn text_effect_outline_extent(
    stroke_width: f32,
    scale_factor: f32,
    entity: Entity,
    component: &str,
) -> Option<f32> {
    let physical_stroke_width = stroke_width * scale_factor;

    if !physical_stroke_width.is_finite() {
        bevy_log::warn_once!(
            "Skipping {component} on {entity:?}: physical stroke width must be finite, got {physical_stroke_width}"
        );
        return None;
    }

    if physical_stroke_width <= 0.0 {
        return None;
    }

    Some(physical_stroke_width * 0.5)
}

/// Key identifying a glyph
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlyphCacheKey {
    /// Id used to look up the glyph
    pub glyph_id: u16,
}

/// Rasterized glyphs are cached, stored in, and retrieved from, a `FontAtlas`.
///
/// A `FontAtlas` contains one or more textures, each of which contains one or more glyphs packed into them.
///
/// A [`FontAtlasSet`](crate::FontAtlasSet) contains a `FontAtlas` for each font size in the same font face.
///
/// For the same font face and font size, a glyph will be rasterized differently for different subpixel offsets.
/// In practice, ranges of subpixel offsets are grouped into subpixel bins to limit the number of rasterized glyphs,
/// providing a trade-off between visual quality and performance.
///
/// A [`GlyphCacheKey`] encodes all of the information of a subpixel-offset glyph and is used to
/// find that glyphs raster in a [`TextureAtlas`] through its corresponding [`GlyphAtlasLocation`].
pub struct FontAtlas {
    /// Used to update the [`TextureAtlasLayout`].
    pub dynamic_texture_atlas_builder: DynamicTextureAtlasBuilder,
    /// A mapping between subpixel-offset glyphs and their [`GlyphAtlasLocation`].
    pub glyph_to_atlas_index: HashMap<GlyphCacheKey, GlyphAtlasLocation>,
    /// The layout for the font atlas.
    pub texture_atlas: TextureAtlasLayout,
    /// The texture where this font atlas is located
    pub texture: Handle<Image>,
}

impl FontAtlas {
    /// Create a new [`FontAtlas`] with the given size, adding it to the appropriate asset collections.
    pub fn new(
        textures: &mut Assets<Image>,
        size: UVec2,
        font_smoothing: FontSmoothing,
    ) -> FontAtlas {
        let mut image = Image::new_fill(
            size.to_extents(),
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            // Need to keep this image CPU persistent in order to add additional glyphs later on
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        if font_smoothing == FontSmoothing::None {
            image.sampler = ImageSampler::nearest();
        }
        let texture = textures.add(image);
        Self {
            texture_atlas: TextureAtlasLayout::new_empty(size),
            glyph_to_atlas_index: HashMap::default(),
            dynamic_texture_atlas_builder: DynamicTextureAtlasBuilder::new(size, 2),
            texture,
        }
    }

    /// Get the [`GlyphAtlasLocation`] for a subpixel-offset glyph.
    pub fn get_glyph_index(&self, cache_key: GlyphCacheKey) -> Option<GlyphAtlasLocation> {
        self.glyph_to_atlas_index.get(&cache_key).copied()
    }

    /// Checks if the given subpixel-offset glyph is contained in this [`FontAtlas`].
    pub fn has_glyph(&self, cache_key: GlyphCacheKey) -> bool {
        self.glyph_to_atlas_index.contains_key(&cache_key)
    }

    /// Add a glyph to the atlas, updating both its texture and layout.
    ///
    /// The glyph is represented by `glyph`, and its image content is `glyph_texture`.
    /// This content is copied into the atlas texture, and the atlas layout is updated
    /// to store the location of that glyph into the atlas.
    ///
    /// # Returns
    ///
    /// Returns `()` if the glyph is successfully added, or [`TextError::FailedToAddGlyph`] otherwise.
    /// In that case, neither the atlas texture nor the atlas layout are
    /// modified.
    pub fn add_glyph(
        &mut self,
        textures: &mut Assets<Image>,
        key: GlyphCacheKey,
        texture: &Image,
        offset: Vec2,
        is_alpha_mask: bool,
        padding: u32,
    ) -> Result<(), TextError> {
        let mut atlas_texture = textures
            .get_mut(&self.texture)
            .ok_or(TextError::MissingAtlasTexture)?;

        if let Ok(glyph_index) = self.dynamic_texture_atlas_builder.add_texture(
            &mut self.texture_atlas,
            texture,
            &mut atlas_texture,
        ) {
            if padding != 0 {
                let glyph_rect = self
                    .texture_atlas
                    .textures
                    .get_mut(glyph_index)
                    .expect("newly added glyph must have an atlas rect");
                let padding = UVec2::splat(padding);
                glyph_rect.min += padding;
                glyph_rect.max -= padding;
            }

            self.glyph_to_atlas_index.insert(
                key,
                GlyphAtlasLocation {
                    glyph_index,
                    offset,
                    is_alpha_mask,
                    padding,
                },
            );
            Ok(())
        } else {
            Err(TextError::FailedToAddGlyph(key.glyph_id))
        }
    }
}

impl core::fmt::Debug for FontAtlas {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FontAtlas")
            .field("glyph_to_atlas_index", &self.glyph_to_atlas_index)
            .field("texture_atlas", &self.texture_atlas)
            .field("texture", &self.texture)
            .field("dynamic_texture_atlas_builder", &"[...]")
            .finish()
    }
}

/// Adds the given subpixel-offset glyph to the given font atlases
pub fn add_glyph_to_atlas(
    font_atlases: &mut Vec<FontAtlas>,
    textures: &mut Assets<Image>,
    scaler: &mut Scaler,
    font_smoothing: FontSmoothing,
    glyph_id: u16,
    text_effect_padding: bool,
    outline_width: Option<f32>,
) -> Result<GlyphAtlasInfo, TextError> {
    let (glyph_texture, offset, is_alpha_mask, padding) = get_outlined_glyph_texture(
        scaler,
        glyph_id,
        font_smoothing,
        text_effect_padding,
        outline_width,
    )?;
    let mut add_char_to_font_atlas = |atlas: &mut FontAtlas| -> Result<(), TextError> {
        atlas.add_glyph(
            textures,
            GlyphCacheKey { glyph_id },
            &glyph_texture,
            offset,
            is_alpha_mask,
            padding,
        )
    };
    if !font_atlases
        .iter_mut()
        .any(|atlas| add_char_to_font_atlas(atlas).is_ok())
    {
        // Find the largest dimension of the glyph, either its width or its height
        let glyph_max_size: u32 = glyph_texture
            .texture_descriptor
            .size
            .height
            .max(glyph_texture.width());
        // Pick the higher of 512 or the smallest power of 2 greater than glyph_max_size
        let containing = (1u32 << (32 - glyph_max_size.leading_zeros())).max(512);

        let mut new_atlas = FontAtlas::new(textures, UVec2::splat(containing), font_smoothing);

        new_atlas.add_glyph(
            textures,
            GlyphCacheKey { glyph_id },
            &glyph_texture,
            offset,
            is_alpha_mask,
            padding,
        )?;

        font_atlases.push(new_atlas);
    }

    get_glyph_atlas_info(font_atlases, GlyphCacheKey { glyph_id })
        .ok_or(TextError::InconsistentAtlasState)
}

/// Get the glyph texture, its offset, whether it is an alpha mask, and its actual padding.
pub fn get_outlined_glyph_texture(
    scaler: &mut Scaler,
    glyph_id: u16,
    font_smoothing: FontSmoothing,
    text_effect_padding: bool,
    outline_width: Option<f32>,
) -> Result<(Image, Vec2, bool, u32), TextError> {
    let mut image = swash::scale::Render::new(&[
        swash::scale::Source::ColorOutline(0),
        swash::scale::Source::ColorBitmap(swash::scale::StrikeWith::BestFit),
        swash::scale::Source::Outline,
    ])
    .format(swash::zeno::Format::Alpha)
    .render(scaler, glyph_id)
    .ok_or(TextError::FailedToGetGlyphImage(glyph_id))?;

    let mut padding = u32::from(text_effect_padding);
    let left = image.placement.left;
    let top = image.placement.top;
    let mut width = image.placement.width;
    let mut height = image.placement.height;

    let (rgba, is_alpha_mask) = match image.content {
        swash::scale::image::Content::Mask => {
            let stroke = outline_width
                .filter(|outline_width| 0.0 < *outline_width)
                .map(|outline_width| {
                    swash::scale::Render::new(&[swash::scale::Source::Outline])
                        .format(swash::zeno::Format::Alpha)
                        .style(
                            swash::zeno::Stroke::new(outline_width * 2.0)
                                .join(swash::zeno::Join::Round),
                        )
                        .render(scaler, glyph_id)
                        .ok_or(TextError::FailedToGetGlyphImage(glyph_id))
                })
                .transpose()?;
            if let Some(stroke) = &stroke {
                // Curved strokes can extend beyond the nominal radius. Measure the
                // rasterized bounds before allocating, retaining a transparent texel
                // on every side and symmetric padding to preserve fill positioning.
                let left_padding = left - stroke.placement.left;
                let top_padding = stroke.placement.top - top;
                let right_padding = stroke.placement.width as i32 - width as i32 - left_padding;
                let bottom_padding = stroke.placement.height as i32 - height as i32 - top_padding;
                let stroke_padding = left_padding
                    .max(top_padding)
                    .max(right_padding)
                    .max(bottom_padding)
                    .max(0) as u32;
                padding = stroke_padding + 1;
            }
            apply_font_smoothing(&mut image.data, font_smoothing);
            let fill_alpha = if padding != 0 {
                width += padding * 2;
                height += padding * 2;
                pad_glyph::<1>(
                    &image.data,
                    image.placement.width,
                    image.placement.height,
                    padding,
                )
            } else {
                image.data
            };

            let outline_alpha = stroke.map(|stroke| {
                let padding = padding as i32;
                let mut alpha = vec![0; (width * height) as usize];
                let offset_x = stroke.placement.left - left + padding;
                let offset_y = top - stroke.placement.top + padding;
                assert!(
                    offset_x >= 0
                        && offset_y >= 0
                        && offset_x as u32 + stroke.placement.width <= width
                        && offset_y as u32 + stroke.placement.height <= height,
                    "text stroke exceeds glyph atlas padding"
                );
                let stroke_width = stroke.placement.width as usize;
                for y in 0..stroke.placement.height as usize {
                    let target = (y + offset_y as usize) * width as usize + offset_x as usize;
                    let source = y * stroke_width;
                    alpha[target..target + stroke_width]
                        .copy_from_slice(&stroke.data[source..source + stroke_width]);
                }
                apply_font_smoothing(&mut alpha, font_smoothing);
                alpha
            });

            (
                pack_text_glyph_texture(&fill_alpha, outline_alpha.as_deref()),
                true,
            )
        }
        swash::scale::image::Content::Color | swash::scale::image::Content::SubpixelMask => {
            let data = if padding != 0 {
                width += padding * 2;
                height += padding * 2;
                pad_glyph::<4>(
                    &image.data,
                    image.placement.width,
                    image.placement.height,
                    padding,
                )
            } else {
                image.data
            };

            (data, false)
        }
    };

    Ok((
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD,
        ),
        Vec2::new(left as f32, -top as f32),
        is_alpha_mask,
        padding,
    ))
}

fn apply_font_smoothing(alpha: &mut [u8], font_smoothing: FontSmoothing) {
    if font_smoothing == FontSmoothing::None {
        for alpha in alpha {
            *alpha = if *alpha > 127 { 255 } else { 0 };
        }
    }
}

fn pad_glyph<const CHANNELS: usize>(data: &[u8], width: u32, height: u32, padding: u32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let padding = padding as usize;
    let stride = (width + padding * 2) * CHANNELS;
    let row_bytes = width * CHANNELS;
    let mut padded = vec![0; stride * (height + padding * 2)];

    for y in 0..height {
        let source = y * row_bytes;
        let target = (y + padding) * stride + padding * CHANNELS;
        padded[target..target + row_bytes].copy_from_slice(&data[source..source + row_bytes]);
    }
    padded
}

fn pack_text_glyph_texture(fill_alpha: &[u8], outline_alpha: Option<&[u8]>) -> Vec<u8> {
    let mut rgba = vec![0; fill_alpha.len() * 4];

    for (i, fill_alpha) in fill_alpha.iter().enumerate() {
        rgba[i * 4] =
            outline_alpha.map_or(0, |outline_alpha| linear_u8_to_srgb_u8(outline_alpha[i]));
        rgba[i * 4 + 1] = 255;
        rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = *fill_alpha;
    }

    rgba
}

fn linear_u8_to_srgb_u8(value: u8) -> u8 {
    if value == 0 || value == 255 {
        return value;
    }

    let value = value as f32 / 255.0;
    let srgb = Srgba::gamma_function_inverse(value);

    (srgb * 255.0).round() as u8
}

/// Generates the [`GlyphAtlasInfo`] for the given subpixel-offset glyph.
pub fn get_glyph_atlas_info(
    font_atlases: &mut [FontAtlas],
    cache_key: GlyphCacheKey,
) -> Option<GlyphAtlasInfo> {
    font_atlases.iter().find_map(|atlas| {
        atlas
            .get_glyph_index(cache_key)
            .map(|location| GlyphAtlasInfo {
                offset: location.offset,
                rect: atlas.texture_atlas.textures[location.glyph_index].as_rect(),
                texture: atlas.texture.id(),
                is_alpha_mask: location.is_alpha_mask,
                padding: location.padding,
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_stroke_width_converts_to_physical_outward_extent() {
        for (stroke_width, scale, extent) in [(0.5, 1.0, 0.25), (2.0, 1.0, 1.0), (2.0, 2.0, 2.0)] {
            assert_eq!(
                text_effect_outline_extent(stroke_width, scale, Entity::PLACEHOLDER, "test"),
                Some(extent)
            );
        }
        for stroke_width in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                text_effect_outline_extent(stroke_width, 1.0, Entity::PLACEHOLDER, "test"),
                None
            );
        }
    }

    fn glyph(width: f32) -> (Image, u32) {
        let font = swash::FontRef::from_index(include_bytes!("FiraMono-subset.ttf"), 0).unwrap();
        let mut context = swash::scale::ScaleContext::new();
        let mut scaler = context.builder(font).size(32.0).hint(false).build();
        let (image, _, _, padding) = get_outlined_glyph_texture(
            &mut scaler,
            font.charmap().map('A'),
            FontSmoothing::AntiAliased,
            true,
            Some(width),
        )
        .unwrap();
        (image, padding)
    }

    #[test]
    fn fractional_strokes_grow_smoothly_without_changing_the_fill() {
        let mut previous_coverage = 0;
        let face = |image: &Image, padding: u32| {
            let padding = padding as usize;
            let stride = image.width() as usize;
            let data = image.data.as_ref().unwrap();
            (padding..image.height() as usize - padding)
                .flat_map(|y| {
                    (padding..stride - padding).map(move |x| data[(y * stride + x) * 4 + 3])
                })
                .collect::<Vec<_>>()
        };
        let (image, padding) = glyph(0.1);
        let fill = face(&image, padding);
        for width in [0.1, 0.2, 0.3, 0.49, 0.51, 0.7, 1.0, 2.0] {
            let (image, padding) = glyph(width);
            let data = image.data.as_ref().unwrap();
            let coverage: u64 = data.chunks_exact(4).map(|pixel| u64::from(pixel[0])).sum();
            assert!(
                coverage > previous_coverage,
                "stroke did not grow at {width}"
            );
            assert!(
                face(&image, padding) == fill,
                "stroke changed the glyph face"
            );
            previous_coverage = coverage;
        }
    }

    #[test]
    fn stroke_covers_antialiased_fill_edges_without_subtracting_them() {
        let data = glyph(2.0).0.data.unwrap();
        let mut edges = 0;
        for pixel in data.chunks_exact(4) {
            if (1..255).contains(&pixel[3]) {
                edges += 1;
                assert_eq!(
                    pixel[0], 255,
                    "full stroke must back the antialiased fill edge"
                );
            }
        }
        assert!(edges > 0);
    }

    #[test]
    fn large_strokes_have_transparent_atlas_borders() {
        assert_transparent_border(&glyph(24.5).0);
    }

    fn assert_transparent_border(image: &Image) {
        let stride = image.width() as usize;
        let height = image.height() as usize;
        let data = image.data.as_ref().unwrap();
        for (i, pixel) in data.chunks_exact(4).enumerate() {
            if i < stride
                || i >= stride * (height - 1)
                || i % stride == 0
                || i % stride == stride - 1
            {
                assert_eq!(pixel[0], 0, "stroke touches the atlas border");
                assert_eq!(pixel[3], 0, "fill touches the atlas border");
            }
        }
    }

    #[test]
    fn curved_stroke_bounds_determine_atlas_padding() {
        let font = swash::FontRef::from_index(include_bytes!("FiraMono-subset.ttf"), 0).unwrap();
        let mut context = swash::scale::ScaleContext::new();
        let mut scaler = context.builder(font).size(12.0).hint(false).build();
        let glyph_id = font.charmap().map('$');
        let extent = 24.5;
        let (image, offset, is_alpha_mask, padding) = get_outlined_glyph_texture(
            &mut scaler,
            glyph_id,
            FontSmoothing::AntiAliased,
            true,
            Some(extent),
        )
        .unwrap();
        // This glyph exceeds the old nominal stroke-radius-plus-one margin.
        assert!(padding > extent.ceil() as u32 + 1);
        assert_transparent_border(&image);

        let mut textures = Assets::default();
        let mut atlases = Vec::new();
        let atlas = add_glyph_to_atlas(
            &mut atlases,
            &mut textures,
            &mut scaler,
            FontSmoothing::AntiAliased,
            glyph_id,
            true,
            Some(extent),
        )
        .unwrap();
        assert!(is_alpha_mask);
        assert_eq!(atlas.padding, padding);
        assert_eq!(atlas.offset, offset);
        assert_eq!(
            atlas.rect.size(),
            Vec2::new(image.width() as f32, image.height() as f32)
                - Vec2::splat(2.0 * padding as f32)
        );
    }

    #[test]
    fn effects_without_strokes_only_need_a_filtering_texel() {
        let font = swash::FontRef::from_index(include_bytes!("FiraMono-subset.ttf"), 0).unwrap();
        let mut context = swash::scale::ScaleContext::new();
        let mut scaler = context.builder(font).size(32.0).hint(false).build();
        for enabled in [false, true] {
            let (image, _, _, padding) = get_outlined_glyph_texture(
                &mut scaler,
                font.charmap().map('A'),
                FontSmoothing::AntiAliased,
                enabled,
                None,
            )
            .unwrap();
            assert_eq!(padding, u32::from(enabled));
            if enabled {
                assert_transparent_border(&image);
            }
        }
    }

    #[test]
    fn shadow_displacement_is_not_limited_by_atlas_padding() {
        let entity = Entity::PLACEHOLDER;
        let offset = Vec2::new(80.0, -48.0);
        assert_eq!(
            text_effect_shadow_offset(offset, 2.0, entity, "test"),
            Some(offset * 2.0)
        );
    }
}
