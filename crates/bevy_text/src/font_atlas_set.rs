use bevy_asset::{AssetEvent, AssetId, Assets, RenderAssetUsages};
use bevy_ecs::{message::MessageReader, resource::Resource, system::ResMut};
use bevy_image::prelude::*;
use bevy_math::{IVec2, UVec2};
use bevy_platform::collections::HashMap;
use bevy_reflect::TypePath;
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

use crate::{
    error::TextError, Font, FontAtlas, FontSmoothing, GlyphAtlasInfo, TEXT_EFFECT_PADDING,
};

/// A map of font faces to their corresponding [`FontAtlasSet`]s.
#[derive(Debug, Default, Resource)]
pub struct FontAtlasSets {
    // PERF: in theory this could be optimized with Assets storage ... consider making some fast "simple" AssetMap
    pub(crate) sets: HashMap<AssetId<Font>, FontAtlasSet>,
}

impl FontAtlasSets {
    /// Get a reference to the [`FontAtlasSet`] with the given font asset id.
    pub fn get(&self, id: impl Into<AssetId<Font>>) -> Option<&FontAtlasSet> {
        let id: AssetId<Font> = id.into();
        self.sets.get(&id)
    }
    /// Get a mutable reference to the [`FontAtlasSet`] with the given font asset id.
    pub fn get_mut(&mut self, id: impl Into<AssetId<Font>>) -> Option<&mut FontAtlasSet> {
        let id: AssetId<Font> = id.into();
        self.sets.get_mut(&id)
    }
}

/// A system that cleans up [`FontAtlasSet`]s for removed [`Font`]s
pub fn remove_dropped_font_atlas_sets(
    mut font_atlas_sets: ResMut<FontAtlasSets>,
    mut font_events: MessageReader<AssetEvent<Font>>,
) {
    for event in font_events.read() {
        if let AssetEvent::Removed { id } = event {
            font_atlas_sets.sets.remove(id);
        }
    }
}

/// Identifies a font size and effect configuration in a [`FontAtlasSet`].
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct FontAtlasKey {
    font_size_bits: u32,
    font_smoothing: FontSmoothing,
    text_effect_padding: bool,
    outline_width_bits: Option<u32>,
}

impl FontAtlasKey {
    pub(crate) fn new(
        font_size_bits: u32,
        font_smoothing: FontSmoothing,
        text_effect_padding: bool,
        outline_width: Option<f32>,
    ) -> Self {
        Self {
            font_size_bits,
            font_smoothing,
            text_effect_padding,
            outline_width_bits: outline_width.map(f32::to_bits),
        }
    }
}

/// A map of font sizes to their corresponding [`FontAtlas`]es, for a given font face.
///
/// Provides the interface for adding and retrieving rasterized glyphs, and manages the [`FontAtlas`]es.
///
/// There is at most one `FontAtlasSet` for each font, stored in the `FontAtlasSets` resource.
/// `FontAtlasSet`s are added and updated by the [`queue_text`](crate::pipeline::TextPipeline::queue_text) function.
///
/// A `FontAtlasSet` contains one or more [`FontAtlas`]es for each font size.
#[derive(Debug, TypePath)]
pub struct FontAtlasSet {
    font_atlases: HashMap<FontAtlasKey, Vec<FontAtlas>>,
}

impl Default for FontAtlasSet {
    fn default() -> Self {
        FontAtlasSet {
            font_atlases: HashMap::with_capacity_and_hasher(1, Default::default()),
        }
    }
}

impl FontAtlasSet {
    /// Returns an iterator over the [`FontAtlas`]es in this set
    pub fn iter(&self) -> impl Iterator<Item = (&FontAtlasKey, &Vec<FontAtlas>)> {
        self.font_atlases.iter()
    }

    /// Checks if the given subpixel-offset glyph is contained in any of the [`FontAtlas`]es in this set
    pub fn has_glyph(&self, cache_key: cosmic_text::CacheKey, font_size: &FontAtlasKey) -> bool {
        self.font_atlases
            .get(font_size)
            .is_some_and(|font_atlas| font_atlas.iter().any(|atlas| atlas.has_glyph(cache_key)))
    }

    /// Adds the given subpixel-offset glyph to the [`FontAtlas`]es in this set
    pub fn add_glyph_to_atlas(
        &mut self,
        texture_atlases: &mut Assets<TextureAtlasLayout>,
        textures: &mut Assets<Image>,
        font_system: &mut cosmic_text::FontSystem,
        swash_cache: &mut cosmic_text::SwashCache,
        layout_glyph: &cosmic_text::LayoutGlyph,
        font_smoothing: FontSmoothing,
        text_effect_padding: bool,
        outline_width: Option<f32>,
    ) -> Result<GlyphAtlasInfo, TextError> {
        let physical_glyph = layout_glyph.physical((0., 0.), 1.0);
        let font_atlas_key = FontAtlasKey::new(
            physical_glyph.cache_key.font_size_bits,
            font_smoothing,
            text_effect_padding,
            outline_width,
        );

        let font_atlases = self.font_atlases.entry(font_atlas_key).or_insert_with(|| {
            vec![FontAtlas::new(
                textures,
                texture_atlases,
                UVec2::splat(512),
                font_smoothing,
            )]
        });

        let (glyph_texture, offset) = Self::get_glyph_texture(
            font_system,
            swash_cache,
            &physical_glyph,
            font_smoothing,
            text_effect_padding,
            outline_width,
        )?;
        let mut add_char_to_font_atlas = |atlas: &mut FontAtlas| -> Result<(), TextError> {
            atlas.add_glyph(
                textures,
                texture_atlases,
                physical_glyph.cache_key,
                &glyph_texture,
                offset,
                text_effect_padding,
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
            font_atlases.push(FontAtlas::new(
                textures,
                texture_atlases,
                UVec2::splat(containing),
                font_smoothing,
            ));

            font_atlases.last_mut().unwrap().add_glyph(
                textures,
                texture_atlases,
                physical_glyph.cache_key,
                &glyph_texture,
                offset,
                text_effect_padding,
            )?;
        }

        Ok(self
            .get_glyph_atlas_info(physical_glyph.cache_key, &font_atlas_key)
            .unwrap())
    }

    /// Generates the [`GlyphAtlasInfo`] for the given subpixel-offset glyph.
    pub fn get_glyph_atlas_info(
        &mut self,
        cache_key: cosmic_text::CacheKey,
        font_atlas_key: &FontAtlasKey,
    ) -> Option<GlyphAtlasInfo> {
        self.font_atlases
            .get(font_atlas_key)
            .and_then(|font_atlases| {
                font_atlases.iter().find_map(|atlas| {
                    atlas
                        .get_glyph_index(cache_key)
                        .map(|location| GlyphAtlasInfo {
                            location,
                            texture_atlas: atlas.texture_atlas.id(),
                            texture: atlas.texture.id(),
                        })
                })
            })
    }

    /// Returns the number of font atlases in this set.
    pub fn len(&self) -> usize {
        self.font_atlases.len()
    }

    /// Returns `true` if the set has no font atlases.
    pub fn is_empty(&self) -> bool {
        self.font_atlases.len() == 0
    }

    /// Get the texture of the glyph as a rendered image, and its offset.
    pub fn get_glyph_texture(
        font_system: &mut cosmic_text::FontSystem,
        swash_cache: &mut cosmic_text::SwashCache,
        physical_glyph: &cosmic_text::PhysicalGlyph,
        font_smoothing: FontSmoothing,
        text_effect_padding: bool,
        outline_width: Option<f32>,
    ) -> Result<(Image, IVec2), TextError> {
        // NOTE: Ideally, we'd ask COSMIC Text to honor the font smoothing setting directly.
        // However, since it currently doesn't support that, we render the glyph with antialiasing
        // and apply a threshold to the alpha channel to simulate the effect.
        //
        // This has the side effect of making regular vector fonts look quite ugly when font smoothing
        // is turned off, but for fonts that are specifically designed for pixel art, it works well.
        //
        // See: https://github.com/pop-os/cosmic-text/issues/279
        let image = swash_cache
            .get_image_uncached(font_system, physical_glyph.cache_key)
            .ok_or(TextError::FailedToGetGlyphImage(physical_glyph.cache_key))?;

        let cosmic_text::Placement {
            left,
            top,
            width,
            height,
        } = image.placement;

        let fill_alpha = match image.content {
            cosmic_text::SwashContent::Mask => apply_font_smoothing(&image.data, font_smoothing),
            cosmic_text::SwashContent::Color => {
                image.data.chunks_exact(4).map(|pixel| pixel[3]).collect()
            }
            cosmic_text::SwashContent::SubpixelMask => {
                // TODO: implement
                todo!()
            }
        };

        let mut width = width;
        let mut height = height;
        let fill_alpha = if text_effect_padding {
            width += TEXT_EFFECT_PADDING * 2;
            height += TEXT_EFFECT_PADDING * 2;
            pad_mask(
                &fill_alpha,
                image.placement.width,
                image.placement.height,
                TEXT_EFFECT_PADDING,
            )
        } else {
            fill_alpha
        };

        let outline_alpha = outline_width
            .filter(|outline_width| 0.0 < *outline_width)
            .map(|outline_width| build_outline_mask(&fill_alpha, width, height, outline_width));
        let data = pack_text_glyph_texture(&fill_alpha, outline_alpha.as_deref());

        Ok((
            Image::new(
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                data,
                TextureFormat::Rgba8Unorm,
                RenderAssetUsages::MAIN_WORLD,
            ),
            IVec2::new(left, top),
        ))
    }
}

fn apply_font_smoothing(alpha: &[u8], font_smoothing: FontSmoothing) -> Vec<u8> {
    match font_smoothing {
        FontSmoothing::AntiAliased => alpha.to_vec(),
        FontSmoothing::None => alpha
            .iter()
            .map(|alpha| if 127 < *alpha { 255 } else { 0 })
            .collect(),
    }
}

fn pad_mask(mask: &[u8], width: u32, height: u32, padding: u32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let padding = padding as usize;
    let padded_width = width + padding * 2;
    let padded_height = height + padding * 2;
    let mut padded = vec![0; padded_width * padded_height];

    for y in 0..height {
        let src_start = y * width;
        let dst_start = (y + padding) * padded_width + padding;
        padded[dst_start..dst_start + width].copy_from_slice(&mask[src_start..src_start + width]);
    }

    padded
}

fn build_outline_mask(fill_alpha: &[u8], width: u32, height: u32, outline_width: f32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let max_radius = outline_width.ceil().max(1.0) as i32;
    let max_distance_squared = (outline_width + 0.5) * (outline_width + 0.5);
    let mut offsets = Vec::new();

    for y in -max_radius..=max_radius {
        for x in -max_radius..=max_radius {
            let distance_squared = (x * x + y * y) as f32;
            if distance_squared <= max_distance_squared {
                offsets.push((x, y));
            }
        }
    }

    let mut outline_alpha = vec![0; fill_alpha.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let mut dilated = 0;

            for &(offset_x, offset_y) in &offsets {
                let sample_x = x as i32 + offset_x;
                let sample_y = y as i32 + offset_y;
                if !(0..width as i32).contains(&sample_x) || !(0..height as i32).contains(&sample_y)
                {
                    continue;
                }

                let sample_index = sample_y as usize * width + sample_x as usize;
                dilated = dilated.max(fill_alpha[sample_index]);
            }

            outline_alpha[index] = dilated.saturating_sub(fill_alpha[index]);
        }
    }

    outline_alpha
}

fn pack_text_glyph_texture(fill_alpha: &[u8], outline_alpha: Option<&[u8]>) -> Vec<u8> {
    let mut rgba = vec![0; fill_alpha.len() * 4];

    for (i, fill_alpha) in fill_alpha.iter().enumerate() {
        rgba[i * 4] = outline_alpha.map_or(0, |outline_alpha| outline_alpha[i]);
        rgba[i * 4 + 3] = *fill_alpha;
    }

    rgba
}
