use crate::{
    ExtractedSlice, ExtractedSlices, ExtractedSprite, ExtractedSpriteKind, ExtractedSprites,
    ExtractedTextEffect,
};
use bevy_asset::{AssetId, Assets};
use bevy_camera::visibility::ViewVisibility;
use bevy_color::LinearRgba;
use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::prelude::*;
use bevy_math::{Rect, Vec2};
use bevy_render::sync_world::TemporaryRenderEntity;
use bevy_render::Extract;
use bevy_sprite::{Anchor, Text2dOutline, Text2dShadow};
use bevy_text::{
    ComputedTextBlock, PositionedGlyph, TextBackgroundColor, TextBounds, TextColor, TextLayoutInfo,
    TEXT_EFFECT_PADDING,
};
use bevy_transform::prelude::GlobalTransform;

const TEXT2D_BACKGROUND_Z_OFFSET: f32 = -0.0003;

/// This system extracts the sprites from the 2D text components and adds them to the
/// "render world".
pub fn extract_text2d_sprite(
    mut commands: Commands,
    mut extracted_sprites: ResMut<ExtractedSprites>,
    mut extracted_slices: ResMut<ExtractedSlices>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    text2d_query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &ComputedTextBlock,
            &TextLayoutInfo,
            &TextBounds,
            &Anchor,
            Option<&Text2dShadow>,
            Option<&Text2dOutline>,
            &GlobalTransform,
        )>,
    >,
    text_colors: Extract<Query<&TextColor>>,
    text_background_colors_query: Extract<Query<&TextBackgroundColor>>,
) {
    for (
        main_entity,
        view_visibility,
        computed_block,
        text_layout_info,
        text_bounds,
        anchor,
        maybe_shadow,
        maybe_outline,
        global_transform,
    ) in text2d_query.iter()
    {
        let inverse_scale_factor = text_layout_info.scale_factor.recip();
        let scaling = GlobalTransform::from_scale(Vec2::splat(inverse_scale_factor).extend(1.));
        if !view_visibility.get() {
            continue;
        }

        let size = Vec2::new(
            text_bounds.width.unwrap_or(text_layout_info.size.x),
            text_bounds.height.unwrap_or(text_layout_info.size.y),
        );

        let top_left = (Anchor::TOP_LEFT.0 - anchor.as_vec()) * size;
        let base_transform =
            *global_transform * GlobalTransform::from_translation(top_left.extend(0.));
        let text_transform = base_transform * scaling;

        for &(section_entity, rect) in text_layout_info.section_rects.iter() {
            let Ok(text_background_color) = text_background_colors_query.get(section_entity) else {
                continue;
            };
            let render_entity = commands.spawn(TemporaryRenderEntity).id();
            let offset = Vec2::new(rect.center().x, -rect.center().y);
            let transform = text_transform
                * GlobalTransform::from_translation(offset.extend(TEXT2D_BACKGROUND_Z_OFFSET));
            extracted_sprites.sprites.push(ExtractedSprite {
                main_entity,
                render_entity,
                transform,
                color: text_background_color.0.into(),
                image_handle_id: AssetId::default(),
                flip_x: false,
                flip_y: false,
                text_effect: ExtractedTextEffect::default(),
                kind: ExtractedSpriteKind::Single {
                    anchor: Vec2::ZERO,
                    rect: None,
                    scaling_mode: None,
                    custom_size: Some(rect.size()),
                },
            });
        }

        let shadow_effect = maybe_shadow.map(|shadow| {
            (
                LinearRgba::from(shadow.color),
                clamp_text2d_shadow_offset(shadow.offset, text_layout_info.scale_factor),
            )
        });
        let outline_effect = maybe_outline.and_then(|outline| {
            let width = clamp_outline_width(outline.width * text_layout_info.scale_factor)?;
            Some((LinearRgba::from(outline.color), width))
        });
        let glyph_text_effect = ExtractedTextEffect::text(
            shadow_effect.map(|(color, offset)| (color, Vec2::new(offset.x, -offset.y))),
            outline_effect.map(|(color, _)| color),
        );
        let glyph_padding = combined_text_effect_padding(
            shadow_effect.map(|(_, offset)| offset),
            outline_effect.map(|(_, width)| width),
        );

        let transform = text_transform;
        let mut color = LinearRgba::WHITE;
        let mut current_span = usize::MAX;
        let mut start = extracted_slices.slices.len();

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                span_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            if *span_index != current_span {
                color = text_colors
                    .get(
                        computed_block
                            .entities()
                            .get(*span_index)
                            .map(|t| t.entity)
                            .unwrap_or(Entity::PLACEHOLDER),
                    )
                    .map(|text_color| LinearRgba::from(text_color.0))
                    .unwrap_or_default();
                current_span = *span_index;
            }
            let rect = texture_atlases
                .get(atlas_info.texture_atlas)
                .unwrap()
                .textures[atlas_info.location.glyph_index]
                .as_rect();
            extracted_slices.slices.push(ExtractedSlice {
                offset: Vec2::new(position.x, -position.y),
                rect: glyph_padding
                    .map(|padding| expanded_effect_rect(rect, padding))
                    .unwrap_or(rect),
                size: rect.size() + glyph_padding.unwrap_or(Vec2::ZERO) * 2.0,
            });

            if text_layout_info.glyphs.get(i + 1).is_none_or(|info| {
                info.span_index != current_span || info.atlas_info.texture != atlas_info.texture
            }) {
                let render_entity = commands.spawn(TemporaryRenderEntity).id();
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity,
                    transform,
                    color,
                    image_handle_id: atlas_info.texture,
                    flip_x: false,
                    flip_y: false,
                    text_effect: glyph_text_effect,
                    kind: ExtractedSpriteKind::Slices {
                        indices: start..extracted_slices.slices.len(),
                    },
                });
                start = extracted_slices.slices.len();
            }
        }
    }
}

fn expanded_effect_rect(fill_rect: Rect, padding: Vec2) -> Rect {
    Rect::from_corners(fill_rect.min - padding, fill_rect.max + padding)
}

fn clamp_text2d_shadow_offset(offset: Vec2, scale_factor: f32) -> Vec2 {
    let sampled_offset = offset * scale_factor;
    let limit = TEXT_EFFECT_PADDING as f32;
    if sampled_offset.x.abs() <= limit && sampled_offset.y.abs() <= limit {
        return sampled_offset;
    }

    sampled_offset.clamp(Vec2::splat(-limit), Vec2::splat(limit))
}

fn clamp_outline_width(width: f32) -> Option<f32> {
    if width <= 0.0 {
        return None;
    }

    let limit = TEXT_EFFECT_PADDING as f32;
    Some(width.min(limit))
}

fn combined_text_effect_padding(
    shadow_offset: Option<Vec2>,
    outline_width: Option<f32>,
) -> Option<Vec2> {
    let shadow_padding =
        shadow_offset.map_or(Vec2::ZERO, |shadow_offset| shadow_offset.abs().ceil());
    let outline_padding = outline_width.map_or(Vec2::ZERO, |outline_width| {
        Vec2::splat(outline_width.ceil().max(1.0))
    });
    let padding = shadow_padding.max(outline_padding);

    if padding == Vec2::ZERO {
        None
    } else {
        Some(padding)
    }
}
