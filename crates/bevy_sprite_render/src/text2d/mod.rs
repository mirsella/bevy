use crate::{
    ExtractedSlice, ExtractedSlices, ExtractedSprite, ExtractedSpriteKind, ExtractedSprites,
    ExtractedTextEffect,
};
use bevy_asset::AssetId;
use bevy_camera::visibility::ViewVisibility;
use bevy_color::{Alpha, LinearRgba};
use bevy_ecs::{
    entity::Entity,
    query::Has,
    system::{Commands, Query, ResMut},
};
use bevy_math::{Vec2, Vec3};
use bevy_render::sync_world::TemporaryRenderEntity;
use bevy_render::Extract;
use bevy_sprite::{Anchor, Text2dOutline, Text2dShadow};
use bevy_text::{
    combined_text_effect_padding, expanded_text_effect_rect, text_effect_outline_width,
    text_effect_shadow_offset, ComputedTextBlock, PositionedGlyph, Strikethrough,
    StrikethroughColor, TextBackgroundColor, TextBounds, TextColor, TextLayoutInfo, Underline,
    UnderlineColor,
};
use bevy_transform::prelude::GlobalTransform;

/// This system extracts the sprites from the 2D text components and adds them to the
/// "render world".
pub fn extract_text2d_sprite(
    mut commands: Commands,
    mut extracted_sprites: ResMut<ExtractedSprites>,
    mut extracted_slices: ResMut<ExtractedSlices>,
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
    decoration_query: Extract<
        Query<(
            &TextColor,
            Has<Strikethrough>,
            Has<Underline>,
            Option<&StrikethroughColor>,
            Option<&UnderlineColor>,
        )>,
    >,
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
        let scaling =
            GlobalTransform::from_scale(Vec3::new(inverse_scale_factor, -inverse_scale_factor, 1.));
        if !view_visibility.get() {
            continue;
        }

        let size = Vec2::new(
            text_bounds.width.unwrap_or(text_layout_info.size.x),
            text_bounds.height.unwrap_or(text_layout_info.size.y),
        );

        let top_left = (Anchor::TOP_LEFT.0 - anchor.as_vec()) * size;

        for run in text_layout_info.run_geometry.iter() {
            let section_entity = computed_block.entities()[run.section_index].entity;
            let Ok(text_background_color) = text_background_colors_query.get(section_entity) else {
                continue;
            };
            let render_entity = commands.spawn(TemporaryRenderEntity).id();
            let offset = run.bounds.center();
            let transform = *global_transform
                * GlobalTransform::from_translation(top_left.extend(0.))
                * scaling
                * GlobalTransform::from_translation(offset.extend(0.));
            extracted_sprites.sprites.push(ExtractedSprite {
                main_entity,
                render_entity,
                transform,
                color: text_background_color.0.into(),
                image_handle_id: AssetId::default(),
                flip_x: false,
                flip_y: true,
                kind: ExtractedSpriteKind::Single {
                    anchor: Vec2::ZERO,
                    rect: None,
                    scaling_mode: None,
                    custom_size: Some(run.bounds.size()),
                },
            });
        }

        let shadow = maybe_shadow
            .filter(|shadow| !shadow.color.is_fully_transparent())
            .and_then(|shadow| {
                text_effect_shadow_offset(
                    shadow.offset,
                    text_layout_info.scale_factor,
                    main_entity,
                    "Text2dShadow",
                )
                .map(|offset| (LinearRgba::from(shadow.color), offset))
            });

        if let Some((color, shadow_offset)) = shadow {
            let shadow_transform = *global_transform
                * GlobalTransform::from_translation(
                    (top_left + shadow_offset * inverse_scale_factor).extend(0.),
                )
                * scaling;

            for run in text_layout_info.run_geometry.iter() {
                let section_entity = computed_block.entities()[run.section_index].entity;
                let Ok((_, has_strikethrough, has_underline, _, _)) =
                    decoration_query.get(section_entity)
                else {
                    continue;
                };

                if has_strikethrough {
                    let render_entity = commands.spawn(TemporaryRenderEntity).id();
                    let offset = run.strikethrough_position();
                    let transform =
                        shadow_transform * GlobalTransform::from_translation(offset.extend(0.));
                    extracted_sprites.sprites.push(ExtractedSprite {
                        main_entity,
                        render_entity,
                        transform,
                        color,
                        image_handle_id: AssetId::default(),
                        flip_x: false,
                        flip_y: false,
                        kind: ExtractedSpriteKind::Single {
                            anchor: Vec2::ZERO,
                            rect: None,
                            scaling_mode: None,
                            custom_size: Some(run.strikethrough_size()),
                        },
                    });
                }

                if has_underline {
                    let render_entity = commands.spawn(TemporaryRenderEntity).id();
                    let offset = run.underline_position();
                    let transform =
                        shadow_transform * GlobalTransform::from_translation(offset.extend(0.));
                    extracted_sprites.sprites.push(ExtractedSprite {
                        main_entity,
                        render_entity,
                        transform,
                        color,
                        image_handle_id: AssetId::default(),
                        flip_x: false,
                        flip_y: false,
                        kind: ExtractedSpriteKind::Single {
                            anchor: Vec2::ZERO,
                            rect: None,
                            scaling_mode: None,
                            custom_size: Some(run.underline_size()),
                        },
                    });
                }
            }
        }

        let transform =
            *global_transform * GlobalTransform::from_translation(top_left.extend(0.)) * scaling;

        let outline = maybe_outline
            .filter(|outline| !outline.color.is_fully_transparent())
            .and_then(|outline| {
                text_effect_outline_width(
                    outline.width,
                    text_layout_info.scale_factor,
                    main_entity,
                    "Text2dOutline",
                )
                .map(|width| (outline.color.to_linear(), width))
            });
        let glyph_padding = combined_text_effect_padding(
            shadow.map(|(_, offset)| offset),
            outline.map(|(_, width)| width),
        );

        let mut color = LinearRgba::WHITE;
        let mut current_section = usize::MAX;
        let mut start = extracted_slices.slices.len();

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            if *section_index != current_section {
                let Some(section_entity) = computed_block
                    .entities()
                    .get(*section_index)
                    .map(|text_entity| text_entity.entity)
                else {
                    tracing::warn!(
                        "Skipping Text2d glyph for {main_entity:?}: missing text section {section_index}"
                    );
                    continue;
                };

                let Ok(text_color) = text_colors.get(section_entity) else {
                    tracing::warn!(
                        "Skipping Text2d glyph for {main_entity:?}: missing TextColor on section entity {section_entity:?}"
                    );
                    continue;
                };

                color = LinearRgba::from(text_color.0);
                current_section = *section_index;
            }
            let shadow = shadow.map(|(color, offset)| (color, Vec2::new(offset.x, -offset.y)));
            let text_effect = if atlas_info.is_alpha_mask {
                ExtractedTextEffect::text(shadow, outline.map(|(color, _)| color))
            } else {
                ExtractedTextEffect::shadow(shadow)
            };
            let rect = if atlas_info.is_alpha_mask || shadow.is_some() {
                glyph_padding
                    .map(|padding| expanded_text_effect_rect(atlas_info.rect, padding))
                    .unwrap_or(atlas_info.rect)
            } else {
                atlas_info.rect
            };

            extracted_slices.slices.push(ExtractedSlice {
                offset: *position,
                rect,
                size: rect.size(),
                text_effect,
            });

            if text_layout_info.glyphs.get(i + 1).is_none_or(|info| {
                info.section_index != current_section
                    || info.atlas_info.texture != atlas_info.texture
            }) {
                let end = extracted_slices.slices.len();
                let render_entity = commands.spawn(TemporaryRenderEntity).id();
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity,
                    transform,
                    color,
                    image_handle_id: atlas_info.texture,
                    flip_x: false,
                    flip_y: true,
                    kind: ExtractedSpriteKind::Slices {
                        indices: start..end,
                    },
                });
                start = end;
            }
        }

        for run in text_layout_info.run_geometry.iter() {
            let section_entity = computed_block.entities()[run.section_index].entity;
            let Ok((
                text_color,
                has_strike_through,
                has_underline,
                maybe_strikethrough_color,
                maybe_underline_color,
            )) = decoration_query.get(section_entity)
            else {
                continue;
            };
            if has_strike_through {
                let color = maybe_strikethrough_color
                    .map(|c| c.0)
                    .unwrap_or(text_color.0)
                    .to_linear();
                let render_entity = commands.spawn(TemporaryRenderEntity).id();
                let offset = run.strikethrough_position();
                let transform = *global_transform
                    * GlobalTransform::from_translation(top_left.extend(0.))
                    * scaling
                    * GlobalTransform::from_translation(offset.extend(0.));
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity,
                    transform,
                    color,
                    image_handle_id: AssetId::default(),
                    flip_x: false,
                    flip_y: false,
                    kind: ExtractedSpriteKind::Single {
                        anchor: Vec2::ZERO,
                        rect: None,
                        scaling_mode: None,
                        custom_size: Some(run.strikethrough_size()),
                    },
                });
            }

            if has_underline {
                let color = maybe_underline_color
                    .map(|c| c.0)
                    .unwrap_or(text_color.0)
                    .to_linear();
                let render_entity = commands.spawn(TemporaryRenderEntity).id();
                let offset = run.underline_position();
                let transform = *global_transform
                    * GlobalTransform::from_translation(top_left.extend(0.))
                    * scaling
                    * GlobalTransform::from_translation(offset.extend(0.));
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity,
                    transform,
                    color,
                    image_handle_id: AssetId::default(),
                    flip_x: false,
                    flip_y: false,
                    kind: ExtractedSpriteKind::Single {
                        anchor: Vec2::ZERO,
                        rect: None,
                        scaling_mode: None,
                        custom_size: Some(run.underline_size()),
                    },
                });
            }
        }
    }
}
