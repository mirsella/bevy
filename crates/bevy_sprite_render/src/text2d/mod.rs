use crate::text_group::{TextGroupCache, TextGroupEffects, TextGroupStyleQuery};
use crate::{
    ExtractedSlice, ExtractedSlices, ExtractedSprite, ExtractedSpriteKind, ExtractedSprites,
    ExtractedTextEffect,
};
use bevy_asset::{AssetId, Assets};
use bevy_camera::visibility::ViewVisibility;
use bevy_color::{Alpha, LinearRgba};
use bevy_ecs::change_detection::Ref;
use bevy_ecs::{
    entity::Entity,
    query::Has,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{Vec2, Vec3};
use bevy_render::sync_world::TemporaryRenderEntity;
use bevy_render::Extract;
use bevy_sprite::{Anchor, Text2dOutline, Text2dShadow};
use bevy_text::{
    glyph_effect_padding, text_effect_outline_width, text_effect_shadow_offset, ComputedTextBlock,
    PositionedGlyph, Strikethrough, StrikethroughColor, TextBackgroundColor, TextBounds, TextColor,
    TextEffectLayer, TextLayoutInfo, Underline, UnderlineColor,
};
use bevy_text::{InheritedTextOpacity, TextOpacity};
use bevy_transform::prelude::GlobalTransform;

/// This system extracts the sprites from the 2D text components and adds them to the
/// "render world".
pub fn extract_text2d_sprite(
    mut commands: Commands,
    mut groups: ResMut<TextGroupCache>,
    group_styles: Extract<Query<TextGroupStyleQuery>>,
    images: Extract<Res<Assets<Image>>>,
    mut extracted_sprites: ResMut<ExtractedSprites>,
    mut extracted_slices: ResMut<ExtractedSlices>,
    text2d_query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &ComputedTextBlock,
            Ref<TextLayoutInfo>,
            Option<&TextOpacity>,
            Option<&InheritedTextOpacity>,
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
        opacity,
        inherited_opacity,
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

        let text_transform =
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
        if let Some(opacity) = opacity {
            let alpha = opacity.get() * inherited_opacity.map_or(1., |inherited| inherited.get());
            if alpha == 0. {
                groups.keep_cached(main_entity);
                continue;
            }
            let effects = TextGroupEffects {
                shadow: shadow.map(|(c, o)| (c, Vec2::new(o.x, -o.y))),
                outline,
                tint_color_glyphs: true,
                ..Default::default()
            };
            for tile in groups.extract(
                main_entity,
                &images,
                &text_layout_info,
                computed_block.entities(),
                &group_styles,
                effects,
            ) {
                let index = extracted_slices.slices.len();
                extracted_slices.slices.push(ExtractedSlice {
                    offset: tile.bounds.center(),
                    rect: tile.rect,
                    size: tile.bounds.size(),
                    text_effect: ExtractedTextEffect::Group,
                });
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    transform: text_transform,
                    color: LinearRgba::WHITE.with_alpha(alpha),
                    image_handle_id: tile.image,
                    flip_x: false,
                    flip_y: true,
                    kind: ExtractedSpriteKind::Slices {
                        indices: index..index + 1,
                    },
                });
            }
            continue;
        }

        for run in text_layout_info.run_geometry.iter() {
            let section_entity = computed_block.entities()[run.section_index].entity;
            let Ok(text_background_color) = text_background_colors_query.get(section_entity) else {
                continue;
            };
            let render_entity = commands.spawn(TemporaryRenderEntity).id();
            extracted_sprites.sprites.push(ExtractedSprite {
                main_entity,
                render_entity,
                transform: text_transform
                    * GlobalTransform::from_translation(run.bounds.center().extend(0.)),
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

                for (position, size) in [
                    has_strikethrough
                        .then(|| (run.strikethrough_position(), run.strikethrough_size())),
                    has_underline.then(|| (run.underline_position(), run.underline_size())),
                ]
                .into_iter()
                .flatten()
                {
                    extracted_sprites.sprites.push(ExtractedSprite {
                        main_entity,
                        render_entity: commands.spawn(TemporaryRenderEntity).id(),
                        transform: shadow_transform
                            * GlobalTransform::from_translation(position.extend(0.)),
                        color,
                        image_handle_id: AssetId::default(),
                        flip_x: false,
                        flip_y: false,
                        kind: ExtractedSpriteKind::Single {
                            anchor: Vec2::ZERO,
                            rect: None,
                            scaling_mode: None,
                            custom_size: Some(size),
                        },
                    });
                }
            }
        }

        let glyph_padding = glyph_effect_padding(
            shadow.is_some() || outline.is_some(),
            outline.map(|(_, width)| width),
        );

        // Keep every shadow behind every stroke, and every stroke behind every fill.
        for layer in TextEffectLayer::ALL {
            let (paint, offset) = match layer {
                TextEffectLayer::Shadow => {
                    let Some((color, offset)) = shadow else {
                        continue;
                    };
                    (Some(color), Vec2::new(offset.x, -offset.y))
                }
                TextEffectLayer::Outline => {
                    let Some((color, _)) = outline else { continue };
                    (Some(color), Vec2::ZERO)
                }
                TextEffectLayer::Fill => (None, Vec2::ZERO),
            };
            let mut color = LinearRgba::WHITE;
            let mut current_section = usize::MAX;
            let layer_padding = layer.glyph_padding(glyph_padding);
            let mut start = extracted_slices.slices.len();

            let mut glyphs = text_layout_info
                .glyphs
                .iter()
                .filter(|glyph| layer != TextEffectLayer::Outline || glyph.atlas_info.is_alpha_mask)
                .peekable();
            while let Some(PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            }) = glyphs.next()
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
                let text_effect = match layer {
                    TextEffectLayer::Shadow if atlas_info.is_alpha_mask && outline.is_some() => {
                        ExtractedTextEffect::OutlinedShadow
                    }
                    TextEffectLayer::Shadow => ExtractedTextEffect::Alpha,
                    TextEffectLayer::Outline => ExtractedTextEffect::Outline,
                    TextEffectLayer::Fill if atlas_info.is_alpha_mask => ExtractedTextEffect::Alpha,
                    TextEffectLayer::Fill => ExtractedTextEffect::None,
                };
                let rect = if atlas_info.is_alpha_mask || layer == TextEffectLayer::Shadow {
                    atlas_info.rect.inflate(layer_padding)
                } else {
                    atlas_info.rect
                };

                extracted_slices.slices.push(ExtractedSlice {
                    // Offset the geometry without changing the block's depth sort key.
                    // A translated layer transform would reorder effects on tilted text.
                    offset: *position + offset,
                    rect,
                    size: rect.size(),
                    text_effect,
                });

                if glyphs.peek().is_none_or(|info| {
                    info.section_index != current_section
                        || info.atlas_info.texture != atlas_info.texture
                }) {
                    let end = extracted_slices.slices.len();
                    let render_entity = commands.spawn(TemporaryRenderEntity).id();
                    extracted_sprites.sprites.push(ExtractedSprite {
                        main_entity,
                        render_entity,
                        transform: text_transform,
                        color: paint.unwrap_or(color),
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
            for (color, position, size) in [
                has_strike_through.then(|| {
                    (
                        maybe_strikethrough_color.map_or(text_color.0, |c| c.0),
                        run.strikethrough_position(),
                        run.strikethrough_size(),
                    )
                }),
                has_underline.then(|| {
                    (
                        maybe_underline_color.map_or(text_color.0, |c| c.0),
                        run.underline_position(),
                        run.underline_size(),
                    )
                }),
            ]
            .into_iter()
            .flatten()
            {
                extracted_sprites.sprites.push(ExtractedSprite {
                    main_entity,
                    render_entity: commands.spawn(TemporaryRenderEntity).id(),
                    transform: text_transform
                        * GlobalTransform::from_translation(position.extend(0.)),
                    color: color.to_linear(),
                    image_handle_id: AssetId::default(),
                    flip_x: false,
                    flip_y: false,
                    kind: ExtractedSpriteKind::Single {
                        anchor: Vec2::ZERO,
                        rect: None,
                        scaling_mode: None,
                        custom_size: Some(size),
                    },
                });
            }
        }
    }
}
