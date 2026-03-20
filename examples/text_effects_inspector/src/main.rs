//! Interactively tunes `TextShadow` / `Text2dShadow` and `TextOutline` / `Text2dOutline`
//! values with bevy-inspector-egui.

use bevy::{
    asset::AssetPlugin,
    color::Srgba,
    math::ops,
    prelude::*,
    sprite::{Anchor, Text2dOutline, Text2dShadow},
};
use bevy_egui::EguiPlugin;
use bevy_inspector_egui::{
    prelude::*,
    quick::{ResourceInspectorPlugin, WorldInspectorPlugin},
    DefaultInspectorConfigPlugin,
};
use std::path::PathBuf;

const INSPECTOR_PANEL_WIDTH: f32 = 320.0;
const PREVIEW_X_OFFSET: f32 = INSPECTOR_PANEL_WIDTH * 0.5;
const TEXT2D_PREVIEW_Y: f32 = -160.0;
const BEFORE_PREVIEW_Y: f32 = -340.0;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Text Effects Inspector".into(),
                        resolution: (1400, 900).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: asset_path(),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(DefaultInspectorConfigPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .register_type::<TextEffectSettings>()
        .init_resource::<TextEffectSettings>()
        .add_plugins(ResourceInspectorPlugin::<TextEffectSettings>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, sync_preview_text)
        .run();
}

fn asset_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets")
        .display()
        .to_string()
}

#[derive(Component)]
struct PreviewText2d;

#[derive(Component)]
struct PreviewUiText;

#[derive(Component)]
struct PreviewBeforeFill;

#[derive(Component)]
struct PreviewBeforeOutline {
    direction: Vec2,
}

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct TextEffectSettings {
    text: String,
    #[inspector(min = 32.0, max = 180.0, speed = 1.0)]
    font_size: f32,
    font_color: Srgba,
    animate_alpha: bool,
    #[inspector(min = 0.0, max = 1.0, speed = 0.01)]
    min_alpha: f32,
    #[inspector(min = 0.0, max = 1.0, speed = 0.01)]
    max_alpha: f32,
    #[inspector(min = 0.1, max = 4.0, speed = 0.01)]
    alpha_speed: f32,
    shadow_enabled: bool,
    shadow_color: Srgba,
    #[inspector(min = -16.0, max = 16.0, speed = 0.1)]
    shadow_x: f32,
    #[inspector(min = -16.0, max = 16.0, speed = 0.1)]
    shadow_y: f32,
    outline_enabled: bool,
    outline_color: Srgba,
    #[inspector(min = 0.0, max = 16.0, speed = 0.1)]
    outline_width: f32,
}

impl Default for TextEffectSettings {
    fn default() -> Self {
        Self {
            text: "outline gap test".to_string(),
            font_size: 110.0,
            font_color: Srgba::WHITE,
            animate_alpha: true,
            min_alpha: 0.0,
            max_alpha: 1.0,
            alpha_speed: 1.0,
            shadow_enabled: true,
            shadow_color: Srgba::new(0.0, 0.0, 0.0, 0.85),
            shadow_x: 4.0,
            shadow_y: -4.0,
            outline_enabled: true,
            outline_color: Srgba::BLACK,
            outline_width: 2.0,
        }
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((Name::new("Preview Camera"), Camera2d));
    let font = asset_server.load("fonts/FiraSans-Bold.ttf");

    let tile_size = 96.0;
    for y in -5..=5 {
        for x in -8..=8 {
            let is_dark = (x + y) % 2 == 0;
            commands.spawn((
                Name::new(format!("Checkerboard Tile ({x}, {y})")),
                Sprite::from_color(
                    if is_dark {
                        Color::srgb(0.12, 0.13, 0.16)
                    } else {
                        Color::srgb(0.24, 0.25, 0.29)
                    },
                    Vec2::splat(tile_size),
                ),
                Transform::from_xyz(x as f32 * tile_size, y as f32 * tile_size, -10.0),
            ));
        }
    }

    commands.spawn((
        Name::new("Text2d Preview"),
        Text2d::new(labeled_text("text2d", "outline gap test")),
        TextFont {
            font: font.clone(),
            font_size: 110.0,
            ..default()
        },
        TextLayout::new_with_justify(Justify::Center),
        TextColor(Color::WHITE),
        Text2dShadow {
            offset: Vec2::new(4.0, -4.0),
            color: Color::srgba(0.0, 0.0, 0.0, 0.85),
        },
        Text2dOutline {
            color: Color::BLACK,
            width: 2.0,
        },
        Anchor::CENTER,
        Transform::from_xyz(PREVIEW_X_OFFSET, TEXT2D_PREVIEW_Y, 10.0),
        PreviewText2d,
    ));

    commands
        .spawn((
            Name::new("UI Preview Container"),
            Node {
                position_type: PositionType::Absolute,
                top: px(120.0),
                left: px(INSPECTOR_PANEL_WIDTH),
                right: px(0.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
        ))
        .with_child((
            Name::new("UI Preview Text"),
            Text::new(labeled_text("textui", "outline gap test")),
            TextFont {
                font: font.clone(),
                font_size: 110.0,
                ..default()
            },
            TextLayout::new_with_justify(Justify::Center),
            TextColor(Color::WHITE),
            TextShadow {
                offset: Vec2::new(4.0, -4.0),
                color: Color::srgba(0.0, 0.0, 0.0, 0.85),
            },
            TextOutline {
                color: Color::BLACK,
                width: 2.0,
            },
            PreviewUiText,
        ));

    let before_position = Vec3::new(PREVIEW_X_OFFSET, BEFORE_PREVIEW_Y, 10.0);
    for (index, direction) in before_outline_directions().into_iter().enumerate() {
        commands.spawn((
            Name::new(format!("Before Outline Copy {index}")),
            Text2d::new(labeled_text("before", "outline gap test")),
            TextFont {
                font: font.clone(),
                font_size: 110.0,
                ..default()
            },
            TextLayout::new_with_justify(Justify::Center),
            TextColor(Color::BLACK),
            Anchor::CENTER,
            Transform::from_translation(before_position + direction.extend(-1.0) * 2.0),
            PreviewBeforeOutline { direction },
        ));
    }

    commands.spawn((
        Name::new("Before Fill Preview"),
        Text2d::new(labeled_text("before", "outline gap test")),
        TextFont {
            font,
            font_size: 110.0,
            ..default()
        },
        TextLayout::new_with_justify(Justify::Center),
        TextColor(Color::WHITE),
        Anchor::CENTER,
        Transform::from_translation(before_position),
        PreviewBeforeFill,
    ));
}

fn sync_preview_text(
    time: Res<Time>,
    mut settings: ResMut<TextEffectSettings>,
    mut commands: Commands,
    mut previews: ParamSet<(
        Query<(Entity, &mut Text2d, &mut TextFont, &mut TextColor), With<PreviewText2d>>,
        Query<(Entity, &mut Text, &mut TextFont, &mut TextColor), With<PreviewUiText>>,
        Query<(&mut Text2d, &mut TextFont, &mut TextColor), With<PreviewBeforeFill>>,
        Query<
            (
                &PreviewBeforeOutline,
                &mut Text2d,
                &mut TextFont,
                &mut TextColor,
                &mut Transform,
                &mut Visibility,
            ),
            With<PreviewBeforeOutline>,
        >,
    )>,
) {
    if settings.animate_alpha {
        let wave = 0.5 + 0.5 * ops::sin(settings.alpha_speed * time.elapsed_secs());
        let alpha = settings.min_alpha + (settings.max_alpha - settings.min_alpha) * wave;
        settings.font_color.alpha = alpha;
        settings.shadow_color.alpha = alpha;
        settings.outline_color.alpha = alpha;
    }

    let font_color = Color::Srgba(settings.font_color);
    let shadow_color = Color::Srgba(settings.shadow_color);
    let outline_color = Color::Srgba(settings.outline_color);
    let text2d_label = labeled_text("text2d", &settings.text);
    let textui_label = labeled_text("textui", &settings.text);
    let before_label = labeled_text("before", &settings.text);
    let before_outline_visible = settings.outline_enabled && settings.outline_width > 0.0;

    if let Ok((entity, mut text, mut text_font, mut text_color)) = previews.p0().single_mut() {
        *text = Text2d::new(text2d_label.clone());
        text_font.font_size = settings.font_size;
        text_color.0 = font_color;

        if settings.shadow_enabled {
            commands.entity(entity).insert(Text2dShadow {
                offset: Vec2::new(settings.shadow_x, settings.shadow_y),
                color: shadow_color,
            });
        } else {
            commands.entity(entity).remove::<Text2dShadow>();
        }

        if settings.outline_enabled {
            commands.entity(entity).insert(Text2dOutline {
                color: outline_color,
                width: settings.outline_width,
            });
        } else {
            commands.entity(entity).remove::<Text2dOutline>();
        }
    }

    if let Ok((entity, mut text, mut text_font, mut text_color)) = previews.p1().single_mut() {
        *text = Text::new(textui_label);
        text_font.font_size = settings.font_size;
        text_color.0 = font_color;

        if settings.shadow_enabled {
            commands.entity(entity).insert(TextShadow {
                offset: Vec2::new(settings.shadow_x, settings.shadow_y),
                color: shadow_color,
            });
        } else {
            commands.entity(entity).remove::<TextShadow>();
        }

        if settings.outline_enabled {
            commands.entity(entity).insert(TextOutline {
                color: outline_color,
                width: settings.outline_width,
            });
        } else {
            commands.entity(entity).remove::<TextOutline>();
        }
    }

    if let Ok((mut text, mut text_font, mut text_color)) = previews.p2().single_mut() {
        *text = Text2d::new(before_label.clone());
        text_font.font_size = settings.font_size;
        text_color.0 = font_color;
    }

    for (before_outline, mut text, mut text_font, mut text_color, mut transform, mut visibility) in
        &mut previews.p3()
    {
        *text = Text2d::new(before_label.clone());
        text_font.font_size = settings.font_size;
        text_color.0 = outline_color;
        transform.translation = Vec3::new(PREVIEW_X_OFFSET, BEFORE_PREVIEW_Y, 9.0)
            + (before_outline.direction * settings.outline_width).extend(0.0);
        *visibility = if before_outline_visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn labeled_text(prefix: &str, text: &str) -> String {
    format!("{prefix}: {text}")
}

fn before_outline_directions() -> [Vec2; 8] {
    [
        Vec2::X,
        -Vec2::X,
        Vec2::Y,
        -Vec2::Y,
        Vec2::new(1.0, 1.0).normalize(),
        Vec2::new(1.0, -1.0).normalize(),
        Vec2::new(-1.0, 1.0).normalize(),
        Vec2::new(-1.0, -1.0).normalize(),
    ]
}
