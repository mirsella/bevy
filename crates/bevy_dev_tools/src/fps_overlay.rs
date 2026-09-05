//! Module containing logic for FPS overlay.

use bevy_app::{Plugin, Startup, Update};
use bevy_asset::Assets;
use bevy_color::Color;
use bevy_diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without},
    reflect::ReflectResource,
    resource::Resource,
    schedule::{common_conditions::resource_changed, IntoScheduleConfigs, SystemSet},
    system::{Commands, Query, Res, ResMut, Single},
};
use bevy_picking::Pickable;
use bevy_reflect::Reflect;
use bevy_render::storage::ShaderBuffer;
use bevy_text::{RemSize, TextColor, TextFont, TextSpan};
use bevy_time::{Real, Time, Timer, TimerMode};
use bevy_ui::{
    widget::{Text, TextUiWriter},
    ComputedUiRenderTargetInfo, FlexDirection, GlobalZIndex, Node, PositionType, Val,
};
#[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
use bevy_ui_render::prelude::MaterialNode;
use core::time::Duration;
use tracing::warn;

#[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
use crate::frame_time_graph::FrameTimeGraphConfigUniform;
use crate::frame_time_graph::{FrameTimeGraphPlugin, FrametimeGraphMaterial};

/// [`GlobalZIndex`] used to render the fps overlay.
///
/// We use a number slightly under `i32::MAX` so you can render on top of it if you really need to.
pub const FPS_OVERLAY_ZINDEX: i32 = i32::MAX - 32;

// Warn the user if the interval is below this threshold.
const MIN_SAFE_INTERVAL: Duration = Duration::from_millis(50);

// Used to scale the frame time graph based on the fps text size
const FRAME_TIME_GRAPH_WIDTH_SCALE: f32 = 6.0;
const FRAME_TIME_GRAPH_HEIGHT_SCALE: f32 = 2.0;

/// A plugin that adds an FPS overlay to the Bevy application.
///
/// This plugin will add the [`FrameTimeDiagnosticsPlugin`] if it wasn't added before.
///
/// Note: It is recommended to use native overlay of rendering statistics when possible for lower overhead and more accurate results.
/// The correct way to do this will vary by platform:
/// - **Metal**: setting env variable `MTL_HUD_ENABLED=1`
#[derive(Default)]
pub struct FpsOverlayPlugin {
    /// Starting configuration of overlay, this can be later be changed through [`FpsOverlayConfig`] resource.
    pub config: FpsOverlayConfig,
}

/// System sets for FPS overlay updates.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum FpsOverlaySystems {
    /// Applies config changes to the overlay UI.
    Customize,
    /// Updates the overlay contents.
    UpdateText,
}

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        // TODO: Use plugin dependencies, see https://github.com/bevyengine/bevy/issues/69
        if !app.is_plugin_added::<FrameTimeDiagnosticsPlugin>() {
            app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        }

        if !app.is_plugin_added::<FrameTimeGraphPlugin>() {
            app.add_plugins(FrameTimeGraphPlugin);
        }

        if self.config.refresh_interval < MIN_SAFE_INTERVAL {
            warn!(
                "Low refresh interval ({:?}) may degrade performance. \
                Min recommended: {:?}.",
                self.config.refresh_interval, MIN_SAFE_INTERVAL
            );
        }

        app.insert_resource(self.config.clone())
            .init_resource::<FpsOverlaySample>()
            .configure_sets(
                Update,
                FpsOverlaySystems::Customize.before(FpsOverlaySystems::UpdateText),
            )
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    (toggle_display, customize_overlay)
                        .run_if(resource_changed::<FpsOverlayConfig>)
                        .in_set(FpsOverlaySystems::Customize),
                    sample_frames
                        .after(FpsOverlaySystems::Customize)
                        .before(FpsOverlaySystems::UpdateText),
                    update_text
                        .after(FrameTimeDiagnosticsPlugin::diagnostic_system)
                        .in_set(FpsOverlaySystems::UpdateText),
                ),
            );
    }
}

/// Configuration options for the FPS overlay.
#[derive(Resource, Clone, Reflect)]
#[reflect(Resource)]
pub struct FpsOverlayConfig {
    /// Configuration of text in the overlay.
    pub text_config: TextFont,
    /// Color of text in the overlay.
    pub text_color: Color,
    /// Displays the FPS overlay if true.
    pub enabled: bool,
    /// The period after which the FPS overlay re-renders.
    ///
    /// Defaults to once every 100 ms.
    pub refresh_interval: Duration,
    /// Display frames divided by elapsed real time over each refresh interval,
    /// instead of the diagnostic's exponentially smoothed FPS. Defaults to false.
    pub average_over_interval: bool,
    /// Configuration of the frame time graph
    pub frame_time_graph_config: FrameTimeGraphConfig,
}

impl Default for FpsOverlayConfig {
    fn default() -> Self {
        FpsOverlayConfig {
            text_config: TextFont::from_font_size(32.),
            text_color: Color::WHITE,
            enabled: true,
            refresh_interval: Duration::from_millis(100),
            average_over_interval: false,
            // TODO set this to display refresh rate if possible
            frame_time_graph_config: FrameTimeGraphConfig::target_fps(60.0),
        }
    }
}

/// Configuration of the frame time graph
#[derive(Clone, Copy, Reflect)]
pub struct FrameTimeGraphConfig {
    /// Is the graph visible
    pub enabled: bool,
    /// The minimum acceptable FPS
    ///
    /// Anything below this will show a red bar
    pub min_fps: f32,
    /// The target FPS
    ///
    /// Anything above this will show a green bar
    pub target_fps: f32,
}

impl FrameTimeGraphConfig {
    /// Constructs a default config for a given target fps
    pub fn target_fps(target_fps: f32) -> Self {
        Self {
            target_fps,
            ..Self::default()
        }
    }
}

impl Default for FrameTimeGraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_fps: 30.0,
            target_fps: 60.0,
        }
    }
}

#[derive(Component)]
struct FpsText;

#[derive(Component)]
struct FrameTimeGraph;

fn setup(
    mut commands: Commands,
    overlay_config: Res<FpsOverlayConfig>,
    #[cfg_attr(
        all(target_arch = "wasm32", not(feature = "webgpu")),
        expect(unused, reason = "Unused variables in wasm32 without webgpu feature")
    )]
    (mut frame_time_graph_materials, mut buffers): (
        ResMut<Assets<FrametimeGraphMaterial>>,
        ResMut<Assets<ShaderBuffer>>,
    ),
) {
    commands
        .spawn((
            Node {
                // We need to make sure the overlay doesn't affect the position of other UI nodes
                position_type: PositionType::Absolute,
                flex_direction: FlexDirection::Column,
                ..Default::default()
            },
            // Render overlay on top of everything
            GlobalZIndex(FPS_OVERLAY_ZINDEX),
            Pickable::IGNORE,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("FPS: "),
                Node {
                    display: if overlay_config.enabled {
                        bevy_ui::Display::DEFAULT
                    } else {
                        bevy_ui::Display::None
                    },
                    ..Default::default()
                },
                overlay_config.text_config.clone(),
                TextColor(overlay_config.text_color),
                FpsText,
                Pickable::IGNORE,
            ))
            .with_child((TextSpan::default(), overlay_config.text_config.clone()));

            #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
            {
                if overlay_config.frame_time_graph_config.enabled {
                    use tracing::warn;

                    warn!("Frame time graph is not supported with WebGL. Consider if WebGPU is viable for your usecase.");
                }
            }
            #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
            {
                // Todo: Needs a better design that works with responsive sizing.
                let font_size = 20.;
                p.spawn((
                    Node {
                        width: Val::Px(font_size * FRAME_TIME_GRAPH_WIDTH_SCALE),
                        height: Val::Px(font_size * FRAME_TIME_GRAPH_HEIGHT_SCALE),
                        display: if overlay_config.enabled && overlay_config.frame_time_graph_config.enabled {
                            bevy_ui::Display::DEFAULT
                        } else {
                            bevy_ui::Display::None
                        },
                        ..Default::default()
                    },
                    Pickable::IGNORE,
                    MaterialNode::from(frame_time_graph_materials.add(FrametimeGraphMaterial {
                        values: buffers.add(ShaderBuffer {
                            // Initialize with dummy data because the default (`data: None`) will
                            // cause a panic in the shader if the frame time graph is constructed
                            // with `enabled: false`.
                            data: Some(vec![0, 0, 0, 0]),
                            ..Default::default()
                        }),
                        config: FrameTimeGraphConfigUniform::new(
                            overlay_config.frame_time_graph_config.target_fps,
                            overlay_config.frame_time_graph_config.min_fps,
                            true,
                        ),
                    })),
                    FrameTimeGraph,
                ));
            }
        });
}

#[derive(Resource, Default)]
pub(crate) struct FpsOverlaySample {
    timer: Timer,
    frames: u64,
    config: Option<(Duration, bool, bool)>,
    pub(crate) average: Option<f64>,
}

impl FpsOverlaySample {
    fn tick(&mut self, delta: Duration, config: &FpsOverlayConfig) {
        let settings = (
            config.refresh_interval,
            config.average_over_interval,
            config.enabled,
        );
        if self.config != Some(settings) {
            *self = Self {
                timer: Timer::new(config.refresh_interval, TimerMode::Repeating),
                config: Some(settings),
                ..Default::default()
            };
        }
        self.average = None;
        if !config.enabled || delta.is_zero() {
            return;
        }
        let elapsed = self.timer.elapsed() + delta;
        self.frames += 1;
        if self.timer.tick(delta).just_finished() {
            self.average = Some(self.frames as f64 / elapsed.as_secs_f64());
            if config.average_over_interval {
                // Averages consume whole frames and their full elapsed interval.
                self.timer.reset();
            }
            self.frames = 0;
        }
    }
}

fn sample_frames(
    time: Res<Time<Real>>,
    config: Res<FpsOverlayConfig>,
    mut sample: ResMut<FpsOverlaySample>,
) {
    sample.tick(time.delta(), &config);
}

fn update_text(
    config: Res<FpsOverlayConfig>,
    sample: Res<FpsOverlaySample>,
    diagnostic: Res<DiagnosticsStore>,
    query: Query<Entity, With<FpsText>>,
    mut writer: TextUiWriter,
) {
    let Some(average) = sample.average else {
        return;
    };
    let value = if config.average_over_interval {
        Some(average)
    } else {
        diagnostic
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed())
    };
    if let Ok(entity) = query.single()
        && let Some(value) = value
    {
        *writer.text(entity, 1) = format!("{value:.2}");
    }
}

fn customize_overlay(
    overlay_config: Res<FpsOverlayConfig>,
    query: Query<Entity, With<FpsText>>,
    mut writer: TextUiWriter,
) {
    for entity in &query {
        writer.for_each_font(entity, |mut font| {
            *font = overlay_config.text_config.clone();
        });
        writer.for_each_color(entity, |mut color| color.0 = overlay_config.text_color);
    }
}

fn toggle_display(
    overlay_config: Res<FpsOverlayConfig>,
    mut text_node: Single<
        (&mut Node, &ComputedUiRenderTargetInfo),
        (With<FpsText>, Without<FrameTimeGraph>),
    >,
    mut graph_nodes: Query<&mut Node, (With<FrameTimeGraph>, Without<FpsText>)>,
    rem_size: Res<RemSize>,
) {
    if overlay_config.enabled {
        text_node.0.display = bevy_ui::Display::DEFAULT;
    } else {
        text_node.0.display = bevy_ui::Display::None;
    }

    for mut graph_node in &mut graph_nodes {
        if overlay_config.enabled && overlay_config.frame_time_graph_config.enabled {
            // Scale the frame time graph based on the font size of the overlay
            let font_size = overlay_config
                .text_config
                .font_size
                .eval(text_node.1.logical_size(), rem_size.0);
            graph_node.width = Val::Px(font_size * FRAME_TIME_GRAPH_WIDTH_SCALE);
            graph_node.height = Val::Px(font_size * FRAME_TIME_GRAPH_HEIGHT_SCALE);

            graph_node.display = bevy_ui::Display::DEFAULT;
        } else {
            graph_node.display = bevy_ui::Display::None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_smoothed_refresh_matches_native_repeating_timer() {
        let config = FpsOverlayConfig::default();
        let mut sample = FpsOverlaySample::default();
        let mut native_timer = Timer::new(config.refresh_interval, TimerMode::Repeating);
        let mut refreshes = 0;
        for _ in 0..100 {
            let delta = Duration::from_millis(33);
            sample.tick(delta, &config);
            native_timer.tick(delta);
            assert_eq!(sample.average.is_some(), native_timer.just_finished());
            refreshes += usize::from(sample.average.is_some());
        }
        assert_eq!(refreshes, 33);
    }

    #[test]
    fn fps_interval_averaging_and_runtime_changes() {
        let mut sample = FpsOverlaySample::default();
        let mut config = FpsOverlayConfig {
            refresh_interval: Duration::from_secs(1),
            average_over_interval: true,
            ..Default::default()
        };
        sample.tick(Duration::ZERO, &config);
        assert_eq!(sample.average, None);
        for _ in 0..50 {
            sample.tick(Duration::from_millis(10), &config);
            assert_eq!(sample.average, None);
        }
        for _ in 0..24 {
            sample.tick(Duration::from_millis(20), &config);
            assert_eq!(sample.average, None);
        }
        sample.tick(Duration::from_millis(20), &config);
        assert_eq!(sample.average, Some(75.0));
        sample.tick(Duration::from_millis(1100), &config);
        assert_eq!(sample.average, Some(1.0 / 1.1));
        sample.tick(Duration::from_millis(900), &config);
        config.refresh_interval = Duration::from_millis(100);
        config.average_over_interval = false;
        sample.tick(Duration::from_millis(50), &config);
        assert_eq!(sample.average, None);
        sample.tick(Duration::from_millis(50), &config);
        assert_eq!(sample.average, Some(20.0));
        config.enabled = false;
        sample.tick(Duration::from_secs(10), &config);
        assert_eq!(sample.average, None);
        config.enabled = true;
        sample.tick(Duration::from_millis(50), &config);
        assert_eq!(sample.average, None);
        sample.tick(Duration::from_millis(50), &config);
        assert_eq!(sample.average, Some(20.0));
    }
}
