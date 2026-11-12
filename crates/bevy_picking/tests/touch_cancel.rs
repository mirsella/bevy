//! Regression test for touch cancellation dispatch.

use bevy_app::{App, PreUpdate};
use bevy_ecs::prelude::*;
use bevy_input::{
    touch::{TouchInput, TouchPhase},
    InputPlugin,
};
use bevy_math::Vec2;
use bevy_picking::{
    backend::{HitData, PointerHits},
    events::{PointerCancel, PointerPress},
    pointer::{PointerId, PointerLocation},
    DefaultPickingPlugins, PickingSystems,
};
use bevy_window::{PrimaryWindow, WindowEvent, WindowPlugin};

#[test]
fn touch_cancellation_reaches_pressed_target() {
    let mut app = App::new();
    app.add_plugins((InputPlugin, WindowPlugin::default(), DefaultPickingPlugins))
        .add_systems(
            PreUpdate,
            (|pointers: Query<(&PointerId, &PointerLocation)>,
              window: Single<Entity, With<PrimaryWindow>>,
              mut hits: MessageWriter<PointerHits>| {
                for (id, location) in &pointers {
                    if location.location().is_some() {
                        hits.write(PointerHits::new(
                            *id,
                            vec![(*window, HitData::new(*window, 0., None, None))],
                            0.,
                        ));
                    }
                }
            })
            .in_set(PickingSystems::Backend),
        );
    app.update();
    let window = app
        .world_mut()
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .single(app.world())
        .unwrap();
    for phase in [TouchPhase::Started, TouchPhase::Canceled] {
        app.world_mut()
            .write_message(WindowEvent::TouchInput(TouchInput {
                window,
                phase,
                position: Vec2::splat(50.),
                force: None,
                id: 42,
            }));
        app.update();
    }
    assert_eq!(app.world().resource::<Messages<PointerPress>>().len(), 1);
    assert_eq!(app.world().resource::<Messages<PointerCancel>>().len(), 1);
}
