use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_input::touch::{TouchInput, TouchPhase};
use bevy_math::{Rect, Vec2};
use bevy_window::{PrimaryWindow, Window, WindowEvent};

use super::{Cancel, Click, Pointer, PointerState, Press, Release};
use crate::{
    backend::{HitData, PointerHits},
    pointer::{
        Location, PointerAction, PointerButton, PointerId, PointerInput, PointerLocation,
        PointerPress,
    },
    DefaultPickingPlugins, PickingSettings, PickingSystems,
};

#[derive(Component)]
struct HitBox(Rect);

#[derive(Resource, Default)]
struct Events(Vec<(&'static str, PointerId, Entity)>);

#[derive(Resource, Default)]
struct BackendPasses(usize);

// Geometry is deliberately simple; native touch ingestion, pointer lifetimes,
// hover generation and observer dispatch all use the real picking plugins.
fn hit_test(
    pointers: Query<(&PointerId, &PointerLocation)>,
    targets: Query<(Entity, &HitBox)>,
    mut hits: MessageWriter<PointerHits>,
    mut passes: ResMut<BackendPasses>,
) {
    passes.0 += 1;
    for (pointer, location) in &pointers {
        let Some(location) = location.location() else {
            continue;
        };
        for (entity, bounds) in &targets {
            if bounds.0.contains(location.position) {
                hits.write(PointerHits::new(
                    *pointer,
                    vec![(entity, HitData::new(entity, 0., None, None))],
                    1.,
                ));
            }
        }
    }
}

fn setup() -> (App, Entity, Entity, Entity) {
    let mut app = App::new();
    app.add_message::<WindowEvent>()
        .add_message::<TouchInput>()
        .add_plugins(DefaultPickingPlugins)
        .init_resource::<Events>()
        .init_resource::<BackendPasses>()
        .add_systems(crate::Picking, hit_test.in_set(PickingSystems::Backend))
        .add_observer(
            |mut event: On<Pointer<Press>>, mut events: ResMut<Events>| {
                event.propagate(false);
                events.0.push(("press", event.pointer_id, event.entity));
            },
        )
        .add_observer(
            |mut event: On<Pointer<Click>>, mut events: ResMut<Events>| {
                event.propagate(false);
                events.0.push(("click", event.pointer_id, event.entity));
            },
        )
        .add_observer(
            |mut event: On<Pointer<Release>>, mut events: ResMut<Events>| {
                event.propagate(false);
                events.0.push(("release", event.pointer_id, event.entity));
            },
        )
        .add_observer(
            |mut event: On<Pointer<Cancel>>, mut events: ResMut<Events>| {
                event.propagate(false);
                events.0.push(("cancel", event.pointer_id, event.entity));
            },
        );
    app.world_mut()
        .resource_mut::<PickingSettings>()
        .is_window_picking_enabled = false;
    let window = app
        .world_mut()
        .spawn((Window::default(), PrimaryWindow))
        .id();
    let first = app
        .world_mut()
        .spawn(HitBox(Rect::new(0., 0., 100., 100.)))
        .id();
    let second = app
        .world_mut()
        .spawn(HitBox(Rect::new(200., 0., 300., 100.)))
        .id();
    (app, window, first, second)
}

fn touch(app: &mut App, window: Entity, id: u64, phase: TouchPhase, x: f32) {
    let input = TouchInput {
        window,
        id,
        phase,
        position: Vec2::new(x, 50.),
        force: None,
    };
    app.world_mut()
        .write_message(WindowEvent::TouchInput(input));
    app.world_mut().write_message(input);
}

#[test]
fn touch_tap_can_begin_and_end_in_one_frame() {
    for held_frames in 0..=2 {
        let (mut app, window, target, _) = setup();
        touch(&mut app, window, 1, TouchPhase::Started, 50.);
        for _ in 0..held_frames {
            app.update();
        }
        touch(&mut app, window, 1, TouchPhase::Ended, 50.);
        app.update();
        assert_eq!(
            app.world().resource::<Events>().0,
            [
                ("press", PointerId::Touch(1), target),
                ("click", PointerId::Touch(1), target),
                ("release", PointerId::Touch(1), target),
            ],
            "held_frames={held_frames}"
        );
        assert_eq!(
            app.world_mut()
                .query::<&PointerId>()
                .iter(app.world())
                .filter(|id| id.is_touch())
                .count(),
            0
        );
    }
}

#[test]
fn release_uses_its_position_and_disarms_the_original_target() {
    for (release_x, over_second) in [(150., false), (250., true)] {
        let (mut app, window, first, second) = setup();
        touch(&mut app, window, 1, TouchPhase::Started, 50.);
        app.update();
        // No Move sample: release itself must update the location before hit testing.
        touch(&mut app, window, 1, TouchPhase::Ended, release_x);
        app.update();
        let mut expected = vec![("press", PointerId::Touch(1), first)];
        if over_second {
            expected.push(("release", PointerId::Touch(1), second));
        }
        expected.push(("release", PointerId::Touch(1), first));
        assert_eq!(app.world().resource::<Events>().0, expected);
    }
}

#[test]
fn same_frame_displacement_never_retargets_the_press() {
    for moved in [false, true] {
        for start in [50., 150.] {
            let (mut app, window, first, second) = setup();
            touch(&mut app, window, 1, TouchPhase::Started, start);
            if moved {
                touch(&mut app, window, 1, TouchPhase::Moved, 250.);
            }
            touch(&mut app, window, 1, TouchPhase::Ended, 250.);
            app.update();
            let mut expected = Vec::new();
            if start < 100. {
                expected.push(("press", PointerId::Touch(1), first));
            }
            expected.push(("release", PointerId::Touch(1), second));
            if start < 100. {
                expected.push(("release", PointerId::Touch(1), first));
            }
            assert_eq!(app.world().resource::<Events>().0, expected);
            assert_eq!(app.world().resource::<BackendPasses>().0, 2);
        }
    }
}

#[test]
fn movement_samples_share_hit_testing_without_losing_drag_events() {
    #[derive(Resource, Default)]
    struct Drags(usize);

    let (mut app, window, _, _) = setup();
    app.init_resource::<Drags>().add_observer(
        |mut event: On<Pointer<super::Drag>>, mut drags: ResMut<Drags>| {
            event.propagate(false);
            drags.0 += 1;
        },
    );
    touch(&mut app, window, 1, TouchPhase::Started, 50.);
    app.update();
    app.world_mut().resource_mut::<BackendPasses>().0 = 0;
    for sample in 1..=64 {
        touch(&mut app, window, 1, TouchPhase::Moved, 50. + sample as f32);
    }
    touch(&mut app, window, 1, TouchPhase::Ended, 114.);
    app.update();
    assert_eq!(app.world().resource::<BackendPasses>().0, 1);
    assert_eq!(app.world().resource::<Drags>().0, 64);
}

#[test]
fn drag_release_without_a_move_uses_the_release_target() {
    #[derive(Resource, Default)]
    struct DragEvents {
        dropped: Vec<(Entity, Entity, Vec2)>,
        ended: Vec<(Entity, Vec2)>,
    }
    let (mut app, window, first, second) = setup();
    app.init_resource::<DragEvents>().add_observer(
        |mut event: On<Pointer<super::DragDrop>>, mut events: ResMut<DragEvents>| {
            event.propagate(false);
            events
                .dropped
                .push((event.entity, event.dropped, event.pointer_location.position));
        },
    );
    app.add_observer(
        |mut event: On<Pointer<super::DragEnd>>, mut events: ResMut<DragEvents>| {
            event.propagate(false);
            events.ended.push((event.entity, event.distance));
        },
    );
    touch(&mut app, window, 1, TouchPhase::Started, 50.);
    touch(&mut app, window, 1, TouchPhase::Moved, 60.);
    touch(&mut app, window, 1, TouchPhase::Ended, 250.);
    app.update();
    assert_eq!(
        app.world().resource::<DragEvents>().dropped,
        [(second, first, Vec2::new(250., 50.))]
    );
    assert_eq!(
        app.world().resource::<DragEvents>().ended,
        [(first, Vec2::new(200., 0.))]
    );
    let state = app
        .world()
        .resource::<PointerState>()
        .get(PointerId::Touch(1), PointerButton::Primary)
        .unwrap();
    assert!(
        state.pressing.is_empty() && state.dragging.is_empty() && state.dragging_over.is_empty()
    );
}

#[test]
fn same_frame_cancellation_is_ordered_after_press() {
    let (mut app, window, target, _) = setup();
    touch(&mut app, window, 1, TouchPhase::Started, 50.);
    touch(&mut app, window, 1, TouchPhase::Canceled, 50.);
    app.update();
    assert_eq!(
        app.world().resource::<Events>().0,
        [
            ("press", PointerId::Touch(1), target),
            ("cancel", PointerId::Touch(1), target)
        ]
    );
}

#[test]
fn mouse_buttons_keep_independent_press_targets() {
    let (mut app, window, first, second) = setup();
    let location = |x| Location {
        target: bevy_camera::RenderTarget::default()
            .normalize(Some(window))
            .unwrap(),
        position: Vec2::new(x, 50.),
    };
    for (x, action) in [
        (50., PointerAction::Press(PointerButton::Primary)),
        (250., PointerAction::Press(PointerButton::Secondary)),
        (250., PointerAction::Release(PointerButton::Primary)),
    ] {
        app.world_mut()
            .write_message(PointerInput::new(PointerId::Mouse, location(x), action));
    }
    app.update();
    let mouse = app
        .world_mut()
        .query::<(&PointerId, &PointerPress)>()
        .iter(app.world())
        .find(|(id, _)| **id == PointerId::Mouse)
        .unwrap()
        .1;
    assert!(!mouse.is_primary_pressed());
    assert!(mouse.is_secondary_pressed());
    app.world_mut().write_message(PointerInput::new(
        PointerId::Mouse,
        location(250.),
        PointerAction::Release(PointerButton::Secondary),
    ));
    app.update();
    assert_eq!(
        app.world().resource::<Events>().0,
        [
            ("press", PointerId::Mouse, first),
            ("press", PointerId::Mouse, second),
            ("release", PointerId::Mouse, second),
            ("release", PointerId::Mouse, first),
            ("click", PointerId::Mouse, second),
            ("release", PointerId::Mouse, second),
        ]
    );
}

#[test]
fn cancellation_reaches_original_press_even_after_leaving() {
    for move_out in [false, true] {
        let (mut app, window, target, _) = setup();
        touch(&mut app, window, 1, TouchPhase::Started, 50.);
        app.update();
        if move_out {
            touch(&mut app, window, 1, TouchPhase::Moved, 150.);
            app.update();
            app.update();
        }
        touch(
            &mut app,
            window,
            1,
            TouchPhase::Canceled,
            if move_out { 150. } else { 50. },
        );
        app.update();
        assert_eq!(
            app.world().resource::<Events>().0,
            [
                ("press", PointerId::Touch(1), target),
                ("cancel", PointerId::Touch(1), target),
            ]
        );
    }
}

#[test]
fn cancelling_one_touch_does_not_cancel_another() {
    let (mut app, window, first, second) = setup();
    touch(&mut app, window, 1, TouchPhase::Started, 50.);
    touch(&mut app, window, 2, TouchPhase::Started, 250.);
    app.update();
    touch(&mut app, window, 1, TouchPhase::Canceled, 50.);
    touch(&mut app, window, 2, TouchPhase::Ended, 250.);
    app.update();
    assert_eq!(
        app.world().resource::<Events>().0,
        [
            ("press", PointerId::Touch(1), first),
            ("press", PointerId::Touch(2), second),
            ("cancel", PointerId::Touch(1), first),
            ("click", PointerId::Touch(2), second),
            ("release", PointerId::Touch(2), second),
        ]
    );
}

#[test]
fn cancellation_deduplicates_targets_and_clears_all_buttons_on_a_retained_pointer() {
    let (mut app, window, first, second) = setup();
    let id = PointerId::Custom(uuid::Uuid::from_u128(1));
    let mut location = Location {
        target: bevy_camera::RenderTarget::default()
            .normalize(Some(window))
            .unwrap(),
        position: Vec2::splat(50.),
    };
    let pointer = app
        .world_mut()
        .spawn((id, PointerLocation::new(location.clone())))
        .id();
    for button in PointerButton::iter() {
        app.world_mut().write_message(PointerInput::new(
            id,
            location.clone(),
            PointerAction::Press(button),
        ));
    }
    app.update();
    let press = app.world().get::<PointerPress>(pointer).unwrap();
    assert!(
        press.is_primary_pressed() && press.is_secondary_pressed() && press.is_middle_pressed()
    );

    location.position.x = 250.;
    app.world_mut().write_message(PointerInput::new(
        id,
        location.clone(),
        PointerAction::Move {
            delta: Vec2::new(200., 0.),
        },
    ));
    app.update();
    app.world_mut().resource_mut::<Events>().0.clear();
    app.world_mut()
        .write_message(PointerInput::new(id, location, PointerAction::Cancel));
    app.update();

    assert_eq!(
        app.world().resource::<Events>().0,
        [("cancel", id, first), ("cancel", id, second)]
    );
    assert!(!app
        .world()
        .get::<PointerPress>(pointer)
        .unwrap()
        .is_any_pressed());
    assert!(app
        .world()
        .get::<PointerLocation>(pointer)
        .unwrap()
        .location
        .is_none());
    let state = app.world().resource::<PointerState>();
    for button in PointerButton::iter() {
        let state = state.get(id, button).unwrap();
        assert!(state.pressing.is_empty());
        assert!(state.dragging.is_empty());
        assert!(state.dragging_over.is_empty());
    }
    app.update();
    assert!(app
        .world()
        .resource::<crate::hover::HoverMap>()
        .get(&id)
        .unwrap()
        .is_empty());
}
