//! This module contains systems that update the UI when something changes

use crate::{
    experimental::{UiChildren, UiRootNodes},
    ui_transform::UiGlobalTransform,
    CalculatedClip, ComputedUiRenderTargetInfo, ComputedUiTargetCamera, DefaultUiCamera, Display,
    Node, OverrideClip, UiScale, UiTargetCamera,
};

use super::ComputedNode;
use bevy_app::Propagate;
use bevy_camera::Camera;
use bevy_ecs::{
    entity::Entity,
    query::{Has, Or, With},
    system::{Commands, Query, Res},
};
use bevy_math::UVec2;

/// Updates clipping for all nodes
pub fn update_clipping_system(
    mut commands: Commands,
    root_nodes: UiRootNodes,
    mut node_query: Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Has<OverrideClip>,
    )>,
    ui_children: UiChildren,
) {
    for root_node in root_nodes.iter() {
        update_clipping(
            &mut commands,
            &ui_children,
            &mut node_query,
            root_node,
            None,
        );
    }
}

fn update_clipping(
    commands: &mut Commands,
    ui_children: &UiChildren,
    node_query: &mut Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Has<OverrideClip>,
    )>,
    entity: Entity,
    maybe_inherited_clip: Option<&CalculatedClip>,
) {
    let Ok((node, computed_node, transform, maybe_calculated_clip, has_override_clip)) =
        node_query.get_mut(entity)
    else {
        return;
    };

    // Hidden nodes clip themselves even when they override ancestor clipping.
    let fully_clipped = CalculatedClip::FullyClipped;
    let maybe_inherited_clip = if node.display == Display::None {
        Some(&fully_clipped)
    } else if has_override_clip {
        None
    } else {
        maybe_inherited_clip
    };

    // Update this node's CalculatedClip component
    if let Some(mut calculated_clip) = maybe_calculated_clip {
        if let Some(inherited_clip) = maybe_inherited_clip {
            // Replace the previous calculated clip with the inherited clipping rect
            if *calculated_clip != *inherited_clip {
                *calculated_clip = inherited_clip.clone();
            }
        } else {
            // No inherited clipping rect, remove the component
            commands.entity(entity).remove::<CalculatedClip>();
        }
    } else if let Some(inherited_clip) = maybe_inherited_clip {
        // No previous calculated clip, add a new CalculatedClip component with the inherited clipping rect
        commands.entity(entity).try_insert(inherited_clip.clone());
    }

    // Borrow unchanged inheritance through visible nodes. Only a clipping node
    // needs an owned list to append its rectangle; siblings share that list.
    let owned_children_clip;
    let children_clip = if maybe_inherited_clip.is_some_and(CalculatedClip::is_fully_clipped)
        || node.overflow.is_visible()
    {
        // The current node doesn't clip, propagate the optional inherited clipping rect to any children
        maybe_inherited_clip
    } else if let Some(clip_from_world) = transform.try_inverse() {
        let mut clip = maybe_inherited_clip.cloned().unwrap_or_default();
        clip.push_rect(
            computed_node.resolve_clip_rect(node.overflow, node.overflow_clip_margin),
            clip_from_world,
        );
        owned_children_clip = clip;
        Some(&owned_children_clip)
    } else {
        Some(&fully_clipped)
    };

    for child in ui_children.iter_ui_children(entity) {
        update_clipping(commands, ui_children, node_query, child, children_clip);
    }
}

pub fn propagate_ui_target_cameras(
    mut commands: Commands,
    default_ui_camera: DefaultUiCamera,
    ui_scale: Res<UiScale>,
    camera_query: Query<&Camera>,
    target_camera_query: Query<&UiTargetCamera>,
    ui_root_nodes: UiRootNodes,
    ui_children: UiChildren,
    propagate_query: Query<
        Entity,
        Or<(
            With<Propagate<ComputedUiTargetCamera>>,
            With<Propagate<ComputedUiRenderTargetInfo>>,
        )>,
    >,
) {
    let default_camera_entity = default_ui_camera.get();

    for entity in propagate_query.iter() {
        if ui_children.get_parent(entity).is_some() {
            commands.entity(entity).remove::<(
                Propagate<ComputedUiTargetCamera>,
                Propagate<ComputedUiRenderTargetInfo>,
            )>();
        }
    }

    for root_entity in ui_root_nodes.iter() {
        let camera = target_camera_query
            .get(root_entity)
            .ok()
            .map(UiTargetCamera::entity)
            .or(default_camera_entity)
            .unwrap_or(Entity::PLACEHOLDER);

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiTargetCamera { camera }));

        let (scale_factor, physical_size) = camera_query
            .get(camera)
            .ok()
            .map(|camera| {
                (
                    camera.target_scaling_factor().unwrap_or(1.) * ui_scale.0,
                    camera.physical_viewport_size().unwrap_or(UVec2::ZERO),
                )
            })
            .unwrap_or((1., UVec2::ZERO));

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiRenderTargetInfo {
                scale_factor,
                physical_size,
            }));
    }
}

#[cfg(test)]
mod tests {
    use crate::update::propagate_ui_target_cameras;
    use crate::ComputedUiRenderTargetInfo;
    use crate::ComputedUiTargetCamera;
    use crate::IsDefaultUiCamera;
    use crate::Node;
    use crate::UiScale;
    use crate::UiTargetCamera;
    use bevy_app::App;
    use bevy_app::HierarchyPropagatePlugin;
    use bevy_app::PostUpdate;
    use bevy_app::PropagateSet;
    use bevy_camera::Camera;
    use bevy_camera::Camera2d;
    use bevy_camera::ComputedCameraValues;
    use bevy_camera::RenderTargetInfo;
    use bevy_ecs::hierarchy::ChildOf;
    use bevy_math::UVec2;
    use bevy_utils::default;

    #[test]
    fn clipping_preserves_heap_backed_inheritance_across_siblings() {
        use super::update_clipping_system;
        use crate::{CalculatedClip, ComputedNode, Overflow, OverrideClip};
        use bevy_ecs::prelude::*;
        use bevy_math::Vec2;

        let mut world = World::new();
        let mut ancestors: Vec<Entity> = Vec::new();
        for size in [100., 80., 60.] {
            let entity = world
                .spawn((
                    Node {
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    ComputedNode {
                        size: Vec2::splat(size),
                        ..default()
                    },
                ))
                .id();
            if let Some(&parent) = ancestors.last() {
                world.entity_mut(parent).add_child(entity);
            }
            ancestors.push(entity);
        }
        let parent = *ancestors.last().unwrap();
        let first = world.spawn((Node::default(), ChildOf(parent))).id();
        let second = world.spawn((Node::default(), ChildOf(parent))).id();
        let override_node = world
            .spawn((Node::default(), OverrideClip, ChildOf(parent)))
            .id();
        let override_child = world.spawn((Node::default(), ChildOf(override_node))).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(update_clipping_system);

        for _ in 0..3 {
            schedule.run(&mut world);
            let first_clip = world.get::<CalculatedClip>(first).unwrap();
            let widths: Vec<_> = first_clip
                .rects()
                .unwrap()
                .iter()
                .map(|r| r.rect.width())
                .collect();
            assert_eq!(widths, [100., 80., 60.]);
            assert_eq!(Some(first_clip), world.get::<CalculatedClip>(second));
            assert_eq!(
                world
                    .get::<CalculatedClip>(parent)
                    .unwrap()
                    .rects()
                    .unwrap()
                    .len(),
                2
            );
            assert!(world.get::<CalculatedClip>(ancestors[0]).is_none());
            assert!(world.get::<CalculatedClip>(override_node).is_none());
            assert!(world.get::<CalculatedClip>(override_child).is_none());
        }
    }

    #[test]
    fn clipping_tracks_bypassed_geometry_and_inherited_inputs() {
        use super::update_clipping_system;
        use crate::{
            CalculatedClip, ComputedNode, Display, Overflow, OverflowClipMargin, OverrideClip,
            UiGlobalTransform,
        };
        use bevy_ecs::{change_detection::DetectChangesMut, prelude::*};
        use bevy_math::{Affine2, Vec2};
        let mut world = World::new();
        let root = world
            .spawn((
                Node {
                    overflow: Overflow::clip(),
                    overflow_clip_margin: OverflowClipMargin::content_box().with_margin(2.),
                    ..default()
                },
                ComputedNode {
                    size: Vec2::splat(100.),
                    ..default()
                },
            ))
            .id();
        let child = world.spawn(Node::default()).id();
        let grandchild = world.spawn(Node::default()).id();
        world.entity_mut(root).add_child(child);
        world.entity_mut(child).add_child(grandchild);
        let mut schedule = Schedule::default();
        schedule.add_systems(update_clipping_system);
        schedule.run(&mut world);
        schedule.run(&mut world); // Observe deferred CalculatedClip insertion.
        assert_eq!(
            world.get::<CalculatedClip>(child),
            world.get::<CalculatedClip>(grandchild)
        );

        // Clipping must observe geometry fields that deliberately bypass ticks.
        for field in 0..6 {
            let before = world.get::<CalculatedClip>(grandchild).unwrap().clone();
            let mut node = world.get_mut::<ComputedNode>(root).unwrap();
            let node = node.bypass_change_detection();
            match field {
                0 => node.border.min_inset.x += 3.,
                1 => node.padding.min_inset.y += 4.,
                2 => node.scrollbar_size.x += 5.,
                3 => node.size.x += 20.,
                4 => node.inverse_scale_factor = 0.5,
                5 => node.border.max_inset.y += 6.,
                _ => unreachable!(),
            }
            schedule.run(&mut world);
            let after = world.get::<CalculatedClip>(grandchild).unwrap();
            assert_ne!(*after, before, "geometry field {field}");
            assert_eq!(Some(after), world.get::<CalculatedClip>(child));
        }

        let before = world.get::<CalculatedClip>(child).unwrap().clone();
        world
            .entity_mut(root)
            .insert(UiGlobalTransform::from(Affine2::from_translation(
                Vec2::new(9., 7.),
            )));
        schedule.run(&mut world);
        assert_ne!(*world.get::<CalculatedClip>(child).unwrap(), before);
        let before = world.get::<CalculatedClip>(child).unwrap().clone();
        world.get_mut::<Node>(root).unwrap().overflow_clip_margin =
            OverflowClipMargin::border_box();
        schedule.run(&mut world);
        assert_ne!(*world.get::<CalculatedClip>(child).unwrap(), before);

        world.entity_mut(child).insert(OverrideClip);
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(child).is_none());
        assert!(world.get::<CalculatedClip>(grandchild).is_none());
        world.entity_mut(child).remove::<OverrideClip>();
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(grandchild).is_some());
        world.get_mut::<Node>(root).unwrap().display = Display::None;
        schedule.run(&mut world);
        assert_eq!(
            world.get::<CalculatedClip>(grandchild),
            Some(&CalculatedClip::FullyClipped)
        );
        world.get_mut::<Node>(root).unwrap().display = Display::Flex;
        schedule.run(&mut world);
        assert!(!world
            .get::<CalculatedClip>(grandchild)
            .unwrap()
            .is_fully_clipped());
        world.get_mut::<Node>(root).unwrap().overflow = Overflow::visible();
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(grandchild).is_none());
        world.get_mut::<Node>(root).unwrap().overflow = Overflow::clip();
        schedule.run(&mut world);
        world.entity_mut(child).remove::<ChildOf>();
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(child).is_none());
        assert!(world.get::<CalculatedClip>(grandchild).is_none());
        world.entity_mut(root).add_child(child);
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(grandchild).is_some());
        world.entity_mut(grandchild).remove::<CalculatedClip>();
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(grandchild).is_some());
    }

    #[cfg(feature = "ghost_nodes")]
    #[test]
    fn clipping_tracks_ghost_root_promotion() {
        use super::update_clipping_system;
        use crate::{experimental::GhostNode, CalculatedClip, Display};
        use bevy_ecs::prelude::*;
        let mut world = World::new();
        let child = world.spawn(Node::default()).id();
        let ghost = world.spawn(GhostNode).add_child(child).id();
        let root = world
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .add_child(ghost)
            .id();
        let mut schedule = Schedule::default();
        schedule.add_systems(update_clipping_system);
        schedule.run(&mut world);
        assert_eq!(
            world.get::<CalculatedClip>(child),
            Some(&CalculatedClip::FullyClipped)
        );
        world.entity_mut(ghost).remove::<ChildOf>();
        schedule.run(&mut world);
        assert!(world.get::<CalculatedClip>(child).is_none());
        world.entity_mut(root).add_child(ghost);
        schedule.run(&mut world);
        assert_eq!(
            world.get::<CalculatedClip>(child),
            Some(&CalculatedClip::FullyClipped)
        );
    }

    fn setup_test_app() -> App {
        let mut app = App::new();

        app.init_resource::<UiScale>();

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiTargetCamera>::default(),
        );

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiRenderTargetInfo>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiRenderTargetInfo>::default(),
        );

        app.add_systems(bevy_app::Update, propagate_ui_target_cameras);

        app
    }

    #[test]
    fn update_context_for_single_ui_root() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale_factor = 10.;
        let physical_size = UVec2::new(1000, 500);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size,
                            scale_factor,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
            ComputedUiTargetCamera { camera }
        );

        assert_eq!(
            *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
            ComputedUiRenderTargetInfo {
                physical_size,
                scale_factor,
            }
        );
    }

    #[test]
    fn update_multiple_context_for_multiple_ui_roots() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode1a = world.spawn(Node::default()).id();
        let uinode2a = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2b = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2c = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode1b = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        for (uinode, camera, scale_factor, physical_size) in [
            (uinode1a, camera1, scale1, size1),
            (uinode1b, camera1, scale1, size1),
            (uinode2a, camera2, scale2, size2),
            (uinode2b, camera2, scale2, size2),
            (uinode2c, camera2, scale2, size2),
        ] {
            assert_eq!(
                *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
                ComputedUiTargetCamera { camera }
            );

            assert_eq!(
                *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
                ComputedUiRenderTargetInfo {
                    physical_size,
                    scale_factor,
                }
            );
        }
    }

    #[test]
    fn update_context_on_changed_camera() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        world.entity_mut(uinode).insert(UiTargetCamera(camera2));

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );
    }

    #[test]
    fn update_context_after_parented() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let camera1 = world.spawn((Camera2d, IsDefaultUiCamera)).id();
        let camera2 = world.spawn(Camera2d).id();
        let parent = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let child = world.spawn(Node::default()).id();

        app.update();

        assert_eq!(
            app.world()
                .get::<ComputedUiTargetCamera>(child)
                .unwrap()
                .get(),
            Some(camera1)
        );

        app.world_mut().entity_mut(parent).add_child(child);
        app.update();

        assert_eq!(
            app.world()
                .get::<ComputedUiTargetCamera>(child)
                .unwrap()
                .get(),
            Some(camera2)
        );
    }

    #[test]
    fn update_context_after_parent_removed() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        // `UiTargetCamera` is ignored on non-root UI nodes
        let uinode1 = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2 = world.spawn(Node::default()).add_child(uinode1).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        // Now `uinode1` is a root UI node its `UiTargetCamera` component will be used and its camera target set to `camera2`.
        world.entity_mut(uinode1).remove::<ChildOf>();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );
    }

    #[test]
    fn update_great_grandchild() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale = 1.;
        let size = UVec2::new(100, 100);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size,
                            scale_factor: scale,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();
        world.spawn(Node::default()).with_children(|builder| {
            builder.spawn(Node::default()).with_children(|builder| {
                builder.spawn(Node::default()).add_child(uinode);
            });
        });

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera
        );

        world.resource_mut::<UiScale>().0 = 2.;

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor(),
            2.
        );
    }
}
