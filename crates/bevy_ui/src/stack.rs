//! This module contains the systems that update the stored UI nodes stack

#[cfg(feature = "ghost_nodes")]
use crate::experimental::GhostNode;
use crate::{
    experimental::{UiChildren, UiRootNodes},
    GlobalZIndex, Node, ZIndex,
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{entity::EntityHashSet, prelude::*, system::SystemParam};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use core::ops::Range;

/// The order of the node in the UI layout.
/// Nodes with a higher stack index are drawn on top of and receive interactions before nodes with lower stack indices.
///
/// Automatically calculated in [`UiSystems::Stack`](`super::UiSystems::Stack`).
#[derive(Component, Default, PartialEq, Eq, Deref, DerefMut, Reflect)]
#[reflect(Component, Default)]
pub struct ComputedStackIndex(pub u32);

/// The current UI stack, which contains all UI nodes ordered by their depth (back-to-front).
///
/// The first entry is the furthest node from the camera and is the first one to get rendered
/// while the last entry is the first node to receive interactions.
#[derive(Debug, Resource, Default, Reflect)]
#[reflect(Resource, Default)]
pub struct UiStack {
    /// Partition of the `uinodes` list into disjoint slices of nodes that all share the same camera target.
    pub partition: Vec<Range<usize>>,
    /// List of UI nodes ordered from back-to-front
    pub uinodes: Vec<Entity>,
}

#[derive(Default)]
pub(crate) struct ChildBufferCache {
    pub inner: Vec<Vec<(Entity, i32)>>,
}

impl ChildBufferCache {
    fn pop(&mut self) -> Vec<(Entity, i32)> {
        self.inner.pop().unwrap_or_default()
    }

    fn push(&mut self, vec: Vec<(Entity, i32)>) {
        self.inner.push(vec);
    }
}

#[doc(hidden)]
#[derive(SystemParam)]
pub struct UiStackChanges<'w, 's> {
    // Unfiltered hierarchy changes include entities entering or leaving the UI.
    changed: Query<
        'w,
        's,
        (),
        Or<(
            Added<Node>,
            Changed<Children>,
            Changed<ChildOf>,
            Changed<ZIndex>,
            Changed<GlobalZIndex>,
        )>,
    >,
    removed_nodes: RemovedComponents<'w, 's, Node>,
    removed_children: RemovedComponents<'w, 's, Children>,
    removed_parents: RemovedComponents<'w, 's, ChildOf>,
    #[cfg(feature = "ghost_nodes")]
    added_ghosts: Query<'w, 's, (), Added<GhostNode>>,
    #[cfg(feature = "ghost_nodes")]
    removed_ghosts: RemovedComponents<'w, 's, GhostNode>,
    removed_z: RemovedComponents<'w, 's, ZIndex>,
    removed_global_z: RemovedComponents<'w, 's, GlobalZIndex>,
    removed_index: RemovedComponents<'w, 's, ComputedStackIndex>,
    roots: Local<'s, Vec<(Entity, (i32, i32))>>,
}

/// Generates the render stack for UI nodes.
///
/// Create a list of root nodes from parentless entities and entities with a `GlobalZIndex` component.
/// Then build the `UiStack` from a walk of the existing layout trees starting from each root node,
/// filtering branches by `Without<GlobalZIndex>`so that we don't revisit nodes.
pub fn ui_stack_system(
    mut cache: Local<ChildBufferCache>,
    mut root_nodes: Local<Vec<(Entity, (i32, i32))>>,
    mut visited_root_nodes: Local<EntityHashSet>,
    mut ui_stack: ResMut<UiStack>,
    ui_root_nodes: UiRootNodes,
    root_node_query: Query<(Entity, Option<&GlobalZIndex>, Option<&ZIndex>)>,
    zindex_global_node_query: Query<
        (Entity, &GlobalZIndex, Option<&ZIndex>),
        With<ComputedStackIndex>,
    >,
    ui_children: UiChildren,
    zindex_query: Query<Option<&ZIndex>, (With<ComputedStackIndex>, Without<GlobalZIndex>)>,
    mut update_query: Query<&mut ComputedStackIndex>,
    mut changes: UiStackChanges,
) {
    visited_root_nodes.clear();

    for (id, maybe_global_zindex, maybe_zindex) in root_node_query.iter_many(ui_root_nodes.iter()) {
        root_nodes.push((
            id,
            (
                maybe_global_zindex.map(|zindex| zindex.0).unwrap_or(0),
                maybe_zindex.map(|zindex| zindex.0).unwrap_or(0),
            ),
        ));
        visited_root_nodes.insert(id);
    }

    for (id, global_zindex, maybe_zindex) in zindex_global_node_query.iter() {
        if visited_root_nodes.contains(&id) {
            continue;
        }

        root_nodes.push((
            id,
            (
                global_zindex.0,
                maybe_zindex.map(|zindex| zindex.0).unwrap_or(0),
            ),
        ));
    }

    // Query iteration order breaks equal-z root ties, and can change when an
    // unrelated component moves a root between archetypes. Compare the unsorted
    // candidates, rather than silently retaining a different paint/hit order.
    let dirty = !changes.changed.is_empty()
        || !changes.removed_nodes.is_empty()
        || !changes.removed_children.is_empty()
        || !changes.removed_parents.is_empty()
        || !changes.removed_z.is_empty()
        || !changes.removed_global_z.is_empty()
        || !changes.removed_index.is_empty()
        || ui_stack.is_changed()
        || *changes.roots != *root_nodes
        || update_query.iter_mut().any(|index| index.is_changed());
    changes.removed_nodes.clear();
    changes.removed_children.clear();
    changes.removed_parents.clear();
    changes.removed_z.clear();
    changes.removed_global_z.clear();
    changes.removed_index.clear();
    #[cfg(feature = "ghost_nodes")]
    let dirty = {
        let dirty = dirty || !changes.added_ghosts.is_empty() || !changes.removed_ghosts.is_empty();
        changes.removed_ghosts.clear();
        dirty
    };
    if !dirty {
        root_nodes.clear();
        return;
    }
    changes.roots.clone_from(&root_nodes);
    ui_stack.partition.clear();
    ui_stack.uinodes.clear();
    root_nodes.sort_by_key(|(_, z)| *z);

    for (root_entity, _) in root_nodes.drain(..) {
        let start = ui_stack.uinodes.len();
        update_uistack_recursive(
            &mut cache,
            root_entity,
            &ui_children,
            &zindex_query,
            &mut ui_stack.uinodes,
        );
        let end = ui_stack.uinodes.len();
        ui_stack.partition.push(start..end);
    }

    for (i, entity) in ui_stack.uinodes.iter().enumerate() {
        if let Ok(mut stack_index) = update_query.get_mut(*entity) {
            stack_index.set_if_neq(ComputedStackIndex(i as u32));
        }
    }
}

fn update_uistack_recursive(
    cache: &mut ChildBufferCache,
    node_entity: Entity,
    ui_children: &UiChildren,
    zindex_query: &Query<Option<&ZIndex>, (With<ComputedStackIndex>, Without<GlobalZIndex>)>,
    ui_stack: &mut Vec<Entity>,
) {
    ui_stack.push(node_entity);

    let mut child_buffer = cache.pop();
    child_buffer.extend(
        ui_children
            .iter_ui_children(node_entity)
            .filter_map(|child_entity| {
                zindex_query
                    .get(child_entity)
                    .ok()
                    .map(|zindex| (child_entity, zindex.map(|zindex| zindex.0).unwrap_or(0)))
            }),
    );
    child_buffer.sort_by_key(|k| k.1);
    for (child_entity, _) in child_buffer.drain(..) {
        update_uistack_recursive(cache, child_entity, ui_children, zindex_query, ui_stack);
    }
    cache.push(child_buffer);
}

#[cfg(test)]
mod tests {
    use bevy_ecs::{
        component::Component,
        schedule::Schedule,
        system::Commands,
        world::{CommandQueue, World},
    };

    use crate::{GlobalZIndex, Node, UiStack, ZIndex};

    use super::ui_stack_system;

    #[derive(Component, PartialEq, Debug, Clone)]
    struct Label(&'static str);

    #[test]
    fn cached_stack_tracks_order_membership_and_removals() {
        use crate::{ComputedStackIndex, UiTransform, Val2};
        use bevy_ecs::prelude::*;
        let mut world = World::new();
        world.init_resource::<UiStack>();
        let root = world.spawn(Node::default()).id();
        let a = world.spawn(Node::default()).id();
        let b = world.spawn(Node::default()).id();
        world.entity_mut(root).add_children(&[a, b]);
        let mut schedule = Schedule::default();
        schedule.add_systems(ui_stack_system);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, a, b]);
        let tick = world.get_resource_ref::<UiStack>().unwrap().last_changed();
        world.get_mut::<UiTransform>(a).unwrap().translation = Val2::px(5., 2.);
        schedule.run(&mut world);
        assert_eq!(
            world.get_resource_ref::<UiStack>().unwrap().last_changed(),
            tick
        );

        world.entity_mut(a).insert(ZIndex(1));
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, b, a]);
        world.get_mut::<ZIndex>(a).unwrap().0 = -1;
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, a, b]);
        world.entity_mut(a).remove::<ZIndex>();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, a, b]);
        world.entity_mut(root).replace_children(&[b, a]);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, b, a]);
        world.entity_mut(a).insert(GlobalZIndex(-1));
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [a, root, b]);
        assert_eq!(world.resource::<UiStack>().partition, [0..1, 1..3]);
        world.get_mut::<GlobalZIndex>(a).unwrap().0 = 1;
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, b, a]);
        assert_eq!(world.resource::<UiStack>().partition, [0..2, 2..3]);
        world.entity_mut(a).remove::<GlobalZIndex>();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, b, a]);
        world.entity_mut(a).remove::<ComputedStackIndex>();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, b]);
        world.entity_mut(a).insert(ComputedStackIndex::default());
        world.entity_mut(b).despawn();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, a]);
    }

    #[test]
    fn cached_stack_preserves_query_order_ties_and_root_promotion() {
        use bevy_ecs::{prelude::*, system::RunSystemOnce};
        let mut world = World::new();
        world.init_resource::<UiStack>();
        let a = world.spawn(Node::default()).id();
        let b = world.spawn(Node::default()).id();
        let c = world.spawn(Node::default()).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(ui_stack_system);
        schedule.run(&mut world);
        // Moving an equal-z root to another archetype must match a fresh build.
        world.entity_mut(a).insert(Label("moved"));
        schedule.run(&mut world);
        let cached = world.resource::<UiStack>().uinodes.clone();
        world.run_system_once(ui_stack_system).unwrap();
        assert_eq!(world.resource::<UiStack>().uinodes, cached);
        world.entity_mut(a).add_children(&[b, c]);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [a, b, c]);
        world.entity_mut(b).remove::<ChildOf>();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().partition.len(), 2);
        assert!(world.resource::<UiStack>().uinodes.contains(&b));
        world.entity_mut(c).insert(GlobalZIndex(3));
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes.last(), Some(&c));
    }

    #[cfg(feature = "ghost_nodes")]
    #[test]
    fn cached_stack_tracks_ghost_root_changes() {
        use crate::experimental::GhostNode;
        let mut world = World::new();
        world.init_resource::<UiStack>();
        let child = world.spawn(Node::default()).id();
        let ghost = world.spawn(GhostNode).add_child(child).id();
        let root = world.spawn(Node::default()).id();
        let mut schedule = Schedule::default();
        schedule.add_systems(ui_stack_system);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().partition.len(), 2);
        world.entity_mut(root).add_child(ghost);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, child]);
        world.entity_mut(ghost).remove::<GhostNode>();
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root]);
        world.entity_mut(ghost).insert(GhostNode);
        schedule.run(&mut world);
        assert_eq!(world.resource::<UiStack>().uinodes, [root, child]);
    }

    fn node_with_global_and_local_zindex(
        name: &'static str,
        global_zindex: i32,
        local_zindex: i32,
    ) -> (Label, Node, GlobalZIndex, ZIndex) {
        (
            Label(name),
            Node::default(),
            GlobalZIndex(global_zindex),
            ZIndex(local_zindex),
        )
    }

    fn node_with_global_zindex(
        name: &'static str,
        global_zindex: i32,
    ) -> (Label, Node, GlobalZIndex) {
        (Label(name), Node::default(), GlobalZIndex(global_zindex))
    }

    fn node_with_zindex(name: &'static str, zindex: i32) -> (Label, Node, ZIndex) {
        (Label(name), Node::default(), ZIndex(zindex))
    }

    fn node_without_zindex(name: &'static str) -> (Label, Node) {
        (Label(name), Node::default())
    }

    /// Tests the UI Stack system.
    ///
    /// This tests for siblings default ordering according to their insertion order, but it
    /// can't test the same thing for UI roots. UI roots having no parents, they do not have
    /// a stable ordering that we can test against. If we test it, it may pass now and start
    /// failing randomly in the future because of some unrelated `bevy_ecs` change.
    #[test]
    fn test_ui_stack_system() {
        let mut world = World::default();
        world.init_resource::<UiStack>();

        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, &world);
        commands.spawn(node_with_global_zindex("0", 2));

        commands
            .spawn(node_with_zindex("1", 1))
            .with_children(|parent| {
                parent
                    .spawn(node_without_zindex("1-0"))
                    .with_children(|parent| {
                        parent.spawn(node_without_zindex("1-0-0"));
                        parent.spawn(node_without_zindex("1-0-1"));
                        parent.spawn(node_with_zindex("1-0-2", -1));
                    });
                parent.spawn(node_without_zindex("1-1"));
                parent
                    .spawn(node_with_global_zindex("1-2", -1))
                    .with_children(|parent| {
                        parent.spawn(node_without_zindex("1-2-0"));
                        parent.spawn(node_with_global_zindex("1-2-1", -3));
                        parent
                            .spawn(node_without_zindex("1-2-2"))
                            .with_children(|_| ());
                        parent.spawn(node_without_zindex("1-2-3"));
                    });
                parent.spawn(node_without_zindex("1-3"));
            });

        commands
            .spawn(node_without_zindex("2"))
            .with_children(|parent| {
                parent
                    .spawn(node_without_zindex("2-0"))
                    .with_children(|_parent| ());
                parent
                    .spawn(node_without_zindex("2-1"))
                    .with_children(|parent| {
                        parent.spawn(node_without_zindex("2-1-0"));
                    });
            });

        commands.spawn(node_with_global_zindex("3", -2));

        queue.apply(&mut world);

        let mut schedule = Schedule::default();
        schedule.add_systems(ui_stack_system);
        schedule.run(&mut world);

        let mut query = world.query::<&Label>();
        let ui_stack = world.resource::<UiStack>();
        let actual_result = ui_stack
            .uinodes
            .iter()
            .map(|entity| query.get(&world, *entity).unwrap().clone())
            .collect::<Vec<_>>();
        let expected_result = vec![
            (Label("1-2-1")), // GlobalZIndex(-3)
            (Label("3")),     // GlobalZIndex(-2)
            (Label("1-2")),   // GlobalZIndex(-1)
            (Label("1-2-0")),
            (Label("1-2-2")),
            (Label("1-2-3")),
            (Label("2")),
            (Label("2-0")),
            (Label("2-1")),
            (Label("2-1-0")),
            (Label("1")), // ZIndex(1)
            (Label("1-0")),
            (Label("1-0-2")), // ZIndex(-1)
            (Label("1-0-0")),
            (Label("1-0-1")),
            (Label("1-1")),
            (Label("1-3")),
            (Label("0")), // GlobalZIndex(2)
        ];
        assert_eq!(actual_result, expected_result);

        // Test partitioning
        let last_part = ui_stack.partition.last().unwrap();
        assert_eq!(last_part.len(), 1);
        let last_entity = ui_stack.uinodes[last_part.start];
        assert_eq!(*query.get(&world, last_entity).unwrap(), Label("0"));

        let actual_result = ui_stack.uinodes[ui_stack.partition[4].clone()]
            .iter()
            .map(|entity| query.get(&world, *entity).unwrap().clone())
            .collect::<Vec<_>>();
        let expected_result = vec![
            (Label("1")), // ZIndex(1)
            (Label("1-0")),
            (Label("1-0-2")), // ZIndex(-1)
            (Label("1-0-0")),
            (Label("1-0-1")),
            (Label("1-1")),
            (Label("1-3")),
        ];
        assert_eq!(actual_result, expected_result);
    }

    #[test]
    fn test_with_equal_global_zindex_zindex_decides_order() {
        let mut world = World::default();
        world.init_resource::<UiStack>();

        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, &world);
        commands.spawn(node_with_global_and_local_zindex("0", -1, 1));
        commands.spawn(node_with_global_and_local_zindex("1", -1, 2));
        commands.spawn(node_with_global_and_local_zindex("2", 1, 3));
        commands.spawn(node_with_global_and_local_zindex("3", 1, -3));
        commands
            .spawn(node_without_zindex("4"))
            .with_children(|builder| {
                builder.spawn(node_with_global_and_local_zindex("5", 0, -1));
                builder.spawn(node_with_global_and_local_zindex("6", 0, 1));
                builder.spawn(node_with_global_and_local_zindex("7", -1, -1));
                builder.spawn(node_with_global_zindex("8", 1));
            });

        queue.apply(&mut world);

        let mut schedule = Schedule::default();
        schedule.add_systems(ui_stack_system);
        schedule.run(&mut world);

        let mut query = world.query::<&Label>();
        let ui_stack = world.resource::<UiStack>();
        let actual_result = ui_stack
            .uinodes
            .iter()
            .map(|entity| query.get(&world, *entity).unwrap().clone())
            .collect::<Vec<_>>();

        let expected_result = vec![
            (Label("7")),
            (Label("0")),
            (Label("1")),
            (Label("5")),
            (Label("4")),
            (Label("6")),
            (Label("3")),
            (Label("8")),
            (Label("2")),
        ];

        assert_eq!(actual_result, expected_result);

        assert_eq!(ui_stack.partition.len(), expected_result.len());
        for (i, part) in ui_stack.partition.iter().enumerate() {
            assert_eq!(*part, i..i + 1);
        }
    }
}
