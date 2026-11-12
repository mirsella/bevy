use crate::{
    experimental::UiChildren,
    prelude::{Button, Label},
    ui_transform::UiGlobalTransform,
    widget::{ImageNode, TextUiReader},
    ComputedNode, UiSystems,
};
use bevy_a11y::{AccessibilityNode, AccessibilityRequested, AccessibilitySystems};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::{
    change_detection::{DetectChanges, Ref},
    component::Component,
    entity_disabling::Disabled,
    hierarchy::{ChildOf, Children},
    lifecycle::{HookContext, RemovedComponents},
    prelude::{Entity, Mut},
    query::{Added, Changed, With, Without},
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Local, Query, Res, SystemParam},
    world::DeferredWorld,
};
use bevy_reflect::prelude::ReflectDefault;

use accesskit::{Affine, Node, Rect, Role};
use bevy_reflect::Reflect;

fn calc_label(
    text_reader: &mut TextUiReader,
    children: impl Iterator<Item = Entity>,
) -> Option<Box<str>> {
    let mut name = None;
    for child in children {
        let values = text_reader
            .iter(child)
            .map(|(_, _, text, _, _, _, _)| text.into())
            .collect::<Vec<String>>();
        if !values.is_empty() {
            name = Some(values.join(" "));
        }
    }
    name.map(String::into_boxed_str)
}

/// Changes that move nodes into another coordinate space without touching them.
#[derive(SystemParam)]
struct AnchorChanges<'w, 's> {
    reparented: Query<'w, 's, Entity, Changed<ChildOf>>,
    accessible: Query<'w, 's, (), With<AccessibilityNode>>,
    children: Query<'w, 's, &'static Children>,
    orphaned: RemovedComponents<'w, 's, ChildOf>,
    removed: RemovedComponents<'w, 's, AccessibilityNode>,
    disabled: Query<'w, 's, (), Added<Disabled>>,
    enabled: RemovedComponents<'w, 's, Disabled>,
}

impl AnchorChanges<'_, '_> {
    fn any(&mut self) -> bool {
        // Moving an entity only matters when it is a node or carries some, and
        // any kind of entity can sit between a node and its accessible ancestor.
        let (accessible, children) = (&self.accessible, &self.children);
        let carries_nodes = |entity| {
            accessible.contains(entity)
                || children
                    .iter_descendants(entity)
                    .any(|descendant| accessible.contains(descendant))
        };
        // `count` drains every reader so that none holds a removal for a later frame.
        let left = self.orphaned.read().filter(|&e| carries_nodes(e)).count()
            + self.removed.read().count()
            + self.enabled.read().count();
        left > 0 || !self.disabled.is_empty() || self.reparented.iter().any(carries_nodes)
    }
}

fn sync_bounds_and_transforms(
    requested: Res<AccessibilityRequested>,
    mut paused: Local<bool>,
    mut accessible_nodes_query: Query<(
        Entity,
        &mut AccessibilityNode,
        Option<Ref<ComputedNode>>,
        Ref<UiGlobalTransform>,
    )>,
    accessible_transform_query: Query<&UiGlobalTransform, With<AccessibilityNode>>,
    semantic_only_nodes: Query<Ref<AccessibilityNode>, Without<UiGlobalTransform>>,
    parents: Query<&ChildOf>,
    mut anchors: AnchorChanges,
) {
    if !requested.get() {
        // Nothing reads the tree, and what changes meanwhile goes untracked.
        *paused = true;
        return;
    }
    // `|` rather than `||`, so that `anchors` drains its readers on every frame.
    let mut all = core::mem::take(&mut *paused)
        | anchors.any()
        | semantic_only_nodes.iter().any(|node| node.is_changed());
    // A node is placed relative to its nearest accessible ancestor, so it only
    // needs placing again when either of them changed.
    let mut stale = Vec::new();
    for (entity, accessible, node, ui_transform) in &mut accessible_nodes_query {
        // A new node can become the nearest accessible ancestor of any other.
        all |= accessible.is_added();
        if all {
            break;
        }
        if accessible.is_changed()
            || ui_transform.is_changed()
            || node.is_some_and(|node| node.is_changed())
        {
            stale.push(entity);
        }
        if ui_transform.is_changed() {
            stale.extend(anchors.children.iter_descendants(entity));
        }
    }
    if all {
        stale = accessible_nodes_query
            .iter()
            .map(|(entity, ..)| entity)
            .collect();
    } else {
        // A moved panel lists each node once per moved ancestor.
        stale.sort_unstable();
        stale.dedup();
    }

    for entity in stale {
        let Ok((_, mut accessible, Some(node), ui_transform)) =
            accessible_nodes_query.get_mut(entity)
        else {
            continue;
        };
        // Semantic-only ancestors inherit their parent's UI coordinate space,
        // with any explicit AccessKit transforms composed along the way.
        let mut parent_transform = Affine::IDENTITY;
        for parent in parents.iter_ancestors(entity) {
            if let Ok(transform) = accessible_transform_query.get(parent) {
                parent_transform = Affine::new(transform.affine().to_cols_array().map(f64::from))
                    * parent_transform;
                break;
            }
            if let Ok(node) = semantic_only_nodes.get(parent)
                && let Some(transform) = node.transform()
            {
                parent_transform = *transform * parent_transform;
            }
        }
        let bounds = Rect::new(
            -0.5 * node.size.x as f64,
            -0.5 * node.size.y as f64,
            0.5 * node.size.x as f64,
            0.5 * node.size.y as f64,
        );

        let transform = parent_transform.inverse()
            * Affine::new(ui_transform.affine().to_cols_array().map(f64::from));

        // Collapsed coordinate spaces have no usable relative transform.
        let transform =
            (transform.is_finite() && transform != Affine::IDENTITY).then_some(transform);
        if accessible.bounds() != Some(bounds) {
            accessible.set_bounds(bounds);
        }
        if accessible.transform() != transform.as_ref() {
            if let Some(transform) = transform {
                accessible.set_transform(transform);
            } else {
                accessible.clear_transform();
            }
        }
    }
}

/// Gives `entity` the role and label, flagging an existing node only when they differ.
///
/// The components that trigger this also change for reasons unrelated to semantics,
/// such as an animated image color, and each flagged node sends the whole tree again.
fn set_role_and_label(
    commands: &mut Commands,
    entity: Entity,
    accessible: Option<Mut<AccessibilityNode>>,
    role: Role,
    label: Option<Box<str>>,
) {
    let Some(mut accessible) = accessible else {
        let mut node = Node::new(role);
        if let Some(label) = label {
            node.set_label(label);
        }
        commands
            .entity(entity)
            .try_insert(AccessibilityNode::from(node));
        return;
    };
    if accessible.role() == role && accessible.label() == label.as_deref() {
        return;
    }
    accessible.set_role(role);
    if let Some(label) = label {
        accessible.set_label(label);
    } else {
        accessible.clear_label();
    }
}

fn button_changed(
    mut commands: Commands,
    mut query: Query<(Entity, Option<&mut AccessibilityNode>), Changed<Button>>,
    ui_children: UiChildren,
    mut text_reader: TextUiReader,
) {
    for (entity, accessible) in &mut query {
        let label = calc_label(&mut text_reader, ui_children.iter_ui_children(entity));
        set_role_and_label(&mut commands, entity, accessible, Role::Button, label);
    }
}

fn image_changed(
    mut commands: Commands,
    mut query: Query<
        (Entity, Option<&mut AccessibilityNode>),
        (Changed<ImageNode>, Without<Button>),
    >,
    ui_children: UiChildren,
    mut text_reader: TextUiReader,
) {
    for (entity, accessible) in &mut query {
        let label = calc_label(&mut text_reader, ui_children.iter_ui_children(entity));
        set_role_and_label(&mut commands, entity, accessible, Role::Image, label);
    }
}

fn label_changed(
    mut commands: Commands,
    mut query: Query<(Entity, Option<&mut AccessibilityNode>), Changed<Label>>,
    mut text_reader: TextUiReader,
) {
    for (entity, accessible) in &mut query {
        let values = text_reader
            .iter(entity)
            .map(|(_, _, text, _, _, _, _)| text.into())
            .collect::<Vec<String>>();
        let label = Some(values.join(" ").into_boxed_str());
        if let Some(mut accessible) = accessible {
            accessible.set_role(Role::Label);
            if let Some(label) = label {
                accessible.set_value(label);
            } else {
                accessible.clear_value();
            }
        } else {
            let mut node = Node::new(Role::Label);
            if let Some(label) = label {
                node.set_value(label);
            }
            commands
                .entity(entity)
                .try_insert(AccessibilityNode::from(node));
        }
    }
}

/// A component which permits the a11y label to be specified independently from other a11y
/// attributes.
///
/// The content of the `label` attribute is typically application-specific, and frequently
/// originates in application code rather than library code. Because the primary mechanism of entity
/// composition in Bevy is component insertion (especially in BSN scenes), and because ``accesskit``
/// mandates that all a11y properties be stored in a single data structure, it can be cumbersome
/// to combine together a11y properties coming from different parts of the code; making the label
/// its own component makes it possible to specify the label as a mixin.
///
/// Internally, what this does is update the [`AccessibilityNode`] component, using component hooks
/// which are automatically registered when this component is used.
#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
#[require(AccessibilityNode)]
#[component(immutable, on_insert = on_label_inserted, on_remove = on_label_removed)]
pub struct AccessibleLabel(pub String);

impl AccessibleLabel {
    /// Makes a new [`AccessibleLabel`] component.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }
}

fn on_label_inserted(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    if let Some(label) = world.get::<AccessibleLabel>(entity) {
        let label_text = label.0.clone().into_boxed_str();
        if let Some(mut accessible) = world.get_mut::<AccessibilityNode>(entity) {
            accessible.set_label(label_text);
        }
    }
}

fn on_label_removed(mut world: DeferredWorld, HookContext { entity, .. }: HookContext) {
    if let Some(mut accessible) = world.get_mut::<AccessibilityNode>(entity) {
        accessible.clear_label();
    }
}

/// `AccessKit` integration for `bevy_ui`.
pub(crate) struct AccessibilityPlugin;

impl Plugin for AccessibilityPlugin {
    fn build(&self, app: &mut App) {
        // The UI also runs without the plugin that owns this resource.
        app.init_resource::<AccessibilityRequested>().add_systems(
            PostUpdate,
            (
                button_changed,
                image_changed,
                label_changed,
                sync_bounds_and_transforms
                    .after(button_changed)
                    .after(image_changed)
                    .after(label_changed),
            )
                .in_set(UiSystems::PostLayout)
                .before(AccessibilitySystems::Update),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::Update;

    #[test]
    fn bounds_follow_the_nearest_semantic_parent_across_layout_entities() {
        let mut app = App::new();
        app.init_resource::<AccessibilityRequested>()
            .add_systems(Update, sync_bounds_and_transforms);
        app.world().resource::<AccessibilityRequested>().set(true);
        let parent = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::TabList)),
                UiGlobalTransform::from_xy(100., 200.),
            ))
            .id();
        let layout = app.world_mut().spawn(ChildOf(parent)).id();
        let tab = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::Tab)),
                ComputedNode::default(),
                UiGlobalTransform::from_xy(130., 240.),
                ChildOf(layout),
            ))
            .id();
        let transform = |app: &App| {
            app.world()
                .get::<AccessibilityNode>(tab)
                .unwrap()
                .transform()
                .copied()
        };
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
        let last_changed = app
            .world()
            .entity(tab)
            .get_ref::<AccessibilityNode>()
            .unwrap()
            .last_changed();
        app.update();
        assert_eq!(
            app.world()
                .entity(tab)
                .get_ref::<AccessibilityNode>()
                .unwrap()
                .last_changed(),
            last_changed
        );
        // Adding/removing a semantic parent changes the relative transform even
        // when none of the global UI transforms or bounds have changed.
        app.world_mut().entity_mut(layout).insert((
            AccessibilityNode::from(Node::new(Role::Group)),
            UiGlobalTransform::from_xy(110., 210.),
        ));
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((20., 30.))));
        app.world_mut()
            .entity_mut(layout)
            .remove::<AccessibilityNode>();
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
        // A semantic-only group inherits the outer UI coordinate space, even
        // though the group has no UI transform of its own.
        app.world_mut()
            .entity_mut(layout)
            .remove::<UiGlobalTransform>()
            .insert(AccessibilityNode::from(Node::new(Role::Group)));
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
        app.world_mut()
            .get_mut::<AccessibilityNode>(layout)
            .unwrap()
            .set_transform(Affine::translate((10., 20.)) * Affine::scale(2.));
        app.update();
        assert_eq!(
            transform(&app),
            Some(Affine::translate((10., 10.)) * Affine::scale(0.5))
        );
        let mut group = Node::new(Role::Group);
        group.set_transform(Affine::translate((4., 6.)));
        let inner = app
            .world_mut()
            .spawn((AccessibilityNode::from(group), ChildOf(layout)))
            .id();
        app.world_mut().entity_mut(tab).insert(ChildOf(inner));
        app.update();
        assert_eq!(
            transform(&app),
            Some(Affine::translate((6., 4.)) * Affine::scale(0.5))
        );

        *app.world_mut()
            .get_mut::<UiGlobalTransform>(parent)
            .unwrap() =
            UiGlobalTransform::from(bevy_math::Affine2::from_scale(bevy_math::Vec2::ZERO));
        app.update();
        assert_eq!(transform(&app), None);
    }

    #[test]
    fn disabling_a_semantic_parent_moves_its_children() {
        let mut app = App::new();
        app.init_resource::<AccessibilityRequested>()
            .add_systems(Update, sync_bounds_and_transforms);
        app.world().resource::<AccessibilityRequested>().set(true);
        let parent = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::TabList)),
                UiGlobalTransform::from_xy(100., 200.),
            ))
            .id();
        let tab = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::Tab)),
                ComputedNode::default(),
                UiGlobalTransform::from_xy(130., 240.),
                ChildOf(parent),
            ))
            .id();
        let transform = |app: &App| {
            app.world()
                .get::<AccessibilityNode>(tab)
                .unwrap()
                .transform()
                .copied()
        };
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
        for expected in [(130., 240.), (30., 40.)] {
            let mut parent = app.world_mut().entity_mut(parent);
            if parent.contains::<Disabled>() {
                parent.remove::<Disabled>();
            } else {
                parent.insert(Disabled);
            }
            app.update();
            assert_eq!(transform(&app), Some(Affine::translate(expected)));
        }
    }

    #[test]
    fn moving_a_layout_entity_moves_the_nodes_below_it() {
        let mut app = App::new();
        app.init_resource::<AccessibilityRequested>()
            .add_systems(Update, sync_bounds_and_transforms);
        app.world().resource::<AccessibilityRequested>().set(true);
        let [near, far] = [(100., 200.), (0., 0.)].map(|(x, y)| {
            app.world_mut()
                .spawn((
                    AccessibilityNode::from(Node::new(Role::TabList)),
                    UiGlobalTransform::from_xy(x, y),
                ))
                .id()
        });
        let layout = app.world_mut().spawn(ChildOf(near)).id();
        let tab = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::Tab)),
                ComputedNode::default(),
                UiGlobalTransform::from_xy(130., 240.),
                ChildOf(layout),
            ))
            .id();
        let transform = |app: &App| {
            app.world()
                .get::<AccessibilityNode>(tab)
                .unwrap()
                .transform()
                .copied()
        };
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
        // The tab itself is untouched: only an entity above it moved.
        app.world_mut().entity_mut(layout).insert(ChildOf(far));
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((130., 240.))));
        app.world_mut().entity_mut(layout).remove::<ChildOf>();
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((130., 240.))));
        app.world_mut().entity_mut(layout).insert(ChildOf(near));
        app.update();
        assert_eq!(transform(&app), Some(Affine::translate((30., 40.))));
    }

    #[test]
    fn bounds_wait_until_accessibility_is_requested() {
        let mut app = App::new();
        app.init_resource::<AccessibilityRequested>()
            .add_systems(Update, sync_bounds_and_transforms);
        let node = app
            .world_mut()
            .spawn((
                AccessibilityNode::from(Node::new(Role::Button)),
                ComputedNode {
                    size: bevy_math::Vec2::new(20., 10.),
                    ..Default::default()
                },
                UiGlobalTransform::default(),
            ))
            .id();
        let bounds = |app: &App| app.world().get::<AccessibilityNode>(node).unwrap().bounds();
        app.update();
        app.update();
        assert_eq!(bounds(&app), None);
        // Nothing changed since the node spawned, yet it still has to be placed.
        app.world().resource::<AccessibilityRequested>().set(true);
        app.update();
        assert_eq!(bounds(&app), Some(Rect::new(-10., -5., 10., 5.)));
    }
}
