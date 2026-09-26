use crate::{
    experimental::UiChildren,
    prelude::{Button, Label},
    ui_transform::UiGlobalTransform,
    widget::{ImageNode, TextUiReader},
    ComputedNode, UiSystems,
};
use bevy_a11y::{AccessibilityNode, AccessibilitySystems};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::{
    component::Component,
    hierarchy::ChildOf,
    lifecycle::HookContext,
    prelude::Entity,
    query::{Changed, With, Without},
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query},
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

fn sync_bounds_and_transforms(
    mut accessible_nodes_query: Query<(
        Entity,
        &mut AccessibilityNode,
        &ComputedNode,
        &UiGlobalTransform,
    )>,
    accessible_transform_query: Query<&UiGlobalTransform, With<AccessibilityNode>>,
    semantic_only_nodes: Query<&AccessibilityNode, Without<UiGlobalTransform>>,
    parents: Query<&ChildOf>,
) {
    for (entity, mut accessible, node, ui_transform) in &mut accessible_nodes_query {
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

fn button_changed(
    mut commands: Commands,
    mut query: Query<(Entity, Option<&mut AccessibilityNode>), Changed<Button>>,
    ui_children: UiChildren,
    mut text_reader: TextUiReader,
) {
    for (entity, accessible) in &mut query {
        let label = calc_label(&mut text_reader, ui_children.iter_ui_children(entity));
        if let Some(mut accessible) = accessible {
            accessible.set_role(Role::Button);
            if let Some(name) = label {
                accessible.set_label(name);
            } else {
                accessible.clear_label();
            }
        } else {
            let mut node = Node::new(Role::Button);
            if let Some(label) = label {
                node.set_label(label);
            }
            commands
                .entity(entity)
                .try_insert(AccessibilityNode::from(node));
        }
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
        if let Some(mut accessible) = accessible {
            accessible.set_role(Role::Image);
            if let Some(label) = label {
                accessible.set_label(label);
            } else {
                accessible.clear_label();
            }
        } else {
            let mut node = Node::new(Role::Image);
            if let Some(label) = label {
                node.set_label(label);
            }
            commands
                .entity(entity)
                .try_insert(AccessibilityNode::from(node));
        }
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
        app.add_systems(
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
    use bevy_ecs::change_detection::DetectChanges;

    #[test]
    fn bounds_follow_the_nearest_semantic_parent_across_layout_entities() {
        let mut app = App::new();
        app.add_systems(Update, sync_bounds_and_transforms);
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
}
