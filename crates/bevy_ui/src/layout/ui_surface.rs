use core::fmt;
use core::ops::{Deref, DerefMut};

use bevy_platform::collections::hash_map::Entry;
use taffy::TaffyTree;

#[cfg(feature = "ghost_nodes")]
use bevy_ecs::entity::EntityHashSet;
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    prelude::Resource,
};
use bevy_math::{UVec2, Vec2};
use bevy_utils::default;

use crate::{layout::convert, LayoutContext, LayoutError, Measure, MeasureArgs, Node, NodeMeasure};
use bevy_text::FontCx;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LayoutNode {
    // Implicit "viewport" node if this `LayoutNode` corresponds to a root UI node entity
    pub(super) viewport_id: Option<taffy::NodeId>,
    // The id of the node in the taffy tree
    pub(super) id: taffy::NodeId,
}

impl From<taffy::NodeId> for LayoutNode {
    fn from(value: taffy::NodeId) -> Self {
        LayoutNode {
            viewport_id: None,
            id: value,
        }
    }
}

pub(crate) struct UiTree<T>(TaffyTree<T>);

#[expect(unsafe_code, reason = "TaffyTree is safe as long as calc is not used")]
// SAFETY: Taffy Tree becomes thread unsafe when you use the calc feature, which we do not implement
unsafe impl Send for UiTree<NodeMeasure> {}

#[expect(unsafe_code, reason = "TaffyTree is safe as long as calc is not used")]
// SAFETY: Taffy Tree becomes thread unsafe when you use the calc feature, which we do not implement
unsafe impl Sync for UiTree<NodeMeasure> {}

impl<T> Deref for UiTree<T> {
    type Target = TaffyTree<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UiTree<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Resource)]
pub struct UiSurface {
    pub root_entity_to_viewport_node: EntityHashMap<taffy::NodeId>,
    pub(super) entity_to_taffy: EntityHashMap<LayoutNode>,
    pub(super) taffy: UiTree<NodeMeasure>,
    // Last successful solve per root. Include the generational viewport ID so a
    // recreated viewport can never reuse the previous viewport's rounded layout.
    computed_viewports: EntityHashMap<(taffy::NodeId, UVec2)>,
    taffy_children_scratch: Vec<taffy::NodeId>,
    #[cfg(feature = "ghost_nodes")]
    pub(super) dirty_ghost_children_scratch: EntityHashSet,
}

fn _assert_send_sync_ui_surface_impl_safe() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<EntityHashMap<taffy::NodeId>>();
    _assert_send_sync::<UiTree<NodeMeasure>>();
    _assert_send_sync::<UiSurface>();
}

impl fmt::Debug for UiSurface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug = f.debug_struct("UiSurface");
        debug
            .field("entity_to_taffy", &self.entity_to_taffy)
            .field("taffy_children_scratch", &self.taffy_children_scratch);
        #[cfg(feature = "ghost_nodes")]
        debug.field(
            "dirty_ghost_children_scratch",
            &self.dirty_ghost_children_scratch,
        );
        debug.finish()
    }
}

impl Default for UiSurface {
    fn default() -> Self {
        let taffy: UiTree<NodeMeasure> = UiTree(TaffyTree::new());
        Self {
            root_entity_to_viewport_node: Default::default(),
            entity_to_taffy: Default::default(),
            taffy,
            computed_viewports: Default::default(),
            taffy_children_scratch: Vec::new(),
            #[cfg(feature = "ghost_nodes")]
            dirty_ghost_children_scratch: EntityHashSet::new(),
        }
    }
}

impl UiSurface {
    /// Retrieves the Taffy node associated with the given UI node entity and updates its style.
    /// If no associated Taffy node exists a new Taffy node is inserted into the Taffy layout.
    pub fn upsert_node(
        &mut self,
        layout_context: &LayoutContext,
        entity: Entity,
        node: &Node,
        new_node_context: Option<NodeMeasure>,
    ) {
        let taffy = &mut self.taffy;
        let style = convert::from_node(node, layout_context);

        match self.entity_to_taffy.entry(entity) {
            Entry::Occupied(entry) => {
                let taffy_node = *entry.get();
                if new_node_context.is_some() {
                    taffy
                        .set_node_context(taffy_node.id, new_node_context)
                        .unwrap();
                }

                if taffy.style(taffy_node.id).unwrap() != &style {
                    taffy.set_style(taffy_node.id, style).unwrap();
                }
            }
            Entry::Vacant(entry) => {
                let taffy_node = if let Some(measure) = new_node_context {
                    taffy.new_leaf_with_context(style, measure)
                } else {
                    taffy.new_leaf(style)
                };
                entry.insert(taffy_node.unwrap().into());
            }
        }
    }

    /// Update the `MeasureFunc` of the taffy node corresponding to the given [`Entity`] if the node exists.
    pub fn update_node_context(&mut self, entity: Entity, context: NodeMeasure) -> Option<()> {
        let taffy_node = self.entity_to_taffy.get(&entity)?;
        self.taffy
            .set_node_context(taffy_node.id, Some(context))
            .ok()
    }

    /// Update the children of the taffy node corresponding to the given [`Entity`].
    pub fn update_children(&mut self, entity: Entity, children: impl Iterator<Item = Entity>) {
        self.taffy_children_scratch.clear();

        for child in children {
            if let Some(taffy_node) = self.entity_to_taffy.get_mut(&child) {
                self.taffy_children_scratch.push(taffy_node.id);
                if let Some(viewport_id) = taffy_node.viewport_id.take() {
                    self.taffy.remove(viewport_id).ok();
                    self.root_entity_to_viewport_node.remove(&child);
                    self.computed_viewports.remove(&child);
                }
            }
        }

        let taffy_node = self.entity_to_taffy.get(&entity).unwrap();
        self.taffy
            .set_children(taffy_node.id, &self.taffy_children_scratch)
            .unwrap();
    }

    /// Removes children from the entity's taffy node if it exists. Does nothing otherwise.
    pub fn try_remove_children(&mut self, entity: Entity) {
        if let Some(taffy_node) = self.entity_to_taffy.get(&entity) {
            self.taffy.set_children(taffy_node.id, &[]).unwrap();
        }
    }

    /// Removes the measure from the entity's taffy node if it exists. Does nothing otherwise.
    pub fn try_remove_node_context(&mut self, entity: Entity) {
        if let Some(taffy_node) = self.entity_to_taffy.get(&entity) {
            self.taffy.set_node_context(taffy_node.id, None).unwrap();
        }
    }

    /// Gets or inserts an implicit taffy viewport node corresponding to the given UI root entity
    pub fn get_or_insert_taffy_viewport_node(&mut self, ui_root_entity: Entity) -> taffy::NodeId {
        *self
            .root_entity_to_viewport_node
            .entry(ui_root_entity)
            .or_insert_with(|| {
                let root_node = self.entity_to_taffy.get_mut(&ui_root_entity).unwrap();
                let implicit_root = self
                    .taffy
                    .new_leaf(taffy::style::Style {
                        display: taffy::style::Display::Grid,
                        // Note: Taffy percentages are floats ranging from 0.0 to 1.0.
                        // So this is setting width:100% and height:100%
                        size: taffy::geometry::Size {
                            width: taffy::style_helpers::percent(1.0_f32),
                            height: taffy::style_helpers::percent(1.0_f32),
                        },
                        align_items: Some(taffy::style::AlignItems::Start),
                        justify_items: Some(taffy::style::JustifyItems::Start),
                        ..default()
                    })
                    .unwrap();
                // Unlike add_child, set_children detaches a promoted root from its
                // previous parent and invalidates that parent's layout cache.
                self.taffy
                    .set_children(implicit_root, &[root_node.id])
                    .unwrap();
                root_node.viewport_id = Some(implicit_root);
                implicit_root
            })
    }

    /// Compute the layout for the given implicit taffy viewport node.
    /// Reuses the rounded layout when both the tree and viewport are unchanged.
    /// Changes to a measurement must be submitted through `update_node_context`
    /// or `upsert_node`, as required by Taffy's layout cache.
    pub fn compute_layout<'a>(
        &mut self,
        ui_root_entity: Entity,
        render_target_resolution: UVec2,
        buffer_query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
        font_system: &'a mut FontCx,
    ) {
        let implicit_viewport_node = self.get_or_insert_taffy_viewport_node(ui_root_entity);
        if self.computed_viewports.get(&ui_root_entity)
            == Some(&(implicit_viewport_node, render_target_resolution))
            && !self.taffy.dirty(implicit_viewport_node).unwrap()
        {
            // Taffy caches the solve, but still traverses the entire tree to round
            // it on every compute_layout_with_measure call.
            return;
        }

        let available_space = taffy::geometry::Size {
            width: taffy::style::AvailableSpace::Definite(render_target_resolution.x as f32),
            height: taffy::style::AvailableSpace::Definite(render_target_resolution.y as f32),
        };

        self.taffy
            .compute_layout_with_measure(
                implicit_viewport_node,
                available_space,
                |known_dimensions: taffy::Size<Option<f32>>,
                 available_space: taffy::Size<taffy::AvailableSpace>,
                 _node_id: taffy::NodeId,
                 context: Option<&mut NodeMeasure>,
                 style: &taffy::Style|
                 -> taffy::Size<f32> {
                    context
                        .map(|ctx| {
                            let buffer = get_text_buffer(
                                crate::widget::TextMeasure::needs_buffer(
                                    known_dimensions.height,
                                    available_space.width,
                                ),
                                ctx,
                                buffer_query,
                            );
                            let size = ctx.measure(MeasureArgs {
                                known_width: known_dimensions.width,
                                known_height: known_dimensions.height,
                                available_width: available_space.width,
                                available_height: available_space.height,
                                font_system,
                                buffer,
                                style,
                            });
                            taffy::Size {
                                width: size.x,
                                height: size.y,
                            }
                        })
                        .unwrap_or(taffy::Size::ZERO)
                },
            )
            .unwrap();
        self.computed_viewports.insert(
            ui_root_entity,
            (implicit_viewport_node, render_target_resolution),
        );
    }

    /// Removes each entity from the internal map and then removes their associated nodes from taffy
    pub fn remove_entities(&mut self, entities: impl IntoIterator<Item = Entity>) {
        for entity in entities {
            if let Some(node) = self.entity_to_taffy.remove(&entity) {
                // Taffy 0.10's remove does not invalidate the former parent.
                if let Some(parent) = self.taffy.parent(node.id) {
                    self.taffy.mark_dirty(parent).unwrap();
                }
                self.taffy.remove(node.id).unwrap();
                if let Some(viewport_node) = node.viewport_id {
                    self.taffy.remove(viewport_node).ok();
                }
            }
            self.root_entity_to_viewport_node.remove(&entity);
            self.computed_viewports.remove(&entity);
        }
    }

    /// Get the layout geometry for the taffy node corresponding to the ui node [`Entity`].
    /// Does not compute the layout geometry, `compute_window_layouts` should be run before using this function.
    /// On success returns a pair consisting of the final resolved layout values after rounding
    /// and the size of the node after layout resolution but before rounding.
    pub fn get_layout(
        &self,
        entity: Entity,
        use_rounding: bool,
    ) -> Result<(taffy::Layout, Vec2), LayoutError> {
        let Some(taffy_node) = self.entity_to_taffy.get(&entity) else {
            return Err(LayoutError::InvalidHierarchy);
        };

        // Keep rounding enabled on the tree: compute_layout must always produce
        // both layouts, including when subsequent calls reuse the cached result.
        let unrounded = self.taffy.unrounded_layout(taffy_node.id);
        let layout = if use_rounding {
            self.taffy
                .layout(taffy_node.id)
                .map_err(LayoutError::TaffyError)?
        } else {
            unrounded
        };
        Ok((
            *layout,
            Vec2::new(unrounded.size.width, unrounded.size.height),
        ))
    }
}

pub fn get_text_buffer<'a>(
    needs_buffer: bool,
    ctx: &mut NodeMeasure,
    query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
) -> Option<&'a mut bevy_text::ComputedTextBlock> {
    // We avoid a query lookup whenever the buffer is not required.
    if !needs_buffer {
        return None;
    }
    let NodeMeasure::Text(crate::widget::TextMeasure { info }) = ctx else {
        return None;
    };
    let Ok(computed) = query.get_mut(info.entity) else {
        return None;
    };
    Some(computed.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentSize, FixedMeasure};
    use bevy_math::Vec2;
    use taffy::TraversePartialTree;

    fn compute(surface: &mut UiSurface, root: Entity, resolution: UVec2) {
        let mut world = bevy_ecs::world::World::new();
        let mut query = world.query::<&mut bevy_text::ComputedTextBlock>();
        surface.compute_layout(
            root,
            resolution,
            &mut query.query_mut(&mut world),
            &mut FontCx::default(),
        );
    }

    fn assert_matches_fresh_layout(surface: &mut UiSurface, root: Entity, resolution: UVec2) {
        let entities: Vec<_> = surface.entity_to_taffy.keys().copied().collect();
        let layouts: Vec<_> = entities
            .iter()
            .map(|entity| {
                (
                    surface.get_layout(*entity, true).unwrap(),
                    surface.get_layout(*entity, false).unwrap(),
                )
            })
            .collect();
        // Clear descendant caches too: dirtying only the viewport would let a
        // stale child cache pass this comparison against a supposedly fresh solve.
        for entity in &entities {
            surface
                .taffy
                .mark_dirty(surface.entity_to_taffy[entity].id)
                .unwrap();
        }
        let viewport = surface.get_or_insert_taffy_viewport_node(root);
        surface.taffy.mark_dirty(viewport).unwrap();
        compute(surface, root, resolution);
        for (entity, (rounded, unrounded)) in entities.into_iter().zip(layouts) {
            assert_eq!(surface.get_layout(entity, true).unwrap(), rounded);
            assert_eq!(surface.get_layout(entity, false).unwrap(), unrounded);
        }
    }

    #[test]
    fn cached_layout_preserves_rounding_and_invalidates_on_resize_and_style() {
        let mut surface = UiSurface::default();
        let root = Entity::from_raw_u32(1).unwrap();
        let mut node = Node {
            width: crate::Val::Percent(33.3),
            height: crate::Val::Px(10.25),
            ..default()
        };
        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &node, None);
        let resolution = UVec2::new(100, 100);
        compute(&mut surface, root, resolution);
        let first = surface.get_layout(root, true).unwrap();
        assert_eq!(first.0.size.width, 33.0);
        assert_ne!(first.0.size.width, first.1.x);
        let viewport = surface.get_or_insert_taffy_viewport_node(root);
        assert!(!surface.taffy.dirty(viewport).unwrap());

        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &node, None);
        assert!(!surface.taffy.dirty(viewport).unwrap());
        // A changed Bevy Node can still convert to the same Taffy style.
        node.border_radius = crate::BorderRadius::all(crate::Val::Px(5.0));
        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &node, None);
        assert!(!surface.taffy.dirty(viewport).unwrap());
        let unrounded = surface.get_layout(root, false).unwrap();
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, true).unwrap(), first);
        assert_eq!(surface.get_layout(root, false).unwrap(), unrounded);
        assert_matches_fresh_layout(&mut surface, root, resolution);

        let resized = UVec2::new(200, 100);
        compute(&mut surface, root, resized);
        assert_eq!(surface.get_layout(root, true).unwrap().0.size.width, 67.0);
        assert_matches_fresh_layout(&mut surface, root, resized);

        node.width = crate::Val::Px(40.25);
        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &node, None);
        assert!(surface.taffy.dirty(viewport).unwrap());
        compute(&mut surface, root, resized);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 40.25);
        assert_matches_fresh_layout(&mut surface, root, resized);

        let scaled_context = LayoutContext {
            scale_factor: 2.0,
            ..LayoutContext::TEST_CONTEXT
        };
        surface.upsert_node(&scaled_context, root, &node, None);
        compute(&mut surface, root, resized);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 80.5);
        assert_matches_fresh_layout(&mut surface, root, resized);
    }

    #[test]
    fn cached_layout_does_not_reuse_a_recreated_clean_viewport() {
        let mut surface = UiSurface::default();
        let root = Entity::from_raw_u32(1).unwrap();
        let resolution = UVec2::splat(100);
        let node = Node {
            width: crate::Val::Percent(50.0),
            ..default()
        };
        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &node, None);
        compute(&mut surface, root, resolution);

        // Recreate the viewport while retaining the old cache entry, then make
        // it clean at another resolution. Identity must prevent an early return.
        let old_viewport = surface.root_entity_to_viewport_node.remove(&root).unwrap();
        surface.taffy.remove(old_viewport).unwrap();
        let viewport = surface.get_or_insert_taffy_viewport_node(root);
        assert_ne!(viewport, old_viewport);
        surface
            .taffy
            .compute_layout(
                viewport,
                taffy::Size {
                    width: taffy::AvailableSpace::Definite(200.0),
                    height: taffy::AvailableSpace::Definite(200.0),
                },
            )
            .unwrap();
        assert!(!surface.taffy.dirty(viewport).unwrap());
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 100.0);

        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 50.0);
        assert_matches_fresh_layout(&mut surface, root, resolution);
    }

    #[test]
    fn cached_layout_invalidates_on_measure_changes_and_removal() {
        let mut surface = UiSurface::default();
        let root = Entity::from_raw_u32(1).unwrap();
        let child = Entity::from_raw_u32(2).unwrap();
        let resolution = UVec2::splat(100);
        surface.upsert_node(&LayoutContext::TEST_CONTEXT, root, &Node::default(), None);
        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            child,
            &Node::default(),
            Some(NodeMeasure::Fixed(FixedMeasure {
                size: Vec2::splat(10.25),
            })),
        );
        surface.update_children(root, [child].into_iter());
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 10.25);

        surface
            .update_node_context(
                child,
                NodeMeasure::Fixed(FixedMeasure {
                    size: Vec2::splat(20.25),
                }),
            )
            .unwrap();
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 20.25);
        assert_matches_fresh_layout(&mut surface, root, resolution);

        surface.try_remove_node_context(child);
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 0.0);

        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            child,
            &Node::default(),
            Some(NodeMeasure::Fixed(FixedMeasure {
                size: Vec2::splat(30.25),
            })),
        );
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 30.25);
        // Exercise removal without a preceding update_children call.
        surface.remove_entities([child]);
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, false).unwrap().0.size.width, 0.0);
        assert_matches_fresh_layout(&mut surface, root, resolution);
    }

    #[test]
    fn cached_layout_invalidates_on_text_measure_changes() {
        let mut surface = UiSurface::default();
        let root = Entity::from_raw_u32(1).unwrap();
        let resolution = UVec2::splat(100);
        // Definite height exercises text intrinsic widths without requiring fonts
        // or a shaped buffer, which are independent of layout invalidation.
        let node = Node {
            height: crate::Val::Px(10.0),
            ..default()
        };
        let measure = |width| {
            NodeMeasure::Text(crate::widget::TextMeasure {
                info: bevy_text::TextMeasureInfo {
                    min: Vec2::new(width, 10.0),
                    max: Vec2::new(width, 10.0),
                    entity: root,
                },
            })
        };
        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            root,
            &node,
            Some(measure(20.0)),
        );
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, true).unwrap().0.size.width, 20.0);
        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            root,
            &node,
            Some(measure(40.0)),
        );
        compute(&mut surface, root, resolution);
        assert_eq!(surface.get_layout(root, true).unwrap().0.size.width, 40.0);
        assert_matches_fresh_layout(&mut surface, root, resolution);
    }

    #[test]
    fn cached_layout_handles_reparenting_and_root_role_changes() {
        let mut surface = UiSurface::default();
        let a = Entity::from_raw_u32(1).unwrap();
        let b = Entity::from_raw_u32(2).unwrap();
        let child = Entity::from_raw_u32(3).unwrap();
        let resolution = UVec2::splat(100);
        for entity in [a, b] {
            surface.upsert_node(&LayoutContext::TEST_CONTEXT, entity, &Node::default(), None);
        }
        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            child,
            &Node {
                width: crate::Val::Px(10.25),
                height: crate::Val::Px(10.25),
                ..default()
            },
            None,
        );
        surface.update_children(a, [child].into_iter());
        compute(&mut surface, a, resolution);
        compute(&mut surface, b, resolution);

        surface.update_children(b, [child].into_iter());
        compute(&mut surface, a, resolution);
        compute(&mut surface, b, resolution);
        assert_eq!(surface.get_layout(a, false).unwrap().0.size.width, 0.0);
        assert_eq!(surface.get_layout(b, false).unwrap().0.size.width, 10.25);
        assert_matches_fresh_layout(&mut surface, b, resolution);

        // Promotion must detach the child from b and invalidate b too.
        compute(&mut surface, child, resolution);
        compute(&mut surface, b, resolution);
        assert_eq!(surface.get_layout(b, false).unwrap().0.size.width, 0.0);
        assert_matches_fresh_layout(&mut surface, child, resolution);
        let old_viewport = surface.get_or_insert_taffy_viewport_node(child);

        // Demotion removes its viewport and cached resolution.
        surface.update_children(a, [child].into_iter());
        assert!(!surface.computed_viewports.contains_key(&child));
        compute(&mut surface, a, resolution);
        surface.try_remove_children(a);
        compute(&mut surface, a, resolution);
        compute(&mut surface, child, resolution);
        assert_ne!(
            surface.get_or_insert_taffy_viewport_node(child),
            old_viewport
        );
        assert_eq!(surface.get_layout(a, false).unwrap().0.size.width, 0.0);
        assert_matches_fresh_layout(&mut surface, child, resolution);

        surface.remove_entities([child]);
        assert!(!surface.computed_viewports.contains_key(&child));
        surface.upsert_node(
            &LayoutContext::TEST_CONTEXT,
            child,
            &Node {
                width: crate::Val::Px(50.25),
                ..default()
            },
            None,
        );
        compute(&mut surface, child, resolution);
        assert_eq!(
            surface.get_layout(child, false).unwrap().0.size.width,
            50.25
        );
        assert_matches_fresh_layout(&mut surface, child, resolution);
    }

    #[test]
    fn test_initialization() {
        let ui_surface = UiSurface::default();
        assert!(ui_surface.entity_to_taffy.is_empty());
        assert_eq!(ui_surface.taffy.total_node_count(), 0);
    }

    #[test]
    fn test_upsert() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        // standard upsert
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // should be inserted into taffy
        assert_eq!(ui_surface.taffy.total_node_count(), 1);
        assert!(ui_surface.entity_to_taffy.contains_key(&root_node_entity));

        // test duplicate insert 1
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // node count should not have increased
        assert_eq!(ui_surface.taffy.total_node_count(), 1);

        // assign root node to camera
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        // each root node will create 2 taffy nodes
        assert_eq!(ui_surface.taffy.total_node_count(), 2);

        // test duplicate insert 2
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // node count should not have increased
        assert_eq!(ui_surface.taffy.total_node_count(), 2);
    }

    #[test]
    fn test_remove_entities() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        assert!(ui_surface.entity_to_taffy.contains_key(&root_node_entity));

        ui_surface.remove_entities([root_node_entity]);
        assert!(!ui_surface.entity_to_taffy.contains_key(&root_node_entity));
    }

    #[test]
    fn test_try_update_measure() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        let mut content_size = ContentSize::default();
        content_size.set(NodeMeasure::Fixed(FixedMeasure { size: Vec2::ONE }));
        let measure_func = content_size.measure.take().unwrap();
        assert!(ui_surface
            .update_node_context(root_node_entity, measure_func)
            .is_some());
    }

    #[test]
    fn test_update_children() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let child_entity = Entity::from_raw_u32(2).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, child_entity, &node, None);

        ui_surface.update_children(root_node_entity, vec![child_entity].into_iter());

        let parent_node = *ui_surface.entity_to_taffy.get(&root_node_entity).unwrap();
        let child_node = *ui_surface.entity_to_taffy.get(&child_entity).unwrap();
        assert_eq!(ui_surface.taffy.parent(child_node.id), Some(parent_node.id));
    }

    #[expect(
        unreachable_code,
        reason = "Certain pieces of code tested here cause the test to fail if made reachable; see #16362 for progress on fixing this"
    )]
    #[test]
    fn test_set_camera_children() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let child_entity = Entity::from_raw_u32(2).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, child_entity, &node, None);

        let root_taffy_node = *ui_surface.entity_to_taffy.get(&root_node_entity).unwrap();
        let child_taffy = *ui_surface.entity_to_taffy.get(&child_entity).unwrap();

        // set up the relationship manually
        ui_surface
            .taffy
            .add_child(root_taffy_node.id, child_taffy.id)
            .unwrap();

        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        assert_eq!(
            ui_surface.taffy.parent(child_taffy.id),
            Some(root_taffy_node.id)
        );
        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );

        // clear camera's root nodes
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        return; // TODO: can't pass the test if we continue - not implemented (remove allow(unreachable_code))

        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );

        // re-associate root node with viewport node
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        let child_taffy = ui_surface.entity_to_taffy.get(&child_entity).unwrap();
        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );
    }
}
