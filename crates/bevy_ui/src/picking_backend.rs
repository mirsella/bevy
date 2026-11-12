//! A picking backend for UI nodes.
//!
//! # Usage
//!
//! This backend does not require markers on cameras or entities to function. It will look for any
//! pointers using the same render target as the UI camera, and run hit tests on the UI node tree.
//!
//! ## Important Note
//!
//! This backend completely ignores [`FocusPolicy`](crate::FocusPolicy). The design of `bevy_ui`'s
//! focus systems and the picking plugin are not compatible. Instead, use the optional [`Pickable`] component
//! to override how an entity responds to picking focus. Nodes without the [`Pickable`] component
//! will still trigger events and block items below it from being hovered.
//!
//! ## Implementation Notes
//!
//! - `bevy_ui` can render on any camera with a flag, it is special, and is not tied to a particular
//!   camera.
//! - To correctly sort picks, the order of `bevy_ui` is set to be the camera order plus 0.5.
//! - The `position` reported in `HitData` is normalized relative to the node, with
//!   `(-0.5, -0.5, 0.)` at the top left and `(0.5, 0.5, 0.)` in the bottom right. Coordinates are
//!   relative to the entire node, not just the visible region. This backend does not provide a `normal`.

use crate::{prelude::*, ui_transform::UiGlobalTransform, ComputedStackIndex, UiStack};
use bevy_app::prelude::*;
use bevy_camera::{visibility::InheritedVisibility, Camera, RenderTarget};
use bevy_ecs::{prelude::*, query::QueryData};
use bevy_math::Vec2;
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_text::{ComputedTextBlock, TextLayoutInfo};
use bevy_window::PrimaryWindow;

use bevy_picking::backend::prelude::*;

/// An optional component that marks cameras that should be used in the [`UiPickingPlugin`].
///
/// Only needed if [`UiPickingSettings::require_markers`] is set to `true`, and ignored
/// otherwise.
#[derive(Debug, Clone, Default, Component, Reflect)]
#[reflect(Debug, Default, Component)]
pub struct UiPickingCamera;

/// Runtime settings for the [`UiPickingPlugin`].
#[derive(Resource, Reflect)]
#[reflect(Resource, Default)]
pub struct UiPickingSettings {
    /// When set to `true` UI picking will only consider cameras marked with
    /// [`UiPickingCamera`] and entities marked with [`Pickable`]. `false` by default.
    ///
    /// This setting is provided to give you fine-grained control over which cameras and entities
    /// should be used by the UI picking backend at runtime.
    pub require_markers: bool,
}

#[expect(
    clippy::allow_attributes,
    reason = "clippy::derivable_impls is not always linted"
)]
#[allow(
    clippy::derivable_impls,
    reason = "Known false positive with clippy: <https://github.com/rust-lang/rust-clippy/issues/13160>"
)]
impl Default for UiPickingSettings {
    fn default() -> Self {
        Self {
            require_markers: false,
        }
    }
}

/// A plugin that adds picking support for UI nodes.
///
/// This is included by default in [`UiPlugin`](crate::UiPlugin).
#[derive(Clone)]
pub struct UiPickingPlugin;
impl Plugin for UiPickingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiPickingSettings>()
            .add_systems(PreUpdate, ui_picking.in_set(PickingSystems::Backend));
    }
}

/// Main query from bevy's `ui_focus_system`
#[derive(QueryData)]
#[query_data(mutable)]
pub struct NodeQuery {
    entity: Entity,
    stack_index: &'static ComputedStackIndex,
    node: &'static ComputedNode,
    transform: &'static UiGlobalTransform,
    pickable: Option<&'static Pickable>,
    inherited_visibility: Option<&'static InheritedVisibility>,
    target_camera: &'static ComputedUiTargetCamera,
    text_node: Option<(&'static TextLayoutInfo, &'static ComputedTextBlock)>,
    calculated_clip: Option<&'static CalculatedClip>,
}

/// Computes the UI node entities under each pointer.
///
/// Bevy's [`UiStack`] orders all nodes in the order they will be rendered, which is the same order
/// we need for determining picking.
pub fn ui_picking(
    pointers: Query<(&PointerId, &PointerLocation)>,
    camera_query: Query<(Entity, &Camera, &RenderTarget, Has<UiPickingCamera>)>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    settings: Res<UiPickingSettings>,
    ui_stack: Res<UiStack>,
    node_query: Query<NodeQuery>,
    markers_query: Query<NodeQuery, Or<(With<Pickable>, With<TextLayoutInfo>)>>,
    mut output: MessageWriter<PointerHits>,
    pickable_query: Query<&Pickable>,
) {
    // Map from each camera to its active pointers and their positions in viewport space
    let mut pointer_pos_by_camera = HashMap::<Entity, HashMap<PointerId, Vec2>>::default();

    for (pointer_id, pointer_location) in
        pointers.iter().filter_map(|(pointer, pointer_location)| {
            Some(*pointer).zip(pointer_location.location().cloned())
        })
    {
        // This pointer is associated with a render target, which could be used by multiple
        // cameras. We want to ensure we return all cameras with a matching target.
        for (entity, camera, _, _) in
            camera_query
                .iter()
                .filter(|(_, _, render_target, cam_can_pick)| {
                    (!settings.require_markers || *cam_can_pick)
                        && render_target
                            .normalize(primary_window.single().ok())
                            .is_some_and(|target| target == pointer_location.target)
                })
        {
            let mut pointer_pos =
                pointer_location.position * camera.target_scaling_factor().unwrap_or(1.);
            if let Some(viewport) = camera.physical_viewport_rect() {
                if !viewport.as_rect().contains(pointer_pos) {
                    // The pointer is outside the viewport, skip it
                    continue;
                }
                pointer_pos -= viewport.min.as_vec2();
            }
            pointer_pos_by_camera
                .entry(entity)
                .or_default()
                .insert(pointer_id, pointer_pos);
        }
    }

    // The nodes under each (camera, pointer) combo, with their stack index.
    let mut hit_nodes = HashMap::<
        (Entity, PointerId),
        Vec<(u32, Entity, Entity, Option<Pickable>, Vec2)>,
    >::default();

    // Only nodes that can be reported are tested, in query order, and the hits are
    // sorted by stack index afterwards. Walking the whole `UiStack` and looking up each
    // node costs far more when most nodes are not pickable.
    // A stack slice shares its root's camera, and is skipped without one, as when the
    // root was despawned after the stack was built.
    let slice_cameras: Vec<_> = ui_stack
        .partition
        .iter()
        .map(|range| {
            let camera = node_query
                .get(ui_stack.uinodes[range.start])
                .ok()
                .and_then(|root| root.target_camera.get());
            (range, camera)
        })
        .collect();

    let mut visit = |node: NodeQueryReadOnlyItem| {
        // Nodes spawned since the stack was built are not pickable yet.
        let index = node.stack_index.0;
        if ui_stack.uinodes.get(index as usize) != Some(&node.entity) {
            return;
        }

        // Nodes with Display::None have a (0., 0.) logical rect and can be ignored
        if node.node.size() == Vec2::ZERO {
            return;
        }

        // Nodes that are not rendered should not be interactable
        if node
            .inherited_visibility
            .map(|inherited_visibility| inherited_visibility.get())
            != Some(true)
        {
            return;
        }

        let slice = slice_cameras.partition_point(|(range, _)| range.end <= index as usize);
        let Some(&(range, Some(camera_entity))) = slice_cameras.get(slice) else {
            return;
        };
        if !range.contains(&(index as usize)) {
            return;
        }
        let Some(pointers_on_this_cam) = pointer_pos_by_camera.get(&camera_entity) else {
            return;
        };

        // Find the normalized cursor position relative to the node.
        // (±0., 0.) is the center with the corners at points (±0.5, ±0.5).
        // Coordinates are relative to the entire node, not just the visible region.
        for (pointer_id, cursor_position) in pointers_on_this_cam.iter() {
            if node.node.contains_point(*node.transform, *cursor_position)
                && node
                    .calculated_clip
                    .is_none_or(|clip| clip.contains_point(*cursor_position))
                && let Some(target) = node
                    .text_node
                    .and_then(|(text_layout_info, text_block)| {
                        pick_ui_text_section(
                            node.node,
                            node.transform,
                            *cursor_position,
                            text_layout_info,
                            text_block,
                        )
                        .filter(|&text_entity| {
                            !settings.require_markers || pickable_query.contains(text_entity)
                        })
                    })
                    .or_else(|| {
                        (!settings.require_markers || node.pickable.is_some())
                            .then_some(node.entity)
                    })
            {
                hit_nodes
                    .entry((camera_entity, *pointer_id))
                    .or_default()
                    .push((
                        index,
                        target,
                        camera_entity,
                        node.pickable.cloned(),
                        node.transform.inverse().transform_point2(*cursor_position)
                            / node.node.size(),
                    ));
            }
        }
    };
    if settings.require_markers {
        markers_query.iter().for_each(&mut visit);
    } else {
        node_query.iter().for_each(&mut visit);
    }

    for ((camera, pointer), hovered) in hit_nodes.iter_mut() {
        // Closest nodes first.
        hovered.sort_unstable_by_key(|(index, ..)| core::cmp::Reverse(*index));

        // As soon as a node with a `Block` focus policy is detected, the iteration will stop on it
        // because it "captures" the interaction.
        let mut picks = Vec::new();
        let mut depth = 0.0;

        for (_, hovered_node, camera_entity, pickable, position) in hovered.iter() {
            picks.push((
                *hovered_node,
                HitData::new(*camera_entity, depth, Some(position.extend(0.0)), None),
            ));

            if let Some(pickable) = pickable {
                // If an entity has a `Pickable` component, we will use that as the source of truth.
                if pickable.should_block_lower {
                    break;
                }
            } else {
                // If the `Pickable` component doesn't exist, default behavior is to block.
                break;
            }

            depth += 0.00001; // keep depth near 0 for precision
        }

        let order = camera_query
            .get(*camera)
            .map(|(_, cam, _, _)| cam.order)
            .unwrap_or_default() as f32
            + 0.5; // bevy ui can run on any camera, it's a special case

        output.write(PointerHits::new(*pointer, picks, order));
    }
}

fn pick_ui_text_section(
    uinode: &ComputedNode,
    global_transform: &UiGlobalTransform,
    point: Vec2,
    text_layout_info: &TextLayoutInfo,
    text_block: &ComputedTextBlock,
) -> Option<Entity> {
    let local_point = global_transform
        .try_inverse()
        .map(|transform| transform.transform_point2(point) - uinode.content_box().min)?;
    let section_index = text_layout_info
        .run_geometry
        .iter()
        .find(|run| run.bounds.contains(local_point))
        .map(|run| run.section_index)?;
    text_block.entities().get(section_index).map(|e| e.entity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_camera::NormalizedRenderTarget;
    use bevy_math::UVec2;
    use bevy_picking::pointer::{Location, PointerId, PointerLocation};

    const PASS_THROUGH: Pickable = Pickable {
        should_block_lower: false,
        is_hoverable: true,
    };

    /// A world with one camera and a mouse pointer at its center, returning the camera.
    fn setup() -> (World, Entity) {
        let mut world = World::new();
        world.init_resource::<Messages<PointerHits>>();
        world.insert_resource(UiPickingSettings {
            require_markers: true,
        });
        let camera = world
            .spawn((
                Camera::default(),
                RenderTarget::None {
                    size: UVec2::splat(100),
                },
                UiPickingCamera,
            ))
            .id();
        world.spawn((
            PointerId::Mouse,
            PointerLocation::new(Location {
                target: NormalizedRenderTarget::None {
                    width: 100,
                    height: 100,
                },
                position: Vec2::splat(50.),
            }),
        ));
        (world, camera)
    }

    /// A node under the pointer at stack `index`.
    fn node(world: &mut World, camera: Entity, index: u32, pickable: Option<Pickable>) -> Entity {
        let mut entity = world.spawn((
            ComputedStackIndex(index),
            ComputedNode {
                size: Vec2::splat(40.),
                ..Default::default()
            },
            UiGlobalTransform::from_xy(50., 50.),
            InheritedVisibility::VISIBLE,
            ComputedUiTargetCamera { camera },
        ));
        if let Some(pickable) = pickable {
            entity.insert(pickable);
        }
        entity.id()
    }

    fn picks(world: &mut World) -> Vec<Entity> {
        world.run_system_cached(ui_picking).unwrap();
        world
            .resource::<Messages<PointerHits>>()
            .iter_current_update_messages()
            .flat_map(|hits| hits.picks.iter().map(|(entity, _)| *entity))
            .collect()
    }

    #[test]
    fn hits_follow_the_stack_whatever_the_query_order() {
        let (mut world, camera) = setup();
        // Spawned top first, so query order is the reverse of the stack order.
        let top = node(&mut world, camera, 3, Some(PASS_THROUGH));
        let unmarked = node(&mut world, camera, 2, None);
        let middle = node(&mut world, camera, 1, Some(Pickable::default()));
        let bottom = node(&mut world, camera, 0, Some(PASS_THROUGH));
        // Claims `middle`'s index without being in the stack.
        node(&mut world, camera, 1, Some(Pickable::default()));
        world.insert_resource(UiStack {
            uinodes: vec![bottom, middle, unmarked, top],
            partition: core::iter::once(0..4).collect(),
        });

        // `middle` blocks the nodes below it; `unmarked` and the unstacked node are never reported.
        assert_eq!(picks(&mut world), [top, middle]);
    }

    #[test]
    fn slices_whose_root_is_gone_are_skipped() {
        let (mut world, camera) = setup();
        let root = node(&mut world, camera, 0, None);
        let orphan = node(&mut world, camera, 1, Some(PASS_THROUGH));
        let other = node(&mut world, camera, 2, Some(PASS_THROUGH));
        world.insert_resource(UiStack {
            uinodes: vec![root, orphan, other],
            partition: [0..2, 2..3].into_iter().collect(),
        });
        // Despawned after the stack was built, as by a system ordered before picking.
        world.despawn(root);

        assert_eq!(picks(&mut world), [other]);
    }
}
