use alloc::vec::Vec;
use bevy_ecs::{message::MessageCursor, prelude::*};
use bevy_platform::collections::HashMap;

use crate::{
    backend::PointerHits,
    pointer::{PointerAction, PointerId, PointerInput, PointerInputBatch},
    Picking,
};

#[derive(Clone, Copy)]
pub(super) enum BatchPosition {
    // Indices into PointerInputBatch. Movement samples share their last hit position;
    // button transitions lock that position for the rest of this pass.
    Moving(usize),
    Fixed(usize),
    Cancelled,
}

pub(super) fn run(
    world: &mut World,
    mut cursor: Local<MessageCursor<PointerInput>>,
    mut pending: Local<Vec<PointerInput>>,
    mut positions: Local<HashMap<PointerId, BatchPosition>>,
) {
    pending.extend(
        cursor
            .read(world.resource::<Messages<PointerInput>>())
            .cloned(),
    );
    let mut pending = pending.drain(..).peekable();
    loop {
        positions.clear();
        {
            let mut batch = world.resource_mut::<PointerInputBatch>();
            batch.inputs.clear();
            while let Some(next) = pending.peek() {
                let previous = positions.get(&next.pointer_id).copied();
                let moving = matches!(next.action, PointerAction::Move { .. });
                let cancelled = matches!(next.action, PointerAction::Cancel);
                match previous {
                    Some(BatchPosition::Cancelled) => break,
                    Some(_) if cancelled => break,
                    Some(BatchPosition::Moving(index)) if !moving => {
                        if batch.inputs[index].location != next.location {
                            break;
                        }
                    }
                    Some(BatchPosition::Fixed(index)) => {
                        if batch.inputs[index].location != next.location {
                            break;
                        }
                    }
                    _ => {}
                }
                let position = if cancelled {
                    BatchPosition::Cancelled
                } else if moving && !matches!(previous, Some(BatchPosition::Fixed(_))) {
                    BatchPosition::Moving(batch.inputs.len())
                } else {
                    BatchPosition::Fixed(batch.inputs.len())
                };
                positions.insert(next.pointer_id, position);
                batch
                    .inputs
                    .push(pending.next().expect("peeked input exists"));
            }
        }

        // Hits belong to this pass, not to a previous position or frame. Backends
        // must run in Picking; PointerInput messages remain available to the app.
        world.resource_mut::<Messages<PointerHits>>().clear();
        world.run_schedule(Picking);
        if pending.peek().is_none() {
            break;
        }
    }
}
