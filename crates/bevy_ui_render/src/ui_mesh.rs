use crate::render_pass::TransparentUi;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    resource::Resource,
    system::{
        lifetimeless::{Read, SRes},
        SystemParamItem,
    },
};
use bevy_render::{
    render_phase::{
        PhaseItem, RenderCommand, RenderCommandResult, SortedPhaseItems, TrackedRenderPass,
    },
    render_resource::{BindGroup, BufferUsages, IndexFormat, RawBufferVec},
    view::ViewUniformOffset,
};
use bytemuck::Pod;
use core::{marker::PhantomData, ops::Range};

/// Shared draw storage for UI meshes. Vertex types keep each pipeline's
/// buffers distinct without duplicating their allocation and draw implementation.
#[derive(Resource)]
pub struct UiMesh<V: Pod + Send + Sync> {
    pub vertices: RawBufferVec<V>,
    pub indices: RawBufferVec<u32>,
    pub view_bind_group: Option<BindGroup>,
}

impl<V: Pod + Send + Sync> Default for UiMesh<V> {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            view_bind_group: None,
        }
    }
}

pub struct DrawUiMesh<V>(PhantomData<V>);

impl<P: PhaseItem, V: Pod + Send + Sync> RenderCommand<P> for DrawUiMesh<V> {
    type Param = SRes<UiMesh<V>>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = Read<UiIndexedBatch>;

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        batch: Option<&'w UiIndexedBatch>,
        mesh: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh = mesh.into_inner();
        let Some(view_bind_group) = mesh.view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let Some(vertices) = mesh.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = mesh.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };
        pass.set_bind_group(0, view_bind_group, &[view_uniform.offset]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}

/// Index-buffer range for a gradient or shadow draw. All other draw data is per vertex.
#[derive(Component)]
pub struct UiIndexedBatch {
    pub range: Range<u32>,
}

impl<V: Pod + Send + Sync> UiMesh<V> {
    pub fn push_triangle_fan(&mut self, vertices: impl ExactSizeIterator<Item = V>) {
        let base = self.vertices.len() as u32;
        let count = vertices.len() as u32;
        self.vertices.extend(vertices);
        for i in 2..count {
            self.indices.extend([base, base + i - 1, base + i]);
        }
    }

    /// Append emitted geometry to the current draw only when no other phase item
    /// intervenes. Reset `first` for each view; empty geometry never creates a batch.
    pub fn finish_batch(
        &self,
        items: &mut SortedPhaseItems<TransparentUi>,
        first: &mut Option<usize>,
        item_index: usize,
        start: u32,
        batches: &mut Vec<(Entity, UiIndexedBatch)>,
    ) {
        let end = self.indices.len() as u32;
        if start == end {
            return;
        }
        if let Some(first) = *first
            && items[first].batch_range.end == item_index as u32
            && items[first].pipeline == items[item_index].pipeline
            && items[first].draw_function == items[item_index].draw_function
        {
            batches.last_mut().unwrap().1.range.end = end;
            items[first].batch_range.end += 1;
        } else {
            batches.push((
                items[item_index].entity.0,
                UiIndexedBatch { range: start..end },
            ));
            items[item_index].batch_range = item_index as u32..item_index as u32 + 1;
            *first = Some(item_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::FloatOrd;
    use bevy_render::{
        render_phase::{DrawFunctionId, PhaseItemExtraIndex},
        render_resource::CachedRenderPipelineId,
    };

    fn item(index: usize, pipeline: usize, draw: u32) -> TransparentUi {
        let entity = Entity::from_raw_u32(index as u32).unwrap();
        TransparentUi {
            sort_key: FloatOrd(index as f32),
            entity: (entity, entity.into()),
            pipeline: CachedRenderPipelineId::new(pipeline),
            draw_function: DrawFunctionId(draw),
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            index,
            indexed: true,
        }
    }

    #[test]
    fn batches_preserve_order_pipeline_draw_and_view_boundaries() {
        let mut items: SortedPhaseItems<_> = [
            item(0, 0, 0),
            item(1, 0, 0),
            item(2, 1, 0),
            item(3, 1, 1),
            item(4, 1, 1),
            item(5, 1, 1),
            item(6, 1, 1),
        ]
        .into_iter()
        .map(|item| (item.entity, item))
        .collect();
        let mut first = None;
        let mut batches = Vec::new();
        let mut mesh = UiMesh::<u32>::default();
        for (index, range) in [
            (0, 0..6),
            (1, 6..18),
            (2, 18..24),
            (3, 24..30),
            (4, 30..30),
            (5, 30..36),
        ] {
            mesh.indices.extend(range.clone());
            mesh.finish_batch(&mut items, &mut first, index, range.start, &mut batches);
        }
        // Item 4 belongs to another preparer, or emitted no geometry. Never skip over it.
        assert_eq!(
            items
                .values()
                .map(|item| item.batch_range.clone())
                .collect::<Vec<_>>(),
            [0..2, 0..0, 2..3, 3..4, 0..0, 5..6, 0..0]
        );
        assert_eq!(
            batches
                .iter()
                .map(|(_, b)| b.range.clone())
                .collect::<Vec<_>>(),
            [0..18, 18..24, 24..30, 30..36]
        );
        first = None;
        mesh.indices.extend(36..42);
        mesh.finish_batch(&mut items, &mut first, 6, 36, &mut batches);
        assert_eq!(batches.len(), 5);
        assert_eq!(items[6].batch_range, 6..7);
    }

    #[test]
    fn triangle_fans_keep_winding_and_offset_each_polygon() {
        let mut mesh = UiMesh::<u32>::default();
        mesh.push_triangle_fan(10..14);
        mesh.push_triangle_fan(20..23);
        assert_eq!(mesh.vertices.len(), 7);
        assert_eq!(mesh.vertices.get(4), Some(&20));
        assert_eq!(
            (0..mesh.indices.len() as u32)
                .map(|i| *mesh.indices.get(i).unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 2, 0, 2, 3, 4, 5, 6]
        );
    }
}
