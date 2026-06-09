use core::ops::Range;

use super::{ImageNodeBindGroups, UiBatch, UiMeta, UiViewTarget};

use crate::{SrgbUiCompositeBindGroup, SrgbUiCompositePipelineId, SrgbUiTexture, UiCameraView};
use bevy_color::LinearRgba;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::{FloatOrd, Rect, Vec2};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_graph::*,
    render_phase::*,
    render_resource::{
        CachedRenderPipelineId, LoadOp, Operations, PipelineCache, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    },
    renderer::*,
    sync_world::MainEntity,
    view::*,
};
use tracing::error;

pub struct UiPassNode {
    ui_view_query: QueryState<(
        &'static ExtractedView,
        &'static UiViewTarget,
        &'static SrgbUiTexture,
    )>,
    ui_view_target_query: QueryState<(&'static ViewTarget, &'static ExtractedCamera)>,
    ui_camera_view_query: QueryState<&'static UiCameraView>,
}

impl UiPassNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            ui_view_query: world.query_filtered(),
            ui_view_target_query: world.query(),
            ui_camera_view_query: world.query(),
        }
    }
}

impl Node for UiPassNode {
    fn update(&mut self, world: &mut World) {
        self.ui_view_query.update_archetypes(world);
        self.ui_view_target_query.update_archetypes(world);
        self.ui_camera_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Extract the UI view.
        let input_view_entity = graph.view_entity();

        let Some(transparent_render_phases) =
            world.get_resource::<ViewSortedRenderPhases<TransparentUi>>()
        else {
            return Ok(());
        };

        // Query the UI view components.
        let Ok((view, ui_view_target, srgb_texture)) =
            self.ui_view_query.get_manual(world, input_view_entity)
        else {
            return Ok(());
        };

        let Ok((_target, camera)) = self
            .ui_view_target_query
            .get_manual(world, ui_view_target.0)
        else {
            return Ok(());
        };

        let Some(transparent_phase) = transparent_render_phases.get(&view.retained_view_entity)
        else {
            return Ok(());
        };

        if transparent_phase.items.is_empty() {
            return Ok(());
        }

        let diagnostics = render_context.diagnostic_recorder();

        // use the UI view entity if it is defined
        let view_entity = if let Ok(ui_camera_view) = self
            .ui_camera_view_query
            .get_manual(world, input_view_entity)
        {
            ui_camera_view.0
        } else {
            input_view_entity
        };
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &srgb_texture.texture.default_view,
                resolve_target: srgb_texture
                    .resolve_texture
                    .as_ref()
                    .map(|t| &*t.default_view),
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::NONE.into()),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let pass_span = diagnostics.pass_span(&mut render_pass, "ui");

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }
        if let Err(err) = transparent_phase.render(&mut render_pass, world, view_entity) {
            error!("Error encountered while rendering the ui phase {err:?}");
        }

        pass_span.end(&mut render_pass);

        Ok(())
    }
}

pub struct SrgbUiCompositePassNode {
    query: QueryState<(
        &'static UiViewTarget,
        &'static SrgbUiTexture,
        &'static SrgbUiCompositeBindGroup,
        &'static SrgbUiCompositePipelineId,
        &'static ExtractedView,
    )>,
    target_query: QueryState<&'static ViewTarget>,
}

impl SrgbUiCompositePassNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query(),
            target_query: world.query(),
        }
    }
}

impl Node for SrgbUiCompositePassNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
        self.target_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let Ok((ui_view_target, _srgb_texture, composite_bind_group, pipeline_id, view)) =
            self.query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        // Check if we should run the composite pass
        let Some(transparent_render_phases) =
            world.get_resource::<ViewSortedRenderPhases<TransparentUi>>()
        else {
            return Ok(());
        };

        let Some(transparent_phase) = transparent_render_phases.get(&view.retained_view_entity)
        else {
            return Ok(());
        };

        if transparent_phase.items.is_empty() {
            return Ok(());
        }

        let Ok(target) = self.target_query.get_manual(world, ui_view_target.0) else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("srgb_ui_composite_pass"),
            color_attachments: &[Some(target.get_unsampled_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &composite_bind_group.bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

pub struct TransparentUi {
    pub sort_key: FloatOrd,
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub index: usize,
    pub indexed: bool,
    /// Clipping rect in UI view coordinates (physical pixels).
    ///
    /// When set, the draw function will apply it using a GPU scissor rect.
    pub clip: Option<Rect>,
}

impl PhaseItem for TransparentUi {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TransparentUi {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        items.sort_by_key(SortedPhaseItem::sort_key);
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub type DrawUi = (
    SetItemPipeline,
    SetUiScissorRect,
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<1>,
    DrawUiNode,
);

pub struct SetUiScissorRect;
impl RenderCommand<TransparentUi> for SetUiScissorRect {
    type Param = ();
    type ViewQuery = Read<ExtractedView>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &TransparentUi,
        view: &'w ExtractedView,
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let view_width = view.viewport.z;
        let view_height = view.viewport.w;

        // wgpu validation requires width/height > 0
        if view_width == 0 || view_height == 0 {
            return RenderCommandResult::Skip;
        }

        let (x0, y0, x1, y1) = if let Some(clip) = item.clip {
            // Clamp clip to the view bounds.
            let max = Vec2::new(view_width as f32, view_height as f32);
            let clip_min = clip.min.clamp(Vec2::ZERO, max);
            let clip_max = clip.max.clamp(Vec2::ZERO, max);

            // Convert float clip rect to an integer scissor rect.
            //
            // We round based on pixel centers (x + 0.5, y + 0.5) to better match how
            // rasterization works for non-MSAA rendering.
            let eps = 1e-4;
            let x0 = (clip_min.x - 0.5 - eps).ceil() as i32;
            let y0 = (clip_min.y - 0.5 - eps).ceil() as i32;
            let x1 = (clip_max.x + 0.5 + eps).floor() as i32;
            let y1 = (clip_max.y + 0.5 + eps).floor() as i32;
            (x0, y0, x1, y1)
        } else {
            (0, 0, view_width as i32, view_height as i32)
        };

        if x1 <= x0 || y1 <= y0 {
            return RenderCommandResult::Skip;
        }

        pass.set_scissor_rect(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32);
        RenderCommandResult::Success
    }
}

pub struct SetUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<I> {
    type Param = SRes<UiMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
pub struct SetUiTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<I> {
    type Param = SRes<ImageNodeBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, image_bind_groups.values.get(&batch.image).unwrap(), &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawUiNode;
impl<P: PhaseItem> RenderCommand<P> for DrawUiNode {
    type Param = SRes<UiMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));
        // Define how to "connect" the vertices
        pass.set_index_buffer(
            indices.slice(..),
            0,
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
