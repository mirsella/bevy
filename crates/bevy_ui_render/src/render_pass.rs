use core::ops::Range;

use super::{ImageNodeBindGroups, UiBatch, UiMeta, UiViewTarget};

use crate::{SrgbUiCompositeBindGroup, SrgbUiCompositePipelineId, SrgbUiTexture, UiCameraView};
use bevy_color::LinearRgba;
use bevy_ecs::{
    entity::EntityHash,
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::FloatOrd;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::*,
    render_resource::{
        CachedRenderPipelineId, LoadOp, Operations, PipelineCache, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    },
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::*,
};
use indexmap::IndexMap;
use tracing::{error, warn};

pub fn ui_pass(
    world: &World,
    view: ViewQuery<(&UiCameraView, &ExtractedCamera), With<ViewTarget>>,
    ui_view_query: Query<(&ExtractedView, Option<&SrgbUiTexture>), With<UiViewTarget>>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    mut ctx: RenderContext,
) {
    let (ui_camera_view, camera) = view.into_inner();
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, srgb_texture)) = ui_view_query.get(ui_view_entity) else {
        return;
    };

    let Some(transparent_phase) =
        transparent_render_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };

    if transparent_phase.items.is_empty() {
        return;
    }

    let Some(srgb_texture) = srgb_texture else {
        warn!("Skipping UI pass for {ui_view_entity:?}: missing sRGB UI texture");
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &srgb_texture.texture.default_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::NONE.into()),
                store: StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "ui");

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = transparent_phase.render(&mut render_pass, world, ui_view_entity) {
        error!("Error encountered while rendering the ui phase {err:?}");
    }

    pass_span.end(&mut render_pass);
}

pub fn srgb_ui_composite_pass(
    view: ViewQuery<(&UiCameraView, &ViewTarget)>,
    ui_view_query: Query<
        (
            &ExtractedView,
            Option<&SrgbUiCompositeBindGroup>,
            Option<&SrgbUiCompositePipelineId>,
        ),
        With<UiViewTarget>,
    >,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (ui_camera_view, target) = view.into_inner();
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, composite_bind_group, pipeline_id)) = ui_view_query.get(ui_view_entity)
    else {
        return;
    };

    let Some(transparent_phase) =
        transparent_render_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };
    if transparent_phase.items.is_empty() {
        return;
    }

    let Some(composite_bind_group) = composite_bind_group else {
        warn!(
            "Skipping sRGB UI composite pass for {ui_view_entity:?}: missing composite bind group"
        );
        return;
    };
    let Some(pipeline_id) = pipeline_id else {
        warn!("Skipping sRGB UI composite pass for {ui_view_entity:?}: missing composite pipeline");
        return;
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("srgb_ui_composite_pass"),
        color_attachments: &[Some(target.get_unsampled_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &composite_bind_group.bind_group, &[]);
    render_pass.draw(0..3, 0..1);
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
    fn sort(items: &mut IndexMap<(Entity, MainEntity), TransparentUi, EntityHash>) {
        items.sort_by_key(|_, value| value.sort_key());
    }

    fn recalculate_sort_keys(
        _: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _: &ExtractedView,
    ) {
        // Sort keys are precalculated for UI phase items.
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
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<1>,
    DrawUiNode,
);

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

        let Some(bind_group) = image_bind_groups.values.get(&batch.image) else {
            error!(
                "Skipping UI draw: missing bind group for image {:?}",
                batch.image
            );
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, bind_group, &[]);
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
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
