use core::ops::Range;

use crate::composite_bounds::{composite_scissor, UiCompositeBounds};
use crate::{ui_mesh::UiMesh, ImageNodeBindGroups, UiBatch, UiVertex};
use crate::{UiCompositeScissor, UiViewTarget};

use crate::{SrgbUiCompositeBindGroup, SrgbUiCompositePipelineId, SrgbUiTexture, UiCameraView};
use bevy_color::LinearRgba;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::{FloatOrd, UVec2};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::*,
    render_resource::{
        CachedRenderPipelineId, Extent3d, LoadOp, Operations, Origin3d, PipelineCache,
        RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
    },
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::*,
};
use tracing::{error, warn};

pub fn ui_pass(
    world: &World,
    view: ViewQuery<(&UiCameraView, &ExtractedCamera), With<ViewTarget>>,
    ui_view_query: Query<(&ExtractedView, Option<&SrgbUiTexture>), With<UiViewTarget>>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    retained: Res<crate::retained_layer::RetainedUiLayers>,
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

    let layer = retained.layers.get(&extracted_view.retained_view_entity);
    if layer.is_some_and(|layer| layer.reuse) {
        return;
    }
    let mut rendered = true;
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    // In-pass timestamps can cover only the final tile on mobile tiled GPUs.
    // Encoder timestamps include the complete pass, including tile load/store.
    let boundary_span = diagnostics.map(|recorder| {
        recorder.time_span(
            ctx.command_encoder(),
            format!(
                "ui_boundary/camera_{}_{}/draw",
                camera.order, *extracted_view.retained_view_entity.main_entity
            ),
        )
    });

    {
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &srgb_texture
                    .raster
                    .as_ref()
                    .unwrap_or(&srgb_texture.texture)
                    .default_view,
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
        // Main-world camera identity survives temporary render-view recreation and separates
        // cameras with the same order. Construct diagnostic labels only when recording.
        let pass_span = diagnostics.map(|diagnostics| {
            diagnostics.pass_span(
                &mut render_pass,
                format!(
                    "ui/camera_{}_{}/draw",
                    camera.order, *extracted_view.retained_view_entity.main_entity
                ),
            )
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if let Err(err) = transparent_phase.render(&mut render_pass, world, ui_view_entity) {
            error!("Error encountered while rendering the ui phase {err:?}");
            rendered = false;
        }

        if let Some(pass_span) = pass_span {
            pass_span.end(&mut render_pass);
        }
    }
    if let (Some(rect), Some(raster)) = (srgb_texture.rect, srgb_texture.raster.as_ref()) {
        let mut source = raster.texture.as_image_copy();
        source.origin = Origin3d {
            x: rect.min.x,
            y: rect.min.y,
            z: 0,
        };
        ctx.command_encoder().copy_texture_to_texture(
            source,
            srgb_texture.texture.texture.as_image_copy(),
            Extent3d {
                width: rect.width(),
                height: rect.height(),
                depth_or_array_layers: 1,
            },
        );
    }
    if let Some(boundary_span) = boundary_span {
        boundary_span.end(ctx.command_encoder());
    }
    if let Some(layer) = layer {
        layer.valid.store(
            rendered && layer.contents.is_some(),
            core::sync::atomic::Ordering::Relaxed,
        );
    }
}

pub fn srgb_ui_composite_pass(
    view: ViewQuery<(
        &UiCameraView,
        &ViewTarget,
        &ExtractedCamera,
        Has<crate::output_fusion::Ready>,
    )>,
    ui_view_query: Query<
        (
            &ExtractedView,
            Option<&SrgbUiCompositeBindGroup>,
            Option<&SrgbUiCompositePipelineId>,
            Has<UiCompositeScissor>,
            &SrgbUiTexture,
        ),
        With<UiViewTarget>,
    >,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    pipeline_cache: Res<PipelineCache>,
    composite_bounds: Res<UiCompositeBounds>,
    mut ctx: RenderContext,
) {
    let (ui_camera_view, target, camera, fused_output) = view.into_inner();
    if fused_output {
        return;
    }
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, composite_bind_group, pipeline_id, scissor_enabled, srgb_texture)) =
        ui_view_query.get(ui_view_entity)
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

    let diagnostics = ctx.diagnostic_recorder();
    let boundary_span = diagnostics.as_deref().map(|recorder| {
        recorder.time_span(
            ctx.command_encoder(),
            format!(
                "ui_boundary/camera_{}_{}/srgb_composite",
                camera.order, *extracted_view.retained_view_entity.main_entity
            ),
        )
    });
    {
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("srgb_ui_composite_pass"),
            color_attachments: &[Some(target.get_unsampled_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pass_span = diagnostics.as_deref().map(|diagnostics| {
            diagnostics.pass_span(
                &mut render_pass,
                format!(
                    "ui/camera_{}_{}/srgb_composite",
                    camera.order, *extracted_view.retained_view_entity.main_entity
                ),
            )
        });

        if let Some(rect) = srgb_texture.rect {
            render_pass.set_viewport(
                rect.min.x as f32,
                rect.min.y as f32,
                rect.width() as f32,
                rect.height() as f32,
                0.,
                1.,
            );
        }
        let scissor = if scissor_enabled {
            let texture = target.main_texture();
            composite_scissor(
                composite_bounds
                    .0
                    .get(&extracted_view.retained_view_entity)
                    .copied(),
                extracted_view.viewport,
                UVec2::new(texture.width(), texture.height()),
            )
        } else {
            None
        };
        // Zero-sized scissors are invalid on some backends. Keep the pass and its
        // diagnostic span balanced, but omit the draw when no pixels can contribute.
        if scissor.is_none_or(|rect| !rect.is_empty()) {
            if let Some(rect) = scissor {
                render_pass.set_scissor_rect(rect.min.x, rect.min.y, rect.width(), rect.height());
            }
            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(0, &composite_bind_group.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        if let Some(pass_span) = pass_span {
            pass_span.end(&mut render_pass);
        }
    }
    if let Some(boundary_span) = boundary_span {
        boundary_span.end(ctx.command_encoder());
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
    fn sort(items: &mut SortedPhaseItems<Self>) {
        items.sort_by_key(|_, value| value.sort_key());
    }

    fn recalculate_sort_keys(_: &mut SortedPhaseItems<Self>, _: &ExtractedView) {
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

pub type DrawUi = (SetItemPipeline, DrawUiNode);

pub struct DrawUiNode;
impl<P: PhaseItem> RenderCommand<P> for DrawUiNode {
    type Param = (
        SRes<UiMesh<UiVertex>>,
        SRes<ImageNodeBindGroups>,
        SRes<PipelineCache>,
    );
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = (
        Read<UiBatch>,
        Option<Read<crate::UiImageInterior>>,
        Has<crate::UiEncodedBatch>,
    );

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        batch: Option<(&'w UiBatch, Option<&'w crate::UiImageInterior>, bool)>,
        (ui_meta, image_bind_groups, pipelines): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some((batch, interior, encoded)) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(view_bind_group) = ui_meta.view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        let bind_group = if encoded {
            assert!(
                interior.is_none(),
                "encoded images cannot share an edge/interior binding"
            );
            image_bind_groups
                .encoded
                .get(&batch.image)
                .map(|(_, binding)| binding)
        } else {
            image_bind_groups.values.get(&batch.image)
        };
        let Some(bind_group) = bind_group else {
            error!(
                "Skipping UI draw: missing bind group for image {:?}",
                batch.image
            );
            return RenderCommandResult::Skip;
        };

        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        pass.set_bind_group(0, view_bind_group, &[view_uniform.offset]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(
            indices.slice(..),
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        if let Some(interior) = interior {
            let Some(pipeline) = pipelines
                .into_inner()
                .get_render_pipeline(interior.pipeline)
            else {
                return RenderCommandResult::Skip;
            };
            assert!(
                interior.range.start >= batch.range.start && interior.range.end == batch.range.end,
                "image interior must be the final nonoverlapping part of its own batch"
            );
            if batch.range.start < interior.range.start {
                pass.draw_indexed(batch.range.start..interior.range.start, 0, 0..1);
            }
            pass.set_render_pipeline(pipeline);
            pass.draw_indexed(interior.range.clone(), 0, 0..1);
        } else {
            pass.draw_indexed(batch.range.clone(), 0, 0..1);
        }
        RenderCommandResult::Success
    }
}
