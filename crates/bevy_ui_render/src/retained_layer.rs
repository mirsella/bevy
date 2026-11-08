//! Opt-in bounded standard-UI layers. Unknown draws retain the reference path.
use crate::composite_bounds::composite_scissor;
use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_math::{URect, UVec2, UVec4};
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::{CachedRenderPipelineId, SamplerId, TextureViewId},
    texture::CachedTexture,
    view::RetainedViewEntity,
};

/// Bounds a camera's standard UI intermediate, retaining its separate blend layer.
///
/// Custom/sliced/surface/shadow draws, offset viewports and non-finite geometry
/// fall back to the reference full-size target. Caching also rejects images that
/// can be written by render passes or storage operations, except completed
/// versioned text-group tiles.
/// Full-view rasterization is preserved; occupied texels are copied to a smaller
/// composite texture. This is opt-in: avoiding transparent sRGB
/// blends can expose target-format rounding differences on some drivers.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct UiRetainedLayer {
    /// Reuse unchanged pixels, not just the allocation. All draw inputs are compared.
    pub cache: bool,
}

/// Render-world acknowledgement of an opted-in layer's actual allocation/reuse.
#[derive(Component, Clone, Copy, Debug)]
pub struct UiRetainedLayerStatus {
    pub rect: URect,
    pub full_size: UVec2,
    pub reused: bool,
    pub cache_state: &'static str,
}

#[derive(Clone, PartialEq)]
pub(crate) struct Contents {
    pub vertices: Vec<u8>,
    pub indices: Vec<u32>,
    pub pipelines: Vec<CachedRenderPipelineId>,
    pub images: Vec<(AssetId<Image>, TextureViewId, SamplerId, u64)>,
}

pub(crate) struct Plan {
    pub rect: URect,
    pub full_size: UVec2,
    pub contents: Option<Contents>,
}

pub(crate) struct Layer {
    pub texture: CachedTexture,
    pub raster: CachedTexture,
    pub full_size: UVec2,
    pub rect: URect,
    pub format: bevy_render::render_resource::TextureFormat,
    pub contents: Option<Contents>,
    pub valid: core::sync::atomic::AtomicBool,
    pub reuse: bool,
}

#[derive(Resource, Default)]
#[doc(hidden)]
pub struct RetainedUiLayers {
    pub(crate) plans: HashMap<RetainedViewEntity, Plan>,
    pub(crate) layers: HashMap<RetainedViewEntity, Layer>,
}

impl RetainedUiLayers {
    pub fn begin_frame(&mut self, changed_images: &[AssetId<Image>]) {
        self.plans.clear();
        for layer in self.layers.values_mut() {
            if layer.contents.as_ref().is_some_and(|c| {
                c.images
                    .iter()
                    .any(|(id, _, _, _)| changed_images.contains(id))
            }) {
                layer
                    .valid
                    .store(false, core::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// Integer-pixel crop with the same bilinear-footprint guard as composite bounds.
pub(crate) fn crop(
    bounds: Option<&crate::composite_bounds::CompositeBounds>,
    viewport: UVec4,
    target: UVec2,
) -> Option<URect> {
    let tight = composite_scissor(bounds.copied(), viewport, target).filter(|r| !r.is_empty())?;
    // Power-of-two dimensions stabilize texel-centre interpolation and reduce
    // allocation churn when a label changes by a few pixels. Keep integer origin.
    let size = UVec2::new(
        tight.width().next_power_of_two(),
        tight.height().next_power_of_two(),
    )
    .min(target);
    let origin = tight.min.min(target - size);
    Some(URect::from_corners(origin, origin + size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite_bounds::CompositeBounds;
    use bevy_math::{Rect, Vec2};

    #[test]
    fn crop_rejects_unknown_empty_and_offset_views() {
        let full = UVec2::new(100, 200);
        let bounds = CompositeBounds::FiniteRect(Rect::from_corners(
            Vec2::new(10.5, 180.5),
            Vec2::new(90., 199.),
        ));
        assert_eq!(
            crop(Some(&bounds), UVec4::new(0, 0, 100, 200), full).unwrap(),
            URect::from_corners(UVec2::new(0, 168), UVec2::new(100, 200))
        );
        assert!(crop(Some(&bounds), UVec4::new(1, 0, 100, 200), full).is_none());
        assert!(crop(
            Some(&CompositeBounds::Full),
            UVec4::new(0, 0, 100, 200),
            full
        )
        .is_none());
        assert!(crop(
            Some(&CompositeBounds::Empty),
            UVec4::new(0, 0, 100, 200),
            full
        )
        .is_none());
    }

    #[test]
    fn content_equality_tracks_geometry_order_and_program() {
        let original = Contents {
            vertices: vec![1, 2],
            indices: vec![0, 1],
            pipelines: vec![CachedRenderPipelineId::new(0)],
            images: vec![],
        };
        assert!(original == original.clone());
        let mut changed = original.clone();
        changed.vertices[0] = 2;
        assert!(changed != original);
        changed = original.clone();
        changed.indices.reverse();
        assert!(changed != original);
        changed = original.clone();
        changed.pipelines.clear();
        assert!(changed != original);
    }
}
