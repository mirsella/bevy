use bevy_ecs::resource::Resource;
use bevy_math::{Rect, URect, UVec2, UVec4, Vec2};
use bevy_platform::collections::HashMap;
use bevy_render::{render_phase::DrawFunctionId, view::RetainedViewEntity};

use crate::UiVertex;

/// Rebuilt during UI preparation; never reuse a previous frame's coverage.
#[derive(Resource, Default)]
pub struct UiCompositeBounds(pub(crate) HashMap<RetainedViewEntity, CompositeBounds>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CompositeBounds {
    Empty,
    FiniteRect(Rect),
    /// Unknown draw functions or nonfinite geometry require the reference draw.
    Full,
}

impl CompositeBounds {
    pub(crate) fn for_draw_functions(
        draw_functions: impl IntoIterator<Item = DrawFunctionId>,
        draw_ui: DrawFunctionId,
    ) -> Self {
        if draw_functions.into_iter().any(|draw| draw != draw_ui) {
            Self::Full
        } else {
            Self::Empty
        }
    }

    pub(crate) fn include_vertices(&mut self, vertices: &[UiVertex]) {
        if matches!(self, Self::Full) {
            return;
        }
        for vertex in vertices {
            // Check every component before min/max, which can silently ignore NaN.
            if !vertex.position.iter().all(|value| value.is_finite()) {
                *self = Self::Full;
                return;
            }
            let position = Vec2::new(vertex.position[0], vertex.position[1]);
            match self {
                Self::Empty => *self = Self::FiniteRect(Rect::from_corners(position, position)),
                Self::FiniteRect(rect) => {
                    rect.min = rect.min.min(position);
                    rect.max = rect.max.max(position);
                }
                Self::Full => unreachable!("full coverage stops vertex accumulation"),
            }
        }
    }
}

/// `None` retains the full-screen reference draw; an empty rectangle skips only
/// the draw, not the render pass. Coordinates are UI texture pixels, top-left down.
pub(crate) fn composite_scissor(
    bounds: Option<CompositeBounds>,
    viewport: UVec4,
    target_size: UVec2,
) -> Option<URect> {
    // The UI texture and composite attachment must have the same pixel mapping.
    // Checking actual sizes also covers explicitly configured full-size viewports.
    if target_size.x == 0
        || target_size.y == 0
        || viewport != UVec4::new(0, 0, target_size.x, target_size.y)
    {
        return None;
    }
    match bounds? {
        CompositeBounds::Empty => Some(URect::default()),
        CompositeBounds::Full => None,
        CompositeBounds::FiniteRect(rect) => {
            if !rect.min.is_finite() || !rect.max.is_finite() || !rect.min.cmple(rect.max).all() {
                return None;
            }
            let size = target_size.as_vec2();
            // The composite samples linearly, so retain a pixel beyond the rounded
            // geometry bounds. Clamp in floating point before converting to u32.
            let min = (rect.min.floor() - Vec2::ONE).clamp(Vec2::ZERO, size);
            let max = (rect.max.ceil() + Vec2::ONE).clamp(Vec2::ZERO, size);
            Some(URect {
                min: min.as_uvec2(),
                max: max.as_uvec2(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(100, 80);
    const VIEWPORT: UVec4 = UVec4::new(0, 0, SIZE.x, SIZE.y);

    fn vertex(position: [f32; 3]) -> UiVertex {
        UiVertex {
            position,
            ..bytemuck::Zeroable::zeroed()
        }
    }

    #[test]
    fn emitted_vertices_union_across_items_including_z_validation() {
        let mut bounds = CompositeBounds::Empty;
        bounds.include_vertices(&[]);
        assert_eq!(bounds, CompositeBounds::Empty);
        bounds.include_vertices(&[vertex([20.25, 30.5, 0.]), vertex([40., 10., 0.])]);
        bounds.include_vertices(&[]);
        bounds.include_vertices(&[vertex([-2., 70., 0.])]);
        assert_eq!(
            bounds,
            CompositeBounds::FiniteRect(Rect::new(-2., 10., 40., 70.))
        );
        assert_eq!(
            composite_scissor(Some(bounds), VIEWPORT, SIZE),
            Some(URect::new(0, 9, 41, 71))
        );
    }

    #[test]
    fn fractional_bounds_expand_for_linear_filtering_and_clamp() {
        for (rect, expected) in [
            (
                Rect::new(10.25, 20.75, 30.5, 40.1),
                URect::new(9, 19, 32, 42),
            ),
            (Rect::new(10., 20., 30., 40.), URect::new(9, 19, 31, 41)),
            (Rect::new(-4.5, -3., 120., 90.), URect::new(0, 0, 100, 80)),
            (Rect::new(-20., 10., -5., 20.), URect::new(0, 9, 0, 21)),
            (
                Rect::new(110., 90., 120., 100.),
                URect::new(100, 80, 100, 80),
            ),
            (
                Rect::new(-f32::MAX, -f32::MAX, f32::MAX, f32::MAX),
                URect::new(0, 0, 100, 80),
            ),
        ] {
            assert_eq!(
                composite_scissor(Some(CompositeBounds::FiniteRect(rect)), VIEWPORT, SIZE),
                Some(expected)
            );
        }
    }

    #[test]
    fn nonfinite_positions_force_full_coverage_before_min_max() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for axis in 0..3 {
                for already_finite in [false, true] {
                    let mut bounds = CompositeBounds::Empty;
                    if already_finite {
                        bounds.include_vertices(&[vertex([10., 20., 0.])]);
                    }
                    let mut position = [1., 2., 0.];
                    position[axis] = invalid;
                    bounds.include_vertices(&[vertex(position)]);
                    bounds.include_vertices(&[vertex([30., 40., 0.])]);
                    assert_eq!(bounds, CompositeBounds::Full);
                    assert_eq!(composite_scissor(Some(bounds), VIEWPORT, SIZE), None);
                }
            }
        }
    }

    #[test]
    fn malformed_rectangles_fall_back() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for corner in 0..2 {
                for axis in 0..2 {
                    let mut corners = [Vec2::ZERO, Vec2::splat(10.)];
                    corners[corner][axis] = invalid;
                    let bounds = CompositeBounds::FiniteRect(Rect {
                        min: corners[0],
                        max: corners[1],
                    });
                    assert_eq!(composite_scissor(Some(bounds), VIEWPORT, SIZE), None);
                }
            }
        }
        let inverted = CompositeBounds::FiniteRect(Rect {
            min: Vec2::splat(20.),
            max: Vec2::splat(10.),
        });
        assert_eq!(composite_scissor(Some(inverted), VIEWPORT, SIZE), None);
    }

    #[test]
    fn foreign_draws_force_fallback_even_without_emitted_standard_vertices() {
        let standard = DrawFunctionId(0);
        let foreign = DrawFunctionId(1);
        for draws in [
            vec![foreign],
            vec![standard, foreign],
            vec![foreign, standard],
        ] {
            let mut bounds = CompositeBounds::for_draw_functions(draws, standard);
            bounds.include_vertices(&[]);
            bounds.include_vertices(&[vertex([10., 20., 0.])]);
            assert_eq!(bounds, CompositeBounds::Full);
            assert_eq!(composite_scissor(Some(bounds), VIEWPORT, SIZE), None);
        }
        for draws in [vec![], vec![standard], vec![standard, standard]] {
            let bounds = CompositeBounds::for_draw_functions(draws, standard);
            assert_eq!(bounds, CompositeBounds::Empty);
            assert_eq!(
                composite_scissor(Some(bounds), VIEWPORT, SIZE),
                Some(URect::default())
            );
        }
        assert_eq!(composite_scissor(None, VIEWPORT, SIZE), None);
    }

    #[test]
    fn offset_viewports_and_mismatched_attachment_sizes_fall_back_even_when_empty() {
        for bounds in [
            CompositeBounds::Empty,
            CompositeBounds::FiniteRect(Rect::new(10., 20., 30., 40.)),
        ] {
            for viewport in [
                UVec4::new(1, 0, SIZE.x, SIZE.y),
                UVec4::new(0, 1, SIZE.x, SIZE.y),
                UVec4::new(0, 0, SIZE.x - 1, SIZE.y),
                UVec4::new(0, 0, SIZE.x, SIZE.y - 1),
            ] {
                assert_eq!(composite_scissor(Some(bounds), viewport, SIZE), None);
            }
            assert_eq!(
                composite_scissor(Some(bounds), VIEWPORT, SIZE + UVec2::ONE),
                None
            );
            assert_eq!(
                composite_scissor(Some(bounds), UVec4::ZERO, UVec2::ZERO),
                None
            );
        }
    }
}
