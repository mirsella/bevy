use bevy_math::{vec2, Affine2, Rect, Vec2};
use bevy_sprite::BorderRect;
use bevy_ui::{CalculatedClip, ResolvedBorderRadius};
use smallvec::{smallvec, SmallVec};

const INLINE_CAPACITY: usize = 16;

/// Conservative rejection before extraction. Check each clip in its own coordinate space,
/// so rotated clips and one-axis overflow clipping work without constructing a polygon.
pub(crate) fn rect_is_clipped(
    clip: Option<&CalculatedClip>,
    transform: Affine2,
    size: Vec2,
) -> bool {
    let Some(clip) = clip else {
        return false;
    };
    let Some(regions) = clip.rects() else {
        return true;
    };
    let corners =
        crate::QUAD_VERTEX_POSITIONS.map(|point| transform.transform_point2(point * size));
    regions.iter().any(|region| {
        let bounds = corners.iter().fold(Rect::EMPTY, |bounds, &point| {
            bounds.union_point(region.world_to_clip_local.transform_point2(point))
        });
        bounds.min.cmpgt(region.rect.max).any() || bounds.max.cmplt(region.rect.min).any()
    })
}

/// Interior beyond rounded corners, borders and the shader's half-pixel antialiasing band.
pub(crate) fn rounded_inner_rect(
    size: Vec2,
    border: BorderRect,
    radius: ResolvedBorderRadius,
) -> Option<Rect> {
    let inset = Vec2::splat(
        radius
            .top_left
            .max(radius.top_right)
            .max(radius.bottom_left)
            .max(radius.bottom_right)
            + 1.,
    );
    let rect = Rect {
        min: -0.5 * size + border.min_inset + inset,
        max: 0.5 * size - border.max_inset - inset,
    };
    (!rect.is_empty()).then_some(rect)
}

/// Partition a centered rectangle around a hole, without overlapping translucent draws.
pub(crate) fn rect_without_hole(bounds: Vec2, hole: Option<Rect>) -> SmallVec<[Rect; 4]> {
    let outer = Rect::from_center_size(Vec2::ZERO, bounds);
    let Some(inner) = hole
        .map(|rect| outer.intersect(rect))
        .filter(|rect| !rect.is_empty())
    else {
        return smallvec![outer];
    };
    [
        Rect {
            min: outer.min,
            max: vec2(outer.max.x, inner.min.y),
        },
        Rect {
            min: vec2(outer.min.x, inner.max.y),
            max: outer.max,
        },
        Rect {
            min: vec2(outer.min.x, inner.min.y),
            max: vec2(inner.min.x, inner.max.y),
        },
        Rect {
            min: vec2(inner.max.x, inner.min.y),
            max: vec2(outer.max.x, inner.max.y),
        },
    ]
    .into_iter()
    .filter(|rect| !rect.is_empty())
    .collect()
}

/// Clips a polygon using the [Sutherland-Hodgman](https://en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm)
/// algorithm and interpolates the attribute values.
///
/// # Arguments
/// * `clip` - The clipping regions to apply. If `None`, the input polygon is returned unchanged.
/// * `vertices` - The polygon vertices and associated attribute values. The vertices should be in boundary order (either direction), forming a convex polygon.
/// * `interpolate` - Interpolates attribute values for new vertices at clip intersections.
///
/// Returns the resulting clipped polygon as a list of vertices forming a triangle fan.
pub fn clip_polygon<T: Copy>(
    clip: Option<&CalculatedClip>,
    vertices: &[(Vec2, T)],
    interpolate: impl Fn(T, T, f32) -> T + Copy,
) -> SmallVec<[(Vec2, T); INLINE_CAPACITY]> {
    // If less than 3 vertices, there's no visible region to clip.
    if vertices.len() < 3 {
        return SmallVec::new();
    }

    let Some(clip) = clip else {
        return SmallVec::from_slice(vertices);
    };
    let Some(rects) = clip.rects() else {
        return SmallVec::new();
    };

    let mut visible_region = SmallVec::from_slice(vertices);
    let mut scratch = SmallVec::new();

    for region in rects {
        if visible_region.len() < 3 {
            break;
        }

        let bounds = visible_region
            .iter()
            .fold(Rect::EMPTY, |bounds, &(point, _)| {
                bounds.union_point(region.world_to_clip_local.transform_point2(point))
            });
        if bounds.min.cmpgt(region.rect.max).any() || bounds.max.cmplt(region.rect.min).any() {
            return SmallVec::new();
        }

        for (edge, distance_normal, needs_clip) in [
            (
                -region.rect.min.x,
                Vec2::X,
                bounds.min.x < region.rect.min.x,
            ),
            (
                region.rect.max.x,
                Vec2::NEG_X,
                bounds.max.x > region.rect.max.x,
            ),
            (
                region.rect.max.y,
                Vec2::NEG_Y,
                bounds.max.y > region.rect.max.y,
            ),
            (
                -region.rect.min.y,
                Vec2::Y,
                bounds.min.y < region.rect.min.y,
            ),
        ] {
            if needs_clip && edge.is_finite() {
                edge_clip(
                    &visible_region,
                    &mut scratch,
                    region.world_to_clip_local,
                    edge,
                    distance_normal,
                    interpolate,
                );
                core::mem::swap(&mut visible_region, &mut scratch);
            }
        }
    }

    if visible_region.len() < 3 {
        visible_region.clear();
    }

    visible_region
}

fn edge_clip<T: Copy>(
    input: &[(Vec2, T)],
    output: &mut SmallVec<[(Vec2, T); INLINE_CAPACITY]>,
    world_to_clip: Affine2,
    edge: f32,
    distance_normal: Vec2,
    interpolate: impl Fn(T, T, f32) -> T + Copy,
) {
    output.clear();

    let Some(mut previous) = input.last().copied() else {
        return;
    };
    // Transform the half-plane once, rather than every vertex on every clipping edge.
    let normal = world_to_clip.matrix2.transpose() * distance_normal;
    let edge = edge + world_to_clip.translation.dot(distance_normal);
    let mut previous_distance = previous.0.dot(normal) + edge;
    let mut is_previous_visible = 0. <= previous_distance;

    for &vertex in input {
        let distance = vertex.0.dot(normal) + edge;
        let is_visible = 0. <= distance;
        // If inside != previous_inside, the previous -> vertex edge crossed the clip rect edge and we
        // add a new vertex at the intersection.
        if is_visible != is_previous_visible {
            let t = previous_distance / (previous_distance - distance);
            output.push((
                previous.0.lerp(vertex.0, t),
                interpolate(previous.1, vertex.1, t),
            ));
        }
        if is_visible {
            output.push(vertex);
        }
        previous = vertex;
        previous_distance = distance;
        is_previous_visible = is_visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Mat2, Rot2};
    use bevy_ui::CalculatedClipRect;

    #[test]
    fn early_rejection_respects_rotated_and_one_axis_clips() {
        let clip_transform = Affine2::from_angle(core::f32::consts::FRAC_PI_4);
        let clip = calculated_clip([
            CalculatedClipRect {
                rect: Rect::new(0., f32::NEG_INFINITY, 10., f32::INFINITY),
                world_to_clip_local: Affine2::IDENTITY,
            },
            CalculatedClipRect {
                rect: Rect::new(-50., -50., 50., 50.),
                world_to_clip_local: clip_transform.inverse(),
            },
        ]);
        let size = Vec2::splat(2.);
        assert!(rect_is_clipped(
            Some(&clip),
            Affine2::from_translation(vec2(20., 0.)),
            size
        ));
        assert!(rect_is_clipped(
            Some(&clip),
            Affine2::from_translation(vec2(5., 100.)),
            size
        ));
        assert!(!rect_is_clipped(
            Some(&clip),
            Affine2::from_translation(vec2(5., 5.)),
            size
        ));
        assert!(!rect_is_clipped(
            Some(&clip),
            Affine2::from_translation(vec2(-1., 5.)),
            size
        ));
        assert!(rect_is_clipped(
            Some(&CalculatedClip::FullyClipped),
            Affine2::IDENTITY,
            size
        ));
        assert!(!rect_is_clipped(None, Affine2::IDENTITY, size));
    }

    fn calculated_clip(regions: impl IntoIterator<Item = CalculatedClipRect>) -> CalculatedClip {
        CalculatedClip::Rects(regions.into_iter().collect())
    }

    fn quad() -> [(Vec2, Vec2); 4] {
        [
            (vec2(-1., -1.), vec2(-1., -1.)),
            (vec2(1., -1.), vec2(1., -1.)),
            (vec2(1., 1.), vec2(1., 1.)),
            (vec2(-1., 1.), vec2(-1., 1.)),
        ]
    }

    #[test]
    fn unclipped_quad_returns_all_vertices() {
        assert_eq!(clip_polygon(None, &quad(), Vec2::lerp).len(), 4);
    }

    #[test]
    fn rectangle_partition_covers_only_the_unoccluded_area_without_overlap() {
        let bounds = vec2(100., 80.);
        let outer = Rect::from_center_size(Vec2::ZERO, bounds);
        for (hole, expected_area) in [
            (None, 8000.),
            (Some(Rect::EMPTY), 8000.),
            (Some(Rect::new(-30., -20., 30., 20.)), 5600.),
            // Offset shadows can move the occluder partially or entirely outside the shadow.
            (Some(Rect::new(-70., -15., -10., 25.)), 6400.),
            (Some(Rect::new(20., 15., 90., 65.)), 7250.),
            (Some(Rect::new(60., 50., 100., 90.)), 8000.),
            (Some(Rect::new(-90., -90., 90., 90.)), 0.),
        ] {
            let rects = rect_without_hole(bounds, hole);
            let area: f32 = rects.iter().map(Rect::area).sum();
            assert_eq!(area, expected_area);
            for (i, rect) in rects.iter().enumerate() {
                assert_eq!(outer.intersect(*rect), *rect);
                assert!(hole.is_none_or(|hole| rect.intersect(hole).is_empty()));
                assert!(rects[i + 1..]
                    .iter()
                    .all(|other| rect.intersect(*other).is_empty()));
            }
        }
    }

    #[test]
    fn clipped_rotated_ring_preserves_area_and_local_shader_coordinates() {
        let transform = Affine2::from_mat2(Mat2::from_cols(2. * Vec2::Y, -2. * Vec2::X));
        let inverse = transform.inverse();
        let clip = calculated_clip([CalculatedClipRect {
            rect: Rect::new(-80., -80., 80., 80.),
            world_to_clip_local: Affine2::IDENTITY,
        }]);
        let mut area = 0.;
        for part in rect_without_hole(vec2(100., 80.), Some(Rect::new(-30., -20., 30., 20.))) {
            let vertices = crate::QUAD_VERTEX_POSITIONS.map(|corner| {
                let point = part.center() + corner * part.size();
                (transform.transform_point2(point), point)
            });
            let clipped = clip_polygon(Some(&clip), &vertices, Vec2::lerp);
            for &(world, local) in &clipped {
                assert_eq!(inverse.transform_point2(world), local);
            }
            area += 0.5
                * clipped
                    .iter()
                    .zip(clipped.iter().cycle().skip(1))
                    .map(|(a, b)| a.0.perp_dot(b.0))
                    .sum::<f32>()
                    .abs();
        }
        // A 160x160 clipped outer square minus the rotated 80x120 hole.
        assert_eq!(area, 160. * 160. - 80. * 120.);
    }

    #[test]
    fn fully_clipped_returns_empty_vertices_list() {
        assert!(clip_polygon(Some(&CalculatedClip::FullyClipped), &quad(), Vec2::lerp).is_empty());
    }

    #[test]
    fn trim_quad_with_axis_aligned_clip() {
        let clip = calculated_clip([CalculatedClipRect {
            rect: Rect {
                min: vec2(0., -0.5),
                max: vec2(0.5, 0.5),
            },
            world_to_clip_local: Affine2::IDENTITY,
        }]);
        let clipped = clip_polygon(Some(&clip), &quad(), Vec2::lerp);

        assert_eq!(clipped.len(), 4);
        assert!(clipped.iter().all(|(v, _)| 0. <= v.x && v.x <= 0.5));
        assert!(clipped.iter().all(|(v, _)| -0.5 <= v.y && v.y <= 0.5));
    }

    #[test]
    fn nested_clip_rects_compose() {
        let vertices = clip_polygon(
            Some(&calculated_clip([
                CalculatedClipRect {
                    rect: Rect {
                        min: vec2(-0.75, -0.75),
                        max: vec2(0.75, 0.75),
                    },
                    world_to_clip_local: Affine2::IDENTITY,
                },
                CalculatedClipRect {
                    rect: Rect {
                        min: vec2(-0.25, -1.),
                        max: vec2(0.25, 1.),
                    },
                    world_to_clip_local: Affine2::from_mat2(Mat2::from(Rot2::radians(0.3)))
                        .inverse(),
                },
            ])),
            &quad(),
            Vec2::lerp,
        );

        assert!(!vertices.is_empty());
        assert!(vertices.iter().all(|(v, _)| -0.75 <= v.x && v.x <= 0.75));
    }

    #[test]
    fn transformed_clip_preserves_boundary_and_interpolated_coordinates() {
        for transform in [
            Affine2::from_scale_angle_translation(vec2(2., 0.5), 0.3, vec2(50., -100.)),
            Affine2::from_mat2_translation(
                Mat2::from_cols(vec2(-2., 0.5), vec2(1., 3.)),
                vec2(-30., 70.),
            ),
        ] {
            let inverse = transform.inverse();
            let clip = calculated_clip([CalculatedClipRect {
                rect: Rect::new(-0.25, -0.5, 0.25, 0.5),
                world_to_clip_local: inverse,
            }]);
            let vertices = quad().map(|(point, _)| (transform.transform_point2(point), point));
            let clipped = clip_polygon(Some(&clip), &vertices, Vec2::lerp);
            assert_eq!(clipped.len(), 4);
            for &(world, point) in &clipped {
                assert!(
                    (inverse.transform_point2(world) - point)
                        .abs()
                        .max_element()
                        < 0.0001
                );
                assert!((point.abs() - vec2(0.25, 0.5)).abs().max_element() < 0.0001);
            }
            let area = 0.5
                * clipped
                    .iter()
                    .zip(clipped.iter().cycle().skip(1))
                    .map(|(a, b)| a.1.perp_dot(b.1))
                    .sum::<f32>()
                    .abs();
            assert!((area - 0.5).abs() < 0.0001);
        }
    }

    #[test]
    fn quad_outside_clip_rect_returns_empty_vertices_list() {
        assert!(clip_polygon(
            Some(&calculated_clip([CalculatedClipRect {
                rect: Rect {
                    min: vec2(2., 2.),
                    max: vec2(3., 3.),
                },
                world_to_clip_local: Affine2::IDENTITY,
            }])),
            &quad(),
            Vec2::lerp
        )
        .is_empty());
    }
}
