use super::append_paint;
use bevy_color::{Alpha, Color, ColorToComponents, Hsla, Hsva, LinearRgba, Oklaba, Oklcha, Srgba};
use bevy_math::{
    ops::{cos, sin},
    FloatOrd, Vec2,
};
use bevy_ui::{ComputedUiRenderTargetInfo, Gradient, InterpolationColorSpace};
use bytemuck::{Pod, Zeroable};
use core::f32::consts::{FRAC_PI_2, TAU};

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuGradient {
    start: [f32; 2],
    direction: [f32; 2],
    kind: u32,
    color_space: u32,
    stops_start: u32,
    stops_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuColorStop {
    color: [f32; 4],
    point: f32,
    hint: f32,
    padding: [f32; 2],
}

// Reuse per-layer scratch space. Only final, texture-addressed records survive extraction.
#[derive(Default)]
pub(super) struct GradientExtractor {
    gradients: Vec<GpuGradient>,
    stops: Vec<GpuColorStop>,
    explicit: Vec<GpuColorStop>,
}

impl GradientExtractor {
    pub fn extract(
        &mut self,
        layers: &[Gradient],
        size: Vec2,
        target: &ComputedUiRenderTargetInfo,
        texels: &mut Vec<[u32; 4]>,
    ) -> [u32; 2] {
        self.gradients.clear();
        let scale = target.scale_factor();
        let viewport = target.physical_size().as_vec2();
        for layer in layers {
            let mut gradient = GpuGradient::default();
            let space = match layer {
                Gradient::Linear(layer) => {
                    let direction = Vec2::new(sin(layer.angle), -cos(layer.angle));
                    let corner = ((layer.angle - FRAC_PI_2).rem_euclid(TAU) / FRAC_PI_2) as usize;
                    gradient.start = (crate::QUAD_VERTEX_POSITIONS[corner] * size).into();
                    gradient.direction = direction.into();
                    let length = direction.abs().dot(size);
                    self.resolve_stops(
                        layer.stops.iter().map(|stop| {
                            (
                                stop.color,
                                stop.point.resolve(scale, length, viewport).ok(),
                                stop.hint,
                            )
                        }),
                        length,
                        layer.color_space,
                    );
                    layer.color_space
                }
                Gradient::Radial(layer) => {
                    let center = layer.position.resolve(scale, size, viewport);
                    let extent = layer.shape.resolve(center, scale, size, viewport);
                    gradient.kind = 1;
                    gradient.start = center.into();
                    gradient.direction = [
                        if extent.y != 0. {
                            extent.x / extent.y
                        } else {
                            1.
                        },
                        0.,
                    ];
                    self.resolve_stops(
                        layer.stops.iter().map(|stop| {
                            (
                                stop.color,
                                stop.point.resolve(scale, extent.x, viewport).ok(),
                                stop.hint,
                            )
                        }),
                        extent.x,
                        layer.color_space,
                    );
                    layer.color_space
                }
                Gradient::Conic(layer) => {
                    gradient.kind = 2;
                    gradient.start = layer.position.resolve(scale, size, viewport).into();
                    gradient.direction = [layer.start, 0.];
                    self.resolve_stops(
                        layer.stops.iter().map(|stop| {
                            (
                                stop.color,
                                stop.angle.map(|angle| angle.clamp(0., TAU)),
                                stop.hint,
                            )
                        }),
                        TAU,
                        layer.color_space,
                    );
                    layer.color_space
                }
            };
            if self.stops.is_empty() {
                continue;
            }
            // These IDs are shared with gradient.wgsl.
            gradient.color_space = match space {
                InterpolationColorSpace::LinearRgba => 0,
                InterpolationColorSpace::Srgba => 1,
                InterpolationColorSpace::Oklaba => 2,
                InterpolationColorSpace::Oklcha => 3,
                InterpolationColorSpace::OklchaLong => 4,
                InterpolationColorSpace::Hsla => 5,
                InterpolationColorSpace::HslaLong => 6,
                InterpolationColorSpace::Hsva => 7,
                InterpolationColorSpace::HsvaLong => 8,
            };
            gradient.stops_start = append_paint(texels, &self.stops);
            gradient.stops_count = self.stops.len() as u32;
            self.gradients.push(gradient);
        }
        [
            append_paint(texels, &self.gradients),
            self.gradients.len() as u32,
        ]
    }

    fn resolve_stops(
        &mut self,
        stops: impl Iterator<Item = (Color, Option<f32>, f32)> + Clone,
        length: f32,
        space: InterpolationColorSpace,
    ) {
        self.stops.clear();
        if stops
            .clone()
            .all(|(color, _, _)| color.is_fully_transparent())
        {
            return;
        }
        self.stops
            .extend(stops.map(|(color, point, hint)| GpuColorStop {
                color: convert_color_to_space(color, space),
                point: point.unwrap_or(f32::NAN),
                hint,
                padding: [0.; 2],
            }));
        self.explicit.clear();
        self.explicit.extend(
            self.stops
                .iter()
                .filter(|stop| !stop.point.is_nan())
                .copied(),
        );
        self.explicit.sort_by_key(|stop| FloatOrd(stop.point));
        let min = self.explicit.first().map_or(0., |stop| stop.point.min(0.));
        let max = self
            .explicit
            .last()
            .map_or(length, |stop| stop.point.max(length));
        let mut sorted = self.explicit.iter();
        for stop in &mut self.stops {
            if !stop.point.is_nan() {
                *stop = *sorted.next().unwrap();
            }
        }
        let stops = &mut self.stops;
        if stops[0].point.is_nan() {
            stops[0].point = min;
        }
        if stops.last().unwrap().point.is_nan() {
            stops.last_mut().unwrap().point = max;
        }
        // Interpolate each run of implicit positions between its explicit neighbors.
        let mut start = 0;
        while start + 1 < stops.len() {
            let end = (start + 1..stops.len())
                .find(|&i| !stops[i].point.is_nan())
                .unwrap();
            let step = (stops[end].point - stops[start].point) / (end - start) as f32;
            for i in start + 1..end {
                stops[i].point = stops[start].point + step * (i - start) as f32;
            }
            start = end;
        }
    }
}

fn convert_color_to_space(color: Color, space: InterpolationColorSpace) -> [f32; 4] {
    // Convert directly: an RGB round trip loses saturation/hue at black or white.
    // Polar spaces use turns in the shader and degrees in Bevy's color types.
    match space {
        InterpolationColorSpace::LinearRgba => LinearRgba::from(color).to_f32_array(),
        InterpolationColorSpace::Srgba => Srgba::from(color).to_f32_array(),
        InterpolationColorSpace::Oklaba => Oklaba::from(color).to_f32_array(),
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let [lightness, chroma, hue, alpha] = Oklcha::from(color).to_f32_array();
            [lightness, chroma, hue / 360., alpha]
        }
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let [hue, saturation, lightness, alpha] = Hsla::from(color).to_f32_array();
            [hue / 360., saturation, lightness, alpha]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let [hue, saturation, value, alpha] = Hsva::from(color).to_f32_array();
            [hue / 360., saturation, value, alpha]
        }
    }
}
