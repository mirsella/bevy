#import bevy_render::view::View
#import bevy_ui::ui_node::{sd_rounded_box, sd_inset_rounded_box}
#ifndef SURFACE_SOLID
#import bevy_ui::gradient::sample_gradient
#endif
#import bevy_ui::paint::read_paint

@group(0) @binding(0) var<uniform> view: View;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
#ifndef SURFACE_SOLID
    @location(1) @interpolate(flat) surface: u32,
#endif
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) radius: vec4<f32>,
    @location(4) @interpolate(flat) border: vec4<f32>,
    @location(5) @interpolate(flat) background: vec4<f32>,
#ifdef SURFACE_SOLID
    @location(6) @interpolate(flat) border_left: vec4<f32>,
    @location(7) @interpolate(flat) border_top: vec4<f32>,
    @location(8) @interpolate(flat) border_right: vec4<f32>,
    @location(9) @interpolate(flat) border_bottom: vec4<f32>,
#else
    @location(6) @interpolate(flat) background_gradients: vec2<u32>,
#endif
}

@vertex
fn vertex(@location(0) position: vec3<f32>, @location(1) point: vec2<f32>, @location(2) surface: u32) -> VertexOutput {
    // Node constants are fetched per vertex and stay flat across the surface.
    let header = read_paint(surface);
    return VertexOutput(view.clip_from_world * vec4(position, 1.0), point,
#ifndef SURFACE_SOLID
        surface,
#endif
        bitcast<vec2<f32>>(header.xy),
        bitcast<vec4<f32>>(read_paint(surface + 1u)),
        bitcast<vec4<f32>>(read_paint(surface + 2u)),
        bitcast<vec4<f32>>(read_paint(surface + 3u)),
#ifdef SURFACE_SOLID
        bitcast<vec4<f32>>(read_paint(surface + 4u)),
        bitcast<vec4<f32>>(read_paint(surface + 5u)),
        bitcast<vec4<f32>>(read_paint(surface + 6u)),
        bitcast<vec4<f32>>(read_paint(surface + 7u)));
#else
        header.zw);
#endif
}

fn coverage(distance: f32) -> f32 {
#ifdef ANTI_ALIAS
    return saturate(0.5 - distance);
#else
    return 1.0 - step(0.0, distance);
#endif
}

fn premultiply(color: vec4<f32>) -> vec4<f32> {
    return vec4(color.rgb * color.a, color.a);
}

// Paint layers without geometric coverage. Otherwise translucent gradients would
// repeatedly apply the same antialiased edge and change the surface's opacity.
#ifndef SURFACE_SOLID
fn paint(base: vec4<f32>, layers: vec2<u32>, point: vec2<f32>) -> vec4<f32> {
    var color = premultiply(base);
    for (var i = 0u; i < layers.y; i += 1u) {
        let layer = sample_gradient(layers.x + i * 2u, point);
#ifdef LINEAR_COLOR
        color = premultiply(layer) + color * (1.0 - layer.a);
#else
        let gamma_layer = vec4(pow(max(layer.rgb, vec3(0.0)), vec3(1.0 / 2.2)), layer.a);
        color = premultiply(gamma_layer) + color * (1.0 - layer.a);
#endif
    }
    return color;
}
#endif

fn border_side(point: vec2<f32>, size: vec2<f32>, border: vec4<f32>) -> u32 {
    let p = clamp(point + size * 0.49999, vec2(0.0), size);
    let distances = select(vec4(1e30), vec4(p, size - p) / max(border, vec4(1e-20)), border > vec4(0.0));
    var side = 0u;
    for (var i = 1u; i < 4u; i += 1u) {
        if distances[i] < distances[side] { side = i; }
    }
    return side;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
#ifdef SURFACE_RECT_FILL
    let q = abs(in.point) - 0.5 * in.size;
#ifdef CHEAP_RECT_DISTANCE
    let distance = max(q.x, q.y);
#else
    let distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
#endif
    return premultiply(in.background) * coverage(distance);
#else ifdef SURFACE_ROUNDED_FILL
    return premultiply(in.background) * coverage(sd_rounded_box(in.point, in.size, in.radius));
#else
    let outer = coverage(sd_rounded_box(in.point, in.size, in.radius));
    let inner = min(outer, coverage(sd_inset_rounded_box(in.point, in.size, in.radius, in.border)));
    // These weights partition the outer shape, even for borders narrower than a pixel.
    let border = outer - inner;
    var color = vec4(0.0);
    if inner > 0.0 {
#ifdef SURFACE_SOLID
        color += premultiply(in.background) * inner;
#else
        color += paint(in.background, in.background_gradients, in.point) * inner;
#endif
    }
    if border > 0.0 {
        let side = border_side(in.point, in.size, in.border);
#ifdef SURFACE_SOLID
        // Same side/tie selection as the reference; no fragment paint-table loads.
        var border_color = in.border_left;
        switch side {
            case 1u: { border_color = in.border_top; }
            case 2u: { border_color = in.border_right; }
            case 3u: { border_color = in.border_bottom; }
            default: {}
        }
        color += premultiply(border_color) * border;
#else
        // Interior pixels never fetch border paint; edge pixels fetch only their side.
        let border_color = bitcast<vec4<f32>>(read_paint(in.surface + 4u + side));
        let layers = read_paint(in.surface + 8u).xy;
        color += paint(border_color, layers, in.point) * border;
#endif
    }
    // One premultiplied source-over blend for the entire surface, including fades.
    return color;
#endif
}
