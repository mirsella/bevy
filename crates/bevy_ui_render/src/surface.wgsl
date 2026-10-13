#import bevy_render::view::View
#import bevy_ui::ui_node::{sd_rounded_box, sd_inset_rounded_box}
#import bevy_ui::gradient::sample_gradient
#import bevy_ui::paint::read_paint

@group(0) @binding(0) var<uniform> view: View;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) @interpolate(flat) surface: u32,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) radius: vec4<f32>,
    @location(4) @interpolate(flat) border: vec4<f32>,
    @location(5) @interpolate(flat) background: vec4<f32>,
    @location(6) @interpolate(flat) background_gradients: vec2<u32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>, @location(1) point: vec2<f32>, @location(2) surface: u32) -> VertexOutput {
    // Node constants are fetched per vertex and stay flat across the surface.
    let header = read_paint(surface);
    return VertexOutput(view.clip_from_world * vec4(position, 1.0), point, surface,
        bitcast<vec2<f32>>(header.xy),
        bitcast<vec4<f32>>(read_paint(surface + 1u)),
        bitcast<vec4<f32>>(read_paint(surface + 2u)),
        bitcast<vec4<f32>>(read_paint(surface + 3u)), header.zw);
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
fn paint(base: vec4<f32>, layers: vec2<u32>, point: vec2<f32>) -> vec4<f32> {
    var color = premultiply(base);
    for (var i = 0u; i < layers.y; i += 1u) {
        let layer = sample_gradient(layers.x + i * 2u, point);
        let gamma_layer = vec4(pow(max(layer.rgb, vec3(0.0)), vec3(1.0 / 2.2)), layer.a);
        color = premultiply(gamma_layer) + color * (1.0 - layer.a);
    }
    return color;
}

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
    let outer = coverage(sd_rounded_box(in.point, in.size, in.radius));
    let inner = min(outer, coverage(sd_inset_rounded_box(in.point, in.size, in.radius, in.border)));
    // These weights partition the outer shape, even for borders narrower than a pixel.
    let border = outer - inner;
    var color = vec4(0.0);
    if inner > 0.0 {
        color += paint(in.background, in.background_gradients, in.point) * inner;
    }
    if border > 0.0 {
        let side = border_side(in.point, in.size, in.border);
        // Interior pixels never fetch border paint; edge pixels fetch only their side.
        let border_color = bitcast<vec4<f32>>(read_paint(in.surface + 4u + side));
        let layers = read_paint(in.surface + 8u).xy;
        color += paint(border_color, layers, in.point) * border;
    }
    // One premultiplied source-over blend for the entire surface, including fades.
    return color;
}
