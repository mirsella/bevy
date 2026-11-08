#import bevy_render::view::View
#import bevy_ui::paint::read_paint
#import bevy_ui::ui_node::sd_rounded_box
@group(0) @binding(0) var<uniform> view: View;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) @interpolate(flat) size: vec2<f32>,
    @location(2) @interpolate(flat) background: vec4<f32>,
#ifdef SURFACE_ROUNDED_FILL
    @location(3) @interpolate(flat) radius: vec4<f32>,
#endif
}
@vertex
fn vertex(@location(0) position: vec3<f32>, @location(1) point: vec2<f32>, @location(2) surface: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4(position, 1.0);
    out.point = point;
    out.size = bitcast<vec2<f32>>(read_paint(surface).xy);
    out.background = bitcast<vec4<f32>>(read_paint(surface + 3u));
#ifdef SURFACE_ROUNDED_FILL
    out.radius = bitcast<vec4<f32>>(read_paint(surface + 1u));
#endif
    return out;
}
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
#ifdef SURFACE_ROUNDED_FILL
    let distance = sd_rounded_box(in.point, in.size, in.radius);
#else
    let q = abs(in.point) - 0.5 * in.size;
#ifdef CHEAP_RECT_DISTANCE
    let distance = max(q.x, q.y);
#else
    let distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
#endif
#endif
#ifdef ANTI_ALIAS
    let coverage = saturate(0.5 - distance);
#else
    let coverage = 1.0 - step(0.0, distance);
#endif
    return vec4(in.background.rgb * in.background.a, in.background.a) * coverage;
}
