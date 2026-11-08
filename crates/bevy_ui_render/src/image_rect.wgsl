#import bevy_render::view::View
@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) point: vec2<f32>,
}
@vertex
fn vertex(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>, @location(6) size: vec2<f32>,
    @location(7) point: vec2<f32>) -> VertexOutput {
    return VertexOutput(view.clip_from_world * vec4(position, 1.0), uv, color, size, point);
}
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSampleGrad(sprite_texture, sprite_sampler, in.uv, dpdx(in.uv), dpdy(in.uv));
    let color = in.color * sample;
    let q = abs(in.point) - 0.5 * in.size;
#ifdef CHEAP_RECT_DISTANCE
    let distance = max(q.x, q.y);
#else
    let distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
#endif
#ifdef ANTI_ALIAS
    let coverage = saturate(0.5 - distance);
#else
    let coverage = 1.0 - step(0.0, distance);
#endif
#ifdef MANUAL_SRGB
    let rgb = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / 2.2));
#else
    let rgb = color.rgb;
#endif
    return vec4(rgb, saturate(color.a * coverage));
}
