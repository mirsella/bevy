#import bevy_render::view::View
@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
}
@vertex
fn vertex(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>) -> VertexOutput {
    var tint = color;
#ifdef IMAGE_BASIC_ENCODED
    tint = vec4(select(1.055 * pow(max(color.rgb, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
        color.rgb * 12.92, color.rgb <= vec3(0.0031308)), color.a);
#endif
    return VertexOutput(view.clip_from_world * vec4(position, 1.0), uv, tint);
}
#ifdef MANUAL_SRGB
fn manual_srgb(color: vec3<f32>) -> vec3<f32> {
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}
#endif
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(sprite_texture, sprite_sampler, in.uv) * in.color;
#ifdef IMAGE_BASIC_ENCODED
    return color;
#else
#ifdef MANUAL_SRGB
    return vec4(manual_srgb(color.rgb), saturate(color.a));
#else
    return vec4(color.rgb, saturate(color.a));
#endif
#endif
}
