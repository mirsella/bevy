#import bevy_render::view::View;

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
}

@vertex
fn vertex(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4(position, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color * textureSample(sprite_texture, sprite_sampler, in.uv);
#ifdef MANUAL_SRGB
    return vec4(pow(max(color.rgb, vec3(0.0)), vec3(1.0 / 2.2)), color.a);
#else
    return color;
#endif
}
