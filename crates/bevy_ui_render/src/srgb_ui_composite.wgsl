// Composite shader for gamma-space UI rendering.
// The source texture stores premultiplied colors after alpha blending in sRGB space.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;

fn gamma_2_2_to_linear(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, screen_sampler, in.uv);
    let alpha = max(color.a, 0.00001);
    let straight_srgb = color.rgb / alpha;
    let straight_linear = gamma_2_2_to_linear(straight_srgb);
    return vec4<f32>(straight_linear, color.a);
}
