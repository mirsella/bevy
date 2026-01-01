// Composite shader for sRGB sprite rendering.
// This shader samples the sRGB sprite texture and blends it onto the main target.
// The texture contains sprites rendered with sRGB blending (in Rgba8Unorm format).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var srgb_texture: texture_2d<f32>;
@group(0) @binding(1) var srgb_sampler: sampler;

// Convert sRGB to linear RGB (Gamma 2.2 approximation)
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(srgb_texture, srgb_sampler, in.uv);

    // color.rgb: sRGB-encoded composite from sprite pass
    // color.a  : composite alpha from sprite pass
    let linear_rgb = srgb_to_linear(color.rgb);

    // Output straight alpha as BlendState::ALPHA_BLENDING (SrcAlpha, OneMinusSrcAlpha)
    // handles the multiplication by alpha.
    return vec4<f32>(linear_rgb, color.a);
}
