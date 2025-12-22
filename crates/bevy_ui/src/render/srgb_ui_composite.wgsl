#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, screen_sampler, in.uv);
    
    // The texture contains sRGB-encoded colors with premultiplied alpha.
    // To convert to linear space correctly, we must:
    // 1. Un-premultiply to get straight sRGB
    // 2. Convert straight sRGB to straight Linear
    // 3. Output Premultiplied Linear (the pipeline's BlendState::ALPHA_BLENDING assumes premultiplied input)
    
    let alpha = max(color.a, 0.00001);
    let straight_srgb = color.rgb / alpha;
    
    // Proper sRGB to Linear conversion
    let straight_linear = select(
        pow((straight_srgb + 0.055) / 1.055, vec3(2.4)),
        straight_srgb / 12.92,
        straight_srgb <= vec3<f32>(0.04045)
    );

    return vec4<f32>(straight_linear, color.a);
}
