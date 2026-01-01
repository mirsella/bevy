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
    // 3. Output straight alpha as BlendState::ALPHA_BLENDING (SrcAlpha, OneMinusSrcAlpha) handles the multiplication by alpha.
    
    let alpha = max(color.a, 0.00001);
    let straight_srgb = color.rgb / alpha;
    
    // Proper sRGB to Linear conversion
    let straight_linear = pow(straight_srgb, vec3(2.2));

    return vec4<f32>(straight_linear, color.a);
}
