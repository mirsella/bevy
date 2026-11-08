// Composite shader for gamma-space UI rendering.
// The source texture stores premultiplied colors after alpha blending in sRGB space.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;

#ifdef UI_COMPOSITE_LOOKUP
@group(0) @binding(2) var gamma_lookup: texture_2d<f32>;
#endif

fn gamma_2_2_to_linear(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, screen_sampler, in.uv);
#ifdef UI_COMPOSITE_LOOKUP
    // Translucent filtered samples are not generally on the 8-bit grid.
    // Preserve their full reference unpremultiplication and gamma arithmetic.
    if color.a == 1.0 {
        let channels = vec3<u32>(round(clamp(color.rgb, vec3(0.0), vec3(1.0)) * 255.0));
        return vec4<f32>(
            textureLoad(gamma_lookup, vec2<u32>(channels.r, 0u), 0).r,
            textureLoad(gamma_lookup, vec2<u32>(channels.g, 0u), 0).r,
            textureLoad(gamma_lookup, vec2<u32>(channels.b, 0u), 0).r,
            color.a,
        );
    }
#endif
    let alpha = max(color.a, 0.00001);
    let straight_srgb = color.rgb / alpha;
    let straight_linear = gamma_2_2_to_linear(straight_srgb);
    return vec4<f32>(straight_linear, color.a);
}
