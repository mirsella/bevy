#import bevy_render::view::View

// Group zero remains the source pipeline's view/paint layout. Cached texels
// deliberately use a different group, never the reference paint binding.
@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var cached_surface: texture_2d<f32>;
@group(1) @binding(1) var<uniform> crop: vec4<u32>;
@group(1) @binding(2) var cache_sampler: sampler;

struct CachedVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ui_position: vec2<f32>,
    @location(1) cache_uv: vec2<f32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>) -> CachedVertexOutput {
    var out: CachedVertexOutput;
    out.clip_position = view.clip_from_world * vec4(position, 1.0);
    out.ui_position = position.xy;
    out.cache_uv = (position.xy - vec2<f32>(crop.xy)) / vec2<f32>(crop.zw);
    return out;
}

@fragment
fn fragment(in: CachedVertexOutput) -> @location(0) vec4<f32> {
#ifdef SURFACE_CACHE_PROBE_1
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
#else ifdef SURFACE_CACHE_PROBE_2
    // Fixed FP3 diagnostic viewport: View is vertex-only in the shared source layout.
    return vec4<f32>(vec2<f32>(crop.xy) / vec2<f32>(1080.0, 2160.0), 0.0, 1.0);
#else ifdef SURFACE_CACHE_PROBE_3
    return vec4<f32>(vec2<f32>(textureDimensions(cached_surface)) / 2048.0, 0.0, 1.0);
#else ifdef SURFACE_CACHE_PROBE_4
    return textureLoad(cached_surface, vec2<i32>(textureDimensions(cached_surface) / 2u), 0);
#else ifdef SURFACE_CACHE_PROBE_5
    let texel = vec2<i32>(in.ui_position - vec2<f32>(crop.xy));
    if any(texel < vec2<i32>(0)) || any(texel >= vec2<i32>(textureDimensions(cached_surface))) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    return textureLoad(cached_surface, texel, 0);
#else
    // Native-size source pixels, sampled at their original pixel centers. Form
    // the crop-relative coordinate in the vertex stage and use nearest sampling.
    return textureSample(cached_surface, cache_sampler, in.cache_uv);
#endif
}
