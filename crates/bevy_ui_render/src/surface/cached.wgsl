#import bevy_render::view::View

// Group zero remains the source pipeline's view/paint layout. Cached texels
// deliberately use a different group, never the reference paint binding.
@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var cached_surface: texture_2d<f32>;
@group(1) @binding(1) var<uniform> crop: vec4<u32>;
@group(1) @binding(2) var cache_sampler: sampler;

struct CachedVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) cache_uv: vec2<f32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>) -> CachedVertexOutput {
    var out: CachedVertexOutput;
    out.clip_position = view.clip_from_world * vec4(position, 1.0);
    out.cache_uv = (position.xy - vec2<f32>(crop.xy)) / vec2<f32>(crop.zw);
    return out;
}

@fragment
fn fragment(in: CachedVertexOutput) -> @location(0) vec4<f32> {
    // Native-size source pixels, sampled at their original pixel centers. Form
    // the crop-relative coordinate in the vertex stage and use nearest sampling.
    return textureSample(cached_surface, cache_sampler, in.cache_uv);
}
