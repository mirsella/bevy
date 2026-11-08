#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::color_operations::{srgb_to_linear, linear_to_srgb, oklab_to_linear_rgb}
@group(0) @binding(0) var ui_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var ui_sampler: sampler;
@group(0) @binding(3) var<uniform> ui_rect: vec4<u32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Match the reference nearest-neighbour output blit. Evaluate the preceding
    // gamma UI composite at that MAIN pixel centre, not the output pixel centre.
    let size = textureDimensions(scene_texture);
    let pixel = min(vec2<u32>(in.uv * vec2<f32>(size)), size - 1u);
    let point = vec2<f32>(pixel) + 0.5;
    let uv = (point - vec2<f32>(ui_rect.xy)) / vec2<f32>(ui_rect.zw);
    var ui = textureSampleLevel(ui_texture, ui_sampler, uv, 0.0);
    if any(uv < vec2(0.0)) || any(uv >= vec2(1.0)) { ui = vec4(0.0); }
    let linear = pow(ui.rgb / max(ui.a, 0.00001), vec3<f32>(2.2));
#ifdef BASE_SCENE
    let scene = textureLoad(scene_texture, vec2<i32>(pixel), 0);
    var color = vec4<f32>(linear * ui.a + scene.rgb * (1.0 - ui.a), ui.a + scene.a * (1.0 - ui.a));
#ifdef UNORM_SCENE
    color = round(clamp(color, vec4(0.0), vec4(1.0)) * 255.0) / 255.0;
#endif
#ifdef SRGB_SCENE
    let encoded = round(vec4(linear_to_srgb(clamp(color.rgb, vec3(0.0), vec3(1.0))), clamp(color.a, 0.0, 1.0)) * 255.0) / 255.0;
    color = vec4(srgb_to_linear(encoded.rgb), encoded.a);
#endif
#ifdef F16_SCENE
    color = vec4(unpack2x16float(pack2x16float(color.rg)), unpack2x16float(pack2x16float(color.ba)));
#endif
#ifdef RGB10A2_SCENE
    let levels = vec4(1023.0, 1023.0, 1023.0, 3.0);
    color = round(clamp(color, vec4(0.0), vec4(1.0)) * levels) / levels;
#endif
#ifdef SRGB_TO_LINEAR
    color = vec4(srgb_to_linear(color.rgb), color.a);
#endif
#ifdef OKLAB_TO_LINEAR
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
#endif
    return color;
#else
    return vec4<f32>(linear, ui.a);
#endif
}
