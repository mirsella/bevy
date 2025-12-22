#import bevy_render::{
    view::View,
    globals::Globals,
}
#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;

@vertex
fn vertex(
    @location(0) vertex_position: vec3<f32>,
    @location(1) vertex_uv: vec2<f32>,
    @location(2) size: vec2<f32>,
    @location(3) border_widths: vec4<f32>,
    @location(4) border_radius: vec4<f32>,
) -> UiVertexOutput {
    var out: UiVertexOutput;
    out.uv = vertex_uv;
    out.position = view.clip_from_world * vec4<f32>(vertex_position, 1.0);
    out.size = size;
    out.border_widths = border_widths;
    out.border_radius = border_radius;
    return out;
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Encode to sRGB so blending in Rgba8Unorm happens in sRGB/gamma space.
    // NOTE: If you are using a custom fragment shader, you must also 
    // encode your output color to sRGB using `pow(color.rgb, vec3(1.0 / 2.2))`.
#ifdef MANUAL_SRGB
    return vec4<f32>(pow(vec3<f32>(1.0), vec3(1.0 / 2.2)), 1.0);
#else
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
#endif
}
