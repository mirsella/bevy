#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif
#ifdef OKLAB_OUTPUT
#import bevy_render::color_operations::linear_rgb_to_oklab
#endif

#ifdef SRGB_OUTPUT
fn linear_to_gamma_2_2(color: vec3<f32>) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}
#endif

#import bevy_render::{
    maths::affine3_to_square,
    view::View,
}

#import bevy_sprite::sprite_view_bindings::view

const TEXT_EFFECT_ALPHA: u32 = 1u;
const TEXT_EFFECT_OUTLINE: u32 = 2u;
const TEXT_EFFECT_OUTLINED_SHADOW: u32 = 3u;
const TEXT_EFFECT_GROUP: u32 = 4u;

struct VertexInput {
    @builtin(vertex_index) index: u32,
    // NOTE: Instance-rate vertex buffer members prefixed with i_
    // NOTE: i_model_transpose_colN are the 3 columns of a 3x4 matrix that is the transpose of the
    // affine 4x3 model matrix.
    @location(0) i_model_transpose_col0: vec4<f32>,
    @location(1) i_model_transpose_col1: vec4<f32>,
    @location(2) i_model_transpose_col2: vec4<f32>,
    @location(3) i_color: vec4<f32>,
    @location(4) i_uv_offset_scale: vec4<f32>,
    @location(5) i_text_effect: vec4<u32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) text_effect: u32,
};

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let vertex_position = vec3<f32>(
        f32(in.index & 0x1u),
        f32((in.index & 0x2u) >> 1u),
        0.0
    );

    out.clip_position = view.clip_from_world * affine3_to_square(mat3x4<f32>(
        in.i_model_transpose_col0,
        in.i_model_transpose_col1,
        in.i_model_transpose_col2,
    )) * vec4<f32>(vertex_position, 1.0);
    out.uv = vec2<f32>(vertex_position.xy) * in.i_uv_offset_scale.zw + in.i_uv_offset_scale.xy;
    out.color = in.i_color;
    out.text_effect = in.i_text_effect.x;

    return out;
}

@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_sample = textureSample(sprite_texture, sprite_sampler, in.uv);
    var color = vec4<f32>(0.0);

    switch in.text_effect {
        case TEXT_EFFECT_ALPHA: {
            color = vec4(in.color.rgb, in.color.a * base_sample.a);
        }
        case TEXT_EFFECT_OUTLINE: {
            color = vec4(in.color.rgb, in.color.a * base_sample.r);
        }
        case TEXT_EFFECT_OUTLINED_SHADOW: {
            let coverage = base_sample.a + base_sample.r * (1.0 - base_sample.a);
            color = vec4(in.color.rgb, in.color.a * coverage);
        }
        case TEXT_EFFECT_GROUP: {
            let straight = base_sample.rgb / max(base_sample.a, 0.00001);
            color = vec4(pow(max(straight, vec3(0.)), vec3(2.2)), base_sample.a * in.color.a);
        }
        default: {
            color = in.color * base_sample;
        }
    }

#ifdef TONEMAP_IN_SHADER
    color = tonemapping::tone_mapping(color, view.color_grading);
#endif

#ifdef SRGB_OUTPUT
    color = vec4(linear_to_gamma_2_2(color.rgb), color.a);
#endif

#ifdef OKLAB_OUTPUT
    color = vec4(linear_rgb_to_oklab(color.rgb), color.a);
#endif

    return color;
}
