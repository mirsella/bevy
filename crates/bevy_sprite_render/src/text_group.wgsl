struct Input {
    @builtin(vertex_index) index: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) coverage: vec4<u32>,
}
struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) coverage: vec4<u32>,
}
@vertex
fn vertex(in: Input) -> Output {
    let corners = array<vec2<f32>, 6>(vec2(0., 0.), vec2(1., 0.), vec2(0., 1.), vec2(0., 1.), vec2(1., 0.), vec2(1., 1.));
    let point = corners[in.index];
    let position = in.rect.xy + point * in.rect.zw;
    var out: Output;
    out.position = vec4(position * vec2(2., -2.) + vec2(-1., 1.), 0., 1.);
    out.uv = in.uv.xy + point * in.uv.zw;
    out.color = in.color;
    out.coverage = in.coverage;
    return out;
}
@fragment
fn clear() -> @location(0) vec4<f32> { return vec4(0.); }
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;
@fragment
fn fragment(in: Output) -> @location(0) vec4<f32> {
    let dx = dpdx(in.uv);
    let dy = dpdy(in.uv);
    if in.coverage.x == 4u {
        let size = bitcast<vec2<f32>>(in.coverage.yz);
        let edge = clamp(min(in.uv + vec2(0.5), size - in.uv + vec2(0.5)), vec2(0.), vec2(1.));
        return vec4(in.color.rgb, in.color.a * edge.x * edge.y);
    }
    let sample = textureSampleGrad(atlas, atlas_sampler, in.uv, dx, dy);
    switch in.coverage.x {
        case 1u: { return vec4(in.color.rgb, in.color.a * sample.a); }
        case 2u: { return vec4(in.color.rgb, in.color.a * sample.r); }
        case 3u: { return vec4(in.color.rgb, in.color.a * (sample.a + sample.r * (1. - sample.a))); }
        default: { return vec4(in.color.rgb * pow(max(sample.rgb, vec3(0.)), vec3(1. / 2.2)), in.color.a * sample.a); }
    }
}
