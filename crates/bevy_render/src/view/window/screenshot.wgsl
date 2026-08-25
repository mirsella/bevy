// This vertex shader will create a triangle that will cover the entire screen
// with minimal effort, avoiding the need for a vertex buffer etc.
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    let x = f32((in_vertex_index & 1u) << 2u);
    let y = f32((in_vertex_index & 2u) << 1u);
    return VertexOutput(
        vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0),
        vec2<f32>(x, 2.0 - y) * 0.5,
    );
}

@group(0) @binding(0) var t: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = textureDimensions(t);
    let coords = min(vec2<u32>(in.uv * vec2<f32>(size)), size - 1u);
    return textureLoad(t, vec2<i32>(coords), 0i);
}
