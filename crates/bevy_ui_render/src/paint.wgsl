#define_import_path bevy_ui::paint

// Exact, unfiltered records shared by every surface in the frame. A single
// integer texture supports native, WebGPU and WebGL2 with the same shader.
@group(0) @binding(1) var paint_data: texture_2d<u32>;

fn read_paint(address: u32) -> vec4<u32> {
    let width = textureDimensions(paint_data).x;
    return textureLoad(paint_data, vec2<i32>(i32(address % width), i32(address / width)), 0);
}
