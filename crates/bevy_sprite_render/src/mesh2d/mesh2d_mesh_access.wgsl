#define_import_path bevy_sprite::mesh2d_mesh_access

#import bevy_sprite::mesh2d_bindings::mesh

// Read members without returning Mesh2d: whole-struct values crash older Adreno Vulkan compilers.
// Use these accessors to preserve backend-specific uniform indexing fixes too.
fn get_world_from_local(instance_index: u32) -> mat3x4<f32> {
    return mesh[instance_index].world_from_local;
}

fn get_local_from_world_transpose_a(instance_index: u32) -> mat2x4<f32> {
    return mesh[instance_index].local_from_world_transpose_a;
}

fn get_local_from_world_transpose_b(instance_index: u32) -> f32 {
    return mesh[instance_index].local_from_world_transpose_b;
}

fn get_tag(instance_index: u32) -> u32 {
    return mesh[instance_index].tag;
}
