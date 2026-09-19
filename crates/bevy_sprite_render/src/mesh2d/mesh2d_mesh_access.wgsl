#define_import_path bevy_sprite::mesh2d_mesh_access

#import bevy_sprite::{mesh2d_types::Mesh2d, mesh2d_bindings::mesh}

// Custom mesh shaders should use this accessor (or mesh2d_functions) rather than
// indexing mesh directly so backend-specific uniform indexing fixes also apply.
fn get_mesh(instance_index: u32) -> Mesh2d {
    return mesh[instance_index];
}
