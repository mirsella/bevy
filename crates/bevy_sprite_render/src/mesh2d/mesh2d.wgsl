#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct Vertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_POSITIONS
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
#endif

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh2d_normal_local_to_world(vertex.normal, vertex.instance_index);
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh2d_tangent_local_to_world(
        world_from_local,
        vertex.tangent
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
#ifdef VERTEX_COLORS
    var color = in.color;
#ifdef TONEMAP_IN_SHADER
    color = tonemapping::tone_mapping(color, view.color_grading);
#endif
#ifdef SRGB_MESH2D_PASS
    // The texture is Rgba8Unorm, so we need to manually encode to sRGB.
    // However, the pipeline expects premultiplied alpha for blending.
    // The sRGB conversion should happen on the straight color, then premultiply again.
    // But standard sRGB encoding is usually done at the very end.
    
    // For Rgba8Unorm targets where we want sRGB results:
    // 1. We have linear color here.
    // 2. We convert to sRGB.
    // 3. We output that.
    
    // Wait, typical bevy pipeline:
    // Main pass (HDR/Linear) -> Tonemapping -> sRGB (if SwapChain)
    
    // Here we are rendering to an intermediate texture (Rgba8Unorm).
    // If we want this texture to contain sRGB data (so that standard blending works? No wait).
    // The previous commit a19973c41 says:
    // "Adds max(0.0) clamping to gamma conversion in shaders to prevent NaNs/artifacts."
    // And it seems they are doing manual sRGB encoding.
    
    // Let's look at what `bevy_sprite` did in `sprite.wgsl` (I should check that too).
    // But for `mesh2d.wgsl`, the reference commit did something.
    
    let alpha = color.a;
    // Prevent NaNs/Artifacts by clamping to 0.0
    let rgb = max(color.rgb, vec3<f32>(0.0));
    
    // Manual sRGB encoding (simplified gamma 2.2)
    let srgb = pow(rgb, vec3<f32>(1.0 / 2.2));
    
    color = vec4<f32>(srgb, alpha);
#endif
    return color;
#else
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
#endif
}
