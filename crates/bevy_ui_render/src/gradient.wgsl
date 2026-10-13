#define_import_path bevy_ui::gradient

#import bevy_render::color_operations::{
    mix_oklch, mix_oklch_long, mix_hsv, mix_hsv_long, mix_hsl, mix_hsl_long,
    oklch_to_linear_rgb, hsv_to_linear_rgb, hsl_to_linear_rgb, oklab_to_linear_rgb,
    srgb_to_linear,
}
#import bevy_render::maths::PI
#import bevy_ui::paint::read_paint

struct Gradient {
    start: vec2<f32>,
    direction: vec2<f32>,
    kind: u32,
    color_space: u32,
    stops_start: u32,
    stops_count: u32,
}

struct ColorStop {
    color: vec4<f32>,
    point: f32,
    hint: f32,
}

fn load_gradient(address: u32) -> Gradient {
    let geometry = bitcast<vec4<f32>>(read_paint(address));
    let metadata = read_paint(address + 1u);
    return Gradient(geometry.xy, geometry.zw, metadata.x, metadata.y, metadata.z, metadata.w);
}

fn load_stop(address: u32, index: u32) -> ColorStop {
    let color = bitcast<vec4<f32>>(read_paint(address + index * 2u));
    let position = bitcast<vec4<f32>>(read_paint(address + index * 2u + 1u));
    return ColorStop(color, position.x, position.y);
}

fn to_linear(color: vec4<f32>, space: u32) -> vec4<f32> {
    var rgb = color.rgb;
    switch space {
        case 1u: { rgb = srgb_to_linear(rgb); }
        case 2u: { rgb = oklab_to_linear_rgb(rgb); }
        case 3u, 4u: { rgb = oklch_to_linear_rgb(rgb); }
        case 5u, 6u: { rgb = hsl_to_linear_rgb(rgb); }
        case 7u, 8u: { rgb = hsv_to_linear_rgb(rgb); }
        default: {}
    }
    return vec4(rgb, color.a);
}

fn mix_colors(a: vec3<f32>, b: vec3<f32>, t: f32, space: u32) -> vec3<f32> {
    switch space {
        case 3u: { return mix_oklch(a, b, t); }
        case 4u: { return mix_oklch_long(a, b, t); }
        case 5u: { return mix_hsl(a, b, t); }
        case 6u: { return mix_hsl_long(a, b, t); }
        case 7u: { return mix_hsv(a, b, t); }
        case 8u: { return mix_hsv_long(a, b, t); }
        default: { return mix(a, b, t); }
    }
}

fn sample_gradient(index: u32, point: vec2<f32>) -> vec4<f32> {
    let gradient = load_gradient(index);
    let offset = point - gradient.start;
    var distance = dot(offset, gradient.direction);
    if gradient.kind == 1u {
        distance = length(vec2(offset.x, offset.y * gradient.direction.x));
    } else if gradient.kind == 2u {
        let angle = atan2(-offset.x, offset.y) + PI - gradient.direction.x;
        distance = ((angle % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
    }
    let last_index = gradient.stops_count - 1u;
    let first = load_stop(gradient.stops_start, 0u);
    let last = load_stop(gradient.stops_start, last_index);
    if distance < first.point { return to_linear(first.color, gradient.color_space); }
    if distance >= last.point || last_index == 0u { return to_linear(last.color, gradient.color_space); }

    // Resolve one segment, including hard stops, rather than rasterizing the node
    // once per segment. Equal-position stops select the color on their right.
    var low = 0u;
    var high = last_index;
    var a = first;
    var b = last;
    while high - low > 1u {
        let mid = low + (high - low) / 2u;
        let stop = load_stop(gradient.stops_start, mid);
        if stop.point <= distance { low = mid; a = stop; }
        else { high = mid; b = stop; }
    }
    var t = (distance - a.point) / (b.point - a.point);
    if t < a.hint { t = 0.5 * t / a.hint; }
    else { t = 0.5 * (1.0 + (t - a.hint) / (1.0 - a.hint)); }
    return to_linear(vec4(mix_colors(a.color.rgb, b.color.rgb, t, gradient.color_space), mix(a.color.a, b.color.a, t)), gradient.color_space);
}
