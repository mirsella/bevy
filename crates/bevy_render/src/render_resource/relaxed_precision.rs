//! Explicit fragment-only Vulkan experiment. No numeric types, control flow,
//! memory operations, layouts or vertex stages are changed.

use super::{RenderDevice, ShaderCacheError, WgpuShaderModule};
use spirv::{Decoration, Op};
use std::collections::HashSet;

pub(super) fn validate_layout(
    descriptor: &super::RenderPipelineDescriptor,
) -> Result<(), ShaderCacheError> {
    let enabled = descriptor.fragment.as_ref().is_some_and(|fragment| {
        fragment.shader_defs.iter().any(|def| {
            matches!(def, super::ShaderDefVal::Bool(name, true)
                if name == bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT)
        })
    });
    if enabled
        && (descriptor.layout.is_empty()
            || descriptor.immediate_size != 0
            || descriptor.layout.iter().any(|group| {
                group
                    .entries
                    .iter()
                    .enumerate()
                    .any(|(index, entry)| entry.binding as usize != index || entry.count.is_some())
            }))
    {
        // wgpu densely renumbers Vulkan descriptor bindings and specializes
        // binding arrays. Passthrough modules bypass those transformations.
        return Err(ShaderCacheError::CreateShaderModule(
            "experimental precision requires explicit ordered dense bindings, without binding arrays or immediates".into(),
        ));
    }
    Ok(())
}

fn instructions(words: &[u32]) -> impl Iterator<Item = (usize, Op, &[u32])> {
    assert_eq!(words[0], 0x0723_0203, "invalid generated SPIR-V");
    let mut offset = 5;
    core::iter::from_fn(move || {
        if offset == words.len() {
            return None;
        }
        let start = offset;
        let count = (words[start] >> 16) as usize;
        assert!(count > 0 && start + count <= words.len());
        offset += count;
        Some((
            start,
            Op::from_u32(words[start] & 0xffff).expect("unknown generated opcode"),
            &words[start..offset],
        ))
    })
}

fn words(module: &naga::Module) -> Result<Vec<u32>, String> {
    let mut fragments = module
        .entry_points
        .iter()
        .filter(|entry| entry.stage == naga::ShaderStage::Fragment);
    let entry = fragments
        .next()
        .ok_or("relaxed shader has no fragment entry point")?;
    if fragments.next().is_some() {
        return Err("relaxed shader has ambiguous fragment entry points".into());
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|error| error.to_string())?;
    let mut words = naga::back::spv::write_vec(
        module,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 0),
            // Stable function names identify semantic color helpers, not IDs or
            // incidental compiler instruction order. Geometry is never relaxed
            // merely because it shares a float type with a color.
            flags: naga::back::spv::Options::default().flags | naga::back::spv::WriterFlags::DEBUG,
            ..Default::default()
        },
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: entry.name.clone(),
        }),
    )
    .map_err(|error| error.to_string())?;
    let mut floats = HashSet::new();
    for (_, op, w) in instructions(&words) {
        match op {
            Op::TypeFloat if w[2] == 32 => {
                floats.insert(w[1]);
            }
            Op::TypeVector if floats.contains(&w[2]) => {
                floats.insert(w[1]);
            }
            _ => {}
        }
    }
    let color_helpers = instructions(&words)
        .filter_map(|(_, op, w)| {
            if op != Op::Name {
                return None;
            }
            let bytes = w[2..]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .take_while(|byte| *byte != 0)
                .collect::<Vec<_>>();
            let name = String::from_utf8(bytes).expect("Naga function name is UTF-8");
            (name.contains("manual_srgb") || name.contains("gamma_2_2_to_linear")).then_some(w[1])
        })
        .collect::<HashSet<_>>();
    let mut annotations = Vec::new();
    let mut in_color_helper = false;
    for (_, op, w) in instructions(&words) {
        if op == Op::Function {
            in_color_helper = color_helpers.contains(&w[2]);
        }
        if op == Op::FunctionEnd {
            in_color_helper = false;
        }
        if !in_color_helper
            && !matches!(op, Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod)
        {
            continue;
        }
        if matches!(
            op,
            Op::ExtInst
                | Op::FNegate
                | Op::FAdd
                | Op::FSub
                | Op::FMul
                | Op::FDiv
                | Op::FRem
                | Op::FMod
                | Op::VectorTimesScalar
                | Op::Dot
                | Op::Load
                | Op::VectorShuffle
                | Op::CompositeConstruct
                | Op::CompositeExtract
                | Op::CopyObject
                | Op::Select
                | Op::Phi
                | Op::FunctionParameter
                | Op::ImageSampleImplicitLod
                | Op::ImageSampleExplicitLod
        ) && w.len() >= 3
            && floats.contains(&w[1])
        {
            annotations.extend([
                (3 << 16) | Op::Decorate as u32,
                w[2],
                Decoration::RelaxedPrecision as u32,
            ]);
        }
    }
    if annotations.is_empty() {
        return Err("relaxed shader changed no floating-point values".into());
    }
    let offset = instructions(&words)
        .find(|(_, op, _)| *op == Op::TypeVoid)
        .ok_or("generated shader has no types")?
        .0;
    words.splice(offset..offset, annotations);
    Ok(words)
}

pub(super) fn load(
    device: &RenderDevice,
    backend: wgpu::Backend,
    module: naga::Module,
) -> Result<WgpuShaderModule, ShaderCacheError> {
    if backend != wgpu::Backend::Vulkan
        || !device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        return Err(ShaderCacheError::CreateShaderModule(
            "experimental precision requires advertised Vulkan SPIR-V passthrough".into(),
        ));
    }
    let words = words(&module).map_err(ShaderCacheError::CreateShaderModule)?;
    let scope = device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: validated Naga emits Vulkan SPIR-V; only RelaxedPrecision
    // annotations on permitted float result IDs are inserted. Interfaces and
    // bounds/control/memory instructions are unchanged. Tests also run spirv-val.
    let shader = unsafe {
        device.wgpu_device().create_shader_module_passthrough(
            wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("experimental relaxed fragment"),
                spirv: Some(words.into()),
                ..Default::default()
            },
        )
    };
    if let Some(Some(wgpu::Error::Validation { description, .. })) =
        bevy_tasks::futures::now_or_never(scope.pop())
    {
        return Err(ShaderCacheError::CreateShaderModule(description));
    }
    Ok(WgpuShaderModule::new(shader))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    #[test]
    fn sparse_bindings_are_rejected_before_vulkan_passthrough() {
        use super::super::{BindGroupLayoutDescriptor, FragmentState, RenderPipelineDescriptor};
        let mut descriptor = RenderPipelineDescriptor {
            fragment: Some(FragmentState {
                shader_defs: vec![bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT.into()],
                ..Default::default()
            }),
            layout: vec![BindGroupLayoutDescriptor::new(
                "test",
                &[wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            )],
            ..Default::default()
        };
        assert!(validate_layout(&descriptor).is_err());
        descriptor.fragment.as_mut().unwrap().shader_defs.clear();
        assert!(
            validate_layout(&descriptor).is_ok(),
            "ordinary pipelines still support sparse bindings"
        );
        descriptor
            .fragment
            .as_mut()
            .unwrap()
            .shader_defs
            .push(bevy_shader::EXPERIMENTAL_RELAXED_COLOR_FRAGMENT.into());
        descriptor.layout[0].entries[0].binding = 0;
        assert!(validate_layout(&descriptor).is_ok());
    }
    #[test]
    fn color_policy_leaves_coordinate_and_coverage_instructions_undecorated() {
        let module = naga::front::wgsl::parse_str("fn manual_srgb(c: vec3<f32>) -> vec3<f32> { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); } @fragment fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { let point = uv * vec2(2000.0) - vec2(1000.0); let coverage = clamp(0.5 - length(point), 0.0, 1.0); return vec4(manual_srgb(vec3(0.2, 0.4, 0.8)), coverage); }").unwrap();
        let words = words(&module).unwrap();
        let annotations = instructions(&words)
            .filter_map(|(_, op, w)| {
                if op == Op::Decorate && w[2] == Decoration::RelaxedPrecision as u32 {
                    Some(w[1])
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
        assert!(!annotations.is_empty(), "gamma helper must be exercised");
        let helper = instructions(&words)
            .find_map(|(_, op, w)| {
                if op != Op::Name {
                    return None;
                }
                let bytes = w[2..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                String::from_utf8(bytes)
                    .unwrap()
                    .contains("manual_srgb")
                    .then_some(w[1])
            })
            .expect("debug function name");
        let mut in_helper = false;
        let mut geometry_operations = 0;
        for (_, op, w) in instructions(&words) {
            if op == Op::Function {
                in_helper = w[2] == helper;
            }
            if op == Op::FunctionEnd {
                in_helper = false;
            }
            if !in_helper && matches!(op, Op::Load | Op::FMul | Op::FSub | Op::ExtInst) {
                geometry_operations += 1;
                assert!(
                    !annotations.contains(&w[2]),
                    "geometry/coordinate result was relaxed"
                );
            }
        }
        assert!(geometry_operations >= 3);
        let mut validation = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.0", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .expect("spirv-val required");
        validation
            .stdin
            .take()
            .unwrap()
            .write_all(
                &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(validation.wait().unwrap().success());
    }
}
