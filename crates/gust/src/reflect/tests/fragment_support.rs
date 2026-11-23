use super::*;

// ---- schema 3: textures, samplers, the stage and a fragment's targets (slice mx.t0b, D91) ----

use crate::{TextureLayout, math::UVec2};

pub(super) const U32_LAYOUT: TypeLayout = <u32 as GpuPod>::LAYOUT;
pub(super) const UVEC2: TypeLayout = <UVec2 as GpuPod>::LAYOUT;

/// A texture or target result type: a scalar for `count == 1`, a vector otherwise.
pub(super) fn result_json(scalar: &str, count: u32) -> String {
    if count == 1 {
        format!(r#"{{"kind":"scalar","scalarType":"{scalar}"}}"#)
    } else {
        format!(r#"{{"kind":"vector","scalarType":"{scalar}","elementCount":{count}}}"#)
    }
}

pub(super) const F4: (&str, u32) = ("float32", 4);

pub(super) fn slot_json(index: u32) -> String {
    format!(r#"{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}}"#)
}

pub(super) fn texture_json(
    name: &str,
    index: u32,
    access: &str,
    result: (&str, u32),
    format: &str,
) -> String {
    format!(
        r#"{{"name":"{name}","binding":{},"type":{{"kind":"texture","shape":"2d","access":"{access}","result":{},"format":"{format}"}}}}"#,
        slot_json(index),
        result_json(result.0, result.1)
    )
}

pub(super) fn sampler_json(name: &str, index: u32) -> String {
    format!(
        r#"{{"name":"{name}","binding":{},"type":{{"kind":"sampler"}}}}"#,
        slot_json(index)
    )
}

pub(super) fn target_json(index: u32, result: (&str, u32)) -> String {
    format!(
        r#"{{"index":{index},"type":{}}}"#,
        result_json(result.0, result.1)
    )
}

/// A schema-3 document whose identity fields match `SOURCE` and `ARTIFACT`: a compute one carries a workgroup
/// size, a fragment one `targets`.
pub(super) fn document3(stage: &str, parameters: &[String], targets: &[String]) -> String {
    let source = fingerprint(SOURCE.as_bytes());
    let artifact = fingerprint(ARTIFACT);
    let (workgroup, targets) = if stage == "compute" {
        (r#""workgroupSize":[1,1,1],"#.to_owned(), String::new())
    } else {
        (
            String::new(),
            format!(r#","targets":[{}]"#, targets.join(",")),
        )
    };
    format!(
        r#"{{"schemaVersion":3,"matrixLayout":"column","compiler":"canned","target":"wgsl","entryPoint":"entry","stage":"{stage}","sourceHash":"{source}","artifactHash":"{artifact}",{workgroup}"parameters":[{}]{targets}}}"#,
        parameters.join(",")
    )
}

pub(super) const fn sampled(class: crate::texture::SampleClass) -> TextureLayout {
    TextureLayout {
        format: None,
        dimension: TextureDimension::D2,
        access: TextureAccess::Sampled,
        class,
    }
}

pub(super) const fn storage(format: TexelFormat, access: TextureAccess) -> TextureLayout {
    TextureLayout {
        format: Some(format),
        dimension: TextureDimension::D2,
        access,
        class: format.sample_class(),
    }
}

pub(super) fn texture_parameter(
    name: &'static str,
    access: Access,
    element: TypeLayout,
    texture: TextureLayout,
    index: u32,
) -> ParameterDescriptor {
    parameter(
        name,
        ParameterKind::Texture,
        access,
        ResourceLayout::texture_v1(element, texture),
        index,
    )
}

pub(super) fn sampler_parameter(name: &'static str, index: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "Sampler",
        kind: ParameterKind::Sampler,
        access: Access::NotApplicable,
        resource_layout: None,
        binding: Some(ResourceBinding {
            group: 0,
            binding: index,
        }),
    }
}

/// An output buffer of a pixel kernel: a storage parameter with no shader binding (a colour target).
pub(super) fn output_parameter(name: &'static str, element: TypeLayout) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "RWStructuredBuffer<T>",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v2(element)),
        binding: None,
    }
}

pub(super) fn pixel_kernel(parameters: Vec<ParameterDescriptor>) -> KernelDescriptor {
    kernel_with(parameters).with_shape(crate::Shape::Pixel)
}

/// A compute kernel with every texture form and a sampler and a buffer: `tex` sampled `float4` at 0, `rw` an
/// `r32f` read-write texture at 1, `wo` an `rg32ui` write-only texture at 2, `samp` at 3, `out` at 4.
pub(super) fn texture_kernel() -> KernelDescriptor {
    kernel_with(vec![
        texture_parameter(
            "tex",
            Access::ReadOnly,
            VEC4,
            sampled(SampleClass::Float),
            0,
        ),
        texture_parameter(
            "rw",
            Access::ReadWrite,
            F32_LAYOUT,
            storage(TexelFormat::R32Float, TextureAccess::ReadWrite),
            1,
        ),
        texture_parameter(
            "wo",
            Access::ReadWrite,
            UVEC2,
            storage(TexelFormat::Rg32Uint, TextureAccess::WriteOnly),
            2,
        ),
        sampler_parameter("samp", 3),
        parameter(
            "out",
            ParameterKind::Storage,
            Access::ReadWrite,
            ResourceLayout::storage_v1(F32_LAYOUT),
            4,
        ),
    ])
}

pub(super) fn texture_parameters() -> Vec<String> {
    vec![
        texture_json("tex", 0, "read", F4, "unknown"),
        texture_json("rw", 1, "readWrite", ("float32", 1), "r32f"),
        texture_json("wo", 2, "write", ("uint32", 2), "rg32ui"),
        sampler_json("samp", 3),
        storage_json("out", 4, F32_LAYOUT, "readWrite"),
    ]
}

pub(super) fn texture_document() -> String {
    document3("compute", &texture_parameters(), &[])
}

/// A fragment kernel: `tex` and `samp`, one `float4` output.
pub(super) fn fragment_kernel() -> KernelDescriptor {
    pixel_kernel(vec![
        texture_parameter(
            "tex",
            Access::ReadOnly,
            VEC4,
            sampled(SampleClass::Float),
            0,
        ),
        sampler_parameter("samp", 1),
        output_parameter("color", VEC4),
    ])
}

pub(super) fn fragment_parameters() -> Vec<String> {
    vec![
        texture_json("tex", 0, "read", F4, "unknown"),
        sampler_json("samp", 1),
    ]
}

pub(super) fn fragment_document() -> String {
    document3("fragment", &fragment_parameters(), &[target_json(0, F4)])
}

/// The message of the `Mismatch` that reading `json` for `kernel` gives, at the read or at the cross-check.
pub(super) fn any_mismatch(json: &str, kernel: &KernelDescriptor) -> String {
    match read(json, kernel) {
        Err(Error::Mismatch(message)) => message,
        Err(other) => panic!("expected a mismatch, got {other:?}\n{json}"),
        Ok(compiled) => match compiled.reflection.cross_check_kernel(kernel) {
            Err(Error::Mismatch(message)) => message,
            other => panic!("expected a mismatch, got {other:?}\n{json}"),
        },
    }
}

pub(super) fn invalid_of(json: &str, kernel: &KernelDescriptor) -> String {
    match read(json, kernel) {
        Err(Error::Invalid(message)) => message,
        other => panic!("expected an invalid document, got {other:?}\n{json}"),
    }
}
