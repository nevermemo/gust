use super::{compile::*, error::*, json::*, model::*, pod::*, read::*};
#[cfg(feature = "slang-runtime")]
use crate::SourceOrigin;
use crate::{
    GpuPod, KernelDescriptor, ScalarKind, ShaderLayoutRules, TypeLayout, TypeLayoutKind,
    abi::{MatrixLayout, TextureAccess, TextureDimension},
    slang::{self, Target},
    texture::{Format as TexelFormat, SampleClass},
};
#[cfg(feature = "slang-runtime")]
use std::{fs, path::Path};

mod fragment_support;
mod helper;
mod schema_four;
mod schema_three;
mod schema_two;
mod tamper;

const SOURCE: &str = "line one\nline two\n";
const LINES: &[crate::LineSpan] = &[crate::LineSpan {
    slang_start: 2,
    slang_end: 2,
    rust_line: 9,
}];
const KERNEL: KernelDescriptor = KernelDescriptor::new("m", "k", "entry", [1, 1, 1], &[], SOURCE)
    .with_source(SourceOrigin {
        file: "src/k.rs",
        lines: LINES,
    });
const ARTIFACT: &[u8] = b"artifact bytes";

// ---- schema 2: vectors, matrices and constant buffers (slice m6.a1) ----

use crate::{
    Access, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout,
    artifact::{Artifact, Format, verify_artifact},
    math::{Mat2, Mat4, Vec2, Vec4},
};

const F32_LAYOUT: TypeLayout = <f32 as GpuPod>::LAYOUT;
const VEC2: TypeLayout = <Vec2 as GpuPod>::LAYOUT;
const VEC4: TypeLayout = <Vec4 as GpuPod>::LAYOUT;
const MAT2: TypeLayout = <Mat2 as GpuPod>::LAYOUT;
const MAT4: TypeLayout = <Mat4 as GpuPod>::LAYOUT;

const fn member(name: &'static str, offset: u32, ty: TypeLayout) -> crate::StructFieldLayout {
    crate::StructFieldLayout { name, offset, ty }
}

/// `TypeLayout::structure` for fields built at run time (its slice must be `'static`).
fn make_struct(
    name: &'static str,
    size: u32,
    alignment: u32,
    fields: &[crate::StructFieldLayout],
) -> TypeLayout {
    TypeLayout::structure(
        name,
        size,
        alignment,
        Box::leak(fields.to_vec().into_boxed_slice()),
    )
}

/// `{a, v2, b}` (storage): a0 v8 b16, size 24.
const STORAGE_STRUCT: TypeLayout = TypeLayout::structure(
    "FV2F",
    24,
    8,
    &[
        member("a", 0, F32_LAYOUT),
        member("v", 8, VEC2),
        member("b", 16, F32_LAYOUT),
    ],
);
const E_LAYOUT: TypeLayout = TypeLayout::structure("E", 16, 16, &[member("a", 0, F32_LAYOUT)]);
/// The review's `cb.json` constant buffer: v16 m32 e96 w112 b120, size 128.
const CB_STRUCT: TypeLayout = TypeLayout::structure(
    "P",
    128,
    16,
    &[
        member("a", 0, F32_LAYOUT),
        member("v", 16, VEC4),
        member("m", 32, MAT4),
        member("e", 96, E_LAYOUT),
        member("w", 112, VEC2),
        member("b", 120, F32_LAYOUT),
    ],
);

fn scalar_name(kind: ScalarKind) -> &'static str {
    match kind {
        ScalarKind::F32 => "float32",
        ScalarKind::I32 => "int32",
        ScalarKind::U32 => "uint32",
    }
}

/// The body (without braces) of one type in the schema-2 shape that plan section 2.4 specifies for the helper.
fn type_body(layout: TypeLayout) -> String {
    let stride = match layout.kind {
        TypeLayoutKind::Matrix { column_stride, .. } => column_stride,
        _ => layout.stride,
    };
    let head = format!(
        r#""size":{},"alignment":{},"stride":{stride},"name":"{}""#,
        layout.size, layout.alignment, layout.name
    );
    let tail = match layout.kind {
        TypeLayoutKind::Scalar(scalar) => {
            format!(r#""kind":"scalar","scalarType":"{}""#, scalar_name(scalar))
        }
        TypeLayoutKind::Vector { scalar, count } => format!(
            r#""kind":"vector","scalarType":"{}","elementCount":{count}"#,
            scalar_name(scalar)
        ),
        TypeLayoutKind::Matrix {
            scalar, rows, cols, ..
        } => format!(
            r#""kind":"matrix","scalarType":"{}","rows":{rows},"columns":{cols},"matrixLayout":"column""#,
            scalar_name(scalar)
        ),
        TypeLayoutKind::Struct(fields) => {
            let fields: Vec<String> = fields
                    .iter()
                    .map(|field| {
                        format!(
                            r#"{{"name":"{}","binding":{{"kind":"uniform","offset":{},"size":{}}},"type":{{{}}}}}"#,
                            field.name,
                            field.offset,
                            field.ty.size,
                            type_body(field.ty)
                        )
                    })
                    .collect();
            format!(r#""kind":"struct","fields":[{}]"#, fields.join(","))
        }
        TypeLayoutKind::Array { element, count } => format!(
            r#""kind":"array","elementCount":{count},"elementStride":{},"elementType":{{{}}}"#,
            element.stride,
            type_body(*element)
        ),
    };
    format!("{head},{tail}")
}

fn storage_json(name: &str, index: u32, layout: TypeLayout, access: &str) -> String {
    format!(
        r#"{{"name":"{name}","binding":{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}},"elementStride":{},"type":{{"kind":"resource","baseShape":"structuredBuffer","access":"{access}","resultType":{{{}}}}}}}"#,
        layout.stride,
        type_body(layout)
    )
}

fn uniform_json(name: &str, index: u32, layout: TypeLayout) -> String {
    format!(
        r#"{{"name":"{name}","binding":{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}},"type":{{"kind":"constantBuffer","size":{},"elementType":{{{}}}}}}}"#,
        layout.size,
        type_body(layout)
    )
}

/// A whole document whose identity fields match `SOURCE` and `ARTIFACT`.
fn document(schema: u32, matrix_layout: Option<&str>, parameters: &[String]) -> String {
    let source = fingerprint(SOURCE.as_bytes());
    let artifact = fingerprint(ARTIFACT);
    let matrix = matrix_layout
        .map(|layout| format!(r#""matrixLayout":"{layout}","#))
        .unwrap_or_default();
    format!(
        r#"{{"schemaVersion":{schema},{matrix}"compiler":"canned","target":"wgsl","entryPoint":"entry","sourceHash":"{source}","artifactHash":"{artifact}","workgroupSize":[1,1,1],"parameters":[{}]}}"#,
        parameters.join(",")
    )
}

fn parameter(
    name: &'static str,
    kind: ParameterKind,
    access: Access,
    layout: ResourceLayout,
    index: u32,
) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "T",
        kind,
        access,
        resource_layout: Some(layout),
        binding: Some(ResourceBinding {
            group: 0,
            binding: index,
        }),
    }
}

fn kernel_with(parameters: Vec<ParameterDescriptor>) -> KernelDescriptor {
    KERNEL.with_parameters(Box::leak(parameters.into_boxed_slice()))
}

fn read(json: &str, kernel: &KernelDescriptor) -> Result<CompiledReflection, Error> {
    CompiledReflection::from_output(kernel, SOURCE, Target::Wgsl, ARTIFACT.to_vec(), json)
}

/// Read `json`, which must parse, and return the message of the cross-check's `Mismatch`.
fn mismatch_of(json: &str, kernel: &KernelDescriptor) -> String {
    let compiled = read(json, kernel).unwrap_or_else(|error| panic!("{json}\n{error}"));
    match compiled.reflection.cross_check_kernel(kernel) {
        Err(Error::Mismatch(message)) => message,
        other => panic!("expected a mismatch, got {other:?}"),
    }
}

/// Storage buffers of a struct with a vector and of `Mat4`, and a constant buffer: the three new things at once.
fn full_kernel() -> KernelDescriptor {
    kernel_with(vec![
        parameter(
            "pts",
            ParameterKind::Storage,
            Access::ReadOnly,
            ResourceLayout::storage_v2(STORAGE_STRUCT),
            0,
        ),
        parameter(
            "mats",
            ParameterKind::Storage,
            Access::ReadWrite,
            ResourceLayout::storage_v2(MAT4),
            1,
        ),
        parameter(
            "cb",
            ParameterKind::Uniform,
            Access::ReadOnly,
            ResourceLayout::std140_v1(CB_STRUCT),
            2,
        ),
    ])
}

fn full_parameters() -> Vec<String> {
    vec![
        storage_json("pts", 0, STORAGE_STRUCT, "read"),
        storage_json("mats", 1, MAT4, "readWrite"),
        uniform_json("cb", 2, CB_STRUCT),
    ]
}
