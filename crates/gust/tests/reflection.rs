use gust::{
    cross_check_pod,
    reflect::{Reflection, compile_reflected, compile_reflected_with_helper, reflect},
    slang::Target,
};

#[gust::gpu]
mod sample {
    pub struct Inner {
        pub value: float,
        pub tag: uint,
    }
    pub struct Outer {
        pub inner: Inner,
        pub weight: int,
    }
    #[kernel]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Outer>,
        mut output: RWStructuredBuffer<Outer>,
    ) {
        if id.x < output.len() {
            output[id.x] = input[id.x];
        }
    }
    #[kernel]
    pub fn scalar(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut output: RWStructuredBuffer<uint>,
    ) {
        if id.x < output.len() {
            output[id.x] = input[id.x];
        }
    }
}

#[test]
fn native_compilation_is_complete_for_both_targets() {
    for target in [Target::Wgsl, Target::Spirv] {
        for kernel in [sample::run::DESCRIPTOR, sample::scalar::DESCRIPTOR] {
            let compiled = compile_reflected(&kernel, target).unwrap();
            assert!(!compiled.artifact.is_empty());
            assert!(!compiled.compiler_version.is_empty());
            assert_eq!(compiled.entry_point, kernel.entry_point);
            assert_eq!(compiled.workgroup_size, kernel.workgroup_size);
            assert_eq!(compiled.reflection.target, target);
            let reports = compiled.reflection.cross_check_kernel(&kernel).unwrap();
            assert_eq!(reports.len(), 2);
            assert!(reports.iter().all(|report| report.is_full_abi_match()));
            if kernel.name == "run" {
                let resource = &compiled.reflection.resources[0];
                assert_eq!(resource.element.size, Some(12));
                assert_eq!(resource.element.alignment, Some(4));
                assert_eq!(resource.element.stride, Some(12));
                assert_eq!(resource.element_stride, Some(12));
                assert_eq!(
                    cross_check_pod::<sample::Outer>(resource)
                        .unwrap()
                        .checked_fields,
                    4
                );
            } else {
                assert!(
                    cross_check_pod::<u32>(&compiled.reflection.resources[0])
                        .unwrap()
                        .is_full_abi_match()
                );
            }
            match target {
                Target::Wgsl => assert!(
                    String::from_utf8(compiled.artifact)
                        .unwrap()
                        .contains(kernel.entry_point)
                ),
                Target::Spirv => {
                    let words: Vec<_> = compiled
                        .artifact
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_le_bytes(*word))
                        .collect();
                    gust::spirv::validate_structure(&words).unwrap();
                }
            }
        }
    }
}

#[test]
fn native_missing_helper_is_an_explicit_error() {
    let directory = gust::slang::TemporaryDirectory::create().unwrap();
    let error = compile_reflected_with_helper(
        &sample::run::DESCRIPTOR,
        Target::Wgsl,
        directory.path().join("missing reflection compiler.exe"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        gust::reflect::Error::HelperUnavailable { .. }
    ));
    assert!(error.to_string().contains("GUST_SLANG_REFLECT"));
}

/// `scripts/probes/layout.slang` as a kernel: eight global resources, among them a padded `float3` struct, a matrix,
/// a non-square matrix, a struct with a `float2` and a constant buffer.
fn layout_kernel() -> gust::KernelDescriptor {
    let mut kernel = sample::run::DESCRIPTOR;
    kernel.slang_source = include_str!("../../../scripts/probes/layout.slang");
    kernel.entry_point = "main";
    kernel.workgroup_size = [64, 1, 1];
    kernel
}

fn struct_fields(resource: &gust::ReflectedResource) -> &[gust::ReflectedField] {
    let gust::ReflectedTypeKind::Struct(fields) = &resource.element.kind else {
        panic!("{}: a struct element expected", resource.name)
    };
    fields
}

/// (offset, size) of each field, by name.
fn field_placement(resource: &gust::ReflectedResource) -> Vec<(&str, u32, u32)> {
    struct_fields(resource)
        .iter()
        .map(|field| (field.name.as_str(), field.offset, field.size))
        .collect()
}

fn vector(count: u32) -> gust::ReflectedTypeKind {
    gust::ReflectedTypeKind::Vector {
        scalar: gust::ScalarKind::F32,
        count,
    }
}

/// A matrix's kind, its layout spelled as the document spells it, and the three byte numbers of its type.
fn matrix_of(ty: &gust::ReflectedType) -> (u32, u32, &'static str, [Option<u32>; 3]) {
    let gust::ReflectedTypeKind::Matrix {
        scalar,
        rows,
        cols,
        layout,
    } = &ty.kind
    else {
        panic!("a matrix expected, got {ty:?}")
    };
    assert_eq!(*scalar, gust::ScalarKind::F32);
    (
        *rows,
        *cols,
        layout.as_str(),
        [ty.size, ty.alignment, ty.stride],
    )
}

/// Slice m6.r: the helper reflects vectors, matrices and a constant buffer, as schema 2, with the numbers V1
/// measured (padded `float3`, the column stride of a matrix, the rounded size of a uniform).
#[test]
fn native_reflects_vectors_matrices_and_a_constant_buffer_as_schema_two() {
    for target in [Target::Wgsl, Target::Spirv] {
        let compiled = compile_reflected(&layout_kernel(), target)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        let reflection = &compiled.reflection;
        assert_eq!(reflection.schema_version, 2, "{target:?}");
        let names: Vec<&str> = reflection
            .resources
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "scalars", "pods", "padded", "output", "matrices", "lits", "frame", "wide"
            ],
            "{target:?}"
        );
        let resource = |name: &str| {
            reflection
                .resources
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("{name}"))
        };

        // A vector member: float3 is 12 bytes at a 16-byte alignment, and the next scalar packs into its tail.
        let padded = resource("padded");
        assert_eq!(
            field_placement(padded),
            [("head", 0, 4), ("direction", 16, 12), ("tail", 28, 4)]
        );
        assert_eq!(struct_fields(padded)[1].ty.kind, vector(3));
        assert_eq!(
            (
                padded.element.size,
                padded.element.alignment,
                padded.element.stride
            ),
            (Some(32), Some(16), Some(32))
        );
        assert_eq!(padded.element_stride, Some(32));

        let lits = resource("lits");
        assert_eq!(
            field_placement(lits),
            [("a", 0, 4), ("pos", 8, 8), ("b", 16, 4)]
        );
        assert_eq!(struct_fields(lits)[1].ty.kind, vector(2));
        assert_eq!(
            (
                lits.element.size,
                lits.element.alignment,
                lits.element_stride
            ),
            (Some(24), Some(8), Some(24))
        );

        // A matrix: pinned column-major; `stride` is the column stride, `elementStride` the whole matrix.
        let matrices = resource("matrices");
        assert_eq!(
            matrices.kind,
            gust::reflect::ReflectedResourceKind::StructuredBuffer
        );
        assert_eq!(
            matrix_of(&matrices.element),
            (4, 4, "column", [Some(64), Some(16), Some(16)])
        );
        assert_eq!(matrices.element_stride, Some(64));
        // Not square: 2 rows by 4 columns, columns of two floats (8 bytes), neither the row stride (16) nor the whole.
        let wide = resource("wide");
        assert_eq!(
            matrix_of(&wide.element),
            (2, 4, "column", [Some(32), Some(8), Some(8)])
        );
        assert_eq!(wide.element_stride, Some(32));
        // The host `Mat4` agrees with the compiler on every number.
        let mat4 = gust::cross_check_pod::<gust::math::Mat4>(matrices)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        assert!(mat4.is_full_abi_match(), "{target:?}");

        // A constant buffer: one descriptor slot, read-only, the element in the Uniform category; a nested struct and
        // the buffer itself round up to 16 bytes.
        let frame = resource("frame");
        assert_eq!(
            frame.kind,
            gust::reflect::ReflectedResourceKind::ConstantBuffer
        );
        assert_eq!(
            (frame.binding, frame.space, frame.access),
            (6, 0, gust::Access::ReadOnly)
        );
        assert_eq!(
            (
                frame.element.size,
                frame.element.alignment,
                frame.element_stride
            ),
            (Some(128), Some(16), Some(128))
        );
        assert_eq!(
            field_placement(frame),
            [
                ("scale", 0, 4),
                ("tint", 16, 16),
                ("view", 32, 64),
                ("inner", 96, 16),
                ("offset", 112, 8),
                ("bias", 120, 4)
            ]
        );
        let fields = struct_fields(frame);
        assert_eq!(fields[1].ty.kind, vector(4));
        assert_eq!(
            matrix_of(&fields[2].ty),
            (4, 4, "column", [Some(64), Some(16), Some(16)])
        );
        assert_eq!(fields[3].ty.size, Some(16), "{target:?}");
        assert_eq!(fields[4].ty.kind, vector(2));
    }
}

/// Slice m6.r (plan R1.1-1): a program without a vector, matrix or constant buffer stays schema 1, so every committed
/// scalar artifact set keeps its bytes.
#[test]
fn native_scalar_programs_stay_schema_one() {
    for target in [Target::Wgsl, Target::Spirv] {
        for kernel in [sample::run::DESCRIPTOR, sample::scalar::DESCRIPTOR] {
            let compiled = compile_reflected(&kernel, target).unwrap();
            assert_eq!(compiled.reflection.schema_version, 1, "{target:?}");
        }
    }
}

/// What the helper still has no layout for is refused by the helper itself, not by the Rust reader afterwards.
#[test]
fn native_still_refuses_what_the_dialect_cannot_describe() {
    const OUTPUT: &str = "[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;\n[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) { ";
    let array = format!(
        "[[vk::binding(0, 0)]] StructuredBuffer<float[4]> input;\n{OUTPUT}output[id.x] = input[id.x][1]; }}\n"
    );
    let wide_scalar = format!(
        "[[vk::binding(0, 0)]] StructuredBuffer<double2> input;\n{OUTPUT}output[id.x] = float(input[id.x].x); }}\n"
    );
    for (source, target, message) in [
        (&array, Target::Wgsl, "unsupported element type"),
        (&array, Target::Spirv, "unsupported element type"),
        // WGSL has no 64-bit float: Slang itself refuses that program, so only SPIR-V reaches the helper.
        (&wide_scalar, Target::Spirv, "unsupported scalar type"),
    ] {
        let mut kernel = layout_kernel();
        kernel.slang_source = Box::leak(source.clone().into_boxed_str());
        let error = compile_reflected(&kernel, target).unwrap_err();
        assert!(
            matches!(
                &error,
                gust::reflect::Error::Compiler(
                    gust::slang::Error::CompilationFailed { target: failed, .. }
                ) if *failed == target
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn compiler_reports_nested_fields_for_both_targets() {
    for target in [Target::Wgsl, Target::Spirv] {
        let reflection = reflect(&sample::run::DESCRIPTOR, target).unwrap();
        assert_eq!(reflection.resources.len(), 2);
        let input = &reflection.resources[0];
        assert_eq!((input.binding, input.space), (0, 0));
        let checked = cross_check_pod::<sample::Outer>(input).unwrap();
        assert_eq!(checked.checked_fields, 4);
        assert!(!checked.alignment_verified);
        assert!(!checked.aggregate_size_verified);
        assert!(!checked.element_stride_verified);
        assert!(!checked.is_full_abi_match());
        assert_eq!(input.access, gust::Access::ReadOnly);
        assert_eq!(reflection.resources[1].access, gust::Access::ReadWrite);
        assert!(cross_check_pod::<u32>(input).is_err());
        let mut mismatch = input.clone();
        mismatch.element_stride = Some(64);
        assert!(cross_check_pod::<sample::Outer>(&mismatch).is_err());
        for change in 0..5 {
            let mut mismatch = input.clone();
            let gust::ReflectedTypeKind::Struct(fields) = &mut mismatch.element.kind else {
                panic!("struct expected")
            };
            match change {
                0 => fields[0].offset = 4,
                1 => fields[0].size = 4,
                2 => fields[0].name = "wrong".into(),
                3 => {
                    fields.pop();
                }
                _ => fields[1].ty.kind = gust::ReflectedTypeKind::Scalar(gust::ScalarKind::U32),
            }
            assert!(
                cross_check_pod::<sample::Outer>(&mismatch).is_err(),
                "mutation {change}"
            );
        }
    }
}

#[test]
fn outer_size_does_not_certify_missing_nested_sizes() {
    let reflection = reflect(&sample::run::DESCRIPTOR, Target::Wgsl).unwrap();
    let mut input = reflection.resources[0].clone();
    input.element.size = Some(12);
    input.element_stride = Some(12);
    let checked = cross_check_pod::<sample::Outer>(&input).unwrap();
    assert!(!checked.aggregate_size_verified);
    assert!(!checked.is_full_abi_match());
}

#[test]
fn compiler_failure_preserves_target_and_kernel_diagnostics() {
    let mut kernel = sample::run::DESCRIPTOR;
    kernel.slang_source = "this is invalid Slang";
    let error = reflect(&kernel, Target::Wgsl).unwrap_err();
    assert!(matches!(
        error,
        gust::reflect::Error::Compiler(gust::slang::Error::CompilationFailed {
            target: Target::Wgsl,
            ..
        })
    ));
    assert!(error.to_string().contains("compiler exit:"));
    assert!(
        error
            .to_string()
            .contains("originating Rust kernel: sample::run")
    );
}

#[test]
fn scalar_fixture_records_nonzero_space_and_rejects_unknown_layouts() {
    let source = r#"{"parameters":[{"name":"a","binding":{"kind":"descriptorTableSlot","index":7,"space":2},"type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"uint32"}}}]}"#;
    let reflection = Reflection::from_json(Target::Spirv, source).unwrap();
    let resource = &reflection.resources[0];
    assert_eq!((resource.binding, resource.space), (7, 2));
    assert_eq!(cross_check_pod::<u32>(resource).unwrap().checked_fields, 0);
    for invalid in [
        source.replace("uint32", "bool"),
        source.replace("structuredBuffer", "texture2D"),
        source.replace("\"space\":2", "\"space\":-1"),
        source.replace("\"space\":2", "\"count\":2"),
        source.replace("\"resultType\"", "\"access\":\"unknown\",\"resultType\""),
    ] {
        assert!(Reflection::from_json(Target::Spirv, &invalid).is_err());
    }
}

#[test]
fn rejects_malformed_or_unsupported_reflection() {
    for json in [
        "{}",
        "{\"parameters\":[],}",
        "{\"parameters\":[],\"parameters\":[]}",
        "{\"parameters\":[]} trailing",
        "{\"parameters\":[{}]}",
    ] {
        assert!(Reflection::from_json(Target::Wgsl, json).is_err(), "{json}");
    }
}

#[test]
fn entry_point_resources_are_not_silently_ignored() {
    for parameter in [
        r#"{"name":"buffer","type":{"kind":"resource"}}"#,
        r#"{"name":"uniforms","type":{"kind":"parameterBlock"}}"#,
        r#"{"name":"value","binding":{"kind":"uniform","offset":0,"size":4},"type":{"kind":"scalar","scalarType":"uint32"}}"#,
        r#"{"name":"value","type":{"kind":"scalar","scalarType":"uint32"}}"#,
    ] {
        let json = format!(r#"{{"parameters":[],"entryPoints":[{{"parameters":[{parameter}]}}]}}"#);
        assert!(
            Reflection::from_json(Target::Wgsl, &json).is_err(),
            "{json}"
        );
    }
}

/// D17 on renamed names (M6 plan 2.5): a kernel whose struct (`class`), field (`new`, `This`) and parameter (`new`,
/// `operator`, `This`) names are renamed. (A parameter and a struct of one name would conflict in Slang's single
/// namespace, renamed or not, so the parameters do not reuse the struct name.) The descriptor
/// carries the emitted names, the compiler reflects them, and the cross-check passes; a descriptor that kept the Rust
/// spelling, or a reflected field with another name, fails it.
#[allow(non_camel_case_types, non_snake_case)]
#[gust::gpu]
mod renamed_reflect {
    pub struct class {
        pub new: uint,
        pub This: float,
    }
    #[kernel]
    pub fn run(
        new: SV_DispatchThreadID,
        operator: StructuredBuffer<class>,
        mut This: RWStructuredBuffer<class>,
    ) {
        if new.x < This.len() {
            This[new.x] = operator[new.x];
        }
    }
}

#[test]
fn the_layout_cross_check_passes_on_renamed_names_and_fails_on_the_rust_spelling() {
    let kernel = renamed_reflect::run::DESCRIPTOR;
    let names: Vec<&str> = kernel.parameters.iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        ["__gust_r_new", "__gust_r_operator", "__gust_r_This"]
    );
    for target in [Target::Wgsl, Target::Spirv] {
        let compiled = compile_reflected(&kernel, target)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        let resources = &compiled.reflection.resources;
        let reflected: Vec<&str> = resources.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            reflected,
            ["__gust_r_operator", "__gust_r_This"],
            "{target:?}"
        );
        let reports = compiled
            .reflection
            .cross_check_kernel(&kernel)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| report.is_full_abi_match()));
        // The host layout names the fields as the compiler does, and the per-field check ran on them.
        let host = <renamed_reflect::class as gust::GpuPod>::LAYOUT;
        let gust::TypeLayoutKind::Struct(fields) = host.kind else {
            panic!("a struct layout");
        };
        let host_fields: Vec<&str> = fields.iter().map(|field| field.name).collect();
        assert_eq!(host_fields, ["__gust_r_new", "__gust_r_This"]);
        let checked = cross_check_pod::<renamed_reflect::class>(&resources[0])
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        assert_eq!(checked.checked_fields, 2);

        // Mutation 1: a reflected field with the Rust spelling is a mismatch.
        let mut mutated = resources[0].clone();
        let gust::reflect::ReflectedTypeKind::Struct(fields) = &mut mutated.element.kind else {
            panic!("a reflected struct");
        };
        fields[0].name = "new".to_owned();
        let error = cross_check_pod::<renamed_reflect::class>(&mutated).unwrap_err();
        assert!(
            error.to_string().contains("field name/order differs"),
            "{error}"
        );

        // Mutation 2: a descriptor parameter with the Rust spelling finds no compiler binding.
        let mut parameters = kernel.parameters.to_vec();
        parameters[1].name = "operator";
        let respelled = gust::KernelDescriptor {
            parameters: Box::leak(parameters.into_boxed_slice()),
            ..kernel
        };
        let error = compiled
            .reflection
            .cross_check_kernel(&respelled)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("missing compiler binding operator"),
            "{error}"
        );
    }
}

/// Slice m6.r fix round 1: the two compile paths agree on a matrix kernel. The CLI path (`compile_spirv`,
/// `compile_wgsl`) pins `-matrix-layout-column-major`, the helper pins `defaultMatrixLayoutMode`; if Slang changes
/// either default (the 2026.20 bump), a string test of the arguments cannot notice but these bytes do.
#[test]
fn the_cli_and_native_paths_compile_the_matrix_kernel_to_the_same_bytes() {
    let kernel = layout_kernel();
    let words = gust::slang::compile_spirv(&kernel).unwrap();
    let cli: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let native = compile_reflected(&kernel, Target::Spirv).unwrap();
    assert!(!cli.is_empty());
    assert_eq!(cli, native.artifact, "SPIR-V");
    // The WGSL of the two paths differs only in line endings (README).
    let cli = gust::slang::compile_wgsl(&kernel)
        .unwrap()
        .replace('\r', "");
    let native = compile_reflected(&kernel, Target::Wgsl).unwrap();
    let native = String::from_utf8(native.artifact)
        .unwrap()
        .replace('\r', "");
    assert!(!cli.is_empty());
    assert_eq!(cli, native, "WGSL");
}

mod host_layouts {
    use gust::{
        GpuPod, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout,
        StructFieldLayout, TypeLayout,
        math::{Mat4, Vec2, Vec4},
    };

    const F32: TypeLayout = <f32 as GpuPod>::LAYOUT;
    const INNER: TypeLayout = TypeLayout::structure(
        "Inner",
        16,
        16,
        &[
            StructFieldLayout {
                name: "value",
                offset: 0,
                ty: F32,
            },
            StructFieldLayout {
                name: "tag",
                offset: 4,
                ty: <u32 as GpuPod>::LAYOUT,
            },
        ],
    );
    /// `Lit` of the source below: a@0 pos@8 b@16, size 24 (storage).
    pub const LIT: TypeLayout = TypeLayout::structure(
        "Lit",
        24,
        8,
        &[
            StructFieldLayout {
                name: "a",
                offset: 0,
                ty: F32,
            },
            StructFieldLayout {
                name: "pos",
                offset: 8,
                ty: <Vec2 as GpuPod>::LAYOUT,
            },
            StructFieldLayout {
                name: "b",
                offset: 16,
                ty: F32,
            },
        ],
    );
    /// `Frame` of the source below, in the uniform rules: size 128.
    pub const FRAME: TypeLayout = TypeLayout::structure(
        "Frame",
        128,
        16,
        &[
            StructFieldLayout {
                name: "scale",
                offset: 0,
                ty: F32,
            },
            StructFieldLayout {
                name: "tint",
                offset: 16,
                ty: <Vec4 as GpuPod>::LAYOUT,
            },
            StructFieldLayout {
                name: "view",
                offset: 32,
                ty: <Mat4 as GpuPod>::LAYOUT,
            },
            StructFieldLayout {
                name: "inner",
                offset: 96,
                ty: INNER,
            },
            StructFieldLayout {
                name: "offset",
                offset: 112,
                ty: <Vec2 as GpuPod>::LAYOUT,
            },
            StructFieldLayout {
                name: "bias",
                offset: 120,
                ty: F32,
            },
        ],
    );

    pub const SOURCE: &str = "struct Inner { float value; uint tag; };\nstruct Lit { float a; float2 pos; float b; };\nstruct Frame { float scale; float4 tint; float4x4 view; Inner inner; float2 offset; float bias; };\n[[vk::binding(0, 0)]] RWStructuredBuffer<float> output;\n[[vk::binding(1, 0)]] StructuredBuffer<Lit> lits;\n[[vk::binding(2, 0)]] ConstantBuffer<Frame> frame;\n[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid main(uint3 id : SV_DispatchThreadID)\n{\n    output[id.x] = lits[id.x].a + lits[id.x].pos.y + lits[id.x].b + mul(frame.view, frame.tint).x\n        + frame.scale * frame.inner.value + float(frame.inner.tag) + frame.offset.x + frame.bias;\n}\n";

    pub fn parameters(frame: TypeLayout) -> Vec<ParameterDescriptor> {
        let parameter = |name, kind, access, layout, binding| ParameterDescriptor {
            name,
            rust_type: "T",
            kind,
            access,
            resource_layout: Some(layout),
            binding: Some(ResourceBinding { group: 0, binding }),
        };
        vec![
            parameter(
                "output",
                ParameterKind::Storage,
                gust::Access::ReadWrite,
                ResourceLayout::storage_v1(F32),
                0,
            ),
            parameter(
                "lits",
                ParameterKind::Storage,
                gust::Access::ReadOnly,
                ResourceLayout::storage_v2(LIT),
                1,
            ),
            parameter(
                "frame",
                ParameterKind::Uniform,
                gust::Access::ReadOnly,
                ResourceLayout::std140_v1(frame),
                2,
            ),
        ]
    }
}

/// Slice m6.r fix round 1: the real helper output end to end through `cross_check_kernel`, with hand-written host
/// layouts: a storage struct with a `Vec2` (`StorageV2`) and a constant buffer with a vector, a matrix, a nested
/// struct and a `Vec2` (`Std140V1`). The fields are checked, and a wrong host layout is refused.
#[test]
fn the_real_helper_output_passes_the_cross_check_for_storage_v2_and_std140_v1() {
    let mut kernel = layout_kernel();
    kernel.slang_source = host_layouts::SOURCE;
    kernel.parameters = Box::leak(host_layouts::parameters(host_layouts::FRAME).into_boxed_slice());
    for target in [Target::Wgsl, Target::Spirv] {
        let compiled = compile_reflected(&kernel, target)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        assert_eq!(compiled.reflection.schema_version, 2);
        let reports = compiled
            .reflection
            .cross_check_kernel(&kernel)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        assert_eq!(reports.len(), 3, "{target:?}");
        assert!(
            reports.iter().all(|report| report.is_full_abi_match()),
            "{target:?}: {reports:?}"
        );
        let checked: Vec<usize> = reports.iter().map(|report| report.checked_fields).collect();
        assert_eq!(checked, [0, 3, 8], "{target:?}");

        // A host layout that differs in one field name is refused (the cross-check is not vacuous).
        let mut moved = host_layouts::FRAME;
        let gust::TypeLayoutKind::Struct(fields) = moved.kind else {
            panic!("a struct layout")
        };
        let mut fields = fields.to_vec();
        fields[5].name = "bias2";
        moved.kind = gust::TypeLayoutKind::Struct(Box::leak(fields.into_boxed_slice()));
        let mut wrong = kernel;
        wrong.parameters = Box::leak(host_layouts::parameters(moved).into_boxed_slice());
        let error = compiled.reflection.cross_check_kernel(&wrong).unwrap_err();
        assert!(
            error.to_string().contains("field name/order differs"),
            "{target:?}: {error}"
        );
    }
}
