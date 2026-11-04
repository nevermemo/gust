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

/// Slice mx.t0b (decision D91): reflection schema 3, the textures, samplers, stage and fragment targets of the native
/// helper, through `cross_check_kernel` and `verify_artifact`. Two kinds of evidence, kept apart:
///
/// - the real helper (`compile_reflected`, both targets) on hand-written Slang, with a hand-written descriptor, and
///   the descriptor tampered one field at a time (the Reflect tier needs the helper);
/// - the committed documents under `tests/fixtures/schema3/`, which are the helper's own output for the probe inputs
///   of `crates/cargo-gust/src/testdata/schema3`, read through `verify_artifact` with the two identity hashes
///   recomputed for the stand-in source and artifact bytes (no Slang needed).
mod schema_three {
    use gust::{
        Access, Artifact, Format as ArtifactFormat, GpuPod, KernelDescriptor, ParameterDescriptor,
        ParameterKind, ReflectedTypeKind, ResourceBinding, ResourceLayout, ScalarKind, Shape,
        SourceOrigin, TextureAccess, TextureDimension, TextureLayout, TypeLayout,
        artifact::{Verified, verify_artifact},
        math::{UVec2, UVec4, Vec4},
        reflect::{Error, Stage, compile_reflected},
        slang::Target,
        texture::{Format, SampleClass},
    };

    const F32: TypeLayout = <f32 as GpuPod>::LAYOUT;
    const U32: TypeLayout = <u32 as GpuPod>::LAYOUT;
    const I32: TypeLayout = <i32 as GpuPod>::LAYOUT;
    const VEC4: TypeLayout = <Vec4 as GpuPod>::LAYOUT;
    const UVEC2: TypeLayout = <UVec2 as GpuPod>::LAYOUT;
    const UVEC4: TypeLayout = <UVec4 as GpuPod>::LAYOUT;

    fn bound(index: u32) -> Option<ResourceBinding> {
        Some(ResourceBinding {
            group: 0,
            binding: index,
        })
    }

    fn sampled(class: SampleClass) -> TextureLayout {
        TextureLayout {
            format: None,
            dimension: TextureDimension::D2,
            access: TextureAccess::Sampled,
            class,
        }
    }

    fn storage(format: Format, access: TextureAccess) -> TextureLayout {
        TextureLayout {
            format: Some(format),
            dimension: TextureDimension::D2,
            access,
            class: format.sample_class(),
        }
    }

    fn texture(
        name: &'static str,
        access: Access,
        element: TypeLayout,
        layout: TextureLayout,
        index: u32,
    ) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "Texture",
            kind: ParameterKind::Texture,
            access,
            resource_layout: Some(ResourceLayout::texture_v1(element, layout)),
            binding: bound(index),
        }
    }

    fn sampler(name: &'static str, index: u32) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "Sampler",
            kind: ParameterKind::Sampler,
            access: Access::NotApplicable,
            resource_layout: None,
            binding: bound(index),
        }
    }

    fn buffer(name: &'static str, element: TypeLayout, index: u32) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "RWStructuredBuffer<T>",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(element)),
            binding: bound(index),
        }
    }

    /// An output of a pixel kernel: a storage parameter without a binding, a colour target by its position.
    fn output(name: &'static str, element: TypeLayout) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "RWStructuredBuffer<T>",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(element)),
            binding: None,
        }
    }

    fn kernel(
        shape: Shape,
        entry_point: &'static str,
        slang_source: &'static str,
        parameters: Vec<ParameterDescriptor>,
    ) -> KernelDescriptor {
        KernelDescriptor {
            module: "m",
            name: "k",
            entry_point,
            shape,
            workgroup_size: [8, 8, 1],
            parameters: Box::leak(parameters.into_boxed_slice()),
            slang_source,
            source: SourceOrigin::UNKNOWN,
            artifacts: &[],
        }
    }

    /// `kernel` with the parameter at `index` changed by `edit`.
    fn with(
        kernel: &KernelDescriptor,
        index: usize,
        edit: impl FnOnce(&mut ParameterDescriptor),
    ) -> KernelDescriptor {
        let mut parameters = kernel.parameters.to_vec();
        edit(&mut parameters[index]);
        KernelDescriptor {
            parameters: Box::leak(parameters.into_boxed_slice()),
            ..*kernel
        }
    }

    /// `kernel` with the parameter at `index` given the texture layout `layout` over `element`.
    fn retextured(
        kernel: &KernelDescriptor,
        index: usize,
        element: TypeLayout,
        layout: TextureLayout,
    ) -> KernelDescriptor {
        with(kernel, index, |p| {
            p.resource_layout = Some(ResourceLayout::texture_v1(element, layout));
        })
    }

    fn message(error: Error) -> String {
        match error {
            Error::Mismatch(message) => message,
            other => panic!("expected a mismatch, got {other:?}"),
        }
    }

    // ---- the real helper ----

    const COMPUTE: &str = "[[vk::binding(0, 0)]] Texture2D<float4> tex;\n[[vk::binding(1, 0)]] [format(\"r32f\")] RWTexture2D<float> rw;\n[[vk::binding(2, 0)]] [format(\"rg32ui\")] WTexture2D<uint2> wo;\n[[vk::binding(3, 0)]] SamplerState samp;\n[[vk::binding(4, 0)]] RWStructuredBuffer<float> out;\n[shader(\"compute\")]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID)\n{\n    float4 t = tex.SampleLevel(samp, float2(0.5, 0.5), 0.0);\n    uint2 p = uint2(id.x, id.y);\n    rw[p] = rw[p] + t.x;\n    wo.Store(p, p);\n    out[id.x] = t.y;\n}\n";

    fn compute_kernel() -> KernelDescriptor {
        kernel(
            Shape::Grid,
            "main",
            COMPUTE,
            vec![
                texture(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    sampled(SampleClass::Float),
                    0,
                ),
                texture(
                    "rw",
                    Access::ReadWrite,
                    F32,
                    storage(Format::R32Float, TextureAccess::ReadWrite),
                    1,
                ),
                texture(
                    "wo",
                    Access::ReadWrite,
                    UVEC2,
                    storage(Format::Rg32Uint, TextureAccess::WriteOnly),
                    2,
                ),
                sampler("samp", 3),
                buffer("out", F32, 4),
            ],
        )
    }

    const FRAGMENT: &str = "[[vk::binding(0, 0)]] Texture2D<float4> tex;\n[[vk::binding(1, 0)]] SamplerState samp;\nstruct Out\n{\n    float4 color : SV_Target0;\n    uint id : SV_Target1;\n};\n[shader(\"fragment\")]\nOut main(float4 pos : SV_Position)\n{\n    Out o;\n    o.color = tex.SampleLevel(samp, float2(pos.x / 64.0, pos.y / 64.0), 0.0);\n    o.id = uint(pos.x);\n    return o;\n}\n";

    fn fragment_kernel() -> KernelDescriptor {
        kernel(
            Shape::Pixel,
            "main",
            FRAGMENT,
            vec![
                texture(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    sampled(SampleClass::Float),
                    0,
                ),
                sampler("samp", 1),
                output("color", VEC4),
                output("id", U32),
            ],
        )
    }

    #[test]
    fn native_reflects_textures_samplers_and_the_compute_stage() {
        let descriptor = compute_kernel();
        for target in [Target::Wgsl, Target::Spirv] {
            let compiled = compile_reflected(&descriptor, target)
                .unwrap_or_else(|error| panic!("{target:?}: {error}"));
            let reflection = &compiled.reflection;
            assert_eq!(reflection.schema_version, 3, "{target:?}");
            assert_eq!(reflection.stage, Stage::Compute, "{target:?}");
            assert_eq!(compiled.workgroup_size, [8, 8, 1], "{target:?}");
            let textures: Vec<_> = reflection
                .textures
                .iter()
                .map(|t| (t.name.as_str(), t.binding, t.access, t.format))
                .collect();
            assert_eq!(
                textures,
                [
                    ("tex", 0, TextureAccess::Sampled, None),
                    ("rw", 1, TextureAccess::ReadWrite, Some(Format::R32Float)),
                    ("wo", 2, TextureAccess::WriteOnly, Some(Format::Rg32Uint)),
                ],
                "{target:?}"
            );
            assert_eq!(
                reflection.textures[2].result.kind,
                ReflectedTypeKind::Vector {
                    scalar: ScalarKind::U32,
                    count: 2
                },
                "{target:?}"
            );
            assert_eq!(reflection.samplers.len(), 1, "{target:?}");
            assert_eq!(
                (
                    reflection.samplers[0].name.as_str(),
                    reflection.samplers[0].binding
                ),
                ("samp", 3),
                "{target:?}"
            );
            // Only the buffer has a layout report.
            let reports = reflection
                .cross_check_kernel(&descriptor)
                .unwrap_or_else(|error| panic!("{target:?}: {error}"));
            assert_eq!(reports.len(), 1, "{target:?}");
            assert!(reports[0].is_full_abi_match(), "{target:?}");
        }
    }

    #[test]
    fn native_reflects_a_fragment_stage_with_its_targets_and_no_workgroup() {
        let descriptor = fragment_kernel();
        for target in [Target::Wgsl, Target::Spirv] {
            let compiled = compile_reflected(&descriptor, target)
                .unwrap_or_else(|error| panic!("{target:?}: {error}"));
            let reflection = &compiled.reflection;
            assert_eq!(reflection.schema_version, 3, "{target:?}");
            assert_eq!(reflection.stage, Stage::Fragment, "{target:?}");
            assert_eq!(compiled.workgroup_size, [0, 0, 0], "{target:?}");
            let targets: Vec<_> = reflection
                .targets
                .iter()
                .map(|t| (t.index, t.ty.kind.clone()))
                .collect();
            assert_eq!(
                targets,
                [
                    (
                        0,
                        ReflectedTypeKind::Vector {
                            scalar: ScalarKind::F32,
                            count: 4
                        }
                    ),
                    (1, ReflectedTypeKind::Scalar(ScalarKind::U32)),
                ],
                "{target:?}"
            );
            assert!(
                reflection
                    .cross_check_kernel(&descriptor)
                    .unwrap()
                    .is_empty(),
                "{target:?}"
            );
        }
    }

    /// The tamper cases of plan section 3.3, on the real helper's output: the program is untouched and the descriptor
    /// is changed one field at a time, so every failure below is the one edit. Each is `Mismatch` and names the field.
    #[test]
    fn native_output_against_a_tampered_descriptor_is_a_mismatch_naming_the_field() {
        let compute = compute_kernel();
        let fragment = fragment_kernel();
        let swapped_outputs = KernelDescriptor {
            parameters: Box::leak(
                vec![
                    fragment.parameters[0],
                    fragment.parameters[1],
                    fragment.parameters[3],
                    fragment.parameters[2],
                ]
                .into_boxed_slice(),
            ),
            ..fragment
        };
        let cases: Vec<(&str, KernelDescriptor, KernelDescriptor, &str)> = vec![
            (
                "a compute reflection against a pixel descriptor",
                compute,
                KernelDescriptor {
                    shape: Shape::Pixel,
                    ..compute
                },
                "stage: host fragment, compiler compute",
            ),
            (
                "a fragment reflection against a grid descriptor",
                fragment,
                KernelDescriptor {
                    shape: Shape::Grid,
                    ..fragment
                },
                "stage: host compute, compiler fragment",
            ),
            (
                "a target type swapped",
                fragment,
                with(&fragment, 3, |p| {
                    p.resource_layout = Some(ResourceLayout::storage_v1(I32));
                }),
                "targets[1].type: host i32, compiler u32",
            ),
            (
                "the targets in the other order",
                fragment,
                swapped_outputs,
                "targets[0].type: host u32, compiler f32x4",
            ),
            (
                "a target dropped",
                fragment,
                KernelDescriptor {
                    parameters: &fragment.parameters[..3],
                    ..fragment
                },
                "targets: host 1 outputs, compiler 2",
            ),
            (
                "a texture access swapped",
                compute,
                retextured(
                    &compute,
                    1,
                    F32,
                    storage(Format::R32Float, TextureAccess::WriteOnly),
                ),
                "rw.access: host write, compiler readWrite",
            ),
            (
                "a texture format with the wrong channel count",
                compute,
                retextured(
                    &compute,
                    2,
                    UVEC2,
                    storage(Format::Rgba32Uint, TextureAccess::WriteOnly),
                ),
                "wo.format: rgba32uint has channel count 4, the result type has width 2",
            ),
            (
                "a storage texture's result type swapped",
                compute,
                retextured(
                    &compute,
                    1,
                    I32,
                    storage(Format::R32Sint, TextureAccess::ReadWrite),
                ),
                "rw.result: host i32, compiler f32",
            ),
            (
                "a sampled texture's result width swapped",
                compute,
                retextured(&compute, 0, F32, sampled(SampleClass::Float)),
                "tex.result: host f32, compiler f32x4",
            ),
            (
                "a sampled texture's class swapped",
                compute,
                retextured(&compute, 0, UVEC4, sampled(SampleClass::Uint)),
                "tex.class: host u32, compiler f32",
            ),
            (
                "a texture slot moved",
                compute,
                with(&compute, 0, |p| p.binding = bound(6)),
                "tex.binding: host 6, compiler 0",
            ),
            (
                "a sampler slot moved",
                compute,
                with(&compute, 3, |p| p.binding = bound(5)),
                "samp.binding: host 5, compiler 3",
            ),
        ];
        for target in [Target::Wgsl, Target::Spirv] {
            let compute_reflection = compile_reflected(&compute, target).unwrap().reflection;
            let fragment_reflection = compile_reflected(&fragment, target).unwrap().reflection;
            for (what, program, tampered, expected) in &cases {
                let reflection = if program.shape == Shape::Grid {
                    &compute_reflection
                } else {
                    &fragment_reflection
                };
                // The control: the same reflection passes against the program's own descriptor.
                reflection
                    .cross_check_kernel(program)
                    .unwrap_or_else(|error| panic!("{what}, {target:?}: control: {error}"));
                let error = reflection.cross_check_kernel(tampered).unwrap_err();
                assert_eq!(&message(error), expected, "{what}, {target:?}");
            }
        }
    }

    /// A target index that is not the output's position: the helper reports `SV_Target0` and `SV_Target2`.
    #[test]
    fn native_target_indices_with_a_gap_are_not_outputs_in_parameter_order() {
        const GAP: &str = "struct Out\n{\n    uint u : SV_Target0;\n    float4 v : SV_Target2;\n};\n[shader(\"fragment\")]\nOut main(float4 pos : SV_Position)\n{\n    Out o;\n    o.u = uint(pos.x);\n    o.v = pos;\n    return o;\n}\n";
        let descriptor = kernel(
            Shape::Pixel,
            "main",
            GAP,
            vec![output("u", U32), output("v", VEC4)],
        );
        for target in [Target::Wgsl, Target::Spirv] {
            let compiled = compile_reflected(&descriptor, target).unwrap();
            let indices: Vec<_> = compiled
                .reflection
                .targets
                .iter()
                .map(|t| t.index)
                .collect();
            assert_eq!(indices, [0, 2], "{target:?}");
            let error = compiled
                .reflection
                .cross_check_kernel(&descriptor)
                .unwrap_err();
            assert_eq!(
                message(error),
                "targets[1].index: host 1, compiler 2",
                "{target:?}"
            );
        }
    }

    // ---- the committed documents, through the precompiled path ----

    const SOURCE: &str = "// schema 3 fixture source\n";
    const BYTES: &[u8] = b"fn main() {}\n";

    fn fnv1a64(bytes: &[u8]) -> String {
        let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        });
        format!("{hash:016x}")
    }

    /// `document` with the value of the string member `key` replaced by `value`.
    fn set_member(document: &str, key: &str, value: &str) -> String {
        let marker = format!("\"{key}\":\"");
        let start = document.find(&marker).expect("the member is there") + marker.len();
        let end = start + document[start..].find('"').expect("a closing quote");
        format!("{}{value}{}", &document[..start], &document[end..])
    }

    /// The committed document `name`, its two identity hashes recomputed for `SOURCE` and `BYTES`: the helper hashed
    /// the probe's own text and the artifact it wrote, neither of which is committed.
    fn fixture(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/schema3")
            .join(name);
        let document = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let document = set_member(&document, "sourceHash", &fnv1a64(SOURCE.as_bytes()));
        set_member(&document, "artifactHash", &fnv1a64(BYTES))
    }

    fn verify_with(
        name: &str,
        descriptor: &KernelDescriptor,
        bytes: &'static [u8],
    ) -> Result<Verified, Error> {
        let reflection: &'static str = Box::leak(fixture(name).into_boxed_str());
        let descriptor = KernelDescriptor {
            slang_source: SOURCE,
            ..*descriptor
        };
        verify_artifact(
            &descriptor,
            &Artifact {
                format: ArtifactFormat::Wgsl,
                key: "0000000000000000",
                bytes,
                reflection,
            },
        )
    }

    fn verify(name: &str, descriptor: &KernelDescriptor) -> Result<Verified, Error> {
        verify_with(name, descriptor, BYTES)
    }

    #[test]
    fn a_committed_sampled_texture_document_verifies_and_is_held_to_its_class() {
        let float = kernel(
            Shape::Grid,
            "gpu_sampled_float4",
            "",
            vec![
                texture(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    sampled(SampleClass::Float),
                    0,
                ),
                buffer("out", F32, 1),
            ],
        );
        let verified = verify("sampled_float4.wgsl.json", &float).unwrap();
        assert_eq!(verified.compiled.reflection.schema_version, 3);
        assert_eq!(verified.compiled.workgroup_size, [8, 8, 1]);
        assert_eq!(verified.cross_checks.len(), 1);
        // The same document is not a `uint` texture, and not a pixel kernel's.
        let uint = retextured(&float, 0, UVEC4, sampled(SampleClass::Uint));
        assert_eq!(
            message(verify("sampled_float4.wgsl.json", &uint).unwrap_err()),
            "tex.class: host u32, compiler f32"
        );
        let pixel = KernelDescriptor {
            shape: Shape::Pixel,
            ..float
        };
        assert_eq!(
            message(verify("sampled_float4.wgsl.json", &pixel).unwrap_err()),
            "stage: host fragment, compiler compute"
        );
        // A scalar `uint` texture.
        let scalar = kernel(
            Shape::Grid,
            "gpu_sampled_uint",
            "",
            vec![
                texture("tex", Access::ReadOnly, U32, sampled(SampleClass::Uint), 0),
                buffer("out", U32, 1),
            ],
        );
        verify("sampled_uint.wgsl.json", &scalar).unwrap();
        let as_float = retextured(&scalar, 0, F32, sampled(SampleClass::Float));
        assert_eq!(
            message(verify("sampled_uint.wgsl.json", &as_float).unwrap_err()),
            "tex.class: host f32, compiler u32"
        );
        // A `float` texture is not the document's `float4` one (D91: the result type for every texture).
        let narrow = retextured(&float, 0, F32, sampled(SampleClass::Float));
        assert_eq!(
            message(verify("sampled_float4.wgsl.json", &narrow).unwrap_err()),
            "tex.result: host f32, compiler f32x4"
        );
    }

    #[test]
    fn a_committed_storage_texture_document_is_held_to_access_result_and_format() {
        let inferred = kernel(
            Shape::Grid,
            "gpu_storage_inferred",
            "",
            vec![texture(
                "rw",
                Access::ReadWrite,
                U32,
                storage(Format::R32Uint, TextureAccess::ReadWrite),
                0,
            )],
        );
        verify("storage_inferred.wgsl.json", &inferred).unwrap();
        for (what, tampered, expected) in [
            (
                "access",
                retextured(
                    &inferred,
                    0,
                    U32,
                    storage(Format::R32Uint, TextureAccess::WriteOnly),
                ),
                "rw.access: host write, compiler readWrite",
            ),
            (
                "result",
                retextured(
                    &inferred,
                    0,
                    F32,
                    storage(Format::R32Float, TextureAccess::ReadWrite),
                ),
                "rw.result: host f32, compiler u32",
            ),
            (
                "result and format",
                retextured(
                    &inferred,
                    0,
                    I32,
                    storage(Format::R32Sint, TextureAccess::ReadWrite),
                ),
                "rw.result: host i32, compiler u32",
            ),
        ] {
            assert_eq!(
                message(verify("storage_inferred.wgsl.json", &tampered).unwrap_err()),
                expected,
                "{what}"
            );
        }
    }

    #[test]
    fn a_committed_document_with_a_wide_read_write_texture_cannot_be_matched_by_a_descriptor() {
        // The helper reports a read-write `rgba32f` texture (Slang allows it); the dialect does not (D120), so no
        // descriptor of the dialect matches it: the read-write host layout is refused as a malformed descriptor, and
        // a write-only one differs in access.
        let declared = kernel(
            Shape::Grid,
            "gpu_storage_declared",
            "",
            vec![
                texture(
                    "rw",
                    Access::ReadWrite,
                    VEC4,
                    storage(Format::Rgba32Float, TextureAccess::ReadWrite),
                    0,
                ),
                texture(
                    "wo",
                    Access::ReadWrite,
                    UVEC2,
                    storage(Format::Rg32Uint, TextureAccess::WriteOnly),
                    1,
                ),
            ],
        );
        assert_eq!(
            message(verify("storage_declared.wgsl.json", &declared).unwrap_err()),
            "rw.format: rgba32float does not fit class f32 and access readWrite"
        );
        let write_only = retextured(
            &declared,
            0,
            VEC4,
            storage(Format::Rgba32Float, TextureAccess::WriteOnly),
        );
        assert_eq!(
            message(verify("storage_declared.wgsl.json", &write_only).unwrap_err()),
            "rw.access: host write, compiler readWrite"
        );
    }

    #[test]
    fn a_committed_fragment_document_verifies_and_its_targets_are_compared() {
        let fragment = kernel(
            Shape::Pixel,
            "gpu_sampler_fragment",
            "",
            vec![
                texture(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    sampled(SampleClass::Float),
                    0,
                ),
                sampler("samp", 1),
                output("color", VEC4),
            ],
        );
        let verified = verify("sampler_fragment.wgsl.json", &fragment).unwrap();
        assert_eq!(verified.compiled.reflection.stage, Stage::Fragment);
        assert_eq!(verified.compiled.workgroup_size, [0, 0, 0]);
        assert!(verified.cross_checks.is_empty());
        for (what, tampered, expected) in [
            (
                "the stage",
                KernelDescriptor {
                    shape: Shape::Grid,
                    ..fragment
                },
                "stage: host compute, compiler fragment",
            ),
            (
                "a target type",
                with(&fragment, 2, |p| {
                    p.resource_layout = Some(ResourceLayout::storage_v1(UVEC4));
                }),
                "targets[0].type: host u32x4, compiler f32x4",
            ),
            (
                "a sampler slot",
                with(&fragment, 1, |p| p.binding = bound(4)),
                "samp.binding: host 4, compiler 1",
            ),
            (
                "a target dropped",
                KernelDescriptor {
                    parameters: &fragment.parameters[..2],
                    ..fragment
                },
                "targets: host 0 outputs, compiler 1",
            ),
        ] {
            assert_eq!(
                message(verify("sampler_fragment.wgsl.json", &tampered).unwrap_err()),
                expected,
                "{what}"
            );
        }
        // The gap in the indices: `SV_Target0` and `SV_Target2`.
        let gap = kernel(
            Shape::Pixel,
            "gpu_fragment_gap",
            "",
            vec![output("u", U32), output("v", VEC4)],
        );
        assert_eq!(
            message(verify("fragment_gap.wgsl.json", &gap).unwrap_err()),
            "targets[1].index: host 1, compiler 2"
        );
    }

    #[test]
    fn a_committed_schema_three_document_is_refused_for_other_artifact_bytes() {
        // The identity check is as strict for schema 3 as for 1 and 2: the document is the one of these bytes.
        let descriptor = kernel(
            Shape::Pixel,
            "gpu_fragment_gap",
            "",
            vec![output("u", U32), output("v", VEC4)],
        );
        match verify_with("fragment_gap.wgsl.json", &descriptor, b"fn other() {}\n") {
            Err(Error::Invalid(message)) => {
                assert!(message.contains("artifact hash"), "{message}");
            }
            other => panic!("expected an identity failure, got {other:?}"),
        }
    }

    /// The committed documents are byte copies of the helper's own output in the cargo-gust probe expectations
    /// (`== <name> wgsl <stage>` heading and one document line). In a checkout the two are compared, so a helper
    /// update that changes one cannot leave the other behind; a packaged crate has no such file and the test ends.
    #[test]
    fn the_committed_documents_are_the_cargo_gust_probe_expectations() {
        let expected = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../cargo-gust/src/testdata/schema3/expected.txt");
        let Ok(expected) = std::fs::read_to_string(&expected) else {
            return;
        };
        for (name, file) in [
            ("sampled_float4", "sampled_float4.wgsl.json"),
            ("sampled_uint", "sampled_uint.wgsl.json"),
            ("storage_declared", "storage_declared.wgsl.json"),
            ("storage_inferred", "storage_inferred.wgsl.json"),
            ("sampler_fragment", "sampler_fragment.wgsl.json"),
            ("fragment_gap", "fragment_gap.wgsl.json"),
        ] {
            let heading = format!("== {name} wgsl ");
            let mut lines = expected.lines();
            let probe = loop {
                match lines.next() {
                    Some(line) if line.starts_with(&heading) => break lines.next().unwrap(),
                    Some(_) => {}
                    None => panic!("no `{heading}` section"),
                }
            };
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/schema3")
                .join(file);
            let committed = std::fs::read_to_string(&path).unwrap();
            assert_eq!(committed.trim_end(), probe.trim_end(), "{file}");
        }
    }
}
