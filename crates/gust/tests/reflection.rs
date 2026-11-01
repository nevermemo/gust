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

#[test]
fn native_rejects_padded_vector_layout_without_widening_storage_v1() {
    let mut kernel = sample::run::DESCRIPTOR;
    kernel.slang_source = include_str!("../../../scripts/probes/layout.slang");
    kernel.entry_point = "main";
    for target in [Target::Wgsl, Target::Spirv] {
        let error = compile_reflected(&kernel, target).unwrap_err();
        // The helper itself must refuse, not the Rust JSON reader after the fact.
        assert!(
            matches!(
                &error,
                gust::reflect::Error::Compiler(
                    gust::slang::Error::CompilationFailed { target: failed, .. }
                ) if *failed == target
            ),
            "{error:?}"
        );
        assert!(
            error.to_string().contains("unsupported element type"),
            "{error}"
        );
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
