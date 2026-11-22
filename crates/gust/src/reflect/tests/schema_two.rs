use super::*;

#[test]
fn schema_two_reads_vectors_matrices_and_constant_buffers() {
    let kernel = full_kernel();
    let compiled = read(&document(2, Some("column"), &full_parameters()), &kernel).unwrap();
    let reflection = &compiled.reflection;
    assert_eq!(reflection.schema_version, 2);
    let kinds: Vec<_> = reflection.resources.iter().map(|r| r.kind).collect();
    assert_eq!(
        kinds,
        [
            ReflectedResourceKind::StructuredBuffer,
            ReflectedResourceKind::StructuredBuffer,
            ReflectedResourceKind::ConstantBuffer,
        ]
    );
    let ReflectedTypeKind::Struct(fields) = &reflection.resources[0].element.kind else {
        panic!("struct expected")
    };
    assert_eq!(
        fields[1].ty.kind,
        ReflectedTypeKind::Vector {
            scalar: ScalarKind::F32,
            count: 2
        }
    );
    assert_eq!(fields[1].ty.size, Some(8));
    let mats = &reflection.resources[1];
    assert_eq!(
        mats.element.kind,
        ReflectedTypeKind::Matrix {
            scalar: ScalarKind::F32,
            rows: 4,
            cols: 4,
            layout: MatrixLayout::Column
        }
    );
    // The matrix `stride` is the column stride; the buffer's element stride is the whole matrix.
    assert_eq!(mats.element.stride, Some(16));
    assert_eq!(mats.element_stride, Some(64));
    let cb = &reflection.resources[2];
    assert_eq!((cb.binding, cb.space, cb.access), (2, 0, Access::ReadOnly));
    assert_eq!(cb.element_stride, Some(128));
    assert_eq!(cb.element.size, Some(128));
    let reports = reflection.cross_check_kernel(&kernel).unwrap();
    assert_eq!(reports.len(), 3);
    assert!(reports.iter().all(PodCrossCheck::is_full_abi_match));
    assert_eq!(reports[0].checked_fields, 3);
    // P has six fields and the nested E has one.
    assert_eq!(reports[2].checked_fields, 7);
}

#[test]
fn schema_two_still_describes_scalar_descriptors() {
    let kernel = kernel_with(vec![parameter(
        "a",
        ParameterKind::Storage,
        Access::ReadOnly,
        ResourceLayout::storage_v1(F32_LAYOUT),
        0,
    )]);
    let json = document(
        2,
        Some("column"),
        &[storage_json("a", 0, F32_LAYOUT, "read")],
    );
    let compiled = read(&json, &kernel).unwrap();
    assert_eq!(
        compiled
            .reflection
            .cross_check_kernel(&kernel)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn matrix_layout_is_mandatory_in_schema_two_and_must_be_the_pin() {
    let kernel = full_kernel();
    let parameters = full_parameters();
    match read(&document(2, None, &parameters), &kernel) {
        Err(Error::Invalid(message)) => assert_eq!(message, "missing matrixLayout"),
        other => panic!("a schema-2 document without matrixLayout was accepted: {other:?}"),
    }
    match read(&document(2, Some("row"), &parameters), &kernel) {
        Err(Error::Mismatch(message)) => {
            assert_eq!(message, "matrixLayout: host column, compiler row")
        }
        other => panic!("a row-major document was accepted: {other:?}"),
    }
    for unknown in ["diagonal", "Column", ""] {
        match read(&document(2, Some(unknown), &parameters), &kernel) {
            Err(Error::Invalid(message)) => assert_eq!(message, "unknown matrixLayout"),
            other => panic!("{unknown:?} was accepted: {other:?}"),
        }
    }
    // Schema 1 has no such field and ignores one: it reads exactly what it read before.
    let scalar = kernel_with(Vec::new());
    assert!(read(&document(1, None, &[]), &scalar).is_ok());
    assert!(read(&document(1, Some("row"), &[]), &scalar).is_ok());
}

#[test]
fn a_matrix_type_in_the_wrong_layout_is_a_mismatch() {
    let json = document(2, Some("column"), &full_parameters());
    let column = r#""rows":4,"columns":4,"matrixLayout":"column""#;
    assert_eq!(json.matches(column).count(), 2, "the Mat4 buffer and P.m");
    let row = json.replacen(column, r#""rows":4,"columns":4,"matrixLayout":"row""#, 1);
    assert_eq!(
        mismatch_of(&row, &full_kernel()),
        "mats.matrixLayout: host column, compiler row"
    );
    let unknown = json.replacen(column, r#""rows":4,"columns":4,"matrixLayout":"zigzag""#, 1);
    assert!(matches!(
        read(&unknown, &full_kernel()),
        Err(Error::Invalid(message)) if message == "unknown matrixLayout"
    ));
}

#[test]
fn schema_one_documents_cannot_describe_new_kinds() {
    let scalar_only = document(1, None, &[storage_json("a", 0, F32_LAYOUT, "read")]);
    let scalar_kernel = kernel_with(vec![parameter(
        "a",
        ParameterKind::Storage,
        Access::ReadOnly,
        ResourceLayout::storage_v1(F32_LAYOUT),
        0,
    )]);
    // The control: a schema-1 document and a scalar descriptor.
    let compiled = read(&scalar_only, &scalar_kernel).unwrap();
    assert_eq!(compiled.reflection.schema_version, 1);
    assert_eq!(
        compiled
            .reflection
            .cross_check_kernel(&scalar_kernel)
            .unwrap()
            .len(),
        1
    );
    let storage_struct_with_matrix = make_struct(
        "WithMatrix",
        80,
        16,
        &[member("a", 0, F32_LAYOUT), member("m", 16, MAT4)],
    );
    let uniform_of_scalars = make_struct("U", 16, 16, &[member("a", 0, F32_LAYOUT)]);
    for (what, kind, layout) in [
        (
            "vector",
            ParameterKind::Storage,
            ResourceLayout::storage_v2(VEC4),
        ),
        (
            "matrix",
            ParameterKind::Storage,
            ResourceLayout::storage_v2(MAT2),
        ),
        (
            "vector",
            ParameterKind::Storage,
            ResourceLayout::storage_v2(STORAGE_STRUCT),
        ),
        (
            "matrix",
            ParameterKind::Storage,
            ResourceLayout::storage_v2(storage_struct_with_matrix),
        ),
        (
            "uniform",
            ParameterKind::Uniform,
            ResourceLayout::std140_v1(uniform_of_scalars),
        ),
        (
            "uniform",
            ParameterKind::Uniform,
            ResourceLayout::std140_v1(CB_STRUCT),
        ),
    ] {
        let kernel = kernel_with(vec![parameter("a", kind, Access::ReadOnly, layout, 0)]);
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&kernel)
                .unwrap_err()
                .to_string(),
            format!(
                "POD reflection mismatch: reflection schema 1 cannot describe {what}; re-export with cargo gust build"
            ),
            "{what}"
        );
    }
    // A schema-1 document that itself contains a new kind is refused by the reader, with the old messages.
    let with_vector = document(1, None, &[storage_json("pts", 0, STORAGE_STRUCT, "read")]);
    assert!(matches!(
        read(&with_vector, &full_kernel()),
        Err(Error::Invalid(message))
            if message == "unsupported element type (only 32-bit scalars and structs)"
    ));
    let with_matrix = document(1, None, &[storage_json("mats", 0, MAT4, "read")]);
    assert!(matches!(
        read(&with_matrix, &full_kernel()),
        Err(Error::Invalid(message))
            if message == "unsupported element type (only 32-bit scalars and structs)"
    ));
    let with_constant_buffer = document(1, None, &[uniform_json("cb", 0, CB_STRUCT)]);
    assert!(matches!(
        read(&with_constant_buffer, &full_kernel()),
        Err(Error::Invalid(message)) if message == "only structured buffers are supported"
    ));
}

#[test]
fn only_schema_versions_one_to_four_are_read() {
    let kernel = kernel_with(Vec::new());
    for version in ["0", "5", "4294967295"] {
        let json = document(1, None, &[]).replacen(
            r#""schemaVersion":1"#,
            &format!(r#""schemaVersion":{version}"#),
            1,
        );
        assert!(matches!(
            read(&json, &kernel),
            Err(Error::Invalid(message)) if message == "unsupported native reflection schema version"
        ));
    }
    let json = document(1, None, &[]).replacen(r#""schemaVersion":1"#, r#""schemaVersion":"2""#, 1);
    assert!(read(&json, &kernel).is_err());
    assert!(read(&document(1, None, &[]), &kernel).is_ok());
    assert!(read(&document(2, Some("column"), &[]), &kernel).is_ok());
}

#[test]
fn schema_two_fields_are_compared_exactly() {
    let json = document(2, Some("column"), &full_parameters());
    let kernel = full_kernel();
    // The control: the unmodified document passes, so each failure below is the one edit.
    let compiled = read(&json, &kernel).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&kernel).is_ok());
    let cases: &[(&str, &str, &str)] = &[
        (
            r#""name":"v","binding":{"kind":"uniform","offset":8,"size":8}"#,
            r#""name":"v","binding":{"kind":"uniform","offset":4,"size":8}"#,
            "pts.v.offset: host 8, compiler 4",
        ),
        (
            r#""elementCount":2"#,
            r#""elementCount":4"#,
            "pts.v: type or field count differs",
        ),
        (
            r#""kind":"vector","scalarType":"float32","elementCount":2"#,
            r#""kind":"vector","scalarType":"int32","elementCount":2"#,
            "pts.v: type or field count differs",
        ),
        (
            r#""size":8,"alignment":8,"stride":8,"name":"Vec2""#,
            r#""size":16,"alignment":8,"stride":8,"name":"Vec2""#,
            "pts.v.size: host 8, compiler 16",
        ),
        (
            r#""rows":4,"columns":4"#,
            r#""rows":4,"columns":3"#,
            "mats: type or field count differs",
        ),
        (
            r#""kind":"matrix","scalarType":"float32""#,
            r#""kind":"matrix","scalarType":"int32""#,
            "mats: type or field count differs",
        ),
        (
            r#""size":64,"alignment":16,"stride":16,"name":"Mat4""#,
            r#""size":64,"alignment":16,"stride":64,"name":"Mat4""#,
            "mats.stride: host 16, compiler 64",
        ),
        (
            r#""size":64,"alignment":16,"stride":16,"name":"Mat4""#,
            r#""size":32,"alignment":16,"stride":16,"name":"Mat4""#,
            "mats.size: host 64, compiler 32",
        ),
        (
            r#""type":{"kind":"constantBuffer","size":128"#,
            r#""type":{"kind":"constantBuffer","size":124"#,
            "cb.stride: host 128, compiler 124",
        ),
        (
            r#""size":128,"alignment":16,"stride":128,"name":"P""#,
            r#""size":128,"alignment":8,"stride":128,"name":"P""#,
            "cb.alignment: host 16, compiler 8",
        ),
        (
            r#""size":128,"alignment":16,"stride":128,"name":"P""#,
            r#""size":128,"alignment":16,"stride":112,"name":"P""#,
            "cb.stride: host 128, compiler 112",
        ),
        (
            r#""name":"m","binding":{"kind":"uniform","offset":32,"size":64}"#,
            r#""name":"m","binding":{"kind":"uniform","offset":16,"size":64}"#,
            "cb.m.offset: host 32, compiler 16",
        ),
        (
            r#""name":"e","binding":{"kind":"uniform","offset":96,"size":16}"#,
            r#""name":"e","binding":{"kind":"uniform","offset":80,"size":16}"#,
            "cb.e.offset: host 96, compiler 80",
        ),
        (
            r#""access":"readWrite""#,
            r#""access":"read""#,
            "mats.access differs",
        ),
        (
            r#""index":2"#,
            r#""index":7"#,
            "cb.binding: host 2, compiler 7",
        ),
    ];
    for (find, replace, expected) in cases {
        assert!(json.contains(find), "the fixture lost `{find}`");
        let edited = json.replacen(find, replace, 1);
        assert_eq!(&mismatch_of(&edited, &kernel), expected, "{find}");
    }
}

#[test]
fn rules_kinds_and_access_must_fit_the_parameter() {
    let json = document(2, Some("column"), &full_parameters());
    let storage_shaped = make_struct("E", 4, 4, &[member("a", 0, F32_LAYOUT)]);
    let cases: Vec<(usize, ParameterDescriptor, &str)> = vec![
        (
            2,
            parameter(
                "cb",
                ParameterKind::Uniform,
                Access::ReadOnly,
                ResourceLayout::storage_v2(CB_STRUCT),
                2,
            ),
            "cb.rules: StorageV2 does not fit this parameter kind",
        ),
        (
            0,
            parameter(
                "pts",
                ParameterKind::Storage,
                Access::ReadOnly,
                ResourceLayout::std140_v1(STORAGE_STRUCT),
                0,
            ),
            "pts.rules: Std140V1 does not fit this parameter kind",
        ),
        (
            2,
            parameter(
                "cb",
                ParameterKind::Uniform,
                Access::ReadWrite,
                ResourceLayout::std140_v1(CB_STRUCT),
                2,
            ),
            "cb.access differs",
        ),
        (
            2,
            parameter(
                "cb",
                ParameterKind::Value,
                Access::ReadOnly,
                ResourceLayout::std140_v1(CB_STRUCT),
                2,
            ),
            "only storage and uniform descriptor resources are supported",
        ),
        (
            2,
            parameter(
                "cb",
                ParameterKind::Uniform,
                Access::ReadOnly,
                ResourceLayout::std140_v1(storage_shaped),
                2,
            ),
            "host layout is not Std140V1",
        ),
        (
            2,
            parameter(
                "cb",
                ParameterKind::Uniform,
                Access::ReadOnly,
                ResourceLayout {
                    rules: ShaderLayoutRules::Std140V1,
                    element: CB_STRUCT,
                    element_stride: 64,
                },
                2,
            ),
            "cb.stride: inconsistent host resource layout",
        ),
        (
            0,
            parameter(
                "pts",
                ParameterKind::Storage,
                Access::ReadOnly,
                ResourceLayout::storage_v1(VEC4),
                0,
            ),
            "host layout is not StorageV1",
        ),
        (
            0,
            parameter(
                "pts",
                ParameterKind::Storage,
                Access::ReadOnly,
                ResourceLayout::storage_v2(CB_STRUCT),
                0,
            ),
            "host layout is not StorageV2",
        ),
    ];
    for (index, replacement, expected) in cases {
        let mut parameters = full_kernel().parameters.to_vec();
        parameters[index] = replacement;
        assert_eq!(
            mismatch_of(&json, &kernel_with(parameters)),
            expected,
            "{expected}"
        );
    }
    // A constant buffer in the document for a storage parameter, and the reverse.
    let kernel = full_kernel();
    let as_buffer = document(
        2,
        Some("column"),
        &[
            storage_json("pts", 0, STORAGE_STRUCT, "read"),
            storage_json("mats", 1, MAT4, "readWrite"),
            storage_json("cb", 2, CB_STRUCT, "read"),
        ],
    );
    assert_eq!(
        mismatch_of(&as_buffer, &kernel),
        "cb.kind: host ConstantBuffer, compiler StructuredBuffer"
    );
    let as_uniform = document(
        2,
        Some("column"),
        &[
            storage_json("pts", 0, STORAGE_STRUCT, "read"),
            uniform_json("mats", 1, MAT4),
            uniform_json("cb", 2, CB_STRUCT),
        ],
    );
    assert_eq!(
        mismatch_of(&as_uniform, &kernel),
        "mats.kind: host StructuredBuffer, compiler ConstantBuffer"
    );
}

#[test]
fn cross_check_pod_checks_the_buffer_math_types() {
    let parameters = [
        storage_json("v4", 0, VEC4, "read"),
        storage_json("v2", 1, VEC2, "read"),
        storage_json("m2", 2, MAT2, "read"),
        storage_json("m4", 3, MAT4, "read"),
        uniform_json("cb", 4, CB_STRUCT),
    ];
    let compiled = read(
        &document(2, Some("column"), &parameters),
        &kernel_with(Vec::new()),
    )
    .unwrap();
    let resources = &compiled.reflection.resources;
    assert!(
        cross_check_pod::<Vec4>(&resources[0])
            .unwrap()
            .is_full_abi_match()
    );
    assert!(
        cross_check_pod::<Vec2>(&resources[1])
            .unwrap()
            .is_full_abi_match()
    );
    assert!(
        cross_check_pod::<Mat2>(&resources[2])
            .unwrap()
            .is_full_abi_match()
    );
    assert!(
        cross_check_pod::<Mat4>(&resources[3])
            .unwrap()
            .is_full_abi_match()
    );
    // Width, scalar kind, matrix size and resource kind each fail.
    assert!(cross_check_pod::<Vec2>(&resources[0]).is_err());
    assert!(cross_check_pod::<crate::math::IVec4>(&resources[0]).is_err());
    assert!(cross_check_pod::<Mat4>(&resources[2]).is_err());
    assert!(cross_check_pod::<Mat2>(&resources[3]).is_err());
    match cross_check_pod::<Vec4>(&resources[4]) {
        Err(Error::Mismatch(message)) => {
            assert_eq!(
                message,
                "cb.kind: host StructuredBuffer, compiler ConstantBuffer"
            )
        }
        other => panic!("a constant buffer was read as storage: {other:?}"),
    }
}

#[test]
fn vector_widths_and_matrix_dimensions_are_two_to_four() {
    let kernel = kernel_with(Vec::new());
    let vector = document(2, Some("column"), &[storage_json("v", 0, VEC4, "read")]);
    for bad in ["0", "1", "5"] {
        let json = vector.replacen(
            r#""elementCount":4"#,
            &format!(r#""elementCount":{bad}"#),
            1,
        );
        assert!(matches!(
            read(&json, &kernel),
            Err(Error::Invalid(message)) if message == format!("unsupported elementCount {bad}")
        ));
    }
    let matrix = document(2, Some("column"), &[storage_json("m", 0, MAT4, "read")]);
    let rows = matrix.replacen(r#""rows":4"#, r#""rows":5"#, 1);
    assert!(matches!(
        read(&rows, &kernel),
        Err(Error::Invalid(message)) if message == "unsupported rows 5"
    ));
    let columns = matrix.replacen(r#""columns":4"#, r#""columns":1"#, 1);
    assert!(matches!(
        read(&columns, &kernel),
        Err(Error::Invalid(message)) if message == "unsupported columns 1"
    ));
    // The same text is a different, still valid, document with the width in range.
    assert!(read(&matrix, &kernel).is_ok());
}

#[gust::gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}

/// Every `*.reflection.json` (WGSL and SPIR-V sets alike) under the workspace, build output, VCS data and hidden
/// directories aside. It walks the file system, not the index, so untracked files are picked up too.
fn committed_reflection_files(directory: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
    for entry in entries {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        let file_type = entry.file_type().unwrap();
        if file_type.is_dir() {
            if name.starts_with('.')
                || matches!(name.as_str(), "target" | "node_modules" | "generated-wgpu")
            {
                continue;
            }
            committed_reflection_files(&entry.path(), found);
        } else if file_type.is_file() && name.ends_with(".reflection.json") {
            found.push(entry.path());
        }
    }
}

/// Schema 1 byte for byte: every committed artifact set that exists still verifies, through the same
/// `verify_artifact` that the run-time path uses, with the reader that now also knows schema 2. No re-export,
/// no re-blessing (plan R1-A.2).
#[test]
fn every_committed_schema_one_artifact_set_still_verifies() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut found = Vec::new();
    committed_reflection_files(&workspace, &mut found);
    found.sort();
    let relative: Vec<String> = found
        .iter()
        .map(|path| {
            path.strip_prefix(&workspace)
                .unwrap()
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/")
        })
        .collect();
    // The sets committed when this slice landed; a vanished one is a failure. Below, the vector-add WGSL sets
    // (the only kernel with a descriptor in this module) are verified and cross-checked through
    // `verify_artifact`; every other set, SPIR-V ones included, is only parsed by the reader, because there is no
    // descriptor to cross-check it against here. The walk is of the file system, so untracked files count too.
    for expected in [
        "crates/gust-wgpu/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json",
        "crates/gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json",
        "tests/ui/profile-native-artifact/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json",
        "tests/ui/profile-native-no-manifest/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json",
    ] {
        assert!(relative.iter().any(|path| path == expected), "{expected}");
    }
    for (path, label) in found.iter().zip(&relative) {
        let reflection = std::fs::read_to_string(path).unwrap();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let key = file.split('.').next().unwrap().to_owned();
        if key == "52e85c5f03f72272" && file.ends_with(".wgsl.reflection.json") {
            let bytes =
                std::fs::read(path.with_file_name(file.trim_end_matches(".reflection.json")))
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
            let artifact = Artifact {
                format: Format::Wgsl,
                key: "52e85c5f03f72272",
                bytes: Box::leak(bytes.into_boxed_slice()),
                reflection: Box::leak(reflection.clone().into_boxed_str()),
            };
            let verified = verify_artifact(&vector_add::add::DESCRIPTOR, &artifact)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(verified.compiled.reflection.schema_version, 1, "{label}");
            assert_eq!(verified.cross_checks.len(), 3, "{label}");
            assert!(
                verified
                    .cross_checks
                    .iter()
                    .all(PodCrossCheck::is_full_abi_match)
            );
            assert!(!reflection.contains("matrixLayout"), "{label}");
        } else {
            // Parsed only: no descriptor here to cross-check against.
            let target = if file.ends_with(".spv.reflection.json") {
                Target::Spirv
            } else {
                Target::Wgsl
            };
            let root = Parser::parse(&reflection).unwrap();
            let schema = read_schema_version(&root).unwrap();
            Reflection::read_json(target, &root, true, schema)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
        }
    }
}
