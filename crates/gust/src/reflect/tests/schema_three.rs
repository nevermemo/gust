use super::{fragment_support::*, *};

#[test]
fn schema_three_reads_textures_samplers_and_the_stage() {
    let kernel = texture_kernel();
    let compiled = read(&texture_document(), &kernel).unwrap();
    let reflection = &compiled.reflection;
    assert_eq!(reflection.schema_version, 3);
    assert_eq!(reflection.stage, Stage::Compute);
    assert_eq!(compiled.workgroup_size, [1, 1, 1]);
    assert!(reflection.targets.is_empty());
    let textures: Vec<_> = reflection
        .textures
        .iter()
        .map(|t| (t.name.as_str(), t.binding, t.space, t.access, t.format))
        .collect();
    assert_eq!(
        textures,
        [
            ("tex", 0, 0, TextureAccess::Sampled, None),
            (
                "rw",
                1,
                0,
                TextureAccess::ReadWrite,
                Some(TexelFormat::R32Float)
            ),
            (
                "wo",
                2,
                0,
                TextureAccess::WriteOnly,
                Some(TexelFormat::Rg32Uint)
            ),
        ]
    );
    assert!(
        reflection
            .textures
            .iter()
            .all(|t| t.dimension == TextureDimension::D2)
    );
    assert_eq!(
        reflection.textures[2].result.kind,
        ReflectedTypeKind::Vector {
            scalar: ScalarKind::U32,
            count: 2
        }
    );
    // The helper reports no byte layout for a texture result.
    assert_eq!(reflection.textures[2].result.size, None);
    let samplers: Vec<_> = reflection
        .samplers
        .iter()
        .map(|s| (s.name.as_str(), s.binding, s.space))
        .collect();
    assert_eq!(samplers, [("samp", 3, 0)]);
    // The buffer is in the buffer list alone, and only buffers have a layout report.
    assert_eq!(reflection.resources.len(), 1);
    let reports = reflection.cross_check_kernel(&kernel).unwrap();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].is_full_abi_match());
}

#[test]
fn a_fragment_document_has_targets_in_order_and_no_workgroup() {
    let kernel = fragment_kernel();
    let compiled = read(&fragment_document(), &kernel).unwrap();
    let reflection = &compiled.reflection;
    assert_eq!(reflection.stage, Stage::Fragment);
    assert_eq!(compiled.workgroup_size, [0, 0, 0]);
    assert_eq!(reflection.targets.len(), 1);
    assert_eq!(reflection.targets[0].index, 0);
    assert_eq!(
        reflection.targets[0].ty.kind,
        ReflectedTypeKind::Vector {
            scalar: ScalarKind::F32,
            count: 4
        }
    );
    assert!(reflection.cross_check_kernel(&kernel).unwrap().is_empty());
    // Two outputs, in parameter order, at indices 0 and 1.
    let two = pixel_kernel(vec![
        output_parameter("a", U32_LAYOUT),
        output_parameter("b", VEC2),
    ]);
    let document = document3(
        "fragment",
        &[],
        &[
            target_json(0, ("uint32", 1)),
            target_json(1, ("float32", 2)),
        ],
    );
    let compiled = read(&document, &two).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&two).is_ok());
}

#[test]
fn schemas_one_and_two_are_compute_documents_without_textures() {
    let kernel = full_kernel();
    for (schema, matrix) in [(1_u32, None), (2, Some("column"))] {
        let parameters = if schema == 1 {
            vec![storage_json("a", 0, F32_LAYOUT, "read")]
        } else {
            full_parameters()
        };
        let kernel = if schema == 1 {
            kernel_with(vec![parameter(
                "a",
                ParameterKind::Storage,
                Access::ReadOnly,
                ResourceLayout::storage_v1(F32_LAYOUT),
                0,
            )])
        } else {
            kernel
        };
        let compiled = read(&document(schema, matrix, &parameters), &kernel).unwrap();
        let reflection = &compiled.reflection;
        assert_eq!(reflection.stage, Stage::Compute);
        assert!(reflection.textures.is_empty() && reflection.samplers.is_empty());
        assert!(reflection.targets.is_empty());
    }
    // A texture in a document of schema 1 or 2 is refused with the message those schemas always gave.
    let texture = texture_json("tex", 0, "read", F4, "unknown");
    for (schema, matrix) in [(1_u32, None), (2, Some("column"))] {
        let json = document(schema, matrix, std::slice::from_ref(&texture));
        assert_eq!(
            invalid_of(&json, &kernel_with(Vec::new())),
            "only structured buffers are supported",
            "schema {schema}"
        );
    }
    // A schema-3 document with nothing new in it is still a schema-3 document.
    assert!(read(&document3("compute", &[], &[]), &kernel_with(Vec::new())).is_ok());
}

#[test]
fn a_schema_one_or_two_reflection_cannot_describe_a_texture_or_a_sampler() {
    let compiled = read(&document(2, Some("column"), &[]), &kernel_with(Vec::new())).unwrap();
    let tex = texture_parameter(
        "tex",
        Access::ReadOnly,
        VEC4,
        sampled(SampleClass::Float),
        0,
    );
    for (what, parameter) in [("texture", tex), ("sampler", sampler_parameter("samp", 1))] {
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&kernel_with(vec![parameter]))
                .unwrap_err()
                .to_string(),
            format!(
                "POD reflection mismatch: reflection schema 2 cannot describe {what}; re-export with cargo gust build"
            ),
            "{what}"
        );
    }
}

#[test]
fn the_schema_three_grammar_is_strict() {
    let kernel = texture_kernel();
    let fragment = fragment_kernel();
    let control = texture_parameters();
    let with = |index: usize, parameter: String| {
        let mut parameters = control.clone();
        parameters[index] = parameter;
        document3("compute", &parameters, &[])
    };
    let cases: Vec<(&str, String, &str)> = vec![
            (
                "an unknown stage",
                texture_document().replacen(r#""stage":"compute""#, r#""stage":"geometry""#, 1),
                "unknown stage",
            ),
            (
                "a stage that is not text",
                texture_document().replacen(r#""stage":"compute""#, r#""stage":1"#, 1),
                "expected string",
            ),
            (
                "a missing stage",
                texture_document().replacen(r#""stage":"compute","#, "", 1),
                "missing stage",
            ),
            (
                "a texture of another shape",
                with(0, texture_json("tex", 0, "read", F4, "unknown").replace("2d", "3d")),
                "unsupported texture shape (only 2d)",
            ),
            (
                "a texture access of another name",
                with(0, texture_json("tex", 0, "sample", F4, "unknown")),
                "unsupported texture access",
            ),
            (
                "a format outside the dialect",
                with(1, texture_json("rw", 1, "readWrite", F4, "rgba8")),
                "unsupported texture format `rgba8`",
            ),
            (
                "a matrix result",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown").replace(
                        r#""kind":"vector","scalarType":"float32","elementCount":4"#,
                        r#""kind":"matrix","scalarType":"float32","rows":2,"columns":2,"matrixLayout":"column""#,
                    ),
                ),
                "unsupported texture result type (a scalar or a vector)",
            ),
            (
                "a texture result of five components",
                with(0, texture_json("tex", 0, "read", ("float32", 5), "unknown")),
                "unsupported elementCount 5",
            ),
            (
                "a texture and a buffer in one slot",
                with(4, storage_json("out", 1, F32_LAYOUT, "readWrite")),
                "duplicate resource name or binding",
            ),
            (
                "a sampler named like a texture",
                with(3, sampler_json("tex", 3)),
                "duplicate resource name or binding",
            ),
            (
                "a sampler in a texture's slot",
                with(3, sampler_json("samp", 0)),
                "duplicate resource name or binding",
            ),
            (
                "a binding array of textures",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown")
                        .replacen(r#""count":1"#, r#""count":2"#, 1),
                ),
                "binding arrays are unsupported",
            ),
            (
                "targets in a compute document",
                texture_document().replacen(
                    r#""parameters""#,
                    r#""targets":[],"parameters""#,
                    1,
                ),
                "a compute reflection has no targets",
            ),
        ];
    for (what, json, expected) in cases {
        assert_eq!(invalid_of(&json, &kernel), expected, "{what}");
    }
    let fragment_cases: Vec<(&str, String, &str)> = vec![
            (
                "a fragment with a workgroup size",
                fragment_document().replacen(
                    r#""parameters""#,
                    r#""workgroupSize":[1,1,1],"parameters""#,
                    1,
                ),
                "a fragment reflection has no workgroupSize",
            ),
            (
                "a fragment without targets",
                fragment_document().replacen(r#","targets":[{"index":0,"type":{"kind":"vector","scalarType":"float32","elementCount":4}}]"#, "", 1),
                "a fragment reflection needs targets",
            ),
            (
                "targets out of order",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[target_json(1, F4), target_json(0, F4)],
                ),
                "targets must be in ascending SV_Target order without repeats",
            ),
            (
                "a repeated target index",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[target_json(0, F4), target_json(0, F4)],
                ),
                "targets must be in ascending SV_Target order without repeats",
            ),
            (
                "a target that is a struct",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[r#"{"index":0,"type":{"kind":"struct","fields":[{"name":"a","binding":{"kind":"uniform","offset":0,"size":4},"type":{"kind":"scalar","scalarType":"float32"}}]}}"#.to_owned()],
                ),
                "unsupported target type (a scalar or a vector)",
            ),
        ];
    for (what, json, expected) in fragment_cases {
        assert_eq!(invalid_of(&json, &fragment), expected, "{what}");
    }
}
