use super::*;
use crate::{
    store::tests::Scratch,
    testdata::{
        PAIRS_REFLECTION, SAMPLE_KEY, SAMPLE_RECORD_JSON, SAMPLE_SLANG, SAMPLE_SLANG_SHA256,
        SITE_LIB_JSON, SITE_LIB_NAME, SITE_OTHER_JSON, SITE_OTHER_NAME, site_record_json,
    },
};

fn sample() -> Record {
    Record::parse(SAMPLE_RECORD_JSON, SAMPLE_SLANG).unwrap()
}

#[test]
fn the_hand_written_golden_reads_field_by_field() {
    let record = sample();
    assert_eq!(record.key, SAMPLE_KEY);
    assert_eq!(key_of("gpu_demo_run", SAMPLE_SLANG), SAMPLE_KEY);
    assert_eq!(record.qualified_name(), "demo::run");
    assert_eq!(record.entry_point, "gpu_demo_run");
    assert_eq!(record.workgroup_size, [64, 1, 1]);
    assert_eq!(record.file, "src/lib.rs");
    assert_eq!(record.lines, [[3, 9, 12], [5, 5, 14]]);
    assert_eq!(record.slang, SAMPLE_SLANG);
    assert_eq!(record.clean_sha256(), SAMPLE_SLANG_SHA256);
    let [id, pairs, out] = &record.parameters[..] else {
        panic!("{:?}", record.parameters);
    };
    assert_eq!(
        (
            id.name.as_str(),
            id.kind,
            id.access,
            id.binding,
            id.layout.is_none()
        ),
        (
            "id",
            ParameterKind::Invocation,
            Access::NotApplicable,
            None,
            true
        )
    );
    assert_eq!(pairs.rust_type, "StructuredBuffer < Pair >");
    assert_eq!(
        (pairs.kind, pairs.access, pairs.binding),
        (ParameterKind::Storage, Access::ReadOnly, Some((0, 0)))
    );
    let layout = pairs.layout.as_ref().unwrap();
    assert_eq!(layout.element_stride, 8);
    assert_eq!(
        (layout.element.name.as_str(), layout.element.size),
        ("Pair", 8)
    );
    let LayoutKindRecord::Struct(fields) = &layout.element.kind else {
        panic!("{:?}", layout.element.kind);
    };
    let shape: Vec<(&str, u32, &LayoutKindRecord)> = fields
        .iter()
        .map(|field| (field.name.as_str(), field.offset, &field.ty.kind))
        .collect();
    assert_eq!(
        shape,
        [
            ("a", 0, &LayoutKindRecord::Scalar(ScalarKind::U32)),
            ("b", 4, &LayoutKindRecord::Scalar(ScalarKind::F32)),
        ]
    );
    assert_eq!((out.access, out.binding), (Access::ReadWrite, Some((0, 1))));
}

const PAIR_FIELDS: &[StructFieldLayout] = &[
    StructFieldLayout {
        name: "a",
        offset: 0,
        ty: TypeLayout::scalar("u32", ScalarKind::U32),
    },
    StructFieldLayout {
        name: "b",
        offset: 4,
        ty: TypeLayout::scalar("f32", ScalarKind::F32),
    },
];

const EXPECTED_PARAMETERS: &[ParameterDescriptor] = &[
    ParameterDescriptor {
        name: "id",
        rust_type: "SV_DispatchThreadID",
        kind: ParameterKind::Invocation,
        access: Access::NotApplicable,
        resource_layout: None,
        binding: None,
    },
    ParameterDescriptor {
        name: "pairs",
        rust_type: "StructuredBuffer < Pair >",
        kind: ParameterKind::Storage,
        access: Access::ReadOnly,
        resource_layout: Some(ResourceLayout::storage_v1(TypeLayout::structure(
            "Pair",
            8,
            4,
            PAIR_FIELDS,
        ))),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    },
    ParameterDescriptor {
        name: "out",
        rust_type: "RWStructuredBuffer < float >",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(TypeLayout::scalar(
            "f32",
            ScalarKind::F32,
        ))),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 1,
        }),
    },
];

#[test]
fn the_rebuilt_descriptor_equals_the_hand_built_one() {
    let expected = KernelDescriptor {
        module: "demo",
        name: "run",
        entry_point: "gpu_demo_run",
        workgroup_size: [64, 1, 1],
        parameters: EXPECTED_PARAMETERS,
        slang_source: SAMPLE_SLANG,
        source: SourceOrigin {
            file: "src/lib.rs",
            lines: &[
                LineSpan {
                    slang_start: 3,
                    slang_end: 9,
                    rust_line: 12,
                },
                LineSpan {
                    slang_start: 5,
                    slang_end: 5,
                    rust_line: 14,
                },
            ],
        },
        artifacts: &[],
    };
    assert_eq!(sample().descriptor(), expected);
}

#[test]
fn a_record_with_no_source_location_rebuilds_with_the_unknown_origin() {
    let mut record = sample();
    record.file.clear();
    record.lines.clear();
    assert_eq!(record.descriptor().source, SourceOrigin::UNKNOWN);
}

#[test]
fn the_empty_source_form_of_the_macros_demo_run_golden_reads_and_rebuilds_with_the_unknown_origin()
{
    // m4.c's `demo-run.kernel.json` differs from `sample.kernel.json` only in these two places: a kernel whose
    // source file is unknown has `"file": ""` and no lines, and its workgroup size is 32, 2, 1.
    let json = SAMPLE_RECORD_JSON
        .replace(
            "\"file\": \"src/lib.rs\",\n    \"lines\": [\n      [3, 9, 12],\n      [5, 5, 14]\n    ]",
            "\"file\": \"\",\n    \"lines\": []",
        )
        .replace("[64, 1, 1]", "[32, 2, 1]");
    assert_ne!(json, SAMPLE_RECORD_JSON);
    let record = Record::parse(&json, SAMPLE_SLANG).unwrap();
    assert_eq!(record.workgroup_size, [32, 2, 1]);
    assert!(record.file.is_empty() && record.lines.is_empty());
    let kernel = record.descriptor();
    assert_eq!(kernel.source, SourceOrigin::UNKNOWN);
    assert_eq!(kernel.workgroup_size, [32, 2, 1]);
}

#[test]
fn a_null_layout_leaves_the_resource_layout_empty_until_the_record_is_completed_from_the_reflection()
 {
    let mut record = sample();
    assert!(record.unchecked_parameters().is_empty());
    record.parameters[1].layout = None;
    assert_eq!(record.unchecked_parameters(), ["pairs"]);
    // The invocation parameter has no layout by design and is not listed; the storage parameter without one is.
    assert!(record.parameters[0].layout.is_none());
    assert_eq!(record.descriptor().parameters[1].resource_layout, None);
    let completed = record.completed(PAIRS_REFLECTION).unwrap();
    assert!(completed.unchecked_parameters().is_empty());
    // The reflection says what the macro would have said.
    assert_eq!(completed.parameters, sample().parameters);
    assert_eq!(completed.descriptor(), sample().descriptor());
    // The original is untouched, and the listing keeps coming from it.
    assert_eq!(record.unchecked_parameters(), ["pairs"]);
}

#[test]
fn completing_a_record_without_unchecked_parameters_does_not_read_the_reflection() {
    assert_eq!(sample().completed("not json").unwrap(), sample());
}

#[test]
fn a_reflection_that_lacks_the_unchecked_parameter_is_an_error_naming_the_kernel_and_the_parameter()
{
    let mut record = sample();
    record.parameters[2].layout = None;
    let error = record.completed(PAIRS_REFLECTION).unwrap_err();
    assert!(
        error.starts_with("demo::run: the reflection reports no resource named `out`"),
        "{error}"
    );
}

fn broken(from: &str, to: &str) -> String {
    assert!(SAMPLE_RECORD_JSON.contains(from), "{from}");
    SAMPLE_RECORD_JSON.replacen(from, to, 1)
}

fn error_of(json: &str) -> String {
    Record::parse(json, SAMPLE_SLANG).unwrap_err()
}

#[test]
fn malformed_records_are_refused_with_the_path_of_the_problem() {
    for (json, expected) in [
        (
            broken("\"schema\": 1", "\"schema\": 3"),
            "unsupported schema",
        ),
        (broken("  \"schema\": 1,\n", ""), "record: missing `schema`"),
        (
            broken("  \"name\": \"run\",\n", ""),
            "record: missing `name`",
        ),
        (
            broken(
                "\"workgroup_size\": [64, 1, 1]",
                "\"workgroup_size\": [64, 1]",
            ),
            "three numbers",
        ),
        (
            broken(
                "\"workgroup_size\": [64, 1, 1]",
                "\"workgroup_size\": [64, 1, -1]",
            ),
            "three numbers",
        ),
        (
            broken("\"access\": \"read_only\"", "\"access\": \"sometimes\""),
            "parameters[1]: unknown access `sometimes`",
        ),
        (
            broken("\"kind\": \"invocation\"", "\"kind\": \"texture\""),
            "parameters[0]: unknown parameter kind `texture`",
        ),
        (
            broken("\"rules\": \"storage_v1\"", "\"rules\": \"std140\""),
            "parameters[1].layout: unknown layout rules `std140`",
        ),
        (
            broken("\"scalar\": \"u32\"", "\"scalar\": \"u64\""),
            "parameters[1].layout.element.fields[0].type: unknown scalar `u64`",
        ),
        (
            broken("\"offset\": 4,", ""),
            "parameters[1].layout.element.fields[1]: missing `offset`",
        ),
        (
            broken("[3, 9, 12]", "[3, 9]"),
            "source.lines[0]: expected [slang_start, slang_end, rust_line]",
        ),
        (
            broken(
                "\"key\": \"022c52d675787198\",",
                "\"key\": \"022c52d675787198\",\n  \"key\": \"x\",",
            ),
            "record: `key` appears more than once",
        ),
        (
            broken("{\n  \"entry_point\"", "[\n  \"entry_point\""),
            "expected",
        ),
        (String::new(), "invalid JSON"),
    ] {
        let message = error_of(&json);
        assert!(
            message.contains(expected),
            "{expected:?} not in {message:?}"
        );
    }
}

#[test]
fn a_json_and_a_slang_text_that_do_not_belong_together_are_refused() {
    let message = Record::parse(SAMPLE_RECORD_JSON, "void other() {}\n").unwrap_err();
    assert!(
        message.starts_with(
            "record: key `022c52d675787198` is not the key of its entry point and Slang text (`"
        ),
        "{message}"
    );
    let renamed = broken(
        "\"entry_point\": \"gpu_demo_run\"",
        "\"entry_point\": \"gpu_demo_other\"",
    );
    assert!(error_of(&renamed).contains("is not the key"));
}

#[test]
fn unknown_keys_are_ignored_so_the_macro_may_add_fields_within_schema_one() {
    let extended = broken(
        "  \"schema\": 1,",
        "  \"schema\": 1,\n  \"future\": {\"x\": [1]},",
    );
    assert_eq!(Record::parse(&extended, SAMPLE_SLANG).unwrap(), sample());
}

// ---- schema 2: the record of a `par_for!` call and its sidecars (R1.1 M5) ----

/// A record directory holding the sample kernel as schema 2, with the given sidecars.
fn site_directory(scratch: &Scratch, sidecars: &[(&str, &str)]) -> std::path::PathBuf {
    let directory = scratch.0.join("records");
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join(format!("{SAMPLE_KEY}.kernel.json")),
        site_record_json(),
    )
    .unwrap();
    fs::write(directory.join(format!("{SAMPLE_KEY}.slang")), SAMPLE_SLANG).unwrap();
    for (name, text) in sidecars {
        fs::write(directory.join(name), text).unwrap();
    }
    directory
}

#[test]
fn a_schema_2_record_has_no_origin_until_its_sidecars_are_read() {
    let record = Record::parse(&site_record_json(), SAMPLE_SLANG).unwrap();
    assert_eq!(record.key, SAMPLE_KEY);
    assert_eq!((record.file.as_str(), record.lines.len()), ("", 0));
    assert_eq!(record.parameters, sample().parameters);
    assert_eq!(record.descriptor().source, SourceOrigin::UNKNOWN);
}

#[test]
fn two_sidecars_give_the_record_the_origin_of_the_first_by_file_and_line() {
    let scratch = Scratch::new("two sites");
    let directory = site_directory(
        &scratch,
        &[
            (SITE_LIB_NAME, SITE_LIB_JSON),
            (SITE_OTHER_NAME, SITE_OTHER_JSON),
        ],
    );
    let records = read_records(&directory).unwrap();
    let [record] = &records[..] else {
        panic!("{records:?}");
    };
    // The sidecar name is the hash of the call site, so its order is arbitrary: `440c...` (tests/other.rs) sorts
    // before `6b7d...` (src/lib.rs). The sites are sorted by file, not by name, and the origin is the first.
    assert!(SITE_OTHER_NAME < SITE_LIB_NAME);
    let files: Vec<&str> = record.sites.iter().map(|site| site.file.as_str()).collect();
    assert_eq!(files, ["src/lib.rs", "tests/other.rs"]);
    assert_eq!(record.file, "src/lib.rs");
    assert_eq!(record.lines, [[3, 9, 12], [5, 5, 14]]);
    let descriptor = record.descriptor();
    assert_eq!(descriptor.source.file, "src/lib.rs");
    assert_eq!(descriptor.source.lines.len(), 2);
    assert_eq!(descriptor.source.lines[0].rust_line, 12);
    // Everything but the origin is the schema 1 sample's.
    let schema_1 = sample();
    assert_eq!(record.key, schema_1.key);
    assert_eq!(record.parameters, schema_1.parameters);
    assert_eq!(record.slang, schema_1.slang);
}

#[test]
fn sites_of_one_file_are_sorted_by_their_first_line() {
    // A third call in `src/lib.rs`, above the first one; its name sorts after both others.
    let earlier = SITE_LIB_JSON
        .replace("[3, 9, 12]", "[3, 9, 4]")
        .replace("[5, 5, 14]", "[5, 5, 6]");
    let name = "022c52d675787198.ffffffffffffffff.site.json";
    let scratch = Scratch::new("three sites");
    let directory = site_directory(
        &scratch,
        &[
            (SITE_LIB_NAME, SITE_LIB_JSON),
            (SITE_OTHER_NAME, SITE_OTHER_JSON),
            (name, &earlier),
        ],
    );
    let records = read_records(&directory).unwrap();
    let origins: Vec<(&str, u32)> = records[0]
        .sites
        .iter()
        .map(|site| (site.file.as_str(), site.lines[0][2]))
        .collect();
    assert_eq!(
        origins,
        [
            ("src/lib.rs", 4),
            ("src/lib.rs", 12),
            ("tests/other.rs", 40)
        ]
    );
    assert_eq!(records[0].lines, [[3, 9, 4], [5, 5, 6]]);
    // A schema 1 record has no sites.
    assert!(sample().sites.is_empty());
}

#[test]
fn one_sidecar_is_enough_and_a_record_without_any_is_refused() {
    let scratch = Scratch::new("one site");
    let directory = site_directory(&scratch, &[(SITE_LIB_NAME, SITE_LIB_JSON)]);
    let records = read_records(&directory).unwrap();
    assert_eq!(records[0].file, "src/lib.rs");
    assert_eq!(records[0].lines, [[3, 9, 12], [5, 5, 14]]);
    let bare = Scratch::new("no site");
    let directory = site_directory(&bare, &[]);
    let error = read_records(&directory).unwrap_err();
    assert!(error.contains("at least one call site"), "{error}");
}

#[test]
fn a_sidecar_that_does_not_belong_is_refused() {
    // A schema 1 record has its own source: a sidecar next to it is a mixed-up collection.
    let scratch = Scratch::new("schema 1 with a site");
    let directory = scratch.0.join("records");
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join(format!("{SAMPLE_KEY}.kernel.json")),
        SAMPLE_RECORD_JSON,
    )
    .unwrap();
    fs::write(directory.join(format!("{SAMPLE_KEY}.slang")), SAMPLE_SLANG).unwrap();
    fs::write(directory.join(SITE_LIB_NAME), SITE_LIB_JSON).unwrap();
    let error = read_records(&directory).unwrap_err();
    assert!(error.contains("has no call sites"), "{error}");
    // A sidecar of another key.
    let scratch = Scratch::new("foreign site");
    let foreign = SITE_LIB_JSON.replace(SAMPLE_KEY, "ffffffffffffffff");
    let directory = site_directory(&scratch, &[(SITE_LIB_NAME, &foreign)]);
    let error = read_records(&directory).unwrap_err();
    assert!(
        error.contains("is not the key of its kernel record"),
        "{error}"
    );
    // A sidecar that is not schema 2, one that is malformed and files that only look like sidecars.
    let not_schema_2 = SITE_LIB_JSON.replace("\"schema\": 2", "\"schema\": 1");
    let short_line = SITE_LIB_JSON.replace("[3, 9, 12]", "[3, 9]");
    let no_file = SITE_LIB_JSON.replace("\"file\"", "\"path\"");
    for (name, text, wanted) in [
        (SITE_LIB_NAME, &not_schema_2, "unsupported schema"),
        (SITE_LIB_NAME, &short_line, "site.lines[0]"),
        (SITE_LIB_NAME, &no_file, "file"),
        (
            "022c52d675787198.site.json",
            &SITE_LIB_JSON.to_owned(),
            "unexpected file",
        ),
        (
            "022c52d675787198.6B7D03DA5734408D.site.json",
            &SITE_LIB_JSON.to_owned(),
            "unexpected file",
        ),
    ] {
        let scratch = Scratch::new("bad site");
        let directory = site_directory(&scratch, &[(name, text)]);
        let error = read_records(&directory).unwrap_err();
        assert!(error.contains(wanted), "{name}: {error}");
    }
}

#[test]
fn a_schema_2_record_that_names_a_source_is_refused() {
    let with_source = site_record_json().replace(
        "  \"schema\": 2,",
        "  \"schema\": 2,\n  \"source\": {\"file\": \"src/lib.rs\", \"lines\": []},",
    );
    assert!(with_source.contains("\"source\""));
    let error = Record::parse(&with_source, SAMPLE_SLANG).unwrap_err();
    assert!(error.contains("has no `source`"), "{error}");
}

#[test]
fn an_unknown_schema_is_refused_naming_both_this_tool_reads() {
    let error = error_of(&broken("  \"schema\": 1,", "  \"schema\": 3,"));
    assert!(
        error.contains("unsupported schema") && error.contains("schemas 1 and 2"),
        "{error}"
    );
}
