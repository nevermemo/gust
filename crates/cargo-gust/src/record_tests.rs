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
    let expected = KernelDescriptor::new(
        "demo",
        "run",
        "gpu_demo_run",
        [64, 1, 1],
        EXPECTED_PARAMETERS,
        SAMPLE_SLANG,
    )
    .with_source(SourceOrigin {
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
    });
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
            broken("\"schema\": 1", "\"schema\": 4"),
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
            broken("\"kind\": \"invocation\"", "\"kind\": \"image\""),
            "parameters[0]: unknown parameter kind `image`",
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
fn an_unknown_schema_is_refused_naming_the_three_this_tool_reads() {
    let error = error_of(&broken("  \"schema\": 1,", "  \"schema\": 4,"));
    assert!(
        error.contains("unsupported schema") && error.contains("schemas 1, 2 and 3"),
        "{error}"
    );
}

/// The sample with the layout block of the `out` buffer replaced: `element` is the element's JSON object (inside
/// braces already) and `rules` the rules string.
fn with_out_layout(element: &str, stride: u32, rules: &str) -> String {
    let start = SAMPLE_RECORD_JSON
        .find(concat!(
            "      \"layout\": {\n        \"element\": {\n",
            "          \"alignment\": 4,\n          \"kind\": \"scalar\""
        ))
        .unwrap();
    let end = SAMPLE_RECORD_JSON.find("      \"name\": \"out\"").unwrap();
    let layout = format!(
        "      \"layout\": {{\n        \"element\": {element},\n        \"element_stride\": {stride},\n\
         \"rules\": \"{rules}\"\n      }},\n"
    );
    format!(
        "{}{layout}{}",
        &SAMPLE_RECORD_JSON[..start],
        &SAMPLE_RECORD_JSON[end..]
    )
}

const VEC4_JSON: &str = concat!(
    "{\"alignment\": 16, \"count\": 4, \"kind\": \"vector\", \"name\": \"Vec4\", ",
    "\"scalar\": \"f32\", \"size\": 16, \"stride\": 16}"
);
const MAT4_JSON: &str = concat!(
    "{\"alignment\": 16, \"cols\": 4, \"column_stride\": 16, \"kind\": \"matrix\", ",
    "\"name\": \"Mat4\", \"rows\": 4, \"scalar\": \"f32\", \"size\": 64, \"stride\": 64}"
);

#[test]
fn a_vector_layout_is_read_and_rebuilt_with_storage_v2_rules() {
    let record =
        Record::parse(&with_out_layout(VEC4_JSON, 16, "storage_v2"), SAMPLE_SLANG).unwrap();
    let layout = record.parameters[2].layout.as_ref().unwrap();
    assert_eq!(layout.rules, ShaderLayoutRules::StorageV2);
    assert_eq!(layout.element_stride, 16);
    assert_eq!(
        layout.element.kind,
        LayoutKindRecord::Vector {
            scalar: ScalarKind::F32,
            count: 4
        }
    );
    assert_eq!((layout.element.size, layout.element.alignment), (16, 16));
    let rebuilt = record.descriptor().parameters[2].resource_layout.unwrap();
    assert_eq!(
        rebuilt,
        ResourceLayout::storage_v2(TypeLayout::vector("Vec4", ScalarKind::F32, 4))
    );
    assert!(layout.element.holds_math_type());
}

#[test]
fn a_matrix_layout_is_read_and_rebuilt_under_storage_v2_and_std140_v1() {
    for (rules, expected) in [
        (
            "storage_v2",
            ResourceLayout::storage_v2(TypeLayout::matrix("Mat4", ScalarKind::F32, 4, 4)),
        ),
        (
            "std140_v1",
            ResourceLayout::std140_v1(TypeLayout::matrix("Mat4", ScalarKind::F32, 4, 4)),
        ),
    ] {
        let record = Record::parse(&with_out_layout(MAT4_JSON, 64, rules), SAMPLE_SLANG).unwrap();
        let layout = record.parameters[2].layout.as_ref().unwrap();
        assert_eq!(
            layout.element.kind,
            LayoutKindRecord::Matrix {
                scalar: ScalarKind::F32,
                rows: 4,
                cols: 4,
                column_stride: 16
            },
            "{rules}"
        );
        assert_eq!(
            record.descriptor().parameters[2].resource_layout.unwrap(),
            expected,
            "{rules}"
        );
    }
}

#[test]
fn a_reader_from_before_this_slice_would_refuse_the_new_rules_by_path() {
    // The new strings are read here; an unknown one (what a reader without them says to `storage_v2` and `std140_v1`)
    // is refused naming the path, so an older cargo-gust fails clearly, not silently, on a new record.
    for known in ["storage_v1", "storage_v2", "std140_v1"] {
        Record::parse(&with_out_layout(VEC4_JSON, 16, known), SAMPLE_SLANG).unwrap();
    }
    let error = error_of(&with_out_layout(VEC4_JSON, 16, "std430"));
    assert!(
        error.contains("parameters[2].layout: unknown layout rules `std430`"),
        "{error}"
    );
}

#[test]
fn malformed_vector_and_matrix_layouts_are_refused_with_the_path_of_the_problem() {
    for (element, expected) in [
        (
            VEC4_JSON.replace("\"count\": 4, ", ""),
            "parameters[2].layout.element: missing `count`",
        ),
        (
            VEC4_JSON.replace("\"scalar\": \"f32\"", "\"scalar\": \"f64\""),
            "unknown scalar `f64`",
        ),
        (
            MAT4_JSON.replace("\"column_stride\": 16, ", ""),
            "parameters[2].layout.element: missing `column_stride`",
        ),
        (
            MAT4_JSON.replace("\"rows\": 4, ", ""),
            "parameters[2].layout.element: missing `rows`",
        ),
        (
            VEC4_JSON.replace("\"vector\"", "\"tensor\""),
            "parameters[2].layout.element: unknown layout kind `tensor`",
        ),
    ] {
        let error = error_of(&with_out_layout(&element, 16, "storage_v2"));
        assert!(error.contains(expected), "{expected:?} not in {error:?}");
    }
}

// ---- schema 3: the record of a kernel whose shape is not `grid` (mx.k, D89) ----

/// The sample record as a pixel kernel's `#[gpu]` record: schema 3, `shape` between `schema` and `source`.
fn pixel_record_json() -> String {
    broken(
        "  \"schema\": 1,\n",
        "  \"schema\": 3,\n  \"shape\": \"pixel\",\n",
    )
}

/// The `par_for!` form of the same: schema 3 without a `source`.
fn pixel_site_record_json() -> String {
    site_record_json().replace(
        "  \"schema\": 2,\n",
        "  \"schema\": 3,\n  \"shape\": \"pixel\",\n",
    )
}

#[test]
fn a_record_without_a_shape_reads_as_grid() {
    // Schema 1 and schema 2 never carry one; the macro writes `shape` only for another shape (D89).
    assert!(!SAMPLE_RECORD_JSON.contains("shape") && !site_record_json().contains("shape"));
    assert_eq!(sample().shape, Shape::Grid);
    assert_eq!(sample().descriptor().shape, Shape::Grid);
    let site = Record::parse(&site_record_json(), SAMPLE_SLANG).unwrap();
    assert_eq!(site.shape, Shape::Grid);
    // A schema 3 record that has no `shape` either is `grid` as well (the key is optional), and one that says so
    // explicitly reads the same.
    let bare = broken("  \"schema\": 1,\n", "  \"schema\": 3,\n");
    assert_eq!(
        Record::parse(&bare, SAMPLE_SLANG).unwrap().shape,
        Shape::Grid
    );
    let explicit = broken(
        "  \"schema\": 1,\n",
        "  \"schema\": 3,\n  \"shape\": \"grid\",\n",
    );
    assert_eq!(
        Record::parse(&explicit, SAMPLE_SLANG).unwrap().shape,
        Shape::Grid
    );
}

#[test]
fn a_schema_3_record_carries_its_shape_into_the_descriptor() {
    let (record, schema, sited) = Record::parse_any(&pixel_record_json(), SAMPLE_SLANG).unwrap();
    assert_eq!((record.shape, schema, sited), (Shape::Pixel, 3, false));
    // A schema 3 record that has its own `source` is read like schema 1: the origin is in the record.
    assert_eq!(record.file, "src/lib.rs");
    assert_eq!(record.lines, [[3, 9, 12], [5, 5, 14]]);
    assert_eq!(record.descriptor().shape, Shape::Pixel);
    // Everything else is the sample's.
    assert_eq!(record.key, sample().key);
    assert_eq!(record.parameters, sample().parameters);
    // Without a `source` it is read like schema 2: no origin until its sidecars are read.
    let (site, schema, sited) = Record::parse_any(&pixel_site_record_json(), SAMPLE_SLANG).unwrap();
    assert_eq!((site.shape, schema, sited), (Shape::Pixel, 3, true));
    assert_eq!((site.file.as_str(), site.lines.len()), ("", 0));
}

#[test]
fn a_schema_3_site_record_takes_its_origin_from_its_sidecars() {
    let scratch = Scratch::new("pixel site");
    let directory = scratch.0.join("records");
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join(format!("{SAMPLE_KEY}.kernel.json")),
        pixel_site_record_json(),
    )
    .unwrap();
    fs::write(directory.join(format!("{SAMPLE_KEY}.slang")), SAMPLE_SLANG).unwrap();
    fs::write(directory.join(SITE_LIB_NAME), SITE_LIB_JSON).unwrap();
    let records = read_records(&directory).unwrap();
    assert_eq!(records[0].shape, Shape::Pixel);
    assert_eq!(records[0].file, "src/lib.rs");
    assert_eq!(records[0].lines, [[3, 9, 12], [5, 5, 14]]);
    // Without a sidecar it is refused, as a schema 2 record is.
    fs::remove_file(directory.join(SITE_LIB_NAME)).unwrap();
    let error = read_records(&directory).unwrap_err();
    assert!(
        error.contains("is schema 3 but has no") && error.contains("at least one call site"),
        "{error}"
    );
    // A schema 3 record with its own `source` takes no sidecar, like schema 1.
    fs::write(
        directory.join(format!("{SAMPLE_KEY}.kernel.json")),
        pixel_record_json(),
    )
    .unwrap();
    fs::write(directory.join(SITE_LIB_NAME), SITE_LIB_JSON).unwrap();
    let error = read_records(&directory).unwrap_err();
    assert!(error.contains("has no call sites"), "{error}");
}

#[test]
fn a_reader_of_schemas_1_and_2_refuses_a_record_with_a_shape() {
    // The number moves so that a reader from before mx.k cannot read a pixel kernel as `grid`.
    let before = [RECORD_SCHEMA, SITE_RECORD_SCHEMA];
    for json in [pixel_record_json(), pixel_site_record_json()] {
        let error = Record::parse_schemas(&json, SAMPLE_SLANG, &before).unwrap_err();
        assert!(
            error.contains("unsupported schema") && error.contains("schemas 1 and 2"),
            "{error}"
        );
        // The reader of this tool accepts the same text.
        assert!(Record::parse_any(&json, SAMPLE_SLANG).is_ok());
    }
    // The old readers still read what they always read.
    for json in [SAMPLE_RECORD_JSON.to_owned(), site_record_json()] {
        assert!(Record::parse_schemas(&json, SAMPLE_SLANG, &before).is_ok());
    }
}

#[test]
fn a_shape_under_schema_1_or_2_is_refused_and_a_bad_shape_is_named() {
    let under_1 = broken(
        "  \"schema\": 1,\n",
        "  \"schema\": 1,\n  \"shape\": \"pixel\",\n",
    );
    let under_2 = site_record_json().replace(
        "  \"schema\": 2,\n",
        "  \"schema\": 2,\n  \"shape\": \"pixel\",\n",
    );
    for json in [under_1, under_2] {
        let error = Record::parse(&json, SAMPLE_SLANG).unwrap_err();
        assert!(
            error.contains("has no `shape`") && error.contains("schema 3 key"),
            "{error}"
        );
    }
    for (value, wanted) in [
        ("\"cpu\"", "unknown shape `cpu`"),
        ("\"Pixel\"", "unknown shape `Pixel`"),
        ("7", "`shape` must be a string"),
        ("null", "`shape` must be a string"),
    ] {
        let json = broken(
            "  \"schema\": 1,\n",
            &format!("  \"schema\": 3,\n  \"shape\": {value},\n"),
        );
        let error = Record::parse(&json, SAMPLE_SLANG).unwrap_err();
        assert!(error.contains(wanted), "{value}: {error}");
    }
}

// --- Texture and sampler parameters (slice mx.t3, D62, D64, D91) ---

/// The record the macro's artifact writer produced for a sampled `Texture2D<Vec4>`, a `Sampler` and a write-only
/// `WriteTexture2D<Rgba32Float>`: the committed golden of `gust-macros`
/// (`the_texture_and_sampler_record_matches_the_golden`), so this reader and that writer are held to one file.
const TEXTURE_GOLDEN: &str = include_str!("../../gust-macros/tests/golden/texture.kernel.json");

/// The Slang the golden's kernel stands for. The golden's key is a placeholder, so the record is read with the key of
/// this text.
const TEXTURE_SLANG: &str = concat!(
    "[[vk::binding(0, 0)]]\n",
    "Texture2D<float4> image;\n",
    "\n",
    "[[vk::binding(1, 0)]]\n",
    "SamplerState bilinear;\n",
    "\n",
    "[[vk::binding(2, 0)]]\n",
    "[format(\"rgba32f\")]\n",
    "WriteTexture2D<float4> out;\n",
    "\n",
    "[shader(\"compute\")]\n",
    "[numthreads(8, 8, 1)]\n",
    "void gpu_textures_run(uint3 id : SV_DispatchThreadID)\n",
    "{\n",
    "    out[id.xy] = image.SampleLevel(bilinear, float2(0.5, 0.5), 0);\n",
    "}\n"
);

fn texture_record_json() -> String {
    let golden = TEXTURE_GOLDEN.replace("\r\n", "\n");
    let placeholder = "\"key\": \"0123456789abcdef\"";
    assert!(golden.contains(placeholder));
    golden.replacen(
        placeholder,
        &format!("\"key\": \"{}\"", key_of("gpu_textures_run", TEXTURE_SLANG)),
        1,
    )
}

#[test]
fn a_record_with_a_texture_and_a_sampler_reads_and_rebuilds_the_descriptor() {
    let record = Record::parse(&texture_record_json(), TEXTURE_SLANG).unwrap();
    assert_eq!(record.qualified_name(), "textures::run");
    assert_eq!(record.workgroup_size, [8, 8, 1]);
    let [id, image, sampler, out] = &record.parameters[..] else {
        panic!("{:?}", record.parameters);
    };
    assert_eq!(id.kind, ParameterKind::Invocation);
    assert_eq!(
        (image.kind, image.access, image.binding),
        (ParameterKind::Texture, Access::ReadOnly, Some((0, 0)))
    );
    // A sampler has a binding and no layout.
    assert_eq!(
        (
            sampler.kind,
            sampler.access,
            sampler.binding,
            sampler.layout.is_none()
        ),
        (
            ParameterKind::Sampler,
            Access::NotApplicable,
            Some((0, 1)),
            true
        )
    );
    assert_eq!(
        (out.kind, out.access, out.binding),
        (ParameterKind::Texture, Access::ReadWrite, Some((0, 2)))
    );
    let sampled = TextureLayout {
        format: None,
        dimension: TextureDimension::D2,
        access: TextureAccess::Sampled,
        class: SampleClass::Float,
    };
    let written = TextureLayout {
        format: Some(Format::Rgba32Float),
        dimension: TextureDimension::D2,
        access: TextureAccess::WriteOnly,
        class: SampleClass::Float,
    };
    assert_eq!(
        image.layout.as_ref().unwrap().rules,
        ShaderLayoutRules::TextureV1(sampled)
    );
    assert_eq!(
        out.layout.as_ref().unwrap().rules,
        ShaderLayoutRules::TextureV1(written)
    );
    // The descriptor carries the same facts a `#[gpu]` module's own descriptor carries (`ResourceLayout::texture_v1`
    // over the texel type), and the sampler has no layout.
    let descriptor = record.descriptor();
    let rebuilt: Vec<_> = descriptor
        .parameters
        .iter()
        .map(|parameter| (parameter.kind, parameter.binding, parameter.resource_layout))
        .collect();
    let element = <gust::math::Vec4 as gust::GpuPod>::LAYOUT;
    assert_eq!(
        rebuilt,
        [
            (ParameterKind::Invocation, None, None),
            (
                ParameterKind::Texture,
                Some(ResourceBinding {
                    group: 0,
                    binding: 0
                }),
                Some(ResourceLayout::texture_v1(element, sampled))
            ),
            (
                ParameterKind::Sampler,
                Some(ResourceBinding {
                    group: 0,
                    binding: 1
                }),
                None
            ),
            (
                ParameterKind::Texture,
                Some(ResourceBinding {
                    group: 0,
                    binding: 2
                }),
                Some(ResourceLayout::texture_v1(element, written))
            ),
        ]
    );
}

#[test]
fn a_texture_layout_without_its_facts_or_with_unknown_facts_is_refused_with_the_path() {
    let json = texture_record_json();
    // The `texture` object of the first texture parameter, with the comma that leads to it.
    let start = json.find(",\n        \"texture\": {").unwrap();
    let end = start + json[start..].find("\n        }").unwrap() + "\n        }".len();
    let without_facts = format!("{}{}", &json[..start], &json[end..]);
    for (changed, expected) in [
        (
            json.replacen("\"access\": \"sampled\"", "\"access\": \"mapped\"", 1),
            "unknown texture access `mapped`",
        ),
        (
            json.replacen("\"class\": \"f32\"", "\"class\": \"f16\"", 1),
            "unknown sample class `f16`",
        ),
        (
            json.replacen("\"dimension\": \"2d\"", "\"dimension\": \"3d\"", 1),
            "unsupported texture dimension `3d`",
        ),
        (
            json.replacen(
                "\"format\": \"rgba32float\"",
                "\"format\": \"rgba8unorm\"",
                1,
            ),
            "unknown texture format `rgba8unorm`",
        ),
        (without_facts, "texture"),
    ] {
        assert_ne!(changed, json, "{expected}");
        let error = Record::parse(&changed, TEXTURE_SLANG).unwrap_err();
        assert!(error.contains(expected), "{expected} in {error}");
        assert!(error.contains("parameters["), "{error}");
    }
    // A kind the reader does not know is still refused.
    let error = Record::parse(
        &json.replacen("\"kind\": \"sampler\"", "\"kind\": \"image\"", 1),
        TEXTURE_SLANG,
    )
    .unwrap_err();
    assert!(
        error.contains("parameters[2]: unknown parameter kind `image`"),
        "{error}"
    );
}

/// A layout and a parameter kind that disagree are refused with the path: a sampler has no layout, a `texture_v1`
/// layout belongs to a texture, and buffer rules do not (the descriptor would otherwise hold a texture that is read
/// as a buffer, or the other way round).
#[test]
fn a_layout_that_does_not_fit_the_parameter_kind_is_refused() {
    let json = texture_record_json();
    // A sampler given the layout of the texture before it.
    let sampler_with_layout = json.replacen(
        "      \"kind\": \"sampler\",\n      \"layout\": null,",
        &format!(
            concat!(
                "      \"kind\": \"sampler\",\n",
                "      \"layout\": {{\"element\": {}, \"element_stride\": 16, \"rules\": \"storage_v2\"}},"
            ),
            VEC4_JSON
        ),
        1,
    );
    assert_ne!(sampler_with_layout, json);
    let error = Record::parse(&sampler_with_layout, TEXTURE_SLANG).unwrap_err();
    assert!(
        error.contains("parameters[2].layout: a sampler has no layout"),
        "{error}"
    );
    // A texture whose layout is a buffer's, and a buffer whose layout is a texture's.
    let buffer_texture = json.replacen("\"rules\": \"texture_v1\"", "\"rules\": \"storage_v2\"", 1);
    let error = Record::parse(&buffer_texture, TEXTURE_SLANG).unwrap_err();
    assert!(
        error.contains("parameters[1].layout: `buffer` rules do not belong to a texture parameter"),
        "{error}"
    );
    let texture_buffer = json.replacen("\"kind\": \"texture\"", "\"kind\": \"storage\"", 1);
    let error = Record::parse(&texture_buffer, TEXTURE_SLANG).unwrap_err();
    assert!(
        error.contains(
            "parameters[1].layout: `texture_v1` rules do not belong to a non-texture parameter"
        ),
        "{error}"
    );
}
