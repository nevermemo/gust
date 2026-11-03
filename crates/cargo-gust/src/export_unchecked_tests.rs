//! `export_unchecked`: storage parameters whose element layout the macro could not write (`layout: null`).
//!
//! These tests use the real vector-add artifact and the reflection JSON the native helper wrote for it (gust's
//! fixture, spike S1) with the descriptor `#[gust::gpu]` makes of the same kernel, so the cross-check that runs here is
//! gust's own and no helper is needed.

use gust::{Access, ParameterKind, ScalarKind, artifact::verify_artifact};

use super::*;
use crate::{
    record::{LayoutKindRecord, LayoutRecord, ParameterRecord, ResourceRecord},
    testdata::{VECTOR_ADD_KEY, VECTOR_ADD_REFLECTION, VECTOR_ADD_WGSL},
};

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

fn f32_layout() -> Option<ResourceRecord> {
    Some(ResourceRecord {
        element_stride: 4,
        element: LayoutRecord {
            name: "f32".to_owned(),
            size: 4,
            alignment: 4,
            stride: 4,
            kind: LayoutKindRecord::Scalar(ScalarKind::F32),
        },
    })
}

/// The record of the vector-add kernel as the macro writes it, except that the named storage parameters have
/// `layout: null`.
fn vector_add_record(null_layouts: &[&str]) -> Record {
    let kernel = vector_add::add::DESCRIPTOR;
    assert_eq!(
        format!(
            "{:016x}",
            gust::artifact::content_key(kernel.entry_point, kernel.slang_source)
        ),
        VECTOR_ADD_KEY,
        "the macro's output drifted from the fixture"
    );
    let storage = |name: &str, binding: u32, access: Access| ParameterRecord {
        name: name.to_owned(),
        rust_type: "StructuredBuffer < float >".to_owned(),
        kind: ParameterKind::Storage,
        access,
        binding: Some((0, binding)),
        layout: if null_layouts.contains(&name) {
            None
        } else {
            f32_layout()
        },
    };
    Record {
        key: VECTOR_ADD_KEY.to_owned(),
        module: "vector_add".to_owned(),
        name: "add".to_owned(),
        entry_point: kernel.entry_point.to_owned(),
        workgroup_size: [64, 1, 1],
        parameters: vec![
            ParameterRecord {
                name: "id".to_owned(),
                rust_type: "SV_DispatchThreadID".to_owned(),
                kind: ParameterKind::Invocation,
                access: Access::NotApplicable,
                binding: None,
                layout: None,
            },
            storage("a", 0, Access::ReadOnly),
            storage("b", 1, Access::ReadOnly),
            storage("out", 2, Access::ReadWrite),
        ],
        file: String::new(),
        lines: Vec::new(),
        slang: kernel.slang_source.to_owned(),
        sites: Vec::new(),
    }
}

fn verify(record: &Record) -> Result<String, String> {
    verify_stored(record, Format::Wgsl, VECTOR_ADD_WGSL, VECTOR_ADD_REFLECTION)
}

#[test]
fn a_record_with_every_layout_verifies_and_lists_nothing_unchecked() {
    let record = vector_add_record(&[]);
    assert!(record.unchecked_parameters().is_empty());
    assert_eq!(verify(&record).unwrap(), "2026.20");
}

#[test]
fn null_layouts_are_filled_from_the_reflection_and_the_artifact_still_verifies() {
    let record = vector_add_record(&["a", "b", "out"]);
    assert_eq!(record.unchecked_parameters(), ["a", "b", "out"]);
    assert_eq!(verify(&record).unwrap(), "2026.20");
    // Only the named parameter is unchecked; the others keep their layouts.
    let one = vector_add_record(&["b"]);
    assert_eq!(one.unchecked_parameters(), ["b"]);
    assert_eq!(verify(&one).unwrap(), "2026.20");
}

#[test]
fn the_export_cross_check_completes_a_null_layout_first_and_never_sees_a_null_one() {
    let reflection = verify_artifact(
        &vector_add::add::DESCRIPTOR,
        &Artifact {
            format: Format::Wgsl,
            key: VECTOR_ADD_KEY,
            bytes: VECTOR_ADD_WGSL,
            reflection: VECTOR_ADD_REFLECTION,
        },
    )
    .unwrap()
    .compiled
    .reflection;
    for null_layouts in [&[][..], &["a"], &["a", "b", "out"]] {
        let record = vector_add_record(null_layouts);
        let reports = cross_check_completed(&record, &reflection, VECTOR_ADD_REFLECTION)
            .unwrap_or_else(|error| panic!("{null_layouts:?}: {error}"));
        assert_eq!(reports.len(), 3);
        assert!(reports.iter().all(PodCrossCheck::is_full_abi_match));
    }
    // The raw cross-check of a descriptor that still has a null layout does fail, which is why it is completed first.
    let raw = reflection
        .cross_check_kernel(&vector_add_record(&["a"]).descriptor())
        .unwrap_err();
    assert!(
        raw.to_string()
            .contains("missing descriptor element layout"),
        "{raw}"
    );
}

#[test]
fn skipping_the_layout_of_one_parameter_does_not_skip_the_layouts_of_the_others() {
    // `b` is described wrongly (as i32) while `a` is unchecked: the wrong one is still caught, by name.
    let mut record = vector_add_record(&["a"]);
    let wrong = &mut record.parameters[2].layout.as_mut().unwrap().element;
    wrong.name = "i32".to_owned();
    wrong.kind = LayoutKindRecord::Scalar(ScalarKind::I32);
    let error = verify(&record).unwrap_err();
    assert_eq!(
        error,
        "POD reflection mismatch: b: type or field count differs"
    );
    let wrong_stride = {
        let mut record = vector_add_record(&["a"]);
        record.parameters[3].layout.as_mut().unwrap().element_stride = 8;
        verify(&record).unwrap_err()
    };
    assert_eq!(
        wrong_stride,
        "POD reflection mismatch: out.stride: inconsistent host resource layout"
    );
}

#[test]
fn an_unchecked_parameter_still_has_its_binding_and_access_compared() {
    let mut moved = vector_add_record(&["a"]);
    moved.parameters[1].binding = Some((0, 7));
    let error = verify(&moved).unwrap_err();
    assert!(error.contains("a.binding"), "{error}");
    let mut group = vector_add_record(&["a"]);
    group.parameters[1].binding = Some((3, 0));
    let error = verify(&group).unwrap_err();
    assert!(error.contains("a.group"), "{error}");
    let mut access = vector_add_record(&["a"]);
    access.parameters[1].access = Access::ReadWrite;
    let error = verify(&access).unwrap_err();
    assert!(error.contains("a.access"), "{error}");
}

#[test]
fn a_reflection_without_the_unchecked_parameter_fails_the_verification_by_name() {
    let mut record = vector_add_record(&["a"]);
    record.parameters[1].name = "renamed".to_owned();
    record.parameters[1].layout = None;
    let error = verify(&record).unwrap_err();
    assert!(
        error.contains("vector_add::add") && error.contains("no resource named `renamed`"),
        "{error}"
    );
}

#[test]
fn the_plan_records_the_unchecked_parameters_on_the_manifest_entry_in_parameter_order() {
    let mut record = crate::export::tests::sample();
    // Parameter 1 (`pairs`) and 2 (`out`) are the storage parameters of the sample.
    record.parameters[2].layout = None;
    record.parameters[1].layout = None;
    let planned = plan(
        vec![record],
        &[Format::Wgsl],
        &crate::export::tests::FakeCompiler::new(),
    )
    .unwrap();
    assert_eq!(
        planned.manifest.kernels[0].export_unchecked,
        ["pairs", "out"]
    );
    let checked = plan(
        vec![crate::export::tests::sample()],
        &[Format::Wgsl],
        &crate::export::tests::FakeCompiler::new(),
    )
    .unwrap();
    assert!(checked.manifest.kernels[0].export_unchecked.is_empty());
    assert!(!checked.manifest_file().bytes.is_empty());
    assert!(
        !String::from_utf8(checked.manifest_file().bytes)
            .unwrap()
            .contains("export_unchecked")
    );
    assert!(
        String::from_utf8(planned.manifest_file().bytes)
            .unwrap()
            .contains("\"export_unchecked\": [\n        \"pairs\",\n        \"out\"\n      ]")
    );
}
