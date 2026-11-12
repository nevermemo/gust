//! Array layouts in the record and in the export cross-check (slice m9.pa, review S1).
//!
//! `testdata/array/pre_arrays.rs` is the module of `crates/gust-wgpu/tests/precompiled.rs` (same Slang, same keys) and
//! of the scratch crate that `cargo gust build` ran on: an array element (`[u32; 4]`), a struct with an array field
//! (`Bag`), an array of vectors (`[Vec2; 2]`) and a struct with arrays of structs and of vectors (`Poly`). The records
//! are what the macro wrote for it (`GUST_EXPORT_DIR`, `testdata/array/<key>.kernel.json`), the WGSL and the
//! reflection JSON what the native helper wrote, so the cross-check that runs here is gust's own and no helper is
//! needed. The record is compared with the reflection, and a record whose layouts are all `null` is filled from it.

use gust::{ResourceLayout, ShaderLayoutRules, artifact::verify_artifact};

use super::*;
use crate::record::{LayoutKindRecord, ResourceRecord};

#[allow(clippy::assign_op_pattern)]
mod fixtures {
    include!("testdata/array/pre_arrays.rs");
}

struct Stored {
    kernel: &'static KernelDescriptor,
    key: &'static str,
    name: &'static str,
    record_json: &'static str,
    wgsl: &'static [u8],
    reflection: &'static str,
}

const ELEMENT: Stored = Stored {
    kernel: &fixtures::pre_arrays::element::DESCRIPTOR,
    key: "54bb2958cc5fe568",
    name: "pre_arrays::element",
    record_json: include_str!("testdata/array/54bb2958cc5fe568.kernel.json"),
    wgsl: include_bytes!("testdata/array/54bb2958cc5fe568.wgsl"),
    reflection: include_str!("testdata/array/54bb2958cc5fe568.wgsl.reflection.json"),
};
const FIELD: Stored = Stored {
    kernel: &fixtures::pre_arrays::field::DESCRIPTOR,
    key: "794f2959e8738056",
    name: "pre_arrays::field",
    record_json: include_str!("testdata/array/794f2959e8738056.kernel.json"),
    wgsl: include_bytes!("testdata/array/794f2959e8738056.wgsl"),
    reflection: include_str!("testdata/array/794f2959e8738056.wgsl.reflection.json"),
};
const SHAPES: Stored = Stored {
    kernel: &fixtures::pre_arrays::shapes::DESCRIPTOR,
    key: "baf2462e890b0200",
    name: "pre_arrays::shapes",
    record_json: include_str!("testdata/array/baf2462e890b0200.kernel.json"),
    wgsl: include_bytes!("testdata/array/baf2462e890b0200.wgsl"),
    reflection: include_str!("testdata/array/baf2462e890b0200.wgsl.reflection.json"),
};
const ALL: [Stored; 3] = [ELEMENT, FIELD, SHAPES];

impl Stored {
    /// The record as the macro writes it (the stored `.kernel.json`, read by the tool's own reader).
    fn record(&self) -> Record {
        assert_eq!(
            format!(
                "{:016x}",
                gust::artifact::content_key(self.kernel.entry_point, self.kernel.slang_source)
            ),
            self.key,
            "the macro's output drifted from the stored artifact"
        );
        Record::parse(self.record_json, self.kernel.slang_source).unwrap()
    }

    /// The record with every layout `null`.
    fn null_record(&self) -> Record {
        let mut record = self.record();
        for parameter in &mut record.parameters {
            parameter.layout = None;
        }
        record
    }

    fn verify(&self, record: &Record) -> Result<String, String> {
        verify_stored(record, Format::Wgsl, self.wgsl, self.reflection)
    }

    fn reflection(&self) -> Reflection {
        verify_artifact(
            self.kernel,
            &Artifact {
                format: Format::Wgsl,
                key: self.key,
                bytes: self.wgsl,
                reflection: self.reflection,
            },
        )
        .unwrap()
        .compiled
        .reflection
    }
}

struct StoredCompiler;

impl Compiler for StoredCompiler {
    fn compile(
        &self,
        record: &Record,
        _kernel: &KernelDescriptor,
        format: Format,
    ) -> Result<Compiled, String> {
        let stored = ALL
            .into_iter()
            .find(|stored| stored.key == record.key)
            .ok_or("no stored artifact for this key")?;
        cross_check_completed(record, &stored.reflection(), stored.reflection)?;
        let compiler = stored.verify(record)?;
        Ok(Compiled {
            bytes: stored.wgsl.to_vec(),
            reflection_json: stored.reflection.to_owned(),
            compiler,
            validated: (format == Format::Spirv).then_some(false),
            warnings: Vec::new(),
        })
    }
}

fn layouts(kernel: &KernelDescriptor) -> Vec<Option<ResourceLayout>> {
    kernel
        .parameters
        .iter()
        .map(|parameter| parameter.resource_layout)
        .collect()
}

fn layout_of<'a>(record: &'a mut Record, parameter: &str) -> &'a mut ResourceRecord {
    record
        .parameters
        .iter_mut()
        .find(|candidate| candidate.name == parameter)
        .unwrap()
        .layout
        .as_mut()
        .unwrap()
}

#[test]
fn the_records_carry_the_array_kind_and_the_rules_of_their_elements() {
    let mut element = ELEMENT.record();
    let quads = layout_of(&mut element, "quads").clone();
    assert_eq!(quads.rules, ShaderLayoutRules::StorageV1);
    assert_eq!((quads.element_stride, quads.element.size), (16, 16));
    assert!(matches!(
        quads.element.kind,
        LayoutKindRecord::Array { count: 4, .. }
    ));

    let mut field = FIELD.record();
    let bags = layout_of(&mut field, "bags").clone();
    assert_eq!(bags.rules, ShaderLayoutRules::StorageV1);
    let LayoutKindRecord::Struct(fields) = &bags.element.kind else {
        panic!("Bag is a struct");
    };
    assert!(matches!(
        fields[0].ty.kind,
        LayoutKindRecord::Array { count: 4, .. }
    ));

    // Vectors inside arrays make the element `StorageV2`.
    let mut shapes = SHAPES.record();
    for name in ["pairs", "polys"] {
        assert_eq!(
            layout_of(&mut shapes, name).rules,
            ShaderLayoutRules::StorageV2,
            "{name}"
        );
    }
}

#[test]
fn the_records_rebuild_the_macros_own_descriptors_and_pass_the_cross_check() {
    for stored in ALL {
        let record = stored.record();
        assert!(record.unchecked_parameters().is_empty(), "{}", stored.name);
        assert_eq!(
            layouts(&record.descriptor()),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
        stored
            .reflection()
            .cross_check_kernel(&record.descriptor())
            .unwrap();
        assert_eq!(
            stored.verify(&record).unwrap(),
            crate::slang::SLANG_VERSION,
            "{}",
            stored.name
        );
    }
}

#[test]
fn a_null_record_is_filled_with_the_macros_layouts_for_arrays() {
    for stored in ALL {
        let record = stored.null_record();
        assert!(!record.unchecked_parameters().is_empty(), "{}", stored.name);
        let completed = completed_descriptor(&record, stored.reflection).unwrap();
        assert_eq!(
            layouts(&completed),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
        assert_eq!(
            stored.verify(&record).unwrap(),
            crate::slang::SLANG_VERSION,
            "{}",
            stored.name
        );
    }
}

#[test]
fn null_array_buffers_are_listed_in_export_unchecked_and_macro_layouts_list_nothing() {
    let planned = plan(
        vec![
            ELEMENT.null_record(),
            FIELD.null_record(),
            SHAPES.null_record(),
        ],
        &[Format::Wgsl],
        &StoredCompiler,
    )
    .unwrap();
    let listed = |key: &str| {
        planned
            .manifest
            .kernels
            .iter()
            .find(|entry| entry.key == key)
            .unwrap()
            .export_unchecked
            .clone()
    };
    assert_eq!(listed(ELEMENT.key), ["quads"]);
    assert_eq!(listed(FIELD.key), ["bags"]);
    assert_eq!(listed(SHAPES.key), ["pairs", "polys"]);
    let planned = plan(
        vec![ELEMENT.record(), FIELD.record(), SHAPES.record()],
        &[Format::Wgsl],
        &StoredCompiler,
    )
    .unwrap();
    assert!(
        planned
            .manifest
            .kernels
            .iter()
            .all(|entry| entry.export_unchecked.is_empty())
    );
}

#[test]
fn a_tampered_array_layout_in_the_record_is_caught_at_export() {
    type Edit = fn(&mut Record);
    // (what, stored kernel, edit of the record, text the error must contain)
    let cases: [(&str, &Stored, Edit, &str); 3] = [
        (
            "the array count changed",
            &ELEMENT,
            |record| {
                let layout = layout_of(record, "quads");
                let LayoutKindRecord::Array { count, .. } = &mut layout.element.kind else {
                    panic!("quads is an array");
                };
                *count = 3;
            },
            "host layout is not StorageV1",
        ),
        (
            "an array field is a scalar in the record",
            &FIELD,
            |record| {
                let LayoutKindRecord::Struct(fields) = &mut layout_of(record, "bags").element.kind
                else {
                    panic!("Bag is a struct");
                };
                fields[0].ty.kind = LayoutKindRecord::Scalar(gust::ScalarKind::U32);
            },
            "host layout is not StorageV1",
        ),
        (
            "an array of vectors declared with the scalar rules",
            &SHAPES,
            |record| layout_of(record, "pairs").rules = ShaderLayoutRules::StorageV1,
            "host layout is not StorageV1",
        ),
    ];
    for (what, stored, edit, expected) in cases {
        let mut tampered = stored.record();
        edit(&mut tampered);
        let error =
            cross_check_completed(&tampered, &stored.reflection(), stored.reflection).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
        let error = stored.verify(&tampered).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
        let error = plan(vec![tampered], &[Format::Wgsl], &StoredCompiler).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
    }
}
