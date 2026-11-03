//! Vector, matrix and std140 layouts in the record (slice m7.lv).
//!
//! The fixtures `tests/fixtures/uniforms.rs` (a `Uniform<Frame>` with a `Mat4`, `Vec4`s, a nested struct and `Vec2`
//! fields, and two `Vec4` buffers) and `tests/fixtures/vectors.rs` (buffers of `Vec4`, `UVec2`, `IVec4`, `UVec4`,
//! `Vec2`, a struct with `Vec2` fields and a scalar one) are compiled by `#[gust::gpu]` here, and the records the macro
//! wrote for them (`testdata/uniform/<key>.kernel.json`, `cargo gust build` on a crate with the same modules) are read
//! by the tool's own reader. The WGSL and the reflection JSON are what the native helper wrote for them, so the
//! cross-check that runs here is gust's own and no helper is needed.

use gust::{ResourceLayout, ShaderLayoutRules, artifact::verify_artifact};

use super::*;
use crate::record::{LayoutKindRecord, LayoutRecord, ResourceRecord};

// The fixtures are shared with other crates; `vectors.rs` spells `w = w * s` on purpose (the operator lowering).
#[allow(clippy::assign_op_pattern)]
mod fixtures {
    include!("../../../tests/fixtures/uniforms.rs");
    include!("../../../tests/fixtures/vectors.rs");
    pub(super) use {uniforms::run as frame_run, vectors::run as vectors_run};
}

const FRAME_KEY: &str = "98a66a209fa92eb1";
const FRAME_RECORD: &str = include_str!("testdata/uniform/98a66a209fa92eb1.kernel.json");
const FRAME_WGSL: &[u8] = include_bytes!("testdata/uniform/98a66a209fa92eb1.wgsl");
const FRAME_REFLECTION: &str =
    include_str!("testdata/uniform/98a66a209fa92eb1.wgsl.reflection.json");
const VECTORS_KEY: &str = "7ac7b445a03f9ce0";
const VECTORS_RECORD: &str = include_str!("testdata/uniform/7ac7b445a03f9ce0.kernel.json");
const VECTORS_WGSL: &[u8] = include_bytes!("testdata/uniform/7ac7b445a03f9ce0.wgsl");
const VECTORS_REFLECTION: &str =
    include_str!("testdata/uniform/7ac7b445a03f9ce0.wgsl.reflection.json");

struct Stored {
    kernel: &'static KernelDescriptor,
    key: &'static str,
    name: &'static str,
    record_json: &'static str,
    wgsl: &'static [u8],
    reflection: &'static str,
}

const FRAME: Stored = Stored {
    kernel: &fixtures::frame_run::DESCRIPTOR,
    key: FRAME_KEY,
    name: "uniforms::run",
    record_json: FRAME_RECORD,
    wgsl: FRAME_WGSL,
    reflection: FRAME_REFLECTION,
};
const VECTORS: Stored = Stored {
    kernel: &fixtures::vectors_run::DESCRIPTOR,
    key: VECTORS_KEY,
    name: "vectors::run",
    record_json: VECTORS_RECORD,
    wgsl: VECTORS_WGSL,
    reflection: VECTORS_REFLECTION,
};

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
        let stored = [FRAME, VECTORS]
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

/// Every layout in a tree, the root first.
fn walk<'a>(layout: &'a LayoutRecord, into: &mut Vec<&'a LayoutRecord>) {
    into.push(layout);
    if let LayoutKindRecord::Struct(fields) = &layout.kind {
        for field in fields {
            walk(&field.ty, into);
        }
    }
}

#[test]
fn the_records_carry_the_vector_matrix_and_std140_kinds_and_rules() {
    let mut frame = FRAME.record();
    let rules: Vec<(&str, ShaderLayoutRules)> = frame
        .parameters
        .iter()
        .filter_map(|parameter| {
            parameter
                .layout
                .as_ref()
                .map(|layout| (parameter.name.as_str(), layout.rules))
        })
        .collect();
    assert_eq!(
        rules,
        [
            ("frame", ShaderLayoutRules::Std140V1),
            ("points", ShaderLayoutRules::StorageV2),
            ("out", ShaderLayoutRules::StorageV2),
        ]
    );
    // `Frame`: a matrix, vectors and a nested struct in one uniform; std140 keeps the element size as its stride.
    let mut seen = Vec::new();
    let element = layout_of(&mut frame, "frame").element.clone();
    walk(&element, &mut seen);
    assert!(seen.iter().any(|layout| matches!(
        layout.kind,
        LayoutKindRecord::Matrix {
            scalar: gust::ScalarKind::F32,
            rows: 4,
            cols: 4,
            column_stride: 16
        }
    )));
    assert!(seen.iter().any(|layout| matches!(
        layout.kind,
        LayoutKindRecord::Vector {
            scalar: gust::ScalarKind::F32,
            count: 4
        }
    )));
    assert!(seen.iter().any(|layout| matches!(
        layout.kind,
        LayoutKindRecord::Vector {
            scalar: gust::ScalarKind::F32,
            count: 2
        }
    )));
    assert_eq!((element.size, element.alignment), (128, 16));
    assert_eq!(layout_of(&mut frame, "frame").element_stride, 128);
    // The storage element `Vec4`: a vector at its 16-byte stride.
    let points = layout_of(&mut frame, "points").clone();
    assert_eq!(points.element_stride, 16);
    assert_eq!(
        points.element.kind,
        LayoutKindRecord::Vector {
            scalar: gust::ScalarKind::F32,
            count: 4
        }
    );

    // The vectors fixture: scalar buffers stay `StorageV1`, everything that holds a vector is `StorageV2`.
    let mut vectors = VECTORS.record();
    let sums = layout_of(&mut vectors, "sums").clone();
    assert_eq!(sums.rules, ShaderLayoutRules::StorageV1);
    for (name, count, scalar) in [
        ("counts", 2, gust::ScalarKind::U32),
        ("signed_in", 4, gust::ScalarKind::I32),
        ("wide_in", 4, gust::ScalarKind::U32),
        ("velocities", 2, gust::ScalarKind::F32),
    ] {
        let layout = layout_of(&mut vectors, name).clone();
        assert_eq!(layout.rules, ShaderLayoutRules::StorageV2, "{name}");
        assert_eq!(
            layout.element.kind,
            LayoutKindRecord::Vector { scalar, count },
            "{name}"
        );
        assert_eq!(layout.element_stride, count * 4, "{name}");
    }
    let particles = layout_of(&mut vectors, "particles").clone();
    assert_eq!(particles.rules, ShaderLayoutRules::StorageV2);
    assert_eq!(particles.element_stride, 24);
    assert!(
        matches!(&particles.element.kind, LayoutKindRecord::Struct(fields) if fields.len() == 4),
        "{:?}",
        particles.element.kind
    );
}

#[test]
fn the_records_rebuild_the_macros_own_descriptors() {
    for stored in [FRAME, VECTORS] {
        let record = stored.record();
        assert!(record.unchecked_parameters().is_empty(), "{}", stored.name);
        assert_eq!(
            layouts(&record.descriptor()),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
        // The descriptor of the record passes the cross-check against the stored reflection as it is, and the stored
        // artifact verifies as a program would verify it, in the record's descriptor and in the macro's.
        stored
            .reflection()
            .cross_check_kernel(&record.descriptor())
            .unwrap();
        assert_eq!(
            stored.verify(&record).unwrap(),
            "2026.20",
            "{}",
            stored.name
        );
    }
}

#[test]
fn a_null_record_is_filled_with_the_macros_layouts_for_vectors_and_matrices() {
    for stored in [FRAME, VECTORS] {
        let record = stored.null_record();
        let completed = completed_descriptor(&record, stored.reflection).unwrap();
        // Rules, strides and every vector and matrix kind of the reflection's fill are the macro's numbers.
        assert_eq!(
            layouts(&completed),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
        assert_eq!(
            stored.verify(&record).unwrap(),
            "2026.20",
            "{}",
            stored.name
        );
    }
}

#[test]
fn a_null_uniform_and_null_vector_buffers_are_listed_in_export_unchecked() {
    let mut record = FRAME.record();
    for parameter in &mut record.parameters {
        if parameter.name == "frame" {
            parameter.layout = None;
        }
    }
    assert_eq!(record.unchecked_parameters(), ["frame"]);
    let planned = plan(
        vec![record, VECTORS.null_record()],
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
    assert_eq!(listed(FRAME_KEY), ["frame"]);
    assert_eq!(
        listed(VECTORS_KEY),
        [
            "points",
            "particles",
            "counts",
            "signed_in",
            "wide_in",
            "moved",
            "sums",
            "cells",
            "velocities",
            "signed_out"
        ]
    );
    // With the layouts the macro wrote, nothing is listed.
    let planned = plan(
        vec![FRAME.record(), VECTORS.record()],
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
fn a_tampered_vector_matrix_or_uniform_layout_in_the_record_is_caught_at_export() {
    let genuine = FRAME.reflection();
    // (what, edit of the record, text the error must contain)
    type Edit = fn(&mut Record);
    let cases: [(&str, Edit, &str); 4] = [
        (
            "a std140 offset moved",
            |record| {
                let LayoutKindRecord::Struct(fields) = &mut layout_of(record, "frame").element.kind
                else {
                    panic!("Frame is a struct");
                };
                fields
                    .iter_mut()
                    .find(|field| field.name == "tint")
                    .unwrap()
                    .offset = 72;
            },
            "host layout is not Std140V1",
        ),
        (
            "a matrix column stride changed",
            |record| {
                let LayoutKindRecord::Struct(fields) = &mut layout_of(record, "frame").element.kind
                else {
                    panic!("Frame is a struct");
                };
                let view = fields
                    .iter_mut()
                    .find(|field| field.name == "view")
                    .unwrap();
                let LayoutKindRecord::Matrix { column_stride, .. } = &mut view.ty.kind else {
                    panic!("view is a matrix");
                };
                *column_stride = 12;
            },
            "host layout is not Std140V1",
        ),
        (
            "a vector element count changed",
            |record| {
                let layout = layout_of(record, "points");
                layout.element.kind = LayoutKindRecord::Vector {
                    scalar: gust::ScalarKind::F32,
                    count: 2,
                };
            },
            "host layout is not StorageV2",
        ),
        (
            "a vector buffer declared with the scalar rules",
            |record| layout_of(record, "points").rules = ShaderLayoutRules::StorageV1,
            "host layout is not StorageV1",
        ),
    ];
    for (what, edit, expected) in cases {
        let mut tampered = FRAME.record();
        edit(&mut tampered);
        let error = cross_check_completed(&tampered, &genuine, FRAME.reflection).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
        let error = FRAME.verify(&tampered).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
        let error = plan(vec![tampered], &[Format::Wgsl], &StoredCompiler).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
    }
}

#[test]
fn a_vector_buffer_fails_when_the_record_says_it_is_a_scalar() {
    // The scalar a reader would produce for a vector element: the same size, no vector kind.
    let genuine = VECTORS.reflection();
    let mut record = VECTORS.record();
    let counts = layout_of(&mut record, "counts");
    counts.element.kind = LayoutKindRecord::Scalar(gust::ScalarKind::U32);
    let error = cross_check_completed(&record, &genuine, VECTORS.reflection).unwrap_err();
    assert!(error.contains("host layout is not StorageV2"), "{error}");
}
