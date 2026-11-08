//! Uniform kernels through the export path (slices m6.u2f and m7.lv).
//!
//! The macro writes the std140 layout of a `Uniform<T>` whose type it can describe (m7.lv), so the export checks it in
//! full against the reflection and lists nothing. A uniform whose layout is `null` (a type from outside the module) is
//! still filled from the reflection's constant buffer as `Std140V1` ([`completed_descriptor`]) and is listed in
//! `export_unchecked`; D104's exemption for it is retired. These tests use the records the macro wrote for two uniform
//! kernels (`testdata/uniform/<key>.kernel.json`, made by `cargo gust build` on a crate with the same module), the WGSL
//! and the reflection JSON the native helper wrote for them, and the descriptor `#[gust::gpu]` makes of the same
//! kernels, so the cross-check that runs here is gust's own and no helper is needed.
//! `the_helper_builds_the_stored_artifacts` needs the helper and checks that the stored bytes are what it writes today.

use gust::{
    ParameterDescriptor, ParameterKind, ResourceLayout, ShaderLayoutRules,
    artifact::verify_artifact,
};

use super::*;
use crate::record::{LayoutKindRecord, ResourceRecord};

#[gust::gpu]
mod probes {
    // `gain` and three explicit padding fields: 16 bytes, no implicit padding (D80).
    pub struct Gain {
        pub gain: f32,
        pub pad0: f32,
        pub pad1: f32,
        pub pad2: f32,
    }

    pub struct Inner {
        pub count: u32,
        pub bias: i32,
        pub pad0: f32,
        pub pad1: f32,
    }

    // `scale` and padding 0..16, the nested `inner` 16..32 (a nested struct starts at a multiple of 16): 32 bytes.
    pub struct Outer {
        pub scale: f32,
        pub pad0: f32,
        pub pad1: f32,
        pub pad2: f32,
        pub inner: Inner,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scale(
        id: SV_DispatchThreadID,
        g: Uniform<Gain>,
        src: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = src[i] * g.gain;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn shifted(
        id: SV_DispatchThreadID,
        o: Uniform<Outer>,
        g: Uniform<Gain>,
        src: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = src[i] * o.scale * g.gain + o.inner.count as f32 + o.inner.bias as f32;
        }
    }
}

const SCALE_KEY: &str = "bfb6e436140611c0";
const SCALE_RECORD: &str = include_str!("testdata/uniform/bfb6e436140611c0.kernel.json");
const SCALE_WGSL: &[u8] = include_bytes!("testdata/uniform/bfb6e436140611c0.wgsl");
const SCALE_REFLECTION: &str =
    include_str!("testdata/uniform/bfb6e436140611c0.wgsl.reflection.json");
const SHIFTED_KEY: &str = "f6dd8e3fe6e29694";
const SHIFTED_RECORD: &str = include_str!("testdata/uniform/f6dd8e3fe6e29694.kernel.json");
const SHIFTED_WGSL: &[u8] = include_bytes!("testdata/uniform/f6dd8e3fe6e29694.wgsl");
const SHIFTED_REFLECTION: &str =
    include_str!("testdata/uniform/f6dd8e3fe6e29694.wgsl.reflection.json");

/// One stored kernel: its descriptor from the macro, the record the macro wrote for it, the key it must have and the
/// helper's bytes for it.
struct Stored {
    kernel: &'static KernelDescriptor,
    key: &'static str,
    name: &'static str,
    record_json: &'static str,
    wgsl: &'static [u8],
    reflection: &'static str,
}

const SCALE: Stored = Stored {
    kernel: &probes::scale::DESCRIPTOR,
    key: SCALE_KEY,
    name: "scale",
    record_json: SCALE_RECORD,
    wgsl: SCALE_WGSL,
    reflection: SCALE_REFLECTION,
};
const SHIFTED: Stored = Stored {
    kernel: &probes::shifted::DESCRIPTOR,
    key: SHIFTED_KEY,
    name: "shifted",
    record_json: SHIFTED_RECORD,
    wgsl: SHIFTED_WGSL,
    reflection: SHIFTED_REFLECTION,
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

    /// The record with every layout `null`: what the macro writes for types it cannot describe.
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

    /// The parsed reflection of the stored bytes (what `HelperCompiler` gets from the helper), read by the macro's own
    /// descriptor, which carries the std140 layouts: the stored bytes are the genuine ones.
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

/// `HelperCompiler::compile` with the stored bytes in place of the helper's run: the same cross-check, the same
/// verification as a program would do, the same `Compiled`.
struct StoredCompiler;

impl Compiler for StoredCompiler {
    fn compile(
        &self,
        record: &Record,
        _kernel: &KernelDescriptor,
        format: Format,
    ) -> Result<Compiled, String> {
        let stored = [SCALE, SHIFTED]
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

/// The uniform parameters of a descriptor, by name.
fn uniforms(kernel: &KernelDescriptor) -> Vec<&ParameterDescriptor> {
    kernel
        .parameters
        .iter()
        .filter(|parameter| parameter.kind == ParameterKind::Uniform)
        .collect()
}

/// The resource layout of every parameter of a descriptor, in order.
fn layouts(kernel: &KernelDescriptor) -> Vec<Option<ResourceLayout>> {
    kernel
        .parameters
        .iter()
        .map(|parameter| parameter.resource_layout)
        .collect()
}

#[test]
fn the_macro_writes_the_uniform_layouts_and_the_record_rebuilds_its_descriptor() {
    for stored in [SCALE, SHIFTED] {
        let record = stored.record();
        let uniform_count = record
            .parameters
            .iter()
            .filter(|parameter| parameter.kind == ParameterKind::Uniform)
            .count();
        assert_eq!(uniform_count, uniforms(stored.kernel).len());
        assert!(uniform_count >= 1);
        assert!(
            record
                .parameters
                .iter()
                .filter(|parameter| parameter.kind == ParameterKind::Uniform)
                .all(|parameter| parameter
                    .layout
                    .as_ref()
                    .is_some_and(|layout| layout.rules == ShaderLayoutRules::Std140V1)),
            "{}",
            stored.name
        );
        // Nothing is left to fill, so nothing is unchecked.
        assert!(record.unchecked_parameters().is_empty(), "{}", stored.name);
        // The record round-trips: its descriptor carries the macro's own resource layouts, parameter by parameter.
        assert_eq!(
            layouts(&record.descriptor()),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
        // And the descriptor of the record passes the cross-check against the stored reflection as it is.
        stored
            .reflection()
            .cross_check_kernel(&record.descriptor())
            .unwrap();
    }
}

#[test]
fn a_uniform_kernel_verifies_as_a_program_verifies_it() {
    for stored in [SCALE, SHIFTED] {
        assert_eq!(
            stored.verify(&stored.record()).unwrap(),
            crate::slang::SLANG_VERSION
        );
        // A record with null layouts verifies too: they are filled from the reflection.
        assert_eq!(
            stored.verify(&stored.null_record()).unwrap(),
            crate::slang::SLANG_VERSION
        );
    }
}

#[test]
fn uniform_and_storage_layouts_are_completed_together() {
    let record = SHIFTED.null_record();
    assert_eq!(record.unchecked_parameters(), ["o", "g", "src", "out"]);
    assert_eq!(
        SHIFTED.verify(&record).unwrap(),
        crate::slang::SLANG_VERSION
    );
}

#[test]
fn the_completed_uniform_layouts_are_the_macros_own_std140_layouts() {
    for stored in [SCALE, SHIFTED] {
        let completed = completed_descriptor(&stored.null_record(), stored.reflection).unwrap();
        let from_macro = uniforms(stored.kernel);
        let from_reflection = uniforms(&completed);
        assert_eq!(from_macro.len(), from_reflection.len());
        for (macro_side, filled) in from_macro.into_iter().zip(from_reflection) {
            let layout = filled.resource_layout.unwrap();
            assert_eq!(layout.rules, ShaderLayoutRules::Std140V1);
            assert_eq!(layout.element_stride, layout.element.size);
            // The macro's own descriptor is independent evidence: its offsets are const-asserted against `offset_of!`.
            assert_eq!(
                macro_side.resource_layout.unwrap().element,
                layout.element,
                "{}",
                filled.name
            );
            assert_eq!(macro_side.resource_layout.unwrap(), layout);
        }
        // Every filled layout is what the macro wrote, the storage parameters (`StorageV1`) included.
        assert_eq!(
            layouts(&completed),
            layouts(stored.kernel),
            "{}",
            stored.name
        );
    }
    // Spelled out for the nested one (`o`: 32 bytes, `inner` at 16 inside it) so the numbers are not only compared.
    let completed = completed_descriptor(&SHIFTED.null_record(), SHIFTED.reflection).unwrap();
    let outer = completed.parameters[1].resource_layout.unwrap();
    assert_eq!((outer.element.size, outer.element.alignment), (32, 16));
    assert_eq!(outer.element_stride, 32);
    let gain = completed.parameters[2].resource_layout.unwrap();
    assert_eq!((gain.element.size, gain.element.alignment), (16, 16));
}

#[test]
fn the_plan_of_a_uniform_kernel_is_built_by_the_export_path() {
    let planned = plan(
        vec![SCALE.record(), SHIFTED.record()],
        &[Format::Wgsl],
        &StoredCompiler,
    )
    .unwrap();
    assert_eq!(
        planned.manifest.slang.as_deref(),
        Some(crate::slang::SLANG_VERSION)
    );
    assert_eq!(planned.manifest.kernels.len(), 2);
    let names: Vec<&str> = planned
        .files
        .iter()
        .map(|file| file.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "bfb6e436140611c0.wgsl",
            "bfb6e436140611c0.wgsl.reflection.json",
            "f6dd8e3fe6e29694.wgsl",
            "f6dd8e3fe6e29694.wgsl.reflection.json",
        ]
    );
    // The macro described every layout, so the descriptor-side check ran in full and nothing is unchecked.
    assert!(
        planned
            .manifest
            .kernels
            .iter()
            .all(|entry| entry.export_unchecked.is_empty())
    );
}

#[test]
fn a_null_uniform_is_listed_in_export_unchecked() {
    // The retired exemption (D104): a uniform the macro could not describe is filled from the reflection and, like a
    // storage buffer in that case, named in the manifest, because its layout was not checked against the macro's.
    let mut record = SHIFTED.record();
    for parameter in &mut record.parameters {
        if parameter.name == "o" {
            parameter.layout = None;
        }
    }
    assert_eq!(record.unchecked_parameters(), ["o"]);
    let planned = plan(vec![record], &[Format::Wgsl], &StoredCompiler).unwrap();
    assert_eq!(planned.manifest.kernels.len(), 1);
    assert_eq!(planned.manifest.kernels[0].export_unchecked, ["o"]);
    // Every layout null: all four are listed, in parameter order, the uniforms with the storage buffers.
    let planned = plan(
        vec![SHIFTED.null_record()],
        &[Format::Wgsl],
        &StoredCompiler,
    )
    .unwrap();
    assert_eq!(
        planned.manifest.kernels[0].export_unchecked,
        ["o", "g", "src", "out"]
    );
}

#[test]
fn a_uniform_offset_tampered_in_the_record_is_caught_at_export() {
    let record = SHIFTED.record();
    let genuine = SHIFTED.reflection();
    // Move the nested `inner` (a struct inside the uniform `o`) from 16 to 20, and the `pad0` field inside `inner`
    // from 8 to 20. Neither is where std140 puts it, so the rules refuse the record before the reflection is read.
    for (what, edit) in [
        (
            "the offset of a nested struct",
            tamper_offset as fn(&mut Record, &str, u32) -> String,
        ),
        ("the offset of a scalar inside it", tamper_inner_offset),
    ] {
        let mut tampered = record.clone();
        let target = edit(&mut tampered, "o", 20);
        let error = cross_check_completed(&tampered, &genuine, SHIFTED.reflection).unwrap_err();
        assert!(
            error.contains("host layout is not Std140V1"),
            "{what} ({target}): {error}"
        );
        // And the whole export refuses the record, with the same cause, before anything is written.
        let error = plan(vec![tampered], &[Format::Wgsl], &StoredCompiler).unwrap_err();
        assert!(
            error.contains("host layout is not Std140V1"),
            "{what}: {error}"
        );
    }
}

/// Set the offset of the field named `inner` of the uniform `parameter` to `to`; returns what was changed.
fn tamper_offset(record: &mut Record, parameter: &str, to: u32) -> String {
    let layout = uniform_layout(record, parameter);
    let LayoutKindRecord::Struct(fields) = &mut layout.element.kind else {
        panic!("the uniform element is a struct");
    };
    let field = fields
        .iter_mut()
        .find(|field| field.name == "inner")
        .unwrap();
    let from = field.offset;
    field.offset = to;
    format!("inner {from} -> {to}")
}

/// Set the offset of `pad0` inside the nested `inner` of the uniform `parameter` to `to`.
fn tamper_inner_offset(record: &mut Record, parameter: &str, to: u32) -> String {
    let layout = uniform_layout(record, parameter);
    let LayoutKindRecord::Struct(fields) = &mut layout.element.kind else {
        panic!("the uniform element is a struct");
    };
    let inner = fields
        .iter_mut()
        .find(|field| field.name == "inner")
        .unwrap();
    let LayoutKindRecord::Struct(inner_fields) = &mut inner.ty.kind else {
        panic!("inner is a struct");
    };
    let field = inner_fields
        .iter_mut()
        .find(|field| field.name == "pad0")
        .unwrap();
    let from = field.offset;
    field.offset = to;
    format!("inner.pad0 {from} -> {to}")
}

fn uniform_layout<'a>(record: &'a mut Record, parameter: &str) -> &'a mut ResourceRecord {
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
fn a_uniform_layout_the_reflection_does_not_confirm_is_refused() {
    let record = SHIFTED.record();
    // The stored reflection with one change each; the all-in-one verification judges the changed document alone.
    for (what, from, to, expected) in [
        (
            "an offset that is not where std140 puts it",
            "\"name\":\"pad0\",\"binding\":{\"kind\":\"uniform\",\"offset\":4,",
            "\"name\":\"pad0\",\"binding\":{\"kind\":\"uniform\",\"offset\":8,",
            "o.pad0.offset: host 4, compiler 8",
        ),
        (
            "a constant buffer whose size is not its element's",
            "\"type\":{\"kind\":\"constantBuffer\",\"size\":32,",
            "\"type\":{\"kind\":\"constantBuffer\",\"size\":48,",
            "o.stride: host 32, compiler 48",
        ),
        (
            "an alignment that is not rounded up to 16",
            "\"type\":{\"size\":16,\"alignment\":16,\"stride\":16,\"name\":\"Inner\"",
            "\"type\":{\"size\":16,\"alignment\":4,\"stride\":16,\"name\":\"Inner\"",
            "o.inner.alignment: host 16, compiler 4",
        ),
    ] {
        let tampered = SHIFTED.reflection.replacen(from, to, 1);
        assert_ne!(
            tampered, SHIFTED.reflection,
            "{what}: the edit did not apply"
        );
        let error = verify_stored(&record, Format::Wgsl, SHIFTED.wgsl, &tampered).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
    }
    // A field named differently in the record's layout than in the reflection that is checked.
    let renamed = SHIFTED
        .reflection
        .replacen("\"name\":\"bias\"", "\"name\":\"offset\"", 1);
    assert_ne!(renamed, SHIFTED.reflection);
    let error = verify_stored(&record, Format::Wgsl, SHIFTED.wgsl, &renamed).unwrap_err();
    assert!(error.contains("field name/order differs"), "{error}");
}

#[test]
fn a_std140_valid_difference_from_the_macro_is_now_caught_at_export() {
    // The limitation m6.u2f pinned is gone: with the macro's layout in the record, a Slang side that names a padding
    // field differently (the same offsets and sizes, so still valid std140) fails the export-time check, where it
    // used to pass the export and fail only in a program. `null_record` shows the old behaviour for contrast.
    for stored in [SCALE, SHIFTED] {
        let renamed = stored
            .reflection
            .replacen("\"name\":\"pad0\"", "\"name\":\"padx\"", 1);
        assert_ne!(renamed, stored.reflection, "{}", stored.name);
        let error =
            verify_stored(&stored.record(), Format::Wgsl, stored.wgsl, &renamed).unwrap_err();
        assert!(
            error.contains("field name/order differs"),
            "{}: {error}",
            stored.name
        );
        // A null layout is filled from the document itself, so that check cannot see it (and says so in the manifest).
        assert_eq!(
            verify_stored(&stored.null_record(), Format::Wgsl, stored.wgsl, &renamed).unwrap(),
            crate::slang::SLANG_VERSION,
            "{}",
            stored.name
        );
        // D17 still holds at run time for exactly that case (`export_unchecked`): a program that loads the artifact
        // checks it against the macro's own Std140V1 descriptor, a real `#[gpu]` kernel descriptor, and refuses it by
        // the field name.
        let error = verify_artifact(
            stored.kernel,
            &Artifact {
                format: Format::Wgsl,
                key: stored.key,
                bytes: stored.wgsl,
                reflection: leak_str(&renamed),
            },
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("field name/order differs"),
            "{}: {error}",
            stored.name
        );
    }
}

#[test]
fn a_uniform_layout_declared_with_storage_rules_is_refused() {
    // The record's rules are the descriptor's rules: `StorageV1` does not fit a constant buffer.
    let mut record = SCALE.record();
    let layout = uniform_layout(&mut record, "g");
    assert_eq!(layout.rules, ShaderLayoutRules::Std140V1);
    assert_eq!(layout.element_stride, 16);
    layout.rules = ShaderLayoutRules::StorageV1;
    let error = SCALE.verify(&record).unwrap_err();
    assert!(
        error.contains("g.rules: StorageV1 does not fit this parameter kind"),
        "{error}"
    );
}

#[test]
fn a_reflection_that_does_not_report_a_constant_buffer_for_a_uniform_is_refused_by_name() {
    // The uniform `g` is renamed to the storage buffer `src`, which the reflection reports as a structured buffer.
    let mut record = SCALE.null_record();
    record.parameters[1].name = "src".to_owned();
    let error = SCALE.verify(&record).unwrap_err();
    assert!(
        error.contains("probes::scale")
            && error.contains("`src`")
            && error.contains("not a constantBuffer"),
        "{error}"
    );
    record.parameters[1].name = "missing".to_owned();
    let error = SCALE.verify(&record).unwrap_err();
    assert!(error.contains("no resource named `missing`"), "{error}");
}

#[test]
#[ignore = "needs the native helper: cargo test -p cargo-gust -- --ignored the_helper_builds"]
fn the_helper_builds_the_stored_artifacts() {
    // The real `cargo gust build` compile step (`HelperCompiler`, which runs the helper and the whole cross-check and
    // verification) on the records the macro writes, and the stored fixtures are what it writes today.
    let compiler = HelperCompiler {
        helper: gust::reflect::native_compiler_path(),
        spirv_val: None,
    };
    for stored in [SCALE, SHIFTED] {
        let record = stored.record();
        let compiled = compiler
            .compile(&record, &record.descriptor(), Format::Wgsl)
            .unwrap();
        assert_eq!(compiled.bytes, stored.wgsl, "{}", stored.name);
        assert_eq!(
            compiled.reflection_json, stored.reflection,
            "{}",
            stored.name
        );
        assert_eq!(compiled.compiler, crate::slang::SLANG_VERSION);
    }
    // And the artifacts verify in the macro's own descriptor, which carries the std140 layouts.
    for stored in [SCALE, SHIFTED] {
        verify_artifact(
            stored.kernel,
            &Artifact {
                format: Format::Wgsl,
                key: stored.key,
                bytes: stored.wgsl,
                reflection: stored.reflection,
            },
        )
        .unwrap();
    }
}
