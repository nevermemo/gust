//! Uniform kernels through the export path (slice m6.u2f).
//!
//! The macro writes `layout: null` for a `Uniform<T>` parameter (the record schema has no uniform layout), so the
//! export completes it from the reflection's constant buffer as `Std140V1` ([`completed_descriptor`]). These tests use
//! the WGSL and the reflection JSON the native helper wrote for two uniform kernels (`testdata/uniform/`, made by
//! `cargo gust build` on a crate with the same module) and the descriptor `#[gust::gpu]` makes of the same kernels, so
//! the cross-check that runs here is gust's own and no helper is needed. `the_helper_builds_the_stored_artifacts`
//! needs the helper and checks that the stored bytes are what it writes today.

use gust::{ParameterDescriptor, ScalarKind, artifact::verify_artifact};

use super::*;
use crate::record::{LayoutKindRecord, LayoutRecord, ParameterRecord, ResourceRecord};

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
const SCALE_WGSL: &[u8] = include_bytes!("testdata/uniform/bfb6e436140611c0.wgsl");
const SCALE_REFLECTION: &str =
    include_str!("testdata/uniform/bfb6e436140611c0.wgsl.reflection.json");
const SHIFTED_KEY: &str = "f6dd8e3fe6e29694";
const SHIFTED_WGSL: &[u8] = include_bytes!("testdata/uniform/f6dd8e3fe6e29694.wgsl");
const SHIFTED_REFLECTION: &str =
    include_str!("testdata/uniform/f6dd8e3fe6e29694.wgsl.reflection.json");

/// One stored kernel: its descriptor from the macro, the key it must have and the helper's bytes for it.
struct Stored {
    kernel: &'static KernelDescriptor,
    key: &'static str,
    name: &'static str,
    wgsl: &'static [u8],
    reflection: &'static str,
}

const SCALE: Stored = Stored {
    kernel: &probes::scale::DESCRIPTOR,
    key: SCALE_KEY,
    name: "scale",
    wgsl: SCALE_WGSL,
    reflection: SCALE_REFLECTION,
};
const SHIFTED: Stored = Stored {
    kernel: &probes::shifted::DESCRIPTOR,
    key: SHIFTED_KEY,
    name: "shifted",
    wgsl: SHIFTED_WGSL,
    reflection: SHIFTED_REFLECTION,
};

fn f32_layout() -> ResourceRecord {
    ResourceRecord {
        element_stride: 4,
        element: LayoutRecord {
            name: "f32".to_owned(),
            size: 4,
            alignment: 4,
            stride: 4,
            kind: LayoutKindRecord::Scalar(ScalarKind::F32),
        },
    }
}

impl Stored {
    /// The record as the macro writes it: every storage parameter has its f32 layout, every uniform `layout: null`.
    fn record(&self) -> Record {
        assert_eq!(
            format!(
                "{:016x}",
                gust::artifact::content_key(self.kernel.entry_point, self.kernel.slang_source)
            ),
            self.key,
            "the macro's output drifted from the stored artifact"
        );
        Record {
            key: self.key.to_owned(),
            module: "probes".to_owned(),
            name: self.name.to_owned(),
            entry_point: self.kernel.entry_point.to_owned(),
            workgroup_size: [64, 1, 1],
            parameters: self
                .kernel
                .parameters
                .iter()
                .map(|parameter| ParameterRecord {
                    name: parameter.name.to_owned(),
                    rust_type: parameter.rust_type.to_owned(),
                    kind: parameter.kind,
                    access: parameter.access,
                    binding: parameter
                        .binding
                        .map(|binding| (binding.group, binding.binding)),
                    layout: (parameter.kind == ParameterKind::Storage).then(f32_layout),
                })
                .collect(),
            file: String::new(),
            lines: Vec::new(),
            slang: self.kernel.slang_source.to_owned(),
        }
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

#[test]
fn the_macro_leaves_the_uniform_layouts_null_and_the_stored_artifacts_are_the_macros_kernels() {
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
                .all(|parameter| parameter.layout.is_none()),
        );
        // Without the completion the descriptor has no element layout for them: the failure of `cargo gust build`
        // on main, as the runtime's own cross-check words it.
        let raw = stored
            .reflection()
            .cross_check_kernel(&record.descriptor())
            .unwrap_err();
        assert!(
            raw.to_string()
                .contains("missing descriptor element layout"),
            "{raw}"
        );
    }
}

#[test]
fn a_uniform_kernel_verifies_as_a_program_verifies_it() {
    for stored in [SCALE, SHIFTED] {
        assert_eq!(stored.verify(&stored.record()).unwrap(), "2026.20");
    }
}

#[test]
fn uniform_and_storage_layouts_are_completed_together() {
    let mut record = SHIFTED.record();
    for parameter in &mut record.parameters {
        parameter.layout = None;
    }
    assert_eq!(record.unchecked_parameters(), ["src", "out"]);
    assert_eq!(SHIFTED.verify(&record).unwrap(), "2026.20");
}

#[test]
fn the_completed_uniform_layouts_are_the_macros_own_std140_layouts() {
    for stored in [SCALE, SHIFTED] {
        let completed = completed_descriptor(&stored.record(), stored.reflection).unwrap();
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
        // The storage parameters keep their `StorageV1` layouts.
        for parameter in completed
            .parameters
            .iter()
            .filter(|parameter| parameter.kind == ParameterKind::Storage)
        {
            assert_eq!(
                parameter.resource_layout.unwrap().rules,
                ShaderLayoutRules::StorageV1
            );
        }
    }
    // Spelled out for the nested one (`o`: 32 bytes, `inner` at 16 inside it) so the numbers are not only compared.
    let completed = completed_descriptor(&SHIFTED.record(), SHIFTED.reflection).unwrap();
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
    assert_eq!(planned.manifest.slang.as_deref(), Some("2026.20"));
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
    // The macro lists the storage parameters it could not describe; a uniform is filled from the reflection without
    // being listed, because its layout is never in the record (and the run-time check of the real `GpuPod::LAYOUT`
    // still holds, as for the others).
    assert!(
        planned
            .manifest
            .kernels
            .iter()
            .all(|entry| entry.export_unchecked.is_empty())
    );
}

#[test]
fn a_uniform_layout_the_reflection_does_not_confirm_is_refused() {
    let genuine = SHIFTED.reflection();
    let record = SHIFTED.record();
    // The stored reflection with one change each; the all-in-one verification judges the changed document alone.
    for (what, from, to, expected) in [
        (
            "an offset that is not where std140 puts it",
            "\"name\":\"pad0\",\"binding\":{\"kind\":\"uniform\",\"offset\":4,",
            "\"name\":\"pad0\",\"binding\":{\"kind\":\"uniform\",\"offset\":8,",
            "host layout is not Std140V1",
        ),
        (
            "a constant buffer whose size is not its element's",
            "\"type\":{\"kind\":\"constantBuffer\",\"size\":32,",
            "\"type\":{\"kind\":\"constantBuffer\",\"size\":48,",
            "o.stride: inconsistent host resource layout",
        ),
        (
            "an alignment that is not rounded up to 16",
            "\"type\":{\"size\":16,\"alignment\":16,\"stride\":16,\"name\":\"Inner\"",
            "\"type\":{\"size\":16,\"alignment\":4,\"stride\":16,\"name\":\"Inner\"",
            "host layout is not Std140V1",
        ),
    ] {
        let tampered = SHIFTED.reflection.replacen(from, to, 1);
        assert_ne!(
            tampered, SHIFTED.reflection,
            "{what}: the edit did not apply"
        );
        let error = verify_stored(&record, Format::Wgsl, SHIFTED.wgsl, &tampered).unwrap_err();
        assert!(error.contains(expected), "{what}: {error}");
        // The export-time cross-check judges a layout filled from the changed document against the genuine
        // reflection, which is what the helper wrote: they disagree, so it never passes.
        let error = cross_check_completed(&record, &genuine, &tampered).unwrap_err();
        assert!(!error.is_empty(), "{what}");
    }
    // A field named differently in the filled layout than in the reflection that is checked.
    let renamed = SHIFTED
        .reflection
        .replacen("\"name\":\"bias\"", "\"name\":\"offset\"", 1);
    assert_ne!(renamed, SHIFTED.reflection);
    let error = cross_check_completed(&record, &genuine, &renamed).unwrap_err();
    assert!(error.contains("field name/order differs"), "{error}");
}

#[test]
fn a_uniform_layout_declared_with_storage_rules_is_refused() {
    // A record that carries a layout for its uniform is not re-declared: the rules `descriptor()` gives it are
    // `StorageV1`, which does not fit a constant buffer. (The macro writes no such layout today.)
    let mut record = SCALE.record();
    let layout = crate::fill::reflected_uniform_layout(SCALE.reflection, "g").unwrap();
    assert_eq!(layout.element_stride, 16);
    record.parameters[1].layout = Some(layout);
    let error = SCALE.verify(&record).unwrap_err();
    assert!(
        error.contains("g.rules: StorageV1 does not fit this parameter kind"),
        "{error}"
    );
}

#[test]
fn a_reflection_that_does_not_report_a_constant_buffer_for_a_uniform_is_refused_by_name() {
    // The uniform `g` is renamed to the storage buffer `src`, which the reflection reports as a structured buffer.
    let mut record = SCALE.record();
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
        assert_eq!(compiled.compiler, "2026.20");
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
