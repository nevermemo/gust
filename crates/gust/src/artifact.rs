//! Precompiled kernel artifacts and their run-time verification (M4, D45).
//!
//! An [`Artifact`] is compiler output (WGSL or SPIR-V) for one kernel together with the
//! raw reflection JSON the native Slang helper wrote when it produced the output. Nothing
//! here starts a process: [`verify_artifact`] re-derives every check from the descriptor
//! and the stored evidence, so a program can run a precompiled kernel on a machine with no
//! Slang installed.
//!
//! # What the identity checks protect
//!
//! The helper compiles the kernel's *clean* Slang text (no `#line` directives) when it
//! exports an artifact, so the stored `sourceHash` is the hash of `KernelDescriptor::slang_source`
//! exactly. [`verify_artifact`] checks the target, the entry point, that source hash and the
//! artifact hash from the stored JSON, then cross-checks the stored layouts against the
//! descriptor ([`crate::reflect::Reflection::cross_check_kernel`]).
//!
//! The hashes and [`content_key`] are FNV-1a 64. They name files and detect accidents:
//! corruption, truncation, a line-ending rewrite, a swapped or stale file, descriptor drift.
//! FNV-1a is not collision-resistant, so none of this protects against an adversary who crafts
//! a colliding artifact, and two different kernels could in principle share a 64-bit key.
//! The export tool (`cargo gust build`, slice m4.d) rejects two kernels with one key and different clean text, and
//! its manifest records SHA-256 of the clean text and of each artifact.

use crate::{
    KernelDescriptor,
    reflect::{CompiledReflection, Error, PodCrossCheck},
    slang::Target,
};

/// The compiler output format of an [`Artifact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wgsl,
    Spirv,
}

impl Format {
    /// The compile target this format is the output of.
    pub const fn target(self) -> Target {
        match self {
            Self::Wgsl => Target::Wgsl,
            Self::Spirv => Target::Spirv,
        }
    }
}

impl From<Format> for Target {
    fn from(format: Format) -> Self {
        format.target()
    }
}

/// Compiler output for one kernel plus the helper's raw reflection JSON for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Artifact {
    pub format: Format,
    /// [`content_key`] of the kernel the artifact was compiled from, as 16 lower-case hex
    /// digits: the stem of the artifact's file name. It names the file; the identity check in
    /// [`verify_artifact`] is what ties the bytes to the descriptor.
    pub key: &'static str,
    /// The compiler output, byte for byte as the helper wrote it.
    pub bytes: &'static [u8],
    /// The helper's reflection JSON, byte for byte as the helper wrote it.
    pub reflection: &'static str,
}

/// What [`verify_artifact`] established about an artifact.
#[derive(Debug)]
pub struct Verified {
    /// The artifact bytes with the stored reflection read from the JSON.
    pub compiled: CompiledReflection,
    /// The cross-check report for each storage parameter of the descriptor.
    pub cross_checks: Vec<PodCrossCheck>,
}

/// Verify `artifact` against `kernel` without starting any process: the stored identity
/// (target, entry point, hash of `kernel.slang_source`, hash of the bytes), the workgroup
/// size, and the complete stored layouts against the descriptor's parameters.
///
/// Identity failures are [`Error::Invalid`] and name the value that differed; a descriptor
/// that disagrees with the stored layouts is [`Error::Mismatch`] or [`Error::Incomplete`].
pub fn verify_artifact(kernel: &KernelDescriptor, artifact: &Artifact) -> Result<Verified, Error> {
    let compiled = CompiledReflection::from_output(
        kernel,
        kernel.slang_source,
        artifact.format.target(),
        artifact.bytes.to_vec(),
        artifact.reflection,
    )?;
    let cross_checks = compiled.reflection.cross_check_kernel(kernel)?;
    Ok(Verified {
        compiled,
        cross_checks,
    })
}

/// The key of a kernel's artifacts: FNV-1a 64 over the entry point, one zero byte and the
/// kernel's clean Slang text (`KernelDescriptor::slang_source`). Format it with `{:016x}`.
pub fn content_key(entry_point: &str, slang_source: &str) -> u64 {
    fnv1a64(&[entry_point.as_bytes(), &[0], slang_source.as_bytes()])
}

/// FNV-1a 64 over the concatenation of `parts`.
pub(crate) fn fnv1a64(parts: &[&[u8]]) -> u64 {
    parts
        .iter()
        .flat_map(|part| part.iter())
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_the_published_vectors() {
        assert_eq!(fnv1a64(&[]), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(&[b"a"]), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(&[b"foobar"]), 0x85944171f73967e8);
        assert_eq!(fnv1a64(&[b"foo", b"bar"]), 0x85944171f73967e8);
    }

    #[test]
    fn content_key_separates_the_entry_point_from_the_source_with_a_zero_byte() {
        // Computed independently (a node script over the same bytes), not with this code.
        assert_eq!(content_key("foo", "bar"), 0x8a5f40d8243888aa);
        assert_eq!(content_key("", ""), 0xaf63bd4c8601b7df);
        assert_ne!(content_key("ab", "c"), content_key("a", "bc"));
        assert_ne!(content_key("foo", "bar"), content_key("foobar", ""));
    }

    /// The macro (`gust-macros/src/expand/artifact.rs`) cannot call [`content_key`], so it carries a copy of the
    /// definition. Both crates assert these same pairs and strings; they were computed independently of either
    /// implementation (a node script over the UTF-8 bytes). A key that differs between the two would name files the
    /// other side never looks for.
    #[test]
    fn content_key_matches_the_golden_values_the_macro_asserts_too() {
        for (entry_point, text, expected) in [
            ("foo", "bar", "8a5f40d8243888aa"),
            ("", "", "af63bd4c8601b7df"),
            (
                "gpu_vector_add_add",
                "// Generated by GUST from Rust syntax. Slang is the compiler backend.\n\n[shader(\"compute\")]\nvoid gpu_vector_add_add() {}\n",
                "6cdc27171dd8a55e",
            ),
            // CRLF is hashed as given: no line-ending normalisation.
            ("gpu_demo_run", "uint a;\r\nuint b;\r\n", "0513779dad82ba6c"),
            (
                "gpu_\u{fc}n\u{ef}_\u{e9}",
                "// \u{fc}n\u{ef} \u{65e5}\u{672c} \u{1f600}\n",
                "f3edb671848d99cc",
            ),
            // One body under two shapes (mx.k, D89): the same entry point and body line, the stage attribute and
            // the signature differ, and so do the keys. The formula has no shape in it.
            (
                "gpu_demo_run",
                "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_demo_run(uint3 id : SV_DispatchThreadID)\n{\n    uint i = id.x;\n}\n",
                "4be1a569519007e0",
            ),
            (
                "gpu_demo_run",
                "[shader(\"fragment\")]\nOut gpu_demo_run(float4 pos : SV_Position)\n{\n    uint i = id.x;\n}\n",
                "76d7388ee1e8f383",
            ),
        ] {
            assert_eq!(
                format!("{:016x}", content_key(entry_point, text)),
                expected,
                "{entry_point:?}"
            );
        }
    }

    /// The key formula is unchanged by shapes (D89): the stage is in the Slang text, so the same entry point and body
    /// under two shapes are two keys, and `KernelDescriptor::shape` is not an input of the key.
    #[test]
    fn one_body_under_two_shapes_gives_two_keys() {
        let entry = "gpu_demo_run";
        let grid = "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid gpu_demo_run(uint3 id : SV_DispatchThreadID)\n{\n    uint i = id.x;\n}\n";
        let pixel = "[shader(\"fragment\")]\nOut gpu_demo_run(float4 pos : SV_Position)\n{\n    uint i = id.x;\n}\n";
        assert_ne!(content_key(entry, grid), content_key(entry, pixel));
        assert_eq!(
            format!("{:016x}", content_key(entry, grid)),
            "4be1a569519007e0"
        );
        assert_eq!(
            format!("{:016x}", content_key(entry, pixel)),
            "76d7388ee1e8f383"
        );
    }

    #[test]
    fn format_converts_to_its_compile_target() {
        assert_eq!(Target::from(Format::Wgsl), Target::Wgsl);
        assert_eq!(Format::Spirv.target(), Target::Spirv);
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

    /// The compile copy carries `#line` directives, the stored artifact was compiled from the clean
    /// text: a stored `sourceHash` of the copy must be rejected (the clean-text rule is load-bearing).
    #[test]
    fn a_source_hash_of_the_directive_copy_is_rejected() {
        const ARTIFACT: Artifact = Artifact {
            format: Format::Wgsl,
            key: "52e85c5f03f72272",
            bytes: include_bytes!("../tests/fixtures/precompiled/52e85c5f03f72272.wgsl"),
            reflection: include_str!(
                "../tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json"
            ),
        };
        let kernel = vector_add::add::DESCRIPTOR;
        verify_artifact(&kernel, &ARTIFACT).expect("the clean text verifies");
        let copy = crate::slang::compile_copy(kernel.slang_source, &kernel.source);
        assert_ne!(copy, kernel.slang_source, "the copy must carry directives");
        let clean_hash = format!("{:016x}", fnv1a64(&[kernel.slang_source.as_bytes()]));
        let copy_hash = format!("{:016x}", fnv1a64(&[copy.as_bytes()]));
        assert_eq!(ARTIFACT.reflection.matches(&clean_hash).count(), 1);
        let replaced = Artifact {
            reflection: Box::leak(
                ARTIFACT
                    .reflection
                    .replace(&clean_hash, &copy_hash)
                    .into_boxed_str(),
            ),
            ..ARTIFACT
        };
        match verify_artifact(&kernel, &replaced) {
            Err(Error::Invalid(message)) => assert!(
                message.contains(&format!(
                    "source hash (stored {copy_hash:?}, expected {clean_hash:?})"
                )),
                "{message}"
            ),
            other => panic!("expected an identity mismatch, got {other:?}"),
        }
    }
}
