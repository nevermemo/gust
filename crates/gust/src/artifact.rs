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
#[non_exhaustive]
pub enum Format {
    /// WGSL text, the shader language of the WebGPU API.
    Wgsl,
    /// SPIR-V words stored as little-endian bytes.
    Spirv,
}

impl Format {
    /// Every format this release has, in the order of the enum. The enum is `#[non_exhaustive]`, so a test outside this
    /// crate can walk this list to learn that a later release added a format its `match` does not know.
    pub const ALL: &'static [Self] = &[Self::Wgsl, Self::Spirv];

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
    /// The compiler output format of `bytes`.
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
/// The stage is the descriptor's: `kernel.shape` says compute or fragment, and the stored reflection has to say the
/// same. Under schema 3 (mx.t0b, D91) the stored document also carries the texture and sampler bindings and a
/// fragment's targets, and the descriptor's textures, samplers and outputs are compared with them in the same call;
/// nothing else about the artifact changes, since the identity fields and the key formula are the same for every
/// stage (the stage is in the Slang text, so one body under two shapes has two keys, D89).
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
    fn format_all_lists_every_variant_once() {
        // Inside the defining crate `#[non_exhaustive]` does not apply, so this `match` has no wildcard arm and stops
        // compiling when a variant is added: add it to `Format::ALL` then.
        let index = |format: Format| match format {
            Format::Wgsl => 0,
            Format::Spirv => 1,
        };
        assert_eq!(Format::ALL.len(), 2);
        for (position, format) in Format::ALL.iter().enumerate() {
            assert_eq!(index(*format), position);
        }
    }

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

    /// Entry point, Slang text and key, in the order the macro asserts them (`GOLDEN_KEYS` there).
    const GOLDEN_KEYS: [(&str, &str, &str); 7] = [
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
    ];

    /// The macro (`gust-macros/src/expand/artifact.rs`) cannot call [`content_key`], so it carries a copy of the
    /// definition. Both crates assert these same pairs and strings; they were computed independently of either
    /// implementation (a node script over the UTF-8 bytes). A key that differs between the two would name files the
    /// other side never looks for.
    #[test]
    fn content_key_matches_the_golden_values_the_macro_asserts_too() {
        for (entry_point, text, expected) in GOLDEN_KEYS {
            assert_eq!(
                format!("{:016x}", content_key(entry_point, text)),
                expected,
                "{entry_point:?}"
            );
        }
    }

    /// The key formula is unchanged by shapes (D89): the stage is in the Slang text, so the same entry point and body
    /// under two shapes are two keys, and `KernelDescriptor::shape` is not an input of the key. The pair is the last
    /// two rows of the table.
    #[test]
    fn one_body_under_two_shapes_gives_two_keys() {
        let [
            ..,
            (grid_entry, grid, grid_key),
            (pixel_entry, pixel, pixel_key),
        ] = GOLDEN_KEYS;
        assert_eq!(grid_entry, pixel_entry);
        assert!(grid.contains("[shader(\"compute\")]") && pixel.contains("[shader(\"fragment\")]"));
        assert_ne!(grid_key, pixel_key);
        assert_eq!(format!("{:016x}", content_key(grid_entry, grid)), grid_key);
        assert_eq!(
            format!("{:016x}", content_key(pixel_entry, pixel)),
            pixel_key
        );
        assert_ne!(
            content_key(grid_entry, grid),
            content_key(pixel_entry, pixel)
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

    /// The stage of the stored reflection has to be the descriptor's: a compute artifact does not verify for a pixel
    /// kernel, and the message names the field (D91). The control is the same artifact against the grid descriptor.
    #[test]
    fn a_compute_artifact_does_not_verify_for_a_pixel_descriptor() {
        const ARTIFACT: Artifact = Artifact {
            format: Format::Wgsl,
            key: "52e85c5f03f72272",
            bytes: include_bytes!("../tests/fixtures/precompiled/52e85c5f03f72272.wgsl"),
            reflection: include_str!(
                "../tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json"
            ),
        };
        let grid = vector_add::add::DESCRIPTOR;
        verify_artifact(&grid, &ARTIFACT).expect("the control verifies");
        let pixel = KernelDescriptor {
            shape: crate::Shape::Pixel,
            ..grid
        };
        match verify_artifact(&pixel, &ARTIFACT) {
            Err(Error::Mismatch(message)) => {
                assert_eq!(message, "stage: host fragment, compiler compute")
            }
            other => panic!("expected a stage mismatch, got {other:?}"),
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
