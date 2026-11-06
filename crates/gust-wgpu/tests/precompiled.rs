//! The precompiled path of `gust-wgpu` (M4, slice m4.e1, D45): a kernel whose descriptor carries a WGSL artifact runs
//! from it, after `gust::artifact::verify_artifact` (D17), and never falls back to compiling the Slang.
//!
//! The artifact is a real compiler product, `cargo gust build` of the `vector_add::add` kernel below (the kernel crate
//! was a scratch copy outside the repository; the files are byte-identical to what the tool wrote, and
//! `.gitattributes` keeps git from rewriting them). The kernel's macro-generated descriptor has no artifact (this crate
//! has no `gust-artifacts/` directory), so [`PRE`] is that descriptor with the artifact added by struct update syntax.
//! The macro's own lookup is exercised by the fixture workspace of m4.e2.
//!
//! These tests run with the `slang-runtime` feature on (the self dev-dependency of the crate), so a compile at run
//! time would succeed for every kernel here. That is what makes the mutated-artifact tests meaningful: a kernel that
//! fell back to the compiler would pass them. Regenerate the fixture with `cargo gust build` on a copy of the kernel
//! when `fixture_is_current` fails.
//!
//! Every test runs on the GPU and fails without an adapter, like the rest of the Gpu suite.

use gust::{Artifact, Format, KernelDescriptor, gpu, reflect::Error as ReflectionError};
use gust_wgpu::{BufferBinding, Error, GpuBufferAccess, HeadlessDevice};

#[gpu]
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

const KEY: &str = "52e85c5f03f72272";
const WGSL: &[u8] = include_bytes!("fixtures/precompiled/52e85c5f03f72272.wgsl");
const REFLECTION: &str = include_str!("fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json");

const ARTIFACT: Artifact = Artifact {
    format: Format::Wgsl,
    key: KEY,
    bytes: WGSL,
    reflection: REFLECTION,
};

/// The macro's descriptor of the kernel plus the artifact.
const PRE: KernelDescriptor = vector_add::add::DESCRIPTOR.with_artifacts(&[ARTIFACT]);

const COUNT: usize = 1000;

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// A descriptor equal to [`PRE`] in every field but the artifact.
fn with_artifact(artifact: Artifact) -> KernelDescriptor {
    PRE.with_artifacts(Box::leak(Box::new([artifact])))
}

fn leaked_bytes(bytes: Vec<u8>) -> &'static [u8] {
    Box::leak(bytes.into_boxed_slice())
}

fn leaked_str(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

struct Inputs {
    a: Vec<f32>,
    b: Vec<f32>,
    expected: Vec<f32>,
}

/// Inputs whose sums are exact in f32, and the host reference.
fn inputs() -> Inputs {
    let a: Vec<f32> = (0..COUNT).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..COUNT).map(|i| 100.0 - i as f32 * 0.25).collect();
    let expected = a.iter().zip(&b).map(|(a, b)| a + b).collect();
    Inputs { a, b, expected }
}

/// Run `kernel` on `device` over [`inputs`] and return the output buffer.
fn run(device: &HeadlessDevice, kernel: &KernelDescriptor) -> Result<Vec<f32>, Error> {
    let inputs = inputs();
    let a = device.create_typed_buffer("a", &inputs.a, GpuBufferAccess::ReadOnly);
    let b = device.create_typed_buffer("b", &inputs.b, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &[0.0_f32; COUNT], GpuBufferAccess::ReadWrite);
    device.dispatch_buffers(
        kernel,
        COUNT as u32,
        &[
            BufferBinding::read_only(&a),
            BufferBinding::read_only(&b),
            BufferBinding::read_write(&out),
        ],
    )?;
    device.read_typed_buffer(&out)
}

fn stats(device: &HeadlessDevice) -> (usize, u64, u64) {
    let stats = device.pipeline_cache_stats();
    (stats.entries, stats.misses, stats.hits)
}

fn identity_message(error: Error) -> String {
    match error {
        Error::Reflection(ReflectionError::Invalid(message)) => message,
        other => panic!("expected the identity error (Reflection(Invalid)), got {other:?}"),
    }
}

#[test]
fn fixture_is_current() {
    // The artifact belongs to the kernel above. If the kernel's Slang changes, the key changes and this fails with the
    // way to regenerate the fixture, before any other test fails with an identity error.
    assert!(
        vector_add::add::DESCRIPTOR.artifacts.is_empty(),
        "this crate has no gust-artifacts/ directory: the descriptor must not carry an artifact of its own"
    );
    let key = gust::artifact::content_key(PRE.entry_point, PRE.slang_source);
    assert_eq!(
        format!("{key:016x}"),
        KEY,
        "the kernel changed: run `cargo gust build` on a scratch copy of it and replace the files under tests/fixtures/precompiled/"
    );
    gust::artifact::verify_artifact(&PRE, &ARTIFACT)
        .expect("the committed artifact verifies against the descriptor without any process");
}

#[test]
fn precompiled_path_dispatches_and_matches_the_host_reference() {
    let device = device();
    let out = run(&device, &PRE).expect("the verified artifact runs");
    let expected = inputs().expected;
    assert_eq!(
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "GPU output of the precompiled kernel versus the host reference"
    );
    assert_eq!(stats(&device), (1, 1, 0));

    // The second dispatch of the same descriptor is a cache hit, not a second verification.
    assert_eq!(run(&device, &PRE).unwrap(), expected);
    assert_eq!(stats(&device), (1, 1, 1));
}

#[test]
fn a_mutated_artifact_is_the_identity_error_and_never_a_fallback_compile() {
    let device = device();
    // The run-time compiler is available in this build, so a fallback would succeed and fail this test.
    let at = WGSL
        .iter()
        .position(|byte| *byte == b'+')
        .expect("the kernel adds");
    let mut flipped = WGSL.to_vec();
    flipped[at] = b'-';
    let mutated = with_artifact(Artifact {
        bytes: leaked_bytes(flipped),
        ..ARTIFACT
    });
    let message =
        identity_message(run(&device, &mutated).expect_err("a flipped byte must not run"));
    assert!(message.contains("artifact hash"), "{message}");

    // A truncated artifact and an artifact for another source are the same error.
    let truncated = with_artifact(Artifact {
        bytes: &WGSL[..WGSL.len() - 1],
        ..ARTIFACT
    });
    let message = identity_message(run(&device, &truncated).expect_err("a truncated artifact"));
    assert!(message.contains("artifact hash"), "{message}");

    let drifted = PRE.with_slang_source(leaked_str(format!("{}\n// drift\n", PRE.slang_source)));
    let message = identity_message(run(&device, &drifted).expect_err("descriptor drift"));
    assert!(message.contains("source hash"), "{message}");

    // Nothing was compiled or cached for any of them.
    assert_eq!(stats(&device), (0, 0, 0));
}

#[test]
fn a_changed_reflection_is_an_error_not_a_fallback() {
    let device = device();
    // The valid descriptor first, so that the cache holds a pipeline the second request could wrongly be served by.
    assert_eq!(run(&device, &PRE).unwrap(), inputs().expected);
    assert_eq!(stats(&device), (1, 1, 0));
    let other = REFLECTION.replace("\"workgroupSize\":[64,1,1]", "\"workgroupSize\":[32,1,1]");
    assert_ne!(other, REFLECTION);
    let error = run(
        &device,
        &with_artifact(Artifact {
            reflection: leaked_str(other),
            ..ARTIFACT
        }),
    )
    .expect_err("the stored workgroup size disagrees with the descriptor");
    assert!(
        matches!(&error, Error::Reflection(ReflectionError::Mismatch(message)) if message == "compute workgroup dimensions differ"),
        "{error:?}"
    );
    // No cache hit, and entries and misses are unchanged: a request that fails verification creates no entry and is
    // not counted as a compile (reflection.rs pins the same rule for every other compile error).
    assert_eq!(stats(&device), (1, 1, 0));
}

/// FNV-1a 64 over `bytes`, written out here so that the test does not use the code it tests.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn the_gpu_runs_the_bytes_of_the_artifact_not_a_recompile_of_the_slang() {
    let device = device();
    // A verified artifact whose output differs from the kernel's Slang: the `+` of the WGSL becomes `*`, the stored
    // artifact hash is recomputed. If the backend verified the artifact and then compiled the Slang, it would add.
    let text = std::str::from_utf8(WGSL).expect("WGSL is UTF-8");
    assert_eq!(text.matches('+').count(), 1, "the fixture adds once");
    let product = text.replace('+', "*").into_bytes();
    let old_hash = format!("\"artifactHash\":\"{:016x}\"", fnv1a64(WGSL));
    let new_hash = format!("\"artifactHash\":\"{:016x}\"", fnv1a64(&product));
    assert!(REFLECTION.contains(&old_hash), "{old_hash}");
    let kernel = with_artifact(Artifact {
        bytes: leaked_bytes(product),
        reflection: leaked_str(REFLECTION.replace(&old_hash, &new_hash)),
        ..ARTIFACT
    });
    gust::artifact::verify_artifact(&kernel, &kernel.artifacts[0])
        .expect("the multiplying artifact is a verified artifact");

    let inputs = inputs();
    let products: Vec<f32> = inputs.a.iter().zip(&inputs.b).map(|(a, b)| a * b).collect();
    assert_ne!(
        products, inputs.expected,
        "the two kernels must be told apart"
    );
    assert_eq!(run(&device, &kernel).unwrap(), products);
    // The kernel's own Slang still adds.
    assert_eq!(run(&device, &PRE).unwrap(), inputs.expected);
    assert_eq!(stats(&device), (2, 2, 0));
}

#[test]
fn descriptors_with_the_same_fields_and_different_artifacts_get_their_own_verified_pipelines() {
    let device = device();
    let expected = inputs().expected;
    let same_bytes_elsewhere = with_artifact(Artifact {
        bytes: leaked_bytes(WGSL.to_vec()),
        ..ARTIFACT
    });
    let same_reflection_elsewhere = with_artifact(Artifact {
        reflection: leaked_str(REFLECTION.to_string()),
        ..ARTIFACT
    });
    // Equal in every field of the descriptor and in the content of the artifact; the addresses differ.
    for kernel in [&PRE, &same_bytes_elsewhere, &same_reflection_elsewhere] {
        assert_eq!(kernel.slang_source.as_ptr(), PRE.slang_source.as_ptr());
        assert_eq!(kernel.parameters.as_ptr(), PRE.parameters.as_ptr());
        assert_eq!(kernel.entry_point, PRE.entry_point);
        assert_eq!(run(&device, kernel).unwrap(), expected);
    }
    assert_ne!(
        same_bytes_elsewhere.artifacts[0].bytes.as_ptr(),
        PRE.artifacts[0].bytes.as_ptr()
    );
    assert_ne!(
        same_reflection_elsewhere.artifacts[0].reflection.as_ptr(),
        PRE.artifacts[0].reflection.as_ptr()
    );
    // Three artifacts, three verifications, three pipelines, no hit.
    assert_eq!(stats(&device), (3, 3, 0));
    // Each one hits its own entry afterwards.
    for kernel in [&PRE, &same_bytes_elsewhere, &same_reflection_elsewhere] {
        assert_eq!(run(&device, kernel).unwrap(), expected);
    }
    assert_eq!(stats(&device), (3, 3, 3));
}

#[test]
fn an_unverified_artifact_cannot_borrow_the_pipeline_of_a_verified_one() {
    let device = device();
    let expected = inputs().expected;
    assert_eq!(run(&device, &PRE).unwrap(), expected);
    assert_eq!(stats(&device), (1, 1, 0));

    // Same length, same key text, same reflection and same slices in the descriptor: one flipped byte at another
    // address. A cache that ignored the artifact would answer with the valid kernel's pipeline.
    let at = WGSL.iter().position(|byte| *byte == b'+').unwrap();
    let mut flipped = WGSL.to_vec();
    flipped[at] = b'-';
    let mutated = with_artifact(Artifact {
        bytes: leaked_bytes(flipped),
        ..ARTIFACT
    });
    let message = identity_message(run(&device, &mutated).expect_err("not a cache hit"));
    assert!(message.contains("artifact hash"), "{message}");
    // No hit. A request that fails verification creates no entry and is not counted as a compile (the same rule as
    // every other compile error of this crate).
    assert_eq!(stats(&device), (1, 1, 0));

    // The valid descriptor still hits its own entry.
    assert_eq!(run(&device, &PRE).unwrap(), expected);
    assert_eq!(stats(&device), (1, 1, 1));
}

#[test]
fn the_generated_host_snippet_prefers_the_embedded_artifact() {
    let source = gust_wgpu::render_wgpu_source(&PRE).expect("the kernel renders");
    let embedded = source
        .find("gust::artifact::verify_artifact(descriptor, artifact)")
        .expect("the snippet verifies the embedded artifact");
    let compiled = source
        .find("gust::slang::compile_wgsl(descriptor)")
        .expect("the snippet keeps the run-time compile for a kernel without an artifact");
    assert!(
        embedded < compiled,
        "the artifact arm comes before the run-time compile"
    );
    assert!(source.contains("`slang-runtime` feature"));
    assert!(source.contains("cargo gust build"));
}
