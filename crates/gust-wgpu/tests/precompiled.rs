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

// ---- stored arrays through the precompiled path (m9.pa) ----
//
// A buffer element that is, or holds, a fixed-size array. The artifacts are `cargo gust build` of this module on a
// scratch copy, as for `vector_add`. Before m9.pa the build refused them ("unsupported element type `array`").

#[gpu]
mod pre_arrays {
    pub struct Bag {
        pub items: [u32; 4],
        pub count: u32,
    }

    pub struct Pt {
        pub x: u32,
        pub y: f32,
    }

    pub struct Poly {
        pub pts: [Pt; 2],
        pub tip: [Vec2; 2],
        pub n: u32,
        pub pad: u32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn element(id: SV_DispatchThreadID, mut quads: RWStructuredBuffer<[u32; 4]>) {
        let i = id.x;
        if i < quads.len() {
            let q = quads[i];
            quads[i][0] = q[3] + 100u32;
            quads[i][3] = q[0] + q[1] + q[2];
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn field(id: SV_DispatchThreadID, mut bags: RWStructuredBuffer<Bag>) {
        let i = id.x;
        if i < bags.len() {
            bags[i].items[1] += 7u32;
            bags[i].count = bags[i].items.len() as u32 + bags[i].items[0];
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn shapes(
        id: SV_DispatchThreadID,
        mut pairs: RWStructuredBuffer<[Vec2; 2]>,
        mut polys: RWStructuredBuffer<Poly>,
    ) {
        let i = id.x;
        if i < pairs.len() {
            let p = pairs[i];
            pairs[i][0] = p[1];
            pairs[i][1] = p[0];
            polys[i].pts[1].x = polys[i].pts[0].x + 1u32;
            polys[i].tip[0] = polys[i].tip[1];
            polys[i].n = polys[i].pts[1].x;
        }
    }
}

const ELEMENT_KEY: &str = "54bb2958cc5fe568";
const FIELD_KEY: &str = "794f2959e8738056";
const SHAPES_KEY: &str = "baf2462e890b0200";

const ELEMENT: KernelDescriptor = pre_arrays::element::DESCRIPTOR.with_artifacts(&[Artifact {
    format: Format::Wgsl,
    key: ELEMENT_KEY,
    bytes: include_bytes!("fixtures/precompiled/54bb2958cc5fe568.wgsl"),
    reflection: include_str!("fixtures/precompiled/54bb2958cc5fe568.wgsl.reflection.json"),
}]);
const FIELD: KernelDescriptor = pre_arrays::field::DESCRIPTOR.with_artifacts(&[Artifact {
    format: Format::Wgsl,
    key: FIELD_KEY,
    bytes: include_bytes!("fixtures/precompiled/794f2959e8738056.wgsl"),
    reflection: include_str!("fixtures/precompiled/794f2959e8738056.wgsl.reflection.json"),
}]);
const SHAPES: KernelDescriptor = pre_arrays::shapes::DESCRIPTOR.with_artifacts(&[Artifact {
    format: Format::Wgsl,
    key: SHAPES_KEY,
    bytes: include_bytes!("fixtures/precompiled/baf2462e890b0200.wgsl"),
    reflection: include_str!("fixtures/precompiled/baf2462e890b0200.wgsl.reflection.json"),
}]);

/// More than one workgroup of 64, so the second workgroup is checked too.
const ARRAY_ROWS: u32 = 70;

#[test]
fn the_array_fixtures_are_current_and_verify_without_a_process() {
    for (kernel, key) in [
        (&ELEMENT, ELEMENT_KEY),
        (&FIELD, FIELD_KEY),
        (&SHAPES, SHAPES_KEY),
    ] {
        assert_eq!(
            format!(
                "{:016x}",
                gust::artifact::content_key(kernel.entry_point, kernel.slang_source)
            ),
            key,
            "the kernel changed: run `cargo gust build` on a scratch copy of the module and replace its fixtures"
        );
        gust::artifact::verify_artifact(kernel, &kernel.artifacts[0])
            .unwrap_or_else(|error| panic!("{}: {error:?}", kernel.entry_point));
    }
}

#[test]
fn a_buffer_of_arrays_runs_from_its_precompiled_artifact() {
    let device = device();
    let rows: Vec<[u32; 4]> = (0..ARRAY_ROWS).map(|i| [i, i + 1, i + 2, i + 3]).collect();
    let quads = device.create_typed_buffer("quads", &rows, GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(&ELEMENT, ARRAY_ROWS, &[BufferBinding::read_write(&quads)])
        .expect("the verified artifact of an array element runs");
    let out: Vec<[u32; 4]> = device.read_typed_buffer(&quads).unwrap();
    // Written by hand: element 0 becomes q[3] + 100 = i + 103, element 3 the sum of the first three = 3 i + 3.
    assert_eq!(out[0], [103, 1, 2, 3]);
    assert_eq!(out[1], [104, 2, 3, 6]);
    assert_eq!(out[69], [172, 70, 71, 210]);
    for (i, row) in out.iter().enumerate() {
        let i = i as u32;
        assert_eq!(*row, [i + 103, i + 1, i + 2, 3 * i + 3], "row {i}");
    }
}

#[test]
fn a_struct_with_an_array_field_runs_from_its_precompiled_artifact() {
    use pre_arrays::Bag;
    let device = device();
    let bags: Vec<Bag> = (0..ARRAY_ROWS)
        .map(|i| Bag {
            items: [i, 2 * i, 3 * i, 4 * i],
            count: 0,
        })
        .collect();
    let buffer = device.create_typed_buffer("bags", &bags, GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(&FIELD, ARRAY_ROWS, &[BufferBinding::read_write(&buffer)])
        .expect("the verified artifact of an array field runs");
    let out: Vec<Bag> = device.read_typed_buffer(&buffer).unwrap();
    // Written by hand: items[1] gains 7 (2 i + 7); count is the array length 4 plus items[0] = i + 4.
    assert_eq!((out[0].items, out[0].count), ([0, 7, 0, 0], 4));
    assert_eq!((out[1].items, out[1].count), ([1, 9, 3, 4], 5));
    assert_eq!((out[69].items, out[69].count), ([69, 145, 207, 276], 73));
    for (i, bag) in out.iter().enumerate() {
        let i = i as u32;
        assert_eq!(bag.items, [i, 2 * i + 7, 3 * i, 4 * i], "row {i}");
        assert_eq!(bag.count, i + 4, "row {i}");
    }
}

#[test]
fn arrays_of_vectors_and_of_structs_run_from_their_precompiled_artifact() {
    use gust::math::Vec2;
    use pre_arrays::{Poly, Pt};
    let device = device();
    let pairs: Vec<[Vec2; 2]> = (0..ARRAY_ROWS)
        .map(|i| [Vec2::new(i as f32, 1.0), Vec2::new(2.0, i as f32 + 0.5)])
        .collect();
    let polys: Vec<Poly> = (0..ARRAY_ROWS)
        .map(|i| Poly {
            pts: [Pt { x: i, y: 1.5 }, Pt { x: 0, y: 2.5 }],
            tip: [Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)],
            n: 0,
            pad: 9,
        })
        .collect();
    let g_pairs = device.create_typed_buffer("pairs", &pairs, GpuBufferAccess::ReadWrite);
    let g_polys = device.create_typed_buffer("polys", &polys, GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &SHAPES,
            ARRAY_ROWS,
            &[
                BufferBinding::read_write(&g_pairs),
                BufferBinding::read_write(&g_polys),
            ],
        )
        .expect("the verified artifact of vector and struct arrays runs");
    let pairs: Vec<[Vec2; 2]> = device.read_typed_buffer(&g_pairs).unwrap();
    let polys: Vec<Poly> = device.read_typed_buffer(&g_polys).unwrap();
    // Written by hand: the pair swaps, so row 69 is [(2.0, 69.5), (69.0, 1.0)]; the poly's second point takes the first
    // point's x plus one, its first tip becomes the second, and n is that new x (row 69: x 70, tip (3, 4), n 70).
    let show = |pair: &[Vec2; 2]| ((pair[0].x, pair[0].y), (pair[1].x, pair[1].y));
    assert_eq!(show(&pairs[0]), ((2.0, 0.5), (0.0, 1.0)));
    assert_eq!(show(&pairs[69]), ((2.0, 69.5), (69.0, 1.0)));
    assert_eq!((polys[69].pts[1].x, polys[69].pts[1].y), (70, 2.5));
    assert_eq!((polys[69].tip[0].x, polys[69].tip[0].y), (3.0, 4.0));
    assert_eq!((polys[69].n, polys[69].pad), (70, 9));
    for i in 0..ARRAY_ROWS as usize {
        let x = i as u32;
        assert_eq!(
            show(&pairs[i]),
            ((2.0, x as f32 + 0.5), (x as f32, 1.0)),
            "pair {i}"
        );
        let poly = &polys[i];
        assert_eq!((poly.pts[0].x, poly.pts[0].y), (x, 1.5), "poly {i}");
        assert_eq!((poly.pts[1].x, poly.pts[1].y), (x + 1, 2.5), "poly {i}");
        for tip in &poly.tip {
            assert_eq!((tip.x, tip.y), (3.0, 4.0), "poly {i}");
        }
        assert_eq!((poly.n, poly.pad), (x + 1, 9), "poly {i}");
    }
}
