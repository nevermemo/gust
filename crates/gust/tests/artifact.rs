//! `gust::artifact::verify_artifact` on a real precompiled artifact (slice m4.a, D45).
//!
//! The fixture under `tests/fixtures/precompiled/` is the vector-add kernel's WGSL and the raw
//! reflection JSON the native Slang helper wrote for it (spike S1, kept verbatim). No Slang and
//! no GPU are needed: the whole path is `verify_artifact`, which starts no process. The kernel
//! is declared below with the macro, so its descriptor (the clean Slang text, the layouts) is
//! the real one; if the macro's output ever drifts from the fixture, the key test fails first.
//!
//! `manifest.json` in the fixture directory is spike S1's interim shape (kernel name and key only), pinned
//! verbatim below; it is not the schema-1 manifest of D45, which the export tool (m4.d) writes.

use gust::{
    Access, KernelDescriptor, ParameterDescriptor, ResourceBinding, ResourceLayout, ScalarKind,
    StructFieldLayout, TypeLayout, TypeLayoutKind,
    artifact::{Artifact, Format, content_key, verify_artifact},
    reflect::Error,
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

const DESCRIPTOR: KernelDescriptor = vector_add::add::DESCRIPTOR;
const KEY: &str = "52e85c5f03f72272";
const WGSL: &[u8] = include_bytes!("fixtures/precompiled/52e85c5f03f72272.wgsl");
const REFLECTION: &str = include_str!("fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json");
const MANIFEST: &str = include_str!("fixtures/precompiled/manifest.json");

const ARTIFACT: Artifact = Artifact {
    format: Format::Wgsl,
    key: KEY,
    bytes: WGSL,
    reflection: REFLECTION,
};

fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
    Box::leak(bytes.into_boxed_slice())
}

fn leak_str(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// The error text of an identity failure, which must be exactly that kind of failure.
fn identity_error(kernel: &KernelDescriptor, artifact: &Artifact) -> String {
    match verify_artifact(kernel, artifact) {
        Err(Error::Invalid(message))
            if message.starts_with(
                "native target, entry point, source or artifact identity mismatch",
            ) =>
        {
            message
        }
        other => panic!("expected an identity mismatch, got {other:?}"),
    }
}

fn mismatch(kernel: &KernelDescriptor, artifact: &Artifact) -> String {
    match verify_artifact(kernel, artifact) {
        Err(Error::Mismatch(message)) => message,
        other => panic!("expected a layout mismatch, got {other:?}"),
    }
}

#[test]
fn the_fixture_verifies_through_the_public_entry() {
    let verified = verify_artifact(&DESCRIPTOR, &ARTIFACT).expect("the real artifact verifies");
    assert_eq!(verified.compiled.entry_point, "gpu_vector_add_add");
    assert_eq!(verified.compiled.workgroup_size, [64, 1, 1]);
    assert_eq!(verified.compiled.artifact, WGSL);
    assert_eq!(verified.compiled.compiler_version, "2026.20");
    let names: Vec<_> = verified
        .compiled
        .reflection
        .resources
        .iter()
        .map(|resource| (resource.name.as_str(), resource.access, resource.binding))
        .collect();
    assert_eq!(
        names,
        [
            ("a", Access::ReadOnly, 0),
            ("b", Access::ReadOnly, 1),
            ("out", Access::ReadWrite, 2),
        ]
    );
    assert_eq!(verified.cross_checks.len(), 3);
    assert!(verified.cross_checks.iter().all(|c| c.is_full_abi_match()));
}

#[test]
fn the_fixture_is_the_kernels_artifact_by_key_and_manifest() {
    assert_eq!(
        format!(
            "{:016x}",
            content_key(DESCRIPTOR.entry_point, DESCRIPTOR.slang_source)
        ),
        KEY
    );
    assert_eq!(
        MANIFEST.trim_end(),
        format!(r#"{{"artifacts":[{{"kernel":"vector_add::add","key":"{KEY}"}}]}}"#)
    );
}

/// Git must hand the fixture back byte for byte (`.gitattributes` marks it `-text`; this machine
/// runs `core.autocrlf=true`). A rewritten WGSL file also fails the artifact hash in the test above.
#[test]
fn the_fixture_files_are_not_line_ending_converted() {
    assert!(!WGSL.contains(&b'\r'), "the WGSL has carriage returns");
    assert!(!REFLECTION.contains('\r'));
    assert!(!MANIFEST.contains('\r'));
    assert_eq!(WGSL.len(), 478);
    assert_eq!(REFLECTION.len(), 1040);
}

#[test]
fn one_flipped_wgsl_byte_is_an_artifact_hash_mismatch() {
    let mut bytes = WGSL.to_vec();
    let at = bytes.iter().position(|b| *b == b'+').expect("the addition");
    bytes[at] = b'-';
    let message = identity_error(
        &DESCRIPTOR,
        &Artifact {
            bytes: leak_bytes(bytes),
            ..ARTIFACT
        },
    );
    assert!(
        message.contains("artifact hash (stored \"282df814ca1e66dd\""),
        "{message}"
    );
    assert!(!message.contains("source hash"), "{message}");
    assert!(!message.contains("entry point ("), "{message}");
}

#[test]
fn a_truncated_wgsl_is_an_artifact_hash_mismatch() {
    let message = identity_error(
        &DESCRIPTOR,
        &Artifact {
            bytes: &WGSL[..WGSL.len() - 1],
            ..ARTIFACT
        },
    );
    assert!(
        message.contains("artifact hash (stored \"282df814ca1e66dd\""),
        "{message}"
    );
    assert!(!message.contains("source hash"), "{message}");
}

#[test]
fn a_crlf_rewritten_wgsl_is_an_artifact_hash_mismatch() {
    let crlf = std::str::from_utf8(WGSL).unwrap().replace('\n', "\r\n");
    let message = identity_error(
        &DESCRIPTOR,
        &Artifact {
            bytes: leak_bytes(crlf.into_bytes()),
            ..ARTIFACT
        },
    );
    assert!(
        message.contains("artifact hash (stored \"282df814ca1e66dd\""),
        "{message}"
    );
    assert!(!message.contains("source hash"), "{message}");
}

#[test]
fn a_replaced_source_hash_is_a_source_hash_mismatch() {
    assert_eq!(REFLECTION.matches("563ba40500e91764").count(), 1);
    let message = identity_error(
        &DESCRIPTOR,
        &Artifact {
            reflection: leak_str(REFLECTION.replace("563ba40500e91764", "0123456789abcdef")),
            ..ARTIFACT
        },
    );
    assert!(
        message
            .contains("source hash (stored \"0123456789abcdef\", expected \"563ba40500e91764\")"),
        "{message}"
    );
    assert!(!message.contains("artifact hash"), "{message}");
    assert!(!message.contains("entry point ("), "{message}");
}

#[test]
fn a_renamed_entry_point_is_an_entry_point_mismatch() {
    let kernel = KernelDescriptor {
        entry_point: "gpu_vector_add_other",
        ..DESCRIPTOR
    };
    let message = identity_error(&kernel, &ARTIFACT);
    assert!(
        message.contains(
            "entry point (stored \"gpu_vector_add_add\", expected \"gpu_vector_add_other\")"
        ),
        "{message}"
    );
    assert!(!message.contains("artifact hash"), "{message}");
}

#[test]
fn a_wrong_target_is_a_target_mismatch() {
    let message = identity_error(
        &DESCRIPTOR,
        &Artifact {
            format: Format::Spirv,
            ..ARTIFACT
        },
    );
    assert!(
        message.contains("target (stored \"wgsl\", expected \"spirv\")"),
        "{message}"
    );
}

#[test]
fn every_differing_identity_value_is_named() {
    let kernel = KernelDescriptor {
        entry_point: "gpu_vector_add_other",
        ..DESCRIPTOR
    };
    let message = identity_error(
        &kernel,
        &Artifact {
            bytes: &WGSL[..WGSL.len() - 1],
            reflection: leak_str(REFLECTION.replace("563ba40500e91764", "0123456789abcdef")),
            ..ARTIFACT
        },
    );
    for what in ["entry point (", "source hash (", "artifact hash ("] {
        assert!(message.contains(what), "{what} missing from {message}");
    }
    assert!(!message.contains("target ("), "{message}");
}

fn with_parameter(index: usize, change: impl Fn(&mut ParameterDescriptor)) -> KernelDescriptor {
    let mut parameters = DESCRIPTOR.parameters.to_vec();
    change(&mut parameters[index]);
    KernelDescriptor {
        parameters: Box::leak(parameters.into_boxed_slice()),
        ..DESCRIPTOR
    }
}

#[test]
fn a_changed_descriptor_binding_is_a_mismatch() {
    let kernel = with_parameter(3, |p| {
        p.binding = Some(ResourceBinding {
            group: 0,
            binding: 5,
        });
    });
    assert_eq!(
        mismatch(&kernel, &ARTIFACT),
        "out.binding: host 5, compiler 2"
    );
}

#[test]
fn a_changed_descriptor_access_is_a_mismatch() {
    let kernel = with_parameter(2, |p| p.access = Access::ReadWrite);
    assert_eq!(mismatch(&kernel, &ARTIFACT), "b.access differs");
}

#[test]
fn a_changed_descriptor_scalar_kind_is_a_mismatch() {
    let kernel = with_parameter(1, |p| {
        let mut layout = p.resource_layout.expect("a storage parameter has a layout");
        layout.element.kind = TypeLayoutKind::Scalar(ScalarKind::U32);
        p.resource_layout = Some(layout);
    });
    assert_eq!(
        mismatch(&kernel, &ARTIFACT),
        "a: type or field count differs"
    );
}

#[test]
fn a_changed_descriptor_element_size_is_a_mismatch() {
    // A valid StorageV1 element twice as large as the stored one (two floats in place of one).
    const PAIR: TypeLayout = TypeLayout::structure(
        "Pair",
        8,
        4,
        &[
            StructFieldLayout {
                name: "x",
                offset: 0,
                ty: TypeLayout::scalar("float", ScalarKind::F32),
            },
            StructFieldLayout {
                name: "y",
                offset: 4,
                ty: TypeLayout::scalar("float", ScalarKind::F32),
            },
        ],
    );
    let kernel = with_parameter(2, |p| {
        p.resource_layout = Some(ResourceLayout::storage_v1(PAIR))
    });
    assert_eq!(mismatch(&kernel, &ARTIFACT), "b.size: host 8, compiler 4");
}

#[test]
fn a_changed_workgroup_size_is_a_mismatch() {
    let kernel = KernelDescriptor {
        workgroup_size: [32, 1, 1],
        ..DESCRIPTOR
    };
    assert_eq!(
        mismatch(&kernel, &ARTIFACT),
        "compute workgroup dimensions differ"
    );
}

#[test]
fn a_descriptor_with_artifacts_keeps_them() {
    let with = KernelDescriptor {
        artifacts: &[ARTIFACT],
        ..DESCRIPTOR
    };
    assert_ne!(with, DESCRIPTOR);
    assert!(
        DESCRIPTOR.artifacts.is_empty(),
        "the macro emits no artifacts"
    );
    assert_eq!(with.artifacts[0].key, KEY);
    verify_artifact(&with, &with.artifacts[0]).expect("verifies from the descriptor itself");
}
