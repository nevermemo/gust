use super::*;

/// The CLI reflection path names the matrix layout (the native helper pins the same one).
#[cfg(feature = "slang-runtime")]
#[test]
fn the_cli_reflection_command_pins_the_column_major_matrix_layout() {
    for (target, spirv_flag) in [
        (Target::Wgsl, None),
        (Target::Spirv, Some("-fvk-use-entrypoint-name")),
    ] {
        for (shape, stage) in [
            (crate::Shape::Grid, "compute"),
            (crate::Shape::Pixel, "fragment"),
        ] {
            let command = slangc_reflection_command(
                std::path::Path::new("slangc"),
                std::path::Path::new("kernel.slang"),
                target,
                "entry",
                shape,
            );
            let mut expected = vec![
                "kernel.slang",
                "-target",
                target_name(target),
                "-entry",
                "entry",
                "-stage",
                stage,
                "-matrix-layout-column-major",
            ];
            expected.extend(spirv_flag);
            let actual: Vec<String> = command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect();
            assert_eq!(actual, expected, "{target:?} {shape:?}");
        }
    }
}

/// The helper is told the stage explicitly, as its seventh argument (mx.t0a), whatever the shape: a helper that
/// defaults to compute and a kernel that is not compute can no longer agree by accident.
#[cfg(feature = "slang-runtime")]
#[test]
fn the_helper_is_given_the_stage_of_the_shape_explicitly() {
    for (shape, stage) in [
        (crate::Shape::Grid, "compute"),
        (crate::Shape::Pixel, "fragment"),
    ] {
        let kernel = KERNEL.with_shape(shape);
        let command = helper_command(
            Path::new("helper"),
            &kernel,
            Target::Spirv,
            Path::new("kernel.slang"),
            Path::new("kernel.artifact"),
            Path::new("reflection.json"),
        );
        let arguments: Vec<String> = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            arguments,
            [
                "kernel.slang",
                "entry",
                "spirv",
                "kernel.artifact",
                "reflection.json",
                stage
            ],
            "{shape:?}"
        );
    }
}

/// A failed helper run that printed `stderr`.
#[cfg(feature = "slang-runtime")]
fn failure_output(stderr: &str) -> std::process::Output {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt;
    #[cfg(unix)]
    let status = std::process::ExitStatus::from_raw(1 << 8);
    #[cfg(windows)]
    let status = std::process::ExitStatus::from_raw(1);
    std::process::Output {
        status,
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
    }
}

/// A helper from before schema 3 answers the seventh argument with its usage text, which has no stage: the error
/// says the helper must be rebuilt (mx.k). Any other failure, and the usage text of a current helper, are not
/// given the note.
#[cfg(feature = "slang-runtime")]
#[test]
fn a_helper_that_refuses_the_stage_argument_is_reported_as_outdated() {
    let old = "usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json>, or <layout.slang>, or \
                   --version";
    let current = "usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json> [compute|fragment], \
                       or <layout.slang>, or --version";
    assert!(outdated_helper_note(&failure_output(old)).contains("predates reflection schema 3"));
    assert_eq!(outdated_helper_note(&failure_output(current)), "");
    assert_eq!(
        outdated_helper_note(&failure_output("error: unsupported stage: x")),
        ""
    );
    let error = compile_clean_with(
        &KERNEL,
        Target::Wgsl,
        Path::new("fake-helper"),
        |_, _, _| Ok(failure_output(old)),
    )
    .unwrap_err();
    match error {
        Error::Compiler(slang::Error::CompilationFailed { diagnostics, .. }) => {
            assert!(
                diagnostics.contains("predates reflection schema 3")
                    && diagnostics.contains("cargo xtask build-slang-reflect --force"),
                "{diagnostics}"
            );
        }
        other => panic!("expected the helper's failure, got {other:?}"),
    }
}

/// A canned helper result (a real one cannot be forced to lie): only the
/// identity fields, which `from_output` checks before anything else.
fn canned(target: &str, entry: &str, source_hash: &str, artifact_hash: &str) -> String {
    format!(
        "{{\"schemaVersion\":1,\"compiler\":\"canned\",\"target\":\"{target}\",\"entryPoint\":\"{entry}\",\"sourceHash\":\"{source_hash}\",\"artifactHash\":\"{artifact_hash}\"}}"
    )
}

fn identity_error(compiled_source: &str, json: &str) -> bool {
    let result = CompiledReflection::from_output(
        &KERNEL,
        compiled_source,
        Target::Wgsl,
        ARTIFACT.to_vec(),
        json,
    );
    matches!(result, Err(Error::Invalid(message)) if message.contains("identity mismatch"))
}

#[test]
fn helper_identity_is_checked_field_by_field_against_the_compiled_copy() {
    let copy = slang::compile_copy(SOURCE, &KERNEL.source);
    assert_ne!(copy, SOURCE, "the fixture must have directives");
    let source = fingerprint(copy.as_bytes());
    let artifact = fingerprint(ARTIFACT);
    // The control: every field right gets past the identity check (and fails
    // later, on the missing workgroup size).
    assert!(!identity_error(
        &copy,
        &canned("wgsl", "entry", &source, &artifact)
    ));
    for (what, json) in [
        ("target", canned("spirv", "entry", &source, &artifact)),
        ("entry point", canned("wgsl", "other", &source, &artifact)),
        (
            "stale source hash (the clean text)",
            canned("wgsl", "entry", &fingerprint(SOURCE.as_bytes()), &artifact),
        ),
        (
            "mismatched source hash",
            canned("wgsl", "entry", "0000000000000000", &artifact),
        ),
        (
            "mismatched artifact hash",
            canned("wgsl", "entry", &source, &fingerprint(b"other bytes")),
        ),
    ] {
        assert!(identity_error(&copy, &json), "{what}");
    }
}

/// What the real helper writes for a WGSL artifact whose bytes are `text`, computed from the text it was given.
#[cfg(feature = "slang-runtime")]
fn fake_helper_json(text: &str, source_hash_of: &str) -> String {
    format!(
        "{{\"schemaVersion\":1,\"compiler\":\"canned\",\"target\":\"wgsl\",\"entryPoint\":\"entry\",\"sourceHash\":\"{}\",\"artifactHash\":\"{}\",\"workgroupSize\":[1,1,1],\"parameters\":[]}}\n",
        fingerprint(source_hash_of.as_bytes()),
        fingerprint(text.as_bytes()),
    )
}

#[cfg(feature = "slang-runtime")]
fn success_output() -> std::process::Output {
    std::process::Output {
        status: std::process::ExitStatus::default(),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

/// The clean compile hands the helper `kernel.slang_source` exactly, never the `#line` copy, and checks the
/// helper's `sourceHash` against that text, so the stored artifact verifies at run time (D45).
#[cfg(feature = "slang-runtime")]
#[test]
fn the_clean_compile_gives_the_helper_the_clean_text_never_the_directive_copy() {
    let copy = slang::compile_copy(SOURCE, &KERNEL.source);
    assert_ne!(copy, SOURCE, "the fixture must have directives");
    let given = std::cell::RefCell::new(None);
    let written = std::cell::RefCell::new(String::new());
    let result = compile_clean_with(
        &KERNEL,
        Target::Wgsl,
        Path::new("fake-helper"),
        |source, artifact, json| {
            let text = fs::read_to_string(source).unwrap();
            fs::write(artifact, &text).unwrap();
            let report = fake_helper_json(&text, &text);
            fs::write(json, &report).unwrap();
            *written.borrow_mut() = report;
            *given.borrow_mut() = Some(text);
            Ok(success_output())
        },
    )
    .unwrap();
    assert_eq!(given.borrow().as_deref(), Some(SOURCE));
    assert_eq!(result.compiled.artifact, SOURCE.as_bytes());
    assert_eq!(result.reflection_json, *written.borrow());
    assert_eq!(result.compiled.compiler_version, "canned");
    assert!(
        result
            .reflection_json
            .contains(&fingerprint(SOURCE.as_bytes()))
    );
    assert!(
        !result
            .reflection_json
            .contains(&fingerprint(copy.as_bytes()))
    );
}

/// A helper that reports the hash of the directive copy is rejected: the identity is the clean text's.
#[cfg(feature = "slang-runtime")]
#[test]
fn the_clean_compile_rejects_a_source_hash_of_the_directive_copy() {
    let copy = slang::compile_copy(SOURCE, &KERNEL.source);
    let error = compile_clean_with(
        &KERNEL,
        Target::Wgsl,
        Path::new("fake-helper"),
        |source, artifact, json| {
            let text = fs::read_to_string(source).unwrap();
            fs::write(artifact, &text).unwrap();
            fs::write(json, fake_helper_json(&text, &copy)).unwrap();
            Ok(success_output())
        },
    )
    .unwrap_err();
    match error {
        Error::Invalid(message) => assert!(message.contains("source hash"), "{message}"),
        other => panic!("expected an identity mismatch, got {other:?}"),
    }
}

#[cfg(feature = "slang-runtime")]
#[test]
fn the_clean_compile_without_a_helper_is_the_helper_unavailable_error() {
    let error = compile_reflected_clean(
        &KERNEL,
        Target::Wgsl,
        "no-such-helper-for-the-clean-compile-test",
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::HelperUnavailable { .. }),
        "{error:?}"
    );
}
