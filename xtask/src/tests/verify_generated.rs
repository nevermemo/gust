use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    ADAPTER_INFO_ARGS, GENERATED_DIR, VerifyMode, adapter_from_check, clear_generated_output,
    clears_generated_output,
};
use crate::{process::CheckRecord, workspace::EXPECTED_SPIRV};

const ALL_MODES: [VerifyMode; 6] = [
    VerifyMode::Smoke,
    VerifyMode::Fast,
    VerifyMode::Gpu,
    VerifyMode::Examples,
    VerifyMode::Artifacts,
    VerifyMode::Full,
];

/// A fresh root under the cargo target directory (`<target>/<profile>/deps/<test binary>`).
fn temp_root() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(Path::to_path_buf))
        .unwrap_or_else(std::env::temp_dir);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let root = base.join("xtask-tests").join(format!(
        "generated-{}-{stamp}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

/// The names in a directory, sorted.
fn names(directory: &Path) -> Vec<String> {
    let mut names = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn the_artifact_modes_clear_a_planted_stale_kernel_output() {
    for mode in [VerifyMode::Artifacts, VerifyMode::Full] {
        let root = temp_root();
        let directory = root.join(GENERATED_DIR);
        fs::create_dir_all(directory.join("nested")).unwrap();
        // The scenario of the follow-up: a kernel was deleted from an example, EXPECTED_SPIRV still lists it, and its
        // old SPIR-V is still on disk. Every expected name is stale here, and so are other files.
        for name in EXPECTED_SPIRV {
            fs::write(directory.join(name), b"stale").unwrap();
        }
        fs::write(directory.join("deleted_kernel__run.slang"), b"stale").unwrap();
        fs::write(directory.join("nested/old.wgsl"), b"stale").unwrap();

        clear_generated_output(&root, mode).unwrap();

        assert!(directory.is_dir(), "{mode:?}: the directory must remain");
        assert_eq!(names(&directory), Vec::<String>::new(), "{mode:?}");
        let _ = fs::remove_dir_all(&root);
    }
}

#[test]
fn only_the_artifact_modes_clear_and_the_others_leave_the_directory_alone() {
    for mode in ALL_MODES {
        let expected = matches!(mode, VerifyMode::Artifacts | VerifyMode::Full);
        assert_eq!(clears_generated_output(mode), expected, "{mode:?}");
        let root = temp_root();
        let directory = root.join(GENERATED_DIR);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("kept.spv"), b"kept").unwrap();

        clear_generated_output(&root, mode).unwrap();

        let kept = if expected { vec![] } else { vec!["kept.spv"] };
        assert_eq!(names(&directory), kept, "{mode:?}");
        let _ = fs::remove_dir_all(&root);
    }
}

#[test]
fn a_missing_output_directory_is_created_and_a_file_in_its_place_fails() {
    let root = temp_root();
    clear_generated_output(&root, VerifyMode::Full).unwrap();
    assert!(root.join(GENERATED_DIR).is_dir());
    // Cleared once more it is still an empty directory.
    clear_generated_output(&root, VerifyMode::Full).unwrap();
    assert_eq!(names(&root.join(GENERATED_DIR)), Vec::<String>::new());

    // A plain file where the directory should be is not silently accepted.
    let other = temp_root();
    fs::write(other.join(GENERATED_DIR), b"not a directory").unwrap();
    let outcome = clear_generated_output(&other, VerifyMode::Full);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&other);
    let error = outcome.expect_err("a file in place of the directory must fail the step");
    assert!(
        error.starts_with("could not clear generated-wgpu: "),
        "{error}"
    );
}

#[test]
fn the_adapter_step_runs_the_gust_wgpu_example_locked_and_quiet() {
    assert_eq!(
        ADAPTER_INFO_ARGS,
        [
            "run",
            "--locked",
            "--quiet",
            "-p",
            "gust-wgpu",
            "--example",
            "adapter_info"
        ]
    );
    // The example exists where the step expects it.
    let example =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/gust-wgpu/examples/adapter_info.rs");
    assert!(
        example.is_file(),
        "crates/gust-wgpu/examples/adapter_info.rs is missing"
    );
}

/// A source pin, not a behavioural test: `verify_inner` needs cargo, slangc and a GPU, so a unit test cannot run it.
/// This pins the call sites in its source instead: the adapter step and the clearing both come after the
/// prerequisite checks and before the first cargo call of the mode, and the artifact scan reads the directory the
/// clearing empties.
#[test]
fn verify_inner_names_the_adapter_and_clears_before_its_cargo_calls() {
    let source = include_str!("../verify.rs");
    let body = &source[source.find("fn verify_inner(").expect("verify_inner")..];
    let position = |needle: &str| {
        body.find(needle)
            .unwrap_or_else(|| panic!("verify_inner no longer contains `{needle}`"))
    };
    let reflect = position("build_slang_reflect(BuildOptions::default())?;");
    let adapter = position("*adapter = Some(record_adapter(root, checks)?);");
    let clear = position("clear_generated_output(root, mode)?;");
    let calls = position("for args in cargo_calls(mode) {");
    let scan = position("fs::read_dir(root.join(GENERATED_DIR))");
    assert!(reflect < adapter && adapter < clear && clear < calls && calls < scan);
}

fn finished_adapter_check(stdout: &str) -> CheckRecord {
    CheckRecord {
        program: "cargo".to_owned(),
        arguments: ADAPTER_INFO_ARGS.iter().map(|a| (*a).to_owned()).collect(),
        exit_code: Some(0),
        stdout: stdout.to_owned(),
        stderr: String::new(),
    }
}

const GOOD_ADAPTER_LINE: &str = r#"{"name":"Test GPU","vendor":4318,"device":1,"device_type":"DiscreteGpu","backend":"Vulkan","driver":"d","driver_info":"1.0"}"#;

#[test]
fn an_adapter_check_that_names_no_adapter_loses_its_exit_code_and_says_why() {
    let two_lines = format!("{GOOD_ADAPTER_LINE}\n{GOOD_ADAPTER_LINE}\n");
    for (stdout, reason) in [
        (
            "",
            "adapter_info must print exactly one JSON line on stdout (no adapter?)",
        ),
        (
            "\n",
            "adapter_info must print exactly one JSON line on stdout (no adapter?)",
        ),
        (
            two_lines.as_str(),
            "adapter_info must print exactly one JSON line on stdout (no adapter?)",
        ),
    ] {
        let mut record = finished_adapter_check(stdout);
        let error = adapter_from_check(&mut record).expect_err(stdout);
        assert_eq!(error, reason);
        assert_eq!(record.exit_code, None, "{stdout:?}");
        assert_eq!(
            record.stderr,
            format!("adapter guard: {reason}\n"),
            "{stdout:?}"
        );
    }
}

#[test]
fn an_adapter_check_with_a_named_adapter_keeps_its_record_untouched() {
    let mut record = finished_adapter_check(&format!("{GOOD_ADAPTER_LINE}\n"));
    let adapter = adapter_from_check(&mut record).unwrap();
    assert_eq!(adapter.name, "Test GPU");
    assert_eq!(adapter.vendor, "4318");
    assert_eq!(adapter.backend, "Vulkan");
    assert_eq!(record.exit_code, Some(0));
    assert_eq!(record.stderr, "");
}
