use std::{fs, path::Path};

use crate::{
    arch::check_arch,
    checks::check_dual,
    docs::docs_check,
    lint::lint,
    process::{CheckRecord, report_check, require_command, run_checked, run_checked_owned},
    reflect::{BuildOptions, build_slang_reflect},
    time::now_utc,
    ui::check_ui,
    validation::{ArtifactRecord, GitStamp, RunInfo, sha256_hex, write_validation},
    workspace::{EXAMPLES, EXPECTED_SPIRV, SMOKE_EXAMPLES, workspace_root, workspace_test_args},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifyMode {
    Smoke,
    Fast,
    Gpu,
    Examples,
    Artifacts,
    Full,
}

impl VerifyMode {
    pub(crate) fn from_name(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "fast" => Ok(Self::Fast),
            "gpu" => Ok(Self::Gpu),
            "examples" => Ok(Self::Examples),
            "artifacts" => Ok(Self::Artifacts),
            "full" => Ok(Self::Full),
            _ => Err(format!("unknown verify mode `{value}`")),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Fast => "fast",
            Self::Gpu => "gpu",
            Self::Examples => "examples",
            Self::Artifacts => "artifacts",
            Self::Full => "full",
        }
    }
}

pub(crate) struct VerifyCommand {
    pub(crate) mode: VerifyMode,
    pub(crate) record: bool,
}

impl VerifyCommand {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut mode = VerifyMode::Smoke;
        let mut record = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--full" | "-Full" | "--Full" => mode = VerifyMode::Full,
                "--mode" | "-Mode" => {
                    index += 1;
                    let Some(value) = args.get(index) else {
                        return Err("--mode needs a value".into());
                    };
                    mode = VerifyMode::from_name(value)?;
                }
                "--record" => record = Some(true),
                "--no-record" => record = Some(false),
                other => mode = VerifyMode::from_name(other)?,
            }
            index += 1;
        }
        Ok(Self {
            mode,
            record: record.unwrap_or(mode == VerifyMode::Full),
        })
    }
}

pub(crate) fn verify(mode: VerifyMode, record: bool) -> Result<(), String> {
    let root = workspace_root()?;
    let started = now_utc();
    // Taken before the checks run: the record names the tree they ran against.
    let git = GitStamp::read(&root);
    let mut checks = Vec::new();
    let mut artifacts = Vec::new();
    let result = verify_inner(&root, mode, &mut checks, &mut artifacts);
    if record {
        let run = RunInfo {
            mode,
            started: &started,
            git: &git,
        };
        write_validation(&root, &run, result.as_ref().err(), &checks, &artifacts)?;
    }
    result?;
    println!("GUST verification passed.");
    Ok(())
}

fn verify_inner(
    root: &Path,
    mode: VerifyMode,
    checks: &mut Vec<CheckRecord>,
    artifacts: &mut Vec<ArtifactRecord>,
) -> Result<(), String> {
    // Cheapest checks first; they need only git, and check-ui needs cargo (no slangc, no GPU).
    if matches!(
        mode,
        VerifyMode::Smoke | VerifyMode::Fast | VerifyMode::Full
    ) {
        lint(root, checks)?;
        docs_check(root, checks)?;
        check_arch(root, checks)?;
        check_ui(root, checks)?;
    }
    require_command("cargo")?;
    require_command("rustc")?;
    require_command("slangc")?;
    if matches!(mode, VerifyMode::Artifacts | VerifyMode::Full) {
        require_command("spirv-val")?;
    }
    run_checked(root, checks, "rustc", &["--version"])?;
    run_checked(root, checks, "slangc", &["-version"])?;
    build_slang_reflect(BuildOptions::default())?;
    for args in cargo_calls(mode) {
        run_checked_owned(root, checks, "cargo", &args)?;
    }
    if runs_check_dual(mode) {
        // The dual build and the example runs with the CPU twin (D26): its cargo calls are its own, with their own
        // target directory and rustflags, so it runs in process and is recorded like the other xtask checks.
        let outcome = check_dual(&[])
            .map(|()| "check-dual passed (full mode)\n".to_owned())
            .map_err(|error| vec![error]);
        report_check(checks, "check-dual", outcome)?;
    }
    if matches!(mode, VerifyMode::Artifacts | VerifyMode::Full) {
        run_checked(root, checks, "spirv-val", &["--version"])?;
        let mut spv = fs::read_dir(root.join("generated-wgpu"))
            .map_err(|error| error.to_string())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "spv"))
            .collect::<Vec<_>>();
        spv.sort();
        let found = spv
            .iter()
            .filter_map(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        compare_artifact_names(EXPECTED_SPIRV, &found)?;
        for artifact in spv {
            let path = artifact.to_string_lossy().into_owned();
            run_checked(
                root,
                checks,
                "spirv-val",
                &["--target-env", "vulkan1.2", &path],
            )?;
            let file = artifact
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<invalid>")
                .to_owned();
            let bytes = fs::read(&artifact).map_err(|error| error.to_string())?;
            artifacts.push(ArtifactRecord {
                file: file.clone(),
                sha256: sha256_hex(&bytes),
            });
            println!("Validated {file}");
        }
    }
    Ok(())
}

/// The Slang tests of the core library in the release profile (the Full mode and suite).
/// `cfg(not(debug_assertions))` tests, such as the source-map fallback, run only here.
pub(crate) const RELEASE_SLANG_TESTS: [&str; 7] = [
    "test",
    "--locked",
    "--release",
    "-p",
    "gust",
    "--lib",
    "slang",
];

/// The known-mismatch ledger in the release profile (the Full mode). The ledger asserts the plain-Rust outcome
/// of every row for the `overflow-checks` mode it detects at run time (D38); the debug run (Gpu suite) covers
/// checks on, this run covers checks off, so both branches of every mode-keyed row run on every full gate.
pub(crate) const RELEASE_LEDGER_TEST: [&str; 7] = [
    "test",
    "--locked",
    "--release",
    "-p",
    "gust-wgpu",
    "--test",
    "mismatch",
];

/// Every cargo call a mode makes after its prerequisite checks, in order, as the arguments
/// after `cargo`. `verify_inner` runs exactly this list, so a test of it is a test of the mode:
/// the Full mode is the release gate and must not lose a step unnoticed.
pub(crate) fn cargo_calls(mode: VerifyMode) -> Vec<Vec<String>> {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    let mut calls = Vec::new();
    if matches!(
        mode,
        VerifyMode::Smoke | VerifyMode::Fast | VerifyMode::Full
    ) {
        calls.push(owned(&["fmt", "--all", "--", "--check"]));
    }
    if matches!(mode, VerifyMode::Smoke | VerifyMode::Full) {
        calls.push(owned(&[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]));
    }
    match mode {
        VerifyMode::Fast => {
            calls.push(owned(&["test", "--locked", "-p", "gust-macros"]));
            calls.push(owned(&["test", "--locked", "-p", "gust", "--lib"]));
            calls.push(owned(&["test", "--locked", "-p", "gust-wgpu"]));
        }
        VerifyMode::Gpu => calls.push(owned(&["test", "--locked", "-p", "gust-wgpu"])),
        VerifyMode::Smoke => calls.push(workspace_test_args()),
        VerifyMode::Full => {
            calls.push(workspace_test_args());
            calls.push(owned(&RELEASE_SLANG_TESTS));
            calls.push(owned(&RELEASE_LEDGER_TEST));
        }
        VerifyMode::Examples | VerifyMode::Artifacts => {}
    }
    if matches!(mode, VerifyMode::Examples | VerifyMode::Full) {
        for example in EXAMPLES {
            calls.push(owned(&["test", "-p", example, "--", "--ignored"]));
        }
    }
    let examples_to_run = match mode {
        VerifyMode::Smoke => SMOKE_EXAMPLES,
        VerifyMode::Examples | VerifyMode::Artifacts | VerifyMode::Full => EXAMPLES,
        VerifyMode::Fast | VerifyMode::Gpu => &[],
    };
    for example in examples_to_run {
        calls.push(owned(&example_run_args(example)));
    }
    calls
}

/// Whether a mode also runs `cargo xtask check-dual` (full mode, GPU): only the release gate does, after its cargo
/// calls. It is not part of [`cargo_calls`] because it is a check of its own, not one cargo call.
pub(crate) fn runs_check_dual(mode: VerifyMode) -> bool {
    mode == VerifyMode::Full
}

/// The cargo call that runs one example binary and exports its SPIR-V (the Spirv suite).
pub(crate) fn example_run_args(example: &str) -> [&str; 5] {
    ["run", "--locked", "--quiet", "-p", example]
}

/// Compares the exported SPIR-V file names with the expected kernels and names the
/// differences separately: a renamed kernel is one missing and one unexpected file.
fn compare_artifact_names(expected: &[&str], found: &[String]) -> Result<(), String> {
    let mut missing = expected
        .iter()
        .copied()
        .filter(|name| !found.iter().any(|found| found == name))
        .collect::<Vec<_>>();
    let mut unexpected = found
        .iter()
        .map(String::as_str)
        .filter(|name| !expected.contains(name))
        .collect::<Vec<_>>();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    unexpected.sort_unstable();
    let mut message = format!(
        "Exported SPIR-V kernels differ from the {} expected ({} found).",
        expected.len(),
        found.len()
    );
    if !missing.is_empty() {
        message.push_str(&format!(" Missing: {}.", missing.join(", ")));
    }
    if !unexpected.is_empty() {
        message.push_str(&format!(" Unexpected: {}.", unexpected.join(", ")));
    }
    message.push_str(
        " Update EXPECTED_SPIRV in xtask/src/workspace.rs deliberately for new examples.",
    );
    Err(message)
}

#[cfg(test)]
#[path = "tests/artifact_names.rs"]
mod artifact_names_tests;

#[cfg(test)]
#[path = "tests/verify_steps.rs"]
mod verify_steps_tests;
