//! E3, E4 and E5: the committed artifact changed, deleted, and the fixture built strict and dual together.

use std::fs;

use super::{
    PRECOMPILED_DIR, STRICT_PROFILE,
    fixture::{Fixture, body_wgsl_artifact, copy_fixture, require_success},
    stripped::VERIFIED_PREFIX,
};
use crate::{
    checks::{dual_target_dir, precompiled_dual_steps, run_dual_step},
    process::{PROFILE_VARIABLE, run_capture_in},
    ui::{Expectation, judge, parse_expected},
};

/// The pinned error of E4, relative to [`PRECOMPILED_DIR`]: the keys of a `check-ui` `expected.txt`.
pub(super) const E4_EXPECTED: &str = "e4-expected.txt";

// ---------------------------------------------------------------------------------------------------------------
// E3

/// E3: a mutated artifact in a scratch copy gives the identity error at run time. Returns the summary.
pub(super) fn e3_mutated_artifact(fixture: &Fixture, check_only: bool) -> Result<String, String> {
    let scratch = fixture.scratch.join("e3");
    let scratch_target = fixture.scratch_target_dir();
    copy_fixture(fixture, &scratch)?;
    let artifact = body_wgsl_artifact(&scratch.join("kernels").join("gust-artifacts"))?;
    let original = fs::read(&artifact).map_err(|error| format!("check-precompiled E3: {error}"))?;
    let mutated =
        mutate_wgsl(&original).map_err(|error| format!("check-precompiled E3: {error}"))?;
    fs::write(&artifact, &mutated).map_err(|error| format!("check-precompiled E3: {error}"))?;
    println!(
        "check-precompiled E3: scratch copy {}: the first `+` of the committed WGSL ({} bytes) is now `-`",
        fixture.shown(&scratch),
        original.len()
    );

    let target_dir = Fixture::target_dir_arg(&scratch_target)?;
    let manifest = scratch.join("Cargo.toml");
    let manifest = manifest
        .to_str()
        .ok_or("check-precompiled E3: the scratch path is not valid UTF-8")?;
    let output = run_capture_in(
        &mut fixture.cargo(&[
            "build",
            "--release",
            "--locked",
            "--color",
            "never",
            "--manifest-path",
            manifest,
            "-p",
            "app",
            "--target-dir",
            target_dir,
        ]),
        "check-precompiled E3",
        "cargo",
    )?;
    require_success(
        "E3",
        "cargo build --release -p app (mutated copy; the macro embeds the bytes without checking them)",
        &output,
    )?;

    if check_only {
        return Ok("passed (build only: --check-only skips the run, which needs an adapter): with one operator of the artifact changed, the build succeeded".to_owned());
    }
    let binary = Fixture::app_binary(&scratch_target);
    let run = fixture
        .command(&binary)
        .output()
        .map_err(|error| format!("check-precompiled E3: could not run the mutated app: {error}"))?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    let message = identity_error(run.status.success(), &stdout, &stderr)
        .map_err(|problems| format!("check-precompiled E3: {}", problems.join("; ")))?;
    println!("check-precompiled E3: the mutated `app` ended with: {message}");
    Ok("passed: with one operator of the artifact changed, the build succeeded and `app` ended at run time with the identity error naming the artifact hash (D17), without a `verified:` line".to_owned())
}

/// The artifact hash error that `verify_artifact` gives, as the app prints it (`error: dispatch failed: could not
/// validate kernel reflection: native target, entry point, source or artifact identity mismatch: artifact hash
/// (stored "..", expected "..")`).
const IDENTITY_TEXT: &str = "identity mismatch: artifact hash (stored ";

/// The problems of the run of the mutated `app`: it must fail with the identity error that names the artifact hash
/// alone, and never print the `verified:` line. Returns the error line.
pub(crate) fn identity_error(
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<String, Vec<String>> {
    let mut problems = Vec::new();
    if success {
        problems.push("the mutated `app` exited 0: the changed artifact was accepted".to_owned());
    }
    if stdout.lines().any(|line| line.starts_with(VERIFIED_PREFIX)) {
        problems.push("the mutated `app` printed the `verified:` line".to_owned());
    }
    let line = stderr
        .lines()
        .find(|line| line.contains(IDENTITY_TEXT))
        .map(str::trim);
    match line {
        None => problems.push(format!(
            "no identity error (`{IDENTITY_TEXT}..`) on stderr; stderr was: {}",
            stderr.trim()
        )),
        Some(line) => {
            for other in ["source hash (", "entry point (", "target ("] {
                if line.contains(other) {
                    problems.push(format!(
                        "the identity error names more than the artifact hash (`{other}..`): {line}"
                    ));
                }
            }
        }
    }
    match (problems.is_empty(), line) {
        (true, Some(line)) => Ok(line.to_owned()),
        _ => Err(problems),
    }
}

/// `wgsl` with its first `+` turned into `-`: the changed artifact would compute `a - b` where the kernel says
/// `a + b`, so a run that accepted it would print a wrong result. Fails when there is no `+`.
pub(crate) fn mutate_wgsl(wgsl: &[u8]) -> Result<Vec<u8>, String> {
    let position = wgsl
        .iter()
        .position(|byte| *byte == b'+')
        .ok_or("the WGSL artifact has no `+` to change")?;
    let mut mutated = wgsl.to_vec();
    mutated[position] = b'-';
    Ok(mutated)
}

// ---------------------------------------------------------------------------------------------------------------
// E4

/// E4: a missing artifact under `native` is a compile error at the kernel line. Returns the summary.
pub(super) fn e4_missing_artifact(fixture: &Fixture) -> Result<String, String> {
    let scratch = fixture.scratch.join("e4");
    let scratch_target = fixture.scratch_target_dir();
    copy_fixture(fixture, &scratch)?;
    let directory = scratch.join("kernels").join("gust-artifacts");
    let mut removed = Vec::new();
    for entry in
        fs::read_dir(&directory).map_err(|error| format!("check-precompiled E4: {error}"))?
    {
        let entry = entry.map_err(|error| format!("check-precompiled E4: {error}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "manifest.json" {
            fs::remove_file(entry.path())
                .map_err(|error| format!("check-precompiled E4: {error}"))?;
            removed.push(name);
        }
    }
    if removed.is_empty() {
        return Err(
            "check-precompiled E4: the scratch copy had no artifact file to remove".to_owned(),
        );
    }
    println!(
        "check-precompiled E4: scratch copy {}: removed {} (the manifest stays)",
        fixture.shown(&scratch),
        removed.join(", ")
    );

    let expected_path = fixture.root.join(PRECOMPILED_DIR).join(E4_EXPECTED);
    let expected = fs::read_to_string(&expected_path).map_err(|error| {
        format!("check-precompiled E4: {PRECOMPILED_DIR}/{E4_EXPECTED}: {error}")
    })?;
    let wanted = parse_expected(&expected).map_err(|error| {
        format!("check-precompiled E4: {PRECOMPILED_DIR}/{E4_EXPECTED}: {error}")
    })?;

    let target_dir = Fixture::target_dir_arg(&scratch_target)?;
    let manifest = scratch.join("Cargo.toml");
    let manifest = manifest
        .to_str()
        .ok_or("check-precompiled E4: the scratch path is not valid UTF-8")?;
    let output = run_capture_in(
        &mut fixture.cargo(&[
            "check",
            "--locked",
            "--color",
            "never",
            "--message-format",
            "json",
            "--manifest-path",
            manifest,
            "-p",
            "kernels",
            "--target-dir",
            target_dir,
        ]),
        "check-precompiled E4",
        "cargo",
    )?;
    let problems = judge(
        "kernels",
        "kernels",
        &Expectation::Fails(wanted.clone()),
        output.status.success(),
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    );
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E4 found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    Ok(format!(
        "passed: with the artifact files removed, `cargo check -p kernels` under {PROFILE_VARIABLE}={STRICT_PROFILE} failed with exactly one {} error at {}:{} ({PRECOMPILED_DIR}/{E4_EXPECTED})",
        wanted.code, wanted.file, wanted.line
    ))
}

// ---------------------------------------------------------------------------------------------------------------
// E5

/// E5: the fixture built with the dual cfg and the strict profile together, through `check-dual`'s own command
/// ([`precompiled_dual_steps`]), so the rustflags rules and the dual target directory of `check-dual` apply and the
/// profile comes from the step. Returns the summary.
pub(super) fn e5_strict_plus_dual(fixture: &Fixture) -> Result<String, String> {
    let directory = dual_target_dir(&fixture.root);
    let target_dir = Fixture::target_dir_arg(&directory)?;
    let steps = precompiled_dual_steps(target_dir);
    for step in &steps {
        println!("check-precompiled E5 {}: {}", step.id, step.shown());
        run_dual_step(&fixture.root, step)
            .map_err(|error| format!("check-precompiled E5 (strict plus dual): {error}"))?;
    }
    Ok(format!(
        "passed: the fixture built with `--cfg gust_dual` and {PROFILE_VARIABLE}={STRICT_PROFILE} together ({}), `app`'s unit test dispatched the CPU twin of the fixture's own kernel on `gust::cpu::CpuDevice` and it equalled the host reference (the test exists only under the cfg), and `cargo tree -e features -p app` shows no `slang-runtime`",
        steps
            .iter()
            .map(|step| step.id)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

#[cfg(test)]
#[path = "../tests/precompiled_artifacts.rs"]
mod tests;
