//! E7: the macro expands the `#[gpu_module]` module again when `GUST_EXPORT_DIR` changes, and `cargo gust build` is
//! reproducible.

use std::{
    collections::BTreeMap,
    env::consts::EXE_SUFFIX,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use super::{
    PRECOMPILED_DIR,
    fixture::{Fixture, copy_fixture, require_success},
};
use crate::{
    checks::RUSTFLAGS_VARIABLES,
    json::{self, Json},
    process::{PROFILE_VARIABLE, exit_code, run_capture_in, set_workspace_profile},
};

// ---------------------------------------------------------------------------------------------------------------
// E7

/// What a step of E7a expects of the expansion of `kernels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Expansion {
    /// No expectation (the first build only warms the target directory).
    Any,
    /// `kernels` is fresh: cargo did not compile it again.
    Fresh,
    /// `kernels` is compiled again, and, when a record directory is set, the macro wrote its records.
    Again,
}

/// One `cargo check -p kernels` of E7a.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExportStep {
    pub(crate) label: &'static str,
    /// The record directory (a name below the scratch records directory) that `GUST_EXPORT_DIR` is set to; `None`
    /// leaves the variable unset.
    pub(crate) export: Option<&'static str>,
    pub(crate) expect: Expansion,
}

/// The steps of E7a. The macro emits `option_env!("GUST_EXPORT_DIR")` in every expansion, so cargo tracks the variable
/// and expands the `#[gpu_module]` module again when its value changes (the nonce of `cargo gust build`, R2-3), and not
/// when it does not. `gust`'s build script tracks `GUST_PROFILE` only, so nothing else could explain the second kind.
pub(crate) fn e7_steps() -> [ExportStep; 6] {
    [
        ExportStep {
            label: "variable unset (warm-up)",
            export: None,
            expect: Expansion::Any,
        },
        ExportStep {
            label: "variable unset again (control: nothing changed)",
            export: None,
            expect: Expansion::Fresh,
        },
        ExportStep {
            label: "GUST_EXPORT_DIR set to a new directory",
            export: Some("a"),
            expect: Expansion::Again,
        },
        ExportStep {
            label: "the same value again",
            export: Some("a"),
            expect: Expansion::Fresh,
        },
        ExportStep {
            label: "GUST_EXPORT_DIR set to another new directory",
            export: Some("b"),
            expect: Expansion::Again,
        },
        ExportStep {
            label: "variable unset again",
            export: None,
            expect: Expansion::Again,
        },
    ]
}

/// The `fresh` flag of every `compiler-artifact` message of the target `name` in the JSON output of cargo.
pub(crate) fn artifact_fresh_flags(stdout: &str, name: &str) -> Vec<bool> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('{'))
        .filter_map(|line| json::parse(line).ok())
        .filter(|message| {
            message.get("reason").and_then(Json::as_str) == Some("compiler-artifact")
                && message
                    .get("target")
                    .and_then(|target| target.get("name"))
                    .and_then(Json::as_str)
                    == Some(name)
        })
        .filter_map(|message| match message.get("fresh") {
            Some(Json::Bool(fresh)) => Some(*fresh),
            _ => None,
        })
        .collect()
}

/// The problem of a step of E7a for the `fresh` flags it saw, or `None`. Exactly one message must name the target,
/// else the observation is blind.
pub(crate) fn expansion_problem(step: &ExportStep, flags: &[bool]) -> Option<String> {
    let [fresh] = flags else {
        return Some(format!(
            "{}: expected one compiler-artifact message for `kernels`, saw {}, so the observation is blind",
            step.label,
            flags.len()
        ));
    };
    match (step.expect, *fresh) {
        (Expansion::Fresh, false) => Some(format!(
            "{}: `kernels` was compiled again, but nothing it tracks changed",
            step.label
        )),
        (Expansion::Again, true) => Some(format!(
            "{}: `kernels` was fresh, but the value of `GUST_EXPORT_DIR` changed, so cargo must expand its `#[gpu_module]` module again",
            step.label
        )),
        _ => None,
    }
}

/// The number of files below `directory` that end in `.slang` and in `.kernel.json`.
fn record_counts(directory: &Path) -> (usize, usize) {
    let mut slang = 0;
    let mut json = 0;
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".kernel.json") {
                json += 1;
            } else if name.ends_with(".slang") {
                slang += 1;
            }
        }
    }
    (slang, json)
}

/// E7a: a change of `GUST_EXPORT_DIR` expands the `#[gpu_module]` module again. Returns the summary.
pub(super) fn e7_reexpansion(fixture: &Fixture) -> Result<String, String> {
    let copy = fixture.scratch.join("e7-expand");
    copy_fixture(fixture, &copy)?;
    let records = fixture.scratch.join("e7-records");
    if records.exists() {
        fs::remove_dir_all(&records).map_err(|error| format!("check-precompiled E7: {error}"))?;
    }
    let target = fixture.scratch.join("e7-target");
    let target_dir = Fixture::target_dir_arg(&target)?;
    let manifest = copy.join("Cargo.toml");
    let manifest = manifest
        .to_str()
        .ok_or("check-precompiled E7: the scratch path is not valid UTF-8")?;

    let mut problems = Vec::new();
    for (number, step) in e7_steps().iter().enumerate() {
        let mut command = fixture.cargo(&[
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
        ]);
        let export = step.export.map(|name| records.join(name));
        if let Some(export) = &export {
            fs::create_dir_all(export).map_err(|error| format!("check-precompiled E7: {error}"))?;
            command.env("GUST_EXPORT_DIR", export);
        }
        let output = run_capture_in(&mut command, "check-precompiled E7", "cargo")?;
        require_success("E7", "cargo check -p kernels", &output)?;
        let flags = artifact_fresh_flags(&String::from_utf8_lossy(&output.stdout), "kernels");
        let mut found = expansion_problem(step, &flags);
        if found.is_none()
            && step.expect == Expansion::Again
            && let Some(export) = &export
        {
            let (slang, json) = record_counts(&export.join("kernels"));
            if slang == 0 || json == 0 {
                found = Some(format!(
                    "{}: `kernels` was expanded again, but the macro wrote {slang} `.slang` and {json} `.kernel.json` record(s) under the export directory, so it did not see the variable",
                    step.label
                ));
            }
        }
        println!(
            "check-precompiled E7 step {}: {}: kernels {}{}",
            number + 1,
            step.label,
            match flags.as_slice() {
                [true] => "fresh",
                [false] => "compiled again",
                _ => "unobserved",
            },
            if found.is_none() {
                ""
            } else {
                " (NOT AS EXPECTED)"
            }
        );
        problems.extend(found);
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E7 found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    Ok("passed (re-expansion): cargo compiled `kernels` again exactly when `GUST_EXPORT_DIR` changed (set to a new directory, set to another, unset), not when it stayed the same, and the macro wrote its `.slang` and `.kernel.json` records for each new directory".to_owned())
}

/// The files below `directory` as relative paths (forward slashes) and their bytes.
pub(crate) fn snapshot(directory: &Path) -> io::Result<BTreeMap<String, Vec<u8>>> {
    fn walk(base: &Path, directory: &Path, into: &mut BTreeMap<String, Vec<u8>>) -> io::Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(base, &path, into)?;
            } else {
                let relative = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                into.insert(relative, fs::read(&path)?);
            }
        }
        Ok(())
    }
    let mut files = BTreeMap::new();
    walk(directory, directory, &mut files)?;
    Ok(files)
}

/// How two snapshots differ, one line per file; empty when they are byte-identical.
pub(crate) fn snapshot_differences(
    first: &BTreeMap<String, Vec<u8>>,
    second: &BTreeMap<String, Vec<u8>>,
) -> Vec<String> {
    let mut found = Vec::new();
    for (name, bytes) in first {
        match second.get(name) {
            None => found.push(format!("`{name}` is only in the first")),
            Some(other) if other != bytes => found.push(format!(
                "`{name}` differs ({} and {} bytes)",
                bytes.len(),
                other.len()
            )),
            Some(_) => {}
        }
    }
    for name in second.keys().filter(|name| !first.contains_key(*name)) {
        found.push(format!("`{name}` is only in the second"));
    }
    found
}

/// The number N of the tool's summary line `.. N file(s) written, M unchanged`, if the output has one.
pub(crate) fn files_written(stdout: &str) -> Option<usize> {
    stdout.lines().find_map(|line| {
        let (before, _) = line.split_once(" file(s) written")?;
        before.rsplit(' ').next()?.parse().ok()
    })
}

/// The problems of the output of one `cargo gust build` run: it must have succeeded and must not have pruned.
pub(crate) fn gust_build_problems(
    label: &str,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    if !success {
        problems.push(format!(
            "{label}: `cargo gust build` failed:\n{stdout}{stderr}"
        ));
    }
    for line in stdout.lines().chain(stderr.lines()) {
        if line.contains("pruned") {
            problems.push(format!(
                "{label}: the tool pruned something: `{}`",
                line.trim()
            ));
        }
    }
    problems
}

/// Builds `cargo-gust` in the repository workspace (under `dev`, as every workspace child) and returns the binary.
fn build_cargo_gust(fixture: &Fixture) -> Result<PathBuf, String> {
    let mut build = Command::new(&fixture.cargo);
    build
        .current_dir(&fixture.root)
        .args(["build", "--locked", "--color", "never", "-p", "cargo-gust"])
        .stdin(Stdio::null());
    set_workspace_profile(&mut build);
    let output = run_capture_in(&mut build, "check-precompiled E7", "cargo")?;
    require_success("E7", "cargo build -p cargo-gust", &output)?;
    let binary = fixture
        .target
        .join("debug")
        .join(format!("cargo-gust{EXE_SUFFIX}"));
    if !binary.is_file() {
        return Err(format!(
            "check-precompiled E7: cargo built cargo-gust, but {} does not exist",
            fixture.shown(&binary)
        ));
    }
    Ok(binary)
}

/// E7b: `cargo gust build` twice on a copy of the fixture without its artifacts gives byte-identical output and no prune message.
/// It needs Slang, so `--check-only` skips it. Returns the summary.
pub(super) fn e7_build_twice(fixture: &Fixture, check_only: bool) -> Result<String, String> {
    if check_only {
        return Ok("skipped by --check-only (`cargo gust build` needs Slang): the full mode runs it on a machine with Slang".to_owned());
    }
    let tool = build_cargo_gust(fixture)?;
    let copy = fixture.scratch.join("e7-gust");
    copy_fixture(fixture, &copy)?;
    let manifest = copy.join("kernels").join("Cargo.toml");
    let artifacts = copy.join("kernels").join("gust-artifacts");
    let committed = snapshot(
        &fixture
            .root
            .join(PRECOMPILED_DIR)
            .join("kernels")
            .join("gust-artifacts"),
    )
    .map_err(|error| format!("check-precompiled E7: {error}"))?;

    // The copy starts without its artifacts, so run 1 has to write all of them and run 2 has to find them unchanged:
    // a rebuild that wrote nothing in both runs would prove only that the committed files equal themselves.
    fs::remove_dir_all(&artifacts).map_err(|error| format!("check-precompiled E7: {error}"))?;

    let mut snapshots = Vec::new();
    let mut problems = Vec::new();
    for run in 1..=2 {
        let mut command = Command::new(&tool);
        command
            .current_dir(&copy)
            .stdin(Stdio::null())
            .arg("build")
            .arg("--manifest-path")
            .arg(&manifest);
        // The tool sets the profile of its collect step itself; a target directory of the parent would put the build
        // into the repository's cache, so the copy gets its own `target`.
        for name in RUSTFLAGS_VARIABLES.iter().copied().chain([
            PROFILE_VARIABLE,
            "GUST_EXPORT_DIR",
            "CARGO_TARGET_DIR",
        ]) {
            command.env_remove(name);
        }
        let output = run_capture_in(&mut command, "check-precompiled E7", "cargo-gust")?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        println!(
            "check-precompiled E7 cargo gust build, run {run}: exit {}",
            exit_code(output.status)
        );
        for line in stdout.lines().filter(|line| line.starts_with("gust:")) {
            println!("  {line}");
        }
        let mut found = gust_build_problems(
            &format!("run {run}"),
            output.status.success(),
            &stdout,
            &stderr,
        );
        match (run, files_written(&stdout)) {
            (1, Some(0)) => found.push("run 1: the tool wrote no file into a directory without artifacts, so the comparison below is vacuous".to_owned()),
            (2, Some(0)) | (1, Some(_)) => {}
            (2, other) => found.push(format!("run 2: the tool wrote {other:?} file(s) into an unchanged directory, not 0")),
            (_, None) => found.push(format!("run {run}: the tool printed no `N file(s) written` line")),
            _ => {}
        }
        let failed = !found.is_empty();
        problems.extend(found);
        if failed {
            break;
        }
        snapshots
            .push(snapshot(&artifacts).map_err(|error| format!("check-precompiled E7: {error}"))?);
    }
    if problems.is_empty()
        && let [first, second] = snapshots.as_slice()
    {
        if !first.contains_key("manifest.json") || !first.keys().any(|name| name.ends_with(".wgsl"))
        {
            problems.push("the output of the first run has no manifest.json or no .wgsl file, so the comparison is vacuous".to_owned());
        }
        for difference in snapshot_differences(first, second) {
            problems.push(format!(
                "the second run differs from the first: {difference}"
            ));
        }
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E7 (cargo gust build twice) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    let first = &snapshots[0];
    let against_committed = snapshot_differences(&committed, first);
    Ok(format!(
        "passed (build twice): `cargo gust build` ran twice on a copy whose artifacts were deleted first (run 1 wrote them, run 2 wrote nothing); both outputs are byte-identical ({} files, no `pruned` line); against the committed artifacts: {}",
        first.len(),
        if against_committed.is_empty() {
            "identical".to_owned()
        } else {
            format!(
                "DIFFERENT ({}), which `cargo gust check` judges, not E7",
                against_committed.join("; ")
            )
        }
    ))
}

#[cfg(test)]
#[path = "../tests/precompiled_reexpand.rs"]
mod tests;
