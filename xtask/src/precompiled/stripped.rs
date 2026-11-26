//! E1 and E2: the release `app` built and run with Slang taken out of the environment, and its binary scanned for the
//! strings of the run-time Slang bridge.

use std::{
    env::consts::EXE_SUFFIX,
    fs,
    path::{Path, PathBuf},
};

use super::{
    CONTROL_PROFILE, Mode, PRECOMPILED_DIR,
    environment::{directory_has_slang_tool, slang_free_problems},
    fixture::{Fixture, names_or_none, require_success},
};
use crate::process::{run_capture_in, slangc_cannot_be_found};

/// The line `app` prints after the GPU result equals its host reference.
pub(crate) const VERIFIED_PREFIX: &str = "verified: GPU result equals the host reference for ";

/// The line `app` prints after a launch past a lowered workgroup limit ran as pieces of the kernel's split artifact
/// (D208, D209; there is no run-time compiler in this binary) and its result equalled the host reference.
pub(crate) const SPLIT_PREFIX: &str = "split: a launch past a lowered workgroup limit ran as pieces of the split artifact and equals the host reference for ";

/// The line `app` prints when the same launch of a kernel whose split has no precompiled artifact is refused with
/// `DeviceLimit` (D208): this binary has no run-time compiler either, so nothing can run the pieces. The name of the
/// limit follows.
pub(crate) const SPLIT_REFUSED_PREFIX: &str = "split fallback: no split artifact and no run-time compiler, so the launch past the limit is refused with DeviceLimit on ";

/// The line `app` prints with the name of the adapter it ran on.
pub(crate) const ADAPTER_PREFIX: &str = "adapter: ";

/// The line `counted` (the counted-stage binary of the fixture's `app` package, m14.s2i) prints after its counted stage ran
/// from the precompiled artifacts and three counts equalled the host model.
pub(crate) const COUNTED_PREFIX: &str =
    "counted: a counted stage ran from its precompiled artifacts";

/// The line the same binary prints after its second batch, whose element is a struct (m14.s2s, D279), ran from the
/// precompiled artifacts and three frames equalled the host model.
pub(crate) const STRUCT_PREFIX: &str = "counted: a struct stage ran from its precompiled artifacts";
/// The byte strings that the run-time Slang bridge brings into a binary (E2). `gust-slang-reflect` (the name of the
/// native helper, `HELPER_NAME` of `gust::toolchain`) and `kernel.artifact` (`compile_reflected_with_helper`) are
/// string literals of code gated by the `slang-runtime` feature of `gust-compute`. `kernel.slang` is not gated:
/// `KERNEL_FILE` in `gust/src/slang/mod.rs` is used by the ungated `compile_copy` and `directive_file_is_usable` too,
/// so it is absent from the measured binary only because the release linker drops dead code. It stays a marker for
/// that measured reason and is the weakest of the three. They were picked by measurement: the same `app` built with the bridge (the
/// control of E2) holds all three, the one without holds none.
///
/// `slangc` and `GUST_SLANG_REFLECT` are not markers, although they look like the obvious ones. The ungated `Display`
/// of `slang::Error::CompilerNotFound` spells `slangc`, and the ungated `Display` of `reflect::Error::HelperUnavailable`
/// spells `GUST_SLANG_REFLECT`; `gust-wgpu`'s error type reaches both without the feature, and both were found in the
/// measured binary without the bridge.
pub(crate) const BRIDGE_MARKERS: [&str; 3] =
    ["gust-slang-reflect", "kernel.slang", "kernel.artifact"];

// ---------------------------------------------------------------------------------------------------------------
// E1

/// E1: release build and run of `app` with Slang taken out of the environment. Returns the summary.
pub(super) fn e1_stripped_build_and_run(fixture: &Fixture, mode: Mode) -> Result<String, String> {
    e1_build(fixture)?;
    if !mode.runs() {
        return Ok("passed (build only: this mode skips the run, which needs an adapter): the release `app` built with Slang taken out of the environment".to_owned());
    }
    e1_run(fixture, &Fixture::app_binary(&fixture.target_dir))
}

/// E1's check that the environment the children get has no Slang; `--adapter-runs` repeats it before the runs.
pub(super) fn check_slang_free(fixture: &Fixture) -> Result<(), String> {
    let problems = slang_free_problems(&fixture.environment, &directory_has_slang_tool);
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E1: the stripped environment still has Slang: {}",
            problems.join("; ")
        ));
    }
    slangc_cannot_be_found(&fixture.environment).map_err(|error| {
        format!("check-precompiled E1: the stripped environment still has Slang: {error}")
    })
}

/// The build half of E1: the stripped environment, the clean, and the release build.
fn e1_build(fixture: &Fixture) -> Result<(), String> {
    check_slang_free(fixture)?;
    println!(
        "check-precompiled E1: environment stripped ({} variable(s) removed: {}; {} PATH director{} removed); `slangc` is not found with it",
        fixture.report.variables.len(),
        names_or_none(&fixture.report.variables),
        fixture.report.directories.len(),
        if fixture.report.directories.len() == 1 {
            "y"
        } else {
            "ies"
        },
    );

    let target_dir = Fixture::target_dir_arg(&fixture.target_dir)?;
    let manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    // Cargo's fingerprints track `GUST_PROFILE` only, so in a warm target directory nothing would be compiled again
    // in the stripped environment. Clean every package that the build compiles from this repository.
    let output = run_capture_in(
        &mut fixture.cargo(&[
            "clean",
            "--release",
            "--color",
            "never",
            "--manifest-path",
            &manifest,
            "-p",
            "gust-compute",
            "-p",
            "gust-compute-macros",
            "-p",
            "kernels",
            "-p",
            "app",
            "--target-dir",
            target_dir,
        ]),
        "check-precompiled E1",
        "cargo",
    )?;
    require_success(
        "E1",
        "cargo clean -p gust-compute -p gust-compute-macros -p kernels -p app",
        &output,
    )?;
    let output = run_capture_in(
        &mut fixture.cargo(&[
            "build",
            "--release",
            "--locked",
            "--color",
            "never",
            "--manifest-path",
            &manifest,
            "-p",
            "app",
            "--target-dir",
            target_dir,
        ]),
        "check-precompiled E1",
        "cargo",
    )?;
    require_success("E1", "cargo build --release -p app", &output)?;
    println!(
        "check-precompiled E1: cargo clean of gust-compute, gust-compute-macros, kernels and app, then cargo build --release -p app, succeeded"
    );
    Ok(())
}

/// The counted-stage binary that E1 builds beside `binary` (`release/app`), `release/counted` (m14.s2i).
pub(super) fn counted_binary_of(binary: &Path) -> std::path::PathBuf {
    binary.with_file_name(format!("counted{EXE_SUFFIX}"))
}

/// The run half of E1: the binary built by [`e1_build`] on the adapter.
pub(super) fn e1_run(fixture: &Fixture, binary: &Path) -> Result<String, String> {
    let run = fixture.command(binary).output().map_err(|error| {
        format!(
            "check-precompiled E1: could not run {}: {error}",
            fixture.shown(binary)
        )
    })?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    print!("{stdout}");
    let report = judge_run(run.status.success(), &stdout, &stderr)
        .map_err(|problems| format!("check-precompiled E1: {}", problems.join("; ")))?;
    let counted = counted_binary_of(binary);
    let counted_run = fixture.command(&counted).output().map_err(|error| {
        format!(
            "check-precompiled E1: could not run the counted-stage binary {}: {error}",
            fixture.shown(&counted)
        )
    })?;
    let counted_out = String::from_utf8_lossy(&counted_run.stdout).into_owned();
    print!("{counted_out}");
    let counted_adapter = judge_counted(
        counted_run.status.success(),
        &counted_out,
        &String::from_utf8_lossy(&counted_run.stderr),
    )
    .map_err(|problems| format!("check-precompiled E1: {}", problems.join("; ")))?;
    Ok(format!(
        "passed: the release `app`, built and run with Slang taken out of the environment, printed `{VERIFIED_PREFIX}{} elements` on the adapter `{}`, and the release `counted` ran a counted stage from its precompiled artifacts on `{counted_adapter}`",
        report.elements, report.adapter
    ))
}

/// The problems of the run of the counted-stage binary in E1: it must exit 0, name its adapter and print the `counted:`
/// line (printed only after three counts, one above the capacity, equalled the host model) and the struct line (printed
/// only after three frames of a struct batch equalled the host model). Returns the adapter.
pub(crate) fn judge_counted(
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<String, Vec<String>> {
    let mut problems = Vec::new();
    if !success {
        problems.push(format!("`counted` failed; stderr was: {}", stderr.trim()));
    }
    let adapter = stdout
        .lines()
        .find_map(|line| line.strip_prefix(ADAPTER_PREFIX))
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty());
    if adapter.is_none() {
        problems.push(format!(
            "`counted` printed no `{ADAPTER_PREFIX}` line with a name"
        ));
    }
    if !stdout.lines().any(|line| line.starts_with(COUNTED_PREFIX)) {
        problems.push(format!(
            "`counted` printed no `{COUNTED_PREFIX}` line, so a counted stage did not run from its precompiled artifacts"
        ));
    }
    if !stdout.lines().any(|line| line.starts_with(STRUCT_PREFIX)) {
        problems.push(format!(
            "`counted` printed no `{STRUCT_PREFIX}` line, so a struct stage did not run from its precompiled artifacts"
        ));
    }
    match (problems.is_empty(), adapter) {
        (true, Some(adapter)) => Ok(adapter),
        _ => Err(problems),
    }
}

/// What a successful run of `app` printed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RunReport {
    pub(crate) adapter: String,
    pub(crate) elements: String,
}

/// The problems of the run of `app` in E1: it must exit 0, name its adapter and a WGSL artifact, and print the
/// `verified:` line and the `split:` line (D209). Returns what it printed.
pub(crate) fn judge_run(
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<RunReport, Vec<String>> {
    let mut problems = Vec::new();
    if !success {
        problems.push(format!("`app` failed; stderr was: {}", stderr.trim()));
    }
    let adapter = stdout
        .lines()
        .find_map(|line| line.strip_prefix(ADAPTER_PREFIX))
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty());
    if adapter.is_none() {
        problems.push(format!(
            "`app` printed no `{ADAPTER_PREFIX}` line with a name"
        ));
    }
    if !stdout
        .lines()
        .any(|line| line.starts_with("artifact: Wgsl "))
    {
        problems.push("`app` printed no `artifact: Wgsl` line, so the kernel did not carry its precompiled artifact".to_owned());
    }
    let elements = stdout
        .lines()
        .find_map(|line| line.strip_prefix(VERIFIED_PREFIX))
        .map(|rest| rest.trim_end_matches(" elements").trim().to_owned())
        .filter(|count| count.parse::<u64>().is_ok_and(|count| count > 0));
    if elements.is_none() {
        problems.push(format!(
            "`app` printed no `{VERIFIED_PREFIX}<n> elements` line"
        ));
    }
    let split = stdout
        .lines()
        .find_map(|line| line.strip_prefix(SPLIT_PREFIX))
        .map(|rest| rest.trim_end_matches(" elements").trim())
        .filter(|count| count.parse::<u64>().is_ok_and(|count| count > 0));
    if split.is_none() {
        problems.push(format!(
            "`app` printed no `{SPLIT_PREFIX}<n> elements` line, so a launch past a workgroup limit did not run from the split artifact"
        ));
    }
    if !stdout.lines().any(|line| {
        line.strip_prefix(SPLIT_REFUSED_PREFIX)
            .is_some_and(|rest| !rest.trim().is_empty())
    }) {
        problems.push(format!(
            "`app` printed no `{SPLIT_REFUSED_PREFIX}<limit>` line, so a launch past a workgroup limit with no split artifact and no compiler was not refused"
        ));
    }
    match (problems.is_empty(), adapter, elements) {
        (true, Some(adapter), Some(elements)) => Ok(RunReport { adapter, elements }),
        _ => Err(problems),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// E2

/// E2: the binary of E1, `cargo tree -e features`, and the manifests. Returns the summary.
pub(super) fn e2_binary_and_features(fixture: &Fixture) -> Result<String, String> {
    let binary = Fixture::app_binary(&fixture.target_dir);
    let bytes = fs::read(&binary).map_err(|error| {
        format!(
            "check-precompiled E2: could not read {}: {error}",
            fixture.shown(&binary)
        )
    })?;
    let present = markers_present(&marker_counts(&bytes, &BRIDGE_MARKERS));
    if !present.is_empty() {
        return Err(format!(
            "check-precompiled E2: the release binary ({} bytes) contains the run-time Slang bridge: {}",
            bytes.len(),
            present.join(", ")
        ));
    }
    println!(
        "check-precompiled E2: the release binary ({} bytes) holds none of {}",
        bytes.len(),
        quoted(&BRIDGE_MARKERS, ", ")
    );

    // The control: the scan above finds nothing in a binary without the bridge, which proves nothing unless it finds
    // all markers in a binary with it.
    let control = build_control(fixture)?;
    let control_bytes = fs::read(&control).map_err(|error| {
        format!(
            "check-precompiled E2: could not read the control binary {}: {error}",
            fixture.shown(&control)
        )
    })?;
    let control_counts = marker_counts(&control_bytes, &BRIDGE_MARKERS);
    let blind = markers_missing(&control_counts);
    if !blind.is_empty() {
        return Err(format!(
            "check-precompiled E2: the scan is blind: the control binary ({} bytes, built with `gust-wgpu/slang-runtime`) lacks {}, so {} says nothing about the release binary",
            control_bytes.len(),
            blind.join(", "),
            if blind.len() == 1 {
                "that marker"
            } else {
                "those markers"
            }
        ));
    }
    println!(
        "check-precompiled E2: control: the same app built with `gust-wgpu/slang-runtime` ({} bytes) holds every marker ({})",
        control_bytes.len(),
        control_counts
            .iter()
            .map(|(marker, count)| format!("`{marker}` x{count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let output = run_capture_in(
        &mut fixture.cargo(&[
            "tree",
            "--locked",
            "--color",
            "never",
            "-e",
            "features",
            "--manifest-path",
            &manifest,
            "-p",
            "app",
        ]),
        "check-precompiled E2",
        "cargo",
    )?;
    require_success("E2", "cargo tree -e features -p app", &output)?;
    let tree = String::from_utf8_lossy(&output.stdout).into_owned();
    let problems = tree_problems(&tree);
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E2: cargo tree -e features -p app: {}",
            problems.join("; ")
        ));
    }

    let mut manifest_problems = Vec::new();
    for (file, dependencies) in [
        ("app/Cargo.toml", &["gust", "gust-wgpu"][..]),
        ("kernels/Cargo.toml", &["gust"][..]),
    ] {
        let path = fixture.root.join(PRECOMPILED_DIR).join(file);
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("check-precompiled E2: {PRECOMPILED_DIR}/{file}: {error}"))?;
        for dependency in dependencies {
            if !dependency_disables_defaults(&text, dependency) {
                manifest_problems.push(format!(
                    "{PRECOMPILED_DIR}/{file} must declare `{dependency}` on one line with `default-features = false` and without the `slang-runtime` feature"
                ));
            }
        }
    }
    if !manifest_problems.is_empty() {
        return Err(format!(
            "check-precompiled E2: {}",
            manifest_problems.join("; ")
        ));
    }
    Ok(format!(
        "passed: the release `app` binary ({} bytes) contains none of {} (the same app built with the bridge holds all of them), `cargo tree -e features -p app` ({} lines) shows no `slang-runtime`, and the manifests declare `gust` and `gust-wgpu` with `default-features = false`",
        bytes.len(),
        quoted(&BRIDGE_MARKERS, ", "),
        tree.lines().count()
    ))
}

/// Builds `app` once more with the bridge (the control of E2) and returns the binary. The profile is `dev` because a
/// strict one refuses the feature, and the target directory is its own so that the build of E1 is never invalidated.
fn build_control(fixture: &Fixture) -> Result<PathBuf, String> {
    let directory = fixture.control_dir();
    let target_dir = directory
        .to_str()
        .ok_or("check-precompiled E2: the control target directory is not valid UTF-8")?;
    let manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let output = run_capture_in(
        &mut fixture.cargo_under_profile(
            &[
                "build",
                "--release",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "app",
                "--features",
                "gust-wgpu/slang-runtime",
                "--target-dir",
                target_dir,
            ],
            CONTROL_PROFILE,
        ),
        "check-precompiled E2",
        "cargo",
    )?;
    require_success(
        "E2",
        "cargo build --release -p app --features gust-wgpu/slang-runtime (the control)",
        &output,
    )?;
    Ok(directory.join("release").join(format!("app{EXE_SUFFIX}")))
}

/// The markers with a count above zero, each written with its count.
pub(crate) fn markers_present(counts: &[(&str, usize)]) -> Vec<String> {
    counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(marker, count)| format!("`{marker}` x{count}"))
        .collect()
}

/// The markers that do not occur.
pub(crate) fn markers_missing(counts: &[(&str, usize)]) -> Vec<String> {
    counts
        .iter()
        .filter(|(_, count)| *count == 0)
        .map(|(marker, _)| format!("`{marker}`"))
        .collect()
}

fn quoted(markers: &[&str], separator: &str) -> String {
    markers
        .iter()
        .map(|marker| format!("`{marker}`"))
        .collect::<Vec<_>>()
        .join(separator)
}

/// How often each marker occurs in `bytes`, in the order of `markers`.
pub(crate) fn marker_counts<'a>(bytes: &[u8], markers: &[&'a str]) -> Vec<(&'a str, usize)> {
    markers
        .iter()
        .map(|marker| (*marker, count_occurrences(bytes, marker.as_bytes())))
        .collect()
}

/// How often `needle` occurs in `haystack`, not overlapping. An empty needle occurs never.
pub(crate) fn count_occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    let Some((first, rest)) = needle.split_first() else {
        return 0;
    };
    let mut count = 0;
    let mut index = 0;
    while index < haystack.len() {
        let Some(offset) = haystack[index..].iter().position(|byte| byte == first) else {
            break;
        };
        let start = index + offset;
        if haystack[start + 1..].starts_with(rest) {
            count += 1;
            index = start + needle.len();
        } else {
            index = start + 1;
        }
    }
    count
}

/// The problems of the output of `cargo tree -e features -p app`: the feature must not appear, and the tree must show
/// the two crates that carry it, so that an empty or unreadable output cannot pass.
pub(crate) fn tree_problems(tree: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for line in tree.lines().filter(|line| line.contains("slang-runtime")) {
        problems.push(format!("the feature is on: `{}`", line.trim()));
    }
    for package in ["gust-compute", "gust-compute-wgpu", "kernels"] {
        let listed = tree.lines().any(|line| {
            line.trim_start_matches(|c: char| !c.is_alphanumeric())
                .starts_with(&format!("{package} v"))
        });
        if !listed {
            problems.push(format!(
                "the tree has no `{package} v..` line, so it was not read correctly"
            ));
        }
    }
    problems
}

/// Whether `manifest` declares `dependency` inline with `default-features = false` and without the `slang-runtime`
/// feature. Only the one-line form (`name = { ... }`) is read; a table form is reported as missing, which is the safe
/// answer for a guard.
pub(crate) fn dependency_disables_defaults(manifest: &str, dependency: &str) -> bool {
    manifest.lines().any(|line| {
        let line = line.trim();
        let Some(rest) = line.strip_prefix(dependency) else {
            return false;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            return false;
        };
        let compact: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
        compact.starts_with('{')
            && compact.contains("default-features=false")
            && !compact.contains("slang-runtime")
    })
}

#[cfg(test)]
#[path = "../tests/precompiled_stripped.rs"]
mod tests;
