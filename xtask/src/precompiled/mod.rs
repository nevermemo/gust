//! `cargo xtask check-precompiled [--check-only | --prepare | --adapter-runs] [--work-dir PATH]`: the evidence that a
//! production-shaped consumer of precompiled kernels needs no Slang (D45, D46). The fixture is `tests/precompiled`, a
//! workspace of its own like `tests/dual`: the `kernels` crate holds one `#[gpu_module]` vector-add and its committed
//! `gust-artifacts/` (written by `cargo gust build`), and `app` dispatches it through `gust-wgpu` with
//! `default-features = false` and checks the result.
//!
//! Every cargo child runs under `GUST_PROFILE=native`, the strict profile, set explicitly (R4.2), except the control
//! build of E2, which needs the bridge and runs under `dev` in its own target directory, the profile checks of E6
//! (their point is the profile), the dual steps of E5 (they run through `check-dual`'s own command, which sets the
//! profile from the step) and `cargo gust build` of E7, an author tool that needs Slang. The checks:
//!
//! - **E1**: release build and run of `app` in an environment with Slang taken out. The child gets a copy of the
//!   parent's environment without every variable that names Slang (any name containing `SLANG`, `VULKAN_SDK`,
//!   `VK_SDK_PATH`) and without every `PATH` directory that holds `slangc` or `gust-slang-reflect`. The environment
//!   is then asserted slang-free: no such variable, no such directory, and the platform's lookup (`where` under `cmd`,
//!   `command -v` under `sh`) does not find `slangc` with the child's `PATH`. The build starts from `cargo clean -p gust-compute
//!   -p gust-compute-macros -p kernels -p app --release` in the same environment, because cargo's fingerprints track only
//!   `GUST_PROFILE`, so a warm target directory would relink nothing and prove nothing about the stripped build. `app`
//!   must print the `verified:` line, which it prints only after it read the GPU result back and compared it with a
//!   host reference, and the line `adapter:` that names the GPU it ran on.
//! - **E2**: the release binary of E1 holds none of the strings that only the run-time Slang bridge contains
//!   ([`BRIDGE_MARKERS`]), `cargo tree -e features` of `app` shows no `slang-runtime`, and the fixture's manifests keep
//!   `default-features = false` on `gust` and `gust-wgpu`. A control keeps the scan from passing vacuously: the same
//!   `app` is built under `GUST_PROFILE=dev` with `--features gust-wgpu/slang-runtime` into its own target directory
//!   (`<target>/precompiled-control`, built, never run), and its binary must hold every marker.
//! - **E3**: in a scratch copy of the fixture under the target directory, one operator of the committed WGSL is
//!   changed (`+` to `-`, so that an unchecked artifact would compute a wrong answer). It builds, and at run time
//!   `app` ends with the artifact-identity error of D17 and prints no `verified:` line.
//! - **E4** (E6f of the plan): in another scratch copy, the artifact files are deleted. Under `native` the build fails
//!   with the one error of `gust::profile` (E0080), reported at the line of the kernel in `kernels/src/lib.rs`, which
//!   `tests/precompiled/e4-expected.txt` pins.
//! - **E5**: strict plus dual (R4.2). The fixture is built through `check-dual`'s own command
//!   ([`crate::checks::precompiled_dual_steps`]) with `--cfg gust_dual` and `GUST_PROFILE=native` together; `app`'s unit
//!   test dispatches the CPU twin of the fixture's kernel and compares it with the host reference, and `cargo tree -e
//!   features` shows no `slang-runtime`. `check-dual` runs the same steps.
//! - **E6a to E6e**: the refusals of `gust`'s build script, each judged on its exact message (the `build-error:` lines
//!   of `e6-native-expected.txt` and `e6-inferred-expected.txt`, judged by [`crate::ui::judge`]) and never on the exit
//!   status alone: a `native` build with `gust/slang-runtime` (E6a), with `gust-wgpu/slang-runtime` (E6b, feature
//!   unification), and with a dependency crate that enables the feature (E6c, in a copy, also in a release build with
//!   no variable). E6d (a release build with no variable and the artifacts present) and E6e (a release build under
//!   `GUST_PROFILE=dev` with the feature) are the positive controls and must compile.
//! - **E6g to E6j**: the internal feature `cargo-gust-bridge` of `gust` (D108, mr.c). With it, a release build without
//!   `GUST_PROFILE` is `dev` and prints the warning of `e6-bridge-expected.txt` (E6g: `app` with
//!   `gust/cargo-gust-bridge` and `gust/slang-runtime`, which the plain inference of E6c-release refuses); an explicit
//!   `GUST_PROFILE=native` still refuses the bridge (E6h, the text of `e6-native-expected.txt`); and (R1 S8) the
//!   release check of the whole repository workspace without the variable unifies the feature through `cargo-gust`,
//!   so it is `dev` and prints the same warning (E6i, `--workspace` where the other cases check `-p app`; it is a
//!   `cargo check` because the build script that the case judges runs in a check as in a build). E6j (D117): in a
//!   copy, `app` has `cargo-gust` as a normal dependency, a mistake that cargo warns about and ignores
//!   (`cargo-warning:` in the expectation) but that still unifies the feature, so the release build is `dev` with the
//!   same warning. Each warning is matched as a whole line.
//! - **E7**: a change of `GUST_EXPORT_DIR`, which the macro tracks through `option_env!`, makes cargo expand the
//!   `#[gpu_module]` module of `kernels` again (and the macro writes its records), and the same value does not. Then,
//!   with Slang, `cargo gust build` runs twice on a copy of the fixture whose artifacts were deleted first: run 1 writes them, run 2 (the run on unchanged input) writes nothing, and the output is byte-identical and the
//!   tool prints no `pruned` line (R2-3, S3).
//! - **E8**: distribution (R2-4, R4.6). `cargo package` of `gust-compute-macros`, `gust-compute`, `gust-compute-wgpu` and `kernels` must list
//!   only allowed files; the four packages are unpacked outside the repository and a consumer with plain registry-style
//!   dependencies and a `[patch.crates-io]` on the four trees is built under `native` in the stripped environment of E1
//!   and, unless `--check-only`, run on the GPU. The packaging of `kernels` runs with a `[patch.crates-io]` of its own,
//!   given with `--config`, for `gust-compute` and `gust-compute-macros` onto this checkout, so that its lock resolves
//!   these trees and not whatever crates.io holds under a GUST name (mr.n: before the rename it was the unrelated
//!   charting crate `gust`).
//!
//! `--check-only` skips the parts that need an adapter (the runs of E1, E3 and E8) and Slang (the second half of E7);
//! it is what the CI job `precompiled` runs. `--prepare` and `--adapter-runs` split the full check in two (see
//! [`Mode`]).
//! Nothing but `cargo gust build` of E7 runs `slangc` or the native helper.
//! The environment of the xtask process is never changed; the stripped environment exists only in the children.
//! `CARGO_NET_OFFLINE` and the rest of the parent's environment pass through. E8 needs the registry (the consumer
//! needs `wgpu` from crates.io), so it is no offline proof.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use crate::{process::PROFILE_VARIABLE, validation::sha256_hex, workspace::workspace_root};

mod artifacts;
mod distribution;
mod environment;
mod fixture;
#[cfg(test)]
#[path = "../tests/precompiled_modes.rs"]
mod modes_tests;
mod profiles;
mod reexpand;
mod refusals;
mod stripped;
#[cfg(test)]
#[path = "../tests/precompiled_support.rs"]
mod test_support;

use self::{
    artifacts::{e3_mutated_artifact, e3_run, e4_missing_artifact, e5_strict_plus_dual},
    distribution::{e8_distribution, e8_run, kept_consumer},
    fixture::Fixture,
    reexpand::{e7_build_twice, e7_reexpansion},
    refusals::e6_refusals,
    stripped::{
        VERIFIED_PREFIX, check_slang_free, e1_run, e1_stripped_build_and_run,
        e2_binary_and_features,
    },
};
pub(crate) use self::{
    distribution::{is_inside, lock_packages, unused_patch_lines},
    environment::{StripReport, child_environment, directory_has_slang_tool, slang_free_problems},
    reexpand::snapshot,
    stripped::{count_occurrences, tree_problems},
};

/// The fixture workspace, relative to the repository root.
pub(crate) const PRECOMPILED_DIR: &str = "tests/precompiled";

/// The profile of every child: strict, so that a missing artifact is a build error and the bridge is refused.
const STRICT_PROFILE: &str = "native";

/// The profile of the control build of E2, the only child that is not strict: the bridge needs a development profile.
const CONTROL_PROFILE: &str = "dev";

/// Which part of `cargo xtask check-precompiled` runs. Each step is in exactly one of `Prepare` and `AdapterRuns`, and
/// the two together are `Full`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// No flag: every step, builds and runs.
    #[default]
    Full,
    /// `--check-only`: skip what needs an adapter (the runs of E1, E3 and E8) and Slang (the second half of E7).
    CheckOnly,
    /// `--prepare`: no adapter, but Slang: every step but the three runs, E7b included; E8's consumer is copied out of
    /// its temporary directory to `<target>/precompiled-dist`; then a stamp (HEAD, a hash of the working tree and a
    /// hash of each of the three binaries) that `AdapterRuns` reads.
    Prepare,
    /// `--adapter-runs`: only the three runs of E1, E3 and E8, on the binaries `Prepare` left and only if the stamp
    /// still matches; no build and no `cargo clean`.
    AdapterRuns,
}

impl Mode {
    /// Whether the runs of E1, E3 and E8, which need an adapter, run.
    pub(crate) const fn runs(self) -> bool {
        matches!(self, Self::Full | Self::AdapterRuns)
    }

    /// Whether E7b (`cargo gust build` twice), which needs Slang, runs.
    pub(crate) const fn slang(self) -> bool {
        matches!(self, Self::Full | Self::Prepare)
    }
}

/// What `cargo xtask check-precompiled` was asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) mode: Mode,
    /// `--work-dir PATH`: where E8 builds its consumer; a directory that does not exist yet or is empty.
    pub(crate) work_dir: Option<PathBuf>,
}

/// The options of `check-precompiled`.
pub(crate) fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut rest = args.iter();
    while let Some(argument) = rest.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (argument.as_str(), None),
        };
        match name {
            "--check-only" | "--prepare" | "--adapter-runs" => {
                if inline.is_some() {
                    return Err(format!("`{name}` takes no value"));
                }
                if options.mode != Mode::Full {
                    return Err("`--check-only`, `--prepare` and `--adapter-runs` exclude each other and can be given once".into());
                }
                options.mode = match name {
                    "--check-only" => Mode::CheckOnly,
                    "--prepare" => Mode::Prepare,
                    _ => Mode::AdapterRuns,
                };
            }
            "--work-dir" => {
                let value = match inline {
                    Some(value) => value,
                    None => rest
                        .next()
                        .cloned()
                        .ok_or("`--work-dir` needs a directory")?,
                };
                if value.is_empty() {
                    return Err("`--work-dir` needs a non-empty directory".into());
                }
                if options.work_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("`--work-dir` was given twice".into());
                }
            }
            _ => {
                return Err(format!(
                    "check-precompiled takes only `--check-only`, `--prepare`, `--adapter-runs` and `--work-dir PATH`, not `{argument}`"
                ));
            }
        }
    }
    if options.mode == Mode::AdapterRuns && options.work_dir.is_some() {
        return Err("`--adapter-runs` builds nothing, so it takes no `--work-dir`".into());
    }
    Ok(options)
}

/// What `--prepare` leaves in `<target>/precompiled-prepared.stamp` for `--adapter-runs`.
const STAMP_FILE: &str = "precompiled-prepared.stamp";

/// The three binaries the runs need (E1's and E3's `app`, E8's kept consumer), with the stamp key of each.
fn stamped_binaries(fixture: &Fixture) -> [(&'static str, PathBuf); 3] {
    [
        (
            "sha256 of the E1 app",
            Fixture::app_binary(&fixture.target_dir),
        ),
        (
            "sha256 of the E3 app",
            Fixture::app_binary(&fixture.scratch_target_dir()),
        ),
        ("sha256 of the E8 consumer", kept_consumer(fixture)),
    ]
}

/// The stamp: HEAD, the tree hash and the hash of each binary, one `key=value` line each.
fn stamp_text(fixture: &Fixture, head: &str, tree: &str) -> Result<String, String> {
    let mut text = format!("head={head}\ntree={tree}\n");
    for (key, path) in stamped_binaries(fixture) {
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read the binary {}: {error}", fixture.shown(&path)))?;
        text.push_str(&format!("{key}={}\n", sha256_hex(&bytes)));
    }
    Ok(text)
}

/// The first line of the stamp `now` that the stamp of `--prepare` does not have, as a refusal naming both values.
/// `what` is the noun of the prepared files and `command` the one that prepares them (`check-difftest` shares this).
pub(crate) fn stamp_refusal(
    prepared: &str,
    now: &str,
    what: &str,
    command: &str,
) -> Option<String> {
    let line = now
        .lines()
        .find(|line| !prepared.lines().any(|p| p == *line))?;
    let (key, value) = line.split_once('=')?;
    let before = prepared
        .lines()
        .find_map(|p| p.strip_prefix(key)?.strip_prefix('='))
        .unwrap_or("(none)");
    Some(format!(
        "the prepared {what} are stale: {key} was {before} when `--prepare` ran and is {value} now; run `cargo xtask {command} --prepare` again"
    ))
}

/// HEAD and a hash of the working tree's state (status, the binary diff against HEAD, and the path, length and bytes of
/// each untracked file).
pub(crate) fn source_state(root: &Path) -> Result<(String, String), String> {
    let git = |args: &[&str]| -> Result<Vec<u8>, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| format!("could not run git {}: {error}", args.join(" ")))?;
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(format!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    };
    let head = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"])?)
        .trim()
        .to_owned();
    let mut state = git(&["status", "--porcelain", "-z", "--untracked-files=all"])?;
    state.extend(git(&[
        "diff",
        "HEAD",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
    ])?);
    for path in git(&["ls-files", "--others", "--exclude-standard", "-z"])?
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let name = String::from_utf8_lossy(path);
        let bytes = fs::read(root.join(name.as_ref()))
            .map_err(|error| format!("cannot read the untracked file {name}: {error}"))?;
        state.extend(format!("\n{name} {}\n", bytes.len()).bytes());
        state.extend(bytes);
    }
    Ok((head, sha256_hex(&state)))
}

/// `--adapter-runs`: the three runs of E1, E3 and E8 on the binaries that `--prepare` built, if its stamp is current.
fn adapter_runs(fixture: &Fixture) -> Result<String, String> {
    let refuse = |message: String| format!("check-precompiled --adapter-runs: {message}");
    let file = fixture.target.join(STAMP_FILE);
    let prepared = fs::read_to_string(&file).map_err(|error| {
        refuse(format!(
            "no stamp at {} ({error}); run `cargo xtask check-precompiled --prepare` first",
            fixture.shown(&file)
        ))
    })?;
    let (head, tree) = source_state(&fixture.root)?;
    let now = stamp_text(fixture, &head, &tree).map_err(|error| {
        refuse(format!(
            "{error}; run `cargo xtask check-precompiled --prepare` again"
        ))
    })?;
    if let Some(message) = stamp_refusal(&prepared, &now, "binaries", "check-precompiled") {
        return Err(refuse(message));
    }
    check_slang_free(fixture)?;
    let [(_, e1), (_, e3), (_, e8)] = stamped_binaries(fixture);
    let report = e1_run(fixture, &e1)?;
    let identity = e3_run(fixture, &e3)?;
    let consumer = e8_run(fixture, &e8)?;
    Ok(format!(
        "check-precompiled --adapter-runs passed (binaries prepared at HEAD {head}): E1 {report}. E3 {identity}. E8 the consumer printed `{VERIFIED_PREFIX}{} elements` on the adapter `{}`.",
        consumer.elements, consumer.adapter
    ))
}

/// `cargo xtask check-precompiled [--check-only | --prepare | --adapter-runs] [--work-dir PATH]`.
pub(crate) fn check_precompiled(args: &[String]) -> Result<(), String> {
    println!("{}", check_precompiled_report(args)?);
    Ok(())
}

/// [`check_precompiled`] returning the pass summary (the E steps and the adapter they ran on) instead of printing
/// it, so `verify --mode full` can record it.
pub(crate) fn check_precompiled_report(args: &[String]) -> Result<String, String> {
    let options = parse_args(args)?;
    let root = workspace_root()?;
    let fixture = Fixture::new(&root);
    println!(
        "check-precompiled: cargo target directory {}{}",
        fixture.shown(&fixture.target_dir),
        match options.mode {
            Mode::Full => "",
            Mode::CheckOnly => " (--check-only: no adapter, no Slang)",
            Mode::Prepare => " (--prepare: no adapter; the stamp for --adapter-runs at the end)",
            Mode::AdapterRuns => " (--adapter-runs: the three runs only)",
        }
    );
    if options.mode == Mode::AdapterRuns {
        return adapter_runs(&fixture);
    }
    // Taken before the builds, and the old stamp removed, so that a failed `--prepare` leaves none.
    let prepared = if options.mode == Mode::Prepare {
        let _ = fs::remove_file(fixture.target.join(STAMP_FILE));
        Some(source_state(&root)?)
    } else {
        None
    };

    let report = e1_stripped_build_and_run(&fixture, options.mode)?;
    let binary = e2_binary_and_features(&fixture)?;
    let identity = e3_mutated_artifact(&fixture, options.mode)?;
    let located = e4_missing_artifact(&fixture)?;
    let strict_dual = e5_strict_plus_dual(&fixture)?;
    let refusals = e6_refusals(&fixture)?;
    let reexpanded = e7_reexpansion(&fixture)?;
    let reproducible = e7_build_twice(&fixture, options.mode)?;
    let distribution = e8_distribution(&fixture, &options)?;

    let headline = match prepared {
        Some((head, tree)) => {
            let stamp = stamp_text(&fixture, &head, &tree)?;
            fs::write(fixture.target.join(STAMP_FILE), stamp).map_err(|error| {
                format!("check-precompiled --prepare: could not write the stamp: {error}")
            })?;
            format!("check-precompiled --prepare passed (runs pending, stamp at HEAD {head})")
        }
        None => "check-precompiled passed".to_owned(),
    };
    Ok(format!(
        "{headline}: E1 {report}. E2 {binary}. E3 {identity}. E4 (E6f) {located}. E5 {strict_dual}. E6a to E6o {refusals}. E7 {reexpanded}. E7 {reproducible}. E8 {distribution}. The fixture is tests/precompiled; every child ran under {PROFILE_VARIABLE}={STRICT_PROFILE} except the E2 control and E6e ({PROFILE_VARIABLE}={CONTROL_PROFILE}), E6k to E6o (each under its own value, `native+subgroups` to `native+f16`), the profile-free builds of E6c, E6d, E6g, E6i and E6j, and cargo gust of E7."
    ))
}
