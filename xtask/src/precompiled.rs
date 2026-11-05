//! `cargo xtask check-precompiled [--check-only] [--work-dir PATH]`: the evidence that a production-shaped consumer of
//! precompiled kernels needs no Slang (D45, D46). The fixture is `tests/precompiled`, a workspace of its own like
//! `tests/dual`: the `kernels` crate holds one `#[gpu]` vector-add and its committed `gust-artifacts/` (written by
//! `cargo gust build`), and `app` dispatches it through `gust-wgpu` with `default-features = false` and checks the
//! result.
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
//!   `#[gpu]` module of `kernels` again (and the macro writes its records), and the same value does not. Then, with
//!   Slang, `cargo gust build` runs twice on a copy of the fixture whose artifacts were deleted first: run 1 writes them, run 2 (the run on unchanged input) writes nothing, and the output is byte-identical and the
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
//! it is what the CI job `precompiled` runs. Nothing but `cargo gust build` of E7 runs `slangc` or the native helper.
//! The environment of the xtask process is never changed; the stripped environment exists only in the children.
//! `CARGO_NET_OFFLINE` and the rest of the parent's environment pass through. E8 needs the registry (the consumer
//! needs `wgpu` from crates.io), so it is no offline proof.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    env::consts::EXE_SUFFIX,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use crate::{
    checks::{RUSTFLAGS_VARIABLES, dual_target_dir, precompiled_dual_steps, run_dual_step},
    json::{self, Json},
    process::{PROFILE_VARIABLE, exit_code, set_workspace_profile},
    release::{PackageSpec, WorkDir, package_files, package_problems},
    ui::{Expectation, judge, parse_expectation, parse_expected},
    workspace::{cargo_target_dir, workspace_root},
};

/// The fixture workspace, relative to the repository root.
pub(crate) const PRECOMPILED_DIR: &str = "tests/precompiled";

/// The pinned error of E4, relative to [`PRECOMPILED_DIR`]: the keys of a `check-ui` `expected.txt`.
const E4_EXPECTED: &str = "e4-expected.txt";

/// The profile of every child: strict, so that a missing artifact is a build error and the bridge is refused.
const STRICT_PROFILE: &str = "native";

/// The line `app` prints after the GPU result equals its host reference.
pub(crate) const VERIFIED_PREFIX: &str = "verified: GPU result equals the host reference for ";

/// The line `app` prints with the name of the adapter it ran on.
pub(crate) const ADAPTER_PREFIX: &str = "adapter: ";

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

/// The files that make a directory of `PATH` a place Slang can be launched from.
const SLANG_TOOLS: [&str; 4] = [
    "slangc",
    "slangc.exe",
    "gust-slang-reflect",
    "gust-slang-reflect.exe",
];

/// The variables, besides those with `SLANG` in the name, that E1 takes out of the children: the Vulkan SDK ships
/// `slangc`, and a stray export directory belongs to `cargo gust` alone (the macro would stay in export mode).
const STRIPPED_VARIABLES: [&str; 3] = ["VULKAN_SDK", "VK_SDK_PATH", "GUST_EXPORT_DIR"];

/// The profile of the control build of E2, the only child that is not strict: the bridge needs a development profile.
const CONTROL_PROFILE: &str = "dev";

/// The variable names of the child that E1 must not contain.
pub(crate) fn is_slang_variable(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.contains("SLANG") || STRIPPED_VARIABLES.contains(&upper.as_str())
}

/// Whether `name` is a variable that cargo reads rustflags from (such a variable would change the builds).
fn is_rustflags_variable(name: &str) -> bool {
    RUSTFLAGS_VARIABLES
        .iter()
        .any(|variable| variable.eq_ignore_ascii_case(name))
}

/// A directory of `PATH` that holds a Slang tool.
pub(crate) fn directory_has_slang_tool(directory: &Path) -> bool {
    SLANG_TOOLS
        .iter()
        .any(|tool| directory.join(tool).is_file())
}

/// `path` without the directories for which `has_tool` is true, and those directories in their order.
pub(crate) fn strip_path(
    path: &OsStr,
    has_tool: &dyn Fn(&Path) -> bool,
) -> (OsString, Vec<PathBuf>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for directory in env::split_paths(path) {
        if has_tool(&directory) {
            dropped.push(directory);
        } else {
            kept.push(directory);
        }
    }
    // `join_paths` fails only for an entry that holds the separator, which `split_paths` never produces.
    let joined = env::join_paths(kept).unwrap_or_else(|_| path.to_owned());
    (joined, dropped)
}

/// The environment of a child: the parent's, without the Slang variables, the rustflags variables and the Slang
/// directories of `PATH`, and with `GUST_PROFILE=native`. `report` collects what was taken out.
pub(crate) fn child_environment(
    parent: &[(OsString, OsString)],
    has_tool: &dyn Fn(&Path) -> bool,
    report: &mut StripReport,
) -> Vec<(OsString, OsString)> {
    let mut child = Vec::new();
    for (name, value) in parent {
        let text = name.to_string_lossy();
        if is_slang_variable(&text) {
            report.variables.push(text.into_owned());
        } else if is_rustflags_variable(&text) || text.eq_ignore_ascii_case(PROFILE_VARIABLE) {
            // Set again below (the profile) or left to cargo's own sources (rustflags).
        } else if text.eq_ignore_ascii_case("PATH") {
            let (stripped, dropped) = strip_path(value, has_tool);
            report.directories.extend(dropped);
            child.push((name.clone(), stripped));
        } else {
            child.push((name.clone(), value.clone()));
        }
    }
    child.push((
        OsString::from(PROFILE_VARIABLE),
        OsString::from(STRICT_PROFILE),
    ));
    child
}

/// What E1 took out of the children's environment.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct StripReport {
    /// The variables that were removed (names only; the values may be paths).
    pub(crate) variables: Vec<String>,
    /// The `PATH` directories that were removed.
    pub(crate) directories: Vec<PathBuf>,
}

/// The problems of a child environment that is supposed to have no Slang: a variable that names Slang, a `PATH`
/// directory that holds a Slang tool, a missing `GUST_PROFILE=native`.
pub(crate) fn slang_free_problems(
    environment: &[(OsString, OsString)],
    has_tool: &dyn Fn(&Path) -> bool,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (name, value) in environment {
        let text = name.to_string_lossy();
        if is_slang_variable(&text) {
            problems.push(format!("the variable {text} is set"));
        }
        if text.eq_ignore_ascii_case("PATH") {
            for directory in env::split_paths(value).filter(|directory| has_tool(directory)) {
                problems.push(format!(
                    "the PATH directory {} holds a Slang tool",
                    directory.display()
                ));
            }
        }
    }
    let profile = environment.iter().find(|(name, _)| {
        name.to_string_lossy()
            .eq_ignore_ascii_case(PROFILE_VARIABLE)
    });
    if profile.is_none_or(|(_, value)| value != STRICT_PROFILE) {
        problems.push(format!("{PROFILE_VARIABLE} is not {STRICT_PROFILE}"));
    }
    problems
}

/// Whether the shell of the platform finds `slangc` with the `PATH` of `environment`: `Ok(())` when it does not, an
/// error when it does or when the lookup itself fails. The lookup is `where slangc` under `cmd` on Windows and
/// `command -v slangc` under `sh` elsewhere. It runs through a shell because `Command::new("slangc")` is no test:
/// on Windows std falls back to the parent's `PATH` after the one it was given, so it would find the tool that the
/// stripping removed from the child.
fn slangc_cannot_be_found(environment: &[(OsString, OsString)]) -> Result<(), String> {
    let mut probe = if cfg!(windows) {
        let mut probe = Command::new("cmd");
        probe.args(["/d", "/c", "where", "slangc"]);
        probe
    } else {
        let mut probe = Command::new("sh");
        probe.args(["-c", "command -v slangc"]);
        probe
    };
    probe
        .env_clear()
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null());
    let output = probe
        .output()
        .map_err(|error| format!("could not run the lookup of `slangc`: {error}"))?;
    let code = output.status.code();
    // `where` answers 1 when it finds nothing and 2 for an error; `command -v` answers non-zero when it finds nothing.
    let absent = if cfg!(windows) {
        code == Some(1)
    } else {
        !output.status.success()
    };
    if absent {
        return Ok(());
    }
    if output.status.success() {
        return Err(format!(
            "`slangc` is found with the stripped environment: {}",
            String::from_utf8_lossy(&output.stdout).trim()
        ));
    }
    Err(format!(
        "the lookup of `slangc` failed (exit {}): {}",
        exit_code(output.status),
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

/// What the fixture's children share.
struct Fixture {
    root: PathBuf,
    /// The cargo target directory, which holds `precompiled`, `precompiled-control` and `precompiled-scratch`.
    target: PathBuf,
    /// The cargo target directory of every child, `<target>/precompiled`.
    target_dir: PathBuf,
    /// The scratch copies of E3 and E4, `<target>/precompiled-scratch`.
    scratch: PathBuf,
    cargo: OsString,
    environment: Vec<(OsString, OsString)>,
    report: StripReport,
}

impl Fixture {
    fn new(root: &Path) -> Self {
        let parent: Vec<(OsString, OsString)> = env::vars_os().collect();
        let target = root.join(cargo_target_dir(root));
        // Resolved before the stripping: the directory of the toolchain may be the one that is stripped on a machine
        // where Slang was unpacked beside it, and then this says so instead of failing to find cargo.
        let cargo = cargo_program(&parent);
        let mut report = StripReport::default();
        let environment = child_environment(&parent, &directory_has_slang_tool, &mut report);
        Self {
            root: root.to_path_buf(),
            target_dir: target.join("precompiled"),
            scratch: target.join("precompiled-scratch"),
            target,
            cargo,
            environment,
            report,
        }
    }

    /// A command that runs `program` in the stripped environment from the repository root.
    fn command(&self, program: impl AsRef<OsStr>) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(&self.root)
            .env_clear()
            .envs(self.environment.iter().map(|(name, value)| (name, value)))
            .stdin(Stdio::null());
        command
    }

    fn cargo(&self, args: &[&str]) -> Command {
        let mut command = self.command(&self.cargo);
        command.args(args);
        command
    }

    /// [`Self::cargo`] with `GUST_PROFILE` set to `profile` instead of the strict one (the control of E2, E6).
    fn cargo_under_profile(&self, args: &[&str], profile: &str) -> Command {
        let mut command = self.cargo(args);
        command.env(PROFILE_VARIABLE, profile);
        command
    }

    /// [`Self::cargo`] with `GUST_PROFILE` removed: the build decides from cargo's `PROFILE` alone (E6c, E6d).
    fn cargo_without_profile(&self, args: &[&str]) -> Command {
        let mut command = self.cargo(args);
        command.env_remove(PROFILE_VARIABLE);
        command
    }

    /// [`Self::cargo`] run in `directory` instead of the repository root, so that cargo reads no configuration of
    /// the repository (it looks from the current directory, not from `--manifest-path`).
    fn cargo_in(&self, directory: &Path, args: &[&str]) -> Command {
        let mut command = self.cargo(args);
        command.current_dir(directory);
        command
    }

    /// `path` as `<target>/..` when it is under the target directory, relative to the repository when it is under that,
    /// else as given, with forward slashes: output without the machine's own paths.
    fn shown(&self, path: &Path) -> String {
        let text = |path: &Path| path.to_string_lossy().replace('\\', "/");
        if let Ok(rest) = path.strip_prefix(&self.target) {
            return format!("<target>/{}", text(rest));
        }
        text(path.strip_prefix(&self.root).unwrap_or(path))
    }

    /// `<target>/precompiled-control`, the target directory of the control build of E2.
    fn control_dir(&self) -> PathBuf {
        self.target_dir.with_file_name("precompiled-control")
    }

    /// `<target>/precompiled-scratch/target`, the target directory of the builds in the scratch copies (E3, E4). It is
    /// not the one of E1: both builds of `app` write `release/app`, and cargo does not link a fresh unit again, so a
    /// second run of the check would run the binary of the mutated copy as the binary of E1.
    fn scratch_target_dir(&self) -> PathBuf {
        self.scratch.join("target")
    }

    /// `directory` as the argument of `--target-dir`; cargo takes it as text, so a path that is not valid UTF-8 would
    /// be altered, not passed.
    fn target_dir_arg(directory: &Path) -> Result<&str, String> {
        directory.to_str().ok_or_else(|| {
            "check-precompiled: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path"
                .to_owned()
        })
    }

    /// `<directory>/release/app`.
    fn app_binary(directory: &Path) -> PathBuf {
        directory.join("release").join(format!("app{EXE_SUFFIX}"))
    }
}

/// `$CARGO` when set (it is under `cargo xtask`), else `cargo`.
fn cargo_program(parent: &[(OsString, OsString)]) -> OsString {
    parent
        .iter()
        .find(|(name, _)| name.to_string_lossy().eq_ignore_ascii_case("CARGO"))
        .map_or_else(|| OsString::from("cargo"), |(_, value)| value.clone())
}

/// What `cargo xtask check-precompiled` was asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    /// `--check-only`: skip what needs an adapter (the runs of E1, E3 and E8) and Slang (the second half of E7).
    pub(crate) check_only: bool,
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
            "--check-only" => {
                if inline.is_some() {
                    return Err("`--check-only` takes no value".into());
                }
                options.check_only = true;
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
                    "check-precompiled takes only `--check-only` and `--work-dir PATH`, not `{argument}`"
                ));
            }
        }
    }
    Ok(options)
}

/// `cargo xtask check-precompiled [--check-only] [--work-dir PATH]`.
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
        if options.check_only {
            " (--check-only: no adapter, no Slang)"
        } else {
            ""
        }
    );

    let report = e1_stripped_build_and_run(&fixture, options.check_only)?;
    let binary = e2_binary_and_features(&fixture)?;
    let identity = e3_mutated_artifact(&fixture, options.check_only)?;
    let located = e4_missing_artifact(&fixture)?;
    let strict_dual = e5_strict_plus_dual(&fixture)?;
    let refusals = e6_refusals(&fixture)?;
    let reexpanded = e7_reexpansion(&fixture)?;
    let reproducible = e7_build_twice(&fixture, options.check_only)?;
    let distribution = e8_distribution(&fixture, &options)?;

    Ok(format!(
        "check-precompiled passed: E1 {report}. E2 {binary}. E3 {identity}. E4 (E6f) {located}. E5 {strict_dual}. E6a to E6j {refusals}. E7 {reexpanded}. E7 {reproducible}. E8 {distribution}. The fixture is tests/precompiled; every child ran under {PROFILE_VARIABLE}={STRICT_PROFILE} except the E2 control and E6e ({PROFILE_VARIABLE}={CONTROL_PROFILE}), the profile-free builds of E6c, E6d, E6g, E6i and E6j, and cargo gust of E7."
    ))
}

// ---------------------------------------------------------------------------------------------------------------
// E1

/// E1: release build and run of `app` with Slang taken out of the environment. Returns the summary.
fn e1_stripped_build_and_run(fixture: &Fixture, check_only: bool) -> Result<String, String> {
    let problems = slang_free_problems(&fixture.environment, &directory_has_slang_tool);
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E1: the stripped environment still has Slang: {}",
            problems.join("; ")
        ));
    }
    slangc_cannot_be_found(&fixture.environment).map_err(|error| {
        format!("check-precompiled E1: the stripped environment still has Slang: {error}")
    })?;
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
    let output = fixture
        .cargo(&[
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
        ])
        .output()
        .map_err(|error| format!("check-precompiled E1: could not launch cargo: {error}"))?;
    require_success(
        "E1",
        "cargo clean -p gust-compute -p gust-compute-macros -p kernels -p app",
        &output,
    )?;
    let output = fixture
        .cargo(&[
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
        ])
        .output()
        .map_err(|error| format!("check-precompiled E1: could not launch cargo: {error}"))?;
    require_success("E1", "cargo build --release -p app", &output)?;
    println!(
        "check-precompiled E1: cargo clean of gust-compute, gust-compute-macros, kernels and app, then cargo build --release -p app, succeeded"
    );

    if check_only {
        return Ok("passed (build only: --check-only skips the run, which needs an adapter): the release `app` built with Slang taken out of the environment".to_owned());
    }
    let binary = Fixture::app_binary(&fixture.target_dir);
    let run = fixture.command(&binary).output().map_err(|error| {
        format!(
            "check-precompiled E1: could not run {}: {error}",
            fixture.shown(&binary)
        )
    })?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    print!("{stdout}");
    let report = judge_run(run.status.success(), &stdout, &stderr)
        .map_err(|problems| format!("check-precompiled E1: {}", problems.join("; ")))?;
    Ok(format!(
        "passed: the release `app`, built and run with Slang taken out of the environment, printed `{VERIFIED_PREFIX}{} elements` on the adapter `{}`",
        report.elements, report.adapter
    ))
}

/// What a successful run of `app` printed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RunReport {
    pub(crate) adapter: String,
    pub(crate) elements: String,
}

/// The problems of the run of `app` in E1: it must exit 0, name its adapter and a WGSL artifact, and print the
/// `verified:` line. Returns what it printed.
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
    match (problems.is_empty(), adapter, elements) {
        (true, Some(adapter), Some(elements)) => Ok(RunReport { adapter, elements }),
        _ => Err(problems),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// E2

/// E2: the binary of E1, `cargo tree -e features`, and the manifests. Returns the summary.
fn e2_binary_and_features(fixture: &Fixture) -> Result<String, String> {
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
    let output = fixture
        .cargo(&[
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
        ])
        .output()
        .map_err(|error| format!("check-precompiled E2: could not launch cargo: {error}"))?;
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
    let output = fixture
        .cargo_under_profile(
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
        )
        .output()
        .map_err(|error| format!("check-precompiled E2: could not launch cargo: {error}"))?;
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

// ---------------------------------------------------------------------------------------------------------------
// E3

/// E3: a mutated artifact in a scratch copy gives the identity error at run time. Returns the summary.
fn e3_mutated_artifact(fixture: &Fixture, check_only: bool) -> Result<String, String> {
    let scratch = fixture.scratch.join("e3");
    let scratch_target = fixture.scratch_target_dir();
    copy_fixture(fixture, &scratch)?;
    let artifact = single_artifact(&scratch.join("kernels").join("gust-artifacts"), "wgsl")?;
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
    let output = fixture
        .cargo(&[
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
        ])
        .output()
        .map_err(|error| format!("check-precompiled E3: could not launch cargo: {error}"))?;
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
fn e4_missing_artifact(fixture: &Fixture) -> Result<String, String> {
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
    let output = fixture
        .cargo(&[
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
        ])
        .output()
        .map_err(|error| format!("check-precompiled E4: could not launch cargo: {error}"))?;
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
fn e5_strict_plus_dual(fixture: &Fixture) -> Result<String, String> {
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

// ---------------------------------------------------------------------------------------------------------------
// E6

/// The expectation of the explicit `native` refusals: the `build-error:` text of `gust`'s build script, relative to
/// [`PRECOMPILED_DIR`].
const E6_NATIVE_EXPECTED: &str = "e6-native-expected.txt";

/// The same for the refusal of a release build without a variable: the text of the first plus the sentence that says
/// the profile was inferred.
const E6_INFERRED_EXPECTED: &str = "e6-inferred-expected.txt";

/// The expectation of the cases in which `cargo-gust-bridge` made the profile `dev`: the `warning:` lines that cargo
/// must print, relative to [`PRECOMPILED_DIR`].
const E6_BRIDGE_EXPECTED: &str = "e6-bridge-expected.txt";

/// The features of the bridge cases: the internal feature, and the bridge that a strict profile refuses.
const BRIDGE_FEATURES: &str = "gust/cargo-gust-bridge,gust/slang-runtime";

/// The sentence of the refusal that only an inferred profile has (`crates/gust/build.rs`).
const INFERRED_SENTENCE: &str =
    "This profile was inferred from a release build without GUST_PROFILE.";

/// What the build script prints for cargo when a release build had no `GUST_PROFILE` and was treated as `native`.
const INFERRED_WARNING: &str = "gust: release build without GUST_PROFILE treated as native";

/// Whether cargo's stderr `text` has the build-script warning `warning` of the package `gust-compute` as a whole line: cargo
/// prints it as `warning: gust-compute@<version>: <warning>`, so a line that continues after the words (`... as devX`) does not
/// count.
pub(crate) fn has_build_script_warning(text: &str, warning: &str) -> bool {
    text.lines().any(|line| {
        line.trim_end()
            .strip_prefix("warning: gust-compute@")
            .and_then(|rest| rest.split_once(": "))
            .is_some_and(|(version, printed)| {
                !version.is_empty() && !version.contains(' ') && printed == warning
            })
    })
}

/// Whether cargo's stderr `text` has the warning of cargo itself `warning` as a whole line: `warning: <warning>`, or,
/// for a warning about a dependency of the case's `app`, `warning: app v<version> (<path>) <warning>` (cargo names
/// the package that declares the dependency; the path is the machine's and is not matched).
pub(crate) fn has_cargo_warning(text: &str, warning: &str) -> bool {
    text.lines().any(|line| {
        line.trim_end()
            .strip_prefix("warning: ")
            .is_some_and(|rest| {
                rest == warning
                    || (rest.starts_with("app v") && rest.ends_with(&format!(") {warning}")))
            })
    })
}

/// The `GUST_PROFILE` of an E6 case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum E6Profile {
    Native,
    Dev,
    /// The variable is removed, so a release build infers its profile.
    Unset,
}

/// What an E6 case must do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum E6Outcome {
    /// The build script of `gust` refuses. `expected` holds the `build-error:` lines (relative to
    /// [`PRECOMPILED_DIR`]); `inferred` says whether the message must (or must not) carry [`INFERRED_SENTENCE`].
    Refused {
        expected: &'static str,
        inferred: bool,
    },
    /// The build compiles. With `warning`, cargo must also have printed that text (the inference was visible).
    Compiles { warning: Option<&'static str> },
    /// The build compiles with the run-time bridge in it because `cargo-gust-bridge` made the profile `dev`. `expected`
    /// holds the `warning:` lines (relative to [`PRECOMPILED_DIR`]) that cargo must print; the plain inference text
    /// ([`INFERRED_WARNING`], `native`) must not be there. With `ignored_tool`, the `cargo-warning:` lines of the file
    /// must be there too (cargo's own warning about a `cargo-gust` dependency, E6j).
    CompilesAsDev {
        expected: &'static str,
        ignored_tool: bool,
    },
}

/// One `cargo check -p app` of E6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct E6Case {
    pub(crate) id: &'static str,
    /// `--features`, if any.
    pub(crate) features: Option<&'static str>,
    pub(crate) release: bool,
    pub(crate) profile: E6Profile,
    /// E6c: built in a scratch copy in which `app` depends on a crate that enables `gust-wgpu/slang-runtime`; that
    /// copy's lockfile changes, so the case runs without `--locked` and never in the fixture.
    pub(crate) bridge_user: bool,
    /// E6i: checks the whole workspace of the repository (`--workspace`, the root manifest) instead of `-p app`.
    pub(crate) workspace: bool,
    /// E6j: built in a scratch copy in which `app` has `cargo-gust` as a normal dependency (a mistake: it has no
    /// library, so cargo ignores the dependency, but the features of the edge still unify). Like E6c, never `--locked`.
    pub(crate) tool_dependency: bool,
    pub(crate) outcome: E6Outcome,
}

impl E6Case {
    /// Whether the case builds in a scratch copy of the fixture (whose lockfile then changes).
    pub(crate) fn in_copy(&self) -> bool {
        self.bridge_user || self.tool_dependency
    }
}

/// The cases of E6, in order. E6f is E4 and is not repeated.
pub(crate) fn e6_cases() -> [E6Case; 10] {
    let native = E6Outcome::Refused {
        expected: E6_NATIVE_EXPECTED,
        inferred: false,
    };
    [
        E6Case {
            id: "E6a",
            features: Some("gust/slang-runtime"),
            release: false,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6b",
            features: Some("gust-wgpu/slang-runtime"),
            release: false,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6c",
            features: None,
            release: false,
            profile: E6Profile::Native,
            bridge_user: true,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6c-release",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: true,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Refused {
                expected: E6_INFERRED_EXPECTED,
                inferred: true,
            },
        },
        E6Case {
            id: "E6d",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Compiles {
                warning: Some(INFERRED_WARNING),
            },
        },
        E6Case {
            id: "E6e",
            features: Some("gust/slang-runtime"),
            release: true,
            profile: E6Profile::Dev,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Compiles { warning: None },
        },
        E6Case {
            id: "E6g",
            features: Some(BRIDGE_FEATURES),
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: false,
            },
        },
        E6Case {
            id: "E6h",
            features: Some(BRIDGE_FEATURES),
            release: true,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6i",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: true,
            tool_dependency: false,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: false,
            },
        },
        E6Case {
            id: "E6j",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: true,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: true,
            },
        },
    ]
}

/// The arguments of the `cargo check` of one E6 case.
pub(crate) fn e6_args(case: &E6Case, manifest: &str, target_dir: &str) -> Vec<String> {
    let mut args: Vec<String> = ["check"].iter().map(|arg| (*arg).to_owned()).collect();
    if !case.in_copy() {
        args.push("--locked".to_owned());
    }
    for arg in [
        "--color",
        "never",
        "--message-format",
        "json",
        "--manifest-path",
        manifest,
    ] {
        args.push(arg.to_owned());
    }
    if case.workspace {
        args.push("--workspace".to_owned());
    } else {
        args.push("-p".to_owned());
        args.push("app".to_owned());
    }
    if case.release {
        args.push("--release".to_owned());
    }
    if let Some(features) = case.features {
        args.push("--features".to_owned());
        args.push(features.to_owned());
    }
    args.push("--target-dir".to_owned());
    args.push(target_dir.to_owned());
    args
}

/// The problems of one E6 case for its outcome. `expectation` is the text of the case's expectation file (`None` for a
/// case that must compile). A refusal is judged on the exact message of the build script by [`judge`], never on the
/// exit status alone; the sentence about an inferred profile must be there for the inferred case and absent for the
/// explicit ones.
pub(crate) fn e6_problems(
    case: &E6Case,
    expectation: Option<&str>,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<String>, String> {
    let text = stderr.replace('\\', "/");
    match case.outcome {
        E6Outcome::Refused { expected, inferred } => {
            let source = expectation.ok_or_else(|| format!("{}: no expectation text", case.id))?;
            let wanted =
                parse_expectation(source).map_err(|error| format!("{expected}: {error}"))?;
            if !matches!(wanted, Expectation::BuildFails(_)) {
                return Err(format!(
                    "{expected} must hold `build-error:` lines (the refusal is a build-script failure), not rustc keys"
                ));
            }
            let mut problems = judge(case.id, "app", &wanted, success, stdout, stderr);
            if text.contains(INFERRED_SENTENCE) != inferred {
                problems.push(format!(
                    "{}: the refusal {} the sentence `{INFERRED_SENTENCE}`",
                    case.id,
                    if inferred {
                        "lacks"
                    } else {
                        "must not have, but has,"
                    }
                ));
            }
            Ok(problems)
        }
        E6Outcome::Compiles { warning } => {
            let mut problems = judge(
                case.id,
                "app",
                &Expectation::Compiles,
                success,
                stdout,
                stderr,
            );
            if let Some(warning) = warning
                && !has_build_script_warning(&text, warning)
            {
                problems.push(format!(
                    "{}: cargo did not print `{warning}`, so nothing shows the profile was inferred",
                    case.id
                ));
            }
            Ok(problems)
        }
        E6Outcome::CompilesAsDev {
            expected,
            ignored_tool,
        } => {
            let source = expectation.ok_or_else(|| format!("{}: no expectation text", case.id))?;
            let wanted = parse_warning_expectation(source)
                .map_err(|error| format!("{expected}: {error}"))?;
            if ignored_tool && wanted.cargo.is_empty() {
                return Err(format!(
                    "{expected}: {} needs a `cargo-warning:` line",
                    case.id
                ));
            }
            let mut problems = judge(
                case.id,
                if case.workspace {
                    "the workspace"
                } else {
                    "app"
                },
                &Expectation::Compiles,
                success,
                stdout,
                stderr,
            );
            for warning in &wanted.build_script {
                if !has_build_script_warning(&text, warning) {
                    problems.push(format!(
                        "{}: cargo did not print `{warning}`, so nothing shows that the bridge feature changed \
                         the profile",
                        case.id
                    ));
                }
            }
            if ignored_tool {
                for warning in &wanted.cargo {
                    if !has_cargo_warning(&text, warning) {
                        problems.push(format!(
                            "{}: cargo did not print `warning: {warning}`, so the dependency on `cargo-gust` is not \
                             the one this case pins",
                            case.id
                        ));
                    }
                }
            }
            if has_build_script_warning(&text, INFERRED_WARNING) {
                problems.push(format!(
                    "{}: cargo printed `{INFERRED_WARNING}`, so the profile was inferred as native and not made dev",
                    case.id
                ));
            }
            Ok(problems)
        }
    }
}

/// The lines of an expectation of the bridge cases: `warning:` is the text of the build script's `cargo:warning` that
/// cargo must print (as `warning: gust-compute@<version>: <text>`), `cargo-warning:` a warning of cargo itself (as
/// `warning: <text>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BridgeExpectation {
    pub(crate) build_script: Vec<String>,
    pub(crate) cargo: Vec<String>,
}

/// Parses an expectation of the bridge cases (blank lines and lines that start with `#` are ignored). Any other line,
/// or no `warning:` line, is an error.
pub(crate) fn parse_warning_expectation(text: &str) -> Result<BridgeExpectation, String> {
    let mut wanted = BridgeExpectation {
        build_script: Vec::new(),
        cargo: Vec::new(),
    };
    for raw in text.trim_start_matches('\u{feff}').lines() {
        let entry = raw.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        if let Some(value) = entry.strip_prefix("warning:").map(str::trim)
            && !value.is_empty()
        {
            wanted.build_script.push(value.to_owned());
        } else if let Some(value) = entry.strip_prefix("cargo-warning:").map(str::trim)
            && !value.is_empty()
        {
            wanted.cargo.push(value.to_owned());
        } else {
            return Err(format!(
                "`{entry}` is not a `warning: <text>` or `cargo-warning: <text>` line"
            ));
        }
    }
    if wanted.build_script.is_empty() {
        return Err("no `warning:` line".to_owned());
    }
    Ok(wanted)
}

/// The manifest of the crate that E6c adds: it depends on `gust-compute-wgpu` with the bridge feature. `root` is the
/// repository, with forward slashes.
pub(crate) fn bridge_user_manifest(root: &str) -> String {
    format!(
        "[package]\nname = \"bridge-user\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngust-wgpu = {{ package = \"gust-compute-wgpu\", path = \"{root}/crates/gust-wgpu\", features = [\"slang-runtime\"] }}\n"
    )
}

/// The fixture's workspace manifest with `bridge-user` as a third member.
pub(crate) fn with_bridge_user_member(workspace: &str) -> Result<String, String> {
    const MEMBERS: &str = "members = [\"kernels\", \"app\"]";
    if !workspace.contains(MEMBERS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/Cargo.toml no longer has the line `{MEMBERS}`; E6c cannot add its crate"
        ));
    }
    Ok(workspace.replace(MEMBERS, "members = [\"kernels\", \"app\", \"bridge-user\"]"))
}

/// `app`'s manifest with a normal dependency on `bridge-user`, after its dependency on `kernels`.
pub(crate) fn with_bridge_user_dependency(app: &str) -> Result<String, String> {
    const KERNELS: &str = "kernels = { path = \"../kernels\" }";
    if !app.contains(KERNELS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/app/Cargo.toml no longer has the line `{KERNELS}`; E6c cannot add its dependency"
        ));
    }
    Ok(app.replace(
        KERNELS,
        &format!("{KERNELS}\nbridge-user = {{ path = \"../bridge-user\" }}"),
    ))
}

/// `app`'s manifest with a normal path dependency on `cargo-gust`, after its dependency on `kernels`. `root` is the
/// repository, with forward slashes.
pub(crate) fn with_tool_dependency(app: &str, root: &str) -> Result<String, String> {
    const KERNELS: &str = "kernels = { path = \"../kernels\" }";
    if !app.contains(KERNELS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/app/Cargo.toml no longer has the line `{KERNELS}`; E6j cannot add its dependency"
        ));
    }
    Ok(app.replace(
        KERNELS,
        &format!("{KERNELS}\ncargo-gust = {{ path = \"{root}/crates/cargo-gust\" }}"),
    ))
}

/// Makes `app` of the scratch copy `copy` of the fixture depend on `cargo-gust`.
fn add_tool_dependency(fixture: &Fixture, copy: &Path) -> Result<(), String> {
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    let app = copy.join("app").join("Cargo.toml");
    let text = fs::read_to_string(&app).map_err(|error| format!("{}: {error}", app.display()))?;
    fs::write(&app, with_tool_dependency(&text, &root)?)
        .map_err(|error| format!("{}: {error}", app.display()))
}

/// Adds the crate `bridge-user` to the scratch copy `copy` of the fixture and makes `app` depend on it.
fn add_bridge_user(fixture: &Fixture, copy: &Path) -> Result<(), String> {
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    let write = |path: PathBuf, text: String| {
        fs::write(&path, text).map_err(|error| format!("{}: {error}", path.display()))
    };
    let read = |path: &Path| {
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
    };
    let directory = copy.join("bridge-user");
    fs::create_dir_all(directory.join("src"))
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    write(directory.join("Cargo.toml"), bridge_user_manifest(&root))?;
    write(
        directory.join("src").join("lib.rs"),
        "// Enables `gust-wgpu/slang-runtime` for E6c of `cargo xtask check-precompiled`; it has no code.\n".to_owned(),
    )?;
    let workspace = copy.join("Cargo.toml");
    write(
        workspace.clone(),
        with_bridge_user_member(&read(&workspace)?)?,
    )?;
    let app = copy.join("app").join("Cargo.toml");
    write(app.clone(), with_bridge_user_dependency(&read(&app)?)?)?;
    Ok(())
}

/// E6a to E6j: the refusals of `gust`'s build script, their positive controls, and the three ways in which the internal
/// `cargo-gust-bridge` feature (D108) makes a release build without a variable `dev` (E6g, E6i, E6j; a release build
/// without `GUST_PROFILE` is otherwise strict). Returns the summary.
fn e6_refusals(fixture: &Fixture) -> Result<String, String> {
    let scratch_target = fixture.scratch_target_dir();
    let target_dir = Fixture::target_dir_arg(&scratch_target)?;
    let fixture_manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let copy = fixture.scratch.join("e6c");
    copy_fixture(fixture, &copy)?;
    add_bridge_user(fixture, &copy)?;
    let copy_manifest = copy.join("Cargo.toml");
    let copy_manifest = copy_manifest
        .to_str()
        .ok_or("check-precompiled E6: the scratch path is not valid UTF-8")?;
    let tool_copy = fixture.scratch.join("e6j");
    copy_fixture(fixture, &tool_copy)?;
    add_tool_dependency(fixture, &tool_copy)?;
    let tool_manifest = tool_copy.join("Cargo.toml");
    let tool_manifest = tool_manifest
        .to_str()
        .ok_or("check-precompiled E6: the scratch path is not valid UTF-8")?;

    let mut problems = Vec::new();
    let mut lines = Vec::new();
    for case in e6_cases() {
        let manifest = if case.bridge_user {
            copy_manifest
        } else if case.tool_dependency {
            tool_manifest
        } else if case.workspace {
            "Cargo.toml"
        } else {
            fixture_manifest.as_str()
        };
        let args = e6_args(&case, manifest, target_dir);
        let arguments: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut command = match case.profile {
            E6Profile::Native => fixture.cargo(&arguments),
            E6Profile::Dev => fixture.cargo_under_profile(&arguments, CONTROL_PROFILE),
            E6Profile::Unset => fixture.cargo_without_profile(&arguments),
        };
        let output = command
            .output()
            .map_err(|error| format!("check-precompiled E6: could not launch cargo: {error}"))?;
        let expectation = match case.outcome {
            E6Outcome::Refused { expected, .. } | E6Outcome::CompilesAsDev { expected, .. } => {
                let path = fixture.root.join(PRECOMPILED_DIR).join(expected);
                Some(fs::read_to_string(&path).map_err(|error| {
                    format!("check-precompiled E6: {PRECOMPILED_DIR}/{expected}: {error}")
                })?)
            }
            E6Outcome::Compiles { .. } => None,
        };
        let found = e6_problems(
            &case,
            expectation.as_deref(),
            output.status.success(),
            &String::from_utf8_lossy(&output.stdout),
            &String::from_utf8_lossy(&output.stderr),
        )
        .map_err(|error| format!("check-precompiled E6: {error}"))?;
        let shown = match case.outcome {
            E6Outcome::Refused { .. } => "refused",
            E6Outcome::Compiles { .. } | E6Outcome::CompilesAsDev { .. } => "compiled",
        };
        println!(
            "check-precompiled E6 {}: {} (profile {:?}{}{}): {}",
            case.id,
            args.join(" ")
                .replace(copy_manifest, "<copy>/Cargo.toml")
                .replace(tool_manifest, "<tool-copy>/Cargo.toml")
                .replace(target_dir, "<scratch-target>"),
            case.profile,
            if case.release { ", release" } else { "" },
            if case.bridge_user {
                ", in a copy with bridge-user"
            } else if case.tool_dependency {
                ", in a copy where app depends on cargo-gust"
            } else if case.workspace {
                ", the whole workspace"
            } else {
                ""
            },
            if found.is_empty() {
                shown
            } else {
                "NOT AS EXPECTED"
            }
        );
        lines.push(format!("{} {shown}", case.id));
        problems.extend(found);
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E6 found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    Ok(format!(
        "passed: {}; each refusal was judged on the exact message of `gust`'s build script ({PRECOMPILED_DIR}/{E6_NATIVE_EXPECTED}, {PRECOMPILED_DIR}/{E6_INFERRED_EXPECTED}), E6d and E6e are the positive controls, E6g, E6i and E6j compiled as dev with the warning of {PRECOMPILED_DIR}/{E6_BRIDGE_EXPECTED}",
        lines.join(", ")
    ))
}

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
/// and expands the `#[gpu]` module again when its value changes (the nonce of `cargo gust build`, R2-3), and not
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
            "{}: `kernels` was fresh, but the value of `GUST_EXPORT_DIR` changed, so cargo must expand its `#[gpu]` module again",
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

/// E7a: a change of `GUST_EXPORT_DIR` expands the `#[gpu]` module again. Returns the summary.
fn e7_reexpansion(fixture: &Fixture) -> Result<String, String> {
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
        let output = command
            .output()
            .map_err(|error| format!("check-precompiled E7: could not launch cargo: {error}"))?;
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
    let output = build
        .output()
        .map_err(|error| format!("check-precompiled E7: could not launch cargo: {error}"))?;
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
fn e7_build_twice(fixture: &Fixture, check_only: bool) -> Result<String, String> {
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
        let output = command.output().map_err(|error| {
            format!("check-precompiled E7: could not launch cargo-gust: {error}")
        })?;
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

// ---------------------------------------------------------------------------------------------------------------
// E8

/// The packages of the distribution proof, in the order they are unpacked and patched. What their `cargo package
/// --list` may hold is decided by `package_problems` of `release.rs`, the one home of the allow-lists (mr.r); the
/// `kernels` package is the only one with `gust-artifacts/**`.
pub(crate) const E8_PACKAGES: [PackageSpec<'static>; 4] = [
    PackageSpec::library("gust-compute-macros", false, false),
    PackageSpec::library("gust-compute", true, false),
    PackageSpec::library("gust-compute-wgpu", false, false),
    PackageSpec::library("kernels", false, true),
];

/// The strings that the packaged `kernels` must not contain as text: the CPU twin's types and function names.
const TWIN_TEXT: [&str; 3] = ["fn cpu", "gust::cpu", "CpuDevice"];

/// R4.6 (c): the unpacked kernels tree holds no `.slang` file, nothing from `generated-wgpu/` and no twin text.
/// `files` are relative paths with their text.
pub(crate) fn slang_free_package_problems(files: &[(String, String)]) -> Vec<String> {
    let mut problems = Vec::new();
    for (path, text) in files {
        if path.to_ascii_lowercase().ends_with(".slang") {
            problems.push(format!(
                "the kernels package holds the Slang source `{path}`"
            ));
        }
        if path.contains("generated-wgpu") {
            problems.push(format!(
                "the kernels package holds `{path}` from generated-wgpu/"
            ));
        }
        for marker in TWIN_TEXT {
            if text.contains(marker) {
                problems.push(format!(
                    "`{path}` of the kernels package contains the twin text `{marker}`"
                ));
            }
        }
    }
    problems
}

/// The `version` of the first `version = ".."` line in the table `[section]` of `manifest`.
pub(crate) fn section_version(manifest: &str, section: &str) -> Option<String> {
    let header = format!("[{section}]");
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            inside = line == header;
            continue;
        }
        if !inside {
            continue;
        }
        let Some(rest) = line.strip_prefix("version") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some((value, _)) = rest
            .trim()
            .strip_prefix('"')
            .and_then(|rest| rest.split_once('"'))
        else {
            continue;
        };
        return Some(value.to_owned());
    }
    None
}

/// The `Cargo.toml` of the relocated consumer (E8): plain registry-style dependencies on the packages that the
/// workspace calls `gust-compute`, `gust-compute-wgpu` and `kernels` (no `default-features = false`, D45), an empty `[workspace]` table
/// so that it is a root of its own, and a `[patch.crates-io]` for all four packages onto their unpacked trees, which
/// sit beside the consumer as `../unpacked/<name>-<version>`. `versions` has the version of each of
/// [`E8_PACKAGES`], in its order.
pub(crate) fn consumer_manifest(versions: &[String; 4]) -> String {
    let mut text = String::from(
        "# The relocated consumer of E8 in `cargo xtask check-precompiled`: registry-style dependencies, patched onto the\n# unpacked packages.\n[package]\nname = \"dist-consumer\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\n",
    );
    for (spec, version) in E8_PACKAGES.iter().zip(versions) {
        if spec.name != "gust-compute-macros" {
            text.push_str(&format!("{} = \"={version}\"\n", spec.name));
        }
    }
    text.push_str("\n[workspace]\n\n[patch.crates-io]\n");
    for (spec, version) in E8_PACKAGES.iter().zip(versions) {
        text.push_str(&format!(
            "{} = {{ path = \"../unpacked/{}-{version}\" }}\n",
            spec.name, spec.name
        ));
    }
    text
}

/// One `[[package]]` of a `Cargo.lock`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LockPackage {
    pub(crate) name: String,
    pub(crate) source: Option<String>,
}

/// The packages of a `Cargo.lock`.
pub(crate) fn lock_packages(lock: &str) -> Vec<LockPackage> {
    let mut packages: Vec<LockPackage> = Vec::new();
    let mut current: Option<LockPackage> = None;
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            packages.extend(current.take());
            current = Some(LockPackage {
                name: String::new(),
                source: None,
            });
        } else if let Some(package) = current.as_mut() {
            let value = |key: &str| {
                line.strip_prefix(key)
                    .and_then(|rest| rest.trim_start().strip_prefix('='))
                    .and_then(|rest| rest.trim().strip_prefix('"'))
                    .and_then(|rest| rest.split_once('"'))
                    .map(|(value, _)| value.to_owned())
            };
            if let Some(name) = value("name") {
                package.name = name;
            } else if let Some(source) = value("source") {
                package.source = Some(source);
            }
        }
    }
    packages.extend(current);
    packages
}

/// The problems of the consumer's `Cargo.lock`: each of the four packages once and with no `source` line (a path source
/// has none; a registry one reads `registry+https://..`), and at least one package from a registry, so that an
/// unreadable lock cannot pass.
pub(crate) fn lock_problems(lock: &str) -> Vec<String> {
    let packages = lock_packages(lock);
    let mut problems = Vec::new();
    for spec in &E8_PACKAGES {
        let entries: Vec<&LockPackage> = packages
            .iter()
            .filter(|package| package.name == spec.name)
            .collect();
        match entries.as_slice() {
            [] => problems.push(format!(
                "the lock has no `[[package]]` named `{}`",
                spec.name
            )),
            [only] => {
                if let Some(source) = &only.source {
                    problems.push(format!(
                        "the lock's `{}` has the source `{source}`: it came from a registry, not from the unpacked package",
                        spec.name
                    ));
                }
            }
            many => problems.push(format!(
                "the lock has {} `[[package]]` entries named `{}`",
                many.len(),
                spec.name
            )),
        }
    }
    if !packages.iter().any(|package| {
        package
            .source
            .as_deref()
            .is_some_and(|source| source.starts_with("registry+"))
    }) {
        problems.push("the lock has no package from a registry (wgpu should be), so it was not read correctly".to_owned());
    }
    problems
}

/// The lines of cargo's output that report a patch that nothing used.
pub(crate) fn unused_patch_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.contains("was not used"))
        .map(|line| line.trim().to_owned())
        .collect()
}

/// Whether the feature entry `entry` of a `[features]` table turns on `slang-runtime`: the feature itself, or the
/// feature of a dependency (`dep/slang-runtime`, `dep?/slang-runtime`).
fn names_slang_runtime(entry: &str) -> bool {
    entry == "slang-runtime" || entry.ends_with("/slang-runtime")
}

/// The problems of `cargo metadata` output for the packages `names`: each must be there, come from a path (the
/// unpacked tree under `unpacked`, forward slashes), and no feature reachable from its `default` feature may turn on
/// `slang-runtime`.
pub(crate) fn default_feature_problems(
    metadata: &str,
    names: &[&str],
    unpacked: &str,
) -> Result<Vec<String>, String> {
    let parsed =
        json::parse(metadata).map_err(|error| format!("cargo metadata output: {error}"))?;
    let packages = parsed
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata output has no `packages` array")?;
    let mut problems = Vec::new();
    for name in names {
        let Some(package) = packages
            .iter()
            .find(|package| package.get("name").and_then(Json::as_str) == Some(*name))
        else {
            problems.push(format!("cargo metadata lists no package `{name}`"));
            continue;
        };
        let manifest = package
            .get("manifest_path")
            .and_then(Json::as_str)
            .unwrap_or("")
            .replace('\\', "/");
        if !manifest.starts_with(unpacked) {
            problems.push(format!(
                "the package `{name}` of the consumer is not the unpacked one (its manifest is `{manifest}`)"
            ));
        }
        let Some(Json::Object(features)) = package.get("features") else {
            problems.push(format!("cargo metadata gives `{name}` no `features` table"));
            continue;
        };
        let table: BTreeMap<&str, Vec<&str>> = features
            .iter()
            .map(|(feature, list)| {
                (
                    feature.as_str(),
                    list.as_array()
                        .unwrap_or(&[])
                        .iter()
                        .filter_map(Json::as_str)
                        .collect(),
                )
            })
            .collect();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut pending: Vec<&str> = vec!["default"];
        while let Some(feature) = pending.pop() {
            if !seen.insert(feature) {
                continue;
            }
            for entry in table.get(feature).into_iter().flatten() {
                if names_slang_runtime(entry) {
                    problems.push(format!(
                        "a default feature of `{name}` turns on `slang-runtime` (`{feature}` lists `{entry}`)"
                    ));
                } else if table.contains_key(entry) {
                    pending.push(entry);
                }
            }
        }
    }
    Ok(problems)
}

/// Whether `child` is `parent` or below it, compared as text with forward slashes (and without case on Windows).
pub(crate) fn is_inside(parent: &Path, child: &Path) -> bool {
    let normal = |path: &Path| {
        let text = path.to_string_lossy().replace('\\', "/");
        let text = text.trim_end_matches('/').to_owned();
        if cfg!(windows) {
            text.to_ascii_lowercase()
        } else {
            text
        }
    };
    let (parent, child) = (normal(parent), normal(child));
    child == parent || child.starts_with(&format!("{parent}/"))
}

/// The relative paths (forward slashes) of the files below `directory`.
fn relative_files(directory: &Path) -> io::Result<Vec<String>> {
    Ok(snapshot(directory)?.into_keys().collect())
}

/// Runs `command` and returns its output when it succeeds.
fn run_step(step: &str, what: &str, mut command: Command) -> Result<Output, String> {
    let output = command
        .output()
        .map_err(|error| format!("check-precompiled {step}: could not launch cargo: {error}"))?;
    require_success(step, what, &output)?;
    Ok(output)
}

/// The two `--config` values with which `cargo package -p kernels` resolves `gust-compute` and `gust-compute-macros`
/// onto this checkout (E8 step 2): `patch.crates-io.<package>.path='<root>/crates/<directory>'`, with forward
/// slashes. The path is a TOML literal string, which has no escapes, so a root that holds a quote or a line break
/// is refused with a clear error and not written wrongly.
pub(crate) fn kernels_patch_configs(root: &Path) -> Result<[String; 2], String> {
    let root = root.to_string_lossy().replace('\\', "/");
    if root.contains(['\'', '\n', '\r']) {
        return Err(format!(
            "check-precompiled E8: the repository path holds a quote or a line break, which the `--config` patch of \
             `cargo package -p kernels` cannot carry: {root:?}"
        ));
    }
    Ok([
        format!("patch.crates-io.gust-compute.path='{root}/crates/gust'"),
        format!("patch.crates-io.gust-compute-macros.path='{root}/crates/gust-macros'"),
    ])
}

/// E8: the packages and the relocated consumer. Returns the summary. The work directory is deleted at the end, and
/// a failure to delete it fails the check.
fn e8_distribution(fixture: &Fixture, options: &Options) -> Result<String, String> {
    let work = WorkDir::create(
        "check-precompiled E8",
        "gust-dist",
        &fixture.root,
        &env::temp_dir(),
        options.work_dir.as_deref(),
    )?;
    let outcome = e8_steps(fixture, options, &work.path);
    work.finish(outcome)
}

/// E8 in the work directory `work`.
fn e8_steps(fixture: &Fixture, options: &Options, work: &Path) -> Result<String, String> {
    let read = |path: PathBuf| {
        fs::read_to_string(&path)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", path.display()))
    };
    let workspace_version =
        section_version(&read(fixture.root.join("Cargo.toml"))?, "workspace.package")
            .ok_or("check-precompiled E8: Cargo.toml has no [workspace.package] version")?;
    let kernels_version = section_version(
        &read(
            fixture
                .root
                .join(PRECOMPILED_DIR)
                .join("kernels")
                .join("Cargo.toml"),
        )?,
        "package",
    )
    .ok_or("check-precompiled E8: kernels/Cargo.toml has no [package] version")?;
    let versions = [
        workspace_version.clone(),
        workspace_version.clone(),
        workspace_version,
        kernels_version,
    ];
    let fixture_manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let committed: Vec<String> = relative_files(
        &fixture
            .root
            .join(PRECOMPILED_DIR)
            .join("kernels")
            .join("gust-artifacts"),
    )
    .map_err(|error| format!("check-precompiled E8: {error}"))?;

    // 1. `cargo package --list`: the allow-list of each package.
    let mut problems = Vec::new();
    for spec in &E8_PACKAGES {
        let mut args = vec!["package", "--list", "--allow-dirty", "--color", "never"];
        if spec.artifacts {
            args.extend(["--manifest-path", &fixture_manifest]);
        }
        args.extend(["-p", spec.name]);
        let output = run_step(
            "E8",
            &format!("cargo package --list -p {}", spec.name),
            fixture.cargo(&args),
        )?;
        let files = package_files(&String::from_utf8_lossy(&output.stdout));
        let required: &[String] = if spec.artifacts { &committed } else { &[] };
        problems.extend(package_problems(spec, &files, required));
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (cargo package --list) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    println!(
        "check-precompiled E8: the file lists of {} packages are on the allow-list (kernels lists every committed artifact)",
        E8_PACKAGES.len()
    );

    // 2. Package and unpack outside the repository. The three repository crates go in one call: cargo then resolves
    // them against each other, which a single `-p gust-compute` cannot (`gust-compute-macros` is not on crates.io).
    let repo_target = work.join("package-target");
    let kernels_target = work.join("package-kernels-target");
    let repo_target_arg = Fixture::target_dir_arg(&repo_target)?;
    let kernels_target_arg = Fixture::target_dir_arg(&kernels_target)?;
    run_step(
        "E8",
        "cargo package -p gust-compute-macros -p gust-compute -p gust-compute-wgpu",
        fixture.cargo(&[
            "package",
            "--allow-dirty",
            "--no-verify",
            "--color",
            "never",
            "-p",
            "gust-compute-macros",
            "-p",
            "gust-compute",
            "-p",
            "gust-compute-wgpu",
            "--target-dir",
            repo_target_arg,
        ]),
    )?;
    // `cargo package` resolves the dependencies of the packaged manifest against crates.io, where `kernels` asks for
    // `gust-compute = "0.1.0"`. What crates.io holds under a GUST name must not decide this check: before the rename
    // the requirement was for `gust`, which cargo met with the unrelated charting crate of that name (0.1.0 and 0.1.4
    // exist), and once the packages were renamed the placeholder `gust-compute 0.0.0` met nothing. So the two packages
    // that `kernels` reaches are patched onto their trees in the repository (`--config`, which `cargo package`
    // honours and which leaves the packaged manifest as it is): the resolution finds this checkout's `gust-compute`
    // and its `gust-compute-macros`, and `--no-verify` still skips the build of the unpacked package.
    let [patch_compute, patch_macros] = kernels_patch_configs(&fixture.root)?;
    run_step(
        "E8",
        "cargo package -p kernels",
        fixture.cargo(&[
            "package",
            "--allow-dirty",
            "--no-verify",
            "--color",
            "never",
            "--config",
            &patch_compute,
            "--config",
            &patch_macros,
            "--manifest-path",
            &fixture_manifest,
            "-p",
            "kernels",
            "--target-dir",
            kernels_target_arg,
        ]),
    )?;
    let crates = work.join("crates");
    let unpacked = work.join("unpacked");
    fs::create_dir_all(&crates)
        .and_then(|()| fs::create_dir_all(&unpacked))
        .map_err(|error| format!("check-precompiled E8: {error}"))?;
    for (spec, version) in E8_PACKAGES.iter().zip(&versions) {
        let file = format!("{}-{version}.crate", spec.name);
        let from = if spec.artifacts {
            &kernels_target
        } else {
            &repo_target
        }
        .join("package")
        .join(&file);
        fs::copy(&from, crates.join(&file))
            .map_err(|error| format!("check-precompiled E8: {}: {error}", fixture.shown(&from)))?;
        // `tar` runs in the unpacked directory and reads the archive by a relative path: a drive letter in an
        // argument would be read as a host name by some tars.
        let mut tar = Command::new("tar");
        tar.current_dir(&unpacked)
            .args(["-xzf", &format!("../crates/{file}")])
            .stdin(Stdio::null());
        let output = tar
            .output()
            .map_err(|error| format!("check-precompiled E8: could not run tar: {error}"))?;
        require_success("E8", &format!("tar -xzf {file}"), &output)?;
    }

    // 3. The unpacked trees are judged again, and the kernels tree is Slang-free.
    for (spec, version) in E8_PACKAGES.iter().zip(&versions) {
        let tree = unpacked.join(format!("{}-{version}", spec.name));
        let files = relative_files(&tree)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", tree.display()))?;
        let required: &[String] = if spec.artifacts { &committed } else { &[] };
        problems.extend(package_problems(spec, &files, required));
        if spec.artifacts {
            let contents: Vec<(String, String)> = snapshot(&tree)
                .map_err(|error| format!("check-precompiled E8: {error}"))?
                .into_iter()
                .map(|(name, bytes)| (name, String::from_utf8_lossy(&bytes).into_owned()))
                .collect();
            problems.extend(slang_free_package_problems(&contents));
        }
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (unpacked packages) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4. The consumer.
    let consumer = work.join("consumer");
    fs::create_dir_all(consumer.join("src"))
        .map_err(|error| format!("check-precompiled E8: {error}"))?;
    let write = |path: PathBuf, text: &str| {
        fs::write(&path, text)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", path.display()))
    };
    write(consumer.join("Cargo.toml"), &consumer_manifest(&versions))?;
    // The same `main` as E1's `app`, so the consumer proves the same run; the lock is seeded from the fixture's so that
    // `wgpu` and the rest resolve to the versions the fixture is tested with.
    write(
        consumer.join("src").join("main.rs"),
        &read(
            fixture
                .root
                .join(PRECOMPILED_DIR)
                .join("app")
                .join("src")
                .join("main.rs"),
        )?,
    )?;
    write(
        consumer.join("Cargo.lock"),
        &read(fixture.root.join(PRECOMPILED_DIR).join("Cargo.lock"))?,
    )?;
    let consumer_manifest_path = consumer.join("Cargo.toml");
    let consumer_manifest_arg = consumer_manifest_path
        .to_str()
        .ok_or("check-precompiled E8: the work directory is not valid UTF-8")?;

    // 4a. `cargo metadata` first: the features the registry consumer sees are the packaged ones.
    let output = run_step(
        "E8",
        "cargo metadata (the consumer)",
        fixture.cargo_in(
            &consumer,
            &[
                "metadata",
                "--format-version",
                "1",
                "--color",
                "never",
                "--manifest-path",
                consumer_manifest_arg,
            ],
        ),
    )?;
    let unpacked_prefix = unpacked.to_string_lossy().replace('\\', "/");
    let mut unused = unused_patch_lines(&String::from_utf8_lossy(&output.stderr));
    problems.extend(default_feature_problems(
        &String::from_utf8_lossy(&output.stdout),
        &["gust-compute", "gust-compute-wgpu"],
        &unpacked_prefix,
    )?);
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (cargo metadata) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4b. The release build under `native` in the stripped environment of E1.
    let target = work.join("target");
    let target_arg = Fixture::target_dir_arg(&target)?;
    let output = run_step(
        "E8",
        "cargo build --release (the consumer)",
        fixture.cargo_in(
            &consumer,
            &[
                "build",
                "--release",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                consumer_manifest_arg,
                "--target-dir",
                target_arg,
            ],
        ),
    )?;
    unused.extend(unused_patch_lines(&String::from_utf8_lossy(&output.stderr)));
    problems.extend(
        unused
            .into_iter()
            .map(|line| format!("cargo reported an unused patch: {line}")),
    );
    problems.extend(lock_problems(&read(consumer.join("Cargo.lock"))?));
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (patch and lock) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4c. Run it.
    let built = Fixture::app_binary(&target);
    let built = built.with_file_name(format!("dist-consumer{EXE_SUFFIX}"));
    if options.check_only {
        return Ok("passed (build only: --check-only skips the run, which needs an adapter): the four packages are on the allow-list and Slang-free, no default feature of gust-compute or gust-compute-wgpu turns on slang-runtime, and the consumer with registry-style dependencies built in release under native in the stripped environment, with every patch used and no `source` in the lock for the four packages".to_owned());
    }
    let run = fixture
        .command(&built)
        .output()
        .map_err(|error| format!("check-precompiled E8: could not run the consumer: {error}"))?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    print!("{stdout}");
    let report = judge_run(run.status.success(), &stdout, &stderr).map_err(|problems| {
        format!(
            "check-precompiled E8: the consumer: {}",
            problems.join("; ")
        )
    })?;
    Ok(format!(
        "passed: the four packages are on the allow-list and Slang-free, no default feature of gust-compute or gust-compute-wgpu turns on slang-runtime, and the consumer with registry-style dependencies on the unpacked packages built in release under native in the stripped environment (every patch used, no `source` in the lock for the four) and printed `{VERIFIED_PREFIX}{} elements` on the adapter `{}`",
        report.elements, report.adapter
    ))
}

// ---------------------------------------------------------------------------------------------------------------
// Scratch copies

/// Replaces the fixture's relative path to the repository (`path = "../../../`) with the absolute one, so that a
/// copy under the target directory still finds `gust` and `gust-wgpu`. `root` is written with forward slashes.
pub(crate) fn retarget_paths(manifest: &str, root: &str) -> String {
    manifest.replace("path = \"../../../", &format!("path = \"{root}/"))
}

/// Copies `tests/precompiled` to `destination` byte for byte, without a `target` directory, and points the copy's
/// `path` dependencies on the repository at its absolute location. An existing `destination` is deleted first.
fn copy_fixture(fixture: &Fixture, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        fs::remove_dir_all(destination).map_err(|error| {
            format!(
                "could not delete the old scratch copy {}: {error}",
                destination.display()
            )
        })?;
    }
    let source = fixture.root.join(PRECOMPILED_DIR);
    copy_tree(&source, destination).map_err(|error| {
        format!("could not copy {PRECOMPILED_DIR} to the scratch directory: {error}")
    })?;
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    for manifest in [
        destination.join("kernels").join("Cargo.toml"),
        destination.join("app").join("Cargo.toml"),
    ] {
        let text = fs::read_to_string(&manifest)
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
        fs::write(&manifest, retarget_paths(&text, &root))
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
    }
    Ok(())
}

/// Recursive copy of `source` into `destination`, skipping directories called `target`.
fn copy_tree(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            if entry.file_name() != "target" {
                copy_tree(&entry.path(), &target)?;
            }
        } else if kind.is_file() {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// The one file of `directory` that ends in `.extension` (and not in `.reflection.json`).
fn single_artifact(directory: &Path, extension: &str) -> Result<PathBuf, String> {
    let mut found = Vec::new();
    for entry in
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path
            .extension()
            .is_some_and(|candidate| candidate == extension)
        {
            found.push(path);
        }
    }
    match found.as_slice() {
        [one] => Ok(one.clone()),
        other => Err(format!(
            "expected exactly one .{extension} artifact in the scratch copy, found {}",
            other.len()
        )),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers

fn require_success(step: &str, what: &str, output: &Output) -> Result<(), String> {
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "check-precompiled {step}: {what} failed (exit {}):\n{}{}",
        exit_code(output.status),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn names_or_none(names: &[String]) -> String {
    if names.is_empty() {
        "none were set".to_owned()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
#[path = "tests/precompiled.rs"]
mod precompiled_tests;
