//! `cargo xtask check-precompiled`: the evidence that a production-shaped consumer of precompiled kernels needs no
//! Slang (D45, D46). The fixture is `tests/precompiled`, a workspace of its own like `tests/dual`: the `kernels` crate
//! holds one `#[gpu]` vector-add and its committed `gust-artifacts/` (written by `cargo gust build`), and `app`
//! dispatches it through `gust-wgpu` with `default-features = false` and checks the result.
//!
//! Every cargo child runs under `GUST_PROFILE=native`, the strict profile, set explicitly (R4.2), except the control
//! build of E2, which needs the bridge and runs under `dev` in its own target directory. The four checks:
//!
//! - **E1**: release build and run of `app` in an environment with Slang taken out. The child gets a copy of the
//!   parent's environment without every variable that names Slang (any name containing `SLANG`, `VULKAN_SDK`,
//!   `VK_SDK_PATH`) and without every `PATH` directory that holds `slangc` or `gust-slang-reflect`. The environment
//!   is then asserted slang-free: no such variable, no such directory, and the platform's lookup (`where` under `cmd`,
//!   `command -v` under `sh`) does not find `slangc` with the child's `PATH`. The build starts from `cargo clean -p gust
//!   -p gust-macros -p kernels -p app --release` in the same environment, because cargo's fingerprints track only
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
//! - **E4**: in another scratch copy, the artifact files are deleted. Under `native` the build fails with the one
//!   error of `gust::profile` (E0080), reported at the line of the kernel in `kernels/src/lib.rs`, which
//!   `tests/precompiled/e4-expected.txt` pins.
//!
//! Nothing here runs `slangc`, the native helper or `cargo gust`. The environment of the xtask process is never
//! changed; the stripped environment exists only in the children. `CARGO_NET_OFFLINE` and the rest of the parent's
//! environment pass through, so a machine that builds offline builds the fixture offline.

use std::{
    env,
    env::consts::EXE_SUFFIX,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use crate::{
    checks::RUSTFLAGS_VARIABLES,
    process::PROFILE_VARIABLE,
    ui::{Expectation, judge, parse_expected},
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
/// native helper, `reflect::native_compiler_path`) and `kernel.artifact` (`compile_reflected_with_helper`) are string
/// literals of code gated by the `slang-runtime` feature of `gust`. `kernel.slang` is not gated: `KERNEL_FILE` in
/// `gust/src/slang/mod.rs` is used by the ungated `compile_copy` and `directive_file_is_usable` too, so it is absent
/// from the measured binary only because the release linker drops dead code. It stays a marker for that measured
/// reason and is the weakest of the three. They were picked by measurement: the same `app` built with the bridge (the
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
        code.map_or_else(|| "?".to_owned(), |code| code.to_string()),
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

    /// [`Self::cargo`] with `GUST_PROFILE` set to `profile` instead of the strict one (the control of E2 only).
    fn cargo_under_profile(&self, args: &[&str], profile: &str) -> Command {
        let mut command = self.cargo(args);
        command.env(PROFILE_VARIABLE, profile);
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

/// `cargo xtask check-precompiled`.
pub(crate) fn check_precompiled(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("check-precompiled takes no arguments".into());
    }
    let root = workspace_root()?;
    let fixture = Fixture::new(&root);
    println!(
        "check-precompiled: cargo target directory {}",
        fixture.shown(&fixture.target_dir)
    );

    let report = e1_stripped_build_and_run(&fixture)?;
    let binary = e2_binary_and_features(&fixture)?;
    let identity = e3_mutated_artifact(&fixture)?;
    let located = e4_missing_artifact(&fixture)?;

    println!(
        "check-precompiled passed: E1 {report}. E2 {binary}. E3 {identity}. E4 {located}. The fixture is tests/precompiled; every child ran under {PROFILE_VARIABLE}={STRICT_PROFILE} except the E2 control ({PROFILE_VARIABLE}={CONTROL_PROFILE})."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// E1

/// E1: release build and run of `app` with Slang taken out of the environment. Returns the summary.
fn e1_stripped_build_and_run(fixture: &Fixture) -> Result<String, String> {
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
            "gust",
            "-p",
            "gust-macros",
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
        "cargo clean -p gust -p gust-macros -p kernels -p app",
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
        "check-precompiled E1: cargo clean of gust, gust-macros, kernels and app, then cargo build --release -p app, succeeded"
    );

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
    for package in ["gust", "gust-wgpu", "kernels"] {
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
fn e3_mutated_artifact(fixture: &Fixture) -> Result<String, String> {
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
        output
            .status
            .code()
            .map_or_else(|| "?".to_owned(), |code| code.to_string()),
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
