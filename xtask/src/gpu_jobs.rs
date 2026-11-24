//! The Gpu suite's `cargo test --locked -p gust-compute-wgpu` call, run as its parts (m14.s8g, gates S8g).
//!
//! Cargo runs the test executables of the crate one after another, so while the slowest ones finish their last tests the
//! adapter and most cores idle. This module asks cargo for the executables (`--no-run --message-format=json`: the build
//! of the gate's prebuild, so it compiles nothing), runs them and one `--doc` call at most `width` at a time, and prints
//! each job's whole output as one block in a fixed order (cargo's: the unit tests first, then the integration targets by
//! name, then the doc tests), never interleaved. The concatenated standard output of the jobs is what
//! `check_gpu_suite_counts` judges, as for the single call (`--gpu-jobs 1`), so every `test result:` line and every
//! failure stays visible to the same pin.
//!
//! Each executable runs the way cargo runs a test: in the package's directory, with the `CARGO_MANIFEST_DIR` and
//! `CARGO_PKG_*` variables read from `cargo metadata`, `CARGO_PRIMARY_PACKAGE`, the `GUST_PROFILE` and helper variables of
//! every workspace cargo child, and, for the dynamic libraries Windows loads through `PATH`, the executable's `deps`
//! directory and its parent in front of `PATH`. The variables that only the compiler reads (`CARGO_CRATE_NAME`,
//! `CARGO_TARGET_TMPDIR`, `CARGO_BIN_EXE_*`) reach a test through `env!`, which the executable already carries.

use std::{
    ffi::OsString,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use crate::{
    json::{self, Json},
    process::{cargo_program, cargo_test_command, exit_code, set_workspace_profile},
    verify::{GPU_SUITE_CALL, GPU_SUITE_PIN, check_gpu_suite_counts},
};

/// The package whose tests the Gpu suite runs.
const PACKAGE: &str = "gust-compute-wgpu";

/// The name of the package's library for the header of the doc tests (its manifest renames the library).
const LIBRARY: &str = "gust_wgpu";

/// How many jobs the Gpu suite runs at a time when `--gpu-jobs` is not given.
pub(crate) const DEFAULT_GPU_JOBS: usize = 3;

/// One test executable that `cargo test --no-run` built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TestExecutable {
    /// The line printed before its output, like cargo's `Running` line.
    pub(crate) header: String,
    pub(crate) path: PathBuf,
    /// An integration target (`tests/<name>.rs`); the unit tests of the library are not.
    pub(crate) integration: bool,
    pub(crate) name: String,
}

/// One part of the call: a test executable, or the doc tests (a `cargo test --doc` call of its own).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Job {
    Executable(TestExecutable),
    Doc,
}

impl Job {
    pub(crate) fn header(&self) -> String {
        match self {
            Self::Executable(executable) => executable.header.clone(),
            Self::Doc => format!("   Doc-tests {LIBRARY}"),
        }
    }
}

/// What a finished job printed and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct JobOutput {
    pub(crate) success: bool,
    /// How it ended, for a message: `exit 101`, or why it did not start.
    pub(crate) status: String,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// The package as `cargo metadata` describes it: its directory and the variables cargo gives its test executables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Package {
    pub(crate) manifest_dir: PathBuf,
    pub(crate) environment: Vec<(&'static str, String)>,
}

/// `text` ending in a newline (unless empty), so that the blocks of two jobs never run together.
fn terminated(text: &str) -> String {
    if text.is_empty() || text.ends_with('\n') {
        text.to_owned()
    } else {
        format!("{text}\n")
    }
}

/// The version split as cargo's `CARGO_PKG_VERSION_*` variables have it: major, minor, patch and pre-release.
fn version_parts(version: &str) -> [String; 4] {
    let core_end = version.find(['-', '+']).unwrap_or(version.len());
    let (core, rest) = version.split_at(core_end);
    let pre = rest
        .strip_prefix('-')
        .map_or("", |pre| pre.split('+').next().unwrap_or(""));
    let mut numbers = core.splitn(3, '.');
    let mut next = || numbers.next().unwrap_or("").to_owned();
    [next(), next(), next(), pre.to_owned()]
}

/// The package `name` of `cargo metadata --no-deps` output: its directory and the `CARGO_*` variables of a test run.
pub(crate) fn package_of(metadata: &str, name: &str) -> Result<Package, String> {
    let document =
        json::parse(metadata).map_err(|error| format!("cargo metadata output: {error}"))?;
    let package = document
        .get("packages")
        .and_then(Json::as_array)
        .and_then(|packages| {
            packages
                .iter()
                .find(|package| package.get("name").and_then(Json::as_str) == Some(name))
        })
        .ok_or_else(|| format!("cargo metadata names no package `{name}`"))?;
    let text = |key: &str| {
        package
            .get(key)
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let manifest = PathBuf::from(text("manifest_path"));
    let manifest_dir = manifest
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .ok_or_else(|| format!("cargo metadata gives no manifest path for `{name}`"))?
        .to_path_buf();
    let authors = package
        .get("authors")
        .and_then(Json::as_array)
        .map(|authors| {
            authors
                .iter()
                .filter_map(Json::as_str)
                .collect::<Vec<_>>()
                .join(":")
        })
        .unwrap_or_default();
    let version = text("version");
    let [major, minor, patch, pre] = version_parts(&version);
    let environment = vec![
        ("CARGO_MANIFEST_DIR", manifest_dir.display().to_string()),
        ("CARGO_PKG_NAME", text("name")),
        ("CARGO_PKG_VERSION", version),
        ("CARGO_PKG_VERSION_MAJOR", major),
        ("CARGO_PKG_VERSION_MINOR", minor),
        ("CARGO_PKG_VERSION_PATCH", patch),
        ("CARGO_PKG_VERSION_PRE", pre),
        ("CARGO_PKG_AUTHORS", authors),
        ("CARGO_PKG_DESCRIPTION", text("description")),
        ("CARGO_PKG_HOMEPAGE", text("homepage")),
        ("CARGO_PKG_REPOSITORY", text("repository")),
        ("CARGO_PKG_LICENSE", text("license")),
        ("CARGO_PKG_LICENSE_FILE", text("license_file")),
        ("CARGO_PKG_RUST_VERSION", text("rust_version")),
        ("CARGO_PKG_README", text("readme")),
        ("CARGO_PRIMARY_PACKAGE", "1".to_owned()),
    ];
    Ok(Package {
        manifest_dir,
        environment,
    })
}

/// The test executables that `cargo test --no-run --message-format=json` built for the package in `manifest_dir`, in
/// cargo's run order: the unit tests of the library, then the integration targets by name. The record of a test build
/// has `profile.test` and a non-null `executable`; the same executable named twice counts once. None at all is an error,
/// so a changed output format cannot turn the suite into a run of nothing.
pub(crate) fn test_executables(
    stdout: &str,
    manifest_dir: &Path,
) -> Result<Vec<TestExecutable>, String> {
    let manifest = manifest_dir.join("Cargo.toml");
    let mut found: Vec<TestExecutable> = Vec::new();
    for line in stdout.lines().map(str::trim) {
        if !line.starts_with('{') {
            continue;
        }
        let Ok(message) = json::parse(line) else {
            continue;
        };
        let is_test_build = message.get("reason").and_then(Json::as_str)
            == Some("compiler-artifact")
            && message
                .get("profile")
                .and_then(|profile| profile.get("test"))
                == Some(&Json::Bool(true))
            && message
                .get("manifest_path")
                .and_then(Json::as_str)
                .map(Path::new)
                == Some(manifest.as_path());
        let (Some(path), Some(target)) = (
            message.get("executable").and_then(Json::as_str),
            message.get("target"),
        ) else {
            continue;
        };
        if !is_test_build || found.iter().any(|known| known.path == Path::new(path)) {
            continue;
        }
        let integration = target
            .get("kind")
            .and_then(Json::as_array)
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("test")));
        let source = target.get("src_path").and_then(Json::as_str).unwrap_or("");
        let relative = Path::new(source)
            .strip_prefix(manifest_dir)
            .unwrap_or(Path::new(source))
            .display()
            .to_string()
            .replace('\\', "/");
        found.push(TestExecutable {
            header: format!(
                "     Running {}{relative} ({path})",
                if integration { "" } else { "unittests " }
            ),
            path: PathBuf::from(path),
            integration,
            name: target
                .get("name")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_owned(),
        });
    }
    if found.is_empty() {
        return Err(format!(
            "cargo's JSON output names no test executable of {PACKAGE}, so there is nothing to run"
        ));
    }
    found.sort_by(|a, b| (a.integration, &a.name).cmp(&(b.integration, &b.name)));
    Ok(found)
}

/// The jobs of a run, in the order their output is printed: the test executables, then the doc tests.
pub(crate) fn plan_jobs(executables: Vec<TestExecutable>) -> Vec<Job> {
    executables
        .into_iter()
        .map(Job::Executable)
        .chain([Job::Doc])
        .collect()
}

/// `PATH` for a test executable: the `deps` directory it lies in and that directory's parent (where cargo puts the
/// libraries a test may load) in front of `existing`. `None` when the executable has no such directories or the result
/// cannot be joined.
pub(crate) fn library_path(executable: &Path, existing: Option<OsString>) -> Option<OsString> {
    let deps = executable.parent()?;
    let profile = deps.parent()?;
    let rest = existing.unwrap_or_default();
    std::env::join_paths(
        [deps.to_path_buf(), profile.to_path_buf()]
            .into_iter()
            .chain(std::env::split_paths(&rest)),
    )
    .ok()
}

/// One job's block: its header line and its whole output, standard output then standard error.
pub(crate) fn block(job: &Job, output: &JobOutput) -> String {
    format!(
        "{}\n{}{}",
        job.header(),
        terminated(&output.stdout),
        terminated(&output.stderr)
    )
}

/// Judges finished jobs, `results` in the order of `jobs` (`None` for a job that never started). `Ok` carries the
/// concatenation of each started job's header and standard output, which is the text the pin of the single call is
/// checked against; `Err` names every job that failed and says how many never started.
pub(crate) fn judge(jobs: &[Job], results: &[Option<JobOutput>]) -> Result<String, String> {
    let mut counted = String::new();
    let mut failed = Vec::new();
    let mut not_started = 0;
    for (job, result) in jobs.iter().zip(results) {
        let Some(output) = result else {
            not_started += 1;
            continue;
        };
        counted.push_str(&format!("{}\n{}", job.header(), terminated(&output.stdout)));
        if !output.success {
            failed.push(format!("`{}` ({})", job.header().trim(), output.status));
        }
    }
    if failed.is_empty() {
        return Ok(counted);
    }
    Err(format!(
        "{} of the {} jobs of {PACKAGE} failed: {}{}",
        failed.len(),
        jobs.len(),
        failed.join("; "),
        if not_started == 0 {
            String::new()
        } else {
            format!("; {not_started} job(s) were not started after the first failure")
        }
    ))
}

/// Runs `run` on every job of `jobs`, at most `width` at a time, in the order of `jobs`, and calls `emit` with each
/// result on the calling thread in that same order as soon as it and all earlier results exist (so output is never
/// interleaved and appears while later jobs still run). After a result for which `failed` is true no further job starts;
/// the ones running finish. The results are returned in the order of `jobs`, `None` where a job never started.
pub(crate) fn run_in_order<J: Sync, R: Send>(
    jobs: &[J],
    width: usize,
    run: &(impl Fn(&J) -> R + Sync),
    failed: &(impl Fn(&R) -> bool + Sync),
    mut emit: impl FnMut(usize, &R),
) -> Vec<Option<R>> {
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let mut results: Vec<Option<R>> = jobs.iter().map(|_| None).collect();
    let (sender, receiver) = mpsc::channel::<(usize, R)>();
    let mut emitted = 0;
    thread::scope(|scope| {
        for _ in 0..width.clamp(1, jobs.len().max(1)) {
            let sender = sender.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    let Some(job) = jobs.get(index) else {
                        break;
                    };
                    let result = run(job);
                    if failed(&result) {
                        stop.store(true, Ordering::SeqCst);
                    }
                    if sender.send((index, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for (index, result) in receiver {
            results[index] = Some(result);
            while let Some(Some(ready)) = results.get(emitted) {
                emit(emitted, ready);
                emitted += 1;
            }
        }
    });
    results
}

/// `cargo metadata --format-version 1 --locked --no-deps` in `root`.
fn read_package(root: &Path) -> Result<Package, String> {
    let output = Command::new(cargo_program())
        .args(["metadata", "--format-version", "1", "--locked", "--no-deps"])
        .current_dir(root)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("could not launch cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed (exit {}): {}",
            exit_code(output.status),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    package_of(&String::from_utf8_lossy(&output.stdout), PACKAGE)
}

/// The jobs of the call: the executables `cargo test --locked -p gust-compute-wgpu --no-run --message-format=json` names
/// (its standard error is shown, so a build that happens under the lock is in the log) and the doc tests.
fn read_jobs(root: &Path, package: &Package) -> Result<Vec<Job>, String> {
    let output = cargo_test_command(None)
        .args([
            "test",
            "--locked",
            "-p",
            PACKAGE,
            "--no-run",
            "--message-format=json",
        ])
        .current_dir(root)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo test --locked -p {PACKAGE} --no-run failed (exit {})",
            exit_code(output.status)
        ));
    }
    test_executables(
        &String::from_utf8_lossy(&output.stdout),
        &package.manifest_dir,
    )
    .map(plan_jobs)
}

/// The command of a job: the executable with the environment of a cargo test run, or the doc-test call.
fn job_command(root: &Path, package: &Package, job: &Job) -> Command {
    match job {
        Job::Executable(executable) => {
            let mut command = Command::new(&executable.path);
            set_workspace_profile(&mut command);
            command.current_dir(&package.manifest_dir);
            for (name, value) in &package.environment {
                command.env(name, value);
            }
            if let Some(path) = library_path(&executable.path, std::env::var_os("PATH")) {
                command.env("PATH", path);
            }
            command
        }
        Job::Doc => {
            let mut command = cargo_test_command(None);
            command
                .args(["test", "--locked", "-p", PACKAGE, "--doc"])
                .current_dir(root);
            command
        }
    }
}

/// Runs one job to its end and captures its output.
fn run_job(root: &Path, package: &Package, job: &Job) -> JobOutput {
    match job_command(root, package, job)
        .stdin(Stdio::null())
        .output()
    {
        Ok(output) => JobOutput {
            success: output.status.success(),
            status: format!("exit {}", exit_code(output.status)),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        },
        Err(error) => JobOutput {
            success: false,
            status: "did not start".to_owned(),
            stdout: String::new(),
            stderr: format!("could not launch the job: {error}"),
        },
    }
}

/// The Gpu suite's call of [`GPU_SUITE_CALL`] as its parts, `width` at a time (see the module text), judged against
/// [`GPU_SUITE_PIN`] like the single call. A failing job is named.
pub(crate) fn run_parallel_gpu_suite(root: &Path, width: usize) -> Result<(), String> {
    let package = read_package(root)?;
    let jobs = read_jobs(root, &package)?;
    println!(
        "   {} job(s) ({} test executable(s) and the doc tests), {width} at a time",
        jobs.len(),
        jobs.len() - 1
    );
    let results = run_in_order(
        &jobs,
        width,
        &|job| run_job(root, &package, job),
        &|output| !output.success,
        |index, output| {
            let mut stdout = io::stdout().lock();
            // Echoing is best effort; the captured copy is what the guard reads.
            let _ = stdout.write_all(block(&jobs[index], output).as_bytes());
            let _ = stdout.flush();
        },
    );
    let counted = judge(&jobs, &results)?;
    check_gpu_suite_counts(&GPU_SUITE_CALL.map(str::to_owned), &counted, GPU_SUITE_PIN)
}
