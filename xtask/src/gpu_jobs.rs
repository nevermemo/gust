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
//! `CARGO_PKG_*` and `CARGO_MANIFEST_PATH` variables read from `cargo metadata`, the `GUST_PROFILE` and helper variables
//! of every workspace cargo child, and, for the dynamic libraries Windows loads through `PATH`, the executable's `deps`
//! directory and its parent in front of `PATH`. The package variables xtask itself inherited from `cargo xtask` are
//! removed first, so a test never sees xtask's values. The variables that only the compiler reads
//! (`CARGO_PRIMARY_PACKAGE`, `CARGO_CRATE_NAME`, `CARGO_TARGET_TMPDIR`, `CARGO_BIN_EXE_*`) reach a test through `env!`,
//! which the executable already carries, and are not set. A package with a build script is refused (cargo would pass the
//! script's variables to its tests), as is a call with `--` arguments; `--gpu-jobs 1` runs those as one cargo call.
//!
//! A job that runs longer than [`JOB_TIME_LIMIT`] is killed with its child processes and named in the error; after any
//! failure or time-out the jobs still running are killed too, so no GPU test process outlives the run.
//!
//! `check-examples --gpu` (`examples_gpu.rs`) uses the same pieces for the examples' test executables: it passes `--ignored`,
//! gives its jobs a time limit of its own, and refuses a listing call that compiles anything (the build is the gate's
//! prebuild, not part of the run).

use std::{
    ffi::OsString,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    json::{self, Json},
    process::{exit_code, set_workspace_profile, without_colors, workspace_command},
    verify::{GPU_SUITE_CALL, GPU_SUITE_PIN, check_gpu_suite_counts},
};

/// How long one job may run. The slowest test executable takes about a minute and a quarter on the Radeon 890M, so this
/// is generous; a hang is killed and named instead of waiting for the outer time limit of the gate.
pub(crate) const JOB_TIME_LIMIT: Duration = Duration::from_secs(900);

/// How often a running job is checked for its end, the time limit and a cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

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
    let has_build_script = package
        .get("targets")
        .and_then(Json::as_array)
        .is_some_and(|targets| {
            targets.iter().any(|target| {
                target
                    .get("kind")
                    .and_then(Json::as_array)
                    .is_some_and(|kinds| {
                        kinds
                            .iter()
                            .any(|kind| kind.as_str() == Some("custom-build"))
                    })
            })
        });
    if has_build_script {
        return Err(format!(
            "{name} has a build script, whose variables cargo passes to its tests and this runner does not; use --gpu-jobs 1"
        ));
    }
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
        ("CARGO_MANIFEST_PATH", manifest.display().to_string()),
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
            "cargo's JSON output names no test executable of the package in {}, so there is nothing to run",
            manifest_dir.display()
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

/// The order in which jobs are started: the doc tests first (a rustdoc build and a run that lasts about as long as the
/// slowest executable), then the rest in print order. Printing stays in job order; a permutation of `0..jobs.len()`.
pub(crate) fn start_order(jobs: &[Job]) -> Vec<usize> {
    let (doc, rest): (Vec<usize>, Vec<usize>) =
        (0..jobs.len()).partition(|&index| jobs[index] == Job::Doc);
    doc.into_iter().chain(rest).collect()
}

/// Whether `name` is a variable cargo sets for the package under test (so a value inherited from the `cargo xtask`
/// process, which describes xtask, is wrong for a test of another package).
pub(crate) fn is_package_variable(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("CARGO_PKG_")
        || name.starts_with("CARGO_BIN_EXE_")
        || matches!(
            name.as_str(),
            "CARGO_MANIFEST_DIR"
                | "CARGO_MANIFEST_PATH"
                | "CARGO_PRIMARY_PACKAGE"
                | "CARGO_CRATE_NAME"
                | "CARGO_BIN_NAME"
                | "CARGO_TARGET_TMPDIR"
        )
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

/// One job's block: its header line and its whole output, standard output then standard error. The doc-test call's own
/// `Doc-tests` line on standard error is left out, so the header is the only one.
pub(crate) fn block(job: &Job, output: &JobOutput) -> String {
    let header = job.header();
    let stderr: String = output
        .stderr
        .split_inclusive('\n')
        .filter(|line| *job != Job::Doc || line.trim() != header.trim())
        .collect();
    format!(
        "{header}\n{}{}",
        terminated(&output.stdout),
        terminated(&stderr)
    )
}

/// Judges finished jobs, `results` in the order of `jobs` (`None` for a job that never started). `Ok` carries the
/// concatenation of each started job's header and standard output, which is the text the pin of the single call is
/// checked against; `Err` names every job that failed and says how many never started.
pub(crate) fn judge(jobs: &[Job], results: &[Option<JobOutput>]) -> Result<String, String> {
    judge_named(PACKAGE, jobs, results)
}

/// [`judge`] for the jobs of `subject` (a package, or a name for several), which the error names.
pub(crate) fn judge_named(
    subject: &str,
    jobs: &[Job],
    results: &[Option<JobOutput>],
) -> Result<String, String> {
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
        "{} of the {} jobs of {subject} failed: {}{}",
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

/// Runs `run` on every job of `jobs`, at most `width` at a time, starting them in the order `starts` (a permutation of
/// the job indices), and calls `emit` with each result on the calling thread in the order of `jobs` as soon as it and all
/// earlier results exist (so output is never interleaved and appears while later jobs still run). After a result for
/// which `failed` is true no further job starts; the ones running are awaited. The results are returned in the order of
/// `jobs`, `None` where a job never started.
pub(crate) fn run_in_order<J: Sync, R: Send>(
    jobs: &[J],
    starts: &[usize],
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
                    let slot = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&index) = starts.get(slot) else {
                        break;
                    };
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
        // A run that stopped early leaves gaps: print what finished after them too, in job order.
        for (index, result) in results.iter().enumerate().skip(emitted) {
            if let Some(result) = result {
                emit(index, result);
            }
        }
    });
    results
}

/// The header of the suite step when it runs as parts, `width` at a time (the single call has its own, `--gpu-jobs 1`).
pub(crate) fn step_label(width: usize) -> String {
    format!(
        "cargo {} as its test binaries and doc tests, {width} at a time",
        GPU_SUITE_CALL.join(" ")
    )
}

/// [`GPU_SUITE_CALL`] with `extra` arguments after it: the one source of the arguments of the calls this module makes.
pub(crate) fn call_with(extra: &[&str]) -> Vec<String> {
    GPU_SUITE_CALL
        .iter()
        .chain(extra)
        .map(|argument| (*argument).to_owned())
        .collect()
}

/// An error when a call cannot be run as parts: one with `--` arguments (a filter, `--nocapture`, `--include-ignored`)
/// would lose them in the doc-test call or in the executables, so it stays one cargo call.
pub(crate) fn refuse_call(call: &[&str]) -> Result<(), String> {
    if call.contains(&"--") {
        return Err(format!(
            "cargo {} has arguments for the test binaries, which the parallel run does not pass on; use --gpu-jobs 1",
            call.join(" ")
        ));
    }
    Ok(())
}

/// A process that [`supervise`] watches.
pub(crate) trait Process {
    type Exit;
    /// The way it ended, once it has.
    fn exit(&mut self) -> Option<Self::Exit>;
    /// Stops it and everything it started.
    fn kill(&mut self);
}

/// How a supervised process ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum End<T> {
    Finished(T),
    TimedOut,
    Cancelled,
}

/// Waits for `process`, checking it every `poll`: it ends by itself, or is killed after `limit` or as soon as `cancel`
/// is set.
pub(crate) fn supervise<P: Process>(
    process: &mut P,
    limit: Duration,
    poll: Duration,
    cancel: &AtomicBool,
) -> End<P::Exit> {
    let started = Instant::now();
    loop {
        if let Some(exit) = process.exit() {
            return End::Finished(exit);
        }
        if cancel.load(Ordering::SeqCst) {
            process.kill();
            return End::Cancelled;
        }
        if started.elapsed() >= limit {
            process.kill();
            return End::TimedOut;
        }
        thread::sleep(poll);
    }
}

/// Ends `child` and, on Windows, everything it started (`taskkill /T`; killing `cargo` alone leaves its test processes
/// running), then reaps it.
fn kill_tree(child: &mut Child) {
    if cfg!(windows) {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// A child process of a job as [`supervise`] sees it.
struct ChildProcess(Child);

impl Process for ChildProcess {
    /// Whether it succeeded, and how it ended for a message.
    type Exit = (bool, String);

    fn exit(&mut self) -> Option<Self::Exit> {
        match self.0.try_wait() {
            Ok(Some(status)) => Some((status.success(), format!("exit {}", exit_code(status)))),
            Ok(None) => None,
            Err(error) => {
                self.kill();
                Some((false, format!("could not wait for it: {error}")))
            }
        }
    }

    fn kill(&mut self) {
        kill_tree(&mut self.0);
    }
}

/// What one pipe of a child has delivered so far, filled by a thread of its own so that a full pipe never stops the child.
struct Capture {
    buffer: Arc<Mutex<Vec<u8>>>,
    reader: thread::JoinHandle<()>,
}

impl Capture {
    fn start(mut pipe: impl Read + Send + 'static) -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::clone(&buffer);
        let reader = thread::spawn(move || {
            let mut chunk = [0_u8; 8192];
            while let Ok(read) = pipe.read(&mut chunk) {
                if read == 0 {
                    break;
                }
                if let Ok(mut buffer) = shared.lock() {
                    buffer.extend_from_slice(&chunk[..read]);
                }
            }
        });
        Self { buffer, reader }
    }

    /// The text so far. With `complete` the reader is awaited first (the child ended and closed its pipe); without it
    /// (the child was killed, and a grandchild may still hold the pipe) nothing is awaited.
    fn text(self, complete: bool) -> String {
        if complete {
            let _ = self.reader.join();
        }
        self.buffer.lock().map_or_else(
            |_| String::new(),
            |bytes| String::from_utf8_lossy(&bytes).into_owned(),
        )
    }
}

/// `cargo metadata --format-version 1 --locked --no-deps` in `root`: its output, for [`package_of`].
pub(crate) fn read_metadata(root: &Path) -> Result<String, String> {
    let output = workspace_command("cargo")
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
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn read_package(root: &Path) -> Result<Package, String> {
    package_of(&read_metadata(root)?, PACKAGE)
}

/// The line of cargo's standard error that reports a compilation (`Compiling <crate>`), if any.
pub(crate) fn compiling_line(stderr: &str) -> Option<String> {
    without_colors(stderr)
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("Compiling "))
        .map(str::to_owned)
}

/// Runs `cargo <call>` (a `--no-run --message-format=json` call) in `root` and returns its standard output. With
/// `refuse_build` its standard error is shown after it ends and a compilation is an error that says to build first (a
/// build under a lock would be part of the measured run); otherwise it is shown as it arrives.
pub(crate) fn no_run_build(
    root: &Path,
    call: &[String],
    refuse_build: bool,
) -> Result<String, String> {
    let mut command = workspace_command("cargo");
    command.args(call).current_dir(root).stdin(Stdio::null());
    if !refuse_build {
        command.stderr(Stdio::inherit());
    }
    let output = command
        .output()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if refuse_build {
        eprint!("{stderr}");
    }
    if !output.status.success() {
        return Err(format!(
            "cargo {} failed (exit {})",
            call.join(" "),
            exit_code(output.status)
        ));
    }
    if refuse_build && let Some(line) = compiling_line(&stderr) {
        return Err(format!(
            "cargo {} compiled something (`{line}`): that call has built them now, so run this again (before the run, build them with `check-examples --gpu --prebuild`)",
            call.join(" ")
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The jobs of the call: the executables `cargo <GPU_SUITE_CALL> --no-run --message-format=json` names (its standard
/// error is shown, so a build that happens under the lock is in the log) and the doc tests.
fn read_jobs(root: &Path, package: &Package) -> Result<Vec<Job>, String> {
    let call = call_with(&["--no-run", "--message-format=json"]);
    test_executables(&no_run_build(root, &call, false)?, &package.manifest_dir).map(plan_jobs)
}

/// The command of a test executable: the environment of a cargo test run in the package's directory.
pub(crate) fn executable_command(executable: &TestExecutable, package: &Package) -> Command {
    let mut command = Command::new(&executable.path);
    set_workspace_profile(&mut command);
    command.current_dir(&package.manifest_dir);
    for (name, _) in std::env::vars_os() {
        if name.to_str().is_some_and(is_package_variable) {
            command.env_remove(name);
        }
    }
    for (name, value) in &package.environment {
        command.env(name, value);
    }
    if let Some(path) = library_path(&executable.path, std::env::var_os("PATH")) {
        command.env("PATH", path);
    }
    command
}

/// The command of a job: the executable with the environment of a cargo test run, or the doc-test call.
fn job_command(root: &Path, package: &Package, job: &Job) -> Command {
    match job {
        Job::Executable(executable) => executable_command(executable, package),
        Job::Doc => {
            let mut command = workspace_command("cargo");
            command.args(call_with(&["--doc"])).current_dir(root);
            command
        }
    }
}

/// A job that did not run to its end.
fn unfinished(status: String, stdout: String, stderr: String) -> JobOutput {
    JobOutput {
        success: false,
        status,
        stdout,
        stderr,
    }
}

/// Runs one job to its end, killing it (and what it started) after [`JOB_TIME_LIMIT`] or when `cancel` is set, and
/// captures its output.
fn run_job(root: &Path, package: &Package, job: &Job, cancel: &AtomicBool) -> JobOutput {
    run_command(job_command(root, package, job), JOB_TIME_LIMIT, cancel)
}

/// [`run_job`] for a command that is already built.
pub(crate) fn run_command(mut command: Command, limit: Duration, cancel: &AtomicBool) -> JobOutput {
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return unfinished(
                "did not start".to_owned(),
                String::new(),
                format!("could not launch the job: {error}"),
            );
        }
    };
    let stdout = child.stdout.take().map(Capture::start);
    let stderr = child.stderr.take().map(Capture::start);
    let end = supervise(&mut ChildProcess(child), limit, POLL_INTERVAL, cancel);
    let complete = matches!(end, End::Finished(_));
    let stdout = stdout.map_or_else(String::new, |capture| capture.text(complete));
    let stderr = stderr.map_or_else(String::new, |capture| capture.text(complete));
    match end {
        End::Finished((success, status)) => JobOutput {
            success,
            status,
            stdout,
            stderr,
        },
        End::TimedOut => unfinished(
            format!("timed out after {} s and was killed", limit.as_secs()),
            stdout,
            stderr,
        ),
        End::Cancelled => unfinished(
            "killed because another job failed".to_owned(),
            stdout,
            stderr,
        ),
    }
}

/// The Gpu suite's call of [`GPU_SUITE_CALL`] as its parts, `width` at a time (see the module text), judged against
/// [`GPU_SUITE_PIN`] like the single call. A failing or timed-out job is named, and the jobs still running are killed.
pub(crate) fn run_parallel_gpu_suite(root: &Path, width: usize) -> Result<(), String> {
    refuse_call(&GPU_SUITE_CALL)?;
    let package = read_package(root)?;
    let jobs = read_jobs(root, &package)?;
    println!(
        "   {} job(s): {} test executable(s) and the doc tests",
        jobs.len(),
        jobs.len() - 1
    );
    let results = run_printed(
        &jobs,
        &start_order(&jobs),
        width,
        &|job| job.clone(),
        &|job, cancel| run_job(root, &package, job, cancel),
    );
    let counted = judge(&jobs, &results)?;
    check_gpu_suite_counts(&GPU_SUITE_CALL.map(str::to_owned), &counted, GPU_SUITE_PIN)
}

/// Runs every job of `jobs` through `run` (given the cancellation flag, which a failed job sets so that the others are
/// killed), at most `width` at a time in the order `starts`, and prints each job's [`block`] (`job_of` names the [`Job`]
/// it prints under) in the order of `jobs` as soon as it and the ones before it have ended. The results, `None` for a job
/// that never started, are for [`judge`] and [`judge_named`].
pub(crate) fn run_printed<J: Sync>(
    jobs: &[J],
    starts: &[usize],
    width: usize,
    job_of: &impl Fn(&J) -> Job,
    run: &(impl Fn(&J, &AtomicBool) -> JobOutput + Sync),
) -> Vec<Option<JobOutput>> {
    let cancel = AtomicBool::new(false);
    run_in_order(
        jobs,
        starts,
        width,
        &|job| run(job, &cancel),
        &|output| {
            let failed = !output.success;
            if failed {
                cancel.store(true, Ordering::SeqCst);
            }
            failed
        },
        |index, output| {
            let mut stdout = io::stdout().lock();
            // Echoing is best effort; the captured copy is what the guard reads.
            let _ = stdout.write_all(block(&job_of(&jobs[index]), output).as_bytes());
            let _ = stdout.flush();
        },
    )
}

/// The environment variable that sets how many example mains `check-artifacts` (and the other modes that run the
/// examples' mains) start at a time: a whole number from 1 (one after another) up; unset means [`DEFAULT_GPU_JOBS`].
pub(crate) const EXAMPLE_JOBS_VARIABLE: &str = "GUST_EXAMPLE_JOBS";

/// The width named by the value of [`EXAMPLE_JOBS_VARIABLE`] (`None` when it is unset).
pub(crate) fn example_jobs(value: Option<&str>) -> Result<usize, String> {
    let Some(text) = value else {
        return Ok(DEFAULT_GPU_JOBS);
    };
    match text.trim().parse::<usize>() {
        Ok(width) if width >= 1 => Ok(width),
        _ => Err(format!(
            "{EXAMPLE_JOBS_VARIABLE} must be a whole number of at least 1, got `{text}`"
        )),
    }
}

/// Runs each cargo call of `calls` (`cargo run` of one example main) in `root`, at most `width` at a time, and calls
/// `emit` with each call's output and its own run time in the order of `calls`, as soon as it and the ones before it
/// have ended (see [`run_in_order`]). A failing call does not stop the others: every call runs, so the log names every
/// failure. A call is killed after [`JOB_TIME_LIMIT`]. The results are in the order of `calls`.
pub(crate) fn run_example_calls(
    root: &Path,
    calls: &[Vec<String>],
    width: usize,
    emit: impl FnMut(usize, &(JobOutput, Duration)),
) -> Vec<Option<(JobOutput, Duration)>> {
    let never = AtomicBool::new(false);
    let starts = (0..calls.len()).collect::<Vec<_>>();
    run_in_order(
        calls,
        &starts,
        width,
        &|call| {
            let started = Instant::now();
            let mut command = workspace_command("cargo");
            command.args(call).current_dir(root);
            (
                run_command(command, JOB_TIME_LIMIT, &never),
                started.elapsed(),
            )
        },
        &|_| false,
        emit,
    )
}

/// [`run_job`] for a test executable that [`executable_command`] builds with `arguments`, killed after `limit`.
pub(crate) fn run_executable(
    executable: &TestExecutable,
    package: &Package,
    arguments: &[&str],
    limit: Duration,
    cancel: &AtomicBool,
) -> JobOutput {
    let mut command = executable_command(executable, package);
    command.args(arguments);
    run_command(command, limit, cancel)
}
