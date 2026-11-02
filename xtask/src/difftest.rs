//! `cargo xtask check-difftest`: the gate around the differential test harness `tests/difftest` (plan M5, R1-13).
//!
//! The harness is `difftest-run`; this command does not reimplement any of it. It runs the binary for the requested
//! seeds, reads the `report.json` it wrote, and fails unless the report says what was asked for: the mode, the seeds,
//! `programs_executed` equal to `16 x seeds`, no mismatch, and (on the GPU tier) an adapter. On the GPU tier it also
//! runs the `short_circuit` test of the harness crate with `--ignored`, the one test no other gate runs.
//!
//! ```text
//! cargo xtask check-difftest [--seeds <list|range|random:N>] [--cpu-only] [--report PATH]
//! ```
//!
//! The default is all eight seeds of `tests/difftest/seeds.txt` on the machine's adapter. `--cpu-only` compares the
//! interpreter with the CPU twin only (the Unit tier: no adapter, no Slang) and never runs the GPU test. `random:N`
//! draws N seeds from the clock, prints them, and passes them on as a list, so a failure can be reproduced with
//! `--seeds <list>`. The report is compared by fields, never by bytes: it is deterministic except for `timing` and
//! `adapter`.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    checks::tests_not_ok,
    json::{self, Json},
    process::{require_tests_ran, sum_test_results, workspace_command},
    workspace::{cargo_target_dir, workspace_root},
};

/// Programs the generator keeps per seed (`PROGRAMS_PER_SEED` of the harness crate; the report must say so).
pub(crate) const PROGRAMS_PER_SEED: u64 = 16;

/// The corpus of `tests/difftest/seeds.txt`. A test requires this to be the first column of that file.
pub(crate) const DEFAULT_SEEDS: &str = "1-8";

const MANIFEST: &str = "tests/difftest/Cargo.toml";

/// The GPU half of `tests/difftest/tests/short_circuit.rs`; it must be reported `ok`, not merely not failed.
pub(crate) const SHORT_CIRCUIT_GPU_TEST: &str = "the_gpu_skips_an_out_of_bounds_right_operand";

/// `random:N` draws between 1 and this many seeds.
const MAX_RANDOM_SEEDS: usize = 64;

/// A range such as `1-4` may span at most this many seeds (the harness refuses larger ones too).
const MAX_RANGE: u64 = 4096;

const USAGE: &str = "usage: cargo xtask check-difftest [--seeds <list|range|random:N>] [--cpu-only] [--report PATH]";

/// Which parties run: all three on the adapter, or the interpreter against the CPU twin only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Gpu,
    CpuOnly,
}

impl Mode {
    /// The `mode` string of `report.json`.
    pub(crate) fn report_name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::CpuOnly => "cpu-only",
        }
    }
}

/// One step of `check-difftest`, in the order it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DifftestStep {
    /// `difftest-run` for the seeds, then the judgement of its report.
    Batch,
    /// `cargo test --test short_circuit -- --ignored` of the harness crate: the GPU must skip a right operand that
    /// would read out of bounds. Gpu tier only.
    ShortCircuitGpu,
}

/// The steps of a mode. The GPU test belongs to the adapter run and to nothing else.
pub(crate) fn steps(mode: Mode) -> Vec<DifftestStep> {
    match mode {
        Mode::Gpu => vec![DifftestStep::Batch, DifftestStep::ShortCircuitGpu],
        Mode::CpuOnly => vec![DifftestStep::Batch],
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) seeds: Vec<u64>,
    pub(crate) mode: Mode,
    pub(crate) report: Option<PathBuf>,
    /// Set for `random:N`, so the summary can say the seeds were drawn.
    pub(crate) random: bool,
}

impl Options {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        Self::parse_with(args, clock_entropy())
    }

    /// [`Options::parse`] with the entropy of `random:N` given, so a test can fix it.
    pub(crate) fn parse_with(args: &[String], entropy: u64) -> Result<Self, String> {
        let mut spec: Option<String> = None;
        let mut cpu_only = false;
        let mut report = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--seeds" => {
                    let value = iter.next().ok_or("--seeds needs a value")?;
                    if spec.replace(value.clone()).is_some() {
                        return Err(format!("--seeds given twice\n{USAGE}"));
                    }
                }
                "--cpu-only" => cpu_only = true,
                "--report" => {
                    let value = iter.next().ok_or("--report needs a path")?;
                    if value.is_empty() || report.replace(PathBuf::from(value)).is_some() {
                        return Err(format!("--report needs one non-empty path\n{USAGE}"));
                    }
                }
                other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
            }
        }
        let spec = spec.as_deref().unwrap_or(DEFAULT_SEEDS);
        let (seeds, random) = match spec.strip_prefix("random:") {
            Some(count) => (random_seeds(count, entropy)?, true),
            None => (parse_seeds(spec)?, false),
        };
        Ok(Self {
            seeds,
            mode: if cpu_only { Mode::CpuOnly } else { Mode::Gpu },
            report,
            random,
        })
    }

    /// The `programs_executed` a passing run must report.
    pub(crate) fn expected_programs(&self) -> u64 {
        PROGRAMS_PER_SEED * self.seeds.len() as u64
    }
}

/// Parses `1-4`, `3`, `1,3,5-6` into the sorted, duplicate-free list of seeds (the grammar of `difftest-run`, so the
/// list this command expects is the list the harness runs).
pub(crate) fn parse_seeds(text: &str) -> Result<Vec<u64>, String> {
    let mut seeds = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        let number = |s: &str| {
            s.trim()
                .parse::<u64>()
                .map_err(|error| format!("seed {s:?}: {error}"))
        };
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last) = (number(first)?, number(last)?);
                if first > last {
                    return Err(format!("seed range {part:?} is empty"));
                }
                if last - first >= MAX_RANGE {
                    return Err(format!("seed range {part:?} is larger than {MAX_RANGE}"));
                }
                seeds.extend(first..=last);
            }
            None => seeds.push(number(part)?),
        }
    }
    seeds.sort_unstable();
    seeds.dedup();
    if seeds.is_empty() {
        return Err("no seeds".into());
    }
    Ok(seeds)
}

fn clock_entropy() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    nanos ^ (u64::from(std::process::id()) << 32)
}

/// `count` distinct seeds in `10_000..1_000_010_000`, sorted, from `entropy` (splitmix64). The range stays clear of the
/// corpus seeds, so a random run adds programs instead of repeating the pinned ones.
pub(crate) fn random_seeds(count: &str, entropy: u64) -> Result<Vec<u64>, String> {
    let wanted: usize = count
        .trim()
        .parse()
        .map_err(|error| format!("random:{count}: {error}"))?;
    if wanted == 0 || wanted > MAX_RANDOM_SEEDS {
        return Err(format!(
            "random:{count}: the count must be between 1 and {MAX_RANDOM_SEEDS}"
        ));
    }
    let mut state = entropy;
    let mut seeds = Vec::new();
    while seeds.len() < wanted {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let seed = 10_000 + z % 1_000_000_000;
        if !seeds.contains(&seed) {
            seeds.push(seed);
        }
    }
    seeds.sort_unstable();
    Ok(seeds)
}

/// The seeds as `difftest-run --seeds` takes them.
pub(crate) fn seed_list(seeds: &[u64]) -> String {
    seeds
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The arguments of the cargo call that runs the batch.
pub(crate) fn batch_args(options: &Options, target: &Path, report: &Path) -> Vec<String> {
    let mut args: Vec<String> = [
        "run",
        "--locked",
        "--manifest-path",
        MANIFEST,
        "--bin",
        "difftest-run",
        "--",
        "--seeds",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    args.push(seed_list(&options.seeds));
    if options.mode == Mode::CpuOnly {
        args.push("--cpu-only".to_owned());
    }
    args.push("--report".to_owned());
    args.push(report.to_string_lossy().into_owned());
    args.push("--target-dir".to_owned());
    args.push(target.to_string_lossy().into_owned());
    args
}

/// The arguments of the cargo call that runs the GPU short-circuit test of the harness crate.
pub(crate) fn short_circuit_args() -> Vec<String> {
    [
        "test",
        "--locked",
        "--manifest-path",
        MANIFEST,
        "--test",
        "short_circuit",
        "--",
        "--ignored",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect()
}

/// One disagreement of a report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MismatchRecord {
    pub(crate) program: String,
    pub(crate) seed: u64,
    pub(crate) index: u64,
    pub(crate) attempt: u64,
    pub(crate) kind: String,
    pub(crate) elements_differing: u64,
    pub(crate) message: String,
}

/// The fields of `report.json` this command judges and prints.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Report {
    pub(crate) mode: String,
    pub(crate) passed: bool,
    pub(crate) seeds: Vec<u64>,
    pub(crate) programs_requested: u64,
    pub(crate) programs_executed: u64,
    pub(crate) adapter: Option<String>,
    /// `(seed, programs, attempts, discarded)` per seed.
    pub(crate) per_seed: Vec<(u64, u64, u64, u64)>,
    pub(crate) discards_total: u64,
    pub(crate) discards_by_class: Vec<(String, u64)>,
    pub(crate) mismatch_count: u64,
    pub(crate) mismatches: Vec<MismatchRecord>,
    pub(crate) cargo_build_s: f64,
    pub(crate) cargo_run_s: f64,
}

fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, String> {
    json.get(key)
        .ok_or_else(|| format!("report.json has no `{key}`"))
}

fn uint(json: &Json, key: &str) -> Result<u64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a whole number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

fn float(json: &Json, key: &str) -> Result<f64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

fn text(json: &Json, key: &str) -> Result<String, String> {
    field(json, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("report.json `{key}` is not a string"))
}

fn array<'a>(json: &'a Json, key: &str) -> Result<&'a [Json], String> {
    field(json, key)?
        .as_array()
        .ok_or_else(|| format!("report.json `{key}` is not an array"))
}

/// Reads the fields of a `report.json`. A missing or mistyped field is an error: a report this function cannot read
/// proves nothing.
pub(crate) fn parse_report(source: &str) -> Result<Report, String> {
    let json = json::parse(source).map_err(|error| format!("report.json: {error}"))?;
    let tool = text(&json, "tool")?;
    if tool != "difftest-run" {
        return Err(format!("report.json is from `{tool}`, not `difftest-run`"));
    }
    let passed = match field(&json, "passed")? {
        Json::Bool(value) => *value,
        _ => return Err("report.json `passed` is not a boolean".into()),
    };
    let seeds = array(&json, "seeds")?
        .iter()
        .map(|seed| match seed {
            Json::Number(text) => text
                .parse()
                .map_err(|_| format!("report.json seed {text} is not a whole number")),
            _ => Err("report.json `seeds` holds a non-number".to_owned()),
        })
        .collect::<Result<Vec<u64>, _>>()?;
    let adapter = match field(&json, "adapter")? {
        Json::Null => None,
        Json::String(name) => Some(name.clone()),
        _ => return Err("report.json `adapter` is neither a string nor null".into()),
    };
    let per_seed = array(&json, "per_seed")?
        .iter()
        .map(|entry| {
            Ok((
                uint(entry, "seed")?,
                uint(entry, "programs")?,
                uint(entry, "attempts")?,
                uint(entry, "discarded")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let discards = field(&json, "discards")?;
    let by_class = match field(discards, "by_class")? {
        Json::Object(members) => members
            .iter()
            .map(|(name, value)| match value {
                Json::Number(count) => count
                    .parse()
                    .map(|count| (name.clone(), count))
                    .map_err(|_| format!("report.json discard class {name} is not a count")),
                _ => Err(format!("report.json discard class {name} is not a number")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("report.json `discards.by_class` is not an object".into()),
    };
    let mismatches = array(&json, "mismatches")?
        .iter()
        .map(|entry| {
            Ok(MismatchRecord {
                program: text(entry, "program")?,
                seed: uint(entry, "seed")?,
                index: uint(entry, "index")?,
                attempt: uint(entry, "attempt")?,
                kind: text(entry, "kind")?,
                elements_differing: uint(entry, "elements_differing")?,
                message: text(entry, "message")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let timing = field(&json, "timing")?;
    Ok(Report {
        mode: text(&json, "mode")?,
        passed,
        seeds,
        programs_requested: uint(&json, "programs_requested")?,
        programs_executed: uint(&json, "programs_executed")?,
        adapter,
        per_seed,
        discards_total: uint(discards, "total")?,
        discards_by_class: by_class,
        mismatch_count: uint(&json, "mismatch_count")?,
        mismatches,
        cargo_build_s: float(timing, "cargo_build_s")?,
        cargo_run_s: float(timing, "cargo_run_s")?,
    })
}

/// Everything wrong with `report` for a run that asked for `options`; empty when the report is a pass. The count rule
/// is the point: `programs_executed` must equal `16 x seeds`, so a harness that silently ran fewer programs fails here
/// whatever it says about `passed`.
pub(crate) fn report_problems(report: &Report, options: &Options) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = options.expected_programs();
    if report.programs_executed != expected {
        problems.push(format!(
            "programs_executed is {}, expected {PROGRAMS_PER_SEED} x {} seeds = {expected}",
            report.programs_executed,
            options.seeds.len()
        ));
    }
    if report.programs_requested != expected {
        problems.push(format!(
            "programs_requested is {}, expected {PROGRAMS_PER_SEED} x {} seeds = {expected}",
            report.programs_requested,
            options.seeds.len()
        ));
    }
    if report.mode != options.mode.report_name() {
        problems.push(format!(
            "the report is for mode `{}`, this run asked for `{}`",
            report.mode,
            options.mode.report_name()
        ));
    }
    if report.seeds != options.seeds {
        problems.push(format!(
            "the report ran seeds [{}], this run asked for [{}]",
            seed_list(&report.seeds),
            seed_list(&options.seeds)
        ));
    }
    // One entry per seed that ran, in that order: a table that names other seeds, or one twice, proves nothing about
    // the seeds that were asked for even when its counts add up.
    let table_seeds: Vec<u64> = report.per_seed.iter().map(|entry| entry.0).collect();
    if table_seeds != report.seeds {
        problems.push(format!(
            "per_seed names seeds [{}] but the report ran seeds [{}]",
            seed_list(&table_seeds),
            seed_list(&report.seeds)
        ));
    }
    for (seed, programs, _, _) in &report.per_seed {
        if *programs != PROGRAMS_PER_SEED {
            problems.push(format!(
                "seed {seed}: {programs} programs, expected {PROGRAMS_PER_SEED}"
            ));
        }
    }
    let planned: u64 = report.per_seed.iter().map(|entry| entry.1).sum();
    if planned != report.programs_executed {
        problems.push(format!(
            "per_seed holds {planned} programs but programs_executed is {}",
            report.programs_executed
        ));
    }
    if report.mismatch_count != report.mismatches.len() as u64 {
        problems.push(format!(
            "mismatch_count is {} but the report lists {} mismatches",
            report.mismatch_count,
            report.mismatches.len()
        ));
    }
    for mismatch in &report.mismatches {
        problems.push(format!(
            "seed {} program {} (index {}, attempt {}): {}: {} element(s) differ: {}",
            mismatch.seed,
            mismatch.program,
            mismatch.index,
            mismatch.attempt,
            mismatch.kind,
            mismatch.elements_differing,
            first_line(&mismatch.message)
        ));
    }
    match (options.mode, &report.adapter) {
        (Mode::Gpu, None) => problems.push("a gpu report names no adapter".into()),
        (Mode::Gpu, Some(name)) if name.trim().is_empty() => {
            problems.push("a gpu report names an empty adapter".into());
        }
        (Mode::CpuOnly, Some(name)) => {
            problems.push(format!("a cpu-only report names the adapter {name:?}"));
        }
        _ => {}
    }
    if !report.passed && problems.is_empty() {
        problems.push("the report says passed: false and names no cause".into());
    }
    problems
}

/// The first line of `message`, cut to 300 characters.
fn first_line(message: &str) -> String {
    let line = message.lines().next().unwrap_or_default();
    match line.char_indices().nth(300) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line.to_owned(),
    }
}

/// What the batch step says: the problems (an empty list is a pass) and the summary lines to print.
pub(crate) struct Judgement {
    pub(crate) problems: Vec<String>,
    pub(crate) summary: Vec<String>,
}

/// Judges the end of the batch from the exit status of `difftest-run` and the content of its report (or the reason it
/// could not be read). A run that exits non-zero fails whatever the report says; a report that is not a pass fails
/// whatever the exit status says.
pub(crate) fn judge_batch(
    options: &Options,
    exit_code: Option<i32>,
    report: Result<String, String>,
) -> Judgement {
    let mut problems = Vec::new();
    let mut summary = Vec::new();
    match report.and_then(|source| parse_report(&source)) {
        Ok(report) => {
            problems.extend(report_problems(&report, options));
            summary = report_summary(&report, options);
        }
        Err(error) => problems.push(format!("no usable report: {error}")),
    }
    match exit_code {
        Some(0) => {}
        // Status 1 is "a program disagreed"; the report must then say which.
        Some(1) if !problems.is_empty() => {}
        Some(code) => problems.push(format!(
            "difftest-run exited with status {code}{}",
            if code == 2 {
                " (the harness failed: build error, stale template lockfile, a result file missing)"
            } else {
                ""
            }
        )),
        None => problems.push("difftest-run was stopped by a signal".into()),
    }
    Judgement { problems, summary }
}

fn report_summary(report: &Report, options: &Options) -> Vec<String> {
    let adapter = report
        .adapter
        .as_ref()
        .map_or(String::new(), |adapter| format!(" on {adapter}"));
    let mut lines = vec![format!(
        "  mode {}{adapter}, seeds [{}]{}, programs_executed {} \
         (expected {PROGRAMS_PER_SEED} x {} = {}), mismatches {}",
        report.mode,
        seed_list(&report.seeds),
        if options.random {
            " (drawn at random)"
        } else {
            ""
        },
        report.programs_executed,
        options.seeds.len(),
        options.expected_programs(),
        report.mismatch_count
    )];
    for (seed, programs, attempts, discarded) in &report.per_seed {
        lines.push(format!(
            "  seed {seed}: {programs} programs from {attempts} attempts ({discarded} discarded)"
        ));
    }
    let classes = report
        .discards_by_class
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    lines.push(format!(
        "  discards: {} in all ({classes})",
        report.discards_total
    ));
    lines.push(format!(
        "  harness timing: cargo build {:.1} s, cargo run {:.1} s",
        report.cargo_build_s, report.cargo_run_s
    ));
    lines
}

/// What is wrong with the output of the short-circuit test run: it must have run a test, nothing may have failed, and
/// the named test must be reported `ok` (a test that was filtered out or ignored again does not count).
pub(crate) fn short_circuit_problems(args: &[String], stdout: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let totals = sum_test_results(stdout);
    if let Err(error) = require_tests_ran(args, &totals) {
        problems.push(error);
    }
    if !tests_not_ok(stdout, &[SHORT_CIRCUIT_GPU_TEST]).is_empty() {
        problems.push(format!(
            "short_circuit: `{SHORT_CIRCUIT_GPU_TEST}` was not reported ok"
        ));
    }
    problems
}

/// `cargo xtask check-difftest`.
pub(crate) fn check_difftest(args: &[String]) -> Result<(), String> {
    println!("{}", check_difftest_report(args)?);
    Ok(())
}

/// [`check_difftest`] returning the pass summary instead of printing it, so `verify --mode full` can record it.
pub(crate) fn check_difftest_report(args: &[String]) -> Result<String, String> {
    let options = Options::parse(args)?;
    let root = workspace_root()?;
    let target = std::path::absolute(root.join(cargo_target_dir(&root)))
        .map_err(|error| format!("check-difftest: the cargo target directory: {error}"))?;
    let report_path = match &options.report {
        Some(path) => std::path::absolute(path)
            .map_err(|error| format!("check-difftest: --report {}: {error}", path.display()))?,
        None => target.join("difftest-report.json"),
    };
    run_steps(&options, &root, &target, &report_path)
}

fn run_steps(
    options: &Options,
    root: &Path,
    target: &Path,
    report_path: &Path,
) -> Result<String, String> {
    let started = Instant::now();
    println!(
        "check-difftest: {} mode, {} seed(s) [{}]{}, {} programs expected",
        options.mode.report_name(),
        options.seeds.len(),
        seed_list(&options.seeds),
        if options.random {
            " drawn at random; rerun with --seeds <this list> to reproduce"
        } else {
            ""
        },
        options.expected_programs()
    );
    let mut problems = Vec::new();
    let mut summary = Vec::new();
    for step in steps(options.mode) {
        match step {
            DifftestStep::Batch => {
                let judgement = run_batch(options, root, target, report_path)?;
                problems.extend(judgement.problems);
                summary = judgement.summary;
            }
            DifftestStep::ShortCircuitGpu => {
                println!("-- short_circuit (GPU, --ignored)");
                problems.extend(run_short_circuit(root, target)?);
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    if !problems.is_empty() {
        let mut message = format!(
            "check-difftest failed after {seconds:.1} s with {} problem(s):",
            problems.len()
        );
        for problem in &problems {
            message.push_str("\ncheck-difftest: ");
            message.push_str(problem);
        }
        message.push_str(&format!("\nreport: {}", report_path.display()));
        return Err(message);
    }
    let mut passed = format!(
        "check-difftest passed in {seconds:.1} s{}:\n{}",
        if options.mode == Mode::Gpu {
            " (the batch and the short_circuit GPU test)"
        } else {
            " (cpu-only: interpreter against twin; no adapter, no GPU evidence)"
        },
        summary.join("\n")
    );
    passed.push_str(&format!("\n  report: {}", report_path.display()));
    Ok(passed)
}

fn run_batch(
    options: &Options,
    root: &Path,
    target: &Path,
    report_path: &Path,
) -> Result<Judgement, String> {
    // A report left by an earlier run must not be able to pass this one.
    match fs::remove_file(report_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "check-difftest: {}: {error}",
                report_path.display()
            ));
        }
    }
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("check-difftest: {}: {error}", parent.display()))?;
    }
    fs::create_dir_all(target)
        .map_err(|error| format!("check-difftest: {}: {error}", target.display()))?;
    let args = batch_args(options, target, report_path);
    println!("-- cargo {}", args.join(" "));
    let status = workspace_command("cargo")
        .args(&args)
        .current_dir(root)
        .env("CARGO_TARGET_DIR", target)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("check-difftest: could not launch cargo: {error}"))?;
    let report = fs::read_to_string(report_path)
        .map_err(|error| format!("{}: {error}", report_path.display()));
    Ok(judge_batch(options, status.code(), report))
}

fn run_short_circuit(root: &Path, target: &Path) -> Result<Vec<String>, String> {
    let args = short_circuit_args();
    println!("-- cargo {}", args.join(" "));
    // CARGO_TARGET_DIR is what the test reads to build its run crate beside the batch's (without it the test builds
    // the whole dependency tree again under its own `target/tmp`).
    let output = workspace_command("cargo")
        .args(&args)
        .current_dir(root)
        .env("CARGO_TARGET_DIR", target)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("check-difftest: could not launch cargo: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    print!("{stdout}");
    let mut problems = short_circuit_problems(&args, &stdout);
    if !output.status.success() {
        problems.push(format!(
            "cargo {} failed (exit {})",
            args.join(" "),
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok(problems)
}

#[cfg(test)]
#[path = "tests/difftest.rs"]
mod difftest_tests;
