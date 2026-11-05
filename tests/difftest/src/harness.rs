//! The batch runner (`difftest-run`): seeds -> run crate -> `cargo test --locked` -> result files -> classification ->
//! `report.json`.
//!
//! The run crate lives at `<target>/difftest/<name>/` and builds into the same target directory (`CARGO_TARGET_DIR`),
//! under `GUST_PROFILE=dev` with the committed template lockfile and `--locked`. The generated test writes one JSON
//! file per program into `DIFFTEST_OUT`; this module asserts that exactly one file per planned program exists, compares
//! the three outputs (`classify`), and writes the report. Nothing here is GPU evidence by itself: a report says what
//! ran (`mode`, `adapter`, `programs_executed`) and what disagreed.
//!
//! With `--shapes grid,pixel` every program also gets a pixel kernel (`k_px`). The macro's classifier decides which
//! programs may have one: a build that it refuses is followed by exactly one rebuild without the refused kernels, and
//! the report counts those programs as `grid`-only with the classifier's own reason (`shapes` in the report).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::ast::{Expr, Program, Stmt, operator_counts};
use crate::classify::{
    CLASSES, Mismatch, PixelRaw, Raw, classify, discard_class, float_comparison, read_results,
};
use crate::corpus::fnv1a64;
use crate::emit::{manifest, package_name};
use crate::generator::{DIFFTEST_FORMAT, PROGRAMS_PER_SEED, generate_batch};
use crate::interp::{Coverage, run_covered, run_full};
use crate::json::Json;
use crate::runsrc::{Planned, kernel_source_in, write_run_crate_in};
use crate::shapes::{Shape, ShapePlan, module_ranges, refusals, set_aside};

/// Parses `1-4`, `3`, `1,3,5-6` into the sorted, duplicate-free list of seeds.
pub fn parse_seeds(text: &str) -> Result<Vec<u64>, String> {
    let mut seeds = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        let number = |s: &str| {
            s.trim()
                .parse::<u64>()
                .map_err(|e| format!("seed {s:?}: {e}"))
        };
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last) = (number(first)?, number(last)?);
                if first > last {
                    return Err(format!("seed range {part:?} is empty"));
                }
                if last - first >= 4096 {
                    return Err(format!("seed range {part:?} is larger than 4096"));
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

/// The identity of a batch crate: the package name and the directory name under `<target>/difftest/`.
///
/// Crates that share a target directory must not share a package name (cargo gives them one fingerprint unit, so a
/// stale check can pass), so the name is derived from everything that makes the crate's content differ: the seeds
/// and the format. The same seeds give the same name on every run, which keeps the build incremental.
pub fn crate_identity(seeds: &[u64], format: u32) -> Result<(String, String), String> {
    crate_identity_in(seeds, format, &[Shape::Grid])
}

/// [`crate_identity`] for the shapes of a batch: a batch with `pixel` is another crate than the same seeds without it.
/// The key of a `grid`-only batch is the one it always had.
pub fn crate_identity_in(
    seeds: &[u64],
    format: u32,
    shapes: &[Shape],
) -> Result<(String, String), String> {
    let mut key = format!("format={format};seeds=");
    for seed in seeds {
        key.push_str(&format!("{seed},"));
    }
    if shapes.contains(&Shape::Pixel) {
        key.push_str(";shapes=grid,pixel");
    }
    let hash = fnv1a64(key.as_bytes());
    let suffix = format!("run-{hash:016x}");
    Ok((package_name(&suffix)?, suffix))
}

/// The `name = "..."` of the `[package]` table of a generated `Cargo.toml`.
fn manifest_package(text: &str) -> Option<String> {
    text.lines()
        .find_map(|line| line.strip_prefix("name = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .map(str::to_owned)
}

/// Checks that no two crate directories under `root` (each holding a `Cargo.toml`) share a package name, and returns
/// the names found. The harness runs this on `<target>/difftest/` before it builds.
pub fn unique_package_names(root: &Path) -> Result<Vec<String>, String> {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return Ok(Vec::new());
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("Cargo.toml").is_file())
        .collect();
    dirs.sort();
    for dir in dirs {
        let text = std::fs::read_to_string(dir.join("Cargo.toml")).map_err(|e| e.to_string())?;
        let name = manifest_package(&text)
            .ok_or_else(|| format!("{}: no package name", dir.join("Cargo.toml").display()))?;
        let here = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(other) = seen.insert(name.clone(), here.clone()) {
            return Err(format!(
                "package name {name:?} is used by both {other} and {here}; crates that share a target directory need distinct names"
            ));
        }
    }
    Ok(seen.into_keys().collect())
}

/// What `difftest-run` was asked to do.
#[derive(Clone, Debug)]
pub struct Options {
    pub seeds: Vec<u64>,
    pub cpu_only: bool,
    /// The shapes to run (`--shapes`): `[Grid]`, or `[Grid, Pixel]` to run every program's `k_px` too.
    pub shapes: Vec<Shape>,
    /// The cargo target directory, the one the batch crate also builds into.
    pub target_dir: PathBuf,
    /// The repository root (the generated manifest points at its `crates/`).
    pub repo_root: PathBuf,
}

/// The planned programs of one seed, with the discard counts and the time the planning took.
pub struct SeedPlan {
    pub seed: u64,
    pub attempts: u64,
    pub discarded: BTreeMap<&'static str, u32>,
    pub planned: Vec<Planned>,
    pub generate_ms: f64,
    pub interpret_ms: f64,
}

pub fn plan_seed_timed(seed: u64) -> Result<SeedPlan, String> {
    let start = Instant::now();
    let batch = generate_batch(seed)?;
    let generate_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut interpret_ms = 0.0;
    let mut planned = Vec::new();
    for (index, program) in batch.programs.into_iter().enumerate() {
        let start = Instant::now();
        let run = run_full(&program).map_err(|f| {
            format!("seed {seed} program {index}: {f:?} after the batch accepted it")
        })?;
        interpret_ms += start.elapsed().as_secs_f64() * 1e3;
        planned.push(Planned {
            program,
            index,
            expected: run.values,
            intervals: run.intervals,
        });
    }
    Ok(SeedPlan {
        seed,
        attempts: batch.attempts,
        discarded: batch.discarded,
        planned,
        generate_ms,
        interpret_ms,
    })
}

fn constant_ifs(stmts: &[Stmt]) -> u64 {
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::If { cond, then, els } => {
                u64::from(matches!(cond, Expr::Lit(..))) + constant_ifs(then) + constant_ifs(els)
            }
            Stmt::For { body, .. } => constant_ifs(body),
            _ => 0,
        })
        .sum()
}

/// Operator coverage over a set of programs: for each operator the number of programs that contain it (syntactic,
/// dead code included) and that execute it, and its total evaluations; the `if` statements by the sides taken; the
/// `&&`/`||` evaluations by whether the right operand ran. The programs are the in-domain ones, so this is coverage
/// under the domain filter.
pub fn coverage_json(programs: &[&Program]) -> Result<Json, String> {
    let mut syntactic: BTreeMap<String, u64> = BTreeMap::new();
    let mut executed_programs: BTreeMap<String, u64> = BTreeMap::new();
    let mut evaluations: BTreeMap<String, u64> = BTreeMap::new();
    let mut total = Coverage::default();
    let mut constant_conditions = 0;
    for program in programs {
        for name in operator_counts(program).keys() {
            *syntactic.entry(name.clone()).or_default() += 1;
        }
        let (_, coverage) = run_covered(program)
            .map_err(|f| format!("seed {} attempt {}: {f:?}", program.seed, program.attempt))?;
        for (name, n) in &coverage.operators {
            *executed_programs.entry(name.clone()).or_default() += 1;
            *evaluations.entry(name.clone()).or_default() += n;
        }
        total.if_statements += coverage.if_statements;
        total.if_evaluated += coverage.if_evaluated;
        total.if_then_taken += coverage.if_then_taken;
        total.if_else_taken += coverage.if_else_taken;
        total.if_both_sides += coverage.if_both_sides;
        total.rhs_evaluated += coverage.rhs_evaluated;
        total.rhs_skipped += coverage.rhs_skipped;
        constant_conditions += constant_ifs(&program.body);
    }
    let mut names: Vec<&String> = syntactic.keys().chain(executed_programs.keys()).collect();
    names.sort();
    names.dedup();
    let mut operators = Vec::new();
    let mut never = Vec::new();
    for name in names {
        let in_text = syntactic.get(name).copied().unwrap_or(0);
        let executed = executed_programs.get(name).copied().unwrap_or(0);
        if executed == 0 {
            never.push(Json::str(name));
        }
        operators.push(Json::obj(vec![
            ("operator", Json::str(name)),
            ("programs_in_text", Json::uint(in_text)),
            ("programs_executing", Json::uint(executed)),
            (
                "evaluations",
                Json::uint(evaluations.get(name).copied().unwrap_or(0)),
            ),
        ]));
    }
    Ok(Json::obj(vec![
        ("programs", Json::uint(programs.len() as u64)),
        ("operators", Json::Arr(operators)),
        ("never_executed", Json::Arr(never)),
        (
            "if_statements",
            Json::obj(vec![
                ("in_text", Json::uint(total.if_statements)),
                ("constant_condition", Json::uint(constant_conditions)),
                ("evaluated", Json::uint(total.if_evaluated)),
                ("then_taken", Json::uint(total.if_then_taken)),
                ("else_taken", Json::uint(total.if_else_taken)),
                ("both_sides", Json::uint(total.if_both_sides)),
            ]),
        ),
        (
            "logical_operators",
            Json::obj(vec![
                ("right_operand_evaluated", Json::uint(total.rhs_evaluated)),
                ("right_operand_skipped", Json::uint(total.rhs_skipped)),
            ]),
        ),
    ]))
}

/// What the cargo steps took.
#[derive(Clone, Debug, Default)]
pub struct CargoRun {
    pub build_s: f64,
    pub run_s: f64,
}

/// A cargo child. `CARGO_TERM_COLOR=never` because the harness parses its output (`shapes::refusals`) and writes it to
/// logs: CI sets `CARGO_TERM_COLOR=always`, which puts escape sequences inside the diagnostics.
fn cargo_command() -> Command {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command.env("CARGO_TERM_COLOR", "never");
    command
}

fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

/// Whether cargo's output says that `--locked` rejected the lockfile.
pub fn lock_was_rejected(output: &str) -> bool {
    output.contains("needs to be updated but --locked was passed")
        || output.contains("cannot update the lock file")
}

/// The message that replaces cargo's when the template lockfile no longer fits the manifest.
pub const STALE_LOCK_MESSAGE: &str = "template lockfile stale; run `difftest-run --refresh-lock` and commit tests/difftest/template/Cargo.lock";

/// The start of the error of a step that ran past its deadline; the reducer counts such a candidate as not reproducing.
pub const TIMED_OUT: &str = "timed out";

/// Kills `child` and everything it started (cargo's rustc and the test binary): `taskkill /T` on Windows, the child's
/// process group elsewhere (`run_with_deadline` makes the child a group leader).
fn kill_tree(child: &mut Child) {
    let _ = if cfg!(windows) {
        Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .output()
    } else {
        Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .output()
    };
    let _ = child.kill();
}

/// Copies a pipe into `buffer` as it arrives, so that the output so far survives a thread that is never joined.
fn drain<R: Read + Send + 'static>(
    pipe: Option<R>,
    buffer: &Arc<Mutex<Vec<u8>>>,
) -> std::thread::JoinHandle<()> {
    let buffer = Arc::clone(buffer);
    std::thread::spawn(move || {
        let (Some(mut pipe), mut chunk) = (pipe, [0u8; 8192]) else {
            return;
        };
        while let Ok(n @ 1..) = pipe.read(&mut chunk) {
            buffer.lock().unwrap().extend_from_slice(&chunk[..n]);
        }
    })
}

/// Runs `command` with its stdout and stderr captured; kills it and its children after `deadline`. The status is
/// `None` on a timeout, and then the text is what the command had written (the drains are not waited for, because a
/// child that survived the kill would hold the pipes open).
fn run_with_deadline(
    mut command: Command,
    deadline: Option<Duration>,
) -> Result<(Option<ExitStatus>, String), String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    if deadline.is_some() {
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot start cargo: {e}"))?;
    let buffers = [Arc::default(), Arc::default()];
    let drains = [
        drain(child.stdout.take(), &buffers[0]),
        drain(child.stderr.take(), &buffers[1]),
    ];
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            for drain in drains {
                let _ = drain.join();
            }
            break Some(status);
        }
        if deadline.is_some_and(|d| start.elapsed() >= d) {
            kill_tree(&mut child);
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let text = |b: &Arc<Mutex<Vec<u8>>>| String::from_utf8_lossy(&b.lock().unwrap()).into_owned();
    Ok((status, text(&buffers[0]) + &text(&buffers[1])))
}

/// Runs one cargo step of the run crate, with its output in `log`; returns the seconds it took. `deadline` bounds the
/// step (the reducer's 120 s); the batch runner passes `None`.
#[allow(clippy::too_many_arguments)]
fn cargo_step(
    crate_dir: &Path,
    target_dir: &Path,
    args: &[&str],
    out_dir: Option<&Path>,
    cpu_only: bool,
    log: &Path,
    env: &[(&str, &Path)],
    deadline: Option<Duration>,
) -> Result<f64, String> {
    let mut command = cargo_command();
    command
        .args(args)
        .arg("--locked")
        .arg("--manifest-path")
        .arg(crate_dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target_dir)
        .env("GUST_PROFILE", "dev")
        .env("DIFFTEST_CPU_ONLY", if cpu_only { "1" } else { "0" })
        .stdin(Stdio::null());
    if let Some(out_dir) = out_dir {
        command.env("DIFFTEST_OUT", out_dir);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    let start = Instant::now();
    let (finished, text) = run_with_deadline(command, deadline)?;
    let seconds = start.elapsed().as_secs_f64();
    std::fs::write(log, &text).map_err(|e| format!("{}: {e}", log.display()))?;
    let Some(status) = finished else {
        return Err(format!(
            "{TIMED_OUT}: cargo {} ran past {} s; the output so far is in {}",
            args.join(" "),
            deadline.map_or(0, |d| d.as_secs()),
            log.display()
        ));
    };
    if status.success() {
        return Ok(seconds);
    }
    if lock_was_rejected(&text) {
        return Err(STALE_LOCK_MESSAGE.into());
    }
    Err(format!(
        "cargo {} failed ({}); log {}:\n{}",
        args.join(" "),
        status,
        log.display(),
        tail(&text, 40)
    ))
}

/// Everything a finished run produced.
pub struct Outcome {
    pub report: Json,
    pub passed: bool,
}

fn milliseconds(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// The `float_pixel_vs_grid` observation of the `shapes` key (D128): over the `f32` programs that ran in both shapes on
/// the GPU, how many differ in a bit between the two shapes, in how many elements, by at most how many ulps, and the
/// first such element. Never a mismatch: each shape is judged against the interpreter's interval.
pub fn float_observation_json(raws: &[Raw]) -> Json {
    let mut compared = 0u64;
    let mut differing_programs = 0u64;
    let mut elements = 0u64;
    let mut max_ulp = 0u32;
    let mut first: Option<Json> = None;
    for raw in raws {
        let Some(found) = float_comparison(raw) else {
            continue;
        };
        compared += 1;
        if found.elements == 0 {
            continue;
        }
        differing_programs += 1;
        elements += found.elements as u64;
        max_ulp = max_ulp.max(found.max_ulp);
        if first.is_none()
            && let Some((element, grid, pixel)) = found.first
        {
            first = Some(Json::obj(vec![
                ("program", Json::str(&raw.program)),
                ("element", Json::uint(element as u64)),
                ("grid_bits", Json::uint(u64::from(grid))),
                ("pixel_bits", Json::uint(u64::from(pixel))),
            ]));
        }
    }
    Json::obj(vec![
        ("programs_compared", Json::uint(compared)),
        ("programs_with_difference", Json::uint(differing_programs)),
        ("elements_differing", Json::uint(elements)),
        ("max_ulp", Json::uint(u64::from(max_ulp))),
        ("first", first.unwrap_or(Json::Null)),
    ])
}

/// The `shapes` key of the report: what was requested, how many programs ran as `pixel`, the programs that stayed
/// `grid`-only with the macro classifier's reason (counted by reason, and listed), and the `f32` observation.
pub fn shapes_json(shapes: &ShapePlan, programs: usize, raws: &[Raw]) -> Json {
    let mut by_reason: BTreeMap<&str, u64> = BTreeMap::new();
    for only in shapes.grid_only.values() {
        *by_reason.entry(only.reason.as_str()).or_default() += 1;
    }
    let grid_only = shapes.grid_only.len();
    let pixel = if shapes.pixel {
        programs.saturating_sub(grid_only)
    } else {
        0
    };
    Json::obj(vec![
        (
            "requested",
            Json::Arr(shapes.requested().into_iter().map(Json::str).collect()),
        ),
        ("float_pixel_vs_grid", float_observation_json(raws)),
        (
            "programs_by_shape",
            Json::obj(vec![
                ("grid", Json::uint(programs as u64)),
                ("pixel", Json::uint(pixel as u64)),
            ]),
        ),
        (
            "grid_only",
            Json::obj(vec![
                ("total", Json::uint(grid_only as u64)),
                (
                    "by_reason",
                    Json::Obj(
                        by_reason
                            .iter()
                            .map(|(k, v)| ((*k).to_owned(), Json::uint(*v)))
                            .collect(),
                    ),
                ),
                (
                    "programs",
                    Json::Arr(
                        shapes
                            .grid_only
                            .iter()
                            .map(|(program, only)| {
                                Json::obj(vec![
                                    ("program", Json::str(program)),
                                    ("reason", Json::str(&only.reason)),
                                    ("message", Json::str(&only.message)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
    ])
}

/// Builds the report from the pieces of a run. Pure: the tests build reports from synthetic results.
pub fn build_report(
    options: &Options,
    seed_plans: &[SeedPlan],
    raws: &[Raw],
    mismatches: &[Mismatch],
    coverage: Json,
    cargo: &CargoRun,
    shapes: &ShapePlan,
) -> Outcome {
    let requested = options.seeds.len() * PROGRAMS_PER_SEED;
    let executed = raws.len();
    let mut by_reason: BTreeMap<&str, u32> = BTreeMap::new();
    let mut by_class: BTreeMap<&str, u32> = CLASSES.iter().map(|c| (*c, 0)).collect();
    for plan in seed_plans {
        for (reason, count) in &plan.discarded {
            *by_reason.entry(reason).or_default() += count;
            *by_class.entry(discard_class(reason)).or_default() += count;
        }
    }
    let discarded_total: u32 = by_reason.values().sum();
    let class_json = Json::Obj(
        CLASSES
            .iter()
            .map(|c| ((*c).to_owned(), Json::uint(u64::from(by_class[c]))))
            .collect(),
    );
    let reason_json = Json::Obj(
        by_reason
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Json::uint(u64::from(*v))))
            .collect(),
    );
    let per_seed = seed_plans
        .iter()
        .map(|plan| {
            let discarded: u32 = plan.discarded.values().sum();
            Json::obj(vec![
                ("seed", Json::uint(plan.seed)),
                ("programs", Json::uint(plan.planned.len() as u64)),
                ("attempts", Json::uint(plan.attempts)),
                ("discarded", Json::uint(u64::from(discarded))),
            ])
        })
        .collect();
    let adapter = raws.iter().find_map(|r| r.adapter.clone());
    let sum = |f: fn(&Raw) -> u64| raws.iter().map(f).sum::<u64>();
    let max = |f: fn(&Raw) -> u64| raws.iter().map(f).max().unwrap_or(0);
    let mean = |total: u64| {
        if raws.is_empty() {
            0.0
        } else {
            milliseconds(total) / raws.len() as f64
        }
    };
    let pixel_raws: Vec<&PixelRaw> = raws.iter().filter_map(|r| r.pixel.as_ref()).collect();
    let pixel_ms = |f: fn(&PixelRaw) -> u64| milliseconds(pixel_raws.iter().map(|p| f(p)).sum());
    let mut timing = vec![
        (
            "generation_ms",
            Json::Float(seed_plans.iter().map(|p| p.generate_ms).sum()),
        ),
        (
            "interpreter_ms",
            Json::Float(seed_plans.iter().map(|p| p.interpret_ms).sum()),
        ),
        ("cargo_build_s", Json::Float(cargo.build_s)),
        ("cargo_run_s", Json::Float(cargo.run_s)),
        (
            "twin_ms_total",
            Json::Float(milliseconds(sum(|r| r.twin_ns))),
        ),
        ("twin_ms_max", Json::Float(milliseconds(max(|r| r.twin_ns)))),
        (
            "gpu_first_dispatch_ms_total",
            Json::Float(milliseconds(sum(|r| r.gpu_first_ns))),
        ),
        (
            "gpu_first_dispatch_ms_mean",
            Json::Float(mean(sum(|r| r.gpu_first_ns))),
        ),
        (
            "gpu_first_dispatch_ms_max",
            Json::Float(milliseconds(max(|r| r.gpu_first_ns))),
        ),
        (
            "gpu_warm_dispatch_ms_total",
            Json::Float(milliseconds(sum(|r| r.gpu_warm_ns))),
        ),
        (
            "gpu_warm_dispatch_ms_max",
            Json::Float(milliseconds(max(|r| r.gpu_warm_ns))),
        ),
    ];
    if shapes.pixel {
        timing.push(("pixel_twin_ms_total", Json::Float(pixel_ms(|p| p.twin_ns))));
        timing.push((
            "pixel_gpu_first_dispatch_ms_total",
            Json::Float(pixel_ms(|p| p.gpu_first_ns)),
        ));
        timing.push((
            "pixel_gpu_warm_dispatch_ms_total",
            Json::Float(pixel_ms(|p| p.gpu_warm_ns)),
        ));
    }
    let timing = Json::obj(timing);
    let passed = mismatches.is_empty() && executed == requested;
    let report = Json::obj(vec![
        ("tool", Json::str("difftest-run")),
        ("format", Json::uint(u64::from(DIFFTEST_FORMAT))),
        (
            "mode",
            Json::str(if options.cpu_only { "cpu-only" } else { "gpu" }),
        ),
        ("passed", Json::Bool(passed)),
        (
            "seeds",
            Json::Arr(options.seeds.iter().map(|s| Json::uint(*s)).collect()),
        ),
        ("programs_requested", Json::uint(requested as u64)),
        ("programs_executed", Json::uint(executed as u64)),
        ("adapter", adapter.map_or(Json::Null, Json::Str)),
        ("per_seed", Json::Arr(per_seed)),
        (
            "discards",
            Json::obj(vec![
                ("total", Json::uint(u64::from(discarded_total))),
                ("by_class", class_json),
                ("by_reason", reason_json),
            ]),
        ),
        ("shapes", shapes_json(shapes, requested, raws)),
        ("mismatch_count", Json::uint(mismatches.len() as u64)),
        (
            "mismatches",
            Json::Arr(mismatches.iter().map(Mismatch::to_json).collect()),
        ),
        ("coverage", coverage),
        ("timing", timing),
    ]);
    Outcome { report, passed }
}

/// Writes the run crate `<target>/difftest/<name>/` for `plan`, builds and runs it, and returns the parsed result files
/// (exactly one per planned program, or an error) with the time the cargo steps took. `env` is added to the environment
/// of the test run (for example `DIFFTEST_WGSL_OUT`).
pub fn execute_plan(
    options: &Options,
    package: &str,
    name: &str,
    plan: &[Planned],
    env: &[(&str, &Path)],
) -> Result<(Vec<Raw>, CargoRun), String> {
    execute_plan_within(options, package, name, plan, env, None)
}

/// [`execute_plan`] with a deadline for each of the two cargo steps (the build and the test run).
pub fn execute_plan_within(
    options: &Options,
    package: &str,
    name: &str,
    plan: &[Planned],
    env: &[(&str, &Path)],
    deadline: Option<Duration>,
) -> Result<(Vec<Raw>, CargoRun), String> {
    let (raws, cargo, _) = execute_shapes(options, package, name, plan, env, deadline)?;
    Ok((raws, cargo))
}

/// [`execute_plan_within`] that also returns the [`ShapePlan`] the run ended with: which programs ran a `k_px` and which
/// stayed `grid`-only. When `options.shapes` holds `pixel` and the first build is refused by the macro's classifier
/// (`GUST-E072x`), the refused programs are set aside and the crate is built once more; any other build failure is an
/// error.
pub fn execute_shapes(
    options: &Options,
    package: &str,
    name: &str,
    plan: &[Planned],
    env: &[(&str, &Path)],
    deadline: Option<Duration>,
) -> Result<(Vec<Raw>, CargoRun, ShapePlan), String> {
    let batches = options.target_dir.join("difftest");
    let dir = batches.join(name);
    let mut shapes = ShapePlan::new(&options.shapes);
    write_run_crate_in(&dir, package, plan, &options.repo_root, &shapes)
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    unique_package_names(&batches)?;

    let out = dir.join("out");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let build = || {
        cargo_step(
            &dir,
            &options.target_dir,
            &["test", "--no-run"],
            None,
            options.cpu_only,
            &dir.join("build.log"),
            &[],
            deadline,
        )
    };
    let first = Instant::now();
    let build_s = match build() {
        Ok(seconds) => seconds,
        Err(error) if shapes.pixel && !error.starts_with(TIMED_OUT) => {
            let log = std::fs::read_to_string(dir.join("build.log")).unwrap_or_default();
            let found = refusals(&log);
            let ranges = module_ranges(&kernel_source_in(plan, &shapes));
            let Ok(refused) = set_aside(&found, &ranges) else {
                // Not (only) the classifier: a failure of another kind, or a refusal that belongs to no module.
                return Err(error);
            };
            shapes.grid_only = refused;
            write_run_crate_in(&dir, package, plan, &options.repo_root, &shapes)
                .map_err(|e| format!("{}: {e}", dir.display()))?;
            let seconds = first.elapsed().as_secs_f64();
            seconds
                + build().map_err(|e| {
                    format!(
                        "{e}\n(the rebuild after setting aside {} grid-only programs)",
                        shapes.grid_only.len()
                    )
                })?
        }
        Err(error) => return Err(error),
    };
    let run_s = cargo_step(
        &dir,
        &options.target_dir,
        &["test"],
        Some(&out),
        options.cpu_only,
        &dir.join("run.log"),
        env,
        deadline,
    )?;
    let raws = read_results(&out, plan)?;
    check_pixel_presence(plan, &raws, &shapes)?;
    Ok((raws, CargoRun { build_s, run_s }, shapes))
}

/// Every program that was built with a `k_px` has pixel results in its file, and no other program has: a run that
/// silently dropped the pixel half would otherwise pass as `grid`-only.
pub fn check_pixel_presence(
    plan: &[Planned],
    raws: &[Raw],
    shapes: &ShapePlan,
) -> Result<(), String> {
    for (planned, raw) in plan.iter().zip(raws) {
        let module = crate::emit::module_name(&planned.program);
        let expected = shapes.runs_pixel(&module);
        if expected != raw.pixel.is_some() {
            return Err(format!(
                "{}: the result file {} pixel outputs, and the plan {} a `k_px` for it",
                raw.program,
                if raw.pixel.is_some() {
                    "holds"
                } else {
                    "lacks"
                },
                if expected { "built" } else { "did not build" },
            ));
        }
    }
    Ok(())
}

/// Runs a batch end to end. `Err` is a harness failure (build, test run, missing files); a program that disagrees is
/// an `Ok` outcome with `passed == false`.
pub fn run(options: &Options) -> Result<Outcome, String> {
    let mut seed_plans = Vec::new();
    for seed in &options.seeds {
        seed_plans.push(plan_seed_timed(*seed)?);
    }
    let plan: Vec<Planned> = seed_plans
        .iter()
        .flat_map(|p| p.planned.iter().cloned())
        .collect();
    let requested = options.seeds.len() * PROGRAMS_PER_SEED;
    if plan.len() != requested {
        return Err(format!(
            "planned {} programs for {requested} requested",
            plan.len()
        ));
    }

    let (package, name) = crate_identity_in(&options.seeds, DIFFTEST_FORMAT, &options.shapes)?;
    let (raws, cargo, shapes) = execute_shapes(options, &package, &name, &plan, &[], None)?;
    let mut mismatches = Vec::new();
    for (planned, raw) in plan.iter().zip(&raws) {
        mismatches.extend(classify(planned, raw, options.cpu_only));
    }
    let programs: Vec<&Program> = plan.iter().map(|p| &p.program).collect();
    let coverage = coverage_json(&programs)?;
    Ok(build_report(
        options,
        &seed_plans,
        &raws,
        &mismatches,
        coverage,
        &cargo,
        &shapes,
    ))
}

/// Regenerates `tests/difftest/template/Cargo.lock` from the root `Cargo.lock`: the root lock is copied in as the
/// starting point, so every registry package keeps the version and checksum the workspace uses, and cargo then
/// resolves the template manifest without `--locked`, which drops what the template does not need and adds its own
/// root package. A test requires the result to pin the same registry packages as the root lock.
pub fn refresh_lock(repo_root: &Path) -> Result<String, String> {
    let template = repo_root.join("tests").join("difftest").join("template");
    let lock = template.join("Cargo.lock");
    let root_lock = std::fs::read(repo_root.join("Cargo.lock"))
        .map_err(|e| format!("the root Cargo.lock: {e}"))?;
    let old = std::fs::read(&lock).ok();
    std::fs::write(&lock, root_lock).map_err(|e| format!("{}: {e}", lock.display()))?;
    let output = cargo_command()
        .args([
            "metadata",
            "--format-version",
            "1",
            "--offline",
            "--manifest-path",
        ])
        .arg(template.join("Cargo.toml"))
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("cannot start cargo: {e}"))?;
    if !output.status.success() {
        if let Some(old) = old {
            let _ = std::fs::write(&lock, old);
        }
        return Err(format!(
            "cargo metadata failed (the old lockfile is restored):\n{}",
            tail(&String::from_utf8_lossy(&output.stderr), 20)
        ));
    }
    let new = std::fs::read(&lock).map_err(|e| e.to_string())?;
    let same = old.as_deref().is_some_and(|old| {
        String::from_utf8_lossy(old).replace("\r\n", "\n")
            == String::from_utf8_lossy(&new).replace("\r\n", "\n")
    });
    Ok(if same {
        "template lockfile unchanged".to_owned()
    } else {
        "template lockfile rewritten; review the diff and commit it".to_owned()
    })
}

/// The manifest of the template, for a test that compares it with the committed one.
pub fn template_manifest() -> String {
    manifest("../../..", crate::emit::CRATE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runsrc::{plan_seed, write_run_crate};

    /// Not a test: the process that `a_hung_step_...` starts (this test binary again, with the variables set). It
    /// writes its pid, says it is ready and sleeps; the "outer" role also starts an "inner" copy of itself.
    #[test]
    fn sleeper_helper() {
        use std::io::Write;
        let (Some(dir), Ok(role)) = (
            std::env::var_os("DIFFTEST_SLEEPER_DIR"),
            std::env::var("DIFFTEST_SLEEPER_ROLE"),
        ) else {
            return;
        };
        std::fs::write(Path::new(&dir).join(&role), std::process::id().to_string()).unwrap();
        let mut stdout = std::io::stdout();
        stdout.write_all(b"sleeper-ready\n").unwrap();
        stdout.flush().unwrap();
        if role == "outer" {
            let mut inner = Command::new(std::env::current_exe().unwrap());
            inner.args(["--exact", "harness::tests::sleeper_helper"]);
            let _ = inner.env("DIFFTEST_SLEEPER_ROLE", "inner").status();
        } else {
            std::thread::sleep(Duration::from_secs(120));
        }
    }

    fn alive(pid: &str) -> bool {
        if cfg!(windows) {
            let filter = format!("PID eq {pid}");
            let listing = Command::new("tasklist")
                .args(["/FI", &filter, "/FO", "CSV", "/NH"])
                .output()
                .unwrap();
            String::from_utf8_lossy(&listing.stdout).contains(&format!("\"{pid}\""))
        } else {
            Command::new("kill")
                .args(["-0", pid])
                .output()
                .unwrap()
                .status
                .success()
        }
    }

    #[test]
    fn a_hung_step_is_killed_with_its_children_and_returns_at_the_deadline() {
        let dir = std::env::temp_dir().join(format!("difftest-sleeper-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "harness::tests::sleeper_helper"]);
        command
            .env("DIFFTEST_SLEEPER_DIR", &dir)
            .env("DIFFTEST_SLEEPER_ROLE", "outer");
        let start = Instant::now();
        let (status, text) = run_with_deadline(command, Some(Duration::from_secs(6))).unwrap();
        let waited = start.elapsed();
        assert!(status.is_none(), "the sleeper finished: {text}");
        assert!(
            waited < Duration::from_secs(30),
            "returned after {waited:?}"
        );
        assert!(
            text.contains("sleeper-ready"),
            "the output so far is kept: {text:?}"
        );
        // Both the process and the copy of itself that it started are gone (they would sleep for 120 s).
        let pids: Vec<String> = ["outer", "inner"]
            .iter()
            .map(|role| {
                std::fs::read_to_string(dir.join(role)).unwrap_or_else(|e| panic!("{role}: {e}"))
            })
            .collect();
        let gone = |pids: &[String]| pids.iter().all(|pid| !alive(pid));
        for _ in 0..100 {
            if gone(&pids) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let survivors: Vec<&String> = pids.iter().filter(|pid| alive(pid)).collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(survivors.is_empty(), "still running: {survivors:?}");
    }

    #[test]
    fn a_command_that_finishes_returns_its_status_and_all_its_output() {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "harness::tests::sleeper_helper"]);
        let (status, text) = run_with_deadline(command, Some(Duration::from_secs(60))).unwrap();
        assert!(status.unwrap().success());
        assert!(text.contains("test result: ok"), "{text}");
    }

    #[test]
    fn seeds_parse_as_ranges_and_lists() {
        assert_eq!(parse_seeds("1-4").unwrap(), [1, 2, 3, 4]);
        assert_eq!(parse_seeds("3").unwrap(), [3]);
        assert_eq!(parse_seeds("5-6, 1,3 ,3").unwrap(), [1, 3, 5, 6]);
        for bad in ["", "x", "4-1", "1-", "-1", "1,,2", "1-99999"] {
            assert!(parse_seeds(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn every_distinct_batch_gets_its_own_package_name() {
        let mut seen = BTreeMap::new();
        let sets: Vec<Vec<u64>> = vec![
            vec![1],
            vec![2],
            vec![1, 2],
            vec![1, 3],
            vec![1, 2, 3],
            (1..=8).collect(),
            vec![11],
            vec![1, 11],
        ];
        for format in [2, 3] {
            for set in &sets {
                let (package, dir) = crate_identity(set, format).unwrap();
                assert!(package.starts_with("difftest-case-run-"), "{package}");
                assert_eq!(package, format!("difftest-case-{dir}"));
                let key = (set.clone(), format);
                assert_eq!(seen.insert(package.clone(), key), None, "{package}");
                // Deterministic: the same batch has the same name, which keeps the build incremental.
                assert_eq!(crate_identity(set, format).unwrap().0, package);
            }
        }
        // The template package is not a batch name either.
        assert!(!seen.contains_key(crate::emit::CRATE_NAME));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("difftest-harness-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn two_crate_dirs_with_one_package_name_are_refused() {
        let root = scratch("unique");
        let plan = plan_seed(1).unwrap();
        for seeds in [vec![1u64], vec![1, 2]] {
            let (package, dir) = crate_identity(&seeds, DIFFTEST_FORMAT).unwrap();
            write_run_crate(&root.join(dir), &package, &plan[..2], Path::new("/r")).unwrap();
        }
        let names = unique_package_names(&root).unwrap();
        assert_eq!(names.len(), 2);
        // Give the second directory the first one's name: the check must fail, naming both.
        let (first_package, first_dir) = crate_identity(&[1], DIFFTEST_FORMAT).unwrap();
        let (_, second_dir) = crate_identity(&[1, 2], DIFFTEST_FORMAT).unwrap();
        write_run_crate(
            &root.join(&second_dir),
            &first_package,
            &plan[..2],
            Path::new("/r"),
        )
        .unwrap();
        let error = unique_package_names(&root).unwrap_err();
        assert!(error.contains(&first_package), "{error}");
        assert!(
            error.contains(&first_dir) && error.contains(&second_dir),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_rejected_lockfile_is_reported_as_a_stale_template() {
        let cargo_text = "error: the lock file /x/Cargo.lock needs to be updated but --locked was passed to prevent this\n";
        assert!(lock_was_rejected(cargo_text));
        assert!(!lock_was_rejected("error: could not compile `x`"));
        assert!(STALE_LOCK_MESSAGE.starts_with("template lockfile stale; run"));
        assert!(STALE_LOCK_MESSAGE.contains("--refresh-lock"));
    }

    #[test]
    fn the_template_manifest_is_the_emitted_one() {
        let committed =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/template/Cargo.toml"))
                .unwrap()
                .replace("\r\n", "\n");
        assert_eq!(committed, template_manifest());
    }

    #[test]
    fn the_report_counts_discards_by_reason_and_by_class() {
        let plans = vec![plan_seed_timed(1).unwrap(), plan_seed_timed(2).unwrap()];
        let options = Options {
            seeds: vec![1, 2],
            cpu_only: true,
            shapes: vec![Shape::Grid],
            target_dir: PathBuf::new(),
            repo_root: PathBuf::new(),
        };
        let outcome = build_report(
            &options,
            &plans,
            &[],
            &[],
            Json::Null,
            &CargoRun::default(),
            &ShapePlan::new(&options.shapes),
        );
        // No results were read, so a report cannot pass.
        assert!(!outcome.passed);
        let report = &outcome.report;
        assert_eq!(report.get("programs_requested").unwrap().as_u64(), Some(32));
        assert_eq!(report.get("programs_executed").unwrap().as_u64(), Some(0));
        let discards = report.get("discards").unwrap();
        let by_reason = discards.get("by_reason").unwrap();
        let by_class = discards.get("by_class").unwrap();
        let sum = |json: &Json| match json {
            Json::Obj(fields) => fields.iter().map(|(_, v)| v.as_u64().unwrap()).sum::<u64>(),
            _ => panic!("not an object"),
        };
        let total = sum(by_reason);
        assert_eq!(discards.get("total").unwrap().as_u64(), Some(total));
        assert_eq!(sum(by_class), total);
        let Json::Obj(classes) = by_class else {
            panic!()
        };
        assert_eq!(
            classes.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            CLASSES
        );
        // The domain class is the sum of its reasons.
        let reason = |name: &str| by_reason.get(name).and_then(Json::as_u64).unwrap_or(0);
        let domain: u64 = [
            "overflow",
            "division-by-zero",
            "min-over-minus-one",
            "shift-amount",
            "index",
            "float-domain",
            "static",
        ]
        .iter()
        .map(|r| reason(r))
        .sum();
        assert_eq!(by_class.get("domain").unwrap().as_u64(), Some(domain));
        assert!(total > 0);
    }

    #[test]
    fn coverage_counts_executed_operators_not_only_written_ones() {
        let programs: Vec<Program> = (1..=2)
            .flat_map(|s| plan_seed(s).unwrap())
            .map(|p| p.program)
            .collect();
        let refs: Vec<&Program> = programs.iter().collect();
        let coverage = coverage_json(&refs).unwrap();
        assert_eq!(coverage.get("programs").unwrap().as_u64(), Some(32));
        let ifs = coverage.get("if_statements").unwrap();
        let get = |k: &str| ifs.get(k).unwrap().as_u64().unwrap();
        assert!(get("in_text") >= get("evaluated"));
        assert!(get("evaluated") >= get("both_sides"));
        assert!(get("then_taken") <= get("evaluated") && get("else_taken") <= get("evaluated"));
        let operators = coverage.get("operators").unwrap().as_array().unwrap();
        assert!(!operators.is_empty());
        for op in operators {
            let executing = op.get("programs_executing").unwrap().as_u64().unwrap();
            let evaluations = op.get("evaluations").unwrap().as_u64().unwrap();
            // The evaluations are zero exactly when no program executes the operator.
            assert_eq!(evaluations == 0, executing == 0);
        }
    }

    fn raw_with(program: &str, pixel: bool) -> Raw {
        Raw {
            program: program.into(),
            seed: 1,
            attempt: 0,
            index: 0,
            count: 1,
            out_ty: "u32".into(),
            mode: "cpu-only".into(),
            adapter: None,
            interp: vec![1],
            twin: Ok(vec![1]),
            gpu: None,
            wgsl_validated: false,
            twin_ns: 0,
            gpu_first_ns: 0,
            gpu_warm_ns: 0,
            pixel: pixel.then(|| PixelRaw {
                twin: Ok(vec![1]),
                gpu: None,
                wgsl_validated: false,
                twin_ns: 4_000_000,
                gpu_first_ns: 0,
                gpu_warm_ns: 0,
            }),
        }
    }

    fn pixel_plan() -> ShapePlan {
        ShapePlan::new(&[Shape::Grid, Shape::Pixel])
    }

    #[test]
    fn a_pixel_batch_is_another_crate_than_the_same_seeds_without_pixel() {
        let grid = crate_identity(&[1, 2], DIFFTEST_FORMAT).unwrap();
        // A grid-only batch keeps the name it always had.
        assert_eq!(
            crate_identity_in(&[1, 2], DIFFTEST_FORMAT, &[Shape::Grid]).unwrap(),
            grid
        );
        let pixel =
            crate_identity_in(&[1, 2], DIFFTEST_FORMAT, &[Shape::Grid, Shape::Pixel]).unwrap();
        assert_ne!(pixel, grid);
        assert!(pixel.0.starts_with("difftest-case-run-"), "{}", pixel.0);
    }

    #[test]
    fn the_presence_of_pixel_results_must_match_the_plan_both_ways() {
        let plan = plan_seed(1).unwrap();
        let name = |k: usize| crate::emit::module_name(&plan[k].program);
        let mut shapes = pixel_plan();
        shapes.grid_only.insert(
            name(1),
            crate::shapes::GridOnly {
                reason: "r".into(),
                message: "m".into(),
            },
        );
        let planned = &plan[..3];
        let good = [
            raw_with(&name(0), true),
            raw_with(&name(1), false),
            raw_with(&name(2), true),
        ];
        assert_eq!(check_pixel_presence(planned, &good, &shapes), Ok(()));
        // A pixel half that went missing from the file of a program that was built with a `k_px`.
        let dropped = [
            raw_with(&name(0), false),
            raw_with(&name(1), false),
            raw_with(&name(2), true),
        ];
        let error = check_pixel_presence(planned, &dropped, &shapes).unwrap_err();
        assert!(
            error.contains(&name(0)) && error.contains("lacks"),
            "{error}"
        );
        // Pixel results for a program that the plan set aside.
        let extra = [
            raw_with(&name(0), true),
            raw_with(&name(1), true),
            raw_with(&name(2), true),
        ];
        let error = check_pixel_presence(planned, &extra, &shapes).unwrap_err();
        assert!(
            error.contains(&name(1)) && error.contains("holds"),
            "{error}"
        );
        // Without `pixel` requested, no file may hold pixel results.
        let grid_only = ShapePlan::new(&[Shape::Grid]);
        assert_eq!(
            check_pixel_presence(&planned[..1], &[raw_with(&name(0), false)], &grid_only),
            Ok(())
        );
        assert!(
            check_pixel_presence(&planned[..1], &[raw_with(&name(0), true)], &grid_only).is_err()
        );
    }

    #[test]
    fn the_report_observes_the_f32_bits_that_differ_between_the_shapes() {
        let f32_raw = |name: &str, grid: Vec<u32>, pixel: Vec<u32>| {
            let mut raw = raw_with(name, true);
            raw.out_ty = "f32".into();
            raw.count = grid.len();
            raw.gpu = Some(Ok(grid));
            raw.pixel.as_mut().unwrap().gpu = Some(Ok(pixel));
            raw
        };
        let one = 1.0f32.to_bits();
        let raws = [
            f32_raw("s1_a0", vec![one, one], vec![one, one]),
            f32_raw("s1_a1", vec![one, one, one], vec![one + 1, one, one + 3]),
            f32_raw("s1_a2", vec![one], vec![one + 2]),
            raw_with("s1_a3", true),
        ];
        let seen = float_observation_json(&raws);
        let number = |key: &str| seen.get(key).unwrap().as_u64().unwrap();
        assert_eq!(number("programs_compared"), 3);
        assert_eq!(number("programs_with_difference"), 2);
        assert_eq!(number("elements_differing"), 3);
        assert_eq!(number("max_ulp"), 3);
        let first = seen.get("first").unwrap();
        assert_eq!(first.get("program").unwrap().as_str(), Some("s1_a1"));
        assert_eq!(first.get("element").unwrap().as_u64(), Some(0));
        assert!(float_observation_json(&[]).get("first").unwrap().is_null());
    }

    #[test]
    fn the_report_names_the_shapes_the_pixel_programs_and_the_grid_only_reasons() {
        let plans = vec![plan_seed_timed(1).unwrap()];
        let options = Options {
            seeds: vec![1],
            cpu_only: true,
            shapes: vec![Shape::Grid, Shape::Pixel],
            target_dir: PathBuf::new(),
            repo_root: PathBuf::new(),
        };
        let mut shapes = pixel_plan();
        for (program, reason) in [
            ("s1_a0", "GUST-E0726: a"),
            ("s1_a1", "GUST-E0726: a"),
            ("s1_a2", "GUST-E0724: b"),
        ] {
            shapes.grid_only.insert(
                program.into(),
                crate::shapes::GridOnly {
                    reason: reason.into(),
                    message: format!("{reason} in full"),
                },
            );
        }
        let raws: Vec<Raw> = (0..16)
            .map(|k| raw_with(&format!("s1_a{k}"), true))
            .collect();
        let outcome = build_report(
            &options,
            &plans,
            &raws,
            &[],
            Json::Null,
            &CargoRun::default(),
            &shapes,
        );
        let section = outcome.report.get("shapes").unwrap();
        let requested: Vec<&str> = section
            .get("requested")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(requested, ["grid", "pixel"]);
        let by_shape = section.get("programs_by_shape").unwrap();
        assert_eq!(by_shape.get("grid").unwrap().as_u64(), Some(16));
        assert_eq!(by_shape.get("pixel").unwrap().as_u64(), Some(13));
        let only = section.get("grid_only").unwrap();
        assert_eq!(only.get("total").unwrap().as_u64(), Some(3));
        let by_reason = only.get("by_reason").unwrap();
        assert_eq!(by_reason.get("GUST-E0726: a").unwrap().as_u64(), Some(2));
        assert_eq!(by_reason.get("GUST-E0724: b").unwrap().as_u64(), Some(1));
        assert_eq!(only.get("programs").unwrap().as_array().unwrap().len(), 3);
        // The pixel timing is reported only when pixel ran.
        let timing = outcome.report.get("timing").unwrap();
        assert!(
            matches!(timing.get("pixel_twin_ms_total"), Some(Json::Float(v)) if (v - 64.0).abs() < 1e-9),
            "{timing:?}"
        );
        let plain = build_report(
            &options,
            &plans,
            &[],
            &[],
            Json::Null,
            &CargoRun::default(),
            &ShapePlan::new(&[Shape::Grid]),
        );
        let section = plain.report.get("shapes").unwrap();
        let by_shape = section.get("programs_by_shape").unwrap();
        assert_eq!(by_shape.get("pixel").unwrap().as_u64(), Some(0));
        assert!(
            plain
                .report
                .get("timing")
                .unwrap()
                .get("pixel_twin_ms_total")
                .is_none()
        );
    }
}
