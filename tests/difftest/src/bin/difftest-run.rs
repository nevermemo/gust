//! `difftest-run`: generate a batch of programs, run them on the interpreter, the CPU twin and the GPU, compare, and
//! write `report.json`. See docs/development/DIFFTEST.md.
//!
//! ```text
//! difftest-run --seeds 1-2 [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR]
//! difftest-run --refresh-lock [--repo-root DIR]
//! ```
//!
//! Exit status: 0 every program agreed, 1 at least one disagreement (the report lists them), 2 the harness failed
//! (build error, stale template lockfile, a result file missing).

use std::path::PathBuf;
use std::process::ExitCode;

use difftest::harness::{Options, parse_seeds, refresh_lock, run};

struct Args {
    seeds: Option<String>,
    cpu_only: bool,
    report: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    repo_root: Option<PathBuf>,
    refresh_lock: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        seeds: None,
        cpu_only: false,
        report: None,
        target_dir: None,
        repo_root: None,
        refresh_lock: false,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| iter.next().cloned().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--seeds" => parsed.seeds = Some(value("--seeds")?),
            "--cpu-only" => parsed.cpu_only = true,
            "--report" => parsed.report = Some(PathBuf::from(value("--report")?)),
            "--target-dir" => parsed.target_dir = Some(PathBuf::from(value("--target-dir")?)),
            "--repo-root" => parsed.repo_root = Some(PathBuf::from(value("--repo-root")?)),
            "--refresh-lock" => parsed.refresh_lock = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(parsed)
}

const USAGE: &str = "usage: difftest-run --seeds 1-2 [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR] | --refresh-lock [--repo-root DIR]";

/// The repository root: the `--repo-root` argument, else two levels above `tests/difftest` (this crate's manifest).
fn repo_root(given: Option<PathBuf>) -> Result<PathBuf, String> {
    let root =
        given.unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")));
    std::fs::canonicalize(&root).map_err(|e| format!("{}: {e}", root.display()))
}

fn main_inner() -> Result<ExitCode, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = parse(&args).map_err(|e| format!("{e}\n{USAGE}"))?;
    let repo_root = repo_root(args.repo_root)?;
    if args.refresh_lock {
        println!("{}", refresh_lock(&repo_root)?);
        return Ok(ExitCode::SUCCESS);
    }
    let seeds = parse_seeds(
        args.seeds
            .as_deref()
            .ok_or(format!("--seeds is required\n{USAGE}"))?,
    )?;
    let target_dir = match args.target_dir {
        Some(dir) => dir,
        None => std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_root.join("tests").join("difftest").join("target")),
    };
    std::fs::create_dir_all(&target_dir).map_err(|e| format!("{}: {e}", target_dir.display()))?;
    // Absolute but not canonical: the verbatim prefix that canonicalize adds on Windows does not mix with the forward
    // slashes the generated test builds its data paths with.
    let target_dir = std::path::absolute(&target_dir).map_err(|e| e.to_string())?;
    let options = Options {
        seeds,
        cpu_only: args.cpu_only,
        target_dir,
        repo_root,
    };
    let outcome = run(&options)?;
    let text = outcome.report.pretty();
    match &args.report {
        Some(path) => {
            std::fs::write(path, &text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        None => print!("{text}"),
    }
    let get = |key: &str| {
        outcome
            .report
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    eprintln!(
        "difftest-run: {} of {} programs executed ({}), {} mismatches: {}",
        get("programs_executed"),
        get("programs_requested"),
        if options.cpu_only { "cpu-only" } else { "gpu" },
        get("mismatch_count"),
        if outcome.passed { "PASS" } else { "FAIL" }
    );
    Ok(if outcome.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("difftest-run: {message}");
            ExitCode::from(2)
        }
    }
}
