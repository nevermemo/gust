//! `difftest-run`: generate a batch of programs, run them on the interpreter, the CPU twin and the GPU, compare, and
//! write `report.json`; or shrink one failing program. See docs/development/DIFFTEST.md.
//!
//! ```text
//! difftest-run --seeds 1-2 [--shapes grid,pixel] [--profile P] [--export-dir DIR] [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR]
//! difftest-run --shrink <seed>:<index> [--cpu-only] [--report PATH] [--max-candidates N] [--max-minutes M] [--target-dir DIR] [--repo-root DIR]
//! difftest-run --refresh-lock [--repo-root DIR]
//! ```
//!
//! Exit status of a batch: 0 every program agreed, 1 at least one disagreement (the report lists them), 2 the harness
//! failed (build error, stale template lockfile, a result file missing). Exit status of `--shrink`: 0 the failure
//! record was written, 1 the program does not fail (nothing to shrink), 2 the harness failed. CI never shrinks.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use difftest::harness::{Options, parse_seeds, refresh_lock, run};
use difftest::reduce::Limits;
use difftest::shapes::{Shape, parse_shapes};
use difftest::shrink::{Shrunk, parse_target, shrink};

struct Args {
    seeds: Option<String>,
    shrink: Option<String>,
    shapes: Option<String>,
    cpu_only: bool,
    profile: Option<String>,
    export_dir: Option<PathBuf>,
    report: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    repo_root: Option<PathBuf>,
    refresh_lock: bool,
    max_candidates: Option<usize>,
    max_minutes: Option<u64>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        seeds: None,
        shrink: None,
        shapes: None,
        cpu_only: false,
        profile: None,
        export_dir: None,
        report: None,
        target_dir: None,
        repo_root: None,
        refresh_lock: false,
        max_candidates: None,
        max_minutes: None,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| iter.next().cloned().ok_or(format!("{name} needs a value"));
        let number = |name: &str, text: String| {
            text.parse::<u64>()
                .map_err(|e| format!("{name} {text:?}: {e}"))
        };
        match arg.as_str() {
            "--seeds" => parsed.seeds = Some(value("--seeds")?),
            "--shrink" => parsed.shrink = Some(value("--shrink")?),
            "--shapes" => parsed.shapes = Some(value("--shapes")?),
            "--profile" => parsed.profile = Some(value("--profile")?),
            "--export-dir" => parsed.export_dir = Some(PathBuf::from(value("--export-dir")?)),
            "--cpu-only" => parsed.cpu_only = true,
            "--report" => parsed.report = Some(PathBuf::from(value("--report")?)),
            "--target-dir" => parsed.target_dir = Some(PathBuf::from(value("--target-dir")?)),
            "--repo-root" => parsed.repo_root = Some(PathBuf::from(value("--repo-root")?)),
            "--refresh-lock" => parsed.refresh_lock = true,
            "--max-candidates" => {
                let text = value("--max-candidates")?;
                parsed.max_candidates = Some(number("--max-candidates", text)? as usize);
            }
            "--max-minutes" => {
                let text = value("--max-minutes")?;
                parsed.max_minutes = Some(number("--max-minutes", text)?);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if parsed.shrink.is_some() && parsed.seeds.is_some() {
        return Err("--shrink and --seeds are exclusive".into());
    }
    if parsed.shrink.is_some() && parsed.shapes.is_some() {
        return Err(
            "--shrink runs the grid kernel only; the reducer does not carry a pixel kernel (--shapes)"
                .into(),
        );
    }
    if parsed.shrink.is_some() && parsed.export_dir.is_some() {
        // Every candidate of a shrink is built under the same package name, so one record directory would be shared
        // (and discarded) across candidates: the pair is refused instead of giving records that mean nothing.
        return Err(
            "--shrink and --export-dir are exclusive: the candidates of a shrink would share one record directory"
                .into(),
        );
    }
    if let Some(profile) = &parsed.profile {
        // The run crate compiles Slang at run time, which only `dev` and `dev+<capability>` keep (gust's build script).
        if profile != "dev" && !profile.starts_with("dev+") {
            return Err(format!(
                "--profile {profile:?}: the run crate needs the run-time compiler, so the profile is `dev` or `dev+<capability>`"
            ));
        }
    }
    if (parsed.max_candidates.is_some() || parsed.max_minutes.is_some()) && parsed.shrink.is_none()
    {
        return Err("--max-candidates and --max-minutes belong to --shrink".into());
    }
    Ok(parsed)
}

const USAGE: &str = "usage: difftest-run --seeds 1-2 [--shapes grid,pixel] [--profile P] [--export-dir DIR] [--cpu-only] [--report PATH] [--target-dir DIR] [--repo-root DIR] | --shrink <seed>:<index> [--cpu-only] [--report PATH] [--max-candidates N] [--max-minutes M] | --refresh-lock [--repo-root DIR]";

/// The repository root: the `--repo-root` argument, else two levels above `tests/difftest` (this crate's manifest).
fn repo_root(given: Option<PathBuf>) -> Result<PathBuf, String> {
    let root =
        given.unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")));
    std::fs::canonicalize(&root).map_err(|e| format!("{}: {e}", root.display()))
}

fn write_or_print(report: &Option<PathBuf>, text: &str) -> Result<(), String> {
    match report {
        Some(path) => std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display())),
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn main_inner() -> Result<ExitCode, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = parse(&args).map_err(|e| format!("{e}\n{USAGE}"))?;
    let repo_root = repo_root(args.repo_root)?;
    if args.refresh_lock {
        println!("{}", refresh_lock(&repo_root)?);
        return Ok(ExitCode::SUCCESS);
    }
    let seeds = match (&args.seeds, &args.shrink) {
        (Some(seeds), _) => parse_seeds(seeds)?,
        (None, Some(_)) => Vec::new(),
        (None, None) => return Err(format!("--seeds is required\n{USAGE}")),
    };
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
    let shapes = match &args.shapes {
        Some(text) => parse_shapes(text)?,
        None => vec![Shape::Grid],
    };
    let export_dir = args
        .export_dir
        .map(std::path::absolute)
        .transpose()
        .map_err(|e| format!("--export-dir: {e}"))?;
    let options = Options {
        seeds,
        cpu_only: args.cpu_only,
        shapes,
        target_dir,
        repo_root,
        profile: args.profile.unwrap_or_else(|| "dev".to_owned()),
        export_dir,
    };
    if let Some(target) = &args.shrink {
        let mut limits = Limits::default();
        if let Some(candidates) = args.max_candidates {
            limits.candidates = candidates;
        }
        if let Some(minutes) = args.max_minutes {
            limits.time = Duration::from_secs(minutes * 60);
        }
        return match shrink(&options, parse_target(target)?, limits)? {
            Shrunk::NoFailure(message) => {
                eprintln!("difftest-run: {message}");
                Ok(ExitCode::from(1))
            }
            Shrunk::Record(record) => {
                write_or_print(&args.report, &record.pretty())?;
                let get = |key: &str| record.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                eprintln!(
                    "difftest-run: shrunk {target} in {} candidates ({} accepted, {} cache hits), stopped by {}",
                    get("candidates_used"),
                    get("accepted_steps"),
                    get("cache_hits"),
                    record
                        .get("stopped_by")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                );
                Ok(ExitCode::SUCCESS)
            }
        };
    }
    let outcome = run(&options)?;
    write_or_print(&args.report, &outcome.report.pretty())?;
    let get = |key: &str| {
        outcome
            .report
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    let shape_count = |section: &str, key: &str| {
        outcome
            .report
            .get("shapes")
            .and_then(|s| s.get(section))
            .and_then(|s| s.get(key))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    let shapes_text = if options.shapes.contains(&Shape::Pixel) {
        format!(
            ", shapes grid,pixel: {} pixel, {} grid-only",
            shape_count("programs_by_shape", "pixel"),
            shape_count("grid_only", "total")
        )
    } else {
        String::new()
    };
    eprintln!(
        "difftest-run: {} of {} programs executed ({}{shapes_text}), {} mismatches: {}",
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

#[cfg(test)]
mod tests {
    use super::parse;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn shrink_with_export_dir_is_refused_with_a_clear_message() {
        let message = parse(&args(&["--shrink", "1:0", "--export-dir", "records"])).err();
        assert_eq!(
            message.as_deref(),
            Some(
                "--shrink and --export-dir are exclusive: the candidates of a shrink would share one record directory"
            )
        );
        // The order of the arguments does not matter.
        assert!(parse(&args(&["--export-dir", "records", "--shrink", "1:0"])).is_err());
    }

    #[test]
    fn export_dir_alone_and_shrink_alone_still_parse() {
        assert!(parse(&args(&["--seeds", "1-2", "--export-dir", "records"])).is_ok());
        assert!(parse(&args(&["--shrink", "1:0"])).is_ok());
    }
}
