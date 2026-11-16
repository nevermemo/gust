//! `cargo xtask check-bench` (M10 b1b): the benchmark harness's CI-safe functional run, and the renderer of
//! `docs/BENCHMARKS.md` from filed results.
//!
//! - `--functional [--quick]`: builds and runs `bench/gust-bench` with `run --functional --quick` into a scratch directory
//!   outside the work tree. The run checks every backend's output against the CPU reference on whatever adapter is there
//!   (lavapipe in CI); it times nothing for publication and is never publishable. The command fails unless the harness
//!   exits 0, the file says `"functional": true` and `"publishable": false`, no cell failed and at least one check passed.
//! - `--render [--results DIR] [--check] [--docs FILE]`: renders the tables of the docs page from the aggregate files
//!   `DIR/<adapter-slug>/<commit>-<route>.json` (each with its `.samples.json` beside it) and checks that the page's
//!   header block is the canonical one and its thresholds block is the harness's. A file is refused unless it says
//!   `"publishable": true` with no banner, its sidecar exists, and the commit that added it carries a
//!   `Quiet-window attestation:` line (plan 2.4 item 5). A file is one group of the harness (`publishable`, or `spirv` from the
//!   SPIR-V route, D193): each is a section of its own, the `spirv` one labelled "checks off on both sides", and a file of the
//!   SPIR-V route in the publishable group (pre-D193) is read and labelled legacy, not refused. With no files the tables say
//!   none are filed. `--check` writes
//!   nothing and fails when the page would change.
//!
//! xtask cannot depend on the bench crate (it is a workspace of its own), so the thresholds are read from the `pub const`
//! lines of its `judge.rs`.

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    json::{self, Json},
    process::{exit_code, workspace_command},
    workspace::workspace_root,
};

/// The page the renderer maintains.
const DOCS_PAGE: &str = "docs/BENCHMARKS.md";
/// Where filed results live, relative to the repository root.
const RESULTS_DIR: &str = "bench/results";
/// The harness source whose `pub const` lines are the thresholds.
const JUDGE_SOURCE: &str = "bench/gust-bench/src/judge.rs";
/// The least number of commit characters a results file name carries (the harness prints 12).
const COMMIT_CHARS: usize = 12;
/// Guide pages that mirror the header block (`10-benchmarks*.md` in this directory), when they exist.
const MIRROR_DIR: &str = "crates/gust-wgpu/src/guide";
const MIRROR_PREFIX: &str = "10-benchmarks";
/// The line of a results commit that attests the quiet window.
const ATTESTATION: &str = "Quiet-window attestation:";

/// The header every place that shows benchmark results carries. One source; the page and the mirrors must match it.
const HEADER: &str = "\
> Every comparator (the hand-written WGSL, the rayon CPU backend and the rest) runs inside GUST's harness with the settings stated here, not in its authors' recommended configuration; corrections are welcome as issues against this repository.
>
> A figure describes one adapter, one driver, one operating system and one commit, measured as the methods below say. It is not a ranking, and this page makes no speed claim.";

/// The metrics of a table, in column order.
const METRICS: [(&str, &str); 4] = [
    ("end_to_end", "end to end"),
    ("dispatch_host", "dispatch (host)"),
    ("gpu_single", "GPU, one dispatch"),
    ("gpu_batch_per_dispatch", "GPU, per dispatch of 16"),
];

#[derive(Debug, PartialEq)]
struct Options {
    functional: bool,
    render: bool,
    check: bool,
    results: Option<String>,
    docs: Option<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self {
            functional: false,
            render: false,
            check: false,
            results: None,
            docs: None,
        };
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--functional" => options.functional = true,
                // The functional run is always quick; the flag is accepted so the CI line says what it does.
                "--quick" => {}
                "--render" => options.render = true,
                "--check" => options.check = true,
                "--results" => {
                    options.results =
                        Some(iter.next().ok_or("--results needs a directory")?.clone());
                }
                "--docs" => options.docs = Some(iter.next().ok_or("--docs needs a file")?.clone()),
                other => return Err(format!("unknown check-bench argument `{other}`")),
            }
        }
        if options.functional == options.render {
            return Err("check-bench needs exactly one of --functional and --render".into());
        }
        if options.functional
            && (options.check || options.results.is_some() || options.docs.is_some())
        {
            return Err("--check, --results and --docs belong to --render".into());
        }
        Ok(options)
    }
}

pub(crate) fn check_bench(args: &[String]) -> Result<(), String> {
    let options = Options::parse(args)?;
    let root = workspace_root()?;
    if options.functional {
        functional(&root)
    } else {
        render(&root, &options, &|file| git_attestation(&root, file))
    }
}

// ---- --functional ----

fn functional(root: &Path) -> Result<(), String> {
    let out_dir = std::env::temp_dir().join(format!("gust-check-bench-{}", std::process::id()));
    fs::create_dir_all(&out_dir).map_err(|error| format!("{}: {error}", out_dir.display()))?;
    let out = out_dir.join("functional.json");
    let status = workspace_command("cargo")
        .args([
            "run",
            "--release",
            "--locked",
            "--manifest-path",
            "bench/Cargo.toml",
            "-p",
            "gust-bench",
            "--",
            "run",
            "--functional",
            "--quick",
            "--label",
            "check-bench",
            "--out",
        ])
        .arg(&out)
        .current_dir(root)
        .status()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    let result = if status.success() {
        fs::read_to_string(&out)
            .map_err(|error| format!("the harness wrote no results file: {error}"))
            .and_then(|text| functional_verdict(&text))
    } else {
        Err(format!(
            "gust-bench run --functional failed ({})",
            exit_code(status)
        ))
    };
    let _ = fs::remove_dir_all(&out_dir);
    let adapter = result?;
    println!(
        "check-bench --functional: every correctness check passed on {adapter} (no timing is published)"
    );
    Ok(())
}

/// Reads the results file of a functional run: the adapter's name when the run is what it claims, else the reason not.
fn functional_verdict(text: &str) -> Result<String, String> {
    let file =
        json::parse(text).map_err(|error| format!("the results file is not JSON: {error}"))?;
    if !matches!(file.get("functional"), Some(Json::Bool(true))) {
        return Err("the results file does not say \"functional\": true".into());
    }
    if !matches!(file.get("publishable"), Some(Json::Bool(false))) {
        return Err("a functional run must say \"publishable\": false".into());
    }
    let failures = file
        .get("failures")
        .and_then(Json::as_array)
        .ok_or("the results file has no failures list")?;
    if !failures.is_empty() {
        let first = failures.first().and_then(Json::as_str).unwrap_or("?");
        return Err(format!("{} failure(s), the first: {first}", failures.len()));
    }
    let cells = file
        .get("cells")
        .and_then(Json::as_array)
        .ok_or("the results file has no cells")?;
    let checks: u64 = cells
        .iter()
        .filter_map(|cell| match cell.get("checks_passed") {
            Some(Json::Number(text)) => text.parse::<u64>().ok(),
            _ => None,
        })
        .sum();
    if checks == 0 {
        return Err("no correctness check ran".into());
    }
    Ok(file
        .get("adapter")
        .and_then(|adapter| adapter.get("name"))
        .and_then(Json::as_str)
        .unwrap_or("an unnamed adapter")
        .to_owned())
}

// ---- --render ----

/// One filed aggregate, after the checks.
#[derive(Debug)]
struct Filed {
    adapter: String,
    route: String,
    /// The harness's backend group of the file: `publishable` or `spirv` (D193). A file without the field is read
    /// as `publishable`.
    group: String,
    /// A file of the SPIR-V route judged in the publishable group (pre-D193): read, rendered and labelled, never refused.
    legacy: bool,
    commit: String,
    file: Json,
}

fn render(
    root: &Path,
    options: &Options,
    attestation: &dyn Fn(&Path) -> Option<String>,
) -> Result<(), String> {
    let docs_path = root.join(options.docs.as_deref().unwrap_or(DOCS_PAGE));
    let original = fs::read_to_string(&docs_path)
        .map_err(|error| format!("{}: {error}", docs_path.display()))?;
    let crlf = original.contains("\r\n");
    let text = original.replace("\r\n", "\n");

    let results_dir = root.join(options.results.as_deref().unwrap_or(RESULTS_DIR));
    let filed = load_results(&results_dir, attestation)?;
    let judge = fs::read_to_string(root.join(JUDGE_SOURCE))
        .map_err(|error| format!("{JUDGE_SOURCE}: {error}"))?;

    let mut updated = replace_block(&text, "bench-header", HEADER)?;
    updated = replace_block(&updated, "bench-thresholds", &thresholds_block(&judge)?)?;
    updated = replace_block(
        &updated,
        "bench-tables",
        &render_tables(&filed, &sample_limits(&judge)?),
    )?;

    let mirror_problems = check_mirrors(root)?;
    if !mirror_problems.is_empty() {
        return Err(mirror_problems.join("\n"));
    }
    if updated == text {
        println!(
            "check-bench --render: {} is up to date ({} filed result(s))",
            DOCS_PAGE,
            filed.len()
        );
        return Ok(());
    }
    if options.check {
        return Err(format!(
            "{DOCS_PAGE} is out of date: run `cargo xtask check-bench --render` and commit the result"
        ));
    }
    let output = if crlf {
        updated.replace('\n', "\r\n")
    } else {
        updated
    };
    fs::write(&docs_path, output).map_err(|error| format!("{}: {error}", docs_path.display()))?;
    println!(
        "check-bench --render: wrote {DOCS_PAGE} ({} filed result(s))",
        filed.len()
    );
    Ok(())
}

/// The attestation line of the commit that last touched `file`, `None` when git knows no commit for it.
fn git_attestation(root: &Path, file: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["log", "-1", "--format=%B", "--"])
        .arg(file)
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let message = String::from_utf8_lossy(&output.stdout).into_owned();
    Some(
        message
            .lines()
            .find(|line| line.trim_start().starts_with(ATTESTATION))
            .map(|line| line.trim().to_owned())
            .unwrap_or_default(),
    )
}

/// Every aggregate under `dir` (`<adapter-slug>/<commit>-<route>.json`), in path order. A file that is not publishable,
/// has no sidecar or has no attestation is an error naming it: a refused file never becomes a table.
fn load_results(
    dir: &Path,
    attestation: &dyn Fn(&Path) -> Option<String>,
) -> Result<Vec<Filed>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = Vec::new();
    let mut refusals = Vec::new();
    for slug in sorted_entries(dir)? {
        if !slug.is_dir() {
            let name = slug.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".json") {
                refusals.push(format!(
                    "{}: a results file sits directly under the results directory (file it as <adapter-slug>/<commit>-<route>.json)",
                    slug.display()
                ));
            }
            continue;
        }
        for path in sorted_entries(&slug)? {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".json") && !name.ends_with(".samples.json") {
                paths.push(path);
            }
        }
    }
    let mut filed = Vec::new();
    for path in paths {
        match load_one(&path, attestation) {
            Ok(entry) => filed.push(entry),
            Err(reason) => refusals.push(format!("{}: {reason}", path.display())),
        }
    }
    if refusals.is_empty() {
        Ok(filed)
    } else {
        Err(format!(
            "refusing results that are not publishable:\n  {}",
            refusals.join("\n  ")
        ))
    }
}

fn sorted_entries(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = fs::read_dir(dir)
        .map_err(|error| format!("{}: {error}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .collect::<Vec<_>>();
    entries.sort();
    Ok(entries)
}

fn load_one(path: &Path, attestation: &dyn Fn(&Path) -> Option<String>) -> Result<Filed, String> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let file = json::parse(&text).map_err(|error| format!("not JSON: {error}"))?;
    if !matches!(file.get("publishable"), Some(Json::Bool(true))) {
        let why = file
            .get("publishable_reason")
            .and_then(Json::as_str)
            .unwrap_or("no reason recorded");
        return Err(format!("not publishable: {why}"));
    }
    if !matches!(file.get("banner"), None | Some(Json::Null)) {
        return Err("carries the not-publishable banner".into());
    }
    let sidecar = path.with_extension("samples.json");
    if !sidecar.is_file() {
        return Err(format!(
            "its sampler sidecar {} is missing",
            sidecar.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        ));
    }
    match attestation(path) {
        None => return Err("no commit adds it (file the results in a commit of their own)".into()),
        Some(line) if line.is_empty() => {
            return Err(format!("its commit carries no `{ATTESTATION}` line"));
        }
        Some(_) => {}
    }
    let adapter = file
        .get("adapter")
        .and_then(|adapter| adapter.get("name"))
        .and_then(Json::as_str)
        .ok_or("no adapter name")?
        .to_owned();
    let route = file
        .get("route")
        .and_then(Json::as_str)
        .ok_or("no route")?
        .to_owned();
    let commit = file
        .get("build")
        .and_then(|build| build.get("commit"))
        .and_then(Json::as_str)
        .ok_or("no build commit")?
        .to_owned();
    // The path says what the file is; the file must say the same (a results file filed under another adapter's
    // directory or another commit's name is refused).
    let slug = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if slug != slugify(&adapter) {
        return Err(format!(
            "filed under `{slug}`, but its adapter `{adapter}` belongs under `{}`",
            slugify(&adapter)
        ));
    }
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let Some((named_commit, named_route)) = stem.rsplit_once('-') else {
        return Err("its name is not <commit>-<route>.json".to_owned());
    };
    if named_route != route {
        return Err(format!(
            "named for route `{named_route}` but the file's route is `{route}`"
        ));
    }
    if named_commit.chars().count() < COMMIT_CHARS {
        return Err(format!(
            "named for commit `{named_commit}`, which has fewer than the {COMMIT_CHARS} characters the harness prints"
        ));
    }
    if !(commit.starts_with(named_commit) || named_commit.starts_with(&commit)) {
        return Err(format!(
            "named for commit `{named_commit}` but the file's build commit is `{commit}`"
        ));
    }
    let group = file
        .get("group")
        .and_then(Json::as_str)
        .unwrap_or("publishable")
        .to_owned();
    let legacy = check_group(&file, &group, &route)?;
    Ok(Filed {
        adapter,
        route,
        group,
        legacy,
        commit,
        file,
    })
}

/// The reference backend of the `spirv` group's kill rule (`bench/gust-bench/src/group.rs`).
const SPIRV_REFERENCE: &str = "wgsl-prebuilt-unchecked";

/// The rows a `spirv` file may hold: the group's three GPU backends and the CPU rows that run beside them.
const SPIRV_GROUP_ROWS: [&str; 9] = [
    "gust-gpu-spirv",
    SPIRV_REFERENCE,
    "wgsl-percall-unchecked",
    "gust-twin",
    "gust-cpu",
    "gust-cpu-par",
    "rayon-1",
    "rayon-N",
    "serial",
];

/// The group of a filed aggregate against its route (D193). Only the table groups are filed: `publishable` (the WGSL
/// build) and `spirv` (the SPIR-V route's, checks off on both sides). A `spirv` file must come from the SPIR-V route,
/// hold only its own rows (`gust-gpu-spirv`, `wgsl-prebuilt-unchecked`, `wgsl-percall-unchecked` and the CPU rows),
/// have at least one run, and carry the group's kill rule in every timed run (`group` and `reference` in each run's
/// record; a run marked `"functional": true` has none by design and may have a null one). A publishable file holds no
/// `-unchecked` point. A file of the SPIR-V route in the
/// publishable group, or a publishable file with a `gust-gpu-spirv` point, is **legacy** (pre-D193): it is read and
/// labelled, not refused. Returns whether the file is legacy.
fn check_group(file: &Json, group: &str, route: &str) -> Result<bool, String> {
    let has_spirv_point = file
        .get("points")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .any(|point| point.get("backend").and_then(Json::as_str) == Some("gust-gpu-spirv"));
    let points = || {
        file.get("points")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .filter_map(|point| point.get("backend").and_then(Json::as_str))
    };
    match group {
        "publishable" => {
            if let Some(backend) = points().find(|backend| backend.ends_with("-unchecked")) {
                return Err(format!(
                    "a publishable file holds the checks-off row `{backend}`: those rows belong to the spirv group (D193)"
                ));
            }
            Ok(route == "spirv" || has_spirv_point)
        }
        "spirv" => {
            if route != "spirv" {
                return Err(format!(
                    "group `spirv` but route `{route}`: the spirv group is the SPIR-V route's"
                ));
            }
            if let Some(backend) = points().find(|backend| !SPIRV_GROUP_ROWS.contains(backend)) {
                return Err(format!(
                    "a spirv file holds the row `{backend}`, which is not the spirv group's (gust-gpu-spirv, {SPIRV_REFERENCE}, wgsl-percall-unchecked and the CPU rows)"
                ));
            }
            let runs = file.get("runs").and_then(Json::as_array).unwrap_or(&[]);
            if runs.is_empty() {
                return Err("a spirv file has no runs, so no kill-rule record to read".into());
            }
            for run in runs {
                let rule = run.get("kill_rule");
                if matches!(rule, None | Some(Json::Null)) {
                    if matches!(run.get("functional"), Some(Json::Bool(true))) {
                        continue;
                    }
                    return Err("a timed run of a spirv file has a null kill rule: it must carry the group's record".into());
                }
                let of = |key: &str| rule.and_then(|r| r.get(key)).and_then(Json::as_str);
                if of("group") != Some("spirv") || of("reference") != Some(SPIRV_REFERENCE) {
                    return Err(format!(
                        "a run's kill rule is not the spirv group's (group {:?}, reference {:?}; it must be spirv against {SPIRV_REFERENCE})",
                        of("group"),
                        of("reference")
                    ));
                }
            }
            Ok(false)
        }
        other => Err(format!(
            "group `{other}` is not a table group (only publishable and spirv aggregates are filed)"
        )),
    }
}

/// The directory name of an adapter: lower case, every run of other characters one `-`.
fn slugify(name: &str) -> String {
    let mut slug = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
    }
    slug.trim_end_matches('-').to_owned()
}

fn number(value: Option<&Json>) -> Option<f64> {
    match value {
        Some(Json::Number(text)) => text.parse().ok(),
        _ => None,
    }
}

fn ms(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.2}")
    } else if value >= 1.0 {
        format!("{value:.3}")
    } else {
        format!("{value:.4}")
    }
}

/// workload -> n -> backend -> metric -> (median, min, max), in ms.
type Rows = BTreeMap<String, BTreeMap<u64, BTreeMap<String, BTreeMap<String, (f64, f64, f64)>>>>;

/// The sample target and the per-quantity cap of the harness, read from its `judge.rs` like the thresholds block.
struct SampleLimits {
    target: f64,
    cap_text: String,
    target_text: String,
}

/// The legend of the marks a cell carries (plan 2.4: the conditions travel with the figure).
fn legend(limits: &SampleLimits) -> String {
    format!(
        "Marks after a figure: **S** the quantity has two states (heavy tail or wide gap between the 5th and 95th percentile); \
**D** the figure is above the adapter's DRAM peak, so it is not a DRAM bandwidth; **K** the cell is an offender of the kill rule in at least one run \
(listed, counted only if it recurs with the control in tolerance); **F** fewer than {} samples were collected in at least one run (the {} s cap stopped it).",
        limits.target_text, limits.cap_text
    )
}

fn text_of(value: Option<&Json>) -> String {
    match value {
        Some(Json::String(text)) => text.clone(),
        Some(Json::Number(text)) => text.clone(),
        _ => "unknown".to_owned(),
    }
}

/// `YYYY-MM-DD` of an ISO timestamp (the first ten characters), or `unknown`.
fn day_of(stamp: Option<&Json>) -> String {
    stamp
        .and_then(Json::as_str)
        .map_or_else(|| "unknown".to_owned(), |s| s.chars().take(10).collect())
}

fn is_true(value: Option<&Json>) -> bool {
    matches!(value, Some(Json::Bool(true)))
}

/// The marks of one point: `workload`, `n`, `backend`, `metric`.
fn marks_of(
    entry: &Filed,
    limits: &SampleLimits,
) -> BTreeMap<(String, u64, String, String), String> {
    let mut marks: BTreeMap<(String, u64, String, String), String> = BTreeMap::new();
    let mut mark = |key: (String, u64, String, String), letter: char| {
        let slot = marks.entry(key).or_default();
        if !slot.contains(letter) {
            slot.push(letter);
        }
    };
    let key_of = |item: &Json, metric_key: &str| {
        Some((
            item.get("workload")?.as_str()?.to_owned(),
            number(item.get("n"))? as u64,
            item.get("backend")?.as_str()?.to_owned(),
            item.get(metric_key)?.as_str()?.to_owned(),
        ))
    };
    for point in entry
        .file
        .get("points")
        .and_then(Json::as_array)
        .unwrap_or(&[])
    {
        let Some(key) = key_of(point, "metric") else {
            continue;
        };
        if is_true(point.get("two_state_suspect")) || is_true(point.get("two_state_wide_gap")) {
            mark(key.clone(), 'S');
        }
        let short = point
            .get("samples_per_run")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|count| number(Some(count)).is_none_or(|n| n < limits.target));
        if short {
            mark(key, 'F');
        }
    }
    for item in entry
        .file
        .get("above_dram_peak")
        .and_then(Json::as_array)
        .unwrap_or(&[])
    {
        if let Some(key) = key_of(item, "metric") {
            mark(key, 'D');
        }
    }
    for item in entry
        .file
        .get("kill_rule_reproducibility")
        .and_then(|repro| repro.get("offenders"))
        .and_then(Json::as_array)
        .unwrap_or(&[])
    {
        if let Some(key) = key_of(item, "quantity") {
            mark(key, 'K');
        }
    }
    marks
}

/// The conditions of a section: adapter, driver, operating system, commit, route, run window, seeds, thresholds version.
fn conditions(entry: &Filed) -> String {
    let adapter = entry.file.get("adapter");
    let driver = format!(
        "{} {}",
        text_of(adapter.and_then(|a| a.get("driver"))),
        text_of(adapter.and_then(|a| a.get("driver_info")))
    );
    let runs = entry
        .file
        .get("runs")
        .and_then(Json::as_array)
        .unwrap_or(&[]);
    let first = runs.first().and_then(|run| run.get("window"));
    let last = runs.last().and_then(|run| run.get("window"));
    let (from, to) = (
        day_of(first.and_then(|w| w.get("started_utc"))),
        day_of(last.and_then(|w| w.get("ended_utc"))),
    );
    let seeds = runs
        .iter()
        .map(|run| text_of(run.get("run_seed")))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Adapter {}; driver {}; OS {}; route {}; commit {}; {} runs from {from} to {to}; seeds {seeds}; thresholds version {}.",
        entry.adapter,
        driver.trim(),
        text_of(entry.file.get("meta").and_then(|m| m.get("os"))),
        entry.route,
        entry.commit,
        runs.len(),
        text_of(
            entry
                .file
                .get("thresholds")
                .and_then(|thresholds| thresholds.get("version"))
        ),
    )
}

/// The markdown of the tables block: one section per filed aggregate, one table per workload.
fn render_tables(filed: &[Filed], limits: &SampleLimits) -> String {
    if filed.is_empty() {
        return "No results are filed. Nothing here is publishable until three complete runs of one session pass the \
                rule above and are filed with the attestation line in their commit. The renderer refuses anything else."
            .to_owned();
    }
    let mut out = String::new();
    for entry in filed {
        let short = entry.commit.chars().take(12).collect::<String>();
        // Each group is a section and a table of its own (D193): the `spirv` group's rows, whose checks are off on both
        // sides, are never in the same table as the publishable group's.
        let label = if entry.legacy {
            ", legacy (pre-D193)"
        } else if entry.group == "spirv" {
            ", group spirv (checks off on both sides)"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "### {}, route {}{label}, commit {short}\n",
            entry.adapter, entry.route
        );
        let _ = writeln!(out, "{}\n", conditions(entry));
        if entry.legacy {
            let _ = writeln!(
                out,
                "Legacy file (pre-D193): the SPIR-V route was judged here in the publishable group, against hand WGSL that has wgpu's runtime checks on, which the SPIR-V route does not. The comparison is not the one the owner ruled for (D193: the `spirv` group, checks off on both sides). Its `gust-gpu-spirv` rows are not comparable with a `spirv` group's.\n"
            );
        } else if entry.group == "spirv" {
            let _ = writeln!(
                out,
                "Group `spirv` (D193): checks off on both sides. `gust-gpu-spirv` against hand WGSL built without wgpu's runtime shader checks (`wgsl-prebuilt-unchecked`, with `wgsl-percall-unchecked` as the control). Its rows are not comparable with the publishable group's `wgsl-prebuilt` and `wgsl-percall`, which have the checks on.\n"
            );
        }
        let _ = writeln!(
            out,
            "Median over the per-run medians, in ms; the bracket is the spread of the run medians (min to max).\n"
        );
        let _ = writeln!(out, "{}\n", legend(limits));
        let marks = marks_of(entry, limits);
        let mut rows: Rows = BTreeMap::new();
        for point in entry
            .file
            .get("points")
            .and_then(Json::as_array)
            .unwrap_or(&[])
        {
            let (Some(workload), Some(backend), Some(metric)) = (
                point.get("workload").and_then(Json::as_str),
                point.get("backend").and_then(Json::as_str),
                point.get("metric").and_then(Json::as_str),
            ) else {
                continue;
            };
            let (Some(n), Some(median), Some(low), Some(high)) = (
                number(point.get("n")),
                number(point.get("median_ms")),
                number(point.get("min_run_median_ms")),
                number(point.get("max_run_median_ms")),
            ) else {
                continue;
            };
            rows.entry(workload.to_owned())
                .or_default()
                .entry(n as u64)
                .or_default()
                .entry(backend.to_owned())
                .or_default()
                .insert(metric.to_owned(), (median, low, high));
        }
        for (workload, sizes) in &rows {
            let _ = writeln!(out, "**{workload}**\n");
            let head = METRICS
                .iter()
                .map(|(_, label)| *label)
                .collect::<Vec<_>>()
                .join(" | ");
            let _ = writeln!(out, "| n | backend | {head} |");
            let _ = writeln!(
                out,
                "|---|---|{}|",
                "---|".repeat(METRICS.len() - 1) + "---"
            );
            for (n, backends) in sizes {
                for (backend, metrics) in backends {
                    let cells = METRICS
                        .iter()
                        .map(|(id, _)| {
                            metrics
                                .get(*id)
                                .map_or("-".to_owned(), |(median, low, high)| {
                                    let key =
                                        (workload.clone(), *n, backend.clone(), (*id).to_owned());
                                    let flags = marks
                                        .get(&key)
                                        .map_or(String::new(), |m| format!(" **{m}**"));
                                    format!(
                                        "{} ({} to {}){flags}",
                                        ms(*median),
                                        ms(*low),
                                        ms(*high)
                                    )
                                })
                        })
                        .collect::<Vec<_>>()
                        .join(" | ");
                    let _ = writeln!(out, "| {n} | {backend} | {cells} |");
                }
            }
            out.push('\n');
        }
    }
    out.trim_end().to_owned()
}

/// The value of one `pub const NAME ... = VALUE;` line of the harness's `judge.rs`.
fn judge_value(judge: &str, name: &str) -> Result<String, String> {
    judge
        .lines()
        .find_map(|line| {
            let rest = line
                .trim_end()
                .strip_prefix("pub const ")?
                .strip_prefix(name)?;
            let (_, value) = rest.split_once('=')?;
            Some(value.trim().trim_end_matches(';').trim().to_owned())
        })
        .ok_or_else(|| format!("{JUDGE_SOURCE} has no `pub const {name}` line"))
}

/// `SAMPLES_TARGET` and `SAMPLE_CAP_SECS`, as the marks and the legend of the tables need them.
fn sample_limits(judge: &str) -> Result<SampleLimits, String> {
    let target_text = judge_value(judge, "SAMPLES_TARGET")?;
    let target = target_text
        .parse::<f64>()
        .map_err(|_| format!("{JUDGE_SOURCE}: SAMPLES_TARGET `{target_text}` is not a number"))?;
    Ok(SampleLimits {
        target,
        target_text,
        cap_text: judge_value(judge, "SAMPLE_CAP_SECS")?,
    })
}

/// The thresholds block, from the `pub const` lines of the harness's `judge.rs`.
fn thresholds_block(judge: &str) -> Result<String, String> {
    let value = |name: &str| judge_value(judge, name);
    let t_gpu = value("T_GPU_PERCENT")?;
    let t_cpu = value("T_CPU_FRACTION")?;
    let t_cpu_text = if t_cpu == "None" {
        "not set (no `vmmem` process in any recorded window; a `vmmem` or `vmmemWSL` process present makes a window not quiet)".to_owned()
    } else {
        t_cpu
    };
    Ok(format!(
        "| Constant | Value |\n|---|---|\n| Thresholds version | {} |\n\
         | `T_gpu` (foreign GPU-engine use, percent of the engine, any sample) | {t_gpu} |\n\
         | `T_cpu` (`vmmem` CPU growth, cpu-seconds per second) | {t_cpu_text} |\n\
         | `N` (settle-and-sample attempts per cell) | {} |\n\
         | Runs per aggregate | {} |\n\
         | Session span, hours | {} |\n\
         | Samples per quantity | {} under a cap of {} s per quantity, never below {} |",
        value("THRESHOLDS_VERSION")?,
        value("N_ATTEMPTS")?,
        value("MIN_RUNS")?,
        value("SESSION_MAX_HOURS")?,
        value("SAMPLES_TARGET")?,
        value("SAMPLE_CAP_SECS")?,
        value("SAMPLES_FLOOR")?
    ))
}

fn markers(name: &str) -> (String, String) {
    (
        format!("<!-- {name}:begin -->"),
        format!("<!-- {name}:end -->"),
    )
}

/// `text` with the lines between the `name` markers replaced by `body`.
fn replace_block(text: &str, name: &str, body: &str) -> Result<String, String> {
    let (begin, end) = markers(name);
    let start = text
        .find(&begin)
        .ok_or_else(|| format!("the page has no `{begin}` marker"))?
        + begin.len();
    let stop = text
        .find(&end)
        .ok_or_else(|| format!("the page has no `{end}` marker"))?;
    if stop < start {
        return Err(format!("the `{name}` markers are out of order"));
    }
    Ok(format!("{}\n{body}\n{}", &text[..start], &text[stop..]))
}

/// The text between the `name` markers, trimmed, `None` when a marker is missing.
fn block_of<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let (begin, end) = markers(name);
    let start = text.find(&begin)? + begin.len();
    let stop = text.find(&end)?;
    (stop >= start).then(|| text[start..stop].trim())
}

/// Problems with the guide pages that mirror the header (none when there are none).
fn check_mirrors(root: &Path) -> Result<Vec<String>, String> {
    let dir = root.join(MIRROR_DIR);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut problems = Vec::new();
    for path in sorted_entries(&dir)? {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with(MIRROR_PREFIX) || !name.ends_with(".md") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?
            .replace("\r\n", "\n");
        match block_of(&text, "bench-header") {
            Some(block) if block == HEADER.trim() => {}
            Some(_) => problems.push(format!(
                "{MIRROR_DIR}/{name}: its bench-header block differs from the canonical header (cargo xtask check-bench --render shows it in {DOCS_PAGE})"
            )),
            None => problems.push(format!(
                "{MIRROR_DIR}/{name}: no bench-header block; a page that shows benchmark results carries the canonical header"
            )),
        }
    }
    Ok(problems)
}

#[cfg(test)]
#[path = "tests/bench.rs"]
mod bench_tests;
