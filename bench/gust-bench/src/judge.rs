//! The publishable judgement (plan 2.4, items 1 to 3 and 5; item 4 is the thresholds below).
//!
//! A results file is publishable only when the harness says so, and the harness says so only when every check here
//! passes. Nothing in this module reads a clock, a process table or a git repository: it judges records (the sampler's
//! window, the run's source state, the cells) that other code gathered, so every rule is a plain function with a test.
//!
//! # Thresholds from data (item 4)
//!
//! The constants below come from the recorded sampler sidecars of the M10 premise runs (not in the repository): 9
//! sidecars from quiet-window runs (726 windows, 174 of them before a GPU group) and 17 from shared-machine runs (510
//! windows). `docs/BENCHMARKS.md`, "The thresholds", gives the data and the method. In short:
//!
//! - **The harness's own GPU use inside a window is 0.00 %** in all 1236 windows: the 3 s settle interval drains it, so
//!   the settle interval and the 2 s window at 4 Hz stay as the plan has them.
//! - **Other processes' GPU-engine use in a quiet window** (the largest per-process total over the 8 samples):
//!   median 0.44 %, p90 2.94 %, p95 5.25 %, p99 15.07 %, maximum 17.11 %. The large values are the System process
//!   (Copy engine) and `csrss`, bursts of a few windows, never sustained. The shared-machine runs show no more GPU use
//!   than that (maximum 15.51 %): **no foreign GPU load was ever recorded**, so the separation of noise from foreign
//!   load rests on persistence (retries), not on a measured foreign-load distribution.
//! - **N = 5** attempts per cell (settle, window, judge, repeat). Under the whole rule as implemented (the GPU rule,
//!   the named-process rule and the fail-closed rows) 87 of the 726 quiet windows fail (12.0 %), and a failing window
//!   is followed by another in 22 of 86 cases, so a window sequence (one per CPU cell, one per GPU group) fails all
//!   five attempts with probability about 5e-4. A full run takes 96 such sequences (120 with the CPU twin), so about
//!   5 (6) runs in 100 end not publishable and about 14 (17) aggregates of three runs in 100 contain one, on a machine
//!   where the monitoring scripts of the recording still ran (the recording held one sequence of five failing windows,
//!   where the model expects about 0.4). The quiet windows
//!   of a publishable session are taken with those scripts stopped. The figures come from the real `judge_window` (the
//!   ignored test `judge_recorded_sidecars` below) applied to the M10 premise runs' sidecars (not in the repository).
//! - **`T_cpu` is not set.** No `vmmem` process existed in any recorded window, and a window recorded with the machine's
//!   WSL service up (no distribution confirmed running) also listed none (only `wslservice` and `vmcompute`), so there
//!   is no distribution to set a growth threshold from, and whether the walk lists WSL's VM process by name is not
//!   established. The rule fails closed for what it does see: a
//!   `vmmem` or `vmmemWSL` process present in a window makes the window not quiet, whatever its CPU time does.
//!
//! The data cannot say how a real foreign GPU load looks to the counters; the one known load shape is the harness's own
//! `probe-adapters` busy loop, which reads up to 42 % on its pid. A change of any threshold is a CHANGELOG line of the
//! bench, and the tests below pin the values the documentation states.

use std::collections::BTreeMap;

use serde_json::{Value, json};

/// The version of the thresholds and of the rule they belong to (recorded in every run file and in each rendered
/// section). Any change to a constant below or to what `judge_window` counts as foreign or unjudgeable raises it.
pub const THRESHOLDS_VERSION: u32 = 1;
/// A process other than the harness is foreign when its GPU-engine use on the adapter exceeds this percentage in any
/// sample of the window. Above the median and p90 of the quiet distribution, at its p94: the ambient bursts above it are
/// handled by [`N_ATTEMPTS`], not by a larger threshold.
pub const T_GPU_PERCENT: f64 = 5.0;
/// The CPU-growth threshold of `vmmem` processes, as cpu-seconds per wall second. `None`: not set (no `vmmem` data).
pub const T_CPU_FRACTION: Option<f64> = None;
/// Settle-and-sample cycles per cell before the cell is marked not quiet.
pub const N_ATTEMPTS: u32 = 5;
/// Complete runs of one session an aggregate needs (plan 2.4 item 2).
pub const MIN_RUNS: usize = 3;
/// The longest span from the first run's start to the last run's end that counts as one session, in hours.
pub const SESSION_MAX_HOURS: f64 = 12.0;
/// Samples per quantity the method asks for (plan 2.3: 30 under a cap). A run that asks for fewer is not publishable.
pub const SAMPLES_TARGET: usize = 30;
/// The wall-clock cap per quantity, in seconds (plan 2.3): a quantity stops at this cap with whatever it has.
pub const SAMPLE_CAP_SECS: f64 = 8.0;
/// A quantity is never accepted with fewer collected samples than this (plan 2.3), whatever the cap.
pub const SAMPLES_FLOOR: usize = 5;

/// Process names (lower case, without `.exe`) whose CPU time growth across a window makes it not quiet (plan 2.4
/// item 3), and `vmmem` and `vmmemwsl`, which also count for any GPU-engine activity.
pub const NAMED_PROCESSES: [&str; 13] = [
    "cargo",
    "rustc",
    "rustdoc",
    "slangc",
    "gust-slang-reflect",
    "cargo-gust",
    "lms",
    "lm studio",
    "ollama",
    "python",
    "node",
    "vmmem",
    "vmmemwsl",
];

/// The thresholds in force, as one value (recorded in every run file and sidecar header).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    pub t_gpu_percent: f64,
    pub t_cpu_fraction: Option<f64>,
    pub attempts: u32,
}

/// The thresholds of this harness.
pub const THRESHOLDS: Thresholds = Thresholds {
    t_gpu_percent: T_GPU_PERCENT,
    t_cpu_fraction: T_CPU_FRACTION,
    attempts: N_ATTEMPTS,
};

impl Thresholds {
    pub fn json(&self) -> Value {
        json!({
            "version": THRESHOLDS_VERSION,
            "t_gpu_percent": self.t_gpu_percent,
            "t_cpu_cpu_s_per_wall_s": self.t_cpu_fraction,
            "t_cpu_note": if self.t_cpu_fraction.is_none() {
                "not set: no vmmem process in any recorded window; a vmmem process present makes a window not quiet"
            } else {
                "growth of a vmmem process above this fraction of one core makes a window not quiet"
            },
            "attempts_per_cell": self.attempts,
            "rule": "foreign = another process above t_gpu on the adapter in any sample, or a named process whose CPU time grew, started or ended across the window, or a vmmem process",
            "named_processes": NAMED_PROCESSES,
        })
    }
}

/// The verdict on one window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    pub quiet: bool,
    /// Why the window is not quiet (empty when it is).
    pub reasons: Vec<String>,
    /// The largest per-sample GPU-engine total of any one foreign process on the adapter, in percent (recorded
    /// whether or not it exceeds the threshold, so a sidecar shows the noise floor it was judged against).
    pub max_foreign_gpu_percent: f64,
}

impl Verdict {
    pub fn json(&self) -> Value {
        json!({
            "quiet": self.quiet,
            "reasons": self.reasons,
            "max_foreign_gpu_percent": self.max_foreign_gpu_percent,
        })
    }
}

/// One row of a recorded process table: `[pid, ppid, name, cpu_100ns, created_filetime]`.
struct Row {
    pid: u64,
    name: String,
    cpu_100ns: Option<u64>,
    created: Option<u64>,
}

fn rows(table: &Value) -> Option<Vec<Row>> {
    table
        .as_array()?
        .iter()
        .map(|row| {
            Some(Row {
                pid: row.get(0)?.as_u64()?,
                name: row.get(2)?.as_str()?.to_owned(),
                cpu_100ns: row.get(3).and_then(Value::as_u64),
                created: row.get(4).and_then(Value::as_u64),
            })
        })
        .collect()
}

/// The lower-case name without `.exe`.
fn stem(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    lower.strip_suffix(".exe").unwrap_or(&lower).to_owned()
}

fn is_named(name: &str) -> bool {
    NAMED_PROCESSES.contains(&stem(name).as_str())
}

/// The names the CPU rule and the GPU rule both treat as the virtual-machine memory process (one definition).
fn is_vmmem(name: &str) -> bool {
    matches!(stem(name).as_str(), "vmmem" | "vmmemwsl")
}

fn pids_of(list: &Value) -> Vec<u64> {
    list.as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["pid"].as_u64())
        .collect()
}

/// Judges one recorded window (the object `Sampler::sample_cell` returns) against `thresholds`. `expected_samples` is
/// the number of samples the window should hold (8 for the defaults). The judgement fails closed: a window that cannot
/// be read, a missing or short sample list, a truncated process table or an unknown ancestry is not quiet, with the
/// reason named.
pub fn judge_window(window: &Value, expected_samples: usize, thresholds: &Thresholds) -> Verdict {
    let mut reasons = Vec::new();
    let mut max_foreign = 0.0f64;
    if let Some(why) = window["unavailable"].as_str() {
        reasons.push(format!("cannot judge: {why}"));
        return Verdict {
            quiet: false,
            reasons,
            max_foreign_gpu_percent: 0.0,
        };
    }
    if let Some(error) = window["pdh"]["error"].as_str() {
        reasons.push(format!(
            "cannot judge: the GPU-engine counters failed: {error}"
        ));
    }
    let samples = window["pdh"]["samples"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if samples.len() < expected_samples || samples.is_empty() {
        reasons.push(format!(
            "cannot judge: {} GPU-engine samples, {expected_samples} expected",
            samples.len()
        ));
    }
    if let Some(bad) = samples.iter().find(|s| s["error"].is_string()) {
        reasons.push(format!(
            "cannot judge: a GPU-engine sample failed: {}",
            bad["error"].as_str().unwrap_or("?")
        ));
    }

    // PID ancestry: the harness, its shell-like parents and its own descendants are excluded; the shell's other
    // children (siblings) stay foreign. Without an ancestry nothing can be excluded, so the window is not judged.
    let ancestry = &window["ancestry"];
    let mut excluded = pids_of(&ancestry["chain"]);
    excluded.extend(pids_of(&ancestry["harness_descendants"]));
    if ancestry["error"].is_string() || excluded.is_empty() {
        reasons.push("cannot judge: the PID ancestry of the harness is unknown".to_owned());
    }

    let start = rows(&window["processes_start"]);
    let end = rows(&window["processes_end"]);
    for (label, table, truncated) in [
        ("start", &start, &window["processes_start_truncated"]),
        ("end", &end, &window["processes_end_truncated"]),
    ] {
        if table.is_none() {
            reasons.push(format!(
                "cannot judge: the process table at the {label} is unreadable"
            ));
        }
        if let Some(note) = truncated.as_str() {
            reasons.push(format!(
                "cannot judge: the process table at the {label} is incomplete: {note}"
            ));
        }
    }
    let name_of = |pid: u64| {
        start
            .iter()
            .chain(end.iter())
            .flatten()
            .find(|row| row.pid == pid)
            .map_or_else(|| "?".to_owned(), |row| row.name.clone())
    };

    // GPU: per sample, the sum over a process's engine instances on the adapter (every adapter when the LUID mapping
    // failed, which the sampler records as a null `adapter_luid`).
    let luid = window["adapter_luid"].as_str();
    let mut worst: BTreeMap<u64, f64> = BTreeMap::new();
    let (mut unreadable_rows, mut unparsed, mut invalid_here) = (0usize, 0usize, 0usize);
    for sample in &samples {
        let mut sums: BTreeMap<u64, f64> = BTreeMap::new();
        // Instances the sampler could not attribute are activity nobody judged: fail closed (plan 2.4 item 3).
        unparsed += sample["unparsed_instances"]
            .as_array()
            .map_or(0, std::vec::Vec::len);
        for entry in sample["invalid"].as_array().into_iter().flatten() {
            // An invalid reading is harmless when it names the harness or another adapter; anything else, or a name
            // that does not parse, could hide a foreign process.
            match entry["name"]
                .as_str()
                .and_then(crate::sampler::parse_instance)
            {
                Some(engine)
                    if excluded.contains(&u64::from(engine.pid))
                        || luid.is_some_and(|luid| engine.luid.text() != luid) => {}
                _ => invalid_here += 1,
            }
        }
        for engine in sample["engines"].as_array().into_iter().flatten() {
            let (Some(pid), Some(percent)) = (engine[0].as_u64(), engine[5].as_f64()) else {
                unreadable_rows += 1;
                continue;
            };
            if luid.is_some_and(|luid| engine[1].as_str() != Some(luid)) {
                continue;
            }
            *sums.entry(pid).or_default() += percent;
        }
        for (pid, sum) in sums {
            if excluded.contains(&pid) {
                continue;
            }
            let slot = worst.entry(pid).or_default();
            *slot = slot.max(sum);
        }
    }
    if unreadable_rows > 0 {
        reasons.push(format!(
            "cannot judge: {unreadable_rows} GPU-engine row(s) without a numeric pid or percent"
        ));
    }
    if unparsed > 0 {
        reasons.push(format!(
            "cannot judge: {unparsed} GPU-engine instance name(s) did not parse"
        ));
    }
    if invalid_here > 0 {
        reasons.push(format!(
            "cannot judge: {invalid_here} invalid GPU-engine reading(s) of the adapter under test or of an unparsed name"
        ));
    }
    for (pid, percent) in &worst {
        max_foreign = max_foreign.max(*percent);
        let name = name_of(*pid);
        if *percent > thresholds.t_gpu_percent {
            reasons.push(format!(
                "foreign GPU load: pid {pid} ({name}) used {percent:.1} % of its engines on the adapter, above T_gpu {} %",
                thresholds.t_gpu_percent
            ));
        } else if is_vmmem(&name) && *percent > 0.0 {
            reasons.push(format!(
                "foreign GPU activity: pid {pid} ({name}) shows {percent:.2} % on the adapter"
            ));
        }
    }

    // CPU: named processes across the window. Keyed by (pid, creation time), so a reused pid is a different process.
    if let (Some(start), Some(end)) = (&start, &end) {
        let wall = window["window_wall_s"].as_f64().unwrap_or(0.0).max(0.001);
        let key = |row: &Row| (row.pid, row.created);
        let in_start: BTreeMap<_, _> = start.iter().map(|row| (key(row), row)).collect();
        let in_end: BTreeMap<_, _> = end.iter().map(|row| (key(row), row)).collect();
        let is_excluded = |row: &Row| excluded.contains(&row.pid);
        for (k, before) in &in_start {
            if is_excluded(before) || !is_named(&before.name) {
                continue;
            }
            match in_end.get(k) {
                None => reasons.push(format!(
                    "named process {} (pid {}) ended during the window",
                    before.name, before.pid
                )),
                Some(after) => match (before.cpu_100ns, after.cpu_100ns) {
                    (Some(a), Some(b)) => {
                        let grown = b.saturating_sub(a) as f64 / 1.0e7;
                        if is_vmmem(&before.name) {
                            match thresholds.t_cpu_fraction {
                                None => reasons.push(format!(
                                    "vmmem process {} (pid {}) present and T_cpu is not set, so the window cannot be judged",
                                    before.name, before.pid
                                )),
                                Some(limit) if grown / wall > limit => reasons.push(format!(
                                    "vmmem process {} (pid {}) used {:.3} cpu-s per s, above T_cpu {limit}",
                                    before.name,
                                    before.pid,
                                    grown / wall
                                )),
                                Some(_) => {}
                            }
                        } else if grown > 0.0 {
                            reasons.push(format!(
                                "named process {} (pid {}) used {grown:.3} s of CPU in the window",
                                before.name, before.pid
                            ));
                        }
                    }
                    _ => reasons.push(format!(
                        "named process {} (pid {}) has no readable CPU time, so the window cannot be judged",
                        before.name, before.pid
                    )),
                },
            }
        }
        for (k, after) in &in_end {
            if !in_start.contains_key(k) && !is_excluded(after) && is_named(&after.name) {
                reasons.push(format!(
                    "named process {} (pid {}) started during the window",
                    after.name, after.pid
                ));
            }
        }
    }

    Verdict {
        quiet: reasons.is_empty(),
        reasons,
        max_foreign_gpu_percent: max_foreign,
    }
}

/// Settles, samples and judges up to `thresholds.attempts` times, stopping at the first quiet window (plan 2.4 item
/// 3: the harness waits for a window, it never makes one). `sample` takes one settle-and-window cycle. Returns every
/// window taken (each with its `attempt` number and its `verdict`, for the sidecar) and the cell's quiet record.
pub fn settle_until_quiet(
    thresholds: &Thresholds,
    expected_samples: usize,
    mut sample: impl FnMut() -> Value,
) -> (Vec<Value>, Value) {
    let mut windows = Vec::new();
    let mut earlier: Vec<Vec<String>> = Vec::new();
    let mut last = Verdict::default();
    for attempt in 1..=thresholds.attempts.max(1) {
        let mut window = sample();
        last = judge_window(&window, expected_samples, thresholds);
        if let Some(map) = window.as_object_mut() {
            map.insert("attempt".to_owned(), json!(attempt));
            map.insert("verdict".to_owned(), last.json());
        }
        windows.push(window);
        if last.quiet {
            break;
        }
        earlier.push(last.reasons.clone());
    }
    // A quiet window ended the loop before it was pushed; a last window that was not quiet was, and is the record's own
    // `reasons`, not an earlier attempt.
    if !last.quiet {
        earlier.pop();
    }
    let record = json!({
        "judged": true,
        "quiet": last.quiet,
        "attempts": windows.len(),
        "max_attempts": thresholds.attempts,
        "max_foreign_gpu_percent": last.max_foreign_gpu_percent,
        "reasons": last.reasons,
        "earlier_attempt_reasons": earlier,
    });
    (windows, record)
}

/// The quiet record of a run that took no windows (the sampler is off).
pub fn unjudged_record(why: &str) -> Value {
    json!({ "judged": false, "quiet": false, "attempts": 0, "reasons": [why] })
}

/// What the start of a run knows that decides whether any of its cells can be published (plan 2.4 items 1 and 2, and
/// what a table must be made of). Plain data, so the rule is testable without a repository or a GPU.
#[derive(Clone, Debug, Default)]
pub struct RunInputs {
    pub functional: bool,
    pub quick: bool,
    pub cold_only: bool,
    pub sizes_overridden: bool,
    pub samples: usize,
    pub group_is_publishable: bool,
    pub release_build: bool,
    /// The cargo profile the binary was built with (`meta::build_profile`): only `release` is publishable, because
    /// `functional` also has debug assertions off but links and optimises differently.
    pub cargo_profile: String,
    pub tainting_features: Vec<String>,
    pub probe_backends: bool,
    pub sampler_enabled: bool,
    pub software_adapter: bool,
    /// The flags of this run that change the measured method (settle, window, cap, warm-up) away from the defaults,
    /// each as the flag's name.
    pub overrides: Vec<String>,
    /// The commit the binary was compiled at (`meta::build_head`), `None` when unknown.
    pub build_head: Option<String>,
    /// Whether tracked files had uncommitted changes when the binary was compiled (`meta::build_dirty`), `None` when
    /// unknown.
    pub build_dirty: Option<bool>,
    pub head: Option<String>,
    pub tree_dirty: Option<bool>,
    pub ancestor_of_origin_main: Option<bool>,
}

/// The reasons no cell of this run can be publishable, from facts known before the first cell. Empty when none.
pub fn run_blockers(input: &RunInputs) -> Vec<String> {
    let mut reasons = Vec::new();
    if input.functional {
        reasons.push("functional run: correctness only, no timing is published".to_owned());
    }
    if input.quick {
        reasons.push("--quick run: reduced samples and sizes".to_owned());
    }
    if input.cold_only {
        reasons.push("--cold-only run: the timed quantities were not measured".to_owned());
    }
    if input.sizes_overridden {
        reasons.push("--sizes replaced the size ladder".to_owned());
    }
    if input.samples < SAMPLES_TARGET {
        reasons.push(format!(
            "{} samples per quantity requested, the method asks for {SAMPLES_TARGET}",
            input.samples
        ));
    }
    for flag in &input.overrides {
        reasons.push(format!(
            "`{flag}` differs from the default: the measured method is changed"
        ));
    }
    if !input.group_is_publishable {
        reasons.push("ablation group: method evidence, not a table".to_owned());
    }
    if !input.release_build {
        reasons.push("not a release build".to_owned());
    }
    if input.cargo_profile != "release" {
        reasons.push(format!(
            "built with the cargo profile `{}`, not `release`",
            input.cargo_profile
        ));
    }
    for feature in &input.tainting_features {
        reasons.push(format!("the `{feature}` feature changes what is measured"));
    }
    if input.probe_backends {
        reasons.push("probe backends (--with) are in the rotation".to_owned());
    }
    if !input.sampler_enabled {
        reasons.push("no sampler: the quiet window was not judged".to_owned());
    }
    if input.software_adapter {
        reasons.push("software adapter: times are never published".to_owned());
    }
    if input.head.is_none() {
        reasons.push("cannot read the commit (git failed)".to_owned());
    }
    match (&input.build_head, &input.head) {
        (None, _) => reasons.push("cannot tell which commit the binary was built at".to_owned()),
        (Some(built), Some(now)) if built != now => reasons.push(format!(
            "the binary was built at {} but HEAD is {}: rebuild at HEAD",
            built.chars().take(12).collect::<String>(),
            now.chars().take(12).collect::<String>()
        )),
        _ => {}
    }
    match input.build_dirty {
        Some(false) => {}
        Some(true) => reasons.push(
            "the binary was built from a tree with uncommitted changes: build from a clean tree"
                .to_owned(),
        ),
        None => {
            reasons.push("cannot tell whether the binary was built from a clean tree".to_owned())
        }
    }
    match input.tree_dirty {
        Some(false) => {}
        Some(true) => reasons.push("the tree is not clean (git status --porcelain)".to_owned()),
        None => reasons.push("cannot tell whether the tree is clean (git failed)".to_owned()),
    }
    match input.ancestor_of_origin_main {
        Some(true) => {}
        Some(false) => reasons.push("HEAD is not an ancestor of origin/main".to_owned()),
        None => reasons.push(
            "cannot tell whether HEAD is an ancestor of origin/main (no origin/main or git failed)"
                .to_owned(),
        ),
    }
    reasons
}

/// A cell's `publishable` flag and its reason. `quiet` is the quiet record of the window the cell ran after,
/// `blockers` the run-level reasons.
pub fn judge_cell(cell: &Value, quiet: &Value, blockers: &[String]) -> (bool, String) {
    let mut reasons = blockers.to_vec();
    if quiet["quiet"] != true {
        let first = quiet["reasons"][0].as_str().unwrap_or("no reason recorded");
        reasons.push(format!(
            "not quiet after {} attempt(s): {first}",
            quiet["attempts"]
        ));
    }
    if cell["failed"] != false {
        reasons.push("a correctness check or the cache assertion failed".to_owned());
    }
    for (metric, count) in collected_counts(cell) {
        if count < SAMPLES_FLOOR {
            reasons.push(format!(
                "{metric}: {count} samples collected, never fewer than {SAMPLES_FLOOR}"
            ));
        }
    }
    if cell["backend"].as_str().is_some_and(|id| id.contains('+')) {
        reasons.push("ablation or probe backend: not a table row".to_owned());
    }
    if reasons.is_empty() {
        (
            true,
            "quiet window held, every correctness check and the cache assertion passed".to_owned(),
        )
    } else {
        (false, reasons.join("; "))
    }
}

/// The samples collected per quantity of a cell (`metrics.<quantity>.n`), for the quantities it has.
pub fn collected_counts(cell: &Value) -> Vec<(String, usize)> {
    cell["metrics"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(metric, stats)| {
            Some((metric.clone(), usize::try_from(stats["n"].as_u64()?).ok()?))
        })
        .collect()
}

/// Writes the quiet record and the judgement into a cell, with the collected count of every quantity
/// (`samples_collected`; a quantity under [`SAMPLES_TARGET`] stopped at its cap, and the tables mark it).
pub fn stamp_cell(cell: &mut Value, quiet: &Value, blockers: &[String]) {
    let (publishable, reason) = judge_cell(cell, quiet, blockers);
    let counts = collected_counts(cell)
        .into_iter()
        .map(|(metric, count)| (metric, json!(count)))
        .collect::<serde_json::Map<_, _>>();
    if let Some(map) = cell.as_object_mut() {
        map.insert("samples_collected".to_owned(), Value::Object(counts));
        map.insert("quiet".to_owned(), quiet.clone());
        map.insert("publishable".to_owned(), json!(publishable));
        map.insert("publishable_reason".to_owned(), json!(reason));
    }
}

/// The run-level judgement: every cell publishable, nothing failed, and no source change during the run. `cells`
/// are already stamped. Returns the flag and every reason it is false.
pub fn judge_run(
    blockers: &[String],
    cells: &[Value],
    failures: &[String],
    source_changed: Option<&str>,
) -> (bool, Vec<String>) {
    let mut reasons = blockers.to_vec();
    if cells.is_empty() {
        reasons.push("the run has no cells".to_owned());
    }
    let bad = cells
        .iter()
        .filter(|cell| cell["publishable"] != true)
        .count();
    if bad > 0 {
        // The blockers are already on every cell; name only what is cell-specific.
        let first = cells
            .iter()
            .find(|cell| cell["publishable"] != true)
            .map(|cell| {
                format!(
                    "{} {} n={}: {}",
                    cell["backend"].as_str().unwrap_or("?"),
                    cell["workload"].as_str().unwrap_or("?"),
                    cell["n"],
                    cell["publishable_reason"].as_str().unwrap_or("?")
                )
            })
            .unwrap_or_default();
        reasons.push(format!(
            "{bad} of {} cells are not publishable (first: {first})",
            cells.len()
        ));
    }
    if !failures.is_empty() {
        reasons.push(format!("{} failure(s) in the run", failures.len()));
    }
    if let Some(change) = source_changed {
        reasons.push(change.to_owned());
    }
    (reasons.is_empty(), reasons)
}

/// Whether three or more runs make one publishable aggregate (plan 2.4 item 2): every run publishable, distinct seeds,
/// one session (first start to last end within [`SESSION_MAX_HOURS`]). Runs that disagree on route, build, adapter,
/// group or config never reach this point (`check_labels` refuses them). Returns the flag and every reason it is false.
pub fn judge_aggregate(runs: &[Value]) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();
    if runs.len() < MIN_RUNS {
        reasons.push(format!(
            "{} run(s), at least {MIN_RUNS} are needed for a table",
            runs.len()
        ));
    }
    for (index, run) in runs.iter().enumerate() {
        match run["publishable"].as_bool() {
            Some(true) => {}
            Some(false) => reasons.push(format!(
                "run {index} is not publishable: {}",
                run["publishable_reason"]
                    .as_str()
                    .unwrap_or("no reason recorded")
            )),
            None => reasons.push(format!(
                "run {index} has no publishable judgement (a run file from an older harness)"
            )),
        }
        if run["failures"].as_array().is_some_and(|f| !f.is_empty()) {
            reasons.push(format!("run {index} has failures"));
        }
    }
    let mut seeds = runs
        .iter()
        .filter_map(|run| run["run_seed"].as_u64())
        .collect::<Vec<_>>();
    seeds.sort_unstable();
    seeds.dedup();
    if seeds.len() < runs.len() {
        reasons.push("the runs do not all have distinct seeds".to_owned());
    }
    let starts = runs
        .iter()
        .filter_map(|run| run["window"]["started_unix_s"].as_u64());
    let ends = runs
        .iter()
        .filter_map(|run| run["window"]["ended_unix_s"].as_u64());
    match (starts.clone().min(), ends.clone().max()) {
        (Some(first), Some(last)) if starts.count() == runs.len() && ends.count() == runs.len() => {
            let hours = last.saturating_sub(first) as f64 / 3600.0;
            if hours > SESSION_MAX_HOURS {
                reasons.push(format!(
                    "the runs span {hours:.1} h, more than one session ({SESSION_MAX_HOURS} h)"
                ));
            }
        }
        _ => reasons.push("a run does not record its start and end time".to_owned()),
    }
    (reasons.is_empty(), reasons)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUID: &str = "0x00000000_0x00017E44";
    const OTHER_LUID: &str = "0x00000000_0x0001541E";

    /// A process row as the sidecar writes it.
    fn row(pid: u64, name: &str, cpu: Option<u64>, created: u64) -> Value {
        json!([pid, 1, name, cpu, created])
    }

    fn sample(engines: &[(u64, &str, f64)]) -> Value {
        json!({
            "t_ms": 250.0,
            "engines": engines.iter().map(|(pid, luid, pct)| json!([pid, luid, 0, 0, "3D", pct])).collect::<Vec<_>>(),
        })
    }

    /// A quiet window of the shape `sample_cell` writes: harness pid 100 under shell 50, a desktop around it.
    fn window() -> Value {
        let table = json!([
            row(4, "System", None, 1),
            row(50, "bash.exe", Some(1_000), 5),
            row(100, "gust-bench.exe", Some(5_000), 10),
            row(300, "dwm.exe", Some(9_000), 3),
            row(400, "python.exe", Some(70_000), 4),
            row(500, "node.exe", Some(80_000), 4),
        ]);
        json!({
            "cell": { "backend": "gpu-group", "workload": "W1", "n": 4096 },
            "window_wall_s": 2.05,
            "adapter_luid": LUID,
            "pdh": { "error": null, "samples": (0..8).map(|_| sample(&[(300, LUID, 0.4), (4, LUID, 1.0)])).collect::<Vec<_>>() },
            "processes_start": table.clone(),
            "processes_end": table,
            "processes_start_truncated": null,
            "processes_end_truncated": null,
            "ancestry": { "chain": [{ "pid": 100, "name": "gust-bench.exe" }, { "pid": 50, "name": "bash.exe" }],
                          "harness_descendants": [{ "pid": 101, "name": "nvidia-smi.exe" }], "siblings": [] },
        })
    }

    fn judged(window: &Value) -> Verdict {
        judge_window(window, 8, &THRESHOLDS)
    }

    #[test]
    fn the_thresholds_are_the_ones_the_documentation_states() {
        // docs/BENCHMARKS.md prints these; `cargo xtask check-bench --render` reads them from this file.
        assert_eq!(THRESHOLDS_VERSION, 1);
        assert_eq!(THRESHOLDS.json()["version"], 1);
        assert_eq!(T_GPU_PERCENT, 5.0);
        assert_eq!(T_CPU_FRACTION, None);
        assert_eq!(N_ATTEMPTS, 5);
        assert_eq!(MIN_RUNS, 3);
        assert_eq!(SAMPLES_TARGET, 30);
        assert_eq!(SAMPLE_CAP_SECS, 8.0);
        assert_eq!(SAMPLES_FLOOR, 5);
        assert_eq!(THRESHOLDS.attempts, N_ATTEMPTS);
        assert!(THRESHOLDS.json()["t_cpu_cpu_s_per_wall_s"].is_null());
    }

    #[test]
    fn a_quiet_desktop_is_quiet_and_its_noise_floor_is_recorded() {
        let verdict = judged(&window());
        assert!(verdict.quiet, "{:?}", verdict.reasons);
        assert_eq!(verdict.max_foreign_gpu_percent, 1.0);
    }

    #[test]
    fn a_foreign_process_above_t_gpu_in_any_one_sample_fails_the_window() {
        let mut w = window();
        w["pdh"]["samples"][5] = sample(&[(300, LUID, 5.0001)]);
        let verdict = judged(&w);
        assert!(!verdict.quiet);
        assert_eq!(verdict.max_foreign_gpu_percent, 5.0001);
        assert!(
            verdict.reasons[0].contains("pid 300 (dwm.exe)")
                && verdict.reasons[0].contains("5.0 %")
        );
        // Exactly at the threshold is not above it.
        w["pdh"]["samples"][5] = sample(&[(300, LUID, 5.0)]);
        assert!(judged(&w).quiet);
    }

    #[test]
    fn a_process_total_is_summed_over_its_engines_and_other_adapters_do_not_count() {
        let mut w = window();
        // Two engines of one process add up (3 + 3 > 5); 90 % on another adapter is not the adapter under test.
        w["pdh"]["samples"][0] =
            sample(&[(300, LUID, 3.0), (300, LUID, 3.0), (4, OTHER_LUID, 90.0)]);
        let verdict = judged(&w);
        assert!(!verdict.quiet);
        assert_eq!(verdict.reasons.len(), 1);
        assert!(verdict.reasons[0].contains("6.0 %"));
        // With the LUID mapping failed (a null adapter), every adapter counts: the stricter fallback.
        w["pdh"]["samples"][0] = sample(&[(4, OTHER_LUID, 90.0)]);
        assert!(judged(&w).quiet);
        w["adapter_luid"] = Value::Null;
        assert!(!judged(&w).quiet);
    }

    #[test]
    fn the_harness_its_shell_and_its_children_are_never_foreign_but_a_sibling_is() {
        let mut w = window();
        w["pdh"]["samples"][1] = sample(&[(100, LUID, 80.0), (50, LUID, 40.0), (101, LUID, 30.0)]);
        assert!(
            judged(&w).quiet,
            "the ancestry block must exclude the harness and its helpers"
        );
        // A process that is not in the ancestry at all, even a child of the shell, is foreign.
        w["pdh"]["samples"][1] = sample(&[(102, LUID, 30.0)]);
        assert!(!judged(&w).quiet);
    }

    #[test]
    fn a_named_process_whose_cpu_time_grew_is_foreign_and_an_idle_one_is_not() {
        let mut w = window();
        // python and node are in the table, idle: quiet (that is the base case). Grow python by one scheduler tick.
        let mut end = w["processes_end"].clone();
        end[4] = row(400, "python.exe", Some(70_000 + 156_000), 4);
        w["processes_end"] = end;
        let verdict = judged(&w);
        assert!(!verdict.quiet);
        assert!(
            verdict.reasons[0].contains("python.exe") && verdict.reasons[0].contains("0.016 s"),
            "{:?}",
            verdict.reasons
        );
        // The harness's own CPU growth is never counted: it is not a named process and it is excluded.
        let mut own = window();
        let mut end = own["processes_end"].clone();
        end[2] = row(100, "gust-bench.exe", Some(999_999_999), 10);
        own["processes_end"] = end;
        assert!(judged(&own).quiet);
    }

    #[test]
    fn a_named_process_that_starts_or_ends_in_the_window_is_foreign() {
        let mut w = window();
        let mut end = w["processes_end"].as_array().unwrap().clone();
        end.push(row(900, "rustc.exe", Some(0), 20));
        w["processes_end"] = Value::Array(end);
        let verdict = judged(&w);
        assert!(!verdict.quiet && verdict.reasons[0].contains("rustc.exe (pid 900) started"));
        let mut gone = window();
        let mut start = gone["processes_start"].as_array().unwrap().clone();
        start.push(row(901, "cargo.exe", Some(0), 20));
        gone["processes_start"] = Value::Array(start);
        let verdict = judged(&gone);
        assert!(!verdict.quiet && verdict.reasons[0].contains("cargo.exe (pid 901) ended"));
        // A pid reused by another process (a new creation time) is an end and a start, not an idle process.
        let mut reused = window();
        let mut end = reused["processes_end"].clone();
        end[4] = row(400, "python.exe", Some(0), 77);
        reused["processes_end"] = end;
        assert_eq!(judged(&reused).reasons.len(), 2);
    }

    #[test]
    fn a_named_process_without_a_readable_cpu_time_cannot_be_judged() {
        let mut w = window();
        let mut start = w["processes_start"].clone();
        start[5] = row(500, "node.exe", None, 4);
        w["processes_start"] = start;
        let verdict = judged(&w);
        assert!(!verdict.quiet && verdict.reasons[0].contains("no readable CPU time"));
    }

    #[test]
    fn name_matching_ignores_case_and_the_exe_suffix_and_knows_the_spaced_name() {
        for name in [
            "PYTHON.EXE",
            "Rustc",
            "LM Studio.exe",
            "slangc.exe",
            "gust-slang-reflect.exe",
            "lms.exe",
            "cargo-gust.exe",
            "ollama.exe",
            "rustdoc.exe",
        ] {
            assert!(is_named(name), "{name}");
        }
        for name in [
            "rustup.exe",
            "pythonw.exe",
            "dwm.exe",
            "chrome.exe",
            "nodemon.exe",
        ] {
            assert!(!is_named(name), "{name}");
        }
    }

    #[test]
    fn a_vmmem_process_fails_closed_while_t_cpu_is_not_set_and_is_judged_by_growth_once_it_is() {
        let mut w = window();
        let mut table = w["processes_start"].as_array().unwrap().clone();
        table.push(row(700, "vmmemWSL", Some(1_000_000), 6));
        w["processes_start"] = Value::Array(table.clone());
        w["processes_end"] = Value::Array(table.clone());
        let verdict = judged(&w);
        assert!(!verdict.quiet && verdict.reasons[0].contains("T_cpu is not set"));
        let with_limit = Thresholds {
            t_cpu_fraction: Some(0.02),
            ..THRESHOLDS
        };
        assert!(judge_window(&w, 8, &with_limit).quiet);
        // Growth of 0.1 cpu-s over 2.05 s is 0.049 cpu-s per s, above 0.02.
        let mut grown = table;
        grown[6] = row(700, "vmmemWSL", Some(1_000_000 + 1_000_000), 6);
        w["processes_end"] = Value::Array(grown);
        assert!(!judge_window(&w, 8, &with_limit).quiet);
        // Any GPU activity of a vmmem process counts, however small.
        let mut active = window();
        let mut table = active["processes_start"].as_array().unwrap().clone();
        table.push(row(700, "vmmem", Some(1_000_000), 6));
        active["processes_start"] = Value::Array(table.clone());
        active["processes_end"] = Value::Array(table);
        active["pdh"]["samples"][2] = sample(&[(700, LUID, 0.05)]);
        let verdict = judge_window(&active, 8, &with_limit);
        assert!(!verdict.quiet && verdict.reasons[0].contains("foreign GPU activity"));
    }

    #[test]
    fn a_window_that_cannot_be_read_is_never_quiet() {
        let mut short = window();
        short["pdh"]["samples"] = json!([sample(&[])]);
        assert!(judged(&short).reasons[0].contains("1 GPU-engine samples, 8 expected"));
        let mut errored = window();
        errored["pdh"]["samples"][3] =
            json!({ "t_ms": 1.0, "error": "PDH returned no GPU-engine instances" });
        assert!(!judged(&errored).quiet);
        let mut pdh = window();
        pdh["pdh"]["error"] = json!("PdhOpenQuery failed");
        assert!(!judged(&pdh).quiet);
        let mut table = window();
        table["processes_start"] = Value::Null;
        assert!(
            judged(&table)
                .reasons
                .iter()
                .any(|r| r.contains("unreadable"))
        );
        let mut truncated = window();
        truncated["processes_end_truncated"] = json!("process walk ended with HRESULT 0x80070005");
        assert!(
            judged(&truncated)
                .reasons
                .iter()
                .any(|r| r.contains("incomplete"))
        );
        let mut no_ancestry = window();
        no_ancestry["ancestry"] = json!({ "error": "process table unreadable" });
        assert!(
            judged(&no_ancestry)
                .reasons
                .iter()
                .any(|r| r.contains("ancestry"))
        );
        let off_windows =
            json!({ "cell": {}, "unavailable": "the GPU-engine counters exist on Windows only" });
        let verdict = judged(&off_windows);
        assert!(!verdict.quiet && verdict.reasons[0].contains("Windows only"));
        // A window the sampler never filled in at all is not quiet either.
        assert!(!judged(&json!({})).quiet);
    }

    #[test]
    fn the_harness_retries_up_to_n_times_and_stops_at_the_first_quiet_window() {
        let mut calls = 0;
        let (windows, record) = settle_until_quiet(&THRESHOLDS, 8, || {
            calls += 1;
            let mut w = window();
            if calls < 3 {
                w["pdh"]["samples"][0] = sample(&[(4, LUID, 14.5)]);
            }
            w
        });
        assert_eq!(calls, 3);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0]["attempt"], 1);
        assert_eq!(windows[0]["verdict"]["quiet"], false);
        assert_eq!(windows[2]["verdict"]["quiet"], true);
        assert_eq!(record["quiet"], true);
        assert_eq!(record["attempts"], 3);
        assert_eq!(
            record["earlier_attempt_reasons"].as_array().unwrap().len(),
            2
        );
        // Never quiet: exactly N attempts, then the cell is marked.
        let mut calls = 0;
        let (windows, record) = settle_until_quiet(&THRESHOLDS, 8, || {
            calls += 1;
            let mut w = window();
            w["pdh"]["samples"][0] = sample(&[(4, LUID, 14.5)]);
            w
        });
        assert_eq!(calls, N_ATTEMPTS);
        assert_eq!(windows.len(), N_ATTEMPTS as usize);
        assert_eq!(record["quiet"], false);
        assert_eq!(record["attempts"], N_ATTEMPTS);
        assert_eq!(
            record["earlier_attempt_reasons"].as_array().unwrap().len(),
            N_ATTEMPTS as usize - 1
        );
        assert!(
            record["reasons"][0]
                .as_str()
                .unwrap()
                .contains("foreign GPU load")
        );
    }

    fn clean_inputs() -> RunInputs {
        RunInputs {
            samples: 30,
            group_is_publishable: true,
            release_build: true,
            cargo_profile: "release".to_owned(),
            sampler_enabled: true,
            build_head: Some("0123456789abcdef".to_owned()),
            build_dirty: Some(false),
            head: Some("0123456789abcdef".to_owned()),
            tree_dirty: Some(false),
            ancestor_of_origin_main: Some(true),
            ..RunInputs::default()
        }
    }

    #[test]
    fn a_clean_release_run_on_a_pushed_commit_has_no_blockers_and_each_item_has_its_own_reason() {
        assert_eq!(run_blockers(&clean_inputs()), Vec::<String>::new());
        let cases: Vec<(&str, RunInputs)> = vec![
            (
                "functional",
                RunInputs {
                    functional: true,
                    ..clean_inputs()
                },
            ),
            (
                "--quick",
                RunInputs {
                    quick: true,
                    ..clean_inputs()
                },
            ),
            (
                "--cold-only",
                RunInputs {
                    cold_only: true,
                    ..clean_inputs()
                },
            ),
            (
                "--sizes",
                RunInputs {
                    sizes_overridden: true,
                    ..clean_inputs()
                },
            ),
            (
                "`--settle-secs`",
                RunInputs {
                    overrides: vec!["--settle-secs".to_owned()],
                    ..clean_inputs()
                },
            ),
            (
                "`--window-secs`",
                RunInputs {
                    overrides: vec!["--window-secs".to_owned()],
                    ..clean_inputs()
                },
            ),
            (
                "`--cap-secs`",
                RunInputs {
                    overrides: vec!["--cap-secs".to_owned()],
                    ..clean_inputs()
                },
            ),
            (
                "`--warmup-secs`",
                RunInputs {
                    overrides: vec!["--warmup-secs".to_owned()],
                    ..clean_inputs()
                },
            ),
            (
                "built at 0123456789ab but HEAD is fedcba987654",
                RunInputs {
                    head: Some("fedcba9876543210".to_owned()),
                    ..clean_inputs()
                },
            ),
            (
                "which commit the binary was built at",
                RunInputs {
                    build_head: None,
                    ..clean_inputs()
                },
            ),
            (
                "built from a tree with uncommitted changes",
                RunInputs {
                    build_dirty: Some(true),
                    ..clean_inputs()
                },
            ),
            (
                "whether the binary was built from a clean tree",
                RunInputs {
                    build_dirty: None,
                    ..clean_inputs()
                },
            ),
            (
                "29 samples",
                RunInputs {
                    samples: 29,
                    ..clean_inputs()
                },
            ),
            (
                "ablation",
                RunInputs {
                    group_is_publishable: false,
                    ..clean_inputs()
                },
            ),
            (
                "not a release",
                RunInputs {
                    release_build: false,
                    ..clean_inputs()
                },
            ),
            (
                "cargo profile `functional`, not `release`",
                RunInputs {
                    cargo_profile: "functional".to_owned(),
                    ..clean_inputs()
                },
            ),
            (
                "cargo profile `unknown`, not `release`",
                RunInputs {
                    cargo_profile: "unknown".to_owned(),
                    ..clean_inputs()
                },
            ),
            (
                "`profile-counts`",
                RunInputs {
                    tainting_features: vec!["profile-counts".to_owned()],
                    ..clean_inputs()
                },
            ),
            (
                "probe backends",
                RunInputs {
                    probe_backends: true,
                    ..clean_inputs()
                },
            ),
            (
                "no sampler",
                RunInputs {
                    sampler_enabled: false,
                    ..clean_inputs()
                },
            ),
            (
                "software adapter",
                RunInputs {
                    software_adapter: true,
                    ..clean_inputs()
                },
            ),
            (
                "cannot read the commit",
                RunInputs {
                    head: None,
                    ..clean_inputs()
                },
            ),
            (
                "tree is not clean",
                RunInputs {
                    tree_dirty: Some(true),
                    ..clean_inputs()
                },
            ),
            (
                "whether the tree is clean",
                RunInputs {
                    tree_dirty: None,
                    ..clean_inputs()
                },
            ),
            (
                "not an ancestor of origin/main",
                RunInputs {
                    ancestor_of_origin_main: Some(false),
                    ..clean_inputs()
                },
            ),
            (
                "whether HEAD is an ancestor",
                RunInputs {
                    ancestor_of_origin_main: None,
                    ..clean_inputs()
                },
            ),
        ];
        for (needle, input) in cases {
            let reasons = run_blockers(&input);
            assert_eq!(reasons.len(), 1, "{needle}: {reasons:?}");
            assert!(reasons[0].contains(needle), "{needle}: {reasons:?}");
        }
    }

    fn cell(backend: &str, failed: bool) -> Value {
        json!({ "backend": backend, "workload": "W1", "n": 4096, "failed": failed })
    }

    fn quiet_record(quiet: bool) -> Value {
        json!({ "judged": true, "quiet": quiet, "attempts": 5, "reasons": ["foreign GPU load: pid 4 (System)"] })
    }

    #[test]
    fn a_cell_is_publishable_only_when_quiet_correct_and_unblocked_and_says_why_not() {
        let (ok, reason) = judge_cell(&cell("gust-gpu", false), &quiet_record(true), &[]);
        assert!(ok && reason.contains("quiet window held"));
        let (ok, reason) = judge_cell(&cell("gust-gpu", false), &quiet_record(false), &[]);
        assert!(!ok && reason.contains("not quiet after 5 attempt(s): foreign GPU load"));
        let (ok, reason) = judge_cell(&cell("gust-gpu", true), &quiet_record(true), &[]);
        assert!(!ok && reason.contains("correctness check or the cache assertion"));
        let (ok, reason) = judge_cell(
            &cell("wgsl-percall+bind-group-reuse", false),
            &quiet_record(true),
            &[],
        );
        assert!(!ok && reason.contains("not a table row"));
        let (ok, reason) = judge_cell(
            &cell("gust-gpu", false),
            &quiet_record(true),
            &["the tree is not clean".to_owned()],
        );
        assert!(!ok && reason == "the tree is not clean");
        // A run without a sampler has an unjudged record: never publishable.
        let (ok, _) = judge_cell(&cell("rayon-N", false), &unjudged_record("no sampler"), &[]);
        assert!(!ok);
        // A cell with no `failed` field is not trusted.
        let (ok, _) = judge_cell(&json!({ "backend": "x" }), &quiet_record(true), &[]);
        assert!(!ok);
    }

    #[test]
    fn a_quantity_collected_below_the_floor_blocks_the_cell_and_a_short_one_is_only_recorded() {
        let mut c = cell("serial", false);
        c["metrics"] = json!({
            "end_to_end": { "n": 30 },
            "dispatch_host": { "n": 12 },
            "gpu_single": null,
        });
        let (ok, _) = judge_cell(&c, &quiet_record(true), &[]);
        assert!(ok, "12 samples under the cap are above the floor of 5");
        stamp_cell(&mut c, &quiet_record(true), &[]);
        assert_eq!(c["samples_collected"]["dispatch_host"], 12);
        assert!(c["samples_collected"].get("gpu_single").is_none());
        c["metrics"]["dispatch_host"]["n"] = json!(4);
        let (ok, reason) = judge_cell(&c, &quiet_record(true), &[]);
        assert!(
            !ok && reason.contains("dispatch_host: 4 samples collected, never fewer than 5"),
            "{reason}"
        );
    }

    #[test]
    fn unattributed_gpu_activity_fails_the_window_closed() {
        // An unparsed instance name.
        let mut w = window();
        w["pdh"]["samples"][3]["unparsed_instances"] = json!(["garbage"]);
        let v = judged(&w);
        assert!(
            !v.quiet && v.reasons.iter().any(|r| r.contains("did not parse")),
            "{:?}",
            v.reasons
        );
        // A row without a numeric percent (a NaN serialises as null).
        let mut w = window();
        w["pdh"]["samples"][0]["engines"] = json!([[4, LUID, 0, 0, "3D", null]]);
        let v = judged(&w);
        assert!(
            !v.quiet
                && v.reasons
                    .iter()
                    .any(|r| r.contains("without a numeric pid or percent"))
        );
        // An invalid reading of another process on the adapter under test, of an unparsed name.
        let on_adapter = "pid_4_luid_0x00000000_0x00017E44_phys_0_eng_0_engtype_3D";
        let mut w = window();
        w["pdh"]["samples"][1]["invalid"] = json!([{ "name": on_adapter, "pdh_status": 2 }]);
        assert!(!judged(&w).quiet);
        let mut w = window();
        w["pdh"]["samples"][1]["invalid"] = json!([{ "name": "nonsense", "pdh_status": 2 }]);
        assert!(!judged(&w).quiet);
        // Harmless: the harness's own pid, or another adapter.
        let own = "pid_100_luid_0x00000000_0x00017E44_phys_0_eng_0_engtype_3D";
        let other = "pid_4_luid_0x00000000_0x0001541E_phys_0_eng_0_engtype_3D";
        let mut w = window();
        w["pdh"]["samples"][1]["invalid"] =
            json!([{ "name": own, "pdh_status": 2 }, { "name": other, "pdh_status": 2 }]);
        assert!(judged(&w).quiet, "{:?}", judged(&w).reasons);
    }

    #[test]
    fn stamping_writes_the_record_the_flag_and_the_reason_into_the_cell() {
        let mut c = cell("rayon-1", false);
        stamp_cell(&mut c, &quiet_record(true), &[]);
        assert_eq!(c["publishable"], true);
        assert!(
            c["publishable_reason"]
                .as_str()
                .unwrap()
                .contains("quiet window held")
        );
        assert_eq!(c["quiet"]["attempts"], 5);
    }

    #[test]
    fn a_run_is_publishable_only_when_every_cell_is_and_nothing_failed_or_changed() {
        let mut good = cell("gust-gpu", false);
        stamp_cell(&mut good, &quiet_record(true), &[]);
        let mut bad = cell("rayon-1", false);
        stamp_cell(&mut bad, &quiet_record(false), &[]);
        assert_eq!(judge_run(&[], &[good.clone()], &[], None), (true, vec![]));
        let (ok, reasons) = judge_run(&[], &[good.clone(), bad], &[], None);
        assert!(
            !ok && reasons[0]
                .contains("1 of 2 cells are not publishable (first: rayon-1 W1 n=4096: not quiet")
        );
        assert!(!judge_run(&[], &[], &[], None).0);
        assert!(!judge_run(&[], &[good.clone()], &["x failed".to_owned()], None).0);
        let (ok, reasons) = judge_run(&[], &[good.clone()], &[], Some("HEAD moved during the run"));
        assert!(!ok && reasons == ["HEAD moved during the run"]);
        let (ok, reasons) = judge_run(&["the tree is not clean".to_owned()], &[good], &[], None);
        assert!(!ok && reasons == ["the tree is not clean"]);
    }

    fn run_file(seed: u64, start: u64, end: u64, publishable: Option<bool>) -> Value {
        let mut run = json!({
            "run_seed": seed, "failures": [],
            "window": { "started_unix_s": start, "ended_unix_s": end },
        });
        if let Some(flag) = publishable {
            run["publishable"] = json!(flag);
            run["publishable_reason"] = json!(if flag { "all good" } else { "not quiet" });
        }
        run
    }

    #[test]
    fn three_publishable_runs_of_distinct_seeds_in_one_session_make_a_publishable_aggregate() {
        let three = [
            run_file(1, 0, 1000, Some(true)),
            run_file(2, 1100, 2000, Some(true)),
            run_file(3, 2100, 3000, Some(true)),
        ];
        assert_eq!(judge_aggregate(&three), (true, vec![]));
        // Two runs are not a table.
        assert!(judge_aggregate(&three[..2]).1[0].contains("2 run(s)"));
        // One unpublishable run, a repeated seed, a different session, a run from an older harness.
        let mut one_bad = three.clone();
        one_bad[1] = run_file(2, 1100, 2000, Some(false));
        assert!(judge_aggregate(&one_bad).1[0].contains("run 1 is not publishable: not quiet"));
        let mut same_seed = three.clone();
        same_seed[2] = run_file(2, 2100, 3000, Some(true));
        assert!(judge_aggregate(&same_seed).1[0].contains("distinct seeds"));
        let mut late = three.clone();
        late[2] = run_file(3, 100_000, 100_900, Some(true));
        assert!(judge_aggregate(&late).1[0].contains("more than one session"));
        let mut old = three.clone();
        old[0] = json!({ "run_seed": 1, "failures": [] });
        let reasons = judge_aggregate(&old).1;
        assert!(
            reasons.iter().any(|r| r.contains("older harness"))
                && reasons.iter().any(|r| r.contains("start and end"))
        );
        let mut failing = three.clone();
        failing[0]["failures"] = json!(["W1 failed"]);
        assert!(judge_aggregate(&failing).1[0].contains("run 0 has failures"));
    }

    /// Records one sampler window (no GPU workload: the sampler alone, adapter mapping off so every adapter counts) and
    /// writes it, with the header, to the file named by `GUST_BENCH_WINDOW_OUT`; prints which `vmmem*` rows the process
    /// walk listed. Used once to see whether the walk lists WSL's VM process by name.
    #[test]
    #[ignore = "records one real sampler window, not a unit test"]
    fn record_one_sampler_window() {
        let out = std::env::var("GUST_BENCH_WINDOW_OUT").expect("GUST_BENCH_WINDOW_OUT");
        let sampler = crate::sampler::Sampler::open(crate::sampler::Timing::default(), None);
        let mut window =
            sampler.sample_cell(json!({ "backend": "sampler-only", "workload": "none", "n": 0 }));
        let verdict = judge_window(&window, 8, &THRESHOLDS);
        window["verdict"] = verdict.json();
        let listed = |key: &str| {
            window[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|row| row.get(2).and_then(Value::as_str))
                .filter(|name| name.to_ascii_lowercase().contains("vmmem"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        println!(
            "process rows at start {}, at end {}; vmmem rows at start {:?}, at end {:?}; quiet {}",
            window["processes_start"]
                .as_array()
                .map_or(0, std::vec::Vec::len),
            window["processes_end"]
                .as_array()
                .map_or(0, std::vec::Vec::len),
            listed("processes_start"),
            listed("processes_end"),
            verdict.quiet
        );
        let document = json!({ "header": sampler.header(), "windows": [window] });
        std::fs::write(out, serde_json::to_string(&document).unwrap()).unwrap();
    }

    /// The threshold analysis (not a test of behaviour): judges every window of the sidecars named in the
    /// `GUST_BENCH_SIDECARS` environment variable (paths separated by `;`) with the rule as implemented and prints one
    /// JSON line per sidecar: `{"file", "windows": [[is_gpu_group, quiet, [reason classes], max_foreign_gpu_percent]]}`.
    /// An analysis script over those lines is filed with the M10 plan, not in the repository. Run with
    /// `cargo test -p gust-bench judge_recorded_sidecars -- --ignored --nocapture`.
    #[test]
    #[ignore = "analysis over recorded sidecars, not a unit test"]
    fn judge_recorded_sidecars() {
        let list = std::env::var("GUST_BENCH_SIDECARS").expect("GUST_BENCH_SIDECARS");
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let document: Value =
                serde_json::from_str(&std::fs::read_to_string(path).expect("sidecar"))
                    .expect("json");
            let windows = document["windows"].as_array().expect("windows");
            let judged = windows
                .iter()
                .filter(|w| w.get("pdh").is_some())
                .map(|w| {
                    let verdict = judge_window(w, 8, &THRESHOLDS);
                    let mut classes = verdict
                        .reasons
                        .iter()
                        .map(|reason| {
                            [
                                "foreign GPU load",
                                "foreign GPU activity",
                                "named process",
                                "vmmem",
                                "cannot judge",
                            ]
                            .iter()
                            .find(|class| reason.starts_with(*class))
                            .map_or("other", |class| *class)
                        })
                        .collect::<Vec<_>>();
                    classes.sort_unstable();
                    classes.dedup();
                    json!([
                        w["cell"]["backend"] == "gpu-group",
                        verdict.quiet,
                        classes,
                        verdict.max_foreign_gpu_percent
                    ])
                })
                .collect::<Vec<_>>();
            println!("{}", json!({ "file": path, "windows": judged }));
        }
    }
}
