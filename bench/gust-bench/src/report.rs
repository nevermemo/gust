//! Aggregation of several run files into one JSON file and one Markdown report.

use std::{collections::BTreeMap, fmt::Write as _, fs};

use serde_json::{Value, json};

use crate::{backend::BenchResult, harness::median};

const BANNER: &str = "shared machine, not publishable";

/// Metric ids in report order, with a label and what each one is.
const METRICS: [(&str, &str); 4] = [
    (
        "end_to_end",
        "End to end (host inputs in, host output out; GPU: upload + dispatch + read-back)",
    ),
    (
        "dispatch_host",
        "Dispatch round trip, host wall clock (data resident, includes per-call host work)",
    ),
    (
        "gpu_single",
        "Dispatch only, GPU timestamps around one pass",
    ),
    (
        "gpu_batch_per_dispatch",
        "Dispatch only, GPU timestamps around a batch of passes, per dispatch",
    ),
];

/// Column order of the backends.
const BACKENDS: [&str; 15] = [
    "gust-gpu",
    "gust-gpu-spirv",
    "wgsl-prebuilt",
    "wgsl-percall",
    "wgsl-percall+bind-group-reuse",
    "wgsl-percall+staging-reuse",
    "wgsl-percall+one-submission",
    "wgsl-percall+all-on",
    "gust-twin",
    "gust-cpu",
    "gust-cpu-par",
    "rayon-1",
    "rayon-N",
    "serial",
    "(none)",
];

struct Point {
    per_run_median: Vec<f64>,
    samples: Vec<u64>,
}

impl Point {
    fn median(&self) -> f64 {
        median(&self.per_run_median)
    }

    fn min(&self) -> f64 {
        self.per_run_median.iter().copied().fold(f64::MAX, f64::min)
    }

    fn max(&self) -> f64 {
        self.per_run_median.iter().copied().fold(0.0, f64::max)
    }

    fn spread_pct(&self) -> f64 {
        (self.max() - self.min()) / self.median() * 100.0
    }
}

/// The paired ratios of one (workload, size, backend, metric) to `wgsl-prebuilt` over the runs: each run contributes the
/// median of its per-rotation ratios and the p5 and p95 of them (the spread inside the run).
#[derive(Default)]
struct PairedPoint {
    run_medians: Vec<f64>,
    run_p5: Vec<f64>,
    run_p95: Vec<f64>,
}

impl PairedPoint {
    fn median(&self) -> f64 {
        median(&self.run_medians)
    }

    fn lowest(&self) -> f64 {
        self.run_medians.iter().copied().fold(f64::MAX, f64::min)
    }

    fn highest(&self) -> f64 {
        self.run_medians.iter().copied().fold(0.0, f64::max)
    }
}

type PairedTable = BTreeMap<(String, u64, String, String), PairedPoint>;

fn gather_paired(runs: &[Value]) -> PairedTable {
    let mut table = PairedTable::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            let (Some(workload), Some(n), Some(backend)) = (
                cell["workload"].as_str(),
                cell["n"].as_u64(),
                cell["backend"].as_str(),
            ) else {
                continue;
            };
            for (metric, _) in METRICS {
                let paired = &cell["metrics"][metric]["paired_vs_prebuilt"];
                let Some(ratio) = paired["median_ratio"].as_f64() else {
                    continue;
                };
                let point = table
                    .entry((
                        workload.to_owned(),
                        n,
                        backend.to_owned(),
                        metric.to_owned(),
                    ))
                    .or_default();
                point.run_medians.push(ratio);
                point
                    .run_p5
                    .push(paired["p5_ratio"].as_f64().unwrap_or(ratio));
                point
                    .run_p95
                    .push(paired["p95_ratio"].as_f64().unwrap_or(ratio));
            }
        }
    }
    table
}

/// (workload, n, backend, metric) -> point
type Table = BTreeMap<(String, u64, String, String), Point>;

fn fmt_ms(value: f64) -> String {
    if value >= 100.0 {
        format!("{value:.1}")
    } else if value >= 1.0 {
        format!("{value:.3}")
    } else {
        format!("{value:.4}")
    }
}

fn load(paths: &[String]) -> BenchResult<Vec<Value>> {
    paths
        .iter()
        .map(|path| Ok(serde_json::from_str(&fs::read_to_string(path)?)?))
        .collect()
}

/// The labels every run must agree on (route, build with its commit, adapter, and the config: quick or full, sizes,
/// workloads, backends with their ablations): an aggregate of a WGSL and a SPIR-V run, or of two adapters, is not one
/// measurement, so a mismatch is an error rather than being labelled by the first run.
fn check_labels(runs: &[Value]) -> BenchResult<()> {
    let Some(first) = runs.first() else {
        return Err("no run files to aggregate".into());
    };
    for (index, run) in runs.iter().enumerate().skip(1) {
        for key in ["route", "build", "adapter", "config", "group"] {
            if run[key] != first[key] {
                return Err(format!(
                    "run {} disagrees with run 0 on `{key}`: {} against {}",
                    index, run[key], first[key]
                )
                .into());
            }
        }
    }
    Ok(())
}

fn gather(runs: &[Value]) -> Table {
    let mut table = Table::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            let (Some(workload), Some(n), Some(backend)) = (
                cell["workload"].as_str(),
                cell["n"].as_u64(),
                cell["backend"].as_str(),
            ) else {
                continue;
            };
            for (metric, _) in METRICS {
                let value = &cell["metrics"][metric];
                let Some(median) = value["median_ms"].as_f64() else {
                    continue;
                };
                let point = table
                    .entry((
                        workload.to_owned(),
                        n,
                        backend.to_owned(),
                        metric.to_owned(),
                    ))
                    .or_insert_with(|| Point {
                        per_run_median: Vec::new(),
                        samples: Vec::new(),
                    });
                point.per_run_median.push(median);
                point.samples.push(value["n"].as_u64().unwrap_or(0));
            }
        }
    }
    table
}

/// The cold-start values of one (workload, size, backend) over the runs. The first cold start of a process also pays
/// one-time costs (the Slang helper's compiler check, a cold driver cache in the first process), so it is kept apart
/// from the rest.
#[derive(Default)]
struct ColdPoint {
    /// One value per run whose cell was the first cold start of its backend in that process.
    process_first: Vec<f64>,
    rest: Vec<f64>,
    note: Option<String>,
}

impl ColdPoint {
    fn all(&self) -> Vec<f64> {
        self.process_first
            .iter()
            .chain(&self.rest)
            .copied()
            .collect()
    }
}

fn cold_starts(runs: &[Value]) -> BTreeMap<(String, u64, String), ColdPoint> {
    let mut cold: BTreeMap<_, ColdPoint> = BTreeMap::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            let Some(values) = cell["cold_start_ms"].as_array() else {
                continue;
            };
            let key = (
                cell["workload"].as_str().unwrap_or("").to_owned(),
                cell["n"].as_u64().unwrap_or(0),
                cell["backend"].as_str().unwrap_or("").to_owned(),
            );
            let point = cold.entry(key).or_default();
            let mut values = values.iter().filter_map(Value::as_f64);
            if cell["cold_start_first_in_process"].as_bool() == Some(true) {
                point.process_first.extend(values.next());
            }
            point.rest.extend(values);
            if let Some(note) = cell["cold_start_note"].as_str() {
                point.note = Some(note.to_owned());
            }
        }
    }
    cold
}

/// The clock readings of one (workload, size) over backends and runs.
#[derive(Default)]
struct ClockSpan {
    sm: Vec<f64>,
    pstates: std::collections::BTreeSet<String>,
    gen_now: Vec<f64>,
    gen_max: Vec<f64>,
    mem: Vec<f64>,
    after_pstates: std::collections::BTreeSet<String>,
    after_mem: Vec<f64>,
    cells: usize,
    p0_reached: usize,
}

/// The ablation order of the table: the per-call shape, each switch alone, all three on (result `Vec` still fresh),
/// the careful shape (which also reuses the `Vec`), and GUST for reference.
const ABLATION: [&str; 8] = [
    "wgsl-percall",
    "wgsl-percall+bind-group-reuse",
    "wgsl-percall+staging-reuse",
    "wgsl-percall+one-submission",
    "wgsl-percall+all-on",
    "wgsl-prebuilt",
    "gust-gpu",
    "gust-gpu-spirv",
];

/// One row per (workload, size, timing, backend of [`ABLATION`]): the median and its ratios to `wgsl-prebuilt` and
/// `wgsl-percall`, all on the same host shape of the same run set.
fn ablation_rows(table: &Table, sizes: &BTreeMap<String, Vec<u64>>) -> Vec<Value> {
    let mut rows = Vec::new();
    for (workload, list) in sizes {
        for n in list {
            for metric in ["end_to_end", "dispatch_host"] {
                let get = |backend: &str| {
                    table
                        .get(&(workload.clone(), *n, backend.to_owned(), metric.to_owned()))
                        .map(Point::median)
                };
                let (prebuilt, percall) = (get("wgsl-prebuilt"), get("wgsl-percall"));
                for backend in ABLATION {
                    let Some(ms) = get(backend) else { continue };
                    rows.push(json!({
                        "workload": workload, "n": n, "metric": metric, "backend": backend,
                        "median_ms": ms,
                        "ratio_vs_prebuilt": prebuilt.map(|p| ms / p),
                        "ratio_vs_percall": percall.map(|p| ms / p),
                    }));
                }
            }
        }
    }
    rows
}

/// The GPU clock state read while each GPU cell's time-based warm-up was still running, grouped by (workload, size,
/// backend) over the runs.
fn clock_rows(runs: &[Value]) -> Vec<Value> {
    type Key = (String, u64, String);
    let mut grouped: BTreeMap<Key, Vec<(&Value, &Value)>> = BTreeMap::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            if cell["warmup"]["clocks"].is_object() {
                let key = (
                    cell["workload"].as_str().unwrap_or("").to_owned(),
                    cell["n"].as_u64().unwrap_or(0),
                    cell["backend"].as_str().unwrap_or("").to_owned(),
                );
                grouped
                    .entry(key)
                    .or_default()
                    .push((&cell["warmup"], &cell["clocks_after_timing"]));
            }
        }
    }
    grouped
        .into_iter()
        .map(|((workload, n, backend), pairs)| {
            let warmups = pairs.iter().map(|p| p.0).collect::<Vec<_>>();
            let afters = pairs.iter().map(|p| p.1).collect::<Vec<_>>();
            let field = |name: &str| {
                warmups
                    .iter()
                    .map(|w| w["clocks"][name].clone())
                    .collect::<Vec<_>>()
            };
            json!({
                "workload": workload, "n": n, "backend": backend,
                "sm_mhz": field("sm_mhz"), "mem_mhz": field("mem_mhz"), "pstate": field("pstate"),
                "pcie_gen_now": field("pcie_gen_now"), "pcie_gen_max": field("pcie_gen_max"),
                "pcie_width": field("pcie_width"),
                "warmup_actual_s": warmups.iter().map(|w| w["actual_s"].clone()).collect::<Vec<_>>(),
                "p0_reached": warmups.iter().map(|w| w["p0_reached"].clone()).collect::<Vec<_>>(),
                "pstate_after_timing": afters.iter().map(|a| a["pstate"].clone()).collect::<Vec<_>>(),
                "sm_mhz_after_timing": afters.iter().map(|a| a["sm_mhz"].clone()).collect::<Vec<_>>(),
                "mem_mhz_after_timing": afters.iter().map(|a| a["mem_mhz"].clone()).collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// The PCIe link seen in the GPU cells of each (workload, size): generation now (and the maximum) and width, from the
/// clock readings at the end of the warm-up and after the timed samples. Printed next to every end-to-end number,
/// because the 5090 may sit on a narrower or slower link than the card supports.
fn link_map(runs: &[Value]) -> BTreeMap<(String, u64), String> {
    #[derive(Default)]
    struct Seen {
        gen_now: std::collections::BTreeSet<u64>,
        gen_max: std::collections::BTreeSet<u64>,
        width: std::collections::BTreeSet<u64>,
    }
    let mut seen: BTreeMap<(String, u64), Seen> = BTreeMap::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            let key = (
                cell["workload"].as_str().unwrap_or("").to_owned(),
                cell["n"].as_u64().unwrap_or(0),
            );
            for clocks in [&cell["warmup"]["clocks"], &cell["clocks_after_timing"]] {
                if !clocks.is_object() {
                    continue;
                }
                let slot = seen.entry(key.clone()).or_default();
                if let Some(v) = clocks["pcie_gen_now"].as_f64() {
                    slot.gen_now.insert(v as u64);
                }
                if let Some(v) = clocks["pcie_gen_max"].as_f64() {
                    slot.gen_max.insert(v as u64);
                }
                if let Some(v) = clocks["pcie_width"].as_f64() {
                    slot.width.insert(v as u64);
                }
            }
        }
    }
    let list = |set: &std::collections::BTreeSet<u64>| {
        if set.is_empty() {
            "?".to_owned()
        } else {
            set.iter().map(u64::to_string).collect::<Vec<_>>().join("/")
        }
    };
    seen.into_iter()
        .map(|(key, s)| {
            let text = format!(
                "gen {} (max {}) x{}",
                list(&s.gen_now),
                list(&s.gen_max),
                list(&s.width)
            );
            (key, text)
        })
        .collect()
}

/// The GUST GPU backend of these runs: `gust-gpu-spirv` in a `spirv-route` build, else `gust-gpu` (one build per
/// aggregate, see `check_labels`).
fn gust_backend(table: &Table) -> &'static str {
    if table
        .keys()
        .any(|(_, _, backend, _)| backend == "gust-gpu-spirv")
    {
        "gust-gpu-spirv"
    } else {
        "gust-gpu"
    }
}

/// What the cells say about a workload: its name and its bytes per element.
struct WorkloadInfo {
    name: String,
    bytes: f64,
}

fn workload_info(runs: &[Value]) -> BTreeMap<String, WorkloadInfo> {
    let mut info = BTreeMap::new();
    for run in runs {
        for cell in run["cells"].as_array().into_iter().flatten() {
            let Some(workload) = cell["workload"].as_str() else {
                continue;
            };
            info.entry(workload.to_owned())
                .or_insert_with(|| WorkloadInfo {
                    name: cell["workload_name"]
                        .as_str()
                        .unwrap_or(workload)
                        .to_owned(),
                    bytes: cell["bytes_per_element"].as_f64().unwrap_or(12.0),
                });
        }
    }
    info
}

/// The DRAM peak of an adapter in GB/s and where the figure comes from, for adapters this table knows. Unknown
/// adapters are not flagged (no peak is guessed).
pub fn dram_peak_gbps(adapter: &str) -> Option<(f64, &'static str)> {
    adapter.contains("RTX 5090").then_some((
        1792.0,
        "NVIDIA GeForce RTX 5090 specification: 512-bit GDDR7 at 28 Gbps, 1792 GB/s",
    ))
}

fn is_gpu_backend(id: &str) -> bool {
    id.starts_with("gust-gpu") || id.starts_with("wgsl-")
}

fn bandwidth_gbps(n: f64, bytes: f64, ms: f64) -> f64 {
    n * bytes / (ms * 1e-3) / 1e9
}

/// Every GPU-backend point whose achieved bandwidth (compulsory bytes over the median time) is above the DRAM `peak`.
/// Such a figure is not a DRAM bandwidth (the data sits in cache, or the time is not the kernel's alone), so the
/// report marks it instead of printing it as one (plan R1.2-2).
fn bandwidth_flags(table: &Table, info: &BTreeMap<String, WorkloadInfo>, peak: f64) -> Vec<Value> {
    let mut flags = Vec::new();
    for ((workload, n, backend, metric), point) in table {
        if !is_gpu_backend(backend) {
            continue;
        }
        let bytes = info.get(workload).map_or(12.0, |i| i.bytes);
        let gbps = bandwidth_gbps(*n as f64, bytes, point.median());
        if gbps > peak {
            flags.push(json!({
                "workload": workload, "n": n, "backend": backend, "metric": metric,
                "gb_per_s": gbps, "dram_peak_gb_per_s": peak,
            }));
        }
    }
    flags
}

pub fn aggregate(inputs: &[String], out_json: &str, out_md: &str) -> BenchResult<()> {
    let runs = load(inputs)?;
    check_labels(&runs)?;
    let table = gather(&runs);
    let paired = gather_paired(&runs);
    let cold = cold_starts(&runs);
    let info = workload_info(&runs);
    let gust = gust_backend(&table);
    let peak = dram_peak_gbps(runs[0]["adapter"]["name"].as_str().unwrap_or(""));
    let flags = peak.map_or_else(Vec::new, |(peak, _)| bandwidth_flags(&table, &info, peak));

    let mut sizes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (workload, n, _, _) in table.keys() {
        let list = sizes.entry(workload.clone()).or_default();
        if !list.contains(n) {
            list.push(*n);
        }
    }

    let mut failures = Vec::new();
    for run in &runs {
        failures.extend(
            run["failures"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|f| f.as_str().map(str::to_owned)),
        );
    }
    let checks: u64 = runs
        .iter()
        .flat_map(|run| run["cells"].as_array().into_iter().flatten())
        .filter_map(|cell| cell["checks_passed"].as_u64())
        .sum();

    let links = link_map(&runs);
    let link_of = |workload: &str, n: u64| {
        links
            .get(&(workload.to_owned(), n))
            .cloned()
            .unwrap_or_else(|| "?".to_owned())
    };

    // ---- JSON ----
    let points = table
        .iter()
        .map(|((workload, n, backend, metric), point)| {
            json!({
                "workload": workload, "n": n, "backend": backend, "metric": metric,
                "median_ms": point.median(), "min_run_median_ms": point.min(),
                "max_run_median_ms": point.max(), "spread_pct": point.spread_pct(),
                "per_run_median_ms": point.per_run_median, "samples_per_run": point.samples,
                "paired_vs_prebuilt": paired
                    .get(&(workload.clone(), *n, backend.clone(), metric.clone()))
                    .map(|p| json!({
                        "median_ratio": p.median(), "run_min_ratio": p.lowest(), "run_max_ratio": p.highest(),
                        "per_run_median_ratio": p.run_medians, "per_run_p5_ratio": p.run_p5,
                        "per_run_p95_ratio": p.run_p95,
                    })),
                "pcie_link": (metric == "end_to_end").then(|| link_of(workload, *n)),
            })
        })
        .collect::<Vec<_>>();
    let overhead = overhead_rows(&table, &sizes, gust);
    let cold_json = cold
        .iter()
        .map(|((workload, n, backend), point)| {
            json!({ "workload": workload, "n": n, "backend": backend,
                    "process_first_ms": point.process_first, "rest_ms": point.rest,
                    "median_rest_ms": (!point.rest.is_empty()).then(|| median(&point.rest)),
                    "all_ms": point.all(), "note": point.note })
        })
        .collect::<Vec<_>>();
    let ablation = ablation_rows(&table, &sizes);
    let clocks = clock_rows(&runs);
    let (gpu_cells, p0_cells) = runs
        .iter()
        .flat_map(|run| run["cells"].as_array().into_iter().flatten())
        .filter(|cell| cell["warmup"]["p0_reached"].is_boolean())
        .fold((0usize, 0usize), |(all, reached), cell| {
            (
                all + 1,
                reached + usize::from(cell["warmup"]["p0_reached"] == true),
            )
        });
    let summary = json!({
        "banner": BANNER,
        "publishable": crate::meta::PUBLISHABLE,
        "publishable_reason": crate::meta::PUBLISHABLE_REASON,
        "route": runs[0]["route"],
        "build": runs[0]["build"],
        "adapter": runs[0]["adapter"],
        "group": runs[0]["group"],
        "dram_peak": peak.map(|(gbps, source)| json!({ "gb_per_s": gbps, "source": source })),
        "above_dram_peak": flags,
        "meta": runs[0]["meta"],
        "config": runs[0]["config"],
        "runs": runs.iter().map(|run| json!({
            "label": run["label"], "group": run["group"], "kill_rule": run["kill_rule"], "run_seed": run["run_seed"], "sidecar": run["sampler"]["sidecar"], "load_before": run["load_before"], "load_after": run["load_after"],
            "gpu_before": run["gpu_before"], "gpu_after": run["gpu_after"],
            "wall_s": run["wall_s"], "failures": run["failures"],
        })).collect::<Vec<_>>(),
        "correctness_checks_passed": checks,
        "failures": failures,
        "points": points,
        "gust_vs_hand_overhead": overhead,
        "ablation": ablation,
        "clocks_at_timing_start": clocks,
        "p0": { "gpu_cells": gpu_cells, "p0_reached": p0_cells, "p0_not_reached": gpu_cells - p0_cells },
        "cold_start": cold_json,
    });
    fs::write(out_json, serde_json::to_string_pretty(&summary)? + "\n")?;

    // ---- Markdown ----
    let mut md = String::new();
    writeln!(md, "# M10 premise measurement results\n")?;
    writeln!(
        md,
        "**{BANNER}.** {} The conditions of each run are at the end.\n",
        crate::meta::PUBLISHABLE_REASON
    )?;
    writeln!(
        md,
        "Shader source of gust-gpu's cold start: {}. Warm-up with load to P0 before each GPU cell, capped at {} s.\n",
        runs[0]["config"]["shader_source"].as_str().unwrap_or("?"),
        runs[0]["config"]["warmup_cap_s"]
    )?;
    writeln!(
        md,
        "Each cell is the median over the per-run medians of {} runs; the bracket is the spread of the run medians (min to max). Every sample passed its correctness check ({checks} checks in all){}.\n",
        runs.len(),
        if failures.is_empty() {
            ""
        } else {
            ", EXCEPT the failures listed at the end"
        }
    )?;
    writeln!(
        md,
        "Group: {}; route: {}; GUST profile: {}; features: {}.\n",
        runs[0]["group"].as_str().unwrap_or("?"),
        runs[0]["route"].as_str().unwrap_or("?"),
        runs[0]["build"]["gust_profile"].as_str().unwrap_or("?"),
        runs[0]["build"]["features"]
    )?;
    for run in &runs {
        if run["kill_rule"].is_object() {
            writeln!(
                md,
                "- {}: kill rule, C re-opened = {} ({} paired GPU-time ratios checked, offenders: {})",
                run["label"].as_str().unwrap_or("run"),
                run["kill_rule"]["c_reopened"],
                run["kill_rule"]["checked"],
                run["kill_rule"]["offenders"]
            )?;
        }
    }
    writeln!(md)?;
    machine_block(&mut md, &runs[0]["meta"], &runs[0]["build"])?;
    for (workload, list) in &sizes {
        let name = info
            .get(workload)
            .map_or(workload.as_str(), |i| i.name.as_str());
        let bytes = info.get(workload).map_or(12.0, |i| i.bytes);
        writeln!(md, "\n## {workload} {name}\n")?;
        for (metric, label) in METRICS {
            writeln!(md, "### {label}\n")?;
            let columns = BACKENDS
                .iter()
                .copied()
                .filter(|b| {
                    list.iter().any(|n| {
                        table.contains_key(&(
                            workload.clone(),
                            *n,
                            (*b).to_owned(),
                            metric.to_owned(),
                        ))
                    })
                })
                .collect::<Vec<_>>();
            write!(md, "| elements |")?;
            for c in &columns {
                write!(md, " {c} (ms) |")?;
            }
            if metric == "end_to_end" {
                write!(md, " PCIe link |")?;
            }
            writeln!(md)?;
            write!(md, "|---|")?;
            for _ in &columns {
                write!(md, "---|")?;
            }
            if metric == "end_to_end" {
                write!(md, "---|")?;
            }
            writeln!(md)?;
            for n in list {
                write!(md, "| {n} |")?;
                for c in &columns {
                    match table.get(&(workload.clone(), *n, (*c).to_owned(), metric.to_owned())) {
                        Some(p) => write!(
                            md,
                            " {} [{}-{}] |",
                            fmt_ms(p.median()),
                            fmt_ms(p.min()),
                            fmt_ms(p.max())
                        )?,
                        None => write!(md, " - |")?,
                    }
                }
                if metric == "end_to_end" {
                    write!(md, " {} |", link_of(workload, *n))?;
                }
                writeln!(md)?;
            }
            writeln!(md)?;
        }

        writeln!(
            md,
            "### Achieved bandwidth (GB/s, {bytes} bytes per element), from the batch timestamps and from end to end\n"
        )?;
        writeln!(
            md,
            "| elements | {gust} batch | wgsl-prebuilt batch | wgsl-percall batch | {gust} e2e | wgsl-prebuilt e2e | rayon-N e2e | PCIe link (GPU e2e) |"
        )?;
        writeln!(md, "|---|---|---|---|---|---|---|---|")?;
        for n in list {
            let gbps = |backend: &str, metric: &str| {
                table
                    .get(&(workload.clone(), *n, backend.to_owned(), metric.to_owned()))
                    .map_or("-".to_owned(), |p| {
                        let gbps = bandwidth_gbps(*n as f64, bytes, p.median());
                        let text = if gbps < 10.0 {
                            format!("{gbps:.2}")
                        } else {
                            format!("{gbps:.0}")
                        };
                        if is_gpu_backend(backend) && peak.is_some_and(|(peak, _)| gbps > peak) {
                            format!("{text} (>peak)")
                        } else {
                            text
                        }
                    })
            };
            writeln!(
                md,
                "| {n} | {} | {} | {} | {} | {} | {} | {} |",
                gbps(gust, "gpu_batch_per_dispatch"),
                gbps("wgsl-prebuilt", "gpu_batch_per_dispatch"),
                gbps("wgsl-percall", "gpu_batch_per_dispatch"),
                gbps(gust, "end_to_end"),
                gbps("wgsl-prebuilt", "end_to_end"),
                gbps("rayon-N", "end_to_end"),
                link_of(workload, *n),
            )?;
        }
        let flagged = flags.iter().filter(|f| f["workload"] == *workload).count();
        if let (true, Some((peak, source))) = (flagged > 0, peak) {
            writeln!(
                md,
                "{flagged} points report more than the adapter's DRAM peak of {peak:.0} GB/s ({source}); the batch and end-to-end cells above are marked `(>peak)`. Such a figure is not a DRAM bandwidth: the data sits in cache, or the timing is not the kernel alone.\n"
            )?;
        }
        writeln!(md)?;

        crossover(&mut md, &table, workload, list, &link_of, gust)?;
    }

    writeln!(
        md,
        "\n## GUST against hand WGSL (ratio of medians; above 1.00 means GUST is slower)\n"
    )?;
    writeln!(
        md,
        "| workload | elements | timing | gust-gpu (ms) | wgsl-prebuilt (ms) | ratio | difference (us) | wgsl-percall (ms) | ratio | PCIe link (end to end) |"
    )?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|")?;
    for row in &overhead {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} | {:.3} | {:+.1} | {} | {} | {} |",
            row["workload"].as_str().unwrap_or(""),
            row["n"],
            row["metric"].as_str().unwrap_or(""),
            fmt_ms(row["gust_ms"].as_f64().unwrap_or(0.0)),
            fmt_ms(row["prebuilt_ms"].as_f64().unwrap_or(0.0)),
            row["ratio_vs_prebuilt"].as_f64().unwrap_or(0.0),
            row["diff_us_vs_prebuilt"].as_f64().unwrap_or(0.0),
            row["percall_ms"].as_f64().map_or("-".to_owned(), fmt_ms),
            row["ratio_vs_percall"]
                .as_f64()
                .map_or("-".to_owned(), |r| format!("{r:.3}")),
            if row["metric"] == "end_to_end" {
                link_of(
                    row["workload"].as_str().unwrap_or(""),
                    row["n"].as_u64().unwrap_or(0),
                )
            } else {
                "-".to_owned()
            },
        )?;
    }

    // Paired ratios: the GPU backends were sampled in rotation, so these ratios saw the same power state.
    writeln!(
        md,
        "\n## Paired ratios to wgsl-prebuilt (GPU backends sampled in interleaved random-order rounds, one shared warm-up)\n"
    )?;
    writeln!(
        md,
        "Each run takes the samples of all GPU backends of a (workload, size) in interleaved rounds, each round in a seeded random order (the run seed, every round's order and every sample's predecessor are in the run file), so a P-state drift hits every backend alike and no backend always follows the same neighbour. The ratio of a rotation is the backend's sample over wgsl-prebuilt's sample of the same rotation; a cell shows the median over runs of the per-run median ratio, then [lowest to highest run median] and (mean of the within-run p5 to p95). A ratio of 1.00 means no gap to wgsl-prebuilt.\n"
    )?;
    for (metric, _) in METRICS {
        for workload in sizes.keys() {
            writeln!(md, "### {workload}, {metric}\n")?;
            write!(md, "| elements |")?;
            for backend in ABLATION {
                write!(md, " {backend} |")?;
            }
            writeln!(md)?;
            write!(md, "|---|")?;
            for _ in ABLATION {
                write!(md, "---|")?;
            }
            writeln!(md)?;
            for n in &sizes[workload] {
                write!(md, "| {n} |")?;
                for backend in ABLATION {
                    match paired.get(&(workload.clone(), *n, backend.to_owned(), metric.to_owned()))
                    {
                        Some(p) => {
                            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
                            write!(
                                md,
                                " {:.3} [{:.3}-{:.3}] ({:.2}-{:.2}) |",
                                p.median(),
                                p.lowest(),
                                p.highest(),
                                mean(&p.run_p5),
                                mean(&p.run_p95)
                            )?;
                        }
                        None => write!(md, " - |")?,
                    }
                }
                writeln!(md)?;
            }
            writeln!(md)?;
        }
    }

    // The ablation table: medians of the per-run medians; each cell shows the ratio to wgsl-prebuilt.
    writeln!(
        md,
        "\n## Ablation (hand WGSL host shape; median ms, ratio to wgsl-prebuilt in brackets)\n"
    )?;
    writeln!(
        md,
        "Rows: `wgsl-percall` (every switch off), each switch alone, `wgsl-percall+all-on` (all three on, result `Vec` still fresh per call), `wgsl-prebuilt` (all three on and the `Vec` reused), `gust-gpu`. The step from `+all-on` to `wgsl-prebuilt` is the result `Vec`.\n"
    )?;
    for metric in ["end_to_end", "dispatch_host"] {
        for workload in sizes.keys() {
            writeln!(md, "### {workload}, {metric}\n")?;
            write!(md, "| elements |")?;
            for backend in ABLATION {
                write!(md, " {backend} |")?;
            }
            if metric == "end_to_end" {
                write!(md, " PCIe link |")?;
            }
            writeln!(md)?;
            write!(md, "|---|")?;
            for _ in ABLATION {
                write!(md, "---|")?;
            }
            if metric == "end_to_end" {
                write!(md, "---|")?;
            }
            writeln!(md)?;
            for n in &sizes[workload] {
                write!(md, "| {n} |")?;
                for backend in ABLATION {
                    let row = ablation.iter().find(|r| {
                        r["workload"] == *workload
                            && r["n"] == *n
                            && r["metric"] == metric
                            && r["backend"] == backend
                    });
                    match row {
                        Some(r) => write!(
                            md,
                            " {} ({}) |",
                            fmt_ms(r["median_ms"].as_f64().unwrap_or(0.0)),
                            r["ratio_vs_prebuilt"]
                                .as_f64()
                                .map_or("-".to_owned(), |x| format!("{x:.2}x"))
                        )?,
                        None => write!(md, " - |")?,
                    }
                }
                if metric == "end_to_end" {
                    write!(md, " {} |", link_of(workload, *n))?;
                }
                writeln!(md)?;
            }
            writeln!(md)?;
        }
    }

    // GPU clocks at the start of timing: per (workload, size) the range over backends and runs.
    writeln!(
        md,
        "\n## GPU clocks at the start of timing and after it (P0 warm-up)\n"
    )?;
    writeln!(
        md,
        "Start: the last clock reading of the warm-up (real load until P0, capped). After: read right after the timed samples. A cell that hit the cap is marked `p0_not_reached`, not dropped: {p0_cells} of {gpu_cells} GPU cells reached P0.\n"
    )?;
    writeln!(
        md,
        "| workload | elements | cells reaching P0 | SM MHz at start (min to max) | memory MHz at start | pstates at start | pstates after | memory MHz after | PCIe gen now (max gen) |"
    )?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|")?;
    let mut by_size: BTreeMap<(String, u64), ClockSpan> = BTreeMap::new();
    for row in &clocks {
        let key = (
            row["workload"].as_str().unwrap_or("").to_owned(),
            row["n"].as_u64().unwrap_or(0),
        );
        let slot = by_size.entry(key).or_default();
        let numbers = |name: &str| {
            row[name]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
                .collect::<Vec<_>>()
        };
        slot.sm.extend(numbers("sm_mhz"));
        slot.mem.extend(numbers("mem_mhz"));
        slot.after_mem.extend(numbers("mem_mhz_after_timing"));
        for pstate in row["pstate_after_timing"].as_array().into_iter().flatten() {
            slot.after_pstates
                .insert(pstate.as_str().unwrap_or("?").to_owned());
        }
        for flag in row["p0_reached"].as_array().into_iter().flatten() {
            slot.cells += 1;
            slot.p0_reached += usize::from(*flag == true);
        }
        for pstate in row["pstate"].as_array().into_iter().flatten() {
            slot.pstates
                .insert(pstate.as_str().unwrap_or("?").to_owned());
        }
        slot.gen_now.extend(numbers("pcie_gen_now"));
        slot.gen_max.extend(numbers("pcie_gen_max"));
    }
    for ((workload, n), spans) in &by_size {
        let (sm, gen_now, gen_max) = (&spans.sm, &spans.gen_now, &spans.gen_max);
        let span = |v: &[f64]| {
            if v.is_empty() {
                "?".to_owned()
            } else {
                let low = v.iter().copied().fold(f64::MAX, f64::min);
                let high = v.iter().copied().fold(f64::MIN, f64::max);
                format!("{low:.0} to {high:.0}")
            }
        };
        writeln!(
            md,
            "| {workload} | {n} | {}/{} | {} | {} | {} | {} | {} | {} ({}) |",
            spans.p0_reached,
            spans.cells,
            span(sm),
            span(&spans.mem),
            spans.pstates.iter().cloned().collect::<Vec<_>>().join(", "),
            spans
                .after_pstates
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            span(&spans.after_mem),
            span(gen_now),
            span(gen_max)
        )?;
    }

    writeln!(
        md,
        "\n## Cold start (pipeline state dropped, first dispatch timed)\n"
    )?;
    writeln!(
        md,
        "The first cold start of each kernel on a backend in a process is listed apart from the rest (it also pays one-time costs). Hand-WGSL rows are labelled by their note: the driver's shader cache is warm for them.\n"
    )?;
    writeln!(
        md,
        "| workload | elements | backend | first in process, per run (ms) | rest median (ms) | rest (ms) | note |"
    )?;
    writeln!(md, "|---|---|---|---|---|---|---|")?;
    let list = |values: &[f64]| {
        if values.is_empty() {
            "-".to_owned()
        } else {
            values
                .iter()
                .map(|v| fmt_ms(*v))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };
    for ((workload, n, backend), point) in &cold {
        writeln!(
            md,
            "| {workload} | {n} | {backend} | {} | {} | {} | {} |",
            list(&point.process_first),
            if point.rest.is_empty() {
                "-".to_owned()
            } else {
                fmt_ms(median(&point.rest))
            },
            list(&point.rest),
            point.note.as_deref().unwrap_or("")
        )?;
    }

    writeln!(md, "\n## Run conditions\n")?;
    for run in &runs {
        writeln!(
            md,
            "- {}: GPU before `{}`, after `{}`; load before `{}`, after `{}`; wall {:.0} s",
            run["label"].as_str().unwrap_or("run"),
            run["gpu_before"],
            run["gpu_after"],
            run["load_before"]["process_counts"],
            run["load_after"]["process_counts"],
            run["wall_s"].as_f64().unwrap_or(0.0)
        )?;
        let (mut util, mut sm, mut mem_used) = (Vec::new(), Vec::new(), Vec::new());
        for cell in run["cells"].as_array().into_iter().flatten() {
            for key in ["gpu_before", "gpu_after"] {
                let snapshot = &cell[key];
                util.extend(snapshot["utilization_pct"].as_f64());
                sm.extend(snapshot["sm_mhz"].as_f64());
                mem_used.extend(snapshot["mem_used_mib"].as_f64());
            }
        }
        let range = |v: &[f64]| {
            let low = v.iter().copied().fold(f64::MAX, f64::min);
            let high = v.iter().copied().fold(f64::MIN, f64::max);
            format!("{low:.0}-{high:.0}")
        };
        if !util.is_empty() {
            writeln!(
                md,
                "  - per-cell snapshots taken between cells (GPU idle by construction): utilization % {}, SM clock MHz {}, memory used MiB {}",
                range(&util),
                range(&sm),
                range(&mem_used)
            )?;
        }
    }
    if !failures.is_empty() {
        writeln!(md, "\n## Failures\n")?;
        for failure in &failures {
            writeln!(md, "- {failure}")?;
        }
    }
    fs::write(out_md, md)?;
    Ok(())
}

fn machine_block(md: &mut String, meta: &Value, build: &Value) -> BenchResult<()> {
    writeln!(md, "| | |\n|---|---|")?;
    writeln!(
        md,
        "| adapter | {} ({}, {}) |",
        meta["adapter"]["name"].as_str().unwrap_or("?"),
        meta["adapter"]["device_type"].as_str().unwrap_or("?"),
        meta["adapter"]["backend"].as_str().unwrap_or("?")
    )?;
    writeln!(
        md,
        "| driver | {} {} |",
        meta["adapter"]["driver"].as_str().unwrap_or("?"),
        meta["adapter"]["driver_info"].as_str().unwrap_or("")
    )?;
    writeln!(md, "| OS | {} |", meta["os"].as_str().unwrap_or("?"))?;
    writeln!(
        md,
        "| CPU | {} ({} logical cores, rayon-N = {} threads) |",
        meta["cpu"].as_str().unwrap_or("?"),
        meta["logical_cores"],
        meta["rayon_threads_n"]
    )?;
    writeln!(
        md,
        "| power plan | {} |",
        meta["power_plan"].as_str().unwrap_or("?")
    )?;
    writeln!(
        md,
        "| versions | rustc `{}`, wgpu {}, rayon {} |",
        meta["versions"]["rustc"].as_str().unwrap_or("?"),
        meta["versions"]["wgpu"].as_str().unwrap_or("?"),
        meta["versions"]["rayon"].as_str().unwrap_or("?"),
    )?;
    writeln!(
        md,
        "| commit | {} (tree dirty: {}) |",
        build["commit"].as_str().unwrap_or("?"),
        build["tree_dirty"]
    )?;
    writeln!(
        md,
        "| build | {} profile, gust_dual = {}, GPU timestamps = {} |",
        meta["build"]["profile"].as_str().unwrap_or("?"),
        meta["build"]["gust_dual"],
        meta["adapter"]["gpu_timestamps"]
    )?;
    Ok(())
}

/// The smallest size where the GPU's end-to-end median beats rayon-N's, per GPU backend.
fn crossover(
    md: &mut String,
    table: &Table,
    workload: &str,
    sizes: &[u64],
    link_of: &dyn Fn(&str, u64) -> String,
    gust: &str,
) -> BenchResult<()> {
    let get = |n: u64, backend: &str| {
        table
            .get(&(
                workload.to_owned(),
                n,
                backend.to_owned(),
                "end_to_end".to_owned(),
            ))
            .map(Point::median)
    };
    for gpu in [gust, "wgsl-prebuilt"] {
        let winner = sizes
            .iter()
            .copied()
            .find(|n| matches!((get(*n, gpu), get(*n, "rayon-N")), (Some(g), Some(c)) if g < c));
        match winner {
            Some(n) => writeln!(
                md,
                "End-to-end break-even, {gpu} against rayon-N: first faster at {n} elements (PCIe link {}).",
                link_of(workload, n)
            )?,
            None => writeln!(
                md,
                "End-to-end break-even, {gpu} against rayon-N: never faster on this ladder (PCIe link {}).",
                sizes
                    .last()
                    .map_or("?".to_owned(), |n| link_of(workload, *n))
            )?,
        }
    }
    writeln!(md)?;
    Ok(())
}

fn overhead_rows(table: &Table, sizes: &BTreeMap<String, Vec<u64>>, gust: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    for (workload, list) in sizes {
        for n in list {
            for (metric, _) in METRICS.iter().rev() {
                let get = |backend: &str| {
                    table
                        .get(&(
                            workload.clone(),
                            *n,
                            backend.to_owned(),
                            (*metric).to_owned(),
                        ))
                        .map(Point::median)
                };
                let (Some(gust), Some(prebuilt)) = (get(gust), get("wgsl-prebuilt")) else {
                    continue;
                };
                let percall = get("wgsl-percall");
                rows.push(json!({
                    "workload": workload, "n": n, "metric": metric,
                    "gust_ms": gust, "prebuilt_ms": prebuilt, "percall_ms": percall,
                    "ratio_vs_prebuilt": gust / prebuilt,
                    "diff_us_vs_prebuilt": (gust - prebuilt) * 1000.0,
                    "ratio_vs_percall": percall.map(|p| gust / p),
                    "diff_us_vs_percall": percall.map(|p| (gust - p) * 1000.0),
                }));
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(route: &str, adapter: &str, sidecar: &str) -> Value {
        json!({
            "label": "t", "route": route, "build": { "profile": "release" },
            "adapter": { "name": adapter }, "meta": {}, "config": {},
            "sampler": { "enabled": true, "sidecar": sidecar },
            "cells": [], "failures": [],
        })
    }

    fn write_runs(dir: &std::path::Path, runs: &[Value]) -> Vec<String> {
        runs.iter()
            .enumerate()
            .map(|(index, value)| {
                let path = dir.join(format!("run{index}.json"));
                fs::write(&path, value.to_string()).unwrap();
                path.to_str().unwrap().to_owned()
            })
            .collect()
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("gust-bench-report-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_label_mismatch_between_runs_is_an_error() {
        let same = [
            run("wgsl", "GPU A", "a.json"),
            run("wgsl", "GPU A", "b.json"),
        ];
        assert!(check_labels(&same).is_ok());
        for other in [
            run("spirv", "GPU A", "b.json"),
            run("wgsl", "GPU B", "b.json"),
        ] {
            let error = check_labels(&[run("wgsl", "GPU A", "a.json"), other]).unwrap_err();
            assert!(error.to_string().contains("disagrees"), "{error}");
        }
        let mut build = run("wgsl", "GPU A", "b.json");
        build["build"] = json!({ "profile": "debug" });
        assert!(check_labels(&[run("wgsl", "GPU A", "a.json"), build]).is_err());
        assert!(check_labels(&[]).is_err());
    }

    fn gpu_cell(backend: &str, ms: f64, first: bool, sm: f64) -> Value {
        json!({
            "workload": "W1", "n": 4096, "backend": backend,
            "metrics": { "end_to_end": { "median_ms": ms, "n": 30, "paired_vs_prebuilt": { "reference": "wgsl-prebuilt", "n": 30, "median_ratio": ms / 2.0, "p5_ratio": 0.9, "p95_ratio": 1.1 } } },
            "cold_start_ms": [10.0 + ms, 5.0 + ms], "cold_start_first_in_process": first,
            "cold_start_note": "note",
            "warmup": { "actual_s": 0.8, "p0_reached": sm > 2500.0, "clocks": { "sm_mhz": sm, "pstate": "P0", "pcie_gen_now": 4.0, "pcie_gen_max": 5.0, "pcie_width": 4.0, "mem_mhz": 1.0 } },
            "clocks_after_timing": { "sm_mhz": sm, "pstate": "P5", "pcie_gen_now": 4.0, "pcie_gen_max": 5.0, "pcie_width": 4.0, "mem_mhz": 810.0 },
        })
    }

    #[test]
    fn cold_start_keeps_the_first_in_a_process_apart() {
        let mut one = run("wgsl", "GPU A", "a.json");
        one["cells"] = json!([gpu_cell("gust-gpu", 1.0, true, 2500.0)]);
        let mut two = run("wgsl", "GPU A", "b.json");
        two["cells"] = json!([gpu_cell("gust-gpu", 1.0, false, 2500.0)]);
        let cold = cold_starts(&[one, two]);
        let point = &cold[&("W1".to_owned(), 4096, "gust-gpu".to_owned())];
        assert_eq!(point.process_first, vec![11.0]);
        assert_eq!(point.rest, vec![6.0, 11.0, 6.0]);
        assert_eq!(point.note.as_deref(), Some("note"));
    }

    #[test]
    fn ablation_rows_and_clocks_come_out_of_the_runs() {
        let mut one = run("wgsl", "GPU A", "a.json");
        one["cells"] = json!([
            gpu_cell("wgsl-percall", 2.0, false, 2400.0),
            gpu_cell("wgsl-percall+all-on", 1.5, false, 2500.0),
            gpu_cell("wgsl-prebuilt", 1.0, false, 2600.0),
        ]);
        let table = gather(std::slice::from_ref(&one));
        let sizes = BTreeMap::from([("W1".to_owned(), vec![4096u64])]);
        let rows = ablation_rows(&table, &sizes);
        let all_on = rows
            .iter()
            .find(|r| r["backend"] == "wgsl-percall+all-on" && r["metric"] == "end_to_end")
            .expect("all-on row");
        assert_eq!(all_on["ratio_vs_prebuilt"], 1.5);
        assert_eq!(all_on["ratio_vs_percall"], 0.75);
        let clocks = clock_rows(&[one]);
        assert_eq!(clocks.len(), 3);
        assert_eq!(clocks[0]["sm_mhz"], json!([2400.0]));
        assert_eq!(clocks[0]["pstate"], json!(["P0"]));
    }

    #[test]
    fn the_markdown_has_the_ablation_table_and_no_stale_text() {
        let dir = scratch("markdown");
        let mut one = run("wgsl", "GPU A", "a.json");
        one["cells"] = json!([
            gpu_cell("wgsl-percall", 2.0, true, 2400.0),
            gpu_cell("wgsl-prebuilt", 1.0, true, 2600.0),
        ]);
        let inputs = write_runs(&dir, &[one]);
        let (out_json, out_md) = (dir.join("o.json"), dir.join("o.md"));
        aggregate(
            &inputs,
            out_json.to_str().unwrap(),
            out_md.to_str().unwrap(),
        )
        .unwrap();
        let md = fs::read_to_string(&out_md).unwrap();
        assert!(md.contains("## Ablation"));
        assert!(md.contains("wgsl-percall+all-on"));
        assert!(md.contains("## GPU clocks at the start of timing"));
        assert!(md.contains("first in process"));
        assert!(md.contains("## Paired ratios to wgsl-prebuilt"));
        assert!(md.contains("1.000 [1.000-1.000]"), "paired cell missing");
        assert!(
            md.contains("1 of 2 GPU cells reached P0"),
            "P0 count missing"
        );
        assert!(
            md.contains("| PCIe link |") && md.contains("gen 4 (max 5) x4"),
            "the link must stand next to the end-to-end numbers"
        );
        let summary: Value = serde_json::from_str(&fs::read_to_string(&out_json).unwrap()).unwrap();
        assert_eq!(summary["p0"]["p0_not_reached"], 1);
        assert!(
            summary["points"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|p| p["metric"] == "end_to_end")
                .all(|p| p["pcie_link"] == "gen 4 (max 5) x4")
        );
        assert!(!md.contains("Other lanes"), "stale text");
        assert!(!md.contains("uncommitted bench"), "stale text");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_config_or_commit_mismatch_between_runs_is_an_error() {
        let base = run("wgsl", "GPU A", "a.json");
        let mut quick = run("wgsl", "GPU A", "b.json");
        quick["config"] =
            json!({ "samples": 5, "sizes": [4096], "workloads": ["W1"], "backends": [] });
        let full = {
            let mut v = run("wgsl", "GPU A", "c.json");
            v["config"] =
                json!({ "samples": 30, "sizes": [4096], "workloads": ["W1"], "backends": [] });
            v
        };
        let error = check_labels(&[full.clone(), quick]).unwrap_err();
        assert!(error.to_string().contains("`config`"), "{error}");
        let mut ablated = full.clone();
        ablated["config"]["backends"] = json!([{ "id": "wgsl-percall+staging-reuse" }]);
        assert!(check_labels(&[full, ablated]).is_err());
        let mut other_commit = run("wgsl", "GPU A", "d.json");
        other_commit["build"] = json!({ "profile": "release", "commit": "other" });
        let mut with_commit = run("wgsl", "GPU A", "e.json");
        with_commit["build"] = json!({ "profile": "release", "commit": "abc" });
        assert!(check_labels(&[with_commit, other_commit]).is_err());
        assert!(check_labels(&[base.clone(), base]).is_ok());
    }

    #[test]
    fn the_aggregate_says_not_publishable_with_the_reason_and_lists_the_sidecars() {
        let dir = scratch("aggregate");
        let inputs = write_runs(
            &dir,
            &[
                run("wgsl", "GPU A", "a.samples.json"),
                run("wgsl", "GPU A", "b.samples.json"),
            ],
        );
        let out_json = dir.join("out.json");
        let out_md = dir.join("out.md");
        aggregate(
            &inputs,
            out_json.to_str().unwrap(),
            out_md.to_str().unwrap(),
        )
        .unwrap();
        let summary: Value = serde_json::from_str(&fs::read_to_string(&out_json).unwrap()).unwrap();
        assert_eq!(summary["publishable"], false);
        assert_eq!(
            summary["publishable_reason"],
            crate::meta::PUBLISHABLE_REASON
        );
        assert_eq!(summary["runs"][0]["sidecar"], "a.samples.json");
        assert_eq!(summary["runs"][1]["sidecar"], "b.samples.json");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_aggregate_refuses_runs_of_different_routes() {
        let dir = scratch("mismatch");
        let inputs = write_runs(
            &dir,
            &[
                run("wgsl", "GPU A", "a.json"),
                run("spirv", "GPU A", "b.json"),
            ],
        );
        let out_json = dir.join("out.json");
        let out_md = dir.join("out.md");
        assert!(
            aggregate(
                &inputs,
                out_json.to_str().unwrap(),
                out_md.to_str().unwrap()
            )
            .is_err()
        );
        assert!(!out_json.exists());
        let _ = fs::remove_dir_all(&dir);
    }
    #[test]
    fn runs_of_different_groups_do_not_aggregate() {
        let mut a = run("wgsl", "GPU A", "a.json");
        a["group"] = json!("publishable");
        let mut b = run("wgsl", "GPU A", "b.json");
        b["group"] = json!("ablation");
        let error = check_labels(&[a.clone(), b]).unwrap_err();
        assert!(error.to_string().contains("`group`"), "{error}");
        assert!(check_labels(&[a.clone(), a]).is_ok());
    }

    #[test]
    fn a_gpu_point_above_the_dram_peak_is_flagged_and_a_cpu_or_low_one_is_not() {
        let cell = |backend: &str, ms: f64| {
            json!({
                "workload": "W1", "n": 1048576, "backend": backend, "bytes_per_element": 12,
                "metrics": { "gpu_batch_per_dispatch": { "median_ms": ms, "n": 30 } },
            })
        };
        // 1048576 * 12 B in 0.006 ms is about 2097 GB/s: above 1792. In 0.01 ms it is about 1258: below.
        let mut one = run("wgsl", "NVIDIA GeForce RTX 5090", "a.json");
        one["cells"] = json!([
            cell("wgsl-prebuilt", 0.006),
            cell("gust-gpu", 0.01),
            cell("rayon-N", 0.006)
        ]);
        let table = gather(std::slice::from_ref(&one));
        let info = workload_info(std::slice::from_ref(&one));
        let (peak, source) = dram_peak_gbps("NVIDIA GeForce RTX 5090").unwrap();
        assert_eq!(peak, 1792.0);
        assert!(source.contains("1792"));
        let flags = bandwidth_flags(&table, &info, peak);
        assert_eq!(flags.len(), 1);
        assert_eq!(flags[0]["backend"], "wgsl-prebuilt");
        assert!(dram_peak_gbps("Some Other GPU").is_none());
    }
}
