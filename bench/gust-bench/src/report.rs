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
const BACKENDS: [&str; 11] = [
    "gust-gpu",
    "wgsl-prebuilt",
    "wgsl-percall",
    "wgsl-percall+bind-group-reuse",
    "wgsl-percall+staging-reuse",
    "wgsl-percall+one-submission",
    "gust-twin",
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

/// The three labels every run must agree on: an aggregate of a WGSL and a SPIR-V run, or of two adapters, is not one
/// measurement, so a mismatch is an error rather than being labelled by the first run.
fn check_labels(runs: &[Value]) -> BenchResult<()> {
    let Some(first) = runs.first() else {
        return Err("no run files to aggregate".into());
    };
    for (index, run) in runs.iter().enumerate().skip(1) {
        for key in ["route", "build", "adapter"] {
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

fn cold_starts(runs: &[Value]) -> BTreeMap<(String, u64, String), Vec<f64>> {
    let mut cold = BTreeMap::new();
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
            cold.entry(key)
                .or_insert_with(Vec::new)
                .extend(values.iter().filter_map(Value::as_f64));
        }
    }
    cold
}

pub fn aggregate(inputs: &[String], out_json: &str, out_md: &str) -> BenchResult<()> {
    let runs = load(inputs)?;
    check_labels(&runs)?;
    let table = gather(&runs);
    let cold = cold_starts(&runs);

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

    // ---- JSON ----
    let points = table
        .iter()
        .map(|((workload, n, backend, metric), point)| {
            json!({
                "workload": workload, "n": n, "backend": backend, "metric": metric,
                "median_ms": point.median(), "min_run_median_ms": point.min(),
                "max_run_median_ms": point.max(), "spread_pct": point.spread_pct(),
                "per_run_median_ms": point.per_run_median, "samples_per_run": point.samples,
            })
        })
        .collect::<Vec<_>>();
    let overhead = overhead_rows(&table, &sizes);
    let cold_json = cold
        .iter()
        .map(|((workload, n, backend), values)| {
            json!({ "workload": workload, "n": n, "backend": backend,
                    "median_ms": median(values), "all_ms": values })
        })
        .collect::<Vec<_>>();
    let summary = json!({
        "banner": BANNER,
        "publishable": crate::meta::PUBLISHABLE,
        "publishable_reason": crate::meta::PUBLISHABLE_REASON,
        "route": runs[0]["route"],
        "build": runs[0]["build"],
        "adapter": runs[0]["adapter"],
        "meta": runs[0]["meta"],
        "config": runs[0]["config"],
        "runs": runs.iter().map(|run| json!({
            "label": run["label"], "sidecar": run["sampler"]["sidecar"], "load_before": run["load_before"], "load_after": run["load_after"],
            "gpu_before": run["gpu_before"], "gpu_after": run["gpu_after"],
            "wall_s": run["wall_s"], "failures": run["failures"],
        })).collect::<Vec<_>>(),
        "correctness_checks_passed": checks,
        "failures": failures,
        "points": points,
        "gust_vs_hand_overhead": overhead,
        "cold_start": cold_json,
    });
    fs::write(out_json, serde_json::to_string_pretty(&summary)? + "\n")?;

    // ---- Markdown ----
    let mut md = String::new();
    writeln!(md, "# M10 spike B1 results\n")?;
    writeln!(
        md,
        "**{BANNER}.** Other lanes were running cargo and GPU tests; see the run conditions at the end. A quiet-window run comes later.\n"
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
    machine_block(&mut md, &runs[0]["meta"])?;
    for (workload, list) in &sizes {
        let name = if workload == "W1" {
            "vector-add"
        } else {
            "saxpy"
        };
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
            writeln!(md)?;
            write!(md, "|---|")?;
            for _ in &columns {
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
                writeln!(md)?;
            }
            writeln!(md)?;
        }

        writeln!(
            md,
            "### Achieved bandwidth (GB/s, 12 bytes per element), from the batch timestamps and from end to end\n"
        )?;
        writeln!(
            md,
            "| elements | gust-gpu batch | wgsl-prebuilt batch | wgsl-percall batch | gust-gpu e2e | wgsl-prebuilt e2e | rayon-N e2e |"
        )?;
        writeln!(md, "|---|---|---|---|---|---|---|")?;
        for n in list {
            let gbps = |backend: &str, metric: &str| {
                table
                    .get(&(workload.clone(), *n, backend.to_owned(), metric.to_owned()))
                    .map_or("-".to_owned(), |p| {
                        let gbps = (*n as f64 * 12.0) / (p.median() * 1e-3) / 1e9;
                        if gbps < 10.0 {
                            format!("{gbps:.2}")
                        } else {
                            format!("{gbps:.0}")
                        }
                    })
            };
            writeln!(
                md,
                "| {n} | {} | {} | {} | {} | {} | {} |",
                gbps("gust-gpu", "gpu_batch_per_dispatch"),
                gbps("wgsl-prebuilt", "gpu_batch_per_dispatch"),
                gbps("wgsl-percall", "gpu_batch_per_dispatch"),
                gbps("gust-gpu", "end_to_end"),
                gbps("wgsl-prebuilt", "end_to_end"),
                gbps("rayon-N", "end_to_end"),
            )?;
        }
        writeln!(md)?;

        crossover(&mut md, &table, workload, list)?;
    }

    writeln!(
        md,
        "\n## GUST against hand WGSL (ratio of medians; above 1.00 means GUST is slower)\n"
    )?;
    writeln!(
        md,
        "| workload | elements | timing | gust-gpu (ms) | wgsl-prebuilt (ms) | ratio | difference (us) | wgsl-percall (ms) | ratio |"
    )?;
    writeln!(md, "|---|---|---|---|---|---|---|---|---|")?;
    for row in &overhead {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} | {:.3} | {:+.1} | {} | {} |",
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
        )?;
    }

    writeln!(
        md,
        "\n## Cold start (first dispatch after the pipeline is dropped; includes compile)\n"
    )?;
    writeln!(
        md,
        "| workload | elements | backend | median (ms) | all samples (ms) |"
    )?;
    writeln!(md, "|---|---|---|---|---|")?;
    for ((workload, n, backend), values) in &cold {
        let all = values
            .iter()
            .map(|v| fmt_ms(*v))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            md,
            "| {workload} | {n} | {backend} | {} | {all} |",
            fmt_ms(median(values))
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

fn machine_block(md: &mut String, meta: &Value) -> BenchResult<()> {
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
        "| versions | rustc `{}`, wgpu {}, rayon {}, GUST commit {} (plus the uncommitted bench) |",
        meta["versions"]["rustc"].as_str().unwrap_or("?"),
        meta["versions"]["wgpu"].as_str().unwrap_or("?"),
        meta["versions"]["rayon"].as_str().unwrap_or("?"),
        meta["versions"]["gust_commit"].as_str().unwrap_or("?")
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
fn crossover(md: &mut String, table: &Table, workload: &str, sizes: &[u64]) -> BenchResult<()> {
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
    for gpu in ["gust-gpu", "wgsl-prebuilt"] {
        let winner = sizes
            .iter()
            .copied()
            .find(|n| matches!((get(*n, gpu), get(*n, "rayon-N")), (Some(g), Some(c)) if g < c));
        match winner {
            Some(n) => writeln!(
                md,
                "End-to-end break-even, {gpu} against rayon-N: first faster at {n} elements."
            )?,
            None => writeln!(
                md,
                "End-to-end break-even, {gpu} against rayon-N: never faster on this ladder."
            )?,
        }
    }
    writeln!(md)?;
    Ok(())
}

fn overhead_rows(table: &Table, sizes: &BTreeMap<String, Vec<u64>>) -> Vec<Value> {
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
                let (Some(gust), Some(prebuilt)) = (get("gust-gpu"), get("wgsl-prebuilt")) else {
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
}
