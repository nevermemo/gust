//! Warm-up, sampling, statistics and the per-cell protocol.
//!
//! A cell is one (backend, workload, size). Four timed quantities are collected per cell, each with a correctness
//! check on every sample that runs outside the timed region (the output is poisoned with NaN first, so a dispatch
//! that does not run, or that writes the wrong thing, fails):
//!
//! - `end_to_end`: host inputs in, host output out, everything complete (GPU: upload, dispatch, read-back).
//! - `dispatch_host`: one dispatch on resident data, host wall clock from the call to completion.
//! - `gpu_single`: GPU timestamps around one compute pass.
//! - `gpu_batch_per_dispatch`: GPU timestamps around `batch` back-to-back passes in one submission, divided by `batch`.
//!   The ratio to `gpu_single` shows how much of a small dispatch is fixed pass cost.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::{
    backend::{Backend, BenchResult, Kind},
    meta,
    workloads::{Inputs, Workload, check, gen_inputs, reference},
};

#[derive(Clone, Debug)]
pub struct Config {
    /// Target sample count per quantity.
    pub samples: usize,
    /// Fewer samples are accepted only when `cap` runs out first.
    pub min_samples: usize,
    /// Wall-clock cap per quantity, checks and poisoning included, after warm-up.
    pub cap: Duration,
    /// Passes per submission for `gpu_batch_per_dispatch`.
    pub batch: usize,
    pub warmup_end_to_end: usize,
    pub warmup_dispatch: usize,
    pub cold_repeats: usize,
    /// Time-based warm-up of a GPU cell before its first timed sample: the cell's own dispatches run for at least
    /// this long, so the clocks have left idle (the sampler idles the GPU for the settle and window before every
    /// cell). Zero turns it off.
    pub warmup_time: Duration,
    /// Upper bound of the warm-up when the clock query is slow.
    pub warmup_max: Duration,
    /// Measure only the cold start (GPU cells at the first size), none of the four timings.
    pub cold_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            samples: 30,
            min_samples: 5,
            cap: Duration::from_secs(8),
            batch: 16,
            warmup_end_to_end: 3,
            warmup_dispatch: 5,
            cold_repeats: 3,
            warmup_time: Duration::from_millis(750),
            warmup_max: Duration::from_secs(3),
            cold_only: false,
        }
    }
}

/// Summary of one sample set, in milliseconds.
pub struct Stats {
    pub n: usize,
    pub median: f64,
    pub min: f64,
    pub p5: f64,
    pub p95: f64,
    pub mad: f64,
    /// Samples further than 3 MAD from the median. They are kept in the statistics, only counted.
    pub outliers: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = p * (sorted.len() - 1) as f64;
    let (low, high) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile(&sorted, 0.5)
}

pub fn stats(samples_ms: &[f64]) -> Stats {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = percentile(&sorted, 0.5);
    let deviations = sorted
        .iter()
        .map(|v| (v - median).abs())
        .collect::<Vec<_>>();
    let mad = self::median(&deviations);
    Stats {
        n: sorted.len(),
        median,
        min: sorted[0],
        p5: percentile(&sorted, 0.05),
        p95: percentile(&sorted, 0.95),
        mad,
        outliers: deviations.iter().filter(|d| **d > 3.0 * mad).count(),
    }
}

fn stats_json(samples_ms: &[f64]) -> Value {
    let s = stats(samples_ms);
    json!({
        "n": s.n, "median_ms": s.median, "min_ms": s.min, "p5_ms": s.p5, "p95_ms": s.p95,
        "mad_ms": s.mad, "outliers_gt_3mad": s.outliers, "samples_ms": samples_ms,
    })
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Runs `sample` until `samples` values are in or the cap is hit (but never fewer than `min_samples`). A `None` from
/// the first call means the quantity does not exist for this backend.
fn collect(
    config: &Config,
    mut sample: impl FnMut() -> BenchResult<Option<f64>>,
) -> BenchResult<Option<Vec<f64>>> {
    let started = Instant::now();
    let mut values = Vec::with_capacity(config.samples);
    while values.len() < config.samples {
        if values.len() >= config.min_samples && started.elapsed() > config.cap {
            break;
        }
        match sample()? {
            Some(value) => values.push(value),
            None => return Ok(None),
        }
    }
    Ok(Some(values))
}

/// Warms a GPU backend up by time: dispatches on resident data for at least `config.warmup_time`, and then reads the
/// GPU clock state (SM clock, pstate, PCIe link) while the dispatches go on, so the record shows the clock at the end
/// of the warm-up, which is the clock the first timed sample sees. (A query started at the beginning would read the
/// idle clock: `nvidia-smi` takes a few hundred ms to answer.) Returns the record.
fn warm_up_by_time(backend: &mut dyn Backend, config: &Config) -> BenchResult<Value> {
    let started = Instant::now();
    let mut dispatches = 0u64;
    while started.elapsed() < config.warmup_time {
        backend.dispatch()?;
        dispatches += 1;
    }
    let busy_until_query = Instant::now();
    let clocks = std::thread::scope(|scope| -> BenchResult<Value> {
        let query = scope.spawn(meta::gpu_clocks);
        while !query.is_finished() && started.elapsed() < config.warmup_max {
            backend.dispatch()?;
            dispatches += 1;
        }
        // `join` only fails when the query thread panicked.
        Ok(query.join().unwrap_or(Value::Null))
    })?;
    Ok(json!({
        "target_s": config.warmup_time.as_secs_f64(),
        "actual_s": started.elapsed().as_secs_f64(),
        "clock_query_s": busy_until_query.elapsed().as_secs_f64(),
        "dispatches": dispatches,
        "clocks": clocks,
    }))
}

/// Process-wide set of (backend, workload) pairs whose cold start has been measured once already: the first cold start of a
/// kernel in a process also pays one-time costs (the Slang helper's compiler check, a first module), so it is reported apart.
fn first_cold_in_process(backend_id: &'static str, workload: Workload) -> bool {
    use std::{collections::HashSet, sync::Mutex};
    static SEEN: Mutex<Option<HashSet<(&'static str, Workload)>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .get_or_insert_with(HashSet::new)
        .insert((backend_id, workload))
}

/// What a backend's cold start does and does not include, written next to its numbers.
fn cold_note(backend_id: &str) -> &'static str {
    if backend_id.starts_with("gust") {
        "pipeline cache cleared, first dispatch timed: includes the run-time Slang compile (or the artifact load) and naga/driver pipeline creation; the driver's shader cache may be warm from an earlier process"
    } else {
        "driver shader cache warm: prepare already created this pipeline on the same device, so this is module and pipeline creation plus one dispatch, not a first-ever compile"
    }
}

/// Everything one cell produced.
pub struct Cell {
    pub json: Value,
    pub failures: Vec<String>,
}

pub struct CellInput<'a> {
    pub workload: Workload,
    pub inputs: &'a Inputs,
    pub reference: &'a [f32],
    /// Measure the cold start in this cell (once per workload, at the smallest size).
    pub cold: bool,
}

fn verify(
    workload: Workload,
    inputs: &Inputs,
    want: &[f32],
    got: &[f32],
    checks: &mut u64,
) -> BenchResult<()> {
    check(workload, inputs, want, got)
        .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;
    *checks += 1;
    Ok(())
}

/// The full protocol for one cell. A failed correctness check ends the cell and is returned in `failures`.
pub fn run_cell(backend: &mut dyn Backend, input: &CellInput<'_>, config: &Config) -> Cell {
    let n = input.inputs.x.len();
    let gpu_before = meta::gpu_snapshot();
    let started = Instant::now();
    let mut checks = 0u64;
    let mut failures = Vec::new();
    let mut metrics = serde_json::Map::new();
    let mut cold_ms = None;
    let mut cold_first = false;
    let mut warmup = Value::Null;
    let mut misses = None;

    let outcome = (|| -> BenchResult<()> {
        backend.prepare(input.workload, input.inputs)?;

        if input.cold && backend.kind() == Kind::Gpu {
            let mut values = Vec::new();
            cold_first =
                config.cold_repeats > 0 && first_cold_in_process(backend.id(), input.workload);
            for _ in 0..config.cold_repeats {
                if let Some(duration) = backend.cold_start()? {
                    values.push(ms(duration));
                }
            }
            cold_ms = Some(values);
        }
        if config.cold_only {
            return Ok(());
        }

        // Warm-up: real dispatches, so the driver's lazy pipeline work and the clocks have settled.
        let mut out = vec![f32::NAN; n];
        for _ in 0..config.warmup_end_to_end {
            backend.end_to_end(input.inputs, &mut out)?;
        }
        for _ in 0..config.warmup_dispatch {
            backend.dispatch()?;
        }
        backend.gpu_time(config.batch)?;
        if backend.kind() == Kind::Gpu && !config.warmup_time.is_zero() {
            warmup = warm_up_by_time(backend, config)?;
        }
        let misses_before = backend.cache_misses();

        let mut out = vec![f32::NAN; n];
        let samples = collect(config, || {
            backend.poison()?;
            out.fill(f32::NAN);
            let started = Instant::now();
            backend.end_to_end(input.inputs, &mut out)?;
            let elapsed = started.elapsed();
            verify(
                input.workload,
                input.inputs,
                input.reference,
                &out,
                &mut checks,
            )?;
            Ok(Some(ms(elapsed)))
        })?;
        if let Some(samples) = samples {
            metrics.insert("end_to_end".to_owned(), stats_json(&samples));
        }

        let samples = collect(config, || {
            backend.poison()?;
            let started = Instant::now();
            backend.dispatch()?;
            let elapsed = started.elapsed();
            backend.read_back(&mut out)?;
            verify(
                input.workload,
                input.inputs,
                input.reference,
                &out,
                &mut checks,
            )?;
            Ok(Some(ms(elapsed)))
        })?;
        if let Some(samples) = samples {
            metrics.insert("dispatch_host".to_owned(), stats_json(&samples));
        }

        for (name, batch) in [("gpu_single", 1), ("gpu_batch_per_dispatch", config.batch)] {
            let samples = collect(config, || {
                backend.poison()?;
                let Some(elapsed) = backend.gpu_time(batch)? else {
                    return Ok(None);
                };
                backend.read_back(&mut out)?;
                verify(
                    input.workload,
                    input.inputs,
                    input.reference,
                    &out,
                    &mut checks,
                )?;
                Ok(Some(ms(elapsed) / batch as f64))
            })?;
            if let Some(samples) = samples {
                metrics.insert(name.to_owned(), stats_json(&samples));
            }
        }

        if let (Some(before), Some(after)) = (misses_before, backend.cache_misses()) {
            misses = Some((before, after));
            if before != after {
                failures.push(format!(
                    "{} {} {n}: pipeline cache misses moved from {before} to {after} inside the timed region",
                    backend.id(),
                    input.workload.id()
                ));
            }
        }
        Ok(())
    })();
    if let Err(error) = outcome {
        failures.push(format!(
            "{} {} {n}: {error}",
            backend.id(),
            input.workload.id()
        ));
    }

    let cell = json!({
        "backend": backend.id(),
        "kind": format!("{:?}", backend.kind()),
        "workload": input.workload.id(),
        "workload_name": input.workload.name(),
        "n": n,
        "bytes_per_element": input.workload.bytes_per_element(),
        "metrics": metrics,
        "checks_passed": checks,
        "cold_start_ms": cold_ms,
        "cold_start_first_in_process": cold_first,
        "cold_start_note": cold_ms.as_ref().map(|_| cold_note(backend.id())),
        "warmup": warmup,
        "cache_misses": misses.map(|(before, after)| json!({ "before": before, "after": after })),
        "failed": !failures.is_empty(),
        "wall_s": started.elapsed().as_secs_f64(),
        "gpu_before": gpu_before,
        "gpu_after": meta::gpu_snapshot(),
    });
    Cell {
        json: cell,
        failures,
    }
}

/// Inputs and the serial reference for one (workload, size).
pub fn prepare_case(n: usize, workload: Workload) -> (Inputs, Vec<f32>) {
    let inputs = gen_inputs(n);
    let want = reference(workload, &inputs);
    (inputs, want)
}
