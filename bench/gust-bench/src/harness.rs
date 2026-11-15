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

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

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
    /// Warm-up of a GPU group before its first timed sample, with real load (end-to-end samples alternating with
    /// `batch`-pass batches) until the clock query reports P0, and no longer than this cap (1 s). Zero turns it off. The
    /// sampler idles the GPU for the settle and window before every cell, which drops it to P8 (memory 405 MHz).
    pub warmup_cap: Duration,
    /// Passes per submission of the warm-up batches. Far above `batch`: the 16-pass batches of the round 3 brief left a
    /// small kernel too short to raise the clocks (the smoke reached P0 in 1 of 4 cells within 3 s).
    pub warmup_batch: usize,
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
            warmup_cap: Duration::from_secs(1),
            warmup_batch: 1024,
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

/// Whether a clock record (see `meta::gpu_clocks`) says the GPU is in P0 and its clocks are really up. The pstate
/// string alone is not enough: the smoke of round 3 read `P0` with the SM at 225 MHz and the memory at 405 MHz while
/// the GPU was busy, so the memory clock must also be within 90 % of its maximum and the SM clock within 50 % of its
/// maximum (the maxima come from the same query; without them only the pstate counts).
fn is_p0(clocks: &Value) -> bool {
    if clocks["pstate"].as_str() != Some("P0") {
        return false;
    }
    let up =
        |now: &str, max: &str, fraction: f64| match (clocks[now].as_f64(), clocks[max].as_f64()) {
            (Some(now), Some(max)) => now >= max * fraction,
            _ => true,
        };
    up("mem_mhz", "max_mem_mhz", 0.9) && up("sm_mhz", "max_sm_mhz", 0.5)
}

/// One round of real load: for each backend of the group in turn, an end-to-end sample (transfers both ways) and a
/// `batch`-pass batch.
fn load_round(
    group: &mut [&mut dyn Backend],
    alive: &[bool],
    inputs: &Inputs,
    out: &mut Vec<f32>,
    batch: usize,
) -> BenchResult<()> {
    for (backend, _) in group.iter_mut().zip(alive).filter(|(_, alive)| **alive) {
        backend.end_to_end(inputs, out)?;
        backend.gpu_time(batch)?;
    }
    Ok(())
}

/// Warms a group of GPU backends up together with real load (one shared warm-up per workload and size): end-to-end
/// samples alternating with `config.warmup_batch`-pass batches, while a
/// second thread polls the clock query (`nvidia-smi` takes a few hundred ms to answer, so the load must go on while it
/// runs), until the query reports P0 or `config.warmup_cap` is hit. A cell that hits the cap is recorded with
/// `p0_reached: false` and the mark `p0_not_reached`, not dropped: P0 is a record here, not a goal (the paired
/// rotation of [`run_gpu_group`] is what makes the backends comparable). Returns the record; `clocks` is the last poll.
fn warm_up_to_p0(
    group: &mut [&mut dyn Backend],
    alive: &[bool],
    inputs: &Inputs,
    config: &Config,
) -> BenchResult<Value> {
    let started = Instant::now();
    let reached = AtomicBool::new(false);
    let stop = AtomicBool::new(false);
    let mut rounds = 0u64;
    let mut out = vec![f32::NAN; inputs.x.len()];
    let (clocks, polls, time_to_p0) = std::thread::scope(|scope| {
        let poller = scope.spawn(|| {
            let mut last = Value::Null;
            let mut polls = 0u64;
            let mut first_p0 = None;
            while !stop.load(Ordering::Relaxed) {
                last = meta::gpu_clocks();
                polls += 1;
                if is_p0(&last) {
                    first_p0 = Some(started.elapsed().as_secs_f64());
                    reached.store(true, Ordering::Relaxed);
                    break;
                }
            }
            (last, polls, first_p0)
        });
        let mut result = Ok(());
        while !reached.load(Ordering::Relaxed) && started.elapsed() < config.warmup_cap {
            if let Err(error) = load_round(group, alive, inputs, &mut out, config.warmup_batch) {
                result = Err(error);
                break;
            }
            rounds += 1;
        }
        stop.store(true, Ordering::Relaxed);
        // `join` only fails when the poller panicked.
        let polled = poller.join().unwrap_or((Value::Null, 0, None));
        result.map(|()| polled)
    })?;
    let p0 = reached.load(Ordering::Relaxed);
    Ok(json!({
        "cap_s": config.warmup_cap.as_secs_f64(),
        "actual_s": started.elapsed().as_secs_f64(),
        "p0_reached": p0,
        "p0_mark": if p0 { Value::Null } else { json!("p0_not_reached") },
        "time_to_p0_s": time_to_p0,
        "rounds": rounds,
        "polls": polls,
        "clocks": clocks,
    }))
}

/// The clock state right after the timed samples: read while the backend still dispatches, so a GPU that is already
/// falling back to idle during the few hundred ms of `nvidia-smi` is not mistaken for one that never left it.
fn clocks_under_load(backend: &mut dyn Backend) -> BenchResult<Value> {
    std::thread::scope(|scope| {
        let query = scope.spawn(meta::gpu_clocks);
        let mut result = Ok(());
        while !query.is_finished() {
            if let Err(error) = backend.dispatch() {
                result = Err(error);
                break;
            }
        }
        let clocks = query.join().unwrap_or(Value::Null);
        result.map(|()| clocks)
    })
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
    let mut clocks_after = Value::Null;
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

        // Warm-up: real dispatches, so the driver's lazy pipeline work has settled; on a GPU, real load until P0.
        let mut out = vec![f32::NAN; n];
        for _ in 0..config.warmup_end_to_end {
            backend.end_to_end(input.inputs, &mut out)?;
        }
        for _ in 0..config.warmup_dispatch {
            backend.dispatch()?;
        }
        backend.gpu_time(config.batch)?;
        if backend.kind() == Kind::Gpu && !config.warmup_cap.is_zero() {
            warmup = warm_up_to_p0(&mut [&mut *backend], &[true], input.inputs, config)?;
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

        if backend.kind() == Kind::Gpu && !config.warmup_cap.is_zero() {
            clocks_after = clocks_under_load(backend)?;
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
        "clocks_after_timing": clocks_after,
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

/// What one backend of a [`run_gpu_group`] collects.
struct Member {
    checks: u64,
    failures: Vec<String>,
    metrics: serde_json::Map<String, Value>,
    cold_ms: Option<Vec<f64>>,
    cold_first: bool,
    alive: bool,
    misses_before: Option<u64>,
}

/// The backend every other one is paired against.
pub const PAIR_REFERENCE: &str = "wgsl-prebuilt";

/// The paired ratios `a[i] / reference[i]` of two sample sets taken in the same rotations: median, spread (min, p5,
/// p95, max) and the count. `None` when either set is empty.
fn paired_json(samples: &[f64], reference: &[f64]) -> Option<Value> {
    let ratios = samples
        .iter()
        .zip(reference)
        .filter(|(_, r)| **r > 0.0)
        .map(|(a, r)| a / r)
        .collect::<Vec<_>>();
    if ratios.is_empty() {
        return None;
    }
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    Some(json!({
        "reference": PAIR_REFERENCE, "n": ratios.len(), "median_ratio": percentile(&sorted, 0.5),
        "min_ratio": sorted[0], "p5_ratio": percentile(&sorted, 0.05),
        "p95_ratio": percentile(&sorted, 0.95), "max_ratio": sorted[sorted.len() - 1],
    }))
}

/// Takes the samples of one quantity for the whole group in rotation (A B C ... A B C, the starting backend shifting
/// by one each round so no backend always follows the same neighbour), so a P-state drift hits every backend alike.
/// `sample` returns one value, or `None` when the backend has no such quantity. A backend that fails is dropped from
/// the rotation with its failure recorded; the others go on. The cap is per cell, so it scales with the group.
fn rotate_quantity(
    group: &mut [&mut dyn Backend],
    members: &mut [Member],
    config: &Config,
    name: &str,
    mut sample: impl FnMut(&mut dyn Backend, &mut Member) -> BenchResult<Option<f64>>,
) {
    let count = group.len();
    let cap = config.cap * u32::try_from(count.max(1)).unwrap_or(1);
    let started = Instant::now();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); count];
    let mut absent = vec![false; count];
    let mut round = 0;
    while round < config.samples {
        if round >= config.min_samples && started.elapsed() > cap {
            break;
        }
        for step in 0..count {
            let i = (round + step) % count;
            if !members[i].alive || absent[i] {
                continue;
            }
            match sample(&mut *group[i], &mut members[i]) {
                Ok(Some(value)) => values[i].push(value),
                Ok(None) => absent[i] = true,
                Err(error) => {
                    members[i]
                        .failures
                        .push(format!("{} {name}: {error}", group[i].id()));
                    members[i].alive = false;
                }
            }
        }
        round += 1;
    }
    let reference = group
        .iter()
        .position(|b| b.id() == PAIR_REFERENCE)
        .map(|i| values[i].clone());
    for i in 0..count {
        if values[i].is_empty() {
            continue;
        }
        let mut entry = stats_json(&values[i]);
        if let Some(paired) = reference.as_ref().and_then(|r| paired_json(&values[i], r)) {
            entry["paired_vs_prebuilt"] = paired;
        }
        members[i].metrics.insert(name.to_owned(), entry);
    }
}

/// Runs every GPU backend of one (workload, size) as a group: each is prepared (and its cold start measured) on its
/// own, then one shared warm-up, then the four timed quantities one after the other, each taken in rotation across the
/// backends ([`rotate_quantity`]). Each backend keeps its own pipeline and buffers across the rotation, and every
/// call returns only when its work is complete (`poison` and the timed call both wait), so one backend leaves nothing
/// in flight for the next; the pipeline-cache miss counter of each backend must not move across the timed samples
/// (asserted). Returns one cell per backend, in group order, with the shared warm-up and the clocks after the
/// rotation in each, and the paired ratio to [`PAIR_REFERENCE`] inside each timed quantity.
pub fn run_gpu_group(
    group: &mut [&mut dyn Backend],
    input: &CellInput<'_>,
    config: &Config,
) -> Vec<Cell> {
    let n = input.inputs.x.len();
    let started = Instant::now();
    let gpu_before = meta::gpu_snapshot();
    let mut members = (0..group.len())
        .map(|_| Member {
            checks: 0,
            failures: Vec::new(),
            metrics: serde_json::Map::new(),
            cold_ms: None,
            cold_first: false,
            alive: true,
            misses_before: None,
        })
        .collect::<Vec<_>>();
    let mut warmup = Value::Null;
    let mut clocks_after = Value::Null;

    for (backend, member) in group.iter_mut().zip(members.iter_mut()) {
        let outcome = (|| -> BenchResult<()> {
            backend.prepare(input.workload, input.inputs)?;
            if input.cold {
                let mut values = Vec::new();
                member.cold_first =
                    config.cold_repeats > 0 && first_cold_in_process(backend.id(), input.workload);
                for _ in 0..config.cold_repeats {
                    if let Some(duration) = backend.cold_start()? {
                        values.push(ms(duration));
                    }
                }
                member.cold_ms = Some(values);
            }
            Ok(())
        })();
        if let Err(error) = outcome {
            member.failures.push(format!(
                "{} {} {n}: {error}",
                backend.id(),
                input.workload.id()
            ));
            member.alive = false;
        }
    }

    if !config.cold_only {
        let mut out = vec![f32::NAN; n];
        // Per-backend warm-up first (lazy pipeline work), then one shared load warm-up for the group.
        for (backend, member) in group.iter_mut().zip(members.iter_mut()) {
            if !member.alive {
                continue;
            }
            let outcome = (|| -> BenchResult<()> {
                for _ in 0..config.warmup_end_to_end {
                    backend.end_to_end(input.inputs, &mut out)?;
                }
                for _ in 0..config.warmup_dispatch {
                    backend.dispatch()?;
                }
                backend.gpu_time(config.batch)?;
                member.misses_before = backend.cache_misses();
                Ok(())
            })();
            if let Err(error) = outcome {
                member
                    .failures
                    .push(format!("{} warm-up: {error}", backend.id()));
                member.alive = false;
            }
        }
        if !config.warmup_cap.is_zero() && members.iter().any(|m| m.alive) {
            let alive = members.iter().map(|m| m.alive).collect::<Vec<_>>();
            match warm_up_to_p0(group, &alive, input.inputs, config) {
                Ok(record) => warmup = record,
                Err(error) => {
                    for member in members.iter_mut().filter(|m| m.alive) {
                        member.failures.push(format!("shared warm-up: {error}"));
                    }
                }
            }
        }

        let batch = config.batch;
        let (inputs, reference) = (input.inputs, input.reference);
        let workload = input.workload;
        rotate_quantity(
            group,
            &mut members,
            config,
            "end_to_end",
            |backend, member| {
                backend.poison()?;
                out.fill(f32::NAN);
                let started = Instant::now();
                backend.end_to_end(inputs, &mut out)?;
                let elapsed = started.elapsed();
                verify(workload, inputs, reference, &out, &mut member.checks)?;
                Ok(Some(ms(elapsed)))
            },
        );
        rotate_quantity(
            group,
            &mut members,
            config,
            "dispatch_host",
            |backend, member| {
                backend.poison()?;
                let started = Instant::now();
                backend.dispatch()?;
                let elapsed = started.elapsed();
                backend.read_back(&mut out)?;
                verify(workload, inputs, reference, &out, &mut member.checks)?;
                Ok(Some(ms(elapsed)))
            },
        );
        for (name, passes) in [("gpu_single", 1), ("gpu_batch_per_dispatch", batch)] {
            rotate_quantity(group, &mut members, config, name, |backend, member| {
                backend.poison()?;
                let Some(elapsed) = backend.gpu_time(passes)? else {
                    return Ok(None);
                };
                backend.read_back(&mut out)?;
                verify(workload, inputs, reference, &out, &mut member.checks)?;
                Ok(Some(ms(elapsed) / passes as f64))
            });
        }
        if !config.warmup_cap.is_zero()
            && let Some(index) = members.iter().position(|m| m.alive)
        {
            match clocks_under_load(&mut *group[index]) {
                Ok(clocks) => clocks_after = clocks,
                Err(error) => members[index]
                    .failures
                    .push(format!("clocks after rotation: {error}")),
            }
        }
    }

    let mut cells = Vec::new();
    for (backend, member) in group.iter().zip(members.iter_mut()) {
        let mut misses = None;
        if let (Some(before), Some(after)) = (member.misses_before, backend.cache_misses()) {
            misses = Some((before, after));
            if before != after {
                member.failures.push(format!(
                    "{} {} {n}: pipeline cache misses moved from {before} to {after} inside the timed region",
                    backend.id(),
                    input.workload.id()
                ));
            }
        }
        let cell = json!({
            "backend": backend.id(),
            "kind": format!("{:?}", backend.kind()),
            "workload": input.workload.id(),
            "workload_name": input.workload.name(),
            "n": n,
            "bytes_per_element": input.workload.bytes_per_element(),
            "metrics": member.metrics,
            "checks_passed": member.checks,
            "cold_start_ms": member.cold_ms,
            "cold_start_first_in_process": member.cold_first,
            "cold_start_note": member.cold_ms.as_ref().map(|_| cold_note(backend.id())),
            "warmup": warmup,
            "clocks_after_timing": clocks_after,
            "paired_group": group.iter().map(|b| b.id()).collect::<Vec<_>>(),
            "cache_misses": misses.map(|(before, after)| json!({ "before": before, "after": after })),
            "failed": !member.failures.is_empty(),
            "wall_s": started.elapsed().as_secs_f64(),
            "gpu_before": gpu_before,
            "gpu_after": meta::gpu_snapshot(),
        });
        cells.push(Cell {
            json: cell,
            failures: std::mem::take(&mut member.failures),
        });
    }
    cells
}

/// Inputs and the serial reference for one (workload, size).
pub fn prepare_case(n: usize, workload: Workload) -> (Inputs, Vec<f32>) {
    let inputs = gen_inputs(n);
    let want = reference(workload, &inputs);
    (inputs, want)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU-kind backend with a cold start and nothing else: enough for a cold-only cell.
    struct Fake(&'static str);

    impl Backend for Fake {
        fn id(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "fake"
        }
        fn kind(&self) -> Kind {
            Kind::Gpu
        }
        fn prepare(&mut self, _: Workload, _: &Inputs) -> BenchResult<()> {
            Ok(())
        }
        fn poison(&mut self) -> BenchResult<()> {
            Ok(())
        }
        fn end_to_end(&mut self, _: &Inputs, _: &mut Vec<f32>) -> BenchResult<()> {
            Ok(())
        }
        fn dispatch(&mut self) -> BenchResult<()> {
            Ok(())
        }
        fn read_back(&mut self, _: &mut Vec<f32>) -> BenchResult<()> {
            Ok(())
        }
        fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
            Ok(Some(Duration::from_millis(1)))
        }
    }

    fn cold_cell(backend: &mut Fake, workload: Workload) -> Value {
        let (inputs, want) = prepare_case(8, workload);
        let input = CellInput {
            workload,
            inputs: &inputs,
            reference: &want,
            cold: true,
        };
        let config = Config {
            cold_only: true,
            cold_repeats: 2,
            ..Config::default()
        };
        run_cell(backend, &input, &config).json
    }

    #[test]
    fn the_first_cold_start_of_each_kernel_in_the_process_carries_the_flag() {
        let mut a = Fake("fake-first-cold-a");
        let mut b = Fake("fake-first-cold-b");
        // W1 first, then W2: each kernel's first cold start is flagged, the repeat of W1 is not.
        let w1 = cold_cell(&mut a, Workload::VectorAdd);
        let w2 = cold_cell(&mut a, Workload::Saxpy);
        let w1_again = cold_cell(&mut a, Workload::VectorAdd);
        let other_backend = cold_cell(&mut b, Workload::Saxpy);
        assert_eq!(w1["cold_start_first_in_process"], true);
        assert_eq!(w2["cold_start_first_in_process"], true);
        assert_eq!(w1_again["cold_start_first_in_process"], false);
        assert_eq!(other_backend["cold_start_first_in_process"], true);
        assert_eq!(w2["cold_start_ms"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn samples_are_taken_in_rotation_and_paired_against_the_reference() {
        let mut prebuilt = Fake(PAIR_REFERENCE);
        let mut other = Fake("fake-paired-other");
        let mut group: Vec<&mut dyn Backend> = vec![&mut prebuilt, &mut other];
        let mut members = (0..2)
            .map(|_| Member {
                checks: 0,
                failures: Vec::new(),
                metrics: serde_json::Map::new(),
                cold_ms: None,
                cold_first: false,
                alive: true,
                misses_before: None,
            })
            .collect::<Vec<_>>();
        let config = Config {
            samples: 4,
            min_samples: 2,
            ..Config::default()
        };
        let mut order = Vec::new();
        rotate_quantity(
            &mut group,
            &mut members,
            &config,
            "end_to_end",
            |backend, _| {
                order.push(backend.id());
                // A drifting clock: every later sample is slower, the other backend is always 3x the reference.
                let drift = order.len() as f64;
                Ok(Some(if backend.id() == PAIR_REFERENCE {
                    drift
                } else {
                    drift * 3.0
                }))
            },
        );
        // The starting backend shifts by one each round: P O, O P, P O, O P.
        assert_eq!(
            order,
            [
                PAIR_REFERENCE,
                "fake-paired-other",
                "fake-paired-other",
                PAIR_REFERENCE,
                PAIR_REFERENCE,
                "fake-paired-other",
                "fake-paired-other",
                PAIR_REFERENCE,
            ]
        );
        let entry = &members[1].metrics["end_to_end"];
        assert_eq!(entry["n"], 4);
        let paired = &entry["paired_vs_prebuilt"];
        assert_eq!(paired["n"], 4);
        // Neighbours in a round differ by one step of drift, so the paired ratio is near 3 only in the mean; it is
        // never the unpaired ratio of the medians, and it is computed per rotation.
        assert!(paired["min_ratio"].as_f64().unwrap() <= paired["median_ratio"].as_f64().unwrap());
        assert!(paired["median_ratio"].as_f64().unwrap() <= paired["max_ratio"].as_f64().unwrap());
        // The reference paired with itself is exactly 1.
        assert_eq!(
            members[0].metrics["end_to_end"]["paired_vs_prebuilt"]["median_ratio"],
            1.0
        );
    }

    #[test]
    fn paired_ratios_are_per_rotation() {
        let paired = paired_json(&[2.0, 6.0, 3.0], &[1.0, 2.0, 1.0]).unwrap();
        // 2, 3, 3
        assert_eq!(paired["median_ratio"], 3.0);
        assert_eq!(paired["min_ratio"], 2.0);
        assert_eq!(paired["max_ratio"], 3.0);
        assert!(paired_json(&[], &[1.0]).is_none());
    }

    #[test]
    fn p0_is_the_pstate_string() {
        assert!(is_p0(&json!({ "pstate": "P0" })));
        assert!(!is_p0(&json!({ "pstate": "P8" })));
        assert!(!is_p0(&Value::Null));
        // P0 by name but idle clocks (what the smoke saw) is not P0.
        let idle = json!({ "pstate": "P0", "sm_mhz": 225.0, "mem_mhz": 405.0, "max_sm_mhz": 3090.0, "max_mem_mhz": 14001.0 });
        assert!(!is_p0(&idle));
        let up = json!({ "pstate": "P0", "sm_mhz": 2900.0, "mem_mhz": 14001.0, "max_sm_mhz": 3090.0, "max_mem_mhz": 14001.0 });
        assert!(is_p0(&up));
    }
}
