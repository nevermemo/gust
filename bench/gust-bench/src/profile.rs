//! `gust-bench profile`: host-side counts per call. Counts, not timings.
//!
//! Three counters, none of which touches `crates/`:
//!
//! - **Rust heap allocations.** (The Vulkan driver's own allocations are invisible.) A counting global allocator (this module, only in a `profile-counts` build, so a timing build
//!   never pays for it). It counts `alloc`/`alloc_zeroed` calls and bytes, `realloc` calls, `dealloc` calls and the
//!   largest single request, for every thread of the process.
//! - **wgpu API calls.** wgpu-core logs every API call it receives at `trace` level through the `log` crate
//!   (`api_log!`). A `log::Log` installed here counts the calls by name (queue submissions, buffer creations, bind-group
//!   creations, `write_buffer`, polls, map requests, dispatches) for every device of the process, GUST's included, which
//!   no public API of `gust-wgpu` exposes. Each cell is measured in two passes: heap with logging off (what a timing build
//!   allocates; wgpu-core copies resource labels into `String`s only to build log arguments), then API calls and the
//!   sequence with the tap on (its heap is kept as `heap_with_tap`). The logger formats into a fixed stack buffer and mutes the allocation counter
//!   while it runs, so it adds no allocation to the counts. It also records the ordered call sequence of one call.
//! - Nothing else: no timer is read.
//!
//! What the API counts are not: `Queue::submit` is wgpu-core's call, one `vkQueueSubmit` or fewer per call (wgpu folds
//! the pending `write_buffer` copies into the next submission); the internal staging buffer of a `Queue::write_buffer`
//! is not a `Device::create_buffer` call and is not counted as a buffer creation here (the `write_buffer` calls are
//! counted on their own line).

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    fmt::Write as _,
    rc::Rc,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use gust_wgpu::HeadlessDevice;
use serde_json::{Value, json};

use crate::{
    Options,
    backend::{Backend, BenchResult},
    gpu_backends,
    group::Group,
    gust_gpu,
    harness::prepare_case,
    meta,
    wgsl_wgpu::{Switch, WgpuContext},
    workloads::{self, Inputs, Workload},
};

// ---------------------------------------------------------------------------------------------------------------
// The counting allocator.

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static LARGEST: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Set while the logger runs on this thread: its own allocations are not counted.
    static MUTE: Cell<bool> = const { Cell::new(false) };
}

fn muted() -> bool {
    MUTE.try_with(Cell::get).unwrap_or(false)
}

struct Mute(bool);

impl Mute {
    fn on() -> Self {
        Mute(MUTE.try_with(|m| m.replace(true)).unwrap_or(false))
    }
}

impl Drop for Mute {
    fn drop(&mut self) {
        let _ = MUTE.try_with(|m| m.set(self.0));
    }
}

fn note_alloc(size: usize) {
    if !muted() {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(size as u64, Relaxed);
        LARGEST.fetch_max(size as u64, Relaxed);
    }
}

pub struct CountingAlloc;

// SAFETY: every method forwards to `System` with the arguments it received; the counters are atomics and the mute flag
// is a const-initialised thread-local `Cell<bool>` without a destructor, so counting allocates nothing and cannot
// re-enter the allocator.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: the caller's contract for `alloc` is passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: the caller's contract for `alloc_zeroed` is passed on unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !muted() {
            FREES.fetch_add(1, Relaxed);
        }
        // SAFETY: `ptr` and `layout` are the caller's, unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !muted() {
            REALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(new_size as u64, Relaxed);
            LARGEST.fetch_max(new_size as u64, Relaxed);
        }
        // SAFETY: `ptr`, `layout` and `new_size` are the caller's, unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

// ---------------------------------------------------------------------------------------------------------------
// The wgpu API tap.

/// The wgpu-core API calls that are counted, by the first word of their log message.
const NAMES: [&str; 12] = [
    "Queue::submit",
    "Device::create_buffer",
    "Buffer::drop",
    "Device::create_bind_group",
    "Device::create_command_encoder",
    "Queue::write_buffer",
    "Device::poll",
    "Buffer::map_async",
    "Buffer::unmap",
    "ComputePass::dispatch",
    "Queue::on_submitted_work_done",
    "Device::create_query_set",
];

static API: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];
/// The ordered tokens of the call being recorded, or `None` when nothing is recording.
static SEQUENCE: Mutex<Option<Vec<String>>> = Mutex::new(None);
/// Every message the tap saw, so a run can tell that the tap works at all.
static SEEN: AtomicU64 = AtomicU64::new(0);

/// A fixed buffer that takes what fits and drops the rest, so formatting a log message allocates nothing.
struct Stack {
    bytes: [u8; 256],
    len: usize,
}

impl std::fmt::Write for Stack {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let room = self.bytes.len() - self.len;
        let mut take = text.len().min(room);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

impl Stack {
    fn text(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

struct Tap;

impl log::Log for Tap {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.target().starts_with("wgpu")
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let _mute = Mute::on();
        let mut stack = Stack {
            bytes: [0; 256],
            len: 0,
        };
        let _ = match record.args().as_str() {
            Some(text) => stack.write_str(text),
            None => stack.write_fmt(*record.args()),
        };
        let text = stack.text();
        let token = text
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("");
        // "Queue::submit returned submit index N" is the same call's second message.
        if token == "Queue::submit" && text.len() > token.len() {
            return;
        }
        SEEN.fetch_add(1, Relaxed);
        if let Some(index) = NAMES.iter().position(|name| *name == token) {
            API[index].fetch_add(1, Relaxed);
        }
        if let Ok(mut guard) = SEQUENCE.lock()
            && let Some(sequence) = guard.as_mut()
        {
            // The whole message (up to 256 bytes), arguments included: labels, indexes, dispatch and copy sizes.
            sequence.push(text.to_owned());
        }
    }

    fn flush(&self) {}
}

static TAP: Tap = Tap;

fn install_tap() -> BenchResult<()> {
    log::set_logger(&TAP).map_err(|e| format!("a logger is already installed: {e}"))?;
    log::set_max_level(log::LevelFilter::Off);
    Ok(())
}

/// Heap pass: logging off, so wgpu-core builds no log arguments (its trace-level sites copy resource labels into
/// `String`s, which the allocator would count). API pass: the tap at `Trace`.
fn set_logging(on: bool) {
    log::set_max_level(if on {
        log::LevelFilter::Trace
    } else {
        log::LevelFilter::Off
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Snapshots and one measured call.

#[derive(Clone, Copy, Default)]
struct Snap {
    allocs: u64,
    bytes: u64,
    reallocs: u64,
    frees: u64,
    largest: u64,
    api: [u64; NAMES.len()],
}

fn snap() -> Snap {
    let mut api = [0; NAMES.len()];
    for (slot, counter) in api.iter_mut().zip(&API) {
        *slot = counter.load(Relaxed);
    }
    Snap {
        allocs: ALLOCS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        reallocs: REALLOCS.load(Relaxed),
        frees: FREES.load(Relaxed),
        largest: LARGEST.load(Relaxed),
        api,
    }
}

fn delta(before: &Snap, after: &Snap) -> Snap {
    let mut api = [0; NAMES.len()];
    for (i, slot) in api.iter_mut().enumerate() {
        *slot = after.api[i] - before.api[i];
    }
    Snap {
        allocs: after.allocs - before.allocs,
        bytes: after.bytes - before.bytes,
        reallocs: after.reallocs - before.reallocs,
        frees: after.frees - before.frees,
        largest: after.largest,
        api,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    EndToEnd,
    DispatchHost,
    /// One `gpu_time(16)` call: a 16-pass batch in one submission with timestamps (what `gpu_batch_per_dispatch` runs).
    Batch16,
}

impl Op {
    fn id(self) -> &'static str {
        match self {
            Op::EndToEnd => "end_to_end",
            Op::DispatchHost => "dispatch_host",
            Op::Batch16 => "batch16",
        }
    }
}

/// min, median and max of a count over the calls.
fn spread(values: &[u64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    json!({
        "min": sorted.first().copied().unwrap_or(0),
        "median": sorted.get(sorted.len() / 2).copied().unwrap_or(0),
        "max": sorted.last().copied().unwrap_or(0),
    })
}

/// A call sequence with runs of one name folded: `["a x2", "b"]`.
fn fold_sequence(tokens: &[String]) -> Vec<String> {
    let mut folded: Vec<(String, usize)> = Vec::new();
    for token in tokens {
        match folded.last_mut() {
            Some((name, count)) if name == token => *count += 1,
            _ => folded.push((token.clone(), 1)),
        }
    }
    folded
        .into_iter()
        .map(|(name, count)| {
            if count > 1 {
                format!("{name} x{count}")
            } else {
                name
            }
        })
        .collect()
}

struct CellSpec<'a> {
    workload: Workload,
    inputs: &'a Inputs,
    reference: &'a workloads::Reference,
    op: Op,
    warm: usize,
    calls: usize,
}

/// One pass of one cell: `warm` unmeasured calls, then `calls` measured ones, with logging `on` or off for the whole
/// pass. Before each call the output is poisoned (as the harness does, outside the counted region). Returns the
/// per-call deltas and, in a logging pass, the whole-message sequence of the first measured call.
fn pass(
    backend: &mut dyn Backend,
    cell: &CellSpec<'_>,
    out: &mut Vec<f32>,
    logging: bool,
) -> BenchResult<(Vec<Snap>, Vec<String>)> {
    set_logging(logging);
    let mut deltas: Vec<Snap> = Vec::with_capacity(cell.calls);
    let mut sequence: Vec<String> = Vec::new();
    for i in 0..cell.warm + cell.calls {
        backend.poison()?;
        out.fill(cell.workload.poison_word());
        let measured = i >= cell.warm;
        let record = logging && i == cell.warm;
        if record && let Ok(mut guard) = SEQUENCE.lock() {
            *guard = Some(Vec::with_capacity(256));
        }
        LARGEST.store(0, Relaxed);
        let before = snap();
        match cell.op {
            Op::EndToEnd => backend.end_to_end(cell.inputs, out)?,
            Op::DispatchHost => backend.dispatch()?,
            Op::Batch16 => {
                backend.gpu_time(16)?;
            }
        }
        let after = snap();
        if record && let Ok(mut guard) = SEQUENCE.lock() {
            sequence = guard.take().unwrap_or_default();
        }
        if measured {
            deltas.push(delta(&before, &after));
        }
    }
    set_logging(false);
    Ok((deltas, sequence))
}

fn heap_json(deltas: &[Snap]) -> Value {
    let column = |f: fn(&Snap) -> u64| deltas.iter().map(f).collect::<Vec<_>>();
    json!({
        "allocs": spread(&column(|d| d.allocs)),
        "reallocs": spread(&column(|d| d.reallocs)),
        "frees": spread(&column(|d| d.frees)),
        "bytes": spread(&column(|d| d.bytes)),
        "largest_single": spread(&column(|d| d.largest)),
    })
}

/// One backend, one workload, one size, one operation, in two passes: heap counts with logging **off** (what a
/// timing build allocates), then API counts and the call sequence with the tap at `Trace`. The second pass also counts
/// heap (`heap_with_tap`), which includes the `String`s wgpu-core builds for its log arguments, so the difference is
/// recorded and the first pass is the one to read.
fn measure(backend: &mut dyn Backend, cell: &CellSpec<'_>) -> BenchResult<Value> {
    let out_len = cell.workload.out_len(cell.inputs.n);
    let mut out: Vec<f32> = vec![cell.workload.poison_word(); out_len];
    let (heap_deltas, _) = pass(backend, cell, &mut out, false)?;
    let (tap_deltas, sequence) = pass(backend, cell, &mut out, true)?;
    // The measured calls must have done the work: check the last result against the reference, outside the counts.
    if cell.op != Op::EndToEnd {
        backend.read_back(&mut out)?;
    }
    workloads::check(cell.workload, cell.inputs, cell.reference, &out).map_err(|e| {
        format!(
            "{} {} {}: {e}",
            backend.id(),
            cell.workload.id(),
            cell.op.id()
        )
    })?;

    let mut api = serde_json::Map::new();
    for (index, name) in NAMES.iter().enumerate() {
        let values = tap_deltas.iter().map(|d| d.api[index]).collect::<Vec<_>>();
        api.insert((*name).to_owned(), spread(&values));
    }
    Ok(json!({
        "backend": backend.id(),
        "workload": cell.workload.id(),
        "n": cell.inputs.n,
        "op": cell.op.id(),
        "calls": cell.calls,
        "warm": cell.warm,
        "heap": heap_json(&heap_deltas),
        "heap_with_tap": heap_json(&tap_deltas),
        "wgpu_api": Value::Object(api),
        "sequence_of_first_measured_call": fold_sequence(&sequence),
    }))
}

// ---------------------------------------------------------------------------------------------------------------

const DEFAULT_CALLS: usize = 25;
const WARM_CALLS: usize = 5;

pub fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    install_tap()?;
    let device = Rc::new(HeadlessDevice::new()?);
    let ctx = Rc::new(WgpuContext::new()?);
    let adapter = device.adapter_info();
    if adapter.name != ctx.info.name {
        return Err(format!(
            "GUST ran on `{}` but the hand-WGSL device on `{}`; refusing to compare",
            adapter.name, ctx.info.name
        )
        .into());
    }
    let calls = options.samples.unwrap_or(DEFAULT_CALLS).max(1);
    let workload_list = options
        .workloads
        .clone()
        .unwrap_or_else(|| [Workload::VectorAdd, Workload::Saxpy].to_vec());

    // GUST first, then the whole hand-WGSL ladder: percall, prebuilt, each switch alone and all-on.
    let mut backends: Vec<Box<dyn Backend>> =
        vec![Box::new(gust_gpu::GustGpu::new(Rc::clone(&device)))];
    backends.extend(gpu_backends(
        Group::Ablation,
        &device,
        &ctx,
        &Switch::ALL,
        true,
    ));
    if let Some(wanted) = &options.backends {
        for name in wanted {
            if !backends.iter().any(|b| b.id() == name) {
                return Err(format!("unknown or unavailable backend {name}").into());
            }
        }
        backends.retain(|b| wanted.iter().any(|w| w == b.id()));
    }

    let mut cells = Vec::new();
    for workload in workload_list {
        let sizes = options
            .sizes
            .clone()
            .unwrap_or_else(|| workload.sizes().to_vec());
        for n in sizes {
            let (inputs, reference) = prepare_case(n, workload);
            for backend in &mut backends {
                backend.prepare(workload, &inputs)?;
                for op in [Op::EndToEnd, Op::DispatchHost, Op::Batch16] {
                    let cell = CellSpec {
                        workload,
                        inputs: &inputs,
                        reference: &reference,
                        op,
                        warm: WARM_CALLS,
                        calls,
                    };
                    let record = measure(backend.as_mut(), &cell)?;
                    // The tap must be live: a submitting call that logged no submission means no log reached it.
                    if record["wgpu_api"]["Queue::submit"]["max"] == 0 {
                        return Err(format!(
                            "{} {} n={n} {}: no `Queue::submit` was logged; the wgpu log tap is not receiving",
                            backend.id(),
                            workload.id(),
                            op.id()
                        )
                        .into());
                    }
                    eprintln!(
                        "{} {} n={n} {}: allocs median {} submissions median {}",
                        backend.id(),
                        workload.id(),
                        op.id(),
                        record["heap"]["allocs"]["median"],
                        record["wgpu_api"]["Queue::submit"]["median"],
                    );
                    cells.push(record);
                }
            }
        }
    }

    let document = json!({
        "kind": "gust-bench profile (host-side counts per call, not timings)",
        "adapter": adapter.name,
        "gust_backend_id": gust_gpu::backend_id(),
        "features": meta::build_features(),
        "gust_profile": gust::profile::NAME,
        "calls_per_cell": calls,
        "warm_calls_per_cell": WARM_CALLS,
        "log_messages_seen": SEEN.load(Relaxed),
        "notes": [
            "Counts per call, warm; min/median/max over the measured calls. No timer is read; other lanes ran at the same time, which does not affect counts.",
            "heap: Rust global-allocator counts only (driver allocations unseen), all threads of the process, measured in a pass with logging Off; heap_with_tap is a second pass with the tap at Trace and includes wgpu-core's log-argument Strings (informational, not the number to read).",
            "wgpu_api: wgpu-core api_log messages seen through the log crate; Queue::submit is one wgpu-core call, not one vkQueueSubmit per se.",
            "A Queue::write_buffer's internal staging buffer is not a Device::create_buffer call.",
            "sequence_of_first_measured_call is not command order: wgpu-core logs pass contents (set_bind_group, dispatch) when the encoder finishes but copies when called; set_pipeline, timestamp writes, resolve_query_set and barriers are not logged. It cannot show the timestamp span.",
            "The profile binary has wgpu/trace compiled in (trace off at run time) when built with the wgpu-trace feature; the build.features list says which features were on.",
        ],
        "cells": cells,
    });
    std::fs::write(out, serde_json::to_string_pretty(&document)?)?;
    eprintln!("wrote {out}");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_is_min_median_max_of_the_calls() {
        let v = spread(&[5, 1, 9, 3, 7]);
        assert_eq!(
            (v["min"].as_u64(), v["median"].as_u64(), v["max"].as_u64()),
            (Some(1), Some(5), Some(9))
        );
    }

    #[test]
    fn a_sequence_folds_runs_of_one_name() {
        let tokens = ["a", "a", "b", "a"].map(str::to_owned);
        assert_eq!(fold_sequence(&tokens), ["a x2", "b", "a"]);
    }

    #[test]
    fn the_stack_buffer_truncates_on_a_char_boundary() {
        let mut stack = Stack {
            bytes: [0; 256],
            len: 0,
        };
        let long = "é".repeat(100);
        stack.write_str(&long).unwrap();
        assert!(stack.len <= 256 && stack.text().chars().all(|c| c == 'é'));
    }

    /// The allocator counts a heap allocation (other test threads may add more, never fewer), and the mute flag is
    /// scoped to its guard.
    #[test]
    fn the_counting_allocator_counts_and_the_mute_flag_is_scoped() {
        let before = ALLOCS.load(Relaxed);
        let kept = std::hint::black_box(vec![0u8; 1024]);
        assert!(
            ALLOCS.load(Relaxed) > before,
            "an allocation was not counted"
        );
        drop(kept);
        assert!(!muted());
        {
            let _mute = Mute::on();
            assert!(muted());
        }
        assert!(!muted());
    }
}
