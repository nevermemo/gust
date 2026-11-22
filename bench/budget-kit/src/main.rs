//! A GPU budget kit: how much compute work of a few common shapes fits in one 16.7 ms frame on this machine's GPU,
//! written with GUST's public API only.
//!
//! ```text
//! GUST_PROFILE=dev cargo run --release --manifest-path bench/Cargo.toml -p budget-kit -- --out PATH [--small]
//! ```
//!
//! `--out PATH` is required (there is no default path) and receives the same text that is printed. `--small` runs only
//! the smallest size of each workload. The output is a one-machine design aid, not a benchmark: it asserts no time.
//!
//! | # | Workload | Shape |
//! | --- | --- | --- |
//! | W1 | Copy (reference) | read one array, write another |
//! | W2 | Field diffusion | 2D grid of four `f32` channels; each cell blends its four neighbours and decays |
//! | W3 | Particle step | position and velocity in `f32`; sample the W2 field at the particle's cell, move, wrap |
//! | W4 | Spatial binning | count particles per cell with atomics, then scan the counts; a second row adds the scatter of the particle indices (each takes its slot from the old value of an atomic add; the order inside a cell is unspecified) |
//! | W5 | Reduction | `f32` sum with `gust-algo`, against `bound::sum_f64` within the documented bound |
//! | W6 | Readback delay | submit a small kernel and wait until its result is on the CPU |
//! | W7 | Composite step | one batch per step: diffuse a 1024 x 1024 field, step the particles, clear and count their cells; a sustained run reporting the GPU time's median, p95 and maximum and two host-side costs |
//! | W8 | Full step with lifecycle | one step on a 1024 x 1024 field: diffuse, move the particles over the field, bin them (clear, count, scan, scatter), deposit a fixed amount per particle into a `u32` grid with an atomic add and fold the grid into the field, then retire some particles by a hash, spawn a fixed number and keep the live set dense with a scan of alive flags and a compaction; a sustained run as W7, with the storage buffers each kernel binds in the note, the GPU time of each of its four batches and the median of each of its kernels run as one-kernel batches (a separate pass each) |
//! | W8a | W8 without bins | W8 with no count, no scan of the counts and no scatter |
//! | W8b | W8 with coarse bins | W8 with the bins on a 256 x 256 grid over the same 1024 x 1024 field |
//! | W8c | W8 with a fused particle kernel | W8 with the step, the count and the deposit in one particle kernel |
//! | W8d | W8 with a sparse lifecycle | W8 with the lifecycle (retire, flag scan, compaction, spawn) on every 4th step; the time is the amortised mean per step |
//!
//! Every row is checked against a CPU computation first; a row that fails its check prints FAILED instead of a time.
//! The median of 30 steps after 5 warm-up steps is reported, from GPU timestamps where the device has them and from the
//! wall clock around a blocking submit otherwise; the table says which, per row.
//!
//! The kernels index a flat array through a grid of rows of `ROW` (8192) invocations (`id.y * ROW + id.x`), so a
//! dispatch never needs more than 65 535 workgroups in one dimension, and every buffer is bound as an `Independent`
//! resident vector. A workload whose largest buffer does not fit the device's storage binding limit is cut and the table says so.
//!
//! W4 puts its own clear and count batch and the `gust-algo` scan in one `timed` scope (the batch goes through
//! `dispatch_batch_in_scope`), so one scope's GPU time covers both; the scatter row adds the scatter batch to the same
//! scope. W7 runs 600 counted steps (120 with `--small`) twice, once through the blocking `dispatch_batch_timed` (GPU
//! time and `host_elapsed`) and once through `submit_batch`, timing only the call and waiting for the job afterwards.
//! W8 (slice m11.kit4) runs the same number of counted steps as W7, at 2^17 and 2^20 particles of capacity (2^17 only
//! with `--small`). A step is four batches in one `timed` scope: the diffusion, the move, the clears, the count, the
//! deposit, the fold and the alive flags; the scan of the counts; the scan of the alive flags; the scatter, the
//! compaction and the spawn. The live counts come from a host model of the retire rule (an integer function of the
//! step and the slot), not from a readback. The helpers `lb_mix` and `lb_hash` (and the GPU forms of them) are
//! copied from `examples/walker-life/src/app.rs`; the retire flags, the scan and the compaction follow its pattern.
//! Its note lists the storage buffers each kernel binds, counted from the generated descriptors.
//!
//! Slice m11.cost asks where that step spends its time. The W8 row's note gains two passes run after the sustained
//! runs: each of the four batches in a `timed` scope of its own, and each kernel (the scans as one scope each) as a
//! one-kernel batch in a scope of its own, with the median of each and its share; the sum of one-kernel batches may
//! differ from the fused batches. The rows W8a to W8d change one thing each (no bins; bins on a 256 x 256 grid; the
//! step, the count and the deposit in one kernel; the lifecycle on every 4th step) and are checked against the host as
//! W8 is.
//!
//! After the table a probe (not a workload) launches a linear kernel with 65 536 workgroups on x and prints one `G4:`
//! line with what happened (a refusal is gust-wgpu's own host-side check against the device limits, made before
//! anything is submitted; since slice m11.g4b the host splits the launch and the success is compared with the host),
//! then shows the remedy: a two-dimensional launch of a count that is not a multiple of the row width, with a tail
//! guard, whose values and untouched sentinel slots are checked exactly against the host.

mod g4;
mod kernels;
mod measure;
mod report;
mod setup;
mod w1;
mod w2;
mod w3;
mod w4;
mod w5;
mod w6;
mod w7;
mod w8;
mod w8_model;

use std::fmt::Write as _;

use gust_wgpu::{DeviceHandle, HeadlessDevice};

use crate::g4::g4;
use crate::report::{Row, variant_text_line};
use crate::setup::{W1_SIZES, W2_SIDES, W3_SIZES, W4_SIZES, W5_SIZES, W6_SIZES};
use crate::w1::w1;
use crate::w2::w2;
use crate::w3::w3;
use crate::w4::{w4, w4_scatter};
use crate::w5::w5;
use crate::w6::w6;
use crate::w7::{W7_SIZES, W7_STEPS, W7_STEPS_SMALL, w7};
use crate::w8::{Var, w8};
use crate::w8_model::W8_SIZES;

// ---- the run ----

struct Args {
    out: String,
    small: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut out = None;
    let mut small = false;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--small" => small = true,
            "--out" => out = Some(iter.next().ok_or("--out needs a path")?),
            other => {
                return Err(format!(
                    "unknown argument {other:?}; usage: budget-kit --out PATH [--small]"
                ));
            }
        }
    }
    Ok(Args {
        out: out.ok_or("--out PATH is required")?,
        small,
    })
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("budget-kit: {message}");
            std::process::exit(2);
        }
    };
    // Opens the device with wgpu's default limits (`LimitsPolicy::Default`); `KIT_LIMITS_POLICY` names that in the G4 line.
    let device =
        DeviceHandle::new(HeadlessDevice::new().expect("no usable adapter on the Vulkan backend"));
    let mut text = report::header(args.small, &device);
    // The adapter, the limits and the table header go out before the first row, which can take a while.
    print!("{text}");
    let pick = |sizes: &[u32]| -> Vec<u32> {
        if args.small {
            sizes[..1].to_vec()
        } else {
            sizes.to_vec()
        }
    };
    // A boxed closure, so a workload can carry data (a flag, a variant) and a row variant needs no copy of its function.
    type Workload = Box<dyn Fn(&DeviceHandle, u32) -> Row>;
    let plan: [(Workload, Vec<u32>); 7] = [
        (Box::new(w1), pick(&W1_SIZES)),
        (Box::new(w2), pick(&W2_SIDES)),
        (Box::new(w3), pick(&W3_SIZES)),
        (Box::new(w5), pick(&W5_SIZES)),
        (Box::new(w4), pick(&W4_SIZES)),
        (Box::new(w4_scatter), pick(&W4_SIZES)),
        (Box::new(w6), pick(&W6_SIZES)),
    ];
    let mut failed = 0;
    let mut emit = |row: Row| {
        failed += usize::from(row.failed());
        let line = row.text();
        println!("{line}");
        let _ = writeln!(text, "{line}");
    };
    for (workload, sizes) in plan {
        for size in sizes {
            emit(workload(&device, size));
        }
    }
    // W7 runs a sustained number of steps, fewer with `--small`.
    let w7_steps = if args.small { W7_STEPS_SMALL } else { W7_STEPS };
    for size in pick(&W7_SIZES) {
        emit(w7(&device, size, w7_steps));
    }
    // W8 (m11.kit4) is sustained as well, with the same step counts.
    // The variants of W8 (m11.cost) follow it, each at the same sizes and step counts.
    for var in [
        Var::Base,
        Var::NoBins,
        Var::CoarseBins,
        Var::Fused,
        Var::Sparse,
    ] {
        for size in pick(&W8_SIZES) {
            emit(w8(&device, size, w7_steps, var));
        }
    }
    let _ = writeln!(text);
    let (g4_lines, g4_failed) = g4(&device);
    failed += usize::from(g4_failed);
    for line in g4_lines {
        println!("{line}");
        let _ = writeln!(text, "{line}");
    }
    // The cost of the split variants (m11.g4b): the Slang text every kernel of this kit carries beside its own.
    let line = variant_text_line();
    println!("{line}");
    let _ = writeln!(text, "{line}");
    let _ = writeln!(text);
    let _ = writeln!(text, "{failed} row(s) FAILED");
    println!("\n{failed} row(s) FAILED");
    std::fs::write(&args.out, &text).unwrap_or_else(|error| panic!("{}: {error}", args.out));
    println!("table written to {}", args.out);
    if failed > 0 {
        std::process::exit(1);
    }
}
