//! A GPU budget kit: how much compute work of a few common shapes fits in one 16.7 ms frame on this machine's GPU,
//! written with GUST's public API only.
//!
//! ```text
//! GUST_PROFILE=dev cargo run --release -p budget-kit -- --out PATH [--small]
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

use std::fmt::Write as _;
use std::time::Instant;

use gust::{Extent, Independent, gpu, math::Vec4};
use gust_algo::{bound, dispatch_batch_in_scope, reduce, scan, timed};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec, HeadlessDevice};

#[gpu]
mod kit {
    pub struct Params {
        pub n: u32,
        pub side: u32,
        pub pad0: u32,
        pub pad1: u32,
    }

    // The width of the invocation grid the kernels index through (`id.y * ROW + id.x`); the host reads the same
    // constant (m11.g3) to build `Extent::new(ROW, rows, 1)`.
    pub const ROW: u32 = 8192;

    // The flat index of an invocation in a grid of rows of `ROW`.
    fn linear(x: u32, y: u32) -> u32 {
        y * ROW + x
    }

    // `a` wrapped once into [0, side) (a is below 2 * side).
    fn wrap_index(a: u32, side: u32) -> u32 {
        if a >= side { a - side } else { a }
    }

    // The cell of a position in [0, side], clamped to the last cell.
    fn cell_of(v: f32, side: u32) -> u32 {
        let c = v as u32;
        if c >= side { side - 1u32 } else { c }
    }

    // `v` wrapped once into [0, side] (v is in [-side, 2 * side)).
    fn wrap_pos(v: f32, side: f32) -> f32 {
        let mut w = v;
        if w >= side {
            w -= side;
        }
        if w < 0.0f32 {
            w += side;
        }
        w
    }

    // W1: read one array, write another.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w1_copy(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            out[i] = input[i];
        }
    }

    // W2: one diffusion step of a `side x side` grid of four channels, wrapping at the edges. Each cell keeps half of
    // itself, takes an eighth of each of its four neighbours, and decays by 0.2 percent.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w2_diffuse(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let x = i % p.side;
            let y = i / p.side;
            let xl = wrap_index(x + p.side - 1u32, p.side);
            let xr = wrap_index(x + 1u32, p.side);
            let yu = wrap_index(y + p.side - 1u32, p.side);
            let yd = wrap_index(y + 1u32, p.side);
            let c = input[i];
            let l = input[y * p.side + xl];
            let r = input[y * p.side + xr];
            let u = input[yu * p.side + x];
            let d = input[yd * p.side + x];
            out[i] = (c * 0.5f32 + (l + r + u + d) * 0.125f32) * 0.998f32;
        }
    }

    // W3: one step of a particle (x, y, vx, vy in the four lanes): sample the field at the particle's cell, accelerate,
    // move, wrap into [0, side). The field is `side x side`.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w3_step(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        field: StructuredBuffer<Vec4>,
        mut parts: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let side = p.side as f32;
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let f = field[cy * p.side + cx];
            let vx = q.z + (f.x - 0.5f32) * 0.05f32;
            let vy = q.w + (f.y - 0.5f32) * 0.05f32;
            let x = wrap_pos(q.x + vx, side);
            let y = wrap_pos(q.y + vy, side);
            parts[i] = Vec4::new(x, y, vx, vy);
        }
    }

    // W4: zero the per-cell counts (`n` is the number of cells).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_clear(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            counts[i] = 0u32;
        }
    }

    // W4: count the particles of each cell of a `side x side` grid with an atomic add. Positions are in [0, side).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_count(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            atomic_add(&mut counts[cy * p.side + cx], 1u32);
        }
    }

    // W4: the scatter. Each particle takes the next free slot of its cell (`old` is the cell's cursor before the add,
    // which starts at zero) and writes its own index at `starts[cell] + old`. `starts` is the exclusive scan of the
    // counts. Which particle gets which slot is schedule-dependent: the order inside a cell is unspecified.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w4_scatter(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        starts: StructuredBuffer<u32>,
        mut cursor: RWStructuredBuffer<u32>,
        mut order: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let c = cy * p.side + cx;
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }

    // G4 probe: a linear launch, `i = id.x`; one invocation per element, so the workgroup count on x is n / 256.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn g4_linear(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < p.n {
            out[i] = i.wrapping_mul(3u32).wrapping_add(1u32);
        }
    }

    // G4 remedy: the same elements through a two-dimensional launch of rows of `p.side` invocations, with the tail
    // guard the kernel owns (the extent is rounded up to whole workgroups in x and y).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn g4_grid(id: SV_DispatchThreadID, p: Uniform<Params>, mut out: RWStructuredBuffer<u32>) {
        let i = id.y * p.side + id.x;
        if i >= p.n {
            return;
        }
        out[i] = i.wrapping_mul(3u32).wrapping_add(1u32);
    }

    // W6: the small kernel whose result the host waits for.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w6_plus_one(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            out[i] = input[i].wrapping_add(1u32);
        }
    }

    // ---- m11.kit4: W8, a full step with a lifecycle ----

    // The parameters of a lifecycle kernel, eight `u32` (32 bytes): the capacity of the particle buffers, the live
    // count before the step, the step number, the survivors of the step and the particles that spawn (both known to the
    // host from its model), the field side, and two pad fields.
    pub struct LifeParams {
        pub cap: u32,
        pub live: u32,
        pub step: u32,
        pub surv: u32,
        pub spawn: u32,
        pub side: u32,
        pub pad0: u32,
        pub pad1: u32,
    }

    // The amount every live particle adds to its cell of the deposit grid.
    pub const DEPOSIT: u32 = 8;
    // A particle retires when the low 16 bits of its hash draw are below this (over 2^16): one in sixteen.
    pub const RETIRE: u32 = 4096;
    // Seeds of the retire draw and of the spawn draw.
    pub const RETIRE_SEED: u32 = 3735928559;
    pub const SPAWN_SEED: u32 = 2654435769;

    // The "lowbias32" finaliser (Chris Wellons' hash-prospector, public domain), as in `examples/walker-life`.
    fn lb_mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16u32);
        h = h.wrapping_mul(2146121005u32);
        h = h ^ (h >> 15u32);
        h = h.wrapping_mul(2221713035u32);
        h ^ (h >> 16u32)
    }

    // A hash of a seed and an index, as in `examples/walker-life`.
    fn lb_hash(seed: u32, index: u32) -> u32 {
        lb_mix(lb_mix(seed) ^ index)
    }

    // A value in [0, 1), exact in `f32` (the high 24 bits of the hash over 2^24).
    fn unit_of(h: u32) -> f32 {
        ((h >> 8u32) as f32) / 16777216.0f32
    }

    // W8: the deposit. Every particle adds `DEPOSIT` to its cell of a `side x side` grid of `u32` with an atomic add.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_deposit(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        parts: StructuredBuffer<Vec4>,
        mut dep: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            atomic_add(&mut dep[cy * p.side + cx], DEPOSIT);
        }
    }

    // W8: folds the deposit grid into the field: lane w of each cell gains its deposit over 1024.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_fold(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        dep: StructuredBuffer<u32>,
        mut field: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let f = field[i];
            field[i] = Vec4::new(f.x, f.y, f.z, f.w + (dep[i] as f32) / 1024.0f32);
        }
    }

    // W8: the alive flags, one invocation per slot of the capacity. A live slot (`i < live`) is alive unless the
    // hash of (step, slot) says it retires; a slot past `live` is not alive, so the scan over the capacity counts
    // the live particles only.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_retire(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        mut alive: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.cap {
            let mut a = 0u32;
            if i < p.live {
                let r = lb_hash(lb_hash(RETIRE_SEED, p.step), i);
                if (r & 65535u32) >= RETIRE {
                    a = 1u32;
                }
            }
            alive[i] = a;
        }
    }

    // W8: the compaction. Each live slot that is alive writes its particle to the slot `starts[i]` of the second
    // buffer, `starts` being the exclusive scan of the alive flags: the survivors stay in order and dense.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_compact(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        parts: StructuredBuffer<Vec4>,
        alive: StructuredBuffer<u32>,
        starts: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.live && alive[i] == 1u32 {
            out[starts[i]] = parts[i];
        }
    }

    // W8: the spawn. Particle `k` of the step takes the slot `surv + k` after the survivors, if that is inside the
    // capacity (the rest are dropped, so the population is clamped at the capacity), with a position in [0, side)
    // and a velocity in [-1, 1) drawn from the hash of (step, k).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_spawn(
        id: SV_DispatchThreadID,
        p: Uniform<LifeParams>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let k = linear(id.x, id.y);
        let t = p.surv + k;
        if k < p.spawn && t < p.cap {
            let h = lb_hash(lb_hash(SPAWN_SEED, p.step), k);
            let side = p.side as f32;
            out[t] = Vec4::new(
                unit_of(lb_hash(h, 0u32)) * side,
                unit_of(lb_hash(h, 1u32)) * side,
                unit_of(lb_hash(h, 2u32)) * 2.0f32 - 1.0f32,
                unit_of(lb_hash(h, 3u32)) * 2.0f32 - 1.0f32,
            );
        }
    }

    // ---- m11.cost: kernels of the W8 variants ----

    // The parameters of a bin kernel on a grid coarser than the field: the particle count, the bins per side and the
    // field side (the positions are in [0, field]).
    pub struct BinParams {
        pub n: u32,
        pub bins: u32,
        pub field: u32,
        pub pad0: u32,
    }

    // The bin of a position in [0, field]: the position scaled to a `bins` wide grid, clamped to the last bin. The scale
    // is exact when both sides are powers of two, so the host computes the same bin from the same position.
    fn bin_of(v: f32, bins: u32, field: u32) -> u32 {
        cell_of(v * ((bins as f32) / (field as f32)), bins)
    }

    // W8b: `w4_count` on a `bins x bins` grid over the field.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_bin_count(
        id: SV_DispatchThreadID,
        p: Uniform<BinParams>,
        parts: StructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = bin_of(q.x, p.bins, p.field);
            let cy = bin_of(q.y, p.bins, p.field);
            atomic_add(&mut counts[cy * p.bins + cx], 1u32);
        }
    }

    // W8b: `w4_scatter` on a `bins x bins` grid over the field.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_bin_scatter(
        id: SV_DispatchThreadID,
        p: Uniform<BinParams>,
        parts: StructuredBuffer<Vec4>,
        starts: StructuredBuffer<u32>,
        mut cursor: RWStructuredBuffer<u32>,
        mut order: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let cx = bin_of(q.x, p.bins, p.field);
            let cy = bin_of(q.y, p.bins, p.field);
            let c = cy * p.bins + cx;
            let old = atomic_add(&mut cursor[c], 1u32);
            order[starts[c] + old] = i;
        }
    }

    // W8c: the particle step of `w3_step`, then the count of `w4_count` and the deposit of `w8_deposit` at the new
    // position, in one kernel.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn w8_step_count_deposit(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        field: StructuredBuffer<Vec4>,
        mut parts: RWStructuredBuffer<Vec4>,
        mut counts: RWStructuredBuffer<u32>,
        mut dep: RWStructuredBuffer<u32>,
    ) {
        let i = linear(id.x, id.y);
        if i < p.n {
            let q = parts[i];
            let side = p.side as f32;
            let cx = cell_of(q.x, p.side);
            let cy = cell_of(q.y, p.side);
            let f = field[cy * p.side + cx];
            let vx = q.z + (f.x - 0.5f32) * 0.05f32;
            let vy = q.w + (f.y - 0.5f32) * 0.05f32;
            let x = wrap_pos(q.x + vx, side);
            let y = wrap_pos(q.y + vy, side);
            parts[i] = Vec4::new(x, y, vx, vy);
            let nx = cell_of(x, p.side);
            let ny = cell_of(y, p.side);
            atomic_add(&mut counts[ny * p.side + nx], 1u32);
            atomic_add(&mut dep[ny * p.side + nx], DEPOSIT);
        }
    }
}

// ---- the plan ----

/// Steps counted per cell, and steps run first and not counted.
const STEPS: usize = 30;
const WARMUP: usize = 5;
/// One frame at 60 Hz, in milliseconds.
const FRAME_MS: f64 = 1000.0 / 60.0;
// The width of the invocation grid the kernels index through: the module-level constant of the `#[gpu]` module.
use kit::ROW;

/// Particle and field sizes (the counts are exact; "1M" is 2^20 and "4M" is 2^22).
const W1_SIZES: [u32; 3] = [1 << 20, 1 << 22, 1 << 24];
const W2_SIDES: [u32; 3] = [512, 1024, 2048];
const W3_SIZES: [u32; 3] = [100_000, 1 << 20, 1 << 22];
const W4_SIZES: [u32; 3] = [100_000, 1 << 20, 1 << 22];
const W5_SIZES: [u32; 3] = [1 << 20, 1 << 22, 1 << 24];
const W6_SIZES: [u32; 3] = [1, 4096, 1 << 20];
/// The field side W3 samples, and the grid side W4 bins into.
const W3_FIELD_SIDE: u32 = 1024;
const W4_GRID_SIDE: u32 = 256;
/// Steps of W2 run before its result is compared with the CPU.
const W2_CHECKED_STEPS: usize = 3;

fn extent(n: u32) -> Extent {
    Extent::new(ROW, n.div_ceil(ROW), 1)
}

fn dbg<E: std::fmt::Debug>(error: E) -> String {
    format!("{error:?}")
}

// ---- host data ----

fn mix(i: u32) -> u32 {
    let mut h = i.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_507);
    h ^ (h >> 13)
}

/// A value in [0, 1), exact in `f32` (24 bits).
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn field_init(i: u32) -> Vec4 {
    Vec4::new(
        unit(mix(i * 4)),
        unit(mix(i * 4 + 1)),
        unit(mix(i * 4 + 2)),
        unit(mix(i * 4 + 3)),
    )
}

/// A particle with its position in [0, side) and its velocity in [-1, 1).
fn particle_init(i: u32, side: u32) -> Vec4 {
    let s = side as f32;
    Vec4::new(
        unit(mix(i * 4)) * s,
        unit(mix(i * 4 + 1)) * s,
        unit(mix(i * 4 + 2)) * 2.0 - 1.0,
        unit(mix(i * 4 + 3)) * 2.0 - 1.0,
    )
}

fn lanes(v: Vec4) -> [f32; 4] {
    [v.x, v.y, v.z, v.w]
}

// ---- CPU references ----

/// The expression of `w2_diffuse`, in the same order, on the host.
fn cpu_diffuse(input: &[Vec4], side: u32) -> Vec<Vec4> {
    let at = |x: u32, y: u32| lanes(input[(y * side + x) as usize]);
    (0..side * side)
        .map(|i| {
            let (x, y) = (i % side, i / side);
            let c = at(x, y);
            let l = at((x + side - 1) % side, y);
            let r = at((x + 1) % side, y);
            let u = at(x, (y + side - 1) % side);
            let d = at(x, (y + 1) % side);
            let lane = |k: usize| (c[k] * 0.5 + (l[k] + r[k] + u[k] + d[k]) * 0.125) * 0.998;
            Vec4::new(lane(0), lane(1), lane(2), lane(3))
        })
        .collect()
}

/// The expression of `w3_step`, on the host, for one particle.
fn cpu_particle(q: Vec4, field: &[Vec4], side: u32) -> Vec4 {
    let s = side as f32;
    let cx = (q.x as u32).min(side - 1);
    let cy = (q.y as u32).min(side - 1);
    let f = field[(cy * side + cx) as usize];
    let vx = q.z + (f.x - 0.5) * 0.05;
    let vy = q.w + (f.y - 0.5) * 0.05;
    let mut x = q.x + vx;
    let mut y = q.y + vy;
    if x >= s {
        x -= s;
    }
    if x < 0.0 {
        x += s;
    }
    if y >= s {
        y -= s;
    }
    if y < 0.0 {
        y += s;
    }
    Vec4::new(x, y, vx, vy)
}

/// The per-cell counts of particles, on the host.
fn cpu_counts(parts: &[Vec4], side: u32) -> Vec<u32> {
    let mut counts = vec![0u32; (side * side) as usize];
    for q in parts {
        let cx = (q.x as u32).min(side - 1);
        let cy = (q.y as u32).min(side - 1);
        counts[(cy * side + cx) as usize] += 1;
    }
    counts
}

/// The bin of a position on a `bins x bins` grid over a field of side `field`: the host's form of `bin_of` (the same
/// scale, exact for powers of two). With `bins == field` it is the cell of `cpu_counts`.
fn bin_of_host(q: &Vec4, bins: u32, field: u32) -> usize {
    let scale = bins as f32 / field as f32;
    let cx = ((q.x * scale) as u32).min(bins - 1);
    let cy = ((q.y * scale) as u32).min(bins - 1);
    (cy * bins + cx) as usize
}

/// The per-bin counts of particles on a `bins x bins` grid over a field of side `field`, on the host.
fn cpu_bin_counts(parts: &[Vec4], bins: u32, field: u32) -> Vec<u32> {
    let mut counts = vec![0u32; (bins * bins) as usize];
    for q in parts {
        counts[bin_of_host(q, bins, field)] += 1;
    }
    counts
}

fn cpu_exclusive(counts: &[u32]) -> Vec<u32> {
    let mut sum = 0u32;
    counts
        .iter()
        .map(|c| {
            let before = sum;
            sum += c;
            before
        })
        .collect()
}

// ---- measuring ----

/// One step's time: the GPU's own, where the device has timestamps, and the wall clock around the step.
struct Sample {
    gpu_ms: Option<f64>,
    wall_ms: f64,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// What a measured row reports.
struct Measured {
    ms: f64,
    method: &'static str,
}

/// The median of `STEPS` steps after `WARMUP` that are not counted. The GPU time is used only if every counted step
/// has one; otherwise the wall clock is, for all of them.
fn measure(mut step: impl FnMut() -> Result<Sample, String>) -> Result<Measured, String> {
    for _ in 0..WARMUP {
        step()?;
    }
    let mut gpu = Vec::new();
    let mut wall = Vec::new();
    let mut every_gpu = true;
    for _ in 0..STEPS {
        let sample = step()?;
        wall.push(sample.wall_ms);
        match sample.gpu_ms {
            Some(ms) => gpu.push(ms),
            None => every_gpu = false,
        }
    }
    Ok(if every_gpu {
        Measured {
            ms: median(&gpu),
            method: "gpu timestamps",
        }
    } else {
        Measured {
            ms: median(&wall),
            method: "wall clock",
        }
    })
}

/// Dispatches a batch once, timed.
fn dispatch(device: &HeadlessDevice, batch: &[BufferDispatch<'_>]) -> Result<Sample, String> {
    let timing = device.dispatch_batch_timed(batch).map_err(dbg)?;
    Ok(Sample {
        gpu_ms: timing.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
        wall_ms: timing.host_elapsed.as_secs_f64() * 1e3,
    })
}

// ---- the table ----

/// What one workload size produced: a table row's pieces.
struct Row {
    workload: &'static str,
    size: String,
    /// A measured time and the bytes one step moves (`None` where a rate has no meaning), or the reason it has none.
    result: Result<(Measured, Option<f64>), String>,
    note: String,
}

impl Row {
    fn text(&self) -> String {
        match &self.result {
            Ok((measured, bytes)) => {
                let rate = bytes.map_or_else(
                    || "n/a".to_owned(),
                    |b| format!("{:.1}", b / (measured.ms * 1e-3) / 1e9),
                );
                let frame = if self.workload.starts_with("W6") {
                    "n/a".to_owned()
                } else {
                    format!("{:.1}", FRAME_MS / measured.ms)
                };
                format!(
                    "| {} | {} | {} | {:.4} | {} | {} | ok | {} |",
                    self.workload, self.size, measured.method, measured.ms, rate, frame, self.note
                )
            }
            Err(reason) => format!(
                "| {} | {} | n/a | n/a | n/a | n/a | {} | {} |",
                self.workload,
                self.size,
                if reason.starts_with("skipped") {
                    "skipped"
                } else {
                    "FAILED"
                },
                if reason.starts_with("skipped") {
                    reason.clone()
                } else {
                    format!("{reason}; {}", self.note)
                }
            ),
        }
    }

    fn failed(&self) -> bool {
        matches!(&self.result, Err(reason) if !reason.starts_with("skipped"))
    }
}

/// Why a buffer of `bytes` cannot be bound on this device, or `None`.
fn skip_reason(device: &HeadlessDevice, bytes: u64) -> Option<String> {
    let limits = device.limits();
    (bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size).then(|| {
        format!(
            "skipped: {bytes} bytes in one buffer exceed the device's storage binding or buffer limit"
        )
    })
}

// ---- workloads ----

fn w1(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W1 copy",
        size: format!("{n} u32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<u32> = (0..n).map(mix).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let out = GpuVec::<u32>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w1",
                &kit::Params {
                    n,
                    side: 0,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call =
            kit::w1_copy::call_extent(extent(n), &params, Independent(&input), Independent(&out));
        let batch = [BufferDispatch::from(&call)];
        dispatch(device, &batch)?;
        if out.to_vec().map_err(dbg)? != host {
            return Err("the copy differs from the input".into());
        }
        let measured = measure(|| dispatch(device, &batch))?;
        Ok((measured, Some(8.0 * f64::from(n))))
    })();
    row.note = "bytes: one read plus one write per element".into();
    row
}

fn w2(device: &DeviceHandle, side: u32) -> Row {
    let n = side * side;
    let mut row = Row {
        workload: "W2 field diffusion",
        size: format!("{side}x{side}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 16) {
            return Err(reason);
        }
        let host: Vec<Vec4> = (0..n).map(field_init).collect();
        let a = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let b = GpuVec::<Vec4>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w2",
                &kit::Params {
                    n,
                    side,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let a_to_b =
            kit::w2_diffuse::call_extent(extent(n), &params, Independent(&a), Independent(&b));
        let b_to_a =
            kit::w2_diffuse::call_extent(extent(n), &params, Independent(&b), Independent(&a));
        let batches = [
            [BufferDispatch::from(&a_to_b)],
            [BufferDispatch::from(&b_to_a)],
        ];
        // The checked steps alternate from `a`; with an odd count the result is in `b`.
        let mut done = 0usize;
        let mut cpu = host;
        for _ in 0..W2_CHECKED_STEPS {
            dispatch(device, &batches[done % 2])?;
            done += 1;
            cpu = cpu_diffuse(&cpu, side);
        }
        let gpu = if done % 2 == 1 { &b } else { &a }.to_vec().map_err(dbg)?;
        // No cancellation (every operand is non-negative), so the relative error of each step is at most five
        // roundings, each 2^-23 on the GPU and 2^-24 on the CPU: 3 steps x 5 x 1.5 x 2^-23 is below 3e-6. The
        // tolerance is 5e-6.
        for (at, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(*c)).enumerate() {
                // Finite and affirmatively within the tolerance, so a NaN cannot pass.
                if !(gv.is_finite() && (gv - cv).abs() <= 5e-6 * cv.abs()) {
                    return Err(format!(
                        "cell {at} lane {lane}: GPU {gv} against CPU {cv} after {W2_CHECKED_STEPS} steps"
                    ));
                }
            }
        }
        let measured = measure(|| {
            let sample = dispatch(device, &batches[done % 2]);
            done += 1;
            sample
        })?;
        Ok((measured, Some(32.0 * f64::from(n))))
    })();
    row.note = "bytes: one read plus one write of the 16-byte cell (neighbour reads assumed cached); checked after 3 steps".into();
    row
}

fn w3(device: &DeviceHandle, count: u32) -> Row {
    let mut row = Row {
        workload: "W3 particle step",
        size: format!("{count} particles"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        let field_n = W3_FIELD_SIDE * W3_FIELD_SIDE;
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let field_host: Vec<Vec4> = (0..field_n).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..count)
            .map(|i| particle_init(i, W3_FIELD_SIDE))
            .collect();
        let field = GpuVec::from_slice_on(device, &field_host).map_err(dbg)?;
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w3",
                &kit::Params {
                    n: count,
                    side: W3_FIELD_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call = kit::w3_step::call_extent(
            extent(count),
            &params,
            Independent(&field),
            Independent(&parts),
        );
        let batch = [BufferDispatch::from(&call)];
        // One step from the initial state, so every particle samples the cell the host also computes. The tolerance is
        // 8 x 2^-23 of max(|expected|, 1): a step is at most four roundings of 2u on the GPU, with a magnitude of at
        // most the field side. The GPU positions must also lie in [0, side] (`side` itself is possible: -tiny + side
        // rounds to side), so a missing wrap shows even though the modulo comparison below hides a difference of a whole
        // side. A wrap decided differently by the two sides differs by the side, so a difference is
        // taken modulo the side.
        dispatch(device, &batch)?;
        let gpu = parts.to_vec().map_err(dbg)?;
        let side = W3_FIELD_SIDE as f32;
        for (at, (g, q)) in gpu.iter().zip(&parts_host).enumerate() {
            let c = cpu_particle(*q, &field_host, W3_FIELD_SIDE);
            for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(c)).enumerate() {
                if lane < 2 && !(0.0..=side).contains(&gv) {
                    return Err(format!(
                        "particle {at} lane {lane}: GPU {gv} is outside [0, {side}] (the wrap)"
                    ));
                }
                let mut diff = (gv - cv).abs();
                if lane < 2 && diff > side / 2.0 {
                    diff = side - diff;
                }
                // Finite and affirmatively within the tolerance, so a NaN cannot pass.
                if !(gv.is_finite() && diff <= 8.0 * (1.0 / 8_388_608.0) * cv.abs().max(1.0)) {
                    return Err(format!(
                        "particle {at} lane {lane}: GPU {gv} against CPU {cv}"
                    ));
                }
            }
        }
        let measured = measure(|| dispatch(device, &batch))?;
        Ok((measured, Some(48.0 * f64::from(count))))
    })();
    row.note = format!(
        "bytes: particle read and write (16 each) plus one 16-byte field sample; field {W3_FIELD_SIDE}x{W3_FIELD_SIDE} held still; checked after 1 step"
    );
    row
}

fn w4(device: &DeviceHandle, count: u32) -> Row {
    let cells = W4_GRID_SIDE * W4_GRID_SIDE;
    let mut row = Row {
        workload: "W4 spatial binning (count + scan)",
        size: format!("{count} particles, {W4_GRID_SIDE}x{W4_GRID_SIDE}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, W4_GRID_SIDE)).collect();
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // Non-zero to begin with (every cell holds a value the histogram cannot produce), so a clear that does
        // nothing leaves a difference the check below sees; zeroed counts would let it pass.
        let poisoned: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &poisoned).map_err(dbg)?;
        let clear_params = device
            .create_uniform(
                "w4 clear",
                &kit::Params {
                    n: cells,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let count_params = device
            .create_uniform(
                "w4 count",
                &kit::Params {
                    n: count,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let clear = kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&counts));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&counts),
        );
        let batch = [BufferDispatch::from(&clear), BufferDispatch::from(&bin)];
        // The scan's output, made once: every step writes all of it (`exclusive_into`), so no step allocates it.
        let positions = GpuVec::<u32>::zeroed_on(device, cells).map_err(dbg)?;
        // One step: the counts the kernel made against the host's histogram, and the scan of them against the host's
        // exclusive prefix, both exactly (integers).
        // The clear and count batch goes through `dispatch_batch_in_scope` inside the same `timed` scope as the scan, so
        // the scope's GPU time covers both and its `batches` is the clear and count batch plus the scan's own.
        let step = || -> Result<Sample, String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &batch)?;
                scan::exclusive_into(&counts, &positions)
            })
            .map_err(dbg)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            if scoped.time.batches != 2 {
                return Err(format!(
                    "the timing scope counted {} batch(es), expected the clear and count batch and the scan",
                    scoped.time.batches
                ));
            }
            let gpu_ms = scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3);
            Ok(Sample { gpu_ms, wall_ms })
        };
        step()?;
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        if counts.to_vec().map_err(dbg)? != cpu {
            return Err("the atomic counts differ from the host histogram".into());
        }
        if positions.to_vec().map_err(dbg)? != cpu_exclusive(&cpu) {
            return Err("the scan differs from the host exclusive prefix".into());
        }
        let measured = measure(step)?;
        // A later step as well: counts that were not cleared between steps have grown by now.
        if counts.to_vec().map_err(dbg)? != cpu {
            return Err(
                "the atomic counts differ from the host histogram after the timed steps".into(),
            );
        }
        // The particle read, the atomic update of one counter (a read and a write), the clear and the scan of the counts.
        let bytes = 16.0 * f64::from(count) + 8.0 * f64::from(count) + 12.0 * f64::from(cells);
        Ok((measured, Some(bytes)))
    })();
    row.note = "one timing scope covers the clear and count batch and the scan, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps, which no longer includes allocating the scan output: it is made once before the steps, so do not compare wall figures with earlier reports); bytes: 16 + 8 per particle, 12 per cell, nominal (the atomic counted as 8), so the GB/s is not a memory bandwidth".into();
    row
}

/// Up to the first 16 entries of a slice, for a failure message.
fn head(values: &[u32]) -> String {
    let shown: Vec<String> = values.iter().take(16).map(u32::to_string).collect();
    let more = if values.len() > 16 { ", .." } else { "" };
    format!("[{}{more}] ({} entries)", shown.join(", "), values.len())
}

/// The scatter against the host: for every cell the slots `starts[c] .. starts[c] + counts[c]` of `order` hold, as a
/// set, the indices of the particles the host puts in that cell (each range is sorted first: the order inside a cell is
/// unspecified), and every index appears once. The host's sets come from a counting sort of the host's own particles,
/// binned on a `bins x bins` grid over a field of side `field` (`grid` is `(bins, field)`).
fn check_scatter(
    order: &[u32],
    starts: &[u32],
    counts: &[u32],
    parts_host: &[Vec4],
    grid: (u32, u32),
) -> Result<(), String> {
    let cell_of_host = |q: &Vec4| bin_of_host(q, grid.0, grid.1);
    if order.len() != parts_host.len() {
        return Err(format!(
            "order holds {} slots, expected {}",
            order.len(),
            parts_host.len()
        ));
    }
    let mut fill = vec![0u32; counts.len()];
    let mut expected = vec![u32::MAX; parts_host.len()];
    for (i, q) in parts_host.iter().enumerate() {
        let c = cell_of_host(q);
        expected[(starts[c] + fill[c]) as usize] = i as u32;
        fill[c] += 1;
    }
    let mut seen = vec![false; parts_host.len()];
    let mut got = Vec::new();
    for (c, &count) in counts.iter().enumerate() {
        let from = starts[c] as usize;
        let to = from + count as usize;
        got.clear();
        got.extend_from_slice(&order[from..to]);
        got.sort_unstable();
        if got != expected[from..to] {
            return Err(format!(
                "cell {c}: GPU set {} against host set {}",
                head(&got),
                head(&expected[from..to])
            ));
        }
        for &index in &got {
            if std::mem::replace(&mut seen[index as usize], true) {
                return Err(format!("cell {c}: index {index} appears twice"));
            }
        }
    }
    if let Some(missing) = seen.iter().position(|s| !s) {
        return Err(format!("index {missing} appears in no cell"));
    }
    Ok(())
}

fn w4_scatter(device: &DeviceHandle, count: u32) -> Row {
    let cells = W4_GRID_SIDE * W4_GRID_SIDE;
    let mut row = Row {
        workload: "W4 spatial binning (count + scan + scatter)",
        size: format!("{count} particles, {W4_GRID_SIDE}x{W4_GRID_SIDE}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, W4_GRID_SIDE)).collect();
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // The counts and the cursors begin non-zero, and `order` begins at a value no index can equal, so a clear that
        // does nothing or a slot never written shows in the check below.
        let counts_init: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let cursor_init: Vec<u32> = (0..cells).map(|i| 0x4000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &counts_init).map_err(dbg)?;
        let cursor = GpuVec::from_slice_on(device, &cursor_init).map_err(dbg)?;
        let order = GpuVec::from_slice_on(device, &vec![u32::MAX; count as usize]).map_err(dbg)?;
        let clear_params = device
            .create_uniform(
                "w4 scatter clear",
                &kit::Params {
                    n: cells,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let count_params = device
            .create_uniform(
                "w4 scatter count",
                &kit::Params {
                    n: count,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let clear_counts =
            kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&counts));
        let clear_cursor =
            kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&cursor));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&counts),
        );
        let first = [
            BufferDispatch::from(&clear_counts),
            BufferDispatch::from(&clear_cursor),
            BufferDispatch::from(&bin),
        ];
        // The scan's output and the scatter call that reads it are made once (`exclusive_into` writes all of `starts`
        // every step), so a step allocates and builds neither; the scan's own calls are still made inside it.
        let starts = GpuVec::<u32>::zeroed_on(device, cells).map_err(dbg)?;
        let scatter = kit::w4_scatter::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&starts),
            Independent(&cursor),
            Independent(&order),
        );
        let second = [BufferDispatch::from(&scatter)];
        // One step: the clears and the count (one batch), the scan, the scatter (one batch): three batches in the one
        // `timed` scope, so its GPU time covers the whole binning.
        let step = || -> Result<Sample, String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &first)?;
                scan::exclusive_into(&counts, &starts)?;
                dispatch_batch_in_scope(device, &second)?;
                Ok(())
            })
            .map_err(dbg)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            if scoped.time.batches != 3 {
                return Err(format!(
                    "the timing scope counted {} batch(es), expected the clear and count batch, the scan and the scatter batch",
                    scoped.time.batches
                ));
            }
            let gpu_ms = scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3);
            Ok(Sample { gpu_ms, wall_ms })
        };
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        let cpu_starts = cpu_exclusive(&cpu);
        let verify = |after: &str| -> Result<(), String> {
            if counts.to_vec().map_err(dbg)? != cpu {
                return Err(format!(
                    "the atomic counts differ from the host histogram{after}"
                ));
            }
            if starts.to_vec().map_err(dbg)? != cpu_starts {
                return Err(format!(
                    "the scan differs from the host exclusive prefix{after}"
                ));
            }
            check_scatter(
                &order.to_vec().map_err(dbg)?,
                &cpu_starts,
                &cpu,
                &parts_host,
                (W4_GRID_SIDE, W4_GRID_SIDE),
            )
            .map_err(|reason| format!("scatter{after}: {reason}"))
        };
        step()?;
        verify("")?;
        let measured = measure(step)?;
        // A later step as well: counters or cursors that were not cleared between steps have grown by now.
        verify(" after the timed steps")?;
        // Per particle: the particle read (16), the count atomic (8), the scatter's particle read (16), the start read
        // (4), the cursor atomic (8) and the order write (4); per cell: the two clears (8) and the scan (12).
        let bytes = 56.0 * f64::from(count) + 20.0 * f64::from(cells);
        Ok((measured, Some(bytes)))
    })();
    row.note = "the order inside a cell is unspecified (checked as a set per cell, each range sorted first, and every index exactly once); one timing scope covers the clears and count batch, the scan and the scatter batch, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps, which no longer includes allocating the scan output or building the scatter call: both are made once before the steps, so do not compare wall figures with earlier reports); the particles never move and `order` is not re-poisoned between steps, so the check after the timed steps does not catch a write lost in a later step; bytes: nominal (each atomic counted as 8), so the GB/s is not a memory bandwidth".into();
    row
}

// ---- W7: a sustained composite step ----

/// W7 particle counts, and the counted steps of its sustained run (after the warm-up, for each of its two runs).
const W7_SIZES: [u32; 2] = [1 << 18, 1 << 20];
const W7_STEPS: usize = 600;
const W7_STEPS_SMALL: usize = 120;

/// The value at or below which at least `p` of the samples lie (nearest rank), for `p` in (0, 1]. `values` must not
/// be empty.
fn percentile(values: &[f64], p: f64) -> f64 {
    debug_assert!(!values.is_empty(), "percentile of no samples");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// W2's comparison of a field with the host's, with W2's tolerance.
fn check_field(gpu: &[Vec4], cpu: &[Vec4]) -> Result<(), String> {
    for (at, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(*c)).enumerate() {
            if !(gv.is_finite() && (gv - cv).abs() <= 5e-6 * cv.abs()) {
                return Err(format!(
                    "field cell {at} lane {lane}: GPU {gv} against CPU {cv} after 1 step"
                ));
            }
        }
    }
    Ok(())
}

/// W3's comparison of particles with the host's one step from `start` over `field`, with W3's tolerance.
fn check_particles(
    gpu: &[Vec4],
    start: &[Vec4],
    field: &[Vec4],
    side_u: u32,
) -> Result<(), String> {
    let side = side_u as f32;
    for (at, (g, q)) in gpu.iter().zip(start).enumerate() {
        let c = cpu_particle(*q, field, side_u);
        for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(c)).enumerate() {
            if lane < 2 && !(0.0..=side).contains(&gv) {
                return Err(format!(
                    "particle {at} lane {lane}: GPU {gv} is outside [0, {side}] (the wrap)"
                ));
            }
            let mut diff = (gv - cv).abs();
            if lane < 2 && diff > side / 2.0 {
                diff = side - diff;
            }
            if !(gv.is_finite() && diff <= 8.0 * (1.0 / 8_388_608.0) * cv.abs().max(1.0)) {
                return Err(format!(
                    "particle {at} lane {lane}: GPU {gv} against CPU {cv}"
                ));
            }
        }
    }
    Ok(())
}

fn w7(device: &DeviceHandle, count: u32, steps: usize) -> Row {
    let side = W3_FIELD_SIDE;
    let cells = side * side;
    let mut row = Row {
        workload: "W7 composite step",
        size: format!("{count} particles, {side}x{side} field"),
        result: Err("not run".into()),
        note: String::new(),
    };
    // What the row says whether or not it succeeds; a success puts the statistics before it.
    let fixed = format!(
        "{steps} counted steps after {WARMUP} warm-up steps, in two runs of that length (blocking, then submit); one batch per step: w2_diffuse on {side}x{side}, w3_step, w4_clear and w4_count into {side}x{side} cells; checked after step 1 and again after step 2, the same three ways (field 5e-6 relative as W2, particles as W3 over the GPU's own field, counts exact against the GPU's own positions), and once more after the runs (particles in range, counts exact)"
    );
    row.note = fixed.clone();
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let field_host: Vec<Vec4> = (0..cells).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, side)).collect();
        let a = GpuVec::from_slice_on(device, &field_host).map_err(dbg)?;
        let b = GpuVec::<Vec4>::zeroed_on(device, cells).map_err(dbg)?;
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // Non-zero to begin with, so a clear that does nothing shows in the count check.
        let poisoned: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &poisoned).map_err(dbg)?;
        let uniform = |label: &str, n: u32| {
            device
                .create_uniform(
                    label,
                    &kit::Params {
                        n,
                        side,
                        pad0: 0,
                        pad1: 0,
                    },
                )
                .map_err(dbg)
        };
        let field_params = uniform("w7 field", cells)?;
        let part_params = uniform("w7 particles", count)?;
        let diffuse_ab = kit::w2_diffuse::call_extent(
            extent(cells),
            &field_params,
            Independent(&a),
            Independent(&b),
        );
        let diffuse_ba = kit::w2_diffuse::call_extent(
            extent(cells),
            &field_params,
            Independent(&b),
            Independent(&a),
        );
        let step_b = kit::w3_step::call_extent(
            extent(count),
            &part_params,
            Independent(&b),
            Independent(&parts),
        );
        let step_a = kit::w3_step::call_extent(
            extent(count),
            &part_params,
            Independent(&a),
            Independent(&parts),
        );
        let clear = kit::w4_clear::call_extent(extent(cells), &field_params, Independent(&counts));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &part_params,
            Independent(&parts),
            Independent(&counts),
        );
        // The field alternates between `a` and `b` as in W2; the particles sample the field the same batch diffused.
        let batches = [
            [
                BufferDispatch::from(&diffuse_ab),
                BufferDispatch::from(&step_b),
                BufferDispatch::from(&clear),
                BufferDispatch::from(&bin),
            ],
            [
                BufferDispatch::from(&diffuse_ba),
                BufferDispatch::from(&step_a),
                BufferDispatch::from(&clear),
                BufferDispatch::from(&bin),
            ],
        ];
        // The first step, checked: the field (in `b`) against the host's diffusion, the particles against the host's
        // step over the GPU's own diffused field (the one they sampled), and the counts exactly against the host's histogram of the GPU's own
        // particle positions (a particle at a cell edge could bin differently from a host position that differs in the
        // last bit).
        dispatch(device, &batches[0])?;
        let cpu_field = cpu_diffuse(&field_host, side);
        let gpu_field = b.to_vec().map_err(dbg)?;
        check_field(&gpu_field, &cpu_field)?;
        let gpu_parts = parts.to_vec().map_err(dbg)?;
        // The particles sampled the GPU's own field, so that is the field of the reference step.
        check_particles(&gpu_parts, &parts_host, &gpu_field, side)?;
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts, side) {
            return Err("the counts differ from the host histogram after 1 step".into());
        }
        // The second step, checked the same way (the field is back in `a`): the host diffuses the GPU's own
        // first-step field, the particles are checked against the host's step from the GPU's first-step particles
        // over the GPU's own second-step field, and the counts against the host histogram of the GPU's own particles.
        dispatch(device, &batches[1])?;
        let cpu_field_2 = cpu_diffuse(&gpu_field, side);
        let gpu_field_2 = a.to_vec().map_err(dbg)?;
        check_field(&gpu_field_2, &cpu_field_2).map_err(|e| format!("second step: {e}"))?;
        let gpu_parts_2 = parts.to_vec().map_err(dbg)?;
        check_particles(&gpu_parts_2, &gpu_parts, &gpu_field_2, side)
            .map_err(|e| format!("second step: {e}"))?;
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts_2, side) {
            return Err("the counts differ from the host histogram after 2 steps".into());
        }
        // Two steps ran, so the alternation continues with `batches[0]` (field in `a`).
        let mut next = 2usize;
        // Run 1: blocking `dispatch_batch_timed`: the GPU time and `host_elapsed` of every counted step.
        for _ in 0..WARMUP {
            dispatch(device, &batches[next % 2])?;
            next += 1;
        }
        let mut blocking = Vec::with_capacity(steps);
        for _ in 0..steps {
            blocking.push(dispatch(device, &batches[next % 2])?);
            next += 1;
        }
        // Run 2: the wall time for `submit_batch` to return (encode and submit, no wait); the wait for the job is
        // outside the measured span.
        for _ in 0..WARMUP {
            device.dispatch_batch(&batches[next % 2]).map_err(dbg)?;
            next += 1;
        }
        let mut submit_ms = Vec::with_capacity(steps);
        for _ in 0..steps {
            let started = Instant::now();
            let job = device.submit_batch(&batches[next % 2]).map_err(dbg)?;
            submit_ms.push(started.elapsed().as_secs_f64() * 1e3);
            job.wait().map_err(dbg)?;
            next += 1;
        }
        // After the sustained runs: the particles are still inside the field, and the counts of the last step are the
        // host's histogram of the particles as they now are.
        let gpu_parts = parts.to_vec().map_err(dbg)?;
        let side_f = side as f32;
        if let Some(at) = gpu_parts.iter().position(|q| {
            !(q.x.is_finite()
                && q.y.is_finite()
                && (0.0..=side_f).contains(&q.x)
                && (0.0..=side_f).contains(&q.y))
        }) {
            return Err(format!(
                "particle {at} is outside [0, {side_f}] after the sustained runs"
            ));
        }
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts, side) {
            return Err(
                "the counts differ from the host histogram after the sustained runs".into(),
            );
        }
        let host_a: Vec<f64> = blocking.iter().map(|s| s.wall_ms).collect();
        let (gpu_series, method) = if blocking.iter().all(|s| s.gpu_ms.is_some()) {
            (
                blocking
                    .iter()
                    .filter_map(|s| s.gpu_ms)
                    .collect::<Vec<f64>>(),
                "gpu timestamps",
            )
        } else {
            (host_a.clone(), "wall clock")
        };
        let measured = Measured {
            ms: median(&gpu_series),
            method,
        };
        row.note = format!(
            "the time column is the median {method} time, p95 {:.4} ms, max {:.4} ms; host (a) host_elapsed of the blocking dispatch_batch_timed (it includes the GPU work and the timestamp readback): median {:.4} ms, p95 {:.4} ms; host (b) wall time for submit_batch to return (encode and submit, no wait; the job is waited for after the span; measured with an idle queue, so it shows no back-pressure): median {:.4} ms, p95 {:.4} ms; {fixed}",
            percentile(&gpu_series, 0.95),
            gpu_series.iter().copied().fold(f64::MIN, f64::max),
            median(&host_a),
            percentile(&host_a, 0.95),
            median(&submit_ms),
            percentile(&submit_ms, 0.95)
        );
        Ok((measured, None))
    })();
    row
}

// ---- W8: a full step with a lifecycle (m11.kit4) ----

/// W8 particle capacities. The live count starts at the capacity over `W8_START_DIV`; every step retires about one
/// particle in sixteen (`kit::RETIRE`) and spawns the capacity over `W8_SPAWN_DIV` new ones, so the population climbs
/// to the capacity and then stays near it, with the spawns that do not fit dropped.
const W8_SIZES: [u32; 2] = [1 << 17, 1 << 20];
const W8_SPAWN_DIV: u32 = 16;
const W8_START_DIV: u32 = 2;

/// The "lowbias32" finaliser (Chris Wellons' hash-prospector, public domain), written again from
/// `examples/walker-life/src/app.rs` (`mix`), as is `lb_hash` (`hash`) after it.
fn lb_mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2_146_121_005);
    h ^= h >> 15;
    h = h.wrapping_mul(2_221_713_035);
    h ^ (h >> 16)
}

fn lb_hash(seed: u32, index: u32) -> u32 {
    lb_mix(lb_mix(seed) ^ index)
}

/// Whether slot `i` retires in step `step`: the host's form of `w8_retire`.
fn retires(step: u32, i: u32) -> bool {
    (lb_hash(lb_hash(kit::RETIRE_SEED, step), i) & 0xffff) < kit::RETIRE
}

/// Particle `k` spawned in step `step`: the host's form of `w8_spawn` (every operation is exact in `f32`).
fn spawn_particle(step: u32, k: u32, side: u32) -> Vec4 {
    let h = lb_hash(lb_hash(kit::SPAWN_SEED, step), k);
    let s = side as f32;
    Vec4::new(
        unit(lb_hash(h, 0)) * s,
        unit(lb_hash(h, 1)) * s,
        unit(lb_hash(h, 2)) * 2.0 - 1.0,
        unit(lb_hash(h, 3)) * 2.0 - 1.0,
    )
}

/// The host model's counts of one step: the live particles before it, the survivors, the live particles after it.
struct LifeStep {
    live: u32,
    surv: u32,
    next: u32,
}

/// The counts of steps 1 to `steps` from `start` live particles: every number is an integer function of the step and
/// the slot, so the host knows them without the GPU.
fn life_plan(cap: u32, spawn: u32, start: u32, steps: usize) -> Vec<LifeStep> {
    let mut live = start;
    let mut plan = Vec::with_capacity(steps);
    for step in 1..=steps as u32 {
        let inner = lb_mix(lb_hash(kit::RETIRE_SEED, step));
        let surv = (0..live)
            .filter(|&i| (lb_mix(inner ^ i) & 0xffff) >= kit::RETIRE)
            .count() as u32;
        let next = (surv + spawn).min(cap);
        plan.push(LifeStep { live, surv, next });
        live = next;
    }
    plan
}

/// The first index at which two integer slices differ, as a message part.
fn first_diff(got: &[u32], want: &[u32]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} entries against {}", got.len(), want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(g, w)| g != w)
        .map(|at| format!("entry {at}: GPU {}, host {}", got[at], want[at]))
}

/// What the GPU left after one W8 step: its own moved live particles (the ones the compaction read), the alive flags
/// and their exclusive scan over the capacity, and the second particle buffer.
struct LifeGpu<'a> {
    stepped: &'a [Vec4],
    alive: &'a [u32],
    astarts: &'a [u32],
    next: &'a [Vec4],
}

/// The lifecycle of one step against the host's model, all exact. The host recomputes the flags from the retire rule,
/// the scan from the flags, the survivors and live count from the scan and the plan, and the live set: the survivors
/// in slot order, then the spawned particles (those that fit), compared slot by slot, which also makes them equal as
/// a set.
fn check_life(
    step: u32,
    cap: u32,
    spawn: u32,
    side: u32,
    plan: &LifeStep,
    gpu: &LifeGpu<'_>,
) -> Result<(), String> {
    let LifeGpu {
        stepped,
        alive,
        astarts,
        next,
    } = *gpu;
    let host_alive: Vec<u32> = (0..cap)
        .map(|i| u32::from(i < plan.live && !retires(step, i)))
        .collect();
    if let Some(reason) = first_diff(alive, &host_alive) {
        return Err(format!("alive flags: {reason}"));
    }
    let host_starts = cpu_exclusive(&host_alive);
    if let Some(reason) = first_diff(astarts, &host_starts) {
        return Err(format!("scan of the alive flags: {reason}"));
    }
    let surv = astarts[cap as usize - 1] + alive[cap as usize - 1];
    if surv != plan.surv {
        return Err(format!(
            "survivors: GPU scan says {surv}, host model {}",
            plan.surv
        ));
    }
    // The live count after the step is `min(surv + spawn, cap)`, from the survivors just compared; the live set below
    // covers exactly that many slots (the spawns past the capacity are dropped).
    let live_next = plan.next;
    let mut expected: Vec<Vec4> = stepped
        .iter()
        .zip(&host_alive)
        .filter(|(_, a)| **a == 1)
        .map(|(q, _)| *q)
        .collect();
    for k in 0..spawn {
        if surv + k < cap {
            expected.push(spawn_particle(step, k, side));
        }
    }
    if let Some(at) = expected
        .iter()
        .zip(next)
        .position(|(e, g)| lanes(*e) != lanes(*g))
    {
        return Err(format!(
            "live set: slot {at} of {live_next}: GPU {:?}, host {:?}",
            lanes(next[at]),
            lanes(expected[at])
        ));
    }
    Ok(())
}

/// How many storage buffers a kernel binds, from its generated descriptor (uniform and invocation parameters are not
/// storage buffers).
fn storage_buffers(kernel: &gust::KernelDescriptor) -> usize {
    kernel
        .parameters
        .iter()
        .filter(|parameter| matches!(parameter.kind, gust::ParameterKind::Storage))
        .count()
}

/// The storage buffers each kernel of the W8 step binds, counted from the generated descriptors, and the largest.
fn w8_buffer_note(cells: u32, cap: u32) -> String {
    use gust_algo::scan_kernels as sk;
    use gust_algo::schedule::TileSchedule;
    use gust_algo::variant::{Operation, Variant, shipped};
    // The scan kernels of the shipped exclusive variant for these two lengths (gust-algo may still fall back to v0 on
    // a device with too little workgroup memory; that decision is inside the library and not visible here).
    let tiles = shipped(Operation::Exclusive) == Variant::V1
        && TileSchedule::new(cells).is_some()
        && TileSchedule::new(cap).is_some();
    let scan_kernels = if tiles {
        vec![
            (
                "scan v1 tiles exclusive",
                storage_buffers(&sk::scan_tiles_exclusive_u32::DESCRIPTOR),
            ),
            (
                "scan v1 tiles sums",
                storage_buffers(&sk::scan_tiles_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v1 tiles top",
                storage_buffers(&sk::scan_tiles_top_u32::DESCRIPTOR),
            ),
            (
                "scan v1 add carry",
                storage_buffers(&sk::add_carry_u32::DESCRIPTOR),
            ),
        ]
    } else {
        vec![
            (
                "scan v0 chunks",
                storage_buffers(&sk::scan_chunks_u32::DESCRIPTOR),
            ),
            (
                "scan v0 sums",
                storage_buffers(&sk::scan_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v0 top",
                storage_buffers(&sk::scan_top_u32::DESCRIPTOR),
            ),
            (
                "scan v0 add back sums",
                storage_buffers(&sk::add_back_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v0 add back exclusive",
                storage_buffers(&sk::add_back_exclusive_u32::DESCRIPTOR),
            ),
        ]
    };
    let mut kernels = vec![
        ("w2_diffuse", storage_buffers(&kit::w2_diffuse::DESCRIPTOR)),
        ("w3_step", storage_buffers(&kit::w3_step::DESCRIPTOR)),
        ("w4_clear", storage_buffers(&kit::w4_clear::DESCRIPTOR)),
        ("w4_count", storage_buffers(&kit::w4_count::DESCRIPTOR)),
        ("w8_deposit", storage_buffers(&kit::w8_deposit::DESCRIPTOR)),
        ("w8_fold", storage_buffers(&kit::w8_fold::DESCRIPTOR)),
        ("w8_retire", storage_buffers(&kit::w8_retire::DESCRIPTOR)),
        ("w4_scatter", storage_buffers(&kit::w4_scatter::DESCRIPTOR)),
        ("w8_compact", storage_buffers(&kit::w8_compact::DESCRIPTOR)),
        ("w8_spawn", storage_buffers(&kit::w8_spawn::DESCRIPTOR)),
    ];
    kernels.extend(scan_kernels);
    let most = kernels.iter().map(|(_, n)| *n).max().unwrap_or(0);
    let list: Vec<String> = kernels
        .iter()
        .map(|(name, n)| format!("{name} {n}"))
        .collect();
    format!(
        "storage buffers bound per kernel (counted from each kernel's generated descriptor: the parameters of kind Storage, uniforms excluded): {}; the most in one kernel is {most}{}",
        list.join(", "),
        if most > 8 {
            " (OVER the 8 per stage GUST opens devices with)"
        } else {
            " (within the 8 per stage GUST opens devices with)"
        }
    )
}

/// The variants of the W8 step (slice m11.cost): `Base` is W8, each other changes one thing.
#[derive(Clone, Copy, PartialEq)]
enum Var {
    Base,
    /// No bins: no count, no clears of the counts and cursors, no scan of the counts and no scatter.
    NoBins,
    /// The bins are a `W8_COARSE_BINS` x `W8_COARSE_BINS` grid over the same field.
    CoarseBins,
    /// The step, the count and the deposit in one particle kernel.
    Fused,
    /// The lifecycle (retire, flag scan, compaction, spawn) on every `W8_LIFE_PERIOD`th step only.
    Sparse,
}

/// The bins per side of `Var::CoarseBins`, and the period of `Var::Sparse`.
const W8_COARSE_BINS: u32 = 256;
const W8_LIFE_PERIOD: usize = 4;

impl Var {
    fn workload(self) -> &'static str {
        match self {
            Self::Base => "W8 full step",
            Self::NoBins => "W8a no bins",
            Self::CoarseBins => "W8b bins 256x256",
            Self::Fused => "W8c fused step, count and deposit",
            Self::Sparse => "W8d lifecycle every 4th step",
        }
    }

    /// Steps per lifecycle: the lifecycle runs on every `period`th step.
    fn period(self) -> usize {
        if self == Self::Sparse {
            W8_LIFE_PERIOD
        } else {
            1
        }
    }

    fn bins(self) -> bool {
        self != Self::NoBins
    }

    /// What differs from W8, for the row's note (empty for W8 itself).
    fn describe(self) -> String {
        match self {
            Self::Base => String::new(),
            Self::NoBins => "variant a, no bins: the step has no w4_count, no clears of the counts and the cursors, no scan of the counts and no scatter, so it is three batches: (1) w2_diffuse, w3_step, the clear of the deposit grid, w8_deposit, w8_fold, w8_retire; (2) the scan of the alive flags; (3) w8_compact and w8_spawn; the checks of the counts, their scan and the scatter are dropped, the rest are W8's; the description below applies except where this says otherwise; ".to_owned(),
            Self::CoarseBins => format!(
                "variant b, the bins are a {W8_COARSE_BINS}x{W8_COARSE_BINS} grid over the same field: w8_bin_count and w8_bin_scatter bin a position scaled by {W8_COARSE_BINS}/{} (exact), so the counts, the cursors and their scan are {} cells; the field and the deposit grid stay {side}x{side}; the checks bin the GPU's own positions the same way on the host, the rest are W8's; storage buffers: w8_bin_count {}, w8_bin_scatter {}; the description below applies except where this says otherwise; ",
                W3_FIELD_SIDE,
                W8_COARSE_BINS * W8_COARSE_BINS,
                storage_buffers(&kit::w8_bin_count::DESCRIPTOR),
                storage_buffers(&kit::w8_bin_scatter::DESCRIPTOR),
                side = W3_FIELD_SIDE
            ),
            Self::Fused => format!(
                "variant c, w8_step_count_deposit does the step, the count and the deposit in one particle kernel (the same three effects, one read of each particle; storage buffers {}), so the first batch has no w3_step, w4_count or w8_deposit; the checks are W8's unchanged; the description below applies except where this says otherwise; ",
                storage_buffers(&kit::w8_step_count_deposit::DESCRIPTOR)
            ),
            Self::Sparse => format!(
                "variant d, the lifecycle (w8_retire, the scan of the alive flags, w8_compact, w8_spawn) runs on every {W8_LIFE_PERIOD}th step only; the steps between move, bin and deposit (three batches: the first without w8_retire, the scan of the counts, the scatter) and the live count changes on a lifecycle step only (the plan is per round of {W8_LIFE_PERIOD} steps and the retire draw uses the round number); the time column is the amortised mean GPU time per step over the counted steps, which are whole rounds from a round boundary (the warm-up is a whole number of rounds too); steps 1 and 2 (no lifecycle), the first clamped lifecycle step and the last step (a lifecycle step) are checked, a step without a lifecycle on everything but the lifecycle; the description below applies except where this says otherwise; "
            ),
        }
    }
}

/// One piece of a W8 step: a kernel (with the name the reports use) or a scan of one `u32` buffer into another.
enum Part<'a> {
    Kernel(&'static str, BufferDispatch<'a>),
    Scan(&'static str, &'a GpuVec<u32>, &'a GpuVec<u32>),
}

impl Part<'_> {
    fn name(&self) -> &'static str {
        match self {
            Self::Kernel(name, _) | Self::Scan(name, ..) => name,
        }
    }
}

/// The part of a kernel's call, or nothing for a call the variant does not make.
fn kernel_part<'a, C>(name: &'static str, call: &'a Option<C>) -> Option<Part<'a>>
where
    BufferDispatch<'a>: From<&'a C>,
{
    call.as_ref()
        .map(|call| Part::Kernel(name, BufferDispatch::from(call)))
}

/// How the parts of a step are grouped into batches (one batch and one timing scope each).
#[derive(Clone, Copy, PartialEq)]
enum Grain {
    /// A run of kernels is one batch and a scan is one of its own: the kit's step.
    Fused,
    /// Every kernel is a batch of its own, and every scan one.
    Kernel,
}

/// The ranges of `parts` that run as one batch each.
fn groups(parts: &[Part<'_>], grain: Grain) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        let joins = grain == Grain::Fused
            && matches!(part, Part::Kernel(..))
            && at > 0
            && matches!(parts[at - 1], Part::Kernel(..));
        match out.last_mut() {
            Some(last) if joins => last.end = at + 1,
            _ => out.push(at..at + 1),
        }
    }
    out
}

/// The dispatches of a group of kernels.
fn batch_of<'a>(group: &[Part<'a>]) -> Vec<BufferDispatch<'a>> {
    group
        .iter()
        .filter_map(|part| match part {
            Part::Kernel(_, dispatch) => Some(*dispatch),
            Part::Scan(..) => None,
        })
        .collect()
}

/// Runs one group: a scan, or a batch of kernels, through `dispatch_batch_in_scope`.
fn run_group(device: &DeviceHandle, group: &[Part<'_>]) -> Result<(), gust_algo::Error> {
    if let [Part::Scan(_, input, out)] = group {
        return scan::exclusive_into(input, out);
    }
    dispatch_batch_in_scope(device, &batch_of(group))?;
    Ok(())
}

/// A group's name in a report: the kernel, the scan, or the first and last kernel of a batch.
fn group_label(group: &[Part<'_>]) -> String {
    match (group.first(), group.last()) {
        (Some(first), Some(last)) if group.len() > 1 => {
            format!("{}..{}", first.name(), last.name())
        }
        (Some(only), _) => only.name().to_owned(),
        _ => String::new(),
    }
}

/// What a per-group pass finds: each group's name and median GPU time, and the median of the per-step sums.
type PassMedians = (Vec<(String, f64)>, f64);

/// The note text of a per-group pass: each group's median GPU time and its share of the sum of the medians, the sum,
/// and the median of the per-step sums.
fn pass_note(what: &str, pass: Option<&PassMedians>) -> String {
    let Some((groups, step_sums)) = pass else {
        return format!("{what}: n/a, this device has no GPU timestamps");
    };
    let total: f64 = groups.iter().map(|(_, ms)| ms).sum();
    let list: Vec<String> = groups
        .iter()
        .map(|(label, ms)| format!("{label} {ms:.4} ms ({:.1}%)", 100.0 * ms / total))
        .collect();
    format!(
        "{what}: {}; sum of the medians {total:.4} ms, median of the per-step sums {step_sums:.4} ms",
        list.join(", ")
    )
}

/// How W8 runs one step.
#[derive(Clone, Copy)]
enum Mode {
    /// Every batch and the scans inside one `timed` scope: the GPU time of the batches and the wall clock.
    Timed,
    /// The same dispatches with no scope: plain blocking dispatches, the wall clock only (used for the checked steps).
    Plain,
    /// As `Plain`, with the first batch sent through `submit_batch` (only the call is timed, then the job is waited
    /// for): the host cost of encoding and submitting it.
    Submit,
    /// Every group of the grain in a `timed` scope of its own: the GPU time of each.
    Split(Grain),
}

/// What one W8 step took.
struct StepTime {
    gpu_ms: Option<f64>,
    wall_ms: f64,
    submit_ms: Option<f64>,
    /// For `Mode::Split`: each group's name and GPU time.
    groups: Vec<(String, Option<f64>)>,
}

fn w8(device: &DeviceHandle, cap: u32, steps: usize, var: Var) -> Row {
    let side = W3_FIELD_SIDE;
    let cells = side * side;
    let bin_side = if var == Var::CoarseBins {
        W8_COARSE_BINS
    } else {
        side
    };
    let bin_cells = bin_side * bin_side;
    let period = var.period();
    // A whole number of lifecycle periods, so the counted steps start on a round boundary.
    let warm = WARMUP.next_multiple_of(period);
    let spawn = cap / W8_SPAWN_DIV;
    let start = cap / W8_START_DIV;
    let mut row = Row {
        workload: var.workload(),
        size: format!("{cap} particles capacity, {side}x{side} field"),
        result: Err("not run".into()),
        note: String::new(),
    };
    let fixed = format!(
        "{steps} counted steps after {warm} warm-up steps, in two runs of that length (the timed scope, then no scope with the first batch submitted through submit_batch); one step is four batches in one timed scope: (1) w2_diffuse on {side}x{side}, w3_step over the diffused field, w4_clear of the counts, the cursors and the deposit grid, w4_count, w8_deposit (atomic_add of {} into a u32 grid), w8_fold (the grid into lane w of the field), w8_retire (alive flags over the capacity); (2) gust-algo exclusive scan of the {cells} counts; (3) the same scan of the {cap} alive flags; (4) w4_scatter, w8_compact (survivors to slots given by the scan), w8_spawn ({spawn} new particles after the survivors, those past the capacity dropped); the population starts at {start} live particles, one in {} retires per step by a hash of (step, slot); the live counts come from a host model, not from a readback (a design that reads the count back each step pays W6's delay on top); checked after step 1, after step 2, on the first step where the survivors and the spawns do not all fit (the capacity clamp; the code requires that step to exist and says which it is in the note) and after the runs (one more step, checked on its own); each checked step re-poisons the scatter's `order` first: the field against the host (diffuse then fold of the GPU's previous field, 5e-6 relative as W2), the moved particles as W3, the deposit grid, the counts and their scan exactly, the scatter as sets per cell as W4, the alive flags, their scan and the survivors (the total of the scan) exactly against the host model, and the live set (survivors in slot order, then the spawned that fit; compared slot by slot, which fixes the live count)",
        kit::DEPOSIT,
        65_536 / kit::RETIRE
    );
    row.note = format!("{}{fixed}; {}", var.describe(), w8_buffer_note(cells, cap));
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(cap.max(cells)) * 16) {
            return Err(reason);
        }
        // The counted steps must be whole lifecycle periods, so the mean covers whole rounds and the last step is a
        // lifecycle step.
        if !steps.is_multiple_of(period) {
            return Err(format!(
                "{steps} counted steps are not a whole number of {period}-step rounds"
            ));
        }
        // The round a step belongs to (a lifecycle period of steps), and whether the step ends it with the lifecycle.
        let round = |g: usize| (g - 1) / period + 1;
        let is_life = |g: usize| g.is_multiple_of(period);
        // The first round where the capacity clamps the population (survivors plus spawns past the capacity), found in
        // a plan long enough to reach it; its lifecycle step is checked on its own. Steps run: 1, 2 and the clamped
        // step checked, two runs of warm-up and counted steps, for W8 itself the two per-group passes, and one last
        // checked step.
        let clamp_round = life_plan(cap, spawn, start, 400)
            .iter()
            .position(|s| s.surv + spawn > cap)
            .map(|at| at + 1)
            .ok_or("no round of the first 400 reaches the capacity clamp")?;
        let clamp_at = clamp_round * period;
        let checked_to = clamp_at.max(2);
        let passes = if var == Var::Base {
            2 * (WARMUP + STEPS)
        } else {
            0
        };
        let total = checked_to + 2 * (warm + steps) + passes + period;
        let plan = life_plan(cap, spawn, start, round(total));
        if plan[clamp_round - 1].surv + spawn <= cap {
            return Err(format!("round {clamp_round} is not clamped"));
        }
        let live_of = |g: usize| plan[round(g) - 1].live;
        let field_host: Vec<Vec4> = (0..cells).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..cap).map(|i| particle_init(i, side)).collect();
        // The field alternates between two buffers each step; the particles between two buffers each round.
        let fields = [
            GpuVec::from_slice_on(device, &field_host).map_err(dbg)?,
            GpuVec::<Vec4>::zeroed_on(device, cells).map_err(dbg)?,
        ];
        let pbufs = [
            GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?,
            GpuVec::from_slice_on(
                device,
                &vec![Vec4::new(-1.0, -1.0, -1.0, -1.0); cap as usize],
            )
            .map_err(dbg)?,
        ];
        // Non-zero to begin with, so a clear or a write that does nothing shows in the checks.
        let poisoned = |tag: u32, n: u32| -> Result<GpuVec<u32>, String> {
            let host: Vec<u32> = (0..n).map(|i| tag | i).collect();
            GpuVec::from_slice_on(device, &host).map_err(dbg)
        };
        // `Var::NoBins` binds none of the bin buffers (counts, cursor, starts, order), so they are one element there.
        let (bin_n, order_n) = if var.bins() { (bin_cells, cap) } else { (1, 1) };
        let counts = poisoned(0x8000_0000, bin_n)?;
        let cursor = poisoned(0x4000_0000, bin_n)?;
        let dep = poisoned(0x2000_0000, cells)?;
        let alive = poisoned(0x8000_0000, cap)?;
        let starts = GpuVec::<u32>::zeroed_on(device, bin_n).map_err(dbg)?;
        let astarts = GpuVec::<u32>::zeroed_on(device, cap).map_err(dbg)?;
        let order =
            GpuVec::from_slice_on(device, &vec![u32::MAX; order_n as usize]).map_err(dbg)?;
        let uniform = |label: &str, n: u32, side: u32| {
            device
                .create_uniform(
                    label,
                    &kit::Params {
                        n,
                        side,
                        pad0: 0,
                        pad1: 0,
                    },
                )
                .map_err(dbg)
        };
        let cell_params = uniform("w8 cells", cells, side)?;
        let bin_clear_params = uniform("w8 bin cells", bin_cells, bin_side)?;
        // One set of uniforms per step: the live count and the round change, so they are made before the runs.
        let part_params = (1..=total)
            .map(|g| uniform("w8 particles", live_of(g), side))
            .collect::<Result<Vec<_>, String>>()?;
        let life_params = (1..=total)
            .map(|g| {
                device
                    .create_uniform(
                        "w8 life",
                        &kit::LifeParams {
                            cap,
                            live: live_of(g),
                            step: round(g) as u32,
                            surv: plan[round(g) - 1].surv,
                            spawn,
                            side,
                            pad0: 0,
                            pad1: 0,
                        },
                    )
                    .map_err(dbg)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let bin_params = if var == Var::CoarseBins {
            (1..=total)
                .map(|g| {
                    device
                        .create_uniform(
                            "w8 bins",
                            &kit::BinParams {
                                n: live_of(g),
                                bins: bin_side,
                                field: side,
                                pad0: 0,
                            },
                        )
                        .map_err(dbg)
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            Vec::new()
        };
        // Step `g` (1-based) diffuses `fields[(g - 1) % 2]` into the other one and moves the particles of
        // `pbufs[(round - 1) % 2]` in place; a lifecycle step compacts them into the other buffer. The calls are built
        // before the clock starts.
        let run_step = |g: usize, mode: Mode| -> Result<StepTime, String> {
            let (f_in, f_out) = (&fields[(g - 1) % 2], &fields[g % 2]);
            let cur = (round(g) - 1) % 2;
            let (p_cur, p_next) = (&pbufs[cur], &pbufs[1 - cur]);
            let (pp, lp) = (&part_params[g - 1], &life_params[g - 1]);
            let live = live_of(g);
            let life = is_life(g);
            let fused = var == Var::Fused;
            let coarse = var == Var::CoarseBins;
            let bins = var.bins();
            let diffuse = kit::w2_diffuse::call_extent(
                extent(cells),
                &cell_params,
                Independent(f_in),
                Independent(f_out),
            );
            let step = (!fused).then(|| {
                kit::w3_step::call_extent(extent(live), pp, Independent(f_out), Independent(p_cur))
            });
            let clear_counts = bins.then(|| {
                kit::w4_clear::call_extent(
                    extent(bin_cells),
                    &bin_clear_params,
                    Independent(&counts),
                )
            });
            let clear_cursor = bins.then(|| {
                kit::w4_clear::call_extent(
                    extent(bin_cells),
                    &bin_clear_params,
                    Independent(&cursor),
                )
            });
            let clear_dep =
                kit::w4_clear::call_extent(extent(cells), &cell_params, Independent(&dep));
            let count = (bins && !fused && !coarse).then(|| {
                kit::w4_count::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&counts),
                )
            });
            let bin_count = coarse.then(|| {
                kit::w8_bin_count::call_extent(
                    extent(live),
                    &bin_params[g - 1],
                    Independent(p_cur),
                    Independent(&counts),
                )
            });
            let step_count_deposit = fused.then(|| {
                kit::w8_step_count_deposit::call_extent(
                    extent(live),
                    pp,
                    Independent(f_out),
                    Independent(p_cur),
                    Independent(&counts),
                    Independent(&dep),
                )
            });
            let deposit = (!fused).then(|| {
                kit::w8_deposit::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&dep),
                )
            });
            let fold = kit::w8_fold::call_extent(
                extent(cells),
                &cell_params,
                Independent(&dep),
                Independent(f_out),
            );
            let retire =
                life.then(|| kit::w8_retire::call_extent(extent(cap), lp, Independent(&alive)));
            let scatter = (bins && !coarse).then(|| {
                kit::w4_scatter::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&starts),
                    Independent(&cursor),
                    Independent(&order),
                )
            });
            let bin_scatter = coarse.then(|| {
                kit::w8_bin_scatter::call_extent(
                    extent(live),
                    &bin_params[g - 1],
                    Independent(p_cur),
                    Independent(&starts),
                    Independent(&cursor),
                    Independent(&order),
                )
            });
            let compact = life.then(|| {
                kit::w8_compact::call_extent(
                    extent(live),
                    lp,
                    Independent(p_cur),
                    Independent(&alive),
                    Independent(&astarts),
                    Independent(p_next),
                )
            });
            let spawn_call =
                life.then(|| kit::w8_spawn::call_extent(extent(spawn), lp, Independent(p_next)));
            let parts: Vec<Part<'_>> = [
                Some(Part::Kernel("diffuse", BufferDispatch::from(&diffuse))),
                kernel_part("step", &step),
                kernel_part("clear counts", &clear_counts),
                kernel_part("clear cursor", &clear_cursor),
                Some(Part::Kernel("clear dep", BufferDispatch::from(&clear_dep))),
                kernel_part("count", &count),
                kernel_part("bin count", &bin_count),
                kernel_part("step count deposit", &step_count_deposit),
                kernel_part("deposit", &deposit),
                Some(Part::Kernel("fold", BufferDispatch::from(&fold))),
                kernel_part("retire", &retire),
                bins.then_some(Part::Scan("scan counts", &counts, &starts)),
                life.then_some(Part::Scan("scan alive", &alive, &astarts)),
                kernel_part("scatter", &scatter),
                kernel_part("bin scatter", &bin_scatter),
                kernel_part("compact", &compact),
                kernel_part("spawn", &spawn_call),
            ]
            .into_iter()
            .flatten()
            .collect();
            let fused_groups = groups(&parts, Grain::Fused);
            let started = Instant::now();
            match mode {
                Mode::Timed => {
                    let scoped = timed(|| -> Result<(), gust_algo::Error> {
                        for range in &fused_groups {
                            run_group(device, &parts[range.clone()])?;
                        }
                        Ok(())
                    })
                    .map_err(dbg)?;
                    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
                    // W8 is four batches: the kernels, the scan of the counts, the scan of the alive flags, the kernels.
                    let expected = if var == Var::Base {
                        4
                    } else {
                        fused_groups.len()
                    };
                    if scoped.time.batches != expected {
                        return Err(format!(
                            "the timing scope counted {} batch(es), expected {expected} (the batches of kernels and the scans)",
                            scoped.time.batches
                        ));
                    }
                    Ok(StepTime {
                        gpu_ms: scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                        wall_ms,
                        submit_ms: None,
                        groups: Vec::new(),
                    })
                }
                Mode::Plain => {
                    for range in &fused_groups {
                        run_group(device, &parts[range.clone()]).map_err(dbg)?;
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: None,
                        groups: Vec::new(),
                    })
                }
                Mode::Submit => {
                    // The batch is built before the clock of the submit span starts.
                    let first = batch_of(&parts[fused_groups[0].clone()]);
                    let started = Instant::now();
                    let job = device.submit_batch(&first).map_err(dbg)?;
                    let submit_ms = started.elapsed().as_secs_f64() * 1e3;
                    job.wait().map_err(dbg)?;
                    for range in &fused_groups[1..] {
                        run_group(device, &parts[range.clone()]).map_err(dbg)?;
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: Some(submit_ms),
                        groups: Vec::new(),
                    })
                }
                Mode::Split(grain) => {
                    let mut timed_groups = Vec::new();
                    for range in groups(&parts, grain) {
                        let group = &parts[range];
                        let scoped = timed(|| run_group(device, group)).map_err(dbg)?;
                        if scoped.time.batches != 1 {
                            return Err(format!(
                                "the timing scope of {} counted {} batch(es), expected 1",
                                group_label(group),
                                scoped.time.batches
                            ));
                        }
                        timed_groups.push((
                            group_label(group),
                            scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                        ));
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: None,
                        groups: timed_groups,
                    })
                }
            }
        };
        let order_poison =
            GpuVec::from_slice_on(device, &vec![u32::MAX; order_n as usize]).map_err(dbg)?;
        let cap_params = uniform("w8 order", order_n, 0)?;
        let repoison = var.bins().then(|| {
            kit::w1_copy::call_extent(
                extent(order_n),
                &cap_params,
                Independent(&order_poison),
                Independent(&order),
            )
        });
        // One checked step: the GPU's state before it is read back, the step runs, and everything it produced is
        // compared with the host (the field and the moved particles from the GPU's own previous state, as W7 does).
        let verify = |g: usize| -> Result<(), String> {
            let live = live_of(g) as usize;
            let cur = (round(g) - 1) % 2;
            let prev_field = fields[(g - 1) % 2].to_vec().map_err(dbg)?;
            let prev_parts = pbufs[cur].to_vec().map_err(dbg)?;
            // The scatter's `order` is poisoned again (a copy of `u32::MAX`s, outside any timed scope), so a slot this
            // step does not write shows in the scatter check.
            if let Some(repoison) = &repoison {
                device
                    .dispatch_batch(&[BufferDispatch::from(repoison)])
                    .map_err(dbg)?;
            }
            run_step(g, Mode::Plain)?;
            let at_step = |e: String| format!("step {g}: {e}");
            let gpu_field = fields[g % 2].to_vec().map_err(dbg)?;
            let moved_all = pbufs[cur].to_vec().map_err(dbg)?;
            let stepped = &moved_all[..live];
            check_particles(stepped, &prev_parts[..live], &gpu_field, side).map_err(at_step)?;
            // The deposit grid: every live particle's amount in its cell, exactly.
            let dep_host: Vec<u32> = cpu_counts(stepped, side)
                .iter()
                .map(|c| c * kit::DEPOSIT)
                .collect();
            if let Some(reason) = first_diff(&dep.to_vec().map_err(dbg)?, &dep_host) {
                return Err(at_step(format!("deposit grid: {reason}")));
            }
            // The field: the host's diffusion of the GPU's previous field, then the deposit folded into lane w.
            let folded: Vec<Vec4> = cpu_diffuse(&prev_field, side)
                .iter()
                .zip(&dep_host)
                .map(|(v, d)| Vec4::new(v.x, v.y, v.z, v.w + *d as f32 / 1024.0))
                .collect();
            check_field(&gpu_field, &folded).map_err(at_step)?;
            // The bins: counts and their scan exactly, the scatter as sets per cell.
            if var.bins() {
                let counts_host = cpu_bin_counts(stepped, bin_side, side);
                if let Some(reason) = first_diff(&counts.to_vec().map_err(dbg)?, &counts_host) {
                    return Err(at_step(format!("counts: {reason}")));
                }
                let starts_host = cpu_exclusive(&counts_host);
                if let Some(reason) = first_diff(&starts.to_vec().map_err(dbg)?, &starts_host) {
                    return Err(at_step(format!("scan of the counts: {reason}")));
                }
                check_scatter(
                    &order.to_vec().map_err(dbg)?[..live],
                    &starts_host,
                    &counts_host,
                    stepped,
                    (bin_side, side),
                )
                .map_err(|reason| at_step(format!("scatter: {reason}")))?;
            }
            // The lifecycle.
            if is_life(g) {
                let (alive_gpu, astarts_gpu) =
                    (alive.to_vec().map_err(dbg)?, astarts.to_vec().map_err(dbg)?);
                let next_gpu = pbufs[1 - cur].to_vec().map_err(dbg)?;
                check_life(
                    round(g) as u32,
                    cap,
                    spawn,
                    side,
                    &plan[round(g) - 1],
                    &LifeGpu {
                        stepped,
                        alive: &alive_gpu,
                        astarts: &astarts_gpu,
                        next: &next_gpu,
                    },
                )
                .map_err(at_step)?;
            }
            Ok(())
        };
        verify(1)?;
        verify(2)?;
        let mut g = 2usize;
        // Plain steps up to the clamped one, which is checked.
        while g + 1 < clamp_at {
            g += 1;
            run_step(g, Mode::Plain)?;
        }
        if clamp_at > 2 {
            g += 1;
            verify(g).map_err(|e| format!("clamped step, {e}"))?;
        }
        // Run 1: the timed scope, the GPU time and the wall clock of every counted step.
        for _ in 0..warm {
            g += 1;
            run_step(g, Mode::Timed)?;
        }
        let mut blocking = Vec::with_capacity(steps);
        for _ in 0..steps {
            g += 1;
            blocking.push(run_step(g, Mode::Timed)?);
        }
        // Run 2: no scope; the first batch is submitted (the call timed, the wait outside), the rest blocks.
        for _ in 0..warm {
            g += 1;
            run_step(g, Mode::Submit)?;
        }
        let mut plain = Vec::with_capacity(steps);
        for _ in 0..steps {
            g += 1;
            plain.push(run_step(g, Mode::Submit)?);
        }
        // The per-group passes (W8 itself): each batch, then each kernel, in a timed scope of its own, the median of
        // `STEPS` steps after `WARMUP`; `None` where the device has no GPU timestamps.
        let pass = |grain: Grain, g: &mut usize| -> Result<Option<PassMedians>, String> {
            for _ in 0..WARMUP {
                *g += 1;
                run_step(*g, Mode::Split(grain))?;
            }
            let mut labels = Vec::new();
            let mut series: Vec<Vec<f64>> = Vec::new();
            let mut sums = Vec::new();
            let mut timestamps = true;
            for at in 0..STEPS {
                *g += 1;
                let step = run_step(*g, Mode::Split(grain))?;
                if at == 0 {
                    labels = step.groups.iter().map(|(label, _)| label.clone()).collect();
                    series = vec![Vec::new(); labels.len()];
                }
                let times: Vec<Option<f64>> = step.groups.iter().map(|(_, ms)| *ms).collect();
                match times.iter().copied().sum::<Option<f64>>() {
                    Some(sum) => sums.push(sum),
                    None => timestamps = false,
                }
                for (values, ms) in series.iter_mut().zip(times) {
                    values.extend(ms);
                }
            }
            Ok(timestamps.then(|| {
                (
                    labels
                        .into_iter()
                        .zip(&series)
                        .map(|(label, values)| (label, median(values)))
                        .collect(),
                    median(&sums),
                )
            }))
        };
        let passes_note = if var == Var::Base {
            let batches = pass(Grain::Fused, &mut g)?;
            let kernels = pass(Grain::Kernel, &mut g)?;
            format!(
                "; {}; {}; the sum of single-kernel batches may differ from the fused batches (a batch of kernels is one compute pass with one timestamp bracket; a one-kernel batch is a pass of its own)",
                pass_note(
                    &format!(
                        "per-batch GPU time (a separate pass after the sustained runs: each of the four batches in a timed scope of its own, median of {STEPS} steps after {WARMUP} warm-up steps)"
                    ),
                    batches.as_ref()
                ),
                pass_note(
                    &format!(
                        "per-kernel GPU time (a separate pass: every kernel a one-kernel batch in a timed scope of its own, each scan one scope, median of {STEPS} steps after {WARMUP} warm-up steps)"
                    ),
                    kernels.as_ref()
                )
            )
        } else {
            String::new()
        };
        // After the sustained runs: steps to the last, which is checked as the first two were, from whatever state the
        // runs left.
        while g + 1 < total {
            g += 1;
            run_step(g, Mode::Plain)?;
        }
        g += 1;
        if g != total {
            return Err(format!("ran {g} steps, planned {total}"));
        }
        verify(g).map_err(|e| format!("after the sustained runs, {e}"))?;
        let host_a: Vec<f64> = blocking.iter().map(|s| s.wall_ms).collect();
        let host_b: Vec<f64> = plain.iter().map(|s| s.wall_ms).collect();
        let submit_ms: Vec<f64> = plain.iter().filter_map(|s| s.submit_ms).collect();
        let (gpu_series, method) = if blocking.iter().all(|s| s.gpu_ms.is_some()) {
            (
                blocking
                    .iter()
                    .filter_map(|s| s.gpu_ms)
                    .collect::<Vec<f64>>(),
                "gpu timestamps",
            )
        } else {
            (host_a.clone(), "wall clock")
        };
        // The counted steps start on a round boundary, so every `period`th one is a lifecycle step. A variant with a
        // sparse lifecycle reports the mean per step (the amortised cost); the others the median.
        let (stat, ms) = if period > 1 {
            (
                "amortised mean",
                gpu_series.iter().sum::<f64>() / steps as f64,
            )
        } else {
            ("median", median(&gpu_series))
        };
        let measured = Measured { ms, method };
        let sparse_note = if period > 1 {
            let (life, rest): (Vec<_>, Vec<_>) = gpu_series
                .iter()
                .enumerate()
                .partition(|(at, _)| (at + 1) % period == 0);
            let medians = |part: &[(usize, &f64)]| {
                median(&part.iter().map(|(_, ms)| **ms).collect::<Vec<f64>>())
            };
            format!(
                "; median of the lifecycle steps {:.4} ms, median of the other steps {:.4} ms",
                medians(&life),
                medians(&rest)
            )
        } else {
            String::new()
        };
        let live_last = plan[round(total) - 1].next;
        row.note = format!(
            "the time column is the {stat} {method} time{}, p95 {:.4} ms, max {:.4} ms{sparse_note}; host (a) host_elapsed-style wall clock around the whole timed step (blocking batches with timestamp readback): median {:.4} ms, p95 {:.4} ms; host (b) wall clock around the whole step of the second run (the first batch through submit_batch and waited for, the scans and the other batches blocking, no timestamps): median {:.4} ms, p95 {:.4} ms; host (c) wall time for submit_batch of the first batch to return (encode and submit, no wait; the job is waited for after the span; measured with an idle queue, so it shows no back-pressure): median {:.4} ms, p95 {:.4} ms (the scans block, so unlike W7 the whole step cannot be submitted without a wait); the capacity clamp is checked at step {clamp_at}; live particles at the last step {live_last} of {cap}{passes_note}; {}",
            if method == "gpu timestamps" {
                " (the sum of the batches' GPU time; the gaps between the batches are not in it)"
            } else {
                " (the wall clock around the whole timed step: this device has no GPU timestamps)"
            },
            percentile(&gpu_series, 0.95),
            gpu_series.iter().copied().fold(f64::MIN, f64::max),
            median(&host_a),
            percentile(&host_a, 0.95),
            median(&host_b),
            percentile(&host_b, 0.95),
            median(&submit_ms),
            percentile(&submit_ms, 0.95),
            row.note
        );
        Ok((measured, None))
    })();
    row
}

fn w5(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W5 reduction (f32 sum)",
        size: format!("{n} f32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<f32> = (0..n).map(|i| unit(mix(i))).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let abs_sum = bound::abs_sum(&host);
        let reference = bound::sum_f64(&host);
        let allowed =
            bound::f32_sum_bound(n, bound::C_GPU, abs_sum) + bound::f64_reference_error(n, abs_sum);
        // The exact companion: a `u32` sum of `n` values, compared bit for bit with the host's wrapping sum, so a lost block
        // or a double-counted one cannot hide inside the float bound.
        let ints: Vec<u32> = (0..n).map(|i| mix(i) >> 8).collect();
        let int_input = GpuVec::from_slice_on(device, &ints).map_err(dbg)?;
        let int_sum = reduce::sum_wrapping(&int_input).map_err(dbg)?;
        let int_host = ints.iter().fold(0u32, |a, v| a.wrapping_add(*v));
        if int_sum != int_host {
            return Err(format!(
                "u32 sum {int_sum} differs from the host's {int_host}"
            ));
        }
        let first = timed(|| reduce::sum(&input)).map_err(dbg)?;
        let error = (f64::from(first.value) - reference).abs();
        row.note = if first.value.is_finite() {
            format!(
                "observed error {:.3e} of an allowed {:.3e} (bound::f32_sum_bound plus bound::f64_reference_error); bytes: one read per element; the f32 bound is wide (about {:.0}x the observed error), so a structural check is the exact u32 sum, which matched",
                error,
                allowed,
                allowed / error.max(f64::MIN_POSITIVE)
            )
        } else {
            format!(
                "the f32 sum was not finite ({}), so there is no observed error to compare with the allowed {:.3e}; the exact u32 sum matched",
                first.value, allowed
            )
        };
        // Finite and affirmatively within the bound, so a NaN cannot pass.
        if !(first.value.is_finite() && error <= allowed) {
            return Err(format!(
                "sum {} differs from the f64 reference {reference} by {error:.3e}, over {allowed:.3e}",
                first.value
            ));
        }
        let measured = measure(|| {
            let started = Instant::now();
            let sum = timed(|| reduce::sum(&input)).map_err(dbg)?;
            Ok(Sample {
                gpu_ms: sum.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                wall_ms: started.elapsed().as_secs_f64() * 1e3,
            })
        })?;
        Ok((measured, Some(4.0 * f64::from(n))))
    })();
    row
}

fn w6(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W6 readback delay",
        size: format!("{n} u32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<u32> = (0..n).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
        let expected: Vec<u32> = host.iter().map(|v| v.wrapping_add(1)).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let out = GpuVec::<u32>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w6",
                &kit::Params {
                    n,
                    side: 0,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call = kit::w6_plus_one::call_extent(
            extent(n),
            &params,
            Independent(&input),
            Independent(&out),
        );
        let batch = [BufferDispatch::from(&call)];
        let mut submit_ms = Vec::new();
        let mut read_ms = Vec::new();
        // Submit, wait for the submission, read the result back: the time until it is on the CPU. The result is compared
        // after the clock stops on every step, but only the first step proves the kernel wrote it (`out` starts zeroed;
        // later steps find the previous step's identical result there)."
        let mut step = || -> Result<Sample, String> {
            let started = Instant::now();
            device.dispatch_batch(&batch).map_err(dbg)?;
            let submitted = started.elapsed();
            let result = out.to_vec().map_err(dbg)?;
            let total = started.elapsed();
            if result != expected {
                return Err("the result differs from the host's".into());
            }
            submit_ms.push(submitted.as_secs_f64() * 1e3);
            read_ms.push((total - submitted).as_secs_f64() * 1e3);
            Ok(Sample {
                gpu_ms: None,
                wall_ms: total.as_secs_f64() * 1e3,
            })
        };
        let measured = measure(&mut step)?;
        // The split of the counted steps (the last STEPS entries).
        let from = submit_ms.len() - STEPS;
        row.note = format!(
            "submit and wait {:.4} ms, readback {:.4} ms (medians of the same steps); wall clock around submit, wait and readback; the result is proven written on the first step only",
            median(&submit_ms[from..]),
            median(&read_ms[from..])
        );
        Ok((measured, None))
    })();
    row
}

// ---- the G4 probe ----

/// The G4 probe launches 65 536 workgroups of 256 on x, one above the portable limit of 65 535.
const G4_WORKGROUPS: u32 = 65_536;
const G4_THREADS: u32 = G4_WORKGROUPS * 256;
/// The remedy's row width, in invocations.
const G4_WIDTH: u32 = 4096;

fn g4_expected(i: u32) -> u32 {
    i.wrapping_mul(3).wrapping_add(1)
}

/// The limits policy of the device `main` opens: `HeadlessDevice::new` requests wgpu's default limits
/// (`LimitsPolicy::Default`). Keep this text in step with the call in `main`.
const KIT_LIMITS_POLICY: &str = "Default (HeadlessDevice::new)";

/// The remedy's element count: not a multiple of the row width or the workgroup size, so the last row is partly past
/// the end and the kernel's `i >= n` guard has work to do.
const G4_REMEDY_N: u32 = G4_THREADS - 1000;
/// What the remedy's buffer holds before the launch, and in the slots past `n` after it.
const G4_SENTINEL: u32 = 0xDEAD_BEEF;

/// The probe of gap G4 (a linear launch above the workgroup limit on x) and its remedy. It is not a workload: it
/// reports what happens on this device. Since slice m11.g4b gust-wgpu splits such a launch into pieces and the probe
/// succeeds; a refusal, where there is one, is its own host-side check of the launch against `limits()`, made before
/// anything is submitted. Returns the lines to print and whether a check failed.
fn g4(device: &DeviceHandle) -> (Vec<String>, bool) {
    let max = device.limits().max_compute_workgroups_per_dimension;
    let context = format!(
        "device max_compute_workgroups_per_dimension {max}, requested limits policy {KIT_LIMITS_POLICY}"
    );
    let n = G4_THREADS;
    let mut failed = false;
    // `Ok(None)` is a launch whose result matched the host exactly; `Ok(Some(reason))` one that ran and differed.
    let run = || -> Result<Option<String>, gust_wgpu::Error> {
        let out = GpuVec::<u32>::zeroed_on(device, n)?;
        let params = device.create_uniform(
            "g4",
            &kit::Params {
                n,
                side: 0,
                pad0: 0,
                pad1: 0,
            },
        )?;
        let call = kit::g4_linear::call_extent(Extent::new(n, 1, 1), &params, Independent(&out));
        device.dispatch_batch(&[BufferDispatch::from(&call)])?;
        let got = out.to_vec()?;
        Ok(got
            .iter()
            .enumerate()
            .find(|(i, v)| **v != g4_expected(*i as u32))
            .map(|(i, v)| format!("element {i} is {v}")))
    };
    let line = match run() {
        Ok(None) => format!(
            "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) succeeded and matched the host exactly; {context}"
        ),
        Ok(Some(reason)) => {
            failed = true;
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) ran but FAILED the host check ({reason}); {context}"
            )
        }
        // The intended rejection and nothing else: the workgroup limit of x, the requested count and the maximum the
        // device reports.
        Err(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) if limit == "max_compute_workgroups_per_dimension"
            && requested == u64::from(G4_WORKGROUPS)
            && maximum == u64::from(max) =>
        {
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) was refused with DeviceLimit: limit {limit}, requested {requested}, maximum {maximum}; {context}"
            )
        }
        Err(other) => {
            failed = true;
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) FAILED with an error other than the intended DeviceLimit: {other:?}; {context}"
            )
        }
    };
    let rows = G4_REMEDY_N.div_ceil(G4_WIDTH);
    let slots = rows * G4_WIDTH;
    let remedy = || -> Result<(), String> {
        let out = GpuVec::from_slice_on(device, &vec![G4_SENTINEL; slots as usize]).map_err(dbg)?;
        let params = device
            .create_uniform(
                "g4 grid",
                &kit::Params {
                    n: G4_REMEDY_N,
                    side: G4_WIDTH,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call =
            kit::g4_grid::call_extent(Extent::new(G4_WIDTH, rows, 1), &params, Independent(&out));
        device
            .dispatch_batch(&[BufferDispatch::from(&call)])
            .map_err(dbg)?;
        let got = out.to_vec().map_err(dbg)?;
        let (values, tail) = got.split_at(G4_REMEDY_N as usize);
        if let Some(at) = values
            .iter()
            .enumerate()
            .position(|(i, v)| *v != g4_expected(i as u32))
        {
            return Err(format!("element {at} differs from the host"));
        }
        if let Some(at) = tail.iter().position(|v| *v != G4_SENTINEL) {
            return Err(format!(
                "slot {} past n was written (the guard did not hold)",
                G4_REMEDY_N as usize + at
            ));
        }
        Ok(())
    };
    let result = remedy();
    failed |= result.is_err();
    let remedy_line = match result {
        Ok(()) => format!(
            "G4 remedy: ok, {G4_REMEDY_N} elements (not a multiple of the width) as Extent::new({G4_WIDTH}, {rows}, 1) with `i = id.y * width + id.x; if i >= n {{ return; }}`: all values exact against the host and all {} slots past n still hold the sentinel",
            slots - G4_REMEDY_N
        ),
        Err(reason) => format!("G4 remedy: FAILED, {reason}"),
    };
    (vec![line, remedy_line], failed)
}

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

/// The total bytes of Slang text of this kit's kernels and of their split variants (slice m11.g4b): what the variants
/// add to the binary. A measurement, with no target.
fn variant_text_line() -> String {
    let kernels = [
        &kit::w1_copy::DESCRIPTOR,
        &kit::w2_diffuse::DESCRIPTOR,
        &kit::w3_step::DESCRIPTOR,
        &kit::w4_clear::DESCRIPTOR,
        &kit::w4_count::DESCRIPTOR,
        &kit::w4_scatter::DESCRIPTOR,
        &kit::g4_linear::DESCRIPTOR,
        &kit::g4_grid::DESCRIPTOR,
        &kit::w6_plus_one::DESCRIPTOR,
        &kit::w8_deposit::DESCRIPTOR,
        &kit::w8_fold::DESCRIPTOR,
        &kit::w8_retire::DESCRIPTOR,
        &kit::w8_compact::DESCRIPTOR,
        &kit::w8_spawn::DESCRIPTOR,
        &kit::w8_bin_count::DESCRIPTOR,
        &kit::w8_bin_scatter::DESCRIPTOR,
        &kit::w8_step_count_deposit::DESCRIPTOR,
    ];
    let base: usize = kernels.iter().map(|kernel| kernel.slang_source.len()).sum();
    let split: usize = kernels
        .iter()
        .filter_map(|kernel| kernel.split)
        .map(|variant| variant.slang_source.len())
        .sum();
    format!(
        "Variant text: {} kernels, {base} bytes of base Slang text and {split} bytes of split-variant Slang text ({:.2}x)",
        kernels.len(),
        split as f64 / base as f64
    )
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
    let info = device.adapter_info();
    let limits = device.limits();
    let mut text = String::new();
    let _ = writeln!(
        text,
        "budget-kit ({}): a one-machine design aid, not a benchmark; no threshold is asserted",
        if args.small {
            "small: the first size of each workload"
        } else {
            "full"
        }
    );
    let _ = writeln!(
        text,
        "adapter {} ({:?}, {:?}, driver {} {})",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );
    let _ = writeln!(
        text,
        "gpu timestamps: {}",
        if device.supports_gpu_timestamps() {
            "available"
        } else {
            "not available (rows fall back to the wall clock)"
        }
    );
    let _ = writeln!(
        text,
        "limits: storage binding {} bytes, buffer {} bytes, workgroups per dimension {}, invocations per workgroup {}, workgroup size x {}",
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x
    );
    let _ = writeln!(
        text,
        "statistic: median of {STEPS} steps after {WARMUP} warm-up steps; a step is one dispatch (W4: a clear, a count and a scan, and in its second row a scatter; W7: one batch of four kernels, a sustained run with its own statistics in the note; W8: a whole step of four batches (two of them scans) in one timing scope, a sustained run with its own statistics in the note; W8a: the same without bins, three batches (one scan); W8b and W8c: four batches as W8; W8d: the lifecycle on every 4th step only, so the steps between are three batches, and its time is the amortised mean per step over whole rounds of 4 steps, not a median; W5: a whole reduction; with GPU timestamps the pooled-buffer clears are outside the timed region, but the wall-clock fallback includes the clears, the allocation (not that of the W4 scan output or the scatter call, which are made once before the steps), the host overhead and the scalar readback; W6: submit, wait and readback); steps per frame is {FRAME_MS:.1} ms over the median; GB/s is the stated bytes over the median"
    );
    let _ = writeln!(
        text,
        "sizes are exact counts; 1M is 2^20 and 4M is 2^22 elements"
    );
    let _ = writeln!(text);
    let _ = writeln!(
        text,
        "| workload | size | timing method | ms per step | GB/s | steps per 16.7 ms | check | note |"
    );
    let _ = writeln!(text, "|---|---|---|---:|---:|---:|---|---|");
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
