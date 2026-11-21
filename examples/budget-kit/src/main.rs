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
        // One step: the counts the kernel made against the host's histogram, and the scan of them against the host's
        // exclusive prefix, both exactly (integers).
        // The clear and count batch goes through `dispatch_batch_in_scope` inside the same `timed` scope as the scan, so
        // the scope's GPU time covers both and its `batches` is the clear and count batch plus the scan's own.
        let step = || -> Result<(Sample, GpuVec<u32>), String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &batch)?;
                scan::exclusive(&counts)
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
            Ok((Sample { gpu_ms, wall_ms }, scoped.value))
        };
        let (_, positions) = step()?;
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        if counts.to_vec().map_err(dbg)? != cpu {
            return Err("the atomic counts differ from the host histogram".into());
        }
        if positions.to_vec().map_err(dbg)? != cpu_exclusive(&cpu) {
            return Err("the scan differs from the host exclusive prefix".into());
        }
        let measured = measure(|| step().map(|(sample, _)| sample))?;
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
    row.note = "one timing scope covers the clear and count batch and the scan, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps); bytes: 16 + 8 per particle, 12 per cell, nominal (the atomic counted as 8), so the GB/s is not a memory bandwidth".into();
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
/// unspecified), and every index appears once. The host's sets come from a counting sort of the host's own particles.
fn check_scatter(
    order: &[u32],
    starts: &[u32],
    counts: &[u32],
    parts_host: &[Vec4],
    side: u32,
) -> Result<(), String> {
    let cell_of_host = |q: &Vec4| {
        let cx = (q.x as u32).min(side - 1);
        let cy = (q.y as u32).min(side - 1);
        (cy * side + cx) as usize
    };
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
        // One step: the clears and the count (one batch), the scan, the scatter (one batch): three batches in the one
        // `timed` scope, so its GPU time covers the whole binning.
        let step = || -> Result<(Sample, GpuVec<u32>), String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &first)?;
                let starts = scan::exclusive(&counts)?;
                let scatter = kit::w4_scatter::call_extent(
                    extent(count),
                    &count_params,
                    Independent(&parts),
                    Independent(&starts),
                    Independent(&cursor),
                    Independent(&order),
                );
                dispatch_batch_in_scope(device, &[BufferDispatch::from(&scatter)])?;
                Ok(starts)
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
            Ok((Sample { gpu_ms, wall_ms }, scoped.value))
        };
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        let cpu_starts = cpu_exclusive(&cpu);
        let verify = |starts: &GpuVec<u32>, after: &str| -> Result<(), String> {
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
                W4_GRID_SIDE,
            )
            .map_err(|reason| format!("scatter{after}: {reason}"))
        };
        let (_, starts) = step()?;
        verify(&starts, "")?;
        let mut last = starts;
        let measured = measure(|| {
            let (sample, starts) = step()?;
            last = starts;
            Ok(sample)
        })?;
        // A later step as well: counters or cursors that were not cleared between steps have grown by now.
        verify(&last, " after the timed steps")?;
        // Per particle: the particle read (16), the count atomic (8), the scatter's particle read (16), the start read
        // (4), the cursor atomic (8) and the order write (4); per cell: the two clears (8) and the scan (12).
        let bytes = 56.0 * f64::from(count) + 20.0 * f64::from(cells);
        Ok((measured, Some(bytes)))
    })();
    row.note = "the order inside a cell is unspecified (checked as a set per cell, each range sorted first, and every index exactly once); one timing scope covers the clears and count batch, the scan and the scatter batch, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps); the particles never move and `order` is not re-poisoned between steps, so the check after the timed steps does not catch a write lost in a later step; bytes: nominal (each atomic counted as 8), so the GB/s is not a memory bandwidth".into();
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
        "{steps} counted steps after {WARMUP} warm-up steps, in two runs of that length (blocking, then submit); one batch per step: w2_diffuse on {side}x{side}, w3_step, w4_clear and w4_count into {side}x{side} cells; checked after step 1 (field 5e-6 relative as W2, particles as W3 over the GPU's own field, counts exact against the GPU's own positions) and again after the runs (particles in range, counts exact)"
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
        let mut next = 1usize;
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
        "statistic: median of {STEPS} steps after {WARMUP} warm-up steps; a step is one dispatch (W4: a clear, a count and a scan, and in its second row a scatter; W7: one batch of four kernels, a sustained run with its own statistics in the note; W5: a whole reduction; with GPU timestamps the pooled-buffer clears are outside the timed region, but the wall-clock fallback includes the clears, the allocation, the host overhead and the scalar readback; W6: submit, wait and readback); steps per frame is {FRAME_MS:.1} ms over the median; GB/s is the stated bytes over the median"
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
    type Workload = fn(&DeviceHandle, u32) -> Row;
    let plan: [(Workload, Vec<u32>); 7] = [
        (w1, pick(&W1_SIZES)),
        (w2, pick(&W2_SIDES)),
        (w3, pick(&W3_SIZES)),
        (w5, pick(&W5_SIZES)),
        (w4, pick(&W4_SIZES)),
        (w4_scatter, pick(&W4_SIZES)),
        (w6, pick(&W6_SIZES)),
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
