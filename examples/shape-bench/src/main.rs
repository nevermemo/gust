//! The same kernels as `grid` and as `pixel` kernels (slice mx.p1c): the bit identity of the two shapes per cell, and the
//! time of each. It prints X-S1's table (five kernels, five square sizes) for storage-write pixel kernels (D120, D127)
//! and asserts no time and no ratio: a `pixel` kernel is not slower than a `grid` one by decree, it is measured.
//!
//! ```text
//! cargo run -p shape-bench                       the smoke: sides 256 and 1024, one run of 3 warm-up + 5 timed dispatches
//! cargo run --release -p shape-bench -- --full [--out PATH]   the table: five sides, 5 runs of 5 warm-up + 20 timed
//! ```
//!
//! What is asserted, in both modes: the descriptors carry the shapes they were declared with; every variant writes
//! every element (two passes over different poison agree) and every `f32` is finite; and for K1, the integer kernel,
//! the three grid kernels (workgroup widths 64, 128, 256) and the pixel kernel write the same bits. What is observed and
//! printed, not asserted: the `f32` bits of the two shapes. A device may fuse `a * b + c` and round `x / y` within
//! 2.5 ulp differently in the compute and the fragment stage (SEMANTICS section 4, D128), and the table says by how
//! much (`N of M words, <= U ulp`). Correctness lives in the difftest and the gust-wgpu tests, not in a benchmark, so no
//! observed maximum is turned into a bound. What is not asserted either: any duration. The numbers of the smoke are noisy
//! (a shared GPU, a debug build, three dispatches) and are labelled so.
//!
//! The kernels are X-S1's, over storage buffers instead of textures (Option S): K1 a hash and a gather (`u32`), K2 a 3x3
//! blur (`f32`), K3 a 64-term polynomial (`f32`), K4 the blur on a `Vec4`, K3L the polynomial with 1024 terms. K3 and K3L
//! are X-S1's form without `fma`: the dialect has none, so `acc * t + c` is an unfused multiply and add (`t` and the
//! coefficients are X-S1's, exact in `f32`), which a device is free to contract. Element `i`
//! is pixel `(i % side, i / side)` of a `side x side` image. A `grid` dispatch covers `8192 x (n / 8192)` invocations
//! (every size here is a whole number of rows of `gust::PIXEL_ROW_WIDTH`, so there is no tail) and indexes
//! `id.y * 8192 + id.x`; a `pixel` dispatch is one call of `n` elements.

use std::fmt::Write as _;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gust::{
    Extent, PIXEL_MAX_ELEMENTS, PIXEL_ROW_WIDTH, Shape, gpu, math::Vec4, pod_slice_as_bytes,
};
use gust_wgpu::{BufferDispatch, DeviceOptions, GpuBuffer, GpuBufferAccess, HeadlessDevice};

// One module per variant: the kernels are written once, in this macro, and differ only in the shape attribute and in how
// the element index is computed from the invocation id. (A module that mixes an explicit workgroup size with a pixel
// kernel is E0707 until mx.p1b's fix; one module per variant does not depend on that fix.) The identifier of the
// invocation id and the index expression come from the call site, so both name the same binding.
macro_rules! variant_module {
    ($m:ident, $id:ident, [$($attr:tt)*], $index:expr) => {
        #[gpu]
        mod $m {
            pub struct Params {
                pub w: u32,
                pub h: u32,
                pub pad0: u32,
                pub pad1: u32,
            }

            fn hash(x: u32, y: u32) -> u32 {
                let mut h = x
                    .wrapping_mul(2654435761u32)
                    .wrapping_add(y.wrapping_mul(2246822507u32))
                    .wrapping_add(668265263u32);
                h = h ^ (h >> 15u32);
                h = h.wrapping_mul(739982445u32);
                h = h ^ (h >> 12u32);
                h = h.wrapping_mul(695872825u32);
                h ^ (h >> 15u32)
            }

            // X-S1's K3 without `fma`: the dialect has none, so `acc * t + c` is an unfused multiply and add, which a
            // device may contract. `t` and `c` are exact in `f32`, as in X-S1.
            fn poly(t: f32, terms: u32) -> f32 {
                let mut acc = 0.0f32;
                for k in 0u32..terms {
                    let c = ((k * 37u32 + 11u32) % 101u32) as f32 * 0.0078125f32 - 0.375f32;
                    acc = acc * t + c;
                }
                acc
            }

            fn poly_t(i: u32, w: u32) -> f32 {
                ((hash(i % w, i / w) & 65535u32) as f32) * 0.000_015_258_789_f32
            }

            #[kernel($($attr)*)]
            pub fn k1(
                $id: SV_DispatchThreadID,
                p: Uniform<Params>,
                input: StructuredBuffer<u32>,
                mut out: RWStructuredBuffer<u32>,
            ) {
                let i = $index;
                if i < out.len() {
                    out[i] = hash(i % p.w, i / p.w) ^ input[(i + 1u32) % input.len()];
                }
            }

            #[kernel($($attr)*)]
            pub fn k2(
                $id: SV_DispatchThreadID,
                p: Uniform<Params>,
                input: StructuredBuffer<f32>,
                mut out: RWStructuredBuffer<f32>,
            ) {
                let i = $index;
                if i < out.len() {
                    let x = i % p.w;
                    let y = i / p.w;
                    let mut s = 0.0f32;
                    for dy in 0u32..3u32 {
                        let mut cy = y + dy;
                        if cy < 1u32 {
                            cy = 1u32;
                        }
                        if cy > p.h {
                            cy = p.h;
                        }
                        for dx in 0u32..3u32 {
                            let mut cx = x + dx;
                            if cx < 1u32 {
                                cx = 1u32;
                            }
                            if cx > p.w {
                                cx = p.w;
                            }
                            s += input[(cy - 1u32) * p.w + (cx - 1u32)];
                        }
                    }
                    out[i] = s * 0.11111111f32;
                }
            }

            #[kernel($($attr)*)]
            pub fn k3(
                $id: SV_DispatchThreadID,
                p: Uniform<Params>,
                mut out: RWStructuredBuffer<f32>,
            ) {
                let i = $index;
                if i < out.len() {
                    out[i] = poly(poly_t(i, p.w), 64u32);
                }
            }

            #[kernel($($attr)*)]
            pub fn k3l(
                $id: SV_DispatchThreadID,
                p: Uniform<Params>,
                mut out: RWStructuredBuffer<f32>,
            ) {
                let i = $index;
                if i < out.len() {
                    out[i] = poly(poly_t(i, p.w), 1024u32);
                }
            }

            #[kernel($($attr)*)]
            pub fn k4(
                $id: SV_DispatchThreadID,
                p: Uniform<Params>,
                input: StructuredBuffer<Vec4>,
                mut out: RWStructuredBuffer<Vec4>,
            ) {
                let i = $index;
                if i < out.len() {
                    let x = i % p.w;
                    let y = i / p.w;
                    let mut s = Vec4::splat(0.0f32);
                    for dy in 0u32..3u32 {
                        let mut cy = y + dy;
                        if cy < 1u32 {
                            cy = 1u32;
                        }
                        if cy > p.h {
                            cy = p.h;
                        }
                        for dx in 0u32..3u32 {
                            let mut cx = x + dx;
                            if cx < 1u32 {
                                cx = 1u32;
                            }
                            if cx > p.w {
                                cx = p.w;
                            }
                            s += input[(cy - 1u32) * p.w + (cx - 1u32)];
                        }
                    }
                    out[i] = s * 0.11111111f32;
                }
            }
        }
    };
}

variant_module!(
    grid64,
    id,
    [workgroup_size(64, 1, 1)],
    id.y * 8192u32 + id.x
);
variant_module!(
    grid128,
    id,
    [workgroup_size(128, 1, 1)],
    id.y * 8192u32 + id.x
);
variant_module!(
    grid256,
    id,
    [workgroup_size(256, 1, 1)],
    id.y * 8192u32 + id.x
);
variant_module!(px, id, [shape = pixel], id.x);

/// The variants of a cell, in the order of the table's columns. The last one is the pixel kernel.
const VARIANTS: [&str; 4] = ["grid64", "grid128", "grid256", "pixel"];

/// The invocations of a `grid` dispatch over `side * side` elements: whole rows of 8192, no tail.
fn grid_extent(side: u32) -> Extent {
    Extent::new(PIXEL_ROW_WIDTH, side * side / PIXEL_ROW_WIDTH, 1)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kern {
    K1,
    K2,
    K3,
    K4,
    K3L,
}

const KERNELS: [Kern; 5] = [Kern::K1, Kern::K2, Kern::K3, Kern::K4, Kern::K3L];

impl Kern {
    fn name(self) -> &'static str {
        match self {
            Self::K1 => "K1",
            Self::K2 => "K2",
            Self::K3 => "K3",
            Self::K4 => "K4",
            Self::K3L => "K3L",
        }
    }

    /// Whether the output is `f32`s (a `Vec4` is four): the compute and the fragment stage of a device may fuse
    /// `a * b + c` and round `x / y` differently, so the bits of the two shapes are observed, not asserted (D128).
    fn is_float(self) -> bool {
        self != Self::K1
    }

    /// Bytes of one element of the output (and of the input, where there is one).
    fn stride(self) -> u64 {
        if self == Self::K4 { 16 } else { 4 }
    }
}

// ---- inputs ----

fn mix(i: u32) -> u32 {
    i.wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

fn input_u32(i: u32) -> u32 {
    mix(i)
}

fn input_f32(i: u32) -> f32 {
    (mix(i) & 0xFFFF) as f32 * (1.0 / 65_536.0)
}

fn input_v4(i: u32) -> Vec4 {
    let s = mix(i);
    Vec4::new(
        (s & 0xFF) as f32 * (1.0 / 256.0),
        ((s >> 8) & 0xFF) as f32 * (1.0 / 256.0),
        ((s >> 16) & 0xFF) as f32 * (1.0 / 256.0),
        (s >> 24) as f32 * (1.0 / 256.0),
    )
}

/// The buffers of one kernel at one size: the input where it has one, and the output.
struct Buffers {
    n: usize,
    u32_in: Option<GpuBuffer<u32>>,
    u32_out: Option<GpuBuffer<u32>>,
    f32_in: Option<GpuBuffer<f32>>,
    f32_out: Option<GpuBuffer<f32>>,
    v4_in: Option<GpuBuffer<Vec4>>,
    v4_out: Option<GpuBuffer<Vec4>>,
}

/// The bits an output starts with, and the bits of a second pass: a word a kernel never wrote keeps the poison it
/// started with, so the two passes differ in exactly the words that were not written.
const POISON: u32 = 0xA5A5_A5A5;
const POISON_B: u32 = 0x5A5A_5A5A;

impl Buffers {
    fn new(device: &HeadlessDevice, kern: Kern, side: u32) -> Self {
        let n = (side * side) as usize;
        let rw = GpuBufferAccess::ReadWrite;
        let mut b = Self {
            n,
            u32_in: None,
            u32_out: None,
            f32_in: None,
            f32_out: None,
            v4_in: None,
            v4_out: None,
        };
        match kern {
            Kern::K1 => {
                let input: Vec<u32> = (0..n as u32).map(input_u32).collect();
                b.u32_in = Some(device.create_typed_buffer("k1 in", &input, rw));
                b.u32_out = Some(device.create_typed_buffer("k1 out", &vec![POISON; n], rw));
            }
            Kern::K2 => {
                let input: Vec<f32> = (0..n as u32).map(input_f32).collect();
                b.f32_in = Some(device.create_typed_buffer("k2 in", &input, rw));
                b.f32_out = Some(device.create_typed_buffer(
                    "k2 out",
                    &vec![f32::from_bits(POISON); n],
                    rw,
                ));
            }
            Kern::K3 | Kern::K3L => {
                b.f32_out = Some(device.create_typed_buffer(
                    "k3 out",
                    &vec![f32::from_bits(POISON); n],
                    rw,
                ));
            }
            Kern::K4 => {
                let input: Vec<Vec4> = (0..n as u32).map(input_v4).collect();
                b.v4_in = Some(device.create_typed_buffer("k4 in", &input, rw));
                let poison = Vec4::splat(f32::from_bits(POISON));
                b.v4_out = Some(device.create_typed_buffer("k4 out", &vec![poison; n], rw));
            }
        }
        b
    }

    /// Writes `word` over every word of the output.
    fn poison(&self, device: &HeadlessDevice, word: u32) -> Result<(), String> {
        let done = if let Some(out) = &self.u32_out {
            device.write_typed_buffer(out, &vec![word; self.n])
        } else if let Some(out) = &self.f32_out {
            device.write_typed_buffer(out, &vec![f32::from_bits(word); self.n])
        } else if let Some(out) = &self.v4_out {
            let poison = Vec4::splat(f32::from_bits(word));
            device.write_typed_buffer(out, &vec![poison; self.n])
        } else {
            return Err("no output buffer".into());
        };
        done.map_err(|error| format!("poison: {error:?}"))
    }

    /// The output as read back, as 32-bit words (a `Vec4` is four of them).
    fn words(&self, device: &HeadlessDevice) -> Result<Vec<u32>, String> {
        let read = |error: gust_wgpu::Error| format!("readback: {error:?}");
        let bytes = if let Some(out) = &self.u32_out {
            pod_slice_as_bytes(&device.read_typed_buffer(out).map_err(read)?).to_vec()
        } else if let Some(out) = &self.f32_out {
            pod_slice_as_bytes(&device.read_typed_buffer(out).map_err(read)?).to_vec()
        } else if let Some(out) = &self.v4_out {
            pod_slice_as_bytes(&device.read_typed_buffer(out).map_err(read)?).to_vec()
        } else {
            return Err("no output buffer".into());
        };
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect())
    }

    fn u32_in(&self) -> &GpuBuffer<u32> {
        self.u32_in.as_ref().expect("K1 input")
    }
    fn u32_out(&self) -> &GpuBuffer<u32> {
        self.u32_out.as_ref().expect("K1 output")
    }
    fn f32_in(&self) -> &GpuBuffer<f32> {
        self.f32_in.as_ref().expect("K2 input")
    }
    fn f32_out(&self) -> &GpuBuffer<f32> {
        self.f32_out.as_ref().expect("f32 output")
    }
    fn v4_in(&self) -> &GpuBuffer<Vec4> {
        self.v4_in.as_ref().expect("K4 input")
    }
    fn v4_out(&self) -> &GpuBuffer<Vec4> {
        self.v4_out.as_ref().expect("K4 output")
    }
}

/// Runs a typed call `warmup + timed` times as a timed batch of one and returns the GPU time of the timed ones in
/// milliseconds; empty when the adapter has no timestamp queries (the call still ran).
macro_rules! time_call {
    ($device:expr, $call:expr, $warmup:expr, $timed:expr) => {{
        let batch = [BufferDispatch::from(&$call)];
        let mut samples = Vec::new();
        for round in 0..($warmup + $timed) {
            let timing = $device
                .dispatch_batch_timed(&batch)
                .map_err(|error| format!("dispatch: {error:?}"))?;
            if round >= $warmup {
                if let Some(elapsed) = timing.gpu_elapsed {
                    samples.push(elapsed.as_secs_f64() * 1e3);
                }
            }
        }
        Ok(samples)
    }};
}

type RunFn = fn(&HeadlessDevice, Kern, u32, &Buffers, usize, usize) -> Result<Vec<f64>, String>;

/// A function that runs one variant of any kernel. `$call` is the typed call (`call_extent` for a grid kernel, `call` for
/// a pixel kernel) and `$first` makes its first argument (the extent, or the element count) from the side.
macro_rules! variant_runner {
    ($name:ident, $m:ident, $call:ident, $first:expr) => {
        fn $name(
            device: &HeadlessDevice,
            kern: Kern,
            side: u32,
            b: &Buffers,
            warmup: usize,
            timed: usize,
        ) -> Result<Vec<f64>, String> {
            let params = $m::Params {
                w: side,
                h: side,
                pad0: 0,
                pad1: 0,
            };
            let uniform = device
                .create_uniform("params", &params)
                .map_err(|error| format!("uniform: {error:?}"))?;
            let first = ($first)(side);
            match kern {
                Kern::K1 => {
                    let call = $m::k1::$call(first, &uniform, b.u32_in(), b.u32_out());
                    time_call!(device, call, warmup, timed)
                }
                Kern::K2 => {
                    let call = $m::k2::$call(first, &uniform, b.f32_in(), b.f32_out());
                    time_call!(device, call, warmup, timed)
                }
                Kern::K3 => {
                    let call = $m::k3::$call(first, &uniform, b.f32_out());
                    time_call!(device, call, warmup, timed)
                }
                Kern::K3L => {
                    let call = $m::k3l::$call(first, &uniform, b.f32_out());
                    time_call!(device, call, warmup, timed)
                }
                Kern::K4 => {
                    let call = $m::k4::$call(first, &uniform, b.v4_in(), b.v4_out());
                    time_call!(device, call, warmup, timed)
                }
            }
        }
    };
}

variant_runner!(run_grid64, grid64, call_extent, grid_extent);
variant_runner!(run_grid128, grid128, call_extent, grid_extent);
variant_runner!(run_grid256, grid256, call_extent, grid_extent);
variant_runner!(run_pixel, px, call, |side: u32| side * side);

const RUNNERS: [RunFn; 4] = [run_grid64, run_grid128, run_grid256, run_pixel];

// ---- the statistic ----

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

/// How a cell is timed: `runs` runs, each `warmup` dispatches that are not counted and `timed` that are; a run's value
/// is the median of its timed dispatches, a cell's is the median of its runs' values. The variants run in an order
/// rotated by one for each run, so none of them always goes first.
#[derive(Clone, Copy, Debug)]
struct Plan {
    runs: usize,
    warmup: usize,
    timed: usize,
}

/// X-S1's statistic.
const FULL: Plan = Plan {
    runs: 5,
    warmup: 5,
    timed: 20,
};

/// The smoke: three warm-up and five timed dispatches, once. Noisy by construction.
const SMOKE: Plan = Plan {
    runs: 1,
    warmup: 3,
    timed: 5,
};

/// How the pixel output compares with a grid output, word by word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Comparison {
    words: usize,
    /// Words whose bits differ.
    differing: usize,
    /// The largest distance in `f32` ulps over the differing words (0 for integer outputs, which are not measured).
    max_ulp: u32,
}

/// The distance in `f32` ulps between two bit patterns on the line of ordered floats; a NaN has none and counts as the
/// largest.
fn ulp_distance(a: u32, b: u32) -> u32 {
    if f32::from_bits(a).is_nan() || f32::from_bits(b).is_nan() {
        return u32::MAX;
    }
    let ordered = |bits: u32| -> i64 {
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    };
    u32::try_from((ordered(a) - ordered(b)).unsigned_abs()).unwrap_or(u32::MAX)
}

fn compare(kern: Kern, grid: &[u32], pixel: &[u32]) -> Comparison {
    let mut found = Comparison {
        words: grid.len().min(pixel.len()),
        ..Comparison::default()
    };
    for (g, p) in grid.iter().zip(pixel) {
        if g != p {
            found.differing += 1;
            if kern.is_float() {
                found.max_ulp = found.max_ulp.max(ulp_distance(*g, *p));
            }
        }
    }
    found
}

/// What one cell measured.
struct Cell {
    /// Median of run medians, milliseconds, per variant; `None` when the adapter has no timestamp queries.
    times: [Option<f64>; 4],
    /// The pixel kernel's run medians' (max - min) / median, in percent.
    pixel_spread: Option<f64>,
    /// Every variant wrote every element (two passes over different poison agree).
    written: bool,
    /// Every `f32` of every variant is finite (trivially true for the integer kernel).
    finite: bool,
    /// The pixel output against the grid outputs: the most differing words and the largest ulp distance over the three
    /// workgroup widths.
    pixel_vs_grid: Comparison,
    /// Words in which the three grid widths differ among themselves (an observation, like the above).
    grid_widths_differ: usize,
    /// The cell meets what is asserted: written, finite, and for the integer kernel bit for bit the same in all four
    /// variants. The `f32` bits of the two shapes are not asserted (D128).
    accepted: bool,
}

impl Cell {
    /// The table's last column.
    fn pixel_vs_grid_text(&self) -> String {
        let c = self.pixel_vs_grid;
        let mut text = if c.differing == 0 {
            "same bits".to_owned()
        } else {
            format!("{} of {} words, <= {} ulp", c.differing, c.words, c.max_ulp)
        };
        if self.grid_widths_differ > 0 {
            let _ = write!(
                text,
                "; grid widths differ in {} words",
                self.grid_widths_differ
            );
        }
        if !self.written {
            text.push_str("; ELEMENTS NOT WRITTEN");
        }
        if !self.finite {
            text.push_str("; NOT FINITE");
        }
        if !self.accepted {
            text.push_str("; NOT ACCEPTED");
        }
        text
    }
}

fn bench_cell(device: &HeadlessDevice, kern: Kern, side: u32, plan: Plan) -> Result<Cell, String> {
    let b = Buffers::new(device, kern, side);
    // Each variant runs twice into an output poisoned with a different word each time. A word the kernel did not write
    // differs between the two passes; the first pass's output is the one compared.
    let mut outputs: Vec<Vec<u32>> = Vec::new();
    let mut written = true;
    for run in RUNNERS {
        b.poison(device, POISON)?;
        run(device, kern, side, &b, 0, 1)?;
        let first = b.words(device)?;
        b.poison(device, POISON_B)?;
        run(device, kern, side, &b, 0, 1)?;
        written &= first == b.words(device)?;
        outputs.push(first);
    }
    let finite = !kern.is_float()
        || outputs
            .iter()
            .all(|out| out.iter().all(|word| f32::from_bits(*word).is_finite()));
    let mut pixel_vs_grid = Comparison::default();
    let mut grid_widths_differ = 0;
    for grid in &outputs[..3] {
        let found = compare(kern, grid, &outputs[3]);
        pixel_vs_grid.words = found.words;
        pixel_vs_grid.differing = pixel_vs_grid.differing.max(found.differing);
        pixel_vs_grid.max_ulp = pixel_vs_grid.max_ulp.max(found.max_ulp);
        grid_widths_differ = grid_widths_differ.max(compare(kern, &outputs[0], grid).differing);
    }
    let accepted = written
        && finite
        && (kern.is_float() || (pixel_vs_grid.differing == 0 && grid_widths_differ == 0));
    let mut run_medians: [Vec<f64>; 4] = Default::default();
    for round in 0..plan.runs {
        for step in 0..VARIANTS.len() {
            let variant = (step + round) % VARIANTS.len();
            let samples = RUNNERS[variant](device, kern, side, &b, plan.warmup, plan.timed)?;
            if !samples.is_empty() {
                run_medians[variant].push(median(&samples));
            }
        }
    }
    let times = run_medians
        .clone()
        .map(|medians| (!medians.is_empty()).then(|| median(&medians)));
    let pixel = &run_medians[3];
    let pixel_spread = (!pixel.is_empty()).then(|| {
        let (low, high) = pixel
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        (high - low) / median(pixel) * 100.0
    });
    Ok(Cell {
        times,
        pixel_spread,
        written,
        finite,
        pixel_vs_grid,
        grid_widths_differ,
        accepted,
    })
}

/// Why a cell cannot run on this adapter, or `None`.
fn skip_reason(device: &HeadlessDevice, kern: Kern, side: u32) -> Option<String> {
    let n = u64::from(side) * u64::from(side);
    if n > u64::from(PIXEL_MAX_ELEMENTS) {
        return Some(format!(
            "n = {n} is over the pixel cap {PIXEL_MAX_ELEMENTS}"
        ));
    }
    let bytes = n * kern.stride();
    let limits = device.limits();
    if bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size {
        return Some(format!(
            "{bytes} bytes per buffer exceed the adapter's storage binding or buffer limit"
        ));
    }
    None
}

fn ms(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |v| format!("{v:.3}"))
}

// ---- conditions ----

/// `YYYY-MM-DDTHH:MM:SSZ` of the clock (civil-from-days, so no date crate).
fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (days, rest) = ((seconds / 86_400) as i64, seconds % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// What `nvidia-smi` says about the GPU right now (VRAM and utilisation held by every process included), or why it
/// could not be asked.
fn gpu_load() -> String {
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        Ok(output) => format!("nvidia-smi failed ({})", output.status),
        Err(error) => format!("nvidia-smi unavailable ({error})"),
    }
}

// ---- the run ----

struct Args {
    full: bool,
    out: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        full: false,
        out: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--full" => args.full = true,
            "--out" => args.out = Some(iter.next().ok_or("--out needs a path")?),
            other => {
                return Err(format!(
                    "unknown argument {other:?}; usage: shape-bench [--full [--out PATH]]"
                ));
            }
        }
    }
    if args.out.is_some() && !args.full {
        return Err("--out belongs to --full".into());
    }
    Ok(args)
}

/// Checks that the descriptors carry the shapes the modules declared.
fn check_shapes() {
    for (name, shape) in [
        ("grid64", grid64::k1::DESCRIPTOR.shape),
        ("grid128", grid128::k4::DESCRIPTOR.shape),
        ("grid256", grid256::k3l::DESCRIPTOR.shape),
    ] {
        assert_eq!(shape, Shape::Grid, "{name}");
    }
    for shape in [
        px::k1::DESCRIPTOR.shape,
        px::k2::DESCRIPTOR.shape,
        px::k3::DESCRIPTOR.shape,
        px::k3l::DESCRIPTOR.shape,
        px::k4::DESCRIPTOR.shape,
    ] {
        assert_eq!(shape, Shape::Pixel);
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("shape-bench: {message}");
            std::process::exit(2);
        }
    };
    check_shapes();
    let device = HeadlessDevice::with_options(DeviceOptions::default())
        .expect("a Vulkan adapter is required");
    let (plan, sides): (Plan, &[u32]) = if args.full {
        (FULL, &[256, 1024, 2048, 4096, 8192])
    } else {
        (SMOKE, &[256, 1024])
    };
    let info = device.adapter_info();
    let mut text = String::new();
    let mode = if args.full {
        "full".to_owned()
    } else {
        "SMOKE, noisy, no threshold".to_owned()
    };
    let _ = writeln!(text, "shape-bench: {mode}");
    let _ = writeln!(
        text,
        "adapter {} ({:?}, {:?}, driver {} {})",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );
    let _ = writeln!(text, "time {}", utc_now());
    let _ = writeln!(text, "gpu before: {}", gpu_load());
    let _ = writeln!(
        text,
        "statistic: {} run(s) of {} warm-up + {} timed dispatches (GPU timestamps), median of run medians, variant order rotated each run; \
         {} sizes ({}), pixel cap {PIXEL_MAX_ELEMENTS} = 8192^2; times in ms; no threshold is asserted; K3 and K3L are X-S1's polynomial without fma (the dialect has none: an unfused multiply and add); f32 bit differences between the shapes are observed, not asserted",
        plan.runs,
        plan.warmup,
        plan.timed,
        sides.len(),
        sides
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !device.supports_gpu_timestamps() {
        let _ = writeln!(
            text,
            "the adapter has no timestamp queries: every time is n/a"
        );
    }
    let _ = writeln!(text);
    let _ = writeln!(
        text,
        "| kernel | side | grid 64 | grid 128 | grid 256 | best grid | pixel | pixel / best grid | pixel spread | pixel vs grid (bits) |"
    );
    let _ = writeln!(text, "|---|---:|---:|---:|---:|---|---:|---:|---:|---|");
    let mut all_accepted = true;
    let mut same_cells = 0;
    let mut observed_cells = 0;
    let mut rejected_cells = 0;
    let mut largest_ulp = 0;
    let mut cells = 0;
    for kern in KERNELS {
        for &side in sides {
            if let Some(reason) = skip_reason(&device, kern, side) {
                let _ = writeln!(
                    text,
                    "| {} | {side} | skipped ({reason}) | | | | | | | |",
                    kern.name()
                );
                continue;
            }
            let cell = bench_cell(&device, kern, side, plan)
                .unwrap_or_else(|error| panic!("{} at side {side}: {error}", kern.name()));
            cells += 1;
            all_accepted &= cell.accepted;
            same_cells += usize::from(cell.pixel_vs_grid.differing == 0);
            observed_cells += usize::from(cell.pixel_vs_grid.differing > 0);
            rejected_cells += usize::from(!cell.accepted);
            largest_ulp = largest_ulp.max(cell.pixel_vs_grid.max_ulp);
            let best = cell.times[..3]
                .iter()
                .enumerate()
                .filter_map(|(at, time)| time.map(|t| (at, t)))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let widths = ["64", "128", "256"];
            let _ = writeln!(
                text,
                "| {} | {side} | {} | {} | {} | {} | {} | {} | {} | {} |",
                kern.name(),
                ms(cell.times[0]),
                ms(cell.times[1]),
                ms(cell.times[2]),
                best.map_or_else(
                    || "n/a".to_owned(),
                    |(at, t)| format!("{t:.3} (wg {})", widths[at])
                ),
                ms(cell.times[3]),
                match (best, cell.times[3]) {
                    (Some((_, grid)), Some(pixel)) => format!("{:.2}", pixel / grid),
                    _ => "n/a".to_owned(),
                },
                cell.pixel_spread
                    .map_or_else(|| "n/a".to_owned(), |s| format!("{s:.1} %")),
                cell.pixel_vs_grid_text()
            );
        }
    }
    let _ = writeln!(text);
    let _ = writeln!(text, "gpu after: {}", gpu_load());
    let _ = writeln!(
        text,
        "{cells} cells run: {same_cells} with the same bits in the pixel and the grid outputs, {observed_cells} with f32 bit \
         differences between the shapes (an observation, D128; the largest is {largest_ulp} ulp), {rejected_cells} outside what is \
         asserted (an element not written, a non-finite f32, or an integer difference)"
    );
    print!("{text}");
    if let Some(path) = &args.out {
        std::fs::write(path, &text).unwrap_or_else(|error| panic!("{path}: {error}"));
        println!("table written to {path}");
    }
    assert!(
        all_accepted,
        "a cell left an element unwritten, wrote a non-finite f32, or the integer kernel differs between the shapes"
    );
    assert!(cells > 0, "no cell ran");
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::cpu::CpuDevice;

    // Two whole rows of 8192, as every size of the table is: a grid dispatch holds exactly its extent (a shared buffer
    // has `x * y * z` elements), and the tail row of a pixel draw is the business of the gust-wgpu tests.
    const W: u32 = 128;
    const H: u32 = 128;
    const N: usize = (W * H) as usize;

    fn host_hash(x: u32, y: u32) -> u32 {
        let mut h = x
            .wrapping_mul(2_654_435_761)
            .wrapping_add(y.wrapping_mul(2_246_822_507))
            .wrapping_add(668_265_263);
        h ^= h >> 15;
        h = h.wrapping_mul(739_982_445);
        h ^= h >> 12;
        h = h.wrapping_mul(695_872_825);
        h ^ (h >> 15)
    }

    fn host_poly(t: f32, terms: u32) -> f32 {
        let mut acc = 0.0f32;
        for k in 0..terms {
            let c = ((k * 37 + 11) % 101) as f32 * 0.0078125 - 0.375;
            acc = acc * t + c;
        }
        acc
    }

    fn clamp(v: u32, high: u32) -> u32 {
        v.clamp(1, high) - 1
    }

    /// The expected output words of each kernel, replayed on the host.
    fn reference(kern: Kern) -> Vec<u32> {
        let i_range = 0..N as u32;
        match kern {
            Kern::K1 => i_range
                .map(|i| host_hash(i % W, i / W) ^ input_u32((i + 1) % N as u32))
                .collect(),
            Kern::K2 => i_range
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    let mut s = 0.0f32;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            s += input_f32(clamp(y + dy, H) * W + clamp(x + dx, W));
                        }
                    }
                    (s * 0.111_111_11f32).to_bits()
                })
                .collect(),
            Kern::K3 | Kern::K3L => {
                let terms = if kern == Kern::K3 { 64 } else { 1024 };
                i_range
                    .map(|i| {
                        let t = (host_hash(i % W, i / W) & 65535) as f32 * 0.000_015_258_789;
                        host_poly(t, terms).to_bits()
                    })
                    .collect()
            }
            Kern::K4 => i_range
                .flat_map(|i| {
                    let (x, y) = (i % W, i / W);
                    let mut s = [0.0f32; 4];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let v = input_v4(clamp(y + dy, H) * W + clamp(x + dx, W));
                            s = [s[0] + v.x, s[1] + v.y, s[2] + v.z, s[3] + v.w];
                        }
                    }
                    s.map(|c| (c * 0.111_111_11f32).to_bits())
                })
                .collect(),
        }
    }

    /// The same kernel on the CPU twin, as words.
    macro_rules! twin_runner {
        ($name:ident, $m:ident, $dispatch:ident, $first:expr) => {
            fn $name(kern: Kern) -> Vec<u32> {
                let cpu = CpuDevice::new();
                let uniform = cpu
                    .uniform($m::Params {
                        w: W,
                        h: H,
                        pad0: 0,
                        pad1: 0,
                    })
                    .expect("Params is padding-free");
                let first = ($first)(N as u32);
                match kern {
                    Kern::K1 => {
                        let input: Vec<u32> = (0..N as u32).map(input_u32).collect();
                        let (input, out) = (cpu.buffer(&input), cpu.buffer(&vec![POISON; N]));
                        $m::k1::$dispatch(&cpu, first, &uniform, &input, &out).unwrap();
                        out.read()
                    }
                    Kern::K2 => {
                        let input: Vec<f32> = (0..N as u32).map(input_f32).collect();
                        let (input, out) = (
                            cpu.buffer(&input),
                            cpu.buffer(&vec![f32::from_bits(POISON); N]),
                        );
                        $m::k2::$dispatch(&cpu, first, &uniform, &input, &out).unwrap();
                        out.read().iter().map(|v| v.to_bits()).collect()
                    }
                    Kern::K3 | Kern::K3L => {
                        let out = cpu.buffer(&vec![f32::from_bits(POISON); N]);
                        if kern == Kern::K3 {
                            $m::k3::$dispatch(&cpu, first, &uniform, &out).unwrap();
                        } else {
                            $m::k3l::$dispatch(&cpu, first, &uniform, &out).unwrap();
                        }
                        out.read().iter().map(|v| v.to_bits()).collect()
                    }
                    Kern::K4 => {
                        let input: Vec<Vec4> = (0..N as u32).map(input_v4).collect();
                        let poison = Vec4::splat(f32::from_bits(POISON));
                        let (input, out) = (cpu.buffer(&input), cpu.buffer(&vec![poison; N]));
                        $m::k4::$dispatch(&cpu, first, &uniform, &input, &out).unwrap();
                        out.read()
                            .iter()
                            .flat_map(|v| [v.x, v.y, v.z, v.w].map(f32::to_bits))
                            .collect()
                    }
                }
            }
        };
    }

    twin_runner!(twin_grid64, grid64, dispatch_extent, |n: u32| Extent::new(
        PIXEL_ROW_WIDTH,
        n.div_ceil(PIXEL_ROW_WIDTH),
        1
    ));
    twin_runner!(
        twin_grid256,
        grid256,
        dispatch_extent,
        |n: u32| Extent::new(PIXEL_ROW_WIDTH, n.div_ceil(PIXEL_ROW_WIDTH), 1)
    );
    twin_runner!(twin_pixel, px, dispatch, |n: u32| n);

    /// The twins of the grid and the pixel form of every kernel write what the host reference computes, bit for bit
    /// (`check-cpu` runs this, so the bodies are covered without a GPU).
    #[test]
    fn twin_matches_host_reference() {
        for kern in KERNELS {
            let want = reference(kern);
            for (name, got) in [
                ("grid64", twin_grid64(kern)),
                ("grid256", twin_grid256(kern)),
                ("pixel", twin_pixel(kern)),
            ] {
                assert_eq!(got.len(), want.len(), "{} {name}", kern.name());
                let first = got.iter().zip(&want).position(|(a, b)| a != b);
                assert_eq!(first, None, "{} {name}: first differing word", kern.name());
            }
        }
    }

    #[test]
    fn the_modules_carry_the_shapes_they_declare() {
        check_shapes();
        assert_eq!(
            grid_extent(256),
            Extent::new(PIXEL_ROW_WIDTH, 8, 1),
            "256^2 is eight rows of 8192"
        );
        assert_eq!(grid_extent(8192).y, 8192);
        assert_eq!(8192u64 * 8192, u64::from(PIXEL_MAX_ELEMENTS));
    }

    #[test]
    fn the_statistic_is_the_median_of_run_medians() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median(&[4.0, 1.0, 2.0, 3.0]) - 2.5).abs() < 1e-12);
        assert_eq!((FULL.runs, FULL.warmup, FULL.timed), (5, 5, 20));
        const { assert!(SMOKE.runs * SMOKE.timed < FULL.runs * FULL.timed) };
    }

    #[test]
    fn the_clock_prints_a_utc_timestamp() {
        let now = utc_now();
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.starts_with("20") && now.ends_with('Z'), "{now}");
    }
}
