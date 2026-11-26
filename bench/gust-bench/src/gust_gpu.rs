//! GUST on the GPU, through `gust-wgpu`'s public API only: typed buffers, the generated typed `call`, and
//! `dispatch_batch` / `dispatch_batch_timed` on a `HeadlessDevice`.
//!
//! The route is a property of the build: a build with the `spirv-route` feature runs this backend on the SPIR-V route
//! (slangc's SPIR-V through passthrough) and names it `gust-gpu-spirv`; the default build is `gust-gpu` on WGSL through
//! naga. One process holds one GUST route (the feature changes the emitted kernels), so the two are separate builds and
//! separate result files, each with its own `wgsl-prebuilt` to pair against.
//!
//! W7 (sum) is not a kernel of this crate: GUST's sum is `gust_algo::reduce` (`sum`, and `sum_wrapping` for `u32`),
//! called on a resident `GpuVec` and read back to the host, which is the public call a program makes. It needs the
//! `algo` feature (`gust-bench-pre` has none: gust-algo ships no precompiled artifacts), takes its GPU time through
//! `gust_algo::timed` (`gpu_time_sum`), and its `f32` result is judged against `gust-algo`'s own bound for the
//! schedule that ran (`Backend::sum_tree_bound`).
//!
//! W6 (staged pipeline) is six kernels of this crate in two forms: `PipelineHand` builds the six typed calls and runs them as one
//! `dispatch_batch` (six passes, one submit); `PipelineGraph` builds a `StagedGraph` of the six calls (derived edges) and
//! executes it inside every timed sample, so graph construction and validation are part of that sample. Its GPU time is
//! `GraphReport.gpu_elapsed` of `execute_graph_timed` over the dispatch nodes only, and its end-to-end sample puts the
//! uploads and the read-back in the same graph.

use std::{
    cell::{Cell, RefCell},
    time::{Duration, Instant},
};

use gust::batch::{Batch, D1, FrameTiming, Stage};
use gust::{Extent, Independent};
#[cfg(feature = "algo")]
use gust_wgpu::GpuVec;
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuBuffer, GpuBufferAccess, StagedGraph};
use serde_json::Value;

use crate::{
    backend::{Backend, BenchResult, Kind},
    route,
    workloads::{
        self, Inputs, Workload, counts_as_words, hist_threads, histogram, matmul, normalize_counts,
        particles, polynomial, saxpy, signal_pipeline, vector_add,
    },
};

/// The GPU time of `runs` runs when the scope counted `counted` units (`gust_algo` dispatch batches for W7, one per sum; graph
/// dispatch passes for W6-graph, six per run): the harness divides by the run count, so a different count would make the
/// per-run time wrong (a silent bias), and is an error instead.
fn per_run(
    counted: usize,
    expected: usize,
    elapsed: Option<Duration>,
) -> BenchResult<Option<Duration>> {
    if counted != expected {
        return Err(format!(
            "the timed span counted {counted} units where {expected} were expected: the GPU time per run would be wrong"
        )
        .into());
    }
    Ok(elapsed)
}

/// The buffers of a prepared workload, in the order of the kernel's parameters.
enum Data {
    Add {
        x: GpuBuffer<f32>,
        y: GpuBuffer<f32>,
        out: GpuBuffer<f32>,
    },
    Saxpy {
        scale: GpuBuffer<f32>,
        x: GpuBuffer<f32>,
        y: GpuBuffer<f32>,
        out: GpuBuffer<f32>,
    },
    Poly {
        coeffs: GpuBuffer<f32>,
        x: GpuBuffer<f32>,
        out: GpuBuffer<f32>,
    },
    Hist {
        keys: GpuBuffer<u32>,
        bins: GpuBuffer<u32>,
    },
    MatMul {
        dims: GpuBuffer<u32>,
        a: GpuBuffer<f32>,
        b: GpuBuffer<f32>,
        c: GpuBuffer<f32>,
    },
    /// W5: the typed particle and acceleration buffers and the output; the typed host copies an end-to-end sample uploads
    /// and the typed poison are built once in `prepare`, so no sample converts the harness's words on the way in.
    Particles {
        input: GpuBuffer<particles::kernel::Particle>,
        acceleration: GpuBuffer<particles::kernel::Pair>,
        out: GpuBuffer<particles::kernel::Particle>,
        host_input: Vec<particles::kernel::Particle>,
        host_acceleration: Vec<particles::kernel::Pair>,
        poison: Vec<particles::kernel::Particle>,
    },
    /// W6: the five resident input streams (`signal`, `baseline`, `gain`, `bias`, `scale`) and the six stage outputs
    /// (`centered`, `amplified`, `biased`, `normalized`, `energy`, `out`). The stage outputs are read-write buffers (a
    /// read-write buffer binds to a read-only parameter), so the graph form can read any of them back.
    Pipeline {
        inputs: Vec<GpuBuffer<f32>>,
        stages: Vec<GpuBuffer<f32>>,
    },
    /// W7 on `gust-algo`: the resident input and the word the last sum returned (the output of this backend, which has
    /// no device-side output buffer to read: the public sum returns its value to the host).
    #[cfg(feature = "algo")]
    SumF32(GpuVec<f32>),
    #[cfg(feature = "algo")]
    SumU32(GpuVec<u32>),
}

struct Resident {
    workload: Workload,
    n: usize,
    data: Data,
    /// The poison words of the f32 output (NaN), or empty for a histogram, whose poison is zeroed bins.
    nan: Vec<f32>,
    zeros: Vec<u32>,
    /// W7: the word the last sum returned, the poison word before the first and after every `poison`.
    #[cfg(feature = "algo")]
    sum_word: Cell<f32>,
    /// W7 (`f32`): the one-word result of the last `dispatch` (`reduce::sum_on_device`), which `read_back` reads; `None`
    /// after a `poison` or an `end_to_end` (whose word is in `sum_word`).
    #[cfg(feature = "algo")]
    sum_dev: RefCell<Option<GpuVec<f32>>>,
}

/// The events a cell produced: those after the `before` that were on the device when its `prepare` began.
#[cfg(any(test, feature = "spirv-route"))]
fn own_events<'a>(events: &'a [&'static str], before: usize) -> &'a [&'static str] {
    &events[before.min(events.len())..]
}

/// The route check of a cell. Route events arise only when a pipeline is compiled, and the pipeline cache is keyed by
/// kernel, so a cell that compiles nothing (for example every cell after a workload's first size) has no events of its own: it runs the pipeline an
/// earlier cell built. The cell therefore fails on any event of its own that is off the build's route; otherwise the
/// kernel's latest event over the device's life (`latest`) must exist and be on the route, which for such a cell is the
/// build of the cached pipeline it ran.
#[cfg(any(test, feature = "spirv-route"))]
fn route_verdict(
    kernel: &str,
    want: &str,
    own: &[&str],
    latest: Option<&str>,
) -> Result<(), String> {
    if let Some(other) = own.iter().find(|route| **route != want) {
        return Err(format!(
            "kernel {kernel} ran on the `{other}` route, but this build's route is `{want}` ({} events of this cell)",
            own.len()
        ));
    }
    match latest {
        None => Err(format!(
            "no route event for {kernel}: the build route `{want}` could not be confirmed"
        )),
        Some(other) if other != want => Err(format!(
            "kernel {kernel} last compiled on the `{other}` route, but this build's route is `{want}`"
        )),
        Some(_) => Ok(()),
    }
}

/// Which form of W6's chain a GUST row runs (slice pf.onepass; probe rows, never publishable). The base `gust-gpu` row is
/// [`Variant::Plain`]; the others are `--with` rows that exist only for W6-hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The base row: every workload, the batch recorder's default (W6: six dispatches in one compute pass).
    Plain,
    /// P': a second copy of the base path on W6-hand, so that the default form can be created at any position of the
    /// rotation (the base row is always created first).
    Default,
    /// U': the six dispatches of the chain recorded one compute pass each, the form batches had before one pass
    /// (`gust_wgpu::force_pass_per_dispatch`).
    PerPass,
    /// H: the hand-fused kernel, one dispatch that writes only the final output.
    Fused,
    /// F': the six stages as one `gust::batch` batch (built once, one frame per chain) recorded in the default form,
    /// the frame path's merged pass.
    Frame,
    /// FU': the same batch with the frame recorded one compute pass per stage (`force_pass_per_dispatch`).
    FramePerPass,
}

impl Variant {
    /// The `--with` name of a variant row.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "gust-default" => Some(Self::Default),
            "gust-perpass" => Some(Self::PerPass),
            "gust-fused" => Some(Self::Fused),
            "gust-frame" => Some(Self::Frame),
            "gust-frame-perpass" => Some(Self::FramePerPass),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Plain => backend_id(),
            Self::Default => "gust-gpu+default",
            Self::PerPass => "gust-gpu+perpass",
            Self::Fused => "gust-gpu+fused",
            Self::Frame => "gust-gpu+frame",
            Self::FramePerPass => "gust-gpu+frame-perpass",
        }
    }

    /// Whether the row records every batch as one pass per dispatch.
    pub fn per_pass(self) -> bool {
        matches!(self, Self::PerPass | Self::FramePerPass)
    }

    /// Whether the row runs the six stages as a `gust::batch` batch (the frame path) and not as `dispatch_batch` calls.
    pub fn is_frame(self) -> bool {
        matches!(self, Self::Frame | Self::FramePerPass)
    }

    pub fn supports(self, workload: Workload) -> bool {
        self == Self::Plain || workload == Workload::PipelineHand
    }
}

/// The batch of the frame rows: the five input streams and the final stage of W6 as handles of one `gust::batch` batch.
struct FrameRig {
    batch: Batch<gust_wgpu::HeadlessDevice>,
    inputs: [Stage<f32, D1>; 5],
    out: Stage<f32, D1>,
}

pub struct GustGpu {
    device: DeviceHandle,
    variant: Variant,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates; see `Backend::dispatch`).
    passes: Cell<u32>,
    route: RefCell<Option<Value>>,
    /// The batch of the frame rows (F', FU'), built by `prepare` for W6-hand.
    frame: RefCell<Option<FrameRig>>,
    /// The final stage's words of the last frame of the last `gpu_time` call (the frame rows only), which `read_back`
    /// returns so the harness judges a timed frame's own output. Cleared by `poison`.
    last_frame: RefCell<Option<Vec<f32>>>,
    /// Route events of the cell's kernel already on the device when `prepare` began: the events are cumulative over the
    /// device's life (the W3 cells share `kernel::horner`), so a cell's own share is the total minus this.
    #[cfg(feature = "spirv-route")]
    events_before_prepare: RefCell<Vec<usize>>,
}

/// Whether the public sum runs its v1 schedule on a device with these limits: a replica of `gust_algo::limits::admits`
/// (private), from the public `device.limits()`: workgroup memory for the v1 kernels (`3 * 256 * 4` bytes), the input
/// (`n * 4` bytes) inside one storage binding, and the first pass's workgroup count inside
/// `max_compute_workgroups_per_dimension`. The replication is an assumption of the bench (the library does not say which
/// schedule it ran); where it is wrong the bound it picks is still the looser one, see [`sum_bound_from`].
#[cfg(feature = "algo")]
pub(crate) fn v1_admitted(
    n: u32,
    workgroup_storage: u32,
    binding: u64,
    per_dimension: u32,
) -> bool {
    use gust_algo::schedule::BlockSchedule;
    BlockSchedule::new(n).is_some_and(|schedule| {
        let first = schedule.passes().first().map_or(1, |pass| pass.workgroups);
        workgroup_storage >= 3 * 256 * 4
            && u64::from(n) * 4 <= binding
            && u64::from(first) <= u64::from(per_dimension)
    })
}

/// The bound of `gust-algo`'s `f32` sum of `n` values with `sum |x_i| <= abs_sum`: `f32_sum_bound_v1` where the shipped
/// variant is v1 and the launch is `admitted` ([`v1_admitted`]), else the looser of the v0 and v1 bounds (the library's own
/// advice, `bound.rs`: v0 runs, or the call errors). `c` is `C_GPU = 2^-23`, the library's GPU constant.
#[cfg(feature = "algo")]
pub(crate) fn sum_bound_from(n: u32, abs_sum: f64, admitted: bool) -> f64 {
    use gust_algo::{
        bound::{C_GPU, f32_sum_bound, f32_sum_bound_v1},
        variant::{Operation, Variant, shipped},
    };
    let v1 = f32_sum_bound_v1(n, C_GPU, abs_sum);
    match shipped(Operation::Sum) {
        Variant::V1 if admitted => v1,
        _ => f32_sum_bound(n, C_GPU, abs_sum).max(v1),
    }
}

/// The id of this backend in the build: the route is part of it.
pub const fn backend_id() -> &'static str {
    if gust_wgpu::SPIRV_ROUTE {
        "gust-gpu-spirv"
    } else {
        "gust-gpu"
    }
}

impl GustGpu {
    pub fn new(device: DeviceHandle) -> Self {
        Self::with_variant(device, Variant::Plain)
    }

    pub fn with_variant(device: DeviceHandle, variant: Variant) -> Self {
        Self {
            device,
            variant,
            resident: None,
            passes: Cell::new(0),
            route: RefCell::new(None),
            frame: RefCell::new(None),
            last_frame: RefCell::new(None),
            #[cfg(feature = "spirv-route")]
            events_before_prepare: RefCell::new(Vec::new()),
        }
    }

    /// One frame of the frame rows' batch (one chain of the six stages). `inputs` are written first, `read` takes a
    /// ticket of the final stage and returns its words, `timed` returns the frame's time at level `Frame`
    /// (`FrameTimes::span`, the first pass's begin to the last pass's end; `None` where the adapter has no timestamps).
    /// The call returns when the submission is complete, so the host time of a call is submit to completion.
    fn frame_chain(
        &self,
        inputs: Option<&Inputs>,
        read: bool,
        timed: bool,
    ) -> BenchResult<(Vec<f32>, Option<Duration>)> {
        self.arm_pass_form();
        let mut guard = self.frame.borrow_mut();
        let rig = guard.as_mut().expect("prepare built the frame batch");
        let mut frame = pollster::block_on(rig.batch.begin()).map_err(|e| e.to_string())?;
        if let Some(inputs) = inputs {
            for (stage, values) in rig.inputs.iter().zip(&inputs.pipeline) {
                frame.write(stage, values).map_err(|e| e.to_string())?;
            }
        }
        let ticket = if read {
            Some(frame.ticket(&rig.out).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let time = timed.then(|| frame.time());
        let submitted = frame.submit().map_err(|e| e.to_string())?;
        submitted.wait().map_err(|e| e.to_string())?;
        let words = match ticket {
            Some(ticket) => pollster::block_on(ticket).map_err(|e| e.to_string())?,
            None => Vec::new(),
        };
        let span = match time {
            Some(time) => pollster::block_on(time)
                .map_err(|e| e.to_string())?
                .map(|times| times.span),
            None => None,
        };
        Ok((words, span))
    }

    /// Builds the batch of the frame rows over `n` elements: the six stages of W6 as `gpu_task!` stages of one batch at
    /// frame timing level `Frame`, one frame in flight.
    fn build_frame_rig(&self, n: usize) -> BenchResult<FrameRig> {
        let n = u32::try_from(n).map_err(|_| "the frame rows' size does not fit u32")?;
        let device = &self.device;
        let (batch, (inputs, out)) =
            gust::gpu_batch!("bench w6 frame", on(device); in_flight(1); |b| {
                b.frame_timing(FrameTiming::Frame);
                let signal = b.input::<f32>(n);
                let baseline = b.input::<f32>(n);
                let gain = b.input::<f32>(n);
                let bias = b.input::<f32>(n);
                let scale = b.input::<f32>(n);
                let centered = gust::gpu_task!(n, |i| -> f32 { signal[i] - baseline[i] });
                let amplified = gust::gpu_task!(n, |i| -> f32 { centered[i] * gain[i] });
                let biased = gust::gpu_task!(n, |i| -> f32 { amplified[i] + bias[i] });
                let normalized = gust::gpu_task!(n, |i| -> f32 { biased[i] / scale[i] });
                let squared = gust::gpu_task!(n, |i| -> f32 { normalized[i] * normalized[i] });
                let out = gust::gpu_task!(n, |i| -> f32 { squared[i] + 0.25f32 * normalized[i] });
                ([signal, baseline, gain, bias, scale], out)
            })
            .map_err(|e| e.to_string())?;
        Ok(FrameRig { batch, inputs, out })
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    /// The resident input of a W7 workload, uploaded through the public `GpuVec`.
    #[cfg(feature = "algo")]
    fn sum_data(d: &DeviceHandle, workload: Workload, inputs: &Inputs) -> BenchResult<Data> {
        Ok(if workload == Workload::SumU32 {
            Data::SumU32(GpuVec::from_slice_on(d, &inputs.values)?)
        } else {
            Data::SumF32(GpuVec::from_slice_on(d, &inputs.x)?)
        })
    }

    #[cfg(not(feature = "algo"))]
    fn sum_data(_: &DeviceHandle, _: Workload, _: &Inputs) -> BenchResult<Data> {
        Err("W7 needs the `algo` feature (gust-algo)".into())
    }

    /// See `Backend::sum_tree_bound`.
    #[cfg(feature = "algo")]
    fn sum_bound(&self, n: usize, abs_sum: f64) -> Option<f64> {
        if self.resident().workload != Workload::SumF32 {
            return None;
        }
        let n = u32::try_from(n).ok()?;
        let limits = self.device.limits();
        let admitted = v1_admitted(
            n,
            limits.max_compute_workgroup_storage_size,
            limits.max_storage_buffer_binding_size,
            limits.max_compute_workgroups_per_dimension,
        );
        Some(sum_bound_from(n, abs_sum, admitted))
    }

    #[cfg(not(feature = "algo"))]
    fn sum_bound(&self, _n: usize, _abs_sum: f64) -> Option<f64> {
        None
    }

    /// Builds the generated typed call for the resident buffers and hands the resulting dispatch to `run`. The call
    /// is the same value a user of `kernel::dispatch` builds on every invocation.
    fn with_dispatch<R>(&self, run: impl FnOnce(BufferDispatch<'_>) -> R) -> R {
        let r = self.resident();
        let n = r.n as u32;
        match &r.data {
            Data::Add { x, y, out } => {
                let call = vector_add::kernel::add::call(n, x, y, out);
                run(BufferDispatch::from(&call))
            }
            Data::Saxpy { scale, x, y, out } => {
                let call = saxpy::kernel::saxpy::call(n, Independent(scale), x, y, out);
                run(BufferDispatch::from(&call))
            }
            Data::Poly { coeffs, x, out } => {
                let call = polynomial::kernel::horner::call(n, Independent(coeffs), x, out);
                run(BufferDispatch::from(&call))
            }
            Data::Hist { keys, bins } => {
                let threads = hist_threads(r.n) as u32;
                let call = histogram::kernel::histogram::call(
                    threads,
                    Independent(keys),
                    Independent(bins),
                );
                run(BufferDispatch::from(&call))
            }
            Data::MatMul { dims, a, b, c } => {
                let call = matmul::kernel::matmul::call_extent(
                    Extent::new(n, n, 1),
                    Independent(dims),
                    Independent(a),
                    Independent(b),
                    Independent(c),
                );
                run(BufferDispatch::from(&call))
            }
            Data::Particles {
                input,
                acceleration,
                out,
                ..
            } => {
                let call = particles::kernel::step::call(n, input, acceleration, out);
                run(BufferDispatch::from(&call))
            }
            Data::Pipeline { .. } => {
                unreachable!("W6 has six calls: see `with_pipeline`")
            }
            #[cfg(feature = "algo")]
            Data::SumF32(_) | Data::SumU32(_) => {
                unreachable!("W7 runs gust-algo's sum, not a kernel call of this crate")
            }
        }
    }

    /// W6: builds the six typed calls over the resident buffers, in stage order, and hands them to `run` as the call
    /// value and the dispatch of each. The `Call` values are the ones a user of `kernel::dispatch` builds.
    fn with_pipeline<R>(&self, run: impl FnOnce(Vec<BufferDispatch<'_>>) -> R) -> R {
        use signal_pipeline::kernel;
        let r = self.resident();
        let n = r.n as u32;
        let Data::Pipeline {
            inputs: i,
            stages: s,
        } = &r.data
        else {
            unreachable!("with_pipeline is for W6")
        };
        if self.variant == Variant::Fused {
            let fused = signal_pipeline::fused::transform::call(
                n,
                &i[0],
                &i[1],
                &i[2],
                &i[3],
                &i[4],
                &s[signal_pipeline::STAGES - 1],
            );
            return run(vec![BufferDispatch::from(&fused)]);
        }
        let c0 = kernel::center::call(n, &i[0], &i[1], &s[0]);
        let c1 = kernel::amplify::call(n, &s[0], &i[2], &s[1]);
        let c2 = kernel::shift::call(n, &s[1], &i[3], &s[2]);
        let c3 = kernel::rescale::call(n, &s[2], &i[4], &s[3]);
        let c4 = kernel::energy::call(n, &s[3], &s[4]);
        let c5 = kernel::combine::call(n, &s[4], &s[3], &s[5]);
        run(vec![
            BufferDispatch::from(&c0),
            BufferDispatch::from(&c1),
            BufferDispatch::from(&c2),
            BufferDispatch::from(&c3),
            BufferDispatch::from(&c4),
            BufferDispatch::from(&c5),
        ])
    }

    /// W6, the `StagedGraph` form: builds the graph of the six calls (derived edges), validates and executes it, all inside
    /// the call. With `inputs` the graph also has an upload node per input stream and a readback of the output, and the
    /// output words come back (the end-to-end sample); without, it is the six dispatches only (the timed `dispatch`).
    fn run_graph(&self, inputs: Option<&Inputs>) -> BenchResult<Option<Vec<f32>>> {
        Ok(self.run_graph_with(inputs, 1, false)?.0)
    }

    /// W6, the `StagedGraph` form with `repeats` copies of the six dispatches in one graph (so one `execute_graph_timed`
    /// span can bracket `6 x repeats` passes, as W6-hand's batch does). With `timed` the graph runs through
    /// `execute_graph_timed` and the second value is `GraphReport.gpu_elapsed` (first dispatch pass begin to last pass end;
    /// uploads, the read-back mapping and the host work of building and validating the graph are outside it); untimed it
    /// is `None`. The third value is `GraphReport.dispatches`, which a timed batch checks against `6 x repeats`.
    fn run_graph_with(
        &self,
        inputs: Option<&Inputs>,
        repeats: usize,
        timed: bool,
    ) -> BenchResult<(Option<Vec<f32>>, Option<Duration>, usize)> {
        use signal_pipeline::kernel;
        let r = self.resident();
        let n = r.n as u32;
        let Data::Pipeline {
            inputs: i,
            stages: s,
        } = &r.data
        else {
            unreachable!("run_graph is for W6")
        };
        let c0 = kernel::center::call(n, &i[0], &i[1], &s[0]);
        let c1 = kernel::amplify::call(n, &s[0], &i[2], &s[1]);
        let c2 = kernel::shift::call(n, &s[1], &i[3], &s[2]);
        let c3 = kernel::rescale::call(n, &s[2], &i[4], &s[3]);
        let c4 = kernel::energy::call(n, &s[3], &s[4]);
        let c5 = kernel::combine::call(n, &s[4], &s[3], &s[5]);
        let mut graph = StagedGraph::new();
        if let Some(host) = inputs {
            for (buffer, values) in i.iter().zip(&host.pipeline) {
                graph.write(buffer, values.as_slice());
            }
        }
        for _ in 0..repeats {
            graph.run(&c0);
            graph.run(&c1);
            graph.run(&c2);
            graph.run(&c3);
            graph.run(&c4);
            graph.run(&c5);
        }
        let read = inputs.map(|_| graph.read(&s[5]));
        let output = if timed {
            self.device.execute_graph_timed(&graph)?
        } else {
            self.device.execute_graph(&graph)?
        };
        let words = match read {
            Some(node) => Some(output.readback::<f32>(node)?),
            None => None,
        };
        Ok((words, output.report.gpu_elapsed, output.report.dispatches))
    }

    /// W7 GPU time: `batch` public sums inside one `gust_algo::timed` scope. `f32` makes the call the host-timed
    /// `dispatch` makes (`sum_on_device`, the word stays on the device), `u32` the public `sum_wrapping`; the last
    /// result is left readable for the harness's `read_back`. One `sum` of a non-empty input is one `dispatch_batch`, so
    /// `GpuTime.batches` must equal `batch`; a different count would make `gpu_elapsed / batch` wrong, so it is an error.
    /// `gpu_elapsed` is the sum over the batches of first-pass begin to last-pass end: the zero-fill of the pooled
    /// partials, the gaps between batches and the read-back are outside it.
    #[cfg(feature = "algo")]
    fn gpu_time_sum(&self, batch: usize) -> BenchResult<Option<Duration>> {
        let timed = self.timed_sums(batch)?;
        per_run(timed.batches, batch, timed.gpu_elapsed)
    }

    /// `batch` public sums of the resident data in one `timed` scope: the scope's `GpuTime` (tests read `batches`).
    #[cfg(feature = "algo")]
    fn timed_sums(&self, batch: usize) -> BenchResult<gust_algo::GpuTime> {
        let r = self.resident();
        let timed = gust_algo::timed(|| {
            match &r.data {
                Data::SumF32(values) => {
                    let mut last = None;
                    for _ in 0..batch {
                        last = Some(gust_algo::reduce::sum_on_device(values)?);
                    }
                    r.sum_dev.replace(last);
                }
                Data::SumU32(values) => {
                    let mut word = 0;
                    for _ in 0..batch {
                        word = gust_algo::reduce::sum_wrapping(values)?;
                    }
                    r.sum_word.set(f32::from_bits(word));
                    r.sum_dev.replace(None);
                }
                _ => unreachable!("timed_sums is for W7"),
            }
            Ok(())
        })?;
        Ok(timed.time)
    }

    /// Sets the process-wide recording form of `gust-wgpu` for the call that follows: a pass per dispatch for the
    /// `perpass` row, the default (one compute pass) for every other row (rows are sampled interleaved, so a row's
    /// setting must not leak into the next row's call).
    fn arm_pass_form(&self) {
        gust_wgpu::force_pass_per_dispatch(self.variant.per_pass());
    }

    /// One complete run of the workload on the resident data: a kernel dispatch, or for W7 the public sum (which waits and
    /// reads its value back, so its word is the result). Used by `end_to_end` and the W7 warm-up; the timed `dispatch`
    /// of W7 `f32` is [`Self::dispatch_sum_f32`] (no read-back inside the timed region).
    fn run_once(&self) -> BenchResult<()> {
        self.arm_pass_form();
        #[cfg(feature = "algo")]
        match &self.resident().data {
            Data::SumF32(values) => {
                let sum = gust_algo::reduce::sum(values)?;
                self.resident().sum_word.set(sum);
                self.resident().sum_dev.replace(None);
                return Ok(());
            }
            Data::SumU32(values) => {
                let sum = gust_algo::reduce::sum_wrapping(values)?;
                self.resident().sum_word.set(f32::from_bits(sum));
                return Ok(());
            }
            _ => {}
        }
        if self.resident().workload.is_pipeline() {
            if self.variant.is_frame() {
                self.frame_chain(None, false, false)?;
            } else if self.resident().workload.is_graph_form() {
                self.run_graph(None)?;
            } else {
                self.with_pipeline(|dispatches| self.device.dispatch_batch(&dispatches))?;
            }
            return Ok(());
        }
        self.with_dispatch(|dispatch| self.device.dispatch_batch(&[dispatch]))?;
        Ok(())
    }

    /// W7 `f32`, the resident data: one `reduce::sum_on_device` call. It waits for the dispatches, and leaves the one-word
    /// result on the device (the same shape as the hand dispatch: no read-back, no host word in the timed region); it
    /// still draws and zeroes its pooled partials per call, which the hand path does not (stated in BENCHMARKS.md).
    /// `read_back` then reads the word. `u32` has no on-device form, so its `dispatch` is `run_once` (with the read-back).
    #[cfg(feature = "algo")]
    fn dispatch_sum_f32(&self, values: &GpuVec<f32>) -> BenchResult<()> {
        let out = gust_algo::reduce::sum_on_device(values)?;
        self.resident().sum_dev.replace(Some(out));
        Ok(())
    }

    fn read_words(&self) -> BenchResult<Vec<f32>> {
        let d = &self.device;
        Ok(match &self.resident().data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => {
                d.read_typed_buffer(out)?
            }
            Data::Hist { bins, .. } => counts_as_words(d.read_typed_buffer(bins)?),
            Data::MatMul { c, .. } => d.read_typed_buffer(c)?,
            // The typed read-back as the harness's words: the conversion is inside an end-to-end sample (see BENCHMARKS.md).
            Data::Particles { out, .. } => particles::from_particles(&d.read_typed_buffer(out)?),
            Data::Pipeline { stages, .. } => {
                d.read_typed_buffer(&stages[signal_pipeline::STAGES - 1])?
            }
            #[cfg(feature = "algo")]
            Data::SumF32(_) | Data::SumU32(_) => {
                let device_word = self.resident().sum_dev.borrow();
                match device_word.as_ref() {
                    Some(word) => word.to_vec()?,
                    None => vec![self.resident().sum_word.get()],
                }
            }
        })
    }

    /// The last route event of the workload's kernel, in a build that keeps events.
    #[cfg(feature = "spirv-route")]
    fn last_event(&self, kernel: &str) -> Option<route::Event> {
        self.device
            .route_events()
            .into_iter()
            .rev()
            .find(|event| event.kernel == kernel)
            .map(|event| route::Event {
                route: match event.route {
                    gust_wgpu::Route::Spirv => "spirv",
                    gust_wgpu::Route::Wgsl => "wgsl",
                    _ => "other",
                },
                reason: event.reason,
            })
    }

    /// The route of every event of the workload's kernel, in a build that keeps events.
    #[cfg(feature = "spirv-route")]
    fn route_events_of(&self, kernel: &str) -> Vec<&'static str> {
        self.device
            .route_events()
            .into_iter()
            .filter(|event| event.kernel == kernel)
            .map(|event| match event.route {
                gust_wgpu::Route::Spirv => "spirv",
                gust_wgpu::Route::Wgsl => "wgsl",
                _ => "other",
            })
            .collect()
    }

    #[cfg(not(feature = "spirv-route"))]
    fn last_event(&self, _kernel: &str) -> Option<route::Event> {
        None
    }
}

impl Backend for GustGpu {
    fn id(&self) -> &'static str {
        self.variant.id()
    }

    fn description(&self) -> &'static str {
        match self.variant {
            Variant::Plain => {}
            Variant::Default => {
                return "W6-hand on GUST as the base row runs it (the six dispatches in one compute pass): the pf.onepass probe row P', created at a rotating position";
            }
            Variant::PerPass => {
                return "W6-hand on GUST with the six dispatches recorded one compute pass each, the form before one pass: the pf.onepass probe row U'";
            }
            Variant::Fused => {
                return "W6-hand as one hand-fused GUST kernel (`transform`, one dispatch): the pf.onepass probe row H";
            }
            Variant::Frame => {
                return "W6-hand as one `gust::batch` batch (six stages, built once, one frame per chain) recorded in the default form: the pf.onepass probe row F'";
            }
            Variant::FramePerPass => {
                return "W6-hand as one `gust::batch` batch with the frame recorded one compute pass per stage: the pf.onepass probe row FU'";
            }
        }
        if gust_wgpu::SPIRV_ROUTE {
            "GUST kernel through gust-wgpu on the opt-in SPIR-V route (slangc SPIR-V through passthrough), typed call, public API only"
        } else {
            "GUST kernel through gust-wgpu (Slang -> WGSL -> wgpu), typed call, public API only"
        }
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    /// W7 needs `gust-algo` (the `algo` feature); every other workload is a kernel of this crate.
    fn supports(&self, workload: Workload) -> bool {
        self.variant.supports(workload) && (workload.family() != "W7" || cfg!(feature = "algo"))
    }

    /// The bound of `gust-algo`'s `f32` sum for the schedule that runs: `f32_sum_bound_v1` where the shipped variant is v1
    /// and the launch is admitted, else the looser of the two bounds. The public API does not say which ran: `limits::admits`
    /// is private, so its test is repeated here from the public `device.limits()` (workgroup memory for the v1 kernels,
    /// the storage binding for the input, the workgroup count of the first pass), an assumption; where it fails, v0 runs
    /// or the call errors, and the larger of the two bounds covers v0. `c` is `C_GPU`, the library's GPU constant.
    fn sum_tree_bound(&self, n: usize, abs_sum: f64) -> Option<f64> {
        self.sum_bound(n, abs_sum)
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        *self.frame.borrow_mut() = None;
        #[cfg(feature = "spirv-route")]
        {
            let before = workloads::descriptors(workload)
                .iter()
                .map(|descriptor| {
                    self.route_events_of(&descriptor.qualified_name().to_string())
                        .len()
                })
                .collect();
            *self.events_before_prepare.borrow_mut() = before;
        }
        let d = &self.device;
        let n = inputs.n;
        let read_only = GpuBufferAccess::ReadOnly;
        let read_write = GpuBufferAccess::ReadWrite;
        let (data, nan, zeros) = match workload.family() {
            "W1" | "W2" => {
                let nan = vec![f32::NAN; n];
                let x = d.create_typed_buffer("bench x", &inputs.x, read_only);
                let y = d.create_typed_buffer("bench y", &inputs.y, read_only);
                let out = d.create_typed_buffer("bench out", &nan, read_write);
                // Flush the initial uploads so no later timed submission carries them.
                d.write_typed_buffer_timed(&x, &inputs.x)?;
                d.write_typed_buffer_timed(&y, &inputs.y)?;
                let data = if workload == Workload::VectorAdd {
                    Data::Add { x, y, out }
                } else {
                    let scale = d.create_typed_buffer("bench scale", &[inputs.scale], read_only);
                    Data::Saxpy { scale, x, y, out }
                };
                (data, nan, Vec::new())
            }
            "W3" => {
                let nan = vec![f32::NAN; n];
                let coeffs = d.create_typed_buffer("bench coeffs", &inputs.coeffs, read_only);
                let x = d.create_typed_buffer("bench x", &inputs.x, read_only);
                let out = d.create_typed_buffer("bench out", &nan, read_write);
                d.write_typed_buffer_timed(&x, &inputs.x)?;
                (Data::Poly { coeffs, x, out }, nan, Vec::new())
            }
            "W5" => {
                let nan = vec![f32::NAN; n * particles::WORDS];
                let host_input = particles::to_particles(&inputs.x);
                let host_acceleration = particles::to_pairs(&inputs.y);
                let poison = particles::to_particles(&nan);
                let input = d.create_typed_buffer("bench particles", &host_input, read_only);
                let acceleration =
                    d.create_typed_buffer("bench acceleration", &host_acceleration, read_only);
                let out = d.create_typed_buffer("bench out", &poison, read_write);
                d.write_typed_buffer_timed(&input, &host_input)?;
                d.write_typed_buffer_timed(&acceleration, &host_acceleration)?;
                let data = Data::Particles {
                    input,
                    acceleration,
                    out,
                    host_input,
                    host_acceleration,
                    poison,
                };
                (data, nan, Vec::new())
            }
            "W6" => {
                let nan = vec![f32::NAN; n];
                let names = ["signal", "baseline", "gain", "bias", "scale"];
                let mut buffers = Vec::new();
                for (name, values) in names.iter().zip(&inputs.pipeline) {
                    let buffer = d.create_typed_buffer(name, values, read_only);
                    d.write_typed_buffer_timed(&buffer, values)?;
                    buffers.push(buffer);
                }
                let stages = [
                    "centered",
                    "amplified",
                    "biased",
                    "normalized",
                    "energy",
                    "out",
                ]
                .iter()
                .map(|name| d.create_typed_buffer(name, &nan, read_write))
                .collect();
                (
                    Data::Pipeline {
                        inputs: buffers,
                        stages,
                    },
                    nan,
                    Vec::new(),
                )
            }
            "W7" => (Self::sum_data(d, workload, inputs)?, Vec::new(), Vec::new()),
            "W9" => {
                let nan = vec![f32::NAN; n * n];
                let dims = d.create_typed_buffer("bench dims", &[n as u32], read_only);
                let a = d.create_typed_buffer("bench a", &inputs.x, read_only);
                let b = d.create_typed_buffer("bench b", &inputs.y, read_only);
                let c = d.create_typed_buffer("bench c", &nan, read_write);
                d.write_typed_buffer_timed(&a, &inputs.x)?;
                d.write_typed_buffer_timed(&b, &inputs.y)?;
                (Data::MatMul { dims, a, b, c }, nan, Vec::new())
            }
            _ => {
                let zeros = vec![0u32; inputs.bins];
                let keys = d.create_typed_buffer("bench keys", &inputs.keys, read_only);
                let bins = d.create_typed_buffer("bench bins", &zeros, read_write);
                d.write_typed_buffer_timed(&keys, &inputs.keys)?;
                (Data::Hist { keys, bins }, Vec::new(), zeros)
            }
        };
        self.passes.set(0);
        self.resident = Some(Resident {
            workload,
            n,
            data,
            nan,
            zeros,
            #[cfg(feature = "algo")]
            sum_word: Cell::new(workload.poison_word()),
            #[cfg(feature = "algo")]
            sum_dev: RefCell::new(None),
        });
        // W7: the first sum builds the pipelines and flushes the upload of the input, so no timed call carries them.
        if workload.family() == "W7" {
            self.run_once()?;
        }
        // The fused row's `transform` pipeline is the one kernel the other GUST rows do not build. The cache and its miss
        // counter belong to the device, and the harness snapshots each row's counter after that row's own warm-up, so a
        // compile at the fused row's warm-up would move the counter of the rows warmed up before it (pf.fuse smoke, 6 to
        // 7). Every row is prepared before any row warms up, so the compile is done here.
        if self.variant == Variant::Fused {
            self.run_once()?;
        }
        // The frame rows build their batch (six task kernels, compiled here for the same reason) and run one frame that
        // uploads the five inputs, waits for it, and takes the frame's time once so the timing objects exist.
        if self.variant.is_frame() {
            let rig = self.build_frame_rig(n)?;
            *self.frame.borrow_mut() = Some(rig);
            self.frame_chain(Some(inputs), true, true)?;
        }
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        *self.last_frame.borrow_mut() = None;
        let r = self.resident();
        // The timed variants wait for the upload, so the copy cannot land inside a timed submission.
        match &r.data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => {
                self.device.write_typed_buffer_timed(out, &r.nan)?;
            }
            Data::Hist { bins, .. } => {
                self.device.write_typed_buffer_timed(bins, &r.zeros)?;
            }
            Data::MatMul { c, .. } => {
                self.device.write_typed_buffer_timed(c, &r.nan)?;
            }
            Data::Particles { out, poison, .. } => {
                self.device.write_typed_buffer_timed(out, poison)?;
            }
            // Every stage output: a stage that did not run leaves its NaN, and the final one never keeps a stale value.
            Data::Pipeline { stages, .. } => {
                for stage in stages {
                    self.device.write_typed_buffer_timed(stage, &r.nan)?;
                }
            }
            #[cfg(feature = "algo")]
            Data::SumF32(_) | Data::SumU32(_) => {
                r.sum_word.set(r.workload.poison_word());
                r.sum_dev.replace(None);
            }
        }
        self.passes.set(0);
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        if self.variant.is_frame() {
            // One frame: the five inputs written, the six stages, a ticket of the final stage.
            (*out, _) = self.frame_chain(Some(inputs), true, false)?;
            self.passes.set(1);
            return Ok(());
        }
        let r = self.resident();
        match &r.data {
            Data::Add { x, y, .. } | Data::Saxpy { x, y, .. } => {
                self.device.write_typed_buffer(x, &inputs.x)?;
                self.device.write_typed_buffer(y, &inputs.y)?;
            }
            Data::Poly { x, .. } => self.device.write_typed_buffer(x, &inputs.x)?,
            Data::Hist { keys, bins } => {
                self.device.write_typed_buffer(keys, &inputs.keys)?;
                self.device.write_typed_buffer(bins, &r.zeros)?;
            }
            Data::MatMul { a, b, .. } => {
                self.device.write_typed_buffer(a, &inputs.x)?;
                self.device.write_typed_buffer(b, &inputs.y)?;
            }
            Data::Particles {
                input,
                acceleration,
                host_input,
                host_acceleration,
                ..
            } => {
                self.device.write_typed_buffer(input, host_input)?;
                self.device
                    .write_typed_buffer(acceleration, host_acceleration)?;
            }
            // W6, graph form: the uploads, the six dispatches and the readback are one graph, built and executed here.
            Data::Pipeline { .. } if r.workload.is_graph_form() => {
                *out = self
                    .run_graph(Some(inputs))?
                    .expect("a graph with a readback returns its words");
                self.passes.set(1);
                return Ok(());
            }
            // W6, hand order: five uploads, then the six passes in one submission, then the readback.
            Data::Pipeline {
                inputs: buffers, ..
            } => {
                for (buffer, values) in buffers.iter().zip(&inputs.pipeline) {
                    self.device.write_typed_buffer(buffer, values)?;
                }
            }
            // W7: the upload is an in-place write into the resident `GpuVec`'s buffer, as the hand path does.
            #[cfg(feature = "algo")]
            Data::SumF32(values) => self
                .device
                .write_typed_buffer(values.as_buffer(), &inputs.x)?,
            #[cfg(feature = "algo")]
            Data::SumU32(values) => self
                .device
                .write_typed_buffer(values.as_buffer(), &inputs.values)?,
        }
        self.run_once()?;
        self.passes.set(1);
        *out = self.read_words()?;
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        #[cfg(feature = "algo")]
        if let Data::SumF32(values) = &self.resident().data {
            self.dispatch_sum_f32(values)?;
            self.passes.set(self.passes.get().wrapping_add(1));
            return Ok(());
        }
        self.run_once()?;
        self.passes.set(self.passes.get().wrapping_add(1));
        Ok(())
    }

    /// GPU time of `batch` runs of the workload in one timed span (the harness divides by `batch`). `None` when the
    /// adapter has no timestamp queries (the existing no-sample path). W7: `batch` public sums inside one
    /// `gust_algo::timed` scope. W6-graph: one graph of `batch` repeats of the six dispatches through
    /// `execute_graph_timed`. Both spans leave out the read-back mapping.
    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        self.arm_pass_form();
        #[cfg(feature = "algo")]
        if self.resident().workload.family() == "W7" {
            return self.gpu_time_sum(batch);
        }
        if self.variant.is_frame() {
            // `batch` frames, one per chain, each with its own time at level `Frame`; their sum is the span of `batch`
            // chains the other rows report (the harness divides by `batch`). None when the adapter has no timestamps.
            let mut total = Duration::ZERO;
            for index in 0..batch {
                // The last frame also takes a ticket of the final stage (its copy lies after the span, outside the time).
                let last = index + 1 == batch;
                let (words, span) = self.frame_chain(None, last, true)?;
                if last {
                    *self.last_frame.borrow_mut() = Some(words);
                }
                let Some(span) = span else {
                    return Ok(None);
                };
                total += span;
            }
            self.passes
                .set(self.passes.get().wrapping_add(batch as u32));
            return Ok(Some(total));
        }
        if self.resident().workload.is_pipeline() {
            if self.resident().workload.is_graph_form() {
                let (_, elapsed, dispatches) = self.run_graph_with(None, batch, true)?;
                return per_run(dispatches, signal_pipeline::STAGES * batch, elapsed);
            }
            let timing = self.with_pipeline(|dispatches| {
                let repeated: Vec<_> = (0..batch).flat_map(|_| dispatches.clone()).collect();
                self.device.dispatch_batch_timed(&repeated)
            })?;
            self.passes
                .set(self.passes.get().wrapping_add(batch as u32));
            return Ok(timing.gpu_elapsed);
        }
        let timing = self.with_dispatch(|dispatch| {
            let batch = vec![dispatch; batch];
            self.device.dispatch_batch_timed(&batch)
        })?;
        self.passes
            .set(self.passes.get().wrapping_add(batch as u32));
        Ok(timing.gpu_elapsed)
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        if self.variant.is_frame() {
            // The frame rows' read-back is a frame of its own (the batch keeps its stage outputs to itself): the six
            // stages over the inputs the batch holds, and a ticket of the final stage. So a poisoned output cannot fail
            // this row. After a `gpu_time` call the words are those of its last timed frame instead (taken with its ticket).
            *out = match self.last_frame.borrow_mut().take() {
                Some(words) => words,
                None => self.frame_chain(None, true, false)?.0,
            };
            return Ok(());
        }
        *out = self.read_words()?;
        if self.resident().workload.accumulates() {
            normalize_counts(out, self.passes.get());
        }
        Ok(())
    }

    #[cfg(feature = "spirv-route")]
    fn route_check(&self) -> Result<(), String> {
        let want = if gust_wgpu::SPIRV_ROUTE {
            "spirv"
        } else {
            "wgsl"
        };
        let before = self.events_before_prepare.borrow();
        for (index, descriptor) in workloads::descriptors(self.resident().workload)
            .iter()
            .enumerate()
        {
            let kernel = descriptor.qualified_name().to_string();
            let events = self.route_events_of(&kernel);
            route_verdict(
                &kernel,
                want,
                own_events(&events, before.get(index).copied().unwrap_or(0)),
                events.last().copied(),
            )?;
        }
        Ok(())
    }

    /// The route evidence of the cell: the sums over its kernels (W6 has six), the last event of the last kernel.
    #[cfg(feature = "spirv-route")]
    fn route_evidence(&self) -> Option<Value> {
        let want = if gust_wgpu::SPIRV_ROUTE {
            "spirv"
        } else {
            "wgsl"
        };
        let before = self.events_before_prepare.borrow();
        let descriptors = workloads::descriptors(self.resident().workload);
        let names: Vec<String> = descriptors
            .iter()
            .map(|descriptor| descriptor.qualified_name().to_string())
            .collect();
        let (mut own_total, mut on_route, mut cumulative) = (0, 0, 0);
        for (index, kernel) in names.iter().enumerate() {
            let events = self.route_events_of(kernel);
            let own = own_events(&events, before.get(index).copied().unwrap_or(0));
            own_total += own.len();
            on_route += own.iter().filter(|route| **route == want).count();
            cumulative += events.len();
        }
        let last = names.last().and_then(|kernel| self.last_event(kernel));
        Some(serde_json::json!({
            "kernel": names.join(","),
            "kernels": names.len(),
            "build_route": want,
            "events": own_total,
            "events_on_build_route": on_route,
            "events_off_build_route": own_total - on_route,
            "events_cumulative": cumulative,
            "last_route": last.as_ref().map(|event| event.route),
            "last_reason": last.and_then(|event| event.reason),
            "source": "HeadlessDevice::route_events, read after the cell (the same events route_check reads), summed over the cell's kernels (one, or six for W6); the three `events` counts are this cell's own (since its prepare; 0 for a warm cell, which runs the pipeline an earlier cell built, and route_check then judges it on each kernel's latest event), `events_cumulative` is the device's whole life",
        }))
    }

    /// The route record of the cold start: the first kernel's (the only one, except for W6, whose record also lists the
    /// kernel count; every one of its six pipelines is compiled by the cold sample, since the cache is cleared first).
    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        // The pipeline cache is the device's, shared by every GUST row: a variant row clearing it would make the other
        // rows rebuild a pipeline inside their timed region (the harness refuses that). Only the base row has a cold start.
        if self.variant != Variant::Plain {
            return Ok(None);
        }
        self.device.clear_pipeline_cache();
        let started = Instant::now();
        self.dispatch()?;
        let elapsed = started.elapsed();
        let workload = self.resident().workload;
        let descriptor = workloads::descriptor(workload);
        let kernel = descriptor.qualified_name().to_string();
        let mut record = route::record(&route::Evidence {
            event: self.last_event(&kernel),
            embedded_artifacts: descriptor.artifacts.len(),
            strict_profile: gust::profile::STRICT,
            kernel: &kernel,
        });
        if let Some(object) = record.as_object_mut().filter(|_| workload.is_pipeline()) {
            object.insert(
                "kernels".into(),
                serde_json::json!(workloads::descriptors(workload).len()),
            );
        }
        *self.route.borrow_mut() = Some(record);
        Ok(Some(elapsed))
    }

    fn cold_route(&self) -> Option<Value> {
        self.route.borrow().clone()
    }

    fn cache_misses(&self) -> Option<u64> {
        Some(self.device.pipeline_cache_stats().misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Judging, Rounding, check_with, gen_inputs, reference};

    #[test]
    fn the_route_check_judges_a_warm_cell_on_the_latest_event_and_a_cold_cell_on_its_own() {
        // Warm cell: it began with 1 event on the device and made none; the earlier on-route event is the pipeline it ran.
        let events = ["spirv"];
        let own = own_events(&events, 1);
        assert!(own.is_empty());
        assert!(route_verdict("horner", "spirv", own, events.last().copied()).is_ok());
        // No event over the device's life: the route cannot be confirmed.
        assert!(route_verdict("horner", "spirv", &[], None).is_err());
        // A cell's own event off the route fails it, even when the latest event is on the route.
        let mixed = ["wgsl", "spirv"];
        assert!(
            route_verdict(
                "horner",
                "spirv",
                own_events(&mixed, 0),
                mixed.last().copied()
            )
            .is_err()
        );
        // A cold cell whose own event is on the route passes.
        assert!(route_verdict("horner", "spirv", own_events(&["spirv"], 0), Some("spirv")).is_ok());
        // A warm cell whose cached pipeline was last built off the route fails too.
        let off = ["spirv", "wgsl"];
        assert!(
            route_verdict("horner", "spirv", own_events(&off, 2), off.last().copied()).is_err()
        );
    }

    /// The GPU tests take turns: creating several Vulkan instances at once made adapter discovery fail now and then.
    fn gpu() -> (std::sync::MutexGuard<'static, ()>, DeviceHandle) {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let device =
            gust_wgpu::HeadlessDevice::new().expect("an adapter is needed for the GUST GPU test");
        (guard, DeviceHandle::new(device))
    }

    /// Every workload on GUST: end to end, a resident dispatch and a 16-pass batch each match the reference, and a
    /// poisoned output that is never dispatched fails the check (so the check can fail on the GPU path too).
    #[test]
    fn every_workload_passes_its_bound_on_gust_and_an_undispatched_output_fails() {
        let (_gpu, device) = gpu();
        for w in Workload::ALL {
            let mut backend = GustGpu::new(DeviceHandle::clone(&device));
            if !backend.supports(w) {
                continue;
            }
            let inputs = gen_inputs(w, workloads::test_size(w));
            let reference = reference(w, &inputs);
            backend.prepare(w, &inputs).unwrap();
            let tree = backend.sum_tree_bound(inputs.n, reference.abs_sum);
            assert_eq!(
                tree.is_some(),
                w == Workload::SumF32 && cfg!(feature = "algo")
            );
            let tree = Judging {
                rounding: Rounding::Either,
                tree,
            };
            let mut out = Vec::new();

            backend.poison().unwrap();
            backend.read_back(&mut out).unwrap();
            assert!(
                check_with(w, &inputs, &reference, &out, tree).is_err(),
                "{}: the poison alone must fail the check",
                w.id()
            );

            backend.poison().unwrap();
            backend.end_to_end(&inputs, &mut out).unwrap();
            check_with(w, &inputs, &reference, &out, tree)
                .unwrap_or_else(|e| panic!("{} end to end: {e}", w.id()));

            backend.poison().unwrap();
            backend.dispatch().unwrap();
            backend.read_back(&mut out).unwrap();
            check_with(w, &inputs, &reference, &out, tree)
                .unwrap_or_else(|e| panic!("{} dispatch: {e}", w.id()));

            backend.poison().unwrap();
            if backend.gpu_time(16).unwrap().is_some() {
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &reference, &out, tree)
                    .unwrap_or_else(|e| panic!("{} 16-pass batch: {e}", w.id()));
            }
        }
    }

    /// W6, both GUST forms: after one dispatch every stage's buffer is inside that stage's own GPU bound
    /// (`signal_pipeline::stage_exact_and_bound`), not only the final output.
    #[test]
    fn every_gust_stage_is_inside_its_own_bound_in_both_forms() {
        let (_gpu, device) = gpu();
        for w in [Workload::PipelineHand, Workload::PipelineGraph] {
            let mut backend = GustGpu::new(DeviceHandle::clone(&device));
            let inputs = gen_inputs(w, workloads::test_size(w));
            backend.prepare(w, &inputs).unwrap();
            backend.poison().unwrap();
            backend.dispatch().unwrap();
            let Data::Pipeline { stages, .. } = &backend.resident().data else {
                panic!("W6 has the pipeline data")
            };
            for (stage, buffer) in stages.iter().enumerate() {
                let got: Vec<f32> = device.read_typed_buffer(buffer).unwrap();
                let (exact, _, bound_gpu) =
                    signal_pipeline::stage_exact_and_bound(&inputs.pipeline, stage);
                for (i, ((got, exact), bound)) in got.iter().zip(&exact).zip(&bound_gpu).enumerate()
                {
                    let err = (f64::from(*got) - exact).abs();
                    assert!(
                        err <= *bound,
                        "{} stage {stage} element {i}: {err} against {bound}",
                        w.id()
                    );
                }
            }
        }
    }

    /// W7 takes its GPU time through `gust_algo::timed`: one public sum is one dispatch batch (`GpuTime.batches`), so
    /// `batch` sums in one scope read `batch` batches (a change in how many batches one `sum` makes would make the
    /// per-sum time wrong, and `gpu_time` refuses it). The time exists exactly where the adapter has timestamp queries; a
    /// `None` is the harness's existing no-sample path.
    #[cfg(feature = "algo")]
    #[test]
    fn a_w7_sum_is_one_batch_and_its_gpu_time_follows_the_adapter() {
        let (_gpu, device) = gpu();
        for w in [Workload::SumU32, Workload::SumF32] {
            let mut backend = GustGpu::new(DeviceHandle::clone(&device));
            let inputs = gen_inputs(w, workloads::test_size(w));
            backend.prepare(w, &inputs).unwrap();
            for batch in [1usize, 2, 16] {
                let time = backend.timed_sums(batch).unwrap();
                assert_eq!(time.batches, batch, "{}: batches of {batch} sums", w.id());
                assert_eq!(
                    time.gpu_elapsed.is_some(),
                    device.supports_gpu_timestamps(),
                    "{}",
                    w.id()
                );
                assert_eq!(
                    backend.gpu_time(batch).unwrap().is_some(),
                    device.supports_gpu_timestamps(),
                    "{}",
                    w.id()
                );
            }
        }
    }

    /// The refusal of a count that is not the run count: the pure check `gpu_time` applies to W7 (`batches`) and to
    /// W6-graph (`dispatches`).
    #[test]
    fn a_count_that_is_not_the_run_count_is_refused() {
        let elapsed = Some(Duration::from_micros(7));
        assert_eq!(per_run(16, 16, elapsed).unwrap(), elapsed);
        assert_eq!(per_run(96, 96, None).unwrap(), None);
        for (counted, expected) in [(32, 16), (15, 16), (0, 16), (6, 96)] {
            let error = per_run(counted, expected, elapsed).unwrap_err().to_string();
            assert!(error.contains("were expected"), "{error}");
        }
    }

    /// W6-graph takes its GPU time from `execute_graph_timed`: a graph of one run and a graph of 16 repeated runs both
    /// report a time exactly where the adapter has timestamp queries, and the repeated graph is a valid graph (the
    /// output is checked by the every-workload test above after `gpu_time(16)`).
    #[test]
    fn a_w6_graph_reports_a_gpu_time_for_one_run_and_for_a_repeated_batch() {
        let (_gpu, device) = gpu();
        let w = Workload::PipelineGraph;
        let mut backend = GustGpu::new(DeviceHandle::clone(&device));
        let inputs = gen_inputs(w, workloads::test_size(w));
        backend.prepare(w, &inputs).unwrap();
        for batch in [1usize, 16] {
            // The batch graph holds `batch` repeats of the six dispatches, so the division by `batch` is per run.
            let (_, _, dispatches) = backend.run_graph_with(None, batch, true).unwrap();
            assert_eq!(dispatches, 6 * batch);
            let time = backend.gpu_time(batch).unwrap();
            assert_eq!(time.is_some(), device.supports_gpu_timestamps(), "{batch}");
            if let Some(time) = time {
                assert!(time < Duration::from_secs(10), "{time:?}");
            }
        }
    }

    /// The route record of a cold start names the path the build can take: a kernel without an embedded artifact
    /// (gust-bench) compiles at run time; a kernel with one (gust-bench-pre, built under a strict profile) reports the
    /// artifact path, and as shown, not inferred.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn a_cold_start_records_the_route_that_ran() {
        let (_gpu, device) = gpu();
        let w = Workload::VectorAdd;
        let inputs = gen_inputs(w, 4096);
        let mut backend = GustGpu::new(device);
        backend.prepare(w, &inputs).unwrap();
        assert!(
            backend.cold_route().is_none(),
            "nothing before a cold start"
        );
        backend.cold_start().unwrap();
        let record = backend
            .cold_route()
            .expect("a cold start records its route");
        let descriptor = workloads::descriptor(w);
        assert_eq!(record["kernel"], descriptor.qualified_name().to_string());
        if descriptor.artifacts.is_empty() {
            assert_eq!(record["path"], "run-time compile");
            assert_eq!(record["shown"], true);
        } else {
            assert_eq!(record["path"], "artifact", "{record}");
            assert_eq!(
                record["shown"], true,
                "the artifact path is shown by the build or an event: {record}"
            );
            assert!(
                gust::profile::STRICT,
                "an embedded artifact is only possible in a strict-profile build"
            );
        }
        // A second cold start records again (one record per sample).
        backend.cold_start().unwrap();
        assert_eq!(
            backend.cold_route().expect("recorded again")["kernel"],
            record["kernel"]
        );
    }

    /// The real tree bound of the public sum (`f32_sum_bound_v1`, as `GustGpu::sum_bound` returns it) judges a result
    /// just inside it as a pass and the next f32 above as a failure (review S6).
    #[cfg(feature = "algo")]
    #[test]
    fn a_sum_just_past_the_real_v1_bound_fails_and_one_just_inside_passes() {
        let w = Workload::SumF32;
        let inputs = gen_inputs(w, 4103);
        let reference = reference(w, &inputs);
        let tree = sum_bound_from(4103, reference.abs_sum, true);
        assert!(tree > 0.0 && tree < 0.1, "{tree}");
        let judging = Judging {
            rounding: Rounding::Either,
            tree: Some(tree),
        };
        let exact = reference.exact[0];
        let limit = tree + reference.ref_error;
        let mut inside = (exact + limit) as f32;
        while f64::from(inside) - exact > limit {
            inside = inside.next_down();
        }
        while f64::from(inside.next_up()) - exact <= limit {
            inside = inside.next_up();
        }
        check_with(w, &inputs, &reference, &[inside], judging).unwrap();
        let outside = inside.next_up();
        assert!(f64::from(outside) - exact > limit);
        assert!(check_with(w, &inputs, &reference, &[outside], judging).is_err());
        // And on the low side.
        let mut inside = (exact - limit) as f32;
        while exact - f64::from(inside) > limit {
            inside = inside.next_up();
        }
        while exact - f64::from(inside.next_down()) <= limit {
            inside = inside.next_down();
        }
        check_with(w, &inputs, &reference, &[inside], judging).unwrap();
        assert!(check_with(w, &inputs, &reference, &[inside.next_down()], judging).is_err());
    }

    /// Where the v1 launch is not admitted (too little workgroup memory, a storage binding too small, too few workgroups
    /// per dimension) the bound is the looser of the v0 and v1 ones; where it is admitted it is v1's (review S6).
    #[cfg(feature = "algo")]
    #[test]
    fn the_bound_falls_back_to_the_looser_of_v0_and_v1_where_v1_is_not_admitted() {
        use gust_algo::bound::{C_GPU, f32_sum_bound, f32_sum_bound_v1};
        let (n, s) = (4103u32, 1234.5f64);
        let (v0, v1) = (f32_sum_bound(n, C_GPU, s), f32_sum_bound_v1(n, C_GPU, s));
        assert_eq!(sum_bound_from(n, s, true), v1);
        assert_eq!(sum_bound_from(n, s, false), v0.max(v1));
        assert!(sum_bound_from(n, s, false) >= sum_bound_from(n, s, true));
        // The replicated admission test, term by term (default limits: 16 KiB workgroup memory, 128 MiB binding, 65535).
        let ok = (16384u32, 128u64 << 20, 65535u32);
        assert!(v1_admitted(n, ok.0, ok.1, ok.2));
        assert!(
            !v1_admitted(n, 3 * 256 * 4 - 1, ok.1, ok.2),
            "workgroup memory"
        );
        assert!(v1_admitted(n, 3 * 256 * 4, ok.1, ok.2));
        assert!(!v1_admitted(n, ok.0, u64::from(n) * 4 - 1, ok.2), "binding");
        assert!(v1_admitted(n, ok.0, u64::from(n) * 4, ok.2));
        // 2^27 values need 16384 workgroups in the first pass.
        assert!(!v1_admitted(1 << 27, ok.0, 1 << 30, 100), "workgroups");
        // On a device the replica agrees with the library: the default headless device admits a small input.
        let (_gpu, device) = gpu();
        let limits = device.limits();
        assert!(v1_admitted(
            n,
            limits.max_compute_workgroup_storage_size,
            limits.max_storage_buffer_binding_size,
            limits.max_compute_workgroups_per_dimension
        ));
    }

    /// pf.onepass: the three `--with` rows parse, are told apart by id, and differ from the base row only as stated
    /// (the per-pass row alone forces the per-dispatch form, the variants exist for W6-hand only; the dispatch counts
    /// are checked on a device, below).
    #[test]
    fn pfonepass_rows_parse_and_differ_only_as_stated() {
        let names = [
            "gust-default",
            "gust-perpass",
            "gust-fused",
            "gust-frame",
            "gust-frame-perpass",
        ];
        let rows = names.map(|name| Variant::parse(name).expect(name));
        assert_eq!(
            rows,
            [
                Variant::Default,
                Variant::PerPass,
                Variant::Fused,
                Variant::Frame,
                Variant::FramePerPass
            ]
        );
        for name in [
            "gust-gpu",
            "gust-shader",
            "gust-setup",
            "gust-unfused",
            "gust-onepass",
            "fused",
            "",
        ] {
            assert!(Variant::parse(name).is_none(), "{name:?}");
        }
        let all = [Variant::Plain, rows[0], rows[1], rows[2], rows[3], rows[4]];
        let ids = all.map(Variant::id);
        assert_eq!(ids[0], backend_id());
        assert_eq!(
            &ids[1..],
            [
                "gust-gpu+default",
                "gust-gpu+perpass",
                "gust-gpu+fused",
                "gust-gpu+frame",
                "gust-gpu+frame-perpass"
            ]
        );
        assert_eq!(
            all.map(Variant::per_pass),
            [false, false, true, false, false, true]
        );
        for workload in Workload::ALL {
            assert!(Variant::Plain.supports(workload));
            for row in rows {
                assert_eq!(row.supports(workload), workload == Workload::PipelineHand);
            }
        }
    }

    /// pf.onepass: every variant row's output passes the W6 bound on a device (end to end, one dispatch and a 16-run
    /// batch), as the base row's does, a poisoned output that is never dispatched fails, and the rows record the
    /// passes they are named for (one for the default and fused rows, one per dispatch for the per-pass row). The
    /// process-wide form is handed back to the default afterwards.
    #[test]
    fn pfonepass_rows_pass_the_w6_bound_and_record_the_passes_they_name() {
        /// Hands the process-wide pass form back to the default, however the test ends.
        struct DefaultForm;
        impl Drop for DefaultForm {
            fn drop(&mut self) {
                gust_wgpu::force_pass_per_dispatch(false);
            }
        }
        let (_gpu, device) = gpu();
        let _form = DefaultForm;
        let w = Workload::PipelineHand;
        let inputs = gen_inputs(w, workloads::test_size(w));
        let want = reference(w, &inputs);
        let judging = Judging {
            rounding: Rounding::Either,
            tree: None,
        };
        for variant in [
            Variant::Default,
            Variant::PerPass,
            Variant::Fused,
            Variant::Frame,
            Variant::FramePerPass,
        ] {
            let mut backend = GustGpu::with_variant(DeviceHandle::clone(&device), variant);
            backend.prepare(w, &inputs).unwrap();
            // The fused row is one dispatch of the one kernel; the others are the six stages.
            let dispatches = if variant.is_frame() {
                signal_pipeline::STAGES
            } else {
                backend.with_pipeline(|dispatches| dispatches.len())
            };
            assert_eq!(
                dispatches,
                if variant == Variant::Fused {
                    1
                } else {
                    signal_pipeline::STAGES
                }
            );
            let mut out = Vec::new();
            backend.poison().unwrap();
            backend.read_back(&mut out).unwrap();
            // A frame row's read-back is a frame of its own, so a poisoned output cannot fail it.
            assert_eq!(
                check_with(w, &inputs, &want, &out, judging).is_err(),
                !variant.is_frame()
            );
            backend.poison().unwrap();
            backend.end_to_end(&inputs, &mut out).unwrap();
            check_with(w, &inputs, &want, &out, judging)
                .unwrap_or_else(|e| panic!("{variant:?} end to end: {e}"));
            backend.poison().unwrap();
            let passes_before = gust_wgpu::batch_passes_recorded();
            backend.dispatch().unwrap();
            assert_eq!(
                gust_wgpu::batch_passes_recorded() - passes_before,
                if variant.per_pass() {
                    signal_pipeline::STAGES as u64
                } else {
                    1
                },
                "{variant:?}: passes of one dispatch"
            );
            backend.read_back(&mut out).unwrap();
            check_with(w, &inputs, &want, &out, judging)
                .unwrap_or_else(|e| panic!("{variant:?} dispatch: {e}"));
            backend.poison().unwrap();
            if backend.gpu_time(16).unwrap().is_some() {
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &want, &out, judging)
                    .unwrap_or_else(|e| panic!("{variant:?} 16-run batch: {e}"));
            }
        }
    }
}
