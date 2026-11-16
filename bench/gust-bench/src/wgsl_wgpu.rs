//! Hand-written WGSL on plain wgpu, the key comparator: GUST emits Slang, compiles it to WGSL and hands that to
//! wgpu, so a human's WGSL on the same adapter goes through the same naga and driver path.
//!
//! Two variants of the same kernel, so a gap to GUST can be attributed:
//! - `Prebuilt`: what a careful hand-written host does. The bind group and the read-back staging buffer are built
//!   once; an end-to-end sample is one submission that dispatches and copies to the staging buffer.
//! - `PerCall`: the same work shaped like `gust-wgpu` does it per call. A bind group is created for every dispatch,
//!   the read-back allocates a fresh staging buffer and a fresh `Vec`, and dispatch and copy are two submissions with
//!   a wait after each.
//!
//! The three host-side differences between the two are the ablation switches (plan 2.6), each a flag of [`Switches`]:
//! bind-group reuse, staging-buffer reuse and one submission instead of two. `Prebuilt` has all three on, `PerCall` all
//! three off (today's behaviour, the default); an ablation backend is `PerCall` with exactly one of them flipped. The
//! fresh result `Vec` of `PerCall` stays per call in every ablation (it is not one of the three switches).
//!
//! The device is a second `wgpu::Device` on the same adapter selection (Vulkan, high performance) with default limits
//! and `TIMESTAMP_QUERY`, exactly what `HeadlessDevice` requests. `HeadlessDevice` does not expose its own device,
//! so sharing one is not possible through the public API. A run with W7 inputs above 2^25 words opens it (and GUST's
//! device) with the adapter's own buffer and storage-binding limits instead (`WgpuContext::open_on`).
//!
//! W7 (sum) is a two-dispatch workgroup-tree reduction in one compute pass: `reduce_blocks` (`workloads::sum::hand_blocks`
//! workgroups, one per 2048 values, capped by the device's workgroup limit: grid-stride sums per lane, then a tree in
//! workgroup memory) and `reduce_total` (one workgroup, each lane adding its share of the partials, then a tree).
//! W9 (naive matrix product) is one dispatch of `N / 16` by `N / 16` workgroups of 16 by 16. W5 (particle step) is one
//! dispatch of 256-wide workgroups over the same 28-byte std430 struct buffer GUST's kernel takes (`shaders/particles.wgsl`).
//! W6 (staged pipeline) is six entry points of `shaders/signal_pipeline.wgsl` over one three-binding layout, one bind group per stage
//! (`STAGE_BINDINGS`) and one compute pass per stage in a single submission, in the hand order; both W6 ids run it.
//! `--adapter` opens this device (and every probe device) on the adapter GUST's device is on, matched by `adapter::match_adapter`.

use std::{
    cell::Cell,
    rc::Rc,
    sync::mpsc,
    time::{Duration, Instant},
};

use bytemuck::cast_slice;

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{self, Inputs, WORKGROUP, Workload, hist_threads, normalize_counts},
};

/// A wgpu device and queue.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub timestamps: bool,
}

impl WgpuContext {
    /// The default device (the high-performance adapter). The run itself opens its device through `open_on`; this one is
    /// for the Windows adapter probe, the profile mode and the tests.
    #[cfg(any(windows, test, feature = "profile-counts"))]
    pub fn new() -> BenchResult<Self> {
        Self::open_with(None, false, false, None, false)
    }

    /// Opens the device; with `trace` (a `wgpu-trace` build) wgpu writes its API trace of this device into that directory.
    #[cfg(feature = "wgpu-trace")]
    pub fn open(trace: Option<&std::path::Path>) -> BenchResult<Self> {
        Self::open_with(trace, false, false, None, false)
    }

    /// A device set up the way `HeadlessDevice` sets its own up (the m10.b1c probe `gust-setup`): it also requests
    /// `MEMORY_DECORATION_COHERENT` when the adapter has it (`gust-wgpu` `lib.rs`, `HeadlessDevice::open`). On the adapter
    /// whose name contains `adapter` (`--adapter`), or the high-performance one with `None`.
    pub fn open_like_gust_on(adapter: Option<&str>) -> BenchResult<Self> {
        Self::open_with(None, true, false, adapter, false)
    }

    /// The default device on the adapter whose name contains `adapter` (`--adapter`), or the high-performance one with
    /// `None`. With `adapter_limits` it requests the adapter's own `max_buffer_size`, `max_storage_buffer_binding_size` and
    /// `max_compute_workgroups_per_dimension` (every other limit at wgpu's default), as `LimitsPolicy::Adapter` does for
    /// `HeadlessDevice`: the one a W7 input above 2^25 words needs.
    pub fn open_on(adapter: Option<&str>, adapter_limits: bool) -> BenchResult<Self> {
        Self::open_with(None, false, adapter_limits, adapter, false)
    }

    /// The device of the `rustgpu-wgpu` backend: its own device (the run's third) on the adapter selection (the `--adapter`
    /// substring or the high-performance choice) with default limits, `TIMESTAMP_QUERY` and `PASSTHROUGH_SHADERS`.
    /// `Err(Refusal)` when the adapter lacks a capability the module needs (see `open_with`); any other `Err` is a failure
    /// to open the device.
    #[cfg(feature = "rust-gpu")]
    pub fn open_passthrough_on(adapter: Option<&str>) -> BenchResult<Self> {
        Self::open_with(None, false, false, adapter, true)
    }

    /// The adapter of a `--adapter` substring on `instance`: the one adapter of the Vulkan enumeration whose name contains
    /// it (`adapter::match_adapter`, the same rule GUST's device is opened by), or wgpu's high-performance choice.
    pub(crate) fn pick_adapter(
        instance: &wgpu::Instance,
        name: Option<&str>,
    ) -> BenchResult<wgpu::Adapter> {
        let Some(name) = name else {
            return Ok(pollster::block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                },
            ))?);
        };
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let names = adapters
            .iter()
            .map(|adapter| adapter.get_info().name)
            .collect::<Vec<_>>();
        let index = crate::adapter::match_adapter(&names, name)?;
        Ok(adapters.swap_remove(index))
    }

    fn open_with(
        trace: Option<&std::path::Path>,
        like_gust: bool,
        adapter_limits: bool,
        adapter_name: Option<&str>,
        passthrough: bool,
    ) -> BenchResult<Self> {
        #[cfg(not(feature = "wgpu-trace"))]
        if trace.is_some() {
            return Err("a wgpu trace needs a build with the wgpu-trace feature".into());
        }
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = Self::pick_adapter(&instance, adapter_name)?;
        // A passthrough device (the rust-gpu comparator) is refused, with the reason and as a `Refusal`, when the adapter
        // lacks a capability the module needs; the backend is then not run (never a fallback to another shader path).
        if passthrough {
            Self::require_passthrough_capabilities(&adapter)?;
        }
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_limits = wgpu::Limits::default();
        if adapter_limits {
            let available = adapter.limits();
            required_limits.max_buffer_size = available.max_buffer_size;
            required_limits.max_storage_buffer_binding_size =
                available.max_storage_buffer_binding_size;
            required_limits.max_compute_workgroups_per_dimension =
                available.max_compute_workgroups_per_dimension;
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bench hand-WGSL device"),
                #[cfg(feature = "wgpu-trace")]
                trace: trace.map_or(wgpu::Trace::Off, |dir| {
                    wgpu::Trace::Directory(dir.to_path_buf())
                }),
                required_features: (if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                }) | if like_gust {
                    adapter.features() & wgpu::Features::MEMORY_DECORATION_COHERENT
                } else {
                    wgpu::Features::empty()
                } | if passthrough {
                    wgpu::Features::PASSTHROUGH_SHADERS
                } else {
                    wgpu::Features::empty()
                },
                required_limits,
                ..Default::default()
            }))?;
        Ok(Self {
            device,
            queue,
            info: adapter.get_info(),
            timestamps,
        })
    }

    /// The capabilities the rust-gpu module needs of an adapter, each a typed `Refusal` when missing:
    /// - wgpu's `PASSTHROUGH_SHADERS` (a defensive check: wgpu-hal 30.0.2 sets it on every Vulkan adapter);
    /// - the Vulkan memory model the module declares (`OpCapability VulkanMemoryModel`). wgpu-hal 30.0.2 creates every
    ///   device with `vulkanMemoryModel` off (it turns it on only with cooperative matrices), which breaks a valid-usage
    ///   rule of the module, so the backend runs only where the physical device's Vulkan API version is 1.3 or later (the
    ///   version that requires the feature to be supported). The adapter's own feature query is not reachable through
    ///   wgpu's HAL access without a Vulkan dependency of the bench, so the version stands for it.
    #[cfg(feature = "rust-gpu")]
    fn require_passthrough_capabilities(
        adapter: &wgpu::Adapter,
    ) -> Result<(), crate::rustgpu_module::Refusal> {
        let passthrough = adapter
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
        // SAFETY: `as_hal` only borrows the adapter's Vulkan HAL object, which is read (a version number), not modified.
        let api_version = unsafe {
            adapter
                .as_hal::<wgpu::hal::api::Vulkan>()
                .map(|hal| hal.physical_device_capabilities().properties().api_version)
        };
        crate::rustgpu_module::check_capabilities(
            &adapter.get_info().name,
            passthrough,
            api_version,
        )
    }

    #[cfg(not(feature = "rust-gpu"))]
    fn require_passthrough_capabilities(_adapter: &wgpu::Adapter) -> BenchResult<()> {
        unreachable!("a passthrough device needs a build with the rust-gpu feature")
    }

    fn wait(&self, submission: wgpu::SubmissionIndex) -> BenchResult<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        Ok(())
    }

    /// Submits nothing and waits, which also flushes the writes queued by `queue.write_buffer`.
    fn flush(&self) -> BenchResult<()> {
        let submission = self.queue.submit([]);
        self.wait(submission)
    }

    fn buffer(&self, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Maps `buffer` for reading, waits for the map, and hands the bytes to `use_bytes`.
    fn map_read<R>(
        &self,
        buffer: &wgpu::Buffer,
        use_bytes: impl FnOnce(&[u8]) -> R,
    ) -> BenchResult<R> {
        let (sender, receiver) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result.map_err(|error| error.to_string()));
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;
        let view = buffer.get_mapped_range(..)?;
        let result = use_bytes(&view);
        drop(view);
        buffer.unmap();
        Ok(result)
    }
}

/// wgpu's limit on the queries of one query set.
const MAX_QUERIES: usize = 4096;

/// Refuses a W7 batch whose `2 * batch` timestamps would not fit one query set (batches over 2048); it is not split.
fn check_span_batch(batch: usize) -> BenchResult<()> {
    if 2 * batch > MAX_QUERIES {
        return Err(format!(
            "a W7 batch of {batch} needs {} timestamp queries, over wgpu's limit of {MAX_QUERIES} per query set (the batch must be at most {})",
            2 * batch,
            MAX_QUERIES / 2
        )
        .into());
    }
    Ok(())
}

/// The sum of the spans of timestamp pairs `(ticks[2i], ticks[2i+1])`, and the number of pairs. `period` is the queue's
/// nanoseconds per tick.
fn span_sum(ticks: &[u64], period: f32) -> (Duration, usize) {
    let (pairs, _) = ticks.as_chunks::<2>();
    let count = pairs.len();
    let nanoseconds: f64 = pairs
        .iter()
        .map(|pair| pair[1].wrapping_sub(pair[0]) as f64 * f64::from(period))
        .sum();
    (
        Duration::from_secs_f64(nanoseconds / 1_000_000_000.0),
        count,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Prebuilt,
    PerCall,
}

/// One host-side difference between `Prebuilt` and `PerCall`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    /// Reuse one bind group instead of creating one per pass.
    BindGroupReuse,
    /// Reuse the read-back staging buffer instead of creating one per read-back.
    StagingReuse,
    /// Dispatch and copy to the staging buffer in one submission, not two with a wait after the first.
    OneSubmission,
}

impl Switch {
    pub const ALL: [Switch; 3] = [
        Switch::BindGroupReuse,
        Switch::StagingReuse,
        Switch::OneSubmission,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Switch::BindGroupReuse => "bind-group-reuse",
            Switch::StagingReuse => "staging-reuse",
            Switch::OneSubmission => "one-submission",
        }
    }

    pub fn parse(text: &str) -> Option<Switch> {
        Self::ALL.into_iter().find(|s| s.id() == text)
    }

    fn backend_id(self) -> &'static str {
        match self {
            Switch::BindGroupReuse => "wgsl-percall+bind-group-reuse",
            Switch::StagingReuse => "wgsl-percall+staging-reuse",
            Switch::OneSubmission => "wgsl-percall+one-submission",
        }
    }
}

/// The three ablation flags; `true` is the careful (`Prebuilt`) behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Switches {
    pub bind_group_reuse: bool,
    pub staging_reuse: bool,
    pub one_submission: bool,
}

impl Switches {
    pub const PREBUILT: Switches = Switches {
        bind_group_reuse: true,
        staging_reuse: true,
        one_submission: true,
    };
    /// Today's `PerCall` behaviour: every switch off.
    pub const PER_CALL: Switches = Switches {
        bind_group_reuse: false,
        staging_reuse: false,
        one_submission: false,
    };

    /// These switches with `switch` turned on.
    pub fn with(mut self, switch: Switch) -> Self {
        match switch {
            Switch::BindGroupReuse => self.bind_group_reuse = true,
            Switch::StagingReuse => self.staging_reuse = true,
            Switch::OneSubmission => self.one_submission = true,
        }
        self
    }
}

/// What the timed paths did, counted where it happens: bind groups created per pass, staging buffers created per
/// read-back and command submissions (the empty flush submissions of the harness are not counted).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub bind_groups: u64,
    pub staging_buffers: u64,
    pub submissions: u64,
}

#[derive(Default)]
struct Counters {
    bind_groups: Cell<u64>,
    staging_buffers: Cell<u64>,
    submissions: Cell<u64>,
}

fn bump(cell: &Cell<u64>) {
    cell.set(cell.get() + 1);
}

struct Pipeline {
    /// One compute pipeline per entry point of the workload (two for the W7 reduction, one for the rest), on one module.
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
}

struct Resident {
    workload: Workload,
    /// Words of the output buffer.
    out_len: usize,
    /// Bytes of the output buffer.
    bytes: u64,
    /// In binding order: `[a, b, out]` for vector-add, `[scale, x, y, out]` for saxpy, `[coeffs, x, out]` for the
    /// polynomial, `[keys, bins]` for the histogram, `[x, partials, out]` for the sum and `[dims, a, b, c]` for the
    /// matrix product. The output is always last.
    buffers: Vec<wgpu::Buffer>,
    /// The buffers (indexes into `buffers`) an end-to-end sample uploads from the host inputs, in the order of
    /// [`HandWgsl::input_bytes`].
    uploads: Vec<usize>,
    staging: wgpu::Buffer,
    pipeline: Pipeline,
    bind_group: wgpu::BindGroup,
    query_set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    timestamp_read: wgpu::Buffer,
    /// The output's poison words (NaN, or zero for the histogram, which is also what a histogram clears to).
    poison: Vec<f32>,
    /// The dispatches of one pass of the workload, in order: the pipeline (index into `Pipeline::pipelines`) and the
    /// workgroup counts.
    steps: Vec<(usize, [u32; 3])>,
    /// W7: the workgroups of the first dispatch (0 for every other workload).
    blocks: u32,
    /// W6: per stage the three buffers (indexes into `buffers`) bound to bindings 0, 1 and 2 (empty for every other
    /// workload, which has the one `bind_group`).
    stage_bindings: Vec<[usize; 3]>,
    /// W6: the bind group of each stage, built once (`bind_group` is the first stage's).
    stage_groups: Vec<wgpu::BindGroup>,
}

pub struct HandWgsl {
    ctx: Rc<WgpuContext>,
    mode: Mode,
    switches: Switches,
    /// The one switch an ablation backend turned on, for its id and description.
    ablation: Option<Switch>,
    /// The m10.b1c probes (`--with`): `Prebuilt` plus what the wgpu trace shows differing from GUST. `gust-setup` is all
    /// of it (GUST's resource creation order, GUST's own generated WGSL as the kernel, and a device that requests what
    /// `HeadlessDevice` requests, which is the context's); `gust-shader` and `gust-device` are the two halves.
    gust_shader: bool,
    gust_order: bool,
    probe_id: Option<&'static str>,
    /// The `spirv` group's hand rows (D193): the hand WGSL compiled without wgpu's runtime checks
    /// (bounds checks, loop bounding), the checks the WGSL route has and the SPIR-V passthrough route does not; `Prebuilt`
    /// is `wgsl-prebuilt-unchecked` (the group's reference), `PerCall` is `wgsl-percall-unchecked` (its control).
    unchecked: bool,
    /// The `rustgpu-wgpu` comparator (m10.b4): `Prebuilt`'s host shape and device shape, with the kernel from the
    /// committed rust-gpu SPIR-V module loaded through wgpu's passthrough route instead of WGSL. Always false in a build
    /// without the `rust-gpu` feature.
    rust_gpu: bool,
    /// All three switches on, the result `Vec` still fresh per call (`Mode::PerCall`): the row between the single
    /// switches and `wgsl-prebuilt`, which also reuses the `Vec`.
    all_on: bool,
    counters: Counters,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates; see `Backend::dispatch`).
    passes: Cell<u32>,
    /// W7: the timestamp pairs the last `gpu_time` read (one per repeat; tests pin it to `batch`).
    last_spans: Cell<usize>,
}

impl HandWgsl {
    pub fn new(ctx: Rc<WgpuContext>, mode: Mode) -> Self {
        let switches = match mode {
            Mode::Prebuilt => Switches::PREBUILT,
            Mode::PerCall => Switches::PER_CALL,
        };
        Self {
            ctx,
            mode,
            switches,
            ablation: None,
            gust_shader: false,
            gust_order: false,
            probe_id: None,
            unchecked: false,
            rust_gpu: false,
            all_on: false,
            counters: Counters::default(),
            resident: None,
            passes: Cell::new(0),
            last_spans: Cell::new(0),
        }
    }

    /// `PerCall` with exactly `switch` turned on and the other two off.
    pub fn ablation(ctx: Rc<WgpuContext>, switch: Switch) -> Self {
        Self {
            switches: Switches::PER_CALL.with(switch),
            ablation: Some(switch),
            ..Self::new(ctx, Mode::PerCall)
        }
    }

    /// A hand row of the `spirv` group (D193): `mode` with the hand WGSL compiled without wgpu's runtime checks (bounds
    /// checks and loop bounding), as the SPIR-V route's shaders are, on the shared hand device. `Mode::Prebuilt` is
    /// `wgsl-prebuilt-unchecked`, `Mode::PerCall` is `wgsl-percall-unchecked`.
    pub fn unchecked(ctx: Rc<WgpuContext>, mode: Mode) -> Self {
        Self {
            unchecked: true,
            ..Self::new(ctx, mode)
        }
    }

    /// The rust-gpu comparator `rustgpu-wgpu`: the host side of `wgsl-prebuilt` (bind group and staging buffer built once,
    /// one submission per end-to-end sample) over a passthrough device (`WgpuContext::open_passthrough_on`), the kernel
    /// the committed rust-gpu module's entry point of the workload.
    #[cfg(feature = "rust-gpu")]
    pub fn rust_gpu(ctx: Rc<WgpuContext>) -> Self {
        Self {
            rust_gpu: true,
            ..Self::new(ctx, Mode::Prebuilt)
        }
    }

    /// The GUST-difference probes, each `Prebuilt` plus what the wgpu trace shows differing from GUST. `gust-shader`: on the
    /// shared hand device, running GUST's generated WGSL. `gust-device`: on a device opened like GUST's (`open_like_gust_on`,
    /// which requests `MEMORY_DECORATION_COHERENT`), creating the timestamp objects before the data buffers as
    /// `HeadlessDevice` does, running the hand WGSL. `gust-setup` (any other name): both.
    pub fn gust_probe(ctx: Rc<WgpuContext>, name: &str) -> Self {
        let (gust_shader, gust_order, probe_id) = match name {
            "gust-shader" => (true, false, "wgsl-prebuilt+gust-shader"),
            "gust-device" => (false, true, "wgsl-prebuilt+gust-device"),
            _ => (true, true, "wgsl-prebuilt+gust-setup"),
        };
        Self {
            gust_shader,
            gust_order,
            probe_id: Some(probe_id),
            ..Self::new(ctx, Mode::Prebuilt)
        }
    }

    /// `PerCall` with every switch on: it differs from `Prebuilt` only in the fresh result `Vec` per call, so the
    /// gap between the two is the `Vec`, and the gap between this and `wgsl-percall` is the three switches together.
    pub fn all_on(ctx: Rc<WgpuContext>) -> Self {
        Self {
            switches: Switches::PREBUILT,
            all_on: true,
            ..Self::new(ctx, Mode::PerCall)
        }
    }

    #[cfg(test)]
    pub fn switches(&self) -> Switches {
        self.switches
    }

    #[cfg(test)]
    pub fn counts(&self) -> Counts {
        Counts {
            bind_groups: self.counters.bind_groups.get(),
            staging_buffers: self.counters.staging_buffers.get(),
            submissions: self.counters.submissions.get(),
        }
    }

    /// Submits one command buffer and counts it.
    fn submit(&self, commands: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        bump(&self.counters.submissions);
        self.ctx.queue.submit([commands])
    }

    /// A fresh read-back staging buffer, counted.
    fn fresh_staging(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        bump(&self.counters.staging_buffers);
        self.ctx.buffer(
            label,
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    /// GUST's own generated WGSL for the workload's kernel and its entry point: the precompiled artifacts that
    /// `gust-bench-pre` embeds (the text the run-time compile hands to wgpu; byte-checked against the shader module of
    /// a GUST wgpu trace for W1 and W2).
    fn gust_source(workload: Workload) -> (&'static str, &'static str) {
        match workload.family() {
            // `supports` leaves the probes out of the workloads whose GUST kernel has no artifact here.
            "W5" | "W6" | "W7" | "W9" => {
                unreachable!("{} has no GUST artifact probe", workload.id())
            }
            "W1" => (
                include_str!("../../gust-bench-pre/gust-artifacts/2446cbf187eae739.wgsl"),
                "gpu_kernel_add",
            ),
            "W2" => (
                include_str!("../../gust-bench-pre/gust-artifacts/fe1197ff8b0197e2.wgsl"),
                "gpu_kernel_saxpy",
            ),
            "W3" => (
                include_str!("../../gust-bench-pre/gust-artifacts/2580c40b11271718.wgsl"),
                "gpu_kernel_horner",
            ),
            _ => (
                include_str!("../../gust-bench-pre/gust-artifacts/48733ffa27c233fd.wgsl"),
                "gpu_kernel_histogram",
            ),
        }
    }

    fn source(workload: Workload) -> &'static str {
        match workload.family() {
            "W1" => include_str!("../../shaders/vector_add.wgsl"),
            "W2" => include_str!("../../shaders/saxpy.wgsl"),
            "W3" => include_str!("../../shaders/polynomial.wgsl"),
            "W5" => include_str!("../../shaders/particles.wgsl"),
            "W6" => include_str!("../../shaders/signal_pipeline.wgsl"),
            "W7" if workload == Workload::SumU32 => include_str!("../../shaders/sum_u32.wgsl"),
            "W7" => include_str!("../../shaders/sum_f32.wgsl"),
            "W9" => include_str!("../../shaders/matmul.wgsl"),
            _ => include_str!("../../shaders/histogram.wgsl"),
        }
    }

    /// Whether binding `index` of the workload is written by the kernel: the last one, in every workload but the sum,
    /// whose partials (binding 1) are written by the first dispatch and read by the second.
    fn writable(workload: Workload, index: usize) -> bool {
        index + 1 == Self::binding_count(workload) || (workload.family() == "W7" && index == 1)
    }

    fn binding_count(workload: Workload) -> usize {
        match workload.family() {
            "W1" | "W3" | "W5" | "W6" | "W7" => 3,
            "W2" | "W9" => 4,
            _ => 2,
        }
    }

    /// The entry points of the workload's module, in dispatch order.
    fn entry_points(workload: Workload, gust_kernel: bool, rust_gpu: bool) -> Vec<&'static str> {
        if gust_kernel {
            return vec![Self::gust_source(workload).1];
        }
        if rust_gpu {
            return Self::rust_gpu_entry_points(workload);
        }
        match workload.family() {
            "W6" => vec!["center", "amplify", "shift", "rescale", "energy", "combine"],
            "W7" => vec!["reduce_blocks", "reduce_total"],
            _ => vec!["main"],
        }
    }

    /// Whether the rust-gpu module has a kernel for the workload (W1 to W4).
    #[cfg(feature = "rust-gpu")]
    fn rust_gpu_supports(workload: Workload) -> bool {
        crate::rustgpu::entry_point(workload).is_some()
    }

    #[cfg(not(feature = "rust-gpu"))]
    fn rust_gpu_supports(_workload: Workload) -> bool {
        false
    }

    /// The rust-gpu module of the `rustgpu-wgpu` backend, created through wgpu's passthrough route on the context's
    /// device (which `WgpuContext::open_passthrough_on` opened with `PASSTHROUGH_SHADERS`).
    #[cfg(feature = "rust-gpu")]
    fn rust_gpu_module(ctx: &WgpuContext) -> wgpu::ShaderModule {
        crate::rustgpu::create_module(&ctx.device)
    }

    #[cfg(not(feature = "rust-gpu"))]
    fn rust_gpu_module(_ctx: &WgpuContext) -> wgpu::ShaderModule {
        unreachable!("the rustgpu-wgpu backend needs a build with the rust-gpu feature")
    }

    /// The one entry point of the rust-gpu module that runs `workload` (empty for a workload the module has no kernel for).
    #[cfg(feature = "rust-gpu")]
    fn rust_gpu_entry_points(workload: Workload) -> Vec<&'static str> {
        crate::rustgpu::entry_point(workload).into_iter().collect()
    }

    #[cfg(not(feature = "rust-gpu"))]
    fn rust_gpu_entry_points(_workload: Workload) -> Vec<&'static str> {
        unreachable!("the rustgpu-wgpu backend needs a build with the rust-gpu feature")
    }

    /// The per-sample host inputs an end-to-end sample uploads, as bytes, in the order of [`Resident::uploads`].
    fn input_bytes(workload: Workload, inputs: &Inputs) -> Vec<&[u8]> {
        match workload.family() {
            // W5: the particles (7 words each) and their accelerations (2 words each), the layout `particles.wgsl` declares.
            "W1" | "W2" | "W5" => vec![cast_slice(&inputs.x), cast_slice(&inputs.y)],
            "W3" => vec![cast_slice(&inputs.x)],
            // W6: signal, baseline, gain, bias, scale: the five input buffers, in the order `prepare_pipeline` creates them.
            "W6" => inputs
                .pipeline
                .iter()
                .map(|s| cast_slice(s.as_slice()))
                .collect(),
            "W7" if workload == Workload::SumU32 => vec![cast_slice(&inputs.values)],
            "W7" => vec![cast_slice(&inputs.x)],
            "W9" => vec![cast_slice(&inputs.x), cast_slice(&inputs.y)],
            _ => vec![cast_slice(&inputs.keys)],
        }
    }

    /// Workgroups of the first W7 dispatch (256 lanes each, a grid-stride loop over the input): one per 2048 values,
    /// capped by the device's per-dimension workgroup limit (a standard grid-stride reduction whose grid grows with `n`).
    fn sum_blocks(ctx: &WgpuContext, n: usize) -> u32 {
        workloads::sum::hand_blocks(n, ctx.device.limits().max_compute_workgroups_per_dimension)
    }

    /// The dispatches of one pass: one per element (or per [`workloads::KEYS_PER_THREAD`] keys) of 256-wide workgroups,
    /// the two reduction dispatches of W7, or the 2-D grid of 16 by 16 workgroups of W9.
    fn steps(workload: Workload, n: usize, blocks: u32) -> Vec<(usize, [u32; 3])> {
        match workload.family() {
            "W7" => vec![(0, [blocks, 1, 1]), (1, [1, 1, 1])],
            "W9" => {
                let side = n.div_ceil(workloads::matmul::TILE) as u32;
                vec![(0, [side, side, 1])]
            }
            // W6: six dispatches, one per stage and pipeline, the same workgroup count each (one invocation per element).
            "W6" => (0..workloads::signal_pipeline::STAGES)
                .map(|stage| (stage, [n.div_ceil(WORKGROUP) as u32, 1, 1]))
                .collect(),
            _ => {
                let threads = if workload.bins().is_some() {
                    hist_threads(n)
                } else {
                    n
                };
                vec![(0, [threads.div_ceil(WORKGROUP) as u32, 1, 1])]
            }
        }
    }

    /// The same objects `gust-wgpu` creates for a kernel: bind group layout, pipeline layout, shader module,
    /// compute pipeline (no driver pipeline cache).
    fn create_pipeline(
        ctx: &WgpuContext,
        workload: Workload,
        gust_kernel: bool,
        unchecked: bool,
        rust_gpu: bool,
    ) -> Pipeline {
        let entries = (0..Self::binding_count(workload))
            .map(|index| wgpu::BindGroupLayoutEntry {
                binding: index as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: !Self::writable(workload, index),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bench layout"),
                entries: &entries,
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bench pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let module = if rust_gpu {
            Self::rust_gpu_module(ctx)
        } else {
            let text = if gust_kernel {
                Self::gust_source(workload).0
            } else {
                Self::source(workload)
            };
            let descriptor = wgpu::ShaderModuleDescriptor {
                label: Some("bench hand WGSL"),
                source: wgpu::ShaderSource::Wgsl(text.into()),
            };
            if unchecked {
                // SAFETY: this builds `wgsl-prebuilt-unchecked` and `wgsl-percall-unchecked`, the reference and the
                // control of the `spirv` group (same kernel text, only the host side differs). The kernels do not rely on
                // wgpu's runtime checks: every kernel guards its indices against `arrayLength` of its own bindings
                // (bench/shaders/*.wgsl), matmul guards against the `dims` the host writes, which match the host-sized
                // buffers, and every loop is bounded by `n`, the degree or a constant. Every sample of both rows is
                // verified against the serial reference.
                unsafe {
                    ctx.device.create_shader_module_trusted(
                        descriptor,
                        wgpu::ShaderRuntimeChecks::unchecked(),
                    )
                }
            } else {
                ctx.device.create_shader_module(descriptor)
            }
        };
        let pipelines = Self::entry_points(workload, gust_kernel, rust_gpu)
            .into_iter()
            .map(|entry| {
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("bench pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .collect();
        Pipeline { pipelines, layout }
    }

    fn create_bind_group(
        ctx: &WgpuContext,
        pipeline: &Pipeline,
        buffers: &[wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench bind group"),
            layout: &pipeline.layout,
            entries: &entries,
        })
    }

    /// W6: the bind group of one stage over its three buffers (`stage_bindings`), on the one three-binding layout.
    fn create_stage_group(
        ctx: &WgpuContext,
        pipeline: &Pipeline,
        buffers: &[wgpu::Buffer],
        bindings: [usize; 3],
    ) -> wgpu::BindGroup {
        let entries = bindings
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffers[*buffer].as_entire_binding(),
            })
            .collect::<Vec<_>>();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench stage bind group"),
            layout: &pipeline.layout,
            entries: &entries,
        })
    }

    /// W6: the index of the first stage output among the buffers (the five inputs come before it).
    const STAGE_FIRST_OUTPUT: usize = 5;

    /// W6: the buffers of the pipeline in creation order (`signal`, `baseline`, `gain`, `bias`, `scale`, then the
    /// outputs of stages 0 to 5, the last being the final output) and, per stage, the three buffers it binds.
    const STAGE_BINDINGS: [[usize; 3]; 6] = [
        [0, 1, 5],
        [5, 2, 6],
        [6, 3, 7],
        [7, 4, 8],
        [8, 8, 9],
        [9, 8, 10],
    ];

    /// W6: records `batch` runs of the six stages, each stage a compute pass of its own in the order written (wgpu
    /// orders the storage writes between passes). The timestamps bracket the first pass of the first run to the last
    /// pass of the last run.
    fn record_staged(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        r: &Resident,
        batch: usize,
        timestamps: bool,
    ) {
        let stages = r.steps.len();
        let passes = batch * stages;
        for index in 0..passes {
            let stage = index % stages;
            let per_call;
            let bind_group = if self.switches.bind_group_reuse {
                &r.stage_groups[stage]
            } else {
                bump(&self.counters.bind_groups);
                per_call = Self::create_stage_group(
                    &self.ctx,
                    &r.pipeline,
                    &r.buffers,
                    r.stage_bindings[stage],
                );
                &per_call
            };
            let timestamp_writes = (timestamps && (index == 0 || index == passes - 1))
                .then_some(r.query_set.as_ref())
                .flatten()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: (index == 0).then_some(0),
                    end_of_pass_write_index: (index == passes - 1).then_some(1),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bench stage pass"),
                timestamp_writes,
            });
            pass.set_bind_group(0, bind_group, &[]);
            let (pipeline, [x, y, z]) = &r.steps[stage];
            pass.set_pipeline(&r.pipeline.pipelines[*pipeline]);
            pass.dispatch_workgroups(*x, *y, *z);
        }
    }

    /// W6's `prepare`: the five input buffers (written once), the six stage outputs, one module with the six entry
    /// points, and one bind group per stage.
    fn prepare_pipeline(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        // `supports` leaves W6 out of the probes; `profile` does not filter by it, so this makes a probe asked for W6 loud
        // instead of running the hand shader under the probe's id.
        assert!(
            !self.gust_shader && !self.gust_order,
            "W6 is not run by the GUST-shader or device-order probes"
        );
        let ctx = &self.ctx;
        let n = inputs.n;
        let bytes = (n * size_of::<f32>()) as u64;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut buffers = Vec::new();
        let mut uploads = Vec::new();
        for (label, values) in [
            "bench signal",
            "bench baseline",
            "bench gain",
            "bench bias",
            "bench scale",
        ]
        .into_iter()
        .zip(Self::input_bytes(workload, inputs))
        {
            let buffer = ctx.buffer(label, values.len() as u64, storage);
            ctx.queue.write_buffer(&buffer, 0, values);
            uploads.push(buffers.len());
            buffers.push(buffer);
        }
        for label in [
            "bench centered",
            "bench amplified",
            "bench biased",
            "bench normalized",
            "bench energy",
        ] {
            buffers.push(ctx.buffer(label, bytes, storage | wgpu::BufferUsages::COPY_SRC));
        }
        buffers.push(ctx.buffer("bench out", bytes, storage | wgpu::BufferUsages::COPY_SRC));
        let pipeline = Self::create_pipeline(ctx, workload, false, self.unchecked, false);
        let stage_bindings = Self::STAGE_BINDINGS.to_vec();
        let stage_groups = stage_bindings
            .iter()
            .map(|bindings| Self::create_stage_group(ctx, &pipeline, &buffers, *bindings))
            .collect::<Vec<_>>();
        let staging = ctx.buffer(
            "bench staging",
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let (query_set, resolve, timestamp_read) = Self::timestamp_objects(ctx);
        ctx.flush()?;
        self.passes.set(0);
        self.resident = Some(Resident {
            workload,
            out_len: n,
            bytes,
            buffers,
            uploads,
            staging,
            pipeline,
            bind_group: stage_groups[0].clone(),
            query_set,
            resolve,
            timestamp_read,
            poison: vec![workload.poison_word(); n],
            steps: Self::steps(workload, n, 0),
            blocks: 0,
            stage_bindings,
            stage_groups,
        });
        Ok(())
    }

    /// Records `batch` passes. A pass with `timestamps` gets the query writes: the first pass writes query 0 at its
    /// beginning and the last pass writes query 1 at its end, as `dispatch_batch_timed` does.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        r: &Resident,
        batch: usize,
        timestamps: bool,
    ) {
        self.record_with(encoder, r, batch, timestamps, None);
    }

    /// [`Self::record`], and with `spans` (W7 only) every pass writes its own begin and end query, `2 * index` and
    /// `2 * index + 1`, so a batch reports the per-repeat spans (first pass begin to last pass end of each repeat).
    fn record_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        r: &Resident,
        batch: usize,
        timestamps: bool,
        spans: Option<&wgpu::QuerySet>,
    ) {
        if r.workload.is_pipeline() {
            return self.record_staged(encoder, r, batch, timestamps);
        }
        for index in 0..batch {
            let per_call;
            let bind_group = if self.switches.bind_group_reuse {
                &r.bind_group
            } else {
                bump(&self.counters.bind_groups);
                per_call = Self::create_bind_group(&self.ctx, &r.pipeline, &r.buffers);
                &per_call
            };
            let timestamp_writes = match spans {
                Some(query_set) => Some(wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2 * index as u32),
                    end_of_pass_write_index: Some(2 * index as u32 + 1),
                }),
                None => (timestamps && (index == 0 || index == batch - 1))
                    .then_some(r.query_set.as_ref())
                    .flatten()
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: (index == 0).then_some(0),
                        end_of_pass_write_index: (index == batch - 1).then_some(1),
                    }),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bench pass"),
                timestamp_writes,
            });
            pass.set_bind_group(0, bind_group, &[]);
            for (pipeline, [x, y, z]) in &r.steps {
                pass.set_pipeline(&r.pipeline.pipelines[*pipeline]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
        }
    }

    /// The query set and the two 16-byte buffers of the timestamp span.
    fn timestamp_objects(
        ctx: &WgpuContext,
    ) -> (Option<wgpu::QuerySet>, wgpu::Buffer, wgpu::Buffer) {
        let query_set = ctx.timestamps.then(|| {
            ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("bench timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let resolve = ctx.buffer(
            "bench timestamp resolve",
            16,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        let timestamp_read = ctx.buffer(
            "bench timestamp read",
            16,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        (query_set, resolve, timestamp_read)
    }

    /// W7 GPU time: every one of the `batch` repeats is a pass with its own begin and end timestamp (a query set of
    /// `2 * batch`, still one submission), and the result is the sum of the per-repeat spans, which is what GUST's W7
    /// batch figure is (`gust_algo::timed` sums per-sum spans): the transitions between repeats are outside it on both sides.
    /// Only W7 does this; every other workload's batch is one pair around all its passes.
    fn gpu_time_per_repeat(&self, batch: usize) -> BenchResult<Option<Duration>> {
        check_span_batch(batch)?;
        let r = self.resident();
        let ctx = &self.ctx;
        let queries = 2 * batch as u32;
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bench W7 span timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let bytes = 8 * u64::from(queries);
        let resolve = ctx.buffer(
            "bench W7 span resolve",
            bytes,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        let read = ctx.buffer(
            "bench W7 span read",
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut encoder = self.encoder();
        self.record_with(&mut encoder, r, batch, true, Some(&query_set));
        encoder.resolve_query_set(&query_set, 0..queries, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
        let submission = self.submit(encoder.finish());
        self.passes
            .set(self.passes.get().wrapping_add(batch as u32));
        ctx.wait(submission)?;
        let ticks = ctx.map_read(&read, |view| {
            view.as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_ne_bytes(*chunk))
                .collect::<Vec<_>>()
        })?;
        let (total, spans) = span_sum(&ticks, ctx.queue.get_timestamp_period());
        self.last_spans.set(spans);
        Ok(Some(total))
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench encoder"),
            })
    }

    fn out_buffer(r: &Resident) -> &wgpu::Buffer {
        r.buffers.last().expect("output buffer")
    }
}

impl Backend for HandWgsl {
    fn id(&self) -> &'static str {
        match (self.mode, self.ablation) {
            _ if self.rust_gpu => "rustgpu-wgpu",
            _ if self.all_on => "wgsl-percall+all-on",
            _ if self.probe_id.is_some() => self.probe_id.unwrap_or("wgsl-prebuilt"),
            (Mode::Prebuilt, _) if self.unchecked => "wgsl-prebuilt-unchecked",
            (Mode::PerCall, _) if self.unchecked => "wgsl-percall-unchecked",
            (Mode::Prebuilt, _) => "wgsl-prebuilt",
            (Mode::PerCall, None) => "wgsl-percall",
            (Mode::PerCall, Some(switch)) => switch.backend_id(),
        }
    }

    fn description(&self) -> &'static str {
        match (self.mode, self.ablation) {
            _ if self.rust_gpu => {
                "rust-gpu 0.11.0 kernels (nightly toolchain, committed SPIR-V module) on plain wgpu through the passthrough route; the host side of wgsl-prebuilt; W1 to W4 only"
            }
            (Mode::Prebuilt, _) if self.unchecked => {
                "wgsl-prebuilt with wgpu's shader runtime checks (bounds checks, loop bounding) off, hand WGSL: the reference of the spirv group (D193)"
            }
            (Mode::PerCall, _) if self.unchecked => {
                "wgsl-percall with wgpu's shader runtime checks (bounds checks, loop bounding) off, hand WGSL: the control of the spirv group (D193)"
            }
            _ if self.probe_id.is_some() => {
                "wgsl-prebuilt plus what the wgpu trace shows differing from GUST (m10.b1c probe): the device's requested features (gust-setup and gust-device), the timestamp objects created before the data buffers (the same two), GUST's generated WGSL as the kernel (gust-setup and gust-shader)"
            }
            _ if self.all_on => {
                "wgsl-percall with all three switches on: bind group and staging buffer built once, one submission; the result Vec is still fresh per call"
            }
            (Mode::PerCall, Some(Switch::BindGroupReuse)) => {
                "wgsl-percall with one switch on: the bind group is built once, everything else per call"
            }
            (Mode::PerCall, Some(Switch::StagingReuse)) => {
                "wgsl-percall with one switch on: the read-back staging buffer is built once, everything else per call"
            }
            (Mode::PerCall, Some(Switch::OneSubmission)) => {
                "wgsl-percall with one switch on: dispatch and copy go in one submission, everything else per call"
            }
            (Mode::Prebuilt, _) => {
                "hand WGSL on plain wgpu; bind group and staging buffer built once, one submission per end-to-end sample"
            }
            (Mode::PerCall, None) => {
                "hand WGSL on plain wgpu shaped like gust-wgpu: bind group, staging buffer and Vec per call, two submissions"
            }
        }
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    /// The worst-case error of the hand tree (`workloads::sum::hand_tree_bound`): its order is known, so it is judged by
    /// its own depth with `c = 2^-23`, not by the serial worst case. Only for the `f32` sum, only after `prepare`.
    fn sum_tree_bound(&self, n: usize, abs_sum: f64) -> Option<f64> {
        let r = self.resident.as_ref()?;
        (r.workload == Workload::SumF32)
            .then(|| workloads::sum::hand_tree_bound(n, r.blocks, abs_sum))
    }

    /// The GUST-shader probes run GUST's generated WGSL, which this crate embeds as precompiled artifacts for W1 to W4;
    /// the probe has no W7 (the library's own kernels), no W5 and no W9 arm (their artifacts exist in `gust-bench-pre` for the
    /// pre-built GUST backend, but the probe path does not take it).
    ///
    /// W6 is left out of every probe: the artifacts for its six kernels exist (in `gust-bench-pre`, as for W5 and W9)
    /// but the shader probe path does not take them, and the device-order probe's resource order (timestamp objects
    /// first) is not built by the staged `prepare`. Both the `gust-shader` and the `gust-device` and `gust-setup`
    /// (`gust_order`) probes skip it.
    fn supports(&self, workload: Workload) -> bool {
        (!self.rust_gpu || Self::rust_gpu_supports(workload))
            && !(self.gust_shader && matches!(workload.family(), "W5" | "W6" | "W7" | "W9"))
            && !(self.gust_order && workload.is_pipeline())
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        if workload.is_pipeline() {
            return self.prepare_pipeline(workload, inputs);
        }
        let ctx = &self.ctx;
        let n = inputs.n;
        let out_len = workload.out_len(n);
        let bytes = (out_len * size_of::<f32>()) as u64;
        // `gust-setup`: the query set and the two timestamp buffers come before the data buffers, as in
        // `HeadlessDevice::open` (the wgpu trace of GUST shows that order).
        let early = self.gust_order.then(|| Self::timestamp_objects(ctx));
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut buffers = Vec::new();
        // A one-element or short read-only buffer that is not re-uploaded per sample: the scalar, the coefficients.
        let dims = [n as u32];
        let fixed: Option<(&str, &[u8])> = match workload.family() {
            "W2" => Some((
                "bench scale",
                cast_slice(std::slice::from_ref(&inputs.scale)),
            )),
            "W3" => Some(("bench coeffs", cast_slice(&inputs.coeffs))),
            "W9" => Some(("bench dims", cast_slice(&dims))),
            _ => None,
        };
        if let Some((label, values)) = fixed {
            let buffer = ctx.buffer(label, values.len() as u64, storage);
            ctx.queue.write_buffer(&buffer, 0, values);
            buffers.push(buffer);
        }
        let mut uploads = Vec::new();
        for (label, values) in ["bench x", "bench y"]
            .into_iter()
            .zip(Self::input_bytes(workload, inputs))
        {
            let buffer = ctx.buffer(label, values.len() as u64, storage);
            ctx.queue.write_buffer(&buffer, 0, values);
            uploads.push(buffers.len());
            buffers.push(buffer);
        }
        let blocks = if workload.family() == "W7" {
            Self::sum_blocks(ctx, n)
        } else {
            0
        };
        if workload.family() == "W7" {
            // One partial sum per workgroup of the first dispatch, read by the second.
            buffers.push(ctx.buffer(
                "bench partials",
                u64::from(blocks) * size_of::<f32>() as u64,
                storage,
            ));
        }
        buffers.push(ctx.buffer("bench out", bytes, storage | wgpu::BufferUsages::COPY_SRC));
        let pipeline = Self::create_pipeline(
            ctx,
            workload,
            self.gust_shader,
            self.unchecked,
            self.rust_gpu,
        );
        let bind_group = Self::create_bind_group(ctx, &pipeline, &buffers);
        let staging = ctx.buffer(
            "bench staging",
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let (query_set, resolve, timestamp_read) =
            early.unwrap_or_else(|| Self::timestamp_objects(ctx));
        ctx.flush()?;
        self.passes.set(0);
        self.resident = Some(Resident {
            workload,
            out_len,
            bytes,
            buffers,
            uploads,
            staging,
            pipeline,
            bind_group,
            query_set,
            resolve,
            timestamp_read,
            poison: vec![workload.poison_word(); out_len],
            steps: Self::steps(workload, n, blocks),
            blocks,
            stage_bindings: Vec::new(),
            stage_groups: Vec::new(),
        });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        if r.workload.is_pipeline() {
            // W6: every stage output (buffers 5 to 10; 0 to 4 are the inputs), as GUST poisons all six, so a pass that
            // stopped writing an intermediate cannot read the previous sample's correct value and still pass.
            for buffer in &r.buffers[Self::STAGE_FIRST_OUTPUT..] {
                self.ctx
                    .queue
                    .write_buffer(buffer, 0, cast_slice(&r.poison));
            }
        } else {
            self.ctx
                .queue
                .write_buffer(Self::out_buffer(r), 0, cast_slice(&r.poison));
        }
        self.passes.set(0);
        self.ctx.flush()
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        let ctx = &self.ctx;
        for (index, values) in r.uploads.iter().zip(Self::input_bytes(r.workload, inputs)) {
            ctx.queue.write_buffer(&r.buffers[*index], 0, values);
        }
        if r.workload.accumulates() {
            // A histogram adds into its bins: a host call starts from clean bins, so the clear is part of the sample.
            ctx.queue
                .write_buffer(Self::out_buffer(r), 0, cast_slice(&r.poison));
        }
        self.passes.set(1);
        let mut encoder = self.encoder();
        self.record(&mut encoder, r, 1, false);
        if !self.switches.one_submission {
            // The dispatch has its own submission and a wait; the copy goes in a second encoder below.
            let submission = self.submit(encoder.finish());
            ctx.wait(submission)?;
            encoder = self.encoder();
        }
        // The staging buffer: the resident one, or one created at the point of use (after the first wait when the
        // dispatch has its own submission, as `gust-wgpu` does).
        let fresh;
        let staging = if self.switches.staging_reuse {
            &r.staging
        } else {
            fresh = self.fresh_staging("bench per-call staging", r.bytes);
            &fresh
        };
        encoder.copy_buffer_to_buffer(Self::out_buffer(r), 0, staging, 0, r.bytes);
        self.submit(encoder.finish());
        match self.mode {
            Mode::Prebuilt => ctx.map_read(staging, |bytes| {
                out.resize(r.out_len, 0.0);
                out.copy_from_slice(cast_slice(bytes));
            })?,
            Mode::PerCall => {
                *out = ctx.map_read(staging, |bytes| cast_slice::<u8, f32>(bytes).to_vec())?;
            }
        }
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        let r = self.resident();
        let mut encoder = self.encoder();
        self.record(&mut encoder, r, 1, false);
        let submission = self.submit(encoder.finish());
        self.passes.set(self.passes.get().wrapping_add(1));
        self.ctx.wait(submission)
    }

    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        let r = self.resident();
        let Some(query_set) = &r.query_set else {
            return Ok(None);
        };
        if r.workload.family() == "W7" {
            return self.gpu_time_per_repeat(batch);
        }
        let ctx = &self.ctx;
        let mut encoder = self.encoder();
        self.record(&mut encoder, r, batch, true);
        encoder.resolve_query_set(query_set, 0..2, &r.resolve, 0);
        encoder.copy_buffer_to_buffer(&r.resolve, 0, &r.timestamp_read, 0, 16);
        let submission = self.submit(encoder.finish());
        self.passes
            .set(self.passes.get().wrapping_add(batch as u32));
        let (sender, receiver) = mpsc::channel();
        r.timestamp_read
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
        ctx.wait(submission)?;
        receiver.recv()??;
        let view = r.timestamp_read.get_mapped_range(..)?;
        let ticks = view
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_ne_bytes(*chunk))
            .collect::<Vec<_>>();
        drop(view);
        r.timestamp_read.unmap();
        let nanoseconds =
            ticks[1].wrapping_sub(ticks[0]) as f64 * f64::from(ctx.queue.get_timestamp_period());
        Ok(Some(Duration::from_secs_f64(nanoseconds / 1_000_000_000.0)))
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        let mut encoder = self.encoder();
        encoder.copy_buffer_to_buffer(Self::out_buffer(r), 0, &r.staging, 0, r.bytes);
        self.submit(encoder.finish());
        self.ctx.map_read(&r.staging, |bytes| {
            out.resize(r.out_len, 0.0);
            out.copy_from_slice(cast_slice(bytes));
        })?;
        if r.workload.accumulates() {
            normalize_counts(out, self.passes.get());
        }
        Ok(())
    }

    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        let workload = self.resident().workload;
        let started = Instant::now();
        let pipeline = Self::create_pipeline(
            &self.ctx,
            workload,
            self.gust_shader,
            self.unchecked,
            self.rust_gpu,
        );
        let r = self.resident.as_mut().expect("prepare was called");
        if workload.is_pipeline() {
            r.stage_groups = r
                .stage_bindings
                .iter()
                .map(|bindings| {
                    Self::create_stage_group(&self.ctx, &pipeline, &r.buffers, *bindings)
                })
                .collect();
            r.bind_group = r.stage_groups[0].clone();
        } else {
            r.bind_group = Self::create_bind_group(&self.ctx, &pipeline, &r.buffers);
        }
        r.pipeline = pipeline;
        self.dispatch()?;
        Ok(Some(started.elapsed()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Judging, Rounding, check, check_with, gen_inputs, reference};

    const N: usize = 4096;
    /// The GPU tests take turns: creating several Vulkan instances at once made adapter discovery fail now and then.
    fn gpu_context() -> (std::sync::MutexGuard<'static, ()>, Rc<WgpuContext>) {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = WgpuContext::new().expect("an adapter is needed for the switch test");
        (guard, Rc::new(ctx))
    }

    /// Runs one end-to-end sample and one 16-pass batch on a fresh backend and returns the counts of each, after
    /// checking the output (an ablation must not change what is computed).
    fn counts_of(mut backend: HandWgsl) -> (Counts, Counts) {
        let inputs = gen_inputs(Workload::VectorAdd, N);
        let want = reference(Workload::VectorAdd, &inputs);
        backend.prepare(Workload::VectorAdd, &inputs).unwrap();
        assert_eq!(
            backend.counts(),
            Counts::default(),
            "prepare counts nothing"
        );
        backend.poison().unwrap();
        let mut out = Vec::new();
        backend.end_to_end(&inputs, &mut out).unwrap();
        check(Workload::VectorAdd, &inputs, &want, &out).unwrap();
        let end_to_end = backend.counts();
        backend.gpu_time(16).unwrap();
        let after = backend.counts();
        let batch = Counts {
            bind_groups: after.bind_groups - end_to_end.bind_groups,
            staging_buffers: after.staging_buffers - end_to_end.staging_buffers,
            submissions: after.submissions - end_to_end.submissions,
        };
        (end_to_end, batch)
    }

    fn counts(bind_groups: u64, staging_buffers: u64, submissions: u64) -> Counts {
        Counts {
            bind_groups,
            staging_buffers,
            submissions,
        }
    }

    /// Every workload on the hand-written WGSL, in both host shapes: end to end, a resident dispatch and a 16-pass
    /// batch match the reference, and the poison alone fails the check.
    #[test]
    fn every_workload_passes_its_bound_on_hand_wgsl_in_both_modes() {
        let (_gpu, ctx) = gpu_context();
        for mode in [Mode::Prebuilt, Mode::PerCall] {
            for w in Workload::ALL {
                let inputs = gen_inputs(w, workloads::test_size(w));
                let reference = reference(w, &inputs);
                let mut backend = HandWgsl::new(Rc::clone(&ctx), mode);
                backend.prepare(w, &inputs).unwrap();
                // A GPU's rounding, and for the f32 sum the hand tree's own bound.
                let judging = Judging {
                    rounding: Rounding::Either,
                    tree: backend.sum_tree_bound(inputs.n, reference.abs_sum),
                };
                assert_eq!(judging.tree.is_some(), w == Workload::SumF32);
                let mut out = Vec::new();
                backend.poison().unwrap();
                backend.read_back(&mut out).unwrap();
                assert!(
                    check_with(w, &inputs, &reference, &out, judging).is_err(),
                    "{} poison",
                    w.id()
                );
                backend.poison().unwrap();
                backend.end_to_end(&inputs, &mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} {mode:?} end to end: {e}", w.id()));
                backend.poison().unwrap();
                backend.dispatch().unwrap();
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} {mode:?} dispatch: {e}", w.id()));
                backend.poison().unwrap();
                if backend.gpu_time(16).unwrap().is_some() {
                    backend.read_back(&mut out).unwrap();
                    check_with(w, &inputs, &reference, &out, judging)
                        .unwrap_or_else(|e| panic!("{} {mode:?} batch: {e}", w.id()));
                }
            }
        }
    }

    /// W7's batch figure is the sum of per-repeat spans (a pair of timestamps per repeat, one submission), the same
    /// shape as GUST's `gust_algo::timed`: `batch` repeats read `batch` pairs, and the pure sum adds the per-pair spans
    /// (and not the gaps between pairs).
    #[test]
    fn a_w7_batch_reads_one_timestamp_pair_per_repeat() {
        assert!(check_span_batch(2048).is_ok());
        let refusal = check_span_batch(2049).unwrap_err().to_string();
        assert!(
            refusal.contains("4096") && refusal.contains("2049"),
            "{refusal}"
        );
        let (total, pairs) = span_sum(&[10, 14, 100, 103, 1_000, 1_005], 2.0);
        assert_eq!(pairs, 3);
        assert_eq!(total, Duration::from_nanos(2 * (4 + 3 + 5)));
        assert_eq!(span_sum(&[], 1.0), (Duration::ZERO, 0));
        let (_gpu, ctx) = gpu_context();
        for mode in [Mode::Prebuilt, Mode::PerCall] {
            for w in [Workload::SumU32, Workload::SumF32] {
                let inputs = gen_inputs(w, workloads::test_size(w));
                let mut backend = HandWgsl::new(Rc::clone(&ctx), mode);
                backend.prepare(w, &inputs).unwrap();
                for batch in [1usize, 5, 16] {
                    let time = backend.gpu_time(batch).unwrap();
                    assert_eq!(time.is_some(), ctx.timestamps, "{}", w.id());
                    if time.is_some() {
                        assert_eq!(backend.last_spans.get(), batch, "{} {mode:?}", w.id());
                    }
                }
            }
        }
    }

    /// W6: the timed path reads only the final stage, so this test reads every stage's buffer after one run and checks each
    /// against its own bound (`signal_pipeline::stage_exact_and_bound`, the GPU bound), and checks the six steps.
    #[test]
    fn every_hand_stage_is_inside_its_own_bound_on_the_gpu() {
        use crate::workloads::signal_pipeline::{STAGES, stage_exact_and_bound};
        let (_gpu, ctx) = gpu_context();
        for mode in [Mode::Prebuilt, Mode::PerCall] {
            let w = Workload::PipelineHand;
            let inputs = gen_inputs(w, workloads::test_size(w));
            let mut backend = HandWgsl::new(Rc::clone(&ctx), mode);
            backend.prepare(w, &inputs).unwrap();
            backend.poison().unwrap();
            backend.dispatch().unwrap();
            let r = backend.resident();
            assert_eq!(r.steps.len(), STAGES);
            for stage in 0..STAGES {
                let source = &r.buffers[5 + stage];
                let staging = ctx.buffer(
                    "test stage staging",
                    r.bytes,
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                );
                let mut encoder = backend.encoder();
                encoder.copy_buffer_to_buffer(source, 0, &staging, 0, r.bytes);
                backend.submit(encoder.finish());
                let got = ctx
                    .map_read(&staging, |bytes| cast_slice::<u8, f32>(bytes).to_vec())
                    .unwrap();
                let (exact, _, bound_gpu) = stage_exact_and_bound(&inputs.pipeline, stage);
                for (i, ((got, exact), bound)) in got.iter().zip(&exact).zip(&bound_gpu).enumerate()
                {
                    let err = (f64::from(*got) - exact).abs();
                    assert!(
                        err <= *bound,
                        "{mode:?} stage {stage} element {i}: {err} against {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn each_switch_changes_exactly_what_it_claims() {
        let (_gpu, ctx) = gpu_context();
        // Defaults are today's behaviour: PerCall has every switch off, Prebuilt every switch on.
        assert_eq!(
            HandWgsl::new(Rc::clone(&ctx), Mode::PerCall).switches(),
            Switches::PER_CALL
        );
        assert_eq!(
            HandWgsl::new(Rc::clone(&ctx), Mode::Prebuilt).switches(),
            Switches::PREBUILT
        );
        assert_eq!(
            Switches::PER_CALL,
            Switches {
                bind_group_reuse: false,
                staging_reuse: false,
                one_submission: false
            }
        );

        let per_call = counts_of(HandWgsl::new(Rc::clone(&ctx), Mode::PerCall));
        // One pass: a bind group, a staging buffer, two submissions. A batch of 16: sixteen bind groups, one submission.
        assert_eq!(per_call, (counts(1, 1, 2), counts(16, 0, 1)));

        let prebuilt = counts_of(HandWgsl::new(Rc::clone(&ctx), Mode::Prebuilt));
        assert_eq!(prebuilt, (counts(0, 0, 1), counts(0, 0, 1)));

        let bind_group = counts_of(HandWgsl::ablation(Rc::clone(&ctx), Switch::BindGroupReuse));
        assert_eq!(bind_group, (counts(0, 1, 2), counts(0, 0, 1)));

        let staging = counts_of(HandWgsl::ablation(Rc::clone(&ctx), Switch::StagingReuse));
        assert_eq!(staging, (counts(1, 0, 2), counts(16, 0, 1)));

        let one_submission = counts_of(HandWgsl::ablation(Rc::clone(&ctx), Switch::OneSubmission));
        assert_eq!(one_submission, (counts(1, 1, 1), counts(16, 0, 1)));
    }

    #[test]
    fn all_on_has_every_switch_on_and_a_fresh_vec_like_percall() {
        let (_gpu, ctx) = gpu_context();
        let backend = HandWgsl::all_on(Rc::clone(&ctx));
        assert_eq!(backend.switches(), Switches::PREBUILT);
        assert_eq!(backend.id(), "wgsl-percall+all-on");
        assert_eq!(backend.mode, Mode::PerCall, "the result Vec stays per call");
        assert_eq!(
            counts_of(backend),
            (counts(0, 0, 1), counts(0, 0, 1)),
            "same host calls as wgsl-prebuilt"
        );
    }

    #[test]
    fn ablation_backends_flip_one_switch_and_have_their_own_ids() {
        let (_gpu, ctx) = gpu_context();
        let mut ids = Vec::new();
        for switch in Switch::ALL {
            let backend = HandWgsl::ablation(Rc::clone(&ctx), switch);
            let s = backend.switches();
            let on = [s.bind_group_reuse, s.staging_reuse, s.one_submission];
            assert_eq!(on.iter().filter(|b| **b).count(), 1, "{}", switch.id());
            assert_eq!(s, Switches::PER_CALL.with(switch));
            assert_eq!(backend.id(), format!("wgsl-percall+{}", switch.id()));
            assert_eq!(Switch::parse(switch.id()), Some(switch));
            ids.push(backend.id());
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3);
        assert_eq!(Switch::parse("nope"), None);
    }

    /// The two hand rows of the `spirv` group (D193): first-class ids without a `+`, the host shape of their checked
    /// twin (percall keeps its per-call switches, prebuilt its reuse).
    #[test]
    fn the_unchecked_rows_have_their_own_ids_and_the_host_shape_of_their_twin() {
        let (_gpu, ctx) = gpu_context();
        let prebuilt = HandWgsl::unchecked(Rc::clone(&ctx), Mode::Prebuilt);
        let percall = HandWgsl::unchecked(Rc::clone(&ctx), Mode::PerCall);
        assert_eq!(prebuilt.id(), "wgsl-prebuilt-unchecked");
        assert_eq!(percall.id(), "wgsl-percall-unchecked");
        assert!(!prebuilt.id().contains('+') && !percall.id().contains('+'));
        assert_eq!(prebuilt.switches(), Switches::PREBUILT);
        assert_eq!(percall.switches(), Switches::PER_CALL);
        assert!(prebuilt.description().contains("reference"));
        assert!(percall.description().contains("control"));
    }
}
