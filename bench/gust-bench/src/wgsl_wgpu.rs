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
//! so sharing one is not possible through the public API.

use std::{
    cell::Cell,
    rc::Rc,
    sync::mpsc,
    time::{Duration, Instant},
};

use bytemuck::cast_slice;

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, WORKGROUP, Workload},
};

/// A wgpu device and queue.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub timestamps: bool,
}

impl WgpuContext {
    pub fn new() -> BenchResult<Self> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bench hand-WGSL device"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            }))?;
        Ok(Self {
            device,
            queue,
            info: adapter.get_info(),
            timestamps,
        })
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
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

struct Resident {
    workload: Workload,
    n: usize,
    bytes: u64,
    /// In binding order: `[a, b, out]` for vector-add, `[scale, x, y, out]` for saxpy.
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    pipeline: Pipeline,
    bind_group: wgpu::BindGroup,
    query_set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    timestamp_read: wgpu::Buffer,
    nan: Vec<f32>,
    groups: u32,
}

pub struct HandWgsl {
    ctx: Rc<WgpuContext>,
    mode: Mode,
    switches: Switches,
    /// The one switch an ablation backend turned on, for its id and description.
    ablation: Option<Switch>,
    counters: Counters,
    resident: Option<Resident>,
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
            counters: Counters::default(),
            resident: None,
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

    fn source(workload: Workload) -> &'static str {
        match workload {
            Workload::VectorAdd => include_str!("../../shaders/vector_add.wgsl"),
            Workload::Saxpy => include_str!("../../shaders/saxpy.wgsl"),
        }
    }

    /// Whether binding `index` of the workload is written by the kernel.
    fn writable(workload: Workload, index: usize) -> bool {
        match workload {
            Workload::VectorAdd => index == 2,
            Workload::Saxpy => index == 3,
        }
    }

    fn binding_count(workload: Workload) -> usize {
        match workload {
            Workload::VectorAdd => 3,
            Workload::Saxpy => 4,
        }
    }

    /// The same objects `gust-wgpu` creates for a kernel: bind group layout, pipeline layout, shader module,
    /// compute pipeline (no driver pipeline cache).
    fn create_pipeline(ctx: &WgpuContext, workload: Workload) -> Pipeline {
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
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bench hand WGSL"),
                source: wgpu::ShaderSource::Wgsl(Self::source(workload).into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bench pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Pipeline { pipeline, layout }
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

    /// Records `batch` passes. A pass with `timestamps` gets the query writes: the first pass writes query 0 at its
    /// beginning and the last pass writes query 1 at its end, as `dispatch_batch_timed` does.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        r: &Resident,
        batch: usize,
        timestamps: bool,
    ) {
        for index in 0..batch {
            let per_call;
            let bind_group = if self.switches.bind_group_reuse {
                &r.bind_group
            } else {
                bump(&self.counters.bind_groups);
                per_call = Self::create_bind_group(&self.ctx, &r.pipeline, &r.buffers);
                &per_call
            };
            let timestamp_writes = (timestamps && (index == 0 || index == batch - 1))
                .then_some(r.query_set.as_ref())
                .flatten()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: (index == 0).then_some(0),
                    end_of_pass_write_index: (index == batch - 1).then_some(1),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bench pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&r.pipeline.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(r.groups, 1, 1);
        }
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
            (Mode::Prebuilt, _) => "wgsl-prebuilt",
            (Mode::PerCall, None) => "wgsl-percall",
            (Mode::PerCall, Some(switch)) => switch.backend_id(),
        }
    }

    fn description(&self) -> &'static str {
        match (self.mode, self.ablation) {
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

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        let ctx = &self.ctx;
        let n = inputs.x.len();
        let bytes = (n * size_of::<f32>()) as u64;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut buffers = Vec::new();
        if workload == Workload::Saxpy {
            let scale = ctx.buffer("bench scale", 4, storage);
            ctx.queue
                .write_buffer(&scale, 0, cast_slice(&[inputs.scale]));
            buffers.push(scale);
        }
        for (label, values) in [("bench x", &inputs.x), ("bench y", &inputs.y)] {
            let buffer = ctx.buffer(label, bytes, storage);
            ctx.queue.write_buffer(&buffer, 0, cast_slice(values));
            buffers.push(buffer);
        }
        buffers.push(ctx.buffer("bench out", bytes, storage | wgpu::BufferUsages::COPY_SRC));
        let pipeline = Self::create_pipeline(ctx, workload);
        let bind_group = Self::create_bind_group(ctx, &pipeline, &buffers);
        let staging = ctx.buffer(
            "bench staging",
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
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
        ctx.flush()?;
        self.resident = Some(Resident {
            workload,
            n,
            bytes,
            buffers,
            staging,
            pipeline,
            bind_group,
            query_set,
            resolve,
            timestamp_read,
            nan: vec![f32::NAN; n],
            groups: n.div_ceil(WORKGROUP) as u32,
        });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        self.ctx
            .queue
            .write_buffer(Self::out_buffer(r), 0, cast_slice(&r.nan));
        self.ctx.flush()
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        let ctx = &self.ctx;
        let x = r.buffers.len() - 3;
        ctx.queue
            .write_buffer(&r.buffers[x], 0, cast_slice(&inputs.x));
        ctx.queue
            .write_buffer(&r.buffers[x + 1], 0, cast_slice(&inputs.y));
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
                out.resize(r.n, 0.0);
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
        self.ctx.wait(submission)
    }

    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        let r = self.resident();
        let Some(query_set) = &r.query_set else {
            return Ok(None);
        };
        let ctx = &self.ctx;
        let mut encoder = self.encoder();
        self.record(&mut encoder, r, batch, true);
        encoder.resolve_query_set(query_set, 0..2, &r.resolve, 0);
        encoder.copy_buffer_to_buffer(&r.resolve, 0, &r.timestamp_read, 0, 16);
        let submission = self.submit(encoder.finish());
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
            out.resize(r.n, 0.0);
            out.copy_from_slice(cast_slice(bytes));
        })
    }

    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        let workload = self.resident().workload;
        let started = Instant::now();
        let pipeline = Self::create_pipeline(&self.ctx, workload);
        let r = self.resident.as_mut().expect("prepare was called");
        r.bind_group = Self::create_bind_group(&self.ctx, &pipeline, &r.buffers);
        r.pipeline = pipeline;
        self.dispatch()?;
        Ok(Some(started.elapsed()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{check, gen_inputs, reference};

    const N: usize = 4096;

    /// Runs one end-to-end sample and one 16-pass batch on a fresh backend and returns the counts of each, after
    /// checking the output (an ablation must not change what is computed).
    fn counts_of(mut backend: HandWgsl) -> (Counts, Counts) {
        let inputs = gen_inputs(N);
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

    #[test]
    fn each_switch_changes_exactly_what_it_claims() {
        let ctx = Rc::new(WgpuContext::new().expect("an adapter is needed for the switch test"));
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
    fn ablation_backends_flip_one_switch_and_have_their_own_ids() {
        let ctx = Rc::new(WgpuContext::new().expect("an adapter is needed for the switch test"));
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
}
