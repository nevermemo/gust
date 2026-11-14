//! GUST on the GPU, through `gust-wgpu`'s public API only: typed buffers, the generated typed `call`, and
//! `dispatch_batch` / `dispatch_batch_timed` on a `HeadlessDevice`.

use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use gust::Independent;
use gust_wgpu::{BufferDispatch, GpuBuffer, GpuBufferAccess, HeadlessDevice};

use crate::{
    backend::{Backend, BenchResult, Kind},
    workloads::{Inputs, Workload, saxpy, vector_add},
};

struct Resident {
    workload: Workload,
    n: usize,
    x: GpuBuffer<f32>,
    y: GpuBuffer<f32>,
    scale: GpuBuffer<f32>,
    out: GpuBuffer<f32>,
    nan: Vec<f32>,
}

pub struct GustGpu {
    device: Rc<HeadlessDevice>,
    resident: Option<Resident>,
}

impl GustGpu {
    pub fn new(device: Rc<HeadlessDevice>) -> Self {
        Self {
            device,
            resident: None,
        }
    }

    fn resident(&self) -> &Resident {
        self.resident.as_ref().expect("prepare was called")
    }

    /// Builds the generated typed call for the resident buffers and hands the resulting dispatch to `run`. The call
    /// is the same value a user of `kernel::dispatch` builds on every invocation.
    fn with_dispatch<R>(&self, run: impl FnOnce(BufferDispatch<'_>) -> R) -> R {
        let r = self.resident();
        let n = r.n as u32;
        match r.workload {
            Workload::VectorAdd => {
                let call = vector_add::kernel::add::call(n, &r.x, &r.y, &r.out);
                run(BufferDispatch::from(&call))
            }
            Workload::Saxpy => {
                let call = saxpy::kernel::saxpy::call(n, Independent(&r.scale), &r.x, &r.y, &r.out);
                run(BufferDispatch::from(&call))
            }
        }
    }
}

impl Backend for GustGpu {
    fn id(&self) -> &'static str {
        "gust-gpu"
    }

    fn description(&self) -> &'static str {
        "GUST kernel through gust-wgpu (Slang -> WGSL -> wgpu), typed call, public API only"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        let d = &self.device;
        let n = inputs.x.len();
        let nan = vec![f32::NAN; n];
        let resident = Resident {
            workload,
            n,
            x: d.create_f32_buffer("bench x", &inputs.x, GpuBufferAccess::ReadOnly),
            y: d.create_f32_buffer("bench y", &inputs.y, GpuBufferAccess::ReadOnly),
            scale: d.create_f32_buffer("bench scale", &[inputs.scale], GpuBufferAccess::ReadOnly),
            out: d.create_f32_buffer("bench out", &nan, GpuBufferAccess::ReadWrite),
            nan,
        };
        // Flush the initial uploads so no later timed submission carries them.
        d.write_f32_buffer_timed(&resident.x, &inputs.x)?;
        d.write_f32_buffer_timed(&resident.y, &inputs.y)?;
        self.resident = Some(resident);
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        // The timed variant waits for the upload, so the copy cannot land inside a timed submission.
        self.device.write_f32_buffer_timed(&r.out, &r.nan)?;
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        self.device.write_f32_buffer(&r.x, &inputs.x)?;
        self.device.write_f32_buffer(&r.y, &inputs.y)?;
        self.with_dispatch(|dispatch| self.device.dispatch_batch(&[dispatch]))?;
        *out = self.device.read_f32_buffer(&r.out)?;
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        self.with_dispatch(|dispatch| self.device.dispatch_batch(&[dispatch]))?;
        Ok(())
    }

    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        let timing = self.with_dispatch(|dispatch| {
            let batch = vec![dispatch; batch];
            self.device.dispatch_batch_timed(&batch)
        })?;
        Ok(timing.gpu_elapsed)
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        *out = self.device.read_f32_buffer(&self.resident().out)?;
        Ok(())
    }

    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        self.device.clear_pipeline_cache();
        let started = Instant::now();
        self.dispatch()?;
        Ok(Some(started.elapsed()))
    }

    fn cache_misses(&self) -> Option<u64> {
        Some(self.device.pipeline_cache_stats().misses)
    }
}
