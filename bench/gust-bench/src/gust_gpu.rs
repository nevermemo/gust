//! GUST on the GPU, through `gust-wgpu`'s public API only: typed buffers, the generated typed `call`, and
//! `dispatch_batch` / `dispatch_batch_timed` on a `HeadlessDevice`.
//!
//! The route is a property of the build: a build with the `spirv-route` feature runs this backend on the SPIR-V route
//! (slangc's SPIR-V through passthrough) and names it `gust-gpu-spirv`; the default build is `gust-gpu` on WGSL through
//! naga. One process holds one GUST route (the feature changes the emitted kernels), so the two are separate builds and
//! separate result files, each with its own `wgsl-prebuilt` to pair against.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::{Duration, Instant},
};

use gust::Independent;
use gust_wgpu::{BufferDispatch, GpuBuffer, GpuBufferAccess, HeadlessDevice};
use serde_json::Value;

use crate::{
    backend::{Backend, BenchResult, Kind},
    route,
    workloads::{
        self, Inputs, Workload, counts_as_words, hist_threads, histogram, normalize_counts,
        polynomial, saxpy, vector_add,
    },
};

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
}

struct Resident {
    workload: Workload,
    n: usize,
    data: Data,
    /// The poison words of the f32 output (NaN), or empty for a histogram, whose poison is zeroed bins.
    nan: Vec<f32>,
    zeros: Vec<u32>,
}

pub struct GustGpu {
    device: Rc<HeadlessDevice>,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates; see `Backend::dispatch`).
    passes: Cell<u32>,
    route: RefCell<Option<Value>>,
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
    pub fn new(device: Rc<HeadlessDevice>) -> Self {
        Self {
            device,
            resident: None,
            passes: Cell::new(0),
            route: RefCell::new(None),
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
        }
    }

    fn read_words(&self) -> BenchResult<Vec<f32>> {
        let d = &self.device;
        Ok(match &self.resident().data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => {
                d.read_f32_buffer(out)?
            }
            Data::Hist { bins, .. } => counts_as_words(d.read_typed_buffer(bins)?),
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

    #[cfg(not(feature = "spirv-route"))]
    fn last_event(&self, _kernel: &str) -> Option<route::Event> {
        None
    }
}

impl Backend for GustGpu {
    fn id(&self) -> &'static str {
        backend_id()
    }

    fn description(&self) -> &'static str {
        if gust_wgpu::SPIRV_ROUTE {
            "GUST kernel through gust-wgpu on the opt-in SPIR-V route (slangc SPIR-V through passthrough), typed call, public API only"
        } else {
            "GUST kernel through gust-wgpu (Slang -> WGSL -> wgpu), typed call, public API only"
        }
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()> {
        self.resident = None;
        let d = &self.device;
        let n = inputs.n;
        let read_only = GpuBufferAccess::ReadOnly;
        let read_write = GpuBufferAccess::ReadWrite;
        let (data, nan, zeros) = match workload.family() {
            "W1" | "W2" => {
                let nan = vec![f32::NAN; n];
                let x = d.create_f32_buffer("bench x", &inputs.x, read_only);
                let y = d.create_f32_buffer("bench y", &inputs.y, read_only);
                let out = d.create_f32_buffer("bench out", &nan, read_write);
                // Flush the initial uploads so no later timed submission carries them.
                d.write_f32_buffer_timed(&x, &inputs.x)?;
                d.write_f32_buffer_timed(&y, &inputs.y)?;
                let data = if workload == Workload::VectorAdd {
                    Data::Add { x, y, out }
                } else {
                    let scale = d.create_f32_buffer("bench scale", &[inputs.scale], read_only);
                    Data::Saxpy { scale, x, y, out }
                };
                (data, nan, Vec::new())
            }
            "W3" => {
                let nan = vec![f32::NAN; n];
                let coeffs = d.create_f32_buffer("bench coeffs", &inputs.coeffs, read_only);
                let x = d.create_f32_buffer("bench x", &inputs.x, read_only);
                let out = d.create_f32_buffer("bench out", &nan, read_write);
                d.write_f32_buffer_timed(&x, &inputs.x)?;
                (Data::Poly { coeffs, x, out }, nan, Vec::new())
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
        });
        Ok(())
    }

    fn poison(&mut self) -> BenchResult<()> {
        let r = self.resident();
        // The timed variants wait for the upload, so the copy cannot land inside a timed submission.
        match &r.data {
            Data::Add { out, .. } | Data::Saxpy { out, .. } | Data::Poly { out, .. } => {
                self.device.write_f32_buffer_timed(out, &r.nan)?;
            }
            Data::Hist { bins, .. } => {
                self.device.write_typed_buffer_timed(bins, &r.zeros)?;
            }
        }
        self.passes.set(0);
        Ok(())
    }

    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()> {
        let r = self.resident();
        match &r.data {
            Data::Add { x, y, .. } | Data::Saxpy { x, y, .. } => {
                self.device.write_f32_buffer(x, &inputs.x)?;
                self.device.write_f32_buffer(y, &inputs.y)?;
            }
            Data::Poly { x, .. } => self.device.write_f32_buffer(x, &inputs.x)?,
            Data::Hist { keys, bins } => {
                self.device.write_typed_buffer(keys, &inputs.keys)?;
                self.device.write_typed_buffer(bins, &r.zeros)?;
            }
        }
        self.with_dispatch(|dispatch| self.device.dispatch_batch(&[dispatch]))?;
        self.passes.set(1);
        *out = self.read_words()?;
        Ok(())
    }

    fn dispatch(&mut self) -> BenchResult<()> {
        self.with_dispatch(|dispatch| self.device.dispatch_batch(&[dispatch]))?;
        self.passes.set(self.passes.get().wrapping_add(1));
        Ok(())
    }

    fn gpu_time(&mut self, batch: usize) -> BenchResult<Option<Duration>> {
        let timing = self.with_dispatch(|dispatch| {
            let batch = vec![dispatch; batch];
            self.device.dispatch_batch_timed(&batch)
        })?;
        self.passes
            .set(self.passes.get().wrapping_add(batch as u32));
        Ok(timing.gpu_elapsed)
    }

    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()> {
        *out = self.read_words()?;
        if self.resident().workload.kind() == workloads::Kind::U32 {
            normalize_counts(out, self.passes.get());
        }
        Ok(())
    }

    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        self.device.clear_pipeline_cache();
        let started = Instant::now();
        self.dispatch()?;
        let elapsed = started.elapsed();
        let descriptor = workloads::descriptor(self.resident().workload);
        let kernel = descriptor.qualified_name().to_string();
        let record = route::record(&route::Evidence {
            event: self.last_event(&kernel),
            embedded_artifacts: descriptor.artifacts.len(),
            strict_profile: gust::profile::STRICT,
            kernel: &kernel,
        });
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
    use crate::workloads::{check, gen_inputs, reference};

    /// The GPU tests take turns: creating several Vulkan instances at once made adapter discovery fail now and then.
    fn gpu() -> (std::sync::MutexGuard<'static, ()>, Rc<HeadlessDevice>) {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let device = HeadlessDevice::new().expect("an adapter is needed for the GUST GPU test");
        (guard, Rc::new(device))
    }

    /// Every workload on GUST: end to end, a resident dispatch and a 16-pass batch each match the reference, and a
    /// poisoned output that is never dispatched fails the check (so the check can fail on the GPU path too).
    #[test]
    fn every_workload_passes_its_bound_on_gust_and_an_undispatched_output_fails() {
        let (_gpu, device) = gpu();
        for w in Workload::ALL {
            let n = if w.family() == "W4" { 5000 } else { 4096 + 7 };
            let inputs = gen_inputs(w, n);
            let reference = reference(w, &inputs);
            let mut backend = GustGpu::new(Rc::clone(&device));
            backend.prepare(w, &inputs).unwrap();
            let mut out = Vec::new();

            backend.poison().unwrap();
            backend.read_back(&mut out).unwrap();
            assert!(
                check(w, &inputs, &reference, &out).is_err(),
                "{}: the poison alone must fail the check",
                w.id()
            );

            backend.poison().unwrap();
            backend.end_to_end(&inputs, &mut out).unwrap();
            check(w, &inputs, &reference, &out)
                .unwrap_or_else(|e| panic!("{} end to end: {e}", w.id()));

            backend.poison().unwrap();
            backend.dispatch().unwrap();
            backend.read_back(&mut out).unwrap();
            check(w, &inputs, &reference, &out)
                .unwrap_or_else(|e| panic!("{} dispatch: {e}", w.id()));

            backend.poison().unwrap();
            if backend.gpu_time(16).unwrap().is_some() {
                backend.read_back(&mut out).unwrap();
                check(w, &inputs, &reference, &out)
                    .unwrap_or_else(|e| panic!("{} 16-pass batch: {e}", w.id()));
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
            assert!(gust::profile::STRICT || cfg!(feature = "spirv-route"));
        }
        // A second cold start records again (one record per sample).
        backend.cold_start().unwrap();
        assert_eq!(
            backend.cold_route().expect("recorded again")["kernel"],
            record["kernel"]
        );
    }
}
