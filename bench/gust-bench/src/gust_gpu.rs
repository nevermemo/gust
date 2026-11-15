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

pub struct GustGpu {
    device: Rc<HeadlessDevice>,
    resident: Option<Resident>,
    /// Dispatches since the bins were last cleared (a histogram accumulates; see `Backend::dispatch`).
    passes: Cell<u32>,
    route: RefCell<Option<Value>>,
    /// Route events of the cell's kernel already on the device when `prepare` began: the events are cumulative over the
    /// device's life (the W3 cells share `kernel::horner`), so a cell's own share is the total minus this.
    #[cfg(feature = "spirv-route")]
    events_before_prepare: Cell<usize>,
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
            #[cfg(feature = "spirv-route")]
            events_before_prepare: Cell::new(0),
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
                d.read_typed_buffer(out)?
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
        #[cfg(feature = "spirv-route")]
        {
            let kernel = workloads::descriptor(workload).qualified_name().to_string();
            self.events_before_prepare
                .set(self.route_events_of(&kernel).len());
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
                self.device.write_typed_buffer_timed(out, &r.nan)?;
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
                self.device.write_typed_buffer(x, &inputs.x)?;
                self.device.write_typed_buffer(y, &inputs.y)?;
            }
            Data::Poly { x, .. } => self.device.write_typed_buffer(x, &inputs.x)?,
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

    #[cfg(feature = "spirv-route")]
    fn route_check(&self) -> Result<(), String> {
        let kernel = workloads::descriptor(self.resident().workload)
            .qualified_name()
            .to_string();
        let events = self.route_events_of(&kernel);
        let want = if gust_wgpu::SPIRV_ROUTE {
            "spirv"
        } else {
            "wgsl"
        };
        route_verdict(
            &kernel,
            want,
            own_events(&events, self.events_before_prepare.get()),
            events.last().copied(),
        )
    }

    #[cfg(feature = "spirv-route")]
    fn route_evidence(&self) -> Option<Value> {
        let kernel = workloads::descriptor(self.resident().workload)
            .qualified_name()
            .to_string();
        let events = self.route_events_of(&kernel);
        let want = if gust_wgpu::SPIRV_ROUTE {
            "spirv"
        } else {
            "wgsl"
        };
        let own = own_events(&events, self.events_before_prepare.get());
        let on_build_route = own.iter().filter(|route| **route == want).count();
        let last = self.last_event(&kernel);
        Some(serde_json::json!({
            "kernel": kernel,
            "build_route": want,
            "events": own.len(),
            "events_on_build_route": on_build_route,
            "events_off_build_route": own.len() - on_build_route,
            "events_cumulative": events.len(),
            "last_route": last.as_ref().map(|event| event.route),
            "last_reason": last.and_then(|event| event.reason),
            "source": "HeadlessDevice::route_events, read after the cell (the same events route_check reads); the three `events` counts are this cell's own (since its prepare; 0 for a warm cell, which runs the pipeline an earlier cell built, and route_check then judges it on the kernel's latest event), `events_cumulative` is the device's whole life",
        }))
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
}
