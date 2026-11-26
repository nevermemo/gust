#[cfg(not(target_arch = "wasm32"))]
use crate::completion::blocking::recv_mapping;
#[allow(deprecated)]
use crate::{
    BufferBinding, BufferDispatch, Error, F32BufferBinding, F32BufferDispatch, HeadlessDevice,
    WgpuJob,
};
#[cfg(not(target_arch = "wasm32"))]
#[allow(deprecated)]
use crate::{
    DispatchOutput, DispatchTiming, F32Binding, PassTimestamps, TIMESTAMP_RESULT_BYTES,
    TimestampProfiler, U32Binding, U32DispatchOutput, f32_as_bytes, validate_dispatch,
    validate_u32_dispatch,
};
use gust::KernelDescriptor;
#[cfg(not(target_arch = "wasm32"))]
use gust::{Access, BindingKind, pod_slice_as_bytes};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(not(target_arch = "wasm32"))]
use wgpu::util::{BufferInitDescriptor, DeviceExt};

impl HeadlessDevice {
    /// Compatibility wrapper for an all-f32 persistent dispatch.
    #[deprecated(since = "0.2.0", note = "use `submit_buffers` with `BufferBinding`")]
    #[allow(deprecated)]
    pub fn submit_f32_buffers<'device>(
        &'device self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        self.submit_f32_batch(&[F32BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Dispatch `kernel` over `element_count` elements with `bindings` and wait until it has completed.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(since = "0.2.0", note = "use `dispatch_buffers` with `BufferBinding`")]
    #[allow(deprecated)]
    pub fn dispatch_f32_buffers(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<(), Error> {
        self.submit_f32_buffers(kernel, element_count, bindings)?
            .wait()
    }

    /// [`HeadlessDevice::dispatch_f32_buffers`], returning the host timing of the dispatch.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(
        since = "0.2.0",
        note = "use `dispatch_buffers_timed` with `BufferBinding`"
    )]
    #[allow(deprecated)]
    pub fn dispatch_f32_buffers_timed(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<DispatchTiming, Error> {
        self.dispatch_f32_batch_timed(&[F32BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Submit `dispatches` as one command buffer and return its job without waiting.
    #[deprecated(since = "0.2.0", note = "use `submit_batch` with `BufferDispatch`")]
    #[allow(deprecated)]
    pub fn submit_f32_batch<'device>(
        &'device self,
        dispatches: &[F32BufferDispatch<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        with_f32_dispatches(dispatches, |typed| self.submit_batch(typed))
    }

    /// Submit `dispatches` as one command buffer and wait until it has completed.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(since = "0.2.0", note = "use `dispatch_batch` with `BufferDispatch`")]
    #[allow(deprecated)]
    pub fn dispatch_f32_batch(&self, dispatches: &[F32BufferDispatch<'_>]) -> Result<(), Error> {
        self.submit_f32_batch(dispatches)?.wait()
    }

    /// [`HeadlessDevice::dispatch_f32_batch`], returning the host timing of the batch.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(
        since = "0.2.0",
        note = "use `dispatch_batch_timed` with `BufferDispatch`"
    )]
    #[allow(deprecated)]
    pub fn dispatch_f32_batch_timed(
        &self,
        dispatches: &[F32BufferDispatch<'_>],
    ) -> Result<DispatchTiming, Error> {
        with_f32_dispatches(dispatches, |typed| self.dispatch_batch_timed(typed))
    }

    /// Submit one persistent-buffer dispatch without waiting for completion.
    pub fn submit_buffers<'device>(
        &'device self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        self.submit_batch(&[BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Dispatch using already-resident typed buffers, without upload or readback.
    ///
    /// This convenience wrapper calls [`Self::submit_buffers`] and waits on
    /// its exact submission fence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_buffers(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<(), Error> {
        self.submit_buffers(kernel, element_count, bindings)?.wait()
    }

    /// Dispatch using persistent buffers and report host time plus an optional GPU timestamp.
    ///
    /// `gpu_elapsed` covers only the compute pass. It is `None` when the
    /// adapter does not expose timestamp queries or when `element_count` is zero.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_buffers_timed(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<DispatchTiming, Error> {
        self.dispatch_batch_timed(&[BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Encode and submit a batch, returning before its GPU work completes.
    ///
    /// Every request is validated before any pipeline, bind group, or command
    /// encoding work begins. Empty requests are valid and do not create GPU work. Encoding a `pixel` call creates its
    /// attachment, so such a call can still fail with [`Error::OutOfMemory`] named `GUST pixel attachment`; nothing
    /// is submitted then.
    ///
    /// This is the recording of [`encode_batch`](Self::encode_batch) into an encoder of its own, one counted
    /// `queue.submit` ([`gust_submissions`](Self::gust_submissions)) and
    /// [`EncodedJob::submitted`](crate::EncodedJob::submitted). A batch with no work makes no submission and returns a
    /// job that is complete at once. To put a batch and your own commands in one submission, use `encode_batch`.
    pub fn submit_batch<'device>(
        &'device self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        let prepared = self.prepare_extent_batch(dispatches)?;
        if prepared.is_empty() {
            return Ok(self.encoded_job(false).nothing_submitted());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST typed batch encoder"),
            });
        self.record_prepared(&mut encoder, &prepared)?;
        let submission = self.submit_counted([encoder.finish()]);
        Ok(self.encoded_job(true).submitted(&submission))
    }

    /// Submit a batch and synchronously wait for its exact submission fence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_batch(&self, dispatches: &[BufferDispatch<'_>]) -> Result<(), Error> {
        self.submit_batch(dispatches)?.wait()
    }

    /// Submit a batch once and measure its total host and GPU time.
    ///
    /// A batch of `grid` dispatches is one compute pass, any other batch a pass per dispatch. `gpu_elapsed` runs from
    /// the beginning of the first pass to the end of the last one. Divide it by
    /// the batch length when the dispatches represent equivalent repeated work.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_batch_timed(
        &self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<DispatchTiming, Error> {
        let started = Instant::now();
        let prepared = self.prepare_extent_batch(dispatches)?;
        if prepared.is_empty() {
            return Ok(DispatchTiming {
                host_elapsed: started.elapsed(),
                gpu_elapsed: None,
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST profiled typed batch encoder"),
            });

        let Some(timestamp_profiler) = self.timestamp_profiler.as_ref() else {
            crate::encode_batch_passes(
                &self.device,
                &mut encoder,
                "GUST profiled typed batched pass",
                &prepared,
                None,
            )?;
            let submission = self.submit_counted([encoder.finish()]);
            self.wait_for_submission(submission)?;
            return Ok(DispatchTiming {
                host_elapsed: started.elapsed(),
                gpu_elapsed: None,
            });
        };

        let profiler = timestamp_profiler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::encode_batch_passes(
            &self.device,
            &mut encoder,
            "GUST profiled typed batched pass",
            &prepared,
            Some(PassTimestamps {
                query_set: &profiler.query_set,
                begin: Some(0),
                end: Some(1),
            }),
        )?;
        encoder.resolve_query_set(&profiler.query_set, 0..2, &profiler.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &profiler.resolve_buffer,
            0,
            &profiler.readback_buffer,
            0,
            TIMESTAMP_RESULT_BYTES,
        );
        let submission = self.submit_counted([encoder.finish()]);
        let gpu_elapsed = self.read_timestamp_duration(&profiler, submission)?;
        Ok(DispatchTiming {
            host_elapsed: started.elapsed(),
            gpu_elapsed: Some(gpu_elapsed),
        })
    }

    /// Dispatch an executable kernel over `element_count` one-dimensional invocations.
    ///
    /// This v1 slice accepts one host binding for each storage parameter and reads
    /// every read-write binding back after the dispatch completes.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(
        since = "0.2.0",
        note = "create `GpuBuffer<f32>`s with `create_typed_buffer`, call `dispatch_buffers` and read back with `read_typed_buffer`"
    )]
    #[allow(deprecated)]
    pub fn dispatch_f32(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32Binding<'_>],
    ) -> Result<DispatchOutput, Error> {
        let resources = validate_dispatch(kernel, element_count, bindings)?;
        if element_count == 0 {
            self.check_requires(kernel)?;
            return Ok(DispatchOutput {
                writable_buffers: bindings
                    .iter()
                    .copied()
                    .filter_map(|binding| match binding {
                        F32Binding::ReadWrite(values) => Some(values.to_vec()),
                        F32Binding::ReadOnly(_) => None,
                    })
                    .collect(),
            });
        }

        let workgroup_count = self.check_launch(
            kernel,
            element_count,
            &resources,
            bindings
                .iter()
                .map(|binding| (size_of_val(binding.values()) as u64, BindingKind::Storage)),
        )?;
        let label = kernel.qualified_name().to_string();
        let cached_kernel = self.cached_kernel(kernel, &resources)?;

        let gpu_buffers = bindings
            .iter()
            .copied()
            .enumerate()
            .map(|(index, binding)| {
                let bytes = f32_as_bytes(binding.values());
                let mut usage = wgpu::BufferUsages::STORAGE;
                if binding.access() == Access::ReadWrite {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(&format!("{label} binding {index}")),
                    contents: bytes,
                    usage,
                })
            })
            .collect::<Vec<_>>();
        let bind_group_entries = resources
            .iter()
            .zip(&gpu_buffers)
            .map(|(parameter, buffer)| wgpu::BindGroupEntry {
                binding: parameter
                    .binding
                    .expect("validated resource binding")
                    .binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout: &cached_kernel.bind_group_layout,
            entries: &bind_group_entries,
        });

        let byte_size = u64::from(element_count) * size_of::<f32>() as u64;
        let readback_buffers = bindings
            .iter()
            .copied()
            .map(|binding| match binding {
                F32Binding::ReadOnly(_) => None,
                F32Binding::ReadWrite(_) => {
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label} readback")),
                        size: byte_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }))
                }
            })
            .collect::<Vec<_>>();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label} command encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} compute pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(cached_kernel.pipeline.compute());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        for (gpu_buffer, readback) in gpu_buffers.iter().zip(&readback_buffers) {
            if let Some(readback) = readback {
                encoder.copy_buffer_to_buffer(gpu_buffer, 0, readback, 0, byte_size);
            }
        }
        self.submit_counted([encoder.finish()]);

        let mut completions = Vec::new();
        for readback in readback_buffers.iter().flatten() {
            let (sender, receiver) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
            completions.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;

        let mut writable_buffers = Vec::with_capacity(completions.len());
        for (readback, completion) in readback_buffers.iter().flatten().zip(completions) {
            recv_mapping(&completion)?;
            let view = readback
                .get_mapped_range(..)
                .map_err(|error| Error::Map(error.to_string()))?;
            let values = view
                .as_chunks::<{ size_of::<f32>() }>()
                .0
                .iter()
                .map(|chunk| f32::from_ne_bytes(*chunk))
                .collect();
            drop(view);
            readback.unmap();
            writable_buffers.push(values);
        }

        Ok(DispatchOutput { writable_buffers })
    }

    /// Dispatch an executable kernel over `element_count` one-dimensional invocations
    /// using `u32` storage buffers.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(
        since = "0.2.0",
        note = "create `GpuBuffer<u32>`s with `create_typed_buffer`, call `dispatch_buffers` and read back with `read_typed_buffer`"
    )]
    #[allow(deprecated)]
    pub fn dispatch_u32(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[U32Binding<'_>],
    ) -> Result<U32DispatchOutput, Error> {
        let resources = validate_u32_dispatch(kernel, element_count, bindings)?;
        if element_count == 0 {
            self.check_requires(kernel)?;
            return Ok(U32DispatchOutput {
                writable_buffers: bindings
                    .iter()
                    .copied()
                    .filter_map(|binding| match binding {
                        U32Binding::ReadWrite(values) => Some(values.to_vec()),
                        U32Binding::ReadOnly(_) => None,
                    })
                    .collect(),
            });
        }

        let workgroup_count = self.check_launch(
            kernel,
            element_count,
            &resources,
            bindings
                .iter()
                .map(|binding| (size_of_val(binding.values()) as u64, BindingKind::Storage)),
        )?;
        let label = kernel.qualified_name().to_string();
        let cached_kernel = self.cached_kernel(kernel, &resources)?;

        let gpu_buffers = bindings
            .iter()
            .copied()
            .enumerate()
            .map(|(index, binding)| {
                let bytes = pod_slice_as_bytes(binding.values());
                let mut usage = wgpu::BufferUsages::STORAGE;
                if binding.access() == Access::ReadWrite {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(&format!("{label} binding {index}")),
                    contents: bytes,
                    usage,
                })
            })
            .collect::<Vec<_>>();
        let bind_group_entries = resources
            .iter()
            .zip(&gpu_buffers)
            .map(|(parameter, buffer)| wgpu::BindGroupEntry {
                binding: parameter
                    .binding
                    .expect("validated resource binding")
                    .binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout: &cached_kernel.bind_group_layout,
            entries: &bind_group_entries,
        });

        let byte_size = u64::from(element_count) * size_of::<u32>() as u64;
        let readback_buffers = bindings
            .iter()
            .copied()
            .map(|binding| match binding {
                U32Binding::ReadOnly(_) => None,
                U32Binding::ReadWrite(_) => {
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label} readback")),
                        size: byte_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }))
                }
            })
            .collect::<Vec<_>>();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label} command encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} compute pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(cached_kernel.pipeline.compute());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        for (gpu_buffer, readback) in gpu_buffers.iter().zip(&readback_buffers) {
            if let Some(readback) = readback {
                encoder.copy_buffer_to_buffer(gpu_buffer, 0, readback, 0, byte_size);
            }
        }
        self.submit_counted([encoder.finish()]);

        let mut completions = Vec::new();
        for readback in readback_buffers.iter().flatten() {
            let (sender, receiver) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
            completions.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;

        let mut writable_buffers = Vec::with_capacity(completions.len());
        for (readback, completion) in readback_buffers.iter().flatten().zip(completions) {
            recv_mapping(&completion)?;
            let view = readback
                .get_mapped_range(..)
                .map_err(|error| Error::Map(error.to_string()))?;
            let values = view
                .as_chunks::<{ size_of::<u32>() }>()
                .0
                .iter()
                .map(|chunk| u32::from_ne_bytes(*chunk))
                .collect();
            drop(view);
            readback.unmap();
            writable_buffers.push(values);
        }

        Ok(U32DispatchOutput { writable_buffers })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wait_for_submission(
        &self,
        submission: wgpu::SubmissionIndex,
    ) -> Result<(), Error> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| Error::Poll(error.to_string()))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read_timestamp_duration(
        &self,
        profiler: &TimestampProfiler,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Duration, Error> {
        let (sender, receiver) = mpsc::channel();
        profiler
            .readback_buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
        self.wait_for_submission(submission)?;
        recv_mapping(&receiver)?;
        let view = profiler
            .readback_buffer
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let mut timestamps = view
            .as_chunks::<{ size_of::<u64>() }>()
            .0
            .iter()
            .map(|chunk| u64::from_ne_bytes(*chunk));
        let beginning = timestamps.next().expect("beginning timestamp");
        let end = timestamps.next().expect("end timestamp");
        drop(view);
        profiler.readback_buffer.unmap();
        let nanoseconds =
            end.wrapping_sub(beginning) as f64 * f64::from(self.queue.get_timestamp_period());
        Ok(Duration::from_secs_f64(nanoseconds / 1_000_000_000.0))
    }
}

#[allow(deprecated)]
fn with_f32_dispatches<R>(
    dispatches: &[F32BufferDispatch<'_>],
    run: impl FnOnce(&[BufferDispatch<'_>]) -> R,
) -> R {
    let bindings = dispatches
        .iter()
        .map(|dispatch| {
            dispatch
                .bindings
                .iter()
                .copied()
                .map(BufferBinding::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let typed = dispatches
        .iter()
        .zip(&bindings)
        .map(|(dispatch, bindings)| {
            BufferDispatch::new(dispatch.kernel, dispatch.element_count, bindings)
        })
        .collect::<Vec<_>>();
    run(&typed)
}
