use crate::cache::PipelineCache;
#[cfg(doc)]
use crate::{DeviceOptions, GpuVec, LimitsPolicy};
use crate::{Error, GpuBuffer, HeadlessDevice, PipelineCacheStats, ShaderChecks};
use gust::capability::Capability;
use std::sync::MutexGuard;
use std::sync::atomic::Ordering;

impl HeadlessDevice {
    /// The wgpu description of the adapter that the device was opened on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// The capabilities this device was opened or adopted with (`DeviceOptions::capabilities`, or the argument of
    /// [`adopt`](Self::adopt)), as asked: a kernel whose
    /// `requires` names a capability not in this set is refused with [`Error::MissingCapability`].
    pub fn capabilities(&self) -> &'static [Capability] {
        self.capabilities
    }

    /// Whether this device creates capture-capable buffers (`DeviceOptions::capture`): read-only storage buffers can be
    /// copied for a debugging capture. A device adopted with [`adopt`](Self::adopt) does not.
    pub fn capture_enabled(&self) -> bool {
        self.capture
    }

    /// Whether this device builds its WGSL shader modules with wgpu's runtime checks: [`ShaderChecks::Checked`] unless
    /// it was opened with [`DeviceOptions::unchecked_kernels`]. A device adopted with [`adopt`](Self::adopt) is always
    /// checked.
    pub fn shader_checks(&self) -> ShaderChecks {
        self.shader_checks
    }

    /// The smallest and the largest subgroup size as wgpu reports them for the adapter (`AdapterInfo`), or `None` when the
    /// adapter does not offer subgroup operations. On Metal wgpu reports a fixed 4..64, whatever the GPU does.
    ///
    /// This is a range, not a promise of one size: the lane count a shader sees can differ between the
    /// dispatches of one device (spike S3: an AMD Radeon 890M on DX12 gave 32 lanes in one kernel and 64 in the
    /// others), so a kernel reads its lane count at run time and never assumes one value from this method. It is
    /// `Some` whether or not the device was opened with [`Capability::Subgroups`].
    pub fn subgroup_sizes(&self) -> Option<(u32, u32)> {
        self.subgroup_sizes
    }

    /// The limits launches are checked against: those the device was opened with
    /// (wgpu's defaults for `new`/`new_async`; the adapter's own buffer and dispatch limits under
    /// [`LimitsPolicy::Adapter`]), or the lower ones from `with_limits`.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// The wgpu device behind this one: an advanced escape hatch, like [`GpuBuffer::as_wgpu_buffer`], for code that
    /// records its own passes over GUST's buffers (a buffer is only valid on the device that created it).
    ///
    /// Work you create here is not tracked by GUST: validation, limits and pooling stay your concern.
    /// Ordering is the queue's ([`Self::wgpu_queue`]).
    ///
    /// A device GUST opened itself lives in a private `wgpu::Instance` and must never be given to a surface: configuring
    /// a surface of another instance with it panics inside wgpu-core, or silently binds the surface to an unrelated
    /// device. A renderer that presents opens its own device in its own instance and shares it with
    /// [`adopt`](Self::adopt) instead.
    ///
    /// Hazards: calling `destroy` on this device destroys GUST's device too, and every call of this `HeadlessDevice`
    /// fails or panics after it. An `on_uncaptured_error` handler you install replaces wgpu's default, so the panic of
    /// [`create_typed_buffer`](Self::create_typed_buffer) on an oversized allocation (and any other uncaptured error)
    /// is then yours to handle. Under [`with_limits`](Self::with_limits) emulation the raw device keeps the limits it was
    /// opened with, so a pipeline or buffer you create here is not held to the lowered ones.
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue behind this device: an advanced escape hatch, like [`GpuBuffer::as_wgpu_buffer`].
    ///
    /// GUST submits every dispatch to this one queue. The submissions of one queue behave as if they ran in
    /// submission order, and wgpu inserts the barriers between them, so your own submission sees the storage-buffer
    /// writes of a kernel submitted before it, with no barrier of yours. That is an ordering, not a promise that the
    /// earlier work has finished before the later work starts. On native targets the typed `dispatch` and `dispatch_*`
    /// wrappers also wait for their submission, so the work is done when they return; on `wasm32` they submit and
    /// return [`Error::WouldBlock`], so that holds on native targets only. `submit_*` returns a job without waiting,
    /// and your submission is still ordered after it.
    /// Submit commands that use a pooled buffer before its owner drops (see [`GpuBuffer::as_wgpu_buffer`]).
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// How many queue submissions GUST itself has made on this device since it was opened or adopted, counting each
    /// `queue.submit` call once, whatever it carried (a submission of no command buffers counts too).
    ///
    /// It counts GUST's own submissions only. On a device shared with a renderer ([`adopt`](Self::adopt)) the
    /// caller's own submissions, including the ones made with [`wgpu_queue`](Self::wgpu_queue) and the one that
    /// carries an [`encode_batch`](Self::encode_batch) encoding, are not counted: a frame that records GUST work
    /// into its own encoder and submits it once leaves this number unchanged. The counter is a diagnostic for
    /// submission-count budgets and tests; reading it takes no lock and orders nothing.
    ///
    /// Calls that submit on their own account, one submission each (a test pins every one named here):
    /// [`submit_batch`](Self::submit_batch) and the typed `dispatch`/`submit` built on it, which make none for a
    /// batch with no work; `dispatch_batch_timed`;
    /// `write_typed_buffer_timed`; `read_typed_buffer`
    /// and `GpuVec::to_vec`; and a pooled buffer handed out recycled and zeroed (`GpuVec::zeroed_on`, the clear is a
    /// submission of its own). The other submitting calls of this crate (pool growth, texture and inspect
    /// read-backs, `execute_graph`) count through the same wrapper. [`submit_batch_read`](Self::submit_batch_read)
    /// makes one for a batch, its clears and a read-back together; [`encode_batch`](Self::encode_batch) and
    /// [`encode_batch_read`](Self::encode_batch_read) make none.
    pub fn gust_submissions(&self) -> u64 {
        self.submissions.load(Ordering::Relaxed)
    }

    /// The one place this crate calls `Queue::submit`, so that [`gust_submissions`](Self::gust_submissions) counts
    /// every submission GUST makes. Never call `self.queue.submit` anywhere else in this crate.
    pub(crate) fn submit_counted(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.submissions.fetch_add(1, Ordering::Relaxed);
        self.queue.submit(commands)
    }

    /// Check launches against `limits` instead of the opened device's, for emulating a
    /// weaker target and for tests. The limits may only be lowered: a value that is
    /// better than the one the device was opened with is rejected with
    /// `Error::LimitRaised` and never clamped, because wgpu would keep enforcing the
    /// device's own. The replacement is compared with the opened device, not with
    /// limits installed earlier, so a lowered limit can be raised back up to it. The
    /// pipeline cache is kept; the checks run before it is consulted. A rejected call
    /// consumes the device, like any failing builder step.
    ///
    /// Only these fields are honoured by the launch checks:
    /// `max_compute_workgroup_size_x`, `max_compute_workgroup_size_y`,
    /// `max_compute_workgroup_size_z`, `max_compute_invocations_per_workgroup`,
    /// `max_compute_workgroups_per_dimension` (for x, y and z alike), `max_storage_buffers_per_shader_stage`,
    /// `max_storage_buffer_binding_size`, `max_uniform_buffers_per_shader_stage`,
    /// `max_uniform_buffer_binding_size`, `max_sampled_textures_per_shader_stage`,
    /// `max_storage_textures_per_shader_stage`, `max_samplers_per_shader_stage`, `max_texture_dimension_2d` (the
    /// sides of every texture a launch binds, and of `create_texture_2d`) and `max_bindings_per_bind_group`
    /// (every binding index of a kernel must be below it). Every other field is accepted if it is not
    /// raised, but has no effect. `max_buffer_size` is read by the fallible buffer functions
    /// ([`try_create_typed_buffer`](Self::try_create_typed_buffer),
    /// [`create_zeroed_buffer`](Self::create_zeroed_buffer),
    /// [`GpuVec`]) and by nothing else; `create_typed_buffer` never checks it.
    pub fn with_limits(mut self, limits: wgpu::Limits) -> Result<Self, Error> {
        let mut raised = None;
        limits.check_limits_with_fail_fn(
            &self.device.limits(),
            true,
            |limit, requested, device| {
                raised = Some(Error::LimitRaised {
                    limit,
                    requested,
                    device,
                });
            },
        );
        if let Some(error) = raised {
            return Err(error);
        }
        self.limits = limits;
        Ok(self)
    }

    /// Whether the device has wgpu's `MEMORY_DECORATION_COHERENT` feature, coherent storage memory.
    pub fn supports_coherent_storage(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::MEMORY_DECORATION_COHERENT)
    }

    /// Whether this device can measure compute passes with hardware timestamp queries.
    pub const fn supports_gpu_timestamps(&self) -> bool {
        self.timestamp_profiler.is_some()
    }

    /// Return hit/miss counts for cached shader modules, layouts, and pipelines.
    pub fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        let cache = self.lock_pipeline_cache();
        PipelineCacheStats {
            entries: cache.kernels.len(),
            hits: cache.hits,
            misses: cache.misses,
        }
    }

    /// Drop the cache's references to every compiled kernel of this device and reset its counters. A graph node
    /// that kept a bind group still holds its kernel until the node is executed again or the graph drops.
    pub fn clear_pipeline_cache(&self) {
        *self.lock_pipeline_cache() = PipelineCache::default();
    }

    pub(crate) fn lock_pipeline_cache(&self) -> MutexGuard<'_, PipelineCache> {
        self.pipeline_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn validate_buffer_device<T>(&self, buffer: &GpuBuffer<T>) -> Result<(), Error> {
        if buffer.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        Ok(())
    }
}
