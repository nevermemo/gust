//! `GpuPool<T>`: the first vector-like resident collection (T08).
//!
//! One typed storage allocation plus separate host-tracked metadata, following
//! the contract in `docs/ENGINE_HURRICANE.md`. Growth allocates a larger
//! buffer, copies the live prefix GPU-to-GPU, and retires the old allocation
//! behind the copy's submission fence. Nothing is read back to grow.

use gust::{GpuPod, pod_slice_as_bytes};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{Error, GpuBuffer, GpuBufferAccess, HeadlessDevice};

/// Evidence for one host-driven growth step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthRecord {
    pub old_capacity: usize,
    pub new_capacity: usize,
    /// Live bytes copied GPU-to-GPU; never read back.
    pub copied_bytes: u64,
}

struct Retired {
    _buffer: wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
}

/// A growable, GPU-resident, element-typed collection with an explicit logical
/// length, capacity, and a one-element count buffer kernels can read.
pub struct GpuPool<T: GpuPod> {
    buffer: GpuBuffer<T>,
    count: GpuBuffer<u32>,
    len: usize,
    generation: u64,
    retired: Vec<Retired>,
    growths: Vec<GrowthRecord>,
}

impl HeadlessDevice {
    /// Create a pool holding `values` with room for at least `capacity` elements.
    pub fn create_pool<T: GpuPod>(&self, label: &str, values: &[T], capacity: usize) -> GpuPool<T> {
        let capacity = capacity.max(values.len()).max(1);
        let buffer = self.allocate_pool_buffer::<T>(label, capacity);
        if !values.is_empty() {
            self.queue
                .write_buffer(&buffer.raw, 0, pod_slice_as_bytes(values));
        }
        let count = self.create_typed_buffer(
            &format!("{label} count"),
            &[values.len() as u32],
            GpuBufferAccess::ReadOnly,
        );
        GpuPool {
            buffer,
            count,
            len: values.len(),
            generation: 0,
            retired: Vec::new(),
            growths: Vec::new(),
        }
    }

    fn allocate_pool_buffer<T: GpuPod>(&self, label: &str, capacity: usize) -> GpuBuffer<T> {
        crate::assert_storage_v1_pod::<T>();
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * u64::from(T::LAYOUT.stride),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        GpuBuffer {
            raw,
            len: capacity,
            access: GpuBufferAccess::ReadWrite,
            layout: T::LAYOUT,
            device_id: self.id,
            indirect: false,
            marker: std::marker::PhantomData,
        }
    }

    /// Allocate a buffer that can drive `StagedGraph::dispatch_indirect`.
    ///
    /// `T` must be exactly three `u32` fields (`x`, `y`, `z` workgroup counts) so
    /// it matches `wgpu::util::DispatchIndirectArgs`. Kernels write it as an
    /// ordinary `RWStructuredBuffer<T>`.
    pub fn create_indirect_buffer<T: GpuPod>(
        &self,
        label: &str,
        values: &[T],
    ) -> Result<GpuBuffer<T>, Error> {
        if !crate::is_indirect_args_layout(T::LAYOUT) {
            return Err(Error::IndirectArgsLayout(T::LAYOUT.name));
        }
        crate::assert_storage_v1_pod::<T>();
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::INDIRECT;
        let raw = if values.is_empty() {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(T::LAYOUT.stride),
                usage,
                mapped_at_creation: false,
            })
        } else {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: pod_slice_as_bytes(values),
                usage,
            })
        };
        Ok(GpuBuffer {
            raw,
            len: values.len(),
            access: GpuBufferAccess::ReadWrite,
            layout: T::LAYOUT,
            device_id: self.id,
            indirect: true,
            marker: std::marker::PhantomData,
        })
    }

    /// Append `values`, growing geometrically when capacity is exceeded.
    pub fn pool_push<T: GpuPod>(
        &self,
        pool: &mut GpuPool<T>,
        values: &[T],
    ) -> Result<Option<GrowthRecord>, Error> {
        self.validate_buffer_device(&pool.buffer)?;
        let required = pool.len + values.len();
        let growth = self.pool_reserve(pool, required)?;
        if !values.is_empty() {
            let offset = pool.len as u64 * u64::from(T::LAYOUT.stride);
            self.queue
                .write_buffer(&pool.buffer.raw, offset, pod_slice_as_bytes(values));
        }
        self.pool_set_len(pool, required);
        Ok(growth)
    }

    /// Ensure capacity for `required` elements without changing the length.
    ///
    /// Growth copies exactly the live prefix from the old allocation to the new
    /// one inside a submission; the old allocation is retired behind that fence.
    pub fn pool_reserve<T: GpuPod>(
        &self,
        pool: &mut GpuPool<T>,
        required: usize,
    ) -> Result<Option<GrowthRecord>, Error> {
        self.validate_buffer_device(&pool.buffer)?;
        let old_capacity = pool.buffer.len;
        if required <= old_capacity {
            return Ok(None);
        }
        let new_capacity = required.max(old_capacity.saturating_mul(2));
        let replacement = self.allocate_pool_buffer::<T>("GUST pool growth", new_capacity);
        let copied_bytes = pool.len as u64 * u64::from(T::LAYOUT.stride);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST pool growth encoder"),
            });
        if copied_bytes > 0 {
            encoder.copy_buffer_to_buffer(&pool.buffer.raw, 0, &replacement.raw, 0, copied_bytes);
        }
        let submission = self.queue.submit([encoder.finish()]);
        let old = std::mem::replace(&mut pool.buffer, replacement);
        pool.retired.push(Retired {
            _buffer: old.raw,
            submission,
        });
        pool.generation += 1;
        let record = GrowthRecord {
            old_capacity,
            new_capacity,
            copied_bytes,
        };
        pool.growths.push(record);
        Ok(Some(record))
    }

    /// Shorten the pool without GPU work; a longer `len` is rejected.
    pub fn pool_truncate<T: GpuPod>(&self, pool: &mut GpuPool<T>, len: usize) -> Result<(), Error> {
        self.validate_buffer_device(&pool.buffer)?;
        if len > pool.len {
            return Err(Error::PoolTruncateGrows {
                len: pool.len,
                requested: len,
            });
        }
        self.pool_set_len(pool, len);
        Ok(())
    }

    /// Copy back the live prefix only.
    pub fn pool_read<T: GpuPod>(&self, pool: &GpuPool<T>) -> Result<Vec<T>, Error> {
        let mut values = self.read_typed_buffer(&pool.buffer)?;
        values.truncate(pool.len);
        Ok(values)
    }

    /// Drop retired allocations whose copy submission has completed; returns how many.
    pub fn pool_reclaim<T: GpuPod>(&self, pool: &mut GpuPool<T>) -> Result<usize, Error> {
        let mut kept = Vec::with_capacity(pool.retired.len());
        let mut reclaimed = 0;
        for retired in pool.retired.drain(..) {
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(retired.submission.clone()),
                timeout: Some(std::time::Duration::ZERO),
            }) {
                Ok(status) if status.wait_finished() => reclaimed += 1,
                Ok(_) | Err(wgpu::PollError::Timeout) => kept.push(retired),
                Err(error) => return Err(Error::Poll(error.to_string())),
            }
        }
        pool.retired = kept;
        Ok(reclaimed)
    }

    fn pool_set_len<T: GpuPod>(&self, pool: &mut GpuPool<T>, len: usize) {
        pool.len = len;
        self.queue
            .write_buffer(&pool.count.raw, 0, pod_slice_as_bytes(&[len as u32]));
    }
}

impl<T: GpuPod> GpuPool<T> {
    /// Live element count; elements at or beyond it hold unspecified bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocation size in elements; kernels observe it as the buffer's `.len()`.
    pub const fn capacity(&self) -> usize {
        self.buffer.len
    }

    /// Incremented by every growth.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Allocations retired by growth and not yet reclaimed.
    pub fn retired_allocations(&self) -> usize {
        self.retired.len()
    }

    pub fn growth_history(&self) -> &[GrowthRecord] {
        &self.growths
    }

    /// The storage allocation. Its length is the capacity, not the logical length.
    pub const fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// One-element `u32` buffer mirroring `len()`, for kernels.
    pub const fn count_buffer(&self) -> &GpuBuffer<u32> {
        &self.count
    }
}
