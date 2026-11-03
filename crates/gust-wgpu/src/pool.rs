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
        crate::assert_pod_matches_storage_layout::<T>();
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
            lease: None,
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
        crate::assert_pod_matches_storage_layout::<T>();
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
            lease: None,
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
    #[cfg(not(target_arch = "wasm32"))]
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

// ---------------------------------------------------------------------------------------------------------------
// `BufferPool` (m7.a0): recycled allocations for `GpuVec` and the element-wise map calls.
// ---------------------------------------------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
use crate::completion::blocking::block_on_mapping;
use crate::{Completion, MappingReady, pod_vec_from_bytes};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

/// Allocations of at most this many bytes are pooled by their exact size; larger ones by the next power of two (or
/// by the device's `max_buffer_size` when that is smaller).
pub const EXACT_CLASS_BYTES: u64 = 64 * 1024;

/// The most bytes a pool keeps idle by default (1 GiB). Past it the least recently returned buffers are dropped.
pub const DEFAULT_POOL_BYTES_MAX: u64 = 1 << 30;

/// Counters of a [`BufferPool`], for tests and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Hand-outs served from an idle buffer.
    pub hits: u64,
    /// Hand-outs that needed a new allocation.
    pub misses: u64,
    /// Buffers put back on the idle lists.
    pub returned: u64,
    /// Idle buffers dropped to stay within `max_bytes` (or returned while larger than the cap).
    pub evicted: u64,
    /// Buffers dropped on return because a mapping was still pending or active: a mapped buffer is never reused.
    pub discarded_mapped: u64,
    /// Idle buffers now.
    pub free_buffers: usize,
    /// Bytes held by the idle buffers now.
    pub free_bytes: u64,
    /// The idle-bytes cap.
    pub max_bytes: u64,
    /// Buffers handed out and not yet returned.
    pub outstanding: usize,
}

/// The size class of a request of `bytes` on a device whose largest buffer is `max_buffer_size`: exact (rounded up to
/// a whole 4 bytes) up to [`EXACT_CLASS_BYTES`], then the next power of two, capped at `max_buffer_size` rounded down
/// to a multiple of 4. `None` when no class holds `bytes`.
pub(crate) fn size_class(bytes: u64, max_buffer_size: u64) -> Option<u64> {
    let bytes = bytes.max(4).checked_next_multiple_of(4)?;
    if bytes <= EXACT_CLASS_BYTES {
        return Some(bytes);
    }
    let class = bytes.checked_next_power_of_two()?;
    if class <= max_buffer_size {
        return Some(class);
    }
    let capped = max_buffer_size & !3;
    (capped >= bytes).then_some(capped)
}

struct PoolState {
    /// Idle buffers by `(usage bits, class bytes)`, oldest first.
    free: HashMap<(u32, u64), Vec<(wgpu::Buffer, u64)>>,
    free_bytes: u64,
    max_bytes: u64,
    tick: u64,
    stats: PoolStats,
}

impl PoolState {
    /// Drop the least recently returned idle buffers until the cap holds.
    fn evict(&mut self) {
        while self.free_bytes > self.max_bytes {
            let oldest = self
                .free
                .iter()
                .filter_map(|(key, list)| list.first().map(|(_, tick)| (*tick, *key)))
                .min();
            let Some((_, key)) = oldest else { break };
            let list = self.free.get_mut(&key).expect("key was just listed");
            list.remove(0);
            if list.is_empty() {
                self.free.remove(&key);
            }
            self.free_bytes -= key.1;
            self.stats.evicted += 1;
        }
    }
}

struct PoolShared {
    state: Mutex<PoolState>,
}

impl PoolShared {
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        // Nothing runs under the lock that can leave the state half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Recycles device allocations in power-of-two byte classes, with an idle-bytes cap (default 1 GiB) enforced
/// least-recently-returned first. One pool belongs to one [`HeadlessDevice`]; [`GpuVec`](crate::GpuVec) takes its
/// storage from it and gives it back on drop. Cloning shares the pool.
///
/// What the pool guarantees, each pinned by a test:
///
/// * A pooled buffer may be larger than the logical length asked for. Every binding exposes the logical bytes only, so
///   `arrayLength` in a kernel and the typed length checks see the logical length.
/// * A buffer handed out as zeroed is zero in its logical range before any later submission reads it: a recycled one
///   is cleared with `CommandEncoder::clear_buffer` in a submission of its own (queue order puts it before every later
///   one), a fresh one is zero by wgpu's guarantee. Buffers filled from a slice are written over their whole logical
///   range.
/// * A buffer with a mapping pending or active is never put back, so it can never be handed out while mapped.
#[derive(Clone)]
pub struct BufferPool {
    shared: Arc<PoolShared>,
}

impl BufferPool {
    // The pool is shared by clones on one thread on wasm32, where wgpu's types are not `Send`; the `Arc` is still right
    // on native, and one definition serves both.
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn new() -> Self {
        Self {
            shared: Arc::new(PoolShared {
                state: Mutex::new(PoolState {
                    free: HashMap::new(),
                    free_bytes: 0,
                    max_bytes: DEFAULT_POOL_BYTES_MAX,
                    tick: 0,
                    stats: PoolStats::default(),
                }),
            }),
        }
    }

    /// The counters now.
    pub fn stats(&self) -> PoolStats {
        let state = self.shared.lock();
        PoolStats {
            free_buffers: state.free.values().map(Vec::len).sum(),
            free_bytes: state.free_bytes,
            max_bytes: state.max_bytes,
            ..state.stats
        }
    }

    /// Change the idle-bytes cap and drop idle buffers down to it.
    pub fn set_max_bytes(&self, bytes: u64) {
        let mut state = self.shared.lock();
        state.max_bytes = bytes;
        state.evict();
    }

    /// Drop every idle buffer. Buffers still handed out are unaffected and return to the (now empty) pool.
    pub fn clear(&self) {
        let mut state = self.shared.lock();
        state.free.clear();
        state.free_bytes = 0;
    }

    /// An idle buffer of this class and usage, or `None` (counted as a miss).
    pub(crate) fn take(&self, usage: wgpu::BufferUsages, class: u64) -> Option<wgpu::Buffer> {
        let mut state = self.shared.lock();
        let key = (usage.bits(), class);
        let taken = state.free.get_mut(&key).and_then(Vec::pop);
        if let Some((buffer, _)) = taken {
            if state.free.get(&key).is_some_and(Vec::is_empty) {
                state.free.remove(&key);
            }
            state.free_bytes -= class;
            state.stats.hits += 1;
            Some(buffer)
        } else {
            state.stats.misses += 1;
            None
        }
    }

    /// A lease that returns `buffer` (of `class` bytes, `usage`) to this pool when it drops.
    pub(crate) fn lease(
        &self,
        buffer: wgpu::Buffer,
        usage: wgpu::BufferUsages,
        class: u64,
    ) -> Lease {
        self.shared.lock().stats.outstanding += 1;
        Lease {
            buffer,
            usage: usage.bits(),
            class,
            mapped: false,
            pool: Arc::clone(&self.shared),
        }
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("BufferPool")
            .field(&self.stats())
            .finish()
    }
}

/// The right to an allocation of a [`BufferPool`]: dropping it puts the allocation back, unless a mapping may still
/// be pending or active (`mapped`), in which case the buffer is dropped instead.
pub(crate) struct Lease {
    buffer: wgpu::Buffer,
    usage: u32,
    class: u64,
    mapped: bool,
    pool: Arc<PoolShared>,
}

impl Lease {
    pub(crate) const fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Record that a `map_async` was requested: until [`recycle`](Self::recycle) says otherwise, the buffer must not
    /// go back to the pool.
    fn mark_mapped(&mut self) {
        self.mapped = true;
    }

    /// Return the buffer to the pool after its mapping finished: unmap it if the mapping succeeded.
    fn recycle(mut self, unmap: bool) {
        if unmap {
            self.buffer.unmap();
        }
        self.mapped = false;
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.pool.lock();
        state.stats.outstanding -= 1;
        if self.mapped {
            state.stats.discarded_mapped += 1;
            return;
        }
        if self.class > state.max_bytes {
            state.stats.evicted += 1;
            return;
        }
        state.tick += 1;
        let tick = state.tick;
        state
            .free
            .entry((self.usage, self.class))
            .or_default()
            .push((self.buffer.clone(), tick));
        state.free_bytes += self.class;
        state.stats.returned += 1;
        state.evict();
    }
}

/// How a pooled buffer starts.
pub(crate) enum Fill<'a, T> {
    /// Zero in the logical range.
    Zeroed,
    /// Holding these values, written over the whole logical range.
    Values(&'a [T]),
}

impl HeadlessDevice {
    /// This device's buffer pool.
    pub const fn buffer_pool(&self) -> &BufferPool {
        &self.pool
    }

    /// The counters of this device's buffer pool.
    pub fn buffer_pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// A read-write persistent buffer of `len` elements from the pool, zeroed or holding `fill`'s values. The
    /// `max_buffer_size` pre-check and the out-of-memory scope apply to a fresh allocation; a recycled buffer is
    /// cleared (zeroed) or written (values) before this returns.
    pub(crate) fn pooled_buffer<T: GpuPod>(
        &self,
        label: &str,
        len: usize,
        fill: Fill<'_, T>,
    ) -> Result<GpuBuffer<T>, Error> {
        crate::assert_pod_matches_storage_layout::<T>();
        if let Fill::Values(values) = &fill {
            debug_assert_eq!(
                values.len(),
                len,
                "Fill::Values must hold exactly `len` elements"
            );
        }
        let bytes = crate::device::zeroed_size(T::LAYOUT.stride, len);
        let class = self.pooled_class(label, bytes)?;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let (raw, recycled) = match self.pool.take(usage, class) {
            Some(raw) => (raw, true),
            None => (self.allocate(label, class, usage)?, false),
        };
        let lease = self.pool.lease(raw.clone(), usage, class);
        match fill {
            Fill::Values(values) if !values.is_empty() => {
                self.queue.write_buffer(&raw, 0, pod_slice_as_bytes(values));
            }
            // Zero elements, or zeroed: a fresh buffer reads as zero; a recycled one holds its previous contents.
            Fill::Values(_) | Fill::Zeroed => {
                if recycled {
                    self.clear_buffer_prefix(&raw, bytes);
                }
            }
        }
        Ok(GpuBuffer {
            raw,
            len,
            access: GpuBufferAccess::ReadWrite,
            layout: T::LAYOUT,
            device_id: self.id,
            indirect: false,
            lease: Some(lease),
            marker: PhantomData,
        })
    }

    /// The pool class for `bytes`, after the `max_buffer_size` pre-check.
    fn pooled_class(&self, label: &str, bytes: u64) -> Result<u64, Error> {
        self.check_buffer_size(label, bytes)?;
        size_class(bytes, self.limits.max_buffer_size).ok_or_else(|| Error::DeviceLimit {
            kernel: label.to_owned(),
            limit: "max_buffer_size",
            requested: bytes,
            maximum: self.limits.max_buffer_size,
        })
    }

    /// Zero the first `bytes` of `buffer` in a submission of its own. The queue runs submissions in order, so every
    /// later submission that reads the buffer sees zeros.
    fn clear_buffer_prefix(&self, buffer: &wgpu::Buffer, bytes: u64) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST pooled buffer clear"),
            });
        encoder.clear_buffer(buffer, 0, Some(bytes));
        self.queue.submit([encoder.finish()]);
    }

    /// Copy `buffer` into a pooled staging buffer, submit the copy and request the mapping. The returned readback
    /// owns the staging buffer until the mapping's callback has fired, whether or not it is awaited.
    // The slot is shared with the mapping callback; on wasm32 that stays on one thread (wgpu is not `Send` there).
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn start_pooled_readback<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<PooledReadback<T>, Error> {
        self.validate_buffer_device(buffer)?;
        if buffer.access != GpuBufferAccess::ReadWrite {
            return Err(Error::PersistentBufferNotReadable);
        }
        if buffer.is_empty() {
            return Ok(PooledReadback {
                slot: Arc::new(Mutex::new(Slot::default())),
                mapped: None,
                bytes: 0,
                marker: PhantomData,
            });
        }
        let bytes = buffer.len as u64 * u64::from(T::LAYOUT.stride);
        let label = "GUST pooled readback";
        let class = self.pooled_class(label, bytes)?;
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let raw = match self.pool.take(usage, class) {
            Some(raw) => raw,
            None => self.allocate(label, class, usage)?,
        };
        let mut lease = self.pool.lease(raw.clone(), usage, class);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST pooled readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&buffer.raw, 0, &raw, 0, bytes);
        self.queue.submit([encoder.finish()]);

        lease.mark_mapped();
        let slot = Arc::new(Mutex::new(Slot {
            lease: Some(lease),
            ..Slot::default()
        }));
        let completion = Completion::new();
        let callback_slot = Arc::clone(&slot);
        let callback_completion = completion.clone();
        raw.map_async(wgpu::MapMode::Read, 0..bytes, move |result| {
            let ok = result.is_ok();
            let leftover = {
                let mut slot = callback_slot.lock().unwrap_or_else(PoisonError::into_inner);
                slot.done = true;
                slot.ok = ok;
                // The readback was dropped before the mapping finished: nobody will read the buffer, so the
                // callback gives it back.
                if slot.abandoned {
                    slot.lease.take()
                } else {
                    None
                }
            };
            if let Some(lease) = leftover {
                lease.recycle(ok);
            }
            let _ = callback_completion.complete(result.map_err(|error| error.to_string()));
        });
        Ok(PooledReadback {
            slot,
            mapped: Some(completion),
            bytes,
            marker: PhantomData,
        })
    }

    /// Read `buffer` back through a pooled staging buffer, blocking until the copy completes. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read_pooled<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        let readback = self.start_pooled_readback(buffer)?;
        // `poll(Wait)` returns when the GPU is done, but a poll of another thread may have collected this mapping's
        // callback and not run it yet (see `completion::blocking`): wait for the callback itself, bounded.
        readback.wait(&self.device)?;
        block_on_mapping(readback.finish(&self.device))
    }

    /// [`read_pooled`](Self::read_pooled) without blocking: each poll polls the device once (see
    /// `read_typed_buffer_async`).
    pub(crate) async fn read_pooled_async<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<Vec<T>, Error> {
        self.start_pooled_readback(buffer)?
            .finish(&self.device)
            .await
    }
}

/// What the staging buffer of a [`PooledReadback`] is doing, shared with its `map_async` callback.
#[derive(Default)]
struct Slot {
    /// The staging buffer, until someone takes it to recycle it.
    lease: Option<Lease>,
    /// The callback has run.
    done: bool,
    /// The mapping succeeded (meaningful once `done`).
    ok: bool,
    /// The readback was dropped before the callback ran: the callback recycles the buffer.
    abandoned: bool,
}

/// A readback whose copy was submitted and mapping requested. It owns the staging buffer until the mapping callback
/// has fired; dropping it early hands the buffer to the callback, which returns it to the pool (never reused while
/// mapped).
pub(crate) struct PooledReadback<T> {
    slot: Arc<Mutex<Slot>>,
    /// `None` for an empty buffer, which has no mapping.
    mapped: Option<Completion<Result<(), String>>>,
    bytes: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T: GpuPod> PooledReadback<T> {
    /// Block until the device has finished the submitted work. This mapping's callback may not have run yet (another
    /// thread's poll can have collected it); `read_pooled` waits for the completion afterwards.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, device: &wgpu::Device) -> Result<(), Error> {
        if self.mapped.is_some() {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| Error::Poll(error.to_string()))?;
        }
        Ok(())
    }

    /// Await the mapping, copy the bytes out, unmap, and give the staging buffer back to the pool.
    async fn finish(mut self, device: &wgpu::Device) -> Result<Vec<T>, Error> {
        let Some(mapped) = self.mapped.take() else {
            return Ok(Vec::new());
        };
        // On an error the buffer stays in the slot and `Drop` recycles it.
        MappingReady { device, mapped }.await?;
        let lease = self
            .slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .lease
            .take()
            .expect("a finished mapping leaves its staging buffer in the slot");
        let values = {
            let view = lease
                .buffer()
                .get_mapped_range(0..self.bytes)
                .map_err(|error| Error::Map(error.to_string()))?;
            pod_vec_from_bytes(&view)?
        };
        lease.recycle(true);
        Ok(values)
    }
}

impl<T> Drop for PooledReadback<T> {
    fn drop(&mut self) {
        let leftover = {
            let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
            if slot.lease.is_none() {
                return;
            }
            if slot.done {
                let ok = slot.ok;
                slot.lease.take().map(|lease| (lease, ok))
            } else {
                slot.abandoned = true;
                None
            }
        };
        if let Some((lease, ok)) = leftover {
            lease.recycle(ok);
        }
    }
}
