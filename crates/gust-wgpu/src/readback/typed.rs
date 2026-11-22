use crate::completion::Completion;
#[cfg(not(target_arch = "wasm32"))]
use crate::completion::blocking::block_on_mapping;
use crate::{
    Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, PendingMapping, TransferTiming,
    TypedReadback,
};
use gust::GpuPod;
use std::marker::PhantomData;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

impl HeadlessDevice {
    /// Copy a persistent read-write POD buffer back to host memory.
    ///
    /// Blocks the calling thread until the copy has completed; not available on `wasm32`, where
    /// [`read_typed_buffer_async`](Self::read_typed_buffer_async) is the readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_typed_buffer<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_typed_buffer_timed(buffer)
            .map(|(values, _)| values)
    }

    /// [`HeadlessDevice::read_typed_buffer`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(since = "0.2.0", note = "use `read_typed_buffer`")]
    pub fn read_f32_buffer(&self, buffer: &GpuBuffer<f32>) -> Result<Vec<f32>, Error> {
        self.read_typed_buffer(buffer)
    }

    /// Copy a persistent read-write POD buffer back to host memory without blocking the caller.
    ///
    /// The returned future requests the copy and the mapping when it is first polled. Each poll then polls the
    /// device once (`wgpu::PollType::Poll`, never `Wait`) and stores the task's waker for the `map_async` callback
    /// to wake.
    ///
    /// # Native: the future completes only while something polls the device
    ///
    /// wgpu runs the `map_async` callback inside a device poll, and on native nothing polls the device in the
    /// background. This future never wakes itself, so awaiting it on an executor that re-polls only after a wake
    /// hangs, unless another thread keeps polling the device. The blocking
    #[cfg_attr(
        not(target_arch = "wasm32"),
        doc = "[`read_typed_buffer`](Self::read_typed_buffer)"
    )]
    #[cfg_attr(target_arch = "wasm32", doc = "`read_typed_buffer`")]
    /// does it correctly: it starts the copy, waits with `poll(Wait)`, then waits on the finished future itself, because
    /// another thread's device poll can run this mapping's callback after this thread's `poll(Wait)` returned
    /// (bounded, then `Error::Map`; `completion::blocking`). On `wasm32` the browser runs the callback from its event
    /// loop, which wakes the task, so a plain `.await` suffices there (measured in Chrome by `cargo xtask check-web`).
    ///
    /// ```no_run
    /// # use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
    /// let device = HeadlessDevice::new()?;
    /// let buffer = device.create_typed_buffer("values", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadWrite);
    /// // On native this is the way to read a buffer back; it polls the device for you.
    /// let values = device.read_typed_buffer(&buffer)?;
    /// assert_eq!(values, [1.0, 2.0, 3.0]);
    /// // A caller who drives the async function on its own executor must keep polling the future (each poll
    /// // polls the device once) or the device itself, as `read_typed_buffer` does.
    /// # Ok::<(), gust_wgpu::Error>(())
    /// ```
    pub async fn read_typed_buffer_async<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<Vec<T>, Error> {
        self.start_typed_readback(buffer)?
            .finish(&self.device)
            .await
    }

    /// Copy a persistent read-write POD buffer back and report its complete host time.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_typed_buffer_timed<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<(Vec<T>, TransferTiming), Error> {
        let started = Instant::now();
        let readback = self.start_typed_readback(buffer)?;
        // The callback runs inside a device poll, but not necessarily this thread's: another thread's poll collects
        // the callbacks of every finished submission, so `poll(Wait)` can return before this mapping's callback has
        // run. The wait is therefore on the completion itself, which the callback signals from whichever thread
        // runs it, for at most `MAPPING_WAIT` (then `Error::Map`).
        readback.wait(&self.device)?;
        let values = block_on_mapping(readback.finish(&self.device))?;
        Ok((
            values,
            TransferTiming {
                bytes: buffer.len as u64 * size_of::<T>() as u64,
                elapsed: started.elapsed(),
            },
        ))
    }

    /// Validate `buffer`, copy it into a mappable staging buffer, submit the copy and request the mapping. The
    /// returned readback completes once a device poll has run the `map_async` callback.
    fn start_typed_readback<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<TypedReadback<T>, Error> {
        self.validate_buffer_device(buffer)?;
        if buffer.access != GpuBufferAccess::ReadWrite {
            return Err(Error::PersistentBufferNotReadable);
        }
        if buffer.is_empty() {
            return Ok(TypedReadback {
                mapping: None,
                element: PhantomData,
            });
        }

        let byte_size = buffer.len as u64 * size_of::<T>() as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST persistent buffer readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST persistent buffer readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&buffer.raw, 0, &readback, 0, byte_size);
        self.queue.submit([encoder.finish()]);

        let mapped = Completion::new();
        let callback = mapped.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            // The callback is `FnOnce`, so it runs at most once; a late or repeated completion would be rejected by
            // the `Completion` anyway.
            let _ = callback.complete(result.map_err(|error| error.to_string()));
        });
        Ok(TypedReadback {
            mapping: Some(PendingMapping { readback, mapped }),
            element: PhantomData,
        })
    }

    /// [`HeadlessDevice::read_typed_buffer_timed`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(since = "0.2.0", note = "use `read_typed_buffer_timed`")]
    pub fn read_f32_buffer_timed(
        &self,
        buffer: &GpuBuffer<f32>,
    ) -> Result<(Vec<f32>, TransferTiming), Error> {
        self.read_typed_buffer_timed(buffer)
    }
}
