use crate::{
    Error, GpuBuffer, GpuBufferAccess, GpuUniform, HeadlessDevice, TransferTiming,
    assert_pod_matches_storage_layout, check_uniform_type,
};
use gust::{GpuPod, pod_slice_as_bytes};
use std::marker::PhantomData;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

impl HeadlessDevice {
    /// Allocate and initialize a persistent storage buffer for any GPU POD type.
    ///
    /// The signature is infallible, so GUST does not check `max_buffer_size` here; an
    /// allocation above it is wgpu's to reject (a panic). Launch limits (workgroup shape, workgroup
    /// count, storage buffer count and binding size) are checked at dispatch time. Use
    /// [`try_create_typed_buffer`](Self::try_create_typed_buffer) or
    /// [`create_zeroed_buffer`](Self::create_zeroed_buffer) for a request that may be too large.
    pub fn create_typed_buffer<T: GpuPod>(
        &self,
        label: &str,
        values: &[T],
        access: GpuBufferAccess,
    ) -> GpuBuffer<T> {
        assert_pod_matches_storage_layout::<T>();
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if access == GpuBufferAccess::ReadWrite {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
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
        GpuBuffer {
            raw,
            len: values.len(),
            access,
            layout: T::LAYOUT,
            device_id: self.id,
            indirect: false,
            lease: None,
            marker: PhantomData,
        }
    }

    /// [`HeadlessDevice::create_typed_buffer`] for `f32` values.
    #[deprecated(since = "0.2.0", note = "use `create_typed_buffer`")]
    pub fn create_f32_buffer(
        &self,
        label: &str,
        values: &[f32],
        access: GpuBufferAccess,
    ) -> GpuBuffer<f32> {
        self.create_typed_buffer(label, values, access)
    }

    /// Allocate a uniform (a constant buffer of one `T`) holding `value`.
    ///
    /// `T` must be a padding-free `Std140V1` type with Rust's size and alignment: scalars, 2- and 4-wide vectors, 4x4
    /// matrices and structs of them whose fields fill a multiple of 16 bytes. Anything else is an
    /// `Error::InvalidUniformType` whose message names the fix (spell the padding out as fields), because a padded
    /// struct would upload uninitialised bytes. The buffer is created with `UNIFORM | COPY_DST | COPY_SRC`.
    pub fn create_uniform<T: GpuPod>(
        &self,
        label: &str,
        value: &T,
    ) -> Result<GpuUniform<T>, Error> {
        check_uniform_type::<T>()?;
        let raw = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: pod_slice_as_bytes(std::slice::from_ref(value)),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        Ok(GpuUniform {
            raw,
            layout: T::LAYOUT,
            device_id: self.id,
            marker: PhantomData,
        })
    }

    /// Replace the value of a uniform with `Queue::write_buffer`: the write is ordered after every earlier submission
    /// and before the next one, so work already submitted sees the old value. Every dispatch of one later submission
    /// (a `submit_batch` or a graph) sees the last value written, whatever the order they were recorded in; use
    /// `StagedGraph::write_uniform`, which encodes a copy in the command stream, when an update must fall between two
    /// dispatches of one submission.
    pub fn write_uniform<T: GpuPod>(
        &self,
        uniform: &GpuUniform<T>,
        value: &T,
    ) -> Result<(), Error> {
        if uniform.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        self.queue.write_buffer(
            &uniform.raw,
            0,
            pod_slice_as_bytes(std::slice::from_ref(value)),
        );
        Ok(())
    }

    /// Replace the complete contents of an existing persistent POD buffer.
    pub fn write_typed_buffer<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
        values: &[T],
    ) -> Result<(), Error> {
        self.validate_buffer_device(buffer)?;
        if buffer.len != values.len() {
            return Err(Error::PersistentBufferLength {
                expected: buffer.len,
                actual: values.len(),
            });
        }
        if !values.is_empty() {
            self.queue
                .write_buffer(&buffer.raw, 0, pod_slice_as_bytes(values));
        }
        Ok(())
    }

    /// [`HeadlessDevice::write_typed_buffer`] for an `f32` buffer.
    #[deprecated(since = "0.2.0", note = "use `write_typed_buffer`")]
    pub fn write_f32_buffer(&self, buffer: &GpuBuffer<f32>, values: &[f32]) -> Result<(), Error> {
        self.write_typed_buffer(buffer, values)
    }

    /// Replace a persistent POD buffer and wait until the upload has completed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_typed_buffer_timed<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
        values: &[T],
    ) -> Result<TransferTiming, Error> {
        let started = Instant::now();
        self.write_typed_buffer(buffer, values)?;
        if !values.is_empty() {
            let submission = self.queue.submit([]);
            self.wait_for_submission(submission)?;
        }
        Ok(TransferTiming {
            bytes: values.len() as u64 * size_of::<T>() as u64,
            elapsed: started.elapsed(),
        })
    }

    /// [`HeadlessDevice::write_typed_buffer_timed`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    #[deprecated(since = "0.2.0", note = "use `write_typed_buffer_timed`")]
    pub fn write_f32_buffer_timed(
        &self,
        buffer: &GpuBuffer<f32>,
        values: &[f32],
    ) -> Result<TransferTiming, Error> {
        self.write_typed_buffer_timed(buffer, values)
    }
}
