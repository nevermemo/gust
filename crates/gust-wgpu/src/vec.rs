//! `GpuVec<T>`: a vector that lives on a device (m7.a0).
//!
//! Residency belongs to buffers, not to functions: a `GpuVec` holds its device ([`DeviceHandle`]) and a pooled
//! allocation, hands out a `GpuBuffer` view for typed calls, and returns the allocation to the device's
//! [`BufferPool`](crate::BufferPool) when dropped. The allocation may be larger than the vector; every binding
//! exposes the logical length only.

use std::fmt;

use gust::{Access, GpuPod, LengthRule, Resource};

use crate::{
    BufferBinding, DeviceHandle, Error, GpuBuffer, HeadlessDevice, default_device, pool::Fill,
};

/// A vector of `T` resident on one device, taken from that device's buffer pool and returned to it on drop.
///
/// Create it with [`from_slice`](Self::from_slice) or [`zeroed`](Self::zeroed) (the
/// [default device](crate::default_device)) or their `_on` forms (a device you hold), read it back with
/// [`to_vec`](Self::to_vec) or [`to_vec_async`](Self::to_vec_async), and pass `&gpu_vec` where a typed call takes a
/// buffer: it is a [`Resource`] of [`HeadlessDevice`]. Its length is `u32`, the type of a dispatch's element count.
///
/// All `GpuVec` arguments of one call must share one device (compared by [`HeadlessDevice::id`]);
/// [`check_device`](Self::check_device) is the test, and a mismatch is [`Error::MixedDevices`]. A `GpuVec` is
/// `Send + Sync` on native; on `wasm32` its device handle is an `Rc`, so it is neither.
pub struct GpuVec<T: GpuPod> {
    buffer: GpuBuffer<T>,
    device: DeviceHandle,
}

impl<T: GpuPod> GpuVec<T> {
    /// Upload `values` to the default device.
    pub fn from_slice(values: &[T]) -> Result<Self, Error> {
        Self::from_slice_on(&default_device()?, values)
    }

    /// Upload `values` to `device`, into a pooled buffer; the logical range is written in full. A slice above
    /// `u32::MAX` elements, or above the device's `max_buffer_size`, is `Error::DeviceLimit`, and an allocation the
    /// adapter cannot serve is `Error::OutOfMemory` (native; the browser applies the pre-check only).
    pub fn from_slice_on(device: &DeviceHandle, values: &[T]) -> Result<Self, Error> {
        check_len(values.len())?;
        let buffer = device.pooled_buffer("GpuVec", values.len(), Fill::Values(values))?;
        Ok(Self {
            buffer,
            device: DeviceHandle::clone(device),
        })
    }

    /// A vector of `len` zeros on the default device.
    pub fn zeroed(len: u32) -> Result<Self, Error> {
        Self::zeroed_on(&default_device()?, len)
    }

    /// A vector of `len` zeros on `device`. A recycled allocation is cleared in a submission of its own before this
    /// returns, so any later read of the vector sees zeros.
    pub fn zeroed_on(device: &DeviceHandle, len: u32) -> Result<Self, Error> {
        let buffer = device.pooled_buffer("GpuVec", len as usize, Fill::Zeroed)?;
        Ok(Self {
            buffer,
            device: DeviceHandle::clone(device),
        })
    }

    /// The number of elements.
    pub fn len(&self) -> u32 {
        u32::try_from(self.buffer.len()).expect("a GpuVec is at most u32::MAX elements long")
    }

    /// Whether there are no elements.
    pub const fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The device this vector lives on.
    pub const fn device(&self) -> &DeviceHandle {
        &self.device
    }

    /// The id of [`device`](Self::device).
    pub fn device_id(&self) -> u64 {
        self.device.id()
    }

    /// Fail with [`Error::MixedDevices`] unless this vector lives on `device`.
    pub fn check_device(&self, device: &HeadlessDevice) -> Result<(), Error> {
        device.check_same_device(self.device.id())
    }

    /// The vector as the buffer a typed call takes. Its allocation may be larger than `len()` elements.
    pub const fn as_buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Copy the elements to the host. Blocks the calling thread until the device has finished the earlier work on
    /// the vector and the copy; native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        self.device.read_pooled(&self.buffer)
    }

    /// Copy the elements to the host without blocking: the copy and the mapping start when the future is first
    /// polled, and each poll polls the device once. On native the future completes only while something polls
    /// the device (see `HeadlessDevice::read_typed_buffer_async`); in the browser a plain `.await` suffices.
    ///
    /// Dropping the future after the mapping was requested is safe: the staging buffer stays with the mapping until
    /// its callback has fired and then returns to the pool; it is never reused while mapped.
    pub async fn to_vec_async(&self) -> Result<Vec<T>, Error> {
        self.device.read_pooled_async(&self.buffer).await
    }
}

/// A vector of more elements than a dispatch can count is a `DeviceLimit` error, like any other too-large request.
fn check_len(len: usize) -> Result<(), Error> {
    if u32::try_from(len).is_err() {
        return Err(Error::DeviceLimit {
            kernel: "GpuVec".to_owned(),
            limit: "element count (u32)",
            requested: len as u64,
            maximum: u64::from(u32::MAX),
        });
    }
    Ok(())
}

impl<T: GpuPod> Resource for GpuVec<T> {
    type Element = T;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Storage;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        length: LengthRule,
    ) -> BufferBinding<'resources> {
        self.buffer.bind(access, length)
    }
}

impl<T: GpuPod> fmt::Debug for GpuVec<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuVec")
            .field("len", &self.buffer.len())
            .field("device", &self.device.id())
            .finish()
    }
}
