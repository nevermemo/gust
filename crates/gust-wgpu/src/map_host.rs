//! `gust::map::MapHost` for the GPU device (m7.a0): upload, zeroed outputs and readback through the device's buffer
//! pool, as the element-wise map calls of M7 need them.
//!
//! The blocking `readback` is native only (the browser cannot block, D69). `readback_async` exists on both targets, so
//! the generic `map_on_async` of M7 is written over `D: MapHost` alone.

use core::future::Future;
use core::ops::Deref;

use gust::{
    Access, Call, GpuPod, Kernel, LengthRule, Resource, Run,
    map::{MapHost, Resident},
};

use crate::{BufferDispatch, Error, GpuBuffer, GpuVec, HeadlessDevice, pool::Fill};

/// A resident buffer is a map argument as it is (`&GpuVec<T>` binds without a copy); the trait gives the generic
/// map call its length. It lives here because `GpuVec` and `GpuBuffer` do (m7.a1).
impl<T: GpuPod> Resident for GpuVec<T> {
    fn resident_len(&self) -> usize {
        self.len() as usize
    }
}

impl<T: GpuPod> Resident for GpuBuffer<T> {
    fn resident_len(&self) -> usize {
        self.len()
    }
}

impl MapHost for HeadlessDevice {
    type Buffer<T: GpuPod> = GpuBuffer<T>;

    /// A pooled buffer holding `values`. The `max_buffer_size` pre-check and, on native, the out-of-memory scope apply
    /// to a fresh allocation.
    fn upload<T: GpuPod>(&self, values: &[T]) -> Result<GpuBuffer<T>, Error> {
        self.pooled_buffer("map upload", values.len(), Fill::Values(values))
    }

    /// A pooled buffer of `len` zeroed elements; a recycled one is cleared before any later submission reads it.
    fn zeroed<T: GpuPod>(&self, len: u32) -> Result<GpuBuffer<T>, Error> {
        self.pooled_buffer("map output", len as usize, Fill::Zeroed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn readback<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_pooled(buffer)
    }

    async fn readback_async<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_pooled_async(buffer).await
    }

    /// The dispatch, the clear of a recycled output and the read-back copy in one command buffer: one counted queue
    /// submission ([`HeadlessDevice::submit_batch_read`]). A zeroed output is an
    /// [`uncleared_buffer`](HeadlessDevice::uncleared_buffer) that the submission clears itself; an output that starts
    /// with values is uploaded with `write_buffer`, which is no submission. The call is checked as `submit_call`
    /// checks it (the batch validates each binding against the kernel). The zeroed output carries the pool label
    /// `"batch output"`, so an oversize task's `DeviceLimit` error names that label.
    fn start_read<'t, H, T, K, C, const N: usize>(
        device: H,
        initial: Option<&[T]>,
        len: u32,
        context: C,
        build: impl for<'o> FnOnce(&'o C, &'o GpuBuffer<T>) -> Call<'o, Self, K, N>,
    ) -> Result<impl Future<Output = Result<Vec<T>, Error>> + 't, Error>
    where
        H: Deref<Target = Self> + 't,
        T: GpuPod + 't,
        K: Kernel,
        Self: Run<K, N> + 't,
        GpuBuffer<T>: 't,
    {
        let (output, clear) = match initial {
            Some(values) => (device.upload(values)?, false),
            None => (device.uncleared_buffer::<T>(len as usize)?, true),
        };
        let pending = {
            let call = build(&context, &output);
            let dispatch =
                BufferDispatch::new_extent(call.kernel(), call.extent(), call.bindings());
            if clear {
                let target = output.bind(Access::ReadWrite, LengthRule::Shared);
                device.submit_batch_read(&[dispatch], &[target], &output)?
            } else {
                device.submit_batch_read(&[dispatch], &[], &output)?
            }
        };
        Ok(async move {
            let values = pending.finish(&device).await;
            drop(output);
            values
        })
    }

    /// Resident buffers of two devices in one call are `MixedDevices`, as the map error names it.
    fn map_error(error: Error) -> gust::map::Error {
        match error {
            // `MixedDevices` is `GpuVec::check_device`'s; a dispatch reports a buffer of another device as foreign.
            Error::MixedDevices { .. } | Error::ForeignPersistentBuffer => {
                gust::map::Error::MixedDevices
            }
            other => gust::map::Error::device(other),
        }
    }
}

/// Compile-time proof, on every target `check-web` builds, that a map call's whole device surface (upload, zeroed
/// output, asynchronous readback) is reachable through `D: MapHost` alone. The web profile's `map_on_async` (m7.a1) is
/// this function plus a dispatch.
#[allow(dead_code)]
async fn upload_and_read_back<D: MapHost>(
    device: &D,
    values: &[u32],
) -> Result<Vec<u32>, D::Error> {
    let input = device.upload(values)?;
    let _output = device.zeroed::<u32>(values.len() as u32)?;
    device.readback_async(&input).await
}
