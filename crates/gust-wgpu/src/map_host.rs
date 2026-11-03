//! `gust::map::MapHost` for the GPU device (m7.a0): upload, zeroed outputs and readback through the device's buffer
//! pool, as the element-wise map calls of M7 need them.
//!
//! The blocking `readback` is native only (the browser cannot block, D69). `readback_async` exists on both targets, so
//! the generic `map_on_async` of M7 is written over `D: MapHost` alone.

use gust::{
    GpuPod,
    map::{MapHost, Resident},
};

use crate::{Error, GpuBuffer, GpuVec, HeadlessDevice, pool::Fill};

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
