//! `gust::map::MapHost` for the GPU device (m7.a0): upload, zeroed outputs and readback through the device's buffer
//! pool, as the element-wise map calls of M7 need them.
//!
//! Native only: the trait's `readback` blocks, and the browser cannot block. The browser's map call (`map_async`) goes
//! through `GpuVec::to_vec_async` instead.

use gust::{GpuPod, map::MapHost};

use crate::{Error, GpuBuffer, HeadlessDevice, pool::Fill};

impl MapHost for HeadlessDevice {
    type Buffer<T: GpuPod> = GpuBuffer<T>;

    /// A pooled buffer holding `values`. The `max_buffer_size` pre-check and the out-of-memory scope apply to a
    /// fresh allocation.
    fn upload<T: GpuPod>(&self, values: &[T]) -> Result<GpuBuffer<T>, Error> {
        self.pooled_buffer("map upload", values.len(), Fill::Values(values))
    }

    /// A pooled buffer of `len` zeroed elements; a recycled one is cleared before any later submission reads it.
    fn zeroed<T: GpuPod>(&self, len: u32) -> Result<GpuBuffer<T>, Error> {
        self.pooled_buffer("map output", len as usize, Fill::Zeroed)
    }

    fn readback<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_pooled(buffer)
    }
}
