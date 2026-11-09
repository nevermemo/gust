//! `HeadlessDevice` as the GPU side of `gust::routed::RoutedDevice` (m9.h1).

use gust::GpuPod;
use gust::routed::GpuBackend;

use crate::{Error, GpuBuffer, GpuBufferAccess, HeadlessDevice};

impl GpuBackend for HeadlessDevice {
    type Buffer<T: GpuPod> = GpuBuffer<T>;

    fn create_buffer<T: GpuPod>(&self, values: &[T]) -> Result<GpuBuffer<T>, Error> {
        self.try_create_typed_buffer("routed", values, GpuBufferAccess::ReadWrite)
    }

    fn write_buffer<T: GpuPod>(&self, buffer: &GpuBuffer<T>, values: &[T]) -> Result<(), Error> {
        self.write_typed_buffer(buffer, values)
    }

    fn read_buffer<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_typed_buffer(buffer)
    }
}
