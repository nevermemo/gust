//! The GPU side of capture (slice g2.cap): copy a call's bound buffers to the host so that the generated `capture` of a
//! kernel can replay the call on its CPU twin. The device-neutral API (`Captured`, `run_element`, `run_until`) is
//! [`gust::capture`]; this file implements its [`CaptureDevice`] for [`HeadlessDevice`].
//!
//! A buffer can be copied when it was created with `COPY_SRC`: every read-write buffer, `GpuVec`, pool buffer, indirect
//! buffer and uniform is, and a read-only storage buffer is when the device was opened with
//! [`DeviceOptions::capture`](crate::DeviceOptions::capture). Any other is [`Error::PersistentBufferNotReadable`],
//! and the error message of the capture names that option. Native only: the copy blocks, like `read_typed_buffer`.

use gust::capture::{BufferCopy, CaptureDevice};
use gust::{Extent, GpuPod, KernelDescriptor, LengthRule};

use crate::completion::blocking::block_on_mapping;
use crate::{BufferBinding, Error, HeadlessDevice};

/// What to say when a read-only buffer cannot be captured.
const NOT_CAPTURABLE_HINT: &str = "open the device with `DeviceOptions::default().with_capture(true)` so read-only \
     buffers can be captured, or create the buffer with `GpuBufferAccess::ReadWrite`";

impl CaptureDevice for HeadlessDevice {
    /// The dispatch's own validation (`validate_dispatch_extent`: kind, layout, device, access, length, aliasing).
    fn check_call(
        &self,
        kernel: &KernelDescriptor,
        extent: Extent,
        bindings: &[BufferBinding<'_>],
    ) -> Result<(), Error> {
        crate::validate_dispatch_extent(
            self.id,
            self.float32_filterable(),
            kernel,
            extent,
            bindings,
            &[],
        )
        .map(|_| ())
    }

    fn capture_buffer<T: GpuPod>(
        &self,
        binding: &BufferBinding<'_>,
    ) -> Result<BufferCopy<T>, Error> {
        if binding.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        let length = if binding.independent_length {
            LengthRule::Independent
        } else {
            LengthRule::Shared
        };
        let raw = binding.buffer();
        if !raw.usage().contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(Error::PersistentBufferNotReadable);
        }
        // `len` is the logical length: a pooled buffer's allocation can be larger.
        let readback = self.start_raw_readback::<T>(raw, binding.len);
        readback.wait(&self.device)?;
        let values = block_on_mapping(readback.finish(&self.device))?;
        Ok(BufferCopy { values, length })
    }

    fn hint(error: &Error) -> Option<&'static str> {
        matches!(error, Error::PersistentBufferNotReadable).then_some(NOT_CAPTURABLE_HINT)
    }
}
