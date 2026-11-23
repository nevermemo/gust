//! The GPU side of capture (slice g2.cap): copy a call's bound buffers to the host so that the generated `capture` of a
//! kernel can replay the call on its CPU twin. The device-neutral API (`Captured`, `run_element`, `run_until`) is
//! [`gust::capture`]; this file implements its [`CaptureDevice`] for [`HeadlessDevice`].
//!
//! A buffer can be copied when it was created with `COPY_SRC`: every read-write buffer, `GpuVec`, pool buffer, indirect
//! buffer and uniform is, and a read-only storage buffer is when the device was opened with
//! [`DeviceOptions::capture`](crate::DeviceOptions::capture). Any other is [`Error::PersistentBufferNotReadable`],
//! and the error message of the capture names that option. Native only: the copy blocks, like `read_typed_buffer`.

use std::marker::PhantomData;

use gust::GpuPod;
use gust::LengthRule;
use gust::ParameterKind;
use gust::capture::{BufferCopy, CaptureDevice};

use crate::completion::blocking::block_on_mapping;
use crate::{BufferBinding, Completion, Error, HeadlessDevice, PendingMapping, TypedReadback};

/// What to say when a read-only buffer cannot be captured.
const NOT_CAPTURABLE_HINT: &str = "open the device with `DeviceOptions::default().with_capture(true)` so read-only \
     buffers can be captured, or create the buffer with `GpuBufferAccess::ReadWrite`";

impl CaptureDevice for HeadlessDevice {
    fn capture_buffer<T: GpuPod>(
        &self,
        binding: &BufferBinding<'_>,
    ) -> Result<BufferCopy<T>, Error> {
        if binding.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        if binding.texture.is_some() {
            return Err(Error::UnsupportedParameter {
                name: "(texture binding)".to_owned(),
                kind: ParameterKind::Texture,
            });
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
        let byte_size = binding.len as u64 * size_of::<T>() as u64;
        if byte_size == 0 {
            return Ok(BufferCopy {
                values: Vec::new(),
                length,
            });
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST capture readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST capture readback encoder"),
            });
        encoder.copy_buffer_to_buffer(raw, 0, &readback, 0, byte_size);
        self.queue.submit([encoder.finish()]);
        let mapped = Completion::new();
        let callback = mapped.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = callback.complete(result.map_err(|error| error.to_string()));
        });
        let pending = TypedReadback::<T> {
            mapping: Some(PendingMapping { readback, mapped }),
            element: PhantomData,
        };
        pending.wait(&self.device)?;
        let values = block_on_mapping(pending.finish(&self.device))?;
        Ok(BufferCopy { values, length })
    }

    fn hint(error: &Error) -> Option<&'static str> {
        matches!(error, Error::PersistentBufferNotReadable).then_some(NOT_CAPTURABLE_HINT)
    }
}
