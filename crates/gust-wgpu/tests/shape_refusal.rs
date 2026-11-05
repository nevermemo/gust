//! A `pixel` kernel on `gust-wgpu` before the fragment path exists (roadmap MX, slice mx.p1a, review S2).
//!
//! `tests/fixtures/pixel.rs` is a real expansion of `shape = pixel` kernels. Until slice mx.p1b gives the crate a
//! fragment pass, a device refuses them with `Error::ShapeUnsupported`, before any shader is compiled and before any
//! pipeline exists; without the refusal `compile_reflected` would build a fragment entry and a compute pipeline over it
//! would be a wgpu validation error, which panics. The test creates a device and the buffers of the call, and nothing
//! else runs (it is in the Gpu suite for that reason; the refusal itself needs no shader).

include!("../../../tests/fixtures/pixel.rs");

use gust::Shape;
use gust_wgpu::{BufferBinding, Error, GpuBufferAccess, HeadlessDevice};

#[test]
fn a_pixel_kernel_is_refused_with_its_shape_before_any_pipeline() {
    let device = HeadlessDevice::new().expect("the test needs an adapter");
    let shade = &pixel::MODULE_DESCRIPTOR.kernels[0];
    assert_eq!(shade.name, "shade");
    assert_eq!(shade.shape, Shape::Pixel);
    let input = device.create_f32_buffer("input", &[0.0; 4], GpuBufferAccess::ReadOnly);
    let scaled = device.create_f32_buffer("scaled", &[0.0; 4], GpuBufferAccess::ReadWrite);
    let flags = device.create_typed_buffer("flags", &[0u32; 4], GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_buffers(
            shade,
            4,
            &[
                BufferBinding::read_only(&input),
                BufferBinding::read_write(&scaled),
                BufferBinding::read_write(&flags),
            ],
        )
        .expect_err("a pixel kernel must be refused");
    match &error {
        Error::ShapeUnsupported { kernel, shape } => {
            assert_eq!(*shape, Shape::Pixel);
            assert!(kernel.contains("shade"), "{kernel}");
        }
        other => panic!("expected ShapeUnsupported, got {other:?}"),
    }
    assert!(error.to_string().contains("mx.p1b"), "{error}");
}
