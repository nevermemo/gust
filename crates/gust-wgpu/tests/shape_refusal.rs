//! What a `pixel` call is refused for, and with which error (roadmap MX, slice mx.p1b; Option S and D127).
//!
//! `tests/fixtures/pixel.rs` is a real expansion of `shape = pixel` kernels, and `pixel.rs` of this directory runs them.
//! Here every refusal is asserted with its variant and its fields, and each one comes before a shader is compiled and
//! before any pipeline exists (the pipeline cache is still cold afterwards). The twin refuses what a GPU refuses, so a
//! program that passes on `cpu` cannot fail on a GPU for these reasons.

#![cfg(not(target_arch = "wasm32"))]

include!("../../../tests/fixtures/pixel.rs");

use gust::{Extent, Independent, PIXEL_MAX_ELEMENTS, Shape, cpu::CpuDevice};
use gust_wgpu::{
    BufferBinding, BufferDispatch, Error, F32Binding, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    StagedGraph, U32Binding, render_wgpu_source,
};

/// A buffer of the layout an indirect launch reads (three `u32`s).
#[gust::gpu]
mod indirect {
    #[derive(Debug, PartialEq)]
    pub struct DispatchArgs {
        pub x: uint,
        pub y: uint,
        pub z: uint,
    }

    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn rewrite(id: SV_DispatchThreadID, mut args: RWStructuredBuffer<DispatchArgs>) {
        if id.x == 0u32 && 0u32 < args.len() {
            args[0u32].x = 1u32;
        }
    }
}

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

fn shade() -> &'static gust::KernelDescriptor {
    let shade = &pixel::MODULE_DESCRIPTOR.kernels[0];
    assert_eq!(shade.name, "shade");
    assert_eq!(shade.shape, Shape::Pixel);
    shade
}

/// The three buffers of one `shade` call of `n` elements: an input and the two outputs.
fn buffers(device: &HeadlessDevice, n: usize) -> (GpuBuffer<f32>, GpuBuffer<f32>, GpuBuffer<u32>) {
    (
        device.create_f32_buffer("input", &vec![0.0; n], GpuBufferAccess::ReadOnly),
        device.create_f32_buffer("scaled", &vec![0.0; n], GpuBufferAccess::ReadWrite),
        device.create_typed_buffer("flags", &vec![0u32; n], GpuBufferAccess::ReadWrite),
    )
}

fn assert_cold(device: &HeadlessDevice) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (0, 0, 0),
        "the refusal must precede compilation and the cache lookup"
    );
}

/// The `ShapeUnsupported` of a `pixel` kernel with `reason` containing `needle`.
fn assert_unsupported(error: &Error, needle: &str) {
    match error {
        Error::ShapeUnsupported {
            kernel,
            shape,
            reason,
        } => {
            assert_eq!(*shape, Shape::Pixel);
            assert!(kernel.contains("shade"), "{kernel}");
            assert!(
                reason.contains(needle),
                "`{reason}` should contain `{needle}`"
            );
            let text = error.to_string();
            assert!(text.contains("Pixel") && text.contains(reason), "{text}");
        }
        other => panic!("expected ShapeUnsupported ({needle}), got {other:?}"),
    }
}

/// n > 2^26 is `DeviceLimit { limit: "pixel_max_elements" }`, decided before the binding sizes.
#[test]
fn a_call_beyond_the_cap_is_the_named_device_limit() {
    let device = device();
    let n = PIXEL_MAX_ELEMENTS as usize + 1;
    let buffers = (
        device.create_zeroed_buffer::<f32>("input", n, GpuBufferAccess::ReadOnly),
        device.create_zeroed_buffer::<f32>("scaled", n, GpuBufferAccess::ReadWrite),
        device.create_zeroed_buffer::<u32>("flags", n, GpuBufferAccess::ReadWrite),
    );
    let (Ok(input), Ok(scaled), Ok(flags)) = buffers else {
        eprintln!(
            "BRANCH cap refusal: the adapter cannot allocate three buffers of 2^26 + 1 elements; the twin's unit test covers the rule"
        );
        return;
    };
    let error = pixel::shade::dispatch(&device, n as u32, &input, &scaled, &flags)
        .expect_err("a call over the cap must be refused");
    match error {
        Error::DeviceLimit {
            kernel,
            limit,
            requested,
            maximum,
        } => {
            assert!(kernel.contains("shade"), "{kernel}");
            assert_eq!(limit, "pixel_max_elements");
            assert_eq!(requested, u64::from(PIXEL_MAX_ELEMENTS) + 1);
            assert_eq!(maximum, u64::from(PIXEL_MAX_ELEMENTS));
        }
        other => panic!("expected the cap DeviceLimit, got {other:?}"),
    }
    assert_cold(&device);
}

/// A lowered `max_texture_dimension_2d` refuses the 8192-wide attachment, and an unlowered one does not.
#[test]
fn a_lowered_texture_dimension_refuses_the_attachment() {
    let device = device();
    let mut limits = device.limits().clone();
    assert!(limits.max_texture_dimension_2d >= 8192);
    limits.max_texture_dimension_2d = 4096;
    let device = device.with_limits(limits).expect("lowering is allowed");
    let (input, scaled, flags) = buffers(&device, 4);
    let error = pixel::shade::dispatch(&device, 4, &input, &scaled, &flags)
        .expect_err("an 8192-wide attachment does not fit a 4096 limit");
    match error {
        Error::DeviceLimit {
            kernel,
            limit,
            requested,
            maximum,
        } => {
            assert!(kernel.contains("shade"), "{kernel}");
            assert_eq!(limit, "max_texture_dimension_2d");
            assert_eq!((requested, maximum), (8192, 4096));
        }
        other => panic!("expected the texture-dimension DeviceLimit, got {other:?}"),
    }
    assert_cold(&device);
    // An empty call is not checked, as for a grid call.
    let (input, scaled, flags) = buffers(&device, 0);
    pixel::shade::dispatch(&device, 0, &input, &scaled, &flags).unwrap();
    assert_cold(&device);
}

/// A read-write output bound `Independent` is `ShapeIndependentOutput`, on the GPU and on the twin; an independent
/// input is fine.
#[test]
fn an_independent_output_is_refused_on_the_gpu_and_on_the_twin() {
    let device = device();
    let (input, scaled, flags) = buffers(&device, 4);
    let error = pixel::shade::dispatch(&device, 4, &input, Independent(&scaled), &flags)
        .expect_err("an independent output must be refused");
    match &error {
        Error::ShapeIndependentOutput { kernel, parameter } => {
            assert!(kernel.contains("shade"), "{kernel}");
            assert_eq!(parameter, "scaled");
        }
        other => panic!("expected ShapeIndependentOutput, got {other:?}"),
    }
    assert!(error.to_string().contains("independent length"), "{error}");
    // The hand-built spelling, `BufferBinding::independent_length`, is the same error.
    let bindings = [
        BufferBinding::read_only(&input),
        BufferBinding::read_write(&scaled),
        BufferBinding::read_write(&flags).independent_length(),
    ];
    match device
        .dispatch_buffers(shade(), 4, &bindings)
        .expect_err("an independent output must be refused")
    {
        Error::ShapeIndependentOutput { parameter, .. } => assert_eq!(parameter, "flags"),
        other => panic!("expected ShapeIndependentOutput, got {other:?}"),
    }
    assert_cold(&device);

    let cpu = CpuDevice::new();
    let (cpu_input, cpu_scaled, cpu_flags) = (
        cpu.buffer(&[0.0f32; 4]),
        cpu.buffer(&[0.0f32; 4]),
        cpu.buffer(&[0u32; 4]),
    );
    match pixel::shade::dispatch(&cpu, 4, &cpu_input, Independent(&cpu_scaled), &cpu_flags)
        .expect_err("the twin refuses an independent output too")
    {
        gust::cpu::Error::PixelIndependentOutput { kernel, parameter } => {
            assert!(kernel.contains("shade"), "{kernel}");
            assert_eq!(parameter, "scaled");
        }
        other => panic!("expected PixelIndependentOutput, got {other:?}"),
    }
}

/// A two- or three-dimensional extent is `ShapeUnsupported`, whatever the buffers hold.
#[test]
fn a_pixel_call_over_an_extent_is_refused() {
    let device = device();
    let (input, scaled, flags) = buffers(&device, 8);
    let bindings = [
        BufferBinding::read_only(&input),
        BufferBinding::read_write(&scaled),
        BufferBinding::read_write(&flags),
    ];
    for extent in [
        Extent::new(4, 2, 1),
        Extent::new(4, 1, 2),
        Extent::new(0, 2, 1),
    ] {
        let error = device
            .dispatch_batch(&[BufferDispatch::new_extent(shade(), extent, &bindings)])
            .expect_err("a pixel call is one-dimensional");
        assert_unsupported(&error, "one-dimensional");
    }
    assert_cold(&device);
    // The same buffers as one dimension of 8 are a call.
    device
        .dispatch_batch(&[BufferDispatch::new(shade(), 8, &bindings)])
        .unwrap();
}

/// An indirect launch of a `pixel` kernel is refused first, so it wins over the binding errors (these bindings are not
/// independent, which an indirect launch would also refuse).
#[test]
fn an_indirect_launch_is_refused_before_anything_else() {
    let device = device();
    let (input, scaled, flags) = buffers(&device, 4);
    let args = device
        .create_indirect_buffer("args", &[indirect::DispatchArgs { x: 1, y: 1, z: 1 }])
        .unwrap();
    let bindings = [
        BufferBinding::read_only(&input),
        BufferBinding::read_write(&scaled),
        BufferBinding::read_write(&flags),
    ];
    let mut graph = StagedGraph::new();
    graph.dispatch_indirect(BufferDispatch::new(shade(), 4, &bindings), &args, &[]);
    let error = device
        .execute_graph(&graph)
        .err()
        .expect("a pixel kernel has no indirect launch");
    assert_unsupported(&error, "indirect");
    assert_cold(&device);

    let call = pixel::shade::call(4, &input, &scaled, &flags);
    let mut typed = StagedGraph::new();
    typed.run_indirect(&call, &args);
    let error = device
        .execute_graph(&typed)
        .err()
        .expect("a pixel kernel has no indirect launch");
    assert_unsupported(&error, "indirect");
    assert_cold(&device);
}

/// The one-shot v1 functions run compute passes: a `pixel` kernel is refused by both, before any buffer is touched.
#[test]
fn the_one_shot_functions_refuse_a_pixel_kernel() {
    let device = device();
    let host = [0.0f32; 4];
    let flags = [0u32; 4];
    let error = device
        .dispatch_f32(
            shade(),
            4,
            &[
                F32Binding::ReadOnly(&host),
                F32Binding::ReadWrite(&host),
                F32Binding::ReadWrite(&host),
            ],
        )
        .expect_err("dispatch_f32 runs compute passes");
    assert_unsupported(&error, "one-shot");
    let error = device
        .dispatch_u32(
            shade(),
            4,
            &[
                U32Binding::ReadOnly(&flags),
                U32Binding::ReadWrite(&flags),
                U32Binding::ReadWrite(&flags),
            ],
        )
        .expect_err("dispatch_u32 runs compute passes");
    assert_unsupported(&error, "one-shot");
    assert_cold(&device);
}

/// The generated host snippet is a compute host; no pixel snippet exists in MX.
#[test]
fn the_generated_host_snippet_refuses_a_pixel_kernel() {
    let error = render_wgpu_source(shade()).expect_err("no pixel snippet exists");
    assert_unsupported(&error, "no pixel snippet");
}

/// The three buffers, run twice: the refusals above leave the device usable.
#[test]
fn a_refused_call_leaves_the_device_able_to_run_the_next_one() {
    let device = device();
    let (input, scaled, flags) = buffers(&device, 4);
    assert!(pixel::shade::dispatch(&device, 4, &input, Independent(&scaled), &flags).is_err());
    pixel::shade::dispatch(&device, 4, &input, &scaled, &flags).unwrap();
    assert_eq!(device.pipeline_cache_stats().entries, 1);
}
