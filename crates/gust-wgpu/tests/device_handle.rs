//! The routed device handle on a real adapter (m9.h1; Gpu tier): one `gust::routed::RoutedDevice` whose calls run on the
//! target each kernel is routed to. `on_gpu::scale2` is a macro kernel (the macro emits `Routed` with the GPU side, so no
//! `impl Routed` is written for it). The macro builds a CPU-routed kernel only under a route, so `Plus1` below is a
//! hand-written kernel that borrows the descriptor and the CPU twin of a macro kernel and says `OnCpu` (the routing
//! check of xtask builds the real thing). The call sites are the same source whatever a kernel's side is. Expected values
//! are worked by hand (D26, D42); the GPU values come from the adapter, never from the twin.

use gust::cpu::{CpuDevice, CpuJob, Error as CpuError};
use gust::routed::{Error, OnCpu, Routed, RoutedBuffer, RoutedDevice};
use gust::{Access, Call, Kernel, KernelDescriptor, LengthRule, Resource, Run};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod on_gpu {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scale2(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}

#[gust::gpu]
mod twin {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn plus1_body(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] + 1.0;
        }
    }
}

/// A CPU-routed kernel: the descriptor and the twin of `plus1_body`, with the target a route would give it.
struct Plus1;

impl Kernel for Plus1 {
    const DESCRIPTOR: KernelDescriptor = twin::plus1_body::DESCRIPTOR;
}

impl Routed for Plus1 {
    type Side = OnCpu;
}

impl Run<Plus1, 2> for CpuDevice {
    fn submit_call(&self, call: &Call<'_, Self, Plus1, 2>) -> Result<CpuJob, CpuError> {
        let twin =
            Call::<CpuDevice, twin::plus1_body, 2>::new_extent(call.extent(), *call.bindings());
        self.submit_call(&twin)
    }
}

type F32Buffer = RoutedBuffer<f32, HeadlessDevice>;

fn plus1(
    device: &RoutedDevice<HeadlessDevice>,
    count: u32,
    input: &F32Buffer,
    out: &F32Buffer,
) -> Result<(), Error> {
    let call = Call::<RoutedDevice<HeadlessDevice>, Plus1, 2>::new(
        count,
        [
            input.bind(Access::ReadOnly, LengthRule::Shared),
            out.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    device.dispatch_call(&call)
}

fn gpu_handle() -> RoutedDevice<HeadlessDevice> {
    RoutedDevice::with_gpu(HeadlessDevice::new().expect("a Vulkan compute adapter"))
}

/// The call-site code of the chain: it names no target, only the handle.
fn pipeline(device: &RoutedDevice<HeadlessDevice>) -> Result<[Vec<f32>; 3], Error> {
    let input = device.buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
    let mid = device.buffer(&[0.0f32; 4])?;
    let out = device.buffer(&[0.0f32; 4])?;
    let fin = device.buffer(&[0.0f32; 4])?;
    on_gpu::scale2::dispatch(device, 4, &input, &mid)?; // GPU
    plus1(device, 4, &mid, &out)?; // CPU, reads what the GPU wrote
    on_gpu::scale2::dispatch(device, 4, &out, &fin)?; // GPU, reads what the CPU wrote
    Ok([mid.read(device)?, out.read(device)?, fin.read(device)?])
}

#[test]
fn a_chain_alternates_targets_on_one_handle() {
    let device = gpu_handle();
    let [mid, out, fin] = pipeline(&device).expect("pipeline");
    assert_eq!(mid, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(out, [3.0, 5.0, 7.0, 9.0]);
    assert_eq!(fin, [6.0, 10.0, 14.0, 18.0]);
}

#[test]
fn a_handle_without_a_gpu_runs_the_cpu_kernel_and_refuses_the_gpu_one_with_a_typed_error() {
    let device = RoutedDevice::<HeadlessDevice>::cpu_only();
    let input = device.buffer(&[1.0f32, 2.0]).unwrap();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    plus1(&device, 2, &input, &out).expect("cpu kernel");
    assert_eq!(out.read(&device).unwrap(), [2.0, 3.0]);
    let refused = on_gpu::scale2::dispatch(&device, 2, &input, &out).unwrap_err();
    assert!(
        matches!(&refused, Error::NoGpu { kernel } if kernel == "on_gpu::scale2"),
        "{refused}"
    );
    assert_eq!(
        out.read(&device).unwrap(),
        [2.0, 3.0],
        "a refused call writes nothing"
    );
}

#[test]
fn a_host_write_reaches_the_gpu_and_a_gpu_result_comes_back_by_read() {
    let device = gpu_handle();
    let input = device.buffer(&[1.0f32, 1.0]).unwrap();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    on_gpu::scale2::dispatch(&device, 2, &input, &out).unwrap();
    assert_eq!(out.read(&device).unwrap(), [2.0, 2.0]);
    // The host write makes the host the only valid copy; the next GPU call must see it, not the older device copy.
    input.write(&[5.0, 6.0]).unwrap();
    on_gpu::scale2::dispatch(&device, 2, &input, &out).unwrap();
    assert_eq!(out.read(&device).unwrap(), [10.0, 12.0]);
    // The device wrote `out`, so a CPU kernel reading it first copies it down.
    let after = device.buffer(&[0.0f32; 2]).unwrap();
    plus1(&device, 2, &out, &after).unwrap();
    assert_eq!(after.read(&device).unwrap(), [11.0, 13.0]);
}

#[test]
fn a_buffer_of_another_handle_is_refused_by_each_side_s_own_device_check() {
    let device = gpu_handle();
    let other = gpu_handle();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    let foreign = other.buffer(&[9.0f32, 9.0]).unwrap();
    // GPU side: gust-wgpu refuses a buffer of another device.
    let error = on_gpu::scale2::dispatch(&device, 2, &foreign, &out).unwrap_err();
    match &error {
        Error::Gpu(inner) => assert!(
            matches!(
                inner.downcast_ref::<gust_wgpu::Error>(),
                Some(gust_wgpu::Error::ForeignPersistentBuffer)
            ),
            "{error}"
        ),
        other => panic!("expected the GPU side's refusal, got {other}"),
    }
    // CPU side: the CPU device refuses a buffer of another instance.
    let error = plus1(&device, 2, &foreign, &out).unwrap_err();
    assert!(
        matches!(error, Error::Cpu(CpuError::WrongDevice { .. })),
        "{error}"
    );
    assert_eq!(out.read(&device).unwrap(), [0.0, 0.0]);
}

#[test]
fn explicit_device_calls_still_work_beside_the_handle() {
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let input = device.create_typed_buffer("in", &[1.0f32, 2.0], GpuBufferAccess::ReadWrite);
    let out = device.create_typed_buffer("out", &[0.0f32; 2], GpuBufferAccess::ReadWrite);
    on_gpu::scale2::dispatch(&device, 2, &input, &out).unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), [2.0, 4.0]);
}

fn foreign_kernel(error: &Error) -> Option<&str> {
    match error {
        Error::ForeignBuffer { kernel } => kernel.as_deref(),
        _ => None,
    }
}

#[test]
fn a_buffer_the_handle_cannot_move_is_a_foreign_buffer_on_all_three_paths() {
    let device = gpu_handle();
    let cpu_only = RoutedDevice::<HeadlessDevice>::cpu_only();
    let out = device.buffer(&[0.0f32; 2]).unwrap();

    // 1. A buffer of a handle without a GPU, written on the host (so the GPU would need an upload), used by a GPU kernel.
    let made_without_gpu = cpu_only.buffer(&[1.0f32, 2.0]).unwrap();
    made_without_gpu.write(&[3.0, 4.0]).unwrap();
    let error = on_gpu::scale2::dispatch(&device, 2, &made_without_gpu, &out).unwrap_err();
    assert_eq!(foreign_kernel(&error), Some("on_gpu::scale2"), "{error}");

    // 2. The same kind of buffer unwritten: valid on both sides, but it has no device copy to bind.
    let unwritten = cpu_only.buffer(&[1.0f32, 2.0]).unwrap();
    let error = on_gpu::scale2::dispatch(&device, 2, &unwritten, &out).unwrap_err();
    assert_eq!(foreign_kernel(&error), Some("on_gpu::scale2"), "{error}");

    // 3. A buffer the GPU wrote, read through a handle that has no GPU.
    let input = device.buffer(&[1.0f32, 2.0]).unwrap();
    on_gpu::scale2::dispatch(&device, 2, &input, &out).unwrap();
    let error = out.read(&cpu_only).unwrap_err();
    assert!(
        matches!(error, Error::ForeignBuffer { kernel: None }),
        "{error}"
    );
    // Through its own handle it reads back what the GPU wrote.
    assert_eq!(out.read(&device).unwrap(), [2.0, 4.0]);
}

/// A CPU-routed kernel that writes its first output element and then panics (D39).
struct Boom;

impl Kernel for Boom {
    const DESCRIPTOR: KernelDescriptor = twin::plus1_body::DESCRIPTOR;
}

impl Routed for Boom {
    type Side = OnCpu;
}

impl Run<Boom, 2> for CpuDevice {
    fn submit_call(&self, call: &Call<'_, Self, Boom, 2>) -> Result<CpuJob, CpuError> {
        let [_, out] = *call.bindings();
        let mut out = gust::cpu::write::<f32>(out, "out")?;
        out[0] = 99.0;
        panic!("a kernel body failed part-way");
    }
}

#[test]
fn after_a_cpu_kernel_panics_part_way_the_gpu_copy_agrees_with_the_host_copy() {
    let device = gpu_handle();
    let input = device.buffer(&[1.0f32, 2.0]).unwrap();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    let call = Call::<RoutedDevice<HeadlessDevice>, Boom, 2>::new(
        2,
        [
            input.bind(Access::ReadOnly, LengthRule::Shared),
            out.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = device.dispatch_call(&call);
    }));
    assert!(caught.is_err(), "the kernel panicked");
    // The host holds the partial write. A GPU kernel reading `out` must see it, not the older device copy [0, 0].
    let doubled = device.buffer(&[0.0f32; 2]).unwrap();
    on_gpu::scale2::dispatch(&device, 2, &out, &doubled).unwrap();
    assert_eq!(doubled.read(&device).unwrap(), [198.0, 0.0]);
    assert_eq!(out.read(&device).unwrap(), [99.0, 0.0]);
}
