//! The routed device handle without a GPU runtime (m9.h1; Unit tier, no adapter, no slangc): `gust` alone offers
//! `RoutedDevice<NoGpu>`. A kernel the macro built is `OnGpu` (the macro emits `Routed`; this file writes none for it),
//! so on a handle with no GPU it is a typed `Error::NoGpu`, never a panic, and leaves its output untouched. A CPU-routed
//! kernel runs on the same handle; the macro makes one only under a route, so `Triple` below is a hand-written kernel
//! that borrows the descriptor and the CPU twin of a macro kernel and says `OnCpu`. Expected values are worked by hand
//! (D26, D42), not read from the twin.

use gust::cpu::{CpuDevice, CpuJob, Error as CpuError};
use gust::routed::{Buffer, Error, NoGpu, OnCpu, OnGpu, Routed, RoutedDevice};
use gust::{Access, Call, Kernel, KernelDescriptor, LengthRule, Resource, Run};

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn triple_body(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 3.0;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn gpu_only(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i];
        }
    }
}

/// A CPU-routed kernel: the descriptor and the twin of `triple_body`, with the target a route would give it.
struct Triple;

impl Kernel for Triple {
    const DESCRIPTOR: KernelDescriptor = kernels::triple_body::DESCRIPTOR;
}

impl Routed for Triple {
    type Side = OnCpu;
}

impl Run<Triple, 2> for CpuDevice {
    fn submit_call(&self, call: &Call<'_, Self, Triple, 2>) -> Result<CpuJob, CpuError> {
        let twin =
            Call::<CpuDevice, kernels::triple_body, 2>::new_extent(call.extent(), *call.bindings());
        self.submit_call(&twin)
    }
}

type F32Buffer = Buffer<f32, NoGpu>;

/// The call site for a kernel with no generated `dispatch` (`Triple`): input read-only, output read-write.
fn triple(
    device: &RoutedDevice<NoGpu>,
    count: u32,
    input: &F32Buffer,
    out: &F32Buffer,
) -> Result<(), Error> {
    let call = Call::<RoutedDevice<NoGpu>, Triple, 2>::new(
        count,
        [
            input.bind(Access::ReadOnly, LengthRule::Shared),
            out.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    device.dispatch_call(&call)
}

/// Compile-time proof that the macro emitted `Routed` with the GPU side (no `impl Routed` is written for it here).
fn side_is_gpu<K: Routed<Side = OnGpu>>() {}

#[test]
fn the_macro_emits_routed_for_a_gpu_kernel() {
    side_is_gpu::<kernels::gpu_only>();
    side_is_gpu::<kernels::triple_body>();
}

#[test]
fn a_cpu_routed_kernel_runs_on_a_handle_without_a_gpu() {
    let device = RoutedDevice::<NoGpu>::cpu_only();
    let input = device.buffer(&[1.0f32, 2.0, 3.0]).unwrap();
    let mid = device.buffer(&[0.0f32; 3]).unwrap();
    let out = device.buffer(&[0.0f32; 3]).unwrap();
    triple(&device, 3, &input, &mid).unwrap();
    triple(&device, 3, &mid, &out).unwrap();
    assert_eq!(mid.read(&device).unwrap(), [3.0, 6.0, 9.0]);
    assert_eq!(out.read(&device).unwrap(), [9.0, 18.0, 27.0]);
    assert_eq!(
        input.read(&device).unwrap(),
        [1.0, 2.0, 3.0],
        "a read-only binding is not written"
    );
}

#[test]
fn a_gpu_routed_kernel_on_a_handle_without_a_gpu_is_a_typed_error_and_writes_nothing() {
    let device = RoutedDevice::<NoGpu>::cpu_only();
    let input = device.buffer(&[1.0f32, 2.0]).unwrap();
    let out = device.buffer(&[7.0f32, 8.0]).unwrap();
    let error = kernels::gpu_only::dispatch(&device, 2, &input, &out).unwrap_err();
    assert!(
        matches!(&error, Error::NoGpu { kernel } if kernel == "kernels::gpu_only"),
        "{error}"
    );
    assert!(error.to_string().contains("no GPU"), "{error}");
    assert_eq!(out.read(&device).unwrap(), [7.0, 8.0]);
    // The refusal is the handle's, so a CPU-routed call on the same buffers still works afterwards.
    triple(&device, 2, &input, &out).unwrap();
    assert_eq!(out.read(&device).unwrap(), [3.0, 6.0]);
}

#[test]
fn a_host_write_is_seen_by_the_next_call_and_a_bad_write_changes_nothing() {
    let device = RoutedDevice::<NoGpu>::cpu_only();
    let input = device.buffer(&[1.0f32, 1.0]).unwrap();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    input.write(&[5.0, 6.0]).unwrap();
    assert!(
        matches!(input.write(&[1.0]), Err(Error::Cpu(_))),
        "a write of another length is refused"
    );
    triple(&device, 2, &input, &out).unwrap();
    assert_eq!(out.read(&device).unwrap(), [15.0, 18.0]);
}

#[test]
fn a_buffer_of_another_handle_is_refused_by_the_cpu_device_check() {
    let device = RoutedDevice::<NoGpu>::cpu_only();
    let other = RoutedDevice::<NoGpu>::cpu_only();
    let input = device.buffer(&[1.0f32, 2.0]).unwrap();
    let out = device.buffer(&[0.0f32; 2]).unwrap();
    let foreign = other.buffer(&[9.0f32, 9.0]).unwrap();
    let error = triple(&device, 2, &foreign, &out).unwrap_err();
    assert!(
        matches!(error, Error::Cpu(CpuError::WrongDevice { .. })),
        "{error}"
    );
    assert_eq!(out.read(&device).unwrap(), [0.0, 0.0]);
    triple(&device, 2, &input, &out).unwrap();
    assert_eq!(out.read(&device).unwrap(), [3.0, 6.0]);
}
