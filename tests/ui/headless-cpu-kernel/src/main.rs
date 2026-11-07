// A hand-written kernel that does not carry the marker `impl gust::target::GpuKernel for Probe {}` has no GPU form in
// this build (the state of a kernel the macro routes to the CPU, D137). Naming `HeadlessDevice` for it does not
// compile, and the error is at the call: the wording of `#[diagnostic::on_unimplemented]` on `GpuKernel` names the
// kernel. The control `headless-gpu-kernel` is not a fixture of its own: add the marker line and this compiles
// (`crates/gust-wgpu/tests/typed.rs` is that control, with the line on every hand-written kernel).

use gust::{
    Access, Arg, Call, GpuPod, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind, Resource,
    ResourceBinding, ResourceLayout, Run,
};
use gust_wgpu::{GpuBuffer, HeadlessDevice};

struct Probe;

impl Kernel for Probe {
    const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
        "ui",
        "probe",
        "ui_probe",
        [64, 1, 1],
        &[ParameterDescriptor {
            name: "p",
            rust_type: "float",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
        }],
        "",
    );
}

// The call the macro generates for a one-parameter kernel.
fn call<'r, R0>(element_count: u32, p: impl Into<Arg<'r, R0>>) -> Call<'r, R0::Device, Probe, 1>
where
    R0: Resource<Element = f32> + 'r,
{
    Call::new(element_count, [p.into().bind(Access::ReadWrite)])
}

fn run(device: &HeadlessDevice, buffer: &GpuBuffer<f32>) {
    let probe = call(1, buffer);
    device.dispatch_call(&probe).unwrap();
}

fn main() {}
