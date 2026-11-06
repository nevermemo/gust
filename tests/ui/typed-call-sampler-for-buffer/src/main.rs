// A sampler where the kernel has a storage buffer is a compile error in the typed call: a sampler is a `SamplerArg`,
// not a `Resource`, so it cannot fill the output parameter, whose type must be a `Resource` of kind `Storage`. The
// kernel and its `blur_call` are written by hand, as `#[gpu]` cannot declare a texture parameter before slice mx.t3.
// The output is taken as a plain `&R2` here: through the macro's `impl Into<Arg<_>>` the same mistake is reported
// twice (once for the conversion and once for the `Resource` bound), and a fixture pins exactly one error.
// expected.txt pins the code and the line of the call.
#![allow(dead_code)]
use std::{marker::PhantomData, time::Duration};

use gust::{
    Access, Arg, Call, Device, GpuPod, Job, JobStatus, Kernel, KernelDescriptor, LengthRule,
    ParameterDescriptor, ParameterKind, Resource, ResourceBinding, Run, SampledCall, SamplerArg,
    SamplerDevice, TypedDevice,
};

struct Mock;
struct Done;

impl Job for Done {
    type Error = ();
    fn status(&self) -> Result<JobStatus, ()> {
        Ok(JobStatus::Complete)
    }
    fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> {
        Ok(JobStatus::Complete)
    }
    fn wait(&self) -> Result<(), ()> {
        Ok(())
    }
}

impl Device for Mock {
    type Error = ();
    type Dispatch<'r> = ();
    type Job<'d> = Done;
    fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> {
        Ok(Done)
    }
}

impl TypedDevice for Mock {
    type Binding<'r> = (Access, LengthRule);
}

impl SamplerDevice for Mock {
    type SamplerBinding<'r> = ();
}

struct Buffer<T>(PhantomData<T>);

impl<T: GpuPod> Resource for Buffer<T> {
    type Element = T;
    type Device = Mock;
    type Kind = ::gust::kind::Storage;
    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> (Access, LengthRule) {
        (access, length)
    }
}

// A texture: a resource of kind `Texture`, with the element type of its texels.
struct Tex<T>(PhantomData<T>);

impl<T: GpuPod> Resource for Tex<T> {
    type Element = T;
    type Device = Mock;
    type Kind = ::gust::kind::Texture;
    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> (Access, LengthRule) {
        (access, length)
    }
}

// A sampler: a `SamplerArg`, not a `Resource`.
struct Smp;

impl SamplerArg for Smp {
    type Device = Mock;
    fn bind(&self) {}
}

impl<K: Kernel, const N: usize> Run<K, N> for Mock {
    fn submit_call<'d>(&'d self, _: &Call<'_, Self, K, N>) -> Result<Done, ()> {
        Ok(Done)
    }
}

// A kernel with a texture, a sampler and an output buffer, written by hand: `#[gpu]` cannot declare a texture
// parameter before slice mx.t3, but the typed call's bounds below are the ones the macro will emit.
struct Blur;

impl Kernel for Blur {
    const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
        "ui",
        "blur",
        "ui_blur",
        [64, 1, 1],
        &[
            ParameterDescriptor {
                name: "tex",
                rust_type: "Texture2D<float>",
                kind: ParameterKind::Texture,
                access: Access::ReadOnly,
                resource_layout: None,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
            },
            ParameterDescriptor {
                name: "samp",
                rust_type: "Sampler",
                kind: ParameterKind::Sampler,
                access: Access::NotApplicable,
                resource_layout: None,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 1,
                }),
            },
            ParameterDescriptor {
                name: "out",
                rust_type: "float",
                kind: ParameterKind::Storage,
                access: Access::ReadWrite,
                resource_layout: None,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 2,
                }),
            },
        ],
        "",
    );
}
impl gust::target::GpuKernel for Blur {}

// The call the macro will generate for it: the texture is a `Texture`-kind resource, the sampler a `SamplerArg` of
// the same device, the output a `Storage`-kind resource.
fn blur_call<'r, R0, S1, R2>(
    element_count: u32,
    tex: impl Into<Arg<'r, R0>>,
    samp: &'r S1,
    out: &'r R2,
) -> SampledCall<'r, R0::Device, Blur, 2, 1>
where
    R0: Resource<Element = f32, Kind = ::gust::kind::Texture> + 'r,
    S1: SamplerArg<Device = R0::Device>,
    R2: Resource<Element = f32, Device = R0::Device, Kind = ::gust::kind::Storage> + 'r,
    R0::Device: SamplerDevice,
{
    Call::new_with_samplers(
        element_count,
        [
            tex.into().bind(Access::ReadOnly),
            Arg::from(out).bind(Access::ReadWrite),
        ],
        [samp.bind()],
    )
}

fn main() {
    let texture = Tex::<f32>(PhantomData);
    let sampler = Smp;
    let not_a_buffer = Smp;
    let _ = blur_call(4, &texture, &sampler, &not_a_buffer);
}
