// A texture where the kernel has a storage buffer is a compile error in the generated typed call: `scale` takes
// storage buffers, the first argument below is a texture (a `Resource` of kind `Texture`) although its element type
// is the right one. The element bound alone would accept it; the generated `Kind = ::gust::kind::Storage` bound
// (D57) is what rejects it. The mock backend is the one of `typed-call-kind-mismatch`, with a texture and a sampler;
// expected.txt pins the code and the line of the call.
#![allow(dead_code)]

use std::{marker::PhantomData, time::Duration};

use gust::{
    Access, Call, Device, GpuPod, Job, JobStatus, Kernel, LengthRule, Resource, Run, SamplerArg,
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

#[gust::gpu]
mod kernels {
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut output: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i] * 2.0f32;
        }
    }
}


fn main() {
    let texture = Tex::<f32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    let _ = kernels::scale::call(4, &texture, &output);
}
