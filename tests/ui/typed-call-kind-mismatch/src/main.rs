// A uniform where the kernel has a storage buffer is a compile error in the generated typed call: `scale` takes
// storage buffers, the first argument below is a constant buffer (a `Resource` of kind `Uniform`, as `GpuUniform`
// and `CpuUniform` are), although its element type is the right one. The element bound alone would accept it;
// the generated `Kind = ::gust::kind::Storage` bound (D57) is what rejects it. The mock backend is the one of
// `typed-call-mismatch`; expected.txt pins the code and the line of the call.

use std::{marker::PhantomData, time::Duration};

use gust::{
    Access, Call, Device, GpuPod, Job, JobStatus, Kernel, LengthRule, Resource, Run, TypedDevice,
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

struct Buffer<T>(PhantomData<T>);

impl<T: GpuPod> Resource for Buffer<T> {
    type Element = T;
    type Device = Mock;
    type Kind = ::gust::kind::Storage;
    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> (Access, LengthRule) {
        (access, length)
    }
}

struct Constant<T>(PhantomData<T>);

impl<T: GpuPod> Resource for Constant<T> {
    type Element = T;
    type Device = Mock;
    type Kind = ::gust::kind::Uniform;
    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> (Access, LengthRule) {
        (access, length)
    }
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
    let constant = Constant::<f32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    let _ = kernels::scale::call(4, &constant, &output);
}
