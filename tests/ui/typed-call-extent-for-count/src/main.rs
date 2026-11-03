// The multi-dimensional forms of the typed call take a `gust::Extent`, not a bare count: `scale::call_extent(4, ..)`
// is a type mismatch at the first argument (the one-dimensional `call` is the form that takes the count). The mock
// backend is the one of `typed-call-mismatch`. expected.txt pins the code and the line of the call.

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
    type Device = Mock; type Kind = ::gust::kind::Storage;
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
    let input = Buffer::<f32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    let _ = kernels::scale::call_extent(4u32, &input, &output);
}
