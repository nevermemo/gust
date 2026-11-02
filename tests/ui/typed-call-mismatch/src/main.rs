// A buffer whose element type is not the kernel parameter's element type is a compile error in the
// generated typed call: `scale` takes `float` buffers, the first argument below holds `u32`. The mock
// backend is the one of `typed-call-ok`. rustc reports the failed `Resource<Element = f32>` bound at the
// call expression in this file (the whole call is the primary span) and notes the bound on the kernel
// parameter; expected.txt pins that code and line.

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
    let wrong = Buffer::<u32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    let _ = kernels::scale::call(4, &wrong, &output);
}
