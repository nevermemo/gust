// A buffer where the kernel has a uniform is a compile error in the generated typed call: `shade` takes a `Uniform`
// parameter, the first argument below is a storage buffer (a `Resource` of kind `Storage`) holding the right element
// type. The element bound alone would accept it; the generated `Kind = ::gust::kind::Uniform` bound (D57) is what
// rejects it. This is the reverse of `typed-call-kind-mismatch`; the mock backend is the one of that case.
// expected.txt pins the code and the line of the call.

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

impl<K: Kernel, const N: usize> Run<K, N> for Mock {
    fn submit_call<'d>(&'d self, _: &Call<'_, Self, K, N>) -> Result<Done, ()> {
        Ok(Done)
    }
}

#[gust::gpu]
mod kernels {
    pub struct Params {
        pub gain: f32,
        pub bias: f32,
        pub pad0: f32,
        pub pad1: f32,
    }

    #[kernel]
    pub fn shade(
        id: SV_DispatchThreadID,
        params: Uniform<Params>,
        mut output: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < output.len() {
            output[i] = params.gain + params.bias;
        }
    }
}

fn main() {
    // `Buffer<Params>` has the uniform's element type and the wrong kind.
    let params = Buffer::<kernels::Params>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    let _ = kernels::shade::call(4, &params, &output);
}
