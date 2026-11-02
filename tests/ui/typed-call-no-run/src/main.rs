// A device that does not implement `Run` for a kernel cannot dispatch it: the generated `dispatch` requires
// the device of its buffers to implement `Run<kernel, N>`. The mock backend is the one of `typed-call-ok`
// without its `Run` impl, so rustc reports the `#[diagnostic::on_unimplemented]` text of `Run`.

use std::{marker::PhantomData, time::Duration};

use gust::{
    Access, Device, GpuPod, Job, JobStatus, LengthRule, Resource, TypedDevice,
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
    let device = Mock;
    let input = Buffer::<f32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);
    kernels::scale::dispatch(&device, 4, &input, &output).unwrap();
}
