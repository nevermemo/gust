// Control case for the typed host API of `#[gust::gpu]`: the generated `call`, `submit` and `dispatch`
// compile for a mock backend, with `Independent`, with user parameters named `element_count` and
// `device`, with user types named like the generated generics (`R0`, `D`), and for a kernel without
// storage parameters. This must compile, so a change to the generated signatures (a missing `'r`
// bound, a wrong element type or a wrong argument count) fails here, not in a user's crate.

use std::{marker::PhantomData, time::Duration};

use gust::{
    Access, Call, Device, GpuPod, Independent, Job, JobStatus, Kernel, LengthRule, Resource, Run,
    TypedDevice,
};

// A 25-line mock backend: it runs nothing, which is all this fixture needs.
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
    // Named like the generics the macro uses internally.
    pub struct R0 {
        pub value: float,
    }

    pub struct D {
        pub value: uint,
    }

    // User parameters named `element_count` and `device`, a settings buffer and a read-write output.
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        element_count: StructuredBuffer<float>,
        settings: StructuredBuffer<float>,
        mut device: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < device.len() && i < element_count.len() && 0u32 < settings.len() {
            device[i] = element_count[i] * settings[0u32];
        }
    }

    #[kernel]
    pub fn copy_r0(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<R0>,
        mut output: RWStructuredBuffer<R0>,
    ) {
        let i = id.x;
        if i < input.len() && i < output.len() {
            output[i] = input[i];
        }
    }

    #[kernel]
    pub fn bump(id: SV_DispatchThreadID, mut values: RWStructuredBuffer<D>) {
        let i = id.x;
        if i < values.len() {
            values[i] = D { value: values[i].value + 1u32 };
        }
    }

    #[kernel]
    pub fn nothing(id: SV_DispatchThreadID) {
        let i = id.x;
        if i < 1u32 {}
    }
}

fn main() {
    let device = Mock;
    let input = Buffer::<f32>(PhantomData);
    let settings = Buffer::<f32>(PhantomData);
    let output = Buffer::<f32>(PhantomData);

    // `R0..R2` are inferred from the arguments; only `settings` has a length of its own.
    let call: Call<'_, Mock, kernels::scale, 3> =
        kernels::scale::call(16, &input, Independent(&settings), &output);
    assert_eq!(call.element_count(), 16);
    kernels::scale::dispatch(&device, 16, &input, Independent(&settings), &output).unwrap();
    let job = kernels::scale::submit(&device, 16, &input, Independent(&settings), &output).unwrap();
    job.wait().unwrap();

    let source = Buffer::<kernels::R0>(PhantomData);
    let target = Buffer::<kernels::R0>(PhantomData);
    kernels::copy_r0::dispatch(&device, 16, &source, &target).unwrap();

    let counters = Buffer::<kernels::D>(PhantomData);
    kernels::bump::dispatch(&device, 16, &counters).unwrap();

    // No storage parameters: the device type cannot be inferred, so `call` takes a turbofish.
    kernels::nothing::dispatch(&device, 16).unwrap();
    let _: Call<'_, Mock, kernels::nothing, 0> = kernels::nothing::call::<Mock>(16);
}
