// A library op of a batch needs the `gust-algo` crate, and this crate does not depend on it (GUST-E1010,
// docs/DIAGNOSTICS.md). expected.txt pins the code and the line of the call. The rest of the program type-checks: the
// refusal is the one error.

use std::sync::Arc;

use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, ()) = gust::gpu_batch!("op-crate", on(&device); |b| {
        let values = b.input::<u32>(4_u32);
        let _total = gust::sum(&values);
    })
    .unwrap();
    let _frame = batch.begin();
}
