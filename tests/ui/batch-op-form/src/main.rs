// `gust::compact` gives two handles, so it takes `let (indices, count) = ..` (GUST-E1009, docs/DIAGNOSTICS.md).
// expected.txt pins the code and the line of the call. The rest of the program type-checks: the refusal is the one
// error.

use std::sync::Arc;

use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, ()) = gust::gpu_batch!("op-form", on(&device); |b| {
        let flags = b.input::<u32>(4_u32);
        let _indices = gust::compact(&flags);
    })
    .unwrap();
    let _frame = batch.begin();
}
