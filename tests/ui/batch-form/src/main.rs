// A batch has at most four frames in flight: `in_flight(5)` is refused (GUST-E1005, docs/DIAGNOSTICS.md).
// expected.txt pins the code and the line of the number. The call is written as a program writes it, with no type
// annotation: the refusal is the one error, and the code after it still type-checks.

use std::sync::Arc;

use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, ()) = gust::gpu_batch!("form", on(&device); in_flight(5); |b| {
        let _a = b.input::<f32>(4_u32);
    })
    .unwrap();
    let _frame = batch.begin();
}
