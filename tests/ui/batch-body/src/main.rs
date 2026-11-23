// A view of a batch stage has one index or a pair: three indices are refused (GUST-E1006, docs/DIAGNOSTICS.md).
// expected.txt pins the code and the line of the index.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("body", on(&device); in_flight(1); |b| {
        let a = b.input::<f32>(4_u32);
        let _ = gust::gpu_task!(4_u32, |i| -> f32 { a[(i, 0u32, 0u32)] });
    });
}
