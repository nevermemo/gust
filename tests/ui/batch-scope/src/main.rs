// A stage of a batch is recorded once, when the block runs: one inside a loop is refused (GUST-E1006,
// docs/DIAGNOSTICS.md). expected.txt pins the code and the line of the stage.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("scope", on(&device); in_flight(1); |b| {
        let a = b.input::<f32>(4_u32);
        for _ in 0..2 {
            let _ = gust::gpu_task!(4_u32, |i| -> f32 { a[i] });
        }
    });
}
