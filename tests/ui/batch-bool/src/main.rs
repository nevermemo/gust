// A `bool` is not the element of a stage of a batch (GUST-E1004, docs/DIAGNOSTICS.md): a buffer holds `u32` words, so
// the stage returns a `u32`. expected.txt pins the code and the line of the result type.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("flags", on(&device); in_flight(1); |b| {
        let _ = gust::gpu_task!(4_u32, |i| -> bool { i > 1_u32 });
    });
}
