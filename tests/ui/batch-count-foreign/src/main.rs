// The count of a counted stage is a handle of its own block (GUST-E1015, D278): a handle that another batch returned is
// not in this block, so `count(foreign)` is refused at the name (no capture list names a count).

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let (_one, foreign) = gust::gpu_batch!("one", on(&device); |b| {
        b.input::<u32>(1_u32)
    })
    .unwrap();
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("two", on(&device); |b| {
        let values = b.input::<u32>(8_u32);
        let _ = gust::gpu_task!(count(foreign); 8_u32, |i| -> u32 { values[i] });
    });
}
