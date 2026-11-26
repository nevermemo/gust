// The count of a counted stage is a handle of the block that holds `u32`s (GUST-E1015, D278). A `Param` is a value the
// host sets, not a buffer a stage wrote, so `count(limit)` is refused at the name.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("limit", on(&device); |b| {
        let limit = b.param::<u32>(2);
        let values = b.input::<u32>(8_u32);
        let _ = gust::gpu_task!(count(limit); 8_u32, |i| -> u32 { values[i] });
    });
}
