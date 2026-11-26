// Control case for a batch with no type of the user's: its parameters are scalars, so `gpu_batch!` lowers the block at
// once, with no callback of an enum (D270). This crate must compile.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("scaled", on(&device); |b| {
        let gain = b.param::<u32>(2);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 { i * gain.get() });
    });
}
