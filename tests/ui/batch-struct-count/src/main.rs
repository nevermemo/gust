// A struct (or a field of one) is no count of a stage (GUST-E1015, D278): the count is a handle of `u32`s. `rows` holds
// rows, so `count(rows)` is refused at the name. expected.txt pins the code and the line of the call.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;
use gust::prelude::*;

#[derive(Clone, Copy, gust::GpuStruct)]
#[repr(C)]
pub struct Row {
    pub pos: Vec2,
    pub life: f32,
    pub flags: u32,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<Row>(8_u32);
        let _ = gust::gpu_task!(count(rows); 8_u32, |i| -> u32 { rows[i].flags });
    });
}
