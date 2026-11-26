// A library op takes no struct operand (GUST-E1022, D279): an op reads a number buffer, and a row is no number.
// expected.txt pins the code and the line of the call.

use std::sync::Arc;

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
    let (mut batch, ()) = gust::gpu_batch!("op-operand", on(&device); |b| {
        let rows = b.input::<Row>(4_u32);
        let _sum = gust::sum(&rows);
    })
    .unwrap();
    let _frame = batch.begin();
}
