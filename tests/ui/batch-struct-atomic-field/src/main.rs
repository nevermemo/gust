// An atomic on a field of a row is refused: an atomic takes an indexed element of an atomic buffer (GUST-E0609, the
// validator's own code; measured at the first cargo run, D279). expected.txt pins the code and the line of the call.

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
    let (mut batch, ()) = gust::gpu_batch!("atomic", on(&device); |b| {
        let rows = b.input::<Row>(4_u32);
        gust::gpu_task!(&mut rows, |i| { atomic_add(&mut rows[i].flags, 1_u32); });
    })
    .unwrap();
    let _frame = batch.begin();
}
