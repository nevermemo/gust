// Control case for a struct stored as an element of a batch (D279, `#[derive(GpuStruct)]`): an input of rows, a stage
// that reads a field, a stage whose result is a row, and an in-place stage that stores a field and a whole row. This
// crate must compile.

use std::sync::Arc;

use gust::batch::{Batch, Error, Stage, D1};
use gust::cpu::CpuDevice;
use gust::prelude::*;

#[derive(Clone, Copy, gust::GpuStruct)]
#[repr(C)]
pub struct Row {
    pub pos: Vec2,
    pub life: f32,
    pub flags: u32,
}

struct Handles {
    rows: Stage<Row, D1>,
    life: Stage<f32, D1>,
    next: Stage<Row, D1>,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, Handles), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<Row>(4_u32);
        let life = gust::gpu_task!(4_u32, |i| -> f32 { rows[i].life });
        let next = gust::gpu_task!(4_u32, |i| -> Row {
            Row { pos: rows[i].pos, life: rows[i].life * 0.5, flags: rows[i].flags + 1_u32 }
        });
        let _damp = gust::gpu_task!(&mut rows, |i| {
            rows[i].life = rows[i].life * 0.5;
            rows[i].flags += 1_u32;
        });
        Handles { rows, life, next }
    });
}
