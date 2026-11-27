// An atomic on a scatter target (GUST-E1025): atomics at a computed place are a later release, and a borrow of a target
// is no store the stage checks.

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
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
    other: Stage<u32, D1>,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, Handles), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<Row>(4_u32);
        let dst = b.input::<u32>(4_u32);
        let out = b.input::<Row>(4_u32);
        let other = b.input::<u32>(4_u32);
        let _s = gust::gpu_task!(4_u32, scatter(&mut other), |i| { atomic_add(&mut other[dst[i]], 1_u32); });
        Handles { rows, dst, out, other }
    });
}
