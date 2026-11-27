// Control case for a scatter stage (m14.s2sc): `scatter(&mut out)` after the extent, a store at a place a handle gives, and a
// stage that stores two targets, one of them on one path only. This crate must compile.

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
        let _s = gust::gpu_task!(4_u32, scatter(&mut out, &mut other), |i| { out[dst[i]] = rows[i]; if rows[i].flags > 1_u32 { other[dst[i]] = rows[i].flags; } });
        Handles { rows, dst, out, other }
    });
}
