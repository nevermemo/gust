// A struct that is no GPU struct as the element of a batch: step one calls the type's own name as a macro (the derive
// writes it), so rustc says "cannot find macro `Row` in this scope" at the type. This is the one error; expected.txt pins
// its line.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;
use gust::prelude::*;

#[derive(Clone, Copy)]
#[repr(C)]
struct Row {
    pos: Vec2,
    life: f32,
    flags: u32,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<Row>(4_u32);
        let _ = gust::gpu_task!(4_u32, |i| -> f32 { rows[i].life });
    });
}
