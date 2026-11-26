// Control case for an enum parameter of a batch (D270): `#[derive(GpuEnum)]` on free-standing enums, a `Param` of each
// in the build block, and a `match` on each `get()` in a stage body. `Mode` and `MODE` differ in case only, so a stage
// that holds both has two decoders that must not share a name. This crate must compile.

#![allow(non_camel_case_types)]

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum Mode {
    Linear,
    Curve,
}

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum MODE {
    Up,
    Down,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("modes", on(&device); |b| {
        let mode = b.param::<Mode>(Mode::Linear);
        let upper = b.param::<MODE>(MODE::Up);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            let picked: u32 = match mode.get() {
                Mode::Linear => i,
                Mode::Curve => i + 1_u32,
            };
            let moved: u32 = match upper.get() {
                MODE::Up => picked,
                MODE::Down => picked + 2_u32,
            };
            moved
        });
    });
}
