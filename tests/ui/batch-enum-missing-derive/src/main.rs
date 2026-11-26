// A parameter type that is not a GPU enum: step one calls the type's own name as a macro (the derive writes it), so
// rustc says "cannot find macro `Mode` in this scope" at the type. The kind of a type, an enum or a struct, is the
// derive's (D279), so the `GpuEnum` bound is checked in step two, which a type with no derive never reaches: this is the
// one error. expected.txt pins its line.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
enum Mode {
    Linear,
    Curve,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("modes", on(&device); |b| {
        let mode = b.param::<Mode>(Mode::Linear);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            let picked: u32 = match mode.get() {
                Mode::Linear => i,
                Mode::Curve => i + 1_u32,
            };
            picked
        });
    });
}
