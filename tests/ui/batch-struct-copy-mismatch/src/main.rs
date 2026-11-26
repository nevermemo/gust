// A stage's copy of a struct that is not the struct the input's type names: the copy here comes from the hidden
// `@enums` form (written by hand, which no user code does) and lists `pos` last, while the real struct has it first.
// The names and the types of the fields all agree, so the destructuring and the typed `let`s hold, and the stage would
// read every field at the wrong offset on the device. The expansion asserts that the copy's fields sit in the order the
// real struct lays them out, so this is a compile error at the input's type.

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
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!(
        @enums [ { struct Row { life: f32, flags: u32, pos: Vec2 } } ];
        "rows", on(&device); |b| {
            let rows = b.input::<Row>(4_u32);
            let _ = gust::gpu_task!(4_u32, |i| -> f32 { rows[i].life });
        }
    );
}
