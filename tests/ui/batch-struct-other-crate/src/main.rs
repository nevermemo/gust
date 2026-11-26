// A struct from another crate as the element of a batch is refused by rustc (D279): the derive's hidden macro is private to
// the crate that declares the struct, so `shapes::Row` has no macro to call from here (an enum is the same). A struct that
// is to be a batch element is declared in the crate (or a module of the crate) that holds the batch. expected.txt pins
// the line.

use std::sync::Arc;

use gust::batch::{Batch, Error, Stage, D1};
use gust::cpu::CpuDevice;

struct Handles {
    rows: Stage<shapes::Row, D1>,
    life: Stage<f32, D1>,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, Handles), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<shapes::Row>(4_u32);
        let life = gust::gpu_task!(4_u32, |i| -> f32 { rows[i].life });
        Handles { rows, life }
    });
}
