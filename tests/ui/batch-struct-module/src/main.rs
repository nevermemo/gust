// A struct declared inside a `#[gpu_module]` is no GPU struct: the module does not keep it as a Rust item for a path to
// reach, so a batch that names `kernels::Row` as an element gets rustc's "cannot find `Row` in `kernels`". expected.txt pins
// the line.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[gust::gpu_module]
mod kernels {
    pub struct Row {
        pub life: f32,
        pub flags: u32,
    }
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let rows = b.input::<kernels::Row>(4_u32);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 { rows[i].flags });
    });
}
