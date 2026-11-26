// A struct declared inside a `#[gpu_module]` is no GPU struct: the module keeps the struct, but it has no `#[derive(GpuStruct)]`,
// so the hidden macro `kernels::Row!` that the batch calls does not exist: rustc's resolution error (E0433, pinned) is the
// refusal, not an error by name, followed by the marker error that names the type (`errors: 2`).

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
