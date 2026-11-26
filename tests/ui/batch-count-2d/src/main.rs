// A counted stage is 1-D in this release (GUST-E1014, D278): the rows are the linear index over a two-dimensional grid
// of groups, so a counted `|x, y|` body (a 2-D stage with a host width) is a later release.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("grid", on(&device); |b| {
        let n = b.input::<u32>(1_u32);
        let size = b.param::<(u32, u32)>((4, 4));
        let image = b.input::<u32>(&size);
        let _ = gust::gpu_task!(count(n); &size, |x, y| -> u32 { image[(x, y)] });
    });
}
