// An atomic stage is one-dimensional (GUST-E1026): its rows are numbered by one index.

use std::sync::Arc;

use gust::batch::{Batch, Error, Stage, D1};
use gust::cpu::CpuDevice;
use gust::prelude::*;

struct Handles {
    key: Stage<u32, D1>,
    counts: Stage<u32, D1>,
    other: Stage<u32, D1>,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, Handles), Error> = gust::gpu_batch!("rows", on(&device); |b| {
        let key = b.input::<u32>(4_u32);
        let counts = b.input::<u32>(4_u32);
        let other = b.input::<u32>(4_u32);
        let _s = gust::gpu_task!((2_u32, 2_u32), atomic(&mut counts), |x, y| { atomic_add(&mut counts[key[x]], 1_u32); });
        Handles { key, counts, other }
    });
}
