// A read of an atomic target (GUST-E1008): a cell other rows are updating on the GPU would be settled on the CPU twin.

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
        let _s = gust::gpu_task!(4_u32, atomic(&mut counts), |i| { let seen = counts[i]; atomic_add(&mut counts[key[i]], seen); });
        Handles { key, counts, other }
    });
}
