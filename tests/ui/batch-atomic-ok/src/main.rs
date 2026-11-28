// Control case for an atomic stage (m14.s2sa): `atomic(&mut a, &mut b)` after the extent, `atomic_add` and `atomic_max` at
// a place a handle gives, and the old value of an add steering a second atomic. This crate must compile.

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
        let _s = gust::gpu_task!(4_u32, atomic(&mut counts, &mut other), |i| { let old = atomic_add(&mut counts[key[i]], 1_u32); if old == 0_u32 { atomic_max(&mut other[0_u32], key[i]); } });
        Handles { key, counts, other }
    });
}
