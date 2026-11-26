// Control case for an enum stored as an element of a batch (D275): an input of enums, a stage that reads it through a
// `match` and a stage whose result type is the enum. The host writes and reads the enum itself. This crate must compile.

use std::sync::Arc;

use gust::batch::{Batch, Error, Stage, D1};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum Phase {
    Idle = 0,
    Run = 3,
}

struct Handles {
    phase: Stage<Phase, D1>,
    next: Stage<Phase, D1>,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, Handles), Error> = gust::gpu_batch!("phases", on(&device); |b| {
        let phase = b.input::<Phase>(4_u32);
        let next = gust::gpu_task!(4_u32, |i| -> Phase {
            match phase[i] {
                Phase::Idle => Phase::Run,
                Phase::Run => Phase::Idle,
            }
        });
        Handles { phase, next }
    });
}
