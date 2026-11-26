// Two enums of one name among the parameters of one batch: a stage declares each enum under its own name, so the batch
// refuses the second (GUST-E1005) and names the fix.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

mod first {
    #[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Mode {
        Linear,
        Curve,
    }
}

mod second {
    #[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Mode {
        Square,
        Cube,
    }
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("modes", on(&device); |b| {
        let one = b.param::<first::Mode>(first::Mode::Linear);
        let two = b.param::<second::Mode>(second::Mode::Square);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            let a: u32 = match one.get() {
                Mode::Linear => i,
                Mode::Curve => i + 1_u32,
            };
            let c: u32 = match two.get() {
                Mode::Square => a,
                Mode::Cube => a + 1_u32,
            };
            c
        });
    });
}
