// A parameter type that is not a GPU enum: step one asks each type of the block for the marker both derives implement,
// so rustc names the type and the derives (E0277), and it also says "cannot find macro `Mode` in this scope" at the type,
// because the derive writes the macro that step one calls. These are the two errors (`errors: 2` in expected.txt, which
// pins the named one and its line).

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
enum Mode {
    Linear,
    Curve,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("modes", on(&device); |b| {
        let mode = b.param::<Mode>(Mode::Linear);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            let picked: u32 = match mode.get() {
                Mode::Linear => i,
                Mode::Curve => i + 1_u32,
            };
            picked
        });
    });
}
