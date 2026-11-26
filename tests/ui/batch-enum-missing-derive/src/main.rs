// A parameter type that is not a GPU enum: rustc says "`Mode` is not a GPU enum" at the type, with the label "add
// `#[derive(GpuEnum)]` to the enum" (the trait's `on_unimplemented`, D270). Without the derive there is no macro of
// that name either, and rustc reports "cannot find macro `Mode`" next to this error; the stand-in macro below takes
// the place of the one the derive would write, so that this fixture has the one error expected.txt pins.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
enum Mode {
    Linear,
    Curve,
}

macro_rules! Mode {
    ($($tokens:tt)*) => {
        ::core::unreachable!()
    };
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
