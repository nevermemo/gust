// A stage's copy of an enum that is not the enum the parameter's type names: the copy here comes from the hidden
// `@enums` form (written by hand, which no user code does) and gives `B` the word 1 and `C` the word 2, while the real
// enum has them the other way round. Nothing else would notice, since the bound and the parameter's packing both hold
// for the real enum, and the stage would swap the two arms on the device and on the CPU twin alike. The stage's
// expansion asserts every word of the copy against the real enum, so this is a compile error at the parameter's type.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum Mode {
    A = 0,
    B = 2,
    C = 1,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!(
        @enums [ { #[repr(u32)] enum Mode { A = 0, B = 1, C = 2 } } ];
        "modes", on(&device); |b| {
            let mode = b.param::<Mode>(Mode::A);
            let _ = gust::gpu_task!(4_u32, |i| -> u32 {
                let picked: u32 = match mode.get() {
                    Mode::A => i,
                    Mode::B => i + 1_u32,
                    Mode::C => i + 2_u32,
                };
                picked
            });
        }
    );
}
