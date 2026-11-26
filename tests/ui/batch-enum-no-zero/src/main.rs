// An enum stored as the element of a batch needs a variant whose word is 0 (GUST-E1012, D275): a buffer starts zeroed,
// so that word must read as a variant. `Shade` has none, so `b.input::<Shade>` is refused at the element type.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum Shade {
    Dim = 2,
    Lit = 5,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("shades", on(&device); |b| {
        let shade = b.input::<Shade>(4_u32);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            match shade[i] {
                Shade::Dim => 1_u32,
                Shade::Lit => 2_u32,
            }
        });
    });
}
