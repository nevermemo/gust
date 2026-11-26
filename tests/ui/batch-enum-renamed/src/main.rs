// An enum brought in under another name (`use shades::Tint as Shade;`) is not supported: the derive's macro hands over
// the enum as `Tint`, so a parameter written `Shade` names none of the enums of the block, and the stage says so rather
// than reporting a parameter a body cannot read.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

mod shades {
    #[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Tint {
        Warm,
        Cool,
    }
}

use shades::Tint as Shade;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("shades", on(&device); |b| {
        let shade = b.param::<Shade>(Shade::Warm);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 {
            let picked: u32 = match shade.get() {
                Shade::Warm => i,
                Shade::Cool => i + 1_u32,
            };
            picked
        });
    });
}
