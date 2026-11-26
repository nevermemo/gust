// A library op takes no enum operand (GUST-E1009, D275): an enum is held as a `u32` word in a batch and is no `GpuPod`.
// expected.txt pins the code and the line of the call.

use std::sync::Arc;

use gust::cpu::CpuDevice;

#[derive(Clone, Copy, PartialEq, gust::GpuEnum)]
#[repr(u32)]
enum Phase {
    Idle = 0,
    Run = 3,
}

fn main() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, ()) = gust::gpu_batch!("op-operand", on(&device); |b| {
        let phase = b.input::<Phase>(4_u32);
        let _sum = gust::sum(&phase);
    })
    .unwrap();
    let _frame = batch.begin();
}
