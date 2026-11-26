// An in-place target of enum elements is not in this release (GUST-E1002, D275): a stage reads enum elements and
// returns an enum result instead. expected.txt pins the code and the line of the call.

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
    let (mut batch, ()) = gust::gpu_batch!("in-place", on(&device); |b| {
        let mut phase = b.input::<Phase>(4_u32);
        gust::gpu_task!(&mut phase, [phase: &mut Stage<Phase>], |i| { phase[i] = Phase::Run; });
    })
    .unwrap();
    let _frame = batch.begin();
}
