// The column form of a batch stage binds one storage buffer per column: nine columns read and the result are ten, over the
// portable floor of 8 (GUST-E0746 at the call, plan D279 check 5). The same data held as rows is two buffers (one read
// and the result, or one in place), which no floor refuses. expected.txt pins the code and the line of the call.

use std::sync::Arc;

use gust::batch::{Batch, Error};
use gust::cpu::CpuDevice;

fn main() {
    let device = Arc::new(CpuDevice::new());
    let _result: Result<(Batch<CpuDevice>, ()), Error> = gust::gpu_batch!("columns", on(&device); |b| {
        let c0 = b.input::<u32>(4_u32);
        let c1 = b.input::<u32>(4_u32);
        let c2 = b.input::<u32>(4_u32);
        let c3 = b.input::<u32>(4_u32);
        let c4 = b.input::<u32>(4_u32);
        let c5 = b.input::<u32>(4_u32);
        let c6 = b.input::<u32>(4_u32);
        let c7 = b.input::<u32>(4_u32);
        let c8 = b.input::<u32>(4_u32);
        let _ = gust::gpu_task!(4_u32, |i| -> u32 { c0[i] + c1[i] + c2[i] + c3[i] + c4[i] + c5[i] + c6[i] + c7[i] + c8[i] });
    });
}
