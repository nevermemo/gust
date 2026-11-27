// `scatter(..)` is a clause of a stage of a `gpu_batch!` (GUST-E1024): a `gpu_task!` that runs alone has host slices and a
// result to read back, and no handle of a block to store into, so it is refused at the keyword.

fn main() {
    let mut slots = [0_u32; 4];
    let _result = gust::gpu_task!(4_u32, scatter(&mut slots), [slots: &mut [u32]], |i| { slots[i] = 1_u32; });
}
