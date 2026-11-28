// `atomic(..)` is a clause of a stage of a `gpu_batch!` (GUST-E1026): a `gpu_task!` that runs alone has host slices and a
// result to read back, and no handle of a block to accumulate into, so it is refused at the keyword.

fn main() {
    let mut counts = vec![0_u32; 4];
    let _result = gust::gpu_task!(4_u32, atomic(&mut counts), [counts: &mut [u32]], |i| { atomic_add(&mut counts[i], 1_u32); });
}
