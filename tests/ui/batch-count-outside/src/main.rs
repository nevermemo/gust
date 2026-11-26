// `count(..)` is a clause of a stage of a `gpu_batch!` (GUST-E1013, D278): a `gpu_task!` that runs alone has its own
// extent, so the clause is refused at the handle's name.

fn main() {
    let sizes = [3_u32, 1, 4, 1];
    let _result = gust::gpu_task!(count(n); 4_u32, [n: &[u32], sizes: &[u32]], |i| -> u32 { sizes[i] });
}
