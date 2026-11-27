// `ScatterMut` is the target of a scatter stage of a `gpu_batch!`, which the macro writes with the stage's own checks
// (GUST-E1024): a kernel of a user's module that takes it has no stage, so it is refused (use `StorageMut`).

#[gust::gpu_module]
mod place {
    #[kernel(shape = grid)]
    pub fn store(id: SV_DispatchThreadID, mut out: ScatterMut<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
