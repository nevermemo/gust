#[gpu_module]
mod histogram {
    /// Every invocation adds one to the bin of its key. Many invocations hit the same few bins at once, so the
    /// increment must be atomic: a plain `bins[k] += 1` would lose counts on a GPU.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn count(
        id: SV_DispatchThreadID,
        keys: StructuredBuffer<uint>,
        mut bins: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < keys.len() {
            let bin = keys[i] % bins.len();
            atomic_add(&mut bins[bin], 1u32);
        }
    }
}
