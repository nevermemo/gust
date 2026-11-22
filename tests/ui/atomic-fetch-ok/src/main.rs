// Control case for m11.fetch: the old value of `atomic_add`, as the whole initializer of a `let`, for a `u32` cell and
// an `i32` cell. This crate must compile, so the validator, the lowering and the preserved Rust all accept it.

#[gust::gpu_module]
mod fetch {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scatter(
        id: SV_DispatchThreadID,
        keys: StructuredBuffer<uint>,
        mut counts: RWStructuredBuffer<uint>,
        mut signed: RWStructuredBuffer<int>,
        mut out: RWStructuredBuffer<uint>,
        mut firsts: RWStructuredBuffer<int>,
    ) {
        let i = id.x;
        if i < keys.len() {
            let cell = keys[i] % counts.len();
            let slot = atomic_add(&mut counts[cell], 1u32);
            out[cell * 16u32 + slot] = i;
            let ticket: i32 = atomic_add(&mut signed[0u32], 1i32);
            firsts[i] = ticket;
        }
    }
}

fn main() {}
