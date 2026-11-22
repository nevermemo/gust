// The old value of an atomic other than `atomic_add` is GUST-E0606 (m11.fetch narrowed the rule to these): the value
// of `atomic_exchange` is not accepted anywhere, a `let` initializer included. expected.txt pins the code and the line.

#[gust::gpu_module]
mod fetch {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn swap(
        id: SV_DispatchThreadID,
        mut cell: RWStructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let previous = atomic_exchange(&mut cell[0u32], id.x);
        out[id.x] = previous;
    }
}

fn main() {}
