// A barrier is a statement of its own: inside a tuple element it is refused like in any expression (the barrier form).
// expected.txt pins the code and the line of the barrier.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn run(
        id: SV_DispatchThreadID,
        lid: LocalId,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        scratch[lid.x] = input[id.x];
        let t = (workgroup_barrier(), 1u32);
        out[id.x] = t.1 + scratch[0u32];
    }
}

fn main() {}
