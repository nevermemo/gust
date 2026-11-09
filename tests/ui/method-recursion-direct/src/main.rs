// A helper that calls itself (m9.a2, GUST-E0120): the call graph has a cycle of one function.

#[gust::gpu]
mod block {
    fn fact(n: u32) -> u32 {
        if n == 0u32 {
            return 1u32;
        }
        n * fact(n - 1u32)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = fact(id.x);
    }
}

fn main() {}
