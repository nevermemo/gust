// `>>=` with a suffixed literal amount of exactly 32 is GUST-E0335: the boundary is the width, not above it.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let mut v = id.x;
        v >>= 32u32;
        out[id.x] = v;
    }
}

fn main() {}
