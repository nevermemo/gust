// The accepted shifts next to GUST-E0335: the boundary amount 31, 0, a signed amount in range, and run-time amounts
// (the GPU masks those, KM-04). This crate must compile.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let n = input[id.x];
        let mut v = id.x << 31u32;
        v >>= 0u32;
        v ^= id.x >> (3i32 as u32);
        v ^= id.x << n;
        v >>= n + 1u32;
        out[id.x] = v;
    }
}

fn main() {}
