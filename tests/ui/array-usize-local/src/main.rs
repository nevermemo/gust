// `usize` is accepted as the cast of an index only, so a `usize` local is GUST-E0334.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let a: [u32; 4] = [1u32, 2u32, 3u32, 4u32];
        let n: usize = 2;
        out[id.x] = a[n];
    }
}

fn main() {}
