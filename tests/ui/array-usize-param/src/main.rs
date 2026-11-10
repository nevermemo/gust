// `usize` is accepted as the cast of an index only, so a `usize` parameter is GUST-E0334.

#[gust::gpu]
mod block {
    fn pick(a: [u32; 4], n: usize) -> u32 {
        a[n]
    }


    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = pick([1u32, 2u32, 3u32, 4u32], 2);
    }
}

fn main() {}
