// `for x in a` over an array is GUST-E0409: only an exclusive integer range is iterable.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let a: [u32; 4] = [1u32, 2u32, 3u32, 4u32];
        for x in a {
            out[id.x] = x;
        }
    }
}

fn main() {}
