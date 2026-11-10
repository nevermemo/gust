// A slice type is GUST-E0111; a helper takes arrays by value with a literal length.

#[gust::gpu]
mod block {
    fn total(a: &[u32]) -> u32 {
        a[0]
    }


    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
