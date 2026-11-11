// A cast to `usize` outside an index is GUST-E0334. expected.txt pins the code and the line of the cast.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let a: [u32; 4] = [1u32, 2u32, 3u32, 4u32];
        out[id.x] = (id.x as usize) as u32 + a[0];
    }
}

fn main() {}
