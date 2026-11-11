// A closure call in the bound of a for range (m9.c, GUST-E0421): refused like a mutating call, conservatively.
// Expected: one error at the bound, line 9.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let bound = |n: u32| -> u32 { n + 1u32 };
        for i in 0u32..bound(3u32) {
            out[id.x] = 1.0f32;
        }
    }
}

fn main() {}
