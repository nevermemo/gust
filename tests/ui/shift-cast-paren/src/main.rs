// A constant amount through parentheses and an `as` cast is still constant: GUST-E0335 names the folded value, 40.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x << ((40i32) as u32);
    }
}

fn main() {}
