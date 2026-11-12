// A negative constant amount (signed amounts are accepted by Rust) is GUST-E0335 too.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<i32>) {
        let v = id.x as i32;
        out[id.x] = v >> -1i32;
    }
}

fn main() {}
