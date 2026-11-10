// `__gust_rep_0` is the lowering's temporary of a repeat; the `__gust_` prefix is reserved, GUST-E0901.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let __gust_rep_0: [u32; 2] = [1u32; 2];
        out[id.x] = __gust_rep_0[0];
    }
}

fn main() {}
