// A generic trait (m9.g1, GUST-E0124); bounds are slice m9.g2.

#[gust::gpu]
mod block {
    pub trait Scale<A> {
        fn scale(&self, by: A) -> u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
