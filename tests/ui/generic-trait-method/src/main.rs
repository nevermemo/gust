// A generic trait method (m9.g2, GUST-E0124): a trait method is not generic, a helper function with the trait as its
// bound is.

#[gust::gpu]
mod block {
    pub trait Tag {
        fn tag(&self) -> u32;
        fn with<X: Tag>(&self, other: X) -> u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
