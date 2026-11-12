// Recursion through an impl (m9.g1, GUST-E0120).

#[gust::gpu]
mod block {
    pub trait Depth {
        fn depth(&self) -> u32;
    }

    pub struct Node {
        pub n: u32,
    }

    impl Depth for Node {
        fn depth(&self) -> u32 {
            self.depth()
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
