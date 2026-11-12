// A barrier in an impl method (m9.g1, D154, GUST-E0414).

#[gust::gpu]
mod block {
    pub trait Sync {
        fn sync(&self) -> u32;
    }

    pub struct Node {
        pub n: u32,
    }

    impl Sync for Node {
        fn sync(&self) -> u32 {
            workgroup_barrier();
            1u32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
