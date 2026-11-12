// A trait method named like an inherent method of the type (m9.g1, GUST-E0328, the P18 ruling): slangc takes both silently.

#[gust::gpu]
mod block {
    pub trait Shape {
        fn area(&self) -> u32;
    }

    pub struct Sq {
        pub side: u32,
    }

    impl Shape for Sq {
        fn area(&self) -> u32 {
            self.side * self.side
        }
    }

    impl Sq {
        pub fn area(&self) -> u32 {
            7u32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
