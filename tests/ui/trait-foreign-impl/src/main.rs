// A trait impl for a foreign type (m9.g1, GUST-E0121).

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

    impl Shape for u32 {
        fn area(&self) -> u32 {
            *self
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
