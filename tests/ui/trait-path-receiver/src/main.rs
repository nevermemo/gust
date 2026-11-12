// A trait path call on a call (m9.g1, GUST-E0126): the receiver must be a place, bind it first.

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

    fn make() -> Sq {
        let s = Sq { side: 2u32 };
        s
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = Shape::area(&make());
    }
}

fn main() {}
