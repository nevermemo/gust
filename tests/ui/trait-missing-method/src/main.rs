// An impl that leaves out a method of its trait (m9.g1): the macro accepts it and rustc refuses it on the CPU twin and
// the typecheck item (E0046), so the error is rustc's and never a slangc message.

#[gust::gpu]
mod block {
    pub trait Shape {
        fn area(&self) -> u32;
        fn perimeter(&self) -> u32;
    }

    pub struct Sq {
        pub side: u32,
    }

    impl Shape for Sq {
        fn area(&self) -> u32 {
            self.side * self.side
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let s = Sq { side: id.x };
        out[id.x] = s.area();
    }
}

fn main() {}
