// Control case for traits (m9.g1): a trait with `&self`, by-value `self` and `&mut self` methods, implemented by a
// struct, called by method syntax and by each path spelling. This crate must compile, and rustc checks the path
// calls as written.

#[gust::gpu]
mod shapes {
    pub trait Shape {
        fn area(&self) -> u32;
        fn grown(self, by: u32) -> Self;
        fn grow(&mut self, by: u32);
    }

    pub struct Sq {
        pub side: u32,
    }

    impl Shape for Sq {
        fn area(&self) -> u32 {
            self.side * self.side
        }

        fn grown(self, by: u32) -> Sq {
            let s = Sq { side: self.side + by };
            s
        }

        fn grow(&mut self, by: u32) {
            self.side += by;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let mut s = Sq { side: id.x };
        s.grow(1u32);
        Shape::grow(&mut s, 1u32);
        let t = Shape::grown(s, 1u32);
        out[id.x] = t.area() + Shape::area(&t) + <Sq as Shape>::area(&t);
    }
}

fn main() {}
