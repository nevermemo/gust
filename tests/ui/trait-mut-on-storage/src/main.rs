// A mutating trait method on a storage element (m9.g1, GUST-E0333).

#[gust::gpu]
mod block {
    pub trait Grow {
        fn grow(&mut self);
    }

    pub struct Sq {
        pub side: u32,
    }

    impl Grow for Sq {
        fn grow(&mut self) {
            self.side += 1u32;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn second(id: SV_DispatchThreadID, mut squares: RWStructuredBuffer<Sq>) {
        squares[id.x].grow();
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
