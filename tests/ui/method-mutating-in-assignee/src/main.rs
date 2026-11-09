// A `&mut self` call in the place of an assignment (m9.a2 review M1, GUST-E0421): Rust evaluates the value first and
// Slang evaluates the call in the index first, so the GPU would write another element than the CPU twin.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: uint,
    }

    impl Counter {
        pub fn next(&mut self) -> uint {
            self.n = self.n + 1u32;
            self.n
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        let mut counter = Counter { n: 5u32 };
        out[counter.next()] = counter.next();
    }
}

fn main() {}
