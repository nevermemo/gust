// A `&mut self` call in the bound of a `for` range (m9.a2, GUST-E0421): the Slang loop evaluates the end bound first.

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
        let mut counter = Counter { n: 2u32 };
        for i in 0u32..counter.next() {
            out[id.x] = i;
        }
    }
}

fn main() {}
