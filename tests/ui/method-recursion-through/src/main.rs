// A method and a helper that reach each other (m9.a2, GUST-E0120): the cycle runs through a method call, which the
// graph draws to every method of that name.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: uint,
    }

    impl Counter {
        pub fn total(&self) -> uint {
            sum_to(self.n)
        }
    }

    fn sum_to(n: u32) -> u32 {
        let counter = Counter { n: n - 1u32 };
        n + counter.total()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = sum_to(id.x);
    }
}

fn main() {}
