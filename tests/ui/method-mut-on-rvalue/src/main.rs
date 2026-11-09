// A `&mut self` method called on a value that is not a place (m9.a2, GUST-E0333): the change would be lost.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: uint,
    }

    impl Counter {
        pub fn bump(&mut self) {
            self.n = self.n + 1u32;
        }
    }

    fn make() -> Counter {
        let counter = Counter { n: 0u32 };
        counter
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        make().bump();
        out[id.x] = id.x;
    }
}

fn main() {}
