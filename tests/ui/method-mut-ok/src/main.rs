// Control case for mutating methods (m9.a2): `&mut self`, `self`, an associated function, `Self`, a second `impl`
// block, a method that calls a method and a helper that calls a method. This crate must compile.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: uint,
    }

    impl Counter {
        pub fn new(start: uint) -> Self {
            let counter = Self { n: start };
            counter
        }

        pub fn bump(&mut self) {
            self.n = self.n + 1u32;
        }

        pub fn bump_twice(&mut self) {
            self.bump();
            self.bump();
        }
    }

    impl Counter {
        pub fn value(self) -> uint {
            self.n
        }
    }

    fn drive(start: uint) -> uint {
        let mut counter = Counter::new(start);
        counter.bump_twice();
        counter.value()
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = drive(id.x);
    }
}

fn main() {}
