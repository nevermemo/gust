// A method whose receiver is `mut self` (m9.a2, GUST-E0116): a method that changes its receiver takes `&mut self`.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: uint,
    }

    impl Counter {
        pub fn bump(mut self) -> uint {
            self.n = self.n + 1u32;
            self.n
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = id.x;
    }
}

fn main() {}
