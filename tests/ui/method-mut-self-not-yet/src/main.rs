// A method that takes `&mut self` is not accepted yet (M9 slice m9.a2): GUST-E0116 at the declaration, reported after
// the rest of the module has been accepted. expected.txt pins the code and the line of the method.

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

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = id.x;
    }
}

fn main() {}
