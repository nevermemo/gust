// X1 (m9.a1, GUST-E0333): a `&mut self` method on a storage element, called under a condition. The write would be a
// shared write hidden in a call, so the receiver rule refuses it: a mutating receiver must be an invocation-local place.

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
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Counter>) {
        if id.x > 0u32 {
            out[0u32].bump();
        }
    }
}

fn main() {}
