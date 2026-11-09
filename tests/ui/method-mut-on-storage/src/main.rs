// A method that takes `&mut self` called on an element of a storage buffer would write through the buffer where no
// statement shows the write: GUST-E0333 at the call (spike M9.s0, fixture X1). It is the one error: the declaration
// of `bump` is GUST-E0116 until M9 slice m9.a2, and that report comes last, only when nothing else is wrong.
// expected.txt pins the code and the line of the call.

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
