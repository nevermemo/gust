// `out[s.n] = s.bump()` on a storage buffer: Rust evaluates the value first and the index after it, and slangc 2026.21
// computes the index first (2026.20 did not), so the element written would differ, GUST-E0421.

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: u32,
    }

    impl Counter {
        pub fn bump(&mut self) -> u32 {
            self.n = self.n + 1u32;
            self.n
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let mut s = Counter { n: 5u32 };
        out[s.n] = s.bump();
    }
}

fn main() {}
