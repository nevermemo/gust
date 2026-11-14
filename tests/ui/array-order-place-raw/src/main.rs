// `a[r#s.n as usize] = r#s.bump()`: a raw identifier is the same variable, so this is GUST-E0421 like the plain form.

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
        let mut s = Counter { n: 0u32 };
        let mut a: [u32; 4] = [0u32; 4];
        a[r#s.n as usize] = r#s.bump();
        out[id.x] = a[1];
    }
}

fn main() {}
