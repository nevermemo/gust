// `rows[c.n].cells[1] = c.next()`: Rust evaluates the value first and the index after it, slangc 2026.21 reads the index
// first, so another element would be written, GUST-E0421 (as for a plain buffer element, here through a field array).

#[gust::gpu]
mod block {
    pub struct Counter {
        pub n: u32,
    }

    impl Counter {
        pub fn next(&mut self) -> u32 {
            self.n = self.n + 1u32;
            self.n
        }
    }

    pub struct Row {
        pub cells: [u32; 4],
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut rows: RWStructuredBuffer<Row>) {
        let mut c = Counter { n: 1u32 };
        rows[c.n].cells[1] = c.next();
    }
}

fn main() {}
