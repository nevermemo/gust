// `.len()` of an array is the constant `N`, so a call inside its receiver would not run on the GPU while the CPU twin
// runs it: GUST-E0306 (bind the element to a local first).

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

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        cells: StructuredBuffer<[u32; 4]>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let mut c = Counter { n: 0u32 };
        out[id.x] = cells[c.next()].len() as u32;
    }
}

fn main() {}
