// A default method body (m9.g1, GUST-E0124).

#[gust::gpu]
mod block {
    pub trait Shape {
        fn area(&self) -> u32 {
            1u32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
