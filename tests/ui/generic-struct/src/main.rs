// A generic struct (m9.g2, GUST-E0103): structs stay concrete.

#[gust::gpu]
mod block {
    pub struct Wrap<T> {
        pub v: T,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
