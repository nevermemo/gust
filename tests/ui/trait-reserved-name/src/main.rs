// A trait method named like a method of the dialect (m9.g1, GUST-E0328).

#[gust::gpu]
mod block {
    pub trait Sized {
        fn len(&self) -> u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
