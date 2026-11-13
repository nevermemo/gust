// A generic kernel (m9.g2, GUST-E0114): kernels are concrete, generics are helpers and inherent methods.

#[gust::gpu]
mod block {
    pub trait Tag {
        fn tag(&self) -> u32;
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run<T: Tag>(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
