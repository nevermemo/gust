// A blanket impl (m9.g1, GUST-E0121): an impl is for a struct or a unit enum of the module.

#[gust::gpu]
mod block {
    pub trait Shape {
        fn area(&self) -> u32;
    }

    impl<A> Shape for A {
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
