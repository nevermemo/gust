// A `&mut self` method in the impl of an enum (m9.g1, GUST-E0124): an enum has no fields to write, and that form is not
// part of this slice.

#[gust::gpu]
mod block {
    pub trait Step {
        fn step(&mut self);
    }

    #[derive(PartialEq, Eq)]
    pub enum Phase {
        A,
        B,
    }

    impl Step for Phase {
        fn step(&mut self) {}
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = 1u32;
    }
}

fn main() {}
