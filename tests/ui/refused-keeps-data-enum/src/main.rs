// A struct with a field that is not known to be `Copy` (here an enum with data, refused as GUST-E0211: an enum is not a struct field) gets no derive
// from the recovery: `derive(Copy)` on it would be an error of its own (E0204). expected.txt pins the one error.

#[gust::gpu_module]
mod shaping {
    pub enum Shape {
        Point(u32),
        Line,
    }

    pub struct Holder {
        pub shape: Shape,
        pub count: u32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
