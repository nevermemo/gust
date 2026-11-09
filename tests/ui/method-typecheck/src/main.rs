// Rustc type-checks the body of a method: the `bool` returned where the signature says `f32` is rustc's E0308 in the
// generated crate, so the validator's shape rules are not the only check on a method (M9 plan 2, F8). expected.txt pins
// the code and the line of the mismatch.

#[gust::gpu]
mod shapes {
    pub struct Circle {
        pub radius: f32,
    }

    impl Circle {
        pub fn big(&self) -> f32 {
            self.radius > 1.0f32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, circles: StructuredBuffer<Circle>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = circles[id.x].big();
    }
}

fn main() {}
