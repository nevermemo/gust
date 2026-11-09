// An `impl` block of a GPU module must be an inherent block of a struct of the same module: `impl super::Outside`
// names a type the module does not define, GUST-E0121 (rustc accepts it, the type is in this crate). expected.txt pins
// the code and the line of the `impl`.

pub struct Outside {
    pub radius: f32,
}

#[gust::gpu]
mod shapes {
    pub struct Circle {
        pub radius: f32,
    }

    impl super::Outside {
        pub fn area(&self) -> f32 {
            1.0f32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, circles: StructuredBuffer<Circle>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = circles[id.x].radius;
    }
}

fn main() {}
