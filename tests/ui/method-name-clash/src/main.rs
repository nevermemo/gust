// A method cannot take the name of a method of the dialect: `len` is the length of a buffer, so `fn len(&self)` is
// GUST-E0328. expected.txt pins the code and the line of the method.

#[gust::gpu]
mod shapes {
    pub struct Circle {
        pub radius: f32,
    }

    impl Circle {
        pub fn len(&self) -> f32 {
            self.radius
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, circles: StructuredBuffer<Circle>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = circles[id.x].len();
    }
}

fn main() {}
