// Control case for methods (m9.a1): an `impl` block of a module struct with `&self` methods that take value parameters,
// called on a storage element, on a field of one and on a local copy. This crate must compile.

#[gust::gpu]
mod shapes {
    pub struct Circle {
        pub radius: f32,
        pub scale: f32,
    }

    impl Circle {
        pub fn area(&self) -> f32 {
            3.0f32 * self.radius * self.radius * self.scale
        }

        pub fn grown(&self, by: f32) -> f32 {
            (self.radius + by) * self.scale
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        circles: StructuredBuffer<Circle>,
        mut areas: RWStructuredBuffer<f32>,
        mut grown: RWStructuredBuffer<f32>,
    ) {
        if id.x < circles.len() {
            areas[id.x] = circles[id.x].area();
            let circle = circles[id.x];
            grown[id.x] = circle.grown(2.0f32);
        }
    }
}

fn main() {}
