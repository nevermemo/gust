#[gust::gpu]
mod methods {
    pub struct Circle {
        pub radius: f32,
        pub scale: f32,
    }

    pub struct Span {
        pub lo: uint,
        pub hi: uint,
    }

    impl Circle {
        // Scaled area with pi taken as 3: exact in f32 for the radii of the test.
        pub fn area(&self) -> f32 {
            3.0f32 * self.radius * self.radius * self.scale
        }

        pub fn grown(&self, by: f32) -> f32 {
            (self.radius + by) * self.scale
        }
    }

    impl Span {
        pub fn extent(&self) -> uint {
            self.hi - self.lo
        }

        // `return` from inside a method, then the tail value.
        pub fn clamp_into(&self, value: uint) -> uint {
            if value < self.lo {
                return self.lo;
            }
            if value > self.hi {
                return self.hi;
            }
            value
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        circles: StructuredBuffer<Circle>,
        spans: StructuredBuffer<Span>,
        input: StructuredBuffer<uint>,
        mut areas: RWStructuredBuffer<f32>,
        mut grown: RWStructuredBuffer<f32>,
        mut clamped: RWStructuredBuffer<uint>,
        mut extents: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < circles.len() {
            // A `&self` method on a storage element reads a copy of it.
            areas[i] = circles[i].area();
            // The same method family on a local copy.
            let circle = circles[i];
            grown[i] = circle.grown(2.0f32);
            clamped[i] = spans[i].clamp_into(input[i]);
            extents[i] = spans[i].extent();
        }
    }
}
