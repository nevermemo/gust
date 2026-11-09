#[gust::gpu]
mod methods {
    pub struct Circle {
        pub radius: f32,
        pub scale: f32,
    }

    // One output buffer: the portable limit is eight storage buffers per shader stage.
    pub struct Results {
        pub area: f32,
        pub grown: f32,
        pub gap: f32,
        pub ring_area: f32,
        pub weight: f32,
        pub clamped: uint,
        pub extent: uint,
    }

    pub struct Span {
        pub lo: uint,
        pub hi: uint,
    }

    // A uniform reaches its struct rounded up to 16 bytes, so the padding is written out (see `uniforms.rs`).
    pub struct Disc {
        pub radius: f32,
        pub scale: f32,
        pub pad0: f32,
        pub pad1: f32,
    }

    pub struct Frame {
        pub disc: Disc,
    }

    impl Circle {
        // Scaled area with pi taken as 3: exact in f32 for the radii of the test.
        pub fn area(&self) -> f32 {
            3.0f32 * self.radius * self.radius * self.scale
        }

        pub fn grown(&self, by: f32) -> f32 {
            (self.radius + by) * self.scale
        }

        // A parameter named `this` is the receiver's Slang spelling: the emitter renames it (M9 review r1, M1).
        pub fn gap(&self, this: Circle) -> f32 {
            self.radius - this.radius
        }
    }

    impl Disc {
        pub fn weight(&self, by: f32) -> f32 {
            self.radius * by + self.scale
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

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        circles: StructuredBuffer<Circle>,
        spans: StructuredBuffer<Span>,
        input: StructuredBuffer<uint>,
        frame: Uniform<Frame>,
        mut ring: RWStructuredBuffer<Circle>,
        mut results: RWStructuredBuffer<Results>,
    ) {
        let i = id.x;
        if i < circles.len() {
            // A `&self` method on a storage element reads a copy of it.
            let area = circles[i].area();
            // The same method family on a local copy.
            let circle = circles[i];
            let grown = circle.grown(2.0f32);
            let clamped = spans[i].clamp_into(input[i]);
            let extent = spans[i].extent();
            let probe = Circle {
                radius: 0.25f32,
                scale: 1.0f32,
            };
            let gap = circles[i].gap(probe);
            // A `&self` call on an element of a read-write buffer, and on a field of a uniform.
            let ring_area = ring[i].area();
            let weight = frame.disc.weight(4.0f32);
            // The write makes `ring` a buffer the kernel changes (`mut`); the value is the one the host stored.
            ring[i].scale = 1.0f32;
            results[i] = Results {
                area,
                grown,
                gap,
                ring_area,
                weight,
                clamped,
                extent,
            };
        }
    }
}
