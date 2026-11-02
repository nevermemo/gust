#[gust::gpu]
mod vectors {
    pub struct Particle {
        pub position: Vec2,
        pub velocity: Vec2,
        pub mass: f32,
        pub tag: u32,
    }

    // A 3-wide type is allowed as a helper parameter and result, and as a local; it is never a buffer field.
    fn tint(color: Vec3, amount: f32) -> Vec3 {
        color + Vec3::splat(amount)
    }

    fn tilt(direction: Vec3) -> Vec3 {
        Mat3::IDENTITY.mul_vec3(direction)
    }

    // Columns are built by `from_cols`, applied by `mul_vec2`; `Mat2::IDENTITY` is a constant, not a call.
    fn spin(angle: Vec2, point: Vec2) -> Vec2 {
        let turn = Mat2::from_cols(Vec2::new(angle.x, angle.y), Vec2::new(-angle.y, angle.x));
        Mat2::IDENTITY.mul_mat2(turn).mul_vec2(point)
    }

    // The fixture writes methods only: `*` on a matrix is a later slice (m6.m2).
    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        points: StructuredBuffer<Vec4>,
        particles: StructuredBuffer<Particle>,
        counts: StructuredBuffer<UVec2>,
        mut moved: RWStructuredBuffer<Vec4>,
        mut sums: RWStructuredBuffer<float>,
        mut cells: RWStructuredBuffer<UVec2>,
        mut velocities: RWStructuredBuffer<Vec2>,
    ) {
        let i: uint = id.x;
        if i < moved.len() {
            let p = points[i];
            // Constructors, swizzles (the result width is the number of letters) and the dot product.
            let q = Vec4::new(p.w, p.z, p.y, p.x);
            let rgb = tint(p.xyz(), 0.5);
            let widened = Vec4::new(rgb.x, rgb.y, rgb.z, p.wzyx().x);
            sums[i] = p.dot(q) + widened.xy().dot(Vec2::splat(1.0)) + tilt(rgb).z;
            // `from_cols` is column-major and `from_rows` row-major; `mul_vec4` is M x v and `mul_mat4` a x b.
            let by_cols = Mat4::from_cols(
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(3.0, 4.0, 5.0, 1.0),
            );
            let by_rows = Mat4::from_rows(
                Vec4::new(2.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 2.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            );
            let both = by_cols.transpose().mul_mat4(by_rows);
            moved[i] = both.mul_vec4(Mat4::IDENTITY.mul_vec4(widened));
            // Integer vectors, and a vector struct literal (fields are assigned in written order).
            let n = counts[i];
            let c = UVec2 { x: n.x + i, y: n.y };
            cells[i] = c.yx();
            let part = particles[i];
            velocities[i] = spin(part.velocity, part.position);
        }
    }
}
