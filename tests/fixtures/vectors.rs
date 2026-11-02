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

    // The integer dot product, kept in its own helper: it is the one form whose SPIR-V needs the `DotProduct`
    // capability, which naga's SPIR-V reader refuses (docs/DIAGNOSTICS.md, E0305 note). Its WGSL is fine.
    fn integer_dots(a: IVec2, b: IVec4, c: UVec2, d: UVec4) -> int {
        a.dot(IVec2::new(3, -2)) + b.dot(IVec4::splat(1)) + (c.dot(UVec2::splat(2)) + d.dot(UVec4::splat(1))) as int
    }

    // Integer constructors and swizzles, including the 3-wide `UVec3` as a helper parameter and local.
    fn integer_mix(wide: UVec3, signed: IVec4) -> IVec4 {
        let low = UVec2::new(wide.x, wide.y).yx();
        let folded = UVec3::new(low.x, low.y, wide.z).zyx();
        let tail = signed.wzyx();
        IVec4::new(folded.x as int, folded.y as int, tail.x, signed.xy().y)
    }

    // Row and column constructors at 2 and 3 wide, `transpose` and `mul_mat` at both, and a swizzle that widens.
    fn matrices(v2: Vec2, v3: Vec3) -> Vec3 {
        let rows2 = Mat2::from_rows(v2, v2.yx());
        let flipped = rows2.transpose().mul_mat2(rows2);
        let cols3 = Mat3::from_cols(v3, v3.zyx(), v3.yzx());
        let rows3 = Mat3::from_rows(v3, v3.yzx(), v3.zxy());
        let product = cols3.transpose().mul_mat3(rows3);
        product.mul_vec3(flipped.mul_vec2(v2).xyx())
    }

    // The fixture writes methods only: `*` on a matrix is a later slice (m6.m2).
    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        points: StructuredBuffer<Vec4>,
        particles: StructuredBuffer<Particle>,
        counts: StructuredBuffer<UVec2>,
        signed_in: StructuredBuffer<IVec4>,
        wide_in: StructuredBuffer<UVec4>,
        mut moved: RWStructuredBuffer<Vec4>,
        mut sums: RWStructuredBuffer<float>,
        mut cells: RWStructuredBuffer<UVec2>,
        mut velocities: RWStructuredBuffer<Vec2>,
        mut signed_out: RWStructuredBuffer<IVec4>,
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
            let signed = signed_in[i];
            let wide = wide_in[i];
            signed_out[i] = integer_mix(wide.xyz(), signed);
            sums[i] = sums[i] + (integer_dots(signed.xy(), signed, wide.xy(), wide) as f32) + matrices(part.position, rgb).x;
        }
    }
}
