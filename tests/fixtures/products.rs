#[gust::gpu]
mod products {
    // Every operand pair of `crates/gust-macros/src/slang/mul_table.rs` written as `a * b`, then the `*=` rows, then
    // the literal operands, then every operand type of `neg_table.rs` written as `-a`, through the macro. A test pins
    // that each row of both tables appears in the golden, and that slangc takes the golden on WGSL and SPIR-V. Each
    // result is read through one component so none is dropped.
    #[allow(clippy::too_many_arguments)]
    fn products(
        fl: f32,
        sn: i32,
        un: u32,
        v2: Vec2,
        v3: Vec3,
        v4: Vec4,
        i2: IVec2,
        i4: IVec4,
        u2: UVec2,
        u3: UVec3,
        u4: UVec4,
        m2: Mat2,
        m3: Mat3,
        m4: Mat4,
    ) -> f32 {
        let mut total = 0.0;
        total += fl * fl;
        total += (sn * sn) as f32;
        total += (un * un) as f32;
        total += (v2 * v2).x;
        total += (v2 * fl).x;
        total += (fl * v2).x;
        total += (v3 * v3).x;
        total += (v3 * fl).x;
        total += (fl * v3).x;
        total += (v4 * v4).x;
        total += (v4 * fl).x;
        total += (fl * v4).x;
        total += (i2 * i2).x as f32;
        total += (i2 * sn).x as f32;
        total += (sn * i2).x as f32;
        total += (i4 * i4).x as f32;
        total += (i4 * sn).x as f32;
        total += (sn * i4).x as f32;
        total += (u2 * u2).x as f32;
        total += (u2 * un).x as f32;
        total += (un * u2).x as f32;
        total += (u3 * u3).x as f32;
        total += (u3 * un).x as f32;
        total += (un * u3).x as f32;
        total += (u4 * u4).x as f32;
        total += (u4 * un).x as f32;
        total += (un * u4).x as f32;
        total += (m2 * m2).mul_vec2(Vec2::splat(1.0)).x;
        total += (m2 * v2).x;
        total += (m2 * fl).mul_vec2(Vec2::splat(1.0)).x;
        total += (fl * m2).mul_vec2(Vec2::splat(1.0)).x;
        total += (m3 * m3).mul_vec3(Vec3::splat(1.0)).x;
        total += (m3 * v3).x;
        total += (m3 * fl).mul_vec3(Vec3::splat(1.0)).x;
        total += (fl * m3).mul_vec3(Vec3::splat(1.0)).x;
        total += (m4 * m4).mul_vec4(Vec4::splat(1.0)).x;
        total += (m4 * v4).x;
        total += (m4 * fl).mul_vec4(Vec4::splat(1.0)).x;
        total += (fl * m4).mul_vec4(Vec4::splat(1.0)).x;

        let mut s0 = v2;
        s0 *= v2;
        total += (s0).x;
        let mut s1 = v2;
        s1 *= fl;
        total += (s1).x;
        let mut s2 = v3;
        s2 *= v3;
        total += (s2).x;
        let mut s3 = v3;
        s3 *= fl;
        total += (s3).x;
        let mut s4 = v4;
        s4 *= v4;
        total += (s4).x;
        let mut s5 = v4;
        s5 *= fl;
        total += (s5).x;
        let mut s6 = i2;
        s6 *= i2;
        total += (s6).x as f32;
        let mut s7 = i2;
        s7 *= sn;
        total += (s7).x as f32;
        let mut s8 = i4;
        s8 *= i4;
        total += (s8).x as f32;
        let mut s9 = i4;
        s9 *= sn;
        total += (s9).x as f32;
        let mut s10 = u2;
        s10 *= u2;
        total += (s10).x as f32;
        let mut s11 = u2;
        s11 *= un;
        total += (s11).x as f32;
        let mut s12 = u3;
        s12 *= u3;
        total += (s12).x as f32;
        let mut s13 = u3;
        s13 *= un;
        total += (s13).x as f32;
        let mut s14 = u4;
        s14 *= u4;
        total += (s14).x as f32;
        let mut s15 = u4;
        s15 *= un;
        total += (s15).x as f32;
        let mut s16 = m2;
        s16 *= fl;
        total += (s16).mul_vec2(Vec2::splat(1.0)).x;
        let mut s17 = m3;
        s17 *= fl;
        total += (s17).mul_vec3(Vec3::splat(1.0)).x;
        let mut s18 = m4;
        s18 *= fl;
        total += (s18).mul_vec4(Vec4::splat(1.0)).x;

        total += (u3 * 3).x as f32;
        total += (-3 * i4).x as f32;
        total += (2.0 * m4).mul_vec4(Vec4::splat(1.0)).x;
        total += (m2 * 0.5).mul_vec2(Vec2::splat(1.0)).x;
        total += (m3 * 2.0).mul_vec3(Vec3::splat(1.0)).x;
        total += (2 * u2).x as f32;
        total += (v3 * 2.0).x;
        total += (un * 3) as f32;
        total += (3 * un) as f32;
        total += (sn * -2) as f32;
        total += (-2 * sn) as f32;
        total += (2 * 3) as f32;

        // Unary minus (slice m6.n): one operand per row of `neg_table.rs`, in table order, then a negation inside a
        // product, a double negation and a float literal (which keeps the plain `-`). `gust::math` has no `Neg` for
        // `IVec2` and `IVec4` yet; slangc compiles this golden, rustc does not.
        total += -fl;
        total += (-sn) as f32;
        total += (-v2).x;
        total += (-v3).x;
        total += (-v4).x;
        total += (-i2).x as f32;
        total += (-i4).x as f32;
        total += (-m2).mul_vec2(Vec2::splat(1.0)).x;
        total += (-m3).mul_vec3(Vec3::splat(1.0)).x;
        total += (-m4).mul_vec4(Vec4::splat(1.0)).x;
        total += (-m4 * v4).x;
        total += -(-fl);
        total += -2.0 * fl;
        total
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
        let i: uint = id.x;
        if i < out.len() {
            out[i] = products(
                1.5,
                -2,
                3,
                Vec2::splat(2.0),
                Vec3::splat(2.0),
                Vec4::splat(2.0),
                IVec2::splat(2),
                IVec4::splat(2),
                UVec2::splat(2),
                UVec3::new(2, 2, 2),
                UVec4::splat(2),
                Mat2::IDENTITY,
                Mat3::IDENTITY,
                Mat4::IDENTITY,
            );
        }
    }
}
