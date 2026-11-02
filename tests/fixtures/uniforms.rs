#[gust::gpu]
mod uniforms {
    // GUST never inserts padding: the fields written here are the whole constant-buffer layout. A struct a uniform
    // reaches is rounded up to 16 bytes, so `Light` fills its last 8 bytes with explicit fields and `Frame` is built so
    // that every field sits where std140 puts it (the 8-byte `offset` before the two scalars, 128 bytes in all).
    pub struct Light {
        pub color: Vec4,
        pub intensity: f32,
        pub falloff: f32,
        pub pad0: f32,
        pub pad1: f32,
    }

    pub struct Frame {
        pub view: Mat4,
        pub tint: Vec4,
        pub light: Light,
        pub offset: Vec2,
        pub scale: f32,
        pub count: u32,
    }

    // The uniform is read one field at a time; a nested struct and a matrix are fields like the others.
    fn shade(base: Vec4, light: Light) -> Vec4 {
        base * light.intensity + light.color * light.falloff
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        frame: Uniform<Frame>,
        points: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.x;
        if i < out.len() {
            let moved = frame.view.mul_vec4(points[i]) + frame.tint;
            let lit = shade(moved, frame.light);
            out[i] = Vec4::new(
                lit.x + frame.offset.x,
                lit.y + frame.offset.y,
                lit.z * frame.scale,
                lit.w + frame.count as f32,
            );
        }
    }
}
