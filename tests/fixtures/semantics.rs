#[gust::gpu_module]
mod semantics {
    fn score(x: float) -> float {
        if x > 0.0f32 {
            x * 2.5f32 + 1.0f32
        } else if x < 0.0f32 {
            -x
        } else {
            7.0f32
        }
    }

    fn mask(x: uint) -> uint {
        let divisor: uint = 2;
        let mut quotient: uint = x / divisor;
        quotient += 1u32;
        if x & 1u32 == 0u32 {
            (!x & 0xff_u32) + 0b10u32 + quotient
        } else {
            x + 1_024u32
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<float>,
        mut floats: RWStructuredBuffer<float>,
        mut integers: RWStructuredBuffer<uint>,
    ) {
        let i: uint = id.x;
        if i < floats.len() {
            floats[i] = score(values[i]);
            integers[i] = mask(i);
            if !(i < 2u32) {
                floats[i] += 1.0f32;
            }
        }
    }
}
