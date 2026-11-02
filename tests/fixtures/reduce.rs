#[gust::gpu]
mod reduce {
    // m7.r1: the passes of `gust_algo::reduce` (plan section 5.2). One module; each element type has, for `sum`,
    // `min` and `max`, a block-strided pass kernel and a terminal kernel.
    //
    // A pass over `input` (N elements, the kernel's `input.len()`) runs one invocation per output slot. Workgroup w
    // owns the block [w * 65536, (w + 1) * 65536); lane t reads `block + t + 256 * k` for k in 0..256 while the
    // index is below N (coalesced across the lanes, at most 256 terms per lane) and writes its own
    // `partials[256 * w + t]` (the identity when it has no element). The terminal kernel is one invocation that
    // combines the at most 256 remaining values into `out[0]`. Both are plain statement sequences: the only
    // operations are `+`, `<`, `>` and assignments, so `sum` is exact and in the domain for integers without
    // overflow of any partial, and `min` and `max` are exact for every input (NaN excepted for `f32`).

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn sum_u32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut partials: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 0u32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn sum_u32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 0u32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn min_u32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut partials: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 4294967295u32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn min_u32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 4294967295u32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn max_u32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut partials: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 0u32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn max_u32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 0u32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn sum_i32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut partials: RWStructuredBuffer<int>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 0i32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn sum_i32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 0i32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn min_i32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut partials: RWStructuredBuffer<int>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 2147483647i32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn min_i32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 2147483647i32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn max_i32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut partials: RWStructuredBuffer<int>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = -2147483647i32 - 1i32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn max_i32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = -2147483647i32 - 1i32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn sum_f32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut partials: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 0.0f32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn sum_f32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 0.0f32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                acc += input[index];
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn min_f32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut partials: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = 3.4028235e38f32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn min_f32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = 3.4028235e38f32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value < acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn max_f32_pass(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut partials: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < partials.len() {
            let n = input.len();
            let first = (i / 256u32) * 65536u32 + i % 256u32;
            let mut acc = -3.4028235e38f32;
            for k in 0u32..256u32 {
                let index = first + k * 256u32;
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            partials[i] = acc;
        }
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn max_f32_final(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        if id.x == 0u32 {
            let n = input.len();
            let mut acc = -3.4028235e38f32;
            for index in 0u32..256u32 {
                if index >= n {
                    break;
                }
                let value = input[index];
                if value > acc {
                    acc = value;
                }
            }
            out[0u32] = acc;
        }
    }
}
