/// Stage one: saturate every sample at 255. The tag `tidy` is declared in `Cargo.toml`, and `gust.toml` routes it to
/// the CPU: a small, cheap stage that is not worth a trip through the GPU.
#[gpu(route = "tidy")]
mod tidy {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn saturate(
        id: SV_DispatchThreadID,
        raw: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < raw.len() && i < out.len() {
            let mut v = raw[i];
            if v > 255u32 {
                v = 255u32;
            }
            out[i] = v;
        }
    }
}

/// Stage two: a weighted sum of each sample and its two neighbours (weights 1, 2, 1; the ends repeat themselves). The
/// tag `smooth` is routed to the GPU.
#[gpu(route = "smooth")]
mod smooth {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn blur(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = input.len();
        if i < n && i < out.len() {
            let mut left = i;
            if i > 0u32 {
                left = i - 1u32;
            }
            let mut right = i;
            if i + 1u32 < n {
                right = i + 1u32;
            }
            out[i] = input[left] + 2u32 * input[i] + input[right];
        }
    }
}
