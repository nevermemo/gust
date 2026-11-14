//! W1: vector-add, `out[i] = a[i] + b[i]`, f32. Memory-bandwidth bound: 12 bytes moved per element.

use gust::gpu;

/// The GUST kernel. Workgroup size 256 (see `workloads::WORKGROUP`).
#[gpu]
pub mod kernel {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}

/// The rayon / serial / reference element function. One IEEE add, so every backend must match it bit for bit.
#[inline(always)]
pub fn element(a: f32, b: f32) -> f32 {
    a + b
}

/// Absolute tolerance for one element: zero, the sum of two f32 values is exact on every backend.
pub fn tolerance(_a: f32, _b: f32, _want: f32) -> f32 {
    0.0
}
