//! W2: saxpy, `out[i] = scale * x[i] + y[i]`, f32, out of place. Memory-bandwidth bound: 12 bytes per element.
//!
//! The scalar travels as a one-element read-only storage buffer on both GPU paths (research section 2, D40).

use gust::gpu;

#[gpu]
pub mod kernel {
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn saxpy(
        id: SV_DispatchThreadID,
        scale: StructuredBuffer<float>,
        x: StructuredBuffer<float>,
        y: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && 0u32 < scale.len() {
            out[i] = scale[0u32] * x[i] + y[i];
        }
    }
}

/// The rayon / serial / reference element function: a multiply and an add, never fused.
#[inline(always)]
pub fn element(scale: f32, x: f32, y: f32) -> f32 {
    scale * x + y
}

/// Absolute tolerance for one element. A GPU compiler may contract the multiply and the add into one fused
/// multiply-add (one rounding instead of two), which differs from `element` by at most about one unit in the last
/// place of the operands' magnitude; allow two f32 epsilons of `|scale * x| + |y|`.
pub fn tolerance(scale: f32, x: f32, y: f32) -> f32 {
    2.0 * f32::EPSILON * ((scale * x).abs() + y.abs())
}
