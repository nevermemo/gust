//! The rust-gpu kernels of the `rustgpu-wgpu` bench comparator (m10.b4): W1 (vector add), W2 (saxpy), W3 (Horner
//! polynomial) and W4 (histogram), each the same algorithm, workgroup size (256) and storage-buffer binding order as the
//! hand-written WGSL of the same workload (`bench/shaders/*.wgsl`). Every binding is in descriptor set 0.
//!
//! Where a kernel differs from its hand WGSL, its doc comment says so. In general rust-gpu ends an invocation whose slice
//! index is out of range early (a bounds-check panic becomes `OpReturn` in the linked module), so the explicit length
//! guards of the hand WGSL are kept and the implicit checks are rust-gpu's own.
#![no_std]

use spirv_std::glam::UVec3;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

/// W1 on `u32` (`out[i] = a[i] + b[i]`, wrapping), guarded by the output length. There is no hand WGSL of this form: the
/// bench's W1 is `f32`. This entry point is built and validated with the module; no workload runs it.
#[spirv(compute(threads(256)))]
pub fn w1_u32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [u32],
) {
    let i = id.x as usize;
    if i < out.len() {
        out[i] = a[i].wrapping_add(b[i]);
    }
}

/// W1 on `f32` (`vector_add.wgsl`): `out[i] = a[i] + b[i]`, guarded by the output length; bindings 0 and 1 read-only, 2
/// read-write. Same as the hand WGSL apart from the implicit index checks.
#[spirv(compute(threads(256)))]
pub fn w1_f32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
) {
    let i = id.x as usize;
    if i < out.len() {
        out[i] = a[i] + b[i];
    }
}

/// W2 (`saxpy.wgsl`): `out[i] = scale[0] * x[i] + y[i]`. The scalar is a one-element read-only storage buffer at binding
/// 0, as in the hand WGSL and the GUST kernel (no uniform, so no side gets a cheaper scalar path); `x`, `y`, `out` are
/// bindings 1, 2 and 3. The guard is the hand WGSL's: the output length and a non-empty `scale`.
#[spirv(compute(threads(256)))]
pub fn w2_saxpy(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scale: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] y: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
) {
    let i = id.x as usize;
    if i < out.len() && !scale.is_empty() {
        out[i] = scale[0] * x[i] + y[i];
    }
}

/// W3 (`polynomial.wgsl`): Horner evaluation, coefficients at binding 0 (the degree is its length minus one), `x` at 1,
/// `out` at 2. The loop is bounded by 64 and leaves at the degree, as the hand WGSL's does, so one entry point serves
/// degree 4, 16 and 64. Guards: the output length, the `x` length and at least two coefficients.
#[spirv(compute(threads(256)))]
pub fn w3_horner(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] coeffs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
) {
    let i = id.x as usize;
    if i < out.len() && i < x.len() && coeffs.len() > 1 {
        let degree = coeffs.len() - 1;
        let v = x[i];
        let mut acc = coeffs[degree];
        let mut k = 0usize;
        while k < 64 {
            if k >= degree {
                break;
            }
            acc = acc * v + coeffs[degree - 1 - k];
            k += 1;
        }
        out[i] = acc;
    }
}

/// W4 (`histogram.wgsl`): keys at binding 0 (read-only), bins at binding 1 (`u32`, atomic add). A workgroup owns a block
/// of 256 * 4 contiguous keys; each lane takes four of them, strided by 256, and ignores a key at or above the bin count.
/// The add is `OpAtomicIAdd` at queue-family scope (the widest scope the Vulkan memory model allows without its
/// device-scope capability, which wgpu does not enable on a device) with no memory semantics, the relaxed add the WGSL
/// `atomicAdd` is; every invocation of the dispatch is inside that scope. The bins
/// are a plain `&mut [u32]` here (rust-gpu has no atomic array type); the atomic is the instruction on the element.
#[spirv(compute(threads(256)))]
pub fn w4_histogram(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] bins: &mut [u32],
) {
    let i = id.x as usize;
    let n = keys.len();
    let first = (i / 256) * 1024 + i % 256;
    let mut k = 0usize;
    while k < 4 {
        let index = first + k * 256;
        if index >= n {
            break;
        }
        let bin = keys[index] as usize;
        if bin < bins.len() {
            // SAFETY: `bin` is in range of the binding, and the add is the only access to that word in this kernel.
            unsafe {
                spirv_std::arch::atomic_i_add::<
                    u32,
                    { Scope::QueueFamily as u32 },
                    { Semantics::NONE.bits() },
                >(&mut bins[bin], 1);
            }
        }
        k += 1;
    }
}
