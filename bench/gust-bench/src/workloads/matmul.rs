//! W9 (naive): the product of two `N x N` f32 matrices, `C = A B`, one output entry per invocation, each entry the dot
//! product of a row of `A` and a column of `B` accumulated in `k` order. `N` is 256, 512, 1024 or 2048 (`Inputs::n` is
//! `N`, the inputs and the output hold `N * N` words, row-major). The tiled form (shared memory) is W9's other half, not
//! this slice's.
//!
//! The GUST kernel is a `grid` kernel with a 16 by 16 workgroup (256 invocations, as every GPU kernel of this harness) and
//! a 2-D extent of `N x N`; the matrix size comes from a one-word buffer (`dims`), the stand-in W2's scalar also uses. The
//! hand WGSL (`shaders/matmul.wgsl`) has the same bindings, guards and loop; `gust-twin` runs the kernel's shadow body;
//! rayon and the serial row are the same loop over host slices (`ijk`, the column of `B` strided: the naive form).
//!
//! # The bound (derived here, plan 2.1)
//!
//! Model every f32 multiply and add of a backend, fused or not, as `fl(a op b) = (a op b)(1 + delta) + eta` with
//! `|delta| <= u` and `|eta| <= 2^-126`. The unit roundoff is the backend's: `u = 2^-24` for the CPU backends (rayon,
//! serial, `gust-twin`: host arithmetic rounds to nearest) and `u = 2^-23` for every GPU backend (GUST, its SPIR-V
//! route, the hand WGSL), because `docs/SEMANTICS.md` section 4 lets a GPU add or multiply round to either neighbour,
//! within 1 ULP (`gust_algo::bound::C_GPU`); the check picks the bound by the backend's `Rounding`. An entry is an inner product of length `N`, `N` multiplies and `N - 1`
//! adds in whatever order the backend groups them (a recursive sum is depth `N - 1`, any other tree is shallower), and
//! Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 3.1 (eq. 3.5 for the inner product, and
//! eq. 3.13 for the product of matrices), gives for the computed `C^`:
//!
//! ```text
//! |C^ - C| <= gamma_N |A| |B|   (entrywise),   gamma_k = k u / (1 - k u)
//! ```
//!
//! for the exact product `C` of the f32 inputs (exact in f64). A fused multiply-add rounds once where the model rounds
//! twice, so it has only less error. The reference is the product in f64 (`c`), which obeys the same theorem with
//! `u64 = 2^-53`, and `s = |A||B|` is computed in f64 beside it, with relative error at most `gamma64_N` (all terms are
//! non-negative, no cancellation), so the code scales by `1 + gamma64_N`. The check compares `C^` with `c_f64`:
//!
//! ```text
//! |c^_ij - c_f64_ij| <= (gamma_N + gamma64_N) * s_ij * (1 + gamma64_N) + 2 N * 2^-126
//! ```
//!
//! The last term is the underflow allowance: each of the `2 N` operations of an entry adds at most `2^-126`. It is not
//! multiplied by `1 + gamma`, so it is a first-order allowance; it is moot here, because the smallest non-zero product is
//! `2^-46` (the inputs are multiples of `2^-23`), so no product, partial sum or result is subnormal and the term never
//! carries error (a test pins the input grid). A backend that fails the bound fails the cell; the bound is never widened
//! to pass. Domain (a test pins it): finite inputs in `[-1, 1)`, so no partial sum passes `N <= 2048`.

use gust::gpu;

use super::{
    mix,
    polynomial::{self, gamma},
};

/// The GUST kernel: `c[row * n + col] = sum_k a[row * n + k] * b[k * n + col]`, the matrix size `n` read from `dims[0]`.
/// Workgroup 16 by 16 (256 invocations). Every buffer is `Independent`: the dispatch is a whole number of workgroups.
#[gpu]
pub mod kernel {
    #[allow(clippy::len_zero, clippy::assign_op_pattern)]
    #[kernel(workgroup_size(16, 16, 1))]
    pub fn matmul(
        id: SV_DispatchThreadID,
        dims: StructuredBuffer<uint>,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut c: RWStructuredBuffer<float>,
    ) {
        if 0u32 < dims.len() {
            let n = dims[0u32];
            let row = id.y;
            let col = id.x;
            if row < n && col < n && row * n + col < c.len() {
                let mut acc = 0.0f32;
                for k in 0u32..n {
                    acc = acc + a[row * n + k] * b[k * n + col];
                }
                c[row * n + col] = acc;
            }
        }
    }
}

/// Workgroup edge of the 2-D kernels: 16 by 16 is 256 invocations, the harness's workgroup size.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub const TILE: usize = 16;

/// The two input matrices (`A`, `B`), row-major, entries `(r24 - 2^23) / 2^23` in `[-1, 1)`: 24-bit integers over a power
/// of two, so each is exact in f32, and every product needs 48 bits, so the f32 products round.
pub fn matrices(n: usize) -> (Vec<f32>, Vec<f32>) {
    let entry = |salt: u64, i: u64| {
        let r24 = (mix(i ^ salt) >> 40) as i32;
        (r24 - 8_388_608) as f32 / 8_388_608.0
    };
    let count = (n * n) as u64;
    (
        (0..count)
            .map(|i| entry(0x1111_0000_0000_0000, i))
            .collect(),
        (0..count)
            .map(|i| entry(0x7777_0000_0000_0000, i))
            .collect(),
    )
}

/// The exact product `c` (in f64) and the derived absolute bound of the f32 result at each entry (`N * N` of each), for
/// a backend that rounds to nearest (`u = 2^-24`) and for a GPU one (`u = 2^-23`).
pub fn exact_and_bound(n: usize, a: &[f32], b: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let factor = gamma(n, polynomial::U32) + gamma(n, polynomial::U64);
    let factor_gpu = gamma(n, polynomial::U32_GPU) + gamma(n, polynomial::U64);
    let s_slack = 1.0 + gamma(n, polynomial::U64);
    let eta = 2.0 * n as f64 * f64::from(f32::MIN_POSITIVE);
    let mut c = vec![0.0f64; n * n];
    let mut s = vec![0.0f64; n * n];
    // `ikj`: for every (i, j) the terms are added in ascending k, one row of C at a time (the cache-friendly order of a
    // reference that runs once per case). The theorem holds for any order.
    for i in 0..n {
        let (c_row, s_row) = (&mut c[i * n..(i + 1) * n], &mut s[i * n..(i + 1) * n]);
        for k in 0..n {
            let aik = f64::from(a[i * n + k]);
            let abs = aik.abs();
            for (j, bkj) in b[k * n..(k + 1) * n].iter().enumerate() {
                let bkj = f64::from(*bkj);
                c_row[j] += aik * bkj;
                s_row[j] += abs * bkj.abs();
            }
        }
    }
    let bound = s.iter().map(|s| factor * s * s_slack + eta).collect();
    let bound_gpu = s.iter().map(|s| factor_gpu * s * s_slack + eta).collect();
    (c, bound, bound_gpu)
}

/// One row of the product in f32, `ijk`: a plain multiply then an add per term, never fused, `k` ascending (the CPU
/// backends' element function; the GPU kernel's order, with the column of `B` strided).
#[inline(always)]
pub fn row(n: usize, a: &[f32], b: &[f32], i: usize, out: &mut [f32]) {
    let a_row = &a[i * n..(i + 1) * n];
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, aik) in a_row.iter().enumerate() {
            acc += aik * b[k * n + j];
        }
        *o = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Rounding, Workload, check, check_on, gen_inputs, reference};

    fn f32_product(n: usize, a: &[f32], b: &[f32], fused: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc = if fused {
                        a[i * n + k].mul_add(b[k * n + j], acc)
                    } else {
                        acc + a[i * n + k] * b[k * n + j]
                    };
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn the_domain_the_derivation_needs_holds_for_the_inputs() {
        let (a, b) = matrices(64);
        assert_eq!((a.len(), b.len()), (64 * 64, 64 * 64));
        assert!(a.iter().chain(&b).all(|v| (-1.0..1.0).contains(v)));
        assert!(a.iter().chain(&b).all(|v| *v == 0.0 || v.is_normal()));
        assert_ne!(a, b, "the two matrices differ");
        // The inputs are multiples of 2^-23 (so every non-zero product is at least 2^-46, far above the subnormals).
        assert!(
            a.iter()
                .chain(&b)
                .all(|v| (f64::from(*v) * 8_388_608.0).fract() == 0.0)
        );
        // The largest N (2048) keeps every partial sum below N (|entry| < 1): far from the f32 range.
        // Full mantissas: the f32 products round, so the bound is exercised.
        let product = f32_product(32, &matrices(32).0, &matrices(32).1, false);
        let (exact, _, _) = exact_and_bound(32, &matrices(32).0, &matrices(32).1);
        assert!(
            product.iter().zip(&exact).any(|(p, e)| f64::from(*p) != *e),
            "an f32 product rounds"
        );
    }

    /// The factor is about `2 N u`: 2^-14 at N = 2048, so the bound is tight against the rounding error it covers (a
    /// bound of 1 would be vacuous).
    #[test]
    fn the_bound_is_a_small_multiple_of_the_unit_roundoff() {
        let factor = gamma(2048, polynomial::U32) + gamma(2048, polynomial::U64);
        assert!((factor - 2048.0 * polynomial::U32).abs() < 1e-6);
        assert!(factor < 1.3e-4);
        // A GPU's is twice that (u = 2^-23).
        let gpu = gamma(2048, polynomial::U32_GPU) + gamma(2048, polynomial::U64);
        assert!((gpu - 2048.0 * polynomial::U32_GPU).abs() < 1e-6);
        assert!(gpu < 2.6e-4);
    }

    /// The plain f32 product, the fused one and the CPU backends' row function are inside the bound.
    #[test]
    fn backends_that_round_differently_stay_inside_the_bound() {
        let w = Workload::MatMul;
        let n = 48;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        let plain = f32_product(n, &inputs.x, &inputs.y, false);
        let fused = f32_product(n, &inputs.x, &inputs.y, true);
        let mut by_rows = vec![0.0f32; n * n];
        for i in 0..n {
            row(n, &inputs.x, &inputs.y, i, &mut by_rows[i * n..(i + 1) * n]);
        }
        assert_eq!(by_rows, plain, "row() is the unfused ijk loop");
        assert_ne!(plain, fused, "fusing changes some entry's rounding");
        check(w, &inputs, &reference, &plain).unwrap();
        check(w, &inputs, &reference, &fused).unwrap();
        for rounding in [Rounding::Nearest, Rounding::Either] {
            check_on(w, &inputs, &reference, &plain, rounding).unwrap();
            check_on(w, &inputs, &reference, &fused, rounding).unwrap();
        }
        // The tight check is a real one: the f32 product is not the f64 one.
        assert!(
            plain
                .iter()
                .zip(&reference.exact)
                .any(|(p, e)| f64::from(*p) != *e)
        );
    }

    #[test]
    fn a_result_just_past_the_bound_fails_and_the_one_just_inside_passes() {
        let w = Workload::MatMul;
        let n = 32;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        let result = f32_product(n, &inputs.x, &inputs.y, false);
        let i = 5 * n + 9;
        let (exact, bound) = (reference.exact[i], reference.bound[i]);
        // The GPU bound (u = 2^-23) is about twice the round-to-nearest one, and a value between the two is the GPU's only.
        let far = reference.bound_gpu[i];
        assert!(far > bound && far < 2.1 * bound);
        let mut between = result.clone();
        between[i] = (exact + 0.5 * (bound + far)) as f32;
        check_on(w, &inputs, &reference, &between, Rounding::Either).unwrap();
        assert!(check_on(w, &inputs, &reference, &between, Rounding::Nearest).is_err());
        assert!(bound > 0.0 && bound < 1e-2, "tight, not vacuous: {bound}");
        // The largest f32 within the bound of entry i, and the next one above it.
        let mut inside = (exact + bound) as f32;
        while f64::from(inside) - exact > bound {
            inside = inside.next_down();
        }
        while f64::from(inside.next_up()) - exact <= bound {
            inside = inside.next_up();
        }
        let outside = inside.next_up();
        assert!(f64::from(outside) - exact > bound);
        let mut ok = result.clone();
        ok[i] = inside;
        check(w, &inputs, &reference, &ok).unwrap();
        let mut bad = result.clone();
        bad[i] = outside;
        let error = check(w, &inputs, &reference, &bad).unwrap_err();
        assert!(
            error.starts_with(&format!("1 of {} elements are wrong; first at {i}", n * n)),
            "{error}"
        );
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = result.clone();
            bad[i] = poison;
            assert!(check(w, &inputs, &reference, &bad).is_err());
        }
        // The transposed product (a wrong index) fails somewhere, the poison fails everywhere.
        let wrong = f32_product(n, &inputs.y, &inputs.x, false);
        assert!(check(w, &inputs, &reference, &wrong).is_err());
        assert!(check(w, &inputs, &reference, &vec![w.poison_word(); n * n]).is_err());
        assert!(check(w, &inputs, &reference, &result[1..]).is_err());
    }
}
