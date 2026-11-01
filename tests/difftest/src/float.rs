//! The interval arithmetic of the float oracle v1 (R1-4 of the M5 plan, `docs/development/DIFFTEST.md`).
//!
//! A float intermediate is an [`Fl`]: an `f64` hull of its exact value, plus the bookkeeping that turns the hull into
//! the interval of every value a conforming WGSL implementation may hold. The interpreter carries the `f32`
//! reference value (written order, round to nearest even) beside it.
//!
//! The rules, with their sources (`docs/SEMANTICS.md` lines 186-200, WGSL sections 15.7.2 to 15.7.6):
//!
//! - `+ - *`: "correctly rounded" means either neighbour of the exact result (SEMANTICS.md:190), so the result lies
//!   within one ULP of the hull of the exact operation over the operand intervals. The hull is computed in `f64`
//!   (rounding `2^-53`, relative) and widened by [`MARGIN`] (`2^-50`, deliberately coarse) per operation.
//! - A maximal chain of `+`/`-` with `n` terms may be regrouped (15.7.5): every grouping performs `n - 1` roundings,
//!   each within one ULP of a partial sum of magnitude at most `S = sum |t_i|max`. The chain is widened by `(n - 1)`
//!   ULPs of `S`, not by one ULP per written operation. `Fl` carries the roundings `r` and the sum `s` of the term
//!   magnitudes so that a chain continues through a variable (a loop accumulator is such a chain).
//! - A fused `a * b + c` (15.7.5, "at least as accurate") lies inside the chain interval, because the exact product
//!   is inside the product's interval.
//! - `x / y`: the divisor interval must lie inside `+-[2^-100, 2^100]` (so the GPU's divisor is non-zero and inside
//!   WGSL's `[2^-126, 2^126]` for the 2.5 ULP bound, SEMANTICS.md:192), and the quotient is widened by 3 ULP
//!   (2.5 rounded up).
//! - Every interval that is produced must be exactly `[0, 0]` or inside `+-[2^-100, 2^100]` on one side of zero:
//!   no subnormal (which 15.7.2 lets an implementation flush) and no infinity (which it may assume absent) can then
//!   occur on any party.

use crate::interp::Violation;

/// The relative margin added to every `f64` hull operation: `2^-50`, eight times `f64`'s own rounding.
pub const MARGIN: f64 = 1.0 / (1u64 << 50) as f64;
/// The smallest magnitude of a non-zero intermediate: `2^-100`.
pub const RANGE_MIN: f64 = 1.0 / (1u128 << 100) as f64;
/// The largest magnitude of an intermediate: `2^100`.
pub const RANGE_MAX: f64 = (1u128 << 100) as f64;
/// The output's relative width limit: `2^-10`.
pub const MAX_RELATIVE_WIDTH: f64 = 1.0 / 1024.0;
/// The largest integer magnitude converted to `f32` without the two-neighbour set of 15.7.6: `2^24`.
pub const EXACT_INT_LIMIT: u64 = 1 << 24;

/// The unit in the last place of an `f32` of magnitude `x` (`x` an `f64` that is at least `2^-126` or zero).
pub fn ulp32(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return 0.0;
    }
    let exponent = ((x.to_bits() >> 52) & 0x7FF) as i64 - 1023;
    let ulp_exponent = exponent.max(-126) - 23;
    f64::from_bits(((ulp_exponent + 1023) as u64) << 52)
}

/// A float intermediate before its roundings are added: the hull `[lo, hi]` of the exact value, the sum `s` of the
/// magnitudes of the terms of its chain, and the number `r` of roundings (an `f64`: a chain through variables can double per statement) of the chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fl {
    pub lo: f64,
    pub hi: f64,
    pub s: f64,
    pub r: f64,
}

impl Fl {
    /// A value known exactly (a literal, an input, an exactly converted integer).
    pub fn exact(value: f32) -> Fl {
        let v = f64::from(value);
        Fl {
            lo: v,
            hi: v,
            s: v.abs(),
            r: 0.0,
        }
    }

    /// A finished interval: no pending roundings.
    pub fn finished(lo: f64, hi: f64) -> Fl {
        Fl {
            lo,
            hi,
            s: lo.abs().max(hi.abs()),
            r: 0.0,
        }
    }

    /// Every value a conforming implementation may hold: the hull widened by the roundings of the chain.
    pub fn interval(&self) -> (f64, f64) {
        // A partial sum can exceed `s` by the roundings before it (each up to one ULP), so a rounding may happen in the
        // next binade up: bound its ULP at `s' = s + 2r ULP(s)` (Opus review of m5.c1a2).
        let s_up = self.s + 2.0 * self.r * ulp32(self.s);
        let w = self.r * (ulp32(s_up) + MARGIN * s_up);
        (self.lo - w, self.hi + w)
    }

    pub fn contains(&self, value: f64) -> bool {
        let (lo, hi) = self.interval();
        lo <= value && value <= hi
    }

    /// `-x`: exact, the chain bookkeeping is unchanged.
    pub fn negated(self) -> Fl {
        Fl {
            lo: -self.hi,
            hi: -self.lo,
            ..self
        }
    }

    /// `x + y`: the chains join (reassociation), one more rounding.
    pub fn plus(self, other: Fl) -> Fl {
        Fl {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
            s: self.s + other.s,
            r: self.r + other.r + 1.0,
        }
    }

    /// `x - y`.
    pub fn minus(self, other: Fl) -> Fl {
        Fl {
            lo: self.lo - other.hi,
            hi: self.hi - other.lo,
            s: self.s + other.s,
            r: self.r + other.r + 1.0,
        }
    }

    /// `x * y`: the hull of the product of the operand intervals, widened by one ULP; a finished interval.
    pub fn times(self, other: Fl) -> Fl {
        let ((al, ah), (bl, bh)) = (self.interval(), other.interval());
        let corners = [al * bl, al * bh, ah * bl, ah * bh];
        widened(&corners, 1.0)
    }

    /// `x / y`; `Err` if the divisor interval is not inside `+-[2^-100, 2^100]`.
    pub fn over(self, other: Fl) -> Result<Fl, Violation> {
        let ((al, ah), (bl, bh)) = (self.interval(), other.interval());
        if !(in_range(bl, bh) && !(bl == 0.0 && bh == 0.0)) {
            return Err(Violation::FloatDomain);
        }
        let corners = [al / bl, al / bh, ah / bl, ah / bh];
        Ok(widened(&corners, 3.0))
    }
}

/// The hull of `corners`, widened outward by `ulps` ULPs of its larger magnitude end and by the margin.
fn widened(corners: &[f64; 4], ulps: f64) -> Fl {
    let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m = lo.abs().max(hi.abs());
    let w = ulps * ulp32(m * (1.0 + MARGIN)) + MARGIN * m;
    Fl::finished(lo - w, hi + w)
}

/// Whether `[lo, hi]` is exactly `[0, 0]` or inside `+-[2^-100, 2^100]` without containing zero.
pub fn in_range(lo: f64, hi: f64) -> bool {
    if lo == 0.0 && hi == 0.0 {
        return true;
    }
    (RANGE_MIN <= lo && hi <= RANGE_MAX) || (-RANGE_MAX <= lo && hi <= -RANGE_MIN)
}

/// The truth value of `a op b` for the comparison `symbol`, if it is the same on every party: the two intervals
/// must be disjoint. `Err(ComparisonOverlap)` otherwise.
pub fn compare(symbol: &str, a: &Fl, b: &Fl) -> Result<bool, Violation> {
    let ((al, ah), (bl, bh)) = (a.interval(), b.interval());
    let (less, greater) = if ah < bl {
        (true, false)
    } else if bh < al {
        (false, true)
    } else {
        return Err(Violation::ComparisonOverlap);
    };
    Ok(match symbol {
        "<" | "<=" => less,
        ">" | ">=" => greater,
        "==" => false,
        "!=" => true,
        other => panic!("not a comparison: {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp32_is_the_f32_spacing() {
        assert_eq!(ulp32(1.0), f64::from(f32::EPSILON));
        assert_eq!(ulp32(1.999_999), f64::from(f32::EPSILON));
        assert_eq!(ulp32(2.0), 2.0 * f64::from(f32::EPSILON));
        assert_eq!(ulp32(0.5), 0.5 * f64::from(f32::EPSILON));
        assert_eq!(ulp32(-3.0), 2.0 * f64::from(f32::EPSILON));
        assert_eq!(ulp32(0.0), 0.0);
        assert_eq!(ulp32(16_777_216.0), 2.0);
        // Against the library: the next f32 above x is x + ulp.
        for bits in [
            0x3F80_0000u32,
            0x3F7F_FFFF,
            0x4B80_0000,
            0x0080_0000,
            0x7F00_0000,
        ] {
            let x = f32::from_bits(bits);
            let next = f32::from_bits(bits + 1);
            assert_eq!(
                f64::from(next) - f64::from(x),
                ulp32(f64::from(x)),
                "{bits:#x}"
            );
        }
    }

    #[test]
    fn the_range_rule() {
        assert!(in_range(0.0, 0.0));
        assert!(in_range(1.0, 2.0));
        assert!(in_range(-2.0, -1.0));
        assert!(in_range(RANGE_MIN, RANGE_MAX));
        assert!(
            !in_range(-1.0, 1.0),
            "an interval holding zero is not [0, 0]"
        );
        assert!(!in_range(0.0, 1.0));
        assert!(!in_range(RANGE_MIN / 2.0, 1.0));
        assert!(!in_range(1.0, RANGE_MAX * 2.0));
        assert!(!in_range(-RANGE_MAX * 2.0, -1.0));
    }

    #[test]
    fn a_single_addition_is_widened_by_one_ulp_of_the_term_sum() {
        let sum = Fl::exact(1.0).plus(Fl::exact(0.5));
        let (lo, hi) = sum.interval();
        let w = ulp32(1.5) + MARGIN * 1.5;
        assert_eq!((lo, hi), (1.5 - w, 1.5 + w));
        assert!(sum.contains(1.5));
        assert!(sum.contains(1.5 + f64::from(f32::EPSILON)));
        // Cancellation keeps the widening of the terms: `1 - 1` is not widened to zero width.
        let zero = Fl::exact(1.0).minus(Fl::exact(1.0));
        assert!(zero.interval().1 > 0.0);
    }

    #[test]
    fn a_chain_of_nine_terms_gets_eight_ulps_of_the_term_sum() {
        // The loop accumulator of R1-4: `acc = acc + t` eight times from a first term, nine terms in all.
        let mut acc = Fl::exact(1.0);
        for _ in 0..8 {
            acc = acc.plus(Fl::exact(1.0));
        }
        assert_eq!((acc.s, acc.r), (9.0, 8.0));
        let (lo, hi) = acc.interval();
        let w = 8.0 * (ulp32(9.0) + MARGIN * 9.0);
        assert_eq!((lo, hi), (9.0 - w, 9.0 + w));
        // Eight roundings of one ULP of 9 (= 2^-20) are 8 * 2^-20, far more than the written-order twin can be off.
        assert!(w >= 8.0 * 2.0f64.powi(-20));
        // Alternating signs: the partial sums stay small in the written order but not in every grouping, and the
        // interval is sized for the worst one (S counts every term's magnitude).
        let mut alt = Fl::exact(1.0);
        for k in 0..8 {
            alt = if k % 2 == 0 {
                alt.minus(Fl::exact(1.0))
            } else {
                alt.plus(Fl::exact(1.0))
            };
        }
        assert_eq!((alt.s, alt.r), (9.0, 8.0));
    }

    #[test]
    fn products_and_quotients_are_finished_intervals() {
        let p = Fl::exact(3.0).times(Fl::exact(0.1_f32));
        assert_eq!(p.r, 0.0);
        let exact = 3.0 * f64::from(0.1_f32);
        assert!(p.contains(exact));
        assert!(p.contains(f64::from(3.0f32 * 0.1_f32)));
        let (lo, hi) = p.interval();
        assert!(hi - lo >= 2.0 * ulp32(exact));
        assert!(hi - lo <= 2.0 * (ulp32(exact) + 4.0 * MARGIN * exact));
        let q = Fl::exact(1.0).over(Fl::exact(3.0)).unwrap();
        assert!(q.contains(f64::from(1.0f32 / 3.0f32)));
        let (lo, hi) = q.interval();
        assert!(hi - lo >= 6.0 * ulp32(1.0 / 3.0), "3 ULP each side");
        // Exact zero stays [0, 0] under a product.
        let z = Fl::exact(0.0).times(Fl::exact(5.0));
        assert_eq!(z.interval(), (0.0, 0.0));
    }

    #[test]
    fn division_needs_a_divisor_away_from_zero_and_inside_the_range() {
        let one = Fl::exact(1.0);
        assert_eq!(one.over(Fl::exact(0.0)), Err(Violation::FloatDomain));
        let straddle = Fl::finished(-1.0, 1.0);
        assert_eq!(one.over(straddle), Err(Violation::FloatDomain));
        let tiny = Fl::exact(2.0f32.powi(-101));
        assert_eq!(one.over(tiny), Err(Violation::FloatDomain));
        let huge = Fl::exact(2.0f32.powi(101));
        assert_eq!(one.over(huge), Err(Violation::FloatDomain));
        assert!(one.over(Fl::exact(-4.0)).is_ok());
        assert!(one.over(Fl::finished(2.0, 4.0)).is_ok());
    }

    #[test]
    fn comparisons_need_disjoint_intervals() {
        let a = Fl::exact(1.0);
        let b = Fl::exact(2.0);
        assert_eq!(compare("<", &a, &b), Ok(true));
        assert_eq!(compare("<=", &a, &b), Ok(true));
        assert_eq!(compare(">", &a, &b), Ok(false));
        assert_eq!(compare(">=", &b, &a), Ok(true));
        assert_eq!(compare("==", &a, &b), Ok(false));
        assert_eq!(compare("!=", &a, &b), Ok(true));
        // The same value, and values whose intervals touch or overlap, are all refused, `==` included.
        assert_eq!(compare("==", &a, &a), Err(Violation::ComparisonOverlap));
        assert_eq!(compare("<=", &a, &a), Err(Violation::ComparisonOverlap));
        let widened = Fl::exact(1.0).plus(Fl::exact(0.0));
        assert_eq!(
            compare("<", &widened, &a),
            Err(Violation::ComparisonOverlap)
        );
        let near = Fl::exact(1.0 + f32::EPSILON);
        assert_eq!(
            compare("<", &a.plus(Fl::exact(0.0)), &near),
            Err(Violation::ComparisonOverlap)
        );
        assert_eq!(
            compare("<", &Fl::exact(0.0), &Fl::exact(0.0)),
            Err(Violation::ComparisonOverlap)
        );
    }

    #[test]
    fn negation_is_exact_and_keeps_the_chain() {
        let chain = Fl::exact(1.0).plus(Fl::exact(2.0));
        let negated = chain.negated();
        let (lo, hi) = chain.interval();
        assert_eq!(negated.interval(), (-hi, -lo));
        assert_eq!((negated.s, negated.r), (chain.s, chain.r));
    }

    /// The chain `a + b + c` at a binade edge (Opus review of m5.c1a2): a conforming implementation may round `a + b`
    /// up to `scale`, and then `scale + c` up to the next `f32`, whose ULP is twice that of `a`. The ULP of the sum of
    /// the magnitudes alone does not cover that second step.
    fn binade_edge_chain(scale: f32, c_exponent: i32) -> (Fl, f32) {
        let a = scale * (1.0 - 2f32.powi(-24));
        let b = scale * 2f32.powi(-26);
        let c = 2f32.powi(c_exponent);
        // The exact values, to make sure the case is what it claims to be.
        assert!(
            f64::from(a) < f64::from(a) + f64::from(b)
                && f64::from(a) + f64::from(b) < f64::from(scale)
        );
        let up = f32::from_bits(scale.to_bits() + 1);
        assert!(
            f64::from(scale) < f64::from(scale) + f64::from(c)
                && f64::from(scale) + f64::from(c) < f64::from(up)
        );
        let chain = Fl::exact(a).plus(Fl::exact(b)).plus(Fl::exact(c));
        (chain, up)
    }

    #[test]
    fn a_chain_rounded_across_a_binade_edge_is_covered() {
        // a = 1 - 2^-24, b = 2^-26, c = 2^-30.
        let (chain, gpu) = binade_edge_chain(1.0, -30);
        assert!(chain.contains(f64::from(gpu)), "{:?}", chain.interval());
    }

    #[test]
    fn the_binade_edge_chain_at_scale_1024_with_an_input_sized_term_is_covered() {
        // a = 1024 (1 - 2^-24), b = 1024 * 2^-26, c = 2^-20 (the smallest input magnitude).
        let (chain, gpu) = binade_edge_chain(1024.0, -20);
        assert!(chain.contains(f64::from(gpu)), "{:?}", chain.interval());
    }
}
