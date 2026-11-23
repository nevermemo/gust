//! The scalar operations of the interpreter, with the domain checks.

use super::*;

/// Evaluates `l op r` for operands of type `ty` (the left operand's type; `r` of a shift is `u32`). Both
/// operands are 32-bit patterns; the result is a pattern (`bool` as 0 or 1). `&&` and `||` are evaluated
/// without short-circuiting here; `Machine::expr` short-circuits before calling this. Integers and `bool` only.
pub fn binary(op: BinOp, ty: Ty, l: u32, r: u32) -> Result<u32, Violation> {
    let (a, b) = (l as i32, r as i32);
    let signed = ty == Ty::I32;
    let value = match op {
        BinOp::Add if signed => a.checked_add(b).map(|v| v as u32),
        BinOp::Add => l.checked_add(r),
        BinOp::Sub if signed => a.checked_sub(b).map(|v| v as u32),
        BinOp::Sub => l.checked_sub(r),
        BinOp::Mul if signed => a.checked_mul(b).map(|v| v as u32),
        BinOp::Mul => l.checked_mul(r),
        BinOp::Div | BinOp::Rem => {
            if r == 0 {
                return Err(Violation::DivisionByZero);
            }
            if signed {
                if a == i32::MIN && b == -1 {
                    return Err(Violation::MinOverMinusOne);
                }
                Some((if op == BinOp::Div { a / b } else { a % b }) as u32)
            } else {
                Some(if op == BinOp::Div { l / r } else { l % r })
            }
        }
        BinOp::BitAnd => Some(l & r),
        BinOp::BitOr => Some(l | r),
        BinOp::BitXor => Some(l ^ r),
        BinOp::Shl | BinOp::Shr => {
            if r >= 32 {
                return Err(Violation::ShiftAmount);
            }
            Some(match (op, signed) {
                (BinOp::Shl, _) => l << r,
                (_, true) => (a >> r) as u32,
                (_, false) => l >> r,
            })
        }
        BinOp::Eq => Some(u32::from(l == r)),
        BinOp::Ne => Some(u32::from(l != r)),
        BinOp::Lt => Some(u32::from(if signed { a < b } else { l < r })),
        BinOp::Le => Some(u32::from(if signed { a <= b } else { l <= r })),
        BinOp::Gt => Some(u32::from(if signed { a > b } else { l > r })),
        BinOp::Ge => Some(u32::from(if signed { a >= b } else { l >= r })),
        BinOp::And => Some(u32::from(l != 0 && r != 0)),
        BinOp::Or => Some(u32::from(l != 0 || r != 0)),
    };
    value.ok_or(Violation::Overflow)
}

/// Evaluates `op e` for an integer or `bool` operand of type `ty`.
pub fn unary(op: UnOp, ty: Ty, e: u32) -> Result<u32, Violation> {
    match (op, ty) {
        (UnOp::Neg, _) => (e as i32)
            .checked_neg()
            .map(|v| v as u32)
            .ok_or(Violation::Overflow),
        (UnOp::Not, Ty::Bool) => Ok(e ^ 1),
        (UnOp::Not, _) => Ok(!e),
    }
}

/// `l.rem_euclid(r)` for operands of type `ty` (`u32` or `i32`), as 32-bit patterns. Rust panics on a zero divisor and
/// on `i32::MIN.rem_euclid(-1)`, so those are the violations of `/` and `%` (D38).
pub fn rem_euclid(ty: Ty, l: u32, r: u32) -> Result<u32, Violation> {
    if r == 0 {
        return Err(Violation::DivisionByZero);
    }
    if ty == Ty::I32 {
        let (a, b) = (l as i32, r as i32);
        if a == i32::MIN && b == -1 {
            return Err(Violation::MinOverMinusOne);
        }
        Ok(a.rem_euclid(b) as u32)
    } else {
        Ok(l.rem_euclid(r))
    }
}

/// The value of module constant `K<k>` of `methods`: its initializer evaluated with the checked operations of Rust,
/// which refuses a constant that overflows, divides by zero or shifts by 32 or more (rustc evaluates the item).
pub fn const_value(methods: &Methods, k: usize) -> Result<u32, Violation> {
    fn value(methods: &Methods, e: &Expr) -> Result<u32, Violation> {
        match e {
            Expr::Lit(_, bits) => Ok(*bits),
            Expr::Const { k, .. } => const_value(methods, usize::from(*k)),
            Expr::Un { op, ty, e } => unary(*op, *ty, value(methods, e)?),
            Expr::Bin { op, ty, l, r } => binary(*op, *ty, value(methods, l)?, value(methods, r)?),
            other => unreachable!("a constant initializer holds {other:?}"),
        }
    }
    value(methods, &methods.consts[k].init)
}

/// A value of the machine: an integer or `bool` pattern, an `f32` with its reference value and its interval, or a
/// vector or matrix of such components (`agg.rs`).
// A `Val` is copied by value on every read of a variable: boxing the aggregate would allocate per vector operation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
pub(super) enum Val {
    I(u32),
    F(f32, Fl),
    A(Agg),
}

impl Val {
    pub(super) fn int(self) -> u32 {
        match self {
            Val::I(bits) => bits,
            Val::F(..) | Val::A(_) => {
                unreachable!("a float or aggregate where the program is typed to hold an integer")
            }
        }
    }

    pub(super) fn float(self) -> (f32, Fl) {
        match self {
            Val::F(reference, fl) => (reference, fl),
            Val::I(_) | Val::A(_) => {
                unreachable!("an integer or aggregate where the program is typed to hold a float")
            }
        }
    }

    pub(super) fn agg(self) -> Agg {
        match self {
            Val::A(agg) => agg,
            _ => unreachable!("a scalar where the program is typed to hold a vector or matrix"),
        }
    }

    /// The components of a value as a slice-able array: a scalar is one component.
    pub(super) fn scalar(self) -> Scalar {
        match self {
            Val::I(bits) => Scalar::I(bits),
            Val::F(reference, fl) => Scalar::F(reference, fl),
            Val::A(_) => unreachable!("an aggregate where the program is typed to hold a scalar"),
        }
    }
}

impl From<Scalar> for Val {
    fn from(scalar: Scalar) -> Val {
        match scalar {
            Scalar::I(bits) => Val::I(bits),
            Scalar::F(reference, fl) => Val::F(reference, fl),
        }
    }
}

/// `Ok` if the finished interval of `fl` obeys the range rule (R1-4: every intermediate).
pub(super) fn in_domain(fl: Fl) -> Result<Fl, Violation> {
    let (lo, hi) = fl.interval();
    if float::in_range(lo, hi) {
        Ok(fl)
    } else {
        Err(Violation::FloatDomain)
    }
}

/// The reference value must lie inside its interval: the interval method's own soundness, checked in debug builds
/// at every operation. A failure is a bug of this module, not a property of the program.
pub(super) fn checked_value(reference: f32, fl: Fl) -> Val {
    debug_assert!(
        fl.contains(f64::from(reference)),
        "reference {reference} outside its interval {:?}",
        fl.interval()
    );
    Val::F(reference, fl)
}

/// `f32` arithmetic of the written operation, in `f32` (round to nearest even, no fusion).
pub(crate) fn float_binary(op: BinOp, a: (f32, Fl), b: (f32, Fl)) -> Result<Scalar, Violation> {
    let (fa, fb) = (a.1, b.1);
    let (reference, fl) = match op {
        BinOp::Add => (a.0 + b.0, fa.plus(fb)),
        BinOp::Sub => (a.0 - b.0, fa.minus(fb)),
        BinOp::Mul => (a.0 * b.0, fa.times(fb)),
        BinOp::Div => (a.0 / b.0, fa.over(fb)?),
        other => unreachable!("float operator {}", other.symbol()),
    };
    match checked_value(reference, in_domain(fl)?) {
        Val::F(reference, fl) => Ok(Scalar::F(reference, fl)),
        _ => unreachable!("a checked float value is a float"),
    }
}

/// `f32` math method `f` of the operands `a` (the receiver first): the reference value is the `std` call, and the
/// interval holds every value the GPU may return. `min`, `max`, `abs`, `floor`, `ceil`, `trunc` and `clamp` are exact
/// and monotone (or `abs`, which folds the interval), so their interval is the image of the operand intervals;
/// `mul_add` is the product's interval plus a rounding, which holds both the fused and the unfused result (15.7.5).
/// `clamp` needs `lo <= hi` on every party, else `ComparisonOverlap` (Rust panics, WGSL leaves it unspecified).
pub(crate) fn float_math(f: MathFn, a: &[(f32, Fl)]) -> Result<Scalar, Violation> {
    let span = |k: usize| a[k].1.interval();
    let (reference, fl) = match f {
        MathFn::MulAdd => {
            let product = in_domain(a[0].1.times(a[1].1))?;
            (a[0].0.mul_add(a[1].0, a[2].0), product.plus(a[2].1))
        }
        MathFn::Abs => {
            let (lo, hi) = span(0);
            let low = if lo <= 0.0 && hi >= 0.0 {
                0.0
            } else {
                lo.abs().min(hi.abs())
            };
            (a[0].0.abs(), Fl::finished(low, lo.abs().max(hi.abs())))
        }
        MathFn::Floor => {
            let (lo, hi) = span(0);
            (a[0].0.floor(), Fl::finished(lo.floor(), hi.floor()))
        }
        MathFn::Ceil => {
            let (lo, hi) = span(0);
            (a[0].0.ceil(), Fl::finished(lo.ceil(), hi.ceil()))
        }
        MathFn::Trunc => {
            let (lo, hi) = span(0);
            (a[0].0.trunc(), Fl::finished(lo.trunc(), hi.trunc()))
        }
        MathFn::Min => {
            let ((al, ah), (bl, bh)) = (span(0), span(1));
            (a[0].0.min(a[1].0), Fl::finished(al.min(bl), ah.min(bh)))
        }
        MathFn::Max => {
            let ((al, ah), (bl, bh)) = (span(0), span(1));
            (a[0].0.max(a[1].0), Fl::finished(al.max(bl), ah.max(bh)))
        }
        MathFn::Clamp => {
            let ((xl, xh), (ll, lh), (hl, hh)) = (span(0), span(1), span(2));
            if lh > hl {
                return Err(Violation::ComparisonOverlap);
            }
            (
                a[0].0.clamp(a[1].0, a[2].0),
                Fl::finished(xl.max(ll).min(hl), xh.max(lh).min(hh)),
            )
        }
        MathFn::RemEuclid => unreachable!("`rem_euclid` of floats is not generated"),
    };
    match checked_value(reference, in_domain(fl)?) {
        Val::F(reference, fl) => Ok(Scalar::F(reference, fl)),
        _ => unreachable!("a checked float value is a float"),
    }
}

/// The truncated integer of the float interval `[lo, hi]`, if every value in it converts to the same one.
pub(super) fn float_to_int(to: Ty, reference: f32, fl: &Fl) -> Result<u32, Violation> {
    let (lo, hi) = fl.interval();
    if lo.trunc() != hi.trunc() {
        return Err(Violation::Unboundable);
    }
    // The target range with a margin of one on each end (R1-4).
    let (min, max) = match to {
        Ty::I32 => (-(i64::from(i32::MAX)), i64::from(i32::MAX) - 1),
        Ty::U32 => (0, i64::from(u32::MAX) - 1),
        other => unreachable!("cast to {}", other.name()),
    };
    if lo.trunc() < min as f64 || hi.trunc() > max as f64 {
        return Err(Violation::FloatDomain);
    }
    // `reference` lies in the interval, so it truncates to the same integer.
    Ok(f64::from(reference).trunc() as i64 as u32)
}
