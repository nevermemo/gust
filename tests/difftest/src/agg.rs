//! Vector and matrix values of the interpreter (slice m6.d): the lane-wise operators, `dot`, `m * v`, `m * m`,
//! swizzles and `transpose`, written over arrays of scalars.
//!
//! A vector is its components in order; a matrix is its `n * n` components column-major (`cols[j][i]` at `j * n + i`,
//! the layout of `gust::math::Mat4 { cols: [Vec4; 4] }`). Every operation is a composition of the scalar operations of
//! the interpreter, in the order the CPU twin performs them, so the integer domain (D38) is tracked per component and
//! the float intervals are those of `float.rs` per component:
//!
//! - a component-wise `+ - * /` is the scalar operation on each pair of components;
//! - `dot` of N components is `p0 + p1 + ... + p(N-1)` with `pk = a_k * b_k`, summed left to right
//!   (`gust::math::Vec4::dot`);
//! - `(m * v)[i]` is `c0[i] * v[0] + c1[i] * v[1] + ...` left to right (`Mat4::mul_vec4`), and `(a * b).cols[j]` is
//!   `a * b.cols[j]` (`Mat4::mul_mat4`).
//!
//! The float bound of a sum of N products is therefore not a constant of this module: it is whatever the interval
//! arithmetic of `float.rs` gives for N products and an (N - 1)-term chain (SEMANTICS section 4, "Vectors and
//! matrices"), and the GPU may fuse and regroup within it (WGSL 15.7.5), which the chain rule already covers.

use crate::ast::{BinOp, Ty};
use crate::float::Fl;
use crate::interp::{Violation, binary, float_binary, unary};

/// One component: an integer pattern, or an `f32` with its reference value and its interval.
#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    I(u32),
    F(f32, Fl),
}

impl Scalar {
    pub fn int(self) -> u32 {
        match self {
            Scalar::I(bits) => bits,
            Scalar::F(..) => unreachable!("a float where the program is typed to hold an integer"),
        }
    }

    pub fn float(self) -> (f32, Fl) {
        match self {
            Scalar::F(reference, fl) => (reference, fl),
            Scalar::I(_) => unreachable!("an integer where the program is typed to hold a float"),
        }
    }
}

/// The most components a value holds: a `Mat4`.
pub const MAX_LANES: usize = 16;

/// A vector or a matrix: `n` components in `c[..n]`.
#[derive(Clone, Copy, Debug)]
pub struct Agg {
    pub n: usize,
    pub c: [Scalar; MAX_LANES],
}

impl Agg {
    pub fn new(components: &[Scalar]) -> Agg {
        assert!(
            (1..=MAX_LANES).contains(&components.len()),
            "{} components",
            components.len()
        );
        let mut c = [Scalar::I(0); MAX_LANES];
        c[..components.len()].copy_from_slice(components);
        Agg {
            n: components.len(),
            c,
        }
    }

    pub fn lanes(&self) -> &[Scalar] {
        &self.c[..self.n]
    }
}

/// `a op b` on two components of scalar type `lane` (`u32`, `i32` or `f32`).
pub fn scalar_binary(op: BinOp, lane: Ty, a: Scalar, b: Scalar) -> Result<Scalar, Violation> {
    if lane == Ty::F32 {
        float_binary(op, a.float(), b.float())
    } else {
        binary(op, lane, a.int(), b.int()).map(Scalar::I)
    }
}

/// `a op b` component by component. An operand of one component is a scalar, used for every component of the other
/// (`v * s`, `s * v`, `m * s`); otherwise both have the same length. Components are done in order, so the first
/// violation is the lowest component's.
pub fn lanewise(op: BinOp, lane: Ty, a: &[Scalar], b: &[Scalar]) -> Result<Agg, Violation> {
    let n = a.len().max(b.len());
    debug_assert!(a.len() == n || a.len() == 1);
    debug_assert!(b.len() == n || b.len() == 1);
    let mut out = [Scalar::I(0); MAX_LANES];
    for (k, slot) in out.iter_mut().enumerate().take(n) {
        let pick = |side: &[Scalar]| if side.len() == 1 { side[0] } else { side[k] };
        *slot = scalar_binary(op, lane, pick(a), pick(b))?;
    }
    Ok(Agg::new(&out[..n]))
}

/// `-a` component by component: an exact sign flip for floats, a checked negation for `i32`.
pub fn negate(lane: Ty, a: &[Scalar]) -> Result<Agg, Violation> {
    let mut out = Vec::with_capacity(a.len());
    for &component in a {
        out.push(if lane == Ty::F32 {
            let (reference, fl) = component.float();
            Scalar::F(-reference, fl.negated())
        } else {
            Scalar::I(unary(crate::ast::UnOp::Neg, lane, component.int())?)
        });
    }
    Ok(Agg::new(&out))
}

/// The sum of the products `a_k * b_k`, left to right.
pub fn dot(lane: Ty, a: &[Scalar], b: &[Scalar]) -> Result<Scalar, Violation> {
    let mut sum = scalar_binary(BinOp::Mul, lane, a[0], b[0])?;
    for k in 1..a.len() {
        let product = scalar_binary(BinOp::Mul, lane, a[k], b[k])?;
        sum = scalar_binary(BinOp::Add, lane, sum, product)?;
    }
    Ok(sum)
}

/// `m * v` for the `n x n` matrix `m` (column-major) and the `n` components of `v`: component `i` is
/// `m[0][i] * v[0] + m[1][i] * v[1] + ...`.
pub fn mat_vec(n: usize, m: &[Scalar], v: &[Scalar]) -> Result<Vec<Scalar>, Violation> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = scalar_binary(BinOp::Mul, Ty::F32, m[i], v[0])?;
        for (k, &vk) in v.iter().enumerate().skip(1) {
            let product = scalar_binary(BinOp::Mul, Ty::F32, m[k * n + i], vk)?;
            sum = scalar_binary(BinOp::Add, Ty::F32, sum, product)?;
        }
        out.push(sum);
    }
    Ok(out)
}

/// `a * b` for two `n x n` matrices: each column of `b` is multiplied by `a`.
pub fn mat_mat(n: usize, a: &[Scalar], b: &[Scalar]) -> Result<Vec<Scalar>, Violation> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        out.extend(mat_vec(n, a, &b[j * n..(j + 1) * n])?);
    }
    Ok(out)
}

/// `a op b` for two operands of types `lt` and `rt` with the result type `result`: a matrix times a vector or a matrix
/// is the linear-algebra product, every other form is lane-wise.
pub fn binary_op(
    op: BinOp,
    result: Ty,
    (lt, rt): (Ty, Ty),
    a: &[Scalar],
    b: &[Scalar],
) -> Result<Agg, Violation> {
    match (lt, rt) {
        (Ty::Mat(n), Ty::Mat(_)) if op == BinOp::Mul => {
            Ok(Agg::new(&mat_mat(usize::from(n), a, b)?))
        }
        (Ty::Mat(n), Ty::Vec(..)) => Ok(Agg::new(&mat_vec(usize::from(n), a, b)?)),
        _ => lanewise(op, result.scalar(), a, b),
    }
}

/// The transpose of an `n x n` matrix.
pub fn transpose(n: usize, m: &[Scalar]) -> Vec<Scalar> {
    (0..n * n).map(|k| m[(k % n) * n + k / n]).collect()
}

/// `Mat::from_rows`: row `i` supplies component `i` of every column.
pub fn from_rows(n: usize, rows: &[Scalar]) -> Vec<Scalar> {
    transpose(n, rows)
}
