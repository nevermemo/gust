//! The interpreter: the independent host reference of the generator (D26). It evaluates the AST directly with
//! checked integer operations and never sees the emitted text.
//!
//! Besides the value it tracks the domain of D38 (docs/SEMANTICS.md section 3, items 1 to 3 and 6): a program run
//! that leaves the domain at any element, in any evaluated sub-expression, returns the first `Violation` and the
//! generator discards the program. Evaluation order and short-circuiting follow Rust, so a sub-expression Rust does
//! not evaluate (the right side of `&&` after `false`) cannot violate the domain.
//!
//! `f32` values (DIFFTEST_FORMAT 2) carry two things: the reference value, computed in `f32` in the written order
//! with round to nearest even (what the CPU twin computes), and an interval from `float.rs` that contains every value
//! a conforming WGSL implementation may hold (R1-4). A float construct whose result the interval cannot pin down is a
//! `Violation` too, and the program is discarded, never asserted.

use std::collections::{BTreeMap, HashMap};

use crate::agg::{self, Agg, Scalar};
use crate::ast::{
    ASSIGN_COMP, Arm, BinOp, Elem, Expr, MATCH_ARM_RETURN_RUN, Pat, Place, Program, Recv, Root,
    SHAPE_EXPR_CALL, SHAPE_STMT_CALL, STRUCT_LIT, Stmt, TUPLE_READ, Ty, UnOp, array_call_names,
    array_expr_names, array_set_names, array_store_names, bin_shape, call_arg_names, call_name,
    call_shape, destruct_names, enum_as_name, enum_call_shapes, match_names, match_shape, set_name,
    set_shape, tuple_field_names, tuple_lit_names, tuple_set_names, value_match_names,
};
use crate::float::{self, Fl};

/// A way to leave the defined domain: the integer conditions of D38 and the float conditions of R1-4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Violation {
    /// `+ - *` or unary `-` whose result does not fit the type.
    Overflow,
    /// `/` or `%` with a zero divisor.
    DivisionByZero,
    /// `i32::MIN / -1` or `i32::MIN % -1`.
    MinOverMinusOne,
    /// A shift amount of 32 or more.
    ShiftAmount,
    /// A buffer index not below `len()`.
    Index,
    /// A float interval outside `+-[2^-100, 2^100]` (or holding zero without being `[0, 0]`), a float divisor
    /// interval outside that range, or a float-to-integer cast whose interval leaves the target range.
    FloatDomain,
    /// A float comparison whose operand intervals are not disjoint.
    ComparisonOverlap,
    /// An output interval wider than `2^-10` relative, or a float-to-integer cast whose interval spans two integers.
    Unboundable,
    /// An integer of magnitude above `2^24` converted to `f32` (the two-neighbour set of 15.7.6 is v2 material).
    FloatConstruct,
}

impl Violation {
    pub const ALL: [Violation; 9] = [
        Violation::Overflow,
        Violation::DivisionByZero,
        Violation::MinOverMinusOne,
        Violation::ShiftAmount,
        Violation::Index,
        Violation::FloatDomain,
        Violation::ComparisonOverlap,
        Violation::Unboundable,
        Violation::FloatConstruct,
    ];

    /// The name used in discard counts and in `seeds.txt` (no spaces: a line is split on spaces).
    pub fn label(self) -> &'static str {
        match self {
            Violation::Overflow => "overflow",
            Violation::DivisionByZero => "division-by-zero",
            Violation::MinOverMinusOne => "min-over-minus-one",
            Violation::ShiftAmount => "shift-amount",
            Violation::Index => "index",
            Violation::FloatDomain => "float-domain",
            Violation::ComparisonOverlap => "comparison-overlap",
            Violation::Unboundable => "unboundable",
            Violation::FloatConstruct => "float-construct-not-in-v1",
        }
    }
}

/// A program run that left the domain: the first element that did, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Failure {
    pub element: u32,
    pub violation: Violation,
}

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

/// A value of the machine: an integer or `bool` pattern, an `f32` with its reference value and its interval, or a
/// vector or matrix of such components (`agg.rs`).
// A `Val` is copied by value on every read of a variable: boxing the aggregate would allocate per vector operation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
enum Val {
    I(u32),
    F(f32, Fl),
    A(Agg),
}

impl Val {
    fn int(self) -> u32 {
        match self {
            Val::I(bits) => bits,
            Val::F(..) | Val::A(_) => {
                unreachable!("a float or aggregate where the program is typed to hold an integer")
            }
        }
    }

    fn float(self) -> (f32, Fl) {
        match self {
            Val::F(reference, fl) => (reference, fl),
            Val::I(_) | Val::A(_) => {
                unreachable!("an integer or aggregate where the program is typed to hold a float")
            }
        }
    }

    fn agg(self) -> Agg {
        match self {
            Val::A(agg) => agg,
            _ => unreachable!("a scalar where the program is typed to hold a vector or matrix"),
        }
    }

    /// The components of a value as a slice-able array: a scalar is one component.
    fn scalar(self) -> Scalar {
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
fn in_domain(fl: Fl) -> Result<Fl, Violation> {
    let (lo, hi) = fl.interval();
    if float::in_range(lo, hi) {
        Ok(fl)
    } else {
        Err(Violation::FloatDomain)
    }
}

/// The reference value must lie inside its interval: the interval method's own soundness, checked in debug builds
/// at every operation. A failure is a bug of this module, not a property of the program.
fn checked_value(reference: f32, fl: Fl) -> Val {
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

/// The truncated integer of the float interval `[lo, hi]`, if every value in it converts to the same one.
fn float_to_int(to: Ty, reference: f32, fl: &Fl) -> Result<u32, Violation> {
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

/// A place resolved: its index evaluated, so that reading and writing it again does not evaluate the index again.
#[derive(Clone, Copy, Debug)]
struct Loc {
    root: Root,
    /// The offset of the part in the root's flat components.
    offset: usize,
    ty: Ty,
    /// The place is the whole root.
    whole: bool,
}

struct Machine<'a> {
    program: &'a Program,
    element: u32,
    vars: Vec<Val>,
    /// The receivers of the calls in progress (a struct as its flat components), innermost last: what `self` is.
    self_stack: Vec<Val>,
    /// Executed-operator counts, kept only by `run_covered`.
    coverage: Option<Box<Counting>>,
    /// The first value of every expression node (by address), kept only by `observed_values`.
    observed: Option<HashMap<usize, u32>>,
    /// Set by `return` in a helper: the value the function ends with; every enclosing block stops while it is set.
    returned: Option<Val>,
    /// `for` bodies and `match` arms the running function is inside (reset per call), for the counted names.
    for_depth: u32,
    arm_depth: u32,
    /// Test hook for the mutation checks (slice m9.d1b); `None` is the Rust semantics.
    mutant: Option<Mutant>,
}

/// The mutant that the test run asks for through `DIFFTEST_MUTANT` (`default-first-arm` or `scrutinee-twice`). Only the
/// hand-vector tests apply it (`enum_tests::value_of`); `run` stays the reference, so an `xfail-` run fails exactly the
/// vector that catches the mutant. Not compiled outside the tests.
#[cfg(test)]
pub(crate) fn env_mutant() -> Option<Mutant> {
    match std::env::var("DIFFTEST_MUTANT").as_deref() {
        Ok("default-first-arm") => Some(Mutant::DefaultTakesFirstArm),
        Ok("scrutinee-twice") => Some(Mutant::ScrutineeTwice),
        Ok("tuple-right-to-left") => Some(Mutant::TupleRightToLeft),
        Ok("value-scrutinee-twice") => Some(Mutant::ValueScrutineeTwice),
        Ok("group-first-only") => Some(Mutant::GroupFirstOnly),
        Ok("compound-place-first") => Some(Mutant::CompoundPlaceFirst),
        Ok("array-lit-right-to-left") => Some(Mutant::ArrayLitRightToLeft),
        Ok("repeat-per-element") => Some(Mutant::RepeatPerElement),
        Ok("index-before-value") => Some(Mutant::IndexBeforeValue),
        Ok("array-arg-by-ref") => Some(Mutant::ArrayArgByRef),
        _ => None,
    }
}

/// A deliberately wrong `match` evaluation, used only by the `xfail-` mutation tests to show the vectors catch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// In a `match` without `_`, the variants of the last arm (the Slang `default`) go to the first arm instead (a
    /// mis-applied `default` rule).
    DefaultTakesFirstArm,
    /// The scrutinee is evaluated again for every arm tried (read twice).
    ScrutineeTwice,
    /// The elements of a tuple literal run right to left (their values are still placed left to right). (m9.d1c)
    TupleRightToLeft,
    /// The scrutinee of a value `match` is evaluated again for every arm after the first. (m9.d1c)
    ValueScrutineeTwice,
    /// An integer pattern group (`1 | 2 | 3`) matches only its first literal. (m9.d1c)
    GroupFirstOnly,
    /// `place op= value` reads the place before it runs the value (the order of an operator-overloaded `AddAssign`, and
    /// of Slang for an indexed place), not after. (m9.d1c fix round 1)
    CompoundPlaceFirst,
    /// The elements of an array literal run right to left (their values are still placed left to right). (m9.d1d)
    ArrayLitRightToLeft,
    /// A repeat `[e; N]` evaluates `e` once per element instead of once. (m9.d1d)
    RepeatPerElement,
    /// `place = value` and `place op= value` locate the place (evaluate its index) before the value, and the output
    /// store evaluates its index before the result. (m9.d1d)
    IndexBeforeValue,
    /// An array argument passed by reference: what the callee leaves in the parameter is written back to the caller's
    /// variable. (m9.d1d)
    ArrayArgByRef,
}

/// What `run_covered` counts while it runs.
#[derive(Default)]
struct Counting {
    /// (kind, head, tail) -> evaluations; see `operator_name`.
    operators: BTreeMap<(u8, &'static str, &'static str), u64>,
    /// Vector and matrix operators and component assignments by their counted name (`Expr::aggregate_name`).
    named: BTreeMap<String, u64>,
    /// `if` statement (by address) -> (evaluations that took the then branch, evaluations that took the else branch).
    branches: HashMap<usize, (u64, u64)>,
    /// `&&` and `||` evaluations whose right operand was evaluated, and whose right operand was skipped.
    rhs_evaluated: u64,
    rhs_skipped: u64,
}

const KIND_UNARY: u8 = 0;
const KIND_BINARY: u8 = 1;
const KIND_CAST: u8 = 2;
const KIND_FOR: u8 = 3;

/// The name of a counted operator, the same spelling as `ast::operator_counts` (`+ u32`, `-f32`, `!bool`, `as f32`, `for`).
fn operator_name(kind: u8, head: &str, tail: &str) -> String {
    match kind {
        KIND_UNARY if head == "!" => format!("!{tail}"),
        KIND_UNARY => head.to_owned(),
        KIND_BINARY => format!("{head} {tail}"),
        KIND_CAST => format!("as {tail}"),
        _ => "for".to_owned(),
    }
}

impl Machine<'_> {
    fn count(&mut self, kind: u8, head: &'static str, tail: &'static str) {
        if let Some(counting) = &mut self.coverage {
            *counting.operators.entry((kind, head, tail)).or_default() += 1;
        }
    }

    /// Counts a vector or matrix operator by its name (only `run_covered` pays for the string).
    fn count_aggregate(&mut self, expr: &Expr) {
        if let Some(counting) = &mut self.coverage
            && let Some(name) = expr.aggregate_name()
        {
            *counting.named.entry(name).or_default() += 1;
        }
    }

    /// Evaluates `expr` and, for `observed_values`, remembers the first value each node takes (a scalar's pattern; a
    /// vector or matrix has no single pattern and is not remembered).
    fn expr(&mut self, expr: &Expr) -> Result<Val, Violation> {
        let value = self.eval(expr)?;
        if let Some(seen) = &mut self.observed {
            let bits = match value {
                Val::I(bits) => Some(bits),
                Val::F(reference, _) => Some(reference.to_bits()),
                Val::A(_) => None,
            };
            if let Some(bits) = bits {
                seen.entry(std::ptr::from_ref(expr) as usize)
                    .or_insert(bits);
            }
        }
        Ok(value)
    }

    /// The components of an operand: a scalar is one, an aggregate its own.
    fn lanes(value: Val) -> Agg {
        match value {
            Val::A(agg) => agg,
            scalar => Agg::new(&[scalar.scalar()]),
        }
    }

    /// The aggregate operators (`AggBin` and the nodes that build, read or rearrange vectors and matrices).
    fn eval_aggregate(&mut self, expr: &Expr) -> Result<Val, Violation> {
        self.count_aggregate(expr);
        match expr {
            Expr::Make { ty, rows, args } => {
                let mut parts = Vec::with_capacity(ty.lanes());
                for arg in args {
                    let value = self.expr(arg)?;
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                if *rows {
                    let Ty::Mat(n) = ty else {
                        unreachable!("from_rows of {}", ty.name())
                    };
                    parts = agg::from_rows(usize::from(*n), &parts);
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Splat { ty, e } => {
                let value = self.expr(e)?.scalar();
                Ok(Val::A(Agg::new(&vec![value; ty.lanes()])))
            }
            Expr::Comp { e, index } => {
                let agg = self.expr(e)?.agg();
                Ok(agg.c[usize::from(*index)].into())
            }
            Expr::Swizzle { e, lanes } => {
                let agg = self.expr(e)?.agg();
                let picked: Vec<Scalar> = lanes.iter().map(|&k| agg.c[usize::from(k)]).collect();
                Ok(Val::A(Agg::new(&picked)))
            }
            Expr::Dot { l, r } => {
                let lane = l.ty().scalar();
                let left = self.expr(l)?.agg();
                let right = self.expr(r)?.agg();
                Ok(agg::dot(lane, left.lanes(), right.lanes())?.into())
            }
            Expr::Transpose { e } => {
                let Ty::Mat(n) = e.ty() else {
                    unreachable!("transpose of a non-matrix")
                };
                let agg = self.expr(e)?.agg();
                Ok(Val::A(Agg::new(&agg::transpose(
                    usize::from(n),
                    agg.lanes(),
                ))))
            }
            Expr::AggNeg { e } => {
                let lane = e.ty().scalar();
                let agg = self.expr(e)?.agg();
                Ok(Val::A(agg::negate(lane, agg.lanes())?))
            }
            Expr::AggBin { op, ty, l, r } => {
                let types = (l.ty(), r.ty());
                let left = Self::lanes(self.expr(l)?);
                let right = Self::lanes(self.expr(r)?);
                Ok(Val::A(agg::binary_op(
                    *op,
                    *ty,
                    types,
                    left.lanes(),
                    right.lanes(),
                )?))
            }
            other => unreachable!("not an aggregate operator: {other:?}"),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Val, Violation> {
        match expr {
            Expr::Lit(Ty::F32, bits) => {
                let value = f32::from_bits(*bits);
                Ok(Val::F(value, in_domain(Fl::exact(value))?))
            }
            Expr::Lit(_, bits) => Ok(Val::I(*bits)),
            Expr::Var(var, _) => Ok(self.vars[*var as usize]),
            Expr::Gid => Ok(Val::I(self.element)),
            Expr::Len(buf) => Ok(Val::I(self.program.inputs[*buf].data.len() as u32)),
            Expr::Input { buf, index, ty } => {
                let index = self.expr(index)?.int();
                let bits = self.program.inputs[*buf]
                    .data
                    .get(index as usize)
                    .copied()
                    .ok_or(Violation::Index)?;
                if *ty == Ty::F32 {
                    let value = f32::from_bits(bits);
                    Ok(Val::F(value, in_domain(Fl::exact(value))?))
                } else {
                    Ok(Val::I(bits))
                }
            }
            Expr::Un {
                op: UnOp::Neg,
                ty: Ty::F32,
                e,
            } => {
                self.count(KIND_UNARY, "-f32", "");
                let (reference, fl) = self.expr(e)?.float();
                Ok(checked_value(-reference, fl.negated()))
            }
            Expr::Un { op, ty, e } => {
                self.count(
                    KIND_UNARY,
                    match op {
                        UnOp::Neg => "-",
                        UnOp::Not => "!",
                    },
                    ty.name(),
                );
                let value = self.expr(e)?.int();
                Ok(Val::I(unary(*op, *ty, value)?))
            }
            Expr::Bin {
                op: op @ (BinOp::And | BinOp::Or),
                l,
                r,
                ..
            } => {
                self.count(KIND_BINARY, op.symbol(), "bool");
                let left = self.expr(l)?.int();
                if (*op == BinOp::And) == (left == 0) {
                    if let Some(counting) = &mut self.coverage {
                        counting.rhs_skipped += 1;
                    }
                    return Ok(Val::I(left));
                }
                if let Some(counting) = &mut self.coverage {
                    counting.rhs_evaluated += 1;
                }
                if self.coverage.is_some()
                    && let Some(shape) = bin_shape(&self.program.methods, *op, l, r)
                {
                    self.count_name(shape);
                }
                self.expr(r)
            }
            Expr::Bin {
                op,
                ty: Ty::F32,
                l,
                r,
            } => {
                self.count(KIND_BINARY, op.symbol(), "f32");
                let left = self.expr(l)?.float();
                let right = self.expr(r)?.float();
                if op.is_comparison() {
                    let truth = float::compare(op.symbol(), &left.1, &right.1)?;
                    debug_assert_eq!(
                        truth,
                        match op {
                            BinOp::Eq => left.0 == right.0,
                            BinOp::Ne => left.0 != right.0,
                            BinOp::Lt => left.0 < right.0,
                            BinOp::Le => left.0 <= right.0,
                            BinOp::Gt => left.0 > right.0,
                            _ => left.0 >= right.0,
                        },
                        "the reference comparison disagrees with the disjoint intervals"
                    );
                    return Ok(Val::I(u32::from(truth)));
                }
                float_binary(*op, left, right).map(Val::from)
            }
            Expr::Make { .. }
            | Expr::Splat { .. }
            | Expr::Comp { .. }
            | Expr::Swizzle { .. }
            | Expr::Dot { .. }
            | Expr::Transpose { .. }
            | Expr::AggBin { .. }
            | Expr::AggNeg { .. } => self.eval_aggregate(expr),
            Expr::Read { place, .. } => {
                if place.root_ty.is_tuple() {
                    self.count_name(TUPLE_READ);
                }
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let loc = self.locate(place)?;
                Ok(self.loc_get(&loc))
            }
            Expr::ArrayLit { elems, .. } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                // The elements run left to right, each completely before the next (Rust Reference, "Array
                // expressions" and "Evaluation order of operands").
                let mut values = vec![None; elems.len()];
                let order: Vec<usize> = if self.mutant == Some(Mutant::ArrayLitRightToLeft) {
                    (0..elems.len()).rev().collect()
                } else {
                    (0..elems.len()).collect()
                };
                for k in order {
                    values[k] = Some(self.expr(&elems[k])?);
                }
                let mut parts = Vec::new();
                for value in values.into_iter().flatten() {
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Repeat { ty, e } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let Ty::Array(a) = ty else {
                    unreachable!("Program::check requires an array type")
                };
                // The element is evaluated once and copied (Reference, "Array expressions": `[expr; N]`).
                let mut parts = Vec::new();
                let mut value = self.expr(e)?;
                for k in 0..a.n {
                    if self.mutant == Some(Mutant::RepeatPerElement) && k > 0 {
                        value = self.expr(e)?;
                    }
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::ArrLen { place } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let Ty::Array(a) = place.root_ty else {
                    unreachable!("Program::check requires an array under `.len()`")
                };
                Ok(Val::I(u32::from(a.n)))
            }
            Expr::Tuple { elems, .. } => {
                if self.coverage.is_some() {
                    for name in tuple_lit_names(&self.program.methods, elems) {
                        self.count_name(name);
                    }
                }
                // The elements run left to right, each completely before the next (Rust Reference, "Tuple
                // expressions" and "Evaluation order of operands").
                let mut values = vec![None; elems.len()];
                let order: Vec<usize> = if self.mutant == Some(Mutant::TupleRightToLeft) {
                    (0..elems.len()).rev().collect()
                } else {
                    (0..elems.len()).collect()
                };
                for k in order {
                    values[k] = Some(self.expr(&elems[k])?);
                }
                let mut parts = Vec::new();
                for value in values.into_iter().flatten() {
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::TupleField { e, index } => {
                if self.coverage.is_some() {
                    for name in tuple_field_names(&self.program.methods, e) {
                        self.count_name(name);
                    }
                }
                let Ty::Tuple(t) = e.ty() else {
                    unreachable!("Program::check requires a tuple under `.k`")
                };
                let agg = self.expr(e)?.agg();
                Ok(self.tuple_element(t, *index, &agg))
            }
            Expr::Variant { e, v } => Ok(Val::I(
                self.program.methods.enums[usize::from(*e)].variants[usize::from(*v)].0,
            )),
            Expr::EnumAs { to, e } => {
                self.count_name(&enum_as_name(*to));
                // The discriminant bits are already the repr value (an `i32` repr holds its two's complement).
                Ok(Val::I(self.expr(e)?.int()))
            }
            Expr::Call { f, recv, args, .. } => {
                if self.program.methods.fns[*f].recv == Recv::Mut {
                    self.count_name(SHAPE_EXPR_CALL);
                }
                self.count_enum_args(args);
                let value = self.call(*f, recv, args)?;
                Ok(value.expect("a value call has a tail"))
            }
            Expr::StructLit { fields, order, .. } => {
                self.count_name(STRUCT_LIT);
                // The fields in the order written, as Rust evaluates them; a nested struct contributes its
                // components. The components are laid out in declaration order.
                let mut by_decl = vec![Vec::new(); fields.len()];
                for (field, &decl) in fields.iter().zip(order) {
                    by_decl[usize::from(decl)] = match self.expr(field)? {
                        Val::A(agg) => agg.lanes().to_vec(),
                        scalar => vec![scalar.scalar()],
                    };
                }
                let parts: Vec<_> = by_decl.into_iter().flatten().collect();
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Bin { op, ty, l, r } => {
                self.count(KIND_BINARY, op.symbol(), ty.name());
                if self.coverage.is_some()
                    && let Some(shape) = bin_shape(&self.program.methods, *op, l, r)
                {
                    self.count_name(shape);
                }
                let left = self.expr(l)?.int();
                let right = self.expr(r)?.int();
                Ok(Val::I(binary(*op, *ty, left, right)?))
            }
            Expr::Cast { to, e } => {
                self.count(KIND_CAST, "as", to.name());
                let from = e.ty();
                let value = self.expr(e)?;
                match (from, *to) {
                    (Ty::F32, Ty::I32 | Ty::U32) => {
                        let (reference, fl) = value.float();
                        Ok(Val::I(float_to_int(*to, reference, &fl)?))
                    }
                    (Ty::I32 | Ty::U32, Ty::F32) => {
                        let bits = value.int();
                        let magnitude = if from == Ty::I32 {
                            u64::from((bits as i32).unsigned_abs())
                        } else {
                            u64::from(bits)
                        };
                        if magnitude > float::EXACT_INT_LIMIT {
                            return Err(Violation::FloatConstruct);
                        }
                        let number = if from == Ty::I32 {
                            (bits as i32) as f32
                        } else {
                            bits as f32
                        };
                        Ok(Val::F(number, Fl::exact(number)))
                    }
                    _ => Ok(value),
                }
            }
        }
    }

    /// Element `k` of a tuple value `agg` of type `t`: a scalar, or a struct as its flat components.
    fn tuple_element(&self, t: crate::ast::TupleTy, k: u8, agg: &Agg) -> Val {
        let methods = &self.program.methods;
        let offset = methods.tuple_offset(t, k);
        match t.elems()[usize::from(k)] {
            Elem::Struct(j) => Val::A(Agg::new(&agg.c[offset..offset + methods.flat_len(j)])),
            _ => agg.c[offset].into(),
        }
    }

    /// Counts the enum-argument shapes of a call.
    fn count_enum_args(&mut self, args: &[Expr]) {
        if self.coverage.is_some() {
            for shape in enum_call_shapes(&self.program.methods, args) {
                self.count_name(shape);
            }
        }
    }

    /// Counts a construct by its name (calls, `set`, struct literals); only `run_covered` pays for it.
    fn count_name(&mut self, name: &str) {
        if let Some(counting) = &mut self.coverage {
            *counting.named.entry(name.to_owned()).or_default() += 1;
        }
    }

    /// Resolves a place: evaluates its index, once, and checks it against the array's length (a `Violation::Index`
    /// otherwise: Rust would panic), and returns where the value lives. A place without an index evaluates nothing.
    fn locate(&mut self, place: &Place) -> Result<Loc, Violation> {
        let program = self.program;
        let methods = &program.methods;
        let mut base = 0;
        let mut ty = place.root_ty;
        if let Some(index) = place.index.as_deref() {
            let Ty::Array(a) = place.root_ty else {
                unreachable!("Program::check requires an array under an index")
            };
            let bits = self.expr(index)?.int();
            let in_range = if index.ty() == Ty::I32 {
                (bits as i32) >= 0 && (bits as usize) < usize::from(a.n)
            } else {
                (bits as usize) < usize::from(a.n)
            };
            if !in_range {
                return Err(Violation::Index);
            }
            base = bits as usize * methods.ty_len(a.e.ty());
            ty = a.e.ty();
        }
        let (ty, offset) = methods.walk(ty, &place.path);
        Ok(Loc {
            root: place.root,
            offset: base + offset,
            ty,
            whole: place.index.is_none() && place.path.is_empty(),
        })
    }

    fn root_val(&self, root: Root) -> Val {
        match root {
            Root::Var(var) => self.vars[var as usize],
            Root::SelfV => *self.self_stack.last().expect("`self` outside a call"),
        }
    }

    /// The value at a resolved place: a scalar, or a struct, a vector, a matrix or an array as its flat components.
    fn loc_get(&self, loc: &Loc) -> Val {
        let root = self.root_val(loc.root);
        if loc.whole {
            return root;
        }
        let Val::A(agg) = root else {
            unreachable!("a part of a scalar: {loc:?} holds {root:?}")
        };
        match loc.ty {
            Ty::U32 | Ty::I32 | Ty::Bool | Ty::F32 | Ty::Enum(_) => agg.c[loc.offset].into(),
            other => {
                let len = self.program.methods.ty_len(other);
                Val::A(Agg::new(&agg.c[loc.offset..loc.offset + len]))
            }
        }
    }

    /// Stores `value` at a resolved place (the root is rewritten whole, as Rust writes the part in place).
    fn loc_set(&mut self, loc: &Loc, value: Val) {
        if loc.whole {
            match loc.root {
                Root::Var(var) => self.vars[var as usize] = value,
                Root::SelfV => *self.self_stack.last_mut().expect("`self` outside a call") = value,
            }
            return;
        }
        let root = match loc.root {
            Root::Var(var) => &mut self.vars[var as usize],
            Root::SelfV => self.self_stack.last_mut().expect("`self` outside a call"),
        };
        let Val::A(agg) = root else {
            unreachable!("a part of a scalar")
        };
        match value {
            Val::A(part) => agg.c[loc.offset..loc.offset + part.n].copy_from_slice(part.lanes()),
            scalar => agg.c[loc.offset] = scalar.scalar(),
        }
    }

    /// A call, in Rust's order: the receiver is read first (a by-value `self` is copied now; for `&self` and `&mut
    /// self` the arguments cannot change it, `ast::Check` refuses a `&mut self` call on its root inside them), then
    /// the arguments left to right, then the body. `&mut self` writes the final receiver back to its place.
    fn call(
        &mut self,
        f: usize,
        recv: &Option<Place>,
        args: &[Expr],
    ) -> Result<Option<Val>, Violation> {
        let program = self.program;
        let def = &program.methods.fns[f];
        self.count_name(call_name(def.recv, def.owner));
        if self.coverage.is_some()
            && let Some(shape) = call_shape(&program.methods, f, recv, args)
        {
            self.count_name(shape);
        }
        if self.coverage.is_some() {
            for name in call_arg_names(def.recv != Recv::None, args) {
                self.count_name(name);
            }
        }
        if self.coverage.is_some() {
            for name in array_call_names(recv)
                .into_iter()
                .chain(crate::ast::array_arg_names(args))
            {
                self.count_name(name);
            }
        }
        let loc = recv.as_ref().map(|place| self.locate(place)).transpose()?;
        let receiver = loc.as_ref().map(|loc| self.loc_get(loc));
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            values.push(self.expr(arg)?);
        }
        // The arguments of a borrowing call (`&self`, `&mut self`) must not change the receiver, or the copy-in and
        // copy-out model of this interpreter would differ from Rust's references (the generator and `ast::Check`
        // keep it so).
        if def.recv != Recv::Value
            && let (Some(loc), Some(before)) = (loc.as_ref(), receiver.as_ref())
        {
            debug_assert_eq!(
                format!("{before:?}"),
                format!("{:?}", self.loc_get(loc)),
                "the arguments of a borrowing call changed its receiver"
            );
        }
        for ((var, _), value) in def.params.iter().zip(values) {
            self.vars[*var as usize] = value;
        }
        if let Some(receiver) = receiver {
            self.self_stack.push(receiver);
        }
        let depths = (self.for_depth, self.arm_depth);
        (self.for_depth, self.arm_depth) = (0, 0);
        let ran = self.block(&def.body);
        (self.for_depth, self.arm_depth) = depths;
        ran?;
        let value = match self.returned.take() {
            Some(value) => Some(value),
            None => def.tail.as_ref().map(|tail| self.expr(tail)).transpose()?,
        };
        if self.mutant == Some(Mutant::ArrayArgByRef) {
            // The wrong model: an array argument that is a variable is the callee's parameter itself.
            for (arg, (var, _)) in args.iter().zip(&def.params) {
                if let Expr::Var(outer, Ty::Array(_)) = arg {
                    self.vars[*outer as usize] = self.vars[*var as usize];
                }
            }
        }
        if let Some(loc) = loc.as_ref() {
            let last = self.self_stack.pop().expect("a receiver was pushed");
            if def.recv == Recv::Mut {
                self.loc_set(loc, last);
            }
        }
        Ok(value)
    }

    /// A value `match` (`tail`: the value of an arm of another one): the scrutinee once, the first arm that names it,
    /// then that arm's block and its value. `None` if the arm returned from the function (`self.returned` is set).
    fn value_match(
        &mut self,
        scrut: &Expr,
        arms: &[Arm],
        tail: bool,
    ) -> Result<Option<Val>, Violation> {
        if self.coverage.is_some() {
            for name in value_match_names(&self.program.methods, scrut, arms, tail) {
                self.count_name(name);
            }
        }
        let bits = self.expr(scrut)?.int();
        let variant = match scrut.ty() {
            Ty::Enum(e) => self.program.methods.enums[usize::from(e)]
                .variants
                .iter()
                .position(|v| v.0 == bits)
                .expect("an enum value is a discriminant") as u8,
            _ => 0,
        };
        let mut taken = None;
        for (k, arm) in arms.iter().enumerate() {
            if self.mutant == Some(Mutant::ValueScrutineeTwice) && k > 0 {
                self.expr(scrut)?;
            }
            let hit = match &arm.pat {
                Pat::Variants(vs) => vs.contains(&variant),
                Pat::Lits(lits) if self.mutant == Some(Mutant::GroupFirstOnly) => lits[0].0 == bits,
                Pat::Lits(lits) => lits.iter().any(|&(lit, _)| lit == bits),
                Pat::Wild => true,
            };
            if hit {
                taken = Some(arm);
                break;
            }
        }
        let arm = taken.expect("a value match is exhaustive");
        self.arm_depth += 1;
        let done = self.arm_value(&arm.body);
        self.arm_depth -= 1;
        done
    }

    /// The block of a value arm: its statements, then its value (`None` after a `return`).
    fn arm_value(&mut self, body: &[Stmt]) -> Result<Option<Val>, Violation> {
        let (last, rest) = body.split_last().expect("a value arm is not empty");
        self.block(rest)?;
        if self.returned.is_some() {
            return Ok(None);
        }
        match last {
            Stmt::Yield { value } => Ok(Some(self.expr(value)?)),
            Stmt::TailMatch { scrut, arms } => self.value_match(scrut, arms, true),
            other => {
                self.block(std::slice::from_ref(other))?;
                Ok(None)
            }
        }
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), Violation> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, init, .. } => {
                    if self.coverage.is_some() {
                        for name in crate::ast::array_value_names(init) {
                            self.count_name(name);
                        }
                    }
                    let value = self.expr(init)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::Assign { var, value } => {
                    if self.coverage.is_some() {
                        for name in crate::ast::array_value_names(value) {
                            self.count_name(name);
                        }
                    }
                    let value = self.expr(value)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::LetTuple {
                    vars,
                    ty,
                    init,
                    annotate,
                } => {
                    if self.coverage.is_some() {
                        for name in destruct_names(&self.program.methods, *annotate, init) {
                            self.count_name(name);
                        }
                    }
                    let Ty::Tuple(t) = ty else {
                        unreachable!("Program::check requires a tuple type")
                    };
                    let agg = self.expr(init)?.agg();
                    for (k, &(var, _)) in vars.iter().enumerate() {
                        self.vars[var as usize] = self.tuple_element(*t, k as u8, &agg);
                    }
                }
                Stmt::LetMatch {
                    var, scrut, arms, ..
                } => {
                    if let Some(value) = self.value_match(scrut, arms, false)? {
                        self.vars[*var as usize] = value;
                    }
                }
                Stmt::Yield { .. } | Stmt::TailMatch { .. } => {
                    unreachable!("a `Yield` or `TailMatch` only ends a value arm (Program::check)")
                }
                Stmt::AssignComp { var, index, value } => {
                    if let Some(counting) = &mut self.coverage {
                        *counting.named.entry(ASSIGN_COMP.to_string()).or_default() += 1;
                    }
                    let value = self.expr(value)?.scalar();
                    let Val::A(mut agg) = self.vars[*var as usize] else {
                        unreachable!("a component assignment to a scalar")
                    };
                    agg.c[usize::from(*index)] = value;
                    self.vars[*var as usize] = Val::A(agg);
                }
                Stmt::Set {
                    place,
                    ty,
                    op,
                    value,
                } => {
                    self.count_name(&set_name(*op));
                    if self.coverage.is_some()
                        && let Some(shape) = set_shape(&self.program.methods, place, *op, value)
                    {
                        self.count_name(&shape);
                    }
                    if self.coverage.is_some() {
                        for name in tuple_set_names(&self.program.methods, place, *op, value) {
                            self.count_name(name);
                        }
                    }
                    // Rust evaluates the assigned value first. For `place op= value` on primitives the right operand
                    // also comes before the place is read (Reference, "Compound assignment expressions"), so the
                    // place holds whatever the value's calls left in it.
                    // That is true of primitive operands only: for a `Vec*`/`Mat*` `AddAssign` the place comes first. The
                    // generator writes a compound set on `u32`/`i32` only; a slice that adds other operand types must
                    // give this arm the type-dependent order.
                    debug_assert!(op.is_none() || ty.is_int(), "a compound set on a primitive");
                    if self.coverage.is_some() {
                        for name in array_set_names(&self.program.methods, place, *op, value) {
                            self.count_name(name);
                        }
                    }
                    // The place is located (its index evaluated, once) after the value, as Rust does; the mutants
                    // `index-before-value` and `compound-place-first` locate it first.
                    let place_first =
                        self.mutant == Some(Mutant::CompoundPlaceFirst) && op.is_some();
                    let mut loc = None;
                    if place_first || self.mutant == Some(Mutant::IndexBeforeValue) {
                        loc = Some(self.locate(place)?);
                    }
                    let early = match (place_first, loc.as_ref()) {
                        (true, Some(loc)) => Some(self.loc_get(loc).int()),
                        _ => None,
                    };
                    let value = self.expr(value)?;
                    let loc = match loc {
                        Some(loc) => loc,
                        None => self.locate(place)?,
                    };
                    let stored = match op {
                        None => value,
                        Some(op) => {
                            let current = early.unwrap_or_else(|| self.loc_get(&loc).int());
                            Val::I(binary(*op, *ty, current, value.int())?)
                        }
                    };
                    self.loc_set(&loc, stored);
                }
                Stmt::Call { f, recv, args, .. } => {
                    if self.program.methods.fns[*f].recv == Recv::Mut {
                        self.count_name(SHAPE_STMT_CALL);
                    }
                    self.count_enum_args(args);
                    self.call(*f, recv, args)?;
                }
                Stmt::Return { value } => {
                    self.count_name("return");
                    if self.arm_depth > 0 {
                        self.count_name(MATCH_ARM_RETURN_RUN);
                    }
                    let value = self.expr(value)?;
                    self.returned = Some(value);
                }
                Stmt::Match { scrut, arms } => {
                    if self.coverage.is_some() {
                        let (in_for, nested) = (self.for_depth > 0, self.arm_depth > 0);
                        for name in match_names(arms, in_for, nested) {
                            self.count_name(name);
                        }
                        if let Some(shape) = match_shape(&self.program.methods, scrut, arms) {
                            self.count_name(shape);
                        }
                    }
                    // The scrutinee is evaluated once, then the first arm naming the variant (`_` names the rest).
                    let bits = self.expr(scrut)?.int();
                    let Ty::Enum(e) = scrut.ty() else {
                        unreachable!("Program::check requires an enum scrutinee")
                    };
                    let variant = self.program.methods.enums[usize::from(e)]
                        .variants
                        .iter()
                        .position(|v| v.0 == bits)
                        .expect("an enum value is a discriminant")
                        as u8;
                    let mut taken = None;
                    for (k, arm) in arms.iter().enumerate() {
                        if self.mutant == Some(Mutant::ScrutineeTwice) && k > 0 {
                            self.expr(scrut)?;
                        }
                        let hit = match &arm.pat {
                            Pat::Variants(vs) => vs.contains(&variant),
                            Pat::Lits(_) => unreachable!("an integer pattern in a statement match"),
                            Pat::Wild => true,
                        };
                        if hit {
                            let no_wild = arms.iter().all(|a| a.pat != Pat::Wild);
                            let wrong = self.mutant == Some(Mutant::DefaultTakesFirstArm)
                                && no_wild
                                && k + 1 == arms.len();
                            taken = Some(if wrong { &arms[0] } else { arm });
                            break;
                        }
                    }
                    let arm = taken.expect("a match is exhaustive");
                    self.arm_depth += 1;
                    let done = self.block(&arm.body);
                    self.arm_depth -= 1;
                    done?;
                }
                Stmt::If { cond, then, els } => {
                    let taken = self.expr(cond)?.int() != 0;
                    if let Some(counting) = &mut self.coverage {
                        let entry = counting
                            .branches
                            .entry(std::ptr::from_ref(stmt) as usize)
                            .or_default();
                        if taken {
                            entry.0 += 1;
                        } else {
                            entry.1 += 1;
                        }
                    }
                    if taken {
                        self.block(then)?;
                    } else {
                        self.block(els)?;
                    }
                }
                Stmt::For { var, trips, body } => {
                    self.count(KIND_FOR, "for", "");
                    self.for_depth += 1;
                    for k in 0..*trips {
                        self.vars[*var as usize] = Val::I(k);
                        if let Err(violation) = self.block(body) {
                            self.for_depth -= 1;
                            return Err(violation);
                        }
                        if self.returned.is_some() {
                            break;
                        }
                    }
                    self.for_depth -= 1;
                }
            }
            if self.returned.is_some() {
                break;
            }
        }
        Ok(())
    }
}

fn variable_slots(program: &Program) -> usize {
    let mut max = 0;
    fn block(stmts: &[Stmt], max: &mut usize) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, .. }
                | Stmt::LetMatch { var, .. }
                | Stmt::Assign { var, .. }
                | Stmt::AssignComp { var, .. }
                | Stmt::For { var, .. } => {
                    *max = (*max).max(*var as usize + 1);
                }
                Stmt::LetTuple { vars, .. } => {
                    for &(var, _) in vars {
                        *max = (*max).max(var as usize + 1);
                    }
                }
                Stmt::If { .. }
                | Stmt::Set { .. }
                | Stmt::Call { .. }
                | Stmt::Match { .. }
                | Stmt::TailMatch { .. }
                | Stmt::Yield { .. }
                | Stmt::Return { .. } => {}
            }
            match stmt {
                Stmt::If { then, els, .. } => {
                    block(then, max);
                    block(els, max);
                }
                Stmt::For { body, .. } => block(body, max),
                Stmt::Match { arms, .. }
                | Stmt::LetMatch { arms, .. }
                | Stmt::TailMatch { arms, .. } => arms.iter().for_each(|arm| block(&arm.body, max)),
                _ => {}
            }
        }
    }
    block(&program.body, &mut max);
    for def in &program.methods.fns {
        block(&def.body, &mut max);
        for (var, _) in &def.params {
            max = max.max(*var as usize + 1);
        }
    }
    crate::ast::visit_exprs(program, &mut |e| {
        if let Expr::Var(var, _) = e {
            max = max.max(*var as usize + 1);
        }
    });
    max
}

/// The outcome of a run that stayed inside the domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The output buffer's bit patterns (`bool` as 0 or 1; an `f32` as the reference value's bits).
    pub values: Vec<u32>,
    /// For an `f32` output, the interval `[lo, hi]` of every element that a conforming implementation may hold;
    /// empty otherwise.
    pub intervals: Vec<(f64, f64)>,
}

/// Runs the kernel body for every element `0..count` and returns the output buffer and, for an `f32` output, the
/// interval of every element, or the first domain violation (lowest element, first violating evaluation in Rust's
/// order). An `f32` output element whose interval is wider than `2^-10` of its magnitude is `Unboundable`.
pub fn run_full(program: &Program) -> Result<Run, Failure> {
    run_machine(program, false, None).map(|(run, _)| run)
}

/// What a run executed, summed over all elements of the program: the evaluations of each operator (the names of
/// `ast::operator_counts`, so a syntactic count and an executed count compare), the `if` statements by the sides they
/// took, and the `&&`/`||` evaluations by whether the right operand ran. Dead code (a branch no element takes, the right
/// side of an `&&` whose left side is always false) shows as a syntactic operator with no evaluations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub operators: BTreeMap<String, u64>,
    /// `if` statements in the program (each counted once; evaluated or not).
    pub if_statements: u64,
    /// `if` statements some element evaluated.
    pub if_evaluated: u64,
    /// ... that took the then branch for some element, and the else branch for some element.
    pub if_then_taken: u64,
    pub if_else_taken: u64,
    /// ... that took both sides over the elements of the program.
    pub if_both_sides: u64,
    pub rhs_evaluated: u64,
    pub rhs_skipped: u64,
}

/// `run_full` with a deliberately wrong `match` evaluation (the mutation checks of slice m9.d1b).
pub fn run_mutant(program: &Program, mutant: Mutant) -> Result<Run, Failure> {
    run_machine(program, false, Some(mutant)).map(|(run, _)| run)
}

/// `run_full` that also counts what was executed.
pub fn run_covered(program: &Program) -> Result<(Run, Coverage), Failure> {
    run_machine(program, true, None).map(|(run, coverage)| (run, coverage.unwrap_or_default()))
}

fn count_ifs(stmts: &[Stmt]) -> u64 {
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::If { then, els, .. } => 1 + count_ifs(then) + count_ifs(els),
            Stmt::For { body, .. } => count_ifs(body),
            Stmt::Match { arms, .. }
            | Stmt::LetMatch { arms, .. }
            | Stmt::TailMatch { arms, .. } => arms.iter().map(|arm| count_ifs(&arm.body)).sum(),
            _ => 0,
        })
        .sum()
}

/// What one element gives: the output bits, the float interval if any, and the output slot of an array store.
type ElementRun = (u32, Option<(f64, f64)>, Option<usize>);

/// One element: the body, the result, and for an array program the slot of the final store. Rust's order for
/// `out[idx] op= result` on a primitive is the result first, then the index; the mutant `index-before-value` evaluates
/// the index first. The slot must be the element's own (the generator keeps it so; the GPU writes of different
/// invocations must not overlap), or the run is outside the checked domain.
fn run_element(machine: &mut Machine, program: &Program) -> Result<ElementRun, Violation> {
    machine.block(&program.body)?;
    let early = match &program.store {
        Some(store) if machine.mutant == Some(Mutant::IndexBeforeValue) => {
            Some(machine.expr(&store.index)?.int())
        }
        _ => None,
    };
    let value = machine.expr(&program.result)?;
    let (bits, interval) = match value {
        Val::I(bits) => (bits, None),
        Val::F(reference, fl) => {
            let (lo, hi) = fl.interval();
            if hi - lo > float::MAX_RELATIVE_WIDTH * lo.abs().max(hi.abs()) {
                return Err(Violation::Unboundable);
            }
            (reference.to_bits(), Some((lo, hi)))
        }
        Val::A(_) => unreachable!("Program::check rejects an aggregate result"),
    };
    let slot = match &program.store {
        Some(store) => {
            if machine.coverage.is_some() {
                for name in array_store_names(&program.methods, &program.result) {
                    machine.count_name(name);
                }
            }
            let idx = match early {
                Some(idx) => idx,
                None => machine.expr(&store.index)?.int(),
            };
            if idx != machine.element {
                return Err(Violation::Index);
            }
            Some(idx as usize)
        }
        None => None,
    };
    Ok((bits, interval, slot))
}

fn run_machine(
    program: &Program,
    covered: bool,
    mutant: Option<Mutant>,
) -> Result<(Run, Option<Coverage>), Failure> {
    let mut machine = Machine {
        program,
        element: 0,
        vars: vec![Val::I(0); variable_slots(program)],
        self_stack: Vec::new(),
        coverage: covered.then(Box::default),
        observed: None,
        returned: None,
        for_depth: 0,
        arm_depth: 0,
        mutant,
    };
    // An array program's output starts as the store's `init`, and element `i` stores into `out[idx]`.
    let init = program.store.as_ref().map_or(0, |store| store.init);
    let mut values = if program.store.is_some() {
        vec![init; program.count as usize]
    } else {
        Vec::with_capacity(program.count as usize)
    };
    let mut intervals = Vec::new();
    for element in 0..program.count {
        machine.element = element;
        machine.self_stack.clear();
        let (bits, interval, slot) = run_element(&mut machine, program)
            .map_err(|violation| Failure { element, violation })?;
        match slot {
            Some(slot) => {
                let stored = match program.store.as_ref().and_then(|store| store.op) {
                    None => bits,
                    Some(op) => binary(op, program.result.ty(), values[slot], bits)
                        .map_err(|violation| Failure { element, violation })?,
                };
                values[slot] = stored;
            }
            None => values.push(bits),
        }
        intervals.extend(interval);
    }
    let coverage = machine.coverage.map(|counting| {
        let mut coverage = Coverage {
            if_statements: count_ifs(&program.body)
                + program
                    .methods
                    .fns
                    .iter()
                    .map(|def| count_ifs(&def.body))
                    .sum::<u64>(),
            rhs_evaluated: counting.rhs_evaluated,
            rhs_skipped: counting.rhs_skipped,
            ..Coverage::default()
        };
        for ((kind, head, tail), n) in counting.operators {
            coverage
                .operators
                .insert(operator_name(kind, head, tail), n);
        }
        coverage.operators.extend(counting.named);
        for (then_taken, else_taken) in counting.branches.values() {
            coverage.if_evaluated += 1;
            coverage.if_then_taken += u64::from(*then_taken > 0);
            coverage.if_else_taken += u64::from(*else_taken > 0);
            coverage.if_both_sides += u64::from(*then_taken > 0 && *else_taken > 0);
        }
        coverage
    });
    Ok((Run { values, intervals }, coverage))
}

/// The first value (the bit pattern; an `f32` as its reference value) that each expression node of `program` takes at
/// `element`, by the node's address, for the reducer's "replace a subexpression by what it evaluated to" pass. A node
/// that element never evaluates is absent. Runs that one element only; a violation ends the walk early (the nodes seen
/// until then stay).
pub fn observed_values(program: &Program, element: u32) -> HashMap<usize, u32> {
    let mut machine = Machine {
        program,
        element,
        vars: vec![Val::I(0); variable_slots(program)],
        self_stack: Vec::new(),
        coverage: None,
        observed: Some(HashMap::new()),
        returned: None,
        for_depth: 0,
        arm_depth: 0,
        mutant: None,
    };
    let _ = machine
        .block(&program.body)
        .and_then(|()| machine.expr(&program.result));
    machine.observed.unwrap_or_default()
}

/// `run_full` without the intervals: the output buffer's bit patterns, or the first domain violation.
pub fn run(program: &Program) -> Result<Vec<u32>, Failure> {
    run_full(program).map(|run| run.values)
}

/// A constant of the static check.
#[allow(clippy::large_enum_variant)] // as `Val`: copied by value, folded once per node
#[derive(Clone, Copy)]
enum Const {
    I(u32),
    F(f32),
    /// A closed vector or matrix; its `f32` components carry exact intervals and are folded with the interpreter's
    /// own arithmetic (`agg`), so a closed aggregate that would leave the domain when run is a static violation too.
    A(Agg),
}

impl Const {
    fn int(self) -> u32 {
        match self {
            Const::I(bits) => bits,
            Const::F(_) | Const::A(_) => unreachable!("typed program"),
        }
    }

    fn float(self) -> f32 {
        match self {
            Const::F(value) => value,
            Const::I(_) | Const::A(_) => unreachable!("typed program"),
        }
    }

    fn agg(self) -> Agg {
        match self {
            Const::A(agg) => agg,
            Const::I(_) | Const::F(_) => unreachable!("typed program"),
        }
    }

    /// This constant as a component of an aggregate (a float with the exact interval of its value).
    fn scalar(self) -> Scalar {
        match self {
            Const::I(bits) => Scalar::I(bits),
            Const::F(value) => Scalar::F(value, Fl::exact(value)),
            Const::A(_) => unreachable!("typed program"),
        }
    }

    /// The constant that a component of an aggregate is (a float as its reference value).
    fn of(scalar: Scalar) -> Const {
        match scalar {
            Scalar::I(bits) => Const::I(bits),
            Scalar::F(reference, _) => Const::F(reference),
        }
    }

    /// The components of an operand of `AggBin`, `Dot` and the like: a scalar is one component.
    fn lanes(self) -> Agg {
        match self {
            Const::A(agg) => agg,
            scalar => Agg::new(&[scalar.scalar()]),
        }
    }
}

/// A closed `f32` constant must be zero or inside `+-[2^-100, 2^100]`.
fn float_constant(value: f32) -> Result<Option<Const>, Violation> {
    let v = f64::from(value);
    if float::in_range(v, v) {
        Ok(Some(Const::F(value)))
    } else {
        Err(Violation::FloatDomain)
    }
}

/// The first domain violation in a closed sub-expression (one made of literals only) anywhere in the program,
/// dead code included, or `None`. Closed sub-expressions are folded eagerly, without short-circuiting, with the same
/// `binary` and `unary` as the interpreter. A shift by a literal 32 or more and a division by a literal zero count
/// even when the other operand is not closed. Such a constant is a compile-time error in some shader languages
/// (const-expression rules), so the generator keeps none, in executed or dead code (labelled `static`). Closed
/// `f32` constants are folded in `f32` and must be zero or inside `+-[2^-100, 2^100]`; a closed float division by
/// zero, an integer above `2^24` converted to `f32` and a closed cast out of the integer range count too. A divisor
/// that slangc folds to zero without being a literal (the first iteration of a loop, `i * 0`, `i - i`) is not counted
/// since DIFFTEST_FORMAT 5: the macro lowers such a `/` and `%` to `__gust_div` and `__gust_rem` (slice m6.dv, D74),
/// which slangc compiles.
pub fn static_violation(program: &Program) -> Option<Violation> {
    fn fold(e: &Expr) -> Result<Option<Const>, Violation> {
        match e {
            Expr::Lit(Ty::F32, bits) => float_constant(f32::from_bits(*bits)),
            Expr::Lit(_, bits) => Ok(Some(Const::I(*bits))),
            Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Variant { .. } => Ok(None),
            // A field, a call or a struct literal is open; its operands are folded as nodes of their own.
            Expr::Read { .. }
            | Expr::Call { .. }
            | Expr::StructLit { .. }
            | Expr::Tuple { .. }
            | Expr::TupleField { .. }
            | Expr::ArrayLit { .. }
            | Expr::Repeat { .. }
            | Expr::ArrLen { .. } => Ok(None),
            Expr::Input { index, .. } => fold(index).map(|_| None),
            Expr::Un {
                op: UnOp::Neg,
                ty: Ty::F32,
                e,
            } => fold(e)?
                .map(|v| float_constant(-v.float()))
                .transpose()
                .map(Option::flatten),
            Expr::Un { op, ty, e } => fold(e)?
                .map(|v| unary(*op, *ty, v.int()).map(Const::I))
                .transpose(),
            Expr::Bin { op, ty, l, r } if *ty == Ty::F32 => {
                let (left, right) = (fold(l)?, fold(r)?);
                if *op == BinOp::Div && right.is_some_and(|v| v.float() == 0.0) {
                    return Err(Violation::DivisionByZero);
                }
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let (a, b) = (a.float(), b.float());
                let value = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Eq => return Ok(Some(Const::I(u32::from(a == b)))),
                    BinOp::Ne => return Ok(Some(Const::I(u32::from(a != b)))),
                    BinOp::Lt => return Ok(Some(Const::I(u32::from(a < b)))),
                    BinOp::Le => return Ok(Some(Const::I(u32::from(a <= b)))),
                    BinOp::Gt => return Ok(Some(Const::I(u32::from(a > b)))),
                    BinOp::Ge => return Ok(Some(Const::I(u32::from(a >= b)))),
                    other => unreachable!("float operator {}", other.symbol()),
                };
                float_constant(value)
            }
            Expr::Bin { op, ty, l, r } => {
                let (left, right) = (fold(l)?, fold(r)?);
                match (op, right) {
                    (BinOp::Div | BinOp::Rem, Some(Const::I(0))) => {
                        return Err(Violation::DivisionByZero);
                    }
                    (BinOp::Shl | BinOp::Shr, Some(Const::I(amount))) if amount >= 32 => {
                        return Err(Violation::ShiftAmount);
                    }
                    _ => {}
                }
                match (left, right) {
                    (Some(l), Some(r)) => {
                        binary(*op, *ty, l.int(), r.int()).map(|v| Some(Const::I(v)))
                    }
                    _ => Ok(None),
                }
            }
            Expr::Make { ty, rows, args } => {
                let mut parts = Vec::with_capacity(ty.lanes());
                let mut closed = true;
                for arg in args {
                    match fold(arg)? {
                        Some(Const::A(agg)) => parts.extend_from_slice(agg.lanes()),
                        Some(scalar) => parts.push(scalar.scalar()),
                        None => closed = false,
                    }
                }
                if !closed {
                    return Ok(None);
                }
                if *rows {
                    let Ty::Mat(n) = ty else {
                        unreachable!("from_rows of {}", ty.name())
                    };
                    parts = agg::from_rows(usize::from(*n), &parts);
                }
                Ok(Some(Const::A(Agg::new(&parts))))
            }
            Expr::Splat { ty, e } => {
                Ok(fold(e)?.map(|value| Const::A(Agg::new(&vec![value.scalar(); ty.lanes()]))))
            }
            Expr::Comp { e, index } => {
                Ok(fold(e)?.map(|value| Const::of(value.agg().c[usize::from(*index)])))
            }
            Expr::Swizzle { e, lanes } => Ok(fold(e)?.map(|value| {
                let agg = value.agg();
                let picked: Vec<Scalar> = lanes.iter().map(|&k| agg.c[usize::from(k)]).collect();
                Const::A(Agg::new(&picked))
            })),
            Expr::Dot { l, r } => {
                let (left, right) = (fold(l)?, fold(r)?);
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let sum = agg::dot(l.ty().scalar(), a.agg().lanes(), b.agg().lanes())?;
                Ok(Some(Const::of(sum)))
            }
            Expr::Transpose { e } => {
                let Ty::Mat(n) = e.ty() else {
                    unreachable!("transpose of a non-matrix")
                };
                Ok(fold(e)?.map(|value| {
                    Const::A(Agg::new(&agg::transpose(
                        usize::from(n),
                        value.agg().lanes(),
                    )))
                }))
            }
            Expr::AggNeg { e } => match fold(e)? {
                Some(value) => Ok(Some(Const::A(agg::negate(
                    e.ty().scalar(),
                    value.agg().lanes(),
                )?))),
                None => Ok(None),
            },
            Expr::AggBin { op, ty, l, r } => {
                let types = (l.ty(), r.ty());
                let (left, right) = (fold(l)?, fold(r)?);
                if matches!(op, BinOp::Div | BinOp::Rem)
                    && right.is_some_and(|divisor| {
                        divisor.lanes().lanes().iter().any(|part| match *part {
                            Scalar::I(bits) => bits == 0,
                            Scalar::F(value, _) => value == 0.0,
                        })
                    })
                {
                    return Err(Violation::DivisionByZero);
                }
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let (a, b) = (a.lanes(), b.lanes());
                Ok(Some(Const::A(agg::binary_op(
                    *op,
                    *ty,
                    types,
                    a.lanes(),
                    b.lanes(),
                )?)))
            }
            Expr::EnumAs { e, .. } => fold(e).map(|_| None),
            Expr::Cast { to, e } => {
                let from = e.ty();
                let Some(value) = fold(e)? else {
                    return Ok(None);
                };
                match (from, *to) {
                    (Ty::I32 | Ty::U32, Ty::F32) => {
                        let bits = value.int();
                        let magnitude = if from == Ty::I32 {
                            u64::from((bits as i32).unsigned_abs())
                        } else {
                            u64::from(bits)
                        };
                        if magnitude > float::EXACT_INT_LIMIT {
                            return Err(Violation::FloatConstruct);
                        }
                        let number = if from == Ty::I32 {
                            (bits as i32) as f32
                        } else {
                            bits as f32
                        };
                        float_constant(number)
                    }
                    (Ty::F32, Ty::I32 | Ty::U32) => {
                        let truncated = f64::from(value.float()).trunc();
                        let (min, max) = if *to == Ty::I32 {
                            (-(f64::from(i32::MAX)), f64::from(i32::MAX) - 1.0)
                        } else {
                            (0.0, f64::from(u32::MAX) - 1.0)
                        };
                        if truncated < min || truncated > max {
                            return Err(Violation::FloatDomain);
                        }
                        Ok(Some(Const::I(truncated as i64 as u32)))
                    }
                    _ => Ok(Some(value)),
                }
            }
        }
    }
    let mut found = None;
    // Every node folds its own subtree: quadratic in depth, but programs are small.
    crate::ast::visit_exprs(program, &mut |e| {
        if found.is_none() {
            found = fold(e).err();
        }
    });
    // Format 10 (m9.d1c): the right-hand side of a compound assignment is the operand of an implicit shift, division or
    // remainder, so a closed constant outside the domain there (`self.f <<= 34u32` in code that is reachable from the
    // kernel but never executes) is a static violation too: naga rejects the module although the interpreter never runs
    // the statement. Every space applies it (fix round 1; round 0 gated it to the value space).
    if found.is_none() {
        fn sets(
            stmts: &[Stmt],
            fold: fn(&Expr) -> Result<Option<Const>, Violation>,
        ) -> Option<Violation> {
            for stmt in stmts {
                let violation = match stmt {
                    Stmt::Set {
                        op: Some(op),
                        value,
                        ..
                    } => match (op, fold(value)) {
                        (BinOp::Shl | BinOp::Shr, Ok(Some(Const::I(amount)))) if amount >= 32 => {
                            Some(Violation::ShiftAmount)
                        }
                        (BinOp::Div | BinOp::Rem, Ok(Some(Const::I(0)))) => {
                            Some(Violation::DivisionByZero)
                        }
                        _ => None,
                    },
                    Stmt::If { then, els, .. } => sets(then, fold).or_else(|| sets(els, fold)),
                    Stmt::For { body, .. } => sets(body, fold),
                    Stmt::Match { arms, .. }
                    | Stmt::LetMatch { arms, .. }
                    | Stmt::TailMatch { arms, .. } => {
                        arms.iter().find_map(|arm| sets(&arm.body, fold))
                    }
                    _ => None,
                };
                if violation.is_some() {
                    return violation;
                }
            }
            None
        }
        found = sets(&program.body, fold).or_else(|| {
            program
                .methods
                .fns
                .iter()
                .find_map(|def| sets(&def.body, fold))
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::InputBuf;

    const MIN: u32 = 0x8000_0000;

    fn u(v: u32) -> Expr {
        Expr::u32(v)
    }

    fn i(v: i32) -> Expr {
        Expr::i32(v)
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::bin(op, l, r)
    }

    /// The program `out[i] = result` over `count` elements of the given `u32` input.
    fn program(data: &[u32], body: Vec<Stmt>, result: Expr) -> Program {
        Program {
            seed: 0,
            methods: Default::default(),
            store: None,
            attempt: 0,
            count: data.len() as u32,
            inputs: vec![InputBuf {
                ty: Ty::U32,
                data: data.to_vec(),
            }],
            body,
            result,
        }
    }

    /// One element, no input read: the value of a closed expression.
    fn eval(expr: Expr) -> Result<u32, Violation> {
        run(&program(&[0], Vec::new(), expr))
            .map(|out| out[0])
            .map_err(|failure| failure.violation)
    }

    #[test]
    fn unsigned_arithmetic_vectors() {
        assert_eq!(eval(bin(BinOp::Add, u(7), u(5))), Ok(12));
        assert_eq!(eval(bin(BinOp::Add, u(u32::MAX), u(0))), Ok(u32::MAX));
        assert_eq!(
            eval(bin(BinOp::Add, u(u32::MAX), u(1))),
            Err(Violation::Overflow)
        );
        assert_eq!(eval(bin(BinOp::Sub, u(5), u(3))), Ok(2));
        assert_eq!(eval(bin(BinOp::Sub, u(3), u(3))), Ok(0));
        assert_eq!(eval(bin(BinOp::Sub, u(0), u(1))), Err(Violation::Overflow));
        assert_eq!(eval(bin(BinOp::Mul, u(65535), u(65535))), Ok(4_294_836_225));
        assert_eq!(eval(bin(BinOp::Mul, u(u32::MAX), u(0))), Ok(0));
        assert_eq!(
            eval(bin(BinOp::Mul, u(65536), u(65536))),
            Err(Violation::Overflow)
        );
        assert_eq!(eval(bin(BinOp::Div, u(7), u(2))), Ok(3));
        assert_eq!(eval(bin(BinOp::Rem, u(7), u(2))), Ok(1));
        assert_eq!(eval(bin(BinOp::Div, u(u32::MAX), u(u32::MAX))), Ok(1));
    }

    #[test]
    fn signed_arithmetic_vectors() {
        assert_eq!(eval(bin(BinOp::Add, i(-5), i(3))), Ok(-2i32 as u32));
        assert_eq!(
            eval(bin(BinOp::Add, i(i32::MAX), i(0))),
            Ok(i32::MAX as u32)
        );
        assert_eq!(
            eval(bin(BinOp::Add, i(i32::MAX), i(1))),
            Err(Violation::Overflow)
        );
        assert_eq!(
            eval(bin(BinOp::Add, Expr::i32_min(), i(-1))),
            Err(Violation::Overflow)
        );
        assert_eq!(eval(Expr::i32_min()), Ok(MIN));
        assert_eq!(eval(bin(BinOp::Sub, i(-3), i(-4))), Ok(1));
        assert_eq!(
            eval(bin(BinOp::Sub, Expr::i32_min(), i(1))),
            Err(Violation::Overflow)
        );
        assert_eq!(
            eval(bin(BinOp::Sub, i(0), Expr::i32_min())),
            Err(Violation::Overflow)
        );
        assert_eq!(eval(bin(BinOp::Mul, i(46340), i(46340))), Ok(2_147_395_600));
        assert_eq!(
            eval(bin(BinOp::Mul, i(-46340), i(46340))),
            Ok((-2_147_395_600i32) as u32)
        );
        assert_eq!(
            eval(bin(BinOp::Mul, i(46341), i(46341))),
            Err(Violation::Overflow)
        );
        assert_eq!(
            eval(bin(BinOp::Mul, Expr::i32_min(), i(-1))),
            Err(Violation::Overflow)
        );
        assert_eq!(eval(bin(BinOp::Mul, Expr::i32_min(), i(1))), Ok(MIN));
    }

    #[test]
    fn negation_vectors() {
        let negate = |e| Expr::Un {
            op: UnOp::Neg,
            ty: Ty::I32,
            e: Box::new(e),
        };
        assert_eq!(eval(negate(bin(BinOp::Add, i(3), i(4)))), Ok(-7i32 as u32));
        assert_eq!(eval(negate(bin(BinOp::Sub, i(3), i(10)))), Ok(7));
        assert_eq!(eval(negate(Expr::i32_min())), Err(Violation::Overflow));
        // Negating the largest value is fine; only the minimum overflows.
        assert_eq!(eval(negate(i(i32::MAX))), Ok((i32::MIN + 1) as u32));
    }

    #[test]
    fn division_and_remainder_vectors() {
        assert_eq!(eval(bin(BinOp::Div, i(-7), i(2))), Ok(-3i32 as u32));
        assert_eq!(eval(bin(BinOp::Rem, i(-7), i(2))), Ok(-1i32 as u32));
        assert_eq!(eval(bin(BinOp::Div, i(7), i(-2))), Ok(-3i32 as u32));
        assert_eq!(eval(bin(BinOp::Rem, i(7), i(-2))), Ok(1));
        assert_eq!(eval(bin(BinOp::Div, i(-1), i(-1))), Ok(1));
        assert_eq!(eval(bin(BinOp::Div, Expr::i32_min(), i(1))), Ok(MIN));
        assert_eq!(eval(bin(BinOp::Rem, Expr::i32_min(), i(2))), Ok(0));
        assert_eq!(
            eval(bin(BinOp::Div, Expr::i32_min(), Expr::i32_min())),
            Ok(1)
        );
        // Each zero divisor, both types and both operators.
        for op in [BinOp::Div, BinOp::Rem] {
            assert_eq!(eval(bin(op, u(7), u(0))), Err(Violation::DivisionByZero));
            assert_eq!(eval(bin(op, i(7), i(0))), Err(Violation::DivisionByZero));
            assert_eq!(eval(bin(op, i(0), i(0))), Err(Violation::DivisionByZero));
        }
    }

    #[test]
    fn min_over_minus_one_is_a_violation_for_division_and_remainder() {
        for op in [BinOp::Div, BinOp::Rem] {
            assert_eq!(
                eval(bin(op, Expr::i32_min(), i(-1))),
                Err(Violation::MinOverMinusOne)
            );
        }
        // The neighbours are inside the domain.
        assert_eq!(
            eval(bin(BinOp::Div, Expr::i32_min(), i(-2))),
            Ok(0x4000_0000)
        );
        assert_eq!(
            eval(bin(BinOp::Div, i(i32::MIN + 1), i(-1))),
            Ok(i32::MAX as u32)
        );
        // `MIN / -1` on u32 operands means 0x8000_0000 / 0xFFFF_FFFF: an ordinary division.
        assert_eq!(eval(bin(BinOp::Div, u(MIN), u(u32::MAX))), Ok(0));
    }

    #[test]
    fn shift_vectors() {
        assert_eq!(eval(bin(BinOp::Shl, u(1), u(31))), Ok(MIN));
        assert_eq!(eval(bin(BinOp::Shl, u(1), u(0))), Ok(1));
        assert_eq!(eval(bin(BinOp::Shl, u(u32::MAX), u(4))), Ok(0xFFFF_FFF0));
        assert_eq!(eval(bin(BinOp::Shr, u(MIN), u(31))), Ok(1));
        assert_eq!(
            eval(bin(BinOp::Shl, u(1), u(32))),
            Err(Violation::ShiftAmount)
        );
        assert_eq!(
            eval(bin(BinOp::Shr, u(1), u(33))),
            Err(Violation::ShiftAmount)
        );
        assert_eq!(
            eval(bin(BinOp::Shl, u(1), u(u32::MAX))),
            Err(Violation::ShiftAmount)
        );
        // i32: bits shifted out are lost without a violation, the sign bit is shifted in by `>>`.
        assert_eq!(eval(bin(BinOp::Shl, i(1), u(31))), Ok(MIN));
        assert_eq!(eval(bin(BinOp::Shl, i(-1), u(31))), Ok(MIN));
        assert_eq!(eval(bin(BinOp::Shr, i(-8), u(1))), Ok(-4i32 as u32));
        assert_eq!(eval(bin(BinOp::Shr, i(-8), u(31))), Ok(u32::MAX));
        assert_eq!(eval(bin(BinOp::Shr, i(5), u(1))), Ok(2));
        assert_eq!(
            eval(bin(BinOp::Shr, i(-8), u(32))),
            Err(Violation::ShiftAmount)
        );
        assert_eq!(
            eval(bin(BinOp::Shl, i(1), u(64))),
            Err(Violation::ShiftAmount)
        );
    }

    #[test]
    fn bit_and_comparison_vectors() {
        assert_eq!(eval(bin(BinOp::BitAnd, u(0xF0F0), u(0xFF00))), Ok(0xF000));
        assert_eq!(eval(bin(BinOp::BitOr, u(0xF0F0), u(0xFF00))), Ok(0xFFF0));
        assert_eq!(eval(bin(BinOp::BitXor, u(0xF0F0), u(0xFF00))), Ok(0x0FF0));
        assert_eq!(eval(bin(BinOp::BitAnd, i(-1), i(0x0F))), Ok(15));
        assert_eq!(eval(Expr::inverted(u(0))), Ok(u32::MAX));
        assert_eq!(eval(Expr::inverted(i(5))), Ok(-6i32 as u32));
        assert_eq!(eval(Expr::inverted(Expr::bool(true))), Ok(0));
        assert_eq!(eval(Expr::inverted(Expr::bool(false))), Ok(1));
        // Signed and unsigned orderings differ for the same bits.
        assert_eq!(eval(bin(BinOp::Lt, i(-1), i(0))), Ok(1));
        assert_eq!(eval(bin(BinOp::Lt, u(u32::MAX), u(0))), Ok(0));
        assert_eq!(eval(bin(BinOp::Lt, Expr::i32_min(), i(i32::MAX))), Ok(1));
        assert_eq!(eval(bin(BinOp::Le, u(3), u(3))), Ok(1));
        assert_eq!(eval(bin(BinOp::Gt, u(3), u(3))), Ok(0));
        assert_eq!(eval(bin(BinOp::Ge, i(-3), i(-3))), Ok(1));
        assert_eq!(
            eval(bin(BinOp::Eq, Expr::bool(true), Expr::bool(true))),
            Ok(1)
        );
        assert_eq!(
            eval(bin(BinOp::Ne, Expr::bool(true), Expr::bool(false))),
            Ok(1)
        );
        assert_eq!(eval(bin(BinOp::Ne, u(4), u(4))), Ok(0));
    }

    #[test]
    fn cast_vectors_reinterpret_the_bits() {
        assert_eq!(eval(Expr::cast(Ty::U32, i(-1))), Ok(u32::MAX));
        assert_eq!(eval(Expr::cast(Ty::I32, u(MIN))), Ok(MIN));
        // The cast at 2^31 changes the comparison result.
        let big = u(MIN);
        assert_eq!(eval(bin(BinOp::Lt, big.clone(), u(1))), Ok(0));
        assert_eq!(eval(bin(BinOp::Lt, Expr::cast(Ty::I32, big), i(0))), Ok(1));
        assert_eq!(eval(Expr::cast(Ty::U32, i(7))), Ok(7));
    }

    #[test]
    fn logical_operators_short_circuit_like_rust() {
        let by_zero = || bin(BinOp::Eq, bin(BinOp::Div, u(1), u(0)), u(0));
        // The right side is not evaluated, so its violation cannot occur.
        assert_eq!(eval(bin(BinOp::And, Expr::bool(false), by_zero())), Ok(0));
        assert_eq!(eval(bin(BinOp::Or, Expr::bool(true), by_zero())), Ok(1));
        // When it is evaluated, it counts.
        assert_eq!(
            eval(bin(BinOp::And, Expr::bool(true), by_zero())),
            Err(Violation::DivisionByZero)
        );
        assert_eq!(
            eval(bin(BinOp::Or, Expr::bool(false), by_zero())),
            Err(Violation::DivisionByZero)
        );
        assert_eq!(
            eval(bin(BinOp::And, Expr::bool(true), Expr::bool(true))),
            Ok(1)
        );
        assert_eq!(
            eval(bin(BinOp::Or, Expr::bool(false), Expr::bool(false))),
            Ok(0)
        );
        // The left side is evaluated first: a violation there wins.
        assert_eq!(
            eval(bin(BinOp::Or, by_zero(), Expr::bool(true))),
            Err(Violation::DivisionByZero)
        );
    }

    #[test]
    fn index_must_be_below_len() {
        let data = [10, 20, 30, 40];
        let read = |index: Expr| program(&data, Vec::new(), Expr::input(0, Ty::U32, index));
        assert_eq!(run(&read(Expr::Gid)), Ok(vec![10, 20, 30, 40]));
        // `len() - 1` is the last element; `len()` is one past it (index == len is a violation).
        let last = bin(BinOp::Sub, Expr::Len(0), u(1));
        assert_eq!(run(&read(last)), Ok(vec![40, 40, 40, 40]));
        assert_eq!(
            run(&read(Expr::Len(0))),
            Err(Failure {
                element: 0,
                violation: Violation::Index
            })
        );
        // `i + 1` is out of bounds exactly at the last element.
        assert_eq!(
            run(&read(bin(BinOp::Add, Expr::Gid, u(1)))),
            Err(Failure {
                element: 3,
                violation: Violation::Index
            })
        );
        // A wrapped index stays inside: `(i + 5) % len()`.
        let wrapped = bin(BinOp::Rem, bin(BinOp::Add, Expr::Gid, u(5)), Expr::Len(0));
        assert_eq!(run(&read(wrapped)), Ok(vec![20, 30, 40, 10]));
    }

    #[test]
    fn the_first_failing_element_is_reported() {
        let result = bin(BinOp::Add, Expr::input(0, Ty::U32, Expr::Gid), u(1));
        assert_eq!(
            run(&program(&[1, u32::MAX, u32::MAX], Vec::new(), result)),
            Err(Failure {
                element: 1,
                violation: Violation::Overflow
            })
        );
    }

    #[test]
    fn statements_loops_and_branches() {
        // let mut acc = 0; for k in 0..4 { acc = acc + a[i]; } acc
        let accumulate = |data: &[u32]| {
            program(
                data,
                vec![
                    Stmt::Let {
                        var: 0,
                        ty: Ty::U32,
                        mutable: true,
                        init: u(0),
                    },
                    Stmt::For {
                        var: 1,
                        trips: 4,
                        body: vec![Stmt::Assign {
                            var: 0,
                            value: bin(
                                BinOp::Add,
                                Expr::Var(0, Ty::U32),
                                Expr::input(0, Ty::U32, Expr::Gid),
                            ),
                        }],
                    },
                ],
                Expr::Var(0, Ty::U32),
            )
        };
        assert_eq!(run(&accumulate(&[1, 2, 3])), Ok(vec![4, 8, 12]));
        // 4 * 2^30 = 2^32: the fourth trip overflows, the third does not.
        assert_eq!(
            run(&accumulate(&[1 << 30])),
            Err(Failure {
                element: 0,
                violation: Violation::Overflow
            })
        );
        assert_eq!(run(&accumulate(&[(1 << 30) - 1])), Ok(vec![u32::MAX - 3]));
        // The loop variable counts 0, 1, 2: sum of k for 3 trips is 3.
        let sum_of_k = program(
            &[0],
            vec![
                Stmt::Let {
                    var: 0,
                    ty: Ty::U32,
                    mutable: true,
                    init: u(0),
                },
                Stmt::For {
                    var: 1,
                    trips: 3,
                    body: vec![Stmt::Assign {
                        var: 0,
                        value: bin(BinOp::Add, Expr::Var(0, Ty::U32), Expr::Var(1, Ty::U32)),
                    }],
                },
            ],
            Expr::Var(0, Ty::U32),
        );
        assert_eq!(run(&sum_of_k), Ok(vec![3]));
        // if a[i] > 1 { r = 10 } else { r = 20 }
        let branch = program(
            &[0, 1, 2, 3],
            vec![
                Stmt::Let {
                    var: 0,
                    ty: Ty::U32,
                    mutable: true,
                    init: u(0),
                },
                Stmt::If {
                    cond: bin(BinOp::Gt, Expr::input(0, Ty::U32, Expr::Gid), u(1)),
                    then: vec![Stmt::Assign {
                        var: 0,
                        value: u(10),
                    }],
                    els: vec![Stmt::Assign {
                        var: 0,
                        value: u(20),
                    }],
                },
            ],
            Expr::Var(0, Ty::U32),
        );
        assert_eq!(run(&branch), Ok(vec![20, 20, 10, 10]));
    }

    #[test]
    fn a_violation_in_a_branch_that_is_not_taken_does_not_count() {
        let program = program(
            &[0, 1],
            vec![Stmt::If {
                cond: bin(BinOp::Eq, Expr::input(0, Ty::U32, Expr::Gid), u(5)),
                then: vec![Stmt::Let {
                    var: 0,
                    ty: Ty::U32,
                    mutable: false,
                    init: bin(BinOp::Div, u(1), u(0)),
                }],
                els: Vec::new(),
            }],
            u(9),
        );
        assert_eq!(run(&program), Ok(vec![9, 9]));
    }

    #[test]
    fn a_bool_result_is_zero_or_one() {
        let result = bin(BinOp::Lt, Expr::input(0, Ty::U32, Expr::Gid), u(2));
        assert_eq!(
            run(&program(&[0, 1, 2, 3], Vec::new(), result)),
            Ok(vec![1, 1, 0, 0])
        );
    }

    /// The program whose body is `if false { let v0 = <init> }` (dead code) and whose result is `a[i]`.
    fn dead_code_with(init: Expr, ty: Ty) -> Program {
        program(
            &[0, 1],
            vec![Stmt::If {
                cond: Expr::bool(false),
                then: vec![Stmt::Let {
                    var: 0,
                    ty,
                    mutable: false,
                    init,
                }],
                els: Vec::new(),
            }],
            Expr::input(0, Ty::U32, Expr::Gid),
        )
    }

    #[test]
    fn static_violation_finds_closed_constants_outside_the_domain_even_in_dead_code() {
        let a0 = Expr::input(0, Ty::U32, Expr::Gid);
        let cases: Vec<(Expr, Ty, Option<Violation>)> = vec![
            // Literal shift amounts: 32 and 34 are out, 31 is in; the left operand need not be closed.
            (
                bin(BinOp::Shl, u(1), u(32)),
                Ty::U32,
                Some(Violation::ShiftAmount),
            ),
            (
                bin(BinOp::Shr, a0.clone(), u(34)),
                Ty::U32,
                Some(Violation::ShiftAmount),
            ),
            (bin(BinOp::Shl, a0.clone(), u(31)), Ty::U32, None),
            // A closed shift amount that is not a bare literal.
            (
                bin(BinOp::Shl, a0.clone(), bin(BinOp::Add, u(30), u(2))),
                Ty::U32,
                Some(Violation::ShiftAmount),
            ),
            // Constant overflow, unsigned and signed, and negating the minimum.
            (
                bin(BinOp::Add, u(u32::MAX), u(1)),
                Ty::U32,
                Some(Violation::Overflow),
            ),
            (
                bin(BinOp::Sub, u(0), u(1)),
                Ty::U32,
                Some(Violation::Overflow),
            ),
            (
                bin(BinOp::Mul, i(65536), i(65536)),
                Ty::I32,
                Some(Violation::Overflow),
            ),
            (
                Expr::negated(Expr::i32_min()),
                Ty::I32,
                Some(Violation::Overflow),
            ),
            (bin(BinOp::Add, i(i32::MAX), i(0)), Ty::I32, None),
            // A literal zero divisor with an open dividend, and a closed one.
            (
                bin(BinOp::Rem, a0.clone(), u(0)),
                Ty::U32,
                Some(Violation::DivisionByZero),
            ),
            (
                bin(BinOp::Div, u(7), bin(BinOp::Sub, u(3), u(3))),
                Ty::U32,
                Some(Violation::DivisionByZero),
            ),
            (bin(BinOp::Div, a0.clone(), u(1)), Ty::U32, None),
            // MIN / -1 as a closed constant.
            (
                bin(BinOp::Div, Expr::i32_min(), i(-1)),
                Ty::I32,
                Some(Violation::MinOverMinusOne),
            ),
            // A violation inside the right operand of a short-circuiting `&&`, and under an index: still static.
            (
                bin(
                    BinOp::And,
                    Expr::bool(false),
                    bin(BinOp::Lt, bin(BinOp::Shl, u(1), u(40)), u(1)),
                ),
                Ty::Bool,
                Some(Violation::ShiftAmount),
            ),
            (
                Expr::input(0, Ty::U32, bin(BinOp::Shl, u(1), u(33))),
                Ty::U32,
                Some(Violation::ShiftAmount),
            ),
            // Open operands never fold: the interpreter, not this check, decides about them.
            (bin(BinOp::Add, a0.clone(), u(u32::MAX)), Ty::U32, None),
        ];
        for (init, ty, expected) in cases {
            let program = dead_code_with(init.clone(), ty);
            assert_eq!(static_violation(&program), expected, "{init:?}");
            // The dead code never runs, so the interpreter accepts every case: only the static check can see it.
            assert!(run(&program).is_ok(), "{init:?}");
        }
    }

    #[test]
    fn static_violation_also_checks_the_result_and_loop_bodies() {
        let in_result = program(&[0], Vec::new(), bin(BinOp::Shr, u(1), u(32)));
        assert_eq!(static_violation(&in_result), Some(Violation::ShiftAmount));
        let in_loop = program(
            &[0],
            vec![Stmt::For {
                var: 0,
                trips: 2,
                body: vec![Stmt::Assign {
                    var: 1,
                    value: bin(BinOp::Add, u(u32::MAX), u(1)),
                }],
            }],
            u(0),
        );
        assert_eq!(static_violation(&in_loop), Some(Violation::Overflow));
    }

    // ---- floats (R1-4) ----

    fn f(v: f32) -> Expr {
        Expr::f32(v)
    }

    fn fbin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::bin(op, l, r)
    }

    /// A program over one `f32` input buffer holding `data`.
    fn float_program(data: &[f32], body: Vec<Stmt>, result: Expr) -> Program {
        Program {
            seed: 0,
            methods: Default::default(),
            store: None,
            attempt: 0,
            count: data.len() as u32,
            inputs: vec![InputBuf {
                ty: Ty::F32,
                data: data.iter().map(|v| v.to_bits()).collect(),
            }],
            body,
            result,
        }
    }

    fn fin(index: Expr) -> Expr {
        Expr::input(0, Ty::F32, index)
    }

    /// The violation of a float result evaluated for one element holding `1.0`.
    fn float_violation(result: Expr) -> Option<Violation> {
        let program = float_program(&[1.0], Vec::new(), result);
        program.check().unwrap();
        run_full(&program).err().map(|failure| failure.violation)
    }

    fn lit_u32_cast(value: u32) -> Expr {
        Expr::cast(Ty::F32, u(value))
    }

    #[test]
    fn the_reference_value_is_the_f32_arithmetic_of_the_written_order() {
        // 0.1 + 0.2 in f32 (not f64), left to right, and the result's bits are the reference value's.
        let result = fbin(BinOp::Add, fbin(BinOp::Add, f(0.1), f(0.2)), fin(Expr::Gid));
        let program = float_program(&[3.0, 4.0], Vec::new(), result);
        let run = run_full(&program).unwrap();
        let expected = [(0.1f32 + 0.2f32) + 3.0f32, (0.1f32 + 0.2f32) + 4.0f32];
        assert_eq!(run.values, expected.map(f32::to_bits));
        assert_eq!(run.intervals.len(), 2);
        for (value, (lo, hi)) in expected.iter().zip(&run.intervals) {
            assert!(*lo <= f64::from(*value) && f64::from(*value) <= *hi);
        }
        // An integer program has no intervals.
        let ints = program_int_input();
        assert!(run_full(&ints).unwrap().intervals.is_empty());
    }

    fn program_int_input() -> Program {
        program(&[1], Vec::new(), Expr::input(0, Ty::U32, Expr::Gid))
    }

    #[test]
    fn a_product_with_a_zero_input_is_exact_zero() {
        let program = float_program(
            &[0.0, 2.0],
            Vec::new(),
            fbin(BinOp::Mul, fin(Expr::Gid), f(0.5)),
        );
        let run = run_full(&program).unwrap();
        assert_eq!(run.values, [0.0f32.to_bits(), 1.0f32.to_bits()]);
        assert_eq!(run.intervals[0], (0.0, 0.0));
    }

    #[test]
    fn the_16777217_cast_is_discarded_and_16777216_is_not() {
        // 16777217 is the first integer that f32 cannot hold: 15.7.6 lets the GPU pick either neighbour.
        assert_eq!(
            float_violation(lit_u32_cast(16_777_217)),
            Some(Violation::FloatConstruct)
        );
        assert_eq!(float_violation(lit_u32_cast(16_777_216)), None);
        assert_eq!(float_violation(lit_u32_cast(16_777_215)), None);
        assert_eq!(
            float_violation(Expr::cast(Ty::F32, i(-16_777_217))),
            Some(Violation::FloatConstruct)
        );
        assert_eq!(float_violation(Expr::cast(Ty::F32, i(-16_777_216))), None);
        // The same through a value read at run time.
        let at_run_time = program(
            &[16_777_216, 16_777_217],
            Vec::new(),
            Expr::cast(Ty::F32, Expr::input(0, Ty::U32, Expr::Gid)),
        );
        assert_eq!(
            run_full(&at_run_time).map_err(|failure| (failure.element, failure.violation)),
            Err((1, Violation::FloatConstruct))
        );
        // Statically, in dead code, where the interpreter never looks.
        let dead = Program {
            body: vec![Stmt::If {
                cond: Expr::bool(false),
                then: vec![Stmt::Let {
                    var: 0,
                    ty: Ty::F32,
                    mutable: false,
                    init: lit_u32_cast(16_777_217),
                }],
                els: Vec::new(),
            }],
            ..float_program(&[1.0], Vec::new(), fin(Expr::Gid))
        };
        assert_eq!(static_violation(&dead), Some(Violation::FloatConstruct));
        assert!(run_full(&dead).is_ok());
    }

    /// The value (0 or 1) of a float comparison, or the violation.
    fn comparison_result(cond: Expr) -> Result<u32, Violation> {
        let program = float_program(&[1.0], Vec::new(), cond);
        program.check().unwrap();
        run_full(&program)
            .map(|run| run.values[0])
            .map_err(|failure| failure.violation)
    }

    #[test]
    fn a_comparison_of_overlapping_intervals_is_discarded_and_a_disjoint_one_is_decided() {
        let sum = fbin(BinOp::Add, f(1.0), f(0.0));
        // `1.0 + 0.0` is an interval around 1.0, so comparing it with 1.0 overlaps, whatever the operator ...
        for op in [
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Le,
            BinOp::Gt,
            BinOp::Ge,
        ] {
            assert_eq!(
                comparison_result(fbin(op, sum.clone(), f(1.0))),
                Err(Violation::ComparisonOverlap),
                "{}",
                op.symbol()
            );
        }
        // ... and with 2.0 it does not.
        let against_two = |op| comparison_result(fbin(op, sum.clone(), f(2.0)));
        assert_eq!(against_two(BinOp::Lt), Ok(1));
        assert_eq!(against_two(BinOp::Le), Ok(1));
        assert_eq!(against_two(BinOp::Gt), Ok(0));
        assert_eq!(against_two(BinOp::Ge), Ok(0));
        assert_eq!(against_two(BinOp::Eq), Ok(0));
        assert_eq!(against_two(BinOp::Ne), Ok(1));
        // A branch on an overlapping comparison is discarded, whichever branch it would have taken.
        let branch = float_program(
            &[1.0],
            vec![Stmt::If {
                cond: fbin(BinOp::Le, fin(Expr::Gid), f(1.0)),
                then: Vec::new(),
                els: Vec::new(),
            }],
            fin(Expr::Gid),
        );
        assert_eq!(
            run_full(&branch).map_err(|failure| failure.violation),
            Err(Violation::ComparisonOverlap)
        );
    }

    #[test]
    fn division_needs_a_divisor_interval_away_from_zero() {
        assert_eq!(
            float_violation(fbin(BinOp::Div, f(1.0), f(0.0))),
            Some(Violation::FloatDomain)
        );
        // 3.0 - 3.0 is an interval around zero, not [0, 0]: the GPU's divisor may be zero or tiny.
        let around_zero = fbin(BinOp::Sub, f(3.0), f(3.0));
        assert_eq!(
            float_violation(fbin(BinOp::Div, f(1.0), around_zero)),
            Some(Violation::FloatDomain)
        );
        assert_eq!(float_violation(fbin(BinOp::Div, f(1.0), f(4.0))), None);
        assert_eq!(float_violation(fbin(BinOp::Div, f(1.0), f(-4.0))), None);
        // A divisor past 2^100 (here 2^120, from a product), and one below 2^-100.
        let big = fbin(BinOp::Mul, f(2f32.powi(60)), f(2f32.powi(60)));
        assert_eq!(
            float_violation(fbin(BinOp::Div, f(1.0), big)),
            Some(Violation::FloatDomain)
        );
        assert_eq!(
            float_violation(fbin(BinOp::Div, f(1.0), f(2f32.powi(-101)))),
            Some(Violation::FloatDomain)
        );
    }

    #[test]
    fn every_float_intermediate_stays_inside_the_range() {
        let big = f(2f32.powi(60));
        // 2^120 > 2^100 and 2^-120 < 2^-100.
        assert_eq!(
            float_violation(fbin(BinOp::Mul, big.clone(), big)),
            Some(Violation::FloatDomain)
        );
        let small = f(2f32.powi(-60));
        assert_eq!(
            float_violation(fbin(BinOp::Mul, small.clone(), small)),
            Some(Violation::FloatDomain)
        );
        assert_eq!(
            float_violation(fbin(BinOp::Mul, f(2f32.powi(40)), f(2f32.powi(40)))),
            None
        );
        // A literal outside the range is out; exactly zero is in.
        assert_eq!(
            float_violation(f(2f32.powi(-101))),
            Some(Violation::FloatDomain)
        );
        assert_eq!(
            float_violation(f(2f32.powi(101))),
            Some(Violation::FloatDomain)
        );
        assert_eq!(float_violation(f(0.0)), None);
    }

    #[test]
    fn a_result_wider_than_two_to_the_minus_ten_relative_is_unboundable() {
        // 1 + 4096 - 4095 = 2. The chain has three terms: two ULPs of 8192 (2^-10 each) are 2^-9 each side, a width of
        // 2^-8 on a result of 2, which is 2^-9 relative: wider than 2^-10.
        let near = fbin(
            BinOp::Sub,
            fbin(BinOp::Add, fin(Expr::Gid), f(4096.0)),
            f(4095.0),
        );
        let program = float_program(&[1.0], Vec::new(), near);
        assert_eq!(
            run_full(&program).map_err(|failure| failure.violation),
            Err(Violation::Unboundable)
        );
        // The same sum with small terms is bounded.
        let small = fbin(BinOp::Sub, fbin(BinOp::Add, fin(Expr::Gid), f(4.0)), f(2.0));
        let run = run_full(&float_program(&[1.0], Vec::new(), small)).unwrap();
        assert_eq!(run.values, [3.0f32.to_bits()]);
        // x - x: the interval holds zero without being [0, 0], so the range rule discards it first.
        let zero = float_program(
            &[1.0],
            Vec::new(),
            fbin(BinOp::Sub, fin(Expr::Gid), fin(Expr::Gid)),
        );
        assert_eq!(
            run_full(&zero).map_err(|failure| failure.violation),
            Err(Violation::FloatDomain)
        );
    }

    /// The integer a float cast gives for one element, or the violation.
    fn float_to_int_result(to: Ty, e: Expr) -> Result<u32, Violation> {
        // The program reads its input in a dead binding; the cast is of the closed expression `e`.
        let mut program = float_program(&[1.0], Vec::new(), Expr::cast(to, e));
        program.body = vec![Stmt::Let {
            var: 0,
            ty: Ty::F32,
            mutable: false,
            init: fin(Expr::Gid),
        }];
        program.check().unwrap();
        run(&program)
            .map(|out| out[0])
            .map_err(|failure| failure.violation)
    }

    #[test]
    fn float_to_integer_casts_need_one_integer_for_the_whole_interval() {
        assert_eq!(float_to_int_result(Ty::I32, f(2.5)), Ok(2));
        assert_eq!(float_to_int_result(Ty::I32, f(-2.5)), Ok((-2i32) as u32));
        assert_eq!(float_to_int_result(Ty::U32, f(2.75)), Ok(2));
        assert_eq!(float_to_int_result(Ty::I32, f(0.0)), Ok(0));
        // 2.0 + 1.0 is an interval around 3.0: below and above it, so it spans 2 and 3.
        let three = fbin(BinOp::Add, f(2.0), f(1.0));
        assert_eq!(
            float_to_int_result(Ty::I32, three.clone()),
            Err(Violation::Unboundable)
        );
        assert_eq!(
            float_to_int_result(Ty::U32, three),
            Err(Violation::Unboundable)
        );
        // 2.5 + 0.25 is around 2.75: one integer.
        assert_eq!(
            float_to_int_result(Ty::I32, fbin(BinOp::Add, f(2.5), f(0.25))),
            Ok(2)
        );
        // A negative value to u32 is outside the target range.
        assert_eq!(
            float_to_int_result(Ty::U32, f(-2.5)),
            Err(Violation::FloatDomain)
        );
        // The range, with a margin of one: 2^31 is out for i32, 2147483520 (the f32 below it) is in.
        assert_eq!(
            float_to_int_result(Ty::I32, f(2_147_483_648.0)),
            Err(Violation::FloatDomain)
        );
        assert_eq!(
            float_to_int_result(Ty::I32, f(2_147_483_520.0)),
            Ok(2_147_483_520)
        );
        assert_eq!(
            float_to_int_result(Ty::I32, f(-2_147_483_648.0)),
            Err(Violation::FloatDomain)
        );
    }

    #[test]
    fn a_loop_accumulator_is_one_chain_of_trips_plus_one_terms() {
        // acc = x; for k in 0..8 { acc = acc + 1.0 }: nine terms, eight roundings of one ULP of 9.
        let body = vec![
            Stmt::Let {
                var: 0,
                ty: Ty::F32,
                mutable: true,
                init: fin(Expr::Gid),
            },
            Stmt::For {
                var: 1,
                trips: 8,
                body: vec![Stmt::Assign {
                    var: 0,
                    value: fbin(BinOp::Add, Expr::Var(0, Ty::F32), f(1.0)),
                }],
            },
        ];
        let program = float_program(&[1.0], body, Expr::Var(0, Ty::F32));
        program.check().unwrap();
        let run = run_full(&program).unwrap();
        assert_eq!(run.values, [9.0f32.to_bits()]);
        let (lo, hi) = run.intervals[0];
        let w = 8.0 * (crate::float::ulp32(9.0) + crate::float::MARGIN * 9.0);
        assert_eq!((lo, hi), (9.0 - w, 9.0 + w));
        // Not eight independent single-operation widenings of the written-order partial sums 2, 3, ..., 9.
        let per_operation: f64 = (2..=9).map(|k| crate::float::ulp32(f64::from(k))).sum();
        assert!(w > per_operation, "{w} <= {per_operation}");
        // A chain through a variable is also a chain: `(x + 2)` stored in a variable, then `+ 4`, is three terms.
        let two_step = float_program(
            &[1.0],
            vec![Stmt::Let {
                var: 0,
                ty: Ty::F32,
                mutable: false,
                init: fbin(BinOp::Add, fin(Expr::Gid), f(2.0)),
            }],
            fbin(BinOp::Add, Expr::Var(0, Ty::F32), f(4.0)),
        );
        let (lo, hi) = run_full(&two_step).unwrap().intervals[0];
        let w = 2.0 * (crate::float::ulp32(7.0) + crate::float::MARGIN * 7.0);
        assert_eq!((lo, hi), (7.0 - w, 7.0 + w));
    }

    #[test]
    fn static_violation_covers_closed_float_constants() {
        let in_dead_code = |init: Expr, ty: Ty| {
            static_violation(&Program {
                body: vec![Stmt::If {
                    cond: Expr::bool(false),
                    then: vec![Stmt::Let {
                        var: 0,
                        ty,
                        mutable: false,
                        init,
                    }],
                    els: Vec::new(),
                }],
                ..float_program(&[1.0], Vec::new(), fin(Expr::Gid))
            })
        };
        assert_eq!(
            in_dead_code(fbin(BinOp::Div, f(1.0), f(0.0)), Ty::F32),
            Some(Violation::DivisionByZero)
        );
        assert_eq!(
            in_dead_code(fbin(BinOp::Div, fin(Expr::Gid), f(0.0)), Ty::F32),
            Some(Violation::DivisionByZero)
        );
        assert_eq!(
            in_dead_code(fbin(BinOp::Mul, f(1e30), f(1e30)), Ty::F32),
            Some(Violation::FloatDomain)
        );
        assert_eq!(
            in_dead_code(fbin(BinOp::Lt, f(1.0), f(2.0)), Ty::Bool),
            None
        );
        assert_eq!(
            in_dead_code(fbin(BinOp::Add, f(1.0), f(2.0)), Ty::F32),
            None
        );
        assert_eq!(
            in_dead_code(Expr::cast(Ty::I32, f(3e9)), Ty::I32),
            Some(Violation::FloatDomain)
        );
        assert_eq!(in_dead_code(Expr::cast(Ty::I32, f(3.5)), Ty::I32), None);
    }
    #[test]
    fn coverage_counts_what_is_executed_not_what_is_written() {
        // Four elements. The first `if` is taken by elements 2 and 3; the second is dead (a literal `false`), so the
        // multiplication in it is written but never executed.
        let a_i = Expr::input(0, Ty::U32, Expr::Gid);
        let body = vec![
            Stmt::Let {
                var: 0,
                ty: Ty::U32,
                mutable: true,
                init: u(0),
            },
            Stmt::If {
                cond: bin(BinOp::Ge, Expr::Gid, u(2)),
                then: vec![Stmt::Assign {
                    var: 0,
                    value: bin(BinOp::Add, a_i, u(1)),
                }],
                els: Vec::new(),
            },
            Stmt::If {
                cond: Expr::bool(false),
                then: vec![Stmt::Assign {
                    var: 0,
                    value: bin(BinOp::Mul, u(2), u(3)),
                }],
                els: Vec::new(),
            },
        ];
        let result = bin(BinOp::BitXor, Expr::Var(0, Ty::U32), u(0));
        let program = program(&[7, 7, 5, 6], body, result);
        let (run, coverage) = run_covered(&program).unwrap();
        assert_eq!(run.values, [0, 0, 6, 7]);
        assert_eq!(coverage.operators[">= u32"], 4);
        assert_eq!(coverage.operators["^ u32"], 4);
        // The addition runs for the two elements that take the branch; the dead multiplication never runs.
        assert_eq!(coverage.operators["+ u32"], 2);
        assert!(!coverage.operators.contains_key("* u32"));
        assert!(crate::ast::operator_counts(&program).contains_key("* u32"));
        assert_eq!((coverage.if_statements, coverage.if_evaluated), (2, 2));
        // First `if`: both sides. Second: only the (empty) else side.
        assert_eq!(coverage.if_both_sides, 1);
        assert_eq!((coverage.if_then_taken, coverage.if_else_taken), (1, 2));
    }

    #[test]
    fn coverage_counts_the_right_operand_of_a_logical_operator_only_when_it_runs() {
        let a_i = Expr::input(0, Ty::U32, Expr::Gid);
        let guard = bin(
            BinOp::And,
            bin(BinOp::Lt, Expr::Gid, u(2)),
            bin(BinOp::Eq, a_i, u(7)),
        );
        let (run, coverage) = run_covered(&program(&[7, 7, 5, 6], Vec::new(), guard)).unwrap();
        assert_eq!(run.values, [1, 1, 0, 0]);
        assert_eq!(coverage.operators["&& bool"], 4);
        assert_eq!((coverage.rhs_evaluated, coverage.rhs_skipped), (2, 2));
        assert_eq!(coverage.operators["== u32"], 2);
        assert_eq!(coverage.operators["< u32"], 4);
    }

    #[test]
    fn covered_and_plain_runs_agree_and_executed_operators_are_written_ones() {
        for seed in 1..=3 {
            for program in crate::generator::generate_batch(seed).unwrap().programs {
                let plain = run_full(&program).unwrap();
                let (covered, coverage) = run_covered(&program).unwrap();
                assert_eq!(plain, covered);
                let written = crate::ast::operator_counts(&program);
                for name in coverage.operators.keys() {
                    // A `return` that runs inside a `match` arm is counted when it runs only.
                    if name == crate::ast::MATCH_ARM_RETURN_RUN {
                        continue;
                    }
                    assert!(
                        written.contains_key(name),
                        "{name} executed but not written"
                    );
                }
            }
        }
    }
}
