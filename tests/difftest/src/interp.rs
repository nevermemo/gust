//! The interpreter: the independent host reference of the generator (D26). It evaluates the AST directly with
//! checked integer operations and never sees the emitted text.
//!
//! Besides the value it tracks the domain of D38 (docs/SEMANTICS.md section 3, items 1 to 3 and 6): a program run
//! that leaves the domain at any element, in any evaluated sub-expression, returns the first `Violation` and the
//! generator discards the program. Evaluation order and short-circuiting follow Rust, so a sub-expression Rust does
//! not evaluate (the right side of `&&` after `false`) cannot violate the domain.

use crate::ast::{BinOp, Expr, Program, Stmt, Ty, UnOp};

/// A way to leave the defined integer domain.
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
}

impl Violation {
    pub const ALL: [Violation; 5] = [
        Violation::Overflow,
        Violation::DivisionByZero,
        Violation::MinOverMinusOne,
        Violation::ShiftAmount,
        Violation::Index,
    ];

    /// The name used in discard counts and in `seeds.txt`.
    pub fn label(self) -> &'static str {
        match self {
            Violation::Overflow => "overflow",
            Violation::DivisionByZero => "division-by-zero",
            Violation::MinOverMinusOne => "min-over-minus-one",
            Violation::ShiftAmount => "shift-amount",
            Violation::Index => "index",
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
/// without short-circuiting here; `Machine::expr` short-circuits before calling this.
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

/// Evaluates `op e` for an operand of type `ty`.
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

struct Machine<'a> {
    program: &'a Program,
    element: u32,
    vars: Vec<u32>,
}

impl Machine<'_> {
    fn expr(&mut self, expr: &Expr) -> Result<u32, Violation> {
        match expr {
            Expr::Lit(_, bits) => Ok(*bits),
            Expr::Var(var, _) => Ok(self.vars[*var as usize]),
            Expr::Gid => Ok(self.element),
            Expr::Len(buf) => Ok(self.program.inputs[*buf].data.len() as u32),
            Expr::Input { buf, index, .. } => {
                let index = self.expr(index)?;
                self.program.inputs[*buf]
                    .data
                    .get(index as usize)
                    .copied()
                    .ok_or(Violation::Index)
            }
            Expr::Un { op, ty, e } => {
                let value = self.expr(e)?;
                unary(*op, *ty, value)
            }
            Expr::Bin {
                op: op @ (BinOp::And | BinOp::Or),
                l,
                r,
                ..
            } => {
                let left = self.expr(l)?;
                if (*op == BinOp::And) == (left == 0) {
                    return Ok(left);
                }
                self.expr(r)
            }
            Expr::Bin { op, ty, l, r } => {
                let left = self.expr(l)?;
                let right = self.expr(r)?;
                binary(*op, *ty, left, right)
            }
            Expr::Cast { e, .. } => self.expr(e),
        }
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), Violation> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, init, .. } => {
                    let value = self.expr(init)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::Assign { var, value } => {
                    let value = self.expr(value)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::If { cond, then, els } => {
                    if self.expr(cond)? != 0 {
                        self.block(then)?;
                    } else {
                        self.block(els)?;
                    }
                }
                Stmt::For { var, trips, body } => {
                    for k in 0..*trips {
                        self.vars[*var as usize] = k;
                        self.block(body)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn variable_slots(program: &Program) -> usize {
    fn expr(e: &Expr, max: &mut usize) {
        match e {
            Expr::Var(var, _) => *max = (*max).max(*var as usize + 1),
            Expr::Lit(..) | Expr::Gid | Expr::Len(_) => {}
            Expr::Input { index, .. } => expr(index, max),
            Expr::Un { e, .. } | Expr::Cast { e, .. } => expr(e, max),
            Expr::Bin { l, r, .. } => {
                expr(l, max);
                expr(r, max);
            }
        }
    }
    fn block(stmts: &[Stmt], max: &mut usize) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, init, .. } => {
                    *max = (*max).max(*var as usize + 1);
                    expr(init, max);
                }
                Stmt::Assign { var, value } => {
                    *max = (*max).max(*var as usize + 1);
                    expr(value, max);
                }
                Stmt::If { cond, then, els } => {
                    expr(cond, max);
                    block(then, max);
                    block(els, max);
                }
                Stmt::For { var, body, .. } => {
                    *max = (*max).max(*var as usize + 1);
                    block(body, max);
                }
            }
        }
    }
    let mut max = 0;
    block(&program.body, &mut max);
    expr(&program.result, &mut max);
    max
}

/// Runs the kernel body for every element `0..count` and returns the output buffer's bit patterns (`bool` results
/// as 0 or 1), or the first domain violation (lowest element, first violating evaluation in Rust's order).
pub fn run(program: &Program) -> Result<Vec<u32>, Failure> {
    let mut machine = Machine {
        program,
        element: 0,
        vars: vec![0; variable_slots(program)],
    };
    let mut out = Vec::with_capacity(program.count as usize);
    for element in 0..program.count {
        machine.element = element;
        let value = machine
            .block(&program.body)
            .and_then(|()| machine.expr(&program.result))
            .map_err(|violation| Failure { element, violation })?;
        out.push(value);
    }
    Ok(out)
}

/// The first domain violation in a closed sub-expression (one made of literals only) anywhere in the program,
/// dead code included, or `None`. Closed sub-expressions are folded eagerly, without short-circuiting, with the same
/// `binary` and `unary` as the interpreter. A shift by a literal 32 or more and a division by a literal zero count
/// even when the other operand is not closed. Such a constant is a compile-time error in some shader languages
/// (const-expression rules), so the generator keeps none, in executed or dead code (labelled `static`).
pub fn static_violation(program: &Program) -> Option<Violation> {
    fn fold(e: &Expr) -> Result<Option<u32>, Violation> {
        match e {
            Expr::Lit(_, bits) => Ok(Some(*bits)),
            Expr::Var(..) | Expr::Gid | Expr::Len(_) => Ok(None),
            Expr::Input { index, .. } => fold(index).map(|_| None),
            Expr::Un { op, ty, e } => fold(e)?.map(|v| unary(*op, *ty, v)).transpose(),
            Expr::Bin { op, ty, l, r } => {
                let (left, right) = (fold(l)?, fold(r)?);
                match (op, right) {
                    (BinOp::Div | BinOp::Rem, Some(0)) => return Err(Violation::DivisionByZero),
                    (BinOp::Shl | BinOp::Shr, Some(amount)) if amount >= 32 => {
                        return Err(Violation::ShiftAmount);
                    }
                    _ => {}
                }
                match (left, right) {
                    (Some(l), Some(r)) => binary(*op, *ty, l, r).map(Some),
                    _ => Ok(None),
                }
            }
            Expr::Cast { e, .. } => fold(e),
        }
    }
    let mut found = None;
    // Every node folds its own subtree: quadratic in depth, but programs are small.
    crate::ast::visit_exprs(program, &mut |e| {
        if found.is_none() {
            found = fold(e).err();
        }
    });
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
}
