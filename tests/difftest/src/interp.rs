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

use crate::ast::{BinOp, Expr, Program, Stmt, Ty, UnOp};
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

/// A value of the machine: an integer or `bool` pattern, or an `f32` with its reference value and its interval.
#[derive(Clone, Copy, Debug)]
enum Val {
    I(u32),
    F(f32, Fl),
}

impl Val {
    fn int(self) -> u32 {
        match self {
            Val::I(bits) => bits,
            Val::F(..) => unreachable!("a float where the program is typed to hold an integer"),
        }
    }

    fn float(self) -> (f32, Fl) {
        match self {
            Val::F(reference, fl) => (reference, fl),
            Val::I(_) => unreachable!("an integer where the program is typed to hold a float"),
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
fn float_binary(op: BinOp, a: (f32, Fl), b: (f32, Fl)) -> Result<Val, Violation> {
    let (fa, fb) = (a.1, b.1);
    let (reference, fl) = match op {
        BinOp::Add => (a.0 + b.0, fa.plus(fb)),
        BinOp::Sub => (a.0 - b.0, fa.minus(fb)),
        BinOp::Mul => (a.0 * b.0, fa.times(fb)),
        BinOp::Div => (a.0 / b.0, fa.over(fb)?),
        other => unreachable!("float operator {}", other.symbol()),
    };
    Ok(checked_value(reference, in_domain(fl)?))
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

struct Machine<'a> {
    program: &'a Program,
    element: u32,
    vars: Vec<Val>,
    /// Executed-operator counts, kept only by `run_covered`.
    coverage: Option<Box<Counting>>,
    /// The first value of every expression node (by address), kept only by `observed_values`.
    observed: Option<HashMap<usize, u32>>,
}

/// What `run_covered` counts while it runs.
#[derive(Default)]
struct Counting {
    /// (kind, head, tail) -> evaluations; see `operator_name`.
    operators: BTreeMap<(u8, &'static str, &'static str), u64>,
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

    /// Evaluates `expr` and, for `observed_values`, remembers the first value each node takes.
    fn expr(&mut self, expr: &Expr) -> Result<Val, Violation> {
        let value = self.eval(expr)?;
        if let Some(seen) = &mut self.observed {
            let bits = match value {
                Val::I(bits) => bits,
                Val::F(reference, _) => reference.to_bits(),
            };
            seen.entry(std::ptr::from_ref(expr) as usize)
                .or_insert(bits);
        }
        Ok(value)
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
                float_binary(*op, left, right)
            }
            Expr::Bin { op, ty, l, r } => {
                self.count(KIND_BINARY, op.symbol(), ty.name());
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
                    for k in 0..*trips {
                        self.vars[*var as usize] = Val::I(k);
                        self.block(body)?;
                    }
                }
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
                Stmt::Let { var, .. } | Stmt::Assign { var, .. } | Stmt::For { var, .. } => {
                    *max = (*max).max(*var as usize + 1);
                }
                Stmt::If { .. } => {}
            }
            match stmt {
                Stmt::If { then, els, .. } => {
                    block(then, max);
                    block(els, max);
                }
                Stmt::For { body, .. } => block(body, max),
                _ => {}
            }
        }
    }
    block(&program.body, &mut max);
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
    run_machine(program, false).map(|(run, _)| run)
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

/// `run_full` that also counts what was executed.
pub fn run_covered(program: &Program) -> Result<(Run, Coverage), Failure> {
    run_machine(program, true).map(|(run, coverage)| (run, coverage.unwrap_or_default()))
}

fn count_ifs(stmts: &[Stmt]) -> u64 {
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::If { then, els, .. } => 1 + count_ifs(then) + count_ifs(els),
            Stmt::For { body, .. } => count_ifs(body),
            _ => 0,
        })
        .sum()
}

fn run_machine(program: &Program, covered: bool) -> Result<(Run, Option<Coverage>), Failure> {
    let mut machine = Machine {
        program,
        element: 0,
        vars: vec![Val::I(0); variable_slots(program)],
        coverage: covered.then(Box::default),
        observed: None,
    };
    let mut values = Vec::with_capacity(program.count as usize);
    let mut intervals = Vec::new();
    for element in 0..program.count {
        machine.element = element;
        let result = machine
            .block(&program.body)
            .and_then(|()| machine.expr(&program.result))
            .and_then(|value| match value {
                Val::I(bits) => Ok((bits, None)),
                Val::F(reference, fl) => {
                    let (lo, hi) = fl.interval();
                    if hi - lo > float::MAX_RELATIVE_WIDTH * lo.abs().max(hi.abs()) {
                        return Err(Violation::Unboundable);
                    }
                    Ok((reference.to_bits(), Some((lo, hi))))
                }
            })
            .map_err(|violation| Failure { element, violation })?;
        values.push(result.0);
        intervals.extend(result.1);
    }
    let coverage = machine.coverage.map(|counting| {
        let mut coverage = Coverage {
            if_statements: count_ifs(&program.body),
            rhs_evaluated: counting.rhs_evaluated,
            rhs_skipped: counting.rhs_skipped,
            ..Coverage::default()
        };
        for ((kind, head, tail), n) in counting.operators {
            coverage
                .operators
                .insert(operator_name(kind, head, tail), n);
        }
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
        coverage: None,
        observed: Some(HashMap::new()),
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
#[derive(Clone, Copy)]
enum Const {
    I(u32),
    F(f32),
}

impl Const {
    fn int(self) -> u32 {
        match self {
            Const::I(bits) => bits,
            Const::F(_) => unreachable!("typed program"),
        }
    }

    fn float(self) -> f32 {
        match self {
            Const::F(value) => value,
            Const::I(_) => unreachable!("typed program"),
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
            Expr::Var(..) | Expr::Gid | Expr::Len(_) => Ok(None),
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
                    assert!(
                        written.contains_key(name),
                        "{name} executed but not written"
                    );
                }
            }
        }
    }
}
