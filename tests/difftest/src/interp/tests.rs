//! Unit tests of the interpreter.

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
