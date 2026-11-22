//! Tests of the math programs (format 14, slice m11.dt2): the interpreter's rules for `rem_euclid`, the `f32` math
//! methods and module constants on hand-built programs (written from the Rust standard library and `docs/SEMANTICS.md`,
//! not from the emitted Slang), the emitted text and its re-parse, and the properties of the generated math space.

use std::collections::BTreeSet;

use crate::ast::{
    BinOp, ConstDef, Expr, InputBuf, MathFn, Methods, Program, Stmt, Ty, operator_counts,
};
use crate::emit::emit_module;
use crate::generator::{
    MATH_ATTEMPT_BASE, MATH_PROGRAMS_PER_SEED, PROGRAMS_PER_SEED, Space, TRAIT_END,
    generate_attempt, generate_batch, generate_batch_in, space_of_attempt,
};
use crate::interp::{Violation, const_value, rem_euclid, run_covered, run_full, static_violation};
use crate::reparse::reparse;

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn i(v: i32) -> Expr {
    Expr::i32(v)
}

fn f(v: f32) -> Expr {
    Expr::f32(v)
}

fn math(f: MathFn, ty: Ty, args: Vec<Expr>) -> Expr {
    Expr::Math { f, ty, args }
}

fn program_of(ty: Ty, data: &[u32], consts: Vec<ConstDef>, result: Expr) -> Program {
    Program {
        seed: 0,
        attempt: MATH_ATTEMPT_BASE,
        count: data.len() as u32,
        inputs: vec![InputBuf {
            ty,
            data: data.to_vec(),
        }],
        body: Vec::new(),
        result,
        methods: Methods {
            consts,
            ..Methods::default()
        },
        store: None,
    }
}

/// The output of an integer expression for one element (the input buffer holds one `u32`).
fn int_result(result: Expr) -> Result<u32, Violation> {
    let program = program_of(Ty::U32, &[1], Vec::new(), result);
    program.check().unwrap();
    run_full(&program)
        .map(|run| run.values[0])
        .map_err(|failure| failure.violation)
}

/// The reference value and the interval of an `f32` expression for one element (the input buffer holds `1.0`).
fn float_result(result: Expr) -> Result<(f32, (f64, f64)), Violation> {
    let program = program_of(Ty::F32, &[1.0f32.to_bits()], Vec::new(), result);
    program.check().unwrap();
    run_full(&program)
        .map(|run| (f32::from_bits(run.values[0]), run.intervals[0]))
        .map_err(|failure| failure.violation)
}

/// `1.0` read from the input, so that no operand is a closed constant.
fn one() -> Expr {
    Expr::input(0, Ty::F32, Expr::Gid)
}

fn rem(ty: Ty, l: Expr, r: Expr) -> Expr {
    math(MathFn::RemEuclid, ty, vec![l, r])
}

#[test]
fn rem_euclid_is_the_non_negative_remainder_and_not_percent() {
    // u32: the same as `%`.
    assert_eq!(int_result(rem(Ty::U32, u(7), u(3))), Ok(1));
    assert_eq!(int_result(rem(Ty::U32, u(u32::MAX), u(7))), Ok(3));
    assert_eq!(int_result(rem(Ty::U32, u(5), u(u32::MAX))), Ok(5));
    // i32: the result is never negative (the divisor's sign does not matter), where `%` follows the dividend.
    let signed = |a: i32, b: i32| int_result(rem(Ty::I32, i(a), i(b))).map(|bits| bits as i32);
    assert_eq!(signed(-7, 2), Ok(1));
    assert_eq!(signed(7, -2), Ok(1));
    assert_eq!(signed(-7, -2), Ok(1));
    assert_eq!(signed(-7, 3), Ok(2));
    assert_eq!(signed(7, 3), Ok(1));
    assert_eq!(signed(i32::MAX, -1), Ok(0));
    assert_eq!(signed(i32::MIN + 1, i32::MIN + 1), Ok(0));
    assert_eq!(signed(-1, i32::MAX), Ok(i32::MAX - 1));
    // `%` of the same operands differs: this is what the method adds.
    let percent = int_result(Expr::bin(BinOp::Rem, i(-7), i(2))).map(|bits| bits as i32);
    assert_eq!(percent, Ok(-1));
    // `i32::MIN` has no literal: it is `-2147483647 - 1`.
    assert_eq!(int_result(rem(Ty::I32, Expr::i32_min(), i(2))), Ok(0));
    assert_eq!(
        int_result(rem(Ty::I32, Expr::i32_min(), i(i32::MAX))).map(|bits| bits as i32),
        Ok(i32::MAX - 1)
    );
}

#[test]
fn rem_euclid_leaves_the_domain_where_rust_panics() {
    let zero = Expr::bin(BinOp::Sub, Expr::Gid, Expr::Gid);
    assert_eq!(
        int_result(rem(Ty::U32, u(7), zero)),
        Err(Violation::DivisionByZero)
    );
    assert_eq!(
        int_result(rem(Ty::I32, Expr::i32_min(), i(-1))),
        Err(Violation::MinOverMinusOne)
    );
    assert_eq!(
        rem_euclid(Ty::I32, i32::MIN as u32, (-1i32) as u32),
        Err(Violation::MinOverMinusOne)
    );
    assert_eq!(rem_euclid(Ty::U32, 1, 0), Err(Violation::DivisionByZero));
    // A literal zero divisor is a static violation, as it is for `%`.
    let program = program_of(Ty::U32, &[1], Vec::new(), rem(Ty::U32, Expr::Gid, u(0)));
    assert_eq!(static_violation(&program), Some(Violation::DivisionByZero));
    let fine = program_of(Ty::U32, &[1], Vec::new(), rem(Ty::U32, Expr::Gid, u(3)));
    assert_eq!(static_violation(&fine), None);
}

#[test]
fn mul_add_is_fused_and_its_interval_holds_the_unfused_result_too() {
    // a = b = 1 + 2^-12: a * b = 1 + 2^-11 + 2^-24, which `f32` rounds to 1 + 2^-11 (a tie, to even). With c = -(1 +
    // 2^-8) the results differ by 2^-24, and the sum stays far enough from zero for the interval to be bounded.
    let a = 1.0f32 + 2f32.powi(-12);
    let c = -(1.0f32 + 2f32.powi(-8));
    let fused = a.mul_add(a, c);
    let unfused = a * a + c;
    assert_ne!(fused, unfused);
    let (reference, (lo, hi)) =
        float_result(math(MathFn::MulAdd, Ty::F32, vec![f(a), f(a), f(c)])).unwrap();
    assert_eq!(reference.to_bits(), fused.to_bits());
    assert!(lo <= f64::from(unfused) && f64::from(unfused) <= hi);
    assert!(lo <= f64::from(fused) && f64::from(fused) <= hi);
}

#[test]
fn the_exact_float_methods_have_the_std_reference_and_the_image_of_the_interval() {
    let on = |m: MathFn, args: Vec<Expr>| float_result(math(m, Ty::F32, args)).unwrap();
    let exact = |value: f32, got: (f32, (f64, f64))| {
        assert_eq!(got.0.to_bits(), value.to_bits(), "{value}");
        assert_eq!(got.1, (f64::from(value), f64::from(value)), "{value}");
    };
    exact(2.0, on(MathFn::Floor, vec![f(2.5)]));
    exact(-3.0, on(MathFn::Floor, vec![f(-2.5)]));
    exact(3.0, on(MathFn::Ceil, vec![f(2.5)]));
    exact(-2.0, on(MathFn::Ceil, vec![f(-2.5)]));
    exact(-2.0, on(MathFn::Trunc, vec![f(-2.5)]));
    exact(2.0, on(MathFn::Trunc, vec![f(2.5)]));
    exact(3.5, on(MathFn::Abs, vec![f(-3.5)]));
    exact(3.5, on(MathFn::Abs, vec![f(3.5)]));
    exact(1.0, on(MathFn::Min, vec![f(1.0), f(2.0)]));
    exact(-2.0, on(MathFn::Min, vec![f(1.0), f(-2.0)]));
    exact(2.0, on(MathFn::Max, vec![f(1.0), f(2.0)]));
    exact(1.0, on(MathFn::Max, vec![f(1.0), f(-2.0)]));
    exact(2.0, on(MathFn::Clamp, vec![f(5.0), f(-1.0), f(2.0)]));
    exact(-1.0, on(MathFn::Clamp, vec![f(-5.0), f(-1.0), f(2.0)]));
    exact(0.5, on(MathFn::Clamp, vec![f(0.5), f(-1.0), f(2.0)]));
    // A clamp whose bounds are not ordered is outside the domain (Rust panics, WGSL leaves it unspecified).
    assert_eq!(
        float_result(math(MathFn::Clamp, Ty::F32, vec![f(0.5), f(2.0), f(-1.0)])),
        Err(Violation::ComparisonOverlap)
    );
    // Equal bounds are ordered.
    exact(2.0, on(MathFn::Clamp, vec![f(0.5), f(2.0), f(2.0)]));
}

#[test]
fn a_floor_of_a_value_that_may_sit_on_either_side_of_an_integer_is_outside_the_float_domain() {
    // `1.0 / 3.0 * 3.0` may be a hair below or above 1 on a GPU: floor is 0 or 1, an interval that holds zero.
    let third = Expr::bin(BinOp::Div, one(), f(3.0));
    let almost_one = Expr::bin(BinOp::Mul, third, f(3.0));
    assert_eq!(
        float_result(math(MathFn::Floor, Ty::F32, vec![almost_one])),
        Err(Violation::FloatDomain)
    );
    // The same value below an integer by more than the interval is fine: floor of 2.9999 + 0.0001 stays 2 on every
    // party.
    let (reference, interval) = float_result(math(
        MathFn::Floor,
        Ty::F32,
        vec![Expr::bin(BinOp::Mul, one(), f(2.5))],
    ))
    .unwrap();
    assert_eq!(reference, 2.0);
    assert_eq!(interval, (2.0, 2.0));
}

#[test]
fn a_module_constant_is_its_initializer_evaluated_with_checked_arithmetic() {
    let consts = vec![
        ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Shl, u(3), u(30)),
        },
        ConstDef {
            ty: Ty::I32,
            init: Expr::bin(BinOp::Sub, i(-5), i(10)),
        },
        ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Rem, Expr::Const { k: 0, ty: Ty::U32 }, u(1000)),
        },
        ConstDef {
            ty: Ty::I32,
            init: Expr::negated(Expr::Const { k: 1, ty: Ty::I32 }),
        },
        ConstDef {
            ty: Ty::U32,
            init: Expr::inverted(Expr::Const { k: 2, ty: Ty::U32 }),
        },
    ];
    let methods = Methods {
        consts,
        ..Methods::default()
    };
    methods.check_tables().unwrap();
    assert_eq!(const_value(&methods, 0), Ok(3 << 30));
    assert_eq!(const_value(&methods, 1), Ok((-15i32) as u32));
    assert_eq!(const_value(&methods, 2), Ok((3u32 << 30) % 1000));
    assert_eq!(const_value(&methods, 3), Ok(15));
    assert_eq!(const_value(&methods, 4), Ok(!((3u32 << 30) % 1000)));
    // A kernel reads the value, and a constant of the program is an operand like any other.
    let program = program_of(
        Ty::U32,
        &[0, 0],
        methods.consts.clone(),
        Expr::bin(BinOp::Add, Expr::Const { k: 2, ty: Ty::U32 }, Expr::Gid),
    );
    program.check().unwrap();
    let run = run_full(&program).unwrap();
    assert_eq!(run.values, [(3u32 << 30) % 1000, (3u32 << 30) % 1000 + 1]);
}

#[test]
fn the_tables_refuse_a_constant_that_rust_would_refuse() {
    let refused = |consts: Vec<ConstDef>| {
        Methods {
            consts,
            ..Methods::default()
        }
        .check_tables()
        .unwrap_err()
    };
    let k = |k: u8, ty: Ty| Expr::Const { k, ty };
    // Overflow, a zero divisor and a shift by 32 or more are errors of the item.
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Mul, u(1 << 20), u(1 << 20)),
        }])
        .contains("does not evaluate: overflow")
    );
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Div, u(1), u(0)),
        }])
        .contains("division-by-zero")
    );
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Shl, u(1), u(32)),
        }])
        .contains("shift-amount")
    );
    // A constant below it only, of its own type, no cast, no comparison, no negation of a `u32`.
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: k(0, Ty::U32),
        }])
        .contains("not a constant expression")
    );
    assert!(
        refused(vec![
            ConstDef {
                ty: Ty::I32,
                init: i(1),
            },
            ConstDef {
                ty: Ty::U32,
                init: Expr::cast(Ty::U32, k(0, Ty::I32)),
            },
        ])
        .contains("not a constant expression")
    );
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Lt, u(1), u(2)),
        }])
        .contains("K0")
    );
    assert!(
        refused(vec![ConstDef {
            ty: Ty::U32,
            init: Expr::Un {
                op: crate::ast::UnOp::Neg,
                ty: Ty::U32,
                e: Box::new(u(1)),
            },
        }])
        .contains("`-` on a u32")
    );
    assert!(
        refused(vec![ConstDef {
            ty: Ty::F32,
            init: f(1.0),
        }])
        .contains("not an integer")
    );
}

/// A program of constants and math methods, as text: the shape of the items, the call syntax, and the round trip.
fn text_program() -> Program {
    let consts = vec![
        ConstDef {
            ty: Ty::U32,
            init: Expr::bin(BinOp::Add, u(3), u(4)),
        },
        ConstDef {
            ty: Ty::I32,
            init: Expr::negated(Expr::bin(BinOp::Mul, i(5), i(6))),
        },
        ConstDef {
            ty: Ty::U32,
            init: Expr::bin(
                BinOp::Shl,
                Expr::Const { k: 0, ty: Ty::U32 },
                Expr::Const { k: 0, ty: Ty::U32 },
            ),
        },
    ];
    let a = Expr::input(0, Ty::U32, Expr::Gid);
    let signed = Expr::cast(Ty::I32, a.clone());
    let result = Expr::bin(
        BinOp::BitXor,
        rem(Ty::U32, a, Expr::Const { k: 0, ty: Ty::U32 }),
        Expr::cast(
            Ty::U32,
            rem(Ty::I32, signed, Expr::Const { k: 1, ty: Ty::I32 }),
        ),
    );
    program_of(Ty::U32, &[1, 2, 3], consts, result)
}

#[test]
fn the_text_of_constants_and_methods_reads_back_to_the_same_program() {
    let program = text_program();
    program.check().unwrap();
    let text = emit_module(&program);
    for needle in [
        "    const K0: u32 = 3u32 + 4u32;\n",
        "    const K1: i32 = -(5i32 * 6i32);\n",
        "    const K2: u32 = K0 << K0;\n",
        ".rem_euclid(K0)",
        ".rem_euclid(K1)",
    ] {
        assert!(text.contains(needle), "{needle} is not in\n{text}");
    }
    let (body, result) = reparse(&text, &program.inputs).unwrap();
    assert_eq!(body, program.body, "{text}");
    assert_eq!(result, program.result, "{text}");
}

#[test]
fn a_literal_receiver_is_parenthesised_so_a_negation_reads_back_as_a_negation() {
    // `-(1.5f32).abs()` is `-(1.5.abs())`; written `-1.5f32.abs()` the re-parser (and a reader) would take `-1.5f32` as
    // the receiver. The program is `-(1.5.abs())`: a negation of the method call, not a call on a negative literal.
    let result = Expr::negated(math(MathFn::Abs, Ty::F32, vec![f(1.5)]));
    let program = program_of(Ty::F32, &[1.0f32.to_bits()], Vec::new(), result);
    program.check().unwrap();
    let text = emit_module(&program);
    assert!(text.contains("-(1.5f32).abs()"), "{text}");
    let (_, back) = reparse(&text, &program.inputs).unwrap();
    assert_eq!(back, program.result);
    let run = run_full(&program).unwrap();
    assert_eq!(f32::from_bits(run.values[0]), -1.5);
    // A call on a negative literal is also a call, parenthesised as an operand.
    let call = math(MathFn::Abs, Ty::F32, vec![f(-1.5)]);
    let program = program_of(Ty::F32, &[1.0f32.to_bits()], Vec::new(), call);
    let text = emit_module(&program);
    assert!(text.contains("(-1.5f32).abs()"), "{text}");
    let (_, back) = reparse(&text, &program.inputs).unwrap();
    assert_eq!(back, program.result);
}

#[test]
fn the_checker_refuses_ill_typed_math() {
    let bad = |result: Expr| {
        program_of(Ty::F32, &[0], Vec::new(), result)
            .check()
            .unwrap_err()
    };
    assert!(bad(math(MathFn::RemEuclid, Ty::F32, vec![f(1.0), f(2.0)])).contains("rem_euclid"));
    assert!(bad(math(MathFn::MulAdd, Ty::F32, vec![f(1.0), f(2.0)])).contains("needs 3"));
    assert!(bad(math(MathFn::Floor, Ty::U32, vec![u(1)])).contains("floor"));
    assert!(bad(math(MathFn::Abs, Ty::F32, vec![u(1)])).contains("abs"));
    assert!(bad(Expr::Const { k: 0, ty: Ty::U32 }).contains("does not exist"));
}

#[test]
fn the_operator_counts_name_the_methods_and_the_constants() {
    let program = text_program();
    let counts = operator_counts(&program);
    assert_eq!(counts.get(".rem_euclid u32"), Some(&1));
    assert_eq!(counts.get(".rem_euclid i32"), Some(&1));
    assert_eq!(counts.get("const u32"), Some(&1));
    assert_eq!(counts.get("const i32"), Some(&1));
}

#[test]
fn the_math_space_is_selected_by_its_attempt_numbers_and_ends_every_batch() {
    assert_eq!(space_of_attempt(MATH_ATTEMPT_BASE - 1), Space::Trait);
    assert_eq!(space_of_attempt(MATH_ATTEMPT_BASE), Space::Math);
    assert_eq!(MATH_PROGRAMS_PER_SEED, 2);
    assert_eq!(
        generate_attempt(3, MATH_ATTEMPT_BASE),
        generate_attempt(3, MATH_ATTEMPT_BASE)
    );
    assert_ne!(
        generate_attempt(3, MATH_ATTEMPT_BASE),
        generate_attempt(3, MATH_ATTEMPT_BASE + 1)
    );
    for seed in 1..=8 {
        let batch = generate_batch(seed).unwrap();
        assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
        let math_start = TRAIT_END;
        assert_eq!(math_start + MATH_PROGRAMS_PER_SEED, PROGRAMS_PER_SEED);
        for program in &batch.programs[math_start..] {
            assert!(program.attempt >= MATH_ATTEMPT_BASE, "seed {seed}");
            assert!(program.has_math_forms(), "seed {seed}");
        }
        for program in &batch.programs[..math_start] {
            assert!(program.attempt < MATH_ATTEMPT_BASE, "seed {seed}");
            assert!(program.methods.consts.is_empty(), "seed {seed}");
            assert!(
                !operator_counts(program)
                    .keys()
                    .any(|name| name.starts_with('.') || name.starts_with("const ")),
                "seed {seed}"
            );
        }
    }
}

/// What the math space reaches over a few seeds: every method on the types it is written for, both constant types, and
/// constants built from other constants and from operators.
#[test]
fn the_math_space_reaches_every_method_and_constant_form() {
    let mut names = BTreeSet::new();
    let mut const_types = BTreeSet::new();
    let (mut chained, mut operators) = (false, false);
    for seed in 1..=8 {
        let batch = generate_batch_in(Space::Math, seed).unwrap();
        for program in &batch.programs {
            program.check().unwrap();
            assert!(program.has_math_forms());
            assert!(!program.methods.consts.is_empty() && program.methods.consts.len() <= 3);
            for (k, def) in program.methods.consts.iter().enumerate() {
                const_types.insert(def.ty);
                let mut uses = false;
                let mut ops = false;
                def.init.for_each_child(&mut |_| ops = true);
                uses |= format!("{:?}", def.init).contains("Const");
                chained |= uses && k > 0;
                operators |= ops;
            }
            names.extend(operator_counts(program).into_keys());
        }
    }
    for f in MathFn::ALL {
        let ty = if f.is_float() { Ty::F32 } else { Ty::U32 };
        assert!(names.contains(&f.counted_name(ty)), "{}", f.name());
    }
    assert!(names.contains(".rem_euclid i32"));
    assert!(names.contains("const u32") && names.contains("const i32"));
    assert_eq!(const_types, BTreeSet::from([Ty::U32, Ty::I32]));
    assert!(chained && operators);
}

/// The 16 math programs of the batch of seeds 1 to 8 are the ones the GPU runs: every math method must execute in them
/// (the interpreter's coverage counts evaluations, not the written forms that `operator_counts` sees).
#[test]
fn the_batch_math_programs_of_seeds_1_to_8_execute_every_math_method() {
    let mut executed = [0u64; MathFn::ALL.len()];
    let mut programs = 0;
    for seed in 1..=8 {
        for program in &generate_batch(seed).unwrap().programs[TRAIT_END..] {
            programs += 1;
            let (_, coverage) = run_covered(program).unwrap();
            for (k, f) in MathFn::ALL.iter().enumerate() {
                executed[k] += [Ty::U32, Ty::I32, Ty::F32]
                    .iter()
                    .filter_map(|ty| coverage.operators.get(&f.counted_name(*ty)))
                    .sum::<u64>();
            }
        }
    }
    assert_eq!(programs, 8 * MATH_PROGRAMS_PER_SEED);
    for (f, n) in MathFn::ALL.iter().zip(executed) {
        assert!(
            n > 0,
            "`{}` never executes in the batch ({executed:?})",
            f.name()
        );
    }
}

/// A module constant folds like a literal in the static check: a shift by `K0` of 33, or a constant whose initializer
/// overflows, in a branch that no element runs, is out of the domain.
#[test]
fn a_constant_in_a_branch_that_never_runs_is_a_static_violation_like_a_literal() {
    let const_u32 = |init: Expr| ConstDef { ty: Ty::U32, init };
    let k0 = || Expr::Const { k: 0, ty: Ty::U32 };
    let dead = |init: Expr| {
        vec![Stmt::If {
            cond: Expr::bin(BinOp::Gt, Expr::Gid, u(100)),
            then: vec![Stmt::Let {
                var: 0,
                ty: Ty::U32,
                mutable: false,
                init,
            }],
            els: Vec::new(),
        }]
    };
    let shift = program_with_body(
        vec![const_u32(u(33))],
        dead(Expr::bin(BinOp::Shl, Expr::Gid, k0())),
    );
    assert_eq!(run_full(&shift).map(|run| run.values[0]).ok(), Some(0));
    assert_eq!(static_violation(&shift), Some(Violation::ShiftAmount));
    let overflow = program_with_body(
        vec![const_u32(Expr::bin(BinOp::Add, u(u32::MAX), u(1)))],
        dead(Expr::bin(BinOp::Add, Expr::Gid, k0())),
    );
    assert_eq!(static_violation(&overflow), Some(Violation::Overflow));
    let fine = program_with_body(
        vec![const_u32(u(31))],
        dead(Expr::bin(BinOp::Shl, Expr::Gid, k0())),
    );
    assert_eq!(static_violation(&fine), None);
}

/// The program of `program_of` with a body (the result is the element index).
fn program_with_body(consts: Vec<ConstDef>, body: Vec<Stmt>) -> Program {
    Program {
        body,
        ..program_of(Ty::U32, &[1], consts, Expr::Gid)
    }
}
