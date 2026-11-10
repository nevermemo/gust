//! Tests of the enum programs (format 9, slice m9.d1b): the interpreter's rules for `match`, `==`, `as` and `return` on
//! hand-built programs (written from the Rust Reference, not from the emitted Slang), the emitted text, the mutation
//! checks (the vectors must catch a wrong `match`), and the properties of the generated enum space.

use std::collections::BTreeMap;

use crate::ast::{
    self, Arm, BinOp, EnumDef, Expr, FnDef, InputBuf, Methods, Pat, Place, Program, Recv, Stmt,
    StructDef, Ty, operator_counts,
};
use crate::emit::emit_module;
use crate::generator::{
    ENUM_ATTEMPT_BASE, ENUM_PROGRAMS_PER_SEED, METHOD_ATTEMPT_BASE, METHOD_END, PROGRAMS_PER_SEED,
    Space, VECTOR_END, generate_batch, generate_batch_in, space_of_attempt,
};
use crate::interp::{Mutant, env_mutant, run, run_covered, run_mutant, static_violation};

const S0: Ty = Ty::Struct(0);
const E0: Ty = Ty::Enum(0);

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn variant(e: u8, v: u8) -> Expr {
    Expr::Variant { e, v }
}

fn local() -> Place {
    Place::var(0, S0)
}

fn n_of(place: &Place) -> Expr {
    Expr::Read {
        place: place.field(0),
        ty: Ty::U32,
    }
}

fn call(f: usize, recv: Option<Place>, args: Vec<Expr>, ty: Ty) -> Expr {
    Expr::Call {
        f,
        owner: None,
        recv,
        args,
        ty,
    }
}

fn add_to_n(by: u32) -> Stmt {
    Stmt::Set {
        place: local().field(0),
        ty: Ty::U32,
        op: Some(BinOp::Add),
        value: u(by),
    }
}

/// `E0` (`u32`): V0 = 0 (the count starts at zero), V1 = 3, V2 = 4 (counts up), V3 = 10. `E1` (`i32`): V0 = -2,
/// V1 = -1 (counts up), V2 = 5.
fn enums() -> Vec<EnumDef> {
    vec![
        EnumDef {
            repr: Ty::U32,
            variants: vec![(0, false), (3, true), (4, false), (10, true)],
        },
        EnumDef {
            repr: Ty::I32,
            variants: vec![((-2i32) as u32, true), ((-1i32) as u32, false), (5, true)],
        },
    ]
}

/// The functions of the hand programs, over `S0 { n: u32 }`:
/// m0 `&mut self` `tick()`: `self.n += 1; self.n`; m1 helper `pick(v) -> E0`: `if v == 6 { return V1 } V0`;
/// m2 `&self` `kind_of() -> E0`: `pick(self.n)`; m3 helper `two(a, b) -> u32`: `a as u32 * 16 + b as u32`;
/// m4 helper `first(e) -> u32`: `for _ in 0..3 { match e { V1 => { return 7 } _ => {} } } 9`.
fn methods() -> Methods {
    let this = Place::this(S0);
    Methods {
        structs: vec![StructDef {
            fields: vec![Ty::U32],
        }],
        fns: vec![
            FnDef {
                owner: Some(0),
                recv: Recv::Mut,
                params: Vec::new(),
                ret: Some(Ty::U32),
                body: vec![Stmt::Set {
                    place: this.field(0),
                    ty: Ty::U32,
                    op: Some(BinOp::Add),
                    value: u(1),
                }],
                tail: Some(n_of(&this)),
            },
            FnDef {
                owner: None,
                recv: Recv::None,
                params: vec![(20, Ty::U32)],
                ret: Some(E0),
                body: vec![Stmt::If {
                    cond: Expr::bin(BinOp::Eq, Expr::Var(20, Ty::U32), u(6)),
                    then: vec![Stmt::Return {
                        value: variant(0, 1),
                    }],
                    els: Vec::new(),
                }],
                tail: Some(variant(0, 0)),
            },
            FnDef {
                owner: Some(0),
                recv: Recv::Shared,
                params: Vec::new(),
                ret: Some(E0),
                body: Vec::new(),
                tail: Some(call(1, None, vec![n_of(&this)], E0)),
            },
            FnDef {
                owner: None,
                recv: Recv::None,
                params: vec![(21, E0), (22, E0)],
                ret: Some(Ty::U32),
                body: Vec::new(),
                tail: Some(Expr::bin(
                    BinOp::Add,
                    Expr::bin(
                        BinOp::Mul,
                        Expr::EnumAs {
                            to: Ty::U32,
                            e: Box::new(Expr::Var(21, E0)),
                        },
                        u(16),
                    ),
                    Expr::EnumAs {
                        to: Ty::U32,
                        e: Box::new(Expr::Var(22, E0)),
                    },
                )),
            },
            FnDef {
                owner: None,
                recv: Recv::None,
                params: vec![(23, E0)],
                ret: Some(Ty::U32),
                // `let mut k = 0; for _ in 0..3 { k += 1; match e { V1 => return k, _ => {} } } k + 100`:
                // an early exit gives 1, a loop that keeps running after the `return` gives 3.
                body: vec![
                    Stmt::Let {
                        var: 32,
                        ty: Ty::U32,
                        mutable: true,
                        init: u(0),
                    },
                    Stmt::For {
                        var: 31,
                        trips: 3,
                        body: vec![
                            Stmt::Assign {
                                var: 32,
                                value: Expr::bin(BinOp::Add, Expr::Var(32, Ty::U32), u(1)),
                            },
                            Stmt::Match {
                                scrut: Expr::Var(23, E0),
                                arms: vec![
                                    Arm {
                                        pat: Pat::Variants(vec![1]),
                                        body: vec![Stmt::Return {
                                            value: Expr::Var(32, Ty::U32),
                                        }],
                                    },
                                    Arm {
                                        pat: Pat::Wild,
                                        body: Vec::new(),
                                    },
                                ],
                            },
                        ],
                    },
                ],
                tail: Some(Expr::bin(BinOp::Add, Expr::Var(32, Ty::U32), u(100))),
            },
        ],
        enums: enums(),
    }
}

/// `let mut v0 = S0 { n: 5 }; <body>; out[i] = <result>` over one element.
fn program(mut body: Vec<Stmt>, result: Expr) -> Program {
    body.insert(
        0,
        Stmt::Let {
            var: 0,
            ty: S0,
            mutable: true,
            init: Expr::StructLit {
                ty: S0,
                fields: vec![u(5)],
                order: vec![0],
            },
        },
    );
    let program = Program {
        seed: 0,
        methods: methods(),
        attempt: ENUM_ATTEMPT_BASE,
        count: 1,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: vec![0],
        }],
        body,
        result,
    };
    program
        .check()
        .unwrap_or_else(|error| panic!("the hand program is refused: {error}"));
    program
}

/// The value of the program's first element. The hand vectors apply the mutant that `DIFFTEST_MUTANT` selects (the
/// `xfail-` runs); `run` is the reference and ignores it.
fn value_of(program: &Program) -> u32 {
    let values = match env_mutant() {
        Some(mutant) => run_mutant(program, mutant).map(|run| run.values),
        None => run(program),
    };
    values.unwrap_or_else(|f| panic!("{f:?}"))[0]
}

/// `match pick(c.tick()) { V0 => c.n += 1000, V1 => c.n += 100, _ => c.n += 10 }; out = c.n`.
fn scrutinee_program() -> Program {
    let scrut = call(
        1,
        None,
        vec![call(0, Some(local()), Vec::new(), Ty::U32)],
        E0,
    );
    let arm = |pat, by| Arm {
        pat,
        body: vec![add_to_n(by)],
    };
    program(
        vec![Stmt::Match {
            scrut,
            arms: vec![
                arm(Pat::Variants(vec![0]), 1000),
                arm(Pat::Variants(vec![1]), 100),
                arm(Pat::Wild, 10),
            ],
        }],
        n_of(&local()),
    )
}

/// A `match` on `E0::V<k>` whose arms assign 1, 2, 3 (or the like) to `v2`, read back as the result.
fn arms_program(k: u8, arms: Vec<(Pat, u32)>) -> Program {
    let arms = arms
        .into_iter()
        .map(|(pat, value)| Arm {
            pat,
            body: vec![Stmt::Assign {
                var: 2,
                value: u(value),
            }],
        })
        .collect();
    program(
        vec![
            Stmt::Let {
                var: 1,
                ty: E0,
                mutable: false,
                init: variant(0, k),
            },
            Stmt::Let {
                var: 2,
                ty: Ty::U32,
                mutable: true,
                init: u(0),
            },
            Stmt::Match {
                scrut: Expr::Var(1, E0),
                arms,
            },
        ],
        Expr::Var(2, Ty::U32),
    )
}

fn wild_arms() -> Vec<(Pat, u32)> {
    vec![
        (Pat::Variants(vec![0, 2]), 1),
        (Pat::Variants(vec![1]), 2),
        (Pat::Wild, 3),
    ]
}

fn default_arms() -> Vec<(Pat, u32)> {
    vec![
        (Pat::Variants(vec![0]), 1),
        (Pat::Variants(vec![1, 2, 3]), 2),
    ]
}

#[test]
fn the_scrutinee_is_evaluated_once_and_the_first_naming_arm_runs() {
    // `tick` makes n 6, `pick(6)` is V1, the second arm adds 100.
    assert_eq!(value_of(&scrutinee_program()), 106);
}

#[test]
fn grouped_wild_and_default_arms_take_the_variants_they_name() {
    for (k, expected) in [1, 2, 1, 3].into_iter().enumerate() {
        assert_eq!(
            value_of(&arms_program(k as u8, wild_arms())),
            expected,
            "wild, V{k}"
        );
    }
    // Without `_` every variant is named; the last arm (the Slang `default`) takes V1, V2 and V3.
    for (k, expected) in [1, 2, 2, 2].into_iter().enumerate() {
        assert_eq!(
            value_of(&arms_program(k as u8, default_arms())),
            expected,
            "default, V{k}"
        );
    }
}

#[test]
fn an_equality_evaluates_the_left_operand_first() {
    // `pick(c.tick()) == c.kind_of()`: left first, n is 6, V1 on both sides: true. Right first, `kind_of` would see 5.
    let left = call(
        1,
        None,
        vec![call(0, Some(local()), Vec::new(), Ty::U32)],
        E0,
    );
    let right = call(2, Some(local()), Vec::new(), E0);
    let equal = program(
        Vec::new(),
        Expr::bin(BinOp::Eq, left.clone(), right.clone()),
    );
    assert_eq!(value_of(&equal), 1);
    let different = program(Vec::new(), Expr::bin(BinOp::Ne, left, right));
    assert_eq!(value_of(&different), 0);
}

#[test]
fn enum_arguments_are_evaluated_left_to_right() {
    // `two(pick(c.tick()), pick(c.tick()))`: the first gets 6 (V1 = 3), the second 7 (V0 = 0): 3 * 16 + 0.
    let tick = || {
        call(
            1,
            None,
            vec![call(0, Some(local()), Vec::new(), Ty::U32)],
            E0,
        )
    };
    let p = program(Vec::new(), call(3, None, vec![tick(), tick()], Ty::U32));
    assert_eq!(value_of(&p), 48);
}

#[test]
fn a_return_in_an_arm_inside_a_for_ends_the_function() {
    let first = |k: u8| program(Vec::new(), call(4, None, vec![variant(0, k)], Ty::U32));
    assert_eq!(value_of(&first(1)), 1);
    assert_eq!(value_of(&first(0)), 103);
    assert_eq!(value_of(&first(3)), 103);
}

#[test]
fn as_reads_the_discriminant_and_a_signed_repr_keeps_its_sign() {
    let as_u32 = |v: u8| value_of(&program(Vec::new(), enum_as(Ty::U32, 0, v)));
    assert_eq!([as_u32(0), as_u32(1), as_u32(2), as_u32(3)], [0, 3, 4, 10]);
    let as_i32 = |v: u8| value_of(&program(Vec::new(), enum_as(Ty::I32, 1, v))) as i32;
    assert_eq!([as_i32(0), as_i32(1), as_i32(2)], [-2, -1, 5]);
}

fn enum_as(to: Ty, e: u8, v: u8) -> Expr {
    Expr::EnumAs {
        to,
        e: Box::new(variant(e, v)),
    }
}

#[test]
fn the_mutants_change_exactly_the_vectors_that_are_meant_to_catch_them() {
    // The reference (`run`) against each mutant, on the hand programs: the `xfail-` runs of the gate set the same
    // mutant for `value_of` and expect exactly the vector named here to fail.
    let reference = |p: &Program| run(p).unwrap()[0];
    let mutated = |p: &Program, m| run_mutant(p, m).unwrap().values[0];
    let scrutinee = scrutinee_program();
    assert_eq!(reference(&scrutinee), 106);
    assert_eq!(mutated(&scrutinee, Mutant::ScrutineeTwice), 107);
    assert_eq!(mutated(&scrutinee, Mutant::DefaultTakesFirstArm), 106);
    // Without `_` the last arm takes V1, V2 and V3; the mutant gives them to the first arm.
    let default_form = arms_program(3, default_arms());
    assert_eq!(reference(&default_form), 2);
    assert_eq!(mutated(&default_form, Mutant::DefaultTakesFirstArm), 1);
    assert_eq!(mutated(&default_form, Mutant::ScrutineeTwice), 2);
    // With `_` the mutant does nothing.
    let wild_form = arms_program(3, wild_arms());
    assert_eq!(reference(&wild_form), 3);
    assert_eq!(mutated(&wild_form, Mutant::DefaultTakesFirstArm), 3);
}

#[test]
fn the_emitted_text_of_the_hand_program_has_the_enum_forms() {
    let text = emit_module(&arms_program(1, wild_arms()));
    for wanted in [
        "#[derive(PartialEq, Eq)]\n    pub enum E0 {\n        V0,\n        V1 = 3,\n        V2,\n        V3 = 10,\n    }",
        "#[derive(PartialEq, Eq)]\n    #[repr(i32)]\n    pub enum E1 {\n        V0 = -2,\n        V1,\n        V2 = 5,\n    }",
        "let v1: E0 = E0::V1;",
        "match v1 {",
        "E0::V0 | E0::V2 => {",
        "E0::V1 => {",
        "_ => {",
        "#[allow(unreachable_code)]",
        "return E0::V1;",
    ] {
        assert!(text.contains(wanted), "no `{wanted}` in:\n{text}");
    }
    let text = emit_module(&program(
        Vec::new(),
        Expr::bin(
            BinOp::Eq,
            call(2, Some(local()), Vec::new(), E0),
            variant(0, 2),
        ),
    ));
    assert!(text.contains("v0.m2() == E0::V2"), "{text}");
    let text = emit_module(&program(Vec::new(), enum_as(Ty::I32, 1, 0)));
    assert!(text.contains("E1::V0 as i32"), "{text}");
}

#[test]
fn the_enum_space_is_selected_by_its_attempt_numbers_and_the_batch_cut_is_seven_five_one_three() {
    assert_eq!(space_of_attempt(METHOD_ATTEMPT_BASE), Space::Method);
    assert_eq!(space_of_attempt(ENUM_ATTEMPT_BASE - 1), Space::Method);
    assert_eq!(space_of_attempt(ENUM_ATTEMPT_BASE), Space::Enum);
    assert_eq!(ENUM_PROGRAMS_PER_SEED, 3);
    for seed in 1..=3 {
        let batch = generate_batch(seed).unwrap();
        assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
        for (k, program) in batch.programs.iter().enumerate() {
            let space = space_of_attempt(program.attempt);
            let expected = if k < 7 {
                Space::Float
            } else if k < VECTOR_END {
                Space::Vector
            } else if k < METHOD_END {
                Space::Method
            } else {
                Space::Enum
            };
            assert_eq!(space, expected, "seed {seed} program {k}");
        }
        assert!(
            batch.enum_attempts == 3 + batch.enum_discarded.values().sum::<u32>() as u64,
            "seed {seed}: {} attempts, {:?}",
            batch.enum_attempts,
            batch.enum_discarded
        );
    }
}

/// The batches of the enum space of the first seeds (16 programs each).
fn enum_programs(seeds: std::ops::RangeInclusive<u64>) -> Vec<Program> {
    seeds
        .flat_map(|seed| generate_batch_in(Space::Enum, seed).unwrap().programs)
        .collect()
}

#[test]
fn every_enum_program_is_well_typed_in_the_domain_and_matches_on_an_enum() {
    let (mut unsigned, mut signed) = (0, 0);
    for p in enum_programs(1..=12) {
        p.check().unwrap_or_else(|e| panic!("seed {} {e}", p.seed));
        assert!(run(&p).is_ok());
        assert!(static_violation(&p).is_none());
        assert!(p.has_call() && p.has_match());
        assert!(!p.methods.enums.is_empty());
        assert!(!p.has_float() && !p.has_aggregate());
        for def in &p.methods.enums {
            assert!((2..=ast::MAX_VARIANTS).contains(&def.variants.len()));
            if def.repr == Ty::I32 {
                signed += 1;
            } else {
                unsigned += 1;
            }
        }
    }
    assert!(
        unsigned > 0 && signed > 0,
        "{unsigned} u32 and {signed} i32 enums"
    );
}

/// The constructs and shapes the generator is biased toward: each must be written, and executed, within seeds 1 to 64
/// of the enum space, or the interpreter's rules have no program behind them.
#[test]
fn the_match_constructs_and_order_sensitive_shapes_are_reachable_within_the_first_64_seeds() {
    let mut written = BTreeMap::<String, usize>::new();
    let mut executed = BTreeMap::<String, u64>::new();
    for p in enum_programs(1..=64) {
        for (name, n) in operator_counts(&p) {
            *written.entry(name).or_default() += n;
        }
        let (_, coverage) = run_covered(&p).unwrap();
        for (name, n) in coverage.operators {
            *executed.entry(name).or_default() += n;
        }
    }
    for wanted in [
        ast::MATCH,
        ast::MATCH_WILD,
        ast::MATCH_DEFAULT,
        ast::MATCH_GROUPED,
        ast::MATCH_NESTED,
        ast::MATCH_IN_FOR,
        ast::MATCH_ARM_RETURN,
        ast::MATCH_ARM_CALL,
        ast::MATCH_ARM_SET,
        ast::SHAPE_MATCH_SCRUTINEE,
        ast::SHAPE_ENUM_EQ,
        ast::SHAPE_ENUM_ARG,
        ast::SHAPE_ENUM_ARGS_ROOT,
        "return",
        "enum as u32",
        "enum as i32",
    ] {
        assert!(
            written.get(wanted).copied().unwrap_or(0) > 0,
            "`{wanted}` is never written in seeds 1-64 of the enum space: {written:?}"
        );
        assert!(
            executed.get(wanted).copied().unwrap_or(0) > 0,
            "`{wanted}` is never executed in seeds 1-64 of the enum space: {executed:?}"
        );
    }
    assert!(
        written
            .keys()
            .any(|name| name.starts_with("== E") || name.starts_with("!= E")),
        "no enum comparison is written: {written:?}"
    );
}

#[test]
fn enum_programs_with_a_default_rule_or_a_scrutinee_that_a_call_mutates_do_not_agree_with_a_wrong_interpreter()
 {
    // The mutants must change some generated program too (the vectors above are hand-built; the corpus must also reach
    // them): over seeds 1 to 64 each mutant differs from the right run in at least one program.
    for mutant in [Mutant::DefaultTakesFirstArm, Mutant::ScrutineeTwice] {
        let differs = enum_programs(1..=64).iter().any(|p| {
            let right = run(p).ok();
            let wrong = run_mutant(p, mutant).ok().map(|r| r.values);
            right != wrong
        });
        assert!(differs, "{mutant:?} changes no program of seeds 1-64");
    }
}

/// The order-sensitive shapes the GPU batch itself holds: the enum programs of the 64-seed batch (three per seed) must
/// write and execute each of them, not only the enum space as a whole.
#[test]
fn the_order_sensitive_enum_shapes_are_written_and_executed_in_the_64_seed_batch() {
    let mut written = BTreeMap::<String, usize>::new();
    let mut executed = BTreeMap::<String, u64>::new();
    for seed in 1..=64 {
        let batch = generate_batch(seed).unwrap();
        for p in &batch.programs[METHOD_END..] {
            for (name, n) in operator_counts(p) {
                *written.entry(name).or_default() += n;
            }
            let (_, coverage) = run_covered(p).unwrap();
            for (name, n) in coverage.operators {
                *executed.entry(name).or_default() += n;
            }
        }
    }
    for wanted in [
        ast::SHAPE_MATCH_SCRUTINEE,
        ast::SHAPE_ENUM_EQ,
        ast::SHAPE_ENUM_ARG,
        ast::SHAPE_ENUM_ARGS_ROOT,
    ] {
        assert!(
            written.get(wanted).copied().unwrap_or(0) > 0
                && executed.get(wanted).copied().unwrap_or(0) > 0,
            "`{wanted}` is not written and executed by the enum programs of the 64-seed batch: {written:?}"
        );
    }
}
