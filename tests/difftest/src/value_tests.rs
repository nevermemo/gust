//! Tests of the value programs (format 10, slice m9.d1c): the interpreter's rules for tuples, the value-form `match` and
//! integer patterns on hand-built programs (written from the Rust Reference, not from the emitted Slang), the emitted
//! text, the mutation checks (the vectors must catch a wrong evaluation order, a scrutinee read twice and a pattern group
//! that matches its first literal only), and the properties of the generated value space.

use std::collections::BTreeMap;

use crate::ast::{
    self, Arm, BinOp, Elem, EnumDef, Expr, FnDef, InputBuf, Methods, Pat, Place, Program, Recv,
    Stmt, StructDef, TupleTy, Ty, operator_counts,
};
use crate::emit::emit_module;
use crate::generator::{
    ENUM_END, PROGRAMS_PER_SEED, Space, VALUE_ATTEMPT_BASE, VALUE_PROGRAMS_PER_SEED,
    generate_batch, generate_batch_in,
};
use crate::interp::{Mutant, env_mutant, run, run_covered, run_mutant, static_violation};

const S0: Ty = Ty::Struct(0);
const E0: Ty = Ty::Enum(0);

fn pair() -> Ty {
    Ty::Tuple(TupleTy::new(&[Elem::U32, Elem::U32]))
}

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn variant(v: u8) -> Expr {
    Expr::Variant { e: 0, v }
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

fn var(id: u32) -> Expr {
    Expr::Var(id, Ty::U32)
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

fn tick() -> Expr {
    call(0, Some(local()), Vec::new(), Ty::U32)
}

fn step() -> Expr {
    call(1, Some(local()), Vec::new(), pair())
}

fn tuple(elems: Vec<Expr>) -> Expr {
    Expr::Tuple { ty: pair(), elems }
}

/// Element `k` (a `u32`) of the tuple local `id`.
fn tread(id: u32, k: u8) -> Expr {
    Expr::Read {
        place: Place::var(id, pair()).field(k),
        ty: Ty::U32,
    }
}

fn mul(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Mul, l, r)
}

fn add(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Add, l, r)
}

fn let_u32(id: u32, init: Expr) -> Stmt {
    Stmt::Let {
        var: id,
        ty: Ty::U32,
        mutable: false,
        init,
    }
}

fn yield_(value: Expr) -> Vec<Stmt> {
    vec![Stmt::Yield { value }]
}

fn arm(pat: Pat, body: Vec<Stmt>) -> Arm {
    Arm { pat, body }
}

fn lits(values: &[u32]) -> Pat {
    Pat::Lits(values.iter().map(|&v| (v, false)).collect())
}

/// `let r: u32 = match <scrut> { <arms> };` as variable `id`.
fn let_match(id: u32, scrut: Expr, arms: Vec<Arm>) -> Stmt {
    Stmt::LetMatch {
        var: id,
        ty: Ty::U32,
        mutable: false,
        scrut,
        arms,
    }
}

/// `E0` (`u32`): V0 = 0, V1 = 3, V2 = 4, V3 = 10.
fn enums() -> Vec<EnumDef> {
    vec![EnumDef {
        repr: Ty::U32,
        variants: vec![(0, false), (3, true), (4, false), (10, true)],
    }]
}

/// The functions of the hand programs, over `S0 { n: u32 }`:
/// m0 `&mut self` `tick()`: `self.n += 1; self.n`;
/// m1 `&mut self` `step() -> (u32, u32)`: `self.n += 1; (self.n, self.n * 100)`;
/// m2 helper `cls(v) -> u32`: `let r: u32 = match v { 1 | 2 => return 7, _ => 5 }; r + 100`.
fn methods() -> Methods {
    let this = Place::this(S0);
    let bump = Stmt::Set {
        place: this.field(0),
        ty: Ty::U32,
        op: Some(BinOp::Add),
        value: u(1),
    };
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
                body: vec![bump.clone()],
                tail: Some(n_of(&this)),
            },
            FnDef {
                owner: Some(0),
                recv: Recv::Mut,
                params: Vec::new(),
                ret: Some(pair()),
                body: vec![bump],
                tail: Some(tuple(vec![n_of(&this), mul(n_of(&this), u(100))])),
            },
            FnDef {
                owner: None,
                recv: Recv::None,
                params: vec![(20, Ty::U32)],
                ret: Some(Ty::U32),
                body: vec![let_match(
                    21,
                    var(20),
                    vec![
                        arm(lits(&[1, 2]), vec![Stmt::Return { value: u(7) }]),
                        arm(Pat::Wild, yield_(u(5))),
                    ],
                )],
                tail: Some(add(var(21), u(100))),
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
        attempt: VALUE_ATTEMPT_BASE,
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

/// `let v1: (u32, u32) = (c.tick(), c.tick()); out = v1.0 * 100 + v1.1`.
fn tuple_order_program() -> Program {
    program(
        vec![Stmt::Let {
            var: 1,
            ty: pair(),
            mutable: false,
            init: tuple(vec![tick(), tick()]),
        }],
        add(mul(tread(1, 0), u(100)), tread(1, 1)),
    )
}

/// `let mut v1 = (1, 2); v1.0 = v1.1 + c.tick(); v1.1 += c.tick(); out = v1.0 * 10 + v1.1`.
fn element_place_program() -> Program {
    let t = || Place::var(1, pair());
    program(
        vec![
            Stmt::Let {
                var: 1,
                ty: pair(),
                mutable: true,
                init: tuple(vec![u(1), u(2)]),
            },
            Stmt::Set {
                place: t().field(0),
                ty: Ty::U32,
                op: None,
                value: add(tread(1, 1), tick()),
            },
            Stmt::Set {
                place: t().field(1),
                ty: Ty::U32,
                op: Some(BinOp::Add),
                value: tick(),
            },
        ],
        add(mul(tread(1, 0), u(10)), tread(1, 1)),
    )
}

/// Destructuring of a mutating call (annotated, then not), `f().1` of a mutating call, and the root read afterwards.
fn destructuring_program() -> Program {
    let t = pair();
    program(
        vec![
            Stmt::LetTuple {
                vars: vec![(1, false), (2, false)],
                ty: t,
                annotate: true,
                init: step(),
            },
            let_u32(
                3,
                Expr::TupleField {
                    e: Box::new(step()),
                    index: 1,
                },
            ),
            Stmt::LetTuple {
                vars: vec![(4, false), (5, false)],
                ty: t,
                annotate: false,
                init: step(),
            },
        ],
        [var(2), var(3), var(4), var(5), n_of(&local())]
            .into_iter()
            .fold(var(1), add),
    )
}

/// `let v1: u32 = match c.tick() & 7 { 1 | 2 => 100, 6 | 7 => c.n, _ => 0 }`.
fn value_scrutinee_program() -> Program {
    program(
        vec![let_match(
            1,
            Expr::bin(BinOp::BitAnd, tick(), u(7)),
            vec![
                arm(lits(&[1, 2]), yield_(u(100))),
                arm(lits(&[6, 7]), yield_(n_of(&local()))),
                arm(Pat::Wild, yield_(u(0))),
            ],
        )],
        var(1),
    )
}

/// `let v1 = 9; let v2: u32 = match v1 { 3 | 5 | 9 => 1, 7 => 2, _ => 3 }`.
fn group_program() -> Program {
    program(
        vec![
            let_u32(1, u(9)),
            let_match(
                2,
                var(1),
                vec![
                    arm(lits(&[3, 5, 9]), yield_(u(1))),
                    arm(lits(&[7]), yield_(u(2))),
                    arm(Pat::Wild, yield_(u(3))),
                ],
            ),
        ],
        var(2),
    )
}

/// A signed scrutinee with negative literal patterns (no group): `match v { 1 => 10, -3i32 => 20, i32::MAX => 40, _ => 30 }`
/// on `-3` and on `i32::MAX`.
fn signed_program() -> Program {
    let signed = |id: u32, value: i32| Stmt::Let {
        var: id,
        ty: Ty::I32,
        mutable: false,
        init: Expr::Lit(Ty::I32, value as u32),
    };
    let arms = || {
        vec![
            arm(lits(&[1]), yield_(u(10))),
            arm(Pat::Lits(vec![((-3i32) as u32, true)]), yield_(u(20))),
            arm(Pat::Lits(vec![(i32::MAX as u32, false)]), yield_(u(40))),
            arm(Pat::Wild, yield_(u(30))),
        ]
    };
    program(
        vec![
            signed(1, -3),
            signed(2, i32::MAX),
            let_match(3, Expr::Var(1, Ty::I32), arms()),
            let_match(4, Expr::Var(2, Ty::I32), arms()),
        ],
        add(mul(var(3), u(100)), var(4)),
    )
}

/// A `return` in an arm of a value `match` ends the helper: `cls(1) * 1000 + cls(9)`.
fn arm_return_program() -> Program {
    program(
        Vec::new(),
        add(
            mul(call(2, None, vec![u(1)], Ty::U32), u(1000)),
            call(2, None, vec![u(9)], Ty::U32),
        ),
    )
}

/// An arm that is itself a value `match`, and a `match` on an enum: `V1` gives `V2`, read back as its value 4.
fn nested_and_enum_program() -> Program {
    let inner = Stmt::TailMatch {
        scrut: var(1),
        arms: vec![
            arm(lits(&[4]), yield_(u(40))),
            arm(Pat::Wild, yield_(u(41))),
        ],
    };
    program(
        vec![
            let_u32(1, u(4)),
            let_match(
                2,
                var(1),
                vec![arm(lits(&[4]), vec![inner]), arm(Pat::Wild, yield_(u(0)))],
            ),
            Stmt::LetMatch {
                var: 3,
                ty: E0,
                mutable: false,
                scrut: variant(1),
                arms: vec![
                    arm(Pat::Variants(vec![0, 2]), yield_(variant(3))),
                    arm(Pat::Variants(vec![1]), yield_(variant(2))),
                    arm(Pat::Wild, yield_(variant(0))),
                ],
            },
        ],
        add(
            mul(var(2), u(100)),
            Expr::EnumAs {
                to: Ty::U32,
                e: Box::new(Expr::Var(3, E0)),
            },
        ),
    )
}

/// A struct in a tuple is a copy: `let mut v0 = S0 { n: 5 }; let v1 = (v0, 1); v0.n += 10; out = v1.0.n`.
fn struct_copy_program() -> Program {
    let t = Ty::Tuple(TupleTy::new(&[Elem::Struct(0), Elem::U32]));
    program(
        vec![
            Stmt::Let {
                var: 1,
                ty: t,
                mutable: false,
                init: Expr::Tuple {
                    ty: t,
                    elems: vec![Expr::Var(0, S0), u(1)],
                },
            },
            Stmt::Set {
                place: local().field(0),
                ty: Ty::U32,
                op: Some(BinOp::Add),
                value: u(10),
            },
        ],
        Expr::Read {
            place: Place::var(1, t).field(0).field(0),
            ty: Ty::U32,
        },
    )
}

/// A hand program with a method nobody calls whose body is `self.n <op>= <amount>`.
fn dead_compound_set(op: BinOp, amount: u32) -> Program {
    let mut p = tuple_order_program();
    p.methods.fns.push(FnDef {
        owner: Some(0),
        recv: Recv::Mut,
        params: Vec::new(),
        ret: None,
        body: vec![Stmt::Set {
            place: Place::this(S0).field(0),
            ty: Ty::U32,
            op: Some(op),
            value: u(amount),
        }],
        tail: None,
    });
    p.check().unwrap();
    p
}

/// The wide run of seeds 1 to 64 found `self.f1 <<= 34u32` in a method that was never called (seed 30 program 15): the
/// interpreter never ran it, so only a static check can discard it, and naga rejects the module. The value space now
/// folds the operand of a compound `<<=`, `>>=`, `/=` and `%=` like that of the binary operators.
#[test]
fn a_closed_constant_outside_the_domain_in_a_compound_set_is_a_static_violation_in_the_value_space()
{
    use crate::interp::Violation;
    for (op, amount, violation) in [
        (BinOp::Shl, 34, Some(Violation::ShiftAmount)),
        (BinOp::Shr, 32, Some(Violation::ShiftAmount)),
        (BinOp::Shl, 31, None),
        (BinOp::Rem, 0, Some(Violation::DivisionByZero)),
        (BinOp::Div, 0, Some(Violation::DivisionByZero)),
        (BinOp::Div, 3, None),
    ] {
        let p = dead_compound_set(op, amount);
        assert_eq!(static_violation(&p), violation, "{op:?} {amount}");
    }
    // The older spaces decode as they did: the same program in the enum space is not refused here.
    let mut older = dead_compound_set(BinOp::Shl, 34);
    older.attempt = crate::generator::ENUM_ATTEMPT_BASE;
    assert_eq!(static_violation(&older), None);
}

#[test]
fn a_tuple_literal_runs_its_elements_left_to_right() {
    // `tick` makes n 6, then 7.
    assert_eq!(value_of(&tuple_order_program()), 607);
}

#[test]
fn an_element_place_is_written_after_its_value_runs_and_a_compound_set_reads_it_then() {
    // n: 5 -> 6, `v1.0 = 2 + 6`; n: 6 -> 7, `v1.1 = 2 + 7`.
    assert_eq!(value_of(&element_place_program()), 8 * 10 + 9);
}

#[test]
fn destructuring_and_a_field_of_a_call_run_the_call_once_and_the_root_is_read_after_it() {
    // step 1: (6, 600); step 2 `.1`: 700; step 3: (8, 800); n is 8.
    assert_eq!(
        value_of(&destructuring_program()),
        6 + 600 + 700 + 8 + 800 + 8
    );
}

#[test]
fn the_scrutinee_of_a_value_match_is_evaluated_once() {
    // `tick` makes n 6, the second arm names 6 and reads n (a scrutinee read again would make it 7).
    assert_eq!(value_of(&value_scrutinee_program()), 6);
}

#[test]
fn an_integer_pattern_group_names_every_literal_of_it() {
    // 9 is the last literal of the first arm.
    assert_eq!(value_of(&group_program()), 1);
}

#[test]
fn integer_patterns_keep_their_sign_and_the_first_equal_arm_runs() {
    // -3 takes the second arm, `i32::MAX` the third.
    assert_eq!(value_of(&signed_program()), 20 * 100 + 40);
}

#[test]
fn a_return_in_an_arm_of_a_value_match_ends_the_helper() {
    assert_eq!(value_of(&arm_return_program()), 7 * 1000 + 105);
}

#[test]
fn a_value_match_nests_in_an_arm_and_matches_an_enum() {
    assert_eq!(value_of(&nested_and_enum_program()), 40 * 100 + 4);
}

#[test]
fn a_struct_element_of_a_tuple_is_a_copy() {
    assert_eq!(value_of(&struct_copy_program()), 5);
}

#[test]
fn the_mutants_change_exactly_the_vectors_that_are_meant_to_catch_them() {
    // The reference (`run`) against each mutant, on the hand programs: the `xfail-` runs of the gate set the same
    // mutant for `value_of` and expect exactly the vector named here to fail.
    let programs: [(&str, Program); 9] = [
        ("tuple order", tuple_order_program()),
        ("element place", element_place_program()),
        ("destructuring", destructuring_program()),
        ("value scrutinee", value_scrutinee_program()),
        ("group", group_program()),
        ("signed", signed_program()),
        ("arm return", arm_return_program()),
        ("nested and enum", nested_and_enum_program()),
        ("struct copy", struct_copy_program()),
    ];
    let expected: [(Mutant, &str, u32); 3] = [
        (Mutant::TupleRightToLeft, "tuple order", 706),
        (Mutant::ValueScrutineeTwice, "value scrutinee", 7),
        (Mutant::GroupFirstOnly, "group", 3),
    ];
    for (mutant, changed, value) in expected {
        for (name, p) in &programs {
            let reference = run(p).unwrap()[0];
            let mutated = run_mutant(p, mutant).unwrap().values[0];
            if *name == changed {
                assert_eq!(mutated, value, "{mutant:?} on `{name}`");
                assert_ne!(reference, mutated);
            } else {
                assert_eq!(mutated, reference, "{mutant:?} changes `{name}`");
            }
        }
    }
}

#[test]
fn the_emitted_text_of_the_hand_programs_has_the_value_forms() {
    let all = [
        emit_module(&tuple_order_program()),
        emit_module(&element_place_program()),
        emit_module(&destructuring_program()),
        emit_module(&value_scrutinee_program()),
        emit_module(&signed_program()),
        emit_module(&nested_and_enum_program()),
        emit_module(&struct_copy_program()),
    ]
    .join("\n");
    for wanted in [
        "let v1: (u32, u32) = (v0.m0(), v0.m0());",
        "v1.0 = v1.1 + v0.m0();",
        "v1.1 += v0.m0();",
        "let (v1, v2): (u32, u32) = v0.m1();",
        "v0.m1().1",
        "let (v4, v5) = v0.m1();",
        "let v1: u32 = match v0.m0() & 7u32 {",
        "1 | 2 => 100u32,",
        "_ => 0u32,",
        "-3i32 =>",
        "let v2: u32 = match v1 {",
        "let v3: E0 = match E0::V1 {",
        "E0::V0 | E0::V2 => E0::V3,",
        "(S0, u32)",
    ] {
        assert!(all.contains(wanted), "no `{wanted}` in:\n{all}");
    }
    let helper = emit_module(&arm_return_program());
    assert!(helper.contains("return 7u32;"), "{helper}");
    assert!(helper.contains("#[allow(unreachable_code)]"), "{helper}");
}

#[test]
fn the_value_space_is_the_last_two_programs_of_the_batch() {
    assert_eq!(PROGRAMS_PER_SEED, 16);
    assert_eq!(VALUE_PROGRAMS_PER_SEED, 2);
    for seed in 1..=3 {
        let batch = generate_batch(seed).unwrap();
        assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
        for (k, p) in batch.programs.iter().enumerate() {
            assert_eq!(
                p.attempt >= VALUE_ATTEMPT_BASE,
                k >= ENUM_END,
                "seed {seed} program {k}"
            );
        }
        assert!(
            batch.value_attempts
                == VALUE_PROGRAMS_PER_SEED as u64
                    + u64::from(batch.value_discarded.values().sum::<u32>()),
            "seed {seed}: {} attempts, {:?}",
            batch.value_attempts,
            batch.value_discarded
        );
    }
}

/// The batches of the value space of the first seeds (16 programs each).
fn value_programs(seeds: std::ops::RangeInclusive<u64>) -> Vec<Program> {
    seeds
        .flat_map(|seed| generate_batch_in(Space::Value, seed).unwrap().programs)
        .collect()
}

#[test]
fn every_value_program_is_well_typed_in_the_domain_and_has_a_value_form_and_a_call() {
    for p in value_programs(1..=12) {
        p.check().unwrap_or_else(|e| panic!("seed {} {e}", p.seed));
        assert!(run(&p).is_ok());
        assert!(static_violation(&p).is_none());
        assert!(p.has_call() && p.has_value_form());
        assert!(!p.has_float() && !p.has_aggregate());
    }
}

type Written = BTreeMap<String, usize>;
type Executed = BTreeMap<String, u64>;

/// The written and executed counts of the names over the given programs.
fn counts(programs: &[Program]) -> (Written, Executed) {
    let mut written = Written::new();
    let mut executed = Executed::new();
    for p in programs {
        for (name, n) in operator_counts(p) {
            *written.entry(name).or_default() += n;
        }
        let (_, coverage) = run_covered(p).unwrap();
        for (name, n) in coverage.operators {
            *executed.entry(name).or_default() += n;
        }
    }
    (written, executed)
}

fn assert_reached(wanted: &[&str], written: &Written, executed: &Executed, what: &str) {
    for name in wanted {
        assert!(
            written.get(*name).copied().unwrap_or(0) > 0,
            "`{name}` is never written in {what}: {written:?}"
        );
        assert!(
            executed.get(*name).copied().unwrap_or(0) > 0,
            "`{name}` is never executed in {what}: {executed:?}"
        );
    }
}

/// The new constructs, all of them: written and executed within seeds 1 to 64 of the value space.
const NEW_FORMS: [&str; 15] = [
    ast::MATCH_VALUE,
    ast::MATCH_VALUE_NESTED,
    ast::MATCH_VALUE_BLOCK,
    ast::MATCH_VALUE_RETURN,
    ast::MATCH_INT,
    ast::MATCH_INT_GROUP,
    ast::TUPLE_LIT,
    ast::TUPLE_FIELD,
    ast::TUPLE_READ,
    ast::TUPLE_SET,
    ast::TUPLE_SET_OP,
    ast::TUPLE_DESTRUCT,
    ast::TUPLE_DESTRUCT_TYPED,
    ast::TUPLE_ARG,
    "return",
];

/// The order-sensitive shapes the value forms add.
const NEW_SHAPES: [&str; 6] = [
    ast::SHAPE_TUPLE_LIT_ROOT,
    ast::SHAPE_TUPLE_SET_CALL,
    ast::SHAPE_TUPLE_FIELD_CALL,
    ast::SHAPE_DESTRUCT_CALL,
    ast::SHAPE_LT_ELEMENT,
    ast::SHAPE_VMATCH_SCRUTINEE,
];

#[test]
fn the_value_forms_and_order_sensitive_shapes_are_reachable_within_the_first_64_seeds() {
    let (written, executed) = counts(&value_programs(1..=64));
    assert_reached(
        &NEW_FORMS,
        &written,
        &executed,
        "seeds 1-64 of the value space",
    );
    assert_reached(
        &NEW_SHAPES,
        &written,
        &executed,
        "seeds 1-64 of the value space",
    );
    assert!(
        executed
            .get(ast::MATCH_ARM_RETURN_RUN)
            .copied()
            .unwrap_or(0)
            > 0,
        "no `return` in an arm of a match ever runs: {executed:?}"
    );
}

/// The shapes the GPU batch itself holds: the value programs of the 64-seed batch (two per seed) must write and execute
/// each of them, not only the value space as a whole.
#[test]
fn the_new_shapes_are_written_and_executed_in_the_64_seed_batch() {
    let mut programs = Vec::new();
    for seed in 1..=64 {
        programs.extend(
            generate_batch(seed)
                .unwrap()
                .programs
                .into_iter()
                .skip(ENUM_END),
        );
    }
    assert_eq!(programs.len(), 64 * VALUE_PROGRAMS_PER_SEED);
    let (written, executed) = counts(&programs);
    let what = "the value programs of the 64-seed batch";
    assert_reached(&NEW_SHAPES, &written, &executed, what);
    assert_reached(
        &[
            ast::MATCH_VALUE,
            ast::MATCH_INT,
            ast::MATCH_INT_GROUP,
            ast::TUPLE_LIT,
            ast::TUPLE_DESTRUCT,
            ast::TUPLE_ARG,
        ],
        &written,
        &executed,
        what,
    );
}

/// Nit R1 of the m9.d1b re-check: a `return` inside a `match` arm ran in 4 of 24 programs. In the value space it runs
/// in far more, and the minimum is pinned.
#[test]
fn a_return_in_a_match_arm_runs_in_at_least_a_fifth_of_the_value_programs() {
    let programs = value_programs(1..=8);
    let running = programs
        .iter()
        .filter(|p| {
            let (_, coverage) = run_covered(p).unwrap();
            coverage
                .operators
                .get(ast::MATCH_ARM_RETURN_RUN)
                .copied()
                .unwrap_or(0)
                > 0
        })
        .count();
    assert!(
        running * 5 >= programs.len(),
        "a return in a match arm runs in {running} of {} programs",
        programs.len()
    );
}

#[test]
fn the_mutants_also_change_some_generated_value_program() {
    // The vectors above are hand-built; the corpus must reach the same rules: over seeds 1 to 64 each mutant differs
    // from the right run in at least one program.
    let programs = value_programs(1..=64);
    for mutant in [
        Mutant::TupleRightToLeft,
        Mutant::ValueScrutineeTwice,
        Mutant::GroupFirstOnly,
    ] {
        let differs = programs.iter().any(|p| {
            let right = run(p).ok();
            let wrong = run_mutant(p, mutant).ok().map(|r| r.values);
            right != wrong
        });
        assert!(differs, "{mutant:?} changes no program of seeds 1-64");
    }
}
