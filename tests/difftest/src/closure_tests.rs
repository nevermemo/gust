//! Tests of the closure programs (format 12, slice m9.d1e): the interpreter's rules for a closure call on hand-built
//! programs (written from the Rust Reference, not from the emitted Slang), the emitted text, the mutation checks (the
//! vectors must catch arguments run right to left, a struct or array parameter passed by reference, and a body run
//! once whose result is reused) and the properties of the generated closure space.
//!
//! The rules (Rust Reference, closure expressions and call expressions): the arguments of a call are evaluated left to
//! right, then the parameters are bound by value (an array or struct argument is copied), the body runs once for
//! every call, and the value of the call is the value of the body's block.

use crate::array_tests::scan_driver_hole;
use crate::ast::{
    ArrTy, BinOp, ClosureDef, Elem, Expr, FnDef, InputBuf, Methods, Place, Program, Recv, Root,
    Stmt, StructDef, Ty, operator_counts,
};
use crate::ast::{
    CLOSURE_ARRAY_RESULT, CLOSURE_BODY_HELPER, CLOSURE_BODY_METHOD, CLOSURE_CALL,
    CLOSURE_IN_HELPER, CLOSURE_IN_KERNEL, CLOSURE_IN_METHOD, CLOSURE_LET, CLOSURE_MUT_ARRAY,
    CLOSURE_MUT_STRUCT, CLOSURE_PARAM_METHOD, SHAPE_CLOSURE_ARGS_ROOT, SHAPE_CLOSURE_ARRAY_LIT,
    SHAPE_CLOSURE_MUT_PARAM, SHAPE_CLOSURE_SET_INDEX, SHAPE_CLOSURE_SET_OP, SHAPE_CLOSURE_STORE,
    SHAPE_CLOSURE_TWICE,
};
use crate::emit::emit_module;
use crate::generator::{
    ARRAY_END, CLOSURE_ATTEMPT_BASE, CLOSURE_PROGRAMS_PER_SEED, Space, generate_batch,
    generate_batch_in,
};
use crate::interp::{Mutant, env_mutant, run, run_covered, run_mutant};

const S0: Ty = Ty::Struct(0);

fn arr() -> Ty {
    Ty::Array(ArrTy::new(Elem::U32, 3))
}

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn add(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Add, l, r)
}

fn mul(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Mul, l, r)
}

/// `c.tick()` on the local `v0`: `self.n += 1; self.n`.
fn tick() -> Expr {
    Expr::Call {
        f: 0,
        owner: None,
        recv: Some(Place::var(0, S0)),
        args: Vec::new(),
        ty: Ty::U32,
    }
}

fn methods() -> Methods {
    let this = Place::this(S0);
    Methods {
        structs: vec![StructDef {
            fields: vec![Ty::U32],
        }],
        fns: vec![FnDef {
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
            tail: Some(Expr::Read {
                place: this.field(0),
                ty: Ty::U32,
            }),
        }],
        enums: Vec::new(),
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
        store: None,
        attempt: CLOSURE_ATTEMPT_BASE,
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

fn closure(var: u32, params: Vec<(u32, Ty, bool)>, body: Vec<Stmt>, tail: Expr, ret: Ty) -> Stmt {
    Stmt::LetClosure {
        def: Box::new(ClosureDef {
            var,
            params,
            ret,
            body,
            tail,
        }),
    }
}

fn call(var: u32, args: Vec<Expr>) -> Expr {
    Expr::ClosureCall {
        var,
        args,
        ty: Ty::U32,
    }
}

fn field(var: u32) -> Place {
    Place::var(var, S0).field(0)
}

fn elem(var: u32, k: u32) -> Place {
    Place {
        index: Some(Box::new(u(k))),
        ..Place::var(var, arr())
    }
}

/// `let cl10 = |a: u32, b: u32| -> u32 { a * 10 + b }; out = cl10(c.tick(), c.tick())`: 6 and 7, so 67.
fn order_program() -> Program {
    program(
        vec![closure(
            10,
            vec![(11, Ty::U32, false), (12, Ty::U32, false)],
            Vec::new(),
            add(mul(Expr::Var(11, Ty::U32), u(10)), Expr::Var(12, Ty::U32)),
            Ty::U32,
        )],
        call(10, vec![tick(), tick()]),
    )
}

/// `let cl10 = |mut s: S0| -> u32 { s.n += 100; s.n }; let v1 = cl10(v0); out = v1 * 1000 + v0.n`: the callee changes
/// its own copy (105) and the caller's `n` stays 5.
fn struct_copy_program() -> Program {
    program(
        vec![
            closure(
                10,
                vec![(11, S0, true)],
                vec![Stmt::Set {
                    place: field(11),
                    ty: Ty::U32,
                    op: Some(BinOp::Add),
                    value: u(100),
                }],
                Expr::Read {
                    place: field(11),
                    ty: Ty::U32,
                },
                Ty::U32,
            ),
            Stmt::Let {
                var: 1,
                ty: Ty::U32,
                mutable: false,
                init: call(10, vec![Expr::Var(0, S0)]),
            },
        ],
        add(
            mul(Expr::Var(1, Ty::U32), u(1000)),
            Expr::Read {
                place: field(0),
                ty: Ty::U32,
            },
        ),
    )
}

/// `let cl10 = |mut a: [u32; 3]| -> u32 { a[0] = 99; a[0] }; let mut v2 = [1, 2, 3]; let v1 = cl10(v2);
/// out = v1 * 1000 + v2[0]`: 99 and the caller's 1.
fn array_copy_program() -> Program {
    program(
        vec![
            closure(
                10,
                vec![(11, arr(), true)],
                vec![Stmt::Set {
                    place: elem(11, 0),
                    ty: Ty::U32,
                    op: None,
                    value: u(99),
                }],
                Expr::Read {
                    place: elem(11, 0),
                    ty: Ty::U32,
                },
                Ty::U32,
            ),
            Stmt::Let {
                var: 2,
                ty: arr(),
                mutable: true,
                init: Expr::ArrayLit {
                    ty: arr(),
                    elems: vec![u(1), u(2), u(3)],
                },
            },
            Stmt::Let {
                var: 1,
                ty: Ty::U32,
                mutable: false,
                init: call(10, vec![Expr::Var(2, arr())]),
            },
        ],
        add(
            mul(Expr::Var(1, Ty::U32), u(1000)),
            Expr::Read {
                place: elem(2, 0),
                ty: Ty::U32,
            },
        ),
    )
}

/// `let cl10 = |a: u32| -> u32 { a + 1 }; out = cl10(1) * 100 + cl10(10)`: 2 and 11, so 211.
fn two_calls_program() -> Program {
    program(
        vec![plus_one()],
        add(mul(call(10, vec![u(1)]), u(100)), call(10, vec![u(10)])),
    )
}

/// `out = cl10(c.tick()) * 100 + cl10(c.tick())`: the arguments are 6 and 7, so the results 7 and 8, so 708.
fn twice_program() -> Program {
    program(
        vec![plus_one()],
        add(mul(call(10, vec![tick()]), u(100)), call(10, vec![tick()])),
    )
}

fn plus_one() -> Stmt {
    closure(
        10,
        vec![(11, Ty::U32, false)],
        Vec::new(),
        add(Expr::Var(11, Ty::U32), u(1)),
        Ty::U32,
    )
}

#[test]
fn closure_arguments_run_left_to_right() {
    assert_eq!(value_of(&order_program()), 67);
}

#[test]
fn a_closure_parameter_is_a_copy() {
    assert_eq!(value_of(&struct_copy_program()), 105 * 1000 + 5);
    assert_eq!(value_of(&array_copy_program()), 99 * 1000 + 1);
}

#[test]
fn a_closure_body_runs_once_per_call() {
    assert_eq!(value_of(&two_calls_program()), 211);
    assert_eq!(value_of(&twice_program()), 708);
}

/// Each mutant changes the vectors that pin its rule, and no other: the right run gives the values above, the
/// mutated one the values of the wrong rule.
#[test]
fn the_closure_mutants_change_exactly_their_vectors() {
    let programs: [(&str, Program); 5] = [
        ("order", order_program()),
        ("struct copy", struct_copy_program()),
        ("array copy", array_copy_program()),
        ("two calls", two_calls_program()),
        ("twice", twice_program()),
    ];
    let reference: [u32; 5] = [67, 105005, 99001, 211, 708];
    // (mutant, the vectors it changes with the value it gives).
    type Changes = &'static [(&'static str, u32)];
    let expected: [(Mutant, Changes); 3] = [
        (Mutant::ClosureArgsRightToLeft, &[("order", 76)]),
        (
            Mutant::ClosureParamByRef,
            &[
                ("struct copy", 105 * 1000 + 105),
                ("array copy", 99 * 1000 + 99),
            ],
        ),
        (Mutant::ClosureMemo, &[("two calls", 202), ("twice", 707)]),
    ];
    for (mutant, changed) in expected {
        for ((name, p), right) in programs.iter().zip(reference) {
            assert_eq!(run(p).unwrap()[0], right, "`{name}`");
            let mutated = run_mutant(p, mutant).map(|run| run.values[0]);
            match changed.iter().find(|(n, _)| n == name) {
                Some((_, value)) => {
                    assert_eq!(mutated.ok(), Some(*value), "{mutant:?} on `{name}`");
                }
                None => assert_eq!(mutated.ok(), Some(right), "{mutant:?} changes `{name}`"),
            }
        }
    }
}

#[test]
fn the_emitted_text_of_the_hand_programs_has_the_closure_forms() {
    let text = emit_module(&order_program());
    assert!(
        text.contains("let cl10 = |v11: u32, v12: u32| -> u32 {"),
        "{text}"
    );
    assert!(text.contains("cl10("), "{text}");
    let text = emit_module(&struct_copy_program());
    assert!(text.contains("|mut v11: S0| -> u32 {"), "{text}");
    let text = emit_module(&array_copy_program());
    assert!(text.contains("|mut v11: [u32; 3]| -> u32 {"), "{text}");
    assert!(text.contains("};"), "{text}");
}

/// A closure may not capture, be a value, be untyped, be nested or sit in a `for` bound: the checker refuses each.
#[test]
fn the_checker_refuses_the_forms_the_generator_never_writes() {
    // A capture of the enclosing local `v0`.
    let capture = Program {
        body: vec![
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
            closure(
                10,
                vec![(11, Ty::U32, false)],
                Vec::new(),
                Expr::Read {
                    place: field(0),
                    ty: Ty::U32,
                },
                Ty::U32,
            ),
        ],
        result: call(10, vec![u(1)]),
        ..order_program()
    };
    assert!(capture.check().is_err());
    // A `&mut self` call on a parameter (E0333).
    let on_param = Program {
        body: vec![
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
            closure(
                10,
                vec![(11, S0, true)],
                Vec::new(),
                Expr::Call {
                    f: 0,
                    owner: None,
                    recv: Some(Place::var(11, S0)),
                    args: Vec::new(),
                    ty: Ty::U32,
                },
                Ty::U32,
            ),
        ],
        result: call(10, vec![Expr::Var(0, S0)]),
        ..order_program()
    };
    assert!(on_param.check().is_err());
    // A closure inside a closure.
    let nested = Program {
        body: vec![
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
            closure(
                10,
                vec![(11, Ty::U32, false)],
                vec![plus_one_as(12, 13)],
                Expr::Var(11, Ty::U32),
                Ty::U32,
            ),
        ],
        result: call(10, vec![u(1)]),
        ..order_program()
    };
    assert!(nested.check().is_err());
}

fn plus_one_as(var: u32, param: u32) -> Stmt {
    closure(
        var,
        vec![(param, Ty::U32, false)],
        Vec::new(),
        add(Expr::Var(param, Ty::U32), u(1)),
        Ty::U32,
    )
}

/// The closure space is selected by its attempt numbers and ends every batch: `CLOSURE_PROGRAMS_PER_SEED` programs
/// after the array programs, each holding a closure and a call.
#[test]
fn a_batch_ends_with_its_closure_programs() {
    assert_eq!(CLOSURE_PROGRAMS_PER_SEED, 2);
    for seed in 1..=4 {
        let batch = generate_batch(seed).unwrap();
        for p in &batch.programs[ARRAY_END..] {
            assert!(p.attempt >= CLOSURE_ATTEMPT_BASE, "seed {seed}");
            assert!(
                p.has_closure() && p.has_array() && p.has_call(),
                "seed {seed}"
            );
        }
        assert_eq!(batch.programs[ARRAY_END..].len(), CLOSURE_PROGRAMS_PER_SEED);
        let space = generate_batch_in(Space::Closure, seed).unwrap();
        assert_eq!(&space.programs[..2], &batch.programs[ARRAY_END..]);
        assert!(batch.closure_attempts == 2 + batch.closure_discarded.values().sum::<u32>() as u64);
    }
}

/// Every generated closure is bound once under a name `cl<N>` that no variable, helper or reserved name has, is
/// written with typed parameters and a written result, and the program passes the checker (no capture, no nesting,
/// no closure as a value).
#[test]
fn generated_closures_have_distinct_names_and_pass_the_checker() {
    for seed in 1..=16 {
        for p in generate_batch_in(Space::Closure, seed).unwrap().programs {
            p.check()
                .unwrap_or_else(|e| panic!("seed {seed} attempt {}: {e}", p.attempt));
            let mut names: Vec<u32> = Vec::new();
            let mut vars: Vec<u32> = Vec::new();
            let mut visit = |s: &Stmt| match s {
                Stmt::LetClosure { def } => names.push(def.var),
                Stmt::Let { var, .. } => vars.push(*var),
                _ => {}
            };
            crate::ast::walk_all_stmts(&p.body, &mut visit);
            for f in &p.methods.fns {
                crate::ast::walk_all_stmts(&f.body, &mut visit);
            }
            let text = emit_module(&p);
            for name in &names {
                assert!(
                    !vars.contains(name),
                    "seed {seed}: cl{name} is also a variable"
                );
                assert_eq!(
                    text.matches(&format!("let cl{name} = |")).count(),
                    1,
                    "seed {seed} attempt {}",
                    p.attempt
                );
            }
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                names.len(),
                "seed {seed}: a closure name is bound twice"
            );
            assert!(!text.contains("__gust_"), "seed {seed}");
            assert!(!text.contains("move |"), "seed {seed}");
        }
    }
}

/// The names the report counts for closures: forms (program counts and written counts) and order shapes.
const FORMS: &[&str] = &[
    CLOSURE_LET,
    CLOSURE_CALL,
    CLOSURE_IN_KERNEL,
    CLOSURE_IN_HELPER,
    CLOSURE_IN_METHOD,
    CLOSURE_BODY_HELPER,
    CLOSURE_BODY_METHOD,
    CLOSURE_MUT_ARRAY,
    CLOSURE_MUT_STRUCT,
    CLOSURE_ARRAY_RESULT,
];

const SHAPES: &[&str] = &[
    SHAPE_CLOSURE_ARGS_ROOT,
    SHAPE_CLOSURE_ARRAY_LIT,
    SHAPE_CLOSURE_SET_OP,
    SHAPE_CLOSURE_SET_INDEX,
    SHAPE_CLOSURE_STORE,
    SHAPE_CLOSURE_TWICE,
    SHAPE_CLOSURE_MUT_PARAM,
    CLOSURE_PARAM_METHOD,
];

/// (programs that write it, evaluations) minima over the closure space of seeds 1 to 64 (1024 programs) and over the
/// 64-seed batch (128 programs). The two numbers are the static and the executed count; the printed
/// `CLOSURE-SHAPE` lines are the ones DIFFTEST.md quotes.
type Minima = &'static [(&'static str, (usize, u64))];

const SPACE_MINIMA: Minima = &[
    (SHAPE_CLOSURE_ARGS_ROOT, (836, 622679)),
    (SHAPE_CLOSURE_ARRAY_LIT, (121, 62645)),
    (SHAPE_CLOSURE_SET_OP, (190, 102081)),
    (SHAPE_CLOSURE_SET_INDEX, (74, 31112)),
    (SHAPE_CLOSURE_STORE, (610, 228199)),
    (SHAPE_CLOSURE_TWICE, (272, 197026)),
    (SHAPE_CLOSURE_MUT_PARAM, (836, 513496)),
    (CLOSURE_PARAM_METHOD, (31, 46413)),
];
const BATCH_MINIMA: Minima = &[
    (SHAPE_CLOSURE_ARGS_ROOT, (102, 108616)),
    (SHAPE_CLOSURE_ARRAY_LIT, (16, 6908)),
    (SHAPE_CLOSURE_SET_OP, (17, 30341)),
    (SHAPE_CLOSURE_SET_INDEX, (6, 259)),
    (SHAPE_CLOSURE_STORE, (75, 26333)),
    (SHAPE_CLOSURE_TWICE, (23, 9147)),
    (SHAPE_CLOSURE_MUT_PARAM, (106, 54253)),
    (CLOSURE_PARAM_METHOD, (2, 65)),
];

fn tally(programs: &[Program]) -> Vec<(&'static str, usize, usize, u64)> {
    let names: Vec<&'static str> = FORMS.iter().chain(SHAPES).copied().collect();
    let mut rows: Vec<(&'static str, usize, usize, u64)> =
        names.iter().map(|&n| (n, 0, 0, 0)).collect();
    for p in programs {
        let counts = operator_counts(p);
        let (_, coverage) = run_covered(p).unwrap();
        for row in &mut rows {
            let n = counts.get(row.0).copied().unwrap_or(0);
            row.1 += usize::from(n > 0);
            row.2 += n;
            row.3 += coverage.operators.get(row.0).copied().unwrap_or(0);
        }
    }
    rows
}

/// The 64-seed closure space and batch write and run every form and every order shape, each causal (the shape counts
/// only where the value shown to the order is a `&mut self` call on a root that the other operand reads or calls).
#[test]
fn the_closure_programs_of_the_64_seed_runs_write_and_run_every_form_and_order_shape() {
    let mut space = Vec::new();
    let mut batch = Vec::new();
    for seed in 1..=64 {
        space.extend(generate_batch_in(Space::Closure, seed).unwrap().programs);
        batch.extend(
            generate_batch(seed)
                .unwrap()
                .programs
                .into_iter()
                .skip(ARRAY_END),
        );
    }
    assert_eq!((space.len(), batch.len()), (1024, 128));
    let mut failures: Vec<String> = Vec::new();
    for (label, programs, minima) in [
        ("space", &space, SPACE_MINIMA),
        ("batch", &batch, BATCH_MINIMA),
    ] {
        let counts = tally(programs);
        for &(name, with, written, executed) in &counts {
            eprintln!(
                "CLOSURE-{label} {name}: programs {with}, written {written}, executed {executed}"
            );
            assert!(
                with > 0 && written > 0 && executed > 0,
                "{label}: `{name}` is not written and executed"
            );
        }
        for (name, (min_programs, min_evals)) in minima {
            let row = counts.iter().find(|r| r.0 == *name).unwrap();
            if row.1 < *min_programs || row.3 < *min_evals {
                failures.push(format!(
                    "{label}: {name}: {} programs, {} evaluations",
                    row.1, row.3
                ));
            }
        }
    }
    // Seeds 1 to 4 (the G5 run): the closure programs are the first 8 of the batch; each order shape is written by at least
    // one and executed. (The method-call-on-a-parameter row is a form, not an order shape: it may be absent here.)
    for &(name, with, _, executed) in &tally(&batch[..8]) {
        eprintln!("CLOSURE-seeds-1-4 {name}: programs {with}, evaluations {executed}");
        if SHAPES.contains(&name) && name != CLOSURE_PARAM_METHOD && (with == 0 || executed == 0) {
            failures.push(format!("seeds 1 to 4: {name} is not written and executed"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Each order mutant changes generated programs, not only the hand vectors, so a GPU lowering that ran the same
/// rule wrong would show in a generated run (counts printed as `CLOSURE-MUTANT`).
#[test]
fn the_closure_mutants_change_generated_programs() {
    let mut space = Vec::new();
    let mut batch = Vec::new();
    for seed in 1..=64 {
        space.extend(generate_batch_in(Space::Closure, seed).unwrap().programs);
        batch.extend(
            generate_batch(seed)
                .unwrap()
                .programs
                .into_iter()
                .skip(ARRAY_END),
        );
    }
    // A change is a different output buffer (or a different domain verdict). A mutated run that breaks an assertion of
    // the interpreter is no change here (it is counted apart as `asserts`).
    let changes = |p: &Program, mutant| {
        let mutated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_mutant(p, mutant).ok().map(|r| r.values)
        }));
        mutated.is_ok_and(|values| run(p).ok() != values)
    };
    let mut failures = Vec::new();
    let breaks = |p: &Program, mutant| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_mutant(p, mutant);
        }))
        .is_err()
    };
    for (mutant, in_space_floor, in_batch_floor) in [
        (Mutant::ClosureArgsRightToLeft, 700, 80),
        (Mutant::ClosureParamByRef, 600, 70),
        (Mutant::ClosureMemo, 700, 70),
    ] {
        let in_space = space.iter().filter(|p| changes(p, mutant)).count();
        let in_batch = batch.iter().filter(|p| changes(p, mutant)).count();
        // The closure programs of the batch of seeds 1 to 4 (the G5 run): each rule is observed by at least one.
        let in_first_four = batch[..8].iter().filter(|p| changes(p, mutant)).count();
        let asserts = space.iter().filter(|p| breaks(p, mutant)).count();
        eprintln!(
            "CLOSURE-MUTANT {mutant:?}: changes {in_space} of 1024 space programs, {in_batch} of 128 batch programs, {in_first_four} of the 8 of seeds 1 to 4 (and {asserts} space programs that break an interpreter assertion)"
        );
        if in_space < in_space_floor || in_batch < in_batch_floor || in_first_four == 0 {
            failures.push(format!("{mutant:?}: {in_space} in the space, {in_batch} in the batch, {in_first_four} in seeds 1 to 4"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Whether `index` is a literal, or a literal modulo `.len()`: the only indices a closure writes an array at (D168).
fn literal_index(index: &Expr) -> bool {
    match index {
        Expr::Lit(..) => true,
        Expr::Bin {
            op: BinOp::Rem,
            l,
            r,
            ..
        } => matches!(**l, Expr::Lit(..)) && matches!(**r, Expr::ArrLen { .. }),
        _ => false,
    }
}

/// D168 in the closure space. (1) The whole-program scan of the array space (identity operands, self-assignments and
/// element calls at a dynamic index, any array of any length) holds for the closure space. (2)
/// Inside a closure body an array is written only at a literal index (or a literal modulo `.len()`), and no array
/// local is declared: a closure never fills an array in the hole form, and never writes back a dynamic element.
#[test]
fn no_closure_fills_or_writes_back_an_array_in_the_driver_hole_form() {
    let counts = scan_driver_hole(Space::Closure);
    eprintln!("CLOSURE-GUARD-SCAN {counts:?}");
    assert!(
        counts.rewritten > 0
            && counts.literal_ops > 0
            && counts.literal_calls > 0
            && counts.plain_dynamic > 0,
        "{counts:?}"
    );
    let (mut array_sets, mut closures) = (0, 0);
    for seed in 1..=64 {
        for p in generate_batch_in(Space::Closure, seed).unwrap().programs {
            let mut defs: Vec<ClosureDef> = Vec::new();
            let mut collect = |s: &Stmt| {
                if let Stmt::LetClosure { def } = s {
                    defs.push((**def).clone());
                }
            };
            crate::ast::walk_all_stmts(&p.body, &mut collect);
            for f in &p.methods.fns {
                crate::ast::walk_all_stmts(&f.body, &mut collect);
            }
            for def in defs {
                closures += 1;
                crate::ast::walk_all_stmts(&def.body, &mut |s: &Stmt| match s {
                    Stmt::Let {
                        ty: Ty::Array(_), ..
                    } => panic!("seed {seed}: an array local inside cl{}", def.var),
                    Stmt::Set { place, .. } => {
                        if let (Root::Var(_), Ty::Array(_), Some(index)) =
                            (place.root, place.root_ty, place.index.as_deref())
                        {
                            array_sets += 1;
                            assert!(
                                literal_index(index),
                                "seed {seed}: cl{} writes an array at {index:?}",
                                def.var
                            );
                        }
                    }
                    _ => {}
                });
            }
        }
    }
    eprintln!("CLOSURE-GUARD-SCAN closures {closures}, array writes inside closures {array_sets}");
    assert!(closures > 0 && array_sets > 0);
}

/// `let cl10 = |a: u32, b: u32| -> u32 { ((a & 255) * 1) ^ ((b & 255) * 3) }; out = cl10(<x>, <y>)`.
fn weighted_program(x: Expr, y: Expr) -> Program {
    let term = |var: u32, w: u32| {
        mul(
            Expr::bin(BinOp::BitAnd, Expr::Var(var, Ty::U32), u(255)),
            u(w),
        )
    };
    program(
        vec![closure(
            10,
            vec![(11, Ty::U32, false), (12, Ty::U32, false)],
            Vec::new(),
            Expr::bin(BinOp::BitXor, term(11, 1), term(12, 3)),
            Ty::U32,
        )],
        call(10, vec![x, y]),
    )
}

/// `let cl10 = |mut s: S0| -> u32 { s.n ^= 1; s.n }; out = <result>`.
fn writer_program(result: Expr) -> Program {
    program(
        vec![closure(
            10,
            vec![(11, S0, true)],
            vec![Stmt::Set {
                place: field(11),
                ty: Ty::U32,
                op: Some(BinOp::BitXor),
                value: u(1),
            }],
            Expr::Read {
                place: field(11),
                ty: Ty::U32,
            },
            Ty::U32,
        )],
        result,
    )
}

/// The order shapes count only where the order is observable: the arguments shape and the store shape need a closure
/// that weights both arguments and two arguments that call `&mut self` on one root; the by-value shape needs the call,
/// then (a `^`, which does not short-circuit) a read of the variable it was given, in a closure that writes it. The
/// static and the executed counts agree.
#[test]
fn the_order_shapes_count_only_where_the_order_is_observable() {
    let counted = |program: &Program, name: &str| {
        let text = operator_counts(program).get(name).copied().unwrap_or(0);
        let ran = run_covered(program)
            .unwrap()
            .1
            .operators
            .get(name)
            .copied()
            .unwrap_or(0);
        assert_eq!(
            text > 0,
            ran > 0,
            "{name}: {text} in the text, {ran} executed"
        );
        text > 0
    };
    // Two calls on one root, both arguments weighted: counted, as the store shape too.
    let both = weighted_program(tick(), tick());
    assert!(counted(&both, SHAPE_CLOSURE_ARGS_ROOT));
    crate::ast::register_closures(&both);
    assert!(!crate::ast::closure_store_names(&both.methods, &both.result).is_empty());
    // One argument is a literal: nothing to order.
    let one = weighted_program(tick(), u(5));
    assert!(!counted(&one, SHAPE_CLOSURE_ARGS_ROOT));
    crate::ast::register_closures(&one);
    assert!(crate::ast::closure_store_names(&one.methods, &one.result).is_empty());
    // A closure that does not weight its arguments (`a * 10 + b`) is not counted either.
    assert!(!counted(&order_program(), SHAPE_CLOSURE_ARGS_ROOT));
    // The by-value shape: `cl10(v0) ^ v0.n` counts; the same call summed (`cl10(v0) + v0.n`) or without a read does not.
    let read = Expr::Read {
        place: field(0),
        ty: Ty::U32,
    };
    let observed = writer_program(Expr::bin(
        BinOp::BitXor,
        call(10, vec![Expr::Var(0, S0)]),
        read.clone(),
    ));
    assert!(counted(&observed, SHAPE_CLOSURE_MUT_PARAM));
    let summed = writer_program(add(call(10, vec![Expr::Var(0, S0)]), read));
    assert!(!counted(&summed, SHAPE_CLOSURE_MUT_PARAM));
    let unread = writer_program(call(10, vec![Expr::Var(0, S0)]));
    assert!(!counted(&unread, SHAPE_CLOSURE_MUT_PARAM));
}
