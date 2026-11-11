//! Tests of the array programs (format 11, slice m9.d1d): the interpreter's rules for array values and indexed places on
//! hand-built programs (written from the Rust Reference, not from the emitted Slang), the emitted text, the mutation
//! checks (the vectors must catch literal elements run right to left, a repeat evaluated per element, an index
//! evaluated before the value, and an array passed by reference) and the properties of the generated array space.
//!
//! Scope: the generator draws `u32`, `i32`, `bool` and struct elements. Arrays of enums, vectors and matrices are
//! accepted by the checker, the interpreter and the emitter but never generated; the `Vec*`/`Mat*` place-first rule of
//! a compound set has no vector here because no accepted program can write a compound set on those types yet.

use crate::ast::{
    ArrTy, BinOp, Elem, Expr, FnDef, InputBuf, Methods, Place, Program, Recv, Stmt, Store,
    StructDef, Ty, operator_counts,
};
use crate::emit::emit_module;
use crate::generator::{
    ARRAY_ATTEMPT_BASE, ARRAY_END, ARRAY_PROGRAMS_PER_SEED, Space, VALUE_END, generate_batch,
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

fn local() -> Place {
    Place::var(0, S0)
}

fn tick() -> Expr {
    Expr::Call {
        f: 0,
        owner: None,
        recv: Some(local()),
        args: Vec::new(),
        ty: Ty::U32,
    }
}

/// Element `index` of the array local `id`.
fn at(id: u32, index: Expr) -> Place {
    Place {
        index: Some(Box::new(index)),
        ..Place::var(id, arr())
    }
}

fn read(id: u32, k: u32) -> Expr {
    Expr::Read {
        place: at(id, u(k)),
        ty: Ty::U32,
    }
}

fn let_arr(id: u32, init: Expr) -> Stmt {
    Stmt::Let {
        var: id,
        ty: arr(),
        mutable: true,
        init,
    }
}

fn lit(elems: Vec<Expr>) -> Expr {
    Expr::ArrayLit { ty: arr(), elems }
}

/// `a[0] * 100 + a[1] * 10 + a[2]` over the array local `id`.
fn digits(id: u32) -> Expr {
    add(
        add(mul(read(id, 0), u(100)), mul(read(id, 1), u(10))),
        read(id, 2),
    )
}

/// The functions of the hand programs over `S0 { n: u32 }`:
/// m0 `&mut self` `tick()`: `self.n += 1; self.n`;
/// m1 helper `fill(mut a: [u32; 3]) -> [u32; 3]`: `a[0] = 99; a`.
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
                tail: Some(Expr::Read {
                    place: this.field(0),
                    ty: Ty::U32,
                }),
            },
            FnDef {
                owner: None,
                recv: Recv::None,
                params: vec![(20, arr())],
                ret: Some(arr()),
                body: vec![Stmt::Set {
                    place: at(20, u(0)),
                    ty: Ty::U32,
                    op: None,
                    value: u(99),
                }],
                tail: Some(Expr::Var(20, arr())),
            },
        ],
        enums: Vec::new(),
    }
}

/// `let mut v0 = S0 { n: 5 }; <body>; out[i] = <result>` over one element.
fn program(body: Vec<Stmt>, result: Expr, store: Option<Store>) -> Program {
    let program = raw_program(body, result, store);
    program
        .check()
        .unwrap_or_else(|error| panic!("the hand program is refused: {error}"));
    program
}

/// [`program`] without `Program::check`: the order of an index against a value is unobservable in an accepted program
/// (a `&mut self` call in an assignee place is E0421), so the vectors that pin it are written past the checker. They
/// pin the interpreter's rule (the Rust Reference), which the generator relies on where it is observable.
fn raw_program(mut body: Vec<Stmt>, result: Expr, store: Option<Store>) -> Program {
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
    Program {
        seed: 0,
        methods: methods(),
        store,
        attempt: ARRAY_ATTEMPT_BASE,
        count: 1,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: vec![0],
        }],
        body,
        result,
    }
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

/// `let v1 = [c.tick(), c.tick(), c.tick()]; out = v1[0] * 100 + v1[1] * 10 + v1[2]`: 6, 7, 8.
fn literal_order_program() -> Program {
    program(
        vec![let_arr(1, lit(vec![tick(), tick(), tick()]))],
        digits(1),
        None,
    )
}

/// `let v1 = [c.tick(); 3]; out = ...`: one call, copied.
fn repeat_program() -> Program {
    program(
        vec![let_arr(
            1,
            Expr::Repeat {
                ty: arr(),
                e: Box::new(tick()),
            },
        )],
        digits(1),
        None,
    )
}

/// `let mut v1 = [0; 3]; v1[c.tick() % 3] = c.tick(); out = ...`: the value (6) is computed before the index (7 % 3).
/// (The generator never writes this: a `&mut self` call in an assignee place is E0421. It pins the interpreter.)
fn set_order_program() -> Program {
    raw_program(
        vec![
            let_arr(1, lit(vec![u(0), u(0), u(0)])),
            Stmt::Set {
                place: at(1, Expr::bin(BinOp::Rem, tick(), u(3))),
                ty: Ty::U32,
                op: None,
                value: tick(),
            },
        ],
        digits(1),
        None,
    )
}

/// The final store `out[c.tick() - 7] = c.tick()`: the result is computed first (6), then the index (7 - 7 = 0, the
/// element's own slot). Index first would be `6 - 7`, an overflow.
fn store_order_program() -> Program {
    raw_program(
        Vec::new(),
        tick(),
        Some(Store {
            index: Expr::bin(BinOp::Sub, tick(), u(7)),
            op: None,
            init: 0,
        }),
    )
}

/// `let mut v1 = [1, 2, 3]; let v2 = fill(v1); out = v1[0] * 1000 + v2[0]`: the argument is a copy.
fn by_value_program() -> Program {
    program(
        vec![
            let_arr(1, lit(vec![u(1), u(2), u(3)])),
            Stmt::Let {
                var: 2,
                ty: arr(),
                mutable: false,
                init: Expr::Call {
                    f: 1,
                    owner: None,
                    recv: None,
                    args: vec![Expr::Var(1, arr())],
                    ty: arr(),
                },
            },
        ],
        add(mul(read(1, 0), u(1000)), read(2, 0)),
        None,
    )
}

/// `let mut v1 = [1, 2, 3]; let mut v2 = v1; v2[0] = 9; out = v1[0] * 10 + v2[0]`: a `let` copies.
fn copy_program() -> Program {
    program(
        vec![
            let_arr(1, lit(vec![u(1), u(2), u(3)])),
            let_arr(2, Expr::Var(1, arr())),
            Stmt::Set {
                place: at(2, u(0)),
                ty: Ty::U32,
                op: None,
                value: u(9),
            },
        ],
        add(mul(read(1, 0), u(10)), read(2, 0)),
        None,
    )
}

/// `let mut v1 = [1, 2, 3]; v1[c.tick() % 3] += c.tick() * 10; out = ...`: the value (60) first, then the index
/// (7 % 3 = 1), then the read of the place (2): `v1[1] = 62`.
fn compound_program() -> Program {
    raw_program(
        vec![
            let_arr(1, lit(vec![u(1), u(2), u(3)])),
            Stmt::Set {
                place: at(1, Expr::bin(BinOp::Rem, tick(), u(3))),
                ty: Ty::U32,
                op: Some(BinOp::Add),
                value: mul(tick(), u(10)),
            },
        ],
        digits(1),
        None,
    )
}

#[test]
fn an_array_literal_runs_its_elements_left_to_right() {
    assert_eq!(value_of(&literal_order_program()), 678);
}

#[test]
fn a_repeat_evaluates_its_element_once() {
    assert_eq!(value_of(&repeat_program()), 666);
}

#[test]
fn an_indexed_set_a_compound_set_and_the_output_store_evaluate_the_value_before_the_index() {
    // Value 6 into element 7 % 3 = 1.
    assert_eq!(value_of(&set_order_program()), 60);
    assert_eq!(value_of(&store_order_program()), 6);
    // A compound set: the value 6 * 10 = 60, then the index 7 % 3 = 1, then the read of v1[1] = 2: 62.
    assert_eq!(value_of(&compound_program()), 100 + 620 + 3);
}

#[test]
fn an_array_argument_is_a_copy() {
    assert_eq!(value_of(&by_value_program()), 1000 + 99);
}

#[test]
fn a_let_of_an_array_is_a_copy() {
    assert_eq!(value_of(&copy_program()), 19);
}

/// The vectors a mutant changes, with the value each then gives (`None`: the run is refused).
type Changes = &'static [(&'static str, Option<u32>)];

#[test]
fn the_mutants_change_exactly_the_vectors_that_are_meant_to_catch_them() {
    let programs: [(&str, Program); 7] = [
        ("literal order", literal_order_program()),
        ("repeat", repeat_program()),
        ("set order", set_order_program()),
        ("store order", store_order_program()),
        ("by value", by_value_program()),
        ("copy", copy_program()),
        ("compound", compound_program()),
    ];
    // (mutant, the vectors it changes with the value it gives, or None when the run is refused).
    let expected: [(Mutant, Changes); 4] = [
        (Mutant::ArrayLitRightToLeft, &[("literal order", Some(876))]),
        (Mutant::RepeatPerElement, &[("repeat", Some(678))]),
        (
            Mutant::IndexBeforeValue,
            // `v1[6 % 3 = 0] = 7`; the store index `6 - 7` overflows; the compound adds 7 * 10 to `v1[6 % 3 = 0]`.
            &[
                ("set order", Some(700)),
                ("store order", None),
                ("compound", Some(7123)),
            ],
        ),
        (Mutant::ArrayArgByRef, &[("by value", Some(99 * 1000 + 99))]),
    ];
    for (mutant, changed) in expected {
        for (name, p) in &programs {
            let reference = run(p).unwrap()[0];
            let mutated = run_mutant(p, mutant).map(|run| run.values[0]);
            match changed.iter().find(|(n, _)| n == name) {
                Some((_, Some(value))) => {
                    assert_eq!(mutated.as_ref().ok(), Some(value), "{mutant:?} on `{name}`");
                    assert_ne!(reference, *value);
                }
                Some((_, None)) => assert!(mutated.is_err(), "{mutant:?} on `{name}`"),
                None => assert_eq!(
                    mutated.as_ref().ok(),
                    Some(&reference),
                    "{mutant:?} changes `{name}`"
                ),
            }
        }
    }
}

#[test]
fn the_emitted_text_of_the_hand_programs_has_the_array_forms() {
    let text = emit_module(&literal_order_program());
    assert!(text.contains(": [u32; 3] = ["), "{text}");
    assert!(emit_module(&repeat_program()).contains("; 3]"));
    assert!(emit_module(&set_order_program()).contains("as usize]"));
    let text = emit_module(&by_value_program());
    assert!(text.contains("mut v20: [uint; 3]"), "{text}");
}

/// The array programs of the 64-seed batch (two per seed) write and run every form, and every order shape. The counts
/// are printed (`ARRAY-FORMS`) for DIFFTEST.md.
#[test]
fn the_array_programs_of_the_64_seed_batch_write_and_run_every_form_and_order_shape() {
    use crate::ast;
    use std::collections::BTreeMap;
    let mut written: BTreeMap<String, usize> = BTreeMap::new();
    let mut executed: BTreeMap<String, u64> = BTreeMap::new();
    let mut with_form: BTreeMap<String, usize> = BTreeMap::new();
    let mut programs = 0;
    for seed in 1..=64 {
        for p in &generate_batch(seed).unwrap().programs[VALUE_END..ARRAY_END] {
            assert!(p.attempt >= ARRAY_ATTEMPT_BASE && p.has_array());
            programs += 1;
            for (name, n) in operator_counts(p) {
                *with_form.entry(name.clone()).or_default() += 1;
                *written.entry(name).or_default() += n;
            }
            let (_, coverage) = run_covered(p).unwrap();
            for (name, n) in coverage.operators {
                *executed.entry(name).or_default() += n;
            }
        }
    }
    assert_eq!(programs, 128);
    let wanted = [
        ast::ARRAY_LIT,
        ast::ARRAY_REPEAT,
        ast::ARRAY_READ,
        ast::ARRAY_LEN,
        ast::ARRAY_SET,
        ast::ARRAY_SET_OP,
        ast::ARRAY_COPY,
        ast::ARRAY_ARG,
        ast::ARRAY_ELEM_CALL,
        ast::ARRAY_STORE,
        ast::SHAPE_ARRAY_LIT_ROOT,
        ast::SHAPE_ARRAY_REPEAT_CALL,
        ast::SHAPE_ARRAY_INDEX_CALLS,
        ast::SHAPE_ARRAY_SET_CALL,
        ast::SHAPE_ARRAY_STORE_CALL,
    ];
    for name in wanted {
        eprintln!(
            "ARRAY-FORMS {name}: programs {} written {} executed {}",
            with_form.get(name).copied().unwrap_or(0),
            written.get(name).copied().unwrap_or(0),
            executed.get(name).copied().unwrap_or(0)
        );
    }
    for name in wanted {
        assert!(
            written.get(name).copied().unwrap_or(0) > 0
                && executed.get(name).copied().unwrap_or(0) > 0,
            "`{name}` is not written and executed by the array programs of the 64-seed batch"
        );
    }
}

#[test]
fn a_batch_ends_with_its_array_programs() {
    for seed in 1..=4 {
        let batch = generate_batch(seed).unwrap();
        for p in &batch.programs[VALUE_END..ARRAY_END] {
            assert!(p.has_array() && p.attempt >= ARRAY_ATTEMPT_BASE);
        }
        let space = generate_batch_in(Space::Array, seed).unwrap();
        assert_eq!(&space.programs[..2], &batch.programs[VALUE_END..ARRAY_END]);
    }
}

/// Whether `value` is the operand `Gen::non_identity` makes for `op` (an operand that is never the identity of `op`).
/// `|= &= %=` have no such operand: the generator gives them a literal index in the hole.
fn is_non_identity_operand(op: BinOp, value: &Expr) -> bool {
    let lit = |e: &Expr, want: u32| matches!(e, Expr::Lit(_, n) if *n == want);
    match (op, value) {
        (
            BinOp::Add | BinOp::Sub | BinOp::BitXor,
            Expr::Bin {
                op: BinOp::BitOr,
                r,
                ..
            },
        ) => lit(r, 1),
        (
            BinOp::Shl | BinOp::Shr,
            Expr::Bin {
                op: BinOp::BitOr,
                l,
                r,
                ..
            },
        ) => lit(r, 1) && matches!(&**l, Expr::Bin { op: BinOp::BitAnd, r, .. } if lit(r, 31)),
        (
            BinOp::Mul | BinOp::Div,
            Expr::Bin {
                op: BinOp::BitOr,
                r,
                ..
            },
        ) => lit(r, 2),
        _ => false,
    }
}

/// The two uniformity sets of `Gen::note_uniform`, mirrored: `wide` (a repeat, equal elements or any non-literal element,
/// a copy of a wide array, a call result) for scalar compound sets, `narrow` (a repeat, equal elements, a copy of a narrow
/// array, a call result) for element method calls.
fn note_uniform(var: u32, init: &Expr, wide: &mut Vec<u32>, narrow: &mut Vec<u32>) {
    wide.retain(|v| *v != var);
    narrow.retain(|v| *v != var);
    let (n, w) = match init {
        Expr::Repeat { .. } => (true, true),
        Expr::ArrayLit { elems, .. } => {
            let equal = elems.windows(2).all(|w| w[0] == w[1]);
            (
                equal,
                equal || elems.iter().any(|e| !matches!(e, Expr::Lit(..))),
            )
        }
        Expr::Var(src, _) => (narrow.contains(src), wide.contains(src)),
        _ => (true, true),
    };
    if n {
        narrow.push(var);
    }
    if w {
        wide.push(var);
    }
}

/// The guard for the NVIDIA hole (plans/m9/arrbug.md, D168), by the rules of `Gen::note_uniform` (mirrored here). In a
/// local array of length other than 2 or 4 that may hold one value in every element (the wide set): a compound set at a
/// dynamic index is `+= -= ^= *= /= <<= >>=` with an operand that is never the identity of its operator; `|= &= %=` have
/// a literal index. A `&mut self` call statement on a struct element of an array of the narrow set has a literal index,
/// and a struct array of the wide set but not the narrow one (distinct non-literal elements) keeps a dynamic index on
/// element calls: the scan must see that at least once, so the narrow rule is not secretly the wide one (R2 of fix round
/// 2). Scans the top-level statements of the array space of seeds 1 to 64.
#[test]
fn no_generated_program_gives_the_driver_hole_an_identity_operand_or_a_call_at_a_dynamic_index() {
    let (guarded, _, _, free_calls) = scan_driver_hole(Space::Array);
    assert!(
        guarded > 5,
        "{guarded} guarded dynamic compound sets in seeds 1-64"
    );
    assert!(
        free_calls > 0,
        "no dynamic element call on a distinct-element array of length 1, 3 or 5"
    );
}

/// The scan of the test above over the top-level statements of `space` (seeds 1 to 64): (guarded dynamic compound
/// sets, literal-index `|= &= %=`, guarded element calls, dynamic element calls on non-narrow arrays). It asserts
/// the rules and returns the counts.
pub(crate) fn scan_driver_hole(space: Space) -> (usize, usize, usize, usize) {
    let (mut guarded, mut literal_ops, mut calls, mut free_calls) = (0, 0, 0, 0);
    for seed in 1..=64 {
        for p in generate_batch_in(space, seed).unwrap().programs {
            let (mut wide, mut narrow): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
            let mut arrays: Vec<u32> = Vec::new();
            for stmt in &p.body {
                match stmt {
                    Stmt::Let {
                        var,
                        ty: Ty::Array(_),
                        init,
                        ..
                    } => {
                        arrays.push(*var);
                        note_uniform(*var, init, &mut wide, &mut narrow);
                    }
                    Stmt::Assign { var, value } if arrays.contains(var) => {
                        note_uniform(*var, value, &mut wide, &mut narrow)
                    }
                    Stmt::Set {
                        place,
                        op: Some(op),
                        value,
                        ..
                    } => {
                        if let (crate::ast::Root::Var(v), Ty::Array(aty), Some(index)) =
                            (place.root, place.root_ty, place.index.as_deref())
                            && wide.contains(&v)
                            && !matches!(aty.n, 2 | 4)
                        {
                            if matches!(op, BinOp::BitOr | BinOp::BitAnd | BinOp::Rem) {
                                literal_ops += 1;
                                assert!(
                                    matches!(index, Expr::Lit(..)),
                                    "seed {seed} attempt {}: {op:?} at a dynamic index",
                                    p.attempt
                                );
                            } else if !matches!(index, Expr::Lit(..)) {
                                guarded += 1;
                                assert!(
                                    is_non_identity_operand(*op, value),
                                    "seed {seed} attempt {}: {op:?} {value:?}",
                                    p.attempt
                                );
                            }
                        }
                    }
                    Stmt::Call {
                        recv: Some(place), ..
                    } => {
                        if let (crate::ast::Root::Var(v), Ty::Array(aty), Some(index)) =
                            (place.root, place.root_ty, place.index.as_deref())
                            && !matches!(aty.n, 2 | 4)
                        {
                            if narrow.contains(&v) {
                                calls += 1;
                                assert!(
                                    matches!(index, Expr::Lit(..)),
                                    "seed {seed} attempt {}",
                                    p.attempt
                                );
                            } else if !matches!(index, Expr::Lit(..)) {
                                free_calls += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    eprintln!(
        "GUARD-SCAN dynamic compound sets {guarded}, literal-index |= &= %= {literal_ops}, guarded element calls {calls}, dynamic element calls on non-narrow arrays {free_calls}"
    );
    (guarded, literal_ops, calls, free_calls)
}

/// Fix round 1 (S2): the order mutants change generated programs, not only the hand vectors, so a GPU lowering that
/// ran the same order wrong would show in a generated run. Over the array space of seeds 1 to 64 (1024 programs, 16
/// per seed) each of the four mutants that an accepted program can show changes at least `MIN_CHANGED_*` programs; the
/// 64-seed batch holds the first two array programs of each seed, and the number it changes is printed
/// (`ARRAY-MUTANT`; at least one each, a small count that moves with the draw). `IndexBeforeValue` is unobservable in an accepted
/// program (E0421 refuses the only case where the index could see the value), so it is not asked of generated programs.
/// The order shapes have minima over the batch (programs that write the shape, evaluations that run it).
#[test]
fn the_order_mutants_change_generated_programs_and_the_order_shapes_keep_their_minima() {
    use std::collections::BTreeMap;
    let mut space = Vec::new();
    let mut batch = Vec::new();
    for seed in 1..=64 {
        space.extend(generate_batch_in(Space::Array, seed).unwrap().programs);
        batch.extend(
            generate_batch(seed)
                .unwrap()
                .programs
                .into_iter()
                .skip(VALUE_END)
                .take(ARRAY_PROGRAMS_PER_SEED),
        );
    }
    assert_eq!((space.len(), batch.len()), (1024, 128));
    // A mutant run that ends in a `Violation` counts as a change (a different outcome from the right run). For the
    // place-first mutant the program must also execute `SHAPE_ARRAY_SET_CALL` (an element set with a call on the same
    // array in the value): the enum and method programs of the space have same-root sets that the mutant changes too,
    // and those do not show the array rule (R3 of fix round 2).
    let changes = |p: &crate::ast::Program, mutant| {
        let differs = run(p).ok() != run_mutant(p, mutant).ok().map(|r| r.values);
        differs
            && (mutant != Mutant::CompoundPlaceFirst
                || run_covered(p).is_ok_and(|(_, coverage)| {
                    coverage
                        .operators
                        .get(crate::ast::SHAPE_ARRAY_SET_CALL)
                        .copied()
                        .unwrap_or(0)
                        > 0
                }))
    };
    for (mutant, minimum, in_batch_floor) in [
        (
            Mutant::ArrayLitRightToLeft,
            MIN_CHANGED_LIT,
            BATCH_CHANGED_LIT,
        ),
        (
            Mutant::RepeatPerElement,
            MIN_CHANGED_REPEAT,
            BATCH_CHANGED_REPEAT,
        ),
        (Mutant::ArrayArgByRef, MIN_CHANGED_ARG, BATCH_CHANGED_ARG),
        (
            Mutant::CompoundPlaceFirst,
            MIN_CHANGED_PLACE_FIRST,
            BATCH_CHANGED_PLACE_FIRST,
        ),
    ] {
        let in_space = space.iter().filter(|p| changes(p, mutant)).count();
        let in_batch = batch.iter().filter(|p| changes(p, mutant)).count();
        eprintln!(
            "ARRAY-MUTANT {mutant:?}: changes {in_space} of 1024 space programs, {in_batch} of 128 batch programs"
        );
        assert!(
            in_space >= minimum,
            "{mutant:?} changes {in_space} space programs"
        );
        assert!(
            in_batch >= in_batch_floor,
            "{mutant:?} changes {in_batch} programs of the 64-seed batch"
        );
    }
    let (mut written, mut executed): (BTreeMap<&str, usize>, BTreeMap<&str, u64>) =
        Default::default();
    for p in &batch {
        let counts = operator_counts(p);
        let (_, coverage) = run_covered(p).unwrap();
        for (name, _) in SHAPE_MINIMA {
            if counts.get(*name).copied().unwrap_or(0) > 0 {
                *written.entry(name).or_default() += 1;
            }
            *executed.entry(name).or_default() +=
                coverage.operators.get(*name).copied().unwrap_or(0);
        }
    }
    for (name, (programs, evaluations)) in SHAPE_MINIMA {
        let (w, e) = (
            written.get(name).copied().unwrap_or(0),
            executed.get(name).copied().unwrap_or(0),
        );
        eprintln!("ARRAY-SHAPE {name}: {w} programs, {e} evaluations");
        assert!(
            w >= *programs && e >= *evaluations,
            "{name}: {w} programs, {e} evaluations"
        );
    }
}

const MIN_CHANGED_LIT: usize = 5;
const MIN_CHANGED_REPEAT: usize = 5;
const MIN_CHANGED_ARG: usize = 10;
const MIN_CHANGED_PLACE_FIRST: usize = 13;
/// The same mutants over the 128 programs of the batch (a regression pin of what was measured: `RepeatPerElement` is 0 there,
/// so a GPU run of the batch does not observe that rule by itself).
const BATCH_CHANGED_LIT: usize = 1;
const BATCH_CHANGED_REPEAT: usize = 0;
const BATCH_CHANGED_ARG: usize = 2;
const BATCH_CHANGED_PLACE_FIRST: usize = 1;
/// (shape row, minimum programs that write it, minimum evaluations) over the 64-seed batch.
const SHAPE_MINIMA: &[(&str, (usize, u64))] = &[
    (crate::ast::SHAPE_ARRAY_LIT_ROOT, (3, 363)),
    (crate::ast::SHAPE_ARRAY_REPEAT_CALL, (2, 2194)),
    (crate::ast::SHAPE_ARRAY_INDEX_CALLS, (5, 254)),
    (crate::ast::SHAPE_ARRAY_SET_CALL, (1, 63)),
    (crate::ast::SHAPE_ARRAY_STORE_CALL, (7, 514)),
];
