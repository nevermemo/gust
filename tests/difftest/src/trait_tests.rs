//! Tests of the trait programs (format 13, slice m9.d2): the interpreter's rules for trait calls and generic helpers on
//! hand-built programs (written from the Rust Reference, not from the emitted Slang), the mutation checks, and the
//! properties of the generated trait space.
//!
//! The rules (Rust Reference, method-call and call expressions, trait objects excluded): a call `s.m(a)` of a trait
//! method runs the impl of the receiver's concrete type; a generic function is specialised per call, its type parameter
//! bound to the type of the argument; the arguments run left to right, each completely, then the body runs; a `self`
//! or `T` taken by value is copied before a later argument runs; a `&mut self` method's writes are visible afterwards.

use crate::ast::{
    BinOp, Expr, FnDef, GenParam, Imp, InputBuf, Methods, Place, Program, Recv, Stmt, StructDef,
    TRet, TraitDef, TraitMethod, Ty, operator_counts,
};
use crate::ast::{GENERIC_CALL, SHAPE_GENERIC_ARGS_ROOT, SHAPE_TRAIT_ARGS_ROOT, TRAIT_CALL};
use crate::generator::{Space, TRAIT_ATTEMPT_BASE, generate_batch_in};
use crate::interp::{Mutant, env_mutant, run, run_covered, run_mutant};

const S0: Ty = Ty::Struct(0);
const S1: Ty = Ty::Struct(1);
const S2: Ty = Ty::Struct(2);
const P0: Ty = Ty::Param(0);

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn add(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Add, l, r)
}

fn mul(l: Expr, r: Expr) -> Expr {
    Expr::bin(BinOp::Mul, l, r)
}

fn field(ty: Ty) -> Expr {
    Expr::Read {
        place: Place::this(ty).field(0),
        ty: Ty::U32,
    }
}

fn imp(tr: u8, ty: Ty) -> Option<Imp> {
    Some(Imp { tr, m: 0, ty })
}

fn call(f: usize, recv: Option<Place>, args: Vec<Expr>) -> Expr {
    Expr::Call {
        f,
        owner: None,
        recv,
        args,
        ty: Ty::U32,
    }
}

fn on(var: u32, ty: Ty) -> Option<Place> {
    Some(Place::var(var, ty))
}

fn method(owner: u8, imp: Option<Imp>, params: Vec<(u32, Ty)>, tail: Expr) -> FnDef {
    FnDef {
        imp,
        gens: Vec::new(),
        owner: Some(owner),
        recv: Recv::Shared,
        params,
        ret: Some(Ty::U32),
        body: Vec::new(),
        tail: Some(tail),
    }
}

fn generic(params: Vec<(u32, Ty)>, body: Vec<Stmt>, tail: Expr) -> FnDef {
    FnDef {
        imp: None,
        gens: vec![GenParam {
            bounds: vec![0],
            copy: false,
            where_clause: false,
        }],
        owner: None,
        recv: Recv::None,
        params,
        ret: Some(Ty::U32),
        body,
        tail: Some(tail),
    }
}

/// Structs `S0 { n }`, `S1 { n }`, a counter `S2 { k }`; `Tr0::t0m0(&self) -> u32` (`S0`: `n * 2`, `S1`: `n + 1000`);
/// `Tr1::t1m0(&self, a, b) -> u32` (`S0`: `n * 1000 + a * 10 + b`, `S1`: `n * 2000 + a * 10 + b`); the helpers
/// `gread<T: Tr0>(t) = { let c: T = t; c.t0m0() }` and `gmix<T: Tr1>(t, a, b) = t.t1m0(a, b)` (the bound of the
/// second is set to `Tr1`).
fn methods() -> Methods {
    let one_field = || StructDef {
        fields: vec![Ty::U32],
    };
    let args = || vec![(20, Ty::U32), (21, Ty::U32)];
    let tr0 = TraitDef {
        methods: vec![TraitMethod {
            recv: Recv::Shared,
            params: Vec::new(),
            ret: TRet::Ty(Ty::U32),
        }],
    };
    let tr1 = TraitDef {
        methods: vec![TraitMethod {
            recv: Recv::Shared,
            params: vec![Ty::U32, Ty::U32],
            ret: TRet::Ty(Ty::U32),
        }],
    };
    let mix_tail = |ty: Ty, w: u32| {
        add(
            add(mul(field(ty), u(w)), mul(Expr::Var(20, Ty::U32), u(10))),
            Expr::Var(21, Ty::U32),
        )
    };
    let mix_tail2 = |ty: Ty, w: u32| {
        add(
            add(mul(field(ty), u(w)), mul(Expr::Var(22, Ty::U32), u(10))),
            Expr::Var(23, Ty::U32),
        )
    };
    let bump = FnDef {
        imp: None,
        gens: Vec::new(),
        owner: Some(2),
        recv: Recv::Mut,
        params: Vec::new(),
        ret: Some(Ty::U32),
        body: vec![Stmt::Set {
            place: Place::this(S2).field(0),
            ty: Ty::U32,
            op: Some(BinOp::Add),
            value: u(1),
        }],
        tail: Some(field(S2)),
    };
    let mut gmix = generic(
        vec![(40, P0), (41, Ty::U32), (42, Ty::U32)],
        Vec::new(),
        call(
            3,
            on(40, P0),
            vec![Expr::Var(41, Ty::U32), Expr::Var(42, Ty::U32)],
        ),
    );
    gmix.gens[0].bounds = vec![1];
    Methods {
        structs: vec![one_field(), one_field(), one_field()],
        enums: Vec::new(),
        traits: vec![tr0, tr1],
        fns: vec![
            bump,
            method(0, imp(0, S0), Vec::new(), mul(field(S0), u(2))),
            method(1, imp(0, S1), Vec::new(), add(field(S1), u(1000))),
            method(0, imp(1, S0), args(), mix_tail(S0, 1000)),
            method(
                1,
                imp(1, S1),
                vec![(22, Ty::U32), (23, Ty::U32)],
                mix_tail2(S1, 2000),
            ),
            // f5: gread
            generic(
                vec![(30, P0)],
                vec![Stmt::Let {
                    var: 31,
                    ty: P0,
                    mutable: false,
                    init: Expr::Var(30, P0),
                }],
                call(1, on(31, P0), Vec::new()),
            ),
            // f6: gmix
            gmix,
        ],
    }
}

/// `let v0 = S0 { n: 5 }; let v1 = S1 { n: 7 }; let mut v2 = S2 { k: 0 }; out[i] = <result>`.
fn program(result: Expr) -> Program {
    let lit = |ty: Ty, n: u32| Expr::StructLit {
        ty,
        fields: vec![u(n)],
        order: vec![0],
    };
    let body = vec![
        Stmt::Let {
            var: 0,
            ty: S0,
            mutable: false,
            init: lit(S0, 5),
        },
        Stmt::Let {
            var: 1,
            ty: S1,
            mutable: false,
            init: lit(S1, 7),
        },
        Stmt::Let {
            var: 2,
            ty: S2,
            mutable: true,
            init: lit(S2, 0),
        },
    ];
    let program = Program {
        seed: 0,
        methods: methods(),
        store: None,
        attempt: TRAIT_ATTEMPT_BASE,
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

fn bump() -> Expr {
    call(0, on(2, S2), Vec::new())
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

/// `value_of` for the argument-order vectors: only the argument-order mutant applies, so that the wrong-impl run fails
/// the dispatch vector alone (the argument vectors call two different impls, which the wrong-impl mutant swaps too).
fn value_of_args(program: &Program) -> u32 {
    match env_mutant() {
        Some(Mutant::TraitArgsRightToLeft) => value_of(program),
        _ => run(program).unwrap_or_else(|f| panic!("{f:?}"))[0],
    }
}

/// `v0.t0m0() + v1.t0m0() * 100`: each receiver runs the impl of its own type.
fn dispatch_program() -> Program {
    program(add(
        call(1, on(0, S0), Vec::new()),
        mul(call(2, on(1, S1), Vec::new()), u(100)),
    ))
}

/// `gread(v0) + gread(v1) * 100`: one helper, two instantiations, each with its own impl.
fn instantiation_program() -> Program {
    program(add(
        call(5, None, vec![Expr::Var(0, S0)]),
        mul(call(5, None, vec![Expr::Var(1, S1)]), u(100)),
    ))
}

/// `v0.t1m0(v2.bump(), v2.bump())`: the arguments run in written order.
fn trait_args_program() -> Program {
    program(call(3, on(0, S0), vec![bump(), bump()]))
}

/// `gmix(v1, v2.bump(), v2.bump())`.
fn generic_args_program() -> Program {
    program(call(6, None, vec![Expr::Var(1, S1), bump(), bump()]))
}

#[test]
fn dispatch_runs_the_impl_of_the_receivers_type() {
    assert_eq!(value_of(&dispatch_program()), 10 + 1007 * 100);
}

#[test]
fn each_instantiation_runs_with_its_own_type() {
    assert_eq!(value_of(&instantiation_program()), 10 + 1007 * 100);
}

#[test]
fn trait_and_generic_arguments_run_left_to_right() {
    assert_eq!(value_of_args(&trait_args_program()), 5 * 1000 + 12);
    assert_eq!(value_of_args(&generic_args_program()), 7 * 2000 + 12);
}

/// Each mutant changes the vectors that pin its rule, and no other.
#[test]
fn the_trait_mutants_change_exactly_their_vectors() {
    let programs: [(&str, Program); 4] = [
        ("dispatch", dispatch_program()),
        ("instantiation", instantiation_program()),
        ("trait args", trait_args_program()),
        ("generic args", generic_args_program()),
    ];
    let reference: [u32; 4] = [100_710, 100_710, 5012, 14_012];
    type Changes = &'static [(&'static str, u32)];
    let expected: [(Mutant, Changes); 3] = [
        (
            Mutant::TraitWrongImpl,
            &[("dispatch", 1005 + 14 * 100), ("trait args", 10_012)],
        ),
        (Mutant::GenericReuse, &[("instantiation", 10 + 14 * 100)]),
        (
            Mutant::TraitArgsRightToLeft,
            &[("trait args", 5021), ("generic args", 14_021)],
        ),
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

/// The counts of the hand vectors are causal: the shape counts in the text and at run time exactly where two
/// `&mut self` calls on one root are the arguments of a trait call or a generic call.
#[test]
fn the_argument_shapes_count_where_calls_on_one_root_are_arguments() {
    let cases = [
        (trait_args_program(), TRAIT_CALL, SHAPE_TRAIT_ARGS_ROOT),
        (
            generic_args_program(),
            GENERIC_CALL,
            SHAPE_GENERIC_ARGS_ROOT,
        ),
    ];
    for (p, construct, shape) in cases {
        let text = operator_counts(&p);
        let (_, coverage) = run_covered(&p).unwrap();
        for name in [construct, shape] {
            assert!(
                text.get(name).copied().unwrap_or(0) > 0,
                "{name} in the text"
            );
            assert!(
                coverage.operators.get(name).copied().unwrap_or(0) > 0,
                "{name} executed"
            );
        }
    }
    // The calls with plain arguments do not count the shapes.
    for p in [dispatch_program(), instantiation_program()] {
        let text = operator_counts(&p);
        assert!(!text.contains_key(SHAPE_TRAIT_ARGS_ROOT));
        assert!(!text.contains_key(SHAPE_GENERIC_ARGS_ROOT));
    }
}

/// The 64-seed trait space: every program has a trait call and a generic call, holds no array (so no trait or generic
/// body holds one, and the D168/D169 guard has nothing to guard), and over the space the generic helpers are called
/// with many different types.
#[test]
fn the_trait_space_has_trait_and_generic_calls_and_no_array() {
    let mut instantiations = std::collections::BTreeSet::new();
    let mut programs = 0;
    for seed in 1..=64 {
        for p in generate_batch_in(Space::Trait, seed).unwrap().programs {
            programs += 1;
            assert!(p.attempt >= TRAIT_ATTEMPT_BASE);
            assert!(p.methods.has_traits() && p.has_trait_forms());
            assert!(!p.has_array() && !p.has_closure() && !p.has_float());
            let counts = operator_counts(&p);
            assert!(counts.contains_key(TRAIT_CALL) && counts.contains_key(GENERIC_CALL));
            crate::ast::visit_exprs(&p, &mut |e| {
                if let Expr::Call { f, args, .. } = e
                    && !p.methods.fns[*f].gens.is_empty()
                {
                    for (a, (_, pty)) in args.iter().zip(&p.methods.fns[*f].params) {
                        if matches!(pty, Ty::Param(_)) {
                            instantiations.insert((seed, *f, a.ty()));
                        }
                    }
                }
            });
        }
    }
    assert_eq!(programs, 64 * 16);
    assert!(
        instantiations.len() > programs / 2,
        "{} instantiations",
        instantiations.len()
    );
}

/// Each mutant changes generated programs, not only the hand vectors (counts printed as `TRAIT-MUTANT`).
#[test]
fn the_trait_mutants_change_generated_programs() {
    let mut space = Vec::new();
    for seed in 1..=64 {
        space.extend(generate_batch_in(Space::Trait, seed).unwrap().programs);
    }
    let changes = |p: &Program, mutant| {
        let mutated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_mutant(p, mutant).ok().map(|r| r.values)
        }));
        mutated.is_ok_and(|values| run(p).ok() != values)
    };
    let mut failures = Vec::new();
    for (mutant, floor) in [
        (Mutant::TraitWrongImpl, 120),
        (Mutant::GenericReuse, 25),
        (Mutant::TraitArgsRightToLeft, 50),
    ] {
        let n = space.iter().filter(|p| changes(p, mutant)).count();
        let first_four = space[..64].iter().filter(|p| changes(p, mutant)).count();
        eprintln!(
            "TRAIT-MUTANT {mutant:?}: changes {n} of {} space programs, {first_four} of the 64 of seeds 1 to 4",
            space.len()
        );
        if n < floor || first_four == 0 {
            failures.push(format!(
                "{mutant:?}: {n} in the space, {first_four} in seeds 1 to 4"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
