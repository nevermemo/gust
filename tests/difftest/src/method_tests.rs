//! Tests of the method programs (format 8, slice m9.d1a): the interpreter's evaluation order on hand-built programs
//! (written from the Rust Reference, not from the emitted Slang), the emitted text of the forms, and the properties of
//! the generated method space (acyclic calls, unique names, no form the checker refuses).

use crate::ast::{
    BinOp, Expr, FnDef, InputBuf, Methods, Place, Program, Recv, Stmt, StructDef, Ty,
    operator_counts, walk_expr, walk_stmts,
};
use crate::emit::emit_module;
use crate::generator::{
    METHOD_ATTEMPT_BASE, PROGRAMS_PER_SEED, Space, generate_batch, generate_batch_in,
    space_of_attempt,
};
use crate::interp::{run, static_violation};

const S0: Ty = Ty::Struct(0);

fn u(v: u32) -> Expr {
    Expr::u32(v)
}

fn local() -> Place {
    Place::var(0, S0)
}

/// `c.n`, the only field of `S0 { n: u32 }`, as a read.
fn field(place: &Place) -> Expr {
    Expr::Read {
        place: place.field(0),
        ty: Ty::U32,
    }
}

fn call(f: usize, recv: Place, args: Vec<Expr>) -> Expr {
    Expr::Call {
        f,
        owner: None,
        recv: Some(recv),
        args,
        ty: Ty::U32,
    }
}

/// The functions of the hand programs, over `S0 { n: u32 }`:
/// m0 `&mut self` `bump(by)`: `self.n += by; self.n`; m1 `self` `plus(x)`: `self.n + x`; m2 `&self` `twice()`: `self.n *
/// 2`; m3 `&mut self` `bump_twice()`: `self.bump(2)` as a statement (a call through `self`, written back to the caller).
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
                params: vec![(10, Ty::U32)],
                ret: Some(Ty::U32),
                body: vec![Stmt::Set {
                    place: this.field(0),
                    ty: Ty::U32,
                    op: Some(BinOp::Add),
                    value: Expr::Var(10, Ty::U32),
                }],
                tail: Some(field(&this)),
            },
            FnDef {
                owner: Some(0),
                recv: Recv::Value,
                params: vec![(11, Ty::U32)],
                ret: Some(Ty::U32),
                body: Vec::new(),
                tail: Some(Expr::bin(BinOp::Add, field(&this), Expr::Var(11, Ty::U32))),
            },
            FnDef {
                owner: Some(0),
                recv: Recv::Shared,
                params: Vec::new(),
                ret: Some(Ty::U32),
                body: Vec::new(),
                tail: Some(Expr::bin(BinOp::Mul, field(&this), u(2))),
            },
            FnDef {
                owner: Some(0),
                recv: Recv::Mut,
                params: Vec::new(),
                ret: None,
                body: vec![Stmt::Call {
                    f: 0,
                    owner: None,
                    recv: Some(this),
                    args: vec![u(2)],
                }],
                tail: None,
            },
        ],
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
            },
        },
    );
    let program = Program {
        seed: 0,
        methods: methods(),
        attempt: METHOD_ATTEMPT_BASE,
        count: 1,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: vec![0],
        }],
        body,
        result,
    };
    program.check().unwrap();
    program
}

fn value_of(program: &Program) -> u32 {
    run(program).unwrap_or_else(|f| panic!("{f:?}"))[0]
}

#[test]
fn the_left_operand_is_read_before_a_mutating_call_on_its_right() {
    // Rust: `c.n + c.bump(1)` reads `c.n` (5) first, the call then makes it 6 and returns 6: 11. Evaluating the call
    // first would give 12.
    let result = Expr::bin(BinOp::Add, field(&local()), call(0, local(), vec![u(1)]));
    assert_eq!(value_of(&program(Vec::new(), result)), 11);
    // The other way round the call comes first: 6 + 6.
    let result = Expr::bin(BinOp::Add, call(0, local(), vec![u(1)]), field(&local()));
    assert_eq!(value_of(&program(Vec::new(), result)), 12);
}

#[test]
fn the_value_of_an_assignment_is_evaluated_before_the_place_is_read_or_written() {
    // `c.n += c.bump(2)`: for primitives the right operand is evaluated first (Reference, compound assignment): the
    // call leaves 7 and returns 7, then the place is read (7) and 7 + 7 is stored.
    let body = vec![Stmt::Set {
        place: local().field(0),
        ty: Ty::U32,
        op: Some(BinOp::Add),
        value: call(0, local(), vec![u(2)]),
    }];
    assert_eq!(value_of(&program(body, field(&local()))), 14);
    // `c.n = c.bump(2) + 1`: the store comes last: 8.
    let body = vec![Stmt::Set {
        place: local().field(0),
        ty: Ty::U32,
        op: None,
        value: Expr::bin(BinOp::Add, call(0, local(), vec![u(2)]), u(1)),
    }];
    assert_eq!(value_of(&program(body, field(&local()))), 8);
}

#[test]
fn a_by_value_receiver_is_copied_before_the_arguments_run() {
    // `c.plus(c.bump(1))`: `self` is a copy of `c` taken first (n = 5); the argument makes `c.n` 6 and is 6: 5 + 6.
    let result = call(1, local(), vec![call(0, local(), vec![u(1)])]);
    assert_eq!(value_of(&program(Vec::new(), result)), 11);
}

#[test]
fn let_copies_a_struct_and_a_mutating_call_writes_back_to_its_own_local() {
    // `let mut d = c; d.bump(3);` leaves `c` alone: 5 * 100 + 8.
    let body = vec![
        Stmt::Let {
            var: 1,
            ty: S0,
            mutable: true,
            init: Expr::Var(0, S0),
        },
        Stmt::Call {
            f: 0,
            owner: None,
            recv: Some(Place::var(1, S0)),
            args: vec![u(3)],
        },
    ];
    let result = Expr::bin(
        BinOp::Add,
        Expr::bin(BinOp::Mul, field(&local()), u(100)),
        field(&Place::var(1, S0)),
    );
    assert_eq!(value_of(&program(body, result)), 508);
}

#[test]
fn a_mutating_call_through_self_is_written_back_to_the_caller() {
    // `c.bump_twice()` calls `self.bump(2)`: the receiver is `c`, so `c.n` is 7 afterwards; `twice` reads it: 14.
    let body = vec![Stmt::Call {
        f: 3,
        owner: None,
        recv: Some(local()),
        args: Vec::new(),
    }];
    assert_eq!(value_of(&program(body, call(2, local(), Vec::new()))), 14);
}

#[test]
fn the_statements_and_calls_are_counted_by_name() {
    let body = vec![Stmt::Set {
        place: local().field(0),
        ty: Ty::U32,
        op: Some(BinOp::Add),
        value: call(0, local(), vec![u(2)]),
    }];
    let counts = operator_counts(&program(body, call(2, local(), Vec::new())));
    for name in ["set +=", "call &mut self", "call &self", "struct literal"] {
        assert!(
            counts.keys().any(|k| k.starts_with(name)),
            "no `{name}` in {counts:?}"
        );
    }
    let covered = crate::interp::run_covered(&program(Vec::new(), call(1, local(), vec![u(0)])))
        .unwrap()
        .1;
    assert!(
        covered.operators.contains_key("call self"),
        "{:?}",
        covered.operators
    );
}

#[test]
fn the_emitted_text_has_the_struct_the_impl_and_the_calls_in_rust_order() {
    let body = vec![Stmt::Set {
        place: local().field(0),
        ty: Ty::U32,
        op: Some(BinOp::Add),
        value: call(0, local(), vec![u(2)]),
    }];
    let result = Expr::bin(BinOp::Add, field(&local()), call(2, local(), Vec::new()));
    let text = emit_module(&program(body, result));
    for wanted in [
        "struct S0",
        "impl S0",
        "pub fn m0(&mut self, v10: uint) -> uint",
        "pub fn m1(self, v11: uint) -> uint",
        "pub fn m2(&self) -> uint",
        "pub fn m3(&mut self)",
        "let mut v0: S0 = S0 { f0: 5u32 };",
        "v0.f0 += v0.m0(2u32);",
        "v0.f0 + v0.m2()",
    ] {
        assert!(text.contains(wanted), "no `{wanted}` in:\n{text}");
    }
}

/// The batches of the first seeds, method part only (16 programs each).
fn method_programs(seeds: std::ops::RangeInclusive<u64>) -> Vec<Program> {
    seeds
        .flat_map(|seed| generate_batch_in(Space::Method, seed).unwrap().programs)
        .collect()
}

#[test]
fn the_method_space_is_selected_by_its_attempt_numbers_only() {
    assert_eq!(space_of_attempt(0), Space::Float);
    assert_eq!(space_of_attempt(METHOD_ATTEMPT_BASE - 1), Space::Vector);
    assert_eq!(space_of_attempt(METHOD_ATTEMPT_BASE), Space::Method);
    for seed in 1..=3 {
        let batch = generate_batch(seed).unwrap();
        assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
        assert!(
            batch.method_attempts >= 3
                && batch.method_attempts == 3 + batch.method_discarded.values().sum::<u32>() as u64,
            "seed {seed}: {} attempts, {:?}",
            batch.method_attempts,
            batch.method_discarded
        );
    }
}

#[test]
fn every_method_program_is_well_typed_in_the_domain_and_calls_from_the_kernel() {
    for p in method_programs(1..=12) {
        p.check().unwrap_or_else(|e| panic!("seed {} {e}", p.seed));
        assert!(run(&p).is_ok());
        assert!(static_violation(&p).is_none());
        assert!(p.has_call(), "seed {} attempt {}", p.seed, p.attempt);
        assert!(!p.has_float() && !p.has_aggregate());
        assert!(p.reads_input());
    }
}

#[test]
fn calls_only_go_to_lower_functions_and_every_name_is_unique() {
    for p in method_programs(1..=12) {
        for (k, def) in p.methods.fns.iter().enumerate() {
            let mut callees = Vec::new();
            let mut note = |e: &Expr| {
                if let Expr::Call { f, .. } = e {
                    callees.push(*f);
                }
            };
            walk_stmts(&def.body, &mut note);
            if let Some(tail) = &def.tail {
                walk_expr(tail, &mut note);
            }
            stmt_callees(&def.body, &mut callees);
            assert!(callees.iter().all(|&f| f < k), "m{k} calls {callees:?}");
        }
        let text = emit_module(&p);
        for k in 0..p.methods.fns.len() {
            assert_eq!(text.matches(&format!("fn m{k}(")).count(), 1, "m{k}");
        }
        for j in 0..p.methods.structs.len() {
            assert_eq!(text.matches(&format!("struct S{j} ")).count(), 1, "S{j}");
        }
    }
}

fn stmt_callees(stmts: &[Stmt], out: &mut Vec<usize>) {
    for stmt in stmts {
        match stmt {
            Stmt::Call { f, .. } => out.push(*f),
            Stmt::If { then, els, .. } => {
                stmt_callees(then, out);
                stmt_callees(els, out);
            }
            Stmt::For { body, .. } => stmt_callees(body, out),
            _ => {}
        }
    }
}

/// The methods and positions the corpus of the first seeds reaches: this fails if a form becomes unreachable.
#[test]
fn the_method_programs_of_the_first_seeds_write_every_kind_of_call_and_assignment() {
    let mut names = std::collections::BTreeSet::new();
    for p in method_programs(1..=12) {
        names.extend(operator_counts(&p).into_keys());
    }
    for wanted in [
        "call &mut self",
        "call &self",
        "call self",
        "call assoc",
        "call helper",
        "set =",
        "set +=",
    ] {
        assert!(names.contains(wanted), "no `{wanted}` in {names:?}");
    }
}
