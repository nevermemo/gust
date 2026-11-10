//! Test support and tests: the interval method against adversarial evaluations of the same program (R1-4).
//!
//! The interpreter claims that every value a conforming WGSL implementation may hold for the output lies in the
//! interval it returns. This module builds the strongest GPU stand-in that can be written on a CPU and checks the claim
//! against it:
//!
//! - the program is unrolled into one expression of the inputs (variables inlined, loops unrolled), because an
//!   implementation may reassociate through variables (15.7.5);
//! - every maximal `+`/`-` chain is flattened and its terms are added in a random order (the regrouping of 15.7.5);
//!   a term that is a product is sometimes fused with the addition (`fma`, one rounding);
//! - every rounded result is moved to a neighbouring `f32` at random ("either neighbour", SEMANTICS.md:190), a
//!   quotient by up to two (2.5 ULP, SEMANTICS.md:192);
//! - and a second model evaluates in `f64`, with the chains in another order.
//!
//! The twin's own `f32` result must be inside as well. None of this is a GPU measurement.

use std::collections::BTreeMap;

use crate::ast::{BinOp, Expr, InputBuf, Program, Stmt, Ty, UnOp};
use crate::interp::{Violation, run_full};
use crate::prng::Prng;

thread_local! {
    /// When set, every rounding of the GPU model picks the neighbour that is not the nearest one, i.e. the one the
    /// round-to-nearest CPU twin would not pick (an adversarial but conforming choice, SEMANTICS.md:190).
    static AWAY_FROM_THE_TWIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The next `f32` above (`up`) or below `x`, for a normal `x`.
fn next(x: f32, up: bool) -> f32 {
    if x == 0.0 {
        return f32::from_bits(if up { 1 } else { 0x8000_0001 });
    }
    let bits = x.to_bits();
    f32::from_bits(if (x > 0.0) == up { bits + 1 } else { bits - 1 })
}

/// A result "correctly rounded" in the sense of SEMANTICS.md:190: either `f32` neighbour of the exact value `exact`,
/// or the value itself when it is representable. `exact` is an `f64` computed from `f32` operands, so it is exact up
/// to `f64` rounding, far below an `f32` ULP.
fn either_neighbour(exact: f64, rng: &mut Prng) -> f32 {
    let nearest = exact as f32;
    let back = f64::from(nearest);
    if back == exact {
        return nearest;
    }
    let other = next(nearest, back < exact);
    if AWAY_FROM_THE_TWIN.get() {
        return other;
    }
    if rng.below(2) == 0 { nearest } else { other }
}

/// A quotient within 2.5 ULP of the exact value (SEMANTICS.md:192): the nearest `f32` moved by up to two steps.
fn within_two_and_a_half_ulp(exact: f64, rng: &mut Prng) -> f32 {
    let mut value = exact as f32;
    if f64::from(value) == exact {
        return value;
    }
    let steps = if AWAY_FROM_THE_TWIN.get() {
        2
    } else {
        rng.below(3)
    };
    for _ in 0..steps {
        value = next(value, rng.below(2) == 0);
    }
    value
}

/// The result of the program as a single expression of `Input`, `Gid`, literals and the operators: variables
/// inlined, loops unrolled (the loop variable becomes a `u32` literal).
fn unroll(program: &Program) -> Expr {
    fn subst(e: &Expr, env: &BTreeMap<u32, Expr>) -> Expr {
        match e {
            Expr::Var(var, _) => env[var].clone(),
            Expr::Lit(..) | Expr::Gid | Expr::Len(_) => e.clone(),
            Expr::Input { buf, ty, index } => Expr::input(*buf, *ty, subst(index, env)),
            Expr::Un { op, ty, e } => Expr::Un {
                op: *op,
                ty: *ty,
                e: Box::new(subst(e, env)),
            },
            Expr::Bin { op, ty, l, r } => Expr::Bin {
                op: *op,
                ty: *ty,
                l: Box::new(subst(l, env)),
                r: Box::new(subst(r, env)),
            },
            Expr::Cast { to, e } => Expr::cast(*to, subst(e, env)),
            _ => panic!("the soundness programs are scalar: {e:?}"),
        }
    }
    fn exec(stmts: &[Stmt], env: &mut BTreeMap<u32, Expr>) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, init, .. } => {
                    let value = subst(init, env);
                    env.insert(*var, value);
                }
                Stmt::Assign { var, value } => {
                    let value = subst(value, env);
                    env.insert(*var, value);
                }
                Stmt::For { var, trips, body } => {
                    for k in 0..*trips {
                        env.insert(*var, Expr::u32(k));
                        exec(body, env);
                    }
                }
                Stmt::If { .. } => panic!("the soundness programs have no branches"),
                Stmt::AssignComp { .. } => panic!("the soundness programs have no vectors"),
                Stmt::Set { .. }
                | Stmt::Call { .. }
                | Stmt::Match { .. }
                | Stmt::LetTuple { .. }
                | Stmt::LetMatch { .. }
                | Stmt::Yield { .. }
                | Stmt::TailMatch { .. }
                | Stmt::Return { .. } => {
                    panic!("the soundness programs have no methods")
                }
            }
        }
    }
    let mut env = BTreeMap::new();
    exec(&program.body, &mut env);
    subst(&program.result, &env)
}

/// The terms of the maximal `+`/`-` chain at `e`: (negative, term).
fn flatten<'a>(e: &'a Expr, negative: bool, out: &mut Vec<(bool, &'a Expr)>) {
    match e {
        Expr::Bin {
            op: BinOp::Add,
            ty: Ty::F32,
            l,
            r,
        } => {
            flatten(l, negative, out);
            flatten(r, negative, out);
        }
        Expr::Bin {
            op: BinOp::Sub,
            ty: Ty::F32,
            l,
            r,
        } => {
            flatten(l, negative, out);
            flatten(r, !negative, out);
        }
        _ => out.push((negative, e)),
    }
}

fn shuffle<T>(items: &mut [T], rng: &mut Prng) {
    for k in (1..items.len()).rev() {
        items.swap(k, rng.below(k as u32 + 1) as usize);
    }
}

/// A leaf value of the unrolled expression for `element`.
fn leaf(e: &Expr, program: &Program, element: u32) -> Option<f32> {
    match e {
        Expr::Lit(Ty::F32, bits) => Some(f32::from_bits(*bits)),
        Expr::Cast { to: Ty::F32, e } => match **e {
            Expr::Lit(Ty::U32, k) => Some(k as f32),
            _ => panic!("the soundness programs cast only literals"),
        },
        Expr::Input { buf, index, .. } => {
            assert_eq!(**index, Expr::Gid, "the soundness programs index by `i`");
            Some(f32::from_bits(program.inputs[*buf].data[element as usize]))
        }
        _ => None,
    }
}

/// The sum of `values` as a balanced tree, each addition rounded to either neighbour.
fn balanced_sum(values: &[f32], rng: &mut Prng) -> f32 {
    if values.len() == 1 {
        return values[0];
    }
    let (left, right) = values.split_at(values.len() / 2);
    let (a, b) = (balanced_sum(left, rng), balanced_sum(right, rng));
    either_neighbour(f64::from(a) + f64::from(b), rng)
}

/// An adversarial `f32` evaluation of `e` (see the module comment).
fn gpu_model(e: &Expr, program: &Program, element: u32, rng: &mut Prng) -> f32 {
    if let Some(value) = leaf(e, program, element) {
        return value;
    }
    match e {
        Expr::Un {
            op: UnOp::Neg, e, ..
        } => -gpu_model(e, program, element, rng),
        Expr::Bin {
            op: BinOp::Add | BinOp::Sub,
            ty: Ty::F32,
            ..
        } => {
            let mut terms = Vec::new();
            flatten(e, false, &mut terms);
            // Three groupings: a left-deep fold in a random order, a left-deep fold in the written order, and a
            // balanced tree over a random order (15.7.5 allows any regrouping).
            let strategy = rng.below(3);
            if strategy != 1 {
                shuffle(&mut terms, rng);
            }
            let signed = |(negative, term): (bool, &Expr), rng: &mut Prng| {
                let value = gpu_model(term, program, element, rng);
                if negative { -value } else { value }
            };
            if strategy == 2 {
                let values: Vec<f32> = terms.into_iter().map(|t| signed(t, rng)).collect();
                return balanced_sum(&values, rng);
            }
            let mut terms = terms.into_iter();
            let mut acc = signed(terms.next().unwrap(), rng);
            for (negative, term) in terms {
                acc = match term {
                    // A product feeding the sum may be fused: one rounding of `a * b + acc`.
                    Expr::Bin {
                        op: BinOp::Mul,
                        ty: Ty::F32,
                        l,
                        r,
                    } if rng.below(2) == 0 => {
                        let a = gpu_model(l, program, element, rng);
                        let b = gpu_model(r, program, element, rng);
                        let a = if negative { -a } else { a };
                        either_neighbour(f64::from(a).mul_add(f64::from(b), f64::from(acc)), rng)
                    }
                    _ => {
                        let value = signed((negative, term), rng);
                        either_neighbour(f64::from(acc) + f64::from(value), rng)
                    }
                };
            }
            acc
        }
        Expr::Bin {
            op: BinOp::Mul,
            l,
            r,
            ..
        } => {
            let a = gpu_model(l, program, element, rng);
            let b = gpu_model(r, program, element, rng);
            either_neighbour(f64::from(a) * f64::from(b), rng)
        }
        Expr::Bin {
            op: BinOp::Div,
            l,
            r,
            ..
        } => {
            let a = gpu_model(l, program, element, rng);
            let b = gpu_model(r, program, element, rng);
            within_two_and_a_half_ulp(f64::from(a) / f64::from(b), rng)
        }
        other => panic!("not in the soundness subset: {other:?}"),
    }
}

/// An `f64` evaluation of `e` with every chain added in a random order: the exact value up to `f64` rounding.
fn exact_model(e: &Expr, program: &Program, element: u32, rng: &mut Prng) -> f64 {
    if let Some(value) = leaf(e, program, element) {
        return f64::from(value);
    }
    match e {
        Expr::Un {
            op: UnOp::Neg, e, ..
        } => -exact_model(e, program, element, rng),
        Expr::Bin {
            op: BinOp::Add | BinOp::Sub,
            ty: Ty::F32,
            ..
        } => {
            let mut terms = Vec::new();
            flatten(e, false, &mut terms);
            shuffle(&mut terms, rng);
            terms
                .into_iter()
                .map(|(negative, term)| {
                    let value = exact_model(term, program, element, rng);
                    if negative { -value } else { value }
                })
                .sum()
        }
        Expr::Bin {
            op: BinOp::Mul,
            l,
            r,
            ..
        } => exact_model(l, program, element, rng) * exact_model(r, program, element, rng),
        Expr::Bin {
            op: BinOp::Div,
            l,
            r,
            ..
        } => exact_model(l, program, element, rng) / exact_model(r, program, element, rng),
        other => panic!("not in the soundness subset: {other:?}"),
    }
}

fn size(e: &Expr) -> usize {
    e.size()
}

/// Checks `program` (if the interpreter accepts it) against the models; returns whether it was accepted.
fn check_against_models(program: &Program, rng: &mut Prng, rounds: usize) -> bool {
    let Ok(run) = run_full(program) else {
        return false;
    };
    let unrolled = unroll(program);
    for element in 0..program.count {
        let (lo, hi) = run.intervals[element as usize];
        let twin = f64::from(f32::from_bits(run.values[element as usize]));
        assert!(lo <= twin && twin <= hi, "twin {twin} outside [{lo}, {hi}]");
        for round in 0..rounds {
            // Half of the rounds pick, at every rounding, the neighbour away from the twin.
            AWAY_FROM_THE_TWIN.set(round % 2 == 0);
            let adversarial = f64::from(gpu_model(&unrolled, program, element, rng));
            assert!(
                lo <= adversarial && adversarial <= hi,
                "GPU model {adversarial} outside [{lo}, {hi}] (round {round}, element {element}) in {program:#?}"
            );
            AWAY_FROM_THE_TWIN.set(false);
            let exact = exact_model(&unrolled, program, element, rng);
            assert!(
                lo <= exact && exact <= hi,
                "f64 model {exact} outside [{lo}, {hi}] (round {round}, element {element})"
            );
        }
    }
    true
}

const COUNT: usize = 6;

/// Values at the edges of the input set and of the binades: the extremes of the bounded set, one below and one above a
/// power of two, and small terms that a rounding at a binade edge can absorb.
fn edge_values() -> Vec<f32> {
    let mut values = Vec::new();
    for v in [
        2f32.powi(20),
        2f32.powi(-20),
        2f32.powi(20) - 2f32.powi(-4),
        1.0 - 2f32.powi(-24),
        1.0 + 2f32.powi(-23),
        2f32.powi(-26),
        2f32.powi(-30),
        1024.0 * (1.0 - 2f32.powi(-24)),
        3.0,
        0.5,
    ] {
        values.push(v);
        values.push(-v);
    }
    values.push(0.0);
    values
}

fn random_data(rng: &mut Prng) -> Vec<u32> {
    if rng.below(3) == 0 {
        let edges = edge_values();
        return (0..COUNT)
            .map(|_| edges[rng.below(edges.len() as u32) as usize].to_bits())
            .collect();
    }
    (0..COUNT)
        .map(|_| {
            let sign = rng.below(2) << 31;
            let exponent = 127 + rng.below(7) - 3;
            let mantissa = rng.next_u32() & 0x7F_FFFF;
            sign | (exponent << 23) | mantissa
        })
        .collect()
}

struct Builder<'a> {
    rng: &'a mut Prng,
    vars: Vec<u32>,
}

impl Builder<'_> {
    fn leaf(&mut self) -> Expr {
        match self.rng.below(6) {
            0 | 1 => Expr::input(0, Ty::F32, Expr::Gid),
            2 => Expr::input(1, Ty::F32, Expr::Gid),
            3 if !self.vars.is_empty() => {
                let var = self.vars[self.rng.below(self.vars.len() as u32) as usize];
                Expr::Var(var, Ty::F32)
            }
            _ => {
                let value = [0.5, 1.0, 2.0, 3.0, 0.1, 1.5, 7.0][self.rng.below(7) as usize];
                Expr::f32(if self.rng.below(3) == 0 {
                    -value
                } else {
                    value
                })
            }
        }
    }

    fn expr(&mut self, depth: u32) -> Expr {
        if depth == 0 || self.rng.below(5) == 0 {
            return self.leaf();
        }
        match self.rng.below(10) {
            0..=2 => Expr::bin(BinOp::Add, self.expr(depth - 1), self.expr(depth - 1)),
            3 | 4 => Expr::bin(BinOp::Sub, self.expr(depth - 1), self.expr(depth - 1)),
            5..=7 => Expr::bin(BinOp::Mul, self.expr(depth - 1), self.expr(depth - 1)),
            8 => {
                let divisor = match self.rng.below(3) {
                    0 => Expr::f32(-4.0),
                    1 => Expr::f32(3.0),
                    _ => {
                        let e = self.expr(depth - 1);
                        Expr::bin(
                            BinOp::Add,
                            Expr::bin(BinOp::Mul, e.clone(), e),
                            Expr::f32(1.0),
                        )
                    }
                };
                Expr::bin(BinOp::Div, self.expr(depth - 1), divisor)
            }
            _ => Expr::negated(self.expr(depth - 1)),
        }
    }
}

fn random_program(rng: &mut Prng, seed: u64) -> Program {
    let inputs = vec![
        InputBuf {
            ty: Ty::F32,
            data: random_data(rng),
        },
        InputBuf {
            ty: Ty::F32,
            data: random_data(rng),
        },
    ];
    let mut body = Vec::new();
    let mut b = Builder {
        rng,
        vars: Vec::new(),
    };
    let mut next = 0;
    for _ in 0..1 + b.rng.below(3) {
        if b.rng.below(2) == 0 {
            let init = b.expr(3);
            body.push(Stmt::Let {
                var: next,
                ty: Ty::F32,
                mutable: false,
                init,
            });
            b.vars.push(next);
            next += 1;
        } else {
            // A loop accumulator: `let mut acc = e; for k in 0..T { acc = acc +/- term }`.
            let (acc, k) = (next, next + 1);
            next += 2;
            let init = b.expr(2);
            body.push(Stmt::Let {
                var: acc,
                ty: Ty::F32,
                mutable: true,
                init,
            });
            let term = match b.rng.below(3) {
                0 => Expr::cast(Ty::F32, Expr::Var(k, Ty::U32)),
                1 => Expr::bin(
                    BinOp::Mul,
                    Expr::cast(Ty::F32, Expr::Var(k, Ty::U32)),
                    b.expr(1),
                ),
                _ => b.expr(2),
            };
            let op = if b.rng.below(2) == 0 {
                BinOp::Add
            } else {
                BinOp::Sub
            };
            body.push(Stmt::For {
                var: k,
                trips: 1 + b.rng.below(8),
                body: vec![Stmt::Assign {
                    var: acc,
                    value: Expr::bin(op, Expr::Var(acc, Ty::F32), term),
                }],
            });
            b.vars.push(acc);
        }
    }
    let result = b.expr(3);
    Program {
        seed,
        methods: Default::default(),
        attempt: 0,
        count: COUNT as u32,
        inputs,
        body,
        result,
    }
}

#[test]
fn the_interval_contains_the_twin_and_adversarial_gpu_evaluations_of_random_programs() {
    let mut rng = Prng::new(0x5EED_F10A7);
    let (mut accepted, mut attempts, mut with_loop, mut with_div) = (0, 0, 0, 0);
    while attempts < 6000 {
        attempts += 1;
        let program = random_program(&mut rng, attempts);
        program.check().unwrap();
        if size(&unroll(&program)) > 4000 {
            continue;
        }
        if check_against_models(&program, &mut rng, 12) {
            accepted += 1;
            with_loop += usize::from(program.body.iter().any(|s| matches!(s, Stmt::For { .. })));
            let mut divides = false;
            crate::ast::visit_exprs(&program, &mut |e| {
                divides |= matches!(e, Expr::Bin { op: BinOp::Div, .. });
            });
            with_div += usize::from(divides);
        }
    }
    eprintln!(
        "soundness: {accepted} of {attempts} random programs accepted ({with_loop} with a loop, {with_div} with a division)"
    );
    // The test must see a real number of programs, with loops and divisions, not discard its way to a pass.
    assert!(accepted >= 400, "only {accepted} of {attempts} accepted");
    assert!(
        with_loop >= 100,
        "{with_loop} accepted programs with a loop"
    );
    assert!(
        with_div >= 40,
        "{with_div} accepted programs with a division"
    );
}

/// The `evaluate` kernel of `examples/polynomial` (the worked example of SEMANTICS.md): `squared = v * v`,
/// `cubed = squared * v`, `curved = cubed - (2.0 * squared)`, `linear = 0.5 * v`, `(curved + linear) + 1.0`.
fn polynomial(values: &[f32]) -> Program {
    let v = || Expr::input(0, Ty::F32, Expr::Gid);
    let var = |id| Expr::Var(id, Ty::F32);
    let let_ = |id, init| Stmt::Let {
        var: id,
        ty: Ty::F32,
        mutable: false,
        init,
    };
    Program {
        seed: 0,
        methods: Default::default(),
        attempt: 0,
        count: values.len() as u32,
        inputs: vec![InputBuf {
            ty: Ty::F32,
            data: values.iter().map(|x| x.to_bits()).collect(),
        }],
        body: vec![
            let_(0, Expr::bin(BinOp::Mul, v(), v())),
            let_(1, Expr::bin(BinOp::Mul, var(0), v())),
            let_(
                2,
                Expr::bin(
                    BinOp::Sub,
                    var(1),
                    Expr::bin(BinOp::Mul, Expr::f32(2.0), var(0)),
                ),
            ),
            let_(3, Expr::bin(BinOp::Mul, Expr::f32(0.5), v())),
        ],
        result: Expr::bin(
            BinOp::Add,
            Expr::bin(BinOp::Add, var(2), var(3)),
            Expr::f32(1.0),
        ),
    }
}

#[test]
fn the_polynomial_worked_example_lies_inside_its_interval() {
    // u = 2^-24. SEMANTICS.md derives, for |v| <= 1 and the worst grouping, a twin error of at most 15u, a GPU
    // error of at most 31u, and a twin-versus-GPU difference of at most 46u (first order, absolute).
    let u = 2f64.powi(-24);
    let program = polynomial(&[1.0]);
    program.check().unwrap();
    let run = run_full(&program).unwrap();
    let (lo, hi) = run.intervals[0];
    let exact = 0.5; // 1 - 2 + 0.5 + 1
    let twin = f64::from(f32::from_bits(run.values[0]));
    assert_eq!(twin, exact, "the arithmetic at v = 1 is exact in f32");
    // The two sides of the derivation, each around the exact value, lie inside the interval ...
    assert!(
        lo <= exact - 31.0 * u && exact + 31.0 * u <= hi,
        "[{lo}, {hi}]"
    );
    assert!(lo <= twin - 15.0 * u && twin + 15.0 * u <= hi);
    // ... and the interval is not absurdly loose: within a factor of two of the derivation.
    let half_width = (hi - lo) / 2.0;
    assert!(
        half_width <= 2.0 * 31.0 * u,
        "half width {} u",
        half_width / u
    );
    // 15u + 31u = 46u is a bound on the difference between the two parties, which is how far apart two values of
    // the interval can be; the interval is centred on the exact value, so the 46u band around the twin is not
    // inside it (that needs 46u + the twin's error), and the oracle does not need it to be: it asserts the GPU
    // value is inside the interval.
    assert!(hi - lo >= 46.0 * u);
    eprintln!(
        "polynomial v = 1: interval [{lo:e}, {hi:e}], half width {:.2} u (derivation: GPU side 31 u, twin side 15 u)",
        half_width / u
    );
}

#[test]
fn the_polynomial_interval_contains_the_models_over_the_input_domain() {
    // The kernel's inputs are k / 2048 - 1: sample the domain away from the root near -0.55, where the output
    // interval is wider than 2^-10 of the result and the program is (correctly) discarded.
    let mut rng = Prng::new(7);
    let values: Vec<f32> = (0..=63)
        .map(|k| k as f32 / 32.0 - 1.0)
        .filter(|v| *v > -0.30 || *v < -0.80)
        .collect();
    let program = polynomial(&values);
    let run = run_full(&program);
    // v = 0 gives exact zeros; the interval of every sampled point is accepted or the program as a whole is not.
    if let Err(failure) = &run {
        panic!("{failure:?}");
    }
    assert!(check_against_models(&program, &mut rng, 200));
    // Near the root (about -0.5369) the result is a difference of near-equal terms, so its interval is wider than
    // 2^-10 of the result or contains zero: the oracle refuses it instead of guessing. Some point of a fine grid does.
    let near: Vec<f32> = (0..=400)
        .map(|k| -0.5369 + (k as f32 - 200.0) / 400_000.0)
        .collect();
    let mut refused = 0;
    for v in &near {
        if let Err(failure) = run_full(&polynomial(&[*v])) {
            assert!(
                matches!(
                    failure.violation,
                    Violation::Unboundable | Violation::FloatDomain
                ),
                "{failure:?}"
            );
            refused += 1;
        }
    }
    assert!(
        refused >= 20,
        "only {refused} of {} root-region points refused",
        near.len()
    );
}

#[test]
fn a_nine_term_loop_chain_covers_every_grouping_the_models_try() {
    // acc = a[i]; for k in 0..8 { acc = acc +/- (k as f32) * a[i] ... }: nine terms with cancellations.
    let v = || Expr::input(0, Ty::F32, Expr::Gid);
    let body = vec![
        Stmt::Let {
            var: 0,
            ty: Ty::F32,
            mutable: true,
            init: v(),
        },
        Stmt::For {
            var: 1,
            trips: 8,
            body: vec![Stmt::Assign {
                var: 0,
                value: Expr::bin(
                    BinOp::Add,
                    Expr::Var(0, Ty::F32),
                    Expr::bin(BinOp::Mul, Expr::cast(Ty::F32, Expr::Var(1, Ty::U32)), v()),
                ),
            }],
        },
    ];
    let program = Program {
        seed: 0,
        methods: Default::default(),
        attempt: 0,
        count: 5,
        inputs: vec![InputBuf {
            ty: Ty::F32,
            data: [0.1f32, -0.7, 1.3, 1000.5, -3.0].map(f32::to_bits).to_vec(),
        }],
        body,
        result: Expr::Var(0, Ty::F32),
    };
    program.check().unwrap();
    let mut rng = Prng::new(99);
    assert!(check_against_models(&program, &mut rng, 400));
    // The widening is nine terms' worth: 8 roundings of one ULP of the sum of the term magnitudes.
    let run = run_full(&program).unwrap();
    let x = 1000.5f64;
    let sum_of_magnitudes = x * (1.0 + 28.0); // 1 + sum k
    let expected =
        8.0 * (crate::float::ulp32(sum_of_magnitudes) + crate::float::MARGIN * sum_of_magnitudes);
    let (lo, hi) = run.intervals[3];
    assert!(
        ((hi - lo) / 2.0 - expected).abs() < 1e-3 * expected + 4.0 * crate::float::ulp32(x * 29.0),
        "half width {} vs {}",
        (hi - lo) / 2.0,
        expected
    );
}

/// `(a + b) + c` over three one-element inputs.
fn three_term_chain(a: f32, b: f32, c: f32) -> Program {
    let input = |bits: f32| InputBuf {
        ty: Ty::F32,
        data: vec![bits.to_bits()],
    };
    let read = |buf| Expr::input(buf, Ty::F32, Expr::Gid);
    Program {
        seed: 0,
        methods: Default::default(),
        attempt: 0,
        count: 1,
        inputs: vec![input(a), input(b), input(c)],
        body: Vec::new(),
        result: Expr::bin(BinOp::Add, Expr::bin(BinOp::Add, read(0), read(1)), read(2)),
    }
}

#[test]
fn the_binade_edge_chains_of_the_review_hold_under_the_adversarial_model() {
    // Opus review of m5.c1a2, must-fix 1. The GPU may round a + b up to the power of two and then round that plus c
    // up to the next f32, whose ULP is twice as large. Round-to-nearest hardware cannot do it; the model, which picks
    // the neighbour away from the twin and folds in the written order, does. Reverting the widening
    // `S' = S + 2r ULP(S)` of `Fl::interval` makes this test fail (mutation d).
    let mut rng = Prng::new(0xB1AD);
    let one = three_term_chain(1.0 - 2f32.powi(-24), 2f32.powi(-26), 2f32.powi(-30));
    assert!(check_against_models(&one, &mut rng, 600));
    let k = 1024.0;
    let scaled = three_term_chain(
        k * (1.0 - 2f32.powi(-24)),
        k * 2f32.powi(-26),
        2f32.powi(-20),
    );
    assert!(check_against_models(&scaled, &mut rng, 600));
}
