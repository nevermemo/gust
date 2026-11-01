//! Tests of the reducer's pure parts: the size order, each pass's candidates, and the driver with fake predicates (no
//! cargo, no GPU).

use std::collections::HashSet;
use std::time::Duration;

use super::*;
use crate::ast::{BinOp, InputBuf};
use crate::interp::run_full;
use crate::runsrc::plan_seed;

fn program(count: u32, data: &[u32], body: Vec<Stmt>, result: Expr) -> Program {
    Program {
        seed: 1,
        attempt: 0,
        count,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: (0..count as usize).map(|k| data[k % data.len()]).collect(),
        }],
        body,
        result,
    }
}

fn read() -> Expr {
    Expr::input(0, Ty::U32, Expr::Gid)
}

fn let_u32(var: u32, mutable: bool, init: Expr) -> Stmt {
    Stmt::Let {
        var,
        ty: Ty::U32,
        mutable,
        init,
    }
}

/// The candidates the reducer would run: type-valid, in domain, strictly smaller than `p`.
fn valid(p: &Program, candidates: Vec<Program>) -> Vec<Program> {
    candidates
        .into_iter()
        .filter(|c| size(c) < size(p) && c.check().is_ok() && discard_reason(c).is_none())
        .collect()
}

#[test]
fn the_size_is_compared_lexicographically_statements_first() {
    let big_expression = program(64, &[1], Vec::new(), {
        let sum = Expr::bin(BinOp::Add, read(), Expr::u32(1));
        Expr::bin(BinOp::Mul, sum, Expr::u32(3))
    });
    let one_statement = program(64, &[1], vec![let_u32(0, false, Expr::u32(0))], read());
    // More statements is larger, whatever the expressions weigh.
    assert!(size(&big_expression) < size(&one_statement));
    // Same statements: nodes, then trips, then elements, then literal magnitudes.
    let loop_of = |trips: u32| {
        let body = vec![Stmt::For {
            var: 0,
            trips,
            body: Vec::new(),
        }];
        program(64, &[1], body, read())
    };
    assert!(size(&loop_of(2)) < size(&loop_of(3)));
    let fewer_elements = program(63, &[1], Vec::new(), read());
    assert!(size(&fewer_elements) < size(&program(64, &[1], Vec::new(), read())));
    let literal = |value: Expr| program(64, &[1], Vec::new(), Expr::bin(BinOp::Add, read(), value));
    assert!(size(&literal(Expr::u32(5))) < size(&literal(Expr::u32(6))));
    // An `i32` by absolute value, an `f32` by magnitude (1.0 below 2.0), and a size is not strictly below itself.
    assert_eq!(magnitude_of(Ty::I32, (-5i32) as u32), 5);
    assert!(magnitude_of(Ty::F32, 1.0f32.to_bits()) < magnitude_of(Ty::F32, 2.0f32.to_bits()));
    assert_eq!(
        magnitude_of(Ty::F32, 1.0f32.to_bits()),
        magnitude_of(Ty::F32, (-1.0f32).to_bits())
    );
    let p = literal(Expr::u32(5));
    assert_eq!(size(&p), size(&p.clone()));
    assert_eq!(size(&p), (0, 4, 0, 64, 5));
}

#[test]
fn the_statement_pass_drops_chunks_and_unwraps_blocks() {
    // `v2` is dead, `v0` and `v1` are live.
    let body = vec![
        let_u32(0, false, read()),
        let_u32(
            1,
            false,
            Expr::bin(BinOp::Add, Expr::Var(0, Ty::U32), Expr::u32(1)),
        ),
        let_u32(2, false, Expr::u32(9)),
    ];
    let p = program(4, &[1, 2], body, Expr::Var(1, Ty::U32));
    let found = valid(&p, statements(&p, 0));
    assert!(!found.is_empty());
    assert!(
        found
            .iter()
            .all(|c| c.body.len() == 2 && c.body[..2] == p.body[..2])
    );
    // An `if` is replaced by its then block, and a `for` by its body.
    let guarded = vec![
        let_u32(0, true, Expr::u32(1)),
        Stmt::If {
            cond: Expr::bin(BinOp::Lt, Expr::Gid, Expr::u32(3)),
            then: vec![Stmt::Assign {
                var: 0,
                value: Expr::bin(BinOp::Add, Expr::Var(0, Ty::U32), read()),
            }],
            els: Vec::new(),
        },
    ];
    let result = Expr::bin(BinOp::BitXor, Expr::Var(0, Ty::U32), read());
    let q = program(4, &[1, 2], guarded, result);
    let found = valid(&q, statements(&q, 0));
    assert!(
        found
            .iter()
            .any(|c| matches!(c.body.as_slice(), [_, Stmt::Assign { .. }]))
    );
    assert!(found.iter().any(|c| c.body.len() == 1));
}

#[test]
fn the_expression_pass_substitutes_the_value_the_interpreter_observed() {
    // out[i] = (a[i] + 5) * 2 over [10, 20, 30, 40]; at element 2: 30, 35, 70.
    let sum = Expr::bin(BinOp::Add, read(), Expr::u32(5));
    let p = program(
        4,
        &[10, 20, 30, 40],
        Vec::new(),
        Expr::bin(BinOp::Mul, sum, Expr::u32(2)),
    );
    let found = observed(&p, 2);
    // Largest subtree first: the whole result, the sum, the read, `i`; literals are never replaced.
    assert_eq!(found.len(), 4);
    assert_eq!(found[0].result, Expr::u32(70));
    assert!(matches!(&found[1].result, Expr::Bin { l, .. } if **l == Expr::u32(35)));
    for candidate in &found[..3] {
        assert_eq!(run_full(candidate).unwrap().values[2], 70);
    }
    assert!(size(&found[0]) < size(&p));
}

#[test]
fn the_expression_pass_skips_what_the_failing_element_never_evaluates() {
    // `a[i] > 100 && a[i + 40] == 1`: the right operand is skipped for every element, so it has no observed value.
    let guard = Expr::bin(BinOp::Gt, read(), Expr::u32(100));
    let far = Expr::input(0, Ty::U32, Expr::bin(BinOp::Add, Expr::Gid, Expr::u32(40)));
    let result = Expr::bin(BinOp::And, guard, Expr::bin(BinOp::Eq, far, Expr::u32(1)));
    let p = program(64, &[1], Vec::new(), result);
    let seen = observed_values(&p, 3);
    let mut nodes = 0;
    visit_exprs(&p, &mut |e| {
        nodes += usize::from(seen.contains_key(&(std::ptr::from_ref(e) as usize)))
    });
    assert_eq!(nodes, 1 + 4);
}

#[test]
fn the_trip_pass_goes_to_one_and_to_half() {
    let body = vec![
        let_u32(1, true, Expr::u32(0)),
        Stmt::For {
            var: 0,
            trips: 8,
            body: vec![Stmt::Assign {
                var: 1,
                value: Expr::bin(BinOp::Add, Expr::Var(1, Ty::U32), Expr::Var(0, Ty::U32)),
            }],
        },
    ];
    let p = program(
        4,
        &[1],
        body,
        Expr::bin(BinOp::BitXor, Expr::Var(1, Ty::U32), read()),
    );
    let found = trips(&p, 0);
    let counts: Vec<u32> = found
        .iter()
        .filter(|c| size(c) < size(&p))
        .map(|c| match &c.body[1] {
            Stmt::For { trips, .. } => *trips,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(counts, [1, 4]);
}

#[test]
fn the_count_pass_goes_to_the_failing_index_rounded_to_a_workgroup_and_then_exactly() {
    let p = program(257, &[1, 2, 3], Vec::new(), read());
    let counts = |p: &Program, failing| -> Vec<(u32, usize)> {
        element_count(p, failing)
            .iter()
            .map(|c| (c.count, c.inputs[0].data.len()))
            .collect()
    };
    assert_eq!(counts(&p, 70), [(128, 128), (71, 71)]);
    assert_eq!(counts(&p, 200), [(256, 256), (201, 201)]);
    // Already within the failing workgroup: only the exact cut; at the end: nothing.
    assert_eq!(counts(&program(64, &[1], Vec::new(), read()), 3), [(4, 4)]);
    assert_eq!(counts(&program(64, &[1], Vec::new(), read()), 63), []);
    // The input data keeps its prefix.
    assert_eq!(element_count(&p, 2)[1].inputs[0].data, [1, 2, 3]);
}

#[test]
fn the_literal_pass_goes_to_zero_one_and_the_power_of_two_below_with_the_sign_kept() {
    let candidates = |expr: Expr| -> Vec<Expr> {
        let p = program(4, &[1], Vec::new(), Expr::bin(BinOp::BitXor, read(), expr));
        literals(&p, 0)
            .into_iter()
            .map(|c| match c.result {
                Expr::Bin { r, .. } => *r,
                other => panic!("{other:?}"),
            })
            .collect()
    };
    assert_eq!(
        candidates(Expr::u32(1000)),
        [Expr::u32(0), Expr::u32(1), Expr::u32(512)]
    );
    assert_eq!(candidates(Expr::u32(1)), [Expr::u32(0)]);
    assert_eq!(candidates(Expr::u32(0)), Vec::<Expr>::new());
    assert_eq!(candidates(Expr::u32(u32::MAX))[2], Expr::u32(1 << 31));
    // An `i32` keeps its sign; `i32::MIN` has no literal, so it is never a candidate.
    let signed = |v: i32| -> Vec<Expr> {
        let p = program(
            4,
            &[1],
            Vec::new(),
            Expr::bin(BinOp::Add, Expr::cast(Ty::I32, read()), Expr::i32(v)),
        );
        literals(&p, 0)
            .into_iter()
            .map(|c| match c.result {
                Expr::Bin { r, .. } => *r,
                other => panic!("{other:?}"),
            })
            .collect()
    };
    assert_eq!(
        signed(-1000),
        [Expr::i32(0), Expr::i32(-1), Expr::i32(-512)]
    );
    assert_eq!(signed(-1), [Expr::i32(0)]);
    assert_eq!(signed(i32::MAX)[2], Expr::i32(1 << 30));
    // An `f32` keeps its exponent and sign: 3.0 -> 0, 1.0, 2.0; -3.0 -> 0, -1.0, -2.0; 1.0 -> 0.
    let float = |v: f32| -> Vec<Expr> {
        let p = program(
            4,
            &[1],
            Vec::new(),
            Expr::bin(BinOp::Add, Expr::cast(Ty::F32, read()), Expr::f32(v)),
        );
        literals(&p, 0)
            .into_iter()
            .map(|c| match c.result {
                Expr::Bin { r, .. } => *r,
                other => panic!("{other:?}"),
            })
            .collect()
    };
    assert_eq!(float(3.0), [Expr::f32(0.0), Expr::f32(1.0), Expr::f32(2.0)]);
    assert_eq!(
        float(-3.0),
        [Expr::f32(0.0), Expr::f32(-1.0), Expr::f32(-2.0)]
    );
    assert_eq!(float(1.0), [Expr::f32(0.0)]);
    // A `bool` literal is left alone.
    let p = program(
        4,
        &[1],
        Vec::new(),
        Expr::bin(BinOp::Eq, Expr::bool(true), Expr::bool(false)),
    );
    assert!(literals(&p, 0).is_empty());
}

#[test]
fn the_stream_pass_decodes_the_programs_own_stream_with_a_zeroed_tail() {
    let p = plan_seed(1).unwrap().remove(0).program;
    let found = truncated_stream(&p, 0);
    assert_eq!(found.len(), 5);
    assert!(
        found
            .iter()
            .all(|c| (c.seed, c.attempt) == (p.seed, p.attempt))
    );
    // The cut at the length the program consumes is the program itself; shorter cuts differ somewhere.
    assert!(found.iter().any(|c| *c != p));
    assert!(found.iter().any(|c| size(c) < size(&p)));
}

/// A budget that is generous in candidates and bounded in time, so a search that does not terminate fails a test.
fn bounded() -> Limits {
    Limits {
        candidates: 10_000,
        time: Duration::from_secs(20),
        origin: None,
    }
}

fn start_program() -> Program {
    let mut all: Vec<Program> = (1..=2)
        .flat_map(|s| plan_seed(s).unwrap())
        .map(|p| p.program)
        .collect();
    all.sort_by_key(size);
    all.pop().unwrap()
}

#[test]
fn the_search_ends_at_a_fixed_point_and_never_offers_a_candidate_that_is_not_strictly_smaller() {
    let start = start_program();
    let mut best = size(&start);
    let limits = bounded();
    // Everything that is offered "still fails", so the search runs to the end of every pass.
    let found = reduce(&start, 0, limits, |candidate| {
        let now = size(candidate);
        assert!(now < best, "{now:?} is not strictly below {best:?}");
        best = now;
        Some(0)
    });
    assert_eq!(found.stop, Stop::FixedPoint);
    assert!(found.accepted > 3, "{} accepted steps", found.accepted);
    assert_eq!(size(&found.program), best);
    assert!(found.program.check().is_ok() && discard_reason(&found.program).is_none());
    // At the fixed point no pass finds anything smaller.
    let again = reduce(&found.program, 0, limits, |_| Some(0));
    assert_eq!((again.accepted, again.stop), (0, Stop::FixedPoint));
}

#[test]
fn a_candidate_seen_before_is_not_run_again() {
    let start = start_program();
    let mut seen = HashSet::new();
    let mut calls = 0;
    // Accept only candidates that keep at least four expression nodes: the rejected ones come back in later rounds.
    let found = reduce(&start, 0, bounded(), |candidate| {
        calls += 1;
        assert!(seen.insert(key(candidate)), "a candidate was run twice");
        (size(candidate).1 >= 4).then_some(0)
    });
    assert_eq!(found.tried, calls);
    assert!(found.cache_hits > 0, "no candidate came back");
}

#[test]
fn the_budget_counts_predicate_calls_and_time() {
    let start = start_program();
    let mut calls = 0;
    let limits = Limits {
        candidates: 3,
        ..Limits::default()
    };
    let found = reduce(&start, 0, limits, |_| {
        calls += 1;
        None
    });
    assert_eq!((found.tried, calls, found.stop), (3, 3, Stop::Candidates));
    assert_eq!(found.program, start);
    let limits = Limits {
        candidates: 60,
        time: Duration::ZERO,
        origin: None,
    };
    let found = reduce(&start, 0, limits, |_| unreachable!("no time is left"));
    assert_eq!((found.tried, found.stop), (0, Stop::Time));
}

#[test]
fn a_candidate_that_is_not_type_valid_or_not_in_domain_never_reaches_the_predicate() {
    let start = start_program();
    reduce(&start, 0, bounded(), |candidate| {
        assert!(candidate.check().is_ok(), "{:?}", candidate.check());
        assert_eq!(discard_reason(candidate), None);
        (size(candidate).1 >= 6).then_some(0)
    });
}

#[test]
fn no_candidate_is_started_with_less_than_the_minimum_step_left() {
    let start = start_program();
    let mut calls = 0;
    let tiny = Limits {
        time: MIN_STEP / 2,
        ..bounded()
    };
    let r = reduce(&start, 0, tiny, |_| {
        calls += 1;
        Some(0)
    });
    assert_eq!((calls, r.tried, r.stop), (0, 0, Stop::Time));
}
