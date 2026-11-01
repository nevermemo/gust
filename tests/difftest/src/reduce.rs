//! The reducer (M5 R1-12): shrinks a failing program by AST passes, with the harness as the predicate. Std only.
//! The predicate is `FnMut(&Program) -> Option<u32>`: `Some(failing element)` if the candidate still fails the same
//! way (the identity rule is the caller's, `shrink.rs`). A candidate is type-valid (`Program::check`), passes the
//! interpreter (`discard_reason`) and is strictly smaller by `size` (so the search terminates); results are cached by
//! hash of text plus inputs. The six passes run in order, each to a fixed point, until a round changes nothing.

use crate::ast::{Expr, Program, Stmt, Ty, visit_exprs};
use crate::corpus::{fnv1a64, fnv1a64_update};
use crate::emit::emit_module;
use crate::generator::{BYTES_PER_PROGRAM, decode, discard_reason};
use crate::interp::observed_values;
use crate::prng::byte_stream;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const WORKGROUP: u32 = 64;
/// A candidate is not started with less than this left of the time budget: it could not finish.
pub const MIN_STEP: Duration = Duration::from_secs(1);

/// Budget: predicate calls, and wall time from `origin` (default: the call of `reduce`; a caller may share the clock).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub candidates: usize,
    pub time: Duration,
    pub origin: Option<Instant>,
}

impl Default for Limits {
    /// R1-12: 60 candidates or 10 minutes.
    fn default() -> Self {
        Limits {
            candidates: 60,
            time: Duration::from_secs(600),
            origin: None,
        }
    }
}

/// Why the search ended: no pass finds a smaller candidate that still fails, or a budget ran out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    FixedPoint,
    Candidates,
    Time,
}

/// The result: `failing` is the element the predicate last reported; `tried` counts predicate calls.
#[derive(Clone, Debug)]
pub struct Reduction {
    pub program: Program,
    pub failing: u32,
    pub tried: usize,
    pub cache_hits: usize,
    pub accepted: usize,
    pub stop: Stop,
}

/// R1-12's measure, compared lexicographically: statements (nested too), expression nodes, loop trips in all,
/// elements, and the sum of literal magnitudes (an `f32` by its magnitude bits, so 1.0 is below 2.0).
pub type Size = (usize, usize, u64, u32, u64);

pub fn size(p: &Program) -> Size {
    fn trip_total(stmts: &[Stmt]) -> u64 {
        stmts.iter().fold(0, |sum, stmt| match stmt {
            Stmt::If { then, els, .. } => sum + trip_total(then) + trip_total(els),
            Stmt::For { trips, body, .. } => sum + u64::from(*trips) + trip_total(body),
            _ => sum,
        })
    }
    let (mut nodes, mut magnitude) = (0, 0u64);
    visit_exprs(p, &mut |e| {
        nodes += 1;
        if let Expr::Lit(ty, bits) = e {
            magnitude = magnitude.saturating_add(magnitude_of(*ty, *bits));
        }
    });
    let statements = p.body.iter().map(weight).sum();
    (statements, nodes, trip_total(&p.body), p.count, magnitude)
}

fn magnitude_of(ty: Ty, bits: u32) -> u64 {
    match ty {
        Ty::I32 => i64::from(bits as i32).unsigned_abs(),
        Ty::F32 => u64::from(bits & 0x7FFF_FFFF),
        _ => u64::from(bits),
    }
}

/// The cache key: the program's text and its inputs (the text holds the count).
fn key(p: &Program) -> u64 {
    let mut hash = fnv1a64(emit_module(p).as_bytes());
    for input in &p.inputs {
        hash = fnv1a64_update(hash, input.ty.name().as_bytes());
        for word in &input.data {
            hash = fnv1a64_update(hash, &word.to_le_bytes());
        }
    }
    hash
}

/// Statements in pre-order, a statement before its children; the weight of a statement is its subtree's size.
fn weight(stmt: &Stmt) -> usize {
    1 + match stmt {
        Stmt::If { then, els, .. } => then.iter().chain(els).map(weight).sum(),
        Stmt::For { body, .. } => body.iter().map(weight).sum(),
        _ => 0,
    }
}

/// Rebuilds `stmts`; `f(index, stmt)` may replace the statement at pre-order index `index` (and its subtree).
fn edit(
    stmts: &[Stmt],
    at: &mut usize,
    f: &mut dyn FnMut(usize, &Stmt) -> Option<Vec<Stmt>>,
) -> Vec<Stmt> {
    let mut out = Vec::new();
    for stmt in stmts {
        let here = *at;
        if let Some(replacement) = f(here, stmt) {
            *at += weight(stmt);
            out.extend(replacement);
            continue;
        }
        *at += 1;
        out.push(match stmt {
            Stmt::If { cond, then, els } => Stmt::If {
                cond: cond.clone(),
                then: edit(then, at, f),
                els: edit(els, at, f),
            },
            Stmt::For { var, trips, body } => Stmt::For {
                var: *var,
                trips: *trips,
                body: edit(body, at, f),
            },
            other => other.clone(),
        });
    }
    out
}

fn with_body(p: &Program, f: &mut dyn FnMut(usize, &Stmt) -> Option<Vec<Stmt>>) -> Program {
    Program {
        body: edit(&p.body, &mut 0, f),
        ..p.clone()
    }
}

/// Applies `f` to the expression node at pre-order index `target` (the order of `visit_exprs`).
fn at_node(p: &mut Program, target: usize, f: &mut dyn FnMut(&mut Expr)) {
    fn go(e: &mut Expr, n: &mut usize, target: usize, f: &mut dyn FnMut(&mut Expr)) {
        *n += 1;
        if *n - 1 == target {
            return f(e);
        }
        match e {
            Expr::Input { index, .. } => go(index, n, target, f),
            Expr::Un { e, .. } | Expr::Cast { e, .. } => go(e, n, target, f),
            Expr::Bin { l, r, .. } => {
                go(l, n, target, f);
                go(r, n, target, f);
            }
            _ => {}
        }
    }
    fn block(stmts: &mut [Stmt], n: &mut usize, target: usize, f: &mut dyn FnMut(&mut Expr)) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init: e, .. } | Stmt::Assign { value: e, .. } => go(e, n, target, f),
                Stmt::If { cond, then, els } => {
                    go(cond, n, target, f);
                    block(then, n, target, f);
                    block(els, n, target, f);
                }
                Stmt::For { body, .. } => block(body, n, target, f),
            }
        }
    }
    let mut n = 0;
    block(&mut p.body, &mut n, target, f);
    go(&mut p.result, &mut n, target, f);
}

/// Pass 1: remove chunks of statements (ddmin: halves down to single ones), then unwrap an `if` or a `for`.
fn statements(p: &Program, _: u32) -> Vec<Program> {
    let total: usize = p.body.iter().map(weight).sum();
    let mut out = Vec::new();
    let mut chunk = total.div_ceil(2);
    while chunk >= 1 {
        for start in (0..total).step_by(chunk) {
            let gone = start..start + chunk;
            out.push(with_body(p, &mut |k, _| gone.contains(&k).then(Vec::new)));
        }
        chunk /= 2;
    }
    for target in 0..total {
        for branch in 0..2 {
            out.push(with_body(p, &mut |k, stmt| match (k == target, stmt) {
                (true, Stmt::If { then, els, .. }) => {
                    Some(if branch == 0 { then } else { els }.clone())
                }
                (true, Stmt::For { body, .. }) if branch == 0 => Some(body.clone()),
                _ => None,
            }));
        }
    }
    out
}

/// Pass 2: a non-literal subexpression, the largest first, becomes what the interpreter observed at the failing
/// element (not `i32::MIN` or `-0.0`, which have no literal: both are the pattern 0x8000_0000).
fn observed(p: &Program, element: u32) -> Vec<Program> {
    let seen = observed_values(p, element);
    let (mut nodes, mut n) = (Vec::new(), 0);
    visit_exprs(p, &mut |e| {
        let value = seen.get(&(std::ptr::from_ref(e) as usize)).copied();
        if let Some(bits) = value
            && !matches!(e, Expr::Lit(..))
            && !(bits == 0x8000_0000 && e.ty() != Ty::U32)
        {
            nodes.push((e.size(), n, e.ty(), bits));
        }
        n += 1;
    });
    nodes.sort_by_key(|node| std::cmp::Reverse(node.0));
    nodes
        .into_iter()
        .map(|(_, node, ty, bits)| {
            let mut candidate = p.clone();
            at_node(&mut candidate, node, &mut |e| *e = Expr::Lit(ty, bits));
            candidate
        })
        .collect()
}

/// Pass 3: each loop's trip count to 1, or to half.
fn trips(p: &Program, _: u32) -> Vec<Program> {
    let total: usize = p.body.iter().map(weight).sum();
    let mut out = Vec::new();
    for target in 0..total {
        for half in [false, true] {
            out.push(with_body(p, &mut |k, stmt| match stmt {
                Stmt::For { var, trips, body } if k == target && *trips > 1 => {
                    Some(vec![Stmt::For {
                        var: *var,
                        trips: if half { trips.div_ceil(2) } else { 1 },
                        body: body.clone(),
                    }])
                }
                _ => None,
            }));
        }
    }
    out
}

/// Pass 4: fewer elements: the failing index rounded up to a workgroup, then just past the index.
fn element_count(p: &Program, failing: u32) -> Vec<Program> {
    let rounded = (failing + 1).div_ceil(WORKGROUP) * WORKGROUP;
    [rounded, failing + 1]
        .into_iter()
        .filter(|&count| count < p.count)
        .map(|count| {
            let mut candidate = p.clone();
            candidate.count = count;
            for input in &mut candidate.inputs {
                input.data.truncate(count as usize);
            }
            candidate
        })
        .collect()
}

/// Pass 5: each non-`bool` literal toward 0, 1 or the power of two below it (an `f32`: its exponent), the sign kept.
fn literals(p: &Program, _: u32) -> Vec<Program> {
    let mut lits = Vec::new();
    let mut n = 0;
    visit_exprs(p, &mut |e| {
        if let Expr::Lit(ty, bits) = e
            && *ty != Ty::Bool
        {
            lits.push((magnitude_of(*ty, *bits), n, *ty, *bits));
        }
        n += 1;
    });
    lits.sort_by_key(|lit| std::cmp::Reverse(lit.0));
    let mut out = Vec::new();
    for (magnitude, node, ty, bits) in lits {
        let power = if magnitude == 0 {
            0
        } else {
            1u32 << magnitude.ilog2()
        };
        let (one, floor) = match ty {
            Ty::F32 => (1.0f32.to_bits() | (bits & 0x8000_0000), bits & 0xFF80_0000),
            Ty::I32 if (bits as i32) < 0 => ((-1i32) as u32, power.wrapping_neg()),
            _ => (1, power),
        };
        for to in [0, one, floor] {
            // 0x8000_0000 is `i32::MIN`, which has no literal, and `-0.0`, which has none either.
            if magnitude_of(ty, to) < magnitude && (to != 0x8000_0000 || ty == Ty::U32) {
                let mut candidate = p.clone();
                at_node(&mut candidate, node, &mut |e| *e = Expr::Lit(ty, to));
                out.push(candidate);
            }
        }
    }
    out
}

/// Pass 6: the program's own byte stream with its tail zeroed from a few cut points below the length it consumes.
fn truncated_stream(p: &Program, _: u32) -> Vec<Program> {
    let bytes = byte_stream(p.seed, p.attempt, BYTES_PER_PROGRAM);
    let whole = decode(&bytes, p.seed, p.attempt);
    let cut = |len: usize| {
        let mut b = bytes.clone();
        b[len..].fill(0);
        decode(&b, p.seed, p.attempt)
    };
    // The bytes the program consumes: the shortest prefix that still decodes to the same program.
    let lengths: Vec<usize> = (0..=bytes.len()).collect();
    let used = lengths.partition_point(|&len| cut(len) != whole);
    [0, used / 8, used / 4, used / 2, used * 3 / 4]
        .into_iter()
        .map(cut)
        .collect()
}

/// Reduces `start`, which fails at element `failing`. See the module comment.
pub fn reduce(
    start: &Program,
    failing: u32,
    limits: Limits,
    mut fails: impl FnMut(&Program) -> Option<u32>,
) -> Reduction {
    let passes: [fn(&Program, u32) -> Vec<Program>; 6] = [
        statements,
        observed,
        trips,
        element_count,
        literals,
        truncated_stream,
    ];
    let (started, mut cache) = (limits.origin.unwrap_or_else(Instant::now), HashMap::new());
    let mut now = Reduction {
        program: start.clone(),
        failing,
        tried: 0,
        cache_hits: 0,
        accepted: 0,
        stop: Stop::FixedPoint,
    };
    loop {
        let before = now.accepted;
        for pass in passes {
            'fixed_point: loop {
                for candidate in pass(&now.program, now.failing) {
                    if size(&candidate) >= size(&now.program)
                        || candidate.check().is_err()
                        || discard_reason(&candidate).is_some()
                    {
                        continue;
                    }
                    if started.elapsed() + MIN_STEP >= limits.time {
                        now.stop = Stop::Time;
                        return now;
                    }
                    let k = key(&candidate);
                    let verdict = if let Some(&cached) = cache.get(&k) {
                        now.cache_hits += 1;
                        cached
                    } else if now.tried >= limits.candidates {
                        now.stop = Stop::Candidates;
                        return now;
                    } else {
                        now.tried += 1;
                        let verdict = fails(&candidate);
                        cache.insert(k, verdict);
                        verdict
                    };
                    if let Some(element) = verdict {
                        now.program = candidate;
                        now.failing = element;
                        now.accepted += 1;
                        continue 'fixed_point;
                    }
                }
                break;
            }
        }
        if now.accepted == before {
            return now;
        }
    }
}

#[cfg(test)]
#[path = "reduce_tests.rs"]
mod tests;
