//! `difftest-run --shrink <seed>:<index>`: the cargo-driven predicate for the reducer (`reduce.rs`), the identity rule
//! and the failure record.
//!
//! One candidate is one build of the single-program run crate `<target>/difftest/shrink-f<format>-s<seed>-i<index>/`,
//! one test run (the interpreter, the CPU twin and, unless `--cpu-only`, the GPU) and one `classify`. The candidate
//! "still fails" under the identity rule of R1-12 and nothing else: the same classification (`Mismatch::kind`), the
//! same output buffer (there is one, `out`; its element type must not change) and a mismatch at some element. A compile
//! error, a validator rejection, any other classification, an error of a party or a timeout is "not reproducing".
//! CI never calls this; it uploads the seed, the program text and the three outputs instead.

use std::time::{Duration, Instant};

use crate::ast::{Program, Ty};
use crate::classify::{Mismatch, Raw, classify};
use crate::emit::{emit_module, package_name};
use crate::generator::DIFFTEST_FORMAT;
use crate::harness::{Options, TIMED_OUT, execute_plan_within};
use crate::interp::run_full;
use crate::json::Json;
use crate::reduce::{Limits, Reduction, Stop, reduce, size};
use crate::runsrc::{Planned, plan_seed};

/// The subprocess deadline of R1-12 (each cargo step of a candidate).
pub const DEADLINE: Duration = Duration::from_secs(120);

/// What "fails the same way" means: the classification (the set of kinds of the mismatches that name an element, so a
/// candidate that gains or drops one party's disagreement is another failure) and the type of the one output buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The kind of the first mismatch that names an element; the failing element is read from it.
    pub kind: &'static str,
    /// Every kind among the mismatches that name an element, sorted.
    pub kinds: Vec<&'static str>,
    pub out_ty: Ty,
    /// A slangc build failure or a validation failure of the GPU party (no element names it): see [`build_signature`]
    /// and [`validation_signature`]. Then `kind` is
    /// `gpu-error` and a candidate reproduces when it fails to compile in the same way.
    pub build: Option<String>,
}

/// The signature of a slangc compilation failure in the message of a `gpu-error`: its first error code and the first
/// quoted name after it (`E30015 '__gust_keep'`), so a candidate that fails to compile for another reason does not
/// count as the same failure. `None` for any other message.
pub fn build_signature(message: &str) -> Option<String> {
    if !message.contains("CompilationFailed") {
        return None;
    }
    let start = message.find("error[E")? + "error[".len();
    let rest = &message[start..];
    let code = &rest[..rest.find(']')?];
    let quoted = rest.split('\'').nth(1).filter(|name| name.len() < 80);
    Some(match quoted {
        Some(name) => format!("{code} '{name}'"),
        None => code.to_owned(),
    })
}

/// The signature of a wgpu or naga validation failure in the message of a `gpu-error` (the party built the module and
/// the device refused it): the text from `Validation Error` on, whitespace collapsed, every run of digits masked (ids
/// of expressions, handles and `v<n>` names change as the shrinker removes statements), the first 160 characters.
/// `None` for any other message. (m9.d1d, N6)
pub fn validation_signature(message: &str) -> Option<String> {
    let start = message.find("Validation Error")?;
    let mut masked = String::new();
    let mut in_digits = false;
    for word in message[start..].split_whitespace() {
        if !masked.is_empty() {
            masked.push(' ');
        }
        for c in word.chars() {
            if c.is_ascii_digit() {
                if !in_digits {
                    masked.push('#');
                }
                in_digits = true;
            } else {
                in_digits = false;
                masked.push(c);
            }
        }
        in_digits = false;
    }
    Some(masked.chars().take(160).collect())
}

/// The signature of a failure that names no element: a slangc build failure or a validation failure.
fn failure_signature(message: &str) -> Option<String> {
    build_signature(message).or_else(|| validation_signature(message))
}

/// The identity of a failing program, and the element it first fails at: its first mismatch that names an element, or,
/// when none does, a slangc build failure (shrunk from element 0).
pub fn identity_of(program: &Program, found: &[Mismatch]) -> Option<(Identity, u32)> {
    let Some((first, element)) = found
        .iter()
        .find_map(|m| Some((m, u32::try_from(m.first_element?).ok()?)))
    else {
        let signature = found
            .iter()
            .filter(|m| m.kind == "gpu-error")
            .find_map(|m| failure_signature(&m.message))?;
        let identity = Identity {
            kind: "gpu-error",
            kinds: vec!["gpu-error"],
            out_ty: program.out_ty(),
            build: Some(signature),
        };
        return Some((identity, 0));
    };
    let mut kinds: Vec<&'static str> = found
        .iter()
        .filter(|m| m.first_element.is_some())
        .map(|m| m.kind)
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    let identity = Identity {
        kind: first.kind,
        kinds,
        out_ty: program.out_ty(),
        build: None,
    };
    Some((identity, element))
}

/// The failing element if `run` (the classification of a candidate, or the error of its build or run) fails the same
/// way as `identity`, else `None`. A mismatch that names no element (an error of a party, a validator rejection) makes
/// the candidate not reproducing even beside a matching one: the failure is then no longer the original one.
pub fn reproduces(
    identity: &Identity,
    program: &Program,
    run: Result<&[Mismatch], &str>,
) -> Option<u32> {
    let found = run.ok()?;
    if let Some(signature) = &identity.build {
        let same = program.out_ty() == identity.out_ty
            && found.len() == 1
            && found[0].kind == "gpu-error"
            && failure_signature(&found[0].message).as_ref() == Some(signature);
        return same.then_some(0);
    }
    if program.out_ty() != identity.out_ty || found.iter().any(|m| m.first_element.is_none()) {
        return None;
    }
    let mut kinds: Vec<&str> = found.iter().map(|m| m.kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    if kinds != identity.kinds {
        return None;
    }
    let same = found.iter().find(|m| m.kind == identity.kind)?;
    u32::try_from(same.first_element?).ok()
}

/// Parses `<seed>:<index>`.
pub fn parse_target(text: &str) -> Result<(u64, usize), String> {
    let (seed, index) = text
        .split_once(':')
        .ok_or_else(|| format!("--shrink wants <seed>:<index>, got {text:?}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|e| format!("seed {seed:?}: {e}"))?;
    let index = index
        .parse::<usize>()
        .map_err(|e| format!("index {index:?}: {e}"))?;
    if index >= crate::generator::PROGRAMS_PER_SEED {
        return Err(format!("index {index} is not below 16"));
    }
    Ok((seed, index))
}

/// Runs one program through the run crate and returns its result file; the `Duration` bounds each of its cargo steps.
pub type Runner<'a> = dyn FnMut(&Planned, Duration) -> Result<Raw, String> + 'a;

/// The deadline of one cargo step: `DEADLINE`, or what is left of the search's time budget if that is less. The
/// budget starts after the original run (`Limits::origin`), and `reduce` starts no candidate with under `MIN_STEP`
/// left, so a step is never started with a zero deadline.
pub fn step_deadline(budget: Duration, elapsed: Duration) -> Duration {
    DEADLINE.min(budget.saturating_sub(elapsed))
}

/// What `--shrink` found.
pub enum Shrunk {
    /// The program agrees on all parties: nothing to shrink.
    NoFailure(String),
    Record(Box<Json>),
}

fn plan_of(program: &Program, index: usize) -> Option<Planned> {
    let run = run_full(program).ok()?;
    Some(Planned {
        program: program.clone(),
        index,
        expected: run.values,
        intervals: run.intervals,
    })
}

fn side(program: &Program, raw: &Raw, found: &[Mismatch]) -> Json {
    let party = |output: &Result<Vec<u32>, String>| match output {
        Ok(words) => (Json::words(words), Json::Null),
        Err(message) => (Json::Null, Json::str(message)),
    };
    let (twin, twin_error) = party(&raw.twin);
    let (gpu, gpu_error) = raw.gpu.as_ref().map_or((Json::Null, Json::Null), party);
    let inputs = program.inputs.iter().map(|input| {
        Json::obj(vec![
            ("ty", Json::str(input.ty.name())),
            ("data", Json::words(&input.data)),
        ])
    });
    let (statements, nodes, trips, count, magnitude) = size(program);
    Json::obj(vec![
        ("program_text", Json::str(&emit_module(program))),
        (
            "size",
            Json::obj(vec![
                ("statements", Json::uint(statements as u64)),
                ("expression_nodes", Json::uint(nodes as u64)),
                ("loop_trips", Json::uint(trips)),
                ("elements", Json::uint(u64::from(count))),
                ("literal_magnitudes", Json::uint(magnitude)),
            ]),
        ),
        ("inputs", Json::Arr(inputs.collect())),
        ("interpreter", Json::words(&raw.interp)),
        ("twin", twin),
        ("twin_error", twin_error),
        ("gpu", gpu),
        ("gpu_error", gpu_error),
        (
            "mismatches",
            Json::Arr(found.iter().map(Mismatch::to_json).collect()),
        ),
    ])
}

/// Shrinks program `index` of `seed` (`plan`) with `run` as the executor. `Err` is a harness failure on the original
/// program; a failing candidate is only ever "not reproducing".
pub fn shrink_with(
    run: &mut Runner,
    cpu_only: bool,
    plan: &Planned,
    limits: Limits,
) -> Result<Shrunk, String> {
    let started = Instant::now();
    let original_raw = run(plan, DEADLINE)?;
    let original_found = classify(plan, &original_raw, cpu_only);
    let Some((identity, failing)) = identity_of(&plan.program, &original_found) else {
        return Ok(Shrunk::NoFailure(format!(
            "seed {} program {} does not fail: none of its {} mismatches names an element or is a slangc build failure",
            plan.program.seed,
            plan.index,
            original_found.len()
        )));
    };
    let mut best = (
        plan.program.clone(),
        original_raw.clone(),
        original_found.clone(),
    );
    let mut calls = Vec::new();
    // The one clock of the time budget starts here, after the original run, for `reduce` and for the deadlines.
    let search = Instant::now();
    let limits = Limits {
        origin: Some(search),
        ..limits
    };
    let reduction = reduce(&plan.program, failing, limits, |candidate| {
        let t = Instant::now();
        let planned = plan_of(candidate, plan.index)?;
        let result = run(&planned, step_deadline(limits.time, search.elapsed())).map(|raw| {
            let found = classify(&planned, &raw, cpu_only);
            (raw, found)
        });
        let verdict = reproduces(
            &identity,
            candidate,
            result
                .as_ref()
                .map(|(_, found)| &found[..])
                .map_err(|e| e.as_str()),
        );
        let outcome = match (&result, verdict) {
            (_, Some(_)) => "reproduces",
            (Err(e), None) if e.starts_with(TIMED_OUT) => "timed out",
            (Err(_), None) => "build or run failed",
            (Ok(_), None) => "does not reproduce",
        };
        calls.push(Json::obj(vec![
            ("seconds", Json::Float(t.elapsed().as_secs_f64())),
            ("result", Json::str(outcome)),
        ]));
        if let (Some(_), Ok((raw, found))) = (verdict, result) {
            best = (candidate.clone(), raw, found);
        }
        verdict
    });
    Ok(Shrunk::Record(Box::new(record(
        plan,
        (&original_raw, &original_found),
        (&best.0, &best.1, &best.2),
        &identity,
        &reduction,
        calls,
        started.elapsed(),
        cpu_only,
    ))))
}

#[allow(clippy::too_many_arguments)]
fn record(
    plan: &Planned,
    original: (&Raw, &[Mismatch]),
    shrunk: (&Program, &Raw, &[Mismatch]),
    identity: &Identity,
    reduction: &Reduction,
    calls: Vec<Json>,
    elapsed: Duration,
    cpu_only: bool,
) -> Json {
    let stop = match reduction.stop {
        Stop::FixedPoint => "fixed point",
        Stop::Candidates => "candidate budget",
        Stop::Time => "time budget",
    };
    Json::obj(vec![
        ("tool", Json::str("difftest-run --shrink")),
        ("format", Json::uint(u64::from(DIFFTEST_FORMAT))),
        ("mode", Json::str(if cpu_only { "cpu-only" } else { "gpu" })),
        (
            "adapter",
            original.0.adapter.as_deref().map_or(Json::Null, Json::str),
        ),
        ("seed", Json::uint(plan.program.seed)),
        ("index", Json::uint(plan.index as u64)),
        ("attempt", Json::uint(plan.program.attempt)),
        ("classification", Json::str(identity.kind)),
        (
            "build_failure",
            identity.build.as_deref().map_or(Json::Null, Json::str),
        ),
        ("output_buffer", Json::str("out")),
        ("output_type", Json::str(identity.out_ty.name())),
        ("failing_element", Json::uint(u64::from(reduction.failing))),
        ("candidates_used", Json::uint(reduction.tried as u64)),
        ("cache_hits", Json::uint(reduction.cache_hits as u64)),
        ("accepted_steps", Json::uint(reduction.accepted as u64)),
        ("stopped_by", Json::str(stop)),
        ("seconds", Json::Float(elapsed.as_secs_f64())),
        ("candidates", Json::Arr(calls)),
        ("original", side(&plan.program, original.0, original.1)),
        ("shrunk", side(shrunk.0, shrunk.1, shrunk.2)),
    ])
}

/// `--shrink`: the executor builds the single-program run crate and runs it, each cargo step under `step_deadline`.
pub fn shrink(options: &Options, target: (u64, usize), limits: Limits) -> Result<Shrunk, String> {
    let (seed, index) = target;
    let plan = plan_seed(seed)?
        .into_iter()
        .nth(index)
        .ok_or_else(|| format!("seed {seed} has no program {index}"))?;
    let name = format!("shrink-f{DIFFTEST_FORMAT}-s{seed}-i{index}");
    let package = package_name(&name)?;
    let mut run = |planned: &Planned, deadline: Duration| {
        let (mut raws, _) = execute_plan_within(
            options,
            &package,
            &name,
            std::slice::from_ref(planned),
            &[],
            Some(deadline),
        )?;
        raws.pop().ok_or_else(|| "no result file".to_owned())
    };
    shrink_with(&mut run, options.cpu_only, &plan, limits)
}

#[cfg(test)]
#[path = "shrink_tests.rs"]
mod tests;
