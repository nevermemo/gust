//! Tests of the identity rule, the failure record and the whole shrink pipeline with a fake executor (no cargo, no GPU).

use super::*;
use crate::ast::{Expr, InputBuf, operator_counts};
use crate::emit::module_name;

fn mismatch(kind: &'static str, first: Option<usize>) -> Mismatch {
    Mismatch {
        program: "s1_a0".into(),
        seed: 1,
        index: 0,
        attempt: 0,
        kind,
        elements: usize::from(first.is_some()),
        first_element: first,
        message: String::new(),
    }
}

fn program(out: Ty) -> Program {
    let result = match out {
        Ty::I32 => Expr::i32(0),
        _ => Expr::u32(0),
    };
    Program {
        seed: 1,
        attempt: 0,
        count: 4,
        inputs: vec![InputBuf {
            ty: Ty::U32,
            data: vec![0; 4],
        }],
        body: Vec::new(),
        result,
    }
}

/// The identity of a program whose only disagreement is the GPU's.
fn gpu() -> Identity {
    Identity {
        kind: "gpu-vs-interpreter",
        kinds: vec!["gpu-vs-interpreter"],
        out_ty: Ty::U32,
    }
}

#[test]
fn the_identity_is_the_first_mismatch_that_names_an_element() {
    let p = program(Ty::U32);
    let found = [
        mismatch("twin-error", None),
        mismatch("gpu-vs-interpreter", Some(7)),
        mismatch("twin-vs-interpreter", Some(2)),
    ];
    // The kinds are the set of the element-naming ones, sorted; the element is the first one's.
    let both = Identity {
        kind: "gpu-vs-interpreter",
        kinds: vec!["gpu-vs-interpreter", "twin-vs-interpreter"],
        out_ty: Ty::U32,
    };
    assert_eq!(identity_of(&p, &found), Some((both, 7)));
    assert_eq!(identity_of(&p, &found[..1]), None);
    assert_eq!(identity_of(&p, &[]), None);
}

#[test]
fn only_the_same_classification_with_an_element_reproduces() {
    let p = program(Ty::U32);
    let same = [mismatch("gpu-vs-interpreter", Some(3))];
    assert_eq!(reproduces(&gpu(), &p, Ok(&same)), Some(3));
    // A different failing element is still the same failure.
    assert_eq!(
        reproduces(&gpu(), &p, Ok(&[mismatch("gpu-vs-interpreter", Some(0))])),
        Some(0)
    );
    // Other classifications are not reproductions, and neither is a pass.
    for kind in ["twin-vs-interpreter", "gpu-outside-interval"] {
        assert_eq!(
            reproduces(&gpu(), &p, Ok(&[mismatch(kind, Some(3))])),
            None,
            "{kind}"
        );
    }
    assert_eq!(reproduces(&gpu(), &p, Ok(&[])), None);
    // An error of a party or a validator rejection beside the matching mismatch changes the failure.
    for kind in ["gpu-error", "twin-error", "gpu-not-validated", "transport"] {
        let both = [
            mismatch("gpu-vs-interpreter", Some(3)),
            mismatch(kind, None),
        ];
        assert_eq!(reproduces(&gpu(), &p, Ok(&both)), None, "{kind}");
    }
    // The output buffer must keep its type.
    assert_eq!(reproduces(&gpu(), &program(Ty::I32), Ok(&same)), None);
}

#[test]
fn a_candidate_that_gains_or_drops_one_partys_mismatch_is_another_failure() {
    let p = program(Ty::U32);
    let (gpu_only, twin_only) = (
        [mismatch("gpu-vs-interpreter", Some(3))],
        [mismatch("twin-vs-interpreter", Some(3))],
    );
    let both = [
        mismatch("twin-vs-interpreter", Some(1)),
        mismatch("gpu-vs-interpreter", Some(1)),
    ];
    let (id_gpu, _) = identity_of(&p, &gpu_only).unwrap();
    let (id_both, _) = identity_of(&p, &both).unwrap();
    assert_eq!(id_both.kinds.len(), 2);
    // Gains the twin's disagreement.
    assert_eq!(reproduces(&id_gpu, &p, Ok(&both)), None);
    // Drops either party's disagreement.
    assert_eq!(reproduces(&id_both, &p, Ok(&gpu_only)), None);
    assert_eq!(reproduces(&id_both, &p, Ok(&twin_only)), None);
    // The same set reproduces, at the element of the identity's own kind, whatever the order.
    assert_eq!(reproduces(&id_both, &p, Ok(&both)), Some(1));
    let swapped = [both[1].clone(), both[0].clone()];
    assert_eq!(reproduces(&id_both, &p, Ok(&swapped)), Some(1));
    assert_eq!(reproduces(&id_gpu, &p, Ok(&gpu_only)), Some(3));
}

#[test]
fn a_build_failure_and_a_timeout_never_count_as_reproducing() {
    let p = program(Ty::U32);
    for message in [
        format!("{TIMED_OUT}: cargo test --no-run ran past 120 s"),
        "cargo test --no-run failed (exit code: 101)".to_owned(),
        "the result file s1_a0.json is missing".to_owned(),
    ] {
        assert_eq!(reproduces(&gpu(), &p, Err(&message)), None, "{message}");
    }
}

#[test]
fn the_target_is_a_seed_and_an_index_below_sixteen() {
    assert_eq!(parse_target("3:11").unwrap(), (3, 11));
    assert_eq!(parse_target("0:0").unwrap(), (0, 0));
    for bad in ["", "3", "3:", ":3", "3:16", "a:1", "1:-1", "1:2:3"] {
        assert!(parse_target(bad).is_err(), "{bad:?}");
    }
}

/// A fake executor: the twin agrees with the interpreter and the GPU is wrong exactly when the program contains a
/// `u32` multiplication (the way a wrong emitter literal would break one operator).
fn run_fake(planned: &Planned) -> Raw {
    let p = &planned.program;
    let faulty = operator_counts(p).contains_key("* u32");
    let gpu = planned
        .expected
        .iter()
        .map(|w| if faulty { w ^ 1 } else { *w })
        .collect();
    Raw {
        program: module_name(p),
        seed: p.seed,
        attempt: p.attempt,
        index: planned.index,
        count: p.count as usize,
        out_ty: p.out_ty().name().into(),
        mode: "gpu".into(),
        adapter: Some("fake adapter".into()),
        interp: planned.expected.clone(),
        twin: Ok(planned.expected.clone()),
        gpu: Some(Ok(gpu)),
        wgsl_validated: true,
        twin_ns: 0,
        gpu_first_ns: 0,
        gpu_warm_ns: 0,
    }
}

/// R1-12's 60 candidates, with a short time limit so that a search that does not terminate fails a test.
fn limits() -> Limits {
    Limits {
        time: std::time::Duration::from_secs(20),
        ..Limits::default()
    }
}

fn failing_plan() -> Planned {
    (1..=8)
        .flat_map(|seed| plan_seed(seed).unwrap())
        .find(|planned| {
            planned.program.out_ty() != Ty::F32
                && operator_counts(&planned.program).contains_key("* u32")
        })
        .expect("a corpus program multiplies u32")
}

fn number(json: &Json, key: &str) -> u64 {
    json.get(key)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("{key}"))
}

#[test]
fn a_fault_is_shrunk_to_a_small_program_that_still_has_it_and_the_record_says_how() {
    let plan = failing_plan();
    let mut runs = 0;
    let mut run = |planned: &Planned, _: Duration| {
        runs += 1;
        Ok(run_fake(planned))
    };
    let Shrunk::Record(record) = shrink_with(&mut run, false, &plan, limits()).unwrap() else {
        panic!("the program fails")
    };
    let (original, shrunk) = (
        record.get("original").unwrap(),
        record.get("shrunk").unwrap(),
    );
    assert_eq!(
        record.get("classification").and_then(Json::as_str),
        Some("gpu-vs-interpreter")
    );
    assert_eq!(
        record.get("output_buffer").and_then(Json::as_str),
        Some("out")
    );
    assert_eq!(record.get("mode").and_then(Json::as_str), Some("gpu"));
    assert_eq!(
        record.get("adapter").and_then(Json::as_str),
        Some("fake adapter")
    );
    assert_eq!(
        (number(&record, "seed"), number(&record, "attempt")),
        (plan.program.seed, plan.program.attempt)
    );
    // The original run is not a candidate: the executor ran once more than the candidates used.
    let used = number(&record, "candidates_used");
    assert_eq!(runs as u64, used + 1);
    assert_eq!(
        record.get("candidates").unwrap().as_array().unwrap().len() as u64,
        used
    );
    assert!(used <= 60);
    // Smaller by the measure, still the fault, and both programs are in the record with their outputs.
    let statements = |side: &Json| number(side.get("size").unwrap(), "statements");
    let nodes = |side: &Json| number(side.get("size").unwrap(), "expression_nodes");
    assert!((statements(shrunk), nodes(shrunk)) < (statements(original), nodes(original)));
    let text = shrunk.get("program_text").and_then(Json::as_str).unwrap();
    assert!(text.contains(" * "), "{text}");
    assert_ne!(
        Some(text),
        original.get("program_text").and_then(Json::as_str)
    );
    for side in [original, shrunk] {
        let count = side.get("interpreter").unwrap().as_array().unwrap().len();
        for key in ["twin", "gpu"] {
            assert_eq!(
                side.get(key).unwrap().as_array().unwrap().len(),
                count,
                "{key}"
            );
        }
        assert!(side.get("gpu_error").unwrap().is_null());
        assert!(!side.get("inputs").unwrap().as_array().unwrap().is_empty());
        assert!(
            !side
                .get("mismatches")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    // The element count and the failing element are consistent in the shrunk side.
    let failing = number(&record, "failing_element") as usize;
    assert!(failing < shrunk.get("interpreter").unwrap().as_array().unwrap().len());
}

#[test]
fn a_program_that_does_not_fail_is_not_shrunk() {
    let plan = (1..=8)
        .flat_map(|seed| plan_seed(seed).unwrap())
        .find(|planned| !operator_counts(&planned.program).contains_key("* u32"))
        .unwrap();
    let mut run = |planned: &Planned, _: Duration| Ok(run_fake(planned));
    match shrink_with(&mut run, false, &plan, limits()).unwrap() {
        Shrunk::NoFailure(message) => assert!(message.contains("does not fail"), "{message}"),
        Shrunk::Record(_) => panic!("nothing fails"),
    }
}

#[test]
fn candidates_that_time_out_leave_the_program_unshrunk() {
    let plan = failing_plan();
    let mut first = true;
    let mut run = |planned: &Planned, _: Duration| {
        if std::mem::take(&mut first) {
            Ok(run_fake(planned))
        } else {
            Err(format!("{TIMED_OUT}: cargo test ran past 120 s"))
        }
    };
    let five = Limits {
        candidates: 5,
        ..limits()
    };
    let Shrunk::Record(record) = shrink_with(&mut run, false, &plan, five).unwrap() else {
        panic!("the program fails")
    };
    assert_eq!(number(&record, "accepted_steps"), 0);
    assert_eq!(number(&record, "candidates_used"), 5);
    assert_eq!(
        record.get("stopped_by").and_then(Json::as_str),
        Some("candidate budget")
    );
    let calls = record.get("candidates").unwrap().as_array().unwrap();
    assert!(
        calls
            .iter()
            .all(|c| c.get("result").and_then(Json::as_str) == Some("timed out"))
    );
    let (original, shrunk) = (
        record.get("original").unwrap(),
        record.get("shrunk").unwrap(),
    );
    assert_eq!(original.get("program_text"), shrunk.get("program_text"));
    // A harness failure on the original program is an error of the whole run, not a record.
    let mut broken = |_: &Planned, _: Duration| Err::<Raw, _>("cargo test failed".to_owned());
    assert!(shrink_with(&mut broken, false, &plan, limits()).is_err());
}

#[test]
fn a_step_never_gets_more_than_what_is_left_of_the_budget() {
    let secs = Duration::from_secs;
    assert_eq!(step_deadline(secs(600), secs(0)), DEADLINE);
    assert_eq!(step_deadline(secs(600), secs(479)), DEADLINE);
    assert_eq!(step_deadline(secs(600), secs(590)), secs(10));
    assert_eq!(step_deadline(secs(600), secs(600)), Duration::ZERO);
    assert_eq!(step_deadline(secs(600), secs(700)), Duration::ZERO);
    assert_eq!(step_deadline(secs(30), secs(0)), secs(30));
}

#[test]
fn every_candidate_run_is_given_the_remaining_budget_as_its_deadline() {
    let plan = failing_plan();
    let mut deadlines = Vec::new();
    let mut run = |planned: &Planned, deadline: Duration| {
        deadlines.push(deadline);
        Ok(run_fake(planned))
    };
    let short = Limits {
        time: Duration::from_secs(5),
        ..limits()
    };
    let Shrunk::Record(record) = shrink_with(&mut run, false, &plan, short).unwrap() else {
        panic!("the program fails")
    };
    assert_eq!(
        deadlines.len() as u64,
        number(&record, "candidates_used") + 1
    );
    // The original run has the full step deadline; every candidate has at most the 5 s budget, and less as time passes.
    assert_eq!(deadlines[0], DEADLINE);
    assert!(deadlines[1..].iter().all(|d| *d <= Duration::from_secs(5)));
    assert!(deadlines[1..].windows(2).all(|w| w[1] <= w[0]));
}

#[test]
fn an_original_run_that_takes_most_of_the_budget_leaves_no_candidate_a_zero_deadline() {
    let plan = failing_plan();
    let mut deadlines = Vec::new();
    let mut run = |planned: &Planned, deadline: Duration| {
        deadlines.push(deadline);
        if deadlines.len() == 1 {
            std::thread::sleep(Duration::from_millis(2500));
        }
        Ok(run_fake(planned))
    };
    let short = Limits {
        time: Duration::from_secs(3),
        ..limits()
    };
    let Shrunk::Record(record) = shrink_with(&mut run, false, &plan, short).unwrap() else {
        panic!("the program fails")
    };
    // The budget starts after the original run, so the candidates see nearly all of it.
    assert_eq!(deadlines[0], DEADLINE);
    assert!(deadlines.len() > 2);
    assert!(
        deadlines[1..]
            .iter()
            .all(|d| *d > Duration::from_millis(2500)),
        "{deadlines:?}"
    );
    assert_eq!(
        record.get("stopped_by").and_then(Json::as_str),
        Some("fixed point")
    );
}

#[test]
fn a_candidate_is_not_started_when_the_budget_is_spent_and_that_ends_the_search_by_time() {
    let plan = failing_plan();
    let mut deadlines = Vec::new();
    let mut run = |planned: &Planned, deadline: Duration| {
        deadlines.push(deadline);
        if deadlines.len() > 1 {
            std::thread::sleep(Duration::from_millis(1300));
        }
        Ok(run_fake(planned))
    };
    let short = Limits {
        time: Duration::from_secs(3),
        ..limits()
    };
    let Shrunk::Record(record) = shrink_with(&mut run, false, &plan, short).unwrap() else {
        panic!("the program fails")
    };
    // 3 s left, then about 1.7 s, then under MIN_STEP: the third candidate is never started.
    assert_eq!(deadlines.len(), 3, "{deadlines:?}");
    assert!(
        deadlines[1..].iter().all(|d| *d >= crate::reduce::MIN_STEP),
        "{deadlines:?}"
    );
    assert_eq!(
        record.get("stopped_by").and_then(Json::as_str),
        Some("time budget")
    );
    assert_eq!(number(&record, "candidates_used"), 2);
    let calls = record.get("candidates").unwrap().as_array().unwrap();
    assert!(
        calls
            .iter()
            .all(|c| c.get("result").and_then(Json::as_str) == Some("reproduces"))
    );
}
