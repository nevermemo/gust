//! Unit tests of the generator.

use std::collections::BTreeSet;

use super::*;
use crate::ast::{BinOp, operator_counts};
use crate::emit::emit_crate_source;
use crate::interp::Violation;

/// Everything about a program except the seed and attempt labels.
fn content(program: &Program) -> (u32, &[InputBuf], &[Stmt], &Expr) {
    (
        program.count,
        &program.inputs,
        &program.body,
        &program.result,
    )
}

#[test]
fn determinism_the_same_seed_gives_byte_identical_programs() {
    let first = generate_batch(3).unwrap();
    let second = generate_batch(3).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        emit_crate_source(&first.programs).as_bytes(),
        emit_crate_source(&second.programs).as_bytes()
    );
    assert_eq!(generate_attempt(5, 17), generate_attempt(5, 17));
    // The attempt number selects the program, independently of the attempts before it.
    let batch = generate_batch(4).unwrap();
    for program in &batch.programs {
        assert_eq!(*program, generate_attempt(4, program.attempt));
    }
}

#[test]
fn determinism_different_seeds_and_attempts_give_different_programs() {
    let one = generate_batch(1).unwrap();
    let two = generate_batch(2).unwrap();
    assert_ne!(content(&one.programs[0]), content(&two.programs[0]));
    assert_ne!(
        one.programs.iter().map(content).collect::<Vec<_>>(),
        two.programs.iter().map(content).collect::<Vec<_>>()
    );
    assert_ne!(
        content(&generate_attempt(1, 0)),
        content(&generate_attempt(1, 1))
    );
    // All 17 programs of a batch differ from each other.
    let distinct: BTreeSet<String> = one
        .programs
        .iter()
        .map(|p| format!("{:?}", content(p)))
        .collect();
    assert_eq!(distinct.len(), PROGRAMS_PER_SEED);
    // A pinned decode of the integer space (it reads the bytes DIFFTEST_FORMAT 1 read, so these are the m5.c1a
    // values): they change only with a deliberate format change (and then seeds.txt changes too).
    let one = generate_batch_in(Space::Integer, 1).unwrap();
    let two = generate_batch_in(Space::Integer, 2).unwrap();
    let first = &one.programs[0];
    assert_eq!((first.seed, first.attempt, first.count), (1, 0, 1));
    let first = &two.programs[0];
    assert_eq!((first.seed, first.attempt, first.count), (2, 0, 64));
    assert_eq!((one.programs[1].attempt, one.programs[1].count), (3, 2898));
}

#[test]
fn the_integer_space_decodes_to_the_programs_of_m5_c1a() {
    // The discard counts of the m5.c1a corpus (its seeds.txt at 3bcebaf; the hashes are read with the format stamp
    // as 1): the integer space must reproduce them, so it is the same program space. Format 4 relabelled some
    // discards (seeds 1, 3, 4, 5, 7 and 8) with a divisor rule for slangc, and format 5 (m6.dv) took the rule
    // back, so the counts are the m5.c1a counts again, label for label; the accepted programs, and so the hashes
    // below, are those of m5.c1a in every format.
    let expected = [
        "1 16 57 division-by-zero=3 index=6 no-input-read=3 overflow=34 shift-amount=3 static=8",
        "2 16 36 division-by-zero=6 index=2 min-over-minus-one=1 overflow=21 static=6",
        "3 16 38 division-by-zero=8 no-input-read=2 overflow=20 static=8",
        "4 16 53 division-by-zero=7 index=5 no-input-read=1 overflow=25 shift-amount=1 static=14",
        "5 16 42 division-by-zero=5 no-input-read=3 overflow=21 shift-amount=3 static=10",
        "6 16 36 division-by-zero=1 index=3 no-input-read=1 overflow=25 shift-amount=1 static=5",
        "7 16 61 division-by-zero=8 index=3 no-input-read=2 overflow=37 shift-amount=1 static=10",
        "8 16 63 division-by-zero=4 index=1 no-input-read=2 overflow=45 static=11",
    ];
    // The m5.c1a hashes of the same programs: the emitted text and the input data, byte for byte, with the format
    // stamp read as 1 (the stamp is the only difference of format 2 for an integer program).
    let hashes = [
        0x371f_a252_2e66_81c7_u64,
        0x811a_4d82_af66_a078,
        0x7816_c1f1_77b7_08c2,
        0xa2e3_1f20_063e_9b07,
        0xd44f_7f81_932a_d6cd,
        0x3531_64f5_a863_23f9,
        0x6155_afa9_9718_9c32,
        0x9a61_7649_d659_8029,
    ];
    let mut lines = Vec::new();
    for (seed, hash) in (1..=8).zip(hashes) {
        let batch = generate_batch_in(Space::Integer, seed).unwrap();
        assert_eq!(
            crate::corpus::batch_hash_as_format(&batch.programs, 1),
            hash,
            "seed {seed}"
        );
        let mut line = format!(
            "{seed} {} {}",
            batch.programs.len(),
            batch.discarded_total()
        );
        for (reason, count) in &batch.discarded {
            line.push_str(&format!(" {reason}={count}"));
        }
        lines.push(line);
        for program in &batch.programs {
            assert!(!program.has_float());
        }
    }
    assert_eq!(lines, expected);
}

#[test]
fn a_batch_holds_seventeen_in_domain_programs_and_counts_its_discards() {
    for seed in 1..=8 {
        let batch = generate_batch(seed).unwrap();
        assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
        assert_eq!(
            u64::from(batch.discarded_total()) + PROGRAMS_PER_SEED as u64,
            batch.attempts
        );
        for program in &batch.programs {
            assert!(program.reads_input());
            let out = crate::interp::run(program).unwrap();
            assert_eq!(out.len(), program.count as usize);
        }
        let attempts: Vec<u64> = batch.programs.iter().map(|p| p.attempt).collect();
        assert!(attempts.windows(2).all(|w| w[0] < w[1]));
    }
}

#[test]
fn every_attempt_decodes_to_a_well_typed_program() {
    for seed in [1, 2, 3] {
        for attempt in 0..300 {
            let program = generate_attempt(seed, attempt);
            program
                .check()
                .unwrap_or_else(|e| panic!("seed {seed} attempt {attempt}: {e}"));
        }
    }
}

#[test]
fn a_truncated_or_empty_stream_still_decodes_to_a_valid_program() {
    let stream = byte_stream(7, 11, BYTES_PER_PROGRAM);
    for len in [0, 1, 2, 7, 8, 9, 31, 100, 500, 2000, BYTES_PER_PROGRAM - 1] {
        let program = decode(&stream[..len], 7, 11);
        program.check().unwrap_or_else(|e| panic!("len {len}: {e}"));
    }
    let empty = decode(&[], 7, 11);
    assert_eq!(empty.count, 1);
    // The all-zero stream selects the first option everywhere: one element, `u32` input, a literal result.
    assert_eq!(empty.inputs.len(), 1);
}

#[test]
fn discards_cover_the_domain_conditions_the_generator_can_reach() {
    let mut reasons = BTreeSet::new();
    let mut float_reasons = BTreeSet::new();
    for attempt in 0..3000 {
        if let Some(reason) = discard_reason(&generate_attempt_in(Space::Integer, 1, attempt)) {
            reasons.insert(reason);
        }
        if let Some(reason) = discard_reason(&generate_attempt(1, attempt)) {
            float_reasons.insert(reason);
        }
    }
    // The four float labels of R1-4, so the narrowing is visible in every run's counts.
    for violation in [
        Violation::FloatDomain,
        Violation::ComparisonOverlap,
        Violation::Unboundable,
        Violation::FloatConstruct,
    ] {
        assert!(
            float_reasons.contains(violation.label()),
            "{float_reasons:?}"
        );
    }
    assert!(
        !reasons
            .iter()
            .any(|r| r.starts_with("float-") || *r == "unboundable" || *r == "comparison-overlap"),
        "{reasons:?}"
    );
    for violation in [
        Violation::Overflow,
        Violation::DivisionByZero,
        Violation::ShiftAmount,
        Violation::Index,
    ] {
        assert!(reasons.contains(violation.label()), "{reasons:?}");
    }
    assert!(reasons.contains("no-input-read"), "{reasons:?}");
    assert!(reasons.contains("static"), "{reasons:?}");
}

#[test]
fn element_counts_include_the_workgroup_boundaries() {
    let counts: BTreeSet<u32> = (0..200).map(|a| generate_attempt(1, a).count).collect();
    for expected in [1, 63, 64, 65, 257] {
        assert!(counts.contains(&expected), "{counts:?}");
    }
    assert!(counts.iter().any(|c| ![1, 63, 64, 65, 257].contains(c)));
    assert!(counts.iter().all(|c| (1..4096).contains(c)));
}

#[test]
fn the_integer_corpus_covers_the_full_operator_set_for_both_types() {
    let mut total: BTreeMap<String, usize> = BTreeMap::new();
    for seed in 1..=8 {
        for program in generate_batch_in(Space::Integer, seed).unwrap().programs {
            for (key, count) in operator_counts(&program) {
                *total.entry(key).or_default() += count;
            }
        }
    }
    let mut expected = BTreeSet::new();
    for ty in ["u32", "i32"] {
        for op in BinOp::ALL {
            if !op.is_logical() {
                expected.insert(format!("{} {ty}", op.symbol()));
            }
        }
        expected.insert(format!("!{ty}"));
        expected.insert(format!("as {ty}"));
    }
    for op in ["&&", "||", "==", "!="] {
        expected.insert(format!("{op} bool"));
    }
    expected.insert("!bool".into());
    expected.insert("-".into());
    expected.insert("for".into());
    let missing: Vec<&String> = expected
        .iter()
        .filter(|key| !total.contains_key(*key))
        .collect();
    assert!(missing.is_empty(), "not covered: {missing:?}");
}

/// The literal zero divisors of `program`, dead code included.
fn literal_zero_divisors(program: &Program) -> usize {
    let mut found = 0;
    crate::ast::visit_exprs(program, &mut |e| {
        if let Expr::Bin {
            op: BinOp::Div | BinOp::Rem,
            r,
            ..
        } = e
        {
            found += usize::from(matches!(**r, Expr::Lit(_, 0)));
        }
    });
    found
}

/// The literal shift amounts of 32 or more in `program`, dead code included.
fn literal_wide_shifts(program: &Program) -> usize {
    let mut found = 0;
    crate::ast::visit_exprs(program, &mut |e| {
        if let Expr::Bin {
            op: BinOp::Shl | BinOp::Shr,
            r,
            ..
        } = e
        {
            found += usize::from(matches!(**r, Expr::Lit(_, 32..)));
        }
    });
    found
}

#[test]
fn no_decoded_program_has_a_literal_zero_divisor() {
    // 40 seeds x 500 attempts = 20000 raw programs, accepted or not (the validator rejects the literal everywhere).
    for seed in 1..=40 {
        for attempt in 0..500 {
            let program = generate_attempt(seed, attempt);
            assert_eq!(
                literal_zero_divisors(&program),
                0,
                "seed {seed} attempt {attempt}"
            );
        }
    }
}

#[test]
fn no_accepted_program_holds_a_static_constant_outside_the_domain() {
    // 100 seeds x 16 = 1600 accepted programs.
    for seed in 1..=100 {
        for program in generate_batch(seed).unwrap().programs {
            let label = format!("seed {seed} attempt {}", program.attempt);
            assert_eq!(interp::static_violation(&program), None, "{label}");
            assert_eq!(literal_wide_shifts(&program), 0, "{label}");
            assert_eq!(literal_zero_divisors(&program), 0, "{label}");
        }
    }
}

/// Seed 4, attempt 26 (a `v2 % v2` on the counter of an unrolled loop, in a branch no element takes) was accepted
/// until DIFFTEST_FORMAT 3 and made slangc refuse the kernel (E30002, slice m5.c2); formats 4 refused it. Since
/// format 5 the macro lowers that `%` to `__gust_rem`, which slangc compiles, so the program is accepted again
/// (the program of that attempt is the same program: the rule changed acceptance, not decoding).
#[test]
fn the_seed_4_dead_divisor_of_format_3_is_accepted_again() {
    let program = generate_attempt(4, 26);
    let text = crate::emit::emit_module(&program);
    assert!(text.contains("(v2 % v2)"), "{text}");
    assert_eq!(discard_reason(&program), None);
    assert!(
        generate_batch_in(Space::Float, 4)
            .unwrap()
            .programs
            .iter()
            .any(|accepted| accepted.attempt == 26)
    );
}

#[test]
fn the_static_discard_is_live_and_only_counts_what_the_interpreter_cannot_see() {
    let mut static_programs = 0;
    let mut wide_shifts = 0;
    for attempt in 0..3000 {
        let program = generate_attempt(1, attempt);
        wide_shifts += literal_wide_shifts(&program);
        if discard_reason(&program) == Some("static") {
            static_programs += 1;
            assert!(program.reads_input());
        }
    }
    assert!(static_programs > 50, "{static_programs}");
    assert!(
        wide_shifts > 0,
        "the generator no longer writes out-of-range literal shifts"
    );
}

#[test]
fn the_float_corpus_covers_the_float_subset() {
    let mut total: BTreeMap<String, usize> = BTreeMap::new();
    let mut loop_accumulators = 0;
    let mut float_outputs = 0;
    let mut float_to_int = 0;
    // 24 seeds: the batch holds two float programs since format 13 (48 of them, as three per seed over 16 seeds before).
    for seed in 1..=24 {
        for program in generate_batch(seed).unwrap().programs {
            for (key, count) in operator_counts(&program) {
                *total.entry(key).or_default() += count;
            }
            float_outputs += usize::from(program.out_ty() == Ty::F32);
            crate::ast::visit_exprs(&program, &mut |e| {
                if let Expr::Cast { to, e } = e {
                    float_to_int += usize::from(to.is_int() && e.ty() == Ty::F32);
                }
            });
            fn has_accumulator(stmts: &[Stmt]) -> bool {
                stmts.iter().any(|stmt| match stmt {
                    Stmt::For { body, .. } => {
                        body.iter().any(
                            |s| matches!(s, Stmt::Assign { value, .. } if value.ty() == Ty::F32),
                        ) || has_accumulator(body)
                    }
                    Stmt::If { then, els, .. } => has_accumulator(then) || has_accumulator(els),
                    _ => false,
                })
            }
            loop_accumulators += usize::from(has_accumulator(&program.body));
        }
    }
    let mut expected: Vec<String> = ["+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="]
        .iter()
        .map(|op| format!("{op} f32"))
        .collect();
    expected.extend(["-f32".to_owned(), "as f32".to_owned()]);
    let missing: Vec<&String> = expected
        .iter()
        .filter(|key| !total.contains_key(*key))
        .collect();
    assert!(missing.is_empty(), "not covered: {missing:?}\n{total:?}");
    assert!(
        loop_accumulators > 0,
        "no float loop accumulator in the corpus"
    );
    assert!(float_outputs > 0, "no f32 output in the corpus");
    assert!(float_to_int > 0, "no f32 -> integer cast in the corpus");
}

#[test]
fn float_programs_stay_inside_the_subset() {
    let mut floats = 0;
    for seed in 1..=3 {
        for attempt in 0..400 {
            let program = generate_attempt(seed, attempt);
            floats += usize::from(program.has_float());
            for input in program.inputs.iter().filter(|input| input.ty == Ty::F32) {
                for bits in &input.data {
                    let value = f32::from_bits(*bits);
                    assert!(
                        *bits == 0
                            || (value.abs() >= 2f32.powi(-20) && value.abs() < 2f32.powi(20)),
                        "input {value:e} is outside the bounded set"
                    );
                }
            }
            crate::ast::visit_exprs(&program, &mut |e| {
                // At most four written terms per chain; no float `%`; no bool buffer (`check` refuses both).
                assert!(
                    crate::ast::chain_terms(e) <= 4,
                    "seed {seed} attempt {attempt}"
                );
            });
            program.check().unwrap();
        }
    }
    assert!(floats > 400, "{floats} float programs in 1200 attempts");
}

#[test]
fn no_float_divisor_is_a_literal_zero_in_the_float_space_either() {
    // Covered by `no_decoded_program_has_a_literal_zero_divisor`, which scans the float space; this checks the
    // scan can see a float divisor at all.
    let mut float_divisions = 0;
    for attempt in 0..2000 {
        crate::ast::visit_exprs(&generate_attempt(1, attempt), &mut |e| {
            if let Expr::Bin {
                op: BinOp::Div,
                ty: Ty::F32,
                ..
            } = e
            {
                float_divisions += 1;
            }
        });
    }
    assert!(float_divisions > 20, "{float_divisions}");
}
