use super::{
    RELEASE_LEDGER_TEST, RELEASE_LEDGER_TESTS, RELEASE_ONLY_CPU_TEST, RELEASE_ONLY_SLANG_TEST,
    RELEASE_ONLY_TESTS, RELEASE_SLANG_TESTS, STRICT_PORTABLE_PROFILE, STRICT_SUBGROUPS_PACKAGE,
    STRICT_SUBGROUPS_PROFILE, VerifyMode, cargo_calls, check_counted_call, check_ledger,
    counts_tests, example_run_args, strict_subgroups_args, strict_subgroups_portable_args,
    strict_subgroups_problems, strict_subgroups_test_args,
};
use crate::workspace::{EXAMPLES, SMOKE_EXAMPLES, workspace_test_args};

const ALL_MODES: [VerifyMode; 6] = [
    VerifyMode::Smoke,
    VerifyMode::Fast,
    VerifyMode::Gpu,
    VerifyMode::Examples,
    VerifyMode::Artifacts,
    VerifyMode::Full,
];

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

const FMT: [&str; 4] = ["fmt", "--all", "--", "--check"];
const CLIPPY: [&str; 7] = [
    "clippy",
    "--locked",
    "--workspace",
    "--all-targets",
    "--",
    "-D",
    "warnings",
];

fn example_tests() -> Vec<Vec<String>> {
    EXAMPLES
        .iter()
        .map(|example| owned(&["test", "-p", example, "--", "--include-ignored"]))
        .collect()
}

fn example_runs(examples: &[&str]) -> Vec<Vec<String>> {
    examples
        .iter()
        .map(|example| owned(&example_run_args(example)))
        .collect()
}

#[test]
fn the_full_gate_is_fmt_clippy_workspace_tests_release_slang_and_ledger_tests_examples_and_runs_in_order()
 {
    let mut expected = vec![owned(&FMT), owned(&CLIPPY), workspace_test_args()];
    // The harness crate (tests/difftest) is a workspace of its own: its fmt, clippy and tests (m5.c2, fix round 2).
    expected.push(owned(&[
        "fmt",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
        "--",
        "--check",
    ]));
    expected.push(owned(&[
        "clippy",
        "--locked",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]));
    expected.push(owned(&[
        "test",
        "--locked",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
    ]));
    expected.push(owned(&[
        "test",
        "--locked",
        "--release",
        "-p",
        "gust-compute",
        "--lib",
        "--",
        "slang",
        "cpu::tests::invocations_are_capped_at_the_id_range_in_release",
    ]));
    expected.push(owned(&[
        "test",
        "--locked",
        "--release",
        "-p",
        "gust-compute-wgpu",
        "--test",
        "mismatch",
        "--",
        "--nocapture",
    ]));
    expected.extend(example_tests());
    expected.extend(example_runs(EXAMPLES));
    assert_eq!(cargo_calls(VerifyMode::Full), expected);
}

#[test]
fn only_the_full_gate_runs_the_release_slang_tests_and_exactly_once() {
    // The release-only fallback tests (`cfg(not(debug_assertions))`) run nowhere else.
    let release = owned(&RELEASE_SLANG_TESTS);
    for mode in ALL_MODES {
        let count = cargo_calls(mode)
            .iter()
            .filter(|call| **call == release)
            .count();
        assert_eq!(count, usize::from(mode == VerifyMode::Full), "{mode:?}");
    }
    // It follows the debug workspace tests and precedes the examples.
    let calls = cargo_calls(VerifyMode::Full);
    let position = |wanted: &Vec<String>| calls.iter().position(|call| call == wanted).unwrap();
    assert!(position(&workspace_test_args()) < position(&release));
    assert!(position(&release) < position(&example_tests()[0]));
    assert!(position(&release) < position(&example_runs(EXAMPLES)[0]));
}

#[test]
fn only_the_full_gate_runs_the_release_ledger_and_exactly_once() {
    // The known-mismatch ledger's checks-off branch (D38) runs nowhere else: debug builds detect
    // `overflow-checks` on, so a release pass is the only way every mode-keyed row is exercised.
    let ledger = owned(&RELEASE_LEDGER_TEST);
    assert_eq!(
        ledger,
        owned(&[
            "test",
            "--locked",
            "--release",
            "-p",
            "gust-compute-wgpu",
            "--test",
            "mismatch",
            "--",
            "--nocapture"
        ])
    );
    for mode in ALL_MODES {
        let count = cargo_calls(mode)
            .iter()
            .filter(|call| **call == ledger)
            .count();
        assert_eq!(count, usize::from(mode == VerifyMode::Full), "{mode:?}");
    }
    // It follows the release Slang tests and precedes the examples.
    let calls = cargo_calls(VerifyMode::Full);
    let position = |wanted: &Vec<String>| calls.iter().position(|call| call == wanted).unwrap();
    assert!(position(&owned(&RELEASE_SLANG_TESTS)) < position(&ledger));
    assert!(position(&ledger) < position(&example_tests()[0]));
    assert!(position(&ledger) < position(&example_runs(EXAMPLES)[0]));
}

#[test]
fn the_other_modes_keep_their_cargo_calls() {
    let macros = owned(&["test", "--locked", "-p", "gust-compute-macros"]);
    let core = owned(&["test", "--locked", "-p", "gust-compute", "--lib"]);
    let wgpu = owned(&["test", "--locked", "-p", "gust-compute-wgpu"]);
    assert_eq!(
        cargo_calls(VerifyMode::Fast),
        [owned(&FMT), macros, core, wgpu.clone()]
    );
    let algo = owned(&["test", "--locked", "-p", "gust-algo"]);
    assert_eq!(cargo_calls(VerifyMode::Gpu), [wgpu, algo]);
    let mut smoke = vec![owned(&FMT), owned(&CLIPPY), workspace_test_args()];
    smoke.extend(example_runs(SMOKE_EXAMPLES));
    assert_eq!(cargo_calls(VerifyMode::Smoke), smoke);
    let mut examples = example_tests();
    examples.extend(example_runs(EXAMPLES));
    assert_eq!(cargo_calls(VerifyMode::Examples), examples);
    assert_eq!(cargo_calls(VerifyMode::Artifacts), example_runs(EXAMPLES));
}

#[test]
fn only_fmt_and_the_example_tests_are_not_locked() {
    // docs/development/VALIDATION.md names exactly these as the unlocked cargo calls of the
    // verify modes; a new unlocked call must change both.
    for mode in ALL_MODES {
        for call in cargo_calls(mode) {
            let locked = call.iter().any(|arg| arg == "--locked");
            let exempt = call[0] == "fmt" || call.iter().any(|arg| arg == "--include-ignored");
            assert!(locked || exempt, "{mode:?}: cargo {call:?}");
            assert!(
                !(locked && exempt),
                "{mode:?}: cargo {call:?} is exempt but locked"
            );
        }
    }
}

#[test]
fn the_examples_tests_and_the_filtered_release_slang_tests_must_run_a_test() {
    // `cargo test` exits 0 with no test, so these calls are checked against their output (see
    // `require_tests_ran`). The unfiltered package and integration-target calls are not.
    for mode in ALL_MODES {
        let counted = cargo_calls(mode)
            .into_iter()
            .filter(|call| counts_tests(call))
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        if matches!(mode, VerifyMode::Examples | VerifyMode::Full) {
            if mode == VerifyMode::Full {
                expected.push(owned(&RELEASE_SLANG_TESTS));
                expected.push(owned(&RELEASE_LEDGER_TEST));
            }
            expected.extend(example_tests());
        }
        assert_eq!(counted, expected, "{mode:?}");
    }
    assert!(!counts_tests(&owned(&[
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu"
    ])));
    // The release ledger is counted too (m5.f1): `check_counted_call` judges it by name below.
    assert!(counts_tests(&owned(&RELEASE_LEDGER_TEST)));
}

/// The tests of `crates/gust-wgpu/tests/mismatch.rs`, named as literals here: renaming a constant alone must not make
/// the guard follow it.
const LEDGER_NAMES: [&str; 6] = [
    "known_mismatches_pin_the_gpu_the_plain_rust_and_the_cpu_twin_outcome",
    "every_pinned_ledger_id_is_documented_and_every_documented_id_is_pinned",
    "the_emitted_wgsl_uses_plain_operators_and_conversions",
    "the_ledger_mode_is_the_overflow_mode_of_this_crate",
    "the_out_of_bounds_store_twin_stops_at_the_store_after_a_load_that_succeeded",
    "a_kernel_without_storage_parameters_runs_on_the_twin",
];

/// Captured output of the ledger call: one line per test with its outcome, then the summary line.
fn ledger_output(outcomes: [&str; 6], summary: &str) -> String {
    let mut text = String::from("\nrunning 6 tests\n");
    for (name, outcome) in LEDGER_NAMES.iter().zip(outcomes) {
        text.push_str(&format!("test {name} ... {outcome}\n"));
    }
    text.push_str(&format!("\n{summary}\n"));
    text
}

const LEDGER_OK: &str =
    "test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.1s";

#[test]
fn the_release_ledger_call_must_report_every_ledger_test_ok_and_none_ignored() {
    assert_eq!(RELEASE_LEDGER_TESTS, LEDGER_NAMES);
    let call = owned(&RELEASE_LEDGER_TEST);
    let good = ledger_output(["ok"; 6], LEDGER_OK);
    assert_eq!(check_counted_call(&call, &good), Ok(()));
    // Colour sequences (CI forces colour) do not hide an `ok`.
    let coloured = good.replace("... ok", "... \u{1b}[32mok\u{1b}[0m");
    assert_eq!(check_counted_call(&call, &coloured), Ok(()));
    // One test `#[ignore]`d: the summary says so.
    let mut outcomes = ["ok"; 6];
    outcomes[3] = "ignored";
    let ignored = ledger_output(
        outcomes,
        "test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.1s",
    );
    let error = check_counted_call(&call, &ignored).unwrap_err();
    assert!(error.contains("1 test(s) reported ignored"), "{error}");
    // The same lines under a summary that says nothing about it still fail, and by name.
    let error = check_counted_call(&call, &ledger_output(outcomes, LEDGER_OK)).unwrap_err();
    assert!(
        error.contains(&format!("`{}`", LEDGER_NAMES[3])) && !error.contains(LEDGER_NAMES[0]),
        "{error}"
    );
    // A failed test, a missing summary and a run of nothing.
    outcomes[3] = "FAILED";
    let failed = ledger_output(
        outcomes,
        "test result: FAILED. 5 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.1s",
    );
    assert!(check_counted_call(&call, &failed).is_err());
    assert!(check_counted_call(&call, &good.replace(LEDGER_OK, "")).is_err());
    let nothing = "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out\n";
    assert!(check_counted_call(&call, nothing).is_err());
}

#[test]
fn a_renamed_removed_or_filtered_out_ledger_test_fails_by_name() {
    let call = owned(&RELEASE_LEDGER_TEST);
    for name in LEDGER_NAMES {
        // The test is absent from the output (deleted, renamed, or filtered out); five others still pass.
        let lines =
            ledger_output(["ok"; 6], LEDGER_OK).replace(&format!("test {name} ... ok\n"), "");
        let error = check_counted_call(&call, &lines).unwrap_err();
        assert!(error.contains(&format!("`{name}`")), "{name}: {error}");
        for other in LEDGER_NAMES.iter().filter(|other| **other != name) {
            assert!(!error.contains(other), "{name}: {error}");
        }
        assert!(error.contains("RELEASE_LEDGER_TESTS"), "{error}");
    }
    // A name inside a module is not the file's test of that name.
    let renamed = ledger_output(["ok"; 6], LEDGER_OK).replace(
        "test a_kernel_without_storage",
        "test tests::a_kernel_without_storage",
    );
    assert!(check_counted_call(&call, &renamed).is_err());
}

/// What the ledger tests print on stderr with `--nocapture` (mismatch.rs `eprintln!`): the adapter, the detected
/// overflow mode and the per-case rows.
const LEDGER_STDERR: &str = "S2 adapter: name=NVIDIA GeForce RTX 5090 device_type=DiscreteGpu\n\
    S2 overflow checks detected in this build: Wrapping\n\
    S2 | KM-01 | gpu=1 | rust=2 | twin=2\n\
    S2 ledger file: 132 lines in target/tmp/ledger-hardware.txt\n";

#[test]
fn the_release_ledger_runs_with_nocapture_and_the_guard_reads_stdout_only() {
    // The release run keeps its stderr evidence (adapter, overflow mode, `S2 |` rows) in the log (m5.f1, review).
    let call = owned(&RELEASE_LEDGER_TEST);
    assert_eq!(call[call.len() - 2..], ["--", "--nocapture"]);
    let stdout = ledger_output(["ok"; 6], LEDGER_OK);
    // stderr is a separate stream: the guard passes on stdout alone ...
    assert!(LEDGER_STDERR.contains("S2 | KM-01"));
    assert_eq!(check_counted_call(&call, &stdout), Ok(()));
    // ... and also when a console shows both streams, the stderr lines on lines of their own between the tests.
    let merged = stdout.replace(
        "test the_emitted_wgsl",
        &format!("{LEDGER_STDERR}test the_emitted_wgsl"),
    );
    assert_eq!(check_counted_call(&call, &merged), Ok(()));
    // A name line missing from stdout fails by name, however much stderr says: the guard never reads stderr, so a
    // `test .. ok` text that only the stderr stream carries is not a pass.
    let missing = LEDGER_NAMES[2];
    let stdout = stdout.replace(&format!("test {missing} ... ok\n"), "");
    let stderr = format!("{LEDGER_STDERR}test {missing} ... ok\n");
    let error = check_counted_call(&call, &stdout).unwrap_err();
    assert!(error.contains(&format!("`{missing}`")), "{error}");
    assert!(stderr.contains(missing));
}

/// The list names exactly the `#[test]` functions of the ledger file: a test added there but not here would be
/// unguarded, one removed there but still named here would fail the gate.
#[test]
fn the_ledger_names_are_the_test_functions_of_the_ledger_file() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/gust-wgpu/tests/mismatch.rs");
    let source = std::fs::read_to_string(path).unwrap().replace("\r\n", "\n");
    assert!(!source.contains("#[ignore"), "an ignored ledger test");
    let mut found = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "#[test]" {
            // Other attributes (`#[cfg(..)]`, `#[should_panic]`) may sit between the marker and the function.
            let name = lines
                .by_ref()
                .find_map(|next| next.strip_prefix("fn "))
                .and_then(|rest| rest.split('(').next())
                .expect("a `#[test]` without a function after it");
            found.push(name.to_owned());
        }
    }
    assert_eq!(found, LEDGER_NAMES);
}

#[test]
fn check_ledger_makes_the_call_of_the_full_mode_and_takes_no_arguments() {
    assert!(cargo_calls(VerifyMode::Full).contains(&owned(&RELEASE_LEDGER_TEST)));
    assert_eq!(
        check_ledger(&owned(&["extra"])),
        Err("check-ledger takes no arguments".to_owned())
    );
}

#[test]
fn the_release_call_must_report_both_release_only_tests_by_name() {
    // The names are pinned as literals here: renaming a constant alone must not make this test follow it.
    const SLANG: &str = "slang::source_map::tests::unparsable_match_marker_number_falls_back_to_the_map_ordinal_in_release";
    const CPU: &str = "cpu::tests::invocations_are_capped_at_the_id_range_in_release";
    assert_eq!(RELEASE_ONLY_SLANG_TEST, SLANG);
    assert_eq!(RELEASE_ONLY_CPU_TEST, CPU);
    assert_eq!(RELEASE_ONLY_TESTS, [SLANG, CPU]);
    // Both are filters of the release call (libtest ORs them), and the `slang` filter is unchanged.
    let release = owned(&RELEASE_SLANG_TESTS);
    assert_eq!(release[release.len() - 3..], ["--", "slang", CPU]);
    let summary = "test result: ok. 40 passed; 0 failed; 0 ignored; 0 measured; 70 filtered out; finished in 0.1s\n";
    let other = "test slang::emit::tests::something_else ... ok\n";
    let line = |name: &str, outcome: &str| format!("test {name} ... {outcome}\n");
    let both = format!("{other}{}{}{summary}", line(SLANG, "ok"), line(CPU, "ok"));
    assert_eq!(check_counted_call(&release, &both), Ok(()));
    // Dozens of other tests ran but one named test is absent (moved, renamed or filtered out): each case fails.
    let only_slang = format!("{other}{}{summary}", line(SLANG, "ok"));
    let error = check_counted_call(&release, &only_slang).unwrap_err();
    assert!(
        error.contains(&format!("`{CPU}`")) && !error.contains(&format!("`{SLANG}`")),
        "{error}"
    );
    let only_cpu = format!("{other}{}{summary}", line(CPU, "ok"));
    let error = check_counted_call(&release, &only_cpu).unwrap_err();
    assert!(
        error.contains(&format!("`{SLANG}`")) && !error.contains(&format!("`{CPU}`")),
        "{error}"
    );
    assert!(check_counted_call(&release, &format!("{other}{summary}")).is_err());
    // Reported ignored or failed is not ok either.
    let ignored = format!("{}{}{summary}", line(SLANG, "ok"), line(CPU, "ignored"));
    assert!(check_counted_call(&release, &ignored).is_err());
    let failed = format!("{}{}{summary}", line(SLANG, "FAILED"), line(CPU, "ok"));
    assert!(check_counted_call(&release, &failed).is_err());
    // The count still applies first.
    assert!(check_counted_call(&release, &both.replace(summary, "")).is_err());
    // Other counted calls (the examples) do not need the names.
    let example = owned(&["test", "-p", "polynomial", "--", "--include-ignored"]);
    assert_eq!(check_counted_call(&example, summary), Ok(()));
}

// ---- the strict proof of the subgroups capability (m8.e) ----

/// The strict run is the locked, quiet run of one package, in the examples and full modes only, and the package is not
/// one of the examples the other modes run (it writes no SPIR-V and runs under a profile of its own).
#[test]
fn the_strict_subgroups_run_is_one_locked_package_and_not_an_example_of_the_other_modes() {
    assert_eq!(
        strict_subgroups_args(),
        owned(&["run", "--locked", "--quiet", "-p", "subgroup-reduce"])
    );
    assert_eq!(STRICT_SUBGROUPS_PACKAGE, "subgroup-reduce");
    assert_eq!(STRICT_SUBGROUPS_PROFILE, "native+subgroups");
    let tests = strict_subgroups_test_args();
    assert!(tests.contains(&"--locked".to_owned()));
    assert!(counts_tests(&tests), "the test call must be a counted one");
    assert!(!EXAMPLES.contains(&STRICT_SUBGROUPS_PACKAGE));
    for mode in ALL_MODES {
        assert!(
            !cargo_calls(mode)
                .iter()
                .any(|call| call.iter().any(|arg| arg == STRICT_SUBGROUPS_PACKAGE)),
            "{mode:?}"
        );
    }
}

/// The source calls the strict check in the examples and full modes only, after the mode's own cargo calls.
#[test]
fn the_strict_subgroups_check_runs_in_the_examples_and_full_modes_after_the_cargo_calls() {
    let source = include_str!("../verify.rs").replace("\r\n", "\n");
    let body = &source[source.find("fn verify_inner(").expect("verify_inner")..];
    let calls = body.find("for args in cargo_calls(mode) {").unwrap();
    let strict = body
        .find("matches!(mode, VerifyMode::Examples | VerifyMode::Full)")
        .unwrap();
    let call = body.find("check_strict_subgroups_example(root)").unwrap();
    let dual = body.find("if runs_check_dual(mode) {").unwrap();
    assert!(calls < strict && strict < call && call < dual);
}

/// m8.x1: the example is also built under plain `native` (the portable bodies, whose committed artifacts the strict run
/// never embeds), in the same target directory, after the example's tests and before the strict run.
#[test]
fn the_strict_check_also_builds_the_example_under_plain_native_before_the_strict_run() {
    assert_eq!(STRICT_PORTABLE_PROFILE, "native");
    assert_ne!(STRICT_PORTABLE_PROFILE, STRICT_SUBGROUPS_PROFILE);
    assert_eq!(
        strict_subgroups_portable_args(),
        owned(&["build", "--locked", "--quiet", "-p", "subgroup-reduce"])
    );
    let source = include_str!("../verify.rs").replace("\r\n", "\n");
    let body = &source[source
        .find("fn check_strict_subgroups_example(")
        .expect("the check")..];
    let body = &body[..body.find("\n}\n").expect("its end")];
    let tests = body.find("run_cargo_test_captured(root").unwrap();
    let portable = body.find("STRICT_PORTABLE_PROFILE").unwrap();
    let portable_args = body.find("strict_subgroups_portable_args()").unwrap();
    let strict = body.find("strict_subgroups_args()").unwrap();
    assert!(tests < portable && portable < portable_args && portable_args < strict);
    // Both builds share the target directory `target`, so cargo reuses what the profile change leaves.
    assert!(
        body.contains("&target,"),
        "the portable build lost the shared target directory"
    );
}

const STRICT_GOOD: &str = "subgroup-reduce: adapter NVIDIA GeForce RTX 5090\n\
    subgroup-reduce: lane count 32 to 32 over 300 invocations in 10 subgroups\n\
    subgroup-reduce: ballot popcount: 150 elements over 10, as the host counts them\n\
    subgroup-reduce: prefix sum: the subgroups' last prefix sums add up to 4500, the sum of the data\n\
    subgroup-reduce: OK body=subgroups\n";

/// Mutations: each line of the output is needed, and the portable body is never accepted as the proof.
#[test]
fn the_strict_run_is_judged_on_its_printed_lines() {
    assert!(strict_subgroups_problems(STRICT_GOOD).is_empty());
    for dropped in [
        "OK body=subgroups",
        "lane count ",
        "ballot popcount: ",
        "prefix sum: ",
    ] {
        let text: String = STRICT_GOOD
            .lines()
            .filter(|line| !line.contains(dropped))
            .map(|line| format!("{line}\n"))
            .collect();
        let problems = strict_subgroups_problems(&text);
        assert_eq!(problems.len(), 1, "{dropped}: {problems:?}");
        assert!(problems[0].contains(dropped), "{problems:?}");
    }
    let portable = "subgroup-reduce: adapter x\n\
        subgroup-reduce: the portable body ran (this build has no subgroups capability): nothing to check\n\
        subgroup-reduce: OK body=portable\n";
    let problems = strict_subgroups_problems(portable);
    assert!(
        problems.iter().any(|p| p.contains("ran the portable body")),
        "{problems:?}"
    );
    assert!(strict_subgroups_problems("").len() >= 4);
}
