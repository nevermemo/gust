use super::{
    RELEASE_LEDGER_TEST, RELEASE_ONLY_CPU_TEST, RELEASE_ONLY_SLANG_TEST, RELEASE_ONLY_TESTS,
    RELEASE_SLANG_TESTS, VerifyMode, cargo_calls, check_counted_call, counts_tests,
    example_run_args,
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
        "gust",
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
        "gust-wgpu",
        "--test",
        "mismatch",
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
            "gust-wgpu",
            "--test",
            "mismatch"
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
    let macros = owned(&["test", "--locked", "-p", "gust-macros"]);
    let core = owned(&["test", "--locked", "-p", "gust", "--lib"]);
    let wgpu = owned(&["test", "--locked", "-p", "gust-wgpu"]);
    assert_eq!(
        cargo_calls(VerifyMode::Fast),
        [owned(&FMT), macros, core, wgpu.clone()]
    );
    assert_eq!(cargo_calls(VerifyMode::Gpu), [wgpu]);
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
            }
            expected.extend(example_tests());
        }
        assert_eq!(counted, expected, "{mode:?}");
    }
    assert!(!counts_tests(&owned(&[
        "test",
        "--locked",
        "-p",
        "gust-wgpu"
    ])));
    assert!(!counts_tests(&owned(&RELEASE_LEDGER_TEST)));
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
