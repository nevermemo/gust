use super::{Step, Suite};
use crate::{
    checks::{CPU_TESTS, REFLECTION_TESTS},
    verify::{RELEASE_SLANG_TESTS, VerifyMode, cargo_calls as mode_cargo_calls, example_run_args},
    workspace::EXAMPLES,
};

const UNIT: [Step; 10] = [
    Step::Cargo(&["test", "--locked", "-p", "gust-macros"]),
    Step::Cargo(&["test", "--locked", "-p", "xtask"]),
    Step::Lint,
    Step::DocsCheck,
    Step::CheckArch,
    Step::CheckUi,
    Step::CheckCpu,
    Step::CountedCargo(&["test", "--locked", "-p", "gust", "--lib", "cpu::tests::"]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust",
        "--lib",
        "artifact::tests::",
    ]),
    Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "artifact"]),
];

#[test]
fn names_parse_case_insensitively_and_unknown_names_list_the_choices() {
    for (text, suite) in [
        ("Unit", Suite::Unit),
        ("slangc", Suite::Slangc),
        ("SPIRV", Suite::Spirv),
        ("Reflect", Suite::Reflect),
        ("gpu", Suite::Gpu),
        ("Full", Suite::Full),
    ] {
        assert_eq!(Suite::parse(text), Ok(suite), "{text}");
        assert!(Suite::names().contains(&suite.name()));
    }
    let error = Suite::parse("Smoke").unwrap_err();
    assert!(
        error.contains("Unit|Slangc|Spirv|Reflect|Gpu|Full"),
        "{error}"
    );
    assert!(Suite::parse("").is_err());
}

#[test]
fn unit_runs_tests_lint_docs_and_architecture_without_external_tools() {
    assert_eq!(Suite::Unit.steps(), UNIT.to_vec());
}

#[test]
fn slangc_is_unit_plus_the_core_library_tests_and_requires_slangc() {
    let mut expected = vec![Step::RequireTool("slangc")];
    expected.extend(UNIT);
    expected.push(Step::Cargo(&["test", "--locked", "-p", "gust", "--lib"]));
    expected.push(Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust",
        "--test",
        "diagnostics",
    ]));
    assert_eq!(Suite::Slangc.steps(), expected);
}

#[test]
fn the_other_tiers_alias_existing_commands() {
    assert_eq!(
        Suite::Spirv.steps(),
        vec![Step::Verify(VerifyMode::Artifacts, false)]
    );
    assert_eq!(Suite::Reflect.steps(), vec![Step::Feature("reflection")]);
    assert_eq!(
        Suite::Gpu.steps(),
        vec![
            Step::RequireTool("slangc"),
            Step::Cargo(&["test", "--locked", "-p", "gust-wgpu"])
        ]
    );
    assert_eq!(
        Suite::Full.steps(),
        vec![Step::Verify(VerifyMode::Full, true)]
    );
}

/// The cargo calls a step makes, as argument lists. Steps that call cargo through another
/// xtask command are listed by that command's data, so that no call hides behind a step.
/// The Full suite is not expanded: it runs `cargo fmt` (which has no `--locked`) and the
/// examples' `--include-ignored` tests, which are not locked, so it is not checked here.
fn cargo_calls(step: Step) -> Vec<Vec<String>> {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    match step {
        Step::Cargo(args) | Step::CountedCargo(args) => vec![owned(args)],
        Step::CheckCpu => CPU_TESTS.iter().map(|args| owned(args)).collect(),
        Step::Feature("reflection") => REFLECTION_TESTS.iter().map(|args| owned(args)).collect(),
        Step::Verify(VerifyMode::Artifacts, _) => EXAMPLES
            .iter()
            .map(|example| owned(&example_run_args(example)))
            .collect(),
        Step::Verify(VerifyMode::Full, _)
        | Step::RequireTool(_)
        | Step::Lint
        | Step::DocsCheck
        | Step::CheckArch => Vec::new(),
        // Its cargo call is built by `ui::cargo_args`, whose test checks `--locked`.
        Step::CheckUi => Vec::new(),
        other => panic!("{other:?}: list the cargo calls of this step in cargo_calls"),
    }
}

#[test]
fn every_cargo_step_is_locked() {
    let mut seen = Vec::new();
    for suite in Suite::names().into_iter().map(|n| Suite::parse(n).unwrap()) {
        for step in suite.steps() {
            for args in cargo_calls(step) {
                assert!(
                    args.iter().any(|arg| arg == "--locked"),
                    "{suite:?}: cargo {args:?}"
                );
                seen.push((suite, args));
            }
        }
    }
    // The nested calls of Reflect and Spirv are really among the calls checked.
    let count = |wanted: Suite| seen.iter().filter(|(suite, _)| *suite == wanted).count();
    assert_eq!(count(Suite::Reflect), 2);
    assert_eq!(count(Suite::Spirv), EXAMPLES.len());
    assert!(count(Suite::Unit) >= 2 && count(Suite::Gpu) >= 1);
}

#[test]
fn every_suite_says_what_it_does_and_does_not_cover() {
    for name in Suite::names() {
        let suite = Suite::parse(name).unwrap();
        assert!(
            !suite.covers().is_empty() && !suite.not_covered().is_empty(),
            "{name}"
        );
        assert!(!suite.steps().is_empty(), "{name}");
    }
    // The Unit tier must not claim anything that needs a tool.
    assert!(Suite::Unit.not_covered().contains("Slang compiles"));
    assert!(
        Suite::Unit
            .steps()
            .iter()
            .all(|s| !matches!(s, Step::RequireTool(_)))
    );
}

#[test]
fn the_full_gate_runs_the_release_slang_tests() {
    // The release-only fallback tests (`cfg(not(debug_assertions))`) run nowhere else.
    assert_eq!(
        RELEASE_SLANG_TESTS,
        [
            "test",
            "--locked",
            "--release",
            "-p",
            "gust",
            "--lib",
            "--",
            "slang",
            "cpu::tests::invocations_are_capped_at_the_id_range_in_release"
        ]
    );
    assert!(Suite::Full.covers().contains("release-profile"));
    // Suite::Full and check-full both run verify's Full mode (above and in main.rs); the
    // structured list of that mode's cargo calls is tested in tests/verify_steps.rs.
    let release = RELEASE_SLANG_TESTS.map(str::to_owned).to_vec();
    assert_eq!(
        mode_cargo_calls(VerifyMode::Full)
            .iter()
            .filter(|call| **call == release)
            .count(),
        1
    );
}

#[test]
fn check_cpu_is_a_unit_step_of_locked_example_test_runs() {
    let calls: Vec<Vec<String>> = CPU_TESTS
        .iter()
        .map(|args| args.iter().map(|arg| (*arg).to_owned()).collect())
        .collect();
    assert_eq!(
        calls,
        [
            ["test", "--locked", "-p", "vector-add"],
            ["test", "--locked", "-p", "polynomial"],
            ["test", "--locked", "-p", "signal-pipeline"],
            ["test", "--locked", "-p", "particle-step"],
            ["test", "--locked", "-p", "typed-pipeline"],
            ["test", "--locked", "-p", "atomic-counter"],
            ["test", "--locked", "-p", "staged-graph"],
            ["test", "--locked", "-p", "component-pool"]
        ]
    );
    // The step is in Unit (after check-ui) and so in Slangc, which contains Unit; CI runs Unit.
    for suite in [Suite::Unit, Suite::Slangc] {
        let steps = suite.steps();
        let ui = steps
            .iter()
            .position(|step| *step == Step::CheckUi)
            .unwrap();
        assert_eq!(steps[ui + 1], Step::CheckCpu, "{suite:?}");
    }
    assert!(Suite::Unit.covers().contains("check-cpu"));
    assert_eq!(Step::CheckCpu.label(), "cargo xtask check-cpu");
    // The calls the Unit suite makes include both, each locked (every_cargo_step_is_locked checks the flag).
    let unit_calls: Vec<Vec<String>> = Suite::Unit
        .steps()
        .into_iter()
        .flat_map(cargo_calls)
        .collect();
    for call in &calls {
        assert!(unit_calls.contains(call), "{call:?}");
    }
}

#[test]
fn check_cpu_and_check_ui_are_explained() {
    use crate::checks::explain_check;
    assert!(explain_check(Some("check-cpu")).is_ok());
    assert!(explain_check(Some("check-ui")).is_ok());
    assert!(explain_check(Some("check-nothing")).is_err());
}

#[test]
fn the_unit_suite_runs_the_gust_cpu_tests_with_the_count_guard() {
    // `gust::cpu` tests need neither slangc nor a GPU; without this step no CI job ran them.
    let step = Step::CountedCargo(&["test", "--locked", "-p", "gust", "--lib", "cpu::tests::"]);
    let steps = Suite::Unit.steps();
    assert_eq!(steps.iter().filter(|s| **s == step).count(), 1);
    assert!(steps.iter().all(|s| !matches!(s, Step::RequireTool(_))));
    assert_eq!(
        step.label(),
        "cargo test --locked -p gust --lib cpu::tests::"
    );
    assert!(Suite::Unit.covers().contains("gust::cpu"));
}

#[test]
fn the_unit_suite_runs_the_gust_artifact_tests_with_the_count_guard() {
    // Pure tests (no slangc, no GPU): the integration file and the counted library tests.
    let unit = Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust",
        "--lib",
        "artifact::tests::",
    ]);
    let integration = Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "artifact"]);
    let steps = Suite::Unit.steps();
    assert_eq!(steps.iter().filter(|s| **s == unit).count(), 1);
    assert_eq!(steps.iter().filter(|s| **s == integration).count(), 1);
    assert_eq!(
        unit.label(),
        "cargo test --locked -p gust --lib artifact::tests::"
    );
    assert_eq!(
        integration.label(),
        "cargo test --locked -p gust --test artifact"
    );
    assert!(Suite::Unit.covers().contains("gust::artifact"));
}
