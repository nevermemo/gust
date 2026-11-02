use super::{DIFFTEST_UNIT_STEPS, Step, Suite, plan_suite, test_command};
use crate::{
    checks::{CPU_TESTS, GPU_SEMANTICS_TESTS, REFLECTION_TESTS},
    cpu_target::{Dirs, all_calls},
    difftest::{DifftestStep, Mode, Options, batch_args, short_circuit_args, steps},
    verify::{RELEASE_SLANG_TESTS, VerifyMode, cargo_calls as mode_cargo_calls, example_run_args},
    workspace::EXAMPLES,
};
use std::path::Path;

const UNIT: [Step; 17] = [
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
    Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "coretests_twin"]),
    Step::Cargo(&[
        "clippy",
        "--locked",
        "-p",
        "gust",
        "-p",
        "gust-wgpu",
        "--no-default-features",
        "--lib",
        "--",
        "-D",
        "warnings",
    ]),
    Step::Cargo(&[
        "check",
        "--locked",
        "-p",
        "gust",
        "-p",
        "gust-wgpu",
        "--no-default-features",
        "--lib",
    ]),
    Step::Cargo(&[
        "fmt",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
        "--",
        "--check",
    ]),
    Step::Cargo(&[
        "clippy",
        "--locked",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]),
    Step::Cargo(&[
        "test",
        "--locked",
        "--manifest-path",
        "tests/difftest/Cargo.toml",
    ]),
    Step::CheckDifftest(&["--cpu-only", "--seeds", "1-4"]),
];

#[test]
fn names_parse_case_insensitively_and_unknown_names_list_the_choices() {
    for (text, suite) in [
        ("Unit", Suite::Unit),
        ("cpu", Suite::Cpu),
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
        error.contains("Unit|Cpu|Slangc|Spirv|Reflect|Gpu|Full"),
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
fn skip_difftest_leaves_out_exactly_the_harness_steps_and_says_which() {
    let (run, skipped) = plan_suite(Suite::Unit, true).unwrap();
    assert_eq!(skipped, DIFFTEST_UNIT_STEPS.to_vec());
    assert_eq!(run.len() + skipped.len(), Suite::Unit.steps().len());
    assert!(run.iter().all(|step| !DIFFTEST_UNIT_STEPS.contains(step)));
    // The four steps are the last four of the Unit suite, the format check first and the batch last.
    let unit = Suite::Unit.steps();
    assert_eq!(unit[unit.len() - 4..], DIFFTEST_UNIT_STEPS);
    assert_eq!(
        DIFFTEST_UNIT_STEPS.map(|step| step.label()),
        [
            "cargo fmt --manifest-path tests/difftest/Cargo.toml -- --check",
            "cargo clippy --locked --manifest-path tests/difftest/Cargo.toml --all-targets -- -D warnings",
            "cargo test --locked --manifest-path tests/difftest/Cargo.toml",
            "cargo xtask check-difftest --cpu-only --seeds 1-4",
        ]
    );
    // Without the flag nothing is left out; the Slangc suite carries the same four steps.
    let (all, none) = plan_suite(Suite::Unit, false).unwrap();
    assert_eq!((all, none.len()), (Suite::Unit.steps(), 0));
    let (slangc, skipped) = plan_suite(Suite::Slangc, true).unwrap();
    assert_eq!(skipped, DIFFTEST_UNIT_STEPS.to_vec());
    assert!(slangc.contains(&Step::RequireTool("slangc")));
}

#[test]
fn skip_difftest_is_an_error_where_there_is_nothing_to_skip_and_a_flag_after_the_suite() {
    for suite in [
        Suite::Cpu,
        Suite::Spirv,
        Suite::Reflect,
        Suite::Gpu,
        Suite::Full,
    ] {
        let error = plan_suite(suite, true).unwrap_err();
        assert!(error.contains("--skip-difftest"), "{error}");
    }
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    for args in [
        owned(&["--skip-difftest", "--suite", "Unit"]),
        owned(&["--suite", "Unit", "--skip-difftest", "extra"]),
        owned(&["--suite", "Unit", "--skip"]),
        owned(&["--skip-difftest"]),
    ] {
        let error = test_command(&args).unwrap_err();
        assert!(error.contains("[--skip-difftest]"), "{error}");
    }
}

fn job_text(ci: &str, job: &str) -> String {
    let header = format!("  {job}:");
    let mut text = String::new();
    let mut found = false;
    for line in ci.lines() {
        let job_key = line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim_start().starts_with('#')
            && line.ends_with(':');
        if found && job_key {
            break;
        }
        found |= line == header;
        if found {
            text.push_str(line);
            text.push('\n');
        }
    }
    assert!(found, "no job {job} in the workflow");
    text
}

/// The CI jobs: the `unit` and `slang` jobs leave the harness crate's steps out, and the job `difftest` runs every one
/// of them as a step of its own (R4.10: the cold batch build is 88 s on the development PC and the `unit` job must keep
/// its time).
#[test]
fn ci_runs_the_harness_steps_in_a_job_of_their_own_and_the_unit_jobs_skip_them() {
    let ci = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows/ci.yml"),
    )
    .unwrap()
    .replace("\r\n", "\n");
    for (job, command) in [
        (
            "unit",
            "run: cargo --locked xtask test --suite Unit --skip-difftest\n",
        ),
        (
            "slang",
            "run: cargo --locked xtask test --suite Slangc --skip-difftest\n",
        ),
    ] {
        let text = job_text(&ci, job);
        assert!(text.contains(command), "{job}: {text}");
    }
    let difftest = job_text(&ci, "difftest");
    // One target directory for the job and a cache keyed on the harness lockfiles as well (fix round 2).
    assert!(
        difftest.contains("      CARGO_TARGET_DIR: ${{ github.workspace }}/target\n"),
        "{difftest}"
    );
    assert!(difftest.contains("uses: actions/cache@"), "{difftest}");
    assert!(
        difftest.contains("hashFiles('Cargo.lock', 'tests/difftest/**/Cargo.lock')"),
        "{difftest}"
    );
    for step in DIFFTEST_UNIT_STEPS {
        let label = step.label();
        // `cargo xtask check-difftest ...` is written in the workflow as `cargo --locked xtask check-difftest ...`.
        let line = match step {
            Step::CheckDifftest(_) => label.replacen("cargo xtask", "cargo --locked xtask", 1),
            _ => label,
        };
        assert!(
            difftest.contains(&format!("        run: {line}\n")),
            "{line}: {difftest}"
        );
    }
}

/// The text of the step `name` of a workflow job: from its `- name:` line to the next step.
fn step_text(job: &str, name: &str) -> String {
    let header = format!("      - name: {name}");
    let mut text = String::new();
    let mut found = false;
    for line in job.lines() {
        if found && line.starts_with("      - ") {
            break;
        }
        found |= line == header;
        if found {
            text.push_str(line);
            text.push('\n');
        }
    }
    assert!(found, "no step {name} in the job");
    text
}

/// A failing differential test in gpu-software must not skip the nightly evidence behind it (fix round 2, review of
/// m5.c2): the steps after it say `!cancelled()` and not the implicit `success()`, and the expression form is needed
/// because a bare `!` starts a YAML tag. The difftest step itself keeps the default, so the job still fails.
#[test]
fn gpu_software_runs_the_nightly_evidence_after_a_difftest_failure() {
    let workflow = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows/gpu-software.yml"),
    )
    .unwrap()
    .replace("\r\n", "\n");
    let job = job_text(&workflow, "gpu-software");
    for name in [
        "Examples on lavapipe (nightly list)",
        "Mismatch ledger in the release profile (nightly list)",
    ] {
        let step = step_text(&job, name);
        assert!(
            step.contains(
                "        if: ${{ !cancelled() && env.GPU_SOFTWARE_NIGHTLY == 'true' }}\n"
            ),
            "{step}"
        );
    }
    let difftest = step_text(
        &job,
        "Differential test (2 seeds; all 8 in the nightly list)",
    );
    assert!(!difftest.contains("        if:"), "{difftest}");
}

#[test]
fn the_other_tiers_alias_existing_commands() {
    assert_eq!(
        Suite::Spirv.steps(),
        vec![Step::Verify(VerifyMode::Artifacts, false)]
    );
    assert_eq!(Suite::Reflect.steps(), vec![Step::Feature("reflection")]);
    assert_eq!(Suite::Cpu.steps(), vec![Step::CheckCpuTarget]);
    assert_eq!(Step::CheckCpuTarget.label(), "cargo xtask check-cpu-target");
    assert_eq!(
        Suite::Gpu.steps(),
        vec![
            Step::RequireTool("slangc"),
            Step::Cargo(&["test", "--locked", "-p", "gust-wgpu"]),
            Step::CheckDifftest(&[])
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
        // The calls of `check-cpu-target` are listed by its own data (and pinned by its own tests).
        Step::CheckCpuTarget => all_calls(&Dirs::under("TARGET"))
            .into_iter()
            .map(|(_, call)| call.args)
            .collect(),
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
        // The calls of `check-difftest` are built by its own functions; the tier decides which of them run.
        Step::CheckDifftest(args) => {
            let options = Options::parse_with(
                &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
                1,
            )
            .unwrap();
            let mut calls = vec![batch_args(
                &options,
                Path::new("TARGET"),
                Path::new("REPORT"),
            )];
            if steps(options.mode).contains(&DifftestStep::ShortCircuitGpu) {
                calls.push(short_circuit_args());
            }
            calls
        }
        other => panic!("{other:?}: list the cargo calls of this step in cargo_calls"),
    }
}

#[test]
fn every_cargo_step_is_locked() {
    let mut seen = Vec::new();
    for suite in Suite::names().into_iter().map(|n| Suite::parse(n).unwrap()) {
        for step in suite.steps() {
            for args in cargo_calls(step) {
                // `cargo fmt` has no `--locked`: it reads no lockfile.
                assert!(
                    args.first().is_some_and(|first| first == "fmt")
                        || args.iter().any(|arg| arg == "--locked"),
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
fn the_unit_suite_builds_gust_and_gust_wgpu_without_default_features() {
    // The only observers of the `slang-runtime` off state: the workspace build unifies the feature on.
    let clippy =
        "cargo clippy --locked -p gust -p gust-wgpu --no-default-features --lib -- -D warnings";
    let check = "cargo check --locked -p gust -p gust-wgpu --no-default-features --lib";
    for suite in [Suite::Unit, Suite::Slangc] {
        let labels: Vec<String> = suite.steps().iter().map(|step| step.label()).collect();
        assert_eq!(labels.iter().filter(|label| *label == clippy).count(), 1);
        assert_eq!(labels.iter().filter(|label| *label == check).count(), 1);
    }
    assert!(Suite::Unit.covers().contains("slang-runtime"));
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

#[test]
fn the_unit_suite_runs_the_coretests_twin_test() {
    // Pure (no slangc, no adapter): the transcribed Rust core test cases on the CPU twin against their literals.
    let step = Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "coretests_twin"]);
    for suite in [Suite::Unit, Suite::Slangc] {
        assert_eq!(
            suite.steps().iter().filter(|s| **s == step).count(),
            1,
            "{suite:?}"
        );
    }
    assert_eq!(
        step.label(),
        "cargo test --locked -p gust --test coretests_twin"
    );
    assert!(Suite::Unit.covers().contains("coretests_twin"));
    assert!(
        Suite::Unit
            .steps()
            .iter()
            .all(|s| !matches!(s, Step::RequireTool(_)))
    );
}

#[test]
fn check_feature_gpu_semantics_runs_the_gpu_coretests() {
    // The GPU half of the coretests (m5.d2) runs where the other differential tests do; the Gpu suite runs every test
    // file of gust-wgpu anyway, so this list is the only place that names the file for `check-feature`.
    assert_eq!(
        GPU_SEMANTICS_TESTS,
        [
            "semantics",
            "numeric",
            "option",
            "result",
            "loops",
            "struct_assignment",
            "mismatch",
            "coretests",
        ]
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for test in GPU_SEMANTICS_TESTS {
        let file = root
            .join("crates/gust-wgpu/tests")
            .join(format!("{test}.rs"));
        assert!(
            file.is_file(),
            "{test}: crates/gust-wgpu/tests/{test}.rs does not exist"
        );
    }
    let explained =
        crate::checks::explanation(Some("gpu-semantics")).expect("gpu-semantics is explained");
    assert!(
        explained.contains("coretests"),
        "the gpu-semantics explanation names the coretests: {explained}"
    );
}

#[test]
fn the_tiers_wire_the_difftest_batch_in() {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    // Unit (and Slangc, which contains it): seeds 1 to 4, CPU-only, so it needs neither an adapter nor Slang.
    let unit = Step::CheckDifftest(&["--cpu-only", "--seeds", "1-4"]);
    for suite in [Suite::Unit, Suite::Slangc] {
        assert_eq!(
            suite.steps().iter().filter(|step| **step == unit).count(),
            1,
            "{suite:?}"
        );
    }
    assert_eq!(
        unit.label(),
        "cargo xtask check-difftest --cpu-only --seeds 1-4"
    );
    assert!(Suite::Unit.covers().contains("differential batch"));
    let Step::CheckDifftest(args) = unit else {
        unreachable!()
    };
    let options = Options::parse_with(&owned(args), 1).unwrap();
    assert_eq!(
        (options.mode, options.expected_programs()),
        (Mode::CpuOnly, 64)
    );
    assert!(!steps(options.mode).contains(&DifftestStep::ShortCircuitGpu));

    // Gpu: all eight seeds on the adapter, last, and that run includes the short_circuit GPU test.
    let gpu = Suite::Gpu.steps();
    assert_eq!(gpu.last(), Some(&Step::CheckDifftest(&[])));
    assert_eq!(
        Step::CheckDifftest(&[]).label(),
        "cargo xtask check-difftest"
    );
    assert!(Suite::Gpu.covers().contains("short_circuit"));
    let options = Options::parse_with(&[], 1).unwrap();
    assert_eq!(
        (options.mode, options.expected_programs()),
        (Mode::Gpu, 128)
    );
    assert!(steps(options.mode).contains(&DifftestStep::ShortCircuitGpu));

    // Neither the Cpu tier nor the Spirv and Reflect tiers run it; Full runs it through verify (integration checks).
    for suite in [Suite::Cpu, Suite::Spirv, Suite::Reflect, Suite::Full] {
        assert!(
            !suite
                .steps()
                .iter()
                .any(|step| matches!(step, Step::CheckDifftest(_))),
            "{suite:?}"
        );
    }
    assert!(Suite::Full.covers().contains("check-difftest"));
}
