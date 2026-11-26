use super::{
    DIFFTEST_FMT, DIFFTEST_UNIT_STEPS, NamedTests, Step, Suite, TestRequest, parse_test_args,
    plan_suite, run_label, run_step_with, step_time_line, test_command, timing_summary,
};
use crate::gpu_jobs::{
    End, JOB_TIME_LIMIT, Job, JobOutput, Process, block, call_with, is_package_variable, judge,
    library_path, package_of, plan_jobs, refuse_call, run_in_order, start_order, supervise,
    test_executables,
};
use crate::{
    checks::{CPU_TESTS, GPU_SEMANTICS_TESTS, REFLECTION_TESTS},
    cpu_target::{Dirs, all_calls},
    difftest::{DifftestStep, Mode, Options, batch_args, short_circuit_args, steps},
    process::{TestTotals, sum_test_results},
    validation::AdapterRecord,
    verify::{
        ALGO_GPU_CALL, ALGO_GPU_TESTS, GPU_SUITE_CALL, GPU_SUITE_PIN, GPU_TARGET_PINS,
        RELEASE_SLANG_TESTS, VerifyMode, adapter_line, algo_profile_call,
        cargo_calls as mode_cargo_calls, check_gpu_suite_counts, example_run_args, judges_algo_gpu,
        run_gpu_profile_suite, run_gpu_target,
    },
    workspace::EXAMPLES,
};
use std::{cell::RefCell, path::Path};

const TOOL_FREE_CORE_DOC: Step = Step::ToolFreeCargo {
    args: &[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--features",
        "slang-runtime",
        "--doc",
        "--",
        "--skip",
        "slang::compile",
        "--skip",
        "reflect::compile::compile_reflected",
        "--skip",
        "reflect::model::CompiledReflection",
        "--skip",
        "reflect::compile::CleanCompilation",
        "--skip",
        "reflect::compile::reflect",
        "--skip",
        "artifact::Verified",
    ],
    hide_vulkan: false,
};
const TOOL_FREE_WGPU_DOC: Step = Step::ToolFreeCargo {
    args: &[
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu",
        "--features",
        "slang-runtime",
        "--doc",
        "--",
        "guide",
        "device::AdapterChoice",
        "GpuBuffer<T>",
        "GpuUniform<T>",
        "HeadlessDevice::read_typed_buffer_async",
    ],
    hide_vulkan: true,
};

const UNIT: [Step; 32] = [
    Step::Cargo(&["test", "--locked", "-p", "gust-compute-macros"]),
    Step::Cargo(&["test", "--locked", "-p", "xtask"]),
    Step::Lint,
    Step::DocsCheck,
    Step::CheckArch,
    Step::CheckUi,
    Step::CheckCpu,
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--lib",
        "cpu::tests::",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--lib",
        "artifact::tests::",
    ]),
    Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "artifact",
    ]),
    Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "coretests_twin",
    ]),
    Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "uniforms_twin",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "pixel_twin",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "textures_twin",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "parallel",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "routed_handle",
    ]),
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "toolchain",
    ]),
    TOOL_FREE_CORE_DOC,
    TOOL_FREE_WGPU_DOC,
    Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "cargo-gust",
        "--",
        "--skip",
        "export::uniform_tests::the_helper_builds_the_stored_artifacts",
        "--skip",
        "testdata::variant::the_helper_builds_the_stored_artifacts",
        "--exact",
    ]),
    Step::Cargo(&[
        "clippy",
        "--locked",
        "-p",
        "gust-compute",
        "-p",
        "gust-compute-wgpu",
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
        "gust-compute",
        "-p",
        "gust-compute-wgpu",
        "--no-default-features",
        "--lib",
    ]),
    // CI's `precompiled` job, word for word (ci.yml).
    Step::Cargo(&[
        "fmt",
        "--manifest-path",
        "tests/precompiled/Cargo.toml",
        "--all",
        "--",
        "--check",
    ]),
    Step::Cargo(&[
        "clippy",
        "--locked",
        "--manifest-path",
        "tests/precompiled/Cargo.toml",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]),
    // The bench workspace (m14.benchfmt).
    Step::Cargo(&[
        "fmt",
        "--manifest-path",
        "bench/Cargo.toml",
        "--all",
        "--",
        "--check",
    ]),
    Step::Cargo(&[
        "clippy",
        "--locked",
        "--manifest-path",
        "bench/Cargo.toml",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]),
    Step::ToolFreeCargo {
        args: &[
            "test",
            "--locked",
            "--manifest-path",
            "bench/Cargo.toml",
            "-p",
            "gust-bench",
            "--",
            "--exact",
            "--skip",
            "gust_gpu::tests::a_cold_start_records_the_route_that_ran",
            "--skip",
            "gust_gpu::tests::a_w6_graph_reports_a_gpu_time_for_one_run_and_for_a_repeated_batch",
            "--skip",
            "gust_gpu::tests::a_w7_sum_is_one_batch_and_its_gpu_time_follows_the_adapter",
            "--skip",
            "gust_gpu::tests::every_gust_stage_is_inside_its_own_bound_in_both_forms",
            "--skip",
            "gust_gpu::tests::every_workload_passes_its_bound_on_gust_and_an_undispatched_output_fails",
            "--skip",
            "gust_gpu::tests::pfonepass_rows_pass_the_w6_bound_and_record_the_passes_they_name",
            "--skip",
            "gust_gpu::tests::the_bound_falls_back_to_the_looser_of_v0_and_v1_where_v1_is_not_admitted",
            "--skip",
            "wgsl_wgpu::tests::a_w7_batch_reads_one_timestamp_pair_per_repeat",
            "--skip",
            "wgsl_wgpu::tests::ablation_backends_flip_one_switch_and_have_their_own_ids",
            "--skip",
            "wgsl_wgpu::tests::all_on_has_every_switch_on_and_a_fresh_vec_like_percall",
            "--skip",
            "wgsl_wgpu::tests::each_switch_changes_exactly_what_it_claims",
            "--skip",
            "wgsl_wgpu::tests::every_hand_stage_is_inside_its_own_bound_on_the_gpu",
            "--skip",
            "wgsl_wgpu::tests::every_workload_passes_its_bound_on_hand_wgsl_in_both_modes",
            "--skip",
            "wgsl_wgpu::tests::the_unchecked_rows_have_their_own_ids_and_the_host_shape_of_their_twin",
        ],
        hide_vulkan: true,
    },
    Step::Cargo(&[
        "run",
        "--locked",
        "--quiet",
        "-p",
        "cargo-gust",
        "--",
        "check",
        "--manifest-path",
        "bench/gust-bench-pre/Cargo.toml",
        "--allow-slang-mismatch",
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

/// Some doc tests of `gust-compute` launch slangc or the native reflection helper and most of those of
/// `gust-compute-wgpu` need an adapter (CI run 43774079389). Unit, the required check, keeps exactly two filtered doc
/// steps (the tool-free gust-compute ones and the adapter-free wgpu ones); Slangc has those and then the unfiltered gust-compute
/// step right after the Unit steps; the other wgpu doc tests run in the Gpu suite through its
/// `cargo test -p gust-compute-wgpu` call.
#[test]
fn doc_tests_run_where_their_resources_exist() {
    let is_doc = |step: &Step| match step {
        Step::Cargo(args) | Step::CountedCargo(args) | Step::ToolFreeCargo { args, .. } => {
            args.contains(&"--doc")
        }
        _ => false,
    };
    let tool_free = TOOL_FREE_CORE_DOC;
    let guide = TOOL_FREE_WGPU_DOC;
    let unfiltered = Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--features",
        "slang-runtime",
        "--doc",
    ]);
    let unit: Vec<Step> = Suite::Unit.steps().into_iter().filter(is_doc).collect();
    assert_eq!(unit, [tool_free, guide]);
    let slangc = Suite::Slangc.steps();
    assert_eq!(slangc[0], Step::RequireTool("slangc"));
    let docs: Vec<&Step> = slangc.iter().filter(|step| is_doc(step)).collect();
    assert_eq!(docs, [&unit[0], &unit[1], &unfiltered]);
    assert_eq!(slangc[1 + UNIT.len()], unfiltered);
    assert!(Suite::Gpu.covers().contains("doc tests of that crate"));
    assert!(
        Suite::Slangc
            .covers()
            .contains("all the doc tests of gust-compute")
    );
    assert!(
        Suite::Unit
            .covers()
            .contains("doc tests of gust-compute that need no tool")
    );
}

#[test]
fn slangc_is_unit_plus_the_core_library_tests_and_requires_slangc() {
    let mut expected = vec![Step::RequireTool("slangc")];
    expected.extend(UNIT);
    expected.push(Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--features",
        "slang-runtime",
        "--doc",
    ]));
    expected.push(Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--lib",
    ]));
    expected.push(Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "diagnostics",
    ]));
    expected.push(Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu",
        "--test",
        "naga_layout",
    ]));
    // m8.x1: the helper-backed tests of cargo-gust, by name, and the variant gate under the capability.
    expected.push(Step::Named(&NamedTests {
        args: &[
            "test",
            "--locked",
            "-p",
            "cargo-gust",
            "--",
            "--include-ignored",
            "the_helper_builds_the_stored_artifacts",
            "schema_3_output",
        ],
        names: &[
            "export::uniform_tests::the_helper_builds_the_stored_artifacts",
            "helper::tests::schema_3_output_of_the_built_helper_matches_the_committed_expectations",
            "testdata::variant::the_helper_builds_the_stored_artifacts",
        ],
        profile: None,
        lines: &[],
    }));
    expected.push(Step::Named(&NamedTests {
        args: &[
            "test",
            "--locked",
            "-p",
            "cargo-gust",
            "--",
            "--include-ignored",
            "--show-output",
            "testdata::variant::",
        ],
        names: &[
            "testdata::variant::the_helper_builds_the_stored_artifacts",
            "testdata::variant::the_macro_makes_today_what_the_stored_selected_bodies_are",
            "testdata::variant::the_plan_of_the_stored_records_is_the_stored_manifest_byte_for_byte",
            "testdata::variant::the_stored_manifest_reads_back_and_is_what_the_sources_give",
            "testdata::variant::the_stored_records_are_the_macros_and_carry_the_facts_of_their_bodies",
            "testdata::variant::without_the_variants_the_same_kernel_gives_a_schema_1_manifest",
        ],
        profile: Some("dev+subgroups"),
        lines: &["checked under capabilities [Subgroups]: "],
    }));
    assert_eq!(Suite::Slangc.steps(), expected);
}

#[test]
fn skip_difftest_leaves_out_exactly_the_harness_steps_and_says_which() {
    let (run, skipped) = plan_suite(Suite::Unit, true).unwrap();
    assert_eq!(skipped, DIFFTEST_UNIT_STEPS.to_vec());
    assert_eq!(run.len() + skipped.len(), Suite::Unit.steps().len());
    assert!(run.iter().all(|step| !DIFFTEST_UNIT_STEPS.contains(step)));
    // The three steps are the last three of the Unit suite, the batch last; the format check just before them is not
    // skipped (it is cheap, and CI's `unit` and `slang` jobs would otherwise not run it at all).
    let unit = Suite::Unit.steps();
    assert_eq!(unit[unit.len() - 3..], DIFFTEST_UNIT_STEPS);
    assert_eq!(
        DIFFTEST_UNIT_STEPS.map(|step| step.label()),
        [
            "cargo clippy --locked --manifest-path tests/difftest/Cargo.toml --all-targets -- -D warnings",
            "cargo test --locked --manifest-path tests/difftest/Cargo.toml",
            "cargo xtask check-difftest --cpu-only --seeds 1-4",
        ]
    );
    let fmt = "cargo fmt --manifest-path tests/difftest/Cargo.toml -- --check";
    assert_eq!(unit[unit.len() - 4].label(), fmt);
    assert!(run.iter().any(|step| step.label() == fmt));
    // Without the flag nothing is left out; the Slangc suite carries the same steps.
    let (all, none) = plan_suite(Suite::Unit, false).unwrap();
    assert_eq!((all, none.len()), (Suite::Unit.steps(), 0));
    let (slangc, skipped) = plan_suite(Suite::Slangc, true).unwrap();
    assert_eq!(skipped, DIFFTEST_UNIT_STEPS.to_vec());
    assert!(slangc.contains(&Step::RequireTool("slangc")));
    assert!(slangc.iter().any(|step| step.label() == fmt));
}

/// The fixture workspace `tests/precompiled` is checked by the CI job `precompiled` with two commands; the Unit suite
/// runs the same two, and `--skip-difftest` (which is about the harness crate) keeps them. The commands are read from
/// the workflow, so the two cannot drift apart.
#[test]
fn the_precompiled_fixture_fmt_and_clippy_are_unit_steps_with_the_commands_of_ci() {
    let ci = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows/ci.yml"),
    )
    .unwrap()
    .replace("\r\n", "\n");
    let commands = [
        "cargo fmt --manifest-path tests/precompiled/Cargo.toml --all -- --check",
        "cargo clippy --locked --manifest-path tests/precompiled/Cargo.toml --workspace --all-targets -- -D warnings",
    ];
    let job = job_text(&ci, "precompiled");
    let (run, skipped) = plan_suite(Suite::Unit, true).unwrap();
    let labels: Vec<String> = run.iter().map(|step| step.label()).collect();
    for command in commands {
        assert!(job.contains(&format!("          {command}\n")), "{command}");
        assert!(labels.iter().any(|label| label == command), "{command}");
        assert!(skipped.iter().all(|step| step.label() != command));
    }
    let slangc = Suite::Slangc.steps();
    for command in commands {
        assert!(
            slangc.iter().any(|step| step.label() == command),
            "{command}"
        );
    }
}

/// The bench workspace `bench/` (a workspace of its own) is formatted, linted (all members, all targets, warnings
/// denied) and tested by the Unit suite, and so by the `unit` CI job, which runs that suite: the command lines are
/// these, `--skip-difftest` keeps them, and Slangc has them too (m14.benchfmt).
#[test]
fn the_bench_workspace_fmt_clippy_tests_and_pre_check_are_unit_steps() {
    let commands = [
        "cargo fmt --manifest-path bench/Cargo.toml --all -- --check",
        "cargo clippy --locked --manifest-path bench/Cargo.toml --workspace --all-targets -- -D warnings",
        "cargo run --locked --quiet -p cargo-gust -- check --manifest-path bench/gust-bench-pre/Cargo.toml --allow-slang-mismatch",
    ];
    let (run, skipped) = plan_suite(Suite::Unit, true).unwrap();
    let labels: Vec<String> = run.iter().map(|step| step.label()).collect();
    let slangc = Suite::Slangc.steps();
    for command in commands {
        assert!(labels.iter().any(|label| label == command), "{command}");
        assert!(skipped.iter().all(|step| step.label() != command));
        assert!(
            slangc.iter().any(|step| step.label() == command),
            "{command}"
        );
    }
    // The CPU test step skips the tests that open an adapter, by exact name, runs with no Vulkan driver and in the
    // environment of the `unit` job (so a stale list fails in the gate as in CI), and counts.
    let tests = Suite::Unit
        .steps()
        .into_iter()
        .find(|step| {
            matches!(step, Step::ToolFreeCargo { args, hide_vulkan: true } if args.contains(&"bench/Cargo.toml"))
        })
        .expect("a tool-free bench test step");
    let label = tests.label();
    assert!(
        label.starts_with("cargo test --locked --manifest-path bench/Cargo.toml -p gust-bench -- --exact --skip gust_gpu::tests::"),
        "{label}"
    );
    assert!(
        label.ends_with("[tool-free environment: no Slang, no helper, no Vulkan driver]"),
        "{label}"
    );
    assert_eq!(label.matches(" --skip ").count(), 14, "{label}");
}

/// The two doc-test steps and the bench CPU test step of the Unit suite run in the environment of the `unit` job (no
/// Slang, no helper), the adapter-free doc step and the bench step also with no Vulkan driver, and say so in their
/// label; the unfiltered doc step of the Slangc tier (which needs the tools) and every other step do not.
#[test]
fn only_the_two_doc_steps_and_the_bench_tests_run_in_the_tool_free_environment() {
    let tool_free: Vec<Step> = Suite::Slangc
        .steps()
        .into_iter()
        .filter(|step| matches!(step, Step::ToolFreeCargo { .. }))
        .collect();
    assert_eq!(tool_free.len(), 3);
    assert_eq!(tool_free[..2], [TOOL_FREE_CORE_DOC, TOOL_FREE_WGPU_DOC]);
    assert!(matches!(
        tool_free[2],
        Step::ToolFreeCargo { args, hide_vulkan: true } if args.contains(&"bench/Cargo.toml")
    ));
    assert!(
        TOOL_FREE_CORE_DOC
            .label()
            .ends_with("[tool-free environment: no Slang, no helper]")
    );
    assert!(
        TOOL_FREE_WGPU_DOC
            .label()
            .ends_with("[tool-free environment: no Slang, no helper, no Vulkan driver]")
    );
    assert_eq!(
        Suite::Unit
            .steps()
            .iter()
            .filter(|step| matches!(step, Step::ToolFreeCargo { .. }))
            .count(),
        3
    );
}

#[test]
fn skip_difftest_leaves_out_the_batch_of_the_gpu_suite_and_nothing_else() {
    let gpu = Suite::Gpu.steps();
    let batch = Step::CheckDifftest(&[]);
    assert_eq!(
        gpu.last(),
        Some(&batch),
        "the default Gpu suite ends with the batch"
    );
    let (run, skipped) = plan_suite(Suite::Gpu, true).unwrap();
    assert_eq!(skipped, vec![batch]);
    assert_eq!(run, gpu[..gpu.len() - 1].to_vec());
    assert_eq!(batch.label(), "cargo xtask check-difftest");
    // Without the flag nothing is left out, and the flag after the suite name parses for Gpu as it does for Unit.
    assert_eq!(plan_suite(Suite::Gpu, false).unwrap(), (gpu, Vec::new()));
    assert_eq!(
        parse_test_args(&[
            "--suite".to_owned(),
            "Gpu".to_owned(),
            "--skip-difftest".to_owned()
        ])
        .unwrap(),
        TestRequest::Suite {
            suite: Suite::Gpu,
            skip_difftest: true,
            gpu_jobs: None
        }
    );
}

#[test]
fn skip_difftest_is_an_error_where_there_is_nothing_to_skip_and_a_flag_after_the_suite() {
    for suite in [Suite::Cpu, Suite::Spirv, Suite::Reflect, Suite::Full] {
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
    for step in [DIFFTEST_FMT].into_iter().chain(DIFFTEST_UNIT_STEPS) {
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

/// The `semver` job is retired until the first release (D226): it had no baseline once `release/0.1` was abandoned.
/// ci.yml has no such job and keeps a comment that cites D226 and RELEASE.md. Restoring the job is a deliberate edit of
/// this test, as the first-release checklist of RELEASE.md says.
#[test]
fn the_semver_job_is_retired_and_its_place_cites_d226() {
    let ci = workflow_text("ci.yml");
    assert!(
        !ci.lines().any(|line| line.trim_end() == "  semver:"),
        "ci.yml has a semver job again (see RELEASE.md, \"The semver baseline\")"
    );
    assert!(
        ci.contains("The `semver` job") && ci.contains("(D226, 2026-11-22)"),
        "the comment that marks the retired job is missing"
    );
    assert!(
        ci.contains("The semver baseline"),
        "the comment does not name the RELEASE.md section"
    );
    assert!(
        ci.contains("RELEASE.md"),
        "the comment does not point at RELEASE.md"
    );
    assert!(
        !ci.contains("cargo semver-checks"),
        "ci.yml runs cargo semver-checks again"
    );
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

/// The text of a workflow under `.github/workflows`, with LF line ends.
fn workflow_text(file: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../.github/workflows/{file}")),
    )
    .unwrap()
    .replace("\r\n", "\n")
}

/// The nightly ledger step shares the guard of the Full mode (m5.f1): it runs `cargo xtask check-ledger`, which makes
/// exactly the `RELEASE_LEDGER_TEST` call and judges it with `check_counted_call`, and not a cargo call of its own
/// that an `#[ignore]`d ledger test would leave green.
#[test]
fn gpu_software_runs_the_ledger_through_the_guarded_xtask_command() {
    let workflow = workflow_text("gpu-software.yml");
    let job = job_text(&workflow, "gpu-software");
    let step = step_text(
        &job,
        "Mismatch ledger in the release profile (nightly list)",
    );
    // The nightly list only: the ledger is not a step of the pull request list.
    assert!(
        step.contains("        if: ${{ !cancelled() && env.GPU_SOFTWARE_NIGHTLY == 'true' }}\n"),
        "{step}"
    );
    let call = "          cargo --locked xtask check-ledger 2>&1 | tee ";
    assert!(
        step.contains(&format!("{call}\"$RUNNER_TEMP/ledger-release.log\"\n")),
        "{step}"
    );
    let commands = step
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!commands.contains("--test mismatch"), "{step}");
    assert!(!commands.contains("cargo --locked test"), "{step}");
    // Nothing else in the workflow runs the ledger by a cargo call of its own.
    assert_eq!(workflow.matches("--test mismatch").count(), 0);
}

/// The Mesa install survives a short outage of the snapshot service (m5.f1) without changing what is installed: the
/// retry options sit in `apt_get`, the whole install is tried once more after 60 s, and the pinned snapshot, the
/// package list and the identity checks of the next step are the ones of D52.
#[test]
fn gpu_software_retries_the_mesa_install_and_keeps_the_pin() {
    let workflow = workflow_text("gpu-software.yml");
    let job = job_text(&workflow, "gpu-software");
    let step = step_text(&job, "Install Mesa and the tools from the Ubuntu snapshot");
    for needed in [
        "apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 \\\n",
        "-o Acquire::https::Timeout=30 \"$@\"",
        "install_mesa() {\n",
        "if ! install_mesa; then\n",
        "            sleep 60\n            install_mesa\n          fi\n",
        "URIs: https://snapshot.ubuntu.com/ubuntu/$SNAPSHOT_ID/\n",
        "packages=(\"mesa-vulkan-drivers=$MESA_VERSION\" vulkan-tools build-essential pkg-config spirv-tools)\n",
    ] {
        assert!(step.contains(needed), "{needed}: {step}");
    }
    // The retry is bounded: a second failure fails the step (`set -e` is on and the second call is not a condition).
    assert!(step.contains("          set -euo pipefail\n"), "{step}");
    assert_eq!(step.matches("install_mesa\n").count(), 1, "{step}");
    let check = step_text(
        &job,
        "Check that Mesa did not drift, and that lavapipe is the device",
    );
    assert!(
        check.contains("\"$installed\" != \"$MESA_VERSION\""),
        "{check}"
    );
}

/// Every call of the xtask in a workflow is locked (m5.f1): `cargo --locked xtask ...`, as the Unit suite's lines are.
#[test]
fn every_xtask_call_in_the_workflows_is_locked() {
    let mut seen = 0;
    for file in ["ci.yml", "gpu-software.yml"] {
        for line in workflow_text(file).lines() {
            let command = line.trim_start();
            if command.starts_with('#') {
                continue;
            }
            assert!(!command.contains("cargo xtask"), "{file}: {line}");
            if command.contains("--locked xtask") {
                seen += 1;
            }
        }
    }
    // ci.yml: Unit, dual, cpu-target, difftest, Slangc, precompiled, web; gpu-software: gpu, difftest, examples, ledger.
    assert!(seen >= 10, "{seen}");
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
            Step::NameAdapter,
            Step::GpuSuite,
            Step::AlgoGpuSuite,
            Step::CheckDifftest(&[])
        ]
    );
    // The step makes the call of `verify --mode gpu` and, like it, judges the output against GPU_SUITE_PIN (m5.f1).
    assert_eq!(
        Step::GpuSuite.label(),
        "cargo test --locked -p gust-compute-wgpu"
    );
    assert_eq!(
        cargo_calls(Step::GpuSuite),
        vec![
            mode_cargo_calls(VerifyMode::Gpu)
                .first()
                .expect("the Gpu mode makes the wgpu call first")
                .clone()
        ]
    );
    // The algorithms step is the Gpu mode's second call (m7.r1), judged by name; it is not a Unit step (R4.10).
    assert_eq!(
        Step::AlgoGpuSuite.label(),
        "cargo test --locked -p gust-algo"
    );
    assert_eq!(
        mode_cargo_calls(VerifyMode::Gpu),
        vec![
            owned_call(&GPU_SUITE_CALL),
            cargo_calls(Step::AlgoGpuSuite).remove(0)
        ]
    );
    assert!(!Suite::Unit.steps().contains(&Step::AlgoGpuSuite));
    assert!(judges_algo_gpu(
        VerifyMode::Gpu,
        &owned_call(&ALGO_GPU_CALL)
    ));
    assert!(!judges_algo_gpu(
        VerifyMode::Fast,
        &owned_call(&ALGO_GPU_CALL)
    ));
    assert_eq!(
        Suite::Full.steps(),
        vec![Step::Verify(VerifyMode::Full, true)]
    );
}

/// Output of the Gpu suite call with `passed` tests over `lines` result lines, as libtest prints it.
fn gpu_suite_output(passed: u64, lines: u64) -> String {
    let mut text = format!(
        "test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n"
    );
    for _ in 1..lines {
        text.push_str(
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n",
        );
    }
    text
}

/// A test runner that records the calls it gets and answers each with `output`.
fn recording_runner(
    calls: &RefCell<Vec<Vec<String>>>,
    output: String,
) -> impl Fn(&Path, &[String], Option<&str>) -> Result<(TestTotals, String), String> + '_ {
    move |_, args, profile| {
        // The plain steps run under the workspace profile: they never ask for another one.
        assert_eq!(profile, None, "{args:?}");
        calls.borrow_mut().push(args.to_vec());
        Ok((sum_test_results(&output), output.clone()))
    }
}

/// The Gpu step runs the call through the captured-output runner and judges it against `GPU_SUITE_PIN` (m5.f1,
/// review): a `run_direct` in its arm would never ask the runner, and the first assertion below would fail.
#[test]
fn the_gpu_suite_step_runs_the_captured_runner_and_fails_on_a_drifted_count() {
    let calls = RefCell::new(Vec::new());
    let pinned = gpu_suite_output(GPU_SUITE_PIN.passed, GPU_SUITE_PIN.result_lines);
    let root = Path::new("ROOT");
    assert_eq!(
        Step::GpuSuite.run_in(root, &recording_runner(&calls, pinned)),
        Ok(())
    );
    let expected = GPU_SUITE_CALL.map(str::to_owned).to_vec();
    assert_eq!(*calls.borrow(), vec![expected]);
    // One test more, one fewer, and one result line fewer all fail, naming the pin.
    for (passed, lines) in [
        (GPU_SUITE_PIN.passed + 1, GPU_SUITE_PIN.result_lines),
        (GPU_SUITE_PIN.passed - 1, GPU_SUITE_PIN.result_lines),
        (GPU_SUITE_PIN.passed, GPU_SUITE_PIN.result_lines - 1),
    ] {
        let error = Step::GpuSuite
            .run_in(
                root,
                &recording_runner(&calls, gpu_suite_output(passed, lines)),
            )
            .unwrap_err();
        assert!(error.contains("GPU_SUITE_PIN"), "{passed} {lines}: {error}");
    }
    // The runner's own failure (cargo failed, nothing ran) passes through.
    let failing = |_: &Path, _: &[String], _: Option<&str>| Err("cargo failed".to_owned());
    assert_eq!(
        Step::GpuSuite.run_in(root, &failing),
        Err("cargo failed".to_owned())
    );
}

/// The cargo calls a step makes, as argument lists. Steps that call cargo through another
/// xtask command are listed by that command's data, so that no call hides behind a step.
/// The Full suite is not expanded: it runs `cargo fmt` (which has no `--locked`) and the
/// examples' `--include-ignored` tests, which are not locked, so it is not checked here.
fn cargo_calls(step: Step) -> Vec<Vec<String>> {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    match step {
        Step::Cargo(args) | Step::CountedCargo(args) | Step::ToolFreeCargo { args, .. } => {
            vec![owned(args)]
        }
        Step::GpuSuite => vec![owned(&GPU_SUITE_CALL)],
        Step::AlgoGpuSuite => vec![owned(&ALGO_GPU_CALL)],
        Step::Named(tests) => vec![owned(tests.args)],
        // `adapter_info` is a `cargo run` of an example; its call is `ADAPTER_INFO_ARGS`, pinned in verify.
        Step::NameAdapter => Vec::new(),
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
    assert_eq!(count(Suite::Reflect), 4);
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
            "gust-compute",
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
            ["test", "--locked", "-p", "component-pool"],
            ["test", "--locked", "-p", "map-chain"],
            ["test", "--locked", "-p", "texture-threshold"],
            ["test", "--locked", "-p", "shape-bench"],
            ["test", "--locked", "-p", "hash-rng"],
            ["test", "--locked", "-p", "polar-transform"],
            ["test", "--locked", "-p", "buffer-handoff"],
            ["test", "--locked", "-p", "mc-histogram"],
            ["test", "--locked", "-p", "field-relax"],
            ["test", "--locked", "-p", "cell-bins"],
            ["test", "--locked", "-p", "walker-steer"],
            ["test", "--locked", "-p", "walker-transport"],
            ["test", "--locked", "-p", "walker-life"],
            ["test", "--locked", "-p", "event-drain"],
            ["test", "--locked", "-p", "task-basics"],
            ["test", "--locked", "-p", "batch-basics"]
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
    let step = Step::CountedCargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--lib",
        "cpu::tests::",
    ]);
    let steps = Suite::Unit.steps();
    assert_eq!(steps.iter().filter(|s| **s == step).count(), 1);
    assert!(steps.iter().all(|s| !matches!(s, Step::RequireTool(_))));
    assert_eq!(
        step.label(),
        "cargo test --locked -p gust-compute --lib cpu::tests::"
    );
    assert!(Suite::Unit.covers().contains("gust::cpu"));
}

#[test]
fn the_unit_suite_builds_gust_and_gust_wgpu_without_default_features() {
    // The only observers of the `slang-runtime` off state: the workspace build unifies the feature on.
    let clippy = "cargo clippy --locked -p gust-compute -p gust-compute-wgpu --no-default-features --lib -- -D warnings";
    let check =
        "cargo check --locked -p gust-compute -p gust-compute-wgpu --no-default-features --lib";
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
        "gust-compute",
        "--lib",
        "artifact::tests::",
    ]);
    let integration = Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "artifact",
    ]);
    let steps = Suite::Unit.steps();
    assert_eq!(steps.iter().filter(|s| **s == unit).count(), 1);
    assert_eq!(steps.iter().filter(|s| **s == integration).count(), 1);
    assert_eq!(
        unit.label(),
        "cargo test --locked -p gust-compute --lib artifact::tests::"
    );
    assert_eq!(
        integration.label(),
        "cargo test --locked -p gust-compute --test artifact"
    );
    assert!(Suite::Unit.covers().contains("gust::artifact"));
}

#[test]
fn the_unit_suite_runs_the_coretests_twin_test() {
    // Pure (no slangc, no adapter): the transcribed Rust core test cases on the CPU twin against their literals.
    let step = Step::Cargo(&[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "coretests_twin",
    ]);
    for suite in [Suite::Unit, Suite::Slangc] {
        assert_eq!(
            suite.steps().iter().filter(|s| **s == step).count(),
            1,
            "{suite:?}"
        );
    }
    assert_eq!(
        step.label(),
        "cargo test --locked -p gust-compute --test coretests_twin"
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
fn check_feature_reflection_runs_the_naga_layout_gate() {
    // The naga layout gate (mx.n) is a gust-wgpu test that needs slangc and the helper but no adapter;
    // `check-feature reflection` (the Reflect suite) and the Slangc suite (the CI job `slang`) both name it, and its
    // file must exist. Other lanes append to REFLECTION_TESTS, so the position is not pinned.
    let gate = vec![
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu",
        "--test",
        "naga_layout",
    ];
    let calls: Vec<Vec<&str>> = REFLECTION_TESTS.iter().map(|call| call.to_vec()).collect();
    assert!(calls.contains(&gate), "{calls:?}");
    assert!(Suite::Slangc.covers().contains("naga layout gate"));
    assert!(Suite::Reflect.covers().contains("naga layout gate"));
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/gust-wgpu/tests/naga_layout.rs");
    assert!(
        file.is_file(),
        "crates/gust-wgpu/tests/naga_layout.rs does not exist"
    );
    let explained =
        crate::checks::explanation(Some("reflection")).expect("reflection is explained");
    assert!(explained.contains("naga"), "{explained}");
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
        (Mode::CpuOnly, 68)
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
    // Slice m8.e (Opus S2): the second batch, its byte compare and the subgroups requirement are named.
    for suite in [Suite::Gpu, Suite::Full] {
        assert!(suite.covers().contains("dev+subgroups"), "{suite:?}");
        assert!(suite.covers().contains("byte"), "{suite:?}");
        assert!(
            suite.not_covered().contains("subgroup operations"),
            "{suite:?}"
        );
    }
    assert!(Suite::Full.covers().contains("subgroup-reduce"));
    // m8.x1: the gust-algo run under the capability is the profile form now, so the suite no longer lists it as missing.
    assert!(!Suite::Gpu.not_covered().contains("evidence test"));
    assert!(Suite::Gpu.not_covered().contains("--profile dev+subgroups"));
    assert!(Suite::Gpu.covers().contains("adapter_info"));
    let options = Options::parse_with(&[], 1).unwrap();
    assert_eq!(
        (options.mode, options.expected_programs()),
        (Mode::Gpu, 136)
    );
    assert!(steps(options.mode).contains(&DifftestStep::ShortCircuitGpu));

    // No suite asks for the pixel shape (slice mx.p1c): the Unit, Slangc and Gpu suites stay grid-only, and `check-full`
    // adds `--shapes grid,pixel` itself, in `check_difftest_report` (a test in difftest.rs pins that).
    for suite in [Suite::Unit, Suite::Slangc, Suite::Gpu] {
        for step in suite.steps() {
            if let Step::CheckDifftest(args) = step {
                assert!(!args.contains(&"--shapes"), "{suite:?}: {args:?}");
                assert!(!Options::parse_with(&owned(args), 1).unwrap().pixel);
            }
        }
    }

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

fn owned_call(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

/// Libtest output of the gust-algo call: the twins, then the `gpu::` tests named in `gpu`, each `ok`.
fn algo_output(gpu: &[&str]) -> String {
    let mut text = String::from("test tests::twin::twin_empty_inputs ... ok\n");
    for name in gpu {
        text.push_str(&format!("test {name} ... ok\n"));
    }
    text.push_str("test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n");
    text
}

/// The algorithms step is judged by name (m7.r1, review): a deleted `gpu::` test leaves the twins to satisfy a plain
/// "a test ran", so the guard names every adapter test and counts them.
#[test]
fn the_algo_gpu_step_requires_every_adapter_test_by_name_and_no_other() {
    let root = Path::new("ROOT");
    let calls = RefCell::new(Vec::new());
    let all = algo_output(&ALGO_GPU_TESTS);
    assert_eq!(
        Step::AlgoGpuSuite.run_in(root, &recording_runner(&calls, all)),
        Ok(())
    );
    assert_eq!(*calls.borrow(), vec![owned_call(&ALGO_GPU_CALL)]);
    // One adapter test deleted: the twins alone would pass a plain count.
    let error = Step::AlgoGpuSuite
        .run_in(
            root,
            &recording_runner(&calls, algo_output(&ALGO_GPU_TESTS[1..])),
        )
        .unwrap_err();
    assert!(
        error.contains(ALGO_GPU_TESTS[0]) && error.contains("ALGO_GPU_TESTS"),
        "{error}"
    );
    // No adapter test at all.
    assert!(
        Step::AlgoGpuSuite
            .run_in(root, &recording_runner(&calls, algo_output(&[])))
            .is_err()
    );
    // One more than pinned: the list must change in the same commit.
    let mut more = ALGO_GPU_TESTS.to_vec();
    more.push("tests::gpu::gpu_a_new_test");
    let mut more_in_record = ALGO_GPU_TESTS.to_vec();
    more_in_record.push("tests::record::gpu_a_new_test");
    let error = Step::AlgoGpuSuite
        .run_in(root, &recording_runner(&calls, algo_output(&more)))
        .unwrap_err();
    assert!(error.contains("the pinned number"), "{error}");
    // The recording form's adapter tests are counted too (an unpinned `tests::record::gpu_` test fails the same way).
    let error = Step::AlgoGpuSuite
        .run_in(
            root,
            &recording_runner(&calls, algo_output(&more_in_record)),
        )
        .unwrap_err();
    assert!(error.contains("the pinned number"), "{error}");
    // So are the adapter tests of the ops of a batch (an unpinned `tests::batch::gpu_` test fails the same way).
    let mut more_in_batch = ALGO_GPU_TESTS.to_vec();
    more_in_batch.push("tests::batch::gpu_a_new_test");
    let error = Step::AlgoGpuSuite
        .run_in(root, &recording_runner(&calls, algo_output(&more_in_batch)))
        .unwrap_err();
    assert!(error.contains("the pinned number"), "{error}");
    // So are the integration tests of gust-algo (an unpinned `gpu_` test of a `tests/` file fails the same way), and
    // a pinned one that did not run is named.
    let mut more_integration = ALGO_GPU_TESTS.to_vec();
    more_integration.push("gpu_a_new_integration_test");
    let error = Step::AlgoGpuSuite
        .run_in(
            root,
            &recording_runner(&calls, algo_output(&more_integration)),
        )
        .unwrap_err();
    assert!(error.contains("the pinned number"), "{error}");
    let last = ALGO_GPU_TESTS[ALGO_GPU_TESTS.len() - 1];
    let error = Step::AlgoGpuSuite
        .run_in(
            root,
            &recording_runner(
                &calls,
                algo_output(&ALGO_GPU_TESTS[..ALGO_GPU_TESTS.len() - 1]),
            ),
        )
        .unwrap_err();
    assert!(error.contains(last), "{error}");
    // Nothing ran at all.
    assert!(
        Step::AlgoGpuSuite
            .run_in(root, &recording_runner(&calls, String::new()))
            .is_err()
    );
}

fn strings(args: &[&str]) -> Vec<String> {
    owned_call(args)
}

/// The profile form of `cargo xtask test` (m8.w2): `--suite Gpu --profile <p> --test <t>`, in that order and only for
/// the Gpu suite; the other forms are as they were.
#[test]
fn the_profile_argument_is_parsed_checked_and_only_for_the_gpu_suite() {
    assert_eq!(
        parse_test_args(&strings(&[
            "--suite",
            "gpu",
            "--profile",
            "dev+subgroups",
            "--test",
            "subgroup"
        ])),
        Ok(TestRequest::Target {
            profile: "dev+subgroups".to_owned(),
            target: "subgroup".to_owned()
        })
    );
    assert_eq!(
        parse_test_args(&strings(&["--suite", "Unit", "--skip-difftest"])),
        Ok(TestRequest::Suite {
            suite: Suite::Unit,
            skip_difftest: true,
            gpu_jobs: None
        })
    );
    assert_eq!(
        parse_test_args(&strings(&["--suite", "Gpu"])),
        Ok(TestRequest::Suite {
            suite: Suite::Gpu,
            skip_difftest: false,
            gpu_jobs: None
        })
    );
    // Another suite: refused, naming the suite.
    let error = parse_test_args(&strings(&[
        "--suite",
        "Unit",
        "--profile",
        "dev",
        "--test",
        "subgroup",
    ]))
    .unwrap_err();
    assert!(error.contains("--suite Gpu only, not Unit"), "{error}");
    // A profile the gust-wgpu tests cannot build under (no run-time compiler), or a malformed one.
    for profile in [
        "native",
        "native+subgroups",
        "web",
        "cpu",
        "dev+",
        "devx",
        "dev+a b",
        "",
    ] {
        let error = parse_test_args(&strings(&[
            "--suite",
            "Gpu",
            "--profile",
            profile,
            "--test",
            "subgroup",
        ]))
        .unwrap_err();
        assert!(error.contains("--profile"), "{profile}: {error}");
    }
    // A target without a pin, naming the targets that have one.
    let error = parse_test_args(&strings(&[
        "--suite",
        "Gpu",
        "--profile",
        "dev",
        "--test",
        "typed",
    ]))
    .unwrap_err();
    assert!(
        error.contains("GPU_TARGET_PINS") && error.contains("capabilities, subgroup"),
        "{error}"
    );
    // The wrong order, `--test` without a profile and extra words are the usage error, which keeps the old flag in it.
    for args in [
        strings(&["--suite", "Gpu", "--test", "subgroup", "--profile", "dev"]),
        strings(&["--suite", "Gpu", "--test", "subgroup"]),
        strings(&["--profile", "dev", "--test", "subgroup"]),
        strings(&[
            "--suite",
            "Gpu",
            "--profile",
            "dev",
            "--test",
            "subgroup",
            "--skip-difftest",
        ]),
    ] {
        let error = parse_test_args(&args).unwrap_err();
        assert!(
            error.contains("[--skip-difftest]") && error.contains("--profile"),
            "{args:?}: {error}"
        );
    }
}

/// The calls and profiles a [`profile_runner`] was given.
type Seen = RefCell<Vec<(Vec<String>, Option<String>)>>;

/// A runner that records the call and the profile it gets, answering with `output`.
fn profile_runner(
    seen: &Seen,
    output: String,
) -> impl Fn(&Path, &[String], Option<&str>) -> Result<(TestTotals, String), String> + '_ {
    move |_, args, profile| {
        seen.borrow_mut()
            .push((args.to_vec(), profile.map(str::to_owned)));
        Ok((sum_test_results(&output), output.clone()))
    }
}

/// One target under a profile: the call is the target's alone, the runner is given the profile (that is what replaces
/// the `dev` of the workspace steps), and the output must be one `ok` result line with the pinned number of tests.
#[test]
fn a_profile_run_makes_one_targeted_call_and_is_judged_against_its_pin() {
    let root = Path::new("ROOT");
    let seen = RefCell::new(Vec::new());
    for (target, passed) in GPU_TARGET_PINS {
        let output = with_build_capabilities(gpu_suite_output(passed, 1), "dev+subgroups");
        assert_eq!(
            run_gpu_target(
                root,
                "dev+subgroups",
                target,
                &profile_runner(&seen, output)
            ),
            Ok(()),
            "{target}"
        );
    }
    assert_eq!(
        *seen.borrow(),
        GPU_TARGET_PINS
            .iter()
            .map(|(target, _)| (
                strings(&[
                    "test",
                    "--locked",
                    "-p",
                    "gust-compute-wgpu",
                    "--test",
                    target,
                    "--",
                    "--show-output"
                ]),
                Some("dev+subgroups".to_owned())
            ))
            .collect::<Vec<_>>()
    );
    // One test more, one fewer, a second result line (a doc-test run, say) and a failure all fail, naming the pin.
    let (target, passed) = GPU_TARGET_PINS[1];
    for output in [
        gpu_suite_output(passed + 1, 1),
        gpu_suite_output(passed - 1, 1),
        gpu_suite_output(passed, 2),
        gpu_suite_output(passed, 1).replace("0 failed", "1 failed"),
    ] {
        let error =
            run_gpu_target(root, "dev", target, &profile_runner(&seen, output)).unwrap_err();
        assert!(error.contains("GPU_TARGET_PINS"), "{error}");
        assert!(!error.contains("GPU_SUITE_PIN"), "{error}");
    }
    // A profile or a target that is refused never reaches cargo.
    let before = seen.borrow().len();
    assert!(
        run_gpu_target(
            root,
            "native",
            target,
            &profile_runner(&seen, String::new())
        )
        .is_err()
    );
    assert!(run_gpu_target(root, "dev", "typed", &profile_runner(&seen, String::new())).is_err());
    assert_eq!(seen.borrow().len(), before);
    // The runner's own failure passes through.
    let failing = |_: &Path, _: &[String], _: Option<&str>| Err("cargo failed".to_owned());
    assert_eq!(
        run_gpu_target(root, "dev", target, &failing),
        Err("cargo failed".to_owned())
    );
}

/// `output` followed by what `tests/capabilities.rs` prints for a build of `profile`, with the harness's `---- stdout`
/// framing around it (what `--show-output` adds).
fn with_build_capabilities(output: String, profile: &str) -> String {
    let set = profile
        .strip_prefix("dev+")
        .map(|names| {
            names
                .split('+')
                .map(|name| name[..1].to_ascii_uppercase() + &name[1..])
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!(
        "{output}\n---- the_builds_capabilities_follow_the_profile_variable stdout ----\nGUST_PROFILE=\"{profile}\"; capabilities of this build: [{set}]\n"
    )
}

/// The profile must have reached the build, not only the runner (m8.w2 fix round 1, S1): the `capabilities` target prints
/// the build's own set under `--show-output`, and the run fails, naming the profile, when the set is not the profile's or
/// the line is absent. The same output with the right line passes, so the check is the line and not the counts.
#[test]
fn a_profile_run_requires_the_capabilities_line_of_the_build() {
    let root = Path::new("ROOT");
    let seen = RefCell::new(Vec::new());
    let passed = GPU_TARGET_PINS
        .iter()
        .find(|(target, _)| *target == "capabilities")
        .unwrap()
        .1;
    let counts = gpu_suite_output(passed, 1);
    for profile in ["dev", "dev+subgroups"] {
        let output = with_build_capabilities(counts.clone(), profile);
        assert_eq!(
            run_gpu_target(
                root,
                profile,
                "capabilities",
                &profile_runner(&seen, output)
            ),
            Ok(()),
            "{profile}"
        );
    }
    // A build that did not get the profile (it ran under `dev`), a line that is missing, and one with another set.
    for (profile, output) in [
        (
            "dev+subgroups",
            with_build_capabilities(counts.clone(), "dev"),
        ),
        ("dev+subgroups", counts.clone()),
        (
            "dev",
            with_build_capabilities(counts.clone(), "dev+subgroups"),
        ),
        (
            "dev+subgroups",
            counts.clone() + "capabilities of this build: [Subgroups, Other]\n",
        ),
    ] {
        let error = run_gpu_target(
            root,
            profile,
            "capabilities",
            &profile_runner(&seen, output.clone()),
        )
        .unwrap_err();
        assert!(
            error.contains("the build did not receive GUST_PROFILE=")
                && error.contains(&format!("GUST_PROFILE={profile}"))
                && error.contains("capabilities of this build: ["),
            "{profile}: {error}"
        );
    }
    // The message names the line it wanted.
    let error = run_gpu_target(
        root,
        "dev+subgroups",
        "capabilities",
        &profile_runner(&seen, counts),
    )
    .unwrap_err();
    assert!(
        error.contains("`capabilities of this build: [Subgroups]`"),
        "{error}"
    );
    // The other target prints no such line and needs none.
    let (other, other_passed) = GPU_TARGET_PINS[1];
    assert_ne!(other, "capabilities");
    assert_eq!(
        run_gpu_target(
            root,
            "dev+subgroups",
            other,
            &profile_runner(&seen, gpu_suite_output(other_passed, 1))
        ),
        Ok(())
    );
}

/// A runner for the profile form without a target: it answers each call as the real test would under `profile`, the
/// pinned count (and the capabilities line, for the `capabilities` target) for a gust-wgpu target, every `gpu::` test of
/// gust-algo (and the capabilities line its own build prints when `algo_prints` is set) for gust-algo.
fn profile_suite_runner<'a>(
    seen: &'a Seen,
    profile: &'a str,
    algo_prints: bool,
) -> impl Fn(&Path, &[String], Option<&str>) -> Result<(TestTotals, String), String> + 'a {
    move |_, args, run_profile| {
        seen.borrow_mut()
            .push((args.to_vec(), run_profile.map(str::to_owned)));
        let output = if args.iter().any(|arg| arg == "gust-algo") {
            let text = algo_output(&ALGO_GPU_TESTS);
            if algo_prints {
                with_build_capabilities(text, profile)
            } else {
                text
            }
        } else {
            let target = args
                .iter()
                .skip_while(|arg| *arg != "--test")
                .nth(1)
                .expect("a gust-wgpu call names its target");
            let passed = GPU_TARGET_PINS
                .iter()
                .find(|(name, _)| *name == target.as_str())
                .expect("a pinned target")
                .1;
            let text = gpu_suite_output(passed, 1);
            if target == "capabilities" {
                with_build_capabilities(text, profile)
            } else {
                text
            }
        };
        Ok((sum_test_results(&output), output))
    }
}

/// Ruling 2 of m8.x1: `--suite Gpu --profile <p>` without `--test` is a request of its own (the pinned gust-wgpu targets,
/// then gust-algo), and only for the Gpu suite and a profile the run-time compiler allows.
#[test]
fn the_profile_argument_without_a_target_is_a_request_for_the_targets_and_gust_algo() {
    assert_eq!(
        parse_test_args(&strings(&["--suite", "Gpu", "--profile", "dev+subgroups"])),
        Ok(TestRequest::Profile {
            profile: "dev+subgroups".to_owned()
        })
    );
    assert_eq!(
        parse_test_args(&strings(&["--suite", "gpu", "--profile", "dev"])),
        Ok(TestRequest::Profile {
            profile: "dev".to_owned()
        })
    );
    let error = parse_test_args(&strings(&["--suite", "Unit", "--profile", "dev"])).unwrap_err();
    assert!(error.contains("--suite Gpu only, not Unit"), "{error}");
    for profile in ["native", "native+subgroups", "dev+", "devx", ""] {
        let error =
            parse_test_args(&strings(&["--suite", "Gpu", "--profile", profile])).unwrap_err();
        assert!(error.contains("--profile"), "{profile}: {error}");
    }
    // The usage text shows both forms, `--test` optional.
    let usage = parse_test_args(&[]).unwrap_err();
    assert!(
        usage.contains("[--test <gust-wgpu test target>]"),
        "{usage}"
    );
}

/// The profile run makes the pinned gust-wgpu calls and then the gust-algo call, each under the profile, announces them in
/// that order, and fails when gust-algo's own build did not receive the profile, when an adapter test is gone, or when a
/// target before it failed (nothing after it runs).
#[test]
fn the_profile_suite_runs_the_pinned_targets_and_then_gust_algo_under_the_profile() {
    let root = Path::new("ROOT");
    let profile = "dev+subgroups";
    let seen = RefCell::new(Vec::new());
    let mut announced = Vec::new();
    let result = run_gpu_profile_suite(
        root,
        profile,
        &profile_suite_runner(&seen, profile, true),
        &mut |call: &str| announced.push(call.to_owned()),
    );
    assert_eq!(result, Ok(()));
    let mut calls: Vec<Vec<String>> = GPU_TARGET_PINS
        .iter()
        .map(|(target, _)| {
            strings(&[
                "test",
                "--locked",
                "-p",
                "gust-compute-wgpu",
                "--test",
                target,
                "--",
                "--show-output",
            ])
        })
        .collect();
    let mut algo = owned_call(&ALGO_GPU_CALL);
    algo.extend(strings(&["--", "--show-output"]));
    assert_eq!(algo, algo_profile_call());
    calls.push(algo);
    let expected: Vec<(Vec<String>, Option<String>)> = calls
        .iter()
        .map(|call| (call.clone(), Some(profile.to_owned())))
        .collect();
    assert_eq!(*seen.borrow(), expected);
    assert_eq!(
        announced,
        calls
            .iter()
            .map(|call| format!("cargo {} (GUST_PROFILE={profile})", call.join(" ")))
            .collect::<Vec<_>>()
    );

    // gust-algo's build printed no capabilities line, or the line of another profile: the profile did not reach it, and
    // the plain counts of its tests would pass for any profile.
    for wrong in [false, true] {
        let seen = RefCell::new(Vec::new());
        let error = if wrong {
            // The gust-wgpu targets report the profile; only gust-algo's build reports `dev`.
            let runner = |root: &Path, args: &[String], run_profile: Option<&str>| {
                let reported = if args.iter().any(|arg| arg == "gust-algo") {
                    "dev"
                } else {
                    profile
                };
                profile_suite_runner(&seen, reported, true)(root, args, run_profile)
            };
            run_gpu_profile_suite(root, profile, &runner, &mut |_: &str| {})
        } else {
            run_gpu_profile_suite(
                root,
                profile,
                &profile_suite_runner(&seen, profile, false),
                &mut |_: &str| {},
            )
        }
        .unwrap_err();
        assert!(
            error.contains("the build did not receive GUST_PROFILE=dev+subgroups")
                && error.contains("crates/gust-algo")
                && error.contains("`capabilities of this build: [Subgroups]`"),
            "{wrong}: {error}"
        );
    }

    // An adapter test deleted from gust-algo's output is named, as in the plain suite.
    let runner = |root: &Path, args: &[String], run_profile: Option<&str>| {
        if args.iter().any(|arg| arg == "gust-algo") {
            let output = with_build_capabilities(algo_output(&ALGO_GPU_TESTS[1..]), profile);
            Ok((sum_test_results(&output), output))
        } else {
            profile_suite_runner(&RefCell::new(Vec::new()), profile, true)(root, args, run_profile)
        }
    };
    let error = run_gpu_profile_suite(root, profile, &runner, &mut |_: &str| {}).unwrap_err();
    assert!(
        error.contains(ALGO_GPU_TESTS[0]) && error.contains("ALGO_GPU_TESTS"),
        "{error}"
    );

    // A failing first target ends the run: gust-algo is not called.
    let seen = RefCell::new(Vec::new());
    let failing = |_: &Path, args: &[String], run_profile: Option<&str>| {
        seen.borrow_mut()
            .push((args.to_vec(), run_profile.map(str::to_owned)));
        Err("cargo failed".to_owned())
    };
    assert_eq!(
        run_gpu_profile_suite(root, profile, &failing, &mut |_: &str| {}),
        Err("cargo failed".to_owned())
    );
    assert_eq!(seen.borrow().len(), 1);

    // A refused profile never reaches cargo.
    let seen = RefCell::new(Vec::new());
    assert!(
        run_gpu_profile_suite(
            root,
            "native",
            &profile_suite_runner(&seen, "native", true),
            &mut |_: &str| {}
        )
        .is_err()
    );
    assert!(seen.borrow().is_empty());
}

/// The gust-algo test that prints the capabilities line is part of the plain `gust-algo` call too, and its print is in
/// the source the judge reads, so a removed `println!` cannot leave the profile run passing on the counts alone.
#[test]
fn the_gust_algo_test_prints_the_capabilities_of_its_build() {
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/gust-algo/src/tests/variant.rs"),
    )
    .unwrap()
    .replace("\r\n", "\n");
    let function = source
        .find("fn the_evidence_pair_selects_the_body_of_the_profile_and_the_shipped_set_does_not_move()")
        .expect("the evidence pair test");
    assert!(
        source[function..]
            .lines()
            .take(6)
            .any(|line| line.trim() == r#"println!("capabilities of this build: {enabled:?}");"#),
        "the print of the capabilities of gust-algo's build is gone"
    );
}

/// The Slangc tier's two named steps (m8.x1): each needs every one of its tests reported `ok` (an `#[ignore]`d test that
/// `--include-ignored` did not run, a renamed or a deleted one fails), and the variant gate asks for the subgroups
/// capability while the helper step asks for none.
#[test]
fn the_named_steps_need_each_named_test_ok_and_pass_their_profile_to_the_runner() {
    let root = Path::new("ROOT");
    let named: Vec<&NamedTests> = Suite::Slangc
        .steps()
        .into_iter()
        .filter_map(|step| match step {
            Step::Named(tests) => Some(tests),
            _ => None,
        })
        .collect();
    assert_eq!(named.len(), 2);
    assert_eq!(
        named.iter().map(|tests| tests.profile).collect::<Vec<_>>(),
        [None, Some("dev+subgroups")]
    );
    assert!(
        Suite::Unit
            .steps()
            .iter()
            .all(|step| !matches!(step, Step::Named(_) | Step::NameAdapter))
    );
    for tests in named {
        let step = Step::Named(tests);
        let output = |skipped: Option<&str>, state: &str| {
            let mut text = String::new();
            for name in tests.names {
                let state = if skipped == Some(*name) { state } else { "ok" };
                text.push_str(&format!("test {name} ... {state}\n"));
            }
            // What `--show-output` adds: the prints of the passing tests, among them the lines the step requires.
            text.push_str("\nsuccesses:\n\n---- x stdout ----\n");
            for line in tests.lines {
                text.push_str(&format!("{line}double_subgroups (0123456789abcdef)\n"));
            }
            text.push_str(&format!(
                "test result: ok. {} passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 1.0s\n",
                tests.names.len()
            ));
            text
        };
        let seen = RefCell::new(Vec::new());
        assert_eq!(
            step.run_in(root, &profile_runner(&seen, output(None, "ok"))),
            Ok(()),
            "{step:?}"
        );
        assert_eq!(
            *seen.borrow(),
            vec![(owned_call(tests.args), tests.profile.map(str::to_owned))]
        );
        assert!(step.label().ends_with(&tests.args.join(" ")));
        assert_eq!(
            step.label().starts_with("GUST_PROFILE=dev+subgroups "),
            tests.profile.is_some()
        );
        for name in tests.names {
            for state in ["ignored, needs the native helper", "FAILED"] {
                let error = step
                    .run_in(root, &profile_runner(&seen, output(Some(*name), state)))
                    .unwrap_err();
                assert!(error.contains(name), "{name} {state}: {error}");
            }
            // The test is not in the output at all (renamed, moved or deleted).
            let without: String = output(None, "ok")
                .lines()
                .filter(|line| !line.contains(name))
                .map(|line| format!("{line}\n"))
                .collect();
            let error = step
                .run_in(root, &profile_runner(&seen, without))
                .unwrap_err();
            assert!(error.contains(name), "{name}: {error}");
        }
        // The required lines (m8.x1 fix round 1, S1): every test is `ok`, but the print that shows the capability reached
        // the build is missing (a cargo that reused the `dev` binary prints `[]`), or `--show-output` is gone.
        for line in tests.lines {
            let dev = output(None, "ok").replace(line, "checked under capabilities []: ");
            let gone: String = output(None, "ok")
                .lines()
                .filter(|text| !text.starts_with(line))
                .map(|text| format!("{text}\n"))
                .collect();
            for text in [dev, gone] {
                let error = step.run_in(root, &profile_runner(&seen, text)).unwrap_err();
                assert!(
                    error.contains(&format!("no line starting `{line}`"))
                        && error.contains("GUST_PROFILE=dev+subgroups")
                        && error.contains("--show-output"),
                    "{error}"
                );
            }
        }
        assert_eq!(
            tests.lines.is_empty(),
            tests.profile.is_none(),
            "a step with a profile names the line that shows it"
        );
        assert_eq!(
            tests.args.contains(&"--show-output"),
            !tests.lines.is_empty(),
            "the lines are read from the prints of passing tests"
        );
        // The runner's own failure and a call that ran nothing pass through.
        let failing = |_: &Path, _: &[String], _: Option<&str>| Err("cargo failed".to_owned());
        assert_eq!(step.run_in(root, &failing), Err("cargo failed".to_owned()));
    }
}

/// Every test file of `gust-compute` (crates/gust/tests) is run by some step of some suite (m8.x1: four of them were in
/// none), so a file added later without a step fails here, not silently in CI.
#[test]
fn every_test_file_of_gust_compute_is_run_by_a_suite_step() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/gust/tests");
    // Top-level `*.rs` files only: a test target in a subdirectory (`tests/<dir>/main.rs`) is not seen by this guard.
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path.file_stem().unwrap().to_string_lossy().into_owned());
        }
    }
    files.sort();
    assert!(files.len() >= 9, "{files:?}");
    let calls: Vec<Vec<String>> = Suite::names()
        .into_iter()
        .flat_map(|name| Suite::parse(name).unwrap().steps())
        .flat_map(cargo_calls)
        .collect();
    for file in files {
        let run = calls.iter().any(|call| {
            call.iter().any(|arg| arg == "gust-compute")
                && call
                    .windows(2)
                    .any(|pair| pair[0] == "--test" && pair[1] == file)
        });
        assert!(run, "crates/gust/tests/{file}.rs is run by no suite step");
    }
}

/// The adapter line of the Gpu suite's log names the device, its type and backend, and the driver when there is one.
#[test]
fn the_adapter_line_names_the_device_its_type_and_its_driver() {
    let line = |driver: &str, info: &str| {
        adapter_line(
            &AdapterRecord::parse(&format!(
                r#"{{"name":"NVIDIA GeForce RTX 5090","vendor":4318,"device":10,"device_type":"DiscreteGpu","backend":"Vulkan","driver":"{driver}","driver_info":"{info}"}}"#
            ))
            .unwrap(),
        )
    };
    assert_eq!(
        line("NVIDIA", "580.97"),
        "adapter: NVIDIA GeForce RTX 5090 (DiscreteGpu, Vulkan, driver NVIDIA 580.97)"
    );
    assert_eq!(
        line("", ""),
        "adapter: NVIDIA GeForce RTX 5090 (DiscreteGpu, Vulkan, driver unnamed)"
    );
    assert_eq!(
        Step::NameAdapter.label(),
        "name the adapter: cargo run --locked --quiet -p gust-compute-wgpu --example adapter_info"
    );
}

/// The whole-suite output stays judged as before: its messages still name `GPU_SUITE_PIN`.
#[test]
fn the_suite_pin_messages_are_unchanged() {
    let call = owned_call(&GPU_SUITE_CALL);
    let error = check_gpu_suite_counts(
        &call,
        &gpu_suite_output(GPU_SUITE_PIN.passed + 1, GPU_SUITE_PIN.result_lines),
        GPU_SUITE_PIN,
    )
    .unwrap_err();
    assert!(
        error.contains("(GPU_SUITE_PIN.passed)")
            && error.contains("change GPU_SUITE_PIN in xtask/src/verify.rs"),
        "{error}"
    );
}

/// Each pinned target is a test file of gust-wgpu, and its pin is the number of `#[test]` functions in it, so a test
/// added to or removed from the file fails here before it fails a run on a GPU. The count is of lines that are exactly
/// `#[test]`: a one-line `#[test] fn ...` or an `#[ignore]`d test is not seen by this static check (the run-time check,
/// which counts the `test result:` line and rejects an ignored test, still catches both).
#[test]
fn the_target_pins_match_the_tests_of_their_files() {
    for (target, passed) in GPU_TARGET_PINS {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/gust-wgpu/tests")
            .join(format!("{target}.rs"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{target}: {}: {error}", path.display()));
        let tests = text.lines().filter(|line| line.trim() == "#[test]").count() as u64;
        assert_eq!(tests, passed, "{target}: change GPU_TARGET_PINS");
    }
}

/// The lavapipe job runs the `+subgroups` build (m8.w2): after the Gpu suite (which keeps the `dev` pins), both pinned
/// targets under `dev+subgroups` through the xtask, in both lists, with the log kept as evidence.
#[test]
fn gpu_software_runs_the_subgroup_targets_under_the_profile() {
    let workflow = workflow_text("gpu-software.yml");
    let job = job_text(&workflow, "gpu-software");
    let name = "Subgroups on lavapipe (dev+subgroups)";
    let step = step_text(&job, name);
    // Both lists: the condition is `!cancelled()` alone, not the nightly flag (S2: a red differential test must not skip
    // the step, and this step must not skip the differential tests, which come before it).
    let conditions: Vec<&str> = step
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("if:"))
        .collect();
    assert_eq!(conditions, ["if: ${{ !cancelled() }}"], "{step}");
    for target in ["subgroup", "capabilities"] {
        assert!(
            step.contains(&format!(
                "cargo --locked xtask test --suite Gpu --profile dev+subgroups --test {target} 2>&1 | tee"
            )),
            "{target}: {step}"
        );
        assert!(
            GPU_TARGET_PINS.iter().any(|(pinned, _)| *pinned == target),
            "{target} has no pin"
        );
    }
    assert!(step.contains("set -o pipefail\n"), "{step}");
    let position = |wanted: &str| {
        job.find(&format!("      - name: {wanted}\n"))
            .unwrap_or_else(|| panic!("no step {wanted}"))
    };
    // After both differential tests (a red step here must not skip their annotations), before the steps that report.
    assert!(
        position("Gpu suite on lavapipe")
            < position("Differential test (2 seeds; all 8 in the nightly list)")
    );
    assert!(
        position("Cross-shape differential test (pixel against grid, 2 seeds)") < position(name)
    );
    assert!(position(name) < position("Name the failing seeds"));
    assert!(
        step_text(&job, "Collect the evidence").contains("gpu-subgroups.log"),
        "the log is kept"
    );
}

/// Whether `path` (`module::item`) names an item that the file of its module (`<module>.rs` or `<module>/mod.rs` under
/// `crates/gust/src`) declares as a fn, struct, enum, trait or type with exactly that name.
fn skip_path_names_an_item(path: &str) -> bool {
    let (module, item) = path.rsplit_once("::").expect("a path-shaped skip entry");
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/gust/src")
        .join(module.replace("::", "/"));
    [base.with_extension("rs"), base.join("mod.rs")]
        .iter()
        .filter_map(|file| std::fs::read_to_string(file).ok())
        .any(|text| {
            ["fn ", "struct ", "enum ", "trait ", "type "]
                .iter()
                .any(|kind| {
                    text.match_indices(&format!("{kind}{item}"))
                        .any(|(at, found)| {
                            !text[at + found.len()..]
                                .starts_with(|next: char| next.is_alphanumeric() || next == '_')
                        })
                })
        })
}

/// Every `--skip` path of the tool-free doc-test step names an item of `crates/gust` (a stale one skips nothing and
/// lets a helper-launching doc test run on the `unit` runner, as the m13.splitr module move did). The check is exact
/// on the item name, and it fails on a step with no `--` or with a different number of entries than `--skip` flags.
#[test]
fn doc_test_skip_paths_name_items_of_the_crate() {
    let Step::ToolFreeCargo { args, .. } = super::CORE_DOC_TESTS_TOOL_FREE else {
        panic!("CORE_DOC_TESTS_TOOL_FREE is a counted cargo step in the tool-free environment");
    };
    assert!(args.contains(&"--"), "the step passes filters after `--`");
    let paths: Vec<&str> = args
        .windows(2)
        .filter(|pair| pair[0] == "--skip")
        .map(|pair| pair[1])
        .collect();
    assert_eq!(
        paths.len(),
        args.iter().filter(|arg| **arg == "--skip").count()
    );
    assert_eq!(paths.len(), 6, "the six skip entries of the tool-free step");
    for path in paths {
        assert!(
            skip_path_names_an_item(path),
            "skip entry `{path}` names no item in crates/gust/src"
        );
    }
    // The pre-split paths, and a prefix of a real name, must be refused by the same check.
    for stale in [
        "reflect::compile_reflected",
        "reflect::CompiledReflection",
        "reflect::CleanCompilation",
        "reflect::reflect",
        "slang::compil",
    ] {
        assert!(
            !skip_path_names_an_item(stale),
            "`{stale}` must not name an item"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The parallel run of the Gpu suite's gust-compute-wgpu call (m14.s8g): the pure parts. No adapter is opened.

/// The `--gpu-jobs` flag: any order with `--skip-difftest`, a count of at least 1, refused for a suite other than Gpu
/// and without a second value, and refused by name when the value is not a count.
#[test]
fn the_gpu_jobs_flag_is_parsed_and_only_for_the_gpu_suite() {
    let request = |args: &[&str]| parse_test_args(&strings(args));
    let gpu = |skip_difftest, gpu_jobs| {
        Ok(TestRequest::Suite {
            suite: Suite::Gpu,
            skip_difftest,
            gpu_jobs,
        })
    };
    assert_eq!(
        request(&["--suite", "Gpu", "--gpu-jobs", "3"]),
        gpu(false, Some(3))
    );
    assert_eq!(
        request(&["--suite", "Gpu", "--gpu-jobs", "1"]),
        gpu(false, Some(1))
    );
    assert_eq!(
        request(&["--suite", "Gpu", "--skip-difftest", "--gpu-jobs", "2"]),
        gpu(true, Some(2))
    );
    assert_eq!(
        request(&["--suite", "Gpu", "--gpu-jobs", "2", "--skip-difftest"]),
        gpu(true, Some(2))
    );
    assert_eq!(
        request(&["--suite", "gpu", "--skip-difftest"]),
        gpu(true, None)
    );
    // Another suite: refused, naming the suite, with or without the other flag.
    for args in [
        &["--suite", "Unit", "--gpu-jobs", "2"][..],
        &["--suite", "Slangc", "--skip-difftest", "--gpu-jobs", "2"][..],
    ] {
        let error = request(args).unwrap_err();
        assert!(error.contains("--suite Gpu only, not"), "{args:?}: {error}");
    }
    // A value that is not a count of at least 1 is named.
    for value in ["0", "-1", "two", "", "1.5"] {
        let error = request(&["--suite", "Gpu", "--gpu-jobs", value]).unwrap_err();
        assert!(
            error.contains("--gpu-jobs") && error.contains("at least 1"),
            "{value}: {error}"
        );
    }
    // A missing value, a repeated flag and an unknown flag are the usage error, which names the flag.
    for args in [
        &["--suite", "Gpu", "--gpu-jobs"][..],
        &["--suite", "Gpu", "--gpu-jobs", "2", "--gpu-jobs", "3"][..],
        &["--suite", "Gpu", "--skip-difftest", "--skip-difftest"][..],
        &["--suite", "Gpu", "--jobs", "2"][..],
        &["--suite", "Gpu", "--profile", "dev", "--gpu-jobs", "2"][..],
    ] {
        let error = request(args).unwrap_err();
        assert!(
            error.contains("usage:") && error.contains("--gpu-jobs"),
            "{args:?}: {error}"
        );
    }
}

/// One line of `cargo ... --message-format=json` for a `compiler-artifact`.
fn artifact(
    manifest: &str,
    kind: &str,
    name: &str,
    source: &str,
    executable: Option<&str>,
    test: bool,
) -> String {
    let executable = executable.map_or("null".to_owned(), |path| format!("\"{path}\""));
    format!(
        "{{\"reason\":\"compiler-artifact\",\"package_id\":\"x\",\"manifest_path\":\"{manifest}\",\"target\":{{\"kind\":[\"{kind}\"],\"name\":\"{name}\",\"src_path\":\"{source}\"}},\"profile\":{{\"opt_level\":\"0\",\"test\":{test}}},\"executable\":{executable},\"fresh\":true}}"
    )
}

const PACKAGE_DIR: &str = "C:/work/crates/wgpu";
const PACKAGE_MANIFEST: &str = "C:/work/crates/wgpu/Cargo.toml";

fn cargo_output() -> String {
    [
        // Cargo's order of completion is not its run order; the library's own build is not a test build.
        artifact(
            PACKAGE_MANIFEST,
            "test",
            "zeta",
            "C:/work/crates/wgpu/tests/zeta.rs",
            Some("C:/work/target/debug/deps/zeta-1.exe"),
            true,
        ),
        artifact(
            PACKAGE_MANIFEST,
            "lib",
            "gust_wgpu",
            "C:/work/crates/wgpu/src/lib.rs",
            None,
            false,
        ),
        artifact(
            PACKAGE_MANIFEST,
            "test",
            "alpha",
            "C:/work/crates/wgpu/tests/alpha.rs",
            Some("C:/work/target/debug/deps/alpha-2.exe"),
            true,
        ),
        // Another package's test build, a build script and a non-JSON line are not ours.
        artifact(
            "C:/work/crates/other/Cargo.toml",
            "test",
            "beta",
            "C:/work/crates/other/tests/beta.rs",
            Some("C:/work/target/debug/deps/beta-3.exe"),
            true,
        ),
        artifact(
            PACKAGE_MANIFEST,
            "custom-build",
            "build-script-build",
            "C:/work/crates/wgpu/build.rs",
            Some("C:/work/target/debug/build/b.exe"),
            false,
        ),
        "   Finished `test` profile [unoptimized + debuginfo] target(s) in 0.1s".to_owned(),
        "{\"reason\":\"build-finished\",\"success\":true}".to_owned(),
        artifact(
            PACKAGE_MANIFEST,
            "lib",
            "gust_wgpu",
            "C:/work/crates/wgpu/src/lib.rs",
            Some("C:/work/target/debug/deps/gust_wgpu-4.exe"),
            true,
        ),
        // The same executable named twice counts once.
        artifact(
            PACKAGE_MANIFEST,
            "test",
            "alpha",
            "C:/work/crates/wgpu/tests/alpha.rs",
            Some("C:/work/target/debug/deps/alpha-2.exe"),
            true,
        ),
    ]
    .join("\n")
}

/// The executables are those of the package's test builds, once each, in cargo's order: the unit tests, then the
/// integration targets by name; the doc tests are the last job. A header names the source like cargo's `Running` line.
#[test]
fn the_test_executables_are_read_from_cargos_json_in_cargos_order() {
    let found = test_executables(&cargo_output(), Path::new(PACKAGE_DIR)).unwrap();
    let names: Vec<&str> = found
        .iter()
        .map(|executable| executable.name.as_str())
        .collect();
    assert_eq!(names, ["gust_wgpu", "alpha", "zeta"]);
    assert_eq!(
        found[0].header,
        "     Running unittests src/lib.rs (C:/work/target/debug/deps/gust_wgpu-4.exe)"
    );
    assert_eq!(
        found[1].header,
        "     Running tests/alpha.rs (C:/work/target/debug/deps/alpha-2.exe)"
    );
    assert_eq!(
        found[2].path,
        Path::new("C:/work/target/debug/deps/zeta-1.exe")
    );
    let jobs = plan_jobs(found);
    assert_eq!(jobs.len(), 4);
    assert_eq!(jobs[3], Job::Doc);
    assert_eq!(jobs[3].header(), "   Doc-tests gust_wgpu");
    // No executable at all is an error, so a changed format cannot make the suite a run of nothing.
    let error = test_executables(
        "{\"reason\":\"build-finished\"}\nnot json",
        Path::new(PACKAGE_DIR),
    )
    .unwrap_err();
    assert!(error.contains("no test executable"), "{error}");
    let error = test_executables(&cargo_output(), Path::new("C:/elsewhere")).unwrap_err();
    assert!(error.contains("no test executable"), "{error}");
}

const METADATA: &str = r#"{"packages":[
  {"name":"other","version":"9.9.9","manifest_path":"C:/work/crates/other/Cargo.toml"},
  {"name":"gust-compute-wgpu","version":"1.2.3-rc.1+build.5","authors":["A <a@x>","B"],"description":"d",
   "homepage":null,"repository":"https://example.test/r","license":"MIT","license_file":null,"rust_version":"1.90",
   "readme":"README.md","manifest_path":"C:/work/crates/wgpu/Cargo.toml"}]}"#;

/// The environment of a test run is read from `cargo metadata`: the directory, every `CARGO_PKG_*` variable (empty when
/// the manifest has no value, as in cargo) and the manifest path; a package with a build script is refused.
#[test]
fn the_package_environment_comes_from_cargo_metadata() {
    let package = package_of(METADATA, "gust-compute-wgpu").unwrap();
    assert_eq!(package.manifest_dir, Path::new(PACKAGE_DIR));
    let value = |name: &str| {
        package
            .environment
            .iter()
            .find_map(|(key, value)| (*key == name).then_some(value.as_str()))
            .unwrap_or_else(|| panic!("{name} is not set"))
    };
    assert_eq!(value("CARGO_MANIFEST_DIR"), PACKAGE_DIR);
    assert_eq!(value("CARGO_PKG_NAME"), "gust-compute-wgpu");
    assert_eq!(value("CARGO_PKG_VERSION"), "1.2.3-rc.1+build.5");
    assert_eq!(
        [
            value("CARGO_PKG_VERSION_MAJOR"),
            value("CARGO_PKG_VERSION_MINOR"),
            value("CARGO_PKG_VERSION_PATCH"),
            value("CARGO_PKG_VERSION_PRE")
        ],
        ["1", "2", "3", "rc.1"]
    );
    assert_eq!(value("CARGO_PKG_AUTHORS"), "A <a@x>:B");
    assert_eq!(value("CARGO_PKG_HOMEPAGE"), "");
    assert_eq!(value("CARGO_PKG_LICENSE_FILE"), "");
    assert_eq!(value("CARGO_PKG_RUST_VERSION"), "1.90");
    assert_eq!(value("CARGO_PKG_README"), "README.md");
    assert_eq!(
        value("CARGO_MANIFEST_PATH"),
        "C:/work/crates/wgpu/Cargo.toml"
    );
    let error = package_of(METADATA, "missing").unwrap_err();
    assert!(error.contains("no package `missing`"), "{error}");
    assert!(package_of("{", "x").is_err());
}

/// The executable's `deps` directory and its parent go in front of the existing `PATH`.
#[test]
fn the_library_path_puts_the_deps_directory_and_its_parent_first() {
    let existing = std::env::join_paths(["first", "second"]).unwrap();
    let joined = library_path(Path::new("target/debug/deps/t-1.exe"), Some(existing)).unwrap();
    let folders: Vec<_> = std::env::split_paths(&joined).collect();
    assert_eq!(
        folders,
        [
            Path::new("target/debug/deps"),
            Path::new("target/debug"),
            Path::new("first"),
            Path::new("second")
        ]
    );
    // No existing PATH: the two folders alone. No parent directories: nothing to add.
    let alone = library_path(Path::new("target/debug/deps/t-1.exe"), None).unwrap();
    assert_eq!(std::env::split_paths(&alone).count(), 3);
    assert_eq!(library_path(Path::new("t.exe"), None), None);
}

fn finished(success: bool, stdout: &str, stderr: &str) -> JobOutput {
    JobOutput {
        success,
        status: if success { "exit 0" } else { "exit 101" }.to_owned(),
        stdout: stdout.to_owned(),
        stderr: stderr.to_owned(),
    }
}

/// A block is the header, the whole standard output and then standard error, each ending in a newline, so two blocks
/// never run together; the judged text is the headers and the standard outputs of the jobs that ran, in job order, and
/// a failure names every failing binary and the jobs that never started.
#[test]
fn the_output_of_the_jobs_is_assembled_in_job_order_and_a_failure_names_the_binary() {
    let jobs = plan_jobs(test_executables(&cargo_output(), Path::new(PACKAGE_DIR)).unwrap());
    let ok = |text: &str| finished(true, text, "");
    assert_eq!(
        block(
            &jobs[1],
            &finished(true, "test result: ok. 2 passed", "warning\n")
        ),
        format!("{}\ntest result: ok. 2 passed\nwarning\n", jobs[1].header())
    );
    let results = vec![
        Some(ok("test result: ok. 1 passed; 0 failed; 0 ignored\n")),
        Some(ok("test result: ok. 2 passed; 0 failed; 0 ignored")),
        Some(ok("test result: ok. 3 passed; 0 failed; 0 ignored\n")),
        Some(ok("test result: ok. 4 passed; 0 failed; 0 ignored\n")),
    ];
    let counted = judge(&jobs, &results).unwrap();
    let headers: Vec<_> = counted
        .lines()
        .filter(|line| line.trim_start().starts_with("Running") || line.contains("Doc-tests"))
        .collect();
    assert_eq!(headers.len(), 4);
    // The judge of the single call reads the concatenation: one line per job, the sum of the passed tests.
    let totals = sum_test_results(&counted);
    assert_eq!((totals.summaries, totals.passed, totals.failed), (4, 10, 0));
    // A failure names the binary and its exit; later jobs that never started are counted.
    let mut failed = results.clone();
    failed[1] = Some(finished(
        false,
        "test result: FAILED. 0 passed; 1 failed; 0 ignored\n",
        "",
    ));
    failed[3] = None;
    let error = judge(&jobs, &failed).unwrap_err();
    assert!(
        error.contains("tests/alpha.rs")
            && error.contains("exit 101")
            && error.contains("1 job(s) were not started"),
        "{error}"
    );
    assert!(
        !error.contains("zeta") && !error.contains("Doc-tests"),
        "{error}"
    );
}

/// The package variables inherited from `cargo xtask` are recognised (and the ones that are not about the package are
/// not), a package with a build script is refused for the parallel form, and the doc-test job is started first.
#[test]
fn the_inherited_package_variables_and_a_build_script_are_handled() {
    for name in [
        "CARGO_MANIFEST_DIR",
        "CARGO_MANIFEST_PATH",
        "CARGO_PKG_NAME",
        "cargo_pkg_version",
        "CARGO_PRIMARY_PACKAGE",
        "CARGO_CRATE_NAME",
        "CARGO_BIN_NAME",
        "CARGO_TARGET_TMPDIR",
        "CARGO_BIN_EXE_tool",
    ] {
        assert!(is_package_variable(name), "{name}");
    }
    for name in [
        "CARGO",
        "CARGO_HOME",
        "CARGO_TARGET_DIR",
        "CARGO_TERM_COLOR",
        "PATH",
        "GUST_PROFILE",
    ] {
        assert!(!is_package_variable(name), "{name}");
    }
    let with_script = METADATA.replace(
        r#""readme":"README.md","#,
        r#""readme":"README.md","targets":[{"kind":["lib"]},{"kind":["custom-build"]}],"#,
    );
    let error = package_of(&with_script, "gust-compute-wgpu").unwrap_err();
    assert!(
        error.contains("build script") && error.contains("--gpu-jobs 1"),
        "{error}"
    );
    let without_script = METADATA.replace(
        r#""readme":"README.md","#,
        r#""readme":"README.md","targets":[{"kind":["lib"]},{"kind":["test"]}],"#,
    );
    assert!(package_of(&without_script, "gust-compute-wgpu").is_ok());
    let jobs = plan_jobs(test_executables(&cargo_output(), Path::new(PACKAGE_DIR)).unwrap());
    assert_eq!(start_order(&jobs), [3, 0, 1, 2]);
}

/// Both calls of the parallel form carry every argument of `GPU_SUITE_CALL` (one source), and a call with `--`
/// arguments is refused for the parallel form with the way out.
#[test]
fn the_parallel_calls_are_built_from_the_suite_call_and_a_call_with_test_arguments_is_refused() {
    for extra in [&["--no-run", "--message-format=json"][..], &["--doc"][..]] {
        let call = call_with(extra);
        assert!(
            GPU_SUITE_CALL
                .iter()
                .all(|argument| call.iter().any(|have| have == argument)),
            "{call:?}"
        );
        assert_eq!(
            call[..GPU_SUITE_CALL.len()],
            GPU_SUITE_CALL.map(str::to_owned)
        );
        assert_eq!(
            call[GPU_SUITE_CALL.len()..],
            extra.iter().map(|a| (*a).to_owned()).collect::<Vec<_>>()
        );
    }
    assert!(GPU_SUITE_CALL.contains(&"gust-compute-wgpu"));
    assert_eq!(refuse_call(&GPU_SUITE_CALL), Ok(()));
    let error = refuse_call(&["test", "-p", "x", "--", "--nocapture"]).unwrap_err();
    assert!(error.contains("--gpu-jobs 1"), "{error}");
}

/// The doc-test job prints its header once: cargo's own `Doc-tests` line on standard error is left out of its block, and
/// only for that job.
#[test]
fn the_doc_test_block_prints_its_header_once() {
    let text = block(
        &Job::Doc,
        &finished(
            true,
            "test result: ok. 1 passed\n",
            "   Doc-tests gust_wgpu\nnote\n",
        ),
    );
    assert_eq!(text.matches("Doc-tests").count(), 1, "{text}");
    assert!(text.ends_with("note\n"), "{text}");
    let jobs = plan_jobs(test_executables(&cargo_output(), Path::new(PACKAGE_DIR)).unwrap());
    let other = block(&jobs[0], &finished(true, "", "   Doc-tests gust_wgpu\n"));
    assert!(other.contains("Doc-tests"), "{other}");
}

/// A step and its header: the parallel form names its width, the single call keeps its own text, and the routing sends
/// `--gpu-jobs 1` (and every other step) to the whole-call runner, never the parallel one.
#[test]
fn the_gpu_suite_step_is_labelled_and_routed_by_its_jobs() {
    assert_eq!(run_label(Step::GpuSuite, 1), Step::GpuSuite.label());
    let parallel = run_label(Step::GpuSuite, 3);
    assert!(
        parallel.contains("-p gust-compute-wgpu")
            && parallel.contains("3 at a time")
            && parallel != Step::GpuSuite.label(),
        "{parallel}"
    );
    assert_eq!(run_label(Step::AlgoGpuSuite, 3), Step::AlgoGpuSuite.label());
    let calls = RefCell::new(Vec::new());
    let in_parts = |width: usize| {
        calls.borrow_mut().push(format!("parts {width}"));
        Ok(())
    };
    let whole = |step: Step| {
        calls.borrow_mut().push(format!("whole {}", step.label()));
        Ok(())
    };
    run_step_with(Step::GpuSuite, 1, &in_parts, &whole).unwrap();
    run_step_with(Step::GpuSuite, 3, &in_parts, &whole).unwrap();
    run_step_with(Step::AlgoGpuSuite, 3, &in_parts, &whole).unwrap();
    assert_eq!(
        *calls.borrow(),
        [
            format!("whole {}", Step::GpuSuite.label()),
            "parts 3".to_owned(),
            format!("whole {}", Step::AlgoGpuSuite.label())
        ]
    );
    let failing = |_: usize| Err("boom".to_owned());
    assert_eq!(
        run_step_with(Step::GpuSuite, 2, &failing, &whole),
        Err("boom".to_owned())
    );
}

/// The scheduler runs jobs at most `width` at a time and really in parallel (a serial scheduler fails the `>= 2`), hands
/// the results to `emit` in job order whatever the order of completion, and starts jobs in the order it is given.
#[test]
fn the_scheduler_runs_in_parallel_within_the_width_and_emits_in_job_order() {
    use std::{
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        thread::sleep,
        time::Duration,
    };
    let jobs: Vec<u64> = (0..8).collect();
    let identity: Vec<usize> = (0..8).collect();
    let running = AtomicUsize::new(0);
    let widest = AtomicUsize::new(0);
    let mut emitted = Vec::new();
    // Earlier jobs take longer, so completion order is the reverse of job order.
    let results = run_in_order(
        &jobs,
        &identity,
        3,
        &|job: &u64| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            widest.fetch_max(now, Ordering::SeqCst);
            sleep(Duration::from_millis(20 * (8 - job)));
            running.fetch_sub(1, Ordering::SeqCst);
            *job * 10
        },
        &|_: &u64| false,
        |index, result| emitted.push((index, *result)),
    );
    assert_eq!(
        emitted,
        (0..8).map(|n| (n, n as u64 * 10)).collect::<Vec<_>>()
    );
    assert_eq!(
        results,
        (0..8).map(|n| Some(n as u64 * 10)).collect::<Vec<_>>()
    );
    let widest = widest.load(Ordering::SeqCst);
    assert!((2..=3).contains(&widest), "{widest} jobs at once");
    // Started in the order given (one at a time, so the order is exact), printed in job order.
    let starts = [5usize, 0, 1, 2, 3, 4, 6, 7];
    let begun = Mutex::new(Vec::new());
    let mut printed = Vec::new();
    run_in_order(
        &jobs,
        &starts,
        1,
        &|job: &u64| {
            begun.lock().unwrap().push(*job);
            *job
        },
        &|_: &u64| false,
        |index, _| printed.push(index),
    );
    assert_eq!(*begun.lock().unwrap(), [5, 0, 1, 2, 3, 4, 6, 7]);
    assert_eq!(printed, [0, 1, 2, 3, 4, 5, 6, 7]);
    // No job at all, and a width above the number of jobs, are fine.
    assert!(
        run_in_order(
            &[] as &[u64],
            &[],
            4,
            &|job: &u64| *job,
            &|_: &u64| false,
            |_, _| {}
        )
        .is_empty()
    );
    assert_eq!(
        run_in_order(
            &[7u64],
            &[0],
            9,
            &|job: &u64| *job,
            &|_: &u64| false,
            |_, _| {}
        ),
        [Some(7)]
    );
}

/// Waits until `count` jobs have started, for at most 5 s: a scheduler that never reaches the width ends the wait and
/// fails the caller's asserts instead of hanging the test run.
fn wait_for_started(started: &std::sync::atomic::AtomicUsize, count: usize) {
    use std::{
        sync::atomic::Ordering,
        thread::sleep,
        time::{Duration, Instant},
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while started.load(Ordering::SeqCst) < count && Instant::now() < deadline {
        sleep(Duration::from_millis(1));
    }
}

/// After a failure no job starts, and the jobs that were running are awaited and handed on, so every failure is
/// there to be named and the judged result is an error: one at a time, and three at a time with two jobs still running
/// when the first fails.
#[test]
fn a_failure_stops_new_jobs_awaits_the_running_ones_and_names_every_failure() {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread::sleep,
        time::Duration,
    };
    let jobs: Vec<u64> = (0..8).collect();
    let identity: Vec<usize> = (0..8).collect();
    let started = AtomicUsize::new(0);
    let mut seen = Vec::new();
    let results = run_in_order(
        &jobs,
        &identity,
        1,
        &|job: &u64| {
            started.fetch_add(1, Ordering::SeqCst);
            *job
        },
        &|result: &u64| *result == 2,
        |index, _| seen.push(index),
    );
    assert_eq!(started.load(Ordering::SeqCst), 3);
    assert_eq!(seen, [0, 1, 2]);
    assert_eq!(results[..3], [Some(0), Some(1), Some(2)]);
    assert!(results[3..].iter().all(Option::is_none));
    // Width 3: job 0 fails at once, jobs 1 and 2 are still running and both fail later; nothing after them starts.
    let started = AtomicUsize::new(0);
    let mut seen = Vec::new();
    let results = run_in_order(
        &jobs,
        &identity,
        3,
        &|job: &u64| {
            started.fetch_add(1, Ordering::SeqCst);
            if *job == 0 {
                // Fail only once the other two are running (bounded; the asserts below report a serial scheduler).
                wait_for_started(&started, 3);
            } else {
                sleep(Duration::from_millis(150));
            }
            *job
        },
        &|result: &u64| *result <= 2,
        |index, _| seen.push(index),
    );
    assert_eq!(started.load(Ordering::SeqCst), 3);
    assert_eq!(seen, [0, 1, 2]);
    assert_eq!(results[..3], [Some(0), Some(1), Some(2)]);
    assert!(results[3..].iter().all(Option::is_none));
    // The jobs of such a run are all named, and the run is an error.
    let jobs = plan_jobs(test_executables(&cargo_output(), Path::new(PACKAGE_DIR)).unwrap());
    let outcome = vec![
        Some(finished(false, "", "")),
        Some(finished(false, "", "")),
        Some(finished(
            true,
            "test result: ok. 1 passed; 0 failed; 0 ignored\n",
            "",
        )),
        None,
    ];
    let error = judge(&jobs, &outcome).unwrap_err();
    assert!(
        error.contains("2 of the 4 jobs") && error.contains("lib.rs") && error.contains("alpha.rs"),
        "{error}"
    );
}

/// A fake process: it ends after `ends_after` checks (never when `None`), and records that it was killed.
struct FakeProcess {
    ends_after: Option<u32>,
    checks: u32,
    killed: bool,
}

impl Process for FakeProcess {
    type Exit = u32;

    fn exit(&mut self) -> Option<u32> {
        self.checks += 1;
        self.ends_after
            .filter(|after| self.checks > *after)
            .map(|_| 7)
    }

    fn kill(&mut self) {
        self.killed = true;
    }
}

/// A process is awaited until it ends; one that outlasts the limit is killed and reported as timed out; one that is
/// cancelled is killed at the next check. The kill never happens for a process that finished.
#[test]
fn a_job_that_outlasts_its_limit_or_is_cancelled_is_killed() {
    use std::{sync::atomic::AtomicBool, time::Duration};
    let limit = Duration::from_secs(30);
    let poll = Duration::from_millis(1);
    let never = AtomicBool::new(false);
    let mut quick = FakeProcess {
        ends_after: Some(3),
        checks: 0,
        killed: false,
    };
    assert_eq!(supervise(&mut quick, limit, poll, &never), End::Finished(7));
    assert!(!quick.killed);
    let mut hung = FakeProcess {
        ends_after: None,
        checks: 0,
        killed: false,
    };
    assert_eq!(
        supervise(&mut hung, Duration::from_millis(30), poll, &never),
        End::TimedOut
    );
    assert!(hung.killed && hung.checks > 1);
    let mut doomed = FakeProcess {
        ends_after: None,
        checks: 0,
        killed: false,
    };
    assert_eq!(
        supervise(&mut doomed, limit, poll, &AtomicBool::new(true)),
        End::Cancelled
    );
    assert!(doomed.killed);
    // A process that has already ended is reported as finished even when the run was cancelled meanwhile.
    let mut done = FakeProcess {
        ends_after: Some(0),
        checks: 0,
        killed: false,
    };
    assert_eq!(
        supervise(&mut done, limit, poll, &AtomicBool::new(true)),
        End::Finished(7)
    );
    assert!(!done.killed);
    assert!(JOB_TIME_LIMIT >= Duration::from_secs(300));
}

/// When the job that was started first (the doc-test job, last in job order) fails and another running job ends
/// cancelled, both blocks are emitted, in job order, though a gap (a job never started) lies between them.
#[test]
fn a_run_that_stops_early_still_emits_every_finished_job_in_job_order() {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread::sleep,
        time::Duration,
    };
    let jobs: Vec<u64> = (0..4).collect();
    let starts = [3usize, 0, 1, 2];
    let started = AtomicUsize::new(0);
    let mut emitted = Vec::new();
    let results = run_in_order(
        &jobs,
        &starts,
        2,
        &|job: &u64| {
            started.fetch_add(1, Ordering::SeqCst);
            if *job == 3 {
                wait_for_started(&started, 2);
            } else {
                sleep(Duration::from_millis(100));
            }
            *job
        },
        &|result: &u64| *result == 3,
        |index, _| emitted.push(index),
    );
    assert_eq!(emitted, [0, 3]);
    assert_eq!(results, [Some(0), None, None, Some(3)]);
}

#[test]
fn step_time_line_shows_one_decimal_and_the_label() {
    assert_eq!(
        step_time_line("cargo fmt", std::time::Duration::from_millis(12_349)),
        "   step 12.3 s: cargo fmt"
    );
}

#[test]
fn timing_summary_lists_five_slowest_first_with_ties_in_step_order_then_the_sum() {
    use std::time::Duration;
    let secs = |label: &str, s: u64| (label.to_string(), Duration::from_secs(s));
    let timings = [
        secs("a", 2),
        secs("b", 9),
        secs("c", 2),
        secs("d", 5),
        secs("e", 2),
        secs("f", 7),
        secs("g", 1),
    ];
    let lines = timing_summary(&timings, Duration::from_secs(30));
    assert_eq!(
        lines,
        [
            "   slowest: 9.0 s b",
            "   slowest: 7.0 s f",
            "   slowest: 5.0 s d",
            "   slowest: 2.0 s a",
            "   slowest: 2.0 s c",
            "   steps 28.0 s of 30.0 s",
        ]
    );
}

#[test]
fn timing_summary_of_few_steps_lists_all_of_them() {
    use std::time::Duration;
    let timings = [("only".to_string(), Duration::from_millis(1500))];
    assert_eq!(
        timing_summary(&timings, Duration::from_secs(2)),
        ["   slowest: 1.5 s only", "   steps 1.5 s of 2.0 s"]
    );
}
