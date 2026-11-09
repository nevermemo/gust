use crate::{
    arch::check_arch,
    checks::{check_cpu, check_feature, tests_not_ok},
    cpu_target::check_cpu_target,
    difftest::check_difftest,
    docs::docs_check,
    lint::lint,
    process::{
        require_command, run_cargo_test_captured_under, run_counted, run_direct, without_colors,
    },
    ui::check_ui,
    verify::{
        ADAPTER_INFO_ARGS, ALGO_GPU_CALL, GPU_SUITE_CALL, GPU_SUITE_PIN, ProfileTestRunner,
        VerifyMode, check_algo_gpu_counts, check_gpu_suite_counts, check_gpu_target_profile,
        gpu_target_pin, name_adapter, run_gpu_profile_suite, run_gpu_target, verify,
    },
    workspace::workspace_root,
};
use std::path::Path;

/// An evidence tier of docs/development/VALIDATION.md, runnable as one command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Suite {
    Unit,
    Cpu,
    Slangc,
    Spirv,
    Reflect,
    Gpu,
    Full,
}

/// Runs a `cargo test` call in a directory, under a `GUST_PROFILE` when one is given, and returns its totals and
/// captured standard output ([`run_cargo_test_captured_under`]).
type TestRunner<'a> = ProfileTestRunner<'a>;

/// A `cargo test` call whose output must report each test of `names` as `ok` (`test <name> ... ok`), under `profile`
/// when it is given. The name is the one libtest prints, module path included, so a renamed, moved, ignored or deleted
/// test fails the step even when other tests of the call still pass. `lines` are the beginnings of lines that the output
/// must also hold (the call has `--show-output`, so a passing test's prints are in it): the proof that the `profile`
/// reached the build, for tests that pass under any profile. A line is found when a trimmed line of the output starts
/// with it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NamedTests {
    pub(crate) args: &'static [&'static str],
    pub(crate) names: &'static [&'static str],
    pub(crate) profile: Option<&'static str>,
    pub(crate) lines: &'static [&'static str],
}

/// One step of a suite. Every step is an existing command or a `cargo test` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// `cargo <args>` in the workspace root.
    Cargo(&'static [&'static str]),
    /// `cargo test <filter>` in the workspace root that must run at least one test (a filter that
    /// matches nothing exits 0 in cargo).
    CountedCargo(&'static [&'static str]),
    /// [`GPU_SUITE_CALL`] with the `test result:` lines checked against [`GPU_SUITE_PIN`], as `verify --mode gpu`
    /// does (m5.f1): the same guard, so `test --suite Gpu` cannot pass with a dropped Gpu test file.
    GpuSuite,
    /// [`ALGO_GPU_CALL`], every test of `gust-algo`, with its `gpu::` tests required by name (`ALGO_GPU_TESTS` in
    /// verify.rs), as `verify --mode gpu` does (m7.r1). Not a Unit step: the `unit` CI job stays free of wgpu (R4.10).
    AlgoGpuSuite,
    /// A `cargo test` call with named tests required (`NamedTests`), run in the workspace root: the helper-backed
    /// tests of `cargo-gust` and its variant gate (m8.x1).
    Named(&'static NamedTests),
    /// Prints the adapter the Gpu suite's tests will open (`adapter_info`), so a log names the device it ran on.
    NameAdapter,
    /// A tool on `PATH`; a missing tool fails the suite by name.
    RequireTool(&'static str),
    Lint,
    DocsCheck,
    CheckArch,
    /// `cargo xtask check-ui`: `cargo check` of the fixture crates under tests/ui.
    CheckUi,
    /// `cargo xtask check-cpu`: the examples' CPU-twin tests (`checks::CPU_TESTS`).
    CheckCpu,
    /// `cargo xtask check-cpu-target`: the production CPU target on the fixture `tests/cpu-target`.
    CheckCpuTarget,
    /// `cargo xtask check-difftest <args>`: the differential test harness (m5.c2). Without `--cpu-only` it needs an
    /// adapter and Slang.
    CheckDifftest(&'static [&'static str]),
    /// `cargo xtask check-feature <area>`.
    Feature(&'static str),
    /// `cargo xtask verify --mode <mode>`, recording only when the flag is set.
    Verify(VerifyMode, bool),
}

const MACROS_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "gust-compute-macros"]);
const XTASK_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "xtask"]);
const CORE_LIB_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "gust-compute", "--lib"]);
/// The `gust::cpu` runtime tests (the CPU twin's device, `ulps`, call checks). They need neither slangc nor a GPU
/// (the `gust` crate has no build script and no GPU dependency), but share the library test binary with tests that
/// do need slangc, so the name filter carves them out for the Unit suite and CI.
const CORE_CPU_TESTS: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--lib",
    "cpu::tests::",
]);
/// The `gust::artifact` tests (`verify_artifact` on the committed precompiled vector-add fixture, the identity and layout
/// negatives, `content_key` vectors). Pure: no slangc, no process, no GPU. The integration test file is whole; the
/// library tests are carved out by name and counted, as `cpu::tests::` is.
const CORE_ARTIFACT_TESTS: Step = Step::Cargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "artifact",
]);
/// The scalar coretests transcription on the CPU twin (m5.d3): every case of docs/development/CORETESTS.md against its
/// literal, with the id guards. Pure: no slangc, no process, no GPU (the GPU half is `check-feature gpu-semantics`).
const CORE_CORETESTS_TWIN: Step = Step::Cargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "coretests_twin",
]);
/// The uniforms fixture (m6.u2) compiled with rustc and run on the CPU twin against a hand reference, with the
/// `Std140V1` size and offsets of its structs. Pure: no slangc, no process, no GPU.
const CORE_UNIFORMS_TWIN: Step = Step::Cargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "uniforms_twin",
]);
/// The off state of the `slang-runtime` feature (D44). The workspace build unifies the feature on (the examples and the
/// self dev-dependencies ask for it), so these two calls, which build `gust` and `gust-wgpu` alone and without default
/// features, are the only observers of the bridge's cfg attributes: they fail when something the precompiled path needs
/// slips behind the feature, or when something behind the feature leaks into the off state. They need cargo only.
const OFF_STATE_CLIPPY: Step = Step::Cargo(&[
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
]);
const OFF_STATE_CHECK: Step = Step::Cargo(&[
    "check",
    "--locked",
    "-p",
    "gust-compute",
    "-p",
    "gust-compute-wgpu",
    "--no-default-features",
    "--lib",
]);
const CORE_ARTIFACT_UNIT_TESTS: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--lib",
    "artifact::tests::",
]);
const CORE_DIAGNOSTICS_TESTS: Step = Step::Cargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "diagnostics",
]);
/// The naga layout gate (slice mx.n): every fixture's WGSL through wgpu's own naga against the descriptors' layouts.
/// It needs slangc and the native reflection helper but no adapter, so it runs here (the CI job `slang`) as well as in
/// `check-feature reflection` and, counted in `GPU_SUITE_PIN`, in `verify --mode gpu` (the job `gpu-software`).
const NAGA_LAYOUT_GATE: Step = Step::Cargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute-wgpu",
    "--test",
    "naga_layout",
]);
/// The routed device handle without a GPU runtime (m9.h1): `RoutedDevice<NoGpu>` runs a CPU-routed kernel, refuses a
/// GPU-routed one with `Error::NoGpu`, and refuses a buffer of another handle. Counted. The GPU half is the
/// `device_handle` test of `gust-compute-wgpu` (Gpu suite, `GPU_SUITE_PIN`).
const CORE_ROUTED_HANDLE: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "routed_handle",
]);
/// The test binaries of `gust-compute` that need neither Slang nor the helper nor an adapter, and were in no suite
/// (m8.x1, the survey of m8.p): the pixel and texture twins on the CPU device, the parallel CPU dispatch, the toolchain
/// discovery tests (one ignored child process test stays ignored), and the doc tests. Each runs in the `unit` CI job,
/// which has no slangc. Counted: a filter or a deleted file that leaves nothing to run fails.
const CORE_PIXEL_TWIN: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "pixel_twin",
]);
const CORE_TEXTURES_TWIN: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "textures_twin",
]);
const CORE_PARALLEL_TESTS: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "parallel",
]);
const CORE_TOOLCHAIN_TESTS: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust-compute",
    "--test",
    "toolchain",
]);
const CORE_DOC_TESTS: Step =
    Step::CountedCargo(&["test", "--locked", "-p", "gust-compute", "--doc"]);
/// The name filter of the tests of `cargo-gust` that build the stored artifacts with the native helper (the one of
/// `testdata::variant` is not `#[ignore]`d, the one of `export::uniform_tests` is).
const HELPER_BUILD_TEST: &str = "the_helper_builds_the_stored_artifacts";
/// Every test of `cargo-gust` that needs neither Slang nor the helper (295 tests: 292 pass, one is ignored for the helper
/// and two are skipped by their full names with `--exact`, so a test that merely has the text in its name is not
/// dropped; the helper-backed ones are left to [`CARGO_GUST_HELPER_TESTS`] and [`CARGO_GUST_VARIANT_GATE`]).
const CARGO_GUST_TESTS: Step = Step::CountedCargo(&[
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
]);
/// The three helper-backed tests of `cargo-gust` (the Slangc tier: slangc and the reflection helper are on `PATH`; the
/// `#[ignore]`d two run through `--include-ignored`): the real compile step on the stored records of the export and of
/// the variant fixture, and the schema 3 output of the built helper against the committed expectations.
const CARGO_GUST_HELPER_TESTS: NamedTests = NamedTests {
    args: &[
        "test",
        "--locked",
        "-p",
        "cargo-gust",
        "--",
        "--include-ignored",
        HELPER_BUILD_TEST,
        "schema_3_output",
    ],
    names: &[
        "export::uniform_tests::the_helper_builds_the_stored_artifacts",
        "helper::tests::schema_3_output_of_the_built_helper_matches_the_committed_expectations",
        "testdata::variant::the_helper_builds_the_stored_artifacts",
    ],
    profile: None,
    lines: &[],
};
/// The `dev+subgroups` variant gate (m8.b2 R2): the variant fixture of `cargo-gust` (`testdata::variant`, six tests,
/// the helper-backed one included) under the build that selects the subgroup bodies, so the stored four artifacts and
/// the macro's selection agree under the capability as well as without it. These tests pass under any profile (they are
/// "driven by key", testdata.rs), so their `ok` lines do not show that the capability reached the build; the call has
/// `--show-output` and the output must hold the print the fixture makes of the capabilities it checked under.
const CARGO_GUST_VARIANT_GATE: NamedTests = NamedTests {
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
};
/// The Unit tier's differential batch (R1-13): seeds 1 to 4, 64 programs, the interpreter against the CPU twin. No
/// adapter and no Slang; it builds the generated crate (`gust` and `gust-wgpu`) with cargo.
const DIFFTEST_CPU: Step = Step::CheckDifftest(&["--cpu-only", "--seeds", "1-4"]);
/// The harness crate `tests/difftest` is a workspace of its own, so the root's `cargo fmt`, `clippy` and `test` never
/// reach it: its own format check, clippy (all targets, warnings denied) and tests (the generator, interpreter, emitter
/// and shrinker unit tests, the corpus pin and the CPU `short_circuit` test) are Unit steps of their own (m5.c2).
const DIFFTEST_FMT_ARGS: &[&str] = &[
    "fmt",
    "--manifest-path",
    "tests/difftest/Cargo.toml",
    "--",
    "--check",
];
const DIFFTEST_CLIPPY_ARGS: &[&str] = &[
    "clippy",
    "--locked",
    "--manifest-path",
    "tests/difftest/Cargo.toml",
    "--all-targets",
    "--",
    "-D",
    "warnings",
];
const DIFFTEST_TESTS_ARGS: &[&str] = &[
    "test",
    "--locked",
    "--manifest-path",
    "tests/difftest/Cargo.toml",
];
const DIFFTEST_FMT: Step = Step::Cargo(DIFFTEST_FMT_ARGS);
const DIFFTEST_CLIPPY: Step = Step::Cargo(DIFFTEST_CLIPPY_ARGS);
const DIFFTEST_TESTS: Step = Step::Cargo(DIFFTEST_TESTS_ARGS);
/// The three cargo calls of the harness crate (its own workspace, which the root's fmt, clippy and test never reach),
/// as the Full mode of `verify` (`check-full`) runs them after the workspace tests.
pub(crate) const DIFFTEST_CARGO_CALLS: [&[&str]; 3] =
    [DIFFTEST_FMT_ARGS, DIFFTEST_CLIPPY_ARGS, DIFFTEST_TESTS_ARGS];
/// The steps of the harness crate in the Unit and Slangc suites, in order. The CI job `difftest` runs them as steps of
/// its own, so the `unit` and `slang` jobs pass `--skip-difftest` and keep their time (R4.10); a local run of the
/// suite runs them all.
pub(crate) const DIFFTEST_UNIT_STEPS: [Step; 4] =
    [DIFFTEST_FMT, DIFFTEST_CLIPPY, DIFFTEST_TESTS, DIFFTEST_CPU];
/// The Gpu tier's differential batch: all eight seeds on the first adapter, and the `short_circuit` GPU test.
const DIFFTEST_GPU: Step = Step::CheckDifftest(&[]);

const ALL: [Suite; 7] = [
    Suite::Unit,
    Suite::Cpu,
    Suite::Slangc,
    Suite::Spirv,
    Suite::Reflect,
    Suite::Gpu,
    Suite::Full,
];

impl Suite {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        ALL.iter()
            .copied()
            .find(|suite| suite.name().eq_ignore_ascii_case(value))
            .ok_or_else(|| {
                format!(
                    "unknown suite `{value}`; expected {}",
                    Self::names().join("|")
                )
            })
    }

    pub(crate) fn names() -> Vec<&'static str> {
        ALL.iter().map(|suite| suite.name()).collect()
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Unit => "Unit",
            Self::Cpu => "Cpu",
            Self::Slangc => "Slangc",
            Self::Spirv => "Spirv",
            Self::Reflect => "Reflect",
            Self::Gpu => "Gpu",
            Self::Full => "Full",
        }
    }

    /// What a pass of this suite shows.
    pub(crate) fn covers(self) -> &'static str {
        match self {
            Self::Unit => {
                "macro validation and translation tests, xtask tests, lint, docs-check, check-arch, check-ui, check-cpu (the CPU twins of the examples against their host references), the gust::cpu unit tests, the gust::artifact tests (verify_artifact on a real precompiled fixture), the coretests_twin test (the scalar Rust core test cases on the CPU twin against their literals), the uniforms_twin test (the uniforms fixture on the CPU twin against a hand reference), the pixel_twin, textures_twin, parallel, routed_handle (the routed device handle without a GPU runtime) and toolchain test binaries and the doc tests of gust-compute, the tests of cargo-gust that need no helper, clippy and check of gust and gust-wgpu without default features (the slang-runtime off state), the format check, clippy and tests of the harness crate tests/difftest (a workspace of its own; CI runs them in the job `difftest`, and `--skip-difftest` leaves them out of the `unit` and `slang` jobs), and the differential batch of seeds 1 to 4 in its CPU-only mode (64 generated kernels: an independent interpreter against the CPU twin, report.json judged by cargo xtask check-difftest); no external tools"
            }
            Self::Cpu => {
                "the production CPU target (GUST_PROFILE=cpu) on the fixture tests/cpu-target: the kernels build in release, run on gust::cpu::CpuDevice and equal independent host references; the binary holds no compiler, helper or GPU runtime; gust-wgpu refuses the profile; the cfg(test) and --cfg gust_dual builds keep one impl; the profile flips rebuild correctly; cargo only"
            }
            Self::Slangc => {
                "Unit, plus the gust library tests and the diagnostics integration test, which compile generated Slang with slangc, and the naga layout gate (every fixture's WGSL, from slangc and from the native reflection helper, parsed and validated by wgpu's naga and compared with the descriptors' layouts; no adapter), the three helper-backed tests of cargo-gust (the real compile step on the stored export and variant records, the helper's schema 3 output), and the variant fixture of cargo-gust again under GUST_PROFILE=dev+subgroups"
            }
            Self::Spirv => {
                "the example binaries export SPIR-V and spirv-val accepts every file (vulkan1.2)"
            }
            Self::Reflect => {
                "the D17 layout gate: native reflection helper, core and wgpu reflection tests, and the naga layout gate (every fixture's WGSL checked by wgpu's naga against the descriptors' layouts)"
            }
            Self::Gpu => {
                "the adapter named in the log (adapter_info), all gust-wgpu tests on the first Vulkan adapter wgpu finds (their `test result:` lines checked against GPU_SUITE_PIN, as in verify --mode gpu), then all gust-algo tests (the reduce and scan calls, their dispatch counts and the float bound on the adapter; its `gpu::` tests are pinned by name), then the differential batch of all 8 seeds (128 generated kernels: interpreter, CPU twin and GPU compared; report.json judged by cargo xtask check-difftest), the short_circuit GPU test of tests/difftest and the same batch again under GUST_PROFILE=dev+subgroups (judged the same way; every program must report the `subgroups` capability from its own build, and its exported records must be byte for byte those of the `dev` batch). The dev+subgroups batch opens its device with the subgroups capability, so the first Vulkan adapter must offer subgroup operations or the suite fails with MissingCapability"
            }
            Self::Full => {
                "the release gate: lint, fmt, clippy, workspace tests, the fmt, clippy and tests of the harness crate tests/difftest, the release-profile Slang tests of the gust library and the release-profile known-mismatch ledger, examples (including `subgroup-reduce`, run under GUST_PROFILE=native+subgroups from its committed artifacts, which also needs an adapter with subgroup operations), the whole Gpu suite again under GUST_PROFILE=dev+subgroups (the pinned gust-wgpu targets, then all of gust-algo, so that its subgroup body runs in the release gate), artifacts and SPIR-V validation, check-dual in full mode, then the whole check-precompiled, the whole check-web (Chrome with a WebGPU adapter; a missing one fails the suite, only `cargo xtask check-full --skip-web` goes without, and records that) and, last, check-difftest (all 8 seeds on the adapter, the short_circuit GPU test and the dev+subgroups batch with its byte compare); writes .ai/VALIDATION.json"
            }
        }
    }

    /// What a pass of this suite does not show.
    pub(crate) fn not_covered(self) -> &'static str {
        match self {
            Self::Unit => {
                "that generated Slang compiles, SPIR-V validity, reflection layouts, or any GPU execution (the differential batch runs CPU-only here: the interpreter and the twin agreeing says nothing about an adapter)"
            }
            Self::Cpu => {
                "anything about a GPU or an adapter, the Unit tier's tests, or the shadow-semantics twin under cfg(test); a CPU result is never GPU evidence (D42)"
            }
            Self::Slangc => "SPIR-V validation, the reflection gate, or any GPU execution",
            Self::Spirv => {
                "unit tests, lint, the reflection gate, or GPU execution against host references"
            }
            Self::Reflect => "unit tests, SPIR-V validation, or GPU readback of kernel results",
            Self::Gpu => {
                "SPIR-V validation or lint; the device is whatever wgpu picks, so name it in the run record, and the dev+subgroups batch needs that adapter to offer subgroup operations (a first adapter without them fails the suite instead of skipping the batch); the plain suite is a `dev` build, so the gust-wgpu targets and the gust-algo tests under dev+subgroups are the profile form (`--suite Gpu --profile dev+subgroups`), which runs the pinned gust-wgpu targets and then all of gust-algo with the capabilities line of each build required, and is not part of the plain suite"
            }
            Self::Full => {
                "nothing named here is skipped, but a software adapter is only Gpu (software); the dev+subgroups batch and the subgroup-reduce example need an adapter with subgroup operations, and the Chrome WebGPU backend maps no subgroup feature, so the web check shows only that the request is refused"
            }
        }
    }

    pub(crate) fn steps(self) -> Vec<Step> {
        let unit = [
            MACROS_TESTS,
            XTASK_TESTS,
            Step::Lint,
            Step::DocsCheck,
            Step::CheckArch,
            Step::CheckUi,
            Step::CheckCpu,
            CORE_CPU_TESTS,
            CORE_ARTIFACT_UNIT_TESTS,
            CORE_ARTIFACT_TESTS,
            CORE_CORETESTS_TWIN,
            CORE_UNIFORMS_TWIN,
            CORE_PIXEL_TWIN,
            CORE_TEXTURES_TWIN,
            CORE_PARALLEL_TESTS,
            CORE_ROUTED_HANDLE,
            CORE_TOOLCHAIN_TESTS,
            CORE_DOC_TESTS,
            CARGO_GUST_TESTS,
            OFF_STATE_CLIPPY,
            OFF_STATE_CHECK,
            DIFFTEST_FMT,
            DIFFTEST_CLIPPY,
            DIFFTEST_TESTS,
            DIFFTEST_CPU,
        ];
        match self {
            Self::Unit => unit.to_vec(),
            Self::Cpu => vec![Step::CheckCpuTarget],
            Self::Slangc => {
                let mut steps = vec![Step::RequireTool("slangc")];
                steps.extend(unit);
                steps.push(CORE_LIB_TESTS);
                steps.push(CORE_DIAGNOSTICS_TESTS);
                steps.push(NAGA_LAYOUT_GATE);
                steps.push(Step::Named(&CARGO_GUST_HELPER_TESTS));
                steps.push(Step::Named(&CARGO_GUST_VARIANT_GATE));
                steps
            }
            Self::Spirv => vec![Step::Verify(VerifyMode::Artifacts, false)],
            Self::Reflect => vec![Step::Feature("reflection")],
            Self::Gpu => vec![
                Step::RequireTool("slangc"),
                Step::NameAdapter,
                Step::GpuSuite,
                Step::AlgoGpuSuite,
                DIFFTEST_GPU,
            ],
            Self::Full => vec![Step::Verify(VerifyMode::Full, true)],
        }
    }
}

impl Step {
    fn label(self) -> String {
        match self {
            Self::Cargo(args) | Self::CountedCargo(args) => format!("cargo {}", args.join(" ")),
            Self::GpuSuite => format!("cargo {}", GPU_SUITE_CALL.join(" ")),
            Self::AlgoGpuSuite => format!("cargo {}", ALGO_GPU_CALL.join(" ")),
            Self::Named(tests) => match tests.profile {
                Some(profile) => format!("GUST_PROFILE={profile} cargo {}", tests.args.join(" ")),
                None => format!("cargo {}", tests.args.join(" ")),
            },
            Self::NameAdapter => format!("name the adapter: cargo {}", ADAPTER_INFO_ARGS.join(" ")),
            Self::RequireTool(tool) => format!("require {tool} on PATH"),
            Self::Lint => "cargo xtask lint".to_owned(),
            Self::DocsCheck => "cargo xtask docs-check".to_owned(),
            Self::CheckArch => "cargo xtask check-arch".to_owned(),
            Self::CheckUi => "cargo xtask check-ui".to_owned(),
            Self::CheckCpu => "cargo xtask check-cpu".to_owned(),
            Self::CheckCpuTarget => "cargo xtask check-cpu-target".to_owned(),
            Self::CheckDifftest([]) => "cargo xtask check-difftest".to_owned(),
            Self::CheckDifftest(args) => format!("cargo xtask check-difftest {}", args.join(" ")),
            Self::Feature(area) => format!("cargo xtask check-feature {area}"),
            Self::Verify(mode, _) => format!("cargo xtask verify --mode {}", mode.as_str()),
        }
    }

    fn run(self) -> Result<(), String> {
        let root = workspace_root()?;
        self.run_in(&root, &run_cargo_test_captured_under)
    }

    /// [`Step::run`] in `root`, with the runner of the steps that judge a captured `cargo test` output (the Gpu
    /// suite) given, so that a test can show which of them a step uses.
    fn run_in(self, root: &Path, test_runner: &TestRunner<'_>) -> Result<(), String> {
        let root = root.to_path_buf();
        match self {
            Self::Cargo(args) => run_direct(&root, "cargo", args),
            Self::CountedCargo(args) => run_counted(&root, args).map(|_| ()),
            Self::GpuSuite => {
                let call = GPU_SUITE_CALL.map(str::to_owned);
                let (_, stdout) = test_runner(&root, &call, None)?;
                check_gpu_suite_counts(&call, &stdout, GPU_SUITE_PIN)
            }
            Self::AlgoGpuSuite => {
                let call = ALGO_GPU_CALL.map(str::to_owned);
                let (_, stdout) = test_runner(&root, &call, None)?;
                check_algo_gpu_counts(&call, &stdout)
            }
            Self::Named(tests) => {
                let call: Vec<String> = tests.args.iter().map(|arg| (*arg).to_owned()).collect();
                let (_, stdout) = test_runner(&root, &call, tests.profile)?;
                let missing = tests_not_ok(&stdout, tests.names);
                if !missing.is_empty() {
                    return Err(format!(
                        "cargo {}: the test(s) `{}` did not run and pass (renamed, moved, ignored, deleted or filtered out? NamedTests in xtask/src/suite.rs names them)",
                        call.join(" "),
                        missing.join("`, `")
                    ));
                }
                let plain = without_colors(&stdout);
                let absent: Vec<&str> = tests
                    .lines
                    .iter()
                    .copied()
                    .filter(|wanted| !plain.lines().any(|line| line.trim().starts_with(wanted)))
                    .collect();
                if absent.is_empty() {
                    return Ok(());
                }
                Err(format!(
                    "cargo {}{}: the output has no line starting `{}`; the tests passed, but nothing shows that GUST_PROFILE reached the build (or `--show-output`, or the print, is gone)",
                    call.join(" "),
                    tests
                        .profile
                        .map_or_else(String::new, |profile| format!(" (GUST_PROFILE={profile})")),
                    absent.join("`, `")
                ))
            }
            Self::NameAdapter => {
                println!("   {}", name_adapter(&root)?);
                Ok(())
            }
            Self::RequireTool(tool) => require_command(tool),
            Self::Lint => lint(&root, &mut Vec::new()),
            Self::DocsCheck => docs_check(&root, &mut Vec::new()),
            Self::CheckArch => check_arch(&root, &mut Vec::new()),
            Self::CheckUi => check_ui(&root, &mut Vec::new()),
            Self::CheckCpu => check_cpu(),
            Self::CheckCpuTarget => check_cpu_target(&[]),
            Self::CheckDifftest(args) => {
                check_difftest(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>())
            }
            Self::Feature(area) => check_feature(Some(area)),
            Self::Verify(mode, record) => verify(mode, record),
        }
    }
}

/// What `cargo xtask test` was asked to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TestRequest {
    /// A whole suite, with or without `--skip-difftest`.
    Suite { suite: Suite, skip_difftest: bool },
    /// One gust-wgpu test target of the Gpu suite under a `GUST_PROFILE` (m8.w2): `--suite Gpu --profile <p> --test <t>`.
    Target { profile: String, target: String },
    /// The pinned gust-wgpu targets and then `gust-algo`, all under a `GUST_PROFILE` (m8.x1): `--suite Gpu --profile <p>`.
    Profile { profile: String },
}

/// Reads the arguments of `cargo xtask test`. The profile form is fixed in shape and checked here, so a bad profile or
/// an unpinned target fails before anything is built.
pub(crate) fn parse_test_args(args: &[String]) -> Result<TestRequest, String> {
    let usage = || {
        format!(
            "usage: cargo xtask test --suite {} [--skip-difftest]\n       cargo xtask test --suite Gpu --profile <dev|dev+capability> [--test <gust-wgpu test target>]",
            Suite::names().join("|")
        )
    };
    let gpu_only = |suite: Suite| {
        if suite == Suite::Gpu {
            Ok(())
        } else {
            Err(format!(
                "--profile (and --test) run the pinned gpu tests under a profile and exist for --suite Gpu only, not {}",
                suite.name()
            ))
        }
    };
    match args {
        [flag, name] if flag == "--suite" => Ok(TestRequest::Suite {
            suite: Suite::parse(name)?,
            skip_difftest: false,
        }),
        [flag, name, skip] if flag == "--suite" && skip == "--skip-difftest" => {
            Ok(TestRequest::Suite {
                suite: Suite::parse(name)?,
                skip_difftest: true,
            })
        }
        [flag, name, profile_flag, profile, test_flag, target]
            if flag == "--suite" && profile_flag == "--profile" && test_flag == "--test" =>
        {
            gpu_only(Suite::parse(name)?)?;
            check_gpu_target_profile(profile)?;
            gpu_target_pin(target)?;
            Ok(TestRequest::Target {
                profile: profile.clone(),
                target: target.clone(),
            })
        }
        [flag, name, profile_flag, profile] if flag == "--suite" && profile_flag == "--profile" => {
            gpu_only(Suite::parse(name)?)?;
            check_gpu_target_profile(profile)?;
            Ok(TestRequest::Profile {
                profile: profile.clone(),
            })
        }
        _ => Err(usage()),
    }
}

/// `cargo xtask test --suite <Unit|Cpu|Slangc|Spirv|Reflect|Gpu|Full> [--skip-difftest]`, or
/// `cargo xtask test --suite Gpu --profile <dev|dev+capability> --test <target>` (m8.w2): one test target of
/// gust-compute-wgpu on the adapter under that `GUST_PROFILE`, counted against `GPU_TARGET_PINS`. m8.s runs
/// `--profile dev+subgroups --test subgroup`; the whole Gpu suite (its pin, `GPU_SUITE_PIN`, describes `dev`) is
/// run without the two flags. Without `--test` the profile form runs every pinned target and then every test of
/// gust-algo under the profile (m8.x1: its evidence tests select the body of the capability).
pub(crate) fn test_command(args: &[String]) -> Result<(), String> {
    match parse_test_args(args)? {
        TestRequest::Suite {
            suite,
            skip_difftest,
        } => run_suite(suite, skip_difftest),
        TestRequest::Target { profile, target } => run_profile_target(&profile, &target),
        TestRequest::Profile { profile } => run_profile_suite(&profile),
    }
}

/// The steps of the profile form without a target: `slangc` on `PATH`, the adapter named, the pinned gust-wgpu targets
/// and `gust-algo`, each judged against its pin and its capabilities line ([`run_gpu_profile_suite`]).
fn run_profile_suite(profile: &str) -> Result<(), String> {
    println!(
        "== suite Gpu under GUST_PROFILE={profile} (the pinned gust-wgpu targets, then gust-algo)"
    );
    println!("-- require slangc on PATH");
    require_command("slangc").map_err(|error| format!("suite Gpu failed: {error}"))?;
    let root = workspace_root()?;
    println!("-- name the adapter");
    println!(
        "   {}",
        name_adapter(&root).map_err(|error| format!("suite Gpu failed: {error}"))?
    );
    let mut announce = |call: &str| println!("-- {call}");
    run_gpu_profile_suite(
        &root,
        profile,
        &run_cargo_test_captured_under,
        &mut announce,
    )
    .map_err(|error| format!("suite Gpu failed: {error}"))?;
    println!(
        "Suite Gpu passed under GUST_PROFILE={profile} for the pinned gust-wgpu targets and gust-algo. It does not cover the other gust-wgpu targets, the differential batches or the whole-suite count (GPU_SUITE_PIN is for dev)."
    );
    Ok(())
}

/// The steps of the profile form: `slangc` on `PATH` (the tests compile kernels at run time), then the target.
fn run_profile_target(profile: &str, target: &str) -> Result<(), String> {
    println!("== suite Gpu, target {target} under GUST_PROFILE={profile}");
    println!("-- require slangc on PATH");
    require_command("slangc").map_err(|error| format!("suite Gpu failed: {error}"))?;
    let root = workspace_root()?;
    let label = format!("cargo test --locked -p gust-compute-wgpu --test {target}");
    println!("-- {label} (GUST_PROFILE={profile})");
    run_gpu_target(&root, profile, target, &run_cargo_test_captured_under)
        .map_err(|error| format!("suite Gpu failed at `{label}`: {error}"))?;
    println!(
        "Suite Gpu passed for the target {target} under GUST_PROFILE={profile}. It does not cover the other targets or the whole-suite count (GPU_SUITE_PIN is for dev)."
    );
    Ok(())
}

/// The steps a run executes and the steps `--skip-difftest` leaves out (those of [`DIFFTEST_UNIT_STEPS`]). The flag on
/// a suite that has none of them is an error, so it can never be a silent no-op.
pub(crate) fn plan_suite(
    suite: Suite,
    skip_difftest: bool,
) -> Result<(Vec<Step>, Vec<Step>), String> {
    let steps = suite.steps();
    if !skip_difftest {
        return Ok((steps, Vec::new()));
    }
    let (skipped, run): (Vec<Step>, Vec<Step>) = steps
        .into_iter()
        .partition(|step| DIFFTEST_UNIT_STEPS.contains(step));
    if skipped.is_empty() {
        return Err(format!(
            "--skip-difftest: suite {} has no differential-test step to skip (only Unit and Slangc have)",
            suite.name()
        ));
    }
    Ok((run, skipped))
}

fn run_suite(suite: Suite, skip_difftest: bool) -> Result<(), String> {
    let name = suite.name();
    let (steps, skipped) = plan_suite(suite, skip_difftest)?;
    println!("== suite {name}");
    println!("   covers: {}", suite.covers());
    println!("   does not cover: {}", suite.not_covered());
    for step in &skipped {
        println!(
            "-- SKIPPED by --skip-difftest (left to the CI job `difftest`): {}",
            step.label()
        );
    }
    for step in steps {
        println!("-- {}", step.label());
        step.run()
            .map_err(|error| format!("suite {name} failed at `{}`: {error}", step.label()))?;
    }
    if skipped.is_empty() {
        println!(
            "Suite {name} passed. It does not cover {}.",
            suite.not_covered()
        );
    } else {
        println!(
            "Suite {name} passed with {} step(s) skipped by --skip-difftest, which this run does not cover. \
             It does not cover {}.",
            skipped.len(),
            suite.not_covered()
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/suite.rs"]
mod suite_tests;
