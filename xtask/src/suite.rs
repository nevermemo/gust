use crate::{
    arch::check_arch,
    checks::{check_cpu, check_feature},
    cpu_target::check_cpu_target,
    difftest::check_difftest,
    docs::docs_check,
    lint::lint,
    process::{TestTotals, require_command, run_cargo_test_captured, run_counted, run_direct},
    ui::check_ui,
    verify::{GPU_SUITE_CALL, GPU_SUITE_PIN, VerifyMode, check_gpu_suite_counts, verify},
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

/// Runs a `cargo test` call in a directory and returns its totals and captured standard output
/// ([`run_cargo_test_captured`]).
type TestRunner<'a> = dyn Fn(&Path, &[String]) -> Result<(TestTotals, String), String> + 'a;

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

const MACROS_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "gust-macros"]);
const XTASK_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "xtask"]);
const CORE_LIB_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "gust", "--lib"]);
/// The `gust::cpu` runtime tests (the CPU twin's device, `ulps`, call checks). They need neither slangc nor a GPU
/// (the `gust` crate has no build script and no GPU dependency), but share the library test binary with tests that
/// do need slangc, so the name filter carves them out for the Unit suite and CI.
const CORE_CPU_TESTS: Step =
    Step::CountedCargo(&["test", "--locked", "-p", "gust", "--lib", "cpu::tests::"]);
/// The `gust::artifact` tests (`verify_artifact` on the committed precompiled vector-add fixture, the identity and layout
/// negatives, `content_key` vectors). Pure: no slangc, no process, no GPU. The integration test file is whole; the
/// library tests are carved out by name and counted, as `cpu::tests::` is.
const CORE_ARTIFACT_TESTS: Step =
    Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "artifact"]);
/// The scalar coretests transcription on the CPU twin (m5.d3): every case of docs/development/CORETESTS.md against its
/// literal, with the id guards. Pure: no slangc, no process, no GPU (the GPU half is `check-feature gpu-semantics`).
const CORE_CORETESTS_TWIN: Step =
    Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "coretests_twin"]);
/// The off state of the `slang-runtime` feature (D44). The workspace build unifies the feature on (the examples and the
/// self dev-dependencies ask for it), so these two calls, which build `gust` and `gust-wgpu` alone and without default
/// features, are the only observers of the bridge's cfg attributes: they fail when something the precompiled path needs
/// slips behind the feature, or when something behind the feature leaks into the off state. They need cargo only.
const OFF_STATE_CLIPPY: Step = Step::Cargo(&[
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
]);
const OFF_STATE_CHECK: Step = Step::Cargo(&[
    "check",
    "--locked",
    "-p",
    "gust",
    "-p",
    "gust-wgpu",
    "--no-default-features",
    "--lib",
]);
const CORE_ARTIFACT_UNIT_TESTS: Step = Step::CountedCargo(&[
    "test",
    "--locked",
    "-p",
    "gust",
    "--lib",
    "artifact::tests::",
]);
const CORE_DIAGNOSTICS_TESTS: Step =
    Step::Cargo(&["test", "--locked", "-p", "gust", "--test", "diagnostics"]);
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
                "macro validation and translation tests, xtask tests, lint, docs-check, check-arch, check-ui, check-cpu (the CPU twins of the examples against their host references), the gust::cpu unit tests, the gust::artifact tests (verify_artifact on a real precompiled fixture), the coretests_twin test (the scalar Rust core test cases on the CPU twin against their literals), clippy and check of gust and gust-wgpu without default features (the slang-runtime off state), the format check, clippy and tests of the harness crate tests/difftest (a workspace of its own; CI runs them in the job `difftest`, and `--skip-difftest` leaves them out of the `unit` and `slang` jobs), and the differential batch of seeds 1 to 4 in its CPU-only mode (64 generated kernels: an independent interpreter against the CPU twin, report.json judged by cargo xtask check-difftest); no external tools"
            }
            Self::Cpu => {
                "the production CPU target (GUST_PROFILE=cpu) on the fixture tests/cpu-target: the kernels build in release, run on gust::cpu::CpuDevice and equal independent host references; the binary holds no compiler, helper or GPU runtime; gust-wgpu refuses the profile; the cfg(test) and --cfg gust_dual builds keep one impl; the profile flips rebuild correctly; cargo only"
            }
            Self::Slangc => {
                "Unit, plus the gust library tests and the diagnostics integration test, which compile generated Slang with slangc"
            }
            Self::Spirv => {
                "the example binaries export SPIR-V and spirv-val accepts every file (vulkan1.2)"
            }
            Self::Reflect => {
                "the D17 layout gate: native reflection helper, core and wgpu reflection tests"
            }
            Self::Gpu => {
                "all gust-wgpu tests on the first Vulkan adapter wgpu finds (their `test result:` lines checked against GPU_SUITE_PIN, as in verify --mode gpu), then the differential batch of all 8 seeds (128 generated kernels: interpreter, CPU twin and GPU compared; report.json judged by cargo xtask check-difftest) and the short_circuit GPU test of tests/difftest"
            }
            Self::Full => {
                "the release gate: lint, fmt, clippy, workspace tests, the fmt, clippy and tests of the harness crate tests/difftest, the release-profile Slang tests of the gust library and the release-profile known-mismatch ledger, examples, artifacts and SPIR-V validation, check-dual in full mode, then the whole check-precompiled, the whole check-web (Chrome with a WebGPU adapter; a missing one fails the suite, only `cargo xtask check-full --skip-web` goes without, and records that) and, last, check-difftest (all 8 seeds on the adapter and the short_circuit GPU test); writes .ai/VALIDATION.json"
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
                "SPIR-V validation or lint; the device is whatever wgpu picks, so name it in the run record"
            }
            Self::Full => {
                "nothing named here is skipped, but a software adapter is only Gpu (software)"
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
                steps
            }
            Self::Spirv => vec![Step::Verify(VerifyMode::Artifacts, false)],
            Self::Reflect => vec![Step::Feature("reflection")],
            Self::Gpu => vec![Step::RequireTool("slangc"), Step::GpuSuite, DIFFTEST_GPU],
            Self::Full => vec![Step::Verify(VerifyMode::Full, true)],
        }
    }
}

impl Step {
    fn label(self) -> String {
        match self {
            Self::Cargo(args) | Self::CountedCargo(args) => format!("cargo {}", args.join(" ")),
            Self::GpuSuite => format!("cargo {}", GPU_SUITE_CALL.join(" ")),
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
        self.run_in(&root, &run_cargo_test_captured)
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
                let (_, stdout) = test_runner(&root, &call)?;
                check_gpu_suite_counts(&call, &stdout, GPU_SUITE_PIN)
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

/// `cargo xtask test --suite <Unit|Cpu|Slangc|Spirv|Reflect|Gpu|Full> [--skip-difftest]`.
pub(crate) fn test_command(args: &[String]) -> Result<(), String> {
    let usage = || {
        format!(
            "usage: cargo xtask test --suite {} [--skip-difftest]",
            Suite::names().join("|")
        )
    };
    let (suite, skip_difftest) = match args {
        [flag, name] if flag == "--suite" => (Suite::parse(name)?, false),
        [flag, name, skip] if flag == "--suite" && skip == "--skip-difftest" => {
            (Suite::parse(name)?, true)
        }
        _ => return Err(usage()),
    };
    run_suite(suite, skip_difftest)
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
