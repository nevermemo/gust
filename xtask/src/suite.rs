use crate::{
    arch::check_arch,
    checks::{check_cpu, check_feature},
    docs::docs_check,
    lint::lint,
    process::{require_command, run_counted, run_direct},
    ui::check_ui,
    verify::{VerifyMode, verify},
    workspace::workspace_root,
};

/// An evidence tier of docs/development/VALIDATION.md, runnable as one command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Suite {
    Unit,
    Slangc,
    Spirv,
    Reflect,
    Gpu,
    Full,
}

/// One step of a suite. Every step is an existing command or a `cargo test` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// `cargo <args>` in the workspace root.
    Cargo(&'static [&'static str]),
    /// `cargo test <filter>` in the workspace root that must run at least one test (a filter that
    /// matches nothing exits 0 in cargo).
    CountedCargo(&'static [&'static str]),
    /// A tool on `PATH`; a missing tool fails the suite by name.
    RequireTool(&'static str),
    Lint,
    DocsCheck,
    CheckArch,
    /// `cargo xtask check-ui`: `cargo check` of the fixture crates under tests/ui.
    CheckUi,
    /// `cargo xtask check-cpu`: the examples' CPU-twin tests (`checks::CPU_TESTS`).
    CheckCpu,
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
const WGPU_TESTS: Step = Step::Cargo(&["test", "--locked", "-p", "gust-wgpu"]);

const ALL: [Suite; 6] = [
    Suite::Unit,
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
                "macro validation and translation tests, xtask tests, lint, docs-check, check-arch, check-ui, check-cpu (the CPU twins of the examples against their host references), the gust::cpu unit tests, the gust::artifact tests (verify_artifact on a real precompiled fixture); no external tools"
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
            Self::Gpu => "all gust-wgpu tests on the first Vulkan adapter wgpu finds",
            Self::Full => {
                "the release gate: lint, fmt, clippy, workspace tests, the release-profile Slang tests of the gust library and the release-profile known-mismatch ledger, examples, artifacts and SPIR-V validation; writes .ai/VALIDATION.json"
            }
        }
    }

    /// What a pass of this suite does not show.
    pub(crate) fn not_covered(self) -> &'static str {
        match self {
            Self::Unit => {
                "that generated Slang compiles, SPIR-V validity, reflection layouts, or any GPU execution"
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
        ];
        match self {
            Self::Unit => unit.to_vec(),
            Self::Slangc => {
                let mut steps = vec![Step::RequireTool("slangc")];
                steps.extend(unit);
                steps.push(CORE_LIB_TESTS);
                steps.push(CORE_DIAGNOSTICS_TESTS);
                steps
            }
            Self::Spirv => vec![Step::Verify(VerifyMode::Artifacts, false)],
            Self::Reflect => vec![Step::Feature("reflection")],
            Self::Gpu => vec![Step::RequireTool("slangc"), WGPU_TESTS],
            Self::Full => vec![Step::Verify(VerifyMode::Full, true)],
        }
    }
}

impl Step {
    fn label(self) -> String {
        match self {
            Self::Cargo(args) | Self::CountedCargo(args) => format!("cargo {}", args.join(" ")),
            Self::RequireTool(tool) => format!("require {tool} on PATH"),
            Self::Lint => "cargo xtask lint".to_owned(),
            Self::DocsCheck => "cargo xtask docs-check".to_owned(),
            Self::CheckArch => "cargo xtask check-arch".to_owned(),
            Self::CheckUi => "cargo xtask check-ui".to_owned(),
            Self::CheckCpu => "cargo xtask check-cpu".to_owned(),
            Self::Feature(area) => format!("cargo xtask check-feature {area}"),
            Self::Verify(mode, _) => format!("cargo xtask verify --mode {}", mode.as_str()),
        }
    }

    fn run(self) -> Result<(), String> {
        let root = workspace_root()?;
        match self {
            Self::Cargo(args) => run_direct(&root, "cargo", args),
            Self::CountedCargo(args) => run_counted(&root, args).map(|_| ()),
            Self::RequireTool(tool) => require_command(tool),
            Self::Lint => lint(&root, &mut Vec::new()),
            Self::DocsCheck => docs_check(&root, &mut Vec::new()),
            Self::CheckArch => check_arch(&root, &mut Vec::new()),
            Self::CheckUi => check_ui(&root, &mut Vec::new()),
            Self::CheckCpu => check_cpu(),
            Self::Feature(area) => check_feature(Some(area)),
            Self::Verify(mode, record) => verify(mode, record),
        }
    }
}

/// `cargo xtask test --suite <Unit|Slangc|Spirv|Reflect|Gpu|Full>`.
pub(crate) fn test_command(args: &[String]) -> Result<(), String> {
    let suite = match args {
        [flag, name] if flag == "--suite" => Suite::parse(name)?,
        _ => {
            return Err(format!(
                "usage: cargo xtask test --suite {}",
                Suite::names().join("|")
            ));
        }
    };
    run_suite(suite)
}

fn run_suite(suite: Suite) -> Result<(), String> {
    let name = suite.name();
    println!("== suite {name}");
    println!("   covers: {}", suite.covers());
    println!("   does not cover: {}", suite.not_covered());
    for step in suite.steps() {
        println!("-- {}", step.label());
        step.run()
            .map_err(|error| format!("suite {name} failed at `{}`: {error}", step.label()))?;
    }
    println!(
        "Suite {name} passed. It does not cover {}.",
        suite.not_covered()
    );
    Ok(())
}

#[cfg(test)]
#[path = "tests/suite.rs"]
mod suite_tests;
