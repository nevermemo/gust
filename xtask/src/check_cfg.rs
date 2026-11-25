//! `cargo xtask check-cfg` (m13.gates-rt): lint `gust-compute` and `gust-compute-wgpu` on their own, one feature set at a
//! time, and for `wasm32-unknown-unknown`.
//!
//! A workspace build unifies features: the self dev-dependencies and `gust-algo` switch `slang-runtime` on for both
//! crates, and nothing in the workspace switches `spirv-route` on, so the default build, the `spirv-route` builds and
//! the browser build of the runtime are compiled by no workspace-wide build (CI's wasm32 clippy and the precompiled job
//! build some of them); `check-fast` and `check-full` run this check. `cargo clippy ... -- -D warnings` also fails on
//! an unused import, which is the usual error of a build that a feature or a target gates off.

use std::{
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};

use crate::{
    process::{cargo_program, workspace_command},
    web::{WASM_TARGET, wasm_target_installed},
    workspace::workspace_root,
};

/// The flag that turns a missing wasm32 standard library from a skip into a failure (the CI job installs it).
const REQUIRE_WASM: &str = "--require-wasm";

/// One lint run: a name for the report, and the arguments cargo gets between `clippy --locked` and `-- -D warnings`.
pub(crate) struct Combo {
    pub(crate) name: &'static str,
    /// Built for `wasm32-unknown-unknown` under `GUST_PROFILE=web`, the strict profile CI lints it with.
    pub(crate) wasm: bool,
    pub(crate) args: &'static [&'static str],
}

/// The runs, in order. `--lib` where the self dev-dependency would otherwise switch `slang-runtime` on (it applies to
/// builds with tests, examples and benches), so that the library is seen with exactly the features named; the sets that
/// have the feature on anyway, or whose tests need it, use `--all-targets`.
pub(crate) const COMBOS: &[Combo] = &[
    Combo {
        name: "gust-compute: no features",
        wasm: false,
        args: &["-p", "gust-compute", "--lib"],
    },
    Combo {
        name: "gust-compute: spirv-route",
        wasm: false,
        args: &["-p", "gust-compute", "--lib", "--features", "spirv-route"],
    },
    // `gust-algo` forwards `slang-runtime`; its own dev-dependency switches the feature on for builds with tests only.
    Combo {
        name: "gust-algo: no features",
        wasm: false,
        args: &["-p", "gust-algo", "--lib"],
    },
    Combo {
        name: "gust-compute-wgpu: no features",
        wasm: false,
        args: &["-p", "gust-compute-wgpu", "--lib"],
    },
    Combo {
        name: "gust-compute-wgpu: slang-runtime",
        wasm: false,
        args: &[
            "-p",
            "gust-compute-wgpu",
            "--all-targets",
            "--features",
            "slang-runtime",
        ],
    },
    Combo {
        name: "gust-compute-wgpu: slang-runtime + spirv-route",
        wasm: false,
        args: &[
            "-p",
            "gust-compute-wgpu",
            "--all-targets",
            "--features",
            "slang-runtime,spirv-route",
        ],
    },
    Combo {
        name: "gust-compute-wgpu: spirv-route alone",
        wasm: false,
        args: &[
            "-p",
            "gust-compute-wgpu",
            "--lib",
            "--features",
            "spirv-route",
        ],
    },
    Combo {
        name: "wasm32: gust-compute, gust-compute-wgpu, web-vector-add, web-batch-frames",
        wasm: true,
        args: &[
            "--target",
            WASM_TARGET,
            "-p",
            "gust-compute",
            "-p",
            "gust-compute-wgpu",
            "-p",
            "web-vector-add",
            "-p",
            "web-batch-frames",
        ],
    },
    Combo {
        name: "wasm32: gust-compute-wgpu with spirv-route",
        wasm: true,
        args: &[
            "--target",
            WASM_TARGET,
            "-p",
            "gust-compute-wgpu",
            "--features",
            "spirv-route",
        ],
    },
];

/// What a run did with one combination.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Passed,
    Skipped,
}

/// The cargo arguments of `combo`.
pub(crate) fn cargo_args(combo: &Combo) -> Vec<String> {
    ["clippy", "--locked"]
        .into_iter()
        .chain(combo.args.iter().copied())
        .chain(["--", "-D", "warnings"])
        .map(str::to_owned)
        .collect()
}

/// Runs every combination through `run`, in order, and stops at the first failure, which names the combination. A wasm32
/// combination is not run, and is reported as [`Outcome::Skipped`], when the target is not installed; with
/// `require_wasm` that is an error instead, raised before any run starts, so a job that must have the target cannot pass without it.
pub(crate) fn run_combos(
    combos: &[Combo],
    wasm_installed: bool,
    require_wasm: bool,
    run: &mut dyn FnMut(&Combo) -> Result<(), String>,
) -> Result<Vec<(&'static str, Outcome)>, String> {
    if require_wasm && !wasm_installed && combos.iter().any(|combo| combo.wasm) {
        return Err(format!(
            "check-cfg: the {WASM_TARGET} standard library is not installed ({REQUIRE_WASM} was given); run `rustup target add {WASM_TARGET}`"
        ));
    }
    let mut outcomes = Vec::new();
    for combo in combos {
        if combo.wasm && !wasm_installed {
            println!(
                "check-cfg: SKIPPED {}: the {WASM_TARGET} standard library is not installed; run `rustup target add {WASM_TARGET}`",
                combo.name
            );
            outcomes.push((combo.name, Outcome::Skipped));
            continue;
        }
        run(combo).map_err(|error| format!("check-cfg: {} failed: {error}", combo.name))?;
        outcomes.push((combo.name, Outcome::Passed));
    }
    Ok(outcomes)
}

/// The closing line: the passed and the skipped counts, and the names of what was skipped.
pub(crate) fn summary(outcomes: &[(&'static str, Outcome)]) -> String {
    let passed = outcomes
        .iter()
        .filter(|(_, outcome)| *outcome == Outcome::Passed)
        .count();
    let skipped: Vec<&str> = outcomes
        .iter()
        .filter(|(_, outcome)| *outcome == Outcome::Skipped)
        .map(|(name, _)| *name)
        .collect();
    if skipped.is_empty() {
        format!(
            "check-cfg passed: {passed} of {} builds linted",
            outcomes.len()
        )
    } else {
        format!(
            "check-cfg: {passed} builds linted, {} SKIPPED (not passed): {}",
            skipped.len(),
            skipped.join("; ")
        )
    }
}

pub(crate) fn check_cfg(args: &[String]) -> Result<(), String> {
    let require_wasm = match args {
        [] => false,
        [flag] if flag == REQUIRE_WASM => true,
        _ => return Err(format!("check-cfg takes no argument or {REQUIRE_WASM}")),
    };
    let report = check_cfg_report(&workspace_root()?, require_wasm)?;
    println!("{}", report.line);
    Ok(())
}

/// What a whole run did: the closing line, and the names of the runs that were skipped (not passed).
pub(crate) struct Report {
    pub(crate) line: String,
    pub(crate) skipped: Vec<&'static str>,
}

/// The whole check, as the `check-fast` and `check-full` modes run it (m13.sfix2), in the repository `root`. It prints
/// each run's progress but not the closing line (the caller prints or records it). With `require_wasm` a missing wasm32
/// standard library is an error, as in CI; without it the wasm32 runs are skipped and named in the report.
pub(crate) fn check_cfg_report(root: &Path, require_wasm: bool) -> Result<Report, String> {
    let installed = wasm_target_installed("check-cfg")?;
    let started = Instant::now();
    let outcomes = run_combos(COMBOS, installed, require_wasm, &mut |combo| {
        let step = Instant::now();
        println!("check-cfg: {}", combo.name);
        let result = lint(root, combo);
        println!("check-cfg: {} took {:.0?}", combo.name, step.elapsed());
        result
    })?;
    Ok(Report {
        line: format!("{} ({:.0?})", summary(&outcomes), started.elapsed()),
        skipped: outcomes
            .iter()
            .filter(|(_, outcome)| *outcome == Outcome::Skipped)
            .map(|(name, _)| *name)
            .collect(),
    })
}

/// The cargo child of `combo`: the same cargo program for every run, with the workspace profile (`GUST_PROFILE=dev`)
/// that [`workspace_command`] sets, replaced by `web` for a wasm32 run, the strict profile CI lints the browser build
/// with.
pub(crate) fn command(root: &Path, combo: &Combo) -> Command {
    let mut command = workspace_command(cargo_program());
    command.args(cargo_args(combo)).current_dir(root);
    if combo.wasm {
        command.env("GUST_PROFILE", "web");
    }
    command
}

fn lint(root: &Path, combo: &Combo) -> Result<(), String> {
    let status = command(root, combo)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo {} failed (exit {})",
            cargo_args(combo).join(" "),
            status.code().unwrap_or(-1)
        ))
    }
}

#[cfg(test)]
#[path = "tests/check_cfg.rs"]
mod check_cfg_tests;
