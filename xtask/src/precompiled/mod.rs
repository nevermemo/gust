//! `cargo xtask check-precompiled [--check-only] [--work-dir PATH]`: the evidence that a production-shaped consumer of
//! precompiled kernels needs no Slang (D45, D46). The fixture is `tests/precompiled`, a workspace of its own like
//! `tests/dual`: the `kernels` crate holds one `#[gpu_module]` vector-add and its committed `gust-artifacts/` (written
//! by `cargo gust build`), and `app` dispatches it through `gust-wgpu` with `default-features = false` and checks the
//! result.
//!
//! Every cargo child runs under `GUST_PROFILE=native`, the strict profile, set explicitly (R4.2), except the control
//! build of E2, which needs the bridge and runs under `dev` in its own target directory, the profile checks of E6
//! (their point is the profile), the dual steps of E5 (they run through `check-dual`'s own command, which sets the
//! profile from the step) and `cargo gust build` of E7, an author tool that needs Slang. The checks:
//!
//! - **E1**: release build and run of `app` in an environment with Slang taken out. The child gets a copy of the
//!   parent's environment without every variable that names Slang (any name containing `SLANG`, `VULKAN_SDK`,
//!   `VK_SDK_PATH`) and without every `PATH` directory that holds `slangc` or `gust-slang-reflect`. The environment
//!   is then asserted slang-free: no such variable, no such directory, and the platform's lookup (`where` under `cmd`,
//!   `command -v` under `sh`) does not find `slangc` with the child's `PATH`. The build starts from `cargo clean -p gust-compute
//!   -p gust-compute-macros -p kernels -p app --release` in the same environment, because cargo's fingerprints track only
//!   `GUST_PROFILE`, so a warm target directory would relink nothing and prove nothing about the stripped build. `app`
//!   must print the `verified:` line, which it prints only after it read the GPU result back and compared it with a
//!   host reference, and the line `adapter:` that names the GPU it ran on.
//! - **E2**: the release binary of E1 holds none of the strings that only the run-time Slang bridge contains
//!   ([`BRIDGE_MARKERS`]), `cargo tree -e features` of `app` shows no `slang-runtime`, and the fixture's manifests keep
//!   `default-features = false` on `gust` and `gust-wgpu`. A control keeps the scan from passing vacuously: the same
//!   `app` is built under `GUST_PROFILE=dev` with `--features gust-wgpu/slang-runtime` into its own target directory
//!   (`<target>/precompiled-control`, built, never run), and its binary must hold every marker.
//! - **E3**: in a scratch copy of the fixture under the target directory, one operator of the committed WGSL is
//!   changed (`+` to `-`, so that an unchecked artifact would compute a wrong answer). It builds, and at run time
//!   `app` ends with the artifact-identity error of D17 and prints no `verified:` line.
//! - **E4** (E6f of the plan): in another scratch copy, the artifact files are deleted. Under `native` the build fails
//!   with the one error of `gust::profile` (E0080), reported at the line of the kernel in `kernels/src/lib.rs`, which
//!   `tests/precompiled/e4-expected.txt` pins.
//! - **E5**: strict plus dual (R4.2). The fixture is built through `check-dual`'s own command
//!   ([`crate::checks::precompiled_dual_steps`]) with `--cfg gust_dual` and `GUST_PROFILE=native` together; `app`'s unit
//!   test dispatches the CPU twin of the fixture's kernel and compares it with the host reference, and `cargo tree -e
//!   features` shows no `slang-runtime`. `check-dual` runs the same steps.
//! - **E6a to E6e**: the refusals of `gust`'s build script, each judged on its exact message (the `build-error:` lines
//!   of `e6-native-expected.txt` and `e6-inferred-expected.txt`, judged by [`crate::ui::judge`]) and never on the exit
//!   status alone: a `native` build with `gust/slang-runtime` (E6a), with `gust-wgpu/slang-runtime` (E6b, feature
//!   unification), and with a dependency crate that enables the feature (E6c, in a copy, also in a release build with
//!   no variable). E6d (a release build with no variable and the artifacts present) and E6e (a release build under
//!   `GUST_PROFILE=dev` with the feature) are the positive controls and must compile.
//! - **E6g to E6j**: the internal feature `cargo-gust-bridge` of `gust` (D108, mr.c). With it, a release build without
//!   `GUST_PROFILE` is `dev` and prints the warning of `e6-bridge-expected.txt` (E6g: `app` with
//!   `gust/cargo-gust-bridge` and `gust/slang-runtime`, which the plain inference of E6c-release refuses); an explicit
//!   `GUST_PROFILE=native` still refuses the bridge (E6h, the text of `e6-native-expected.txt`); and (R1 S8) the
//!   release check of the whole repository workspace without the variable unifies the feature through `cargo-gust`,
//!   so it is `dev` and prints the same warning (E6i, `--workspace` where the other cases check `-p app`; it is a
//!   `cargo check` because the build script that the case judges runs in a check as in a build). E6j (D117): in a
//!   copy, `app` has `cargo-gust` as a normal dependency, a mistake that cargo warns about and ignores
//!   (`cargo-warning:` in the expectation) but that still unifies the feature, so the release build is `dev` with the
//!   same warning. Each warning is matched as a whole line.
//! - **E7**: a change of `GUST_EXPORT_DIR`, which the macro tracks through `option_env!`, makes cargo expand the
//!   `#[gpu_module]` module of `kernels` again (and the macro writes its records), and the same value does not. Then,
//!   with Slang, `cargo gust build` runs twice on a copy of the fixture whose artifacts were deleted first: run 1 writes them, run 2 (the run on unchanged input) writes nothing, and the output is byte-identical and the
//!   tool prints no `pruned` line (R2-3, S3).
//! - **E8**: distribution (R2-4, R4.6). `cargo package` of `gust-compute-macros`, `gust-compute`, `gust-compute-wgpu` and `kernels` must list
//!   only allowed files; the four packages are unpacked outside the repository and a consumer with plain registry-style
//!   dependencies and a `[patch.crates-io]` on the four trees is built under `native` in the stripped environment of E1
//!   and, unless `--check-only`, run on the GPU. The packaging of `kernels` runs with a `[patch.crates-io]` of its own,
//!   given with `--config`, for `gust-compute` and `gust-compute-macros` onto this checkout, so that its lock resolves
//!   these trees and not whatever crates.io holds under a GUST name (mr.n: before the rename it was the unrelated
//!   charting crate `gust`).
//!
//! `--check-only` skips the parts that need an adapter (the runs of E1, E3 and E8) and Slang (the second half of E7);
//! it is what the CI job `precompiled` runs. Nothing but `cargo gust build` of E7 runs `slangc` or the native helper.
//! The environment of the xtask process is never changed; the stripped environment exists only in the children.
//! `CARGO_NET_OFFLINE` and the rest of the parent's environment pass through. E8 needs the registry (the consumer
//! needs `wgpu` from crates.io), so it is no offline proof.

use std::path::PathBuf;

use crate::{process::PROFILE_VARIABLE, workspace::workspace_root};

mod artifacts;
mod distribution;
mod environment;
mod fixture;
mod profiles;
mod reexpand;
mod refusals;
mod stripped;
#[cfg(test)]
#[path = "../tests/precompiled_support.rs"]
mod test_support;

use self::{
    artifacts::{e3_mutated_artifact, e4_missing_artifact, e5_strict_plus_dual},
    distribution::e8_distribution,
    fixture::Fixture,
    reexpand::{e7_build_twice, e7_reexpansion},
    refusals::e6_refusals,
    stripped::{e1_stripped_build_and_run, e2_binary_and_features},
};
pub(crate) use self::{
    distribution::{is_inside, lock_packages, unused_patch_lines},
    environment::{StripReport, child_environment, directory_has_slang_tool, slang_free_problems},
    reexpand::snapshot,
    stripped::{count_occurrences, tree_problems},
};

/// The fixture workspace, relative to the repository root.
pub(crate) const PRECOMPILED_DIR: &str = "tests/precompiled";

/// The profile of every child: strict, so that a missing artifact is a build error and the bridge is refused.
const STRICT_PROFILE: &str = "native";

/// The profile of the control build of E2, the only child that is not strict: the bridge needs a development profile.
const CONTROL_PROFILE: &str = "dev";

/// What `cargo xtask check-precompiled` was asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    /// `--check-only`: skip what needs an adapter (the runs of E1, E3 and E8) and Slang (the second half of E7).
    pub(crate) check_only: bool,
    /// `--work-dir PATH`: where E8 builds its consumer; a directory that does not exist yet or is empty.
    pub(crate) work_dir: Option<PathBuf>,
}

/// The options of `check-precompiled`.
pub(crate) fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut rest = args.iter();
    while let Some(argument) = rest.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (argument.as_str(), None),
        };
        match name {
            "--check-only" => {
                if inline.is_some() {
                    return Err("`--check-only` takes no value".into());
                }
                options.check_only = true;
            }
            "--work-dir" => {
                let value = match inline {
                    Some(value) => value,
                    None => rest
                        .next()
                        .cloned()
                        .ok_or("`--work-dir` needs a directory")?,
                };
                if value.is_empty() {
                    return Err("`--work-dir` needs a non-empty directory".into());
                }
                if options.work_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("`--work-dir` was given twice".into());
                }
            }
            _ => {
                return Err(format!(
                    "check-precompiled takes only `--check-only` and `--work-dir PATH`, not `{argument}`"
                ));
            }
        }
    }
    Ok(options)
}

/// `cargo xtask check-precompiled [--check-only] [--work-dir PATH]`.
pub(crate) fn check_precompiled(args: &[String]) -> Result<(), String> {
    println!("{}", check_precompiled_report(args)?);
    Ok(())
}

/// [`check_precompiled`] returning the pass summary (the E steps and the adapter they ran on) instead of printing
/// it, so `verify --mode full` can record it.
pub(crate) fn check_precompiled_report(args: &[String]) -> Result<String, String> {
    let options = parse_args(args)?;
    let root = workspace_root()?;
    let fixture = Fixture::new(&root);
    println!(
        "check-precompiled: cargo target directory {}{}",
        fixture.shown(&fixture.target_dir),
        if options.check_only {
            " (--check-only: no adapter, no Slang)"
        } else {
            ""
        }
    );

    let report = e1_stripped_build_and_run(&fixture, options.check_only)?;
    let binary = e2_binary_and_features(&fixture)?;
    let identity = e3_mutated_artifact(&fixture, options.check_only)?;
    let located = e4_missing_artifact(&fixture)?;
    let strict_dual = e5_strict_plus_dual(&fixture)?;
    let refusals = e6_refusals(&fixture)?;
    let reexpanded = e7_reexpansion(&fixture)?;
    let reproducible = e7_build_twice(&fixture, options.check_only)?;
    let distribution = e8_distribution(&fixture, &options)?;

    Ok(format!(
        "check-precompiled passed: E1 {report}. E2 {binary}. E3 {identity}. E4 (E6f) {located}. E5 {strict_dual}. E6a to E6o {refusals}. E7 {reexpanded}. E7 {reproducible}. E8 {distribution}. The fixture is tests/precompiled; every child ran under {PROFILE_VARIABLE}={STRICT_PROFILE} except the E2 control and E6e ({PROFILE_VARIABLE}={CONTROL_PROFILE}), E6k to E6o (each under its own value, `native+subgroups` to `native+f16`), the profile-free builds of E6c, E6d, E6g, E6i and E6j, and cargo gust of E7."
    ))
}
