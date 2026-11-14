//! `cargo xtask check-difftest`: the gate around the differential test harness `tests/difftest` (plan M5, R1-13).
//!
//! The harness is `difftest-run`; this command does not reimplement any of it. It runs the binary for the requested
//! seeds, reads the `report.json` it wrote, and fails unless the report says what was asked for: the mode, the seeds,
//! `programs_executed` equal to `16 x seeds`, no mismatch, and (on the GPU tier) an adapter. On the GPU tier it also
//! runs the `short_circuit` test of the harness crate with `--ignored`, the one test no other gate runs.
//!
//! The GPU tier ends with a second batch of the same seeds under `GUST_PROFILE=dev+subgroups` (M8, slice m8.e): the
//! run crate is built and run with the `subgroups` capability, its report must pass and say that profile, and the
//! records the macro exports for it (`GUST_EXPORT_DIR`, a new directory for each batch) must equal, file for file and
//! byte for byte, the records of the `dev` batch. The corpus has no kernel with a variant, so the capability may not
//! change a byte the compiler writes. A kernel whose variant the capability does select is not in the corpus; the
//! `gust-algo` evidence tests and `examples/subgroup-reduce` cover that case.
//!
//! ```text
//! cargo xtask check-difftest [--seeds <list|range|random:N>] [--shapes grid,pixel] [--cpu-only] [--report PATH]
//! ```
//!
//! `--shapes grid,pixel` (slice mx.p1c) has the harness give every generated program a second kernel, its pixel form,
//! and compare it with the grid run: bit for bit for integers and booleans, and for `f32` through each shape's interval,
//! with the bit differences between the shapes only observed (D128). Without `--shapes` the command is the grid-only run
//! it always was. The
//! Full mode of `verify` (`check-full`) reaches this file through [`check_difftest_report`], which adds
//! `--shapes grid,pixel` when the caller named no shapes; the command line and the Gpu suite do not, and the
//! `gpu-software` workflow names the shapes in one step of its own, seeds 1-2 on both lists (slice mx.z; tests pin all
//! three). The report must say which programs ran as pixel and which stayed grid-only, with the
//! classifier's reason, and at least one program must have run as pixel.
//!
//! The default is all eight seeds of `tests/difftest/seeds.txt` on the machine's adapter. `--cpu-only` compares the
//! interpreter with the CPU twin only (the Unit tier: no adapter, no Slang) and never runs the GPU test. `random:N`
//! draws N seeds from the clock, prints them, and passes them on as a list, so a failure can be reproduced with
//! `--seeds <list>`. The report is compared by fields, never by bytes: it is deterministic except for `timing` and
//! `adapter`.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    checks::tests_not_ok,
    json::{self, Json},
    process::{require_tests_ran, sum_test_results, workspace_command},
    workspace::{cargo_target_dir, workspace_root},
};

/// Setting this to `1` runs the batches and the GPU test with the SPIR-V route (slice m9.rt2, D172): the generated crates
/// depend on `gust-wgpu` with a fixed feature list that no Cargo flag can extend from here, so the route is delivered the
/// way `check-dual` delivers its cfg, through `CARGO_ENCODED_RUSTFLAGS`, as `--cfg feature="spirv-route"` for every crate of
/// the build (the macro and `gust-wgpu` both read `cfg!(feature = "spirv-route")`). `GUST_SPIRV_ROUTE_LOG` (a file) then
/// records the route of each kernel.
pub(crate) const ROUTE_VARIABLE: &str = "GUST_SPIRV_ROUTE";

/// The `CARGO_ENCODED_RUSTFLAGS` of a route run: two flags joined by the unit separator.
pub(crate) const ROUTE_RUSTFLAGS: &str = "--cfg\u{1f}feature=\"spirv-route\"";

/// The environment variables a cargo call of this command gets for the route: none unless [`ROUTE_VARIABLE`] is `1`. It
/// refuses a value other than `0` or `1`, and refuses while a rustflags variable is set, because
/// `CARGO_ENCODED_RUSTFLAGS` takes precedence over them and would silently drop their flags.
pub(crate) fn route_environment(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Vec<(&'static str, &'static str)>, String> {
    match lookup(ROUTE_VARIABLE).as_deref().and_then(|v| v.to_str()) {
        None | Some("0") => return Ok(Vec::new()),
        Some("1") => {}
        Some(other) => {
            return Err(format!(
                "check-difftest: {ROUTE_VARIABLE} is `{other}`; use `1` (route on) or leave it unset"
            ));
        }
    }
    let set: Vec<&str> = crate::checks::RUSTFLAGS_VARIABLES
        .iter()
        .copied()
        .filter(|name| lookup(name).is_some())
        .collect();
    if !set.is_empty() {
        return Err(format!(
            "check-difftest: {ROUTE_VARIABLE}=1 delivers its cfg through CARGO_ENCODED_RUSTFLAGS, which would drop the flags of {}; unset {} and run it again",
            set.join(", "),
            if set.len() == 1 { "it" } else { "them" }
        ));
    }
    Ok(vec![("CARGO_ENCODED_RUSTFLAGS", ROUTE_RUSTFLAGS)])
}

/// The file `gust-wgpu` appends one line to for each kernel it builds, `<kernel> <Spirv|Wgsl> <reason>` (a verification
/// aid of the `spirv-route` feature; it is read when the variable is set, whatever the caller).
pub(crate) const ROUTE_LOG_VARIABLE: &str = "GUST_SPIRV_ROUTE_LOG";

/// The route counts of a log and what is wrong with it: `(spirv, wgsl, problems)`. A route run is evidence only if the
/// log is not empty and every line says `Spirv`; each other line is named with its reason.
pub(crate) fn judge_route_log(text: &str) -> (usize, usize, Vec<String>) {
    let (mut spirv, mut wgsl) = (0, 0);
    let mut problems = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        match line.split_whitespace().nth(1) {
            Some("Spirv") => spirv += 1,
            _ => {
                wgsl += 1;
                if problems.len() < 5 {
                    problems.push(format!("a kernel did not take the SPIR-V route: {line}"));
                }
            }
        }
    }
    if spirv + wgsl == 0 {
        problems.push("the route log has no line: no kernel took any route".to_owned());
    }
    (spirv, wgsl, problems)
}

/// The decision after an attempt to delete the route log of an earlier run (m9.rt2 review, R2): a log that was never
/// there is fine, any other failure is an error, because an old log that stays (a process of a killed run still holds
/// it open, say) could let a run that wrote no line pass on stale evidence.
pub(crate) fn stale_log_removal(result: std::io::Result<()>, log: &Path) -> Result<(), String> {
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "check-difftest: cannot delete the old route log {}: {error}",
            log.display()
        )),
    }
}

/// Delete the route log of an earlier run before a run that writes a new one (see [`stale_log_removal`]).
fn remove_stale_log(log: &Path) -> Result<(), String> {
    stale_log_removal(fs::remove_file(log), log)
}

/// Read the route log of `what` and judge it; prints the counts. A missing log is a problem.
fn route_log_problems(what: &str, log: &Path) -> Vec<String> {
    let text = fs::read_to_string(log).unwrap_or_default();
    let (spirv, wgsl, problems) = judge_route_log(&text);
    println!(
        "check-difftest: route of {what}: {spirv} kernel build(s) on SPIR-V, {wgsl} on WGSL ({})",
        log.display()
    );
    problems
        .into_iter()
        .map(|problem| format!("{what}: {problem}"))
        .collect()
}

/// Programs the generator keeps per seed (`PROGRAMS_PER_SEED` of the harness crate; the report must say so).
pub(crate) const PROGRAMS_PER_SEED: u64 = 16;

/// The corpus of `tests/difftest/seeds.txt`. A test requires this to be the first column of that file.
pub(crate) const DEFAULT_SEEDS: &str = "1-8";

const MANIFEST: &str = "tests/difftest/Cargo.toml";

/// The GPU half of `tests/difftest/tests/short_circuit.rs`; it must be reported `ok`, not merely not failed.
pub(crate) const SHORT_CIRCUIT_GPU_TEST: &str = "the_gpu_skips_an_out_of_bounds_right_operand";

/// `random:N` draws between 1 and this many seeds.
const MAX_RANDOM_SEEDS: usize = 64;

/// A range such as `1-4` may span at most this many seeds (the harness refuses larger ones too).
const MAX_RANGE: u64 = 4096;

const USAGE: &str = "usage: cargo xtask check-difftest [--seeds <list|range|random:N>] [--shapes grid,pixel] [--cpu-only] [--report PATH]";

/// The `GUST_PROFILE` of the batch that proves the capability changes no artifact of the corpus ([`DifftestStep::ProfileArtifacts`]).
pub(crate) const SUBGROUPS_PROFILE: &str = "dev+subgroups";

/// The shapes `--shapes` names for the Full mode of `verify` when the caller names none.
pub(crate) const FULL_SHAPES: &str = "grid,pixel";

/// Which parties run: all three on the adapter, or the interpreter against the CPU twin only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Gpu,
    CpuOnly,
}

impl Mode {
    /// The `mode` string of `report.json`.
    pub(crate) fn report_name(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::CpuOnly => "cpu-only",
        }
    }
}

/// One step of `check-difftest`, in the order it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DifftestStep {
    /// `difftest-run` for the seeds, then the judgement of its report.
    Batch,
    /// `cargo test --test short_circuit -- --ignored` of the harness crate: the GPU must skip a right operand that
    /// would read out of bounds. Gpu tier only.
    ShortCircuitGpu,
    /// `difftest-run --profile dev+subgroups` over the same seeds (M8, slice m8.e, plan section 3): the corpus run crate built
    /// and run with the `subgroups` capability must pass, and the records the macro exports for it (`GUST_EXPORT_DIR`) must
    /// equal, byte for byte, the records of the `dev` batch: no corpus program has a variant, so the capability changes
    /// nothing the compiler writes. Gpu tier only (the run opens a device with the capability).
    ProfileArtifacts,
}

/// The steps of a mode. The GPU test belongs to the adapter run and to nothing else.
pub(crate) fn steps(mode: Mode) -> Vec<DifftestStep> {
    match mode {
        Mode::Gpu => vec![
            DifftestStep::Batch,
            DifftestStep::ShortCircuitGpu,
            DifftestStep::ProfileArtifacts,
        ],
        Mode::CpuOnly => vec![DifftestStep::Batch],
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) seeds: Vec<u64>,
    pub(crate) mode: Mode,
    pub(crate) report: Option<PathBuf>,
    /// Set for `random:N`, so the summary can say the seeds were drawn.
    pub(crate) random: bool,
    /// `--shapes grid,pixel`: every program also runs as a pixel kernel (`grid` always runs: it is the oracle).
    pub(crate) pixel: bool,
}

/// Parses the value of `--shapes`: `grid` or `grid,pixel` (either order, each once). `grid` is the shape every other one
/// is compared with, so it must be named. Returns whether `pixel` is among them.
pub(crate) fn parse_shapes(text: &str) -> Result<bool, String> {
    let (mut grid, mut pixel) = (false, false);
    for part in text.split(',') {
        let seen = match part.trim() {
            "grid" => &mut grid,
            "pixel" => &mut pixel,
            other => {
                return Err(format!(
                    "unknown shape {other:?}; the shapes are `grid` and `pixel`\n{USAGE}"
                ));
            }
        };
        if std::mem::replace(seen, true) {
            return Err(format!("shape {:?} named twice\n{USAGE}", part.trim()));
        }
    }
    if !grid {
        return Err(format!(
            "`--shapes` must name `grid`, the shape every other one is compared with\n{USAGE}"
        ));
    }
    Ok(pixel)
}

impl Options {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        Self::parse_with(args, clock_entropy())
    }

    /// [`Options::parse`] with the entropy of `random:N` given, so a test can fix it.
    pub(crate) fn parse_with(args: &[String], entropy: u64) -> Result<Self, String> {
        let mut spec: Option<String> = None;
        let mut cpu_only = false;
        let mut shapes: Option<String> = None;
        let mut report = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--seeds" => {
                    let value = iter.next().ok_or("--seeds needs a value")?;
                    if spec.replace(value.clone()).is_some() {
                        return Err(format!("--seeds given twice\n{USAGE}"));
                    }
                }
                "--cpu-only" => cpu_only = true,
                "--shapes" => {
                    let value = iter.next().ok_or("--shapes needs a value")?;
                    if shapes.replace(value.clone()).is_some() {
                        return Err(format!("--shapes given twice\n{USAGE}"));
                    }
                }
                "--report" => {
                    let value = iter.next().ok_or("--report needs a path")?;
                    if value.is_empty() || report.replace(PathBuf::from(value)).is_some() {
                        return Err(format!("--report needs one non-empty path\n{USAGE}"));
                    }
                }
                other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
            }
        }
        let spec = spec.as_deref().unwrap_or(DEFAULT_SEEDS);
        let (seeds, random) = match spec.strip_prefix("random:") {
            Some(count) => (random_seeds(count, entropy)?, true),
            None => (parse_seeds(spec)?, false),
        };
        let pixel = shapes
            .as_deref()
            .map(parse_shapes)
            .transpose()?
            .unwrap_or(false);
        Ok(Self {
            seeds,
            mode: if cpu_only { Mode::CpuOnly } else { Mode::Gpu },
            report,
            random,
            pixel,
        })
    }

    /// The `programs_executed` a passing run must report.
    pub(crate) fn expected_programs(&self) -> u64 {
        PROGRAMS_PER_SEED * self.seeds.len() as u64
    }
}

/// Parses `1-4`, `3`, `1,3,5-6` into the sorted, duplicate-free list of seeds (the grammar of `difftest-run`, so the
/// list this command expects is the list the harness runs).
pub(crate) fn parse_seeds(text: &str) -> Result<Vec<u64>, String> {
    let mut seeds = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        let number = |s: &str| {
            s.trim()
                .parse::<u64>()
                .map_err(|error| format!("seed {s:?}: {error}"))
        };
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last) = (number(first)?, number(last)?);
                if first > last {
                    return Err(format!("seed range {part:?} is empty"));
                }
                if last - first >= MAX_RANGE {
                    return Err(format!("seed range {part:?} is larger than {MAX_RANGE}"));
                }
                seeds.extend(first..=last);
            }
            None => seeds.push(number(part)?),
        }
    }
    seeds.sort_unstable();
    seeds.dedup();
    if seeds.is_empty() {
        return Err("no seeds".into());
    }
    Ok(seeds)
}

fn clock_entropy() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    nanos ^ (u64::from(std::process::id()) << 32)
}

/// `count` distinct seeds in `10_000..1_000_010_000`, sorted, from `entropy` (splitmix64). The range stays clear of the
/// corpus seeds, so a random run adds programs instead of repeating the pinned ones.
pub(crate) fn random_seeds(count: &str, entropy: u64) -> Result<Vec<u64>, String> {
    let wanted: usize = count
        .trim()
        .parse()
        .map_err(|error| format!("random:{count}: {error}"))?;
    if wanted == 0 || wanted > MAX_RANDOM_SEEDS {
        return Err(format!(
            "random:{count}: the count must be between 1 and {MAX_RANDOM_SEEDS}"
        ));
    }
    let mut state = entropy;
    let mut seeds = Vec::new();
    while seeds.len() < wanted {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let seed = 10_000 + z % 1_000_000_000;
        if !seeds.contains(&seed) {
            seeds.push(seed);
        }
    }
    seeds.sort_unstable();
    Ok(seeds)
}

/// The seeds as `difftest-run --seeds` takes them.
pub(crate) fn seed_list(seeds: &[u64]) -> String {
    seeds
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The arguments of the cargo call that runs the batch.
pub(crate) fn batch_args(options: &Options, target: &Path, report: &Path) -> Vec<String> {
    let mut args: Vec<String> = [
        "run",
        "--locked",
        "--manifest-path",
        MANIFEST,
        "--bin",
        "difftest-run",
        "--",
        "--seeds",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    args.push(seed_list(&options.seeds));
    if options.pixel {
        args.push("--shapes".to_owned());
        args.push(FULL_SHAPES.to_owned());
    }
    if options.mode == Mode::CpuOnly {
        args.push("--cpu-only".to_owned());
    }
    args.push("--report".to_owned());
    args.push(report.to_string_lossy().into_owned());
    args.push("--target-dir".to_owned());
    args.push(target.to_string_lossy().into_owned());
    args
}

/// The arguments of the cargo call that runs the GPU short-circuit test of the harness crate.
pub(crate) fn short_circuit_args() -> Vec<String> {
    [
        "test",
        "--locked",
        "--manifest-path",
        MANIFEST,
        "--test",
        "short_circuit",
        "--",
        "--ignored",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect()
}

/// The arguments that make a batch run under `profile` and export its records to `export_dir`, appended to
/// [`batch_args`] (M8, slice m8.e).
pub(crate) fn export_args(profile: &str, export_dir: &Path) -> Vec<String> {
    vec![
        "--profile".to_owned(),
        profile.to_owned(),
        "--export-dir".to_owned(),
        export_dir.to_string_lossy().into_owned(),
    ]
}

/// What is wrong with the `profile` of `report` for a run that asked for `expected`. A report from before `--profile`
/// has no such field and ran `dev`.
pub(crate) fn profile_problems(report: &Report, expected: &str) -> Vec<String> {
    let ran = report.profile.as_deref().unwrap_or("dev");
    if ran == expected {
        Vec::new()
    } else {
        vec![format!(
            "the report ran under the profile `{ran}`, this batch asked for `{expected}`"
        )]
    }
}

/// The capabilities a batch under `profile` must report from inside the run crate: the names after the target in the
/// profile (`dev+subgroups` has `subgroups`, `dev` has none).
pub(crate) fn expected_capabilities(profile: &str) -> Vec<String> {
    profile.split('+').skip(1).map(str::to_owned).collect()
}

/// What is wrong with the capabilities the programs of `report` reported (Opus S4): the profile argument is echoed by the
/// harness, but `gust::capability::enabled()` is read by the program itself, so this is the proof that the profile
/// reached the run crate's build. Every result must report exactly the capabilities of `profile` (so the `dev` batch
/// reports none and the `dev+subgroups` batch reports `subgroups` in every result); a report that names none of them
/// proves nothing.
pub(crate) fn capability_problems(report: &Report, profile: &str) -> Vec<String> {
    let expected = expected_capabilities(profile);
    let (Some(every), Some(any)) = (
        &report.capabilities_in_every_result,
        &report.capabilities_in_any_result,
    ) else {
        return vec![format!(
            "the report does not say which capabilities the programs were built with, so the profile `{profile}` is not shown to have reached the build"
        )];
    };
    let mut problems = Vec::new();
    for name in &expected {
        if !every.contains(name) {
            problems.push(format!(
                "the profile `{profile}` builds with `{name}`, but not every program reported it (in every result: {every:?}, in any: {any:?})"
            ));
        }
    }
    for name in any {
        if !expected.contains(name) {
            problems.push(format!(
                "a program reported the capability `{name}`, which the profile `{profile}` does not build with"
            ));
        }
    }
    problems
}

/// Every file under `root`, as (path relative to `root` with `/`, bytes), sorted by path. A directory that does not
/// exist is an empty tree (the macro creates its record directory only when it writes a record).
pub(crate) fn record_tree(root: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    fn walk(base: &Path, dir: &Path, found: &mut Vec<(String, Vec<u8>)>) -> Result<(), String> {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(format!("{}: {error}", dir.display())),
        };
        for entry in entries {
            let path = entry
                .map_err(|error| format!("{}: {error}", dir.display()))?
                .path();
            if path.is_dir() {
                walk(base, &path, found)?;
            } else {
                let relative = path
                    .strip_prefix(base)
                    .map_err(|error| format!("{}: {error}", path.display()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes =
                    fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                found.push((relative, bytes));
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(root, root, &mut found)?;
    found.sort();
    Ok(found)
}

/// What differs between the records of the `dev` batch and those of the `dev+subgroups` batch: both must hold at least
/// one record, the same files, and the same bytes in each (the compiler writes the same Slang and the same kernel
/// description whether or not the build has the capability, because no program of the corpus has a variant).
pub(crate) fn record_problems(
    dev: &[(String, Vec<u8>)],
    subgroups: &[(String, Vec<u8>)],
) -> Vec<String> {
    const LISTED: usize = 5;
    let mut problems = Vec::new();
    if dev.is_empty() {
        problems.push("the dev batch exported no records, so there is nothing to compare".into());
    }
    if subgroups.is_empty() {
        problems.push("the dev+subgroups batch exported no records".into());
    }
    if dev.is_empty() || subgroups.is_empty() {
        return problems;
    }
    let names =
        |tree: &[(String, Vec<u8>)]| tree.iter().map(|entry| entry.0.clone()).collect::<Vec<_>>();
    let (dev_names, subgroup_names) = (names(dev), names(subgroups));
    let only = |from: &[String], other: &[String]| {
        from.iter()
            .filter(|name| !other.contains(name))
            .take(LISTED)
            .cloned()
            .collect::<Vec<_>>()
    };
    let (only_dev, only_subgroups) = (
        only(&dev_names, &subgroup_names),
        only(&subgroup_names, &dev_names),
    );
    if !only_dev.is_empty() || !only_subgroups.is_empty() {
        problems.push(format!(
            "the two batches exported different records: only in dev [{}], only in dev+subgroups [{}] ({} and {} records)",
            only_dev.join(", "),
            only_subgroups.join(", "),
            dev.len(),
            subgroups.len()
        ));
    }
    let mut different = Vec::new();
    for (name, bytes) in dev {
        if let Some((_, other)) = subgroups.iter().find(|entry| &entry.0 == name)
            && other != bytes
        {
            different.push(name.clone());
        }
    }
    if !different.is_empty() {
        problems.push(format!(
            "{} record(s) differ between the dev and the dev+subgroups batch, for example [{}]",
            different.len(),
            different
                .iter()
                .take(LISTED)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    problems
}

/// A directory name that is new for each run: the record directory of the macro must not hold the records of an
/// earlier run, and cargo re-expands the run crate only when `GUST_EXPORT_DIR` changes.
fn fresh_record_dirs(target: &Path) -> (PathBuf, PathBuf) {
    let nonce = format!("{:016x}", clock_entropy());
    let base = target.join("difftest-records").join(nonce);
    (base.join("dev"), base.join("dev-subgroups"))
}

/// One disagreement of a report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MismatchRecord {
    pub(crate) program: String,
    pub(crate) seed: u64,
    pub(crate) index: u64,
    pub(crate) attempt: u64,
    pub(crate) kind: String,
    pub(crate) elements_differing: u64,
    pub(crate) message: String,
    /// `grid` or `pixel`; a report from before `--shapes` has no such field, and its mismatches are `grid`.
    pub(crate) shape: String,
}

/// The `f32` observation of a pixel run (D128): never a mismatch, printed and range-checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FloatObservation {
    pub(crate) programs_compared: u64,
    pub(crate) programs_with_difference: u64,
    pub(crate) elements_differing: u64,
    pub(crate) max_ulp: u64,
}

/// The `shapes` section of a report: what the run was asked for, how many programs ran as pixel, and the programs that
/// stayed `grid`-only, counted by the classifier's reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShapesRecord {
    pub(crate) float_observation: Option<FloatObservation>,
    pub(crate) requested: Vec<String>,
    pub(crate) grid_programs: u64,
    pub(crate) pixel_programs: u64,
    pub(crate) grid_only_total: u64,
    pub(crate) grid_only_by_reason: Vec<(String, u64)>,
}

/// The fields of `report.json` this command judges and prints.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Report {
    pub(crate) mode: String,
    pub(crate) passed: bool,
    pub(crate) seeds: Vec<u64>,
    pub(crate) programs_requested: u64,
    pub(crate) programs_executed: u64,
    pub(crate) adapter: Option<String>,
    /// `(seed, programs, attempts, discarded)` per seed.
    pub(crate) per_seed: Vec<(u64, u64, u64, u64)>,
    pub(crate) discards_total: u64,
    pub(crate) discards_by_class: Vec<(String, u64)>,
    pub(crate) mismatch_count: u64,
    pub(crate) mismatches: Vec<MismatchRecord>,
    pub(crate) cargo_build_s: f64,
    pub(crate) cargo_run_s: f64,
    /// Absent in a report from before `--profile` (m8.e); such a report ran `dev`.
    pub(crate) profile: Option<String>,
    /// The capabilities every program of the batch reported for the run crate's own build (`gust::capability::enabled`, in
    /// lower case as the profile spells them), and those at least one reported. Absent in a report from before m8.e.
    pub(crate) capabilities_in_every_result: Option<Vec<String>>,
    pub(crate) capabilities_in_any_result: Option<Vec<String>>,
    /// Absent in a report from before `--shapes`.
    pub(crate) shapes: Option<ShapesRecord>,
    /// The order-sensitive shapes and the `match` constructs of the program text and of the run, from `coverage.operators`
    /// (empty in a report without coverage).
    pub(crate) shape_rows: Vec<ShapeRow>,
}

/// One row of `coverage.operators` that names a shape or a `match` construct (slice m9.d1b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShapeRow {
    pub(crate) name: String,
    pub(crate) programs_in_text: u64,
    pub(crate) occurrences_in_text: u64,
    pub(crate) programs_executing: u64,
    pub(crate) evaluations: u64,
}

/// Whether a counted operator name is a shape, a `match` construct, (format 10) a tuple construct or (format 11) an
/// array construct (`array element read`; the order shapes start with `shape array`) or (format 12) a closure
/// construct (`closure call`; its order shapes start with `shape closure`).
fn is_shape_row(name: &str) -> bool {
    name.starts_with("shape ")
        || name.starts_with("match")
        || name.starts_with("enum ")
        || name.starts_with("tuple ")
        || name.starts_with("array ")
        || name.starts_with("closure ")
        || name.starts_with("trait ")
        || name.starts_with("generic ")
        || name == "return"
}

fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, String> {
    json.get(key)
        .ok_or_else(|| format!("report.json has no `{key}`"))
}

fn uint(json: &Json, key: &str) -> Result<u64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a whole number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

fn float(json: &Json, key: &str) -> Result<f64, String> {
    match field(json, key)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("report.json `{key}` is not a number: {text}")),
        _ => Err(format!("report.json `{key}` is not a number")),
    }
}

fn text(json: &Json, key: &str) -> Result<String, String> {
    field(json, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("report.json `{key}` is not a string"))
}

fn array<'a>(json: &'a Json, key: &str) -> Result<&'a [Json], String> {
    field(json, key)?
        .as_array()
        .ok_or_else(|| format!("report.json `{key}` is not an array"))
}

/// Reads the fields of a `report.json`. A missing or mistyped field is an error: a report this function cannot read
/// proves nothing.
pub(crate) fn parse_report(source: &str) -> Result<Report, String> {
    let json = json::parse(source).map_err(|error| format!("report.json: {error}"))?;
    let tool = text(&json, "tool")?;
    if tool != "difftest-run" {
        return Err(format!("report.json is from `{tool}`, not `difftest-run`"));
    }
    let passed = match field(&json, "passed")? {
        Json::Bool(value) => *value,
        _ => return Err("report.json `passed` is not a boolean".into()),
    };
    let seeds = array(&json, "seeds")?
        .iter()
        .map(|seed| match seed {
            Json::Number(text) => text
                .parse()
                .map_err(|_| format!("report.json seed {text} is not a whole number")),
            _ => Err("report.json `seeds` holds a non-number".to_owned()),
        })
        .collect::<Result<Vec<u64>, _>>()?;
    let adapter = match field(&json, "adapter")? {
        Json::Null => None,
        Json::String(name) => Some(name.clone()),
        _ => return Err("report.json `adapter` is neither a string nor null".into()),
    };
    let per_seed = array(&json, "per_seed")?
        .iter()
        .map(|entry| {
            Ok((
                uint(entry, "seed")?,
                uint(entry, "programs")?,
                uint(entry, "attempts")?,
                uint(entry, "discarded")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let discards = field(&json, "discards")?;
    let by_class = match field(discards, "by_class")? {
        Json::Object(members) => members
            .iter()
            .map(|(name, value)| match value {
                Json::Number(count) => count
                    .parse()
                    .map(|count| (name.clone(), count))
                    .map_err(|_| format!("report.json discard class {name} is not a count")),
                _ => Err(format!("report.json discard class {name} is not a number")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("report.json `discards.by_class` is not an object".into()),
    };
    let mismatches = array(&json, "mismatches")?
        .iter()
        .map(|entry| {
            Ok(MismatchRecord {
                program: text(entry, "program")?,
                seed: uint(entry, "seed")?,
                index: uint(entry, "index")?,
                attempt: uint(entry, "attempt")?,
                kind: text(entry, "kind")?,
                elements_differing: uint(entry, "elements_differing")?,
                message: text(entry, "message")?,
                shape: match entry.get("shape") {
                    None => "grid".to_owned(),
                    Some(_) => text(entry, "shape")?,
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let timing = field(&json, "timing")?;
    let shapes = match json.get("shapes") {
        None => None,
        Some(section) => Some(parse_shapes_record(section)?),
    };
    let profile = match json.get("profile") {
        None => None,
        Some(_) => Some(text(&json, "profile")?),
    };
    let names = |key: &str| -> Result<Option<Vec<String>>, String> {
        json.get(key)
            .map(|_| {
                array(&json, key)?
                    .iter()
                    .map(|name| {
                        name.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| format!("report.json `{key}` holds a non-text item"))
                    })
                    .collect()
            })
            .transpose()
    };
    let shape_rows = parse_shape_rows(&json)?;
    let capabilities_in_every_result = names("capabilities_in_every_result")?;
    let capabilities_in_any_result = names("capabilities_in_any_result")?;
    Ok(Report {
        mode: text(&json, "mode")?,
        passed,
        seeds,
        programs_requested: uint(&json, "programs_requested")?,
        programs_executed: uint(&json, "programs_executed")?,
        adapter,
        per_seed,
        discards_total: uint(discards, "total")?,
        discards_by_class: by_class,
        mismatch_count: uint(&json, "mismatch_count")?,
        mismatches,
        cargo_build_s: float(timing, "cargo_build_s")?,
        cargo_run_s: float(timing, "cargo_run_s")?,
        shapes,
        profile,
        capabilities_in_every_result,
        capabilities_in_any_result,
        shape_rows,
    })
}

/// The shape and `match` rows of `coverage.operators`; none when the report has no coverage.
fn parse_shape_rows(json: &Json) -> Result<Vec<ShapeRow>, String> {
    let Some(coverage) = json.get("coverage") else {
        return Ok(Vec::new());
    };
    let Some(operators) = coverage.get("operators").and_then(Json::as_array) else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::new();
    for entry in operators {
        let name = text(entry, "operator")?;
        if !is_shape_row(&name) {
            continue;
        }
        let optional = |key: &str| -> Result<u64, String> {
            if entry.get(key).is_some() {
                uint(entry, key)
            } else {
                Ok(0)
            }
        };
        rows.push(ShapeRow {
            programs_in_text: optional("programs_in_text")?,
            occurrences_in_text: optional("occurrences_in_text")?,
            programs_executing: optional("programs_executing")?,
            evaluations: optional("evaluations")?,
            name,
        });
    }
    Ok(rows)
}

fn parse_shapes_record(section: &Json) -> Result<ShapesRecord, String> {
    let requested = array(section, "requested")?
        .iter()
        .map(|shape| {
            shape
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "report.json `shapes.requested` holds a non-string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let by_shape = field(section, "programs_by_shape")?;
    let grid_only = field(section, "grid_only")?;
    let by_reason = match field(grid_only, "by_reason")? {
        Json::Object(members) => members
            .iter()
            .map(|(name, value)| match value {
                Json::Number(count) => count
                    .parse()
                    .map(|count| (name.clone(), count))
                    .map_err(|_| format!("report.json grid-only reason {name:?} is not a count")),
                _ => Err(format!(
                    "report.json grid-only reason {name:?} is not a number"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("report.json `shapes.grid_only.by_reason` is not an object".into()),
    };
    let float_observation = match section {
        Json::Object(members) if members.iter().any(|(k, _)| k == "float_pixel_vs_grid") => {
            let seen = field(section, "float_pixel_vs_grid")?;
            Some(FloatObservation {
                programs_compared: uint(seen, "programs_compared")?,
                programs_with_difference: uint(seen, "programs_with_difference")?,
                elements_differing: uint(seen, "elements_differing")?,
                max_ulp: uint(seen, "max_ulp")?,
            })
        }
        _ => None,
    };
    Ok(ShapesRecord {
        float_observation,
        requested,
        grid_programs: uint(by_shape, "grid")?,
        pixel_programs: uint(by_shape, "pixel")?,
        grid_only_total: uint(grid_only, "total")?,
        grid_only_by_reason: by_reason,
    })
}

/// What is wrong with the `shapes` section of `report` for a run that asked for `options`. A report without the section
/// is the grid-only form and is valid only for a grid-only request.
fn shapes_problems(report: &Report, options: &Options) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = options.expected_programs();
    let Some(shapes) = &report.shapes else {
        if options.pixel {
            problems.push(
                "the report has no `shapes` section, and this run asked for `--shapes grid,pixel`"
                    .into(),
            );
        }
        return problems;
    };
    let wanted: &[&str] = if options.pixel {
        &["grid", "pixel"]
    } else {
        &["grid"]
    };
    if shapes.requested != wanted {
        problems.push(format!(
            "the report ran shapes [{}], this run asked for [{}]",
            shapes.requested.join(","),
            wanted.join(",")
        ));
    }
    if shapes.grid_programs != expected {
        problems.push(format!(
            "shapes.programs_by_shape.grid is {}, expected {expected}",
            shapes.grid_programs
        ));
    }
    let reasons: u64 = shapes.grid_only_by_reason.iter().map(|entry| entry.1).sum();
    if reasons != shapes.grid_only_total {
        problems.push(format!(
            "shapes.grid_only counts {} programs but its reasons add up to {reasons}",
            shapes.grid_only_total
        ));
    }
    if options.pixel {
        if shapes.pixel_programs + shapes.grid_only_total != expected {
            problems.push(format!(
                "{} pixel programs and {} grid-only programs do not add up to the {expected} programs of the run",
                shapes.pixel_programs, shapes.grid_only_total
            ));
        }
        if shapes.pixel_programs == 0 {
            problems
                .push("no program ran as pixel: the run says nothing about the pixel shape".into());
        }
        match &shapes.float_observation {
            None => problems
                .push("shapes has no `float_pixel_vs_grid` observation of the f32 programs".into()),
            Some(seen) => {
                if seen.programs_compared > shapes.pixel_programs
                    || seen.programs_with_difference > seen.programs_compared
                    || (seen.programs_with_difference == 0) != (seen.elements_differing == 0)
                    || seen.elements_differing < seen.programs_with_difference
                {
                    problems.push(format!(
                        "shapes.float_pixel_vs_grid is inconsistent: {} programs compared of {} pixel programs, {} with a difference, {} elements, {} ulp at most",
                        seen.programs_compared,
                        shapes.pixel_programs,
                        seen.programs_with_difference,
                        seen.elements_differing,
                        seen.max_ulp
                    ));
                }
            }
        }
    } else if shapes.pixel_programs != 0 || shapes.grid_only_total != 0 {
        problems.push(format!(
            "a grid-only run reports {} pixel programs and {} grid-only programs",
            shapes.pixel_programs, shapes.grid_only_total
        ));
    }
    problems
}

/// Everything wrong with `report` for a run that asked for `options`; empty when the report is a pass. The count rule
/// is the point: `programs_executed` must equal `16 x seeds`, so a harness that silently ran fewer programs fails here
/// whatever it says about `passed`.
pub(crate) fn report_problems(report: &Report, options: &Options) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = options.expected_programs();
    if report.programs_executed != expected {
        problems.push(format!(
            "programs_executed is {}, expected {PROGRAMS_PER_SEED} x {} seeds = {expected}",
            report.programs_executed,
            options.seeds.len()
        ));
    }
    if report.programs_requested != expected {
        problems.push(format!(
            "programs_requested is {}, expected {PROGRAMS_PER_SEED} x {} seeds = {expected}",
            report.programs_requested,
            options.seeds.len()
        ));
    }
    if report.mode != options.mode.report_name() {
        problems.push(format!(
            "the report is for mode `{}`, this run asked for `{}`",
            report.mode,
            options.mode.report_name()
        ));
    }
    if report.seeds != options.seeds {
        problems.push(format!(
            "the report ran seeds [{}], this run asked for [{}]",
            seed_list(&report.seeds),
            seed_list(&options.seeds)
        ));
    }
    // One entry per seed that ran, in that order: a table that names other seeds, or one twice, proves nothing about
    // the seeds that were asked for even when its counts add up.
    let table_seeds: Vec<u64> = report.per_seed.iter().map(|entry| entry.0).collect();
    if table_seeds != report.seeds {
        problems.push(format!(
            "per_seed names seeds [{}] but the report ran seeds [{}]",
            seed_list(&table_seeds),
            seed_list(&report.seeds)
        ));
    }
    for (seed, programs, _, _) in &report.per_seed {
        if *programs != PROGRAMS_PER_SEED {
            problems.push(format!(
                "seed {seed}: {programs} programs, expected {PROGRAMS_PER_SEED}"
            ));
        }
    }
    let planned: u64 = report.per_seed.iter().map(|entry| entry.1).sum();
    if planned != report.programs_executed {
        problems.push(format!(
            "per_seed holds {planned} programs but programs_executed is {}",
            report.programs_executed
        ));
    }
    if report.mismatch_count != report.mismatches.len() as u64 {
        problems.push(format!(
            "mismatch_count is {} but the report lists {} mismatches",
            report.mismatch_count,
            report.mismatches.len()
        ));
    }
    problems.extend(shapes_problems(report, options));
    for mismatch in &report.mismatches {
        let shape = if mismatch.shape == "grid" {
            String::new()
        } else {
            format!(", shape {}", mismatch.shape)
        };
        // A build failure is named with the commands that reproduce and shrink it.
        let reproduce = if mismatch.kind == "gpu-error" {
            format!(
                " [reproduce: difftest-gen show {} {}; shrink: difftest-run --shrink {}:{}]",
                mismatch.seed, mismatch.index, mismatch.seed, mismatch.index
            )
        } else {
            String::new()
        };
        problems.push(format!(
            "seed {} program {} (index {}, attempt {}{shape}): {}: {} element(s) differ: {}{}",
            mismatch.seed,
            mismatch.program,
            mismatch.index,
            mismatch.attempt,
            mismatch.kind,
            mismatch.elements_differing,
            // A build failure is one line that the reader needs whole; the other messages are cut.
            first_line(
                &mismatch.message,
                if mismatch.kind == "gpu-error" {
                    3000
                } else {
                    300
                }
            ),
            reproduce
        ));
    }
    match (options.mode, &report.adapter) {
        (Mode::Gpu, None) => problems.push("a gpu report names no adapter".into()),
        (Mode::Gpu, Some(name)) if name.trim().is_empty() => {
            problems.push("a gpu report names an empty adapter".into());
        }
        (Mode::CpuOnly, Some(name)) => {
            problems.push(format!("a cpu-only report names the adapter {name:?}"));
        }
        _ => {}
    }
    if !report.passed && problems.is_empty() {
        problems.push("the report says passed: false and names no cause".into());
    }
    problems
}

/// The first line of `message`, cut to `limit` characters.
fn first_line(message: &str, limit: usize) -> String {
    let line = message.lines().next().unwrap_or_default();
    match line.char_indices().nth(limit) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line.to_owned(),
    }
}

/// What the batch step says: the problems (an empty list is a pass) and the summary lines to print.
pub(crate) struct Judgement {
    pub(crate) problems: Vec<String>,
    pub(crate) summary: Vec<String>,
}

/// Judges the end of the batch from the exit status of `difftest-run` and the content of its report (or the reason it
/// could not be read). A run that exits non-zero fails whatever the report says; a report that is not a pass fails
/// whatever the exit status says.
pub(crate) fn judge_batch(
    options: &Options,
    exit_code: Option<i32>,
    report: Result<String, String>,
) -> Judgement {
    let mut problems = Vec::new();
    let mut summary = Vec::new();
    match report.and_then(|source| parse_report(&source)) {
        Ok(report) => {
            problems.extend(report_problems(&report, options));
            summary = report_summary(&report, options);
        }
        Err(error) => problems.push(format!("no usable report: {error}")),
    }
    match exit_code {
        Some(0) => {}
        // Status 1 is "a program disagreed"; the report must then say which.
        Some(1) if !problems.is_empty() => {}
        Some(code) => problems.push(format!(
            "difftest-run exited with status {code}{}",
            if code == 2 {
                " (the harness failed: build error, stale template lockfile, a result file missing)"
            } else {
                ""
            }
        )),
        None => problems.push("difftest-run was stopped by a signal".into()),
    }
    Judgement { problems, summary }
}

fn report_summary(report: &Report, options: &Options) -> Vec<String> {
    let adapter = report
        .adapter
        .as_ref()
        .map_or(String::new(), |adapter| format!(" on {adapter}"));
    let mut lines = vec![format!(
        "  mode {}{adapter}, seeds [{}]{}, programs_executed {} \
         (expected {PROGRAMS_PER_SEED} x {} = {}), mismatches {}",
        report.mode,
        seed_list(&report.seeds),
        if options.random {
            " (drawn at random)"
        } else {
            ""
        },
        report.programs_executed,
        options.seeds.len(),
        options.expected_programs(),
        report.mismatch_count
    )];
    for (seed, programs, attempts, discarded) in &report.per_seed {
        lines.push(format!(
            "  seed {seed}: {programs} programs from {attempts} attempts ({discarded} discarded)"
        ));
    }
    let classes = report
        .discards_by_class
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    lines.push(format!(
        "  discards: {} in all ({classes})",
        report.discards_total
    ));
    if let Some(shapes) = &report.shapes
        && options.pixel
    {
        let mut line = format!(
            "  shapes {}: {} programs ran as pixel and were compared with their grid run, {} stayed grid-only",
            shapes.requested.join(","),
            shapes.pixel_programs,
            shapes.grid_only_total
        );
        if let Some(seen) = &shapes.float_observation {
            line.push_str(&format!(
                "\n    f32 pixel vs grid (an observation, not a mismatch): {} of {} programs differ in a bit, {} elements, at most {} ulp",
                seen.programs_with_difference,
                seen.programs_compared,
                seen.elements_differing,
                seen.max_ulp
            ));
        }
        for (reason, count) in &shapes.grid_only_by_reason {
            line.push_str(&format!("\n    grid-only {count} x {reason}"));
        }
        lines.push(line);
    }
    if !report.shape_rows.is_empty() {
        lines.push("  shapes and match constructs (programs in text, occurrences in text, programs executing, evaluations):".to_owned());
        for row in &report.shape_rows {
            lines.push(format!(
                "    {}: {} programs, {} in text, {} executing, {} evaluations",
                row.name,
                row.programs_in_text,
                row.occurrences_in_text,
                row.programs_executing,
                row.evaluations
            ));
        }
    }
    lines.push(format!(
        "  harness timing: cargo build {:.1} s, cargo run {:.1} s",
        report.cargo_build_s, report.cargo_run_s
    ));
    lines
}

/// What is wrong with the output of the short-circuit test run: it must have run a test, nothing may have failed, and
/// the named test must be reported `ok` (a test that was filtered out or ignored again does not count).
pub(crate) fn short_circuit_problems(args: &[String], stdout: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let totals = sum_test_results(stdout);
    if let Err(error) = require_tests_ran(args, &totals) {
        problems.push(error);
    }
    if !tests_not_ok(stdout, &[SHORT_CIRCUIT_GPU_TEST]).is_empty() {
        problems.push(format!(
            "short_circuit: `{SHORT_CIRCUIT_GPU_TEST}` was not reported ok"
        ));
    }
    problems
}

/// `cargo xtask check-difftest`: grid-only unless `--shapes` says otherwise.
pub(crate) fn check_difftest(args: &[String]) -> Result<(), String> {
    println!("{}", difftest_summary(args)?);
    Ok(())
}

/// The Full mode of `verify` (`check-full`) calls this with no arguments: it returns the pass summary instead of
/// printing it, so the record can hold it, and it runs `--shapes grid,pixel` unless the caller names shapes.
pub(crate) fn check_difftest_report(args: &[String]) -> Result<String, String> {
    difftest_summary(&with_full_shapes(args))
}

/// `args` with `--shapes grid,pixel` added when they name no shapes.
pub(crate) fn with_full_shapes(args: &[String]) -> Vec<String> {
    let mut all = args.to_vec();
    if !all.iter().any(|arg| arg == "--shapes") {
        all.push("--shapes".to_owned());
        all.push(FULL_SHAPES.to_owned());
    }
    all
}

fn difftest_summary(args: &[String]) -> Result<String, String> {
    let options = Options::parse(args)?;
    let root = workspace_root()?;
    let target = std::path::absolute(root.join(cargo_target_dir(&root)))
        .map_err(|error| format!("check-difftest: the cargo target directory: {error}"))?;
    let report_path = match &options.report {
        Some(path) => std::path::absolute(path)
            .map_err(|error| format!("check-difftest: --report {}: {error}", path.display()))?,
        None => target.join("difftest-report.json"),
    };
    run_steps(&options, &root, &target, &report_path)
}

fn run_steps(
    options: &Options,
    root: &Path,
    target: &Path,
    report_path: &Path,
) -> Result<String, String> {
    let started = Instant::now();
    println!(
        "check-difftest: {} mode, {} seed(s) [{}]{}, {} programs expected",
        options.mode.report_name(),
        options.seeds.len(),
        seed_list(&options.seeds),
        if options.random {
            " drawn at random; rerun with --seeds <this list> to reproduce"
        } else {
            ""
        },
        options.expected_programs()
    );
    let mut problems = Vec::new();
    let mut summary = Vec::new();
    let all_steps = steps(options.mode);
    // The record directories are named before the first step: the `dev` batch exports into one of them, the
    // `dev+subgroups` batch into the other, and the last step compares the two.
    let records = all_steps
        .contains(&DifftestStep::ProfileArtifacts)
        .then(|| fresh_record_dirs(target));
    let mut compared = None;
    let mut subgroups_timing = None;
    for step in all_steps {
        match step {
            DifftestStep::Batch => {
                let export = records.as_ref().map(|dirs| dirs.0.as_path());
                let judgement = run_batch(options, root, target, report_path, "dev", export)?;
                problems.extend(judgement.problems);
                summary = judgement.summary;
            }
            DifftestStep::ShortCircuitGpu => {
                println!("-- short_circuit (GPU, --ignored)");
                problems.extend(run_short_circuit(root, target)?);
            }
            DifftestStep::ProfileArtifacts => {
                let (dev_dir, subgroups_dir) = records
                    .as_ref()
                    .ok_or("check-difftest: the profile step has no record directories")?;
                println!("-- difftest-run under GUST_PROFILE={SUBGROUPS_PROFILE}");
                let judgement = run_batch(
                    options,
                    root,
                    target,
                    &subgroups_report_path(report_path),
                    SUBGROUPS_PROFILE,
                    Some(subgroups_dir),
                )?;
                // The last summary line of a passing batch is its harness timing (cargo build, cargo run).
                subgroups_timing = judgement.summary.last().cloned();
                problems.extend(
                    judgement
                        .problems
                        .into_iter()
                        .map(|problem| format!("[{SUBGROUPS_PROFILE}] {problem}")),
                );
                let (dev, subgroups) = (record_tree(dev_dir)?, record_tree(subgroups_dir)?);
                problems.extend(record_problems(&dev, &subgroups));
                compared = Some(dev.len());
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    if !problems.is_empty() {
        let mut message = format!(
            "check-difftest failed after {seconds:.1} s with {} problem(s):",
            problems.len()
        );
        for problem in &problems {
            message.push_str("\ncheck-difftest: ");
            message.push_str(problem);
        }
        message.push_str(&format!("\nreport: {}", report_path.display()));
        return Err(message);
    }
    let mut passed = format!(
        "check-difftest passed in {seconds:.1} s{}:\n{}",
        if options.mode == Mode::Gpu {
            " (the batch, the short_circuit GPU test and the dev+subgroups batch)"
        } else {
            " (cpu-only: interpreter against twin; no adapter, no GPU evidence)"
        },
        summary.join("\n")
    );
    if let Some(count) = compared {
        passed.push_str(&format!(
            "\n  {SUBGROUPS_PROFILE}: the batch passed, every program reported the capabilities of the profile from its own build, and its {count} exported record file(s) are byte for byte those of the dev batch"
        ));
        if let Some(timing) = subgroups_timing {
            passed.push_str(&format!(
                "\n  {SUBGROUPS_PROFILE} batch: {}",
                timing.trim_start()
            ));
        }
    }
    passed.push_str(&format!("\n  report: {}", report_path.display()));
    Ok(passed)
}

/// The report of the `dev+subgroups` batch, beside the report of the `dev` one.
fn subgroups_report_path(report_path: &Path) -> PathBuf {
    let stem = report_path.file_stem().map_or_else(
        || "difftest-report".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    report_path.with_file_name(format!("{stem}-subgroups.json"))
}

/// One `difftest-run` batch under `profile`, exporting its records to `export` when that is given (a profile other
/// than `dev` needs it: the arguments that choose the profile are the ones that name the record directory).
fn run_batch(
    options: &Options,
    root: &Path,
    target: &Path,
    report_path: &Path,
    profile: &str,
    export: Option<&Path>,
) -> Result<Judgement, String> {
    // A report left by an earlier run must not be able to pass this one.
    match fs::remove_file(report_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "check-difftest: {}: {error}",
                report_path.display()
            ));
        }
    }
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("check-difftest: {}: {error}", parent.display()))?;
    }
    fs::create_dir_all(target)
        .map_err(|error| format!("check-difftest: {}: {error}", target.display()))?;
    let mut args = batch_args(options, target, report_path);
    if let Some(dir) = export {
        args.extend(export_args(profile, dir));
    } else if profile != "dev" {
        return Err(format!(
            "check-difftest: a batch under the profile `{profile}` needs a record directory"
        ));
    }
    let route = route_environment(|name| std::env::var_os(name))?;
    // With the route on, every kernel the batch builds appends its route to a log beside the report, and the log is
    // judged below (Opus S1 of m9.rt2): a silent fallback to WGSL must not pass as route evidence.
    let route_log = (!route.is_empty()).then(|| report_path.with_extension("route.log"));
    if let Some(log) = &route_log {
        remove_stale_log(log)?;
    }
    println!("-- cargo {}", args.join(" "));
    let mut command = workspace_command("cargo");
    command
        .args(&args)
        .current_dir(root)
        .envs(route)
        .env("CARGO_TARGET_DIR", target)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if let Some(log) = &route_log {
        command.env(ROUTE_LOG_VARIABLE, log);
    }
    let status = command
        .status()
        .map_err(|error| format!("check-difftest: could not launch cargo: {error}"))?;
    let report = fs::read_to_string(report_path)
        .map_err(|error| format!("{}: {error}", report_path.display()));
    let mut judgement = judge_batch(options, status.code(), report.clone());
    if let Ok(parsed) = report.and_then(|source| parse_report(&source)) {
        judgement
            .problems
            .extend(profile_problems(&parsed, profile));
        judgement
            .problems
            .extend(capability_problems(&parsed, profile));
    }
    if let Some(log) = &route_log {
        judgement
            .problems
            .extend(route_log_problems(&format!("the `{profile}` batch"), log));
    }
    Ok(judgement)
}

fn run_short_circuit(root: &Path, target: &Path) -> Result<Vec<String>, String> {
    let args = short_circuit_args();
    println!("-- cargo {}", args.join(" "));
    let route = route_environment(|name| std::env::var_os(name))?;
    // CARGO_TARGET_DIR is what the test reads to build its run crate beside the batch's (without it the test builds
    // the whole dependency tree again under its own `target/tmp`).
    let route_log = (!route.is_empty()).then(|| target.join("short-circuit.route.log"));
    if let Some(log) = &route_log {
        remove_stale_log(log)?;
    }
    let mut command = workspace_command("cargo");
    command
        .args(&args)
        .current_dir(root)
        .env("CARGO_TARGET_DIR", target)
        .envs(route)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(log) = &route_log {
        command.env(ROUTE_LOG_VARIABLE, log);
    }
    let output = command
        .output()
        .map_err(|error| format!("check-difftest: could not launch cargo: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    print!("{stdout}");
    let mut problems = short_circuit_problems(&args, &stdout);
    if let Some(log) = &route_log {
        problems.extend(route_log_problems("the short_circuit test", log));
    }
    if !output.status.success() {
        problems.push(format!(
            "cargo {} failed (exit {})",
            args.join(" "),
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok(problems)
}

#[cfg(test)]
#[path = "tests/difftest.rs"]
mod difftest_tests;

/// Tests of the shape rows of the summary (slice m9.d1b, round r1; the shared test file is another lane's).
#[cfg(test)]
mod shape_row_tests {
    use super::*;

    #[test]
    fn shape_rows_keep_zero_counts_and_skip_the_operators_that_are_not_shapes() {
        let json = crate::json::parse(
            r#"{"coverage":{"operators":[
                {"operator":"+ u32 u32","programs_in_text":3,"programs_executing":3,"evaluations":9},
                {"operator":"match","programs_in_text":2,"occurrences_in_text":5,"programs_executing":1,"evaluations":40},
                {"operator":"shape enum arguments built from calls on one root"},
                {"operator":"return","programs_in_text":0,"occurrences_in_text":0,"programs_executing":0,"evaluations":0},
                {"operator":"enum as u32","programs_in_text":1}
            ]}}"#,
        )
        .unwrap();
        let rows = parse_shape_rows(&json).unwrap();
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "match",
                "shape enum arguments built from calls on one root",
                "return",
                "enum as u32"
            ]
        );
        assert_eq!(
            rows[0],
            ShapeRow {
                name: "match".into(),
                programs_in_text: 2,
                occurrences_in_text: 5,
                programs_executing: 1,
                evaluations: 40
            }
        );
        // A row with no counts at all is a row of zeros, not an error.
        assert_eq!(rows[1].programs_in_text + rows[1].evaluations, 0);
    }

    /// Format 10 (m9.d1c): the value `match`, integer pattern and tuple names are rows too, in report order, and the
    /// operators of the same programs (`<< u32`, `== E0`, `!u32`) are not.
    #[test]
    fn the_value_match_and_tuple_names_of_format_10_are_shape_rows() {
        let json = crate::json::parse(
            r#"{"coverage":{"operators":[
                {"operator":"< u32","programs_in_text":9,"programs_executing":9,"evaluations":90},
                {"operator":"match value (typed let)","programs_in_text":4,"occurrences_in_text":7,"programs_executing":3,"evaluations":300},
                {"operator":"match integer group arm","programs_in_text":2,"occurrences_in_text":2,"programs_executing":2,"evaluations":60},
                {"operator":"tuple destructuring annotated","programs_in_text":3,"occurrences_in_text":4,"programs_executing":3,"evaluations":120},
                {"operator":"tuple element set","programs_in_text":1,"occurrences_in_text":1,"programs_executing":0,"evaluations":0},
                {"operator":"== E0","programs_in_text":1},
                {"operator":"shape < before the last tuple element or argument","programs_in_text":5,"occurrences_in_text":8,"programs_executing":5,"evaluations":40}
            ]}}"#,
        )
        .unwrap();
        let rows = parse_shape_rows(&json).unwrap();
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "match value (typed let)",
                "match integer group arm",
                "tuple destructuring annotated",
                "tuple element set",
                "shape < before the last tuple element or argument"
            ]
        );
        // A construct that is written but never executed keeps its zero.
        assert_eq!(
            (
                rows[3].programs_in_text,
                rows[3].programs_executing,
                rows[3].evaluations
            ),
            (1, 0, 0)
        );
        assert_eq!(rows[4].evaluations, 40);
    }

    /// Format 11 (m9.d1d): the array constructs (`array element read`, `array literal`) are shape rows, and so are the
    /// order shapes (`shape array ...`); a plain operator is not.
    #[test]
    fn the_array_names_of_format_11_are_shape_rows() {
        let json = crate::json::parse(
            r#"{"coverage":{"operators":[
                {"operator":"^ u32","programs_in_text":9,"programs_executing":9,"evaluations":90},
                {"operator":"array element read","programs_in_text":6,"occurrences_in_text":9,"programs_executing":6,"evaluations":900},
                {"operator":"array copy","programs_in_text":1,"occurrences_in_text":1,"programs_executing":0,"evaluations":0},
                {"operator":"shape array literal of &mut self calls on one root","programs_in_text":2,"occurrences_in_text":2,"programs_executing":2,"evaluations":40}
            ]}}"#,
        )
        .unwrap();
        let rows = parse_shape_rows(&json).unwrap();
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "array element read",
                "array copy",
                "shape array literal of &mut self calls on one root"
            ]
        );
        assert_eq!((rows[1].programs_in_text, rows[1].evaluations), (1, 0));
    }

    /// Format 12 (m9.d1e): the closure constructs (`closure call`, `closure bound in a method`) are shape rows, and so are
    /// the order shapes (`shape closure ...`); a plain operator is not.
    #[test]
    fn the_closure_names_of_format_12_are_shape_rows() {
        let json = crate::json::parse(
            r#"{"coverage":{"operators":[
                {"operator":"+ u32","programs_in_text":9,"programs_executing":9,"evaluations":90},
                {"operator":"closure call","programs_in_text":6,"occurrences_in_text":9,"programs_executing":6,"evaluations":900},
                {"operator":"closure bound in a method","programs_in_text":1,"occurrences_in_text":1,"programs_executing":0,"evaluations":0},
                {"operator":"shape closure called twice with a side effect in an argument","programs_in_text":2,"occurrences_in_text":2,"programs_executing":2,"evaluations":40}
            ]}}"#,
        )
        .unwrap();
        let rows = parse_shape_rows(&json).unwrap();
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "closure call",
                "closure bound in a method",
                "shape closure called twice with a side effect in an argument"
            ]
        );
        assert_eq!((rows[1].programs_in_text, rows[1].evaluations), (1, 0));
    }
    #[test]
    fn a_report_without_coverage_has_no_shape_rows() {
        assert!(
            parse_shape_rows(&crate::json::parse("{}").unwrap())
                .unwrap()
                .is_empty()
        );
    }
}

/// Tests of the route environment (slice m9.rt2).
#[cfg(test)]
mod route_environment_tests {
    use super::*;
    use std::ffi::OsString;

    fn lookup<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            set.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(*value))
        }
    }

    #[test]
    fn unset_or_zero_adds_nothing() {
        assert_eq!(route_environment(lookup(&[])), Ok(Vec::new()));
        assert_eq!(
            route_environment(lookup(&[(ROUTE_VARIABLE, "0")])),
            Ok(Vec::new())
        );
    }

    #[test]
    fn one_delivers_the_cfg_through_the_encoded_rustflags() {
        assert_eq!(
            route_environment(lookup(&[(ROUTE_VARIABLE, "1")])),
            Ok(vec![("CARGO_ENCODED_RUSTFLAGS", ROUTE_RUSTFLAGS)])
        );
        assert_eq!(
            ROUTE_RUSTFLAGS.split('\u{1f}').collect::<Vec<_>>(),
            ["--cfg", "feature=\"spirv-route\""]
        );
    }

    #[test]
    fn another_value_and_preset_rustflags_are_refused() {
        let error = route_environment(lookup(&[(ROUTE_VARIABLE, "yes")])).unwrap_err();
        assert!(error.contains("use `1`"), "{error}");
        let error =
            route_environment(lookup(&[(ROUTE_VARIABLE, "1"), ("RUSTFLAGS", "")])).unwrap_err();
        assert!(
            error.contains("RUSTFLAGS") && error.contains("unset it"),
            "{error}"
        );
        let error = route_environment(lookup(&[
            (ROUTE_VARIABLE, "1"),
            ("CARGO_ENCODED_RUSTFLAGS", "x"),
            ("CARGO_BUILD_RUSTFLAGS", "y"),
        ]))
        .unwrap_err();
        assert!(error.contains("unset them"), "{error}");
    }

    #[test]
    fn the_route_log_passes_only_when_every_line_is_spirv() {
        let (spirv, wgsl, problems) = judge_route_log("a::k Spirv \nb::k Spirv \n");
        assert_eq!((spirv, wgsl), (2, 0));
        assert!(problems.is_empty(), "{problems:?}");
        let (spirv, wgsl, problems) =
            judge_route_log("a::k Spirv \nb::k Wgsl backend is not Vulkan\n");
        assert_eq!((spirv, wgsl), (1, 1));
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].contains("b::k Wgsl backend is not Vulkan"),
            "{problems:?}"
        );
        let (_, _, problems) = judge_route_log("");
        assert_eq!(problems.len(), 1, "{problems:?}");
    }
}
