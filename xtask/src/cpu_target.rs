//! `cargo xtask check-cpu-target`: the evidence of the production `cpu` target (D21, D51, plan M5 section 2.1).
//!
//! The fixture workspace `tests/cpu-target` is built and run with cargo only: no slangc, no GPU, no adapter. Every
//! cargo child goes through `check-dual`'s command builder ([`dual_command`]), so it starts without any rustflags
//! variable and with `GUST_PROFILE` set by the step (or removed, for the `unset` state). Steps, in order, each
//! printing one line with its result; the command stops at the first step that fails and names it.
//!
//! - E-a: `app` built in release under `GUST_PROFILE=cpu` and run: one `CPU target: matches (..)` line per kernel and
//!   the final line, exit 0. Every step that runs or scans a binary (E-a, E-b, E-c, E-h) takes its path from the
//!   `executable` of cargo's own `compiler-artifact` record ([`executable_of`]), never from a guessed
//!   `<dir>/release/app`, so a `build.target` or `CARGO_BUILD_TARGET` cannot make it use an old binary.
//! - E-b: that release binary holds none of `slangc`, `gust-slang-reflect`, `wgpu`, `vulkan` (case-insensitive), and
//!   `cargo tree -e features -p app` shows neither `wgpu` nor `slang-runtime`.
//! - E-c: `app` under `cpu` with `--cfg gust_dual` builds and runs (one `Run` impl, no E0119), judged like E-a.
//! - E-d: `app`'s `single-dispatch` under `dev` fails with the one E0277 of `e-d-expected.txt` (the twin is absent).
//! - E-e: `app-with-wgpu` under `cpu` fails with the one E0080 of `e-e-expected.txt`, raised by `gust-wgpu`.
//! - E-f: `app` under `cpu` with `--features gust/slang-runtime` fails in the build script of `gust` with the text of
//!   `e-f-expected.txt`.
//! - E-g: `optout` under `cpu` fails with the one E0277 of `e-g-expected.txt` (the opted-out kernel).
//! - E-h: in one warm target directory the profile flips `cpu`, unset, `cpu`, unset, `cpu`, unset; each `cpu` state
//!   must build and pass E-a's judgement, each unset state must fail like E-d.
//! - E-i: `cargo test -p kernels` under `cpu` compiles the kernels with `cfg(test)` (still one impl) and exits 0. The
//!   crate has no tests, so this proves the build, not any behaviour.

use std::{
    cell::RefCell,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::Instant,
};

use crate::{
    checks::{DualKind, DualStep, RUSTFLAGS_VARIABLES, dual_command, target_rustflags_variables},
    json::{self, Json},
    precompiled::count_occurrences,
    process::{PROFILE_VARIABLE, without_colors},
    ui::{Expectation, judge, parse_expectation},
    workspace::{cargo_target_dir, workspace_root},
};

/// The fixture workspace, relative to the repository root. It is a workspace of its own.
pub(crate) const CPU_TARGET_DIR: &str = "tests/cpu-target";

/// The value of `GUST_PROFILE` of the production CPU target.
pub(crate) const CPU_PROFILE: &str = "cpu";

/// The kernels `app` runs, as it names them in its `CPU target: matches (<kernel>, ..)` lines.
pub(crate) const KERNELS: [&str; 3] = [
    "vector_add::add",
    "polynomial::evaluate",
    "atomic_counter::increment",
];

/// What `app` prints last, after every kernel matched.
pub(crate) const FINAL_LINE: &str = "CPU target: all kernels match";

const MATCH_PREFIX: &str = "CPU target: matches (";

/// The byte strings that must not occur in the release binary (E-b), in any letter case: the compiler, the native
/// helper, the GPU runtime and the graphics API.
pub(crate) const FORBIDDEN: [&str; 4] = ["slangc", "gust-slang-reflect", "wgpu", "vulkan"];

/// A string the binary is known to hold, lower case: the scan finds it or it is blind (a scan that finds nothing in a
/// binary it cannot read proves nothing).
pub(crate) const SCAN_CONTROL: &str = "cpu target: matches";

/// How many offending lines of the tree a message shows.
const TREE_LINES_SHOWN: usize = 3;

/// Packages that `cargo tree -e features -p app` must list, so that an empty or unreadable tree cannot pass.
const TREE_PACKAGES: [&str; 3] = ["gust", "kernels", "app"];

/// The line of `crates/gust-wgpu/src/lib.rs` that E-e expects the error at (the file is `line: auto` in
/// `e-e-expected.txt`).
pub(crate) const REFUSAL_ITEM: &str = "const _: () = ::gust::profile::require_gpu_target();";

/// The source file of the refusal, relative to the repository root.
const REFUSAL_FILE: &str = "crates/gust-wgpu/src/lib.rs";

/// The profile states E-h walks through in one target directory. Both directions occur, twice and more: after
/// `cpu` the `unset` state must not reuse the `cpu` expansion, after `unset` the `cpu` state must not reuse the other.
pub(crate) const FLIPS: [Profile; 6] = [
    Profile::Cpu,
    Profile::Unset,
    Profile::Cpu,
    Profile::Unset,
    Profile::Cpu,
    Profile::Unset,
];

/// The `GUST_PROFILE` a cargo child gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Profile {
    /// `GUST_PROFILE=cpu`.
    Cpu,
    /// `GUST_PROFILE=dev`, what every workspace command of xtask uses.
    Dev,
    /// The variable is removed: a debug build then infers `dev`, and what matters for E-h is that the variable
    /// changed from `cpu` to nothing.
    Unset,
}

impl Profile {
    fn word(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Dev => "dev",
            Self::Unset => "unset",
        }
    }
}

/// One cargo call of the check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Call {
    /// The arguments after `cargo`.
    pub(crate) args: Vec<String>,
    pub(crate) profile: Profile,
    /// Whether the child gets `CARGO_ENCODED_RUSTFLAGS` with `--cfg gust_dual` (E-c only).
    pub(crate) cfg: bool,
}

impl Call {
    /// The command: `check-dual`'s own builder (no inherited rustflags, `GUST_PROFILE=dev`, then this call's
    /// environment), and the variable removed again for [`Profile::Unset`].
    pub(crate) fn command(&self, root: &Path, id: &'static str) -> Command {
        let step = DualStep {
            id,
            args: self.args.clone(),
            cfg: self.cfg,
            profile: (self.profile == Profile::Cpu).then_some(CPU_PROFILE),
            kind: DualKind::Build,
        };
        let mut command = dual_command(root, &step);
        if self.profile == Profile::Unset {
            command.env_remove(PROFILE_VARIABLE);
        }
        command
    }

    /// The command line for a message: no machine path (the target directory is shown as `<target>/..`).
    pub(crate) fn shown(&self) -> String {
        let mut env = String::new();
        if self.cfg {
            env.push_str("CARGO_ENCODED_RUSTFLAGS=\"--cfg gust_dual\" ");
        }
        match self.profile {
            Profile::Cpu => env.push_str(&format!("{PROFILE_VARIABLE}={CPU_PROFILE} ")),
            Profile::Dev => env.push_str(&format!("{PROFILE_VARIABLE}=dev ")),
            Profile::Unset => env.push_str(&format!("(no {PROFILE_VARIABLE}) ")),
        }
        let mut shown: Vec<String> = Vec::with_capacity(self.args.len());
        let mut args = self.args.iter();
        while let Some(arg) = args.next() {
            shown.push(arg.clone());
            if arg == "--target-dir"
                && let Some(directory) = args.next()
            {
                let last = directory.rsplit(['/', '\\']).next().unwrap_or(directory);
                shown.push(format!("<target>/{last}"));
            }
        }
        format!("{env}cargo {}", shown.join(" "))
    }
}

/// The target directories of the check, all under the cargo target directory so that the machine's own path never
/// reaches a message: one for the builds and checks, one for E-c (its rustflags would invalidate the others), one
/// for E-h (so that its flips start from a state of their own).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dirs {
    pub(crate) main: String,
    pub(crate) dual: String,
    pub(crate) flip: String,
}

impl Dirs {
    /// `<target>/cpu-target`, `<target>/cpu-target-dual` and `<target>/cpu-target-flip`.
    pub(crate) fn under(target: &str) -> Self {
        let target = target.trim_end_matches(['/', '\\']);
        Self {
            main: format!("{target}/cpu-target"),
            dual: format!("{target}/cpu-target-dual"),
            flip: format!("{target}/cpu-target-flip"),
        }
    }

    fn new(root: &Path) -> Result<Self, String> {
        let target = root.join(cargo_target_dir(root));
        let target = target.to_str().ok_or_else(|| {
            "check-cpu-target: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path"
                .to_owned()
        })?;
        Ok(Self::under(target))
    }
}

fn manifest() -> String {
    format!("{CPU_TARGET_DIR}/Cargo.toml")
}

fn call(args: &[&str], directory: Option<&str>, profile: Profile, cfg: bool) -> Call {
    let mut args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    if let Some(directory) = directory {
        args.push("--target-dir".to_owned());
        args.push(directory.to_owned());
    }
    Call { args, profile, cfg }
}

/// A `cargo <verb> --locked --color never --manifest-path <fixture> <rest>` call.
fn fixture_call(
    verb_and_flags: &[&str],
    rest: &[&str],
    directory: Option<&str>,
    profile: Profile,
    cfg: bool,
) -> Call {
    let manifest = manifest();
    let mut args: Vec<&str> = verb_and_flags.to_vec();
    args.extend(["--locked", "--color", "never", "--manifest-path", &manifest]);
    args.extend(rest);
    call(&args, directory, profile, cfg)
}

/// E-a: `app` in release under `cpu`.
pub(crate) fn e_a_build(dirs: &Dirs) -> Call {
    fixture_call(
        &["build", "--release"],
        &[
            "-p",
            "app",
            "--bin",
            "app",
            "--message-format",
            "json-render-diagnostics",
        ],
        Some(&dirs.main),
        Profile::Cpu,
        false,
    )
}

/// E-b: the features tree of `app`.
pub(crate) fn e_b_tree(_dirs: &Dirs) -> Call {
    fixture_call(
        &["tree"],
        &["-e", "features", "-p", "app"],
        None,
        Profile::Cpu,
        false,
    )
}

/// E-c: `app` under `cpu` with `--cfg gust_dual`, in the directory of its own.
pub(crate) fn e_c_build(dirs: &Dirs) -> Call {
    fixture_call(
        &["build"],
        &[
            "-p",
            "app",
            "--bin",
            "app",
            "--message-format",
            "json-render-diagnostics",
        ],
        Some(&dirs.dual),
        Profile::Cpu,
        true,
    )
}

/// E-d: the `single-dispatch` binary under `dev`, which must fail.
pub(crate) fn e_d_check(dirs: &Dirs) -> Call {
    fixture_call(
        &["check"],
        &[
            "-p",
            "app",
            "--bin",
            "single-dispatch",
            "--message-format",
            "json",
        ],
        Some(&dirs.main),
        Profile::Dev,
        false,
    )
}

/// E-e: the crate graph with the GPU runtime under `cpu`, which must fail.
pub(crate) fn e_e_check(dirs: &Dirs) -> Call {
    fixture_call(
        &["check"],
        &["-p", "app-with-wgpu", "--message-format", "json"],
        Some(&dirs.main),
        Profile::Cpu,
        false,
    )
}

/// E-f: `app` under `cpu` with the bridge feature, which the build script of `gust` must refuse.
pub(crate) fn e_f_check(dirs: &Dirs) -> Call {
    fixture_call(
        &["check"],
        &[
            "-p",
            "app",
            "--features",
            "gust/slang-runtime",
            "--message-format",
            "json",
        ],
        Some(&dirs.main),
        Profile::Cpu,
        false,
    )
}

/// E-g: the opted-out kernel's binary under `cpu`, which must fail.
pub(crate) fn e_g_check(dirs: &Dirs) -> Call {
    fixture_call(
        &["check"],
        &["-p", "optout", "--message-format", "json"],
        Some(&dirs.main),
        Profile::Cpu,
        false,
    )
}

/// One state of E-h: a debug build of `app` under `cpu`, or of `single-dispatch` with the variable removed.
pub(crate) fn e_h_build(dirs: &Dirs, profile: Profile) -> Call {
    match profile {
        Profile::Cpu => fixture_call(
            &["build"],
            &[
                "-p",
                "app",
                "--bin",
                "app",
                "--message-format",
                "json-render-diagnostics",
            ],
            Some(&dirs.flip),
            Profile::Cpu,
            false,
        ),
        other => fixture_call(
            &["build"],
            &[
                "-p",
                "app",
                "--bin",
                "single-dispatch",
                "--message-format",
                "json",
            ],
            Some(&dirs.flip),
            other,
            false,
        ),
    }
}

/// E-i: the test build of the kernels crate under `cpu`.
pub(crate) fn e_i_test(dirs: &Dirs) -> Call {
    fixture_call(
        &["test"],
        &["-p", "kernels"],
        Some(&dirs.main),
        Profile::Cpu,
        false,
    )
}

/// Every cargo call of the check, in order, for a test that pins them (E-h once per state of [`FLIPS`]).
#[cfg(test)]
pub(crate) fn all_calls(dirs: &Dirs) -> Vec<(&'static str, Call)> {
    let mut calls = vec![
        ("E-a", e_a_build(dirs)),
        ("E-b", e_b_tree(dirs)),
        ("E-c", e_c_build(dirs)),
        ("E-d", e_d_check(dirs)),
        ("E-e", e_e_check(dirs)),
        ("E-f", e_f_check(dirs)),
        ("E-g", e_g_check(dirs)),
    ];
    for profile in FLIPS {
        calls.push(("E-h", e_h_build(dirs, profile)));
    }
    calls.push(("E-i", e_i_test(dirs)));
    calls
}

// ---------------------------------------------------------------------------------------------------------------
// Judging

/// The path of the executable that a successful `cargo build --message-format json-render-diagnostics` wrote for
/// the binary target `name`: the `executable` of its `compiler-artifact` record. Exactly one such record must name
/// one; none (a blind output) or several (an ambiguous one) is an error, so a step never runs a binary that this
/// build did not write.
pub(crate) fn executable_of(stdout: &str, name: &str) -> Result<PathBuf, String> {
    let mut paths: Vec<String> = Vec::new();
    for line in stdout.lines().map(str::trim) {
        if !line.starts_with('{') {
            continue;
        }
        let Ok(message) = json::parse(line) else {
            continue;
        };
        if message.get("reason").and_then(Json::as_str) != Some("compiler-artifact") {
            continue;
        }
        let Some(target) = message.get("target") else {
            continue;
        };
        let is_binary = target
            .get("kind")
            .and_then(Json::as_array)
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if !is_binary || target.get("name").and_then(Json::as_str) != Some(name) {
            continue;
        }
        if let Some(path) = message.get("executable").and_then(Json::as_str) {
            paths.push(path.to_owned());
        }
    }
    match paths.as_slice() {
        [path] => Ok(PathBuf::from(path)),
        [] => Err(format!(
            "cargo's JSON output names no executable for the binary `{name}`, so there is no binary to run"
        )),
        many => Err(format!(
            "cargo's JSON output names {} executables for the binary `{name}`; expected one",
            many.len()
        )),
    }
}

/// The number of tests a `cargo test` stdout reports as passed, summed over its `test result: ok.` lines; `None`
/// when it has no such line.
pub(crate) fn passed_tests(stdout: &str) -> Option<u64> {
    let mut total = None;
    for line in stdout.lines() {
        let Some(rest) = line.trim().strip_prefix("test result: ok. ") else {
            continue;
        };
        let count = rest
            .split_whitespace()
            .next()
            .and_then(|word| word.parse::<u64>().ok())?;
        total = Some(total.unwrap_or(0) + count);
    }
    total
}

/// One line `CPU target: matches (<kernel>, <count> elements|invocations, max ULP <k>)` of `app`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MatchLine {
    pub(crate) kernel: String,
    pub(crate) count: u64,
    /// `elements` or `invocations` (the atomic counter has no elements).
    pub(crate) unit: String,
    pub(crate) max_ulp: u64,
}

/// The parts of a `matches` line, or `None` for any other text.
pub(crate) fn parse_match_line(line: &str) -> Option<MatchLine> {
    let inner = line.trim().strip_prefix(MATCH_PREFIX)?.strip_suffix(')')?;
    let mut parts = inner.split(", ");
    let kernel = parts.next()?;
    let counted = parts.next()?;
    let ulp = parts.next()?.strip_prefix("max ULP ")?;
    if parts.next().is_some() || kernel.is_empty() || kernel.contains(char::is_whitespace) {
        return None;
    }
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    let (count, unit) = counted.split_once(' ')?;
    if !digits(count) || !digits(ulp) || !matches!(unit, "elements" | "invocations") {
        return None;
    }
    Some(MatchLine {
        kernel: kernel.to_owned(),
        count: count.parse().ok()?,
        unit: unit.to_owned(),
        max_ulp: ulp.parse().ok()?,
    })
}

/// The problems of the standard output of a run of `app`: each of [`KERNELS`] has exactly one well-formed `matches`
/// line with a count above zero, no other kernel has one, no `matches` line is malformed, and [`FINAL_LINE`] occurs
/// exactly once. Empty when the output is what E-a requires.
pub(crate) fn match_problems(stdout: &str) -> Vec<String> {
    let stdout = without_colors(stdout);
    let mut problems = Vec::new();
    let mut found: Vec<MatchLine> = Vec::new();
    for line in stdout.lines().map(str::trim) {
        if !line.starts_with(MATCH_PREFIX) {
            continue;
        }
        match parse_match_line(line) {
            Some(parsed) => found.push(parsed),
            None => problems.push(format!("a malformed `matches` line: `{line}`")),
        }
    }
    for kernel in KERNELS {
        let lines: Vec<&MatchLine> = found.iter().filter(|line| line.kernel == kernel).collect();
        match lines.as_slice() {
            [] => problems.push(format!("no `matches` line for the kernel `{kernel}`")),
            [line] if line.count == 0 => problems.push(format!(
                "the kernel `{kernel}` matched 0 {}, which proves nothing",
                line.unit
            )),
            [_] => {}
            many => problems.push(format!(
                "{} `matches` lines for the kernel `{kernel}`, expected one",
                many.len()
            )),
        }
    }
    for line in &found {
        if !KERNELS.contains(&line.kernel.as_str()) {
            problems.push(format!(
                "a `matches` line for the unknown kernel `{}` (the fixture and the check disagree)",
                line.kernel
            ));
        }
    }
    match stdout
        .lines()
        .filter(|line| line.trim() == FINAL_LINE)
        .count()
    {
        1 => {}
        0 => problems.push(format!("the final line `{FINAL_LINE}` is missing")),
        many => problems.push(format!("the final line `{FINAL_LINE}` occurs {many} times")),
    }
    problems
}

/// The `matches` lines of `stdout` in the form of the summary of a step: `vector_add::add 1000 elements ULP 0`.
fn match_summary(stdout: &str) -> String {
    without_colors(stdout)
        .lines()
        .filter_map(parse_match_line)
        .map(|line| {
            format!(
                "{} {} {}, max ULP {}",
                line.kernel, line.count, line.unit, line.max_ulp
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// How often the control and each forbidden string occur in `bytes`, ignoring ASCII letter case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scan {
    pub(crate) control: usize,
    pub(crate) found: Vec<(&'static str, usize)>,
}

/// Counts [`SCAN_CONTROL`] and each of [`FORBIDDEN`] in `bytes`, ignoring ASCII letter case.
pub(crate) fn scan(bytes: &[u8]) -> Scan {
    let lower = bytes.to_ascii_lowercase();
    Scan {
        control: count_occurrences(&lower, SCAN_CONTROL.as_bytes()),
        found: FORBIDDEN
            .iter()
            .map(|marker| (*marker, count_occurrences(&lower, marker.as_bytes())))
            .collect(),
    }
}

/// The problems of a scan: a blind scan (the control is absent), or any forbidden string present.
pub(crate) fn scan_problems(scan: &Scan) -> Vec<String> {
    let mut problems = Vec::new();
    if scan.control == 0 {
        problems.push(format!(
            "the scan is blind: the binary does not hold `{SCAN_CONTROL}` (any letter case), so finding nothing else says nothing"
        ));
    }
    for (marker, count) in &scan.found {
        if *count > 0 {
            problems.push(format!(
                "the release binary contains `{marker}` ({count} time(s), any letter case)"
            ));
        }
    }
    problems
}

/// The problems of the output of `cargo tree -e features -p app`: no line may name `wgpu` (which also covers
/// `gust-wgpu`) or `slang-runtime`, and the tree must list [`TREE_PACKAGES`], so that an empty or unreadable output
/// cannot pass.
pub(crate) fn tree_problems(tree: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let named: Vec<&str> = tree
        .lines()
        .filter(|line| line.contains("wgpu") || line.contains("slang-runtime"))
        .collect();
    // A tree that holds the GPU runtime names it dozens of times; the first lines say enough.
    for line in named.iter().take(TREE_LINES_SHOWN) {
        problems.push(format!(
            "the GPU runtime or the bridge is in the tree: `{}`",
            line.trim()
        ));
    }
    if named.len() > TREE_LINES_SHOWN {
        problems.push(format!(
            "... and {} more lines of the tree name wgpu or slang-runtime",
            named.len() - TREE_LINES_SHOWN
        ));
    }
    for package in TREE_PACKAGES {
        let listed = tree.lines().any(|line| {
            line.trim_start_matches(|c: char| !c.is_alphanumeric())
                .starts_with(&format!("{package} v"))
        });
        if !listed {
            problems.push(format!(
                "the tree has no `{package} v..` line, so it was not read correctly"
            ));
        }
    }
    problems
}

/// The 1-based line of [`REFUSAL_ITEM`] in the text of `crates/gust-wgpu/src/lib.rs`: exactly one line that is the
/// item (not a comment about it).
pub(crate) fn refusal_line(source: &str) -> Result<usize, String> {
    let lines: Vec<usize> = source
        .lines()
        .enumerate()
        .filter(|(_, line)| line.trim() == REFUSAL_ITEM)
        .map(|(index, _)| index + 1)
        .collect();
    match lines.as_slice() {
        [line] => Ok(*line),
        [] => Err(format!(
            "{REFUSAL_FILE} has no line `{REFUSAL_ITEM}`, so `gust-wgpu` does not refuse GUST_PROFILE=cpu"
        )),
        many => Err(format!(
            "{REFUSAL_FILE} has {} lines `{REFUSAL_ITEM}`, expected one",
            many.len()
        )),
    }
}

/// `text` (an expectation file) with its `line: auto` line replaced by `line: <line>`. A file without that line is
/// returned unchanged.
pub(crate) fn with_located_line(text: &str, line: usize) -> String {
    let mut out = String::with_capacity(text.len());
    for raw in text.split_inclusive('\n') {
        if raw.trim() == "line: auto" {
            let ending = &raw[raw.trim_end().len()..];
            out.push_str(&format!("line: {line}{ending}"));
        } else {
            out.push_str(raw);
        }
    }
    out
}

/// `json` (the standard output of `cargo ... --message-format json`) with the repository root `root` taken off the
/// front of every string value that starts with it: cargo reports a path dependency outside the fixture workspace
/// with an absolute path, and an expectation file cannot name a machine's own path. Both slash spellings are
/// handled, and on Windows the drive letter's case does not matter.
pub(crate) fn relative_to(json: &str, root: &str) -> String {
    let trimmed = root.trim_end_matches(['/', '\\']);
    let escaped = trimmed.replace('\\', "\\\\");
    let forward = trimmed.replace('\\', "/");
    let mut text = json.to_owned();
    for prefix in [
        format!("\"{escaped}\\\\"),
        format!("\"{escaped}/"),
        format!("\"{forward}/"),
    ] {
        text = replace_ignoring_case(&text, &prefix, "\"");
    }
    text
}

/// `text` with every occurrence of `pattern` replaced by `with`, comparing ASCII letters without case (an ASCII
/// pattern only, so the byte offsets stay on character boundaries).
fn replace_ignoring_case(text: &str, pattern: &str, with: &str) -> String {
    let (haystack, needle) = (text.as_bytes(), pattern.as_bytes());
    if needle.is_empty() || !pattern.is_ascii() {
        return text.replace(pattern, with);
    }
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut index = 0;
    while index + needle.len() <= haystack.len() {
        if haystack[index..index + needle.len()].eq_ignore_ascii_case(needle) {
            out.push_str(&text[copied..index]);
            out.push_str(with);
            index += needle.len();
            copied = index;
        } else {
            index += 1;
        }
    }
    out.push_str(&text[copied..]);
    out
}

// ---------------------------------------------------------------------------------------------------------------
// Running

/// What the steps share.
struct Check {
    root: PathBuf,
    dirs: Dirs,
    /// The release `app` that E-a built (the path cargo reported), for E-b's byte scan.
    release_app: RefCell<Option<PathBuf>>,
}

/// The last lines of cargo's standard error, for the message of a failed build.
fn tail(stderr: &[u8]) -> String {
    let text = without_colors(&String::from_utf8_lossy(stderr));
    let lines: Vec<&str> = text.lines().collect();
    let from = lines.len().saturating_sub(40);
    lines[from..].join("\n")
}

fn exit_code(output: &Output) -> String {
    output
        .status
        .code()
        .map_or_else(|| "?".to_owned(), |code| code.to_string())
}

fn problems_error(problems: &[String]) -> String {
    format!(
        "found {} problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    )
}

/// Judges one run that had to fail: `text` is the expectation file called `name` (for messages), `scope` the
/// `(case, target)` of `ui::judge`, `root` the repository root to take off cargo's absolute paths ([`relative_to`]),
/// and `run` the `(success, stdout, stderr)` of the cargo call.
pub(crate) fn judge_run(
    name: &str,
    text: &str,
    scope: (&str, &str),
    root: &str,
    run: (bool, &str, &str),
) -> Result<(), String> {
    let expectation: Expectation =
        parse_expectation(text).map_err(|error| format!("{name}: {error}"))?;
    let (case, target) = scope;
    let (success, stdout, stderr) = run;
    let problems = judge(
        case,
        target,
        &expectation,
        success,
        &relative_to(stdout, root),
        stderr,
    );
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems_error(&problems))
    }
}

impl Check {
    fn output(&self, id: &'static str, call: &Call) -> Result<Output, String> {
        call.command(&self.root, id)
            .output()
            .map_err(|error| format!("could not launch cargo: {error}"))
    }

    /// Runs a call that must succeed.
    fn must_succeed(&self, id: &'static str, call: &Call) -> Result<Output, String> {
        let output = self.output(id, call)?;
        if output.status.success() {
            return Ok(output);
        }
        Err(format!(
            "`{}` failed (exit {}):\n{}",
            call.shown(),
            exit_code(&output),
            tail(&output.stderr)
        ))
    }

    /// The expectation file `name` of the fixture.
    fn expectation_text(&self, name: &str) -> Result<String, String> {
        let path = self.root.join(CPU_TARGET_DIR).join(name);
        fs::read_to_string(&path).map_err(|error| format!("{CPU_TARGET_DIR}/{name}: {error}"))
    }

    /// Judges the output of a call that must fail against the expectation file `name`, as `check-ui` and
    /// `check-dual` do. `case` is the directory the expected `file:` is relative to, `target` the cargo target
    /// whose error counts.
    fn judge_failure(
        &self,
        name: &str,
        case: &str,
        target: &str,
        text: &str,
        output: &Output,
    ) -> Result<(), String> {
        judge_run(
            &format!("{CPU_TARGET_DIR}/{name}"),
            text,
            (case, target),
            &self.root.to_string_lossy(),
            (
                output.status.success(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            ),
        )
    }

    /// A call that must fail, judged against `name`.
    fn fails_like(
        &self,
        id: &'static str,
        call: &Call,
        name: &str,
        case: &str,
        target: &str,
    ) -> Result<(), String> {
        let text = self.expectation_text(name)?;
        let output = self.output(id, call)?;
        self.judge_failure(name, case, target, &text, &output)
    }

    /// Builds `app` with `call` (a `cargo build` that prints cargo's JSON) and returns the path cargo reports for
    /// the binary it wrote.
    fn build_app(&self, id: &'static str, call: &Call) -> Result<PathBuf, String> {
        let output = self.must_succeed(id, call)?;
        executable_of(&String::from_utf8_lossy(&output.stdout), "app")
    }

    /// Runs the built `app` at `path` and judges it with [`match_problems`]; returns the summary of its `matches`
    /// lines.
    fn app_matches(&self, path: &Path) -> Result<String, String> {
        let output = Command::new(path)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| format!("could not run the built `app`: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let mut problems = match_problems(&stdout);
        if !output.status.success() {
            problems.insert(
                0,
                format!(
                    "`app` exited with {}: {}",
                    exit_code(&output),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            );
        }
        if problems.is_empty() {
            Ok(match_summary(&stdout))
        } else {
            Err(problems_error(&problems))
        }
    }

    fn e_a(&self) -> Result<String, String> {
        let path = self.build_app("E-a", &e_a_build(&self.dirs))?;
        let summary = self.app_matches(&path)?;
        *self.release_app.borrow_mut() = Some(path);
        Ok(format!(
            "release `app` under GUST_PROFILE=cpu ran and printed the final line; {summary}"
        ))
    }

    fn e_b(&self) -> Result<String, String> {
        let path = self
            .release_app
            .borrow()
            .clone()
            .ok_or("E-b scans the release `app` that E-a built, and E-a has not built one")?;
        let bytes = fs::read(&path)
            .map_err(|error| format!("could not read the release `app` of E-a: {error}"))?;
        let report = scan(&bytes);
        let problems = scan_problems(&report);
        if !problems.is_empty() {
            return Err(problems_error(&problems));
        }
        let output = self.must_succeed("E-b", &e_b_tree(&self.dirs))?;
        let tree = String::from_utf8_lossy(&output.stdout).into_owned();
        let problems = tree_problems(&tree);
        if !problems.is_empty() {
            return Err(problems_error(&problems));
        }
        Ok(format!(
            "the release binary ({} bytes) holds none of {} in any letter case (the scan found its control string {} time(s)), and `cargo tree -e features -p app` ({} lines) shows no wgpu and no slang-runtime",
            bytes.len(),
            FORBIDDEN.map(|marker| format!("`{marker}`")).join(", "),
            report.control,
            tree.lines().count()
        ))
    }

    fn e_c(&self) -> Result<String, String> {
        let path = self.build_app("E-c", &e_c_build(&self.dirs))?;
        let summary = self.app_matches(&path)?;
        Ok(format!(
            "under cpu with --cfg gust_dual `app` builds (one impl, no E0119) and runs; {summary}"
        ))
    }

    fn e_d(&self) -> Result<String, String> {
        self.fails_like(
            "E-d",
            &e_d_check(&self.dirs),
            "e-d-expected.txt",
            "app",
            "single-dispatch",
        )?;
        Ok("without the profile `single-dispatch` fails with the one E0277 of e-d-expected.txt (no CPU twin in a production build)".to_owned())
    }

    fn e_e(&self) -> Result<String, String> {
        let text = self.expectation_text("e-e-expected.txt")?;
        let output = self.output("E-e", &e_e_check(&self.dirs))?;
        // A build that succeeded is the failure to report, whatever line the item is on (a placeholder, since the
        // expectation needs a number).
        let line = if output.status.success() {
            1
        } else {
            let source = fs::read_to_string(self.root.join(REFUSAL_FILE))
                .map_err(|error| format!("{REFUSAL_FILE}: {error}"))?;
            refusal_line(&source)?
        };
        let text = with_located_line(&text, line);
        self.judge_failure(
            "e-e-expected.txt",
            "crates/gust-wgpu",
            "gust_wgpu",
            &text,
            &output,
        )?;
        Ok(format!(
            "`app-with-wgpu` under cpu fails with the one E0080 of e-e-expected.txt, raised at {REFUSAL_FILE}:{line} by `gust-wgpu`'s `require_gpu_target`"
        ))
    }

    fn e_f(&self) -> Result<String, String> {
        self.fails_like(
            "E-f",
            &e_f_check(&self.dirs),
            "e-f-expected.txt",
            "app",
            "app",
        )?;
        Ok("`app --features gust/slang-runtime` under cpu fails in the build script of `gust` with the text of e-f-expected.txt".to_owned())
    }

    fn e_g(&self) -> Result<String, String> {
        self.fails_like(
            "E-g",
            &e_g_check(&self.dirs),
            "e-g-expected.txt",
            "optout",
            "optout",
        )?;
        Ok("`optout` under cpu fails with the one E0277 of e-g-expected.txt (the kernel opted out of the CPU)".to_owned())
    }

    fn e_h(&self) -> Result<String, String> {
        let mut states = Vec::new();
        for (index, profile) in FLIPS.into_iter().enumerate() {
            let number = index + 1;
            let call = e_h_build(&self.dirs, profile);
            let state = format!("state {number} ({})", profile.word());
            match profile {
                Profile::Cpu => {
                    let path = self
                        .build_app("E-h", &call)
                        .map_err(|error| format!("{state}: {error}"))?;
                    self.app_matches(&path)
                        .map_err(|error| format!("{state}: {error}"))?;
                }
                _ => {
                    self.fails_like("E-h", &call, "e-d-expected.txt", "app", "single-dispatch")
                        .map_err(|error| {
                            format!("{state}: the build did not fail like E-d: {error}")
                        })?;
                }
            }
            states.push(profile.word());
        }
        Ok(format!(
            "in one target directory the profile went {}: every cpu state built and matched, every unset state failed like E-d",
            states.join(" -> ")
        ))
    }

    fn e_i(&self) -> Result<String, String> {
        let output = self.must_succeed("E-i", &e_i_test(&self.dirs))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let Some(passed) = passed_tests(&stdout) else {
            return Err(format!(
                "`cargo test -p kernels` succeeded but printed no `test result: ok.` line:\n{stdout}"
            ));
        };
        Ok(format!(
            "`cargo test -p kernels` under cpu compiled the kernels with cfg(test) (one impl) and ran {passed} test(s){}",
            if passed == 0 {
                "; the crate has none, so this proves that it builds, not any behaviour"
            } else {
                ""
            }
        ))
    }
}

/// A step: it runs against the checked-out workspace and returns the summary of its result.
type StepFunction = fn(&Check) -> Result<String, String>;

/// The steps in order: id, and the function that runs it and returns the summary of its result.
const STEPS: [(&str, StepFunction); 9] = [
    ("E-a", Check::e_a),
    ("E-b", Check::e_b),
    ("E-c", Check::e_c),
    ("E-d", Check::e_d),
    ("E-e", Check::e_e),
    ("E-f", Check::e_f),
    ("E-g", Check::e_g),
    ("E-h", Check::e_h),
    ("E-i", Check::e_i),
];

/// `cargo xtask check-cpu-target`: see the module documentation. Needs cargo only. The fixture builds in
/// `<cargo target dir>/cpu-target`, `cpu-target-dual` and `cpu-target-flip`.
pub(crate) fn check_cpu_target(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("check-cpu-target takes no arguments".into());
    }
    let root = workspace_root()?;
    let set: Vec<&str> = RUSTFLAGS_VARIABLES
        .iter()
        .copied()
        .filter(|name| env::var_os(name).is_some())
        .collect();
    if !set.is_empty() {
        println!(
            "note: {} set in the environment; every child of check-cpu-target starts without any rustflags variable (E-c sets its own), so those flags are not part of this check",
            set.join(", ")
        );
    }
    let target_flags = target_rustflags_variables(env::vars_os().map(|(name, _)| name));
    if !target_flags.is_empty() {
        println!(
            "note: {} set in the environment; cargo reads them as rustflags config for their target, so a flag there applies to the builds of this check (E-c's own cfg replaces them) and can make a step fail",
            target_flags.join(", ")
        );
    }
    let check = Check {
        dirs: Dirs::new(&root)?,
        root,
        release_app: RefCell::new(None),
    };
    let started = Instant::now();
    for (id, step) in STEPS {
        let step_started = Instant::now();
        let summary =
            step(&check).map_err(|error| format!("check-cpu-target {id} failed: {error}"))?;
        println!(
            "check-cpu-target {id}: ok ({:.1} s) {summary}",
            step_started.elapsed().as_secs_f64()
        );
    }
    println!(
        "check-cpu-target passed in {:.1} s: the production CPU target (GUST_PROFILE=cpu) built and ran the fixture kernels against host references with cargo only, its binary holds no compiler, helper or GPU runtime, and `gust-wgpu` refuses the profile. This is Cpu evidence (D42), never GPU evidence",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
#[path = "tests/cpu_target.rs"]
mod cpu_target_tests;
