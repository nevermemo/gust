use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use crate::{
    process::{
        command_exists, print_command, require_command, run_checked, run_direct, run_direct_args,
    },
    reflect::{BuildOptions, build_slang_reflect},
    shader::doctor_smoke,
    ui::{Expectation, judge, parse_expected},
    validation::{compact_validation_summary, json_string},
    verify::{VerifyMode, verify},
    workspace::{EXAMPLES, cargo_target_dir, changed_paths, workspace_root, workspace_test_args},
};

/// The cargo calls of `check-feature reflection` after the native helper is built (the
/// Reflect suite). Kept as data so a test can require `--locked` on each.
pub(crate) const REFLECTION_TESTS: &[&[&str]] = &[
    &["test", "--locked", "-p", "gust", "--test", "reflection"],
    &[
        "test",
        "--locked",
        "-p",
        "gust-wgpu",
        "--test",
        "reflection",
    ],
];

/// The cargo calls of `cargo xtask check-cpu` (a step of the Unit suite): the non-ignored,
/// GPU-free tests of the examples, which run each kernel's CPU twin against its host reference.
/// Later slices append their examples. Kept as data so a test can require `--locked` on each.
pub(crate) const CPU_TESTS: &[&[&str]] = &[
    &["test", "--locked", "-p", "vector-add"],
    &["test", "--locked", "-p", "polynomial"],
];

/// `cargo xtask check-cpu`: the CPU-twin evidence, needing neither slangc nor a GPU.
pub(crate) fn check_cpu() -> Result<(), String> {
    let root = workspace_root()?;
    for args in CPU_TESTS {
        run_direct(&root, "cargo", args)?;
    }
    Ok(())
}

pub(crate) fn check_feature(area: Option<&str>) -> Result<(), String> {
    let root = workspace_root()?;
    let area = area.ok_or("check-feature needs an area")?;
    match area {
        "macro" => run_direct(&root, "cargo", &["test", "-p", "gust-macros"]),
        "core" => {
            run_direct(&root, "cargo", &["test", "-p", "gust", "--lib"])?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust", "--test", "diagnostics"],
            )
        }
        "wgpu" => verify(VerifyMode::Gpu, false),
        "gpu-smoke" => run_direct(&root, "cargo", &["test", "-p", "gust-wgpu", "--lib"]),
        "gpu-semantics" => {
            for test in [
                "semantics",
                "numeric",
                "option",
                "result",
                "loops",
                "struct_assignment",
                "mismatch",
            ] {
                run_direct(&root, "cargo", &["test", "-p", "gust-wgpu", "--test", test])?;
            }
            Ok(())
        }
        "gpu-runtime" => run_direct(
            &root,
            "cargo",
            &[
                "test",
                "-p",
                "gust-wgpu",
                "--test",
                "reflection",
                "--test",
                "limits",
                "--test",
                "typed",
                "--test",
                "graph",
            ],
        ),
        "reflection" => {
            build_slang_reflect(BuildOptions::default())?;
            for args in REFLECTION_TESTS {
                run_direct(&root, "cargo", args)?;
            }
            Ok(())
        }
        "loops" => {
            run_direct(&root, "cargo", &["test", "-p", "gust-macros", "loops"])?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-wgpu", "--test", "loops"],
            )
        }
        "result" => {
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-macros", "--", "result"],
            )?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-wgpu", "--test", "result"],
            )
        }
        "examples" => verify(VerifyMode::Examples, false),
        "artifacts" => verify(VerifyMode::Artifacts, false),
        "full" => verify(VerifyMode::Full, true),
        _ => Err(format!("unknown feature area `{area}`")),
    }
}

pub(crate) fn check_workspace() -> Result<(), String> {
    let root = workspace_root()?;
    run_direct_args(&root, "cargo", &workspace_test_args())
}

pub(crate) fn check_format() -> Result<(), String> {
    run_direct(
        &workspace_root()?,
        "cargo",
        &["fmt", "--all", "--", "--check"],
    )
}

pub(crate) fn check_lints() -> Result<(), String> {
    run_direct(
        &workspace_root()?,
        "cargo",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )
}

pub(crate) fn check_changed() -> Result<(), String> {
    let paths = changed_paths()?;
    if paths.is_empty() {
        println!("No changed files; suggested check: cargo xtask check-fast");
        return Ok(());
    }
    println!("Changed files:");
    for path in &paths {
        println!("  {path}");
    }
    for check in select_changed_checks(&paths) {
        match check {
            ChangedCheck::Fast => {
                println!("Selected check: cargo xtask check-fast");
                verify(VerifyMode::Fast, false)?;
            }
            ChangedCheck::Feature(area) => {
                println!("Selected check: cargo xtask check-feature {area}");
                check_feature(Some(area))?;
            }
            ChangedCheck::Workspace => {
                println!("Selected check: cargo xtask check-workspace");
                check_workspace()?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ChangedCheck {
    Fast,
    Feature(&'static str),
    Workspace,
}

fn select_changed_checks(paths: &[String]) -> Vec<ChangedCheck> {
    if paths.is_empty() {
        return Vec::new();
    }
    let has = |prefix: &str| paths.iter().any(|path| path.starts_with(prefix));
    let any = |needles: &[&str]| {
        paths
            .iter()
            .any(|path| needles.iter().any(|needle| path.contains(needle)))
    };

    if any(&[
        "Cargo.toml",
        "Cargo.lock",
        "xtask/",
        ".cargo/",
        ".github/",
        ".vscode/",
        "AGENTS.md",
    ]) {
        return vec![ChangedCheck::Fast];
    }

    let mut checks = Vec::new();
    if any(&["tests/fixtures/loops", "crates/gust-wgpu/tests/loops.rs"]) {
        checks.push(ChangedCheck::Feature("loops"));
    }
    if (has("crates/gust-macros/") || has("tests/fixtures/"))
        && !checks.contains(&ChangedCheck::Feature("loops"))
    {
        checks.push(ChangedCheck::Feature("macro"));
    }
    if any(&[
        "crates/gust/src/reflect/",
        "reflect.rs",
        "reflection.rs",
        "scripts/probes/",
    ]) {
        checks.push(ChangedCheck::Feature("reflection"));
    }
    if has("crates/gust-wgpu/") {
        checks.push(ChangedCheck::Feature("wgpu"));
    }
    if has("crates/gust/")
        && !has("crates/gust/src/reflect/")
        && !any(&["reflect.rs", "reflection.rs"])
    {
        checks.push(ChangedCheck::Feature("core"));
    }
    if has("examples/") || has("generated-wgpu/") {
        checks.push(ChangedCheck::Feature("examples"));
    }
    if checks.is_empty() {
        vec![ChangedCheck::Workspace]
    } else {
        checks
    }
}

#[cfg(test)]
#[path = "tests/check_routing.rs"]
mod check_routing_tests;

pub(crate) fn status() -> Result<(), String> {
    let root = workspace_root()?;
    println!("Repository: {}", root.display());
    print_command(&root, "git", &["status", "--short", "--branch"])?;
    if let Ok(text) = fs::read_to_string(root.join("docs/development/STATUS.md")) {
        let active = text.lines().find(|line| line.starts_with("## Active:"));
        if let Some(active) = active {
            println!("{active}");
        }
        for line in text.lines().skip_while(|line| !line.starts_with("```text")) {
            println!("{line}");
            if line == "```" {
                break;
            }
        }
    }
    if let Ok(text) = fs::read_to_string(root.join(".ai/VALIDATION.json")) {
        println!("Validation: {}", compact_validation_summary(&text));
    }
    println!("Suggested clean-tree check: cargo xtask check-fast");
    Ok(())
}

/// Reports tool readiness and fails when a required tool is missing. `--smoke` also compiles
/// a hand-written Slang kernel to WGSL and SPIR-V and validates the SPIR-V.
pub(crate) fn doctor(args: &[String]) -> Result<(), String> {
    let smoke = match args {
        [] => false,
        [flag] if flag == "--smoke" => true,
        _ => return Err("doctor takes only `--smoke`".into()),
    };
    let mut missing = Vec::new();
    println!("Toolchain doctor:");
    for command in ["cargo", "rustc", "slangc"] {
        if command_exists(command) {
            println!("  {command}: found");
        } else {
            println!("  {command}: missing");
            missing.push(command);
        }
    }
    println!(
        "  spirv-val: {}",
        if command_exists("spirv-val") {
            "found"
        } else {
            "missing (needed for artifacts/full)"
        }
    );
    match build_slang_reflect(BuildOptions::default()) {
        Ok(()) => println!("  gust-slang-reflect: ready"),
        Err(error) => {
            println!("  gust-slang-reflect: not ready ({error})");
            missing.push("gust-slang-reflect");
        }
    }
    println!(
        "  Vulkan/wgpu adapter: run `cargo xtask check-feature gpu-smoke` for a cheap runtime check"
    );
    if !missing.is_empty() {
        return Err(format!(
            "required tools missing or not ready: {}",
            missing.join(", ")
        ));
    }
    if smoke {
        doctor_smoke(&workspace_root()?)?;
    } else {
        println!("  Slang compile smoke: run `cargo xtask doctor --smoke`");
    }
    Ok(())
}

pub(crate) fn list_tests() -> Result<(), String> {
    let root = workspace_root()?;
    let output = Command::new("cargo")
        .args(["test", "--workspace", "--", "--list"])
        .current_dir(&root)
        .output()
        .map_err(|error| format!("could not launch cargo test -- --list: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut total_tests = 0usize;
    let mut doctests = 0usize;
    let mut ignored_examples = 0usize;
    for line in stdout.lines() {
        if line.ends_with(": test") {
            total_tests += 1;
            if line.contains("tests::") && line.contains("example validation") {
                ignored_examples += 1;
            }
        }
        if line.contains(" - (line ") {
            doctests += 1;
        }
    }
    println!("cargo test --workspace -- --list");
    println!("  listed Rust tests: {total_tests}");
    println!("  listed doctests: {doctests}");
    println!("  example tests are ignored by attribute and run by `cargo xtask check-examples`");
    println!(
        "  categories: macro golden/rejection, core bridge/reflection, wgpu GPU/runtime, ignored examples, doctests"
    );
    if ignored_examples > 0 {
        println!("  ignored example test-name matches in list: {ignored_examples}");
    }
    Ok(())
}

pub(crate) fn explain_check(area: Option<&str>) -> Result<(), String> {
    let area = area.unwrap_or("all");
    let message = match area {
        "macro" => "macro: validator/emitter/golden changes; fastest compiler-front-end proof",
        "core" => {
            "core: ABI, Slang bridge, reflection parser, diagnostic mapping, SPIR-V structure helpers"
        }
        "wgpu" => "wgpu: all runtime/GPU integration tests; about 10s on this machine",
        "gpu-smoke" => "gpu-smoke: wgpu crate unit tests only; cheap runtime/cache sanity",
        "gpu-semantics" => {
            "gpu-semantics: semantics/numeric/option/loops/struct assignment GPU differential tests and the known-mismatch ledger"
        }
        "gpu-runtime" => {
            "gpu-runtime: reflection gate, cache/runtime mismatch tests, launch-limit checks and buffer layout assertions, typed calls and derived graph edges"
        }
        "reflection" => "reflection: native Slang helper + core/wgpu reflection contract",
        "loops" => "loops: bounded-loop golden/rejection test plus real-GPU loop differential test",
        "result" => {
            "result: Result lowering/match regressions plus WGSL, SPIR-V, and real-GPU Result tests"
        }
        "examples" => {
            "examples: ignored example tests and example binaries; use for major behavior confidence"
        }
        "artifacts" => "artifacts: example binaries plus exported SPIR-V validation",
        "full" => "full: release evidence; writes .ai/VALIDATION.json",
        "changed" => "changed: route dirty files to the cheapest likely sufficient check",
        "workspace" => {
            "workspace: non-example workspace tests via --exclude; example tests stay in check-examples"
        }
        "format" => "format: rustfmt only",
        "lints" => "lints: strict Clippy over workspace/all targets",
        "lint" => {
            "lint: no tracked shell scripts (D27) and no machine-specific paths in tracked text (home paths include WSL spellings); also part of check-fast"
        }
        "docs-check" => {
            "docs-check: relative Markdown links resolve to project files (anchors and external links are not checked), no merge-conflict markers, DECISIONS ids unique and increasing; also part of check-fast"
        }
        "check-arch" => {
            "check-arch: per-crate dependency allow-list (D02), no IR module or file outside future/, no SPIR-V library in the compiler crates; also part of check-fast"
        }
        "check-ui" => {
            "check-ui: cargo check of each tests/ui fixture crate; a crate with an expected.txt must fail with exactly one error satisfying every key (`code: none` names a macro error without a rustc code and needs a `message:`); also part of check-fast and the Unit suite"
        }
        "check-cpu" => {
            "check-cpu: `cargo test --locked` of the examples with a CPU twin (vector-add, polynomial), running the non-ignored twin tests against the host references; needs neither slangc nor a GPU; a step of the Unit suite"
        }
        "check-dual" => {
            "check-dual: the dual test build (`--cfg gust_dual`, D26) and its cross-crate proof, with its own target directory <target>/dual: the workspace builds with the cfg; in tests/dual a consumer's test (E1) and production binary (E3) each fail with exactly one E0277 for the missing CPU twin of the dependency `kernel-lib` without the cfg, and with it (E2) the test passes. Refuses to run while RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS or CARGO_BUILD_RUSTFLAGS is set (it delivers the cfg through CARGO_ENCODED_RUSTFLAGS). `--check-only` is the CPU-only mode CI will run; needs cargo only"
        }
        "bless" => {
            "bless: reruns the Slang golden tests with BLESS_GOLDEN=1, which rewrites changed fixtures and fails with the diff; review before committing"
        }
        "test" => {
            "test: `cargo xtask test --suite Unit|Slangc|Spirv|Reflect|Gpu|Full` runs one evidence tier of VALIDATION.md"
        }
        "all" => {
            "known areas: macro, core, wgpu, gpu-smoke, gpu-semantics, gpu-runtime, reflection, loops, examples, artifacts, full, changed, workspace, format, lints, lint, docs-check, check-arch, check-ui, check-cpu, check-dual, bless, test"
        }
        _ => return Err(format!("unknown check area `{area}`")),
    };
    println!("{message}");
    Ok(())
}

pub(crate) fn export_artifacts() -> Result<(), String> {
    let root = workspace_root()?;
    for example in EXAMPLES {
        println!("Exporting artifacts via {example}");
        run_direct(&root, "cargo", &["run", "--quiet", "-p", example])?;
    }
    Ok(())
}

pub(crate) fn measure_tests(args: &[String]) -> Result<(), String> {
    let root = workspace_root()?;
    let mut json_path = None;
    let mut command_args = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                index += 1;
                json_path = Some(PathBuf::from(args.get(index).ok_or("--json needs a path")?));
            }
            other => command_args.push(other.to_owned()),
        }
        index += 1;
    }
    let commands = if command_args.is_empty() {
        let mut workspace = vec!["cargo".to_owned()];
        workspace.extend(workspace_test_args());
        vec![
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust-macros".into(),
            ],
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust".into(),
                "--lib".into(),
            ],
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust-wgpu".into(),
            ],
            workspace,
        ]
    } else {
        vec![command_args]
    };
    let mut records = Vec::new();
    for command in commands {
        if command.is_empty() {
            return Err("measure-tests command cannot be empty".into());
        }
        println!("== {}", command.join(" "));
        let start = Instant::now();
        let status = Command::new(&command[0])
            .args(&command[1..])
            .current_dir(&root)
            .status()
            .map_err(|error| format!("could not launch {}: {error}", command[0]))?;
        let elapsed = start.elapsed().as_millis();
        let exit = status.code().unwrap_or(-1);
        println!("elapsed_ms={elapsed} exit={exit}");
        records.push((command.join(" "), elapsed, exit));
        if !status.success() {
            return Err(format!("{} failed", command.join(" ")));
        }
    }
    if let Some(path) = json_path {
        let mut json = String::from("{\n  \"schema_version\": 1,\n  \"measurements\": [\n");
        for (index, (command, elapsed, exit)) in records.iter().enumerate() {
            if index != 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "    {{ \"command\": {}, \"elapsed_ms\": {elapsed}, \"exit_code\": {exit} }}",
                json_string(command)
            ));
        }
        json.push_str("\n  ]\n}\n");
        fs::write(path, json).map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn hook_format_rust_after_edit() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| error.to_string())?;
    if !input.contains("PostToolUse") || !input.contains(".rs") {
        return Ok(());
    }
    let edit_tool = [
        "create_file",
        "replace_string_in_file",
        "multi_replace_string_in_file",
        "insert_edit_into_file",
        "edit_file",
        "apply_patch",
    ]
    .iter()
    .any(|tool| input.contains(tool));
    if edit_tool {
        run_direct(&workspace_root()?, "cargo", &["fmt", "--all"])?;
    }
    Ok(())
}

pub(crate) fn self_test() -> Result<(), String> {
    require_command("cargo")?;
    require_command("rustc")?;
    require_command("slangc")?;
    let root = workspace_root()?;
    let mut checks = Vec::new();
    run_checked(&root, &mut checks, "rustc", &["--version"])?;
    if checks.len() != 1 || !checks[0].stdout.contains("rustc") {
        return Err("command capture self-test failed".into());
    }
    println!("PASS: xtask command capture and tool lookup work.");
    Ok(())
}

// ---- check-dual (m3.f1): the dual test build and its cross-crate fixture ----

/// The fixture workspace of the cross-crate proof, relative to the repository root. It has a `[workspace]` of its
/// own, so the root workspace neither contains it nor builds it with `--workspace`.
pub(crate) const DUAL_DIR: &str = "tests/dual";

/// The value of `CARGO_ENCODED_RUSTFLAGS` that delivers `--cfg gust_dual`: two flags joined by the unit separator
/// (0x1f) that cargo splits that variable at.
pub(crate) const DUAL_RUSTFLAGS: &str = "--cfg\u{1f}gust_dual";

/// The variable that carries the cfg to the child cargo processes.
pub(crate) const DUAL_RUSTFLAGS_VARIABLE: &str = "CARGO_ENCODED_RUSTFLAGS";

/// The environment variables that decide rustflags. `check-dual` refuses to run while any of them is present, even
/// with an empty value, because it sets `CARGO_ENCODED_RUSTFLAGS` itself and that takes precedence over the rest.
pub(crate) const RUSTFLAGS_VARIABLES: [&str; 3] = [
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS",
];

/// The tests of `consumer` that E2 requires to run and pass: the twin on a dependency's kernel, and the overflow
/// mode of a dependent with a per-package override.
pub(crate) const DUAL_TESTS: &[&str] = &[
    "tests::twin_runs",
    "tests::overflow_mode_is_the_dependents_own",
];

/// What `cargo xtask check-dual` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DualMode {
    /// `--check-only`: CPU only, no GPU; the mode CI runs.
    CheckOnly,
    /// No argument: the same steps; the GPU runs of the examples arrive in m3.f2.
    Full,
}

/// How one step of `check-dual` is judged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DualKind {
    /// Must succeed; its output goes to the terminal.
    Build,
    /// Must fail with exactly the one error of the expectation file (relative to [`DUAL_DIR`]) on the cargo target
    /// `target` of the package in the directory `case`, judged by `ui::judge` like a `check-ui` fixture.
    Fails {
        case: &'static str,
        target: &'static str,
        expected: &'static str,
    },
    /// E2, first half: must compile with the cfg. A failure means the cfg did not reach the dependency.
    Compiles,
    /// E2, second half: must succeed, and each named test must be reported `ok` (a run that finds no test passes
    /// vacuously, so the names are required).
    Passes { tests: &'static [&'static str] },
}

/// One cargo call of `check-dual`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DualStep {
    /// `build`, `E1`, `E2-compile`, `E2-run` or `E3`.
    pub(crate) id: &'static str,
    /// The arguments after `cargo`; the last two are always `--target-dir <dir>`.
    pub(crate) args: Vec<String>,
    /// Whether the child gets `CARGO_ENCODED_RUSTFLAGS` with `--cfg gust_dual`.
    pub(crate) cfg: bool,
    pub(crate) kind: DualKind,
}

impl DualStep {
    /// The environment this step adds to the child's: the cfg, or nothing (E1 and E3 build without it).
    pub(crate) fn env(&self) -> Vec<(&'static str, &'static str)> {
        if self.cfg {
            vec![(DUAL_RUSTFLAGS_VARIABLE, DUAL_RUSTFLAGS)]
        } else {
            Vec::new()
        }
    }

    /// The command line for the terminal: the arguments without the target directory (a machine-specific path).
    fn shown(&self) -> String {
        let keep = self.args.len().saturating_sub(2);
        let env = if self.cfg {
            "CARGO_ENCODED_RUSTFLAGS=\"--cfg gust_dual\" "
        } else {
            ""
        };
        format!(
            "{env}cargo {} --target-dir <target>/dual",
            self.args[..keep].join(" ")
        )
    }
}

/// `check-dual`'s own target directory: `<cargo target dir>/dual`, so that the changed rustflags never invalidate
/// the cache of the normal build. A relative `CARGO_TARGET_DIR` is read against the repository root, where the
/// child cargo processes run.
pub(crate) fn dual_target_dir(root: &Path) -> PathBuf {
    dual_dir_under(root, &cargo_target_dir(root))
}

/// `<target>/dual` for the cargo target directory `target` (absolute, or relative to `root`).
pub(crate) fn dual_dir_under(root: &Path, target: &Path) -> PathBuf {
    root.join(target).join("dual")
}

/// The cargo calls of `check-dual`, in order: the workspace build with the cfg, then E1, E2 and E3 on the
/// `tests/dual` fixture. Every call is `--locked`. Kept as data so a test can require the exact lists.
pub(crate) fn dual_steps(target_dir: &str) -> Vec<DualStep> {
    let target_dir = target_dir.to_owned();
    let manifest = format!("{DUAL_DIR}/Cargo.toml");
    let strings = |args: &[&str]| -> Vec<String> {
        let mut args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
        args.push("--target-dir".to_owned());
        args.push(target_dir.clone());
        args
    };
    vec![
        DualStep {
            id: "build",
            args: strings(&["build", "--locked", "--workspace", "--all-targets"]),
            cfg: true,
            kind: DualKind::Build,
        },
        // E1: without the cfg the consumer's own `cfg(test)` does not give `kernel-lib` its twin.
        DualStep {
            id: "E1",
            args: strings(&[
                "test",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
                "--message-format",
                "json",
            ]),
            cfg: false,
            kind: DualKind::Fails {
                case: "consumer",
                target: "consumer",
                expected: "e1-expected.txt",
            },
        },
        // E2: with the cfg the same test builds, and runs.
        DualStep {
            id: "E2-compile",
            args: strings(&[
                "test",
                "--locked",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
            ]),
            cfg: true,
            kind: DualKind::Compiles,
        },
        DualStep {
            id: "E2-run",
            args: strings(&[
                "test",
                "--locked",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
            ]),
            cfg: true,
            kind: DualKind::Passes { tests: DUAL_TESTS },
        },
        // E3: without the cfg a production binary gets no twin.
        DualStep {
            id: "E3",
            args: strings(&[
                "check",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--bin",
                "prod",
                "--message-format",
                "json",
            ]),
            cfg: false,
            kind: DualKind::Fails {
                case: "consumer",
                target: "prod",
                expected: "e3-expected.txt",
            },
        },
    ]
}

/// `--check-only`, or no argument for the full mode.
pub(crate) fn parse_dual_args(args: &[String]) -> Result<DualMode, String> {
    match args {
        [] => Ok(DualMode::Full),
        [flag] if flag == "--check-only" => Ok(DualMode::CheckOnly),
        _ => Err("check-dual takes only `--check-only` (CPU only, what CI runs)".into()),
    }
}

/// Refuses while any of [`RUSTFLAGS_VARIABLES`] is present in the environment, even with an empty value
/// (`lookup` answers `Some(..)` for a variable that is set). The error names every one that is.
pub(crate) fn refuse_preset_rustflags(
    lookup: impl Fn(&str) -> Option<OsString>,
) -> Result<(), String> {
    let set: Vec<&str> = RUSTFLAGS_VARIABLES
        .iter()
        .copied()
        .filter(|name| lookup(name).is_some())
        .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(format!(
        "check-dual refuses to run because {} {} set in the environment (an empty value counts). It delivers `--cfg gust_dual` through {DUAL_RUSTFLAGS_VARIABLE}, which takes precedence over the other rustflags sources, so flags set there would be dropped from the dual build, and a set {DUAL_RUSTFLAGS_VARIABLE} would be replaced; unset {} and run it again",
        set.join(", "),
        if set.len() == 1 { "is" } else { "are" },
        if set.len() == 1 { "it" } else { "them" },
    ))
}

/// The cargo config files that mention `rustflags`, described without machine paths: the workspace's own
/// `.cargo/config.toml`, those of its parent directories (cargo reads them all) and `config.toml` of `cargo_home`.
/// `CARGO_ENCODED_RUSTFLAGS` replaces such `build.rustflags` and `target.*.rustflags` settings, so on a machine with
/// one the dual build differs from a normal build in those flags.
pub(crate) fn rustflags_config_files(root: &Path, cargo_home: Option<&Path>) -> Vec<String> {
    let mentions = |path: &Path| {
        fs::read_to_string(path).is_ok_and(|text| text.to_ascii_lowercase().contains("rustflags"))
    };
    let mut found = Vec::new();
    for (level, directory) in root.ancestors().enumerate() {
        for name in ["config.toml", "config"] {
            if mentions(&directory.join(".cargo").join(name)) {
                let place = match level {
                    0 => "the workspace".to_owned(),
                    1 => "1 level above the workspace".to_owned(),
                    n => format!("{n} levels above the workspace"),
                };
                found.push(format!("{place}: .cargo/{name}"));
            }
        }
    }
    if let Some(home) = cargo_home {
        for name in ["config.toml", "config"] {
            if mentions(&home.join(name)) {
                found.push(format!("CARGO_HOME: {name}"));
            }
        }
    }
    found
}

/// `$CARGO_HOME`, or `~/.cargo`.
fn cargo_home() -> Option<PathBuf> {
    env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| {
        env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .map(|home| PathBuf::from(home).join(".cargo"))
    })
}

/// `text` without ANSI colour sequences (`ESC [ ... m`), which libtest prints around `ok` when colour is forced
/// (`CARGO_TERM_COLOR=always`, as CI sets it).
fn without_colors(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            for skipped in chars.by_ref() {
                if skipped.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

/// The names of `tests` that `stdout` of `cargo test` does not report as `ok`.
pub(crate) fn tests_not_ok<'a>(stdout: &str, tests: &[&'a str]) -> Vec<&'a str> {
    let stdout = without_colors(stdout);
    tests
        .iter()
        .copied()
        .filter(|name| {
            let wanted = format!("test {name} ... ok");
            !stdout.lines().any(|line| line.trim() == wanted)
        })
        .collect()
}

fn dual_command(root: &Path, step: &DualStep) -> Command {
    let mut command = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")));
    command
        .args(&step.args)
        .current_dir(root)
        .stdin(Stdio::null());
    // Absent, not inherited: whatever the parent has, the child starts without any rustflags variable, then gets
    // the cfg on the steps that build with it.
    for name in RUSTFLAGS_VARIABLES {
        command.env_remove(name);
    }
    for (name, value) in step.env() {
        command.env(name, value);
    }
    command
}

/// The `CARGO_TARGET_<TRIPLE>_RUSTFLAGS` variables among `names` (upper-cased and sorted): cargo config given
/// through the environment, which `CARGO_ENCODED_RUSTFLAGS` replaces like the config files.
pub(crate) fn target_rustflags_variables(names: impl Iterator<Item = OsString>) -> Vec<String> {
    let mut found: Vec<String> = names
        .filter_map(|name| name.into_string().ok())
        .map(|name| name.to_ascii_uppercase())
        .filter(|name| {
            name.len() > "CARGO_TARGET__RUSTFLAGS".len()
                && name.starts_with("CARGO_TARGET_")
                && name.ends_with("_RUSTFLAGS")
        })
        .collect();
    found.sort();
    found.dedup();
    found
}

/// `cargo xtask check-dual [--check-only]`: the dual test build (`--cfg gust_dual`, D26) and its cross-crate
/// proof. Needs cargo only; no slangc and no GPU.
///
/// 1. The workspace builds with the cfg (all targets) in `<target dir>/dual`.
/// 2. E1 (no cfg): `tests/dual`'s `consumer` test fails with exactly one E0277, the missing twin of the dependency.
/// 3. E2 (cfg): the same test compiles and passes, which is the proof that the cfg reached `kernel-lib`.
/// 4. E3 (no cfg): a production binary fails with exactly that one error.
pub(crate) fn check_dual(args: &[String]) -> Result<(), String> {
    let mode = parse_dual_args(args)?;
    refuse_preset_rustflags(|name| env::var_os(name))?;
    let root = workspace_root()?;
    let configs = rustflags_config_files(&root, cargo_home().as_deref());
    if !configs.is_empty() {
        println!(
            "note: {DUAL_RUSTFLAGS_VARIABLE} replaces `build.rustflags` and `target.*.rustflags` of cargo config files, and these mention rustflags ({}); the dual build here differs from a normal build in those flags",
            configs.join("; ")
        );
    }
    let target_variables = target_rustflags_variables(env::vars_os().map(|(name, _)| name));
    if !target_variables.is_empty() {
        println!(
            "note: {DUAL_RUSTFLAGS_VARIABLE} also replaces the rustflags of {}, so the dual build differs from a normal build in those flags",
            target_variables.join(", ")
        );
    }
    // Cargo takes the directory as an argument, so a path that is not valid UTF-8 would be altered, not passed.
    let target_dir = dual_target_dir(&root);
    let target_dir = target_dir.to_str().ok_or_else(|| {
        "check-dual: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path".to_owned()
    })?;
    for step in dual_steps(target_dir) {
        println!("check-dual {}: {}", step.id, step.shown());
        run_dual_step(&root, &step)?;
    }
    println!(
        "check-dual passed: the workspace built with --cfg gust_dual; without it the tests/dual consumer's test (E1) and binary (E3) each failed with exactly one E0277 for the missing CPU twin of the dependency `kernel-lib`, and with it (E2) the test dispatched on `gust::cpu::CpuDevice` and passed, together with the overflow-mode test of a package with its own override. Needs cargo only; no slangc and no GPU was involved."
    );
    if mode == DualMode::Full {
        println!(
            "check-dual: full mode (the vector-add and polynomial runs with and without the cfg) arrives in m3.f2; only the --check-only steps ran"
        );
    }
    Ok(())
}

fn exit_code(status: std::process::ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "?".to_owned(), |code| code.to_string())
}

/// The problems of a `Fails` step (E1 or E3) for the outcome of its cargo call: none when it failed with exactly the
/// one error of the step's expectation file. Any other kind of step has no expectation.
pub(crate) fn dual_problems(
    root: &Path,
    step: &DualStep,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<String>, String> {
    let DualKind::Fails {
        case,
        target,
        expected,
    } = &step.kind
    else {
        return Err(format!("check-dual {}: has no expectation file", step.id));
    };
    let id = step.id;
    let text = fs::read_to_string(root.join(DUAL_DIR).join(expected))
        .map_err(|error| format!("check-dual {id}: {DUAL_DIR}/{expected}: {error}"))?;
    let wanted = parse_expected(&text)
        .map_err(|error| format!("check-dual {id}: {DUAL_DIR}/{expected}: {error}"))?;
    Ok(judge(
        case,
        target,
        &Expectation::Fails(wanted),
        success,
        stdout,
        stderr,
    ))
}

fn run_dual_step(root: &Path, step: &DualStep) -> Result<(), String> {
    let id = step.id;
    let launch = |error: io::Error| format!("check-dual {id}: could not launch cargo: {error}");
    match &step.kind {
        DualKind::Build | DualKind::Compiles => {
            let status = dual_command(root, step)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(launch)?;
            if status.success() {
                return Ok(());
            }
            let code = exit_code(status);
            if step.kind == DualKind::Compiles {
                return Err(format!(
                    "check-dual {id}: cfg did not reach `kernel-lib`, or the fixture failed to compile for another reason; see the cargo output above (the consumer's test did not compile with {DUAL_RUSTFLAGS_VARIABLE} set to `--cfg gust_dual`, cargo exit {code}). Stop: there is no fallback delivery. Run the same cargo command with {DUAL_RUSTFLAGS_VARIABLE} set and `-vv` to see the rustc flags it passed"
                ));
            }
            Err(format!("check-dual {id}: cargo failed (exit {code})"))
        }
        DualKind::Fails { .. } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let problems = dual_problems(
                root,
                step,
                output.status.success(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            )?;
            if problems.is_empty() {
                return Ok(());
            }
            Err(format!(
                "check-dual {id} found {} problem(s):\n{}",
                problems.len(),
                problems.join("\n")
            ))
        }
        DualKind::Passes { tests } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            print!("{stdout}");
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            if !output.status.success() {
                return Err(format!(
                    "check-dual {id}: cargo failed (exit {})",
                    exit_code(output.status)
                ));
            }
            let missing = tests_not_ok(&stdout, tests);
            if missing.is_empty() {
                return Ok(());
            }
            Err(format!(
                "check-dual {id}: cargo succeeded but did not report these tests as ok: {}",
                missing.join(", ")
            ))
        }
    }
}

#[cfg(test)]
#[path = "tests/dual.rs"]
mod dual_tests;
