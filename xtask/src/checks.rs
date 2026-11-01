use std::{
    fs,
    io::{self, Read},
    path::PathBuf,
    process::Command,
    time::Instant,
};

use crate::{
    process::{
        command_exists, print_command, require_command, run_checked, run_direct, run_direct_args,
    },
    reflect::{BuildOptions, build_slang_reflect},
    shader::doctor_smoke,
    validation::{compact_validation_summary, json_string},
    verify::{VerifyMode, verify},
    workspace::{EXAMPLES, changed_paths, workspace_root, workspace_test_args},
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
        "bless" => {
            "bless: reruns the Slang golden tests with BLESS_GOLDEN=1, which rewrites changed fixtures and fails with the diff; review before committing"
        }
        "test" => {
            "test: `cargo xtask test --suite Unit|Slangc|Spirv|Reflect|Gpu|Full` runs one evidence tier of VALIDATION.md"
        }
        "all" => {
            "known areas: macro, core, wgpu, gpu-smoke, gpu-semantics, gpu-runtime, reflection, loops, examples, artifacts, full, changed, workspace, format, lints, lint, docs-check, check-arch, check-ui, check-cpu, bless, test"
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
