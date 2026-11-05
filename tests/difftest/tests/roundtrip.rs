//! The emitter round trip: generated crates pass `cargo check --locked` against the repository's `gust` and
//! `gust-wgpu` path dependencies, under `GUST_PROFILE=dev`, with the committed template lockfile. Needs cargo and
//! the registry packages already in the cargo cache (the template lock pins the same versions as the root lock);
//! needs neither slangc nor a GPU. Generated crates live under the target directory and are never committed.

use std::path::{Path, PathBuf};
use std::process::Command;

use difftest::emit::{lock_for, package_name, write_crate};
use difftest::generator::generate_batch;
use difftest::runsrc::{kernel_source_in, plan_seed, write_run_crate, write_run_crate_in};
use difftest::shapes::{Shape, ShapePlan, module_ranges, refusals, set_aside};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `<target>/tmp/difftest`: cargo creates `CARGO_TARGET_TMPDIR` for integration tests.
fn work_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("difftest")
}

/// Runs `cargo check --locked` on the generated crate in `dir`; returns success and the combined output.
fn cargo_check(dir: &Path) -> (bool, String) {
    cargo_check_with(dir, &[])
}

/// `cargo check --locked <extra>`; the run crate needs `--tests` (its twin and test module exist only under cfg(test)).
fn cargo_check_with(dir: &Path, extra: &[&str]) -> (bool, String) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["check", "--locked"])
        .args(extra)
        .args(["--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        // One build directory for all generated crates and runs, separate from the one this test runs in.
        .env("CARGO_TARGET_DIR", work_dir().join("target"))
        .env("GUST_PROFILE", "dev")
        // The tests parse this output (`refusals`); CI sets CARGO_TERM_COLOR=always, which colours the diagnostics.
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("cargo should start");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(60)..].join("\n")
}

#[test]
fn generated_crates_pass_cargo_check_with_the_template_lockfile() {
    let dir = work_dir().join("roundtrip-ok");
    let mut programs = Vec::new();
    for seed in 1..=8 {
        programs.extend(generate_batch(seed).unwrap().programs);
    }
    assert_eq!(programs.len(), 128);
    write_crate(&dir, &package_name("ok").unwrap(), &programs, &repo_root()).unwrap();
    let (ok, output) = cargo_check(&dir);
    assert!(ok, "cargo check failed:\n{}", tail(&output));
    // `--locked` would have refused a lockfile that needed an update; the crate used the template as written.
    let lock = std::fs::read_to_string(dir.join("Cargo.lock")).unwrap();
    assert_eq!(lock, lock_for("difftest-case-ok").unwrap());
}

#[test]
fn run_crates_pass_cargo_check_with_their_test_module() {
    // The crate that `difftest-run` builds: the 128 kernels of seeds 1 to 8 plus the generated test module (the twin
    // dispatches, the GPU path, the result files) must type-check against the repository's `gust` and `gust-wgpu`.
    let dir = work_dir().join("roundtrip-run");
    let mut plan = Vec::new();
    for seed in 1..=8 {
        plan.extend(plan_seed(seed).unwrap());
    }
    assert_eq!(plan.len(), 128);
    let package = package_name("run-roundtrip").unwrap();
    write_run_crate(&dir, &package, &plan, &repo_root()).unwrap();
    let (ok, output) = cargo_check_with(&dir, &["--tests"]);
    assert!(
        ok,
        "cargo check --tests failed:
{}",
        tail(&output)
    );
    let lock = std::fs::read_to_string(dir.join("Cargo.lock")).unwrap();
    assert_eq!(lock, lock_for(&package).unwrap());
}

#[test]
fn the_round_trip_check_fails_on_a_broken_crate() {
    let dir = work_dir().join("roundtrip-broken");
    let batch = generate_batch(1).unwrap();
    let package = package_name("broken").unwrap();
    write_crate(&dir, &package, &batch.programs[..1], &repo_root()).unwrap();
    let lib = dir.join("src").join("lib.rs");
    let mut text = std::fs::read_to_string(&lib).unwrap();
    text.push_str("\npub const BROKEN: u32 = true;\n");
    std::fs::write(&lib, text).unwrap();
    let (ok, output) = cargo_check(&dir);
    assert!(!ok, "cargo check accepted a type error");
    assert!(output.contains("mismatched types"), "{}", tail(&output));
}

#[test]
fn the_round_trip_check_fails_when_the_gpu_validator_rejects_a_module() {
    let dir = work_dir().join("roundtrip-rejected");
    let batch = generate_batch(1).unwrap();
    write_crate(
        &dir,
        &package_name("rejected").unwrap(),
        &batch.programs[..1],
        &repo_root(),
    )
    .unwrap();
    let lib = dir.join("src").join("lib.rs");
    let mut text = std::fs::read_to_string(&lib).unwrap();
    // `while` is outside the dialect: the macro must refuse it with a GUST diagnostic, not rustc.
    text.push_str(
        "\n#[gust::gpu]\npub mod rejected {\n    #[kernel(workgroup_size(64, 1, 1))]\n    pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {\n        let mut n = 0u32;\n        while n < 3u32 {\n            n = n + 1u32;\n        }\n        out[id.x] = n;\n    }\n}\n",
    );
    std::fs::write(&lib, text).unwrap();
    let (ok, output) = cargo_check(&dir);
    assert!(!ok, "cargo check accepted a module outside the dialect");
    assert!(output.contains("[GUST-E"), "{}", tail(&output));
}

#[test]
fn run_crates_with_a_pixel_kernel_per_module_pass_cargo_check_with_their_test_module() {
    // The crate `difftest-run --shapes grid,pixel` builds first: every module has its `k_px` beside the `workgroup_size`
    // kernel (the E0707 sibling-kernel fix is what lets them share a module) and the generated test module runs both.
    let dir = work_dir().join("roundtrip-run-pixel");
    let mut plan = Vec::new();
    for seed in 1..=8 {
        plan.extend(plan_seed(seed).unwrap());
    }
    assert_eq!(plan.len(), 128);
    let package = package_name("run-roundtrip-pixel").unwrap();
    let shapes = ShapePlan::new(&[Shape::Grid, Shape::Pixel]);
    write_run_crate_in(&dir, &package, &plan, &repo_root(), &shapes).unwrap();
    let source = std::fs::read_to_string(dir.join("src").join("lib.rs")).unwrap();
    assert_eq!(source.matches("fn k_px(").count(), 128);
    let (ok, output) = cargo_check_with(&dir, &["--tests"]);
    assert!(ok, "cargo check --tests failed:\n{}", tail(&output));
    let lock = std::fs::read_to_string(dir.join("Cargo.lock")).unwrap();
    assert_eq!(lock, lock_for(&package).unwrap());
}

#[test]
fn a_return_in_one_pixel_kernel_is_one_refusal_that_the_harness_parses() {
    // Mutation check (slice mx.p1c): a `return;` at the top of the third module's `k_px` is outside the pixel contract
    // (every path writes `out[id.x]`, E0727). The macro refuses that module and no other; `refusals` finds the error at
    // the attribute's line, `set_aside` maps it to that module (not the first, not the last), and the crate without that
    // `k_px` builds, which is what the harness's one rebuild does.
    let dir = work_dir().join("roundtrip-run-pixel-mutant");
    let plan = plan_seed(1).unwrap();
    let package = package_name("run-roundtrip-pixel-mutant").unwrap();
    let mut shapes = ShapePlan::new(&[Shape::Grid, Shape::Pixel]);
    write_run_crate_in(&dir, &package, &plan, &repo_root(), &shapes).unwrap();
    let lib = dir.join("src").join("lib.rs");
    let text = std::fs::read_to_string(&lib).unwrap();
    let ranges = module_ranges(&kernel_source_in(&plan, &shapes));
    assert_eq!(ranges.len(), plan.len());
    let (target, start, end) = ranges[2].clone();
    // The k_px of the third module: the first `fn k_px(` after the module's opening line.
    let module_at: usize = text
        .lines()
        .take(start - 1)
        .map(|line| line.len() + 1)
        .sum();
    let at = module_at + text[module_at..].find("fn k_px(").expect("a k_px");
    let open = at + text[at..].find("{\n").expect("its body") + 2;
    let mutant = format!("{}        return;\n{}", &text[..open], &text[open..]);
    std::fs::write(&lib, mutant).unwrap();
    let (ok, output) = cargo_check_with(&dir, &["--tests"]);
    assert!(!ok, "cargo check accepted a return in a pixel kernel");
    let found = refusals(&output);
    assert_eq!(found.len(), 1, "{found:?}\n{}", tail(&output));
    assert_eq!(found[0].code, "GUST-E0727", "{found:?}");
    assert!((start..=end).contains(&found[0].line), "{found:?}");
    let refused = set_aside(&found, &ranges).unwrap();
    assert_eq!(refused.keys().collect::<Vec<_>>(), [&target], "{refused:?}");
    assert!(refused[&target].reason.starts_with("GUST-E0727: "));
    // The rebuild: the plan without that module's `k_px` (the mutation goes with it) passes `cargo check --tests`, and
    // every other module keeps its `k_px`.
    shapes.grid_only = refused;
    write_run_crate_in(&dir, &package, &plan, &repo_root(), &shapes).unwrap();
    let rebuilt = std::fs::read_to_string(&lib).unwrap();
    assert_eq!(rebuilt.matches("fn k_px(").count(), plan.len() - 1);
    assert!(!shapes.runs_pixel(&target));
    let (ok, output) = cargo_check_with(&dir, &["--tests"]);
    assert!(ok, "the rebuild failed:\n{}", tail(&output));
}
