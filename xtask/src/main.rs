use std::{env, process::ExitCode};

mod arch;
mod bless;
mod checks;
mod cpu_target;
mod difftest;
mod docs;
mod json;
mod lint;
mod precompiled;
mod process;
mod reflect;
mod release;
mod shader;
mod suite;
mod test_exact;
mod time;
mod ui;
mod validation;
mod verify;
mod web;
mod workspace;

use checks::{
    check_changed, check_cpu, check_dual, check_feature, check_format, check_lints,
    check_workspace, doctor, explain_check, export_artifacts, hook_format_rust_after_edit,
    list_tests, measure_tests, self_test, status,
};
use reflect::{BuildOptions, build_slang_reflect};
use shader::{SlangcOptions, SpirvValOptions, slangc, spirv_val};
use test_exact::{TestExactOptions, test_exact};
use verify::{VerifyCommand, VerifyMode, check_full, check_ledger, verify, verify_with};
use workspace::{FEATURE_AREAS, workspace_root};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        usage();
        return Err("missing command".into());
    };
    let args: Vec<_> = args.collect();
    match command.as_str() {
        "build-slang-reflect" => build_slang_reflect(BuildOptions::parse(&args)?)?,
        "verify" => {
            let command = VerifyCommand::parse(&args)?;
            verify_with(command.mode, command.record, command.skip_web)?;
        }
        "check-fast" => verify(VerifyMode::Fast, false)?,
        "check-full" => check_full(&args)?,
        "check-examples" => verify(VerifyMode::Examples, false)?,
        "check-artifacts" => verify(VerifyMode::Artifacts, false)?,
        "check-workspace" => check_workspace()?,
        "check-format" => check_format()?,
        "check-lints" => check_lints()?,
        "check-feature" => check_feature(args.first().map(String::as_str))?,
        "check-changed" => check_changed()?,
        "status" => {
            if !args.is_empty() {
                return Err("status takes no arguments; `status --update` was intentionally not implemented".into());
            }
            status()?;
        }
        "doctor" => doctor(&args)?,
        "lint" => {
            if !args.is_empty() {
                return Err("lint takes no arguments".into());
            }
            lint::lint(&workspace_root()?, &mut Vec::new())?;
        }
        "docs-check" => {
            if !args.is_empty() {
                return Err("docs-check takes no arguments".into());
            }
            docs::docs_check(&workspace_root()?, &mut Vec::new())?;
        }
        "check-arch" => {
            if !args.is_empty() {
                return Err("check-arch takes no arguments".into());
            }
            arch::check_arch(&workspace_root()?, &mut Vec::new())?;
        }
        "check-ui" => {
            if !args.is_empty() {
                return Err("check-ui takes no arguments".into());
            }
            ui::check_ui(&workspace_root()?, &mut Vec::new())?;
        }
        "check-cpu" => {
            if !args.is_empty() {
                return Err("check-cpu takes no arguments".into());
            }
            check_cpu()?;
        }
        "check-difftest" => difftest::check_difftest(&args)?,
        "check-dual" => check_dual(&args)?,
        "check-ledger" => check_ledger(&args)?,
        "check-cpu-target" => cpu_target::check_cpu_target(&args)?,
        "check-precompiled" => precompiled::check_precompiled(&args)?,
        "check-release" => release::check_release(&args)?,
        "check-web" => web::check_web(&args)?,
        "bless" => {
            if !args.is_empty() {
                return Err("bless takes no arguments (it sets BLESS_GOLDEN=1 itself)".into());
            }
            bless::bless()?;
        }
        "test" => suite::test_command(&args)?,
        "slangc" => slangc(&SlangcOptions::parse(&args)?)?,
        "spirv-val" => spirv_val(&SpirvValOptions::parse(&args)?)?,
        "test-exact" => test_exact(&TestExactOptions::parse(&args)?)?,
        "list-tests" => list_tests()?,
        "explain-check" => explain_check(args.first().map(String::as_str))?,
        "export-artifacts" => export_artifacts()?,
        "measure-tests" => measure_tests(&args)?,
        "hook-format-rust-after-edit" => hook_format_rust_after_edit()?,
        "self-test" => self_test()?,
        "help" | "--help" | "-h" => usage(),
        _ => return Err(format!("unknown xtask command `{command}`")),
    }
    Ok(())
}

fn usage() {
    eprintln!(
        "usage: cargo xtask <command>\n\n  build-slang-reflect [--force] [--sdk-root PATH] [--out-dir PATH]\n  verify [--mode smoke|fast|gpu|examples|artifacts|full] [--record|--no-record] [--skip-web (full only)]\n  check-feature <{}>\n  check-changed | check-workspace | check-format | check-lints\n  check-fast | check-examples | check-artifacts\n  check-full [--skip-web]   the release gate, recorded in .ai/VALIDATION.json: lint, docs-check, check-arch and check-ui; fmt and clippy; the whole workspace test run, the release-only tests and every example's tests and run (with a Vulkan GPU); check-dual in full; the export of the SPIR-V artifacts and spirv-val on each (what check-artifacts does); then check-difftest (all 8 seeds on the adapter with --shapes grid,pixel, and the short_circuit GPU test), then the whole check-precompiled (E1 to E8, with a GPU, Slang and the crates.io registry), the whole check-release (R1 to R8, with a GPU, Slang and the registry) and the whole check-web (Chrome with a WebGPU adapter, wasm-bindgen, the wasm32 target). Nothing is skipped silently: a missing tool or adapter fails the run and names the cause. --skip-web leaves out check-web only, prints that, and records `skipped_checks: [\"check-web\"]` and the browser as untested in the record\n  lint   tracked files: no shell scripts or machine-specific paths; release metadata, licence copies, dependency names\n  docs-check   relative Markdown links resolve to project files, no conflict markers, DECISIONS ids; scans tracked and untracked non-ignored files (anchors and external links are not checked)\n  check-arch   dependency allow-list per crate read from `cargo metadata` (D02), no IR module outside future/, no SPIR-V library; and every allow-listed package in Cargo.lock resolves from crates.io or crates/ (a [patch] or [replace] redirect fails); scans tracked and untracked non-ignored files\n  check-ui   `cargo check --locked --message-format json` of each tests/ui fixture crate: one with an expected.txt must fail with exactly one error satisfying every key (code, file and line required, message, label: diagnostic::on_unimplemented wording; `code: none` plus a `message:` for a macro error without a rustc code), one without must compile; needs cargo only\n  check-cpu   `cargo test --locked` of the examples with a CPU twin (vector-add, polynomial, signal-pipeline, particle-step, typed-pipeline, atomic-counter, staged-graph, component-pool, map-chain, texture-threshold, shape-bench): the non-ignored twin tests; needs neither slangc nor a GPU; a step of the Unit suite\n  check-cpu-target   the production CPU target (D21, D51) on the fixture tests/cpu-target, cargo only (no slangc, no GPU), every cargo child under GUST_PROFILE=cpu except where a step needs another state: E-a builds `app` in release under cpu and runs it (one `CPU target: matches (..)` line per kernel, the `CPU target: map calls match (..)` line of the map case exactly once and the final line); E-b requires that binary to hold none of slangc, gust-slang-reflect, wgpu, vulkan (any letter case; the scan must find its control string) and `cargo tree -e features -p app` to show no wgpu and no slang-runtime; E-c builds and runs `app` under cpu with --cfg gust_dual (one impl, no E0119) in <target>/cpu-target-dual; E-d, E-e, E-f and E-g judge the refusals on expectation files (`app` without the profile: the one E0277 of a missing twin; `app-with-wgpu` under cpu: the one E0080 of `gust-wgpu`'s require_gpu_target; `app` under cpu with --features gust/slang-runtime: the build script's refusal; `optout` under cpu: the one E0277 of an opted-out kernel); E-h flips the profile cpu, unset, cpu, unset, cpu, unset in <target>/cpu-target-flip (cpu states must build and match, unset states must fail like E-d); E-i runs `cargo test -p kernels` under cpu; stops at the first failing step and names it; the Cpu suite is this command\n  check-difftest [--seeds <list|range|random:N>] [--shapes grid,pixel] [--cpu-only] [--report PATH]   the differential test harness tests/difftest (D38, D39, D51, D52), run through `difftest-run`: it generates 16 in-domain scalar kernels per seed (default: the 8 seeds of tests/difftest/seeds.txt, 128 programs), runs each on an independent interpreter, the CPU twin and, unless --cpu-only, the GPU, and writes report.json; this command fails unless the report has the requested mode and seeds, `programs_executed` equal to 16 times the seeds, no mismatch (each is named with its seed) and, without --cpu-only, an adapter. Without --cpu-only it also runs the harness crate's `short_circuit` test with --ignored (the GPU must skip a right operand that would read out of bounds), needs a Vulkan adapter, slangc and the reflection helper, and is Gpu evidence only with the adapter the report names. --cpu-only (interpreter against twin, cargo only) is a step of the Unit suite with --seeds 1-4; `random:N` draws N seeds, prints them and runs them as a list; --report PATH keeps the report (default <target>/difftest-report.json); --shapes grid,pixel gives every program a pixel kernel next to its grid kernel and compares the two (integers and booleans bit for bit, `f32` through each shape's interval, with the bit differences only observed; the report names the programs the macro classifier kept grid-only, with its reason, and at least one must run as pixel); without --shapes the run is grid-only; the Gpu suite runs all 8 seeds grid-only, and check-full runs all 8 seeds with --shapes grid,pixel\n  check-ledger   the known-mismatch ledger (crates/gust-wgpu/tests/mismatch.rs) in the release profile, with the guard of the Full mode: at least one test ran, 0 failed, 0 ignored, and each of the six ledger tests reported ok by name (RELEASE_LEDGER_TESTS in xtask/src/verify.rs); needs slangc, the reflection helper and a Vulkan adapter (the nightly list of the gpu-software workflow runs this command)\n  check-dual [--check-only]   the dual test build (`--cfg gust_dual`, D26) and its cross-crate proof: tests/dual is formatted and passes clippy -D warnings, the workspace builds with the cfg into <target>/dual, and in tests/dual a consumer's test (E1) and production binary (E3) each fail with exactly one E0277 without the cfg and the test passes with it (E2); without --check-only it also runs vector-add and polynomial with the cfg (they must print the `CPU twin: matches` line) and vector-add without it (it must not), which needs a Vulkan GPU; refuses to run while RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS or CARGO_BUILD_RUSTFLAGS is set; after its own steps it runs the strict-plus-dual steps E5 of check-precompiled on tests/precompiled; `--check-only` is the CPU-only mode CI runs and needs cargo only\n  check-precompiled [--check-only] [--work-dir PATH]   the precompiled fixture tests/precompiled (D45, D46), every cargo child under GUST_PROFILE=native except the E2 control build (dev, in <target>/precompiled-control): E1 cleans and builds, then runs, `app` in a copy of the environment without Slang (no SLANG variable, no VULKAN_SDK, no PATH directory with slangc or gust-slang-reflect) and needs a Vulkan GPU; E2 scans the release binary for the bridge strings and requires `cargo tree -e features` to show no slang-runtime; E3 runs a scratch copy with a changed artifact and requires the identity error; E4 (E6f) deletes the artifacts in a scratch copy and requires one E0080 at the kernel line; E5 builds the fixture with --cfg gust_dual and the strict profile together through check-dual's own command, runs app's twin test and requires no slang-runtime in cargo tree; E6a-e judge the refusals of the build script of gust on their exact message (explicit native, inferred from a release build, through feature unification, in a copy with a bridge user) with two positive controls; E7 shows that a changed GUST_EXPORT_DIR re-expands the kernels crate and, with Slang, that cargo gust build twice gives byte-identical output without a prune message; E8 checks the cargo package allow-lists of gust-macros, gust, gust-wgpu and kernels, unpacks them outside the repository (--work-dir PATH, default a temp directory; deleted afterwards) and builds and runs a consumer with registry-style dependencies patched onto them, needing the crates.io registry; --check-only skips the runs of E1, E3 and E8 (they need an adapter) and the second half of E7 (it needs Slang), and is what CI runs; the scratch copies are under <target>/precompiled-scratch\n  check-release [--check-only] [--work-dir DIR]   the permanent release gate (mr.r): R1 one multi-package cargo package --locked of every publishable member (from cargo metadata, so it needs no edit across the rename), with verification; R2 the file lists and unpacked trees against the allow-list (E8 uses the same one); R3/R4 metadata, one version, exact internal pins, publish flags; R5 cargo doc of each library crate with every rustdoc warning an error (known debt of mr.b2 admitted by message and file, once each), plus nightly --cfg docsrs when a nightly toolchain is installed; R6 each .crate under 2 MiB; R7 the README recipe word for word (xtask/src/release/hello-gust): the unpacked packages patched into a consumer built with its gust-dev feature, cargo-gust installed from its unpacked package, and, without --check-only, run on the GPU, then cargo gust build and the strict release build and run; R8 cargo publish --dry-run --locked. Needs the crates.io registry. --work-dir DIR is a new or empty directory outside the repository (default: a new directory under the cargo target directory, or under the temporary directory when that lies inside the repository, as on CI), deleted when the run ends (a deletion that fails fails the check). --check-only skips what needs Slang or an adapter and is what CI runs\n  check-web [--wasm-bindgen PATH] [--chrome PATH] [--timeout-secs N] [--chrome-arg ARG]... | --tree-only   the first browser WebGPU run (D45). First a feature guard: `cargo tree -e features -i wgpu` must show the wgpu `webgpu` feature for wasm32 and not for the host (all that `--tree-only` runs, needs only cargo). Then it builds examples/web-vector-add for wasm32 under GUST_PROFILE=web, runs wasm-bindgen --target web (the CLI must equal the wasm-bindgen version in Cargo.lock), serves the output from 127.0.0.1 on an ephemeral port, opens it in headless Chrome with a fresh profile and requires the console line `GUST_OK <vendor> <architecture>` (the real gust-wgpu vector-add and the `map_async` map of `double_add` equalled a host reference on the browser GPU) within the deadline (default 90 s), then kills Chrome and stops the server; a `GUST_FAIL`, a missing line or a timeout fails; needs the wasm32 target, wasm-bindgen-cli, Chrome and a WebGPU adapter; prints the `Gpu (WebGPU, Chrome <version>, <adapter>)` label; not part of any suite\n  bless   rewrite reviewed Slang goldens (BLESS_GOLDEN=1); exits non-zero when a fixture changed\n  test --suite <{}>   evidence tier as one command\n  slangc <file.slang> --entry NAME --target wgsl|spirv --out PATH [-- SLANGC_ARGS]\n  spirv-val <file.spv> --target-env ENV\n  test-exact -p PACKAGE --test TARGET [--features LIST] NAME\n  status | doctor [--smoke] | list-tests | explain-check <area>\n  export-artifacts [--hash-manifest PATH] [--compare PATH]   run the examples; --hash-manifest writes the SHA-256 of every exported .slang, .rs, .wgsl and .spv file (stale ones are deleted first), --compare fails on any missing, unexpected or changed file against such a manifest (the byte-identical gate of mx.p0)\n  measure-tests [--json PATH] [command ...]\n  hook-format-rust-after-edit\n  self-test",
        FEATURE_AREAS.join("|"),
        suite::Suite::names().join("|")
    );
}
