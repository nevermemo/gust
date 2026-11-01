use std::{env, process::ExitCode};

mod arch;
mod bless;
mod checks;
mod docs;
mod json;
mod lint;
mod precompiled;
mod process;
mod reflect;
mod shader;
mod suite;
mod test_exact;
mod time;
mod ui;
mod validation;
mod verify;
mod workspace;

use checks::{
    check_changed, check_cpu, check_dual, check_feature, check_format, check_lints,
    check_workspace, doctor, explain_check, export_artifacts, hook_format_rust_after_edit,
    list_tests, measure_tests, self_test, status,
};
use reflect::{BuildOptions, build_slang_reflect};
use shader::{SlangcOptions, SpirvValOptions, slangc, spirv_val};
use test_exact::{TestExactOptions, test_exact};
use verify::{VerifyCommand, VerifyMode, verify};
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
            verify(command.mode, command.record)?;
        }
        "check-fast" => verify(VerifyMode::Fast, false)?,
        "check-full" => verify(VerifyMode::Full, true)?,
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
        "check-dual" => check_dual(&args)?,
        "check-precompiled" => precompiled::check_precompiled(&args)?,
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
        "usage: cargo xtask <command>\n\n  build-slang-reflect [--force] [--sdk-root PATH] [--out-dir PATH]\n  verify [--mode smoke|fast|gpu|examples|artifacts|full] [--record|--no-record]\n  check-feature <{}>\n  check-changed | check-workspace | check-format | check-lints\n  check-fast | check-examples | check-artifacts | check-full\n  lint   no tracked shell scripts, no machine-specific paths; scans tracked files only\n  docs-check   relative Markdown links resolve to project files, no conflict markers, DECISIONS ids; scans tracked and untracked non-ignored files (anchors and external links are not checked)\n  check-arch   dependency allow-list per crate read from `cargo metadata` (D02), no IR module outside future/, no SPIR-V library; and every allow-listed package in Cargo.lock resolves from crates.io or crates/ (a [patch] or [replace] redirect fails); scans tracked and untracked non-ignored files\n  check-ui   `cargo check --locked --message-format json` of each tests/ui fixture crate: one with an expected.txt must fail with exactly one error satisfying every key (code, file and line required, message, label: diagnostic::on_unimplemented wording; `code: none` plus a `message:` for a macro error without a rustc code), one without must compile; needs cargo only\n  check-cpu   `cargo test --locked` of the examples with a CPU twin (vector-add, polynomial, signal-pipeline, particle-step, typed-pipeline, atomic-counter, staged-graph, component-pool): the non-ignored twin tests; needs neither slangc nor a GPU; a step of the Unit suite\n  check-dual [--check-only]   the dual test build (`--cfg gust_dual`, D26) and its cross-crate proof: tests/dual is formatted and passes clippy -D warnings, the workspace builds with the cfg into <target>/dual, and in tests/dual a consumer's test (E1) and production binary (E3) each fail with exactly one E0277 without the cfg and the test passes with it (E2); without --check-only it also runs vector-add and polynomial with the cfg (they must print the `CPU twin: matches` line) and vector-add without it (it must not), which needs a Vulkan GPU; refuses to run while RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS or CARGO_BUILD_RUSTFLAGS is set; after its own steps it runs the strict-plus-dual steps E5 of check-precompiled on tests/precompiled; `--check-only` is the CPU-only mode CI runs and needs cargo only\n  check-precompiled [--check-only] [--work-dir PATH]   the precompiled fixture tests/precompiled (D45, D46), every cargo child under GUST_PROFILE=native except the E2 control build (dev, in <target>/precompiled-control): E1 cleans and builds, then runs, `app` in a copy of the environment without Slang (no SLANG variable, no VULKAN_SDK, no PATH directory with slangc or gust-slang-reflect) and needs a Vulkan GPU; E2 scans the release binary for the bridge strings and requires `cargo tree -e features` to show no slang-runtime; E3 runs a scratch copy with a changed artifact and requires the identity error; E4 (E6f) deletes the artifacts in a scratch copy and requires one E0080 at the kernel line; E5 builds the fixture with --cfg gust_dual and the strict profile together through check-dual's own command, runs app's twin test and requires no slang-runtime in cargo tree; E6a-e judge the refusals of the build script of gust on their exact message (explicit native, inferred from a release build, through feature unification, in a copy with a bridge user) with two positive controls; E7 shows that a changed GUST_EXPORT_DIR re-expands the kernels crate and, with Slang, that cargo gust build twice gives byte-identical output without a prune message; E8 checks the cargo package allow-lists of gust-macros, gust, gust-wgpu and kernels, unpacks them outside the repository (--work-dir PATH, default a temp directory; deleted afterwards) and builds and runs a consumer with registry-style dependencies patched onto them, needing the crates.io registry; --check-only skips the runs of E1, E3 and E8 (they need an adapter) and the second half of E7 (it needs Slang), and is what CI runs; the scratch copies are under <target>/precompiled-scratch\n  bless   rewrite reviewed Slang goldens (BLESS_GOLDEN=1); exits non-zero when a fixture changed\n  test --suite <{}>   evidence tier as one command\n  slangc <file.slang> --entry NAME --target wgsl|spirv --out PATH [-- SLANGC_ARGS]\n  spirv-val <file.spv> --target-env ENV\n  test-exact -p PACKAGE --test TARGET [--features LIST] NAME\n  status | doctor [--smoke] | list-tests | explain-check <area>\n  export-artifacts\n  measure-tests [--json PATH] [command ...]\n  hook-format-rust-after-edit\n  self-test",
        FEATURE_AREAS.join("|"),
        suite::Suite::names().join("|")
    );
}
