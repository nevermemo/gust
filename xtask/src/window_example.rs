//! `cargo xtask check-window-example`: compile-only evidence that `examples/window-share` builds (m14.devshare.1).
//!
//! The example is a workspace of its own with its own `Cargo.lock`, like `tests/dual`: the root workspace does not
//! contain it, so `cargo build --workspace --all-targets` and every lane gate leave winit out. This command formats
//! it (`cargo fmt --check`; the root `cargo fmt --all` does not reach it) and builds it with `--locked`. It is not
//! part of `verify`, `check-fast` or `check-full`, and it runs nothing: opening the window needs a display and an
//! adapter, and is a manual run (`GUST_WINDOW_ADAPTER=<name> cargo run --manifest-path examples/window-share/Cargo.toml`),
//! which prints `GUST_OK frames=60 adapter=<name>`.

use std::path::{Path, PathBuf};

use crate::{
    process::run_direct,
    workspace::{cargo_target_dir, workspace_root},
};

/// The example's manifest, from the repository root.
const MANIFEST: &str = "examples/window-share/Cargo.toml";

/// The cargo calls of the check, in order. Kept as data so a test can require the exact lists.
///
/// `target_dir` is where the example builds, `<cargo target dir>/window-share` (the way `check-dual` keeps `dual` apart),
/// so winit never lands in the shared target directory and no `target/` appears inside the example.
pub(crate) fn window_example_calls(target_dir: &str) -> [Vec<String>; 3] {
    let call = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect();
    [
        call(&["fmt", "--manifest-path", MANIFEST, "--", "--check"]),
        call(&[
            "build",
            "--locked",
            "--manifest-path",
            MANIFEST,
            "--target-dir",
            target_dir,
        ]),
        call(&[
            "clippy",
            "--locked",
            "--manifest-path",
            MANIFEST,
            "--target-dir",
            target_dir,
            "--",
            "-D",
            "warnings",
        ]),
    ]
}

/// `<cargo target dir>/window-share`.
pub(crate) fn window_target_dir(root: &Path) -> PathBuf {
    root.join(cargo_target_dir(root)).join("window-share")
}

pub(crate) fn check_window_example(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("check-window-example takes no arguments".into());
    }
    let root = workspace_root()?;
    let target_dir = window_target_dir(&root);
    let target_dir = target_dir.to_str().ok_or_else(|| {
        "check-window-example: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path"
            .to_owned()
    })?;
    for call in window_example_calls(target_dir) {
        println!("check-window-example: cargo {}", call.join(" "));
        let borrowed = call.iter().map(String::as_str).collect::<Vec<_>>();
        run_direct(&root, "cargo", &borrowed)?;
    }
    println!(
        "check-window-example passed: examples/window-share is formatted, builds with --locked and is clippy-clean (nothing was run; the window run is manual)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_is_locked_and_both_calls_name_the_example_manifest() {
        let [fmt, build, clippy] = window_example_calls("T/window-share");
        assert_eq!(
            fmt,
            [
                "fmt",
                "--manifest-path",
                "examples/window-share/Cargo.toml",
                "--",
                "--check"
            ]
        );
        assert_eq!(
            build,
            [
                "build",
                "--locked",
                "--manifest-path",
                "examples/window-share/Cargo.toml",
                "--target-dir",
                "T/window-share"
            ]
        );
        assert_eq!(
            clippy,
            [
                "clippy",
                "--locked",
                "--manifest-path",
                "examples/window-share/Cargo.toml",
                "--target-dir",
                "T/window-share",
                "--",
                "-D",
                "warnings"
            ]
        );
    }

    #[test]
    fn arguments_are_refused() {
        assert_eq!(
            check_window_example(&["--now".to_owned()]),
            Err("check-window-example takes no arguments".to_owned())
        );
    }
}
