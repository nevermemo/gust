//! `cargo xtask check-window-example`: compile-only evidence that `examples/window-share` builds (m14.devshare.1).
//!
//! The example is a workspace of its own with its own `Cargo.lock`, like `tests/dual`: the root workspace does not
//! contain it, so `cargo build --workspace --all-targets` and every lane gate leave winit out. This command formats
//! it (`cargo fmt --check`; the root `cargo fmt --all` does not reach it) and builds it with `--locked`. It is not
//! part of `verify`, `check-fast` or `check-full`, and it runs nothing: opening the window needs a display and an
//! adapter, and is a manual run (`GUST_WINDOW_ADAPTER=<name> cargo run --manifest-path examples/window-share/Cargo.toml`),
//! which prints `GUST_OK frames=60 adapter=<name>`.

use crate::{process::run_direct, workspace::workspace_root};

/// The example's manifest, from the repository root.
const MANIFEST: &str = "examples/window-share/Cargo.toml";

/// The cargo calls of the check, in order. Kept as data so a test can require the exact lists.
pub(crate) fn window_example_calls() -> [Vec<String>; 2] {
    let call = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect();
    [
        call(&["fmt", "--manifest-path", MANIFEST, "--", "--check"]),
        call(&["build", "--locked", "--manifest-path", MANIFEST]),
    ]
}

pub(crate) fn check_window_example(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("check-window-example takes no arguments".into());
    }
    let root = workspace_root()?;
    for call in window_example_calls() {
        println!("check-window-example: cargo {}", call.join(" "));
        let borrowed = call.iter().map(String::as_str).collect::<Vec<_>>();
        run_direct(&root, "cargo", &borrowed)?;
    }
    println!(
        "check-window-example passed: examples/window-share is formatted and builds with --locked (nothing was run; the window run is manual)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_is_locked_and_both_calls_name_the_example_manifest() {
        let [fmt, build] = window_example_calls();
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
                "examples/window-share/Cargo.toml"
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
