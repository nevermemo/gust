//! Records the commit the binary was built at (`GUST_BENCH_BUILD_HEAD`) and whether tracked files had uncommitted
//! changes then (`GUST_BENCH_BUILD_DIRTY`, `1` or `0`), so that a run can tell a binary built from another commit than
//! the one now checked out, or from a tree that no commit holds (the harness's publishable judgement blocks both).
//! Shared by `gust-bench-pre` (`build = "../gust-bench/build.rs"`). When git cannot answer, the variable is empty and
//! the run is not publishable.
//!
//! The dirty flag is `git status --porcelain --untracked-files=no` being non-empty. The script re-runs when HEAD, the
//! branch ref, `packed-refs` or the git index change, and when anything under the repository's `crates/` or `bench/`
//! (except `bench/target/`, which every build writes to) changes, so an edit, a build and a revert leave a binary
//! marked dirty until it is rebuilt.

use std::{path::PathBuf, process::Command};

fn manifest_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(["-c", "safe.directory=*"])
        .args(args)
        .current_dir(manifest_dir())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn watch(git_path: &str) {
    if let Some(path) = git(&["rev-parse", "--git-path", git_path]) {
        let path = PathBuf::from(path);
        let path = if path.is_absolute() {
            path
        } else {
            manifest_dir().join(path)
        };
        // A missing path counts as changed to cargo and would rerun the script on every build (a packed ref, no
        // packed-refs file); packing deletes the watched loose ref, which triggers the one rerun that drops it.
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Watches the repository's `crates/` and `bench/` (both two levels above the manifest directory of `gust-bench` and
/// of `gust-bench-pre`). Cargo scans a watched directory for newer files, so `bench/` is watched entry by entry,
/// leaving out its `target/`.
fn watch_sources() {
    let root = manifest_dir().join("..").join("..");
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    if let Ok(entries) = std::fs::read_dir(root.join("bench")) {
        for entry in entries.flatten() {
            if entry.file_name() != "target" {
                println!("cargo:rerun-if-changed={}", entry.path().display());
            }
        }
    }
}

fn main() {
    watch("HEAD");
    watch("packed-refs");
    watch("index");
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watch(&reference);
    }
    watch_sources();
    let head = git(&["rev-parse", "HEAD"]).unwrap_or_default();
    println!("cargo:rustc-env=GUST_BENCH_BUILD_HEAD={head}");
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(status) if status.is_empty() => "0",
        Some(_) => "1",
        None => "",
    };
    println!("cargo:rustc-env=GUST_BENCH_BUILD_DIRTY={dirty}");
}
