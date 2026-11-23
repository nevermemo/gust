//! What the checks of `check-precompiled` share: the fixture's directories and cargo children, the scratch copies of
//! the fixture, and the failure report of a child.

use std::{
    env,
    env::consts::EXE_SUFFIX,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use super::{
    PRECOMPILED_DIR,
    environment::{StripReport, child_environment, directory_has_slang_tool},
};
use crate::{
    process::{PROFILE_VARIABLE, exit_code},
    workspace::cargo_target_dir,
};

/// What the fixture's children share.
pub(super) struct Fixture {
    pub(super) root: PathBuf,
    /// The cargo target directory, which holds `precompiled`, `precompiled-control` and `precompiled-scratch`.
    pub(super) target: PathBuf,
    /// The cargo target directory of every child, `<target>/precompiled`.
    pub(super) target_dir: PathBuf,
    /// The scratch copies of E3 and E4, `<target>/precompiled-scratch`.
    pub(super) scratch: PathBuf,
    pub(super) cargo: OsString,
    pub(super) environment: Vec<(OsString, OsString)>,
    pub(super) report: StripReport,
}

impl Fixture {
    pub(super) fn new(root: &Path) -> Self {
        let parent: Vec<(OsString, OsString)> = env::vars_os().collect();
        let target = root.join(cargo_target_dir(root));
        // Resolved before the stripping: the directory of the toolchain may be the one that is stripped on a machine
        // where Slang was unpacked beside it, and then this says so instead of failing to find cargo.
        let cargo = cargo_program(&parent);
        let mut report = StripReport::default();
        let environment = child_environment(&parent, &directory_has_slang_tool, &mut report);
        Self {
            root: root.to_path_buf(),
            target_dir: target.join("precompiled"),
            scratch: target.join("precompiled-scratch"),
            target,
            cargo,
            environment,
            report,
        }
    }

    /// A command that runs `program` in the stripped environment from the repository root.
    pub(super) fn command(&self, program: impl AsRef<OsStr>) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(&self.root)
            .env_clear()
            .envs(self.environment.iter().map(|(name, value)| (name, value)))
            .stdin(Stdio::null());
        command
    }

    pub(super) fn cargo(&self, args: &[&str]) -> Command {
        let mut command = self.command(&self.cargo);
        command.args(args);
        command
    }

    /// [`Self::cargo`] with `GUST_PROFILE` set to `profile` instead of the strict one (the control of E2, E6).
    pub(super) fn cargo_under_profile(&self, args: &[&str], profile: &str) -> Command {
        let mut command = self.cargo(args);
        command.env(PROFILE_VARIABLE, profile);
        command
    }

    /// [`Self::cargo`] with `GUST_PROFILE` removed: the build decides from cargo's `PROFILE` alone (E6c, E6d).
    pub(super) fn cargo_without_profile(&self, args: &[&str]) -> Command {
        let mut command = self.cargo(args);
        command.env_remove(PROFILE_VARIABLE);
        command
    }

    /// [`Self::cargo`] run in `directory` instead of the repository root, so that cargo reads no configuration of
    /// the repository (it looks from the current directory, not from `--manifest-path`).
    pub(super) fn cargo_in(&self, directory: &Path, args: &[&str]) -> Command {
        let mut command = self.cargo(args);
        command.current_dir(directory);
        command
    }

    /// `path` as `<target>/..` when it is under the target directory, relative to the repository when it is under that,
    /// else as given, with forward slashes: output without the machine's own paths.
    pub(super) fn shown(&self, path: &Path) -> String {
        let text = |path: &Path| path.to_string_lossy().replace('\\', "/");
        if let Ok(rest) = path.strip_prefix(&self.target) {
            return format!("<target>/{}", text(rest));
        }
        text(path.strip_prefix(&self.root).unwrap_or(path))
    }

    /// `<target>/precompiled-control`, the target directory of the control build of E2.
    pub(super) fn control_dir(&self) -> PathBuf {
        self.target_dir.with_file_name("precompiled-control")
    }

    /// `<target>/precompiled-scratch/target`, the target directory of the builds in the scratch copies (E3, E4). It is
    /// not the one of E1: both builds of `app` write `release/app`, and cargo does not link a fresh unit again, so a
    /// second run of the check would run the binary of the mutated copy as the binary of E1.
    pub(super) fn scratch_target_dir(&self) -> PathBuf {
        self.scratch.join("target")
    }

    /// `directory` as the argument of `--target-dir`; cargo takes it as text, so a path that is not valid UTF-8 would
    /// be altered, not passed.
    pub(super) fn target_dir_arg(directory: &Path) -> Result<&str, String> {
        directory.to_str().ok_or_else(|| {
            "check-precompiled: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path"
                .to_owned()
        })
    }

    /// `<directory>/release/app`.
    pub(super) fn app_binary(directory: &Path) -> PathBuf {
        directory.join("release").join(format!("app{EXE_SUFFIX}"))
    }
}

/// `$CARGO` as read from the captured environment `parent` (not from the process), else `cargo`; the name is
/// matched without regard to case, as Windows spells environment variable names.
fn cargo_program(parent: &[(OsString, OsString)]) -> OsString {
    parent
        .iter()
        .find(|(name, _)| name.to_string_lossy().eq_ignore_ascii_case("CARGO"))
        .map_or_else(|| OsString::from("cargo"), |(_, value)| value.clone())
}

// ---------------------------------------------------------------------------------------------------------------
// Scratch copies

/// Replaces the fixture's relative path to the repository (`path = "../../../`) with the absolute one, so that a
/// copy under the target directory still finds `gust` and `gust-wgpu`. `root` is written with forward slashes.
pub(crate) fn retarget_paths(manifest: &str, root: &str) -> String {
    manifest.replace("path = \"../../../", &format!("path = \"{root}/"))
}

/// Copies `tests/precompiled` to `destination` byte for byte, without a `target` directory, and points the copy's
/// `path` dependencies on the repository at its absolute location. An existing `destination` is deleted first.
pub(super) fn copy_fixture(fixture: &Fixture, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        fs::remove_dir_all(destination).map_err(|error| {
            format!(
                "could not delete the old scratch copy {}: {error}",
                destination.display()
            )
        })?;
    }
    let source = fixture.root.join(PRECOMPILED_DIR);
    copy_tree(&source, destination).map_err(|error| {
        format!("could not copy {PRECOMPILED_DIR} to the scratch directory: {error}")
    })?;
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    for manifest in [
        destination.join("kernels").join("Cargo.toml"),
        destination.join("app").join("Cargo.toml"),
    ] {
        let text = fs::read_to_string(&manifest)
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
        fs::write(&manifest, retarget_paths(&text, &root))
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
    }
    Ok(())
}

/// Recursive copy of `source` into `destination`, skipping directories called `target`.
fn copy_tree(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            if entry.file_name() != "target" {
                copy_tree(&entry.path(), &target)?;
            }
        } else if kind.is_file() {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// The body's `.wgsl` artifact of the generated set in `directory`, chosen by its manifest: the key a split entry names
/// in `split_of` (the set holds the body's file and its split's file, so a count cannot tell them apart). A manifest
/// that names no split belongs to a set with exactly one `.wgsl` file, which is the body's.
pub(super) fn body_wgsl_artifact(directory: &Path) -> Result<PathBuf, String> {
    let manifest = directory.join("manifest.json");
    let text = fs::read_to_string(&manifest)
        .map_err(|error| format!("{}: {error}", manifest.display()))?;
    let owner = text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("\"split_of\": \"")?;
        Some(rest.split('"').next()?.to_owned())
    });
    let Some(owner) = owner else {
        return single_artifact(directory, "wgsl");
    };
    let path = directory.join(format!("{owner}.wgsl"));
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "the manifest names `{owner}` as a split's body, but {} does not exist",
            path.display()
        ))
    }
}

/// The one file of `directory` that ends in `.extension` (and not in `.reflection.json`).
fn single_artifact(directory: &Path, extension: &str) -> Result<PathBuf, String> {
    let mut found = Vec::new();
    for entry in
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path
            .extension()
            .is_some_and(|candidate| candidate == extension)
        {
            found.push(path);
        }
    }
    match found.as_slice() {
        [one] => Ok(one.clone()),
        other => Err(format!(
            "expected exactly one .{extension} artifact in the scratch copy, found {}",
            other.len()
        )),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers

pub(super) fn require_success(step: &str, what: &str, output: &Output) -> Result<(), String> {
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "check-precompiled {step}: {what} failed (exit {}):\n{}{}",
        exit_code(output.status),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

pub(super) fn names_or_none(names: &[String]) -> String {
    if names.is_empty() {
        "none were set".to_owned()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
#[path = "../tests/precompiled_fixture.rs"]
mod tests;
