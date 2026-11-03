//! Where the managed Slang toolchain lives, and the one set of rules for finding it (D46, D72, D109).
//!
//! Support surface for `cargo-gust` and the run-time bridge, outside the pre-1.0 promise (hence the
//! `#![doc(hidden)]` below). The module is std-only and has no `crate::` or `super::` path, so an integration test can
//! compile this very file with `#[path = "../src/toolchain.rs"]`; the eventual `mod` line in `lib.rs` is gated on
//! `slang-runtime`, and nothing in here needs the feature.
//!
//! # Where things live
//!
//! The cache root is `GUST_SLANG_HOME` when it is set, else `<CARGO_HOME or ~/.cargo>/gust/slang`:
//!
//! ```text
//! <root>/<pin>/bin/slangc               the extracted release (`cargo gust slang install`)
//! <root>/helper/<pin>/gust-slang-reflect  the native helper (`cargo gust slang build-helper`)
//! ```
//!
//! # Finding things
//!
//! `slangc`: `<root>/<pin>/bin/slangc` when that file exists, else the first `slangc` in an absolute `PATH` entry.
//! When `GUST_SLANG_HOME` is set it is the only root. The pin is not judged here; [`check_slangc_version`] does that.
//!
//! The helper: `GUST_SLANG_REFLECT` when set, else `<root>/helper/<pin>/gust-slang-reflect`, else the repository
//! checkout's `target/slang-reflect/gust-slang-reflect`, which is looked at only when `xtask/Cargo.toml` exists two
//! directories above the crate's compile-time manifest directory (a registry build has no `xtask` there, so it never
//! looks at a path that belongs to someone else's tree).
//!
//! # No current-directory lookup
//!
//! Nothing here resolves a relative path against the current directory: a relative `GUST_SLANG_HOME` or `CARGO_HOME`
//! is refused or ignored, and an empty or relative `PATH` entry is skipped (an empty entry means the current
//! directory to many shells and loaders).
#![doc(hidden)]

use std::{
    env,
    env::consts::EXE_SUFFIX,
    error::Error,
    ffi::{OsStr, OsString},
    fmt,
    path::{Path, PathBuf},
    process::Command,
};

/// The Slang release GUST works with. This is the one home of the pin; `cargo-gust` and `cargo xtask` read it here.
pub const SLANG_PIN: &str = "2026.20";

/// The variable that names the cache root.
pub const HOME_VARIABLE: &str = "GUST_SLANG_HOME";

/// The variable that names the native reflection helper.
pub const HELPER_VARIABLE: &str = "GUST_SLANG_REFLECT";

const HELPER_NAME: &str = "gust-slang-reflect";

/// The part of the process environment the rules read. Tests build it by hand, so they never touch the real one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Env {
    /// `GUST_SLANG_HOME`.
    pub slang_home: Option<OsString>,
    /// `CARGO_HOME`.
    pub cargo_home: Option<OsString>,
    /// `HOME`, or `USERPROFILE` on Windows.
    pub home: Option<OsString>,
    /// `PATH`.
    pub path: Option<OsString>,
    /// `GUST_SLANG_REFLECT`.
    pub helper: Option<OsString>,
}

impl Env {
    /// Read the five variables from the process environment.
    pub fn from_process() -> Self {
        let home = if cfg!(windows) {
            env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"))
        } else {
            env::var_os("HOME")
        };
        Self {
            slang_home: env::var_os(HOME_VARIABLE),
            cargo_home: env::var_os("CARGO_HOME"),
            home,
            path: env::var_os("PATH"),
            helper: env::var_os(HELPER_VARIABLE),
        }
    }
}

/// The executable file name of a tool called `name` on this platform.
pub fn executable_name(name: &str) -> String {
    format!("{name}{EXE_SUFFIX}")
}

/// `GUST_SLANG_HOME`, when it is set: it must be a non-empty absolute path.
fn explicit_root(env: &Env) -> Result<Option<PathBuf>, String> {
    let Some(value) = &env.slang_home else {
        return Ok(None);
    };
    if value.is_empty() {
        return Err(format!(
            "{HOME_VARIABLE} is set but empty; give an absolute directory or unset it"
        ));
    }
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!(
            "{HOME_VARIABLE} must be an absolute path, not `{}`",
            path.display()
        ));
    }
    Ok(Some(path))
}

/// `<CARGO_HOME or ~/.cargo>/gust/slang`, when either is known and absolute.
fn default_root(env: &Env) -> Option<PathBuf> {
    let base = match &env.cargo_home {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from(env.home.as_ref().filter(|home| !home.is_empty())?).join(".cargo"),
    };
    base.is_absolute().then(|| base.join("gust").join("slang"))
}

/// The cache root an install writes to: `GUST_SLANG_HOME`, else `<CARGO_HOME or ~/.cargo>/gust/slang`. An error when
/// `GUST_SLANG_HOME` is empty or relative, or when neither fallback is usable.
pub fn cache_root_in(env: &Env) -> Result<PathBuf, String> {
    if let Some(root) = explicit_root(env)? {
        return Ok(root);
    }
    default_root(env).ok_or_else(|| {
        format!(
            "cannot tell where to install Slang: set {HOME_VARIABLE} to an absolute directory \
             (CARGO_HOME and the home directory are not usable)"
        )
    })
}

/// [`cache_root_in`] for the process environment.
pub fn cache_root() -> Result<PathBuf, String> {
    cache_root_in(&Env::from_process())
}

/// The cache root a lookup reads: as [`cache_root_in`], except that a root nobody can name is `Ok(None)` (there is
/// nothing to look in), not an error. An unusable `GUST_SLANG_HOME` is still an error: it is never replaced by
/// another root.
pub fn lookup_root_in(env: &Env) -> Result<Option<PathBuf>, String> {
    match explicit_root(env)? {
        Some(root) => Ok(Some(root)),
        None => Ok(default_root(env)),
    }
}

/// `bin/slangc` of the release extracted for the pin under `root`.
pub fn managed_slangc(root: &Path) -> PathBuf {
    root.join(SLANG_PIN)
        .join("bin")
        .join(executable_name("slangc"))
}

/// The helper executable `build-helper` writes for the pin under `root`.
pub fn managed_helper(root: &Path) -> PathBuf {
    root.join("helper")
        .join(SLANG_PIN)
        .join(executable_name(HELPER_NAME))
}

/// The first directory of `path_var` that holds a regular file called `file_name`. Empty and relative entries are
/// skipped (no current-directory lookup).
pub fn find_on_path(file_name: &str, path_var: Option<&OsStr>) -> Option<PathBuf> {
    env::split_paths(path_var?)
        .filter(|directory| directory.is_absolute())
        .map(|directory| directory.join(file_name))
        .find(|candidate| candidate.is_file())
}

/// The repository checkout's helper, `<manifest_dir>/../../target/slang-reflect/gust-slang-reflect`, when this
/// really is a checkout: `<manifest_dir>/../../xtask/Cargo.toml` must exist, so a crate unpacked from the registry
/// never looks at a path beside it. `None` when the marker or the helper is missing.
pub fn checkout_helper(manifest_dir: &Path) -> Option<PathBuf> {
    if !manifest_dir.join("../../xtask/Cargo.toml").is_file() {
        return None;
    }
    let helper = manifest_dir
        .join("../../target/slang-reflect")
        .join(executable_name(HELPER_NAME));
    helper.is_file().then_some(helper)
}

/// The `slangc` to run: the managed one for the pin, else the first on `PATH`; `Ok(None)` when there is neither.
/// The version is not judged here ([`check_slangc_version`]). `Err` for an unusable `GUST_SLANG_HOME`.
pub fn slangc_path_in(env: &Env) -> Result<Option<PathBuf>, String> {
    if let Some(root) = lookup_root_in(env)? {
        let managed = managed_slangc(&root);
        if managed.is_file() {
            return Ok(Some(managed));
        }
    }
    Ok(find_on_path(
        &executable_name("slangc"),
        env.path.as_deref(),
    ))
}

/// [`slangc_path_in`] for the process environment.
pub fn slangc_path() -> Result<Option<PathBuf>, String> {
    slangc_path_in(&Env::from_process())
}

/// The native helper to run: `GUST_SLANG_REFLECT` (as given, even when the file is missing, so a typo is reported
/// with that path), else the managed helper for the pin, else the checkout's ([`checkout_helper`], with
/// `manifest_dir` the compile-time manifest directory of the crate); `Ok(None)` when there is none. `Err` for an
/// empty `GUST_SLANG_REFLECT` or an unusable `GUST_SLANG_HOME`.
pub fn helper_path_in(env: &Env, manifest_dir: &Path) -> Result<Option<PathBuf>, String> {
    if let Some(value) = &env.helper {
        if value.is_empty() {
            return Err(format!(
                "{HELPER_VARIABLE} is set but empty; give the helper executable or unset it"
            ));
        }
        return Ok(Some(PathBuf::from(value)));
    }
    if let Some(root) = lookup_root_in(env)? {
        let managed = managed_helper(&root);
        if managed.is_file() {
            return Ok(Some(managed));
        }
    }
    Ok(checkout_helper(manifest_dir))
}

/// [`helper_path_in`] for the process environment and this crate's compile-time manifest directory.
pub fn helper_path() -> Result<Option<PathBuf>, String> {
    helper_path_in(&Env::from_process(), Path::new(env!("CARGO_MANIFEST_DIR")))
}

/// Why a `slangc` is not the pinned one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionError {
    /// `<slangc> -version` could not be run, or exited with a failure.
    Run(String),
    /// The output was not one version token.
    Output(String),
    /// A `slangc` of another version.
    Mismatch {
        /// The executable that was asked.
        slangc: PathBuf,
        /// What it printed.
        found: String,
    },
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Run(reason) | Self::Output(reason) => f.write_str(reason),
            Self::Mismatch { slangc, found } => write!(
                f,
                "{} reports Slang `{found}`, not the pinned {SLANG_PIN}; install the pin with \
                 `cargo gust slang install`",
                slangc.display()
            ),
        }
    }
}

impl Error for VersionError {}

/// The first line of `slangc -version` output, if it is a version.
pub fn parse_version_output(output: &str) -> Result<String, VersionError> {
    let line = output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let looks_like_version =
        line.starts_with(|c: char| c.is_ascii_digit()) && !line.contains(char::is_whitespace);
    if looks_like_version {
        Ok(line.to_owned())
    } else {
        Err(VersionError::Output(format!(
            "unexpected `-version` output: `{line}`"
        )))
    }
}

/// Run `<slangc> -version` and return the version it prints.
pub fn slangc_version(slangc: &Path) -> Result<String, VersionError> {
    let output = Command::new(slangc)
        .arg("-version")
        .output()
        .map_err(|error| {
            VersionError::Run(format!("cannot run `{}`: {error}", slangc.display()))
        })?;
    if !output.status.success() {
        return Err(VersionError::Run(format!(
            "`{} -version` failed: {}",
            slangc.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    // slangc has printed the version on stdout; fall back to stderr rather than fail on a build that uses it.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let text = if stdout.trim().is_empty() {
        String::from_utf8_lossy(&output.stderr)
    } else {
        stdout
    };
    parse_version_output(&text)
}

/// `Ok(version)` when `slangc -version` prints exactly [`SLANG_PIN`]; otherwise a [`VersionError`] that names the
/// executable, what it printed and the pin. Only the exact pin passes (D72: no listed build, no prefix match).
pub fn check_slangc_version(slangc: &Path) -> Result<String, VersionError> {
    let found = slangc_version(slangc)?;
    if found == SLANG_PIN {
        Ok(found)
    } else {
        Err(VersionError::Mismatch {
            slangc: slangc.to_path_buf(),
            found,
        })
    }
}
