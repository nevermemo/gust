//! The pinned-Slang manager (D46, R4.8, section 2.5 of the M4 plan): which Slang `cargo gust` works with, how to get
//! it, and how to find it.
//!
//! Authors who export need Slang; consumers and production builds never call this module. The manager downloads the
//! release archive from Slang's own GitHub release with `curl -fsSL`, checks its SHA-256 (the std-only implementation
//! in `sha256.rs`) before anything is extracted, extracts it with `tar -xf` into a directory that is private to the
//! process, and renames that directory into place. The archive layout is kept, so the bundle's own `LICENSE`,
//! `LICENSES/` and `third-party-notices/` stay beside the binaries (`docs/development/LICENSES-SLANG.md` records the
//! audit of the three archives).
//!
//! # Where things live
//!
//! The cache root is `GUST_SLANG_HOME` when it is set, else `<CARGO_HOME or ~/.cargo>/gust/slang`. Under it:
//!
//! ```text
//! <root>/<version>/                  the extracted release (bin/slangc, lib/, include/, LICENSE, ...)
//! <root>/<version>.partial-<pid>/    an install in progress; renamed to <version> or removed
//! <root>/archives/<file>             the verified release archive (kept only if it was already there)
//! <root>/helper/<version>/           gust-slang-reflect built by `cargo gust slang build-helper`
//! ```
//!
//! # Finding Slang (`locate`)
//!
//! 1. `<GUST_SLANG_HOME>/<pin>` when the variable is set and that directory has `bin/slangc`; when the variable is
//!    set it is the only cache root, and step 2 is skipped;
//! 2. `<CARGO_HOME or ~/.cargo>/gust/slang/<pin>`;
//! 3. the first `slangc` on `PATH`, when its `-version` is the pin ([`matches_pin`]).
//!
//! Anything else is an error that names `cargo gust slang install` and the manual download URL. A `slangc` on `PATH`
//! of another version is accepted only with `--allow-slang-mismatch`, and then it is printed every time. A directory
//! in step 1 or 2 whose `slangc` reports another version is never substituted: it is a hard error. The pin is the
//! exact version or a listed validated build ([`VALIDATED_BUILDS`], empty for 2026.20); `build` and `locate` print a
//! note when the located version is not exactly the pin.
//!
//! # Known limitations
//!
//! - A process that is killed during an install leaves its `<version>.partial-<pid>` directory behind. The drop guard
//!   removes it on every error and panic, but not on a kill; nothing reads such a directory, and it can be deleted by
//!   hand.
//! - An installer that finds a cached archive failing its digest keeps it, but an installer that downloaded its own
//!   archive deletes it after a successful install. If two processes share one `archives/` directory, the first to
//!   finish can delete the file the other is about to extract; the second then fails and a rerun succeeds.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use gust::toolchain;

use crate::{sha256::sha256_hex, tools::executable_name};

/// The environment the rules read: `gust::toolchain` owns the cache-root and lookup rules (D109), and this tool
/// calls them.
pub(crate) use toolchain::Env;
/// The pin rule of `gust::toolchain` (exact equality, or a listed validated build), shared with the run-time bridge and
/// with `cargo xtask`.
pub(crate) use toolchain::matches_pin;

/// The Slang release `cargo gust` works with: `gust::toolchain::SLANG_PIN`, the one home of the pin. Slice m4.r ran
/// the Gpu, Slangc, Spirv and Reflect suites, `check-artifacts`, `check-precompiled` and `check-web` on it (STATUS.md
/// records the run); `cargo xtask check-full` runs at the M4 close. The pin moves only with a new run of those.
/// `cargo xtask` and the `gpu-software` workflow read it from `crates/gust/src/toolchain.rs`.
pub(crate) const SLANG_VERSION: &str = toolchain::SLANG_PIN;

/// Other `slangc -version` tags that count as the pin: `gust::toolchain::VALIDATED_BUILDS`, which ships empty (the
/// release archive prints the bare pin). Tests exercise the mechanism with a synthetic tag.
pub(crate) const VALIDATED_BUILDS: &[&str] = toolchain::VALIDATED_BUILDS;

/// One release archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Asset {
    pub(crate) version: &'static str,
    /// `std::env::consts::OS`: `windows`, `linux` or `macos`.
    pub(crate) os: &'static str,
    /// `std::env::consts::ARCH`.
    pub(crate) arch: &'static str,
    /// The file name in the release.
    pub(crate) file: &'static str,
    /// Lower-case hex SHA-256 of the archive: GitHub's release API digest, checked once by hand against the archive
    /// (slice log of m4.f).
    pub(crate) sha256: &'static str,
    pub(crate) size: u64,
}

/// The pinned release assets, one per `(version, os, arch)`.
pub(crate) const ASSETS: &[Asset] = &[
    Asset {
        version: SLANG_VERSION,
        os: "windows",
        arch: "x86_64",
        file: "slang-2026.20-windows-x86_64.zip",
        sha256: "e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174",
        size: 63_658_255,
    },
    Asset {
        version: SLANG_VERSION,
        os: "linux",
        arch: "x86_64",
        file: "slang-2026.20-linux-x86_64.tar.gz",
        sha256: "e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9",
        size: 83_407_874,
    },
    Asset {
        version: SLANG_VERSION,
        os: "macos",
        arch: "aarch64",
        file: "slang-2026.20-macos-aarch64.tar.gz",
        sha256: "07fa8833fdbb343c3674c84fdd885910f495f3daaaa0c732b32980151e8b58c4",
        size: 62_779_895,
    },
];

/// The asset for a platform, if the table has one.
pub(crate) fn asset_for(version: &str, os: &str, arch: &str) -> Option<&'static Asset> {
    ASSETS
        .iter()
        .find(|asset| asset.version == version && asset.os == os && asset.arch == arch)
}

/// Where the release archive is downloaded from.
pub(crate) fn download_url(asset: &Asset) -> String {
    format!(
        "https://github.com/shader-slang/slang/releases/download/v{}/{}",
        asset.version, asset.file
    )
}

/// The cache root `install` writes to: `GUST_SLANG_HOME`, else the default (`gust::toolchain::cache_root_in`).
pub(crate) fn install_root(env: &Env) -> Result<PathBuf, String> {
    toolchain::cache_root_in(env)
}

/// `bin/slangc` of an extracted release.
pub(crate) fn slangc_in(install_dir: &Path) -> PathBuf {
    install_dir.join("bin").join(executable_name("slangc"))
}

/// The directory `build-helper` writes the helper to for a managed install.
pub(crate) fn helper_dir(root: &Path, version: &str) -> PathBuf {
    root.join("helper").join(version)
}

/// Where a located `slangc` came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// `<GUST_SLANG_HOME>/<pin>`.
    Home,
    /// `<CARGO_HOME or ~/.cargo>/gust/slang/<pin>`.
    Cache,
    /// The first `slangc` on `PATH`.
    Path,
}

impl Source {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Home => "GUST_SLANG_HOME",
            Self::Cache => "cargo cache",
            Self::Path => "PATH",
        }
    }
}

/// A `slangc` that `locate` accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Located {
    pub(crate) slangc: PathBuf,
    /// What `slangc -version` printed.
    pub(crate) version: String,
    pub(crate) source: Source,
    /// Whether [`matches_pin`] holds; false only for a `PATH` `slangc` accepted by `--allow-slang-mismatch`.
    pub(crate) pinned: bool,
    /// The cache root, for a managed install (`Home` or `Cache`).
    pub(crate) root: Option<PathBuf>,
}

impl Located {
    /// The extracted release, for a managed install.
    pub(crate) fn install_dir(&self) -> Option<PathBuf> {
        self.root.as_ref().map(|root| root.join(SLANG_VERSION))
    }
}

/// The one-line note for a located Slang that is not exactly the pin: a validated build (accepted), or another
/// version that `--allow-slang-mismatch` let through. `None` when the version is exactly the pin.
pub(crate) fn version_note(located: &Located) -> Option<String> {
    if located.version == SLANG_VERSION {
        return None;
    }
    let slangc = located.slangc.display();
    Some(if located.pinned {
        format!(
            "note: using Slang `{}` at {slangc}, a validated build of the pinned {SLANG_VERSION}; the manifest records the build that compiled",
            located.version
        )
    } else {
        format!(
            "note: using Slang `{}` at {slangc}, not the pinned {SLANG_VERSION} (--allow-slang-mismatch); the manifest records the build that compiled",
            located.version
        )
    })
}

/// The text that tells the user how to get the pin; part of every "not found" error.
pub(crate) fn install_hint(env: &Env) -> String {
    let manual = asset_for(SLANG_VERSION, env::consts::OS, env::consts::ARCH).map_or_else(
        || {
            format!(
                "no release archive is pinned for {}-{}; install Slang {SLANG_VERSION} yourself and put its bin directory first on PATH",
                env::consts::OS,
                env::consts::ARCH
            )
        },
        |asset| {
            let archives = install_root(env).map_or_else(
                |_| "<cache>/archives".to_owned(),
                |root| root.join("archives").display().to_string(),
            );
            format!(
                "to do it by hand, download {} (SHA-256 {}) into {archives} and run the install command again",
                download_url(asset),
                asset.sha256
            )
        },
    );
    format!("run `cargo gust slang install` ({manual})")
}

/// Find the pinned Slang. `version_of` runs `slangc -version`; tests pass a fake.
pub(crate) fn locate(
    env: &Env,
    allow_mismatch: bool,
    version_of: &dyn Fn(&Path) -> Result<String, String>,
) -> Result<Located, String> {
    locate_with(env, allow_mismatch, version_of, VALIDATED_BUILDS)
}

/// [`locate`] with the validated builds given, so the tests can exercise the mechanism with a synthetic tag.
pub(crate) fn locate_with(
    env: &Env,
    allow_mismatch: bool,
    version_of: &dyn Fn(&Path) -> Result<String, String>,
    validated: &[&str],
) -> Result<Located, String> {
    // `GUST_SLANG_HOME`, when set, is the only cache root (section 2.5 reads "or"): an install elsewhere is not looked at.
    let source = if env.slang_home.is_some() {
        Source::Home
    } else {
        Source::Cache
    };
    if let Some(root) = toolchain::lookup_root_in(env)?
        && toolchain::managed_slangc(&root).is_file()
    {
        let install_dir = root.join(SLANG_VERSION);
        let slangc = slangc_in(&install_dir);
        let version = version_of(&slangc).map_err(|error| {
            format!(
                "{} cannot report its version ({error}); remove {} and run `cargo gust slang install`",
                slangc.display(),
                install_dir.display()
            )
        })?;
        if !matches_pin(&version, validated) {
            return Err(format!(
                "{} reports Slang `{version}`, not {SLANG_VERSION}; remove {} and run `cargo gust slang install`",
                slangc.display(),
                install_dir.display()
            ));
        }
        return Ok(Located {
            slangc,
            version,
            source,
            pinned: true,
            root: Some(root),
        });
    }
    let Some(slangc) = toolchain::find_on_path(&executable_name("slangc"), env.path.as_deref())
    else {
        return Err(format!(
            "Slang {SLANG_VERSION} was not found (no {SLANG_VERSION} directory in the cache and no slangc on PATH): {}",
            install_hint(env)
        ));
    };
    let version = version_of(&slangc).map_err(|error| {
        format!(
            "the slangc on PATH ({}) cannot report its version ({error}): {}",
            slangc.display(),
            install_hint(env)
        )
    })?;
    let pinned = matches_pin(&version, validated);
    if !pinned && !allow_mismatch {
        return Err(format!(
            "the slangc on PATH ({}) is Slang `{version}`, not {SLANG_VERSION}: {}, or pass --allow-slang-mismatch to use it anyway",
            slangc.display(),
            install_hint(env)
        ));
    }
    Ok(Located {
        slangc,
        version,
        source: Source::Path,
        pinned,
        root: None,
    })
}

/// The directories of a release that hold licence texts: 2026.20 ships `LICENSES/` (the SPDX texts of its own and its
/// components' licences) and `third-party-notices/` (one notice file per bundled component) beside `LICENSE`.
const NOTICE_DIRECTORIES: &[&str] = &["licenses", "third-party-notices"];

/// The licence and notice paths beside the binaries of a managed install, sorted by name: `LICENSE` and any notice
/// file a later release adds, and the [`NOTICE_DIRECTORIES`] (listed as directories, not expanded).
pub(crate) fn notice_files(install_dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(install_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if path.is_dir() {
                NOTICE_DIRECTORIES.contains(&name.as_str())
            } else {
                path.is_file()
                    && ["license", "licence", "notice", "copying", "third"]
                        .iter()
                        .any(|prefix| name.starts_with(prefix))
            }
        })
        .map(|entry| entry.path())
        .collect();
    found.sort();
    found
}

/// What `install` needs from the outside world; tests substitute fakes.
pub(crate) trait Fetch {
    /// Download `url` into the file `dest`.
    fn download(&self, url: &str, dest: &Path) -> Result<(), String>;
    /// Extract `archive` into the existing directory `dest`.
    fn extract(&self, archive: &Path, dest: &Path) -> Result<(), String>;
}

/// `curl` and `tar` as subprocesses (no HTTP dependency; both ship with Windows 10+, macOS and the CI runners).
pub(crate) struct SystemTools;

impl SystemTools {
    /// `tar`. On Windows it is the `tar.exe` of the system directory (bsdtar, which reads zip): a `tar` that Git for
    /// Windows puts first on `PATH` is GNU tar, which cannot read the zip archive and reads `C:` as a host name.
    fn tar() -> PathBuf {
        if cfg!(windows)
            && let Some(root) = env::var_os("SystemRoot")
        {
            let system = Path::new(&root).join("System32").join("tar.exe");
            if system.is_file() {
                return system;
            }
        }
        PathBuf::from("tar")
    }
}

impl Fetch for SystemTools {
    fn download(&self, url: &str, dest: &Path) -> Result<(), String> {
        let status = Command::new("curl")
            .args(["-fsSL", "--retry", "2", "--retry-delay", "2", "-o"])
            .arg(dest)
            .arg(url)
            .status()
            .map_err(|error| format!("cannot run `curl`: {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "`curl -fsSL {url}` failed ({})",
                status
                    .code()
                    .map_or_else(|| "no exit code".to_owned(), |code| format!("exit {code}"))
            ))
        }
    }

    fn extract(&self, archive: &Path, dest: &Path) -> Result<(), String> {
        let tar = Self::tar();
        let status = Command::new(&tar)
            .arg("-xf")
            .arg(archive)
            .arg("-C")
            .arg(dest)
            .status()
            .map_err(|error| format!("cannot run `{}`: {error}", tar.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "`tar -xf {}` failed ({})",
                archive.display(),
                status
                    .code()
                    .map_or_else(|| "no exit code".to_owned(), |code| format!("exit {code}"))
            ))
        }
    }
}

/// What `install` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Installed {
    /// `<root>/<version>` was already complete; nothing was touched.
    Already(PathBuf),
    /// The release was extracted and renamed into place.
    Fresh(PathBuf),
}

/// Removes a directory when dropped, unless it was disarmed: the "remove the partial directory on any failure" rule,
/// including a panic or an early return.
struct PartialDir<'a> {
    path: &'a Path,
    armed: bool,
}

impl Drop for PartialDir<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(self.path);
        }
    }
}

/// Check the archive at `path` against the asset's digest. Nothing is extracted before this passes.
fn verify_archive(path: &Path, asset: &Asset) -> Result<(), String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let actual = sha256_hex(&bytes);
    if actual.eq_ignore_ascii_case(asset.sha256) {
        Ok(())
    } else {
        Err(format!(
            "{} has SHA-256 {actual}, expected {} ({} bytes)",
            path.display(),
            asset.sha256,
            bytes.len()
        ))
    }
}

/// Install the release `asset` under `root`: obtain the archive, verify it, extract it into
/// `<root>/<version>.partial-<pid>` and rename that to `<root>/<version>`. On any failure the partial directory is
/// removed and `<root>/<version>` does not exist. `pid` names the partial directory, so two installs do not share one.
pub(crate) fn install(
    root: &Path,
    asset: &Asset,
    tools: &dyn Fetch,
    pid: u32,
) -> Result<Installed, String> {
    let version_dir = root.join(asset.version);
    if slangc_in(&version_dir).is_file() {
        return Ok(Installed::Already(version_dir));
    }
    if version_dir.exists() {
        return Err(format!(
            "{} exists but holds no bin/slangc; remove it and run `cargo gust slang install` again",
            version_dir.display()
        ));
    }
    let archives = root.join("archives");
    fs::create_dir_all(&archives)
        .map_err(|error| format!("cannot create {}: {error}", archives.display()))?;
    let archive = archives.join(asset.file);
    let mut downloaded = false;
    if archive.exists() {
        // A verified archive left by an earlier run, or one the user downloaded by hand. A bad one is reported, not
        // replaced: it is evidence of a damaged disk or a wrong file.
        verify_archive(&archive, asset).map_err(|error| {
            format!("the cached archive failed its check: {error}; delete it and run `cargo gust slang install` again")
        })?;
    } else {
        let url = download_url(asset);
        println!(
            "slang: downloading {} ({} bytes) from {url}",
            asset.file, asset.size
        );
        let part = archives.join(format!("{}.part-{pid}", asset.file));
        let _ = fs::remove_file(&part);
        let fetched = tools
            .download(&url, &part)
            .and_then(|()| verify_archive(&part, asset));
        if let Err(error) = fetched {
            let _ = fs::remove_file(&part);
            return Err(format!(
                "{error}; the download was discarded. To install by hand, download {url} into {} and run the install command again",
                archives.display()
            ));
        }
        fs::rename(&part, &archive).map_err(|error| {
            let _ = fs::remove_file(&part);
            format!(
                "cannot move the verified download to {}: {error}",
                archive.display()
            )
        })?;
        downloaded = true;
    }
    let partial = root.join(format!("{}.partial-{pid}", asset.version));
    let mut guard = PartialDir {
        path: &partial,
        armed: true,
    };
    let _ = fs::remove_dir_all(&partial);
    fs::create_dir_all(&partial)
        .map_err(|error| format!("cannot create {}: {error}", partial.display()))?;
    tools.extract(&archive, &partial)?;
    if !slangc_in(&partial).is_file() {
        return Err(format!(
            "{} has no bin/{}; this is not a Slang release archive",
            asset.file,
            executable_name("slangc")
        ));
    }
    if let Err(error) = fs::rename(&partial, &version_dir) {
        // Another process may have finished the same install between our check and the rename.
        if slangc_in(&version_dir).is_file() {
            drop(guard);
            return Ok(Installed::Already(version_dir));
        }
        return Err(format!(
            "cannot move {} to {}: {error}",
            partial.display(),
            version_dir.display()
        ));
    }
    guard.armed = false;
    if downloaded {
        let _ = fs::remove_file(&archive);
    }
    Ok(Installed::Fresh(version_dir))
}

/// Run `<slangc> -version` (`gust::toolchain::slangc_version`, which retries a busy executable); the error is the
/// text of its [`toolchain::VersionError`].
pub(crate) fn slangc_version(slangc: &Path) -> Result<String, String> {
    toolchain::slangc_version(slangc).map_err(|error| error.to_string())
}

#[cfg(test)]
#[path = "slang_tests.rs"]
mod tests;
