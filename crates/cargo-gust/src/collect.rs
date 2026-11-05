//! Collecting the kernels of a package (R2-3, R4.3).
//!
//! `cargo gust` runs `cargo check -p <package>` in an isolated target directory, `<target>/gust-export`, with
//! `GUST_EXPORT_DIR` pointing at a directory that is new for this run (`records/<nonce>`) and `GUST_PROFILE=dev` in the
//! child environment. The `#[gpu]` macro tracks both variables with `option_env!`, so a new nonce re-expands exactly the
//! crates that contain kernels, and in export mode it writes `<key>.slang` and `<key>.kernel.json` for every kernel
//! into `<GUST_EXPORT_DIR>/<CARGO_PKG_NAME>/`. Only the target package's directory is read: path crates further down
//! the graph write their own sets, which are ignored. The record directory is removed after every run, whatever its
//! outcome, and a nonce is never reused, so a kernel deleted from the source leaves no record.
//!
//! The command constructors here only build a [`Command`]; tests assert on them without running cargo.

use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    json,
    record::{Record, read_records},
};

/// The directory of the export cache inside the normal target directory.
pub(crate) const EXPORT_DIR: &str = "gust-export";

/// The profile the collect step runs under: the user's crate may be configured strict, and collecting is not a
/// production build.
pub(crate) const COLLECT_PROFILE: &str = "dev";

/// A workspace package, as `cargo metadata` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Package {
    pub(crate) name: String,
    pub(crate) manifest_path: PathBuf,
}

impl Package {
    /// The package directory (where `gust-artifacts/` lives).
    pub(crate) fn directory(&self) -> PathBuf {
        self.manifest_path
            .parent()
            .map_or_else(PathBuf::new, Path::to_path_buf)
    }
}

/// What `cargo metadata --no-deps` says about the workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Workspace {
    pub(crate) target_directory: PathBuf,
    pub(crate) packages: Vec<Package>,
}

/// The cargo to run: the `CARGO` variable cargo sets for its subcommands, else `cargo` from `PATH`.
pub(crate) fn cargo_program(env_cargo: Option<OsString>) -> OsString {
    env_cargo
        .filter(|cargo| !cargo.is_empty())
        .unwrap_or_else(|| OsString::from("cargo"))
}

/// `cargo metadata --no-deps --format-version 1 [--manifest-path <path>]`.
pub(crate) fn metadata_command(cargo: &OsString, manifest_path: Option<&Path>) -> Command {
    let mut command = Command::new(cargo);
    command.args(["metadata", "--no-deps", "--format-version", "1"]);
    if let Some(path) = manifest_path {
        command.arg("--manifest-path").arg(path);
    }
    command
}

/// Read the output of [`metadata_command`].
pub(crate) fn parse_metadata(text: &str) -> Result<Workspace, String> {
    let root = json::parse(text).map_err(|error| format!("cargo metadata: {error}"))?;
    let target_directory = root
        .get("target_directory")
        .and_then(json::Json::as_str)
        .ok_or("cargo metadata: no `target_directory`")?;
    let packages = root
        .get("packages")
        .and_then(json::Json::as_array)
        .ok_or("cargo metadata: no `packages` array")?
        .iter()
        .map(|package| {
            let name = package
                .get("name")
                .and_then(json::Json::as_str)
                .ok_or("cargo metadata: a package has no `name`")?;
            let manifest_path = package
                .get("manifest_path")
                .and_then(json::Json::as_str)
                .ok_or_else(|| {
                    format!("cargo metadata: package `{name}` has no `manifest_path`")
                })?;
            Ok(Package {
                name: name.to_owned(),
                manifest_path: PathBuf::from(manifest_path),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Workspace {
        target_directory: PathBuf::from(target_directory),
        packages,
    })
}

/// A comparison key for a path: separators as `/`, no `\?` prefix, lower case on Windows only (its paths ignore case;
/// elsewhere `A` and `a` are different directories).
pub(crate) fn path_key(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let text = text.strip_prefix("//?/").unwrap_or(&text);
    let text = text.trim_end_matches('/');
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text.to_owned()
    }
}

/// The `Cargo.toml` that cargo would use from `start`: the closest one in `start` or an ancestor.
pub(crate) fn find_manifest(start: &Path, exists: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|directory| directory.join("Cargo.toml"))
        .find(|candidate| exists(candidate))
}

/// The package to work on: `name` when given (`-p`), else the package whose manifest is `manifest`.
pub(crate) fn select_package<'a>(
    workspace: &'a Workspace,
    name: Option<&str>,
    manifest: Option<&Path>,
) -> Result<&'a Package, String> {
    let listing = || {
        workspace
            .packages
            .iter()
            .map(|package| package.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(name) = name {
        return workspace
            .packages
            .iter()
            .find(|package| package.name == name)
            .ok_or_else(|| {
                format!(
                    "no workspace package `{name}` (workspace packages: {})",
                    listing()
                )
            });
    }
    let manifest = manifest
        .ok_or("no Cargo.toml found; run from a package directory or pass --manifest-path")?;
    let wanted = path_key(manifest);
    workspace
        .packages
        .iter()
        .find(|package| path_key(&package.manifest_path) == wanted)
        .ok_or_else(|| {
            format!(
                "the manifest here is not a package of the workspace (a virtual workspace root?); pass -p <crate> (workspace packages: {})",
                listing()
            )
        })
}

/// A run's collection plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectPlan {
    pub(crate) cargo: OsString,
    pub(crate) package: String,
    pub(crate) manifest_path: Option<PathBuf>,
    /// The normal target directory (`target_directory` of the metadata).
    pub(crate) target_directory: PathBuf,
    /// A string that is new for this run.
    pub(crate) nonce: String,
}

impl CollectPlan {
    /// `<target>/gust-export`, the isolated target directory of the collect step.
    pub(crate) fn export_target(&self) -> PathBuf {
        self.target_directory.join(EXPORT_DIR)
    }

    /// `<target>/gust-export/records/<nonce>`, the value of `GUST_EXPORT_DIR`.
    pub(crate) fn record_root(&self) -> PathBuf {
        self.export_target().join("records").join(&self.nonce)
    }

    /// Where the macro writes this package's records: `<record root>/<package>`.
    pub(crate) fn package_records(&self) -> PathBuf {
        self.record_root().join(&self.package)
    }
}

/// The collect command: `cargo check [--manifest-path <path>] -p <package> --target-dir <target>/gust-export` with
/// `GUST_EXPORT_DIR=<target>/gust-export/records/<nonce>` and `GUST_PROFILE=dev`.
///
/// Everything else in the environment (`RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `RUSTC_WRAPPER`, ...) is inherited, so a
/// `--cfg gust_dual` build is collected as such (F15).
pub(crate) fn collect_command(plan: &CollectPlan) -> Command {
    let mut command = Command::new(&plan.cargo);
    command.arg("check");
    if let Some(path) = &plan.manifest_path {
        command.arg("--manifest-path").arg(path);
    }
    command
        .arg("-p")
        .arg(&plan.package)
        .arg("--target-dir")
        .arg(plan.export_target())
        .env("GUST_EXPORT_DIR", plan.record_root())
        .env("GUST_PROFILE", COLLECT_PROFILE);
    command
}

/// `cargo clean -p <package> [--manifest-path <path>]` on the normal target directory (no `--target-dir`): the one-time
/// clean at the init transition (R2-3).
pub(crate) fn clean_command(
    cargo: &OsString,
    package: &str,
    manifest_path: Option<&Path>,
) -> Command {
    let mut command = Command::new(cargo);
    command.arg("clean");
    if let Some(path) = manifest_path {
        command.arg("--manifest-path").arg(path);
    }
    command.arg("-p").arg(package);
    command
}

/// A string that is new for every run of every process: process id and the nanoseconds since the epoch.
pub(crate) fn fresh_nonce() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    format!("{}-{nanos}", std::process::id())
}

/// Removes the record root of one run when dropped, so a failed run leaves no records either.
pub(crate) struct RecordRoot(PathBuf);

impl RecordRoot {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self(path)
    }
}

impl Drop for RecordRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The error for a collection that found no kernel.
pub(crate) fn zero_kernels_error(package: &str) -> String {
    format!(
        "zero kernels collected for `{package}`: the crate may have no `#[gpu]` module, or its gust-compute-macros predates export mode (GUST_EXPORT_DIR); nothing was written"
    )
}

/// Run the collect step and read the records of the target package. The record root is removed afterwards.
///
/// `run` executes the command and returns whether it succeeded; production runs it with inherited output.
pub(crate) fn collect(
    plan: &CollectPlan,
    run: &dyn Fn(&mut Command) -> Result<bool, String>,
) -> Result<Vec<Record>, String> {
    let _cleanup = RecordRoot::new(plan.record_root());
    let mut command = collect_command(plan);
    if !run(&mut command)? {
        return Err(format!(
            "`cargo check -p {}` failed in the collect step; fix the errors above",
            plan.package
        ));
    }
    let records = read_records(&plan.package_records())?;
    if records.is_empty() {
        return Err(zero_kernels_error(&plan.package));
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::Scratch;
    use std::ffi::OsStr;

    fn plan() -> CollectPlan {
        CollectPlan {
            cargo: OsString::from("cargo"),
            package: "my-kernels".to_owned(),
            manifest_path: None,
            target_directory: PathBuf::from("/work/target"),
            nonce: "1234-99".to_owned(),
        }
    }

    /// A path built with the platform separator, so the expectations hold on Windows too.
    fn path_text(parts: &[&str]) -> String {
        let mut path = PathBuf::new();
        for part in parts {
            path.push(part);
        }
        path.to_string_lossy().into_owned()
    }

    fn args(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn env_of(command: &Command, name: &str) -> Option<Option<String>> {
        command
            .get_envs()
            .find(|(key, _)| *key == OsStr::new(name))
            .map(|(_, value)| value.map(|value| value.to_string_lossy().into_owned()))
    }

    #[test]
    fn the_collect_command_checks_the_package_in_an_isolated_target_with_a_nonce_and_the_dev_profile()
     {
        let command = collect_command(&plan());
        assert_eq!(command.get_program(), "cargo");
        assert_eq!(
            args(&command),
            [
                "check",
                "-p",
                "my-kernels",
                "--target-dir",
                &path_text(&["/work/target", "gust-export"])
            ]
        );
        assert_eq!(
            env_of(&command, "GUST_EXPORT_DIR"),
            Some(Some(path_text(&[
                "/work/target",
                "gust-export",
                "records",
                "1234-99"
            ])))
        );
        assert_eq!(
            env_of(&command, "GUST_PROFILE"),
            Some(Some("dev".to_owned()))
        );
        // Nothing else is set or removed: RUSTFLAGS and friends are inherited so `--cfg gust_dual` collects as dual.
        assert_eq!(command.get_envs().count(), 2);
        assert_eq!(command.get_current_dir(), None);
    }

    #[test]
    fn a_manifest_path_is_passed_through_and_the_records_live_under_the_package_name() {
        let mut with_manifest = plan();
        with_manifest.manifest_path = Some(PathBuf::from("pkgs/k/Cargo.toml"));
        assert_eq!(
            args(&collect_command(&with_manifest)),
            [
                "check",
                "--manifest-path",
                "pkgs/k/Cargo.toml",
                "-p",
                "my-kernels",
                "--target-dir",
                &path_text(&["/work/target", "gust-export"])
            ]
        );
        assert_eq!(
            with_manifest.package_records(),
            Path::new(&path_text(&[
                "/work/target",
                "gust-export",
                "records",
                "1234-99",
                "my-kernels"
            ]))
        );
    }

    #[test]
    fn the_clean_command_cleans_the_package_in_the_normal_target_directory() {
        let command = clean_command(&OsString::from("cargo"), "my-kernels", None);
        assert_eq!(args(&command), ["clean", "-p", "my-kernels"]);
        assert_eq!(command.get_envs().count(), 0);
        let with_manifest = clean_command(
            &OsString::from("cargo"),
            "k",
            Some(Path::new("a/Cargo.toml")),
        );
        assert_eq!(
            args(&with_manifest),
            ["clean", "--manifest-path", "a/Cargo.toml", "-p", "k"]
        );
        // No --target-dir: the normal cache is the one that holds the stale expansion.
        assert!(!args(&with_manifest).iter().any(|arg| arg == "--target-dir"));
    }

    #[test]
    fn the_cargo_program_prefers_the_cargo_variable() {
        assert_eq!(cargo_program(Some(OsString::from("/x/cargo"))), "/x/cargo");
        assert_eq!(cargo_program(Some(OsString::new())), "cargo");
        assert_eq!(cargo_program(None), "cargo");
    }

    #[test]
    fn nonces_differ_between_calls() {
        let first = fresh_nonce();
        let second = fresh_nonce();
        assert_ne!(first, second);
        assert!(first.starts_with(&format!("{}-", std::process::id())));
    }

    const METADATA: &str = r#"{"packages":[
        {"name":"app","manifest_path":"C:\\ws\\app\\Cargo.toml","version":"0.1.0"},
        {"name":"kern-els","manifest_path":"C:\\ws\\kern\\Cargo.toml"}],
        "target_directory":"C:\\ws\\target","workspace_root":"C:\\ws"}"#;

    #[test]
    fn metadata_is_read_and_a_package_is_selected_by_name_or_by_manifest() {
        let workspace = parse_metadata(METADATA).unwrap();
        assert_eq!(workspace.target_directory, Path::new("C:\\ws\\target"));
        assert_eq!(workspace.packages.len(), 2);
        let by_name = select_package(&workspace, Some("kern-els"), None).unwrap();
        assert_eq!(by_name.name, "kern-els");
        // The manifest of the working directory is compared without regard to separators or `\\?\`, and on Windows case.
        let mut manifests = vec!["C:/ws/app/Cargo.toml", "\\\\?\\C:\\ws\\app\\Cargo.toml"];
        if cfg!(windows) {
            manifests.push("c:\\WS\\app\\cargo.toml");
        }
        for manifest in manifests {
            let found = select_package(&workspace, None, Some(Path::new(manifest))).unwrap();
            assert_eq!(found.name, "app", "{manifest}");
        }
        assert_eq!(by_name.directory(), Path::new("C:\\ws\\kern"));
    }

    #[test]
    fn selection_failures_name_the_available_packages() {
        let workspace = parse_metadata(METADATA).unwrap();
        let missing = select_package(&workspace, Some("nope"), None).unwrap_err();
        assert!(
            missing.contains("no workspace package `nope`") && missing.contains("app, kern-els"),
            "{missing}"
        );
        let root =
            select_package(&workspace, None, Some(Path::new("C:/ws/Cargo.toml"))).unwrap_err();
        assert!(
            root.contains("pass -p <crate>") && root.contains("app, kern-els"),
            "{root}"
        );
        let none = select_package(&workspace, None, None).unwrap_err();
        assert!(none.contains("--manifest-path"), "{none}");
    }

    #[test]
    fn unreadable_metadata_is_an_error() {
        for text in [
            "",
            "{}",
            r#"{"packages":[],"target_directory":1}"#,
            r#"{"target_directory":"/t"}"#,
            r#"{"target_directory":"/t","packages":[{"name":"x"}]}"#,
        ] {
            assert!(parse_metadata(text).is_err(), "{text}");
        }
    }

    #[test]
    fn the_closest_manifest_wins() {
        let exists = |path: &Path| {
            path_key(path).to_lowercase() == "/ws/a/cargo.toml"
                || path_key(path).to_lowercase() == "/cargo.toml"
        };
        assert_eq!(
            find_manifest(Path::new("/ws/a/src/deep"), &exists),
            Some(PathBuf::from("/ws/a").join("Cargo.toml"))
        );
        assert_eq!(find_manifest(Path::new("/other"), &|_| false), None);
    }

    fn write_record(root: &Path, key: &str, json: &str, slang: &str) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join(format!("{key}.kernel.json")), json).unwrap();
        fs::write(root.join(format!("{key}.slang")), slang).unwrap();
    }

    fn sample_plan(scratch: &Scratch) -> CollectPlan {
        CollectPlan {
            target_directory: scratch.0.join("target"),
            ..plan()
        }
    }

    #[test]
    fn collect_reads_only_the_target_package_and_removes_the_record_root_afterwards() {
        let scratch = Scratch::new("collect");
        let plan = sample_plan(&scratch);
        let run = |_: &mut Command| -> Result<bool, String> {
            write_record(
                &plan.package_records(),
                crate::testdata::SAMPLE_KEY,
                crate::testdata::SAMPLE_RECORD_JSON,
                crate::testdata::SAMPLE_SLANG,
            );
            // A path crate further down the graph writes its own set; it must be ignored, even though it is junk.
            let other = plan.record_root().join("dependency");
            fs::create_dir_all(&other).unwrap();
            fs::write(other.join("junk.txt"), "not a record").unwrap();
            Ok(true)
        };
        let records = collect(&plan, &run).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, crate::testdata::SAMPLE_KEY);
        assert!(
            !plan.record_root().exists(),
            "the record root must be removed"
        );
    }

    #[test]
    fn zero_kernels_is_an_error_and_so_is_a_failed_cargo_and_both_remove_the_record_root() {
        let scratch = Scratch::new("zero");
        let plan = sample_plan(&scratch);
        let created = |_: &mut Command| -> Result<bool, String> {
            fs::create_dir_all(plan.package_records()).unwrap();
            Ok(true)
        };
        let error = collect(&plan, &created).unwrap_err();
        assert_eq!(error, zero_kernels_error("my-kernels"));
        assert!(error.starts_with("zero kernels collected for `my-kernels`"));
        assert!(!plan.record_root().exists());
        let failed = |_: &mut Command| -> Result<bool, String> {
            fs::create_dir_all(plan.package_records()).unwrap();
            Ok(false)
        };
        assert!(
            collect(&plan, &failed)
                .unwrap_err()
                .contains("failed in the collect step")
        );
        assert!(!plan.record_root().exists());
    }

    #[test]
    fn stray_files_in_the_record_directory_and_half_records_are_errors() {
        let scratch = Scratch::new("stray");
        let dir = scratch.0.join("records");
        write_record(
            &dir,
            crate::testdata::SAMPLE_KEY,
            crate::testdata::SAMPLE_RECORD_JSON,
            crate::testdata::SAMPLE_SLANG,
        );
        assert_eq!(read_records(&dir).unwrap().len(), 1);
        // A temporary the macro could not remove is not a stray file.
        let leftover = dir.join(format!(".{}.slang.123.0.tmp", crate::testdata::SAMPLE_KEY));
        fs::write(&leftover, "half").unwrap();
        assert_eq!(read_records(&dir).unwrap().len(), 1);
        fs::remove_file(&leftover).unwrap();
        fs::write(dir.join("notes.txt"), "x").unwrap();
        assert!(
            read_records(&dir)
                .unwrap_err()
                .contains("unexpected file `notes.txt`")
        );
        fs::remove_file(dir.join("notes.txt")).unwrap();
        fs::remove_file(dir.join(format!("{}.slang", crate::testdata::SAMPLE_KEY))).unwrap();
        assert!(read_records(&dir).unwrap_err().contains("is incomplete"));
        assert_eq!(
            read_records(&scratch.0.join("missing")).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn a_record_whose_file_name_is_not_its_key_is_refused() {
        let scratch = Scratch::new("renamed");
        let dir = scratch.0.join("records");
        write_record(
            &dir,
            "ffffffffffffffff",
            crate::testdata::SAMPLE_RECORD_JSON,
            crate::testdata::SAMPLE_SLANG,
        );
        let error = read_records(&dir).unwrap_err();
        assert!(error.contains("does not match the record key"), "{error}");
    }
}
