//! `cargo xtask check-release [--check-only] [--work-dir DIR]`: the permanent release gate (slice mr.r, D70 extended,
//! D106, D107, D110). It proves that what `cargo publish` would upload is complete, bounded, documented, mutually
//! consistent and usable by someone who has only the README. It is name-agnostic: the publishable packages and their
//! names come from `cargo metadata` (`publish` is not `[]`), so the command passes unchanged before and after the
//! rename of mr.n. No crate name is spelled here except in tests; the README recipe and its fixture are text.
//!
//! - **R1**: one multi-package `cargo package --locked` of every publishable member, with verification (cargo builds
//!   each unpacked package; per crate it fails because a dependant needs the others on the registry, in one call the
//!   members verify against each other, mr.m0 P1). The output must say `Verifying` for each package. R1 and R8 build
//!   into a target directory inside the work directory, not the caller's: cargo fingerprints a registry crate by name
//!   and version, not by source, so a build that an earlier run left in a shared target would be reused (mr.r2).
//!   Cargo also unpacks the temporary registry into `$CARGO_HOME/registry/src/<hash of the registry's path>`, and the
//!   path (so the hash) is unique to a run's work directory name. After the steps, pass or fail, the run deletes
//!   exactly the entries that the `*.d` dep-info files in its own target name, if they are new and hold only the
//!   publishable `<name>-<version>` directories; it never deletes an `index.*` entry or an entry that no dep-info of
//!   its own names (that may be a concurrent run's). If its builds read such an entry that existed before the run (a
//!   killed run's, under a work directory name used again), cargo may have built from a stale copy and the check
//!   fails, naming the entry ([`clean_registry_src`]).
//! - **R2**: `cargo package --list` and the unpacked `.crate` files against the allow-list ([`package_allows`], the one
//!   home, which E8 of `check-precompiled` calls too), plus `LICENSE-MIT`, `LICENSE-APACHE` and `README.md`, no test
//!   module or test data.
//! - **R3**: the metadata fields, equal versions, and every internal dependency exact (`=<version>`, D107).
//! - **R4**: no member outside `crates/` is publishable, and no publishable member depends on an unpublished one.
//! - **R5**: `cargo doc --no-deps` of every library crate, every rustdoc warning an error ([`KNOWN_DOC_DEBT`] lists
//!   the ones admitted, each by text, none since mr.b2; a listed warning that disappears fails the check), on
//!   stable; and, when a nightly toolchain is installed, with the docs.rs features and `--cfg docsrs`. Without
//!   nightly the second run is printed as skipped.
//! - **R6**: each `.crate` file is under 2 MiB.
//! - **R7**: the README recipe, word for word. The packages are unpacked into the work directory; `hello-gust`, whose
//!   `Cargo.toml` and `src/main.rs` are the README's "Install from crates.io" code blocks (see
//!   `xtask/src/release/hello-gust/`; [`readme_problems`] and a unit test compare them), gets a `[patch.crates-io]` on
//!   the unpacked trees and is built with its dev feature; `cargo install --path` puts the unpacked binary package
//!   into the work directory with the same patches. The full run then uses Slang and the GPU: it runs the dev build,
//!   runs `cargo gust build -p hello-gust` with the installed tool, builds `--release` (the strict `native` profile
//!   inferred from the release build) and runs that.
//! - **R8**: `cargo publish --dry-run --locked` in the one-call form that mr.m0 P2 found to work, into the same
//!   target directory as R1 (the work directory's; same packaged source, so the reuse is sound).
//!
//! `--check-only` skips what needs Slang or an adapter: the runs of the dev build and the release build, and `cargo
//! gust build` (the release build needs its artifacts). Everything else runs, and needs the crates.io registry. The
//! work directory defaults to a new directory under the cargo target directory (under the temporary directory when
//! that lies inside the repository, where a consumer would read the repository's workspace); it must lie outside the
//! repository, not exist or be empty, and is deleted when the command ends. Every cargo child runs without
//! `GUST_PROFILE`, as a user's would; the xtask process's environment is never changed.

use std::{
    collections::BTreeSet,
    env,
    env::consts::EXE_SUFFIX,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    checks::cargo_home,
    json::{self, Json},
    precompiled::{is_inside, lock_packages, snapshot, unused_patch_lines},
    process::{PROFILE_VARIABLE, exit_code},
    workspace::workspace_root,
};

/// The fixture of R7, relative to the repository root: the `Cargo.toml` and `src/main.rs` of the README recipe.
const HELLO_DIR: &str = "xtask/src/release/hello-gust";

/// The heading of the README section whose two code blocks are the fixture.
const README_HEADING: &str = "## Install from crates.io";

/// The line that the recipe prints when the kernel ran on the GPU.
const HELLO_OUTPUT: &str = "hello-gust: [11.0, 22.0, 33.0, 44.0]";

/// R6: the largest `.crate` file (crates.io accepts 10 MiB; a file this large was packaged by mistake).
const MAX_CRATE_BYTES: u64 = 2 * 1024 * 1024;

/// The files, besides the licences, that a package must list (R2).
const REQUIRED_FILES: [&str; 3] = ["LICENSE-MIT", "LICENSE-APACHE", "README.md"];

/// rustdoc warnings that R5 admits as known debt. It is empty since slice mr.b2 (the documentation sweep) fixed the
/// three it owed, so every rustdoc warning fails R5. An entry admits one warning, at a file that ends in `file` (no
/// line number, so an edit above it does not break the entry), and a second match fails. R5 also fails when an entry
/// no longer occurs, so that the entry is deleted with the fix and the debt cannot go stale. `stable` is whether the
/// plain stable run shows it (else only the run with the docs.rs features does).
pub(crate) const KNOWN_DOC_DEBT: [DocDebt; 0] = [];

// ---------------------------------------------------------------------------------------------------------------
// The allow-lists of R2 (E8 of `check-precompiled` calls them)

/// One package and what its `cargo package --list` may hold besides `Cargo.toml`, `Cargo.toml.orig`, `Cargo.lock`,
/// `.cargo_vcs_info.json`, `src/**` (no test module, no `testdata`), `README*` and `LICENSE*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackageSpec<'a> {
    pub(crate) name: &'a str,
    /// `build.rs`.
    pub(crate) build_script: bool,
    /// `gust-artifacts/**` (a package of precompiled kernels only).
    pub(crate) artifacts: bool,
    /// `assets/**` (a binary-only package embeds its helper source from there).
    pub(crate) assets: bool,
    /// The target source that the package must list: `src/lib.rs`, or `src/main.rs` for a binary-only package.
    pub(crate) entry: &'a str,
}

impl<'a> PackageSpec<'a> {
    /// A package with a library at `src/lib.rs` and no `assets/`.
    pub(crate) const fn library(name: &'a str, build_script: bool, artifacts: bool) -> Self {
        Self {
            name,
            build_script,
            artifacts,
            assets: false,
            entry: "src/lib.rs",
        }
    }
}

/// The file names of the output of `cargo package --list`: one per line, backslashes read as slashes.
pub(crate) fn package_files(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(|line| line.trim().replace('\\', "/"))
        .filter(|line| !line.is_empty())
        .collect()
}

/// Whether `file` (relative, forward slashes) is a unit-test module or test data: a path component `tests` or
/// `testdata` (also as the file `testdata.rs`), or a file name ending in `_tests.rs`. Those read repository paths
/// and do not ship (mr.a). A name that merely starts with `testdata` is no test data.
fn is_test_source(file: &str) -> bool {
    file.split('/')
        .any(|part| part == "tests" || part == "testdata" || part == "testdata.rs")
        || file.ends_with("_tests.rs")
}

/// Whether the allow-list of R2 admits `file` (a relative path with forward slashes) in the package `spec`.
pub(crate) fn package_allows(spec: &PackageSpec, file: &str) -> bool {
    if file.split('/').any(|part| part == ".." || part == ".") {
        return false;
    }
    match file {
        "Cargo.toml" | "Cargo.toml.orig" | "Cargo.lock" | ".cargo_vcs_info.json" => true,
        "build.rs" => spec.build_script,
        _ if file.starts_with("src/") => !is_test_source(file),
        _ if file.starts_with("gust-artifacts/") => spec.artifacts,
        _ if file.starts_with("assets/") => spec.assets,
        _ => {
            let upper = file.to_ascii_uppercase();
            !file.contains('/') && (upper.starts_with("README") || upper.starts_with("LICENSE"))
        }
    }
}

/// The problems of a package's files (its `--list` or its unpacked tree): a file off the allow-list (named), a
/// `gust-artifacts/` entry in a package that must have none, a missing `Cargo.toml` or entry source, and, for the
/// package with artifacts, a missing manifest or committed artifact file (`required` holds the names relative to
/// `gust-artifacts/`).
pub(crate) fn package_problems(
    spec: &PackageSpec,
    files: &[String],
    required: &[String],
) -> Vec<String> {
    let mut problems = Vec::new();
    for file in files {
        if package_allows(spec, file) {
            continue;
        }
        problems.push(if file.starts_with("gust-artifacts/") {
            format!(
                "`{}` lists `{file}`, but this package ships no gust-artifacts/ (R2-5)",
                spec.name
            )
        } else if file.starts_with("src/") && is_test_source(file) {
            format!(
                "`{}` lists `{file}`, a test module or test data, which must not ship (add it to `exclude`)",
                spec.name
            )
        } else {
            format!(
                "`{}` lists `{file}`, which is not on the allow-list (src/**, Cargo.toml, Cargo.toml.orig, \
                 Cargo.lock, .cargo_vcs_info.json{}{}{}, README*, LICENSE*)",
                spec.name,
                if spec.build_script { ", build.rs" } else { "" },
                if spec.artifacts {
                    ", gust-artifacts/**"
                } else {
                    ""
                },
                if spec.assets { ", assets/**" } else { "" }
            )
        });
    }
    for needed in ["Cargo.toml", spec.entry] {
        if !files.iter().any(|file| file == needed) {
            problems.push(format!("`{}` does not list `{needed}`", spec.name));
        }
    }
    if spec.artifacts {
        let mut wanted: Vec<String> = vec!["manifest.json".to_owned()];
        wanted.extend(required.iter().cloned());
        wanted.sort();
        wanted.dedup();
        for name in wanted {
            let path = format!("gust-artifacts/{name}");
            if !files.contains(&path) {
                problems.push(format!("`{}` does not list `{path}`", spec.name));
            }
        }
    }
    problems
}

/// What R2 asks of a published package beyond [`package_problems`]: both licence files and the README.
pub(crate) fn required_file_problems(name: &str, files: &[String]) -> Vec<String> {
    REQUIRED_FILES
        .iter()
        .filter(|needed| !files.iter().any(|file| file == *needed))
        .map(|needed| format!("`{name}` does not list `{needed}`"))
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Options

/// What `cargo xtask check-release` was asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    /// `--check-only`: skip what needs Slang or an adapter (the runs and `cargo gust build` of R7).
    pub(crate) check_only: bool,
    /// `--work-dir DIR`: where R1 to R8 unpack and build; a directory that does not exist yet or is empty.
    pub(crate) work_dir: Option<PathBuf>,
}

/// The options of `check-release`.
pub(crate) fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut rest = args.iter();
    while let Some(argument) = rest.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (argument.as_str(), None),
        };
        match name {
            "--check-only" => {
                if inline.is_some() {
                    return Err("`--check-only` takes no value".into());
                }
                options.check_only = true;
            }
            "--work-dir" => {
                let value = match inline {
                    Some(value) => value,
                    None => rest
                        .next()
                        .cloned()
                        .ok_or("`--work-dir` needs a directory")?,
                };
                if value.is_empty() {
                    return Err("`--work-dir` needs a non-empty directory".into());
                }
                if options.work_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("`--work-dir` was given twice".into());
                }
            }
            _ => {
                return Err(format!(
                    "check-release takes only `--check-only` and `--work-dir DIR`, not `{argument}`"
                ));
            }
        }
    }
    Ok(options)
}

// ---------------------------------------------------------------------------------------------------------------
// Members, from `cargo metadata`

/// The kind of a dependency edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DependencyKind {
    Normal,
    Build,
    Dev,
}

/// One dependency of a member, as `cargo metadata` lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dependency {
    pub(crate) name: String,
    pub(crate) requirement: String,
    pub(crate) kind: DependencyKind,
}

/// One workspace member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Member {
    pub(crate) name: String,
    pub(crate) version: String,
    /// `publish` is not `[]`.
    pub(crate) publishable: bool,
    /// The member's directory relative to the workspace root, forward slashes.
    pub(crate) directory: String,
    pub(crate) has_library: bool,
    pub(crate) has_binary: bool,
    /// The name of the first `bin` target.
    pub(crate) binary: Option<String>,
    pub(crate) build_script: bool,
    /// The source file of the library (else of the first binary), relative to the member's directory.
    pub(crate) entry: String,
    pub(crate) description: bool,
    pub(crate) license: bool,
    pub(crate) repository: bool,
    pub(crate) documentation: bool,
    pub(crate) keywords: usize,
    pub(crate) categories: usize,
    pub(crate) rust_version: bool,
    pub(crate) dependencies: Vec<Dependency>,
    /// `[package.metadata.docs.rs] features` and `rustdoc-args`.
    pub(crate) docs_features: Vec<String>,
    pub(crate) docs_rustdoc_args: Vec<String>,
}

impl Member {
    /// The allow-list entry of this member.
    fn spec(&self) -> PackageSpec<'_> {
        PackageSpec {
            name: &self.name,
            build_script: self.build_script,
            artifacts: false,
            assets: self.has_binary && !self.has_library,
            entry: &self.entry,
        }
    }
}

/// What `cargo metadata --no-deps` says about the workspace.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Metadata {
    pub(crate) members: Vec<Member>,
    pub(crate) target_directory: PathBuf,
}

fn text(value: Option<&Json>) -> Option<&str> {
    value.and_then(Json::as_str)
}

fn strings(value: Option<&Json>) -> Vec<String> {
    value
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(Json::string_or_null)
        .collect()
}

fn present(value: Option<&Json>) -> bool {
    text(value).is_some_and(|text| !text.trim().is_empty())
}

fn slashes(path: &str) -> String {
    path.replace('\\', "/")
}

/// The members in the output of `cargo metadata --format-version 1 --no-deps`.
pub(crate) fn parse_metadata(output: &str) -> Result<Metadata, String> {
    let parsed = json::parse(output).map_err(|error| format!("cargo metadata output: {error}"))?;
    let root = slashes(
        text(parsed.get("workspace_root")).ok_or("cargo metadata has no `workspace_root`")?,
    );
    let target_directory = text(parsed.get("target_directory"))
        .map(PathBuf::from)
        .ok_or("cargo metadata has no `target_directory`")?;
    let packages = parsed
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata output has no `packages` array")?;
    let mut members = Vec::new();
    for package in packages {
        let name = text(package.get("name"))
            .ok_or("a package in the cargo metadata output has no name")?
            .to_owned();
        let manifest = slashes(text(package.get("manifest_path")).ok_or_else(|| {
            format!("`{name}` has no manifest_path in the cargo metadata output")
        })?);
        let directory = manifest
            .rsplit_once('/')
            .map_or("", |(directory, _)| directory)
            .to_owned();
        let directory = directory
            .strip_prefix(&root)
            .map(|rest| rest.trim_start_matches('/').to_owned())
            .ok_or_else(|| format!("`{name}` lies outside the workspace root"))?;
        let targets = package
            .get("targets")
            .and_then(Json::as_array)
            .unwrap_or(&[]);
        let kinds = |target: &Json| strings(target.get("kind"));
        let library = targets.iter().find(|target| {
            kinds(target).iter().any(|kind| {
                ["lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"]
                    .contains(&kind.as_str())
            })
        });
        let binary = targets
            .iter()
            .find(|target| kinds(target).iter().any(|kind| kind == "bin"));
        let prefix = if directory.is_empty() {
            format!("{root}/")
        } else {
            format!("{root}/{directory}/")
        };
        let source = |target: Option<&Json>| {
            target
                .and_then(|target| text(target.get("src_path")))
                .map(slashes)
                .and_then(|path| path.strip_prefix(&prefix).map(str::to_owned))
        };
        let entry = source(library)
            .or_else(|| source(binary))
            .unwrap_or_else(|| "src/lib.rs".to_owned());
        let publishable = match package.get("publish") {
            Some(Json::Array(registries)) => !registries.is_empty(),
            _ => true,
        };
        let mut dependencies = Vec::new();
        for dependency in package
            .get("dependencies")
            .and_then(Json::as_array)
            .unwrap_or(&[])
        {
            let kind = match text(dependency.get("kind")) {
                Some("dev") => DependencyKind::Dev,
                Some("build") => DependencyKind::Build,
                _ => DependencyKind::Normal,
            };
            dependencies.push(Dependency {
                name: text(dependency.get("name")).unwrap_or("").to_owned(),
                requirement: text(dependency.get("req")).unwrap_or("*").to_owned(),
                kind,
            });
        }
        let docs = package
            .get("metadata")
            .and_then(|metadata| metadata.get("docs"))
            .and_then(|docs| docs.get("rs"));
        members.push(Member {
            version: text(package.get("version")).unwrap_or("").to_owned(),
            publishable,
            directory,
            has_library: library.is_some(),
            has_binary: binary.is_some(),
            binary: binary
                .and_then(|target| text(target.get("name")))
                .map(str::to_owned),
            build_script: targets
                .iter()
                .any(|target| kinds(target).iter().any(|kind| kind == "custom-build")),
            entry,
            description: present(package.get("description")),
            license: present(package.get("license")) || present(package.get("license_file")),
            repository: present(package.get("repository")),
            documentation: present(package.get("documentation")),
            keywords: strings(package.get("keywords")).len(),
            categories: strings(package.get("categories")).len(),
            rust_version: present(package.get("rust_version")),
            dependencies,
            docs_features: strings(docs.and_then(|docs| docs.get("features"))),
            docs_rustdoc_args: strings(docs.and_then(|docs| docs.get("rustdoc-args"))),
            name,
        });
    }
    members.sort_by(|first, second| first.name.cmp(&second.name));
    Ok(Metadata {
        members,
        target_directory,
    })
}

/// R3 and R4: the metadata of the publishable members and the shape of the publishable set.
pub(crate) fn metadata_problems(members: &[Member]) -> Vec<String> {
    let mut problems = Vec::new();
    let publishable: Vec<&Member> = members.iter().filter(|member| member.publishable).collect();
    if publishable.is_empty() {
        problems.push("no workspace member is publishable (every `publish` is `[]`)".to_owned());
    }
    // R3: fields.
    for member in &publishable {
        let name = &member.name;
        for (present, field) in [
            (member.description, "description"),
            (member.license, "license"),
            (member.repository, "repository"),
            (member.documentation, "documentation"),
            (member.rust_version, "rust-version"),
            (member.keywords > 0, "keywords"),
            (member.categories > 0, "categories"),
        ] {
            if !present {
                problems.push(format!("R3: `{name}` has no `{field}`"));
            }
        }
        if !member.has_library && !member.has_binary {
            problems.push(format!(
                "R3: `{name}` has neither a library nor a binary target"
            ));
        }
    }
    // R3: one version, every internal dependency exact.
    let versions: BTreeSet<&str> = publishable
        .iter()
        .map(|member| member.version.as_str())
        .collect();
    if versions.len() > 1 {
        let listed: Vec<String> = publishable
            .iter()
            .map(|member| format!("{} {}", member.name, member.version))
            .collect();
        problems.push(format!(
            "R3: the published packages must share one version (D107), but they are {}",
            listed.join(", ")
        ));
    }
    for member in &publishable {
        for dependency in &member.dependencies {
            let Some(target) = members
                .iter()
                .find(|other| other.name == dependency.name && other.name != member.name)
            else {
                continue;
            };
            // A dev-dependency without a version is dropped by `cargo publish`.
            if dependency.kind == DependencyKind::Dev && dependency.requirement == "*" {
                continue;
            }
            let exact = format!("={}", target.version);
            if dependency.requirement != exact {
                problems.push(format!(
                    "R3: `{}` depends on `{}` with `{}`, not the exact `{exact}` (D107)",
                    member.name, dependency.name, dependency.requirement
                ));
            }
            // R4: a published package cannot depend on one that is not published.
            if dependency.kind != DependencyKind::Dev && !target.publishable {
                problems.push(format!(
                    "R4: `{}` depends on `{}`, which has `publish = false`",
                    member.name, dependency.name
                ));
            }
        }
    }
    // R4: the publishable set is the crates under `crates/` and nothing else.
    for member in members {
        let under_crates = member.directory.starts_with("crates/");
        if member.publishable && !under_crates {
            problems.push(format!(
                "R4: `{}` ({}) is publishable but lies outside crates/; set `publish = false`",
                member.name, member.directory
            ));
        }
    }
    problems
}

// ---------------------------------------------------------------------------------------------------------------
// R5: rustdoc

/// A rustdoc warning that is known, owned by a later slice, and admitted until then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DocDebt {
    /// Text of the message.
    pub(crate) needle: &'static str,
    /// The end of the path of the file that the warning points at (forward slashes, no line number).
    pub(crate) file: &'static str,
    /// Whether the plain stable run shows it; otherwise only the run with the docs.rs features does.
    pub(crate) stable: bool,
}

/// One diagnostic of `cargo doc --message-format json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DocMessage {
    pub(crate) level: String,
    pub(crate) text: String,
    pub(crate) location: String,
}

/// The compiler diagnostics (warnings and errors) in the JSON lines that `cargo doc --message-format json` printed.
pub(crate) fn doc_messages(stdout: &str) -> Vec<DocMessage> {
    let mut found = Vec::new();
    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('{'))
    {
        let Ok(parsed) = json::parse(line) else {
            continue;
        };
        if text(parsed.get("reason")) != Some("compiler-message") {
            continue;
        }
        let Some(message) = parsed.get("message") else {
            continue;
        };
        let level = text(message.get("level")).unwrap_or("");
        if level != "warning" && level != "error" {
            continue;
        }
        let location = message
            .get("spans")
            .and_then(Json::as_array)
            .and_then(|spans| spans.first())
            .map(|span| {
                let file = text(span.get("file_name")).unwrap_or("?");
                let line = match span.get("line_start") {
                    Some(Json::Number(number)) => number.as_str(),
                    _ => "?",
                };
                format!("{}:{line}", slashes(file))
            })
            .unwrap_or_default();
        found.push(DocMessage {
            level: level.to_owned(),
            text: text(message.get("message")).unwrap_or("").to_owned(),
            location,
        });
    }
    found
}

/// A line of cargo's own summary (`generated 2 warnings`), which is no diagnostic of the code.
fn is_summary(message: &DocMessage) -> bool {
    message.text.contains(" generated ") && message.text.contains("warning")
}

/// Whether the warning `message` is the one that `entry` admits: its text holds the needle and its file ends in the
/// entry's file.
fn debt_matches(entry: &DocDebt, message: &DocMessage) -> bool {
    let file = message
        .location
        .rsplit_once(':')
        .map_or(message.location.as_str(), |(file, _)| file);
    message.level == "warning" && message.text.contains(entry.needle) && file.ends_with(entry.file)
}

/// The problems of one `cargo doc` run: every diagnostic that is not known debt, and every debt entry of this run
/// that did not occur. `with_features` is whether this is the docs.rs-shaped run; `debt` is [`KNOWN_DOC_DEBT`] in
/// production.
pub(crate) fn doc_problems(
    label: &str,
    messages: &[DocMessage],
    with_features: bool,
    debt: &[DocDebt],
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut seen = vec![0_usize; debt.len()];
    for message in messages.iter().filter(|message| !is_summary(message)) {
        let known = debt.iter().position(|entry| debt_matches(entry, message));
        match known {
            Some(index) => {
                seen[index] += 1;
                if seen[index] > 1 {
                    problems.push(format!(
                        "R5 {label}: {}: {} ({}); the known rustdoc warning `{}` admits one occurrence in `{}`, 
                         this is another",
                        message.level, message.text, message.location, debt[index].needle, debt[index].file
                    ));
                }
            }
            None => problems.push(format!(
                "R5 {label}: {}: {} ({})",
                message.level, message.text, message.location
            )),
        }
    }
    for (entry, seen) in debt.iter().zip(seen) {
        if seen == 0 && (entry.stable || with_features) {
            problems.push(format!(
                "R5 {label}: the known rustdoc warning `{}` in `{}` no longer occurs; delete its entry from 
                 KNOWN_DOC_DEBT",
                entry.needle, entry.file
            ));
        }
    }
    problems
}

/// The docs.rs-shaped request of R5: `--features` entries (`package/feature`) and the rustdoc flags, from each
/// library member's `[package.metadata.docs.rs]`.
pub(crate) fn docsrs_request(libraries: &[&Member]) -> (Vec<String>, String) {
    let mut features = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    for member in libraries {
        for feature in &member.docs_features {
            features.push(format!("{}/{feature}", member.name));
        }
        let group = member.docs_rustdoc_args.join(" ");
        if !group.is_empty() && !groups.contains(&group) {
            groups.push(group);
        }
    }
    (features, groups.join(" "))
}

// ---------------------------------------------------------------------------------------------------------------
// R7: the README recipe

/// The two code blocks of the README section "Install from crates.io": the `toml` one and the `rust` one, each as
/// the text between the fences, with `\n` line ends and one at the end.
pub(crate) fn readme_blocks(readme: &str) -> Result<(String, String), String> {
    let readme = readme.replace("\r\n", "\n");
    let mut lines = readme.lines();
    if !lines.any(|line| line.trim_end() == README_HEADING) {
        return Err(format!("the README has no `{README_HEADING}` section"));
    }
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut open: Option<(String, String)> = None;
    for line in lines {
        if let Some((language, body)) = open.as_mut() {
            if line.trim_end() == "```" {
                blocks.push((std::mem::take(language), std::mem::take(body)));
                open = None;
            } else {
                body.push_str(line);
                body.push('\n');
            }
        } else if let Some(language) = line.strip_prefix("```") {
            open = Some((language.trim().to_owned(), String::new()));
        } else if line.starts_with("## ") {
            break;
        }
    }
    let find = |language: &str| {
        let matching: Vec<&(String, String)> = blocks
            .iter()
            .filter(|(found, _)| found == language)
            .collect();
        match matching.as_slice() {
            [(_, body)] => Ok(body.clone()),
            other => Err(format!(
                "the README section `{README_HEADING}` must have exactly one ```{language} block, it has {}",
                other.len()
            )),
        }
    };
    Ok((find("toml")?, find("rust")?))
}

/// The fixture file with `\n` line ends.
fn normalized(text: &str) -> String {
    let mut text = text.replace("\r\n", "\n");
    while text.ends_with("\n\n") {
        text.pop();
    }
    text
}

/// R7's first requirement: the README blocks and the fixture are the same text.
pub(crate) fn readme_problems(
    readme: &str,
    fixture_manifest: &str,
    fixture_source: &str,
) -> Vec<String> {
    let (manifest, source) = match readme_blocks(readme) {
        Ok(blocks) => blocks,
        Err(error) => return vec![error],
    };
    let mut problems = Vec::new();
    for (what, readme_text, fixture_text, file) in [
        ("Cargo.toml", manifest, fixture_manifest, "Cargo.toml"),
        ("main.rs", source, fixture_source, "src/main.rs"),
    ] {
        if normalized(&readme_text) != normalized(fixture_text) {
            problems.push(format!(
                "R7: the README's {what} block differs from {HELLO_DIR}/{file}; the recipe is built word for word, \
                 so change both"
            ));
        }
    }
    problems
}

/// The value of the first `key = "value"` line in the table `[section]` of `manifest`.
pub(crate) fn section_value(manifest: &str, section: &str, key: &str) -> Option<String> {
    let header = format!("[{section}]");
    let mut inside = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            inside = line == header;
            continue;
        }
        if !inside {
            continue;
        }
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        if let Some((value, _)) = rest
            .trim()
            .strip_prefix('"')
            .and_then(|rest| rest.split_once('"'))
        {
            return Some(value.to_owned());
        }
    }
    None
}

/// The `(key, line)` pairs of the table `[section]` of `manifest`.
fn section_entries<'a>(manifest: &'a str, section: &str) -> Vec<(&'a str, &'a str)> {
    let header = format!("[{section}]");
    let mut inside = false;
    let mut entries = Vec::new();
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            inside = line == header;
            continue;
        }
        if !inside || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, _)) = line.split_once('=') {
            entries.push((key.trim(), line));
        }
    }
    entries
}

/// The problems of the fixture's `[dependencies]`: there is at least one, and each is a publishable member of the
/// workspace with the plain or caret requirement that D107 gives users (`"0.1"`) and that the version being released
/// satisfies. Whether the fixture names every crate a user needs is for R7's build to show.
pub(crate) fn fixture_dependency_problems(manifest: &str, members: &[Member]) -> Vec<String> {
    let mut problems = Vec::new();
    let dependencies = section_entries(manifest, "dependencies");
    if dependencies.is_empty() {
        problems.push("R7: the fixture Cargo.toml has no [dependencies]".to_owned());
    }
    for (key, line) in dependencies {
        match members
            .iter()
            .find(|member| member.publishable && member.name == key)
        {
            None => problems.push(format!(
                "R7: the fixture depends on `{key}`, which is no publishable member of the workspace (a rename? \
                 change the README block and the fixture together)"
            )),
            Some(member) => {
                let satisfied = dependency_requirement(line)
                    .is_some_and(|requirement| caret_satisfied(&requirement, &member.version));
                if !satisfied {
                    problems.push(format!(
                        "R7: the fixture's `{key}` line must give a caret requirement such as \"0.1\" that the \
                         version being released ({}) satisfies",
                        member.version
                    ));
                }
            }
        }
    }
    problems
}

/// The version requirement of a dependency line `key = "req"` or `key = { version = "req", .. }`.
fn dependency_requirement(line: &str) -> Option<String> {
    let value = line.split_once('=')?.1.trim();
    let quoted = |text: &str| {
        let text = text.trim_start().strip_prefix('"')?;
        Some(text[..text.find('"')?].to_owned())
    };
    if value.starts_with('"') {
        return quoted(value);
    }
    let after = value
        .split_once("version")?
        .1
        .trim_start()
        .strip_prefix('=')?;
    quoted(after)
}

/// Whether the caret requirement `requirement` (`0.1`, `^0.1.0`; any other operator is refused) is satisfied by
/// `version` (`major.minor.patch`, a pre-release suffix ignored): at least the requirement, below the next version
/// that cargo counts as breaking.
fn caret_satisfied(requirement: &str, version: &str) -> bool {
    let numbers = |text: &str| -> Option<Vec<u64>> {
        let core = text.split(['-', '+']).next()?;
        let parts: Option<Vec<u64>> = core.split('.').map(|part| part.parse().ok()).collect();
        parts.filter(|parts| (1..=3).contains(&parts.len()))
    };
    let requirement = requirement.trim();
    let requirement = requirement.strip_prefix('^').unwrap_or(requirement);
    let (Some(required), Some(actual)) = (numbers(requirement), numbers(version)) else {
        return false;
    };
    if actual.len() != 3 {
        return false;
    }
    let mut lower = [0_u64; 3];
    lower[..required.len()].copy_from_slice(&required);
    let mut upper = lower;
    let bump = if lower[0] > 0 || required.len() == 1 {
        0
    } else if lower[1] > 0 || required.len() == 2 {
        1
    } else {
        2
    };
    upper[bump] += 1;
    for slot in upper.iter_mut().skip(bump + 1) {
        *slot = 0;
    }
    let actual = [actual[0], actual[1], actual[2]];
    actual >= lower && actual < upper
}

/// The fixture's `Cargo.toml` plus a `[patch.crates-io]` table that points every publishable package at its unpacked
/// tree (`trees` has the name and the absolute directory, forward slashes).
pub(crate) fn patched_manifest(fixture: &str, trees: &[(String, String)]) -> String {
    let mut manifest = fixture.replace("\r\n", "\n");
    if !manifest.ends_with('\n') {
        manifest.push('\n');
    }
    // The empty `[workspace]` makes the consumer its own workspace, so that a work directory below some other
    // workspace (a `--work-dir` under a checkout, say) does not fail.
    manifest.push_str(
        "\n# Added by `cargo xtask check-release`: the consumer is its own workspace, and the packages are not on the \
         registry yet; these are the unpacked .crate files.\n[workspace]\n\n[patch.crates-io]\n",
    );
    for (name, tree) in trees {
        manifest.push_str(&format!("{name} = {{ path = \"{tree}\" }}\n"));
    }
    manifest
}

/// The `--config` arguments of `cargo install --path` that patch every package except `skip` onto its tree.
pub(crate) fn patch_config_arguments(trees: &[(String, String)], skip: &str) -> Vec<String> {
    trees
        .iter()
        .filter(|(name, _)| name != skip)
        .flat_map(|(name, tree)| {
            [
                "--config".to_owned(),
                format!("patch.crates-io.{name}.path=\"{tree}\""),
            ]
        })
        .collect()
}

/// The problems of the consumer's `Cargo.lock`: each library package once and with no `source` (a path source has
/// none; a registry one reads `registry+https://..`), and some registry package, so that an unreadable lock cannot
/// pass.
pub(crate) fn consumer_lock_problems(lock: &str, libraries: &[&str]) -> Vec<String> {
    let packages = lock_packages(lock);
    let mut problems = Vec::new();
    for name in libraries {
        let entries: Vec<_> = packages
            .iter()
            .filter(|package| package.name == *name)
            .collect();
        match entries.as_slice() {
            [] => problems.push(format!("R7: the lock has no package `{name}`")),
            [only] => {
                if let Some(source) = &only.source {
                    problems.push(format!(
                        "R7: the lock's `{name}` has the source `{source}`: it came from a registry, not from the \
                         unpacked package"
                    ));
                }
            }
            many => problems.push(format!(
                "R7: the lock has {} packages named `{name}`",
                many.len()
            )),
        }
    }
    if !packages.iter().any(|package| {
        package
            .source
            .as_deref()
            .is_some_and(|source| source.starts_with("registry+"))
    }) {
        problems.push(
            "R7: the lock has no registry package (wgpu should be), so it was not read".to_owned(),
        );
    }
    problems
}

/// The patches that cargo reported unused, for the library packages (a binary-only package is no dependency of the
/// consumer, so its patch is unused by design).
pub(crate) fn unused_library_patches(stderr: &str, libraries: &[&str]) -> Vec<String> {
    unused_patch_lines(stderr)
        .into_iter()
        .filter(|line| {
            libraries
                .iter()
                .any(|name| line.contains(&format!("`{name} v")))
        })
        .collect()
}

/// Whether `stdout` has the recipe's result line.
pub(crate) fn recipe_ran(stdout: &str) -> bool {
    stdout.lines().any(|line| line.trim() == HELLO_OUTPUT)
}

// ---------------------------------------------------------------------------------------------------------------
// R6, R8

/// R6: `sizes` has each `.crate` file's name and byte count.
pub(crate) fn size_problems(sizes: &[(String, u64)]) -> Vec<String> {
    sizes
        .iter()
        .filter(|(_, bytes)| *bytes > MAX_CRATE_BYTES)
        .map(|(name, bytes)| {
            format!(
                "R6: `{name}` is {bytes} bytes, over the {MAX_CRATE_BYTES} limit; find the file with `cargo package \
                 --list`"
            )
        })
        .collect()
}

/// R8: the dry run must end each package with `aborting upload due to dry run`.
pub(crate) fn dry_run_problems(stderr: &str, packages: usize) -> Vec<String> {
    let aborted = stderr.matches("aborting upload due to dry run").count();
    if aborted == packages {
        Vec::new()
    } else {
        vec![format!(
            "R8: the dry run ended {aborted} of {packages} packages with `aborting upload due to dry run`"
        )]
    }
}

/// The lines of `cargo publish --dry-run` that say a crate of that name and version already exists on the registry.
pub(crate) fn existing_crate_lines(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|line| line.contains("already exists on crates.io"))
        .map(|line| line.trim().to_owned())
        .collect()
}

/// Whether the output of `rustup toolchain list` has a nightly toolchain.
pub(crate) fn has_nightly(listing: &str) -> bool {
    listing
        .lines()
        .any(|line| line.trim().starts_with("nightly"))
}

// ---------------------------------------------------------------------------------------------------------------
// The run

/// `$CARGO` when set (it is under `cargo xtask`), else `cargo`.
fn cargo_program() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// The directory that holds a run's files. It is deleted by [`WorkDir::finish`], whose failure is reported; the drop
/// is only the fallback for a run that panics. Shared by `check-release` and E8 of `check-precompiled`.
pub(crate) struct WorkDir {
    pub(crate) path: PathBuf,
    label: String,
    deleted: bool,
}

impl WorkDir {
    /// `requested`, or a new directory `<prefix>-<pid>-<nonce>` under `parent` (under the temporary directory when
    /// `parent` lies inside the repository `root`). It must lie outside the repository and not exist yet or be
    /// empty; a directory with content is never deleted. `label` starts every message.
    pub(crate) fn create(
        label: &str,
        prefix: &str,
        root: &Path,
        parent: &Path,
        requested: Option<&Path>,
    ) -> Result<Self, String> {
        let path = match requested {
            Some(path) => path.to_path_buf(),
            None => {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_nanos());
                let name = format!("{prefix}-{}-{nanos}", std::process::id());
                if is_inside(root, parent) {
                    env::temp_dir().join(name)
                } else {
                    parent.join(name)
                }
            }
        };
        let path = std::path::absolute(&path)
            .map_err(|error| format!("{label}: {}: {error}", path.display()))?;
        if is_inside(root, &path) {
            return Err(format!(
                "{label}: the work directory must lie outside the repository (a consumer inside it would read the \
                 repository's cargo configuration and workspace)"
            ));
        }
        if path.exists() {
            let empty = fs::read_dir(&path)
                .map_err(|error| format!("{label}: {}: {error}", path.display()))?
                .next()
                .is_none();
            if !empty {
                return Err(format!(
                    "{label}: the work directory {} is not empty; the run deletes its work directory when it ends, \
                     so choose a new or empty one",
                    path.display()
                ));
            }
        }
        fs::create_dir_all(&path)
            .map_err(|error| format!("{label}: {}: {error}", path.display()))?;
        Ok(Self {
            path,
            label: label.to_owned(),
            deleted: false,
        })
    }

    /// Deletes the directory. `Ok` only when it is gone; a failure (a locked file, say) names the directory that is
    /// left behind.
    pub(crate) fn delete(&mut self) -> Result<(), String> {
        if self.deleted {
            return Ok(());
        }
        match fs::remove_dir_all(&self.path) {
            Ok(()) => {}
            Err(_) if !self.path.exists() => {}
            Err(error) => {
                return Err(format!(
                    "{}: could not delete the work directory {} ({:.1} MiB left behind): {error}; delete it by hand",
                    self.label,
                    self.path.display(),
                    directory_size(&self.path) as f64 / (1024.0 * 1024.0)
                ));
            }
        }
        self.deleted = true;
        Ok(())
    }

    /// Deletes the directory and returns `outcome`, or the deletion's failure (also when the run failed).
    pub(crate) fn finish(mut self, outcome: Result<String, String>) -> Result<String, String> {
        let deletion = self.delete();
        finish_outcome(outcome, deletion)
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        if !self.deleted {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// The result of a run together with the result of deleting its work directory: a failure of either is a failure,
/// and both are shown when both failed.
pub(crate) fn finish_outcome(
    outcome: Result<String, String>,
    deletion: Result<(), String>,
) -> Result<String, String> {
    match (outcome, deletion) {
        (Ok(summary), Ok(())) => Ok(summary),
        (Ok(_), Err(failure)) => Err(failure),
        (Err(failure), Ok(())) => Err(failure),
        (Err(failure), Err(deleting)) => Err(format!("{failure}\n{deleting}")),
    }
}

/// The bytes of the files below `directory`.
fn directory_size(directory: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => directory_size(&entry.path()),
            Ok(_) => entry.metadata().map_or(0, |metadata| metadata.len()),
            Err(_) => 0,
        })
        .sum()
}

/// The names of the entries of `directory` (none when it cannot be read).
pub(crate) fn registry_entries(directory: &Path) -> BTreeSet<String> {
    fs::read_dir(directory)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// What [`clean_registry_src`] did.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RegistryCleanup {
    /// The entries it deleted, with their sizes in bytes.
    pub(crate) removed: Vec<(String, u64)>,
    /// The entries of this run's builds that it left alone, each with the reason.
    pub(crate) kept: Vec<String>,
    /// The deletions that failed.
    pub(crate) failed: Vec<String>,
    /// The entries that this run's builds read although they existed before the run: cargo may have built from a
    /// stale copy, so the run fails.
    pub(crate) reused: Vec<String>,
    /// Set when there was nothing to look at (no cargo home).
    pub(crate) skipped: Option<String>,
}

impl RegistryCleanup {
    /// Whether the run must fail whatever its steps said: a deletion failed, or a stale entry was read.
    pub(crate) fn is_failure(&self) -> bool {
        !self.failed.is_empty() || !self.reused.is_empty()
    }

    /// One sentence for the summary and the failure text.
    pub(crate) fn describe(&self) -> String {
        if let Some(reason) = &self.skipped {
            return format!("cargo registry/src not cleaned: {reason}");
        }
        let mut parts = Vec::new();
        if !self.removed.is_empty() {
            let list: Vec<String> = self
                .removed
                .iter()
                .map(|(name, bytes)| {
                    format!("{name} ({:.1} MiB)", *bytes as f64 / (1024.0 * 1024.0))
                })
                .collect();
            parts.push(format!(
                "deleted {} cargo registry/src entr{} that this run's builds read: {}",
                list.len(),
                if list.len() == 1 { "y" } else { "ies" },
                list.join(", ")
            ));
        }
        if !self.reused.is_empty() {
            parts.push(format!(
                "this run's builds read the cargo registry/src entr{} {}, which existed before the run (a killed \
                 run, or a work directory name used before): cargo may have built from a stale copy of the \
                 packages, so the result cannot be trusted; delete {} and run again",
                if self.reused.len() == 1 { "y" } else { "ies" },
                self.reused.join(", "),
                if self.reused.len() == 1 { "it" } else { "them" }
            ));
        }
        if !self.kept.is_empty() {
            parts.push(format!(
                "left alone (not deleted by this check): {}",
                self.kept.join("; ")
            ));
        }
        if !self.failed.is_empty() {
            parts.push(format!("could not delete: {}", self.failed.join("; ")));
        }
        if parts.is_empty() {
            return "no dep-info file of this run's builds names a cargo registry/src entry of the run; nothing \
                    deleted"
                .to_owned();
        }
        parts.join("; ")
    }
}

/// The `registry/src` entry names that `dep_info` (the text of one rustc dep-info `.d` file) mentions: the component
/// after `registry/src/` in any path, whichever way the paths are spelled (`\` or `/`).
pub(crate) fn dep_info_entries(dep_info: &str) -> BTreeSet<String> {
    const MARK: &str = "registry/src/";
    let text = dep_info.replace('\\', "/");
    let mut names = BTreeSet::new();
    let mut rest = text.as_str();
    while let Some(at) = rest.find(MARK) {
        rest = &rest[at + MARK.len()..];
        let name: String = rest
            .chars()
            .take_while(|c| *c != '/' && !c.is_whitespace())
            .collect();
        if !name.is_empty() {
            names.insert(name);
        }
    }
    names
}

/// The `registry/src` entries that the `*.d` files below `target` name: the ones this run's own builds read. Each run's
/// temporary registry lies in its own target directory, so the hash in the entry's name is the run's alone; another
/// run's entry is never named here.
pub(crate) fn target_dep_info_entries(target: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut pending = vec![target.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => pending.push(path),
                Ok(kind) if kind.is_file() && path.extension().is_some_and(|ext| ext == "d") => {
                    if let Ok(bytes) = fs::read(&path) {
                        names.extend(dep_info_entries(&String::from_utf8_lossy(&bytes)));
                    }
                }
                _ => {}
            }
        }
    }
    names
}

/// Deletes the entries of `src` (cargo's `registry/src`) that `named` lists (see [`target_dep_info_entries`]), that
/// were not in `before`, are not named `index.*` and hold only directories named like one of `packages`
/// (`<name>-<version>`): the copies that cargo unpacked from the temporary registry of `cargo package` and `cargo
/// publish` for this run's builds. An entry that is named and shaped so, but existed before the run, is in
/// [`RegistryCleanup::reused`] and is not deleted: this run did not make it, and its builds read it. A named entry of
/// another shape is never deleted and is named in the result. An entry that no dep-info names is never touched (it
/// may be another run's).
pub(crate) fn clean_registry_src(
    src: &Path,
    before: &BTreeSet<String>,
    named: &BTreeSet<String>,
    packages: &BTreeSet<String>,
) -> RegistryCleanup {
    let mut result = RegistryCleanup::default();
    for name in named {
        if name.starts_with("index.") {
            continue;
        }
        let path = src.join(name);
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        let children = if meta.file_type().is_dir() {
            registry_entries(&path)
        } else {
            BTreeSet::new()
        };
        let ours = !children.is_empty()
            && children.iter().all(|child| {
                packages.contains(child)
                    && fs::symlink_metadata(path.join(child))
                        .is_ok_and(|meta| meta.file_type().is_dir())
            });
        if !ours {
            // Another registry that the builds read (an older crates.io entry, say): not ours, and not news when it
            // is old.
            if !before.contains(name) {
                let listed: Vec<&str> = children.iter().map(String::as_str).take(4).collect();
                result.kept.push(format!(
                    "{name} (holds {}{})",
                    if listed.is_empty() {
                        "nothing".to_owned()
                    } else {
                        listed.join(", ")
                    },
                    if children.len() > listed.len() {
                        ", .."
                    } else {
                        ""
                    }
                ));
            }
            continue;
        }
        if before.contains(name) {
            result.reused.push(name.clone());
            continue;
        }
        let bytes = directory_size(&path);
        match fs::remove_dir_all(&path) {
            Ok(()) => result.removed.push((name.clone(), bytes)),
            Err(error) => result.failed.push(format!("{name}: {error}")),
        }
    }
    result
}

/// The entries of cargo's `registry/src` before a run, to tell afterwards which ones the run made.
struct RegistryWatch {
    src: Option<PathBuf>,
    before: BTreeSet<String>,
    packages: BTreeSet<String>,
}

impl RegistryWatch {
    /// Lists `registry/src`; `packages` are the `<name>-<version>` directories that may be deleted.
    fn begin(publishable: &[&Member]) -> Self {
        let src = cargo_home().map(|home| home.join("registry").join("src"));
        let before = src.as_deref().map(registry_entries).unwrap_or_default();
        Self {
            src,
            before,
            packages: publishable
                .iter()
                .map(|member| format!("{}-{}", member.name, member.version))
                .collect(),
        }
    }

    /// The cleanup, for the entries that the dep-info files below `target` name.
    fn finish(self, target: &Path) -> RegistryCleanup {
        match self.src {
            Some(src) => clean_registry_src(
                &src,
                &self.before,
                &target_dep_info_entries(target),
                &self.packages,
            ),
            None => RegistryCleanup {
                skipped: Some("neither CARGO_HOME nor the home directory is set".to_owned()),
                ..RegistryCleanup::default()
            },
        }
    }
}

/// The last `limit` characters of `text`, so that a failure shows its end and not a whole build log.
fn tail(text: &str, limit: usize) -> String {
    let text = text.trim_end();
    if text.len() <= limit {
        return text.to_owned();
    }
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("[..]{}", &text[start..])
}

/// The facts of the run that every step shares.
struct Run<'a> {
    root: &'a Path,
    cargo: OsString,
    work: &'a Path,
    check_only: bool,
    /// The reflection helper that the children of the full run get as `GUST_SLANG_REFLECT` when the caller set none.
    helper: Option<PathBuf>,
}

impl Run<'_> {
    /// A command in `directory`, with no stdin and without `GUST_PROFILE`: the build decides from cargo's profile
    /// alone, as it does for a user.
    fn command(&self, program: impl AsRef<std::ffi::OsStr>, directory: &Path) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(directory)
            .env_remove(PROFILE_VARIABLE)
            .stdin(Stdio::null());
        command
    }

    /// `cargo` with `args` in the repository.
    fn cargo(&self, args: &[&str]) -> Command {
        let mut command = self.command(&self.cargo, self.root);
        command.args(args);
        command
    }

    /// `cargo` with `args` in `directory`, building into `target`.
    fn cargo_in(&self, directory: &Path, target: &Path, args: &[&str]) -> Command {
        let mut command = self.command(&self.cargo, directory);
        command.args(args).env("CARGO_TARGET_DIR", target);
        command
    }

    /// `command` with `GUST_SLANG_REFLECT` set to the default helper when the caller set none (see [`helper_choice`]).
    /// That is the child's environment only, not this process's.
    fn bridge(&self, mut command: Command) -> Command {
        if let Some(helper) = &self.helper {
            command.env(HELPER_VARIABLE, helper);
        }
        command
    }

    /// Runs `command` and returns its output when it succeeds.
    fn step(&self, step: &str, what: &str, mut command: Command) -> Result<Output, String> {
        println!("check-release {step}: {what}");
        let output = command
            .output()
            .map_err(|error| format!("check-release {step}: could not run {what}: {error}"))?;
        if output.status.success() {
            return Ok(output);
        }
        Err(format!(
            "check-release {step}: {what} failed (exit {}):\n{}\n{}",
            exit_code(output.status),
            tail(&String::from_utf8_lossy(&output.stdout), 4000),
            tail(&String::from_utf8_lossy(&output.stderr), 8000)
        ))
    }
}

/// `path` as a command-line argument.
fn path_arg(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("check-release: {} is not valid UTF-8", path.display()))
}

fn problems_to_error(step: &str, problems: &[String]) -> Result<(), String> {
    if problems.is_empty() {
        return Ok(());
    }
    Err(format!(
        "check-release {step} found {} problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    ))
}

/// `cargo xtask check-release [--check-only] [--work-dir DIR]`.
pub(crate) fn check_release(args: &[String]) -> Result<(), String> {
    println!("{}", check_release_report(args)?);
    Ok(())
}

/// [`check_release`] returning the pass summary instead of printing it, so `check-full` can record it.
pub(crate) fn check_release_report(args: &[String]) -> Result<String, String> {
    let options = parse_args(args)?;
    let root = workspace_root()?;
    let cargo = cargo_program();

    let mut metadata_command = Command::new(&cargo);
    metadata_command
        .current_dir(&root)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--color",
            "never",
        ])
        .env_remove(PROFILE_VARIABLE)
        .stdin(Stdio::null());
    let output = metadata_command
        .output()
        .map_err(|error| format!("check-release: could not run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "check-release: cargo metadata failed:\n{}",
            tail(&String::from_utf8_lossy(&output.stderr), 4000)
        ));
    }
    let metadata = parse_metadata(&String::from_utf8_lossy(&output.stdout))?;
    let publishable: Vec<&Member> = metadata
        .members
        .iter()
        .filter(|member| member.publishable)
        .collect();

    // R3 and R4 read the metadata only.
    problems_to_error("R3/R4 (metadata)", &metadata_problems(&metadata.members))?;
    let names: Vec<&str> = publishable
        .iter()
        .map(|member| member.name.as_str())
        .collect();
    println!(
        "check-release R3/R4: {} publishable package(s) ({}) share version {}, with the metadata fields and exact \
         internal dependencies; every other member has publish = false",
        names.len(),
        names.join(", "),
        publishable[0].version
    );

    // The full run needs Slang and a reflection helper: refuse before the first step, not after minutes of work.
    let helper = if options.check_only {
        None
    } else {
        slang_environment(&root)?
    };
    let work = WorkDir::create(
        "check-release",
        "gust-release",
        &root,
        &metadata.target_directory,
        options.work_dir.as_deref(),
    )?;
    let outcome = {
        let run = Run {
            root: &root,
            cargo,
            work: &work.path,
            check_only: options.check_only,
            helper,
        };
        release_steps(&run, &metadata, &publishable)
    };
    // "deleted" is said only after the deletion succeeded.
    work.finish(outcome)
        .map(|summary| format!("{summary}, deleted."))
}

/// R1, R2, R5 to R8 in the work directory of `run`, then the removal of the unpacked-registry copies that R1 and R8
/// made in cargo's `registry/src` (pass or fail; see [`clean_registry_src`]). Returns the summary up to the size of
/// the work directory.
fn release_steps(
    run: &Run,
    metadata: &Metadata,
    publishable: &[&Member],
) -> Result<String, String> {
    let watch = RegistryWatch::begin(publishable);
    let outcome = all_steps(run, metadata, publishable);
    let cleanup = watch.finish(&registry_target(run.work));
    let said = cleanup.describe();
    println!("check-release: {said}");
    match outcome {
        Ok(summary) if !cleanup.is_failure() => Ok(format!(
            "{summary}. Registry leftovers: {said}. Work directory {:.1} MiB",
            directory_size(run.work) as f64 / (1024.0 * 1024.0)
        )),
        Ok(_) => Err(format!("check-release: {said}")),
        Err(failure) => Err(format!("{failure}\ncheck-release: {said}")),
    }
}

/// R1, R2, R5 to R8; the summary up to R8.
fn all_steps(run: &Run, metadata: &Metadata, publishable: &[&Member]) -> Result<String, String> {
    let listing = r2_lists(run, publishable)?;
    let crates = r1_package(run, publishable)?;
    let unpacked = r2_unpacked(run, publishable)?;
    let sizes = r6_sizes(&crates, publishable)?;
    let docs = r5_docs(run, publishable)?;
    let recipe = r7_recipe(run, &metadata.members, publishable, &unpacked)?;
    let dry = r8_dry_run(run, publishable)?;

    Ok(format!(
        "check-release passed{}: R1 {} packages built and verified in one cargo package call. R2 {listing}. R3/R4 \
         metadata, one version, exact pins, publish flags. R5 {docs}. R6 {sizes}. R7 {recipe}. R8 {dry}",
        if run.check_only {
            " (--check-only: no Slang, no adapter)"
        } else {
            ""
        },
        publishable.len()
    ))
}

/// R2, first half: `cargo package --list` of each package against the allow-list.
fn r2_lists(run: &Run, publishable: &[&Member]) -> Result<String, String> {
    let mut problems = Vec::new();
    let mut counts = Vec::new();
    for member in publishable {
        let output = run.step(
            "R2",
            &format!("cargo package --list -p {}", member.name),
            run.cargo(&[
                "package",
                "--list",
                "--allow-dirty",
                "--color",
                "never",
                "-p",
                &member.name,
            ]),
        )?;
        let files = package_files(&String::from_utf8_lossy(&output.stdout));
        problems.extend(package_problems(&member.spec(), &files, &[]));
        problems.extend(required_file_problems(&member.name, &files));
        counts.push(format!("{} {}", member.name, files.len()));
    }
    problems_to_error("R2 (cargo package --list)", &problems)?;
    Ok(format!(
        "the file lists are on the allow-list with both licences and a README ({} files)",
        counts.join(", ")
    ))
}

/// R1's judgement: cargo's stderr must show a `Verifying <name> v<version>` line for every package, or the package
/// was not built from its packed tree (`--no-verify`, or a cargo that skips it).
pub(crate) fn verify_problems(stderr: &str, members: &[&Member]) -> Vec<String> {
    members
        .iter()
        .filter(|member| {
            !stderr.contains(&format!("Verifying {} v{}", member.name, member.version))
        })
        .map(|member| {
            format!(
                "R1: cargo did not verify `{}` v{} (no `Verifying` line)",
                member.name, member.version
            )
        })
        .collect()
}

/// The target directory of every build from cargo's temporary registry (R1 and R8), inside the work directory. Cargo
/// fingerprints a registry crate by name and version, not by its source, so a build left by an earlier run in a
/// shared target directory would be reused although the packaged source changed.
fn registry_target(work: &Path) -> PathBuf {
    work.join("package-target")
}

/// The command of R1 and R8: `cargo <leading> --locked --allow-dirty --color never -p <each package>` in the
/// repository, building into [`registry_target`].
fn registry_command(run: &Run, leading: &[&str], publishable: &[&Member]) -> Command {
    let mut args = leading.to_vec();
    args.extend(["--locked", "--allow-dirty", "--color", "never"]);
    for member in publishable {
        args.extend(["-p", &member.name]);
    }
    run.cargo_in(run.root, &registry_target(run.work), &args)
}

/// The command that R1 runs.
fn r1_command(run: &Run, publishable: &[&Member]) -> Command {
    registry_command(run, &["package"], publishable)
}

/// The command that R8 runs.
fn r8_command(run: &Run, publishable: &[&Member]) -> Command {
    registry_command(run, &["publish", "--dry-run"], publishable)
}

/// R1: the one-call package with verification. Returns the directory with a copy of each `.crate`.
fn r1_package(run: &Run, publishable: &[&Member]) -> Result<PathBuf, String> {
    let output = run.step(
        "R1",
        "cargo package --locked (one call, with verification)",
        r1_command(run, publishable),
    )?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    problems_to_error("R1", &verify_problems(&stderr, publishable))?;
    let crates = run.work.join("crates");
    fs::create_dir_all(&crates).map_err(|error| format!("check-release R1: {error}"))?;
    for member in publishable {
        let file = format!("{}-{}.crate", member.name, member.version);
        let from = registry_target(run.work).join("package").join(&file);
        fs::copy(&from, crates.join(&file))
            .map_err(|error| format!("check-release R1: {}: {error}", from.display()))?;
    }
    Ok(crates)
}

/// R2, second half: unpack each `.crate` and judge its tree again. Returns the directory with the trees.
fn r2_unpacked(run: &Run, publishable: &[&Member]) -> Result<PathBuf, String> {
    let unpacked = run.work.join("unpacked");
    fs::create_dir_all(&unpacked).map_err(|error| format!("check-release R2: {error}"))?;
    let mut problems = Vec::new();
    for member in publishable {
        let file = format!("{}-{}.crate", member.name, member.version);
        // `tar` runs in the unpacked directory and reads the archive by a relative path: a drive letter in an
        // argument would be read as a host name by some tars.
        let mut tar = Command::new("tar");
        tar.current_dir(&unpacked)
            .args(["-xzf", &format!("../crates/{file}")])
            .stdin(Stdio::null());
        let output = tar
            .output()
            .map_err(|error| format!("check-release R2: could not run tar: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "check-release R2: tar -xzf {file} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let tree = unpacked.join(format!("{}-{}", member.name, member.version));
        let files: Vec<String> = snapshot(&tree)
            .map_err(|error| format!("check-release R2: {}: {error}", tree.display()))?
            .into_keys()
            .collect();
        problems.extend(package_problems(&member.spec(), &files, &[]));
        problems.extend(required_file_problems(&member.name, &files));
    }
    problems_to_error("R2 (unpacked packages)", &problems)?;
    Ok(unpacked)
}

/// R6: the size of each `.crate`.
fn r6_sizes(crates: &Path, publishable: &[&Member]) -> Result<String, String> {
    let mut sizes = Vec::new();
    for member in publishable {
        let file = format!("{}-{}.crate", member.name, member.version);
        let bytes = fs::metadata(crates.join(&file))
            .map_err(|error| format!("check-release R6: {file}: {error}"))?
            .len();
        sizes.push((file, bytes));
    }
    problems_to_error("R6", &size_problems(&sizes))?;
    let listed: Vec<String> = sizes
        .iter()
        .map(|(file, bytes)| {
            format!(
                "{} {:.0} KiB",
                file.trim_end_matches(".crate"),
                *bytes as f64 / 1024.0
            )
        })
        .collect();
    Ok(format!(
        "each .crate is under 2 MiB ({})",
        listed.join(", ")
    ))
}

/// One `cargo doc` run of R5. Returns its diagnostics.
fn doc_run(
    run: &Run,
    label: &str,
    nightly: bool,
    flags: &str,
    libraries: &[&Member],
    features: &[String],
) -> Result<Vec<DocMessage>, String> {
    let mut args: Vec<String> = Vec::new();
    if nightly {
        args.push("+nightly".to_owned());
    }
    args.extend(
        [
            "doc",
            "--no-deps",
            "--message-format",
            "json",
            "--color",
            "never",
        ]
        .map(str::to_owned),
    );
    for member in libraries {
        args.extend(["-p".to_owned(), member.name.clone()]);
    }
    if !features.is_empty() {
        args.extend(["--features".to_owned(), features.join(",")]);
    }
    let program = if nightly {
        OsString::from("cargo")
    } else {
        run.cargo.clone()
    };
    let mut command = run.command(&program, run.root);
    command
        .args(&args)
        .env("RUSTDOCFLAGS", flags)
        .env_remove("CARGO_ENCODED_RUSTDOCFLAGS");
    let output = run.step("R5", &format!("{label}: cargo {}", args.join(" ")), command)?;
    Ok(doc_messages(&String::from_utf8_lossy(&output.stdout)))
}

/// R5: rustdoc.
fn r5_docs(run: &Run, publishable: &[&Member]) -> Result<String, String> {
    let libraries: Vec<&Member> = publishable
        .iter()
        .copied()
        .filter(|member| member.has_library)
        .collect();
    if libraries.is_empty() {
        return Err(
            "check-release R5: no publishable package has a library to document".to_owned(),
        );
    }
    let mut problems = Vec::new();
    let stable = doc_run(run, "stable", false, "", &libraries, &[])?;
    problems.extend(doc_problems("stable", &stable, false, &KNOWN_DOC_DEBT));
    let known = stable
        .iter()
        .filter(|message| message.level == "warning")
        .count();

    let listing = Command::new("rustup")
        .args(["toolchain", "list"])
        .stdin(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    let nightly = if has_nightly(&listing) {
        let (features, flags) = docsrs_request(&libraries);
        let messages = doc_run(run, "nightly docsrs", true, &flags, &libraries, &features)?;
        problems.extend(doc_problems(
            "nightly docsrs",
            &messages,
            true,
            &KNOWN_DOC_DEBT,
        ));
        "nightly with the docs.rs features and --cfg docsrs"
    } else {
        println!(
            "check-release R5: no nightly toolchain is installed; the docs.rs-shaped run is SKIPPED"
        );
        "nightly docs.rs run SKIPPED (no nightly toolchain installed)"
    };
    problems_to_error("R5", &problems)?;
    Ok(format!(
        "cargo doc --no-deps of {} librar{} on stable ({} warning(s), all admitted as known debt); {nightly}",
        libraries.len(),
        if libraries.len() == 1 { "y" } else { "ies" },
        known
    ))
}

/// R7: the README recipe.
fn r7_recipe(
    run: &Run,
    members: &[Member],
    publishable: &[&Member],
    unpacked: &Path,
) -> Result<String, String> {
    let read = |path: PathBuf| {
        fs::read_to_string(&path)
            .map_err(|error| format!("check-release R7: {}: {error}", path.display()))
    };
    let fixture_dir = run.root.join(HELLO_DIR);
    let fixture_manifest = read(fixture_dir.join("Cargo.toml"))?;
    let fixture_source = read(fixture_dir.join("src").join("main.rs"))?;
    let mut problems = readme_problems(
        &read(run.root.join("README.md"))?,
        &fixture_manifest,
        &fixture_source,
    );
    problems.extend(fixture_dependency_problems(&fixture_manifest, members));
    problems_to_error("R7 (the recipe text)", &problems)?;
    let package = section_value(&fixture_manifest, "package", "name")
        .ok_or("check-release R7: the fixture Cargo.toml has no package name")?;

    let trees: Vec<(String, String)> = publishable
        .iter()
        .map(|member| {
            let tree = unpacked.join(format!("{}-{}", member.name, member.version));
            (member.name.clone(), slashes(&tree.to_string_lossy()))
        })
        .collect();
    let libraries: Vec<&str> = publishable
        .iter()
        .filter(|member| member.has_library)
        .map(|member| member.name.as_str())
        .collect();
    let tool = publishable
        .iter()
        .find(|member| member.has_binary && !member.has_library)
        .ok_or("check-release R7: no publishable package is a binary-only tool to install")?;
    let tool_name = tool
        .binary
        .as_deref()
        .ok_or("check-release R7: the tool package has no bin target")?;

    // The consumer: the README's two files, the patch table, and the lock of the precompiled fixture so that wgpu and
    // the rest resolve to the versions that are tested.
    let consumer = run.work.join("hello");
    fs::create_dir_all(consumer.join("src"))
        .map_err(|error| format!("check-release R7: {error}"))?;
    let write = |path: PathBuf, text: &str| {
        fs::write(&path, text)
            .map_err(|error| format!("check-release R7: {}: {error}", path.display()))
    };
    write(
        consumer.join("Cargo.toml"),
        &patched_manifest(&fixture_manifest, &trees),
    )?;
    write(
        consumer.join("src").join("main.rs"),
        &fixture_source.replace("\r\n", "\n"),
    )?;
    let seed = run
        .root
        .join(crate::precompiled::PRECOMPILED_DIR)
        .join("Cargo.lock");
    match fs::read_to_string(&seed) {
        Ok(lock) => write(consumer.join("Cargo.lock"), &lock)?,
        Err(error) => println!(
            "check-release R7: skipping the lock seed: {} is not readable ({error}); cargo resolves wgpu and the \
             rest afresh, so the run may differ from the tested versions",
            seed.display()
        ),
    }
    let consumer_target = run.work.join("hello-target");

    // The dev build with the README's user feature (compile only here).
    let output = run.step(
        "R7",
        &format!("cargo build --features gust-dev ({package})"),
        run.cargo_in(
            &consumer,
            &consumer_target,
            &["build", "--features", "gust-dev", "--color", "never"],
        ),
    )?;
    let mut problems = unused_library_patches(&String::from_utf8_lossy(&output.stderr), &libraries)
        .into_iter()
        .map(|line| format!("R7: cargo reported an unused patch: {line}"))
        .collect::<Vec<_>>();
    problems.extend(consumer_lock_problems(
        &read(consumer.join("Cargo.lock"))?,
        &libraries,
    ));
    problems_to_error("R7 (patches and lock)", &problems)?;

    // The tool, installed from its unpacked package with the same patches; no variable is set.
    let install_root = run.work.join("install");
    let install_target = run.work.join("install-target");
    let tool_tree = unpacked.join(format!("{}-{}", tool.name, tool.version));
    let patches = patch_config_arguments(&trees, &tool.name);
    let mut install_args: Vec<&str> = vec![
        "install",
        "--path",
        path_arg(&tool_tree)?,
        "--root",
        path_arg(&install_root)?,
    ];
    install_args.extend(["--color", "never"]);
    install_args.extend(patches.iter().map(String::as_str));
    run.step(
        "R7",
        &format!(
            "cargo install --path {} (patched, no GUST_PROFILE)",
            tool.name
        ),
        run.cargo_in(run.work, &install_target, &install_args),
    )?;
    let installed = install_root
        .join("bin")
        .join(format!("{tool_name}{EXE_SUFFIX}"));
    if !installed.is_file() {
        return Err(format!(
            "check-release R7: cargo install left no {}",
            installed.display()
        ));
    }

    if run.check_only {
        return Ok(format!(
            "(build only: --check-only skips the runs, `cargo gust build` and the release build) the README \
             blocks equal {HELLO_DIR}; {package} with the user feature gust-dev built against the unpacked \
             packages (every patch used, no `source` in the lock for the libraries); {tool_name} installed from its \
             unpacked package with no GUST_PROFILE"
        ));
    }

    // The full run: Slang and the GPU (the environment was checked before the first step).
    let dev_binary = consumer_target
        .join("debug")
        .join(format!("{package}{EXE_SUFFIX}"));
    run_recipe(run, "dev build", &dev_binary, &consumer)?;
    run.step(
        "R7",
        &format!("{tool_name} gust build -p {package} (needs Slang)"),
        {
            let mut command = run.bridge(run.command(&installed, &consumer));
            command
                .args(["gust", "build", "-p", &package])
                .env("CARGO_TARGET_DIR", &consumer_target);
            command
        },
    )?;
    if !consumer
        .join("gust-artifacts")
        .join("manifest.json")
        .is_file()
    {
        return Err(
            "check-release R7: `gust build` wrote no gust-artifacts/manifest.json".to_owned(),
        );
    }
    run.step(
        "R7",
        &format!("cargo build --release ({package}, the strict native profile)"),
        run.cargo_in(
            &consumer,
            &consumer_target,
            &["build", "--release", "--color", "never"],
        ),
    )?;
    let release_binary = consumer_target
        .join("release")
        .join(format!("{package}{EXE_SUFFIX}"));
    run_recipe(
        run,
        "release build (native, precompiled)",
        &release_binary,
        &consumer,
    )?;
    Ok(format!(
        "the README blocks equal {HELLO_DIR}; {package} with gust-dev ran on the GPU through the run-time bridge and \
         printed `{HELLO_OUTPUT}`; {tool_name} installed from its unpacked package with no GUST_PROFILE wrote \
         gust-artifacts/ for it; the release build (inferred native, no feature) ran and printed the same line"
    ))
}

/// Runs the recipe's binary and requires its result line.
fn run_recipe(run: &Run, what: &str, binary: &Path, directory: &Path) -> Result<(), String> {
    let output = run.step(
        "R7",
        &format!("run the {what}"),
        run.bridge(run.command(binary, directory)),
    )?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if recipe_ran(&stdout) {
        return Ok(());
    }
    Err(format!(
        "check-release R7: the {what} exited 0 but did not print `{HELLO_OUTPUT}`:\n{}",
        tail(&stdout, 2000)
    ))
}

/// The variable that names the reflection helper (the run-time bridge, README).
const HELPER_VARIABLE: &str = "GUST_SLANG_REFLECT";

/// The helper that `cargo xtask verify` and `build-slang-reflect` build by default.
fn default_helper(root: &Path) -> PathBuf {
    root.join("target")
        .join("slang-reflect")
        .join(format!("gust-slang-reflect{EXE_SUFFIX}"))
}

/// Which reflection helper the children of the full run use. `variable` is `GUST_SLANG_REFLECT` of the caller and
/// `default` the file that `verify` builds. A variable that is set must name a file (the children inherit it:
/// `Ok(None)`); when it is unset the default is handed to the children if it exists, else the run fails naming both.
fn helper_choice(variable: Option<OsString>, default: &Path) -> Result<Option<PathBuf>, String> {
    if let Some(value) = variable {
        let path = PathBuf::from(&value);
        if path.is_file() {
            return Ok(None);
        }
        return Err(format!(
            "check-release R7: {HELPER_VARIABLE} is set to {}, which is not a file; name the reflection helper \
             (`cargo gust slang build-helper` prints it), unset the variable to use {}, or use --check-only",
            path.display(),
            default.display()
        ));
    }
    if default.is_file() {
        return Ok(Some(default.to_path_buf()));
    }
    Err(format!(
        "check-release R7: the full run needs the reflection helper: {HELPER_VARIABLE} is unset and the default {} \
         does not exist (`cargo xtask verify` or `cargo xtask build-slang-reflect` builds it); or use --check-only",
        default.display()
    ))
}

/// The full run needs `slangc` and the reflection helper (README, run-time bridge). Returns the helper to hand to the
/// children, if the caller's environment names none.
fn slang_environment(root: &Path) -> Result<Option<PathBuf>, String> {
    let slangc = Command::new("slangc")
        .arg("-version")
        .stdin(Stdio::null())
        .output();
    if !slangc.is_ok_and(|output| output.status.success()) {
        return Err("check-release R7: the full run needs `slangc` on PATH (the pinned Slang; `cargo gust slang \
                    install`), or use --check-only"
            .to_owned());
    }
    helper_choice(env::var_os(HELPER_VARIABLE), &default_helper(root))
}

/// R8: the publish dry run.
fn r8_dry_run(run: &Run, publishable: &[&Member]) -> Result<String, String> {
    let output = run.step(
        "R8",
        "cargo publish --dry-run --locked (one call)",
        r8_command(run, publishable),
    )?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    problems_to_error("R8", &dry_run_problems(&stderr, publishable.len()))?;
    let existing = existing_crate_lines(&stderr);
    let note = if existing.is_empty() {
        String::new()
    } else {
        format!(
            "; the registry already has {} of these names and versions (a published version cannot be published again)",
            existing.len()
        )
    };
    Ok(format!(
        "cargo publish --dry-run --locked of {} packages in one call ended each with `aborting upload due to dry \
         run`{note}",
        publishable.len()
    ))
}

#[cfg(test)]
#[path = "tests/release.rs"]
mod release_tests;
