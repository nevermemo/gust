use std::{
    env,
    ffi::OsString,
    path::Path,
    process::{Command, Stdio},
};

use crate::{
    json::{self, Json},
    process::{CheckRecord, report_check},
    workspace::{project_files, read_text},
};

/// One checked crate: its manifest, name, and the only dependencies it may declare (all
/// kinds: normal, dev, build, and target-specific). D02 keeps this set fixed. `gust` and
/// `gust-wgpu` list themselves: each has a dev-dependency on itself that turns its own
/// `slang-runtime` feature on for its tests (D44), the one way cargo lets a package's tests
/// differ from its default features. A package can depend on itself only that way, so the entry
/// adds no edge to the architecture.
struct CrateRule {
    manifest: &'static str,
    name: &'static str,
    allowed: &'static [&'static str],
}

const CRATE_RULES: &[CrateRule] = &[
    CrateRule {
        manifest: "crates/gust-macros/Cargo.toml",
        name: "gust-macros",
        allowed: &["proc-macro2", "quote", "syn"],
    },
    CrateRule {
        manifest: "crates/gust/Cargo.toml",
        name: "gust",
        allowed: &["gust-macros", "gust"],
    },
    CrateRule {
        manifest: "crates/gust-wgpu/Cargo.toml",
        name: "gust-wgpu",
        allowed: &["gust", "gust-wgpu", "pollster", "wgpu"],
    },
    CrateRule {
        manifest: "crates/cargo-gust/Cargo.toml",
        name: "cargo-gust",
        allowed: &["gust"],
    },
    CrateRule {
        manifest: "xtask/Cargo.toml",
        name: "xtask",
        allowed: &[],
    },
];

/// The crates that translate Rust to Slang; they must not depend on a SPIR-V library.
const COMPILER_CRATES: &[&str] = &["gust-macros", "gust"];

/// The default registry as `cargo metadata` names it: the git index form, which is what cargo
/// prints for crates.io today, and the sparse form, in case that spelling is ever printed.
const CRATES_IO_SOURCES: &[&str] = &[
    "registry+https://github.com/rust-lang/crates.io-index",
    "sparse+https://index.crates.io/",
];

const CHANGE_RULE: &str = "D02 keeps the architecture fixed; explain the change in DECISIONS and update xtask/src/arch.rs";

/// `cargo xtask check-arch`: dependency allow-list per crate, no IR modules outside `future/`,
/// no SPIR-V emitter dependency. Needs git and cargo; also part of `check-fast` and CI.
pub(crate) fn check_arch(root: &Path, checks: &mut Vec<CheckRecord>) -> Result<(), String> {
    report_check(checks, "check-arch", check_project(root))
}

/// Returns a summary on success, or one line per problem.
fn check_project(root: &Path) -> Result<String, Vec<String>> {
    let files = project_files(root).map_err(|error| vec![error])?;
    let mut problems = Vec::new();
    match read_metadata(root).and_then(|text| parse_metadata(&text)) {
        Ok(packages) => problems.extend(package_problems(&root.display().to_string(), &packages)),
        Err(error) => problems.push(error),
    }
    match read_text(&root.join("Cargo.lock")) {
        Ok(Some(text)) => match parse_lock(&text) {
            Ok(locked) => problems.extend(lock_problems(&locked)),
            Err(error) => problems.push(error),
        },
        Ok(None) => problems.push("Cargo.lock: is not UTF-8 text".to_owned()),
        Err(error) => problems.push(format!(
            "Cargo.lock: could not be read ({error}); the resolved sources of allow-listed dependencies cannot be checked without it"
        )),
    }
    problems.extend(unlisted_crate_problems(&files));
    let mut rust_files = 0usize;
    for relative in files.iter().filter(|path| path.starts_with("crates/")) {
        problems.extend(ir_path_problems(relative));
        if relative.ends_with(".rs") && !in_future(relative) {
            rust_files += 1;
            match read_text(&root.join(relative)) {
                Ok(Some(text)) => problems.extend(ir_module_problems(relative, &text)),
                Ok(None) => {}
                Err(error) => problems.push(format!("{relative}: could not be read: {error}")),
            }
        }
    }
    if !problems.is_empty() {
        return Err(problems);
    }
    Ok(format!(
        "Architecture check passed: {} crate manifests match the dependency allow-list, read from `cargo metadata` so that `workspace = true` entries, `package` renames and declared sources are the ones cargo resolved (example packages are not checked), and every allow-listed package in Cargo.lock resolves from crates.io or crates/ (so a root `[patch]` or `[replace]` redirect fails); no IR module or file under crates/ outside future/ ({rust_files} Rust files scanned); no SPIR-V library in {}.\n",
        CRATE_RULES.len(),
        COMPILER_CRATES.join(" or ")
    ))
}

/// One dependency of a workspace package, as `cargo metadata` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dependency {
    /// The real package name (after any `package = "real"` rename).
    pub(crate) name: String,
    /// The key the manifest declares it under, when it gives `package = "..."`.
    pub(crate) rename: Option<String>,
    /// `registry+...`, `sparse+...`, `git+...`; `None` for a path dependency.
    pub(crate) source: Option<String>,
    /// The alternative registry, if the manifest names one.
    pub(crate) registry: Option<String>,
    /// The directory of a path dependency, as cargo printed it (absolute).
    pub(crate) path: Option<String>,
}

impl Dependency {
    /// The name the manifest writes it under.
    fn key(&self) -> &str {
        self.rename.as_deref().unwrap_or(&self.name)
    }
}

/// One workspace package and its dependencies of every kind (normal, dev, build, target).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Package {
    pub(crate) name: String,
    pub(crate) manifest_path: String,
    pub(crate) dependencies: Vec<Dependency>,
}

/// `cargo metadata --format-version 1 --locked --no-deps` in the workspace root. Cargo itself
/// reads the manifests, so inheritance (`workspace = true`), renames, target tables and
/// `[dependencies.name]` forms need no reader of ours, and a manifest cargo rejects (such as an
/// inherited dependency without a root entry) is reported here with cargo's own message.
fn read_metadata(root: &Path) -> Result<String, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let output = Command::new(&cargo)
        .args(["metadata", "--format-version", "1", "--locked", "--no-deps"])
        .current_dir(root)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cargo metadata: could not launch cargo: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .replace('\n', " | ")
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("cargo metadata printed text that is not UTF-8: {error}"))
}

/// The packages and dependencies of `cargo metadata` output; nothing else is read.
pub(crate) fn parse_metadata(text: &str) -> Result<Vec<Package>, String> {
    let document = json::parse(text).map_err(|error| format!("cargo metadata output: {error}"))?;
    let packages = document
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata output has no `packages` array")?;
    packages.iter().map(read_package).collect()
}

fn read_package(package: &Json) -> Result<Package, String> {
    let text = |value: &Json, key: &str| -> Result<String, String> {
        value
            .get(key)
            .and_then(Json::string_or_null)
            .ok_or_else(|| format!("cargo metadata output: a package or dependency lacks `{key}`"))
    };
    let name = text(package, "name")?;
    let manifest_path = text(package, "manifest_path")?;
    let dependencies = package
        .get("dependencies")
        .and_then(Json::as_array)
        .ok_or_else(|| format!("cargo metadata output: package `{name}` has no `dependencies`"))?
        .iter()
        .map(|dependency| {
            let optional = |key: &str| dependency.get(key).and_then(Json::string_or_null);
            Ok(Dependency {
                name: text(dependency, "name")?,
                rename: optional("rename"),
                source: optional("source"),
                registry: optional("registry"),
                path: optional("path"),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Package {
        name,
        manifest_path,
        dependencies,
    })
}

/// `path` with `\` as `/`, a `\\?\` prefix and trailing `/` dropped.
fn slashed(path: &str) -> String {
    let path = path.strip_prefix(r"\\?\").unwrap_or(path);
    path.replace('\\', "/").trim_end_matches('/').to_owned()
}

/// `path` relative to `root` (`None` when it is not below it). Windows paths ignore case.
fn relative_to(root: &str, path: &str) -> Option<String> {
    let (root, path) = (slashed(root), slashed(path));
    let (head, tail) = (path.get(..root.len())?, path.get(root.len()..)?);
    let same = if cfg!(windows) {
        head.eq_ignore_ascii_case(&root)
    } else {
        head == root
    };
    (same && tail.starts_with('/')).then(|| tail.trim_start_matches('/').to_owned())
}

/// Allow-list and source problems of every rule crate; a rule crate that cargo does not list
/// as a workspace package, or lists at another path, is a problem too.
pub(crate) fn package_problems(root: &str, packages: &[Package]) -> Vec<String> {
    let mut problems = Vec::new();
    for rule in CRATE_RULES {
        match packages.iter().find(|package| package.name == rule.name) {
            None => problems.push(format!(
                "{}: cargo metadata lists no workspace package `{}`; its dependencies would go unchecked ({CHANGE_RULE})",
                rule.manifest, rule.name
            )),
            Some(package) => {
                let found = relative_to(root, &package.manifest_path)
                    .unwrap_or_else(|| slashed(&package.manifest_path));
                if found == rule.manifest {
                    problems.extend(dependency_problems(rule, package, root));
                } else {
                    problems.push(format!(
                        "{}: package `{}` is at `{found}`, not at its manifest ({CHANGE_RULE})",
                        rule.manifest, rule.name
                    ));
                }
            }
        }
    }
    problems
}

fn is_spirv_library(name: &str) -> bool {
    name == "spirv"
        || name.starts_with("rspirv")
        || name.starts_with("spirv-")
        || name.starts_with("spirv_")
}

/// True when some crate may depend on `name`.
fn is_allowed_somewhere(name: &str) -> bool {
    CRATE_RULES.iter().any(|rule| rule.allowed.contains(&name))
}

/// Dependencies outside the crate's allow-list, SPIR-V libraries in the compiler crates, and
/// allow-listed dependencies that are not the real thing: no `package` rename, no git or
/// alternative registry, a workspace crate from its own `crates/<name>` directory, everything
/// else from crates.io. Each dependency is judged by its real package name, not its key.
fn dependency_problems(rule_crate: &CrateRule, package: &Package, root: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for dependency in &package.dependencies {
        let name = dependency.name.as_str();
        if !rule_crate.allowed.contains(&name) {
            if COMPILER_CRATES.contains(&rule_crate.name) && is_spirv_library(name) {
                problems.push(format!(
                    "{}: `{}` depends on SPIR-V library `{name}`; no SPIR-V emitter in the compiler crates ({CHANGE_RULE})",
                    rule_crate.manifest, rule_crate.name
                ));
            } else {
                problems.push(format!(
                    "{}: `{}` has dependency `{name}`, which is not in its allow-list ({}); {CHANGE_RULE}",
                    rule_crate.manifest,
                    rule_crate.name,
                    if rule_crate.allowed.is_empty() {
                        "none".to_owned()
                    } else {
                        rule_crate.allowed.join(", ")
                    }
                ));
            }
        }
        if is_allowed_somewhere(name) || is_allowed_somewhere(dependency.key()) {
            problems.extend(source_problems(rule_crate, dependency, root));
        }
    }
    // A dependency listed under several kinds or targets is reported once.
    let mut seen: Vec<String> = Vec::new();
    problems.retain(|problem| {
        let new = !seen.contains(problem);
        if new {
            seen.push(problem.clone());
        }
        new
    });
    problems
}

/// What an allow-listed dependency must be, beyond its name (see `dependency_problems`).
fn source_problems(rule_crate: &CrateRule, dependency: &Dependency, root: &str) -> Vec<String> {
    let name = dependency.name.as_str();
    let at = format!(
        "{}: `{}` dependency `{}`",
        rule_crate.manifest,
        rule_crate.name,
        dependency.key()
    );
    let mut problems = Vec::new();
    if dependency.rename.is_some() {
        problems.push(format!(
            "{at} is renamed with package = \"{name}\"; an allow-listed dependency must keep its own name ({CHANGE_RULE})"
        ));
    }
    let source = dependency.source.as_deref().unwrap_or_default();
    if source.starts_with("git+") {
        problems.push(format!(
            "{at} has a git source; allow-listed dependencies come from crates.io or crates/ ({CHANGE_RULE})"
        ));
    }
    let foreign_registry = (source.starts_with("registry+") || source.starts_with("sparse+"))
        && !CRATES_IO_SOURCES.contains(&source);
    if dependency.registry.is_some() || foreign_registry {
        problems.push(format!(
            "{at} names a registry; allow-listed dependencies come from crates.io or crates/ ({CHANGE_RULE})"
        ));
    }
    let path = dependency
        .path
        .as_deref()
        .map(|path| relative_to(root, path).unwrap_or_else(|| slashed(path)));
    if CRATE_RULES.iter().any(|rule| rule.name == name) {
        let expected = format!("crates/{name}");
        if path.as_deref() != Some(expected.as_str()) {
            problems.push(format!(
                "{at} must be `path = \"{expected}\"`, found {} ({CHANGE_RULE})",
                path.map_or("no path".to_owned(), |path| format!("`{path}`"))
            ));
        }
    } else if let Some(path) = path {
        problems.push(format!(
            "{at} is a path dependency on `{path}`; it must come from crates.io ({CHANGE_RULE})"
        ));
    }
    problems
}

/// One `[[package]]` of `Cargo.lock`: what cargo resolved, after any root `[patch]` or
/// `[replace]` redirect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LockedPackage {
    pub(crate) name: String,
    pub(crate) version: String,
    /// `registry+...`, `sparse+...`, `git+...`; `None` for a path package (a workspace member
    /// or a path redirect).
    pub(crate) source: Option<String>,
}

/// The fields of the `[[package]]` table being read.
type PackageFields = (Option<String>, Option<String>, Option<String>);

/// The `[[package]]` tables of a `Cargo.lock`. Cargo writes the file in one fixed shape (one
/// `key = "value"` per line, no escapes in these three keys), so a line reader suffices; any
/// other table, such as `[metadata]` or `[[patch.unused]]`, ends the current package.
pub(crate) fn parse_lock(text: &str) -> Result<Vec<LockedPackage>, String> {
    fn finish(
        fields: Option<PackageFields>,
        packages: &mut Vec<LockedPackage>,
    ) -> Result<(), String> {
        if let Some((name, version, source)) = fields {
            let name = name.ok_or("Cargo.lock: a [[package]] has no `name`")?;
            let version =
                version.ok_or_else(|| format!("Cargo.lock: package `{name}` has no `version`"))?;
            packages.push(LockedPackage {
                name,
                version,
                source,
            });
        }
        Ok(())
    }
    let mut packages = Vec::new();
    let mut current: Option<PackageFields> = None;
    for line in text.lines() {
        if line.starts_with('[') {
            finish(current.take(), &mut packages)?;
            if line.trim_end() == "[[package]]" {
                current = Some((None, None, None));
            }
            continue;
        }
        let Some((name, version, source)) = current.as_mut() else {
            continue;
        };
        let Some((key, value)) = line.split_once(" = ") else {
            continue;
        };
        let slot = match key {
            "name" => name,
            "version" => version,
            "source" => source,
            _ => continue,
        };
        let value = value
            .trim()
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .filter(|text| !text.contains(['"', '\\']))
            .ok_or_else(|| format!("Cargo.lock: cannot read the value of `{key}`: {value}"))?;
        *slot = Some(value.to_owned());
    }
    finish(current, &mut packages)?;
    if packages.is_empty() {
        return Err("Cargo.lock lists no [[package]]; is it a lockfile?".to_owned());
    }
    Ok(packages)
}

/// Resolved-source problems: an allow-listed external package must resolve from crates.io, and
/// a workspace crate must be a path package. `cargo metadata --no-deps` shows only declared
/// sources, so a root `[patch.crates-io]` or `[replace]` entry that points `syn` at a git
/// repository or a local directory is invisible there but changes the lockfile's source.
/// Trusting the lockfile is safe enough because the suites' cargo calls pass `--locked`, which
/// fails when the lockfile no longer matches the manifests.
pub(crate) fn lock_problems(locked: &[LockedPackage]) -> Vec<String> {
    let mut problems = Vec::new();
    for package in locked {
        let name = package.name.as_str();
        let workspace_crate = CRATE_RULES.iter().any(|rule| rule.name == name);
        let external = !workspace_crate && is_allowed_somewhere(name);
        let source = package.source.as_deref();
        let at = format!("Cargo.lock: package `{name}` {}", package.version);
        if external && !source.is_some_and(|source| CRATES_IO_SOURCES.contains(&source)) {
            problems.push(format!(
                "{at} resolves from {}, not from crates.io; an allow-listed dependency must not be redirected, for example by a [patch] or [replace] section ({CHANGE_RULE})",
                source.map_or("a local path".to_owned(), |source| format!("`{source}`"))
            ));
        }
        if workspace_crate && let Some(source) = source {
            problems.push(format!(
                "{at} resolves from `{source}`, not from its crates/ directory; a workspace crate must be a path package ({CHANGE_RULE})"
            ));
        }
    }
    problems
}

/// A crate directory under `crates/` that has no rule: its dependencies would go unchecked.
pub(crate) fn unlisted_crate_problems(files: &[String]) -> Vec<String> {
    files
        .iter()
        .filter(|path| {
            path.strip_prefix("crates/")
                .and_then(|rest| rest.strip_suffix("/Cargo.toml"))
                .is_some_and(|name| !name.contains('/'))
        })
        .filter(|path| !CRATE_RULES.iter().any(|rule| rule.manifest == path.as_str()))
        .map(|path| {
            format!("{path}: crate has no dependency allow-list; add one to xtask/src/arch.rs ({CHANGE_RULE})")
        })
        .collect()
}

fn in_future(path: &str) -> bool {
    path.split('/').any(|part| part == "future")
}

/// True for `ir` and `<anything>_ir`, any case.
fn is_ir_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "ir" || name.ends_with("_ir")
}

/// A directory or file named `ir`, `*_ir`, or `ir/` under `crates/`, outside `future/`.
pub(crate) fn ir_path_problems(path: &str) -> Vec<String> {
    if !path.starts_with("crates/") || in_future(path) {
        return Vec::new();
    }
    let parts: Vec<&str> = path.split('/').collect();
    let mut problems = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let name = if index + 1 == parts.len() {
            part.split('.').next().unwrap_or(part)
        } else {
            part
        };
        if is_ir_name(name) {
            problems.push(format!(
                "{path}: `{part}` looks like an instruction IR; D02 allows no custom shader IR outside a future/ path ({CHANGE_RULE})"
            ));
            break;
        }
    }
    problems
}

/// A `mod ir;` or `mod x_ir { .. }` declaration in a Rust file outside `future/`.
pub(crate) fn ir_module_problems(path: &str, text: &str) -> Vec<String> {
    if in_future(path) {
        return Vec::new();
    }
    let mut problems = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line.trim_start();
        if rest.starts_with("//") {
            continue;
        }
        if let Some(after) = rest.strip_prefix("pub") {
            rest = after.trim_start();
            if rest.starts_with('(') {
                rest = rest
                    .split_once(')')
                    .map_or("", |(_, tail)| tail)
                    .trim_start();
            }
        }
        let Some(rest) = rest.strip_prefix("mod ") else {
            continue;
        };
        let name: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if is_ir_name(&name) {
            problems.push(format!(
                "{path}:{}: module `{name}` looks like an instruction IR; D02 allows no custom shader IR outside a future/ path ({CHANGE_RULE})",
                index + 1
            ));
        }
    }
    problems
}

#[cfg(test)]
#[path = "tests/arch.rs"]
mod arch_tests;
