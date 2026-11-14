use std::{
    env,
    ffi::OsString,
    fs,
    path::Path,
    process::{Command, Stdio},
};

use crate::{
    json::{self, Json},
    process::{CheckRecord, report_check},
};

/// Extensions of shell scripts that must not be tracked (D27: workflow commands live in
/// `cargo xtask`).
const SCRIPT_EXTENSIONS: &[&str] = &["ps1", "psm1", "psd1", "bat", "cmd", "sh"];

/// Tracked files that spell machine-specific paths on purpose, as fixtures for path
/// redaction and for this lint. The matcher still runs on them: a path is accepted only when
/// its user name is one of `PLACEHOLDER_USERS`.
const MACHINE_PATH_FIXTURES: &[&str] = &[
    "xtask/src/tests/lint.rs",
    "xtask/src/tests/validation_redaction.rs",
];

/// The only user names a fixture file may spell in a home path.
const PLACEHOLDER_USERS: &[&str] = &["someone", "bob.smith", "José", "a_b-c", "some-one"];

/// `cargo xtask lint`: tracked-file checks that need no external tools. Prints its findings
/// and records the run like any other check.
pub(crate) fn lint(root: &Path, checks: &mut Vec<CheckRecord>) -> Result<(), String> {
    report_check(checks, "lint", lint_tracked_files(root))
}

/// Returns a summary on success, or one line per problem.
fn lint_tracked_files(root: &Path) -> Result<String, Vec<String>> {
    let files = tracked_files(root).map_err(|error| vec![error])?;
    let mut problems = Vec::new();
    let mut scanned = 0usize;
    let mut text_files = 0usize;
    let mut skipped = 0usize;
    let mut manifests: Vec<(String, String)> = Vec::new();
    let mut tagged_documents = 0usize;
    let pin = match crate::verify::pin_rule(root) {
        Ok(rule) => Some(rule.pin),
        Err(error) => {
            problems.push(format!("compiler tags cannot be checked: {error}"));
            None
        }
    };
    for relative in &files {
        // A path in the index whose file is missing from the working tree (deleted, deletion
        // not yet staged) is skipped: there is nothing to read. A plain `git commit` would
        // still keep it, so stage the deletion before relying on this result.
        let path = root.join(relative);
        if !path.is_file() {
            continue;
        }
        scanned += 1;
        if is_forbidden_script(relative) {
            problems.push(format!(
                "{relative}: tracked shell script; workflow commands live in `cargo xtask` (D27)"
            ));
        }
        let Ok(bytes) = fs::read(&path) else {
            problems.push(format!("{relative}: could not be read"));
            continue;
        };
        // A file that carries a compiler tag is judged before the binary skip below: one that is not UTF-8 text is a
        // problem there, never a quiet skip.
        if let Some(pin) = &pin
            && let Some(check) = compiler_tag_check(relative, &bytes, pin)
        {
            tagged_documents += check.documents;
            problems.extend(check.problems);
        }
        // Binary and non-UTF-8 files are not scanned for paths; the summary counts them.
        let Some(text) = std::str::from_utf8(&bytes)
            .ok()
            .filter(|_| !bytes.contains(&0))
        else {
            skipped += 1;
            continue;
        };
        text_files += 1;
        problems.extend(machine_path_problems(relative, text));
        if relative.rsplit('/').next() == Some("Cargo.toml") {
            manifests.push((relative.clone(), text.to_owned()));
        }
    }
    problems.extend(path_dependency_problems(&manifests));
    let (publishable, release) = release_metadata_problems(root);
    problems.extend(release);
    if problems.is_empty() {
        Ok(format!(
            "Lint passed: {scanned} tracked files have no shell scripts; {text_files} text files have no \
             machine-specific paths ({skipped} binary or non-UTF-8 files were skipped, not scanned); {} manifests \
             name the crates/ packages they depend on by path; {publishable} publishable packages carry the \
             release metadata, a README and the licence copies; the compiler tags of {tagged_documents} reflection \
             documents and artifact manifests equal SLANG_PIN.\n",
            manifests.len()
        ))
    } else {
        Err(problems)
    }
}

/// Whether the tracked file `relative` (a `/` path) is a reflection document or an artifact manifest, the files whose
/// `compiler` and `slang` fields name the Slang build that made them: `*.reflection.json` (artifacts, examples
/// included), `*.wgsl.json` (the schema fixtures of `gust`) and `manifest.json`.
fn carries_compiler_tag(relative: &str) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    name == "manifest.json" || name.ends_with(".reflection.json") || name.ends_with(".wgsl.json")
}

/// Whether the tracked file `relative` (a `/` path) is a probe expectation of the reflection helper: an `expected.txt`
/// under a `testdata` folder, whose lines that start with `{` are reflection documents (one per line, with a
/// `compiler` tag) between header and error lines. Today that is `crates/cargo-gust/src/testdata/schema3/expected.txt`
/// alone, the helper's own output for the schema-3 probes (m9.x1 review S1): its WGSL lines are compared with the
/// checked `.wgsl.json` fixtures by `crates/gust/tests/reflection.rs`, but nothing compares its SPIR-V lines, so a pin
/// bump that regenerates the other artifacts could leave their tags stale and lint green. The other tracked
/// `expected.txt` files (`tests/ui`, `tests/dual`, `tests/cpu-target`, `tests/precompiled`) hold compiler or driver
/// diagnostics, never a reflection document. A future `testdata/**/expected.txt` with no `{` line is a lint
/// problem by design (an empty expectation checks nothing), and the match takes a top-level `testdata/` folder too.
fn carries_line_documents(relative: &str) -> bool {
    relative.rsplit('/').next() == Some("expected.txt")
        && (relative.starts_with("testdata/") || relative.contains("/testdata/"))
}

/// What the compiler-tag check found in one tracked file: the number of documents whose tags were read, and the
/// problems.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct TagCheck {
    pub(crate) documents: usize,
    pub(crate) problems: Vec<String>,
}

/// The compiler-tag check of one tracked file, from its bytes: `None` for a file that carries no tag. A file that does
/// carry one but is not UTF-8 text, or holds a NUL, is a problem (an editor that saves a manifest as UTF-16 must not
/// hide a stale tag; m9.x1 review N4), not a skip. A whole-file document counts as one; for the probe expectations
/// (`carries_line_documents`) every line that starts with `{` is a document, each judged by `compiler_tag_problems`
/// under the name `<file>:<line>`, and a file with no such line is a problem too (the check would pass on nothing).
pub(crate) fn compiler_tag_check(relative: &str, bytes: &[u8], pin: &str) -> Option<TagCheck> {
    let whole_file = carries_compiler_tag(relative);
    let by_line = carries_line_documents(relative);
    if !whole_file && !by_line {
        return None;
    }
    let Some(text) = std::str::from_utf8(bytes)
        .ok()
        .filter(|_| !bytes.contains(&0))
    else {
        return Some(TagCheck {
            documents: 0,
            problems: vec![format!(
                "{relative}: not UTF-8 text, or holds a NUL, so its compiler tag was not checked; save it as UTF-8"
            )],
        });
    };
    if whole_file {
        return Some(TagCheck {
            documents: 1,
            problems: compiler_tag_problems(relative, text, pin),
        });
    }
    let mut check = TagCheck::default();
    for (index, line) in text.lines().enumerate() {
        if line.starts_with('{') {
            check.documents += 1;
            check.problems.extend(compiler_tag_problems(
                &format!("{relative}:{}", index + 1),
                line,
                pin,
            ));
        }
    }
    if check.documents == 0 {
        check.problems.push(format!(
            "{relative}: no line starts with `{{`, so no reflection document had its compiler tag checked"
        ));
    }
    Some(check)
}

/// The problems of one tagged document: a top-level `compiler` or `slang` that is not `pin` (the `SLANG_PIN` of
/// `gust::toolchain`). `slang` may be `null` (a manifest with no kernels says so); a missing field is not checked, and a
/// file that is not JSON is a problem, so that a broken document cannot hide a stale tag. A stale committed tag means
/// the artifact was made by another compiler: regenerate it, never edit the tag by hand.
pub(crate) fn compiler_tag_problems(path: &str, text: &str, pin: &str) -> Vec<String> {
    let document = match json::parse(text) {
        Ok(document) => document,
        Err(error) => {
            return vec![format!(
                "{path}: not JSON, so its compiler tag was not checked: {error}"
            )];
        }
    };
    let mut problems = Vec::new();
    for key in ["compiler", "slang"] {
        match document.get(key) {
            None => {}
            Some(Json::String(tag)) if tag == pin => {}
            Some(Json::Null) if key == "slang" => {}
            Some(Json::String(tag)) => problems.push(format!(
                "{path}: `{key}` is `{tag}` but SLANG_PIN is `{pin}`; regenerate the artifact with the pinned compiler"
            )),
            Some(_) => problems.push(format!("{path}: `{key}` is not a string")),
        }
    }
    problems
}

fn tracked_files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not launch git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}

/// True for a repository-relative path whose file name ends in a shell-script extension.
pub(crate) fn is_forbidden_script(path: &str) -> bool {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    name.rsplit_once('.').is_some_and(|(_, extension)| {
        SCRIPT_EXTENSIONS
            .iter()
            .any(|script| extension.eq_ignore_ascii_case(script))
    })
}

/// One `file:line` message per machine-specific path in `text`. In a fixture file a path is
/// accepted when its user name is a placeholder; everywhere else every path is a leak.
pub(crate) fn machine_path_problems(path: &str, text: &str) -> Vec<String> {
    let fixture = MACHINE_PATH_FIXTURES.contains(&path);
    let mut problems = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut from = 0;
        while let Some((start, end)) = find_machine_span(line, from) {
            from = end;
            let found = &line[start..end];
            if fixture && is_placeholder_user(user_name(found)) {
                continue;
            }
            let hint = if fixture {
                format!(
                    "; fixtures may only use the user names {}",
                    PLACEHOLDER_USERS.join(", ")
                )
            } else {
                String::new()
            };
            problems.push(format!(
                "{path}:{}: machine-specific path `{found}`; use `<workspace>`, `~` or a relative path{hint}",
                index + 1
            ));
        }
    }
    problems
}

/// Placeholder names are matched ignoring case, because the redaction tests also check
/// case-insensitive spellings such as `SOMEONE`.
fn is_placeholder_user(name: &str) -> bool {
    let name = name.to_lowercase();
    PLACEHOLDER_USERS
        .iter()
        .any(|placeholder| placeholder.to_lowercase() == name)
}

/// The user name of a matched path: its last non-empty component.
fn user_name(found: &str) -> &str {
    found
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or(found)
}

/// The first absolute user-home path in `line`: `X:\Users\<name>` or `X:/Users/<name>` in any
/// slash style (doubled backslashes included), `/home/<name>` and `/Users/<name>` (followed
/// by `/`, a quote or the end of the line), the WSL spelling `/mnt/<letter>/Users/<name>`,
/// and a WSL network path such as
/// `\\wsl.localhost\<distro>\home\<name>` (also `\\wsl$\...`).
/// Placeholders such as `<name>`, `%USERNAME%`, `~` and `<workspace>` do not match.
#[cfg(test)]
pub(crate) fn find_machine_path(line: &str) -> Option<&str> {
    find_machine_span(line, 0).map(|(start, end)| &line[start..end])
}

/// Byte range of the first home path that starts at or after `from`.
fn find_machine_span(line: &str, from: usize) -> Option<(usize, usize)> {
    (from..line.len()).find_map(|start| home_path_end(line, start).map(|end| (start, end)))
}

fn home_path_end(line: &str, start: usize) -> Option<usize> {
    if !line.is_char_boundary(start) {
        return None;
    }
    let bytes = line.as_bytes();
    let before = line[..start].chars().next_back();
    if bytes[start].is_ascii_alphabetic() && bytes.get(start + 1) == Some(&b':') {
        // A drive letter, not the tail of a word such as `PATH:`.
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }
        let users = skip_separators(bytes, start + 2)?;
        if !starts_with_ignore_case(bytes, users, b"users") {
            return None;
        }
        let name = skip_separators(bytes, users + 5)?;
        return name_end(line, name);
    }
    if matches!(bytes[start], b'/' | b'\\') {
        // The root of an absolute path, not a segment of a relative one (`docs/home/x/`).
        if before.is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '~')) {
            return None;
        }
        if let Some(end) = wsl_network_home_end(line, start) {
            return Some(end);
        }
    }
    if bytes[start] == b'/' {
        if let Some(end) = wsl_mount_home_end(line, start) {
            return Some(end);
        }
        let name = ["/home/", "/Users/"]
            .iter()
            .find(|prefix| line[start..].starts_with(*prefix))
            .map(|prefix| start + prefix.len())?;
        let end = name_end(line, name)?;
        // The name must end the path: a `/`, the end of the line, or a closing quote.
        let ends_name = match bytes.get(end) {
            None => true,
            Some(byte) => matches!(byte, b'/' | b'"' | b'\'' | b'`'),
        };
        return ends_name.then_some(end);
    }
    None
}

/// `/mnt/<letter>/Users/<name>`: a Windows home seen from WSL.
fn wsl_mount_home_end(line: &str, start: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    if !line[start..].starts_with("/mnt/") {
        return None;
    }
    let drive = start + 5;
    if !bytes.get(drive)?.is_ascii_alphabetic() || bytes.get(drive + 1) != Some(&b'/') {
        return None;
    }
    let users = skip_separators(bytes, drive + 1)?;
    if !starts_with_ignore_case(bytes, users, b"users") {
        return None;
    }
    let name = skip_separators(bytes, users + 5)?;
    name_end(line, name)
}

/// `\\wsl.localhost\<distro>\home\<name>` or `\\wsl$\<distro>\home\<name>`, in any slash style.
fn wsl_network_home_end(line: &str, start: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    let host = skip_separators(bytes, start)?;
    let host_end = [&b"wsl.localhost"[..], b"wsl$"]
        .iter()
        .find(|host_name| starts_with_ignore_case(bytes, host, host_name))
        .map(|host_name| host + host_name.len())?;
    let distro = skip_separators(bytes, host_end)?;
    let distro_end = distro
        + line[distro..]
            .find(|c: char| matches!(c, '/' | '\\') || c.is_whitespace() || "\"'`)".contains(c))
            .unwrap_or(line.len() - distro);
    if distro_end == distro {
        return None;
    }
    let home = skip_separators(bytes, distro_end)?;
    if !starts_with_ignore_case(bytes, home, b"home") {
        return None;
    }
    let name = skip_separators(bytes, home + 4)?;
    name_end(line, name)
}

fn starts_with_ignore_case(bytes: &[u8], at: usize, prefix: &[u8]) -> bool {
    bytes
        .get(at..at + prefix.len())
        .is_some_and(|slice| slice.eq_ignore_ascii_case(prefix))
}

/// Index after one or more path separators starting at `from`, or `None` if there is none.
fn skip_separators(bytes: &[u8], from: usize) -> Option<usize> {
    let count = bytes
        .get(from..)?
        .iter()
        .take_while(|byte| matches!(byte, b'\\' | b'/'))
        .count();
    (count > 0).then_some(from + count)
}

/// End of the user name at `from`; it must start with a letter or digit so that
/// placeholders such as `<name>` and `%USERNAME%` are not user names.
fn name_end(line: &str, from: usize) -> Option<usize> {
    let rest = line.get(from..)?;
    if !rest.chars().next()?.is_alphanumeric() {
        return None;
    }
    let length = rest
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || matches!(c, '_' | '-' | '.')))
        .map_or(rest.len(), |(index, _)| index);
    Some(from + length)
}

// ---- Release rules (mr.a): what a crate needs before `cargo publish` ---------------------------------------------
//
// Names come from `cargo metadata`, never from a list in this file, so the rules hold unchanged when a package is
// renamed.

/// The two licence texts every publishable package carries a byte-equal copy of.
const LICENCE_FILES: &[&str] = &["LICENSE-MIT", "LICENSE-APACHE"];

/// What the release rules read of one workspace package (`cargo metadata --no-deps`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Member {
    pub(crate) name: String,
    /// The directory of the package's manifest, as cargo printed it (never shown in a message).
    pub(crate) directory: String,
    /// `publish` is not `false` (or `[]`): `cargo publish` would accept the package.
    pub(crate) publishable: bool,
    pub(crate) description: Option<String>,
    pub(crate) license: Option<String>,
    pub(crate) repository: Option<String>,
    pub(crate) documentation: Option<String>,
    /// The file cargo found or was told for the README, relative to the package directory.
    pub(crate) readme: Option<String>,
    pub(crate) keywords: Vec<String>,
    pub(crate) categories: Vec<String>,
}

/// The workspace packages of `cargo metadata --format-version 1 --no-deps` output.
pub(crate) fn parse_members(text: &str) -> Result<Vec<Member>, String> {
    let document = json::parse(text).map_err(|error| format!("cargo metadata output: {error}"))?;
    let packages = document
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata output has no `packages` array")?;
    packages.iter().map(read_member).collect()
}

fn read_member(package: &Json) -> Result<Member, String> {
    let text = |key: &str| {
        package
            .get(key)
            .and_then(Json::string_or_null)
            .filter(|value| !value.trim().is_empty())
    };
    let list = |key: &str| -> Vec<String> {
        package
            .get(key)
            .and_then(Json::as_array)
            .map(|items| items.iter().filter_map(Json::string_or_null).collect())
            .unwrap_or_default()
    };
    let name = text("name").ok_or("cargo metadata output: a package has no `name`")?;
    let manifest = text("manifest_path")
        .ok_or_else(|| format!("cargo metadata output: package `{name}` has no `manifest_path`"))?;
    let directory = match manifest.rsplit_once(['/', '\\']) {
        Some((directory, _)) => directory.to_owned(),
        None => String::new(),
    };
    // `publish = false` is `[]`; no key, `null` or a non-empty list of registries allows publishing.
    let publishable = match package.get("publish") {
        Some(Json::Array(registries)) => !registries.is_empty(),
        _ => true,
    };
    Ok(Member {
        description: text("description"),
        license: text("license"),
        repository: text("repository"),
        documentation: text("documentation"),
        readme: text("readme"),
        keywords: list("keywords"),
        categories: list("categories"),
        name,
        directory,
        publishable,
    })
}

/// Missing metadata of a publishable package; one message per field. A package that is not meant for crates.io says
/// `publish = false`, which is why every message names it.
pub(crate) fn member_field_problems(member: &Member) -> Vec<String> {
    if !member.publishable {
        return Vec::new();
    }
    let mut missing: Vec<&str> = Vec::new();
    for (field, value) in [
        ("description", &member.description),
        ("license", &member.license),
        ("repository", &member.repository),
        ("documentation", &member.documentation),
        ("readme", &member.readme),
    ] {
        if value.is_none() {
            missing.push(field);
        }
    }
    if member.keywords.is_empty() {
        missing.push("keywords");
    }
    if member.categories.is_empty() {
        missing.push("categories");
    }
    missing
        .into_iter()
        .map(|field| {
            format!(
                "package `{}` can be published but has no `{field}`; add it (`repository`, `keywords` and `categories` \
                 inherit from [workspace.package]), or set `publish = false` if it is not meant for crates.io",
                member.name
            )
        })
        .collect()
}

/// `None` when `actual` equals `expected` byte for byte; otherwise the problem, naming the first differing byte.
pub(crate) fn licence_problem(
    package: &str,
    file: &str,
    expected: &[u8],
    actual: Option<&[u8]>,
) -> Option<String> {
    let Some(actual) = actual else {
        return Some(format!(
            "package `{package}`: {file} is missing; copy the root {file} into the package directory \
             (cp {file} <crate dir>/)"
        ));
    };
    if actual == expected {
        return None;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    Some(format!(
        "package `{package}`: {file} differs from the root {file} (first difference at byte {first}; \
         {} bytes against {}); copy the root file again",
        actual.len(),
        expected.len()
    ))
}

/// All release problems of the workspace and the number of publishable packages, or the problem that stopped the
/// check. A failed `cargo metadata` is a problem, never a pass.
fn release_metadata_problems(root: &Path) -> (usize, Vec<String>) {
    let members = match read_metadata(root).and_then(|text| parse_members(&text)) {
        Ok(members) => members,
        Err(error) => return (0, vec![error]),
    };
    let mut problems = Vec::new();
    let mut publishable = 0usize;
    let licences: Vec<(&str, Option<Vec<u8>>)> = LICENCE_FILES
        .iter()
        .map(|file| (*file, fs::read(root.join(file)).ok()))
        .collect();
    for member in members.iter().filter(|member| member.publishable) {
        publishable += 1;
        problems.extend(member_field_problems(member));
        let directory = Path::new(&member.directory);
        if let Some(readme) = &member.readme
            && !directory.join(readme).is_file()
        {
            problems.push(format!(
                "package `{}`: its readme `{readme}` is not a file in the package directory",
                member.name
            ));
        }
        for (file, expected) in &licences {
            let Some(expected) = expected else {
                problems.push(format!("the root {file} could not be read"));
                continue;
            };
            let actual = fs::read(directory.join(file)).ok();
            problems.extend(licence_problem(
                &member.name,
                file,
                expected,
                actual.as_deref(),
            ));
        }
    }
    (publishable, problems)
}

/// `cargo metadata --format-version 1 --locked --no-deps` in `root`.
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

// ---- Path dependencies on the packages under crates/ -------------------------------------------------------------
//
// A `path =` dependency on a package under `crates/` must call that package by its own name: the key is the package's
// `name`, or `package = "<name>"` says so. Cargo rejects a wrong key in the workspace itself, but not in the fixture
// workspaces under `tests/`, which a lane branched before a rename would still write in the old form. The manifests are
// read as text (the fixtures are not part of the workspace, so `cargo metadata` does not list them).

/// One dependency declaration that has a `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PathDependency {
    key: String,
    path: String,
    package: Option<String>,
    line: usize,
}

/// The problems of every `path =` dependency on a `crates/<dir>` package in `manifests` (repository-relative path and
/// text of each tracked `Cargo.toml`).
pub(crate) fn path_dependency_problems(manifests: &[(String, String)]) -> Vec<String> {
    // The package name of each manifest under crates/, by the directory it is in.
    let mut packages: Vec<(String, String)> = Vec::new();
    for (file, text) in manifests {
        if file.starts_with("crates/")
            && let Some(directory) = file.strip_suffix("/Cargo.toml")
            && let Some(name) = package_name(text)
        {
            packages.push((directory.to_owned(), name));
        }
    }
    let mut problems = Vec::new();
    for (file, text) in manifests {
        let directory = file
            .strip_suffix("Cargo.toml")
            .unwrap_or("")
            .trim_end_matches('/');
        let (dependencies, unsupported) = path_dependencies(text);
        for line in unsupported {
            problems.push(format!(
                "{file}:{line}: a dependency written as an inline table over several lines; the path-dependency rule \
                 reads one line per dependency, so write it on one line"
            ));
        }
        for dependency in dependencies {
            let Some(target) = resolve(directory, &dependency.path) else {
                continue;
            };
            let Some((_, name)) = packages.iter().find(|(package, _)| *package == target) else {
                continue;
            };
            let called = dependency.package.as_deref().unwrap_or(&dependency.key);
            if called != name {
                problems.push(format!(
                    "{file}:{}: dependency `{}` has path `{}`, the package `{name}` ({target}), but calls it \
                     `{called}`; \
                     name the dependency `{name}` or add package = \"{name}\"",
                    dependency.line, dependency.key, dependency.path
                ));
            }
        }
    }
    problems
}

/// The repository-relative directory that `path` names from `directory` (both with `/`), or `None` for an absolute
/// path or one that leaves the repository.
fn resolve(directory: &str, path: &str) -> Option<String> {
    let path = path.replace('\\', "/");
    if path.starts_with('/') || path.as_bytes().get(1) == Some(&b':') {
        return None;
    }
    let mut parts: Vec<&str> = directory
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    Some(parts.join("/"))
}

/// The `name` of the `[package]` table.
fn package_name(text: &str) -> Option<String> {
    let mut in_package = false;
    for line in text.lines() {
        let line = strip_comment(line).trim();
        if let Some(header) = table_header(line) {
            in_package = header == "package";
        } else if in_package
            && let Some((key, value)) = split_key_value(line)
            && key == "name"
        {
            return string_value(value);
        }
    }
    None
}

/// The `[...]` header of a line (not `[[...]]`), without the brackets.
fn table_header(line: &str) -> Option<&str> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    (!inner.starts_with('[')).then(|| inner.trim())
}

/// What a table header introduces, as far as dependencies go.
enum Section {
    Other,
    /// `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, `[workspace.dependencies]` and the `target`
    /// forms.
    Dependencies,
    /// `[dependencies.<key>]` and its kin: one dependency, one key per line.
    Dependency(String),
}

fn section_of(header: &str) -> Section {
    let parts: Vec<String> = split_top(header, '.')
        .into_iter()
        .map(|part| unquote(part.trim()))
        .collect();
    let is_table = |part: &String| {
        matches!(
            part.as_str(),
            "dependencies" | "dev-dependencies" | "build-dependencies"
        )
    };
    match parts.iter().rposition(is_table) {
        Some(index) if index + 1 == parts.len() => Section::Dependencies,
        Some(index) if index + 2 == parts.len() => Section::Dependency(parts[index + 1].clone()),
        _ => Section::Other,
    }
}

/// The dependencies with a `path`, and the lines of dependencies this reader cannot see through (an inline table that
/// does not end on its line).
fn path_dependencies(text: &str) -> (Vec<PathDependency>, Vec<usize>) {
    let mut found: Vec<PathDependency> = Vec::new();
    let mut unsupported = Vec::new();
    let mut section = Section::Other;
    // Dotted and table forms collect one field per line: (key, line of the key, path, package).
    let mut partial: Vec<(String, usize, Option<String>, Option<String>)> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        let number = index + 1;
        if line.is_empty() {
            continue;
        }
        if let Some(header) = table_header(line) {
            section = section_of(header);
            if let Section::Dependency(key) = &section {
                partial.push((key.clone(), number, None, None));
            }
            continue;
        }
        if line.starts_with("[[") {
            section = Section::Other;
            continue;
        }
        let Some((key, value)) = split_key_value(line) else {
            continue;
        };
        match &section {
            Section::Other => {}
            Section::Dependency(_) => {
                if let Some(entry) = partial.last_mut() {
                    match key.as_str() {
                        "path" => entry.2 = string_value(value),
                        "package" => entry.3 = string_value(value),
                        _ => {}
                    }
                }
            }
            Section::Dependencies => {
                let parts: Vec<String> = split_top(&key, '.')
                    .into_iter()
                    .map(|part| unquote(part.trim()))
                    .collect();
                match parts.as_slice() {
                    [name] if value.starts_with('{') => match inline_body(value) {
                        // Refused only when the first line already names a field the rule reads: a table that
                        // names neither is a dependency the rule does not concern as far as can be seen.
                        None => {
                            if split_top(&value[1..], ',').into_iter().any(|pair| {
                                split_key_value(pair.trim())
                                    .is_some_and(|(field, _)| field == "path" || field == "package")
                            }) {
                                unsupported.push(number);
                            }
                        }
                        Some(body) => {
                            let (mut path, mut package) = (None, None);
                            for pair in split_top(body, ',') {
                                if let Some((field, field_value)) = split_key_value(pair.trim()) {
                                    match field.as_str() {
                                        "path" => path = string_value(field_value),
                                        "package" => package = string_value(field_value),
                                        _ => {}
                                    }
                                }
                            }
                            partial.push((name.clone(), number, path, package));
                        }
                    },
                    [name, field] if field == "path" || field == "package" => {
                        let at = match partial.iter().position(|entry| entry.0 == *name) {
                            Some(at) => at,
                            None => {
                                partial.push((name.clone(), number, None, None));
                                partial.len() - 1
                            }
                        };
                        if field == "path" {
                            partial[at].2 = string_value(value);
                        } else {
                            partial[at].3 = string_value(value);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for (key, line, path, package) in partial {
        if let Some(path) = path {
            found.push(PathDependency {
                key,
                path,
                package,
                line,
            });
        }
    }
    (found, unsupported)
}

/// `line` without a trailing `# comment` (a `#` inside a string is kept).
fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (index, c) in line.char_indices() {
        match quote {
            Some(open) => {
                if escaped {
                    escaped = false;
                } else if c == '\\' && open == '"' {
                    escaped = true;
                } else if c == open {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '#' => return &line[..index],
                _ => {}
            },
        }
    }
    line
}

/// `text` split at `separator` where it is outside strings, brackets and braces.
fn split_top(text: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut start = 0;
    for (index, c) in text.char_indices() {
        match quote {
            Some(open) => {
                if escaped {
                    escaped = false;
                } else if c == '\\' && open == '"' {
                    escaped = true;
                } else if c == open {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '[' | '{' => depth += 1,
                ']' | '}' => depth = depth.saturating_sub(1),
                _ if c == separator && depth == 0 => {
                    parts.push(&text[start..index]);
                    start = index + c.len_utf8();
                }
                _ => {}
            },
        }
    }
    parts.push(&text[start..]);
    parts
}

/// `key = value` at the first `=` outside strings; the key is returned as written (dots and quotes kept).
fn split_key_value(line: &str) -> Option<(String, &str)> {
    let mut quote: Option<char> = None;
    for (index, c) in line.char_indices() {
        match quote {
            Some(open) => {
                if c == open {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '=' => return Some((line[..index].trim().to_owned(), line[index + 1..].trim())),
                _ => {}
            },
        }
    }
    None
}

/// The content between the braces of an inline table that starts `value`, when it ends on the same line.
fn inline_body(value: &str) -> Option<&str> {
    let inner = value.strip_prefix('{')?;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, c) in inner.char_indices() {
        match quote {
            Some(open) => {
                if escaped {
                    escaped = false;
                } else if c == '\\' && open == '"' {
                    escaped = true;
                } else if c == open {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '[' | '{' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                '}' if depth == 0 => return Some(&inner[..index]),
                '}' => depth -= 1,
                _ => {}
            },
        }
    }
    None
}

/// The text of a basic (`"..."`) or literal (`'...'`) string that starts `value`.
fn string_value(value: &str) -> Option<String> {
    let mut chars = value.trim().chars();
    let open = chars.next().filter(|c| matches!(c, '"' | '\''))?;
    let mut text = String::new();
    while let Some(c) = chars.next() {
        if c == open {
            return Some(text);
        }
        if c == '\\' && open == '"' {
            match chars.next()? {
                '\\' => text.push('\\'),
                '"' => text.push('"'),
                other => {
                    text.push('\\');
                    text.push(other);
                }
            }
        } else {
            text.push(c);
        }
    }
    None
}

/// A key part without its quotes.
fn unquote(part: &str) -> String {
    string_value(part).unwrap_or_else(|| part.to_owned())
}

#[cfg(test)]
#[path = "tests/lint.rs"]
mod lint_tests;
