//! The JSON stream of one shared `cargo check`, split per fixture package (`check-ui`, m13.uifast).
//!
//! `cargo check --keep-going -p a -p b ... --message-format json` prints the `compiler-message` and `compiler-artifact`
//! lines of every package in one stdout. Each line names its package by `manifest_path`, so the stream can be cut into
//! one piece per fixture and each piece judged like the output of a call of its own. A line is only ever given to the
//! package whose manifest it names: a message of package A never judges fixture B, and a dependency's lines belong to
//! no fixture at all.

use std::collections::BTreeMap;

use crate::{
    json::{self, Json},
    ui::UI_DIR,
};

/// What one fixture package printed in a shared call.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PackageStream {
    /// The package's `compiler-message` lines, one per line, in the order cargo printed them: what a call of its own
    /// would have printed on stdout as far as the judge reads it.
    pub(crate) stdout: String,
    /// How many `compiler-artifact` lines the package had: one per target that was checked to the end.
    pub(crate) artifacts: usize,
    /// How many `compiler-message` lines of any target of the package are at level `error`.
    pub(crate) errors: usize,
}

impl PackageStream {
    /// Whether the package compiled: some target was checked to the end and no target reported an error. A package
    /// that cargo never reached (no artifact) did not compile, and nor did one with an error beside its artifacts.
    pub(crate) fn compiled(&self) -> bool {
        self.artifacts > 0 && self.errors == 0
    }
}

/// The fixture a `manifest_path` belongs to: `NAME` of `<anything>/tests/ui/NAME/Cargo.toml`, with backslashes read
/// as slashes. Anything else (a dependency, a manifest deeper in a fixture, another fixture workspace) is `None`.
pub(crate) fn fixture_of(manifest_path: &str) -> Option<String> {
    let path = manifest_path.replace('\\', "/");
    let marker = format!("/{UI_DIR}/");
    let rest = &path[path.rfind(&marker)? + marker.len()..];
    let (name, file) = rest.split_once('/')?;
    (file == "Cargo.toml" && !name.is_empty()).then(|| name.to_owned())
}

/// The `compiler-message` and `compiler-artifact` lines of `stdout`, grouped by fixture package. A fixture that printed
/// no such line has no entry, so the caller can tell "not checked" from "checked and clean". Every other line (text that
/// is not JSON, a line cut short, `build-script-executed`, `build-finished`, a message without a `manifest_path`) is
/// skipped.
pub(crate) fn split_by_package(stdout: &str) -> BTreeMap<String, PackageStream> {
    let mut packages: BTreeMap<String, PackageStream> = BTreeMap::new();
    for line in stdout.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let Ok(parsed) = json::parse(line) else {
            continue;
        };
        let reason = parsed.get("reason").and_then(Json::as_str);
        if !matches!(reason, Some("compiler-message" | "compiler-artifact")) {
            continue;
        }
        let Some(name) = parsed
            .get("manifest_path")
            .and_then(Json::as_str)
            .and_then(fixture_of)
        else {
            continue;
        };
        let package = packages.entry(name).or_default();
        if reason == Some("compiler-artifact") {
            package.artifacts += 1;
            continue;
        }
        if parsed
            .get("message")
            .and_then(|message| message.get("level"))
            .and_then(Json::as_str)
            == Some("error")
        {
            package.errors += 1;
        }
        package.stdout.push_str(line);
        package.stdout.push('\n');
    }
    packages
}

/// The lines of cargo's stderr that name `case` as a package (`` `case` ``, or `` `case v0.0.0 `` in a build script's
/// failure line), for the report of a fixture that has no diagnostics of its own in a shared call.
pub(crate) fn stderr_of(stderr: &str, case: &str) -> String {
    let quoted = format!("`{case}`");
    let versioned = format!("`{case} ");
    let mut lines: String = stderr
        .lines()
        .filter(|line| line.contains(&quoted) || line.contains(&versioned))
        .collect::<Vec<_>>()
        .join("\n");
    lines.push('\n');
    lines
}

/// How many lines of [`shared_cause`] are shown.
const CAUSE_LINES: usize = 200;

/// The rendered diagnostics of the `compiler-message` lines of `stdout` that belong to no fixture package (`gust`, the
/// macro crate, a dependency), with backslashes read as slashes.
fn foreign_rendered(stdout: &str) -> String {
    let mut rendered = String::new();
    for line in stdout.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let Ok(parsed) = json::parse(line) else {
            continue;
        };
        if parsed.get("reason").and_then(Json::as_str) != Some("compiler-message") {
            continue;
        }
        let fixture = parsed
            .get("manifest_path")
            .and_then(Json::as_str)
            .and_then(fixture_of);
        if fixture.is_none()
            && let Some(text) = parsed
                .get("message")
                .and_then(|message| message.get("rendered"))
                .and_then(Json::as_str)
        {
            rendered.push_str(&text.replace('\\', "/"));
        }
    }
    rendered
}

/// The problem that carries the cause of a call (`label` names it) that did not reach every fixture: the diagnostics
/// of packages that are no fixture, then cargo's whole stderr, cut to the first [`CAUSE_LINES`] lines with a note of
/// how many were left out. Lock drift, a failing build script of `gust` and an error in `gust` or the macro crate name
/// no fixture, so no per-fixture problem can show them.
pub(crate) fn shared_cause(stdout: &str, stderr: &str, label: &str) -> String {
    let mut text = foreign_rendered(stdout);
    text.push_str(&stderr.replace('\\', "/"));
    let lines: Vec<&str> = text.lines().collect();
    let mut shown: Vec<String> = lines
        .iter()
        .take(CAUSE_LINES)
        .map(|line| format!("    {line}"))
        .collect();
    if lines.len() > CAUSE_LINES {
        shown.push(format!(
            "    ... {} more line(s) not shown",
            lines.len() - CAUSE_LINES
        ));
    }
    format!(
        "{UI_DIR}: {label} did not reach every fixture; the diagnostics of packages that are no fixture, then cargo's stderr:\n{}",
        shown.join("\n")
    )
}

#[cfg(test)]
#[path = "tests/ui_stream.rs"]
mod tests;
