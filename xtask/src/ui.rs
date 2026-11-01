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

/// The fixture workspace, relative to the repository root. It has a `[workspace]` of its own
/// and is not a member of the root workspace, so `cargo test --workspace` never builds it.
pub(crate) const UI_DIR: &str = "tests/ui";

/// The value of `code:` for an error that has no rustc code (`"code": null`), such as the
/// `compile_error!` of a proc macro. Exact, lower case; no rustc code is spelled this way.
pub(crate) const NO_CODE: &str = "none";

/// What `cargo check` of one fixture crate must do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Expectation {
    /// No `expected.txt`: the crate must compile.
    Compiles,
    /// `expected.txt` lists keys: the crate must fail with exactly one error, and that error must
    /// satisfy all of them (code, location, message and label), not just the output as a whole.
    Fails(Expected),
}

/// The keys of an `expected.txt`, all to hold in the one error of the crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expected {
    /// `code: E0277`: the diagnostic's `code.code`. Required. [`NO_CODE`] (`code: none`) instead
    /// requires `"code": null`, which is what a macro's `compile_error!` has; it is only accepted
    /// together with a `message:`.
    pub(crate) code: String,
    /// `file: gpupod-missing/src/main.rs`: the `file_name` of a primary span, with slashes.
    /// Required, so that a fixture cannot pass on a diagnostic from somewhere else.
    pub(crate) file: String,
    /// `line: 13`: the `line_start` of the same primary span. Required.
    pub(crate) line: u64,
    /// `message: <text>`, any number: a substring of the diagnostic's `message` or of one of
    /// its `children[].message`.
    pub(crate) messages: Vec<String>,
    /// `label: <text>`, any number: a substring of the `label` of the primary span that carries
    /// `file` and `line` (`#[diagnostic::on_unimplemented]`'s `label` lands there, not in a child).
    pub(crate) labels: Vec<String>,
}

/// One fixture crate under `tests/ui/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Case {
    pub(crate) name: String,
    pub(crate) expectation: Expectation,
}

/// A place in a source file: a span's own, or one of the call sites of its macro expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Site {
    /// `file_name` with backslashes read as slashes.
    pub(crate) file: String,
    pub(crate) line: Option<u64>,
}

/// One span of a rustc diagnostic, as far as `expected.txt` can name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Span {
    /// `file_name` with backslashes read as slashes (relative to the fixture workspace).
    pub(crate) file: String,
    pub(crate) line: Option<u64>,
    pub(crate) primary: bool,
    pub(crate) label: Option<String>,
    /// The macro call sites of the span (`expansion.span`, then that span's own `expansion.span`, and so
    /// on), innermost first. Empty for a span that is not from a macro. A panic in a `const` that a
    /// macro generated has core's `panic.rs` as its primary span and the user's file only here.
    pub(crate) call_sites: Vec<Site>,
}

/// One `compiler-message` of `cargo check --message-format json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Diagnostic {
    /// The cargo target (`target.name`) the message belongs to; a dependency's own diagnostics
    /// carry its name, not the fixture's.
    pub(crate) target: String,
    pub(crate) level: String,
    /// `code.code`; `None` where rustc prints `"code": null` (summary notes, some errors).
    pub(crate) code: Option<String>,
    pub(crate) message: String,
    pub(crate) child_messages: Vec<String>,
    pub(crate) spans: Vec<Span>,
    /// The text rustc rendered for humans, with backslashes read as slashes.
    pub(crate) rendered: String,
}

/// `cargo xtask check-ui`: `cargo check` of each fixture crate under `tests/ui/`, compared with
/// its expectation. Needs cargo only; no slangc and no GPU. Also part of the Unit suite.
pub(crate) fn check_ui(root: &Path, checks: &mut Vec<CheckRecord>) -> Result<(), String> {
    let outcome = run_cases(root, checks);
    report_check(checks, "check-ui", outcome)
}

/// Returns a summary on success, or one line per problem.
fn run_cases(root: &Path, checks: &mut Vec<CheckRecord>) -> Result<String, Vec<String>> {
    let cases = discover_cases(&root.join(UI_DIR)).map_err(|error| vec![error])?;
    let mut problems = Vec::new();
    for case in &cases {
        let args = cargo_args(&case.name);
        let output = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
            .args(&args)
            .current_dir(root)
            .stdin(Stdio::null())
            .output();
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                problems.push(format!("{}: could not launch cargo: {error}", case.name));
                continue;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        problems.extend(judge(
            &case.name,
            &case.expectation,
            output.status.success(),
            &stdout,
            &stderr,
        ));
        // The record keeps what a reader of the output would see: the rendered diagnostics
        // (not the JSON, whose artifact lines carry absolute paths and are of no use here).
        let rendered = parse_diagnostics(&stdout)
            .iter()
            .map(|diagnostic| diagnostic.rendered.as_str())
            .collect::<String>();
        checks.push(CheckRecord {
            program: "cargo".to_owned(),
            arguments: args,
            exit_code: output.status.code(),
            stdout: rendered,
            stderr,
        });
    }
    if !problems.is_empty() {
        return Err(problems);
    }
    let failing = cases
        .iter()
        .filter(|case| matches!(case.expectation, Expectation::Fails(_)))
        .count();
    Ok(format!(
        "UI check passed: {failing} fixture crate(s) failed to compile with exactly the expected rustc error and {} compiled cleanly (tests/ui, `cargo check --locked --message-format json`). Only the keys of each expected.txt (code, file, line, message and label) are compared, all within the one error of the crate itself (warnings and other crates are ignored, a second error fails) read from cargo's JSON, not the whole rustc output; no slangc and no GPU were involved.\n",
        cases.len() - failing
    ))
}

/// The arguments after `cargo` that check one fixture crate. `--locked` keeps the fixture's own
/// `Cargo.lock` honest, `--color never` keeps ANSI codes out of the rendered text and
/// `--message-format json` makes cargo print one JSON object per line on stdout.
pub(crate) fn cargo_args(case: &str) -> Vec<String> {
    [
        "check",
        "--locked",
        "--color",
        "never",
        "--manifest-path",
        "tests/ui/Cargo.toml",
        "-p",
        case,
        "--message-format",
        "json",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect()
}

/// The fixture crates of `ui_root`, sorted by name: every subdirectory with a `Cargo.toml`.
pub(crate) fn discover_cases(ui_root: &Path) -> Result<Vec<Case>, String> {
    let entries =
        fs::read_dir(ui_root).map_err(|error| format!("{UI_DIR}: could not be read: {error}"))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{UI_DIR}: {error}"))?;
        if entry.path().join("Cargo.toml").is_file() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!(
            "{UI_DIR} has no fixture crate (a directory with a Cargo.toml); a check without a case would pass without checking anything"
        ));
    }
    names
        .into_iter()
        .map(|name| {
            let expected = ui_root.join(&name).join("expected.txt");
            let expectation = if expected.is_file() {
                let text = fs::read_to_string(&expected)
                    .map_err(|error| format!("{UI_DIR}/{name}/expected.txt: {error}"))?;
                Expectation::Fails(
                    parse_expected(&text).map_err(|error| format!("{UI_DIR}/{name}: {error}"))?,
                )
            } else {
                Expectation::Compiles
            };
            Ok(Case { name, expectation })
        })
        .collect()
}

/// The keys of an `expected.txt`: `key: value` lines, blank lines and lines starting with `#`
/// ignored. `code`, `file` and `line` are required (a file without them would accept almost any
/// failure) and may appear once; `message` and `label` may repeat. `code: none` names an error
/// without a rustc code (a macro's `compile_error!`) and needs at least one `message:`, since the
/// message text is then the only thing that tells the error from any other code-less one. Unknown
/// keys, empty values, a `line` that is not a positive number and an absolute `file` are errors.
pub(crate) fn parse_expected(text: &str) -> Result<Expected, String> {
    let mut code: Option<String> = None;
    let mut file: Option<String> = None;
    let mut line: Option<u64> = None;
    let mut messages = Vec::new();
    let mut labels = Vec::new();
    let mut listed = 0_usize;
    for raw in text.trim_start_matches('\u{feff}').lines() {
        let entry = raw.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        listed += 1;
        let Some((key, value)) = entry.split_once(':') else {
            return Err(format!(
                "expected.txt line `{entry}` is not a `key: value` pair (keys: code, file, line, message, label)"
            ));
        };
        let (key, value) = (key.trim(), value.trim());
        if !matches!(key, "code" | "file" | "line" | "message" | "label") {
            return Err(format!(
                "expected.txt has an unknown key `{key}` (keys: code, file, line, message, label)"
            ));
        }
        if value.is_empty() {
            return Err(format!("expected.txt: `{key}` has no value"));
        }
        let twice = || format!("expected.txt: `{key}` appears twice");
        match key {
            "code" => {
                if code.replace(value.to_owned()).is_some() {
                    return Err(twice());
                }
            }
            "file" => {
                let path = value.replace('\\', "/");
                if is_absolute(&path) {
                    return Err(format!(
                        "expected.txt: `file: {value}` must be relative to tests/ui (an absolute path is machine specific)"
                    ));
                }
                if file.replace(path).is_some() {
                    return Err(twice());
                }
            }
            "line" => {
                let Some(number) = value.parse::<u64>().ok().filter(|number| *number > 0) else {
                    return Err(format!(
                        "expected.txt: `line: {value}` is not a positive line number"
                    ));
                };
                if line.replace(number).is_some() {
                    return Err(twice());
                }
            }
            "message" => messages.push(value.to_owned()),
            _ => labels.push(value.to_owned()),
        }
    }
    if listed == 0 {
        return Err("expected.txt lists no key; delete it if the crate must compile".to_owned());
    }
    let Some(code) = code else {
        return Err(
            "expected.txt has no `code:` line; without it any diagnostic would do".to_owned(),
        );
    };
    let Some(file) = file else {
        return Err(
            "expected.txt has no `file:` line; without it a diagnostic from anywhere would do"
                .to_owned(),
        );
    };
    let Some(line) = line else {
        return Err(
            "expected.txt has no `line:` line; without it any line of the file would do".to_owned(),
        );
    };
    if code == NO_CODE && messages.is_empty() {
        return Err(format!(
            "expected.txt has `code: {NO_CODE}` but no `message:` line; without it any error without a rustc code would do"
        ));
    }
    Ok(Expected {
        code,
        file,
        line,
        messages,
        labels,
    })
}

/// Whether a slash-separated path is absolute: a leading slash (also `//host/share`) or a
/// drive letter.
fn is_absolute(path: &str) -> bool {
    path.starts_with('/') || path.as_bytes().get(1) == Some(&b':')
}

/// The problems of one fixture run: none when the outcome matches the expectation.
///
/// Only the case's own diagnostics count (`target.name` is the case), so a dependency's error
/// never satisfies an expectation. A failing case must report exactly one error-level message;
/// warnings and the summary `failure-note` are ignored, and a second error is a problem of its
/// own that names it.
pub(crate) fn judge(
    case: &str,
    expectation: &Expectation,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Vec<String> {
    let diagnostics = parse_diagnostics(stdout);
    let errors: Vec<&Diagnostic> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.target == case && diagnostic.level == "error")
        .collect();
    let shown = || indent(&output_shown(&diagnostics, stderr));
    match expectation {
        Expectation::Compiles if success => Vec::new(),
        Expectation::Compiles => vec![format!(
            "{case}: has no expected.txt so it must compile, but `cargo check` failed:\n{}",
            shown()
        )],
        Expectation::Fails(_) if success => vec![format!(
            "{case}: has an expected.txt so `cargo check` must fail, but it compiled"
        )],
        Expectation::Fails(_) if errors.is_empty() => vec![format!(
            "{case}: `cargo check` failed but printed no error-level compiler-message of its own, so expected.txt cannot match; cargo said:\n{}",
            shown()
        )],
        Expectation::Fails(_) if errors.len() > 1 => vec![
            format!(
                "{case}: expected.txt describes one error, but rustc reported {} errors: {}",
                errors.len(),
                errors
                    .iter()
                    .map(|error| describe(case, error))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            format!("{case}: rustc output was:\n{}", shown()),
        ],
        Expectation::Fails(wanted) => {
            let error = errors[0];
            let missing = gaps(case, wanted, error);
            if missing.is_empty() {
                return Vec::new();
            }
            let mut problems: Vec<String> = missing
                .iter()
                .map(|gap| {
                    format!(
                        "{case}: the rustc error does not satisfy `{gap}` of expected.txt (error was: {})",
                        describe(case, error)
                    )
                })
                .collect();
            problems.push(format!("{case}: rustc output was:\n{}", shown()));
            problems
        }
    }
}

/// The keys of `wanted` that `error` does not satisfy, each written as in expected.txt.
fn gaps(case: &str, wanted: &Expected, error: &Diagnostic) -> Vec<String> {
    let mut gaps = Vec::new();
    // `code: none` wants an error with no rustc code; any other value wants exactly that code.
    let code_holds = if wanted.code == NO_CODE {
        error.code.is_none()
    } else {
        error.code.as_deref() == Some(wanted.code.as_str())
    };
    if !code_holds {
        gaps.push(format!("code: {}", wanted.code));
    }
    // `file` and `line` say where the error is reported: one primary span must carry both. A
    // wrong file hides the line, and without a located span there is no label to look at.
    let on_file: Vec<&Span> = error
        .spans
        .iter()
        .filter(|span| span.primary && reported_at(case, span).file == wanted.file)
        .collect();
    let located: Vec<&Span> = on_file
        .iter()
        .copied()
        .filter(|span| reported_at(case, span).line == Some(wanted.line))
        .collect();
    if on_file.is_empty() {
        gaps.push(format!("file: {}", wanted.file));
    } else if located.is_empty() {
        gaps.push(format!("line: {}", wanted.line));
    }
    for text in &wanted.messages {
        let found = error.message.contains(text.as_str())
            || error
                .child_messages
                .iter()
                .any(|child| child.contains(text.as_str()));
        if !found {
            gaps.push(format!("message: {text}"));
        }
    }
    if !located.is_empty() {
        for text in &wanted.labels {
            let found = located.iter().any(|span| {
                span.label
                    .as_deref()
                    .is_some_and(|label| label.contains(text.as_str()))
            });
            if !found {
                gaps.push(format!("label: {text}"));
            }
        }
    }
    gaps
}

/// Whether `file` (slash-separated, as in [`Span`]) is a file of the case: relative to `tests/ui` and under
/// the case's directory. Core's `panic.rs` and every other absolute path are not.
fn in_case(case: &str, file: &str) -> bool {
    !is_absolute(file)
        && file
            .strip_prefix(case)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Where the span is reported, for `file` and `line`. A primary span in the case's own files is reported
/// where it is. One outside them (the `panic.rs` of a `const` panic that a macro generated, say) is
/// reported at the outermost of its macro call sites that lies in the case, which is where the user's
/// attribute or macro call is. Without such a call site it stays where it is, so it never matches.
fn reported_at(case: &str, span: &Span) -> Site {
    let own = Site {
        file: span.file.clone(),
        line: span.line,
    };
    if in_case(case, &span.file) {
        return own;
    }
    span.call_sites
        .iter()
        .rfind(|site| in_case(case, &site.file))
        .cloned()
        .unwrap_or(own)
}

/// `error[E0277] message at file:line`, for naming a diagnostic.
fn describe(case: &str, diagnostic: &Diagnostic) -> String {
    let code = diagnostic
        .code
        .as_deref()
        .map_or_else(String::new, |code| format!("[{code}]"));
    let place = diagnostic
        .spans
        .iter()
        .find(|span| span.primary)
        .map_or_else(String::new, |span| {
            let site = reported_at(case, span);
            let line = site
                .line
                .map_or_else(|| "?".to_owned(), |line| line.to_string());
            format!(" at {}:{line}", site.file)
        });
    format!("{}{code} {}{place}", diagnostic.level, diagnostic.message)
}

/// What a reader sees of a run: the rendered diagnostics, then cargo's own stderr (backslashes
/// read as slashes). Cargo's JSON has no text of its own beyond the diagnostics.
fn output_shown(diagnostics: &[Diagnostic], stderr: &str) -> String {
    let mut shown: String = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.rendered.as_str())
        .collect();
    shown.push_str(&stderr.replace('\\', "/"));
    shown.trim_end().to_owned()
}

/// The `compiler-message` lines of `cargo check --message-format json` stdout. Every other
/// line (artifacts, `build-finished`, text that is not JSON, a JSON line cut short) is skipped,
/// and so is a message without a `target.name` or `message` text.
pub(crate) fn parse_diagnostics(stdout: &str) -> Vec<Diagnostic> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('{'))
        .filter_map(|line| json::parse(line).ok())
        .filter(|line| line.get("reason").and_then(Json::as_str) == Some("compiler-message"))
        .filter_map(|line| {
            let target = line.get("target")?.get("name")?.as_str()?;
            diagnostic_of(target, line.get("message")?)
        })
        .collect()
}

fn diagnostic_of(target: &str, message: &Json) -> Option<Diagnostic> {
    let text = |value: Option<&Json>| value.and_then(Json::as_str).map(str::to_owned);
    let spans = message
        .get("spans")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(span_of)
        .collect();
    let child_messages = message
        .get("children")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(|child| text(child.get("message")))
        .collect();
    Some(Diagnostic {
        target: target.to_owned(),
        level: text(message.get("level")).unwrap_or_default(),
        code: message
            .get("code")
            .and_then(|code| code.get("code"))
            .and_then(Json::string_or_null),
        message: text(message.get("message"))?,
        child_messages,
        spans,
        rendered: text(message.get("rendered"))
            .unwrap_or_default()
            .replace('\\', "/"),
    })
}

fn span_of(span: &Json) -> Option<Span> {
    Some(Span {
        file: span
            .get("file_name")
            .and_then(Json::as_str)?
            .replace('\\', "/"),
        line: match span.get("line_start") {
            Some(Json::Number(text)) => text.parse().ok(),
            _ => None,
        },
        primary: matches!(span.get("is_primary"), Some(Json::Bool(true))),
        label: span.get("label").and_then(Json::string_or_null),
        call_sites: call_sites_of(span),
    })
}

/// The macro call sites of a span, innermost first (see [`Span::call_sites`]).
fn call_sites_of(span: &Json) -> Vec<Site> {
    let mut sites = Vec::new();
    let mut current = span;
    while let Some(site) = current
        .get("expansion")
        .and_then(|expansion| expansion.get("span"))
    {
        let Some(file) = site.get("file_name").and_then(Json::as_str) else {
            break;
        };
        sites.push(Site {
            file: file.replace('\\', "/"),
            line: match site.get("line_start") {
                Some(Json::Number(text)) => text.parse().ok(),
                _ => None,
            },
        });
        current = site;
    }
    sites
}

fn indent(text: &str) -> String {
    text.lines()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[path = "tests/ui.rs"]
mod ui_tests;
