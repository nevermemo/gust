use std::fs;

use super::{
    Expectation, Expected, UI_DIR, cargo_args, case_command, discover_cases, judge,
    parse_diagnostics, parse_env, parse_expectation, parse_expected,
};
use crate::workspace::workspace_root;

const MESSAGE: &str = "`Outside` is not a GUST GPU POD type";
const NOTE: &str = "an `unsafe impl` written by hand must uphold the layout contract in the safety section of `GpuPod`";
const LABEL: &str = "declare a plain struct inside the same `#[gpu]` module so that the macro implements `GpuPod` for it; `u32`, `i32` and `f32` already implement it";

/// The `compiler-message` line of `cargo check --message-format json -p gpupod-missing` on
/// Windows (probe/d0-json-gpupod-missing.log), trimmed from a real run; the fields the matcher
/// reads are unchanged. Cut: `package_id`, `manifest_path`, `explanation`, the long `rendered`
/// text, the `text` arrays, two of the three `abi.rs` child spans and byte offsets were
/// shortened or dropped, and the remaining `abi.rs` path is made relative. Kept: `file_name`
/// is relative to the fixture workspace and uses backslashes, the label sits on the span (not on
/// a child), the `GpuPod` help and the attribute's note are child messages, and the child spans
/// name other files and lines.
const GPUPOD_MISSING_LINE: &str = r#"{"reason":"compiler-message","target":{"kind":["bin"],"crate_types":["bin"],"name":"gpupod-missing"},"message":{"rendered":"error[E0277]: `Outside` is not a GUST GPU POD type\n   --> gpupod-missing\\src\\main.rs:13:65\n","$message_type":"diagnostic","children":[{"children":[],"code":null,"level":"help","message":"the trait `GpuPod` is not implemented for `Outside`","rendered":null,"spans":[{"byte_end":120,"byte_start":106,"column_end":15,"column_start":1,"expansion":null,"file_name":"gpupod-missing\\src\\main.rs","is_primary":true,"label":null,"line_end":6,"line_start":6,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]},{"children":[],"code":null,"level":"note","message":"an `unsafe impl` written by hand must uphold the layout contract in the safety section of `GpuPod`","rendered":null,"spans":[]},{"children":[],"code":null,"level":"help","message":"the following other types implement trait `GpuPod`","rendered":null,"spans":[{"byte_end":4000,"byte_start":3974,"column_end":27,"column_start":1,"expansion":null,"file_name":"crates\\gust\\src\\abi.rs","is_primary":true,"label":"`f32`","line_end":141,"line_start":141,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}],"code":{"code":"E0277","explanation":null},"level":"error","message":"`Outside` is not a GUST GPU POD type","spans":[{"byte_end":459,"byte_start":445,"column_end":79,"column_start":65,"expansion":null,"file_name":"gpupod-missing\\src\\main.rs","is_primary":true,"label":"declare a plain struct inside the same `#[gpu]` module so that the macro implements `GpuPod` for it; `u32`, `i32` and `f32` already implement it","line_end":13,"line_start":13,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

/// A second error of the same crate, trimmed the same way from a run of `gpupod-missing` with
/// `let _ = undefined_name;` added to `main` (probe/d0-json-second-error.log). rustc printed it
/// before the `GpuPod` error.
const SECOND_ERROR_LINE: &str = r#"{"reason":"compiler-message","target":{"kind":["bin"],"crate_types":["bin"],"name":"gpupod-missing"},"message":{"rendered":"error[E0425]: cannot find value `undefined_name` in this scope\n  --> gpupod-missing\\src\\main.rs:20:13\n","$message_type":"diagnostic","children":[],"code":{"code":"E0425","explanation":null},"level":"error","message":"cannot find value `undefined_name` in this scope","spans":[{"byte_end":600,"byte_start":586,"column_end":27,"column_start":13,"expansion":null,"file_name":"gpupod-missing\\src\\main.rs","is_primary":true,"label":"not found in this scope","line_end":20,"line_start":20,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

/// The other lines cargo prints around it: an artifact line, the summary note (`failure-note`
/// level, no code) and the final `build-finished` line.
const ARTIFACT_LINE: &str = r#"{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"gust"},"profile":{"opt_level":"0"},"fresh":false}"#;
const FAILURE_NOTE_LINE: &str = r#"{"reason":"compiler-message","target":{"kind":["bin"],"crate_types":["bin"],"name":"gpupod-missing"},"message":{"rendered":"For more information about this error, try `rustc --explain E0277`.\n","$message_type":"diagnostic","children":[],"code":null,"level":"failure-note","message":"For more information about this error, try `rustc --explain E0277`.","spans":[]}}"#;
const FINISHED_LINE: &str = r#"{"reason":"build-finished","success":false}"#;

fn cargo_stdout(lines: &[&str]) -> String {
    let mut stdout = String::new();
    for line in lines {
        stdout.push_str(line);
        stdout.push('\n');
    }
    stdout
}

fn gpupod_stdout() -> String {
    cargo_stdout(&[
        ARTIFACT_LINE,
        "this is not json",
        GPUPOD_MISSING_LINE,
        FAILURE_NOTE_LINE,
        FINISHED_LINE,
    ])
}

/// A synthetic `compiler-message` line of the crate `a`: `spans` are
/// `(file, line, is_primary, label)`.
fn message_line(
    level: &str,
    code: Option<&str>,
    message: &str,
    spans: &[(&str, u32, bool, Option<&str>)],
    children: &[&str],
) -> String {
    let code = code.map_or_else(
        || "null".to_owned(),
        |code| format!(r#"{{"code":"{code}","explanation":null}}"#),
    );
    let spans: Vec<String> = spans
        .iter()
        .map(|(file, line, primary, label)| {
            let label = label.map_or_else(|| "null".to_owned(), |label| format!("\"{label}\""));
            format!(
                r#"{{"file_name":"{file}","is_primary":{primary},"label":{label},"line_start":{line},"line_end":{line}}}"#
            )
        })
        .collect();
    let children: Vec<String> = children
        .iter()
        .map(|text| format!(r#"{{"level":"help","message":"{text}","spans":[],"children":[]}}"#))
        .collect();
    format!(
        r#"{{"reason":"compiler-message","target":{{"kind":["bin"],"name":"a"}},"message":{{"rendered":"{level}: {message}\n","children":[{}],"code":{code},"level":"{level}","message":"{message}","spans":[{}]}}}}"#,
        children.join(","),
        spans.join(",")
    )
}

fn expected(code: &str, file: &str, line: u64, messages: &[&str], labels: &[&str]) -> Expected {
    Expected {
        code: code.to_owned(),
        file: file.to_owned(),
        line,
        messages: messages.iter().map(|text| (*text).to_owned()).collect(),
        labels: labels.iter().map(|text| (*text).to_owned()).collect(),
    }
}

/// The converted expectation of `gpupod-missing`, spelled out: `tests/ui/gpupod-missing/expected.txt`
/// must parse to exactly this (see the repository test), so editing the file alone fails a test.
fn gpupod_expected() -> Expected {
    expected(
        "E0277",
        "gpupod-missing/src/main.rs",
        13,
        &[MESSAGE, NOTE],
        &[LABEL],
    )
}

fn fails(wanted: Expected) -> Expectation {
    Expectation::Fails(wanted)
}

fn run(wanted: &Expected, stdout: &str) -> Vec<String> {
    judge(
        "gpupod-missing",
        "gpupod-missing",
        &fails(wanted.clone()),
        false,
        stdout,
        "",
    )
}

/// The case `a` of the synthetic lines.
fn run_a(wanted: &Expected, lines: &[String]) -> Vec<String> {
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    judge(
        "a",
        "a",
        &fails(wanted.clone()),
        false,
        &cargo_stdout(&lines),
        "",
    )
}

#[test]
fn expected_txt_accepts_the_keys_and_skips_blank_and_comment_lines() {
    let text = "# the attribute message\n\ncode: E0277\nfile: gpupod-missing/src/main.rs\nline: 13\n  # indented comment\nmessage: `Outside` is not a GUST GPU POD type\nlabel: declare a plain struct: with a colon\nmessage: second\n";
    assert_eq!(
        parse_expected(text),
        Ok(expected(
            "E0277",
            "gpupod-missing/src/main.rs",
            13,
            &["`Outside` is not a GUST GPU POD type", "second"],
            &["declare a plain struct: with a colon"],
        ))
    );
    // `code`, `file` and `line` are enough; CRLF line endings and a byte order mark do not matter.
    assert_eq!(
        parse_expected("\u{feff}code: E0599\r\nfile: a.rs\r\nline: 2\r\n"),
        Ok(expected("E0599", "a.rs", 2, &[], &[]))
    );
    // A path written with backslashes reads as slashes, as the rustc paths do.
    assert_eq!(
        parse_expected("code: E0277\nfile: a\\src\\main.rs\nline: 1\n")
            .unwrap()
            .file,
        "a/src/main.rs"
    );
}

#[test]
fn expected_txt_rejects_what_it_cannot_judge() {
    let rejected = |text: &str| parse_expected(text).unwrap_err();
    // A file that lists nothing, or only comments, would accept any failure.
    for empty in ["", "\n\n", "# only a comment\n"] {
        assert!(rejected(empty).contains("lists no key"), "{empty:?}");
    }
    // `code`, `file` and `line` are each required: without them almost any diagnostic would do.
    assert!(rejected("message: x\n").contains("no `code:` line"));
    assert!(rejected("file: a.rs\nline: 1\n").contains("no `code:` line"));
    assert!(rejected("code: E0277\nline: 1\nmessage: x\n").contains("no `file:` line"));
    assert!(rejected("code: E0277\nfile: a.rs\nlabel: x\n").contains("no `line:` line"));
    assert!(rejected("code: E0277\n").contains("no `file:` line"));
    // Unknown keys are typos that would otherwise pass silently; the old one-substring-per-line
    // format has no key at all.
    assert!(rejected("code: E0277\nlabl: x\n").contains("unknown key `labl`"));
    assert!(rejected("code: E0277\nerror[E0277]\n").contains("`key: value`"));
    assert!(rejected("code: E0277\nMessage: x\n").contains("unknown key `Message`"));
    // Single-valued keys appear once; a value must be there and `line` must be a number.
    assert!(rejected("code: E0277\ncode: E0271\n").contains("`code` appears twice"));
    assert!(rejected("code: E0277\nfile: a\nfile: b\n").contains("`file` appears twice"));
    assert!(rejected("code: E0277\nline: 1\nline: 2\n").contains("`line` appears twice"));
    assert!(rejected("code: E0277\nmessage:   \n").contains("`message` has no value"));
    assert!(rejected("code: \n").contains("`code` has no value"));
    for bad in ["thirteen", "-1", "0", "1.5", "99999999999999999999999"] {
        assert!(
            rejected(&format!("code: E0277\nline: {bad}\n"))
                .contains("is not a positive line number"),
            "{bad}"
        );
    }
    // A fixture path is relative to the fixture workspace; an absolute one is machine specific.
    for absolute in [
        "/opt/a/main.rs",
        "C:/w/main.rs",
        "C:\\a\\main.rs",
        "\\\\host\\share\\a.rs",
    ] {
        let error = rejected(&format!("code: E0277\nfile: {absolute}\n"));
        assert!(error.contains("must be relative"), "{absolute}: {error}");
    }
}

#[test]
fn a_captured_cargo_line_is_read_field_by_field() {
    let diagnostics = parse_diagnostics(&gpupod_stdout());
    // The artifact line, the non-JSON line and `build-finished` are skipped; the error and the
    // summary note remain.
    assert_eq!(diagnostics.len(), 2, "{diagnostics:#?}");
    let error = &diagnostics[0];
    assert_eq!(error.target, "gpupod-missing");
    assert_eq!(error.level, "error");
    assert_eq!(error.code.as_deref(), Some("E0277"));
    assert_eq!(error.message, MESSAGE);
    assert_eq!(error.child_messages.len(), 3);
    assert_eq!(error.child_messages[1], NOTE);
    assert_eq!(error.spans.len(), 1);
    assert_eq!(error.spans[0].file, "gpupod-missing/src/main.rs");
    assert_eq!(error.spans[0].line, Some(13));
    assert!(error.spans[0].primary);
    assert_eq!(error.spans[0].label.as_deref(), Some(LABEL));
    // `code` is JSON null on the summary note.
    assert_eq!(diagnostics[1].level, "failure-note");
    assert_eq!(diagnostics[1].code, None);
    // A message without `target.name` is skipped: it cannot be attributed to a crate.
    let no_target = GPUPOD_MISSING_LINE.replace(
        r#""target":{"kind":["bin"],"crate_types":["bin"],"name":"gpupod-missing"},"#,
        "",
    );
    assert_ne!(no_target, GPUPOD_MISSING_LINE);
    assert!(parse_diagnostics(&no_target).is_empty());
}

#[test]
fn the_converted_expectation_passes_on_the_captured_line_and_on_slash_paths() {
    assert_eq!(
        run(&gpupod_expected(), &gpupod_stdout()),
        Vec::<String>::new()
    );
    // The same line with Unix paths (cargo prints slashes there) passes too.
    let unix = gpupod_stdout().replace("\\\\", "/");
    assert_ne!(unix, gpupod_stdout());
    assert_eq!(run(&gpupod_expected(), &unix), Vec::<String>::new());
    // `code`, `file` and `line` alone are enough when nothing else is asked.
    let bare = expected("E0277", "gpupod-missing/src/main.rs", 13, &[], &[]);
    assert!(run(&bare, &gpupod_stdout()).is_empty());
}

#[test]
fn a_wrong_line_is_named_with_the_error_that_was_found() {
    let mut wanted = gpupod_expected();
    wanted.line = 14;
    let problems = run(&wanted, &gpupod_stdout());
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(
        problems[0].contains("does not satisfy `line: 14`"),
        "{problems:?}"
    );
    assert!(
        problems[0].contains(
            "error was: error[E0277] `Outside` is not a GUST GPU POD type at gpupod-missing/src/main.rs:13"
        ),
        "{problems:?}"
    );
    assert!(problems[1].contains("rustc output was:"), "{problems:?}");
    assert!(
        problems[1].contains("    error[E0277]: `Outside`"),
        "{problems:?}"
    );
    // Line 6 belongs to a child span (the `GpuPod` help), not to the error's own span.
    wanted.line = 6;
    let problems = run(&wanted, &gpupod_stdout());
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(problems[0].contains("`line: 6`"), "{problems:?}");
}

#[test]
fn a_wrong_code_file_message_or_label_is_each_a_failure() {
    let stdout = gpupod_stdout();
    let one_gap = |change: &dyn Fn(&mut Expected), gap: &str| {
        let mut wanted = gpupod_expected();
        change(&mut wanted);
        let problems = run(&wanted, &stdout);
        assert_eq!(problems.len(), 2, "{gap}: {problems:#?}");
        assert!(
            problems[0].contains(&format!("does not satisfy `{gap}`")),
            "{gap}: {problems:?}"
        );
    };
    one_gap(&|w| w.code = "E0271".to_owned(), "code: E0271");
    one_gap(
        &|w| w.file = "kernel-missing/src/main.rs".to_owned(),
        "file: kernel-missing/src/main.rs",
    );
    one_gap(
        &|w| w.messages.push("is not a GUST kernel handle".to_owned()),
        "message: is not a GUST kernel handle",
    );
    one_gap(
        &|w| w.labels.push("pass the zero-sized handle".to_owned()),
        "label: pass the zero-sized handle",
    );
    // The attribute's note and the whole label are pinned: a changed word is a failure.
    one_gap(
        &|w| w.messages[1] = NOTE.replace("layout contract", "layout rule"),
        &format!(
            "message: {}",
            NOTE.replace("layout contract", "layout rule")
        ),
    );
    one_gap(
        &|w| w.labels[0] = LABEL.replace("already implement it", "implement it"),
        &format!(
            "label: {}",
            LABEL.replace("already implement it", "implement it")
        ),
    );
    // Every gap is reported, not only the first. A wrong file hides the line and the labels:
    // there is no located span to look at.
    let mut all = gpupod_expected();
    all.code = "E0271".to_owned();
    all.file = "x.rs".to_owned();
    all.messages = vec!["nope".to_owned()];
    all.labels = vec!["nada".to_owned()];
    let problems = run(&all, &stdout);
    assert_eq!(problems.len(), 4, "{problems:#?}");
    for gap in ["code: E0271", "file: x.rs", "message: nope"] {
        assert!(
            problems
                .iter()
                .any(|p| p.contains(&format!("does not satisfy `{gap}`"))),
            "{gap}: {problems:#?}"
        );
    }
}

#[test]
fn messages_match_the_message_and_the_child_messages_but_labels_only_the_located_span() {
    let stdout = gpupod_stdout();
    let wanted = |messages: &[&str], labels: &[&str]| {
        expected("E0277", "gpupod-missing/src/main.rs", 13, messages, labels)
    };
    // The `GpuPod` help is a child message: `message:` finds it.
    let child = wanted(
        &["the trait `GpuPod` is not implemented for `Outside`"],
        &[],
    );
    assert_eq!(run(&child, &stdout), Vec::<String>::new());
    // The wording of a label is not a message and a message is not a label.
    assert_eq!(run(&wanted(&[LABEL], &[]), &stdout).len(), 2);
    assert_eq!(run(&wanted(&[], &[MESSAGE]), &stdout).len(), 2);
    // A child span's label (the `f32` one, from the "other types" help) does not count as a
    // label of the error: renamed here so that no top-level label contains it.
    let renamed = stdout.replace(r#""label":"`f32`""#, r#""label":"child only""#);
    assert_ne!(renamed, stdout);
    assert_eq!(run(&wanted(&[], &["child only"]), &renamed).len(), 2);
}

#[test]
fn a_label_must_sit_on_the_primary_span_that_carries_file_and_line() {
    let line = |spans: &[(&str, u32, bool, Option<&str>)]| {
        vec![message_line("error", Some("E0277"), "m", spans, &[])]
    };
    let wanted = |labels: &[&str]| expected("E0277", "a/one.rs", 5, &[], labels);
    let lines = line(&[
        ("a/one.rs", 5, true, Some("right label")),
        ("a/two.rs", 9, true, Some("elsewhere")),
        ("a/one.rs", 5, false, Some("secondary only")),
        ("a/one.rs", 7, true, Some("other line")),
    ]);
    assert!(run_a(&wanted(&["right label"]), &lines).is_empty());
    for not_there in ["secondary only", "elsewhere", "other line", "no such label"] {
        let problems = run_a(&wanted(&[not_there]), &lines);
        assert_eq!(problems.len(), 2, "{not_there}: {problems:#?}");
        assert!(
            problems[0].contains(&format!("does not satisfy `label: {not_there}`")),
            "{problems:?}"
        );
    }
    // A label that only a secondary span carries does not satisfy the key either when it is
    // the only span with a label.
    let only_secondary = line(&[
        ("a/one.rs", 5, true, None),
        ("a/one.rs", 5, false, Some("secondary only")),
    ]);
    assert_eq!(
        run_a(&wanted(&["secondary only"]), &only_secondary).len(),
        2
    );
}

#[test]
fn only_the_crates_own_diagnostics_count() {
    // A matching E0277 from another package (a dependency) is ignored: the crate itself printed
    // no error, so there is nothing to compare.
    let other = gpupod_stdout().replace(r#""name":"gpupod-missing""#, r#""name":"gust""#);
    assert_ne!(other, gpupod_stdout());
    let problems = run(&gpupod_expected(), &other);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("printed no error-level compiler-message of its own"),
        "{problems:?}"
    );
    // Another package's matching error does not rescue the crate's own error on the wrong line
    // either, and does not count as a second error.
    let own_wrong_line = GPUPOD_MISSING_LINE
        .replace(r#""line_start":13"#, r#""line_start":99"#)
        .replace(r#""line_end":13"#, r#""line_end":99"#);
    assert_ne!(own_wrong_line, GPUPOD_MISSING_LINE);
    let dependency = GPUPOD_MISSING_LINE.replace(r#""name":"gpupod-missing""#, r#""name":"gust""#);
    let stdout = cargo_stdout(&[&dependency, &own_wrong_line, FAILURE_NOTE_LINE]);
    let problems = run(&gpupod_expected(), &stdout);
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(problems[0].contains("`line: 13`"), "{problems:?}");
    // The other way round: the dependency's error is on the wrong line, the crate's is right.
    let stdout = cargo_stdout(&[
        &own_wrong_line.replace(r#""name":"gpupod-missing""#, r#""name":"gust""#),
        GPUPOD_MISSING_LINE,
    ]);
    assert_eq!(run(&gpupod_expected(), &stdout), Vec::<String>::new());
}

#[test]
fn a_failing_case_must_report_exactly_one_error() {
    // The pinned error plus a second one (a real E0425 line, printed first): a problem that
    // names the second error.
    let stdout = cargo_stdout(&[
        SECOND_ERROR_LINE,
        GPUPOD_MISSING_LINE,
        FAILURE_NOTE_LINE,
        FINISHED_LINE,
    ]);
    let problems = run(&gpupod_expected(), &stdout);
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(
        problems[0].contains("one error, but rustc reported 2 errors"),
        "{problems:?}"
    );
    assert!(
        problems[0].contains(
            "error[E0425] cannot find value `undefined_name` in this scope at gpupod-missing/src/main.rs:20"
        ),
        "{problems:?}"
    );
    assert!(problems[1].contains("rustc output was:"), "{problems:?}");
    assert!(
        problems[1].contains("error[E0425]: cannot find value `undefined_name`"),
        "{problems:?}"
    );
    // The same with the order reversed.
    let stdout = cargo_stdout(&[GPUPOD_MISSING_LINE, SECOND_ERROR_LINE]);
    assert_eq!(run(&gpupod_expected(), &stdout).len(), 2);
    // Warnings and the summary notes do not count as errors.
    let warning = message_line(
        "warning",
        Some("unused"),
        "w",
        &[("a/x.rs", 1, true, None)],
        &[],
    )
    .replace(r#""name":"a""#, r#""name":"gpupod-missing""#);
    let stdout = cargo_stdout(&[
        &warning,
        GPUPOD_MISSING_LINE,
        FAILURE_NOTE_LINE,
        FAILURE_NOTE_LINE,
    ]);
    assert_eq!(run(&gpupod_expected(), &stdout), Vec::<String>::new());
}

#[test]
fn every_key_must_hold_in_the_error_not_in_a_warning_beside_it() {
    // The error has the right code on the wrong line; a warning of the same crate sits on the
    // right line and carries the wanted message and label.
    let lines = [
        message_line(
            "error",
            Some("E0277"),
            "first message",
            &[("a/src/main.rs", 20, true, Some("first label"))],
            &[],
        ),
        message_line(
            "warning",
            Some("E0277"),
            "second message",
            &[("a/src/main.rs", 13, true, Some("second label"))],
            &["second child"],
        ),
    ];
    let wanted = |line: u64, messages: &[&str], labels: &[&str]| {
        expected("E0277", "a/src/main.rs", line, messages, labels)
    };
    assert!(run_a(&wanted(20, &["first message"], &["first label"]), &lines).is_empty());
    let problems = run_a(&wanted(13, &[], &[]), &lines);
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(
        problems[0].contains("does not satisfy `line: 13`"),
        "{problems:?}"
    );
    // The message of the warning, the child of the warning, the label of the warning.
    for (messages, label) in [
        (&["second message"][..], &[][..]),
        (&["second child"][..], &[][..]),
        (&[][..], &["second label"][..]),
    ] {
        let problems = run_a(&wanted(20, messages, label), &lines);
        assert_eq!(problems.len(), 2, "{messages:?} {label:?}: {problems:#?}");
    }
}

#[test]
fn file_and_line_must_sit_on_the_same_primary_span() {
    let lines = [message_line(
        "error",
        Some("E0277"),
        "m",
        &[
            ("a/one.rs", 5, true, None),
            ("a/two.rs", 9, true, None),
            ("a/one.rs", 7, false, Some("secondary")),
        ],
        &[],
    )];
    let judge_with =
        |file: &str, line: u64| run_a(&expected("E0277", file, line, &[], &[]), &lines);
    assert!(judge_with("a/one.rs", 5).is_empty());
    assert!(judge_with("a/two.rs", 9).is_empty());
    // The right file and the right line, but on different spans.
    assert_eq!(judge_with("a/one.rs", 9).len(), 2);
    assert_eq!(judge_with("a/two.rs", 5).len(), 2);
    // A secondary span does not count.
    let problems = judge_with("a/one.rs", 7);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`line: 7`"), "{problems:?}");
    let problems = judge_with("a/three.rs", 5);
    assert!(problems[0].contains("`file: a/three.rs`"), "{problems:?}");
}

#[test]
fn a_null_code_never_matches_and_a_lint_code_is_a_code() {
    let lint = [message_line(
        "error",
        Some("unused_variables"),
        "lint",
        &[("a.rs", 2, true, None)],
        &[],
    )];
    assert!(
        run_a(
            &expected("unused_variables", "a.rs", 2, &["lint"], &[]),
            &lint
        )
        .is_empty()
    );
    let none = [message_line(
        "error",
        None,
        "no code",
        &[("a.rs", 1, true, None)],
        &[],
    )];
    let problems = run_a(&expected("E0277", "a.rs", 1, &["no code"], &[]), &none);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`code: E0277`"), "{problems:?}");
}

#[test]
fn a_failing_run_without_an_error_is_a_problem_naming_cargos_output() {
    let stdout = cargo_stdout(&[ARTIFACT_LINE, FINISHED_LINE]);
    let problems = judge(
        "c",
        "c",
        &fails(gpupod_expected()),
        false,
        &stdout,
        "error: could not find `Cargo.toml`\n",
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains(
            "c: `cargo check` failed but printed no error-level compiler-message of its own"
        ),
        "{problems:?}"
    );
    assert!(
        problems[0].contains("    error: could not find `Cargo.toml`"),
        "{problems:?}"
    );
    // No output at all is the same problem.
    assert_eq!(
        judge("c", "c", &fails(gpupod_expected()), false, "", "").len(),
        1
    );
    // A truncated JSON line does not count as a diagnostic.
    let truncated = &GPUPOD_MISSING_LINE[..GPUPOD_MISSING_LINE.len() / 2];
    assert_eq!(
        judge(
            "gpupod-missing",
            "gpupod-missing",
            &fails(gpupod_expected()),
            false,
            truncated,
            ""
        )
        .len(),
        1
    );
    // Only the summary note, or only a warning, is no error either.
    let notes = cargo_stdout(&[FAILURE_NOTE_LINE]);
    assert_eq!(run(&gpupod_expected(), &notes).len(), 1);
}

#[test]
fn a_crate_that_compiles_when_it_must_fail_is_a_failure() {
    // Exit status zero is a failure even when the output happens to contain the diagnostic.
    let problems = judge(
        "c",
        "c",
        &fails(gpupod_expected()),
        true,
        &gpupod_stdout(),
        "",
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("must fail, but it compiled"),
        "{problems:?}"
    );
}

#[test]
fn a_crate_without_expected_txt_must_compile() {
    assert!(judge("c", "c", &Expectation::Compiles, true, "", "").is_empty());
    // A compiled crate may print warnings.
    let warning = cargo_stdout(&[
        &message_line("warning", Some("unused"), "w", &[], &[]),
        r#"{"reason":"build-finished","success":true}"#,
    ]);
    assert!(judge("c", "c", &Expectation::Compiles, true, &warning, "").is_empty());
    // A failure shows the rendered diagnostics (of any crate) and cargo's own text, with slashes.
    let problems = judge(
        "c",
        "c",
        &Expectation::Compiles,
        false,
        &gpupod_stdout(),
        "error: could not compile `c`\n",
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("must compile, but `cargo check` failed"),
        "{problems:?}"
    );
    assert!(problems[0].contains("GUST GPU POD"), "{problems:?}");
    assert!(
        problems[0].contains("--> gpupod-missing/src/main.rs:13:65"),
        "{problems:?}"
    );
    assert!(
        problems[0].contains("could not compile `c`"),
        "{problems:?}"
    );
}

#[test]
fn the_cargo_call_is_locked_per_crate_colorless_and_json() {
    let args = cargo_args("gpupod-missing");
    assert_eq!(args[0], "check");
    assert!(args.iter().any(|arg| arg == "--locked"), "{args:?}");
    let pair = |key: &str, value: &str| args.windows(2).any(|w| w[0] == key && w[1] == value);
    assert!(pair("--manifest-path", "tests/ui/Cargo.toml"), "{args:?}");
    assert!(pair("-p", "gpupod-missing"), "{args:?}");
    assert!(pair("--color", "never"), "{args:?}");
    assert!(pair("--message-format", "json"), "{args:?}");
    assert!(!args.iter().any(|arg| arg == "short"), "{args:?}");
}

#[test]
fn the_fixture_check_runs_under_the_dev_profile() {
    let command = case_command(std::path::Path::new("ROOT"), "gpupod-missing", &[]);
    let args: Vec<_> = command.get_args().map(ToOwned::to_owned).collect();
    let wanted: Vec<std::ffi::OsString> = cargo_args("gpupod-missing")
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(args, wanted);
    let profile: Vec<_> = command
        .get_envs()
        .filter(|(name, _)| *name == "GUST_PROFILE")
        .map(|(_, value)| value.map(ToOwned::to_owned))
        .collect();
    assert_eq!(profile, [Some(std::ffi::OsString::from("dev"))]);
}

#[test]
fn discovery_reads_each_crate_directory_and_its_expectation() {
    let root = std::env::temp_dir().join(format!("gust-ui-discovery-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    for name in ["b-fails", "a-compiles"] {
        fs::create_dir_all(root.join(name)).unwrap();
        fs::write(root.join(name).join("Cargo.toml"), "[package]\n").unwrap();
    }
    // Directories without a manifest, and plain files, are not cases.
    fs::create_dir_all(root.join("target")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::write(
        root.join("b-fails").join("expected.txt"),
        "code: E0277\nfile: b-fails/src/main.rs\nline: 3\nmessage: message\n",
    )
    .unwrap();
    let cases = discover_cases(&root).unwrap();
    let _ = fs::remove_dir_all(&root);
    assert_eq!(
        cases
            .iter()
            .map(|case| (case.name.as_str(), case.expectation.clone()))
            .collect::<Vec<_>>(),
        [
            ("a-compiles", Expectation::Compiles),
            (
                "b-fails",
                fails(expected(
                    "E0277",
                    "b-fails/src/main.rs",
                    3,
                    &["message"],
                    &[]
                ))
            )
        ]
    );
}

#[test]
fn a_fixture_directory_with_no_crate_is_an_error_not_a_pass() {
    let root = std::env::temp_dir().join(format!("gust-ui-empty-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let error = discover_cases(&root).unwrap_err();
    let _ = fs::remove_dir_all(&root);
    assert!(error.contains("no fixture crate"), "{error}");
    assert!(
        discover_cases(&root).is_err(),
        "a missing directory is an error too"
    );
}

#[test]
fn an_empty_or_old_format_expected_txt_is_an_error_in_discovery() {
    let root = std::env::temp_dir().join(format!("gust-ui-blank-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("c")).unwrap();
    fs::write(root.join("c").join("Cargo.toml"), "[package]\n").unwrap();
    fs::write(root.join("c").join("expected.txt"), "# nothing\n").unwrap();
    let error = discover_cases(&root).unwrap_err();
    assert!(error.contains("lists no key"), "{error}");
    // The pre-JSON format (one substring per line) is refused, not silently half-read.
    fs::write(root.join("c").join("expected.txt"), "error[E0277]\n").unwrap();
    let error = discover_cases(&root).unwrap_err();
    let _ = fs::remove_dir_all(&root);
    assert!(
        error.contains("tests/ui/c") && error.contains("`key: value`"),
        "{error}"
    );
}

#[test]
fn the_repository_fixtures_are_a_workspace_of_their_own_and_every_crate_is_listed() {
    let root = workspace_root().unwrap();
    let ui = root.join(UI_DIR);
    let manifest = fs::read_to_string(ui.join("Cargo.toml")).unwrap();
    // A root of its own, so it can sit inside the repository without joining its workspace.
    assert!(
        manifest.lines().any(|line| line.trim() == "[workspace]"),
        "{manifest}"
    );
    let cases = discover_cases(&ui).unwrap();
    for case in &cases {
        assert!(
            manifest.contains(&format!("\"{}\"", case.name)),
            "{} is not a member of tests/ui/Cargo.toml",
            case.name
        );
    }
    // The root workspace neither lists nor includes the fixtures: no member under tests/.
    let root_manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(!root_manifest.contains("tests/ui"), "{root_manifest}");
    // The committed cases: two that must fail with the exact diagnostic, one that compiles.
    let find = |name: &str| cases.iter().find(|case| case.name == name).unwrap();
    assert_eq!(find("gpupod-missing").expectation, fails(gpupod_expected()));
    assert_eq!(
        find("kernel-missing").expectation,
        fails(expected(
            "E0277",
            "kernel-missing/src/main.rs",
            10,
            &["`u32` is not a GUST kernel handle"],
            &["pass the zero-sized handle that `#[gpu]` generates for a `#[kernel]` function"],
        ))
    );
    assert_eq!(find("gpupod-present").expectation, Expectation::Compiles);
    // The two const layout fixtures: one E0080 reported at the `#[gust::gpu]` line (line 17), pinned by message.
    assert_eq!(
        find("layout-outside-field").expectation,
        fails(expected(
            "E0080",
            "layout-outside-field/src/main.rs",
            17,
            &[
                "evaluation panicked: GUST StorageV1: field `w` of `Holder` is not at its padding-free offset"
            ],
            &[],
        ))
    );
    assert_eq!(
        find("layout-outside-element").expectation,
        fails(expected(
            "E0080",
            "layout-outside-element/src/main.rs",
            17,
            &[
                "evaluation panicked: GUST StorageV1: element `super::Short` layout does not match its Rust representation (parameter `out` of kernel `run`)"
            ],
            &[],
        ))
    );
    // Each fixture crate has a lockfile through the shared one at the fixture root.
    assert!(ui.join("Cargo.lock").is_file());
}

/// The `compiler-message` line of `cargo check --message-format json -p layout-outside-field`
/// (probe/d2-fixture.log), trimmed from a real run on Windows. A `const` panic that a macro generated
/// is reported at core's `panic.rs`, not in the user's file: the primary span is there and the user's
/// `#[gust::gpu]` line is only the call site of its expansion chain. Cut: `package_id`,
/// `manifest_path`, `explanation`, `def_site_span`s, byte offsets, columns and `text`. Changed: the
/// machine path of `panic.rs` is replaced by the `/rustc/<hash>` form that rustc prints with
/// remapped paths. Kept: `file_name` of the call sites is relative to the fixture workspace with
/// backslashes, the label is on the primary span, the chain has two call sites (`panic_2021!`, then
/// `assert!`) and the message names the field.
const CONST_PANIC_LINE: &str = r#"{"reason":"compiler-message","target":{"kind":["bin"],"crate_types":["bin"],"name":"layout-outside-field"},"message":{"rendered":"error[E0080]: evaluation panicked: GUST StorageV1: field `w` of `Holder` is not at its padding-free offset\n --> layout-outside-field/src/main.rs:17:1\n","$message_type":"diagnostic","children":[],"code":{"code":"E0080","explanation":null},"level":"error","message":"evaluation panicked: GUST StorageV1: field `w` of `Holder` is not at its padding-free offset","spans":[{"file_name":"/rustc/0000000000000000000000000000000000000000/library/core/src/panic.rs","is_primary":true,"label":"evaluation of `layouts::_` failed here","line_start":62,"line_end":62,"expansion":{"span":{"file_name":"layout-outside-field\\src\\main.rs","is_primary":false,"label":null,"line_start":17,"line_end":17,"expansion":{"span":{"file_name":"layout-outside-field\\src\\main.rs","is_primary":false,"label":null,"line_start":17,"line_end":17,"expansion":null},"macro_decl_name":"::core::assert!"}},"macro_decl_name":"$crate::panic::panic_2021!"}}]}}"#;

const CORE_PANIC: &str =
    "/rustc/0000000000000000000000000000000000000000/library/core/src/panic.rs";

fn const_panic_expected() -> Expected {
    expected(
        "E0080",
        "layout-outside-field/src/main.rs",
        17,
        &["field `w` of `Holder` is not at its padding-free offset"],
        &["evaluation of `layouts::_` failed here"],
    )
}

fn run_const_panic(wanted: &Expected, line: &str) -> Vec<String> {
    judge(
        "layout-outside-field",
        "layout-outside-field",
        &fails(wanted.clone()),
        false,
        &cargo_stdout(&[line]),
        "",
    )
}

/// A synthetic error of the case `a`: one primary span at `(file, line)` with the call sites `sites`
/// (innermost first) as its expansion chain.
fn expansion_line(primary: (&str, u32), sites: &[(&str, u32)]) -> String {
    let mut chain = "null".to_owned();
    for (file, line) in sites.iter().rev() {
        chain = format!(
            r#"{{"span":{{"file_name":"{file}","is_primary":false,"label":null,"line_start":{line},"line_end":{line},"expansion":{chain}}},"macro_decl_name":"m!"}}"#
        );
    }
    format!(
        r#"{{"reason":"compiler-message","target":{{"kind":["bin"],"name":"a"}},"message":{{"rendered":"error: x\n","children":[],"code":{{"code":"E0080","explanation":null}},"level":"error","message":"x","spans":[{{"file_name":"{}","is_primary":true,"label":"here","line_start":{},"line_end":{},"expansion":{chain}}}]}}}}"#,
        primary.0, primary.1, primary.1
    )
}

#[test]
fn a_primary_span_in_core_is_reported_at_the_expansion_call_site_in_the_case() {
    let diagnostics = parse_diagnostics(CONST_PANIC_LINE);
    let span = &diagnostics[0].spans[0];
    assert_eq!(span.file, CORE_PANIC);
    assert_eq!(span.line, Some(62));
    let sites: Vec<(&str, Option<u64>)> = span
        .call_sites
        .iter()
        .map(|site| (site.file.as_str(), site.line))
        .collect();
    assert_eq!(
        sites,
        [
            ("layout-outside-field/src/main.rs", Some(17)),
            ("layout-outside-field/src/main.rs", Some(17))
        ]
    );
    // `file`, `line` and the message hold; the label is still read from the primary span itself.
    let wanted = const_panic_expected();
    assert_eq!(
        run_const_panic(&wanted, CONST_PANIC_LINE),
        Vec::<String>::new()
    );
    // A wrong line is named, and the error is described where it is reported (no machine path).
    let mut wrong = wanted.clone();
    wrong.line = 62;
    let problems = run_const_panic(&wrong, CONST_PANIC_LINE);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`line: 62`"), "{problems:?}");
    assert!(
        problems[0].contains("at layout-outside-field/src/main.rs:17)"),
        "{problems:?}"
    );
    // The file of core's panic.rs is not the case's: it never satisfies `file`.
    let mut core = wanted.clone();
    core.file = CORE_PANIC.to_owned();
    assert!(
        run_const_panic(&core, CONST_PANIC_LINE)[0].contains("`file: "),
        "an absolute path in expected.txt must not match"
    );
    // A label that is on no primary span is still a gap.
    let mut label = wanted;
    label.labels = vec!["not the label".to_owned()];
    assert!(run_const_panic(&label, CONST_PANIC_LINE)[0].contains("`label: not the label`"));
}

#[test]
fn a_chain_that_never_reaches_the_case_does_not_satisfy_file() {
    let outside = CONST_PANIC_LINE.replace(
        r"layout-outside-field\\src\\main.rs",
        "/rustc/0000000000000000000000000000000000000000/library/core/src/macros/mod.rs",
    );
    assert_ne!(outside, CONST_PANIC_LINE);
    let problems = run_const_panic(&const_panic_expected(), &outside);
    assert!(
        problems[0].contains("`file: layout-outside-field/src/main.rs`"),
        "{problems:?}"
    );
    // Another case's file in the chain is not this case's file either.
    let other = CONST_PANIC_LINE.replace(
        r"layout-outside-field\\src\\main.rs",
        r"other-case\\src\\main.rs",
    );
    let problems = run_const_panic(&const_panic_expected(), &other);
    assert!(
        problems[0].contains("`file: layout-outside-field/src/main.rs`"),
        "{problems:?}"
    );
    // No expansion at all: the primary span is all there is.
    let line = expansion_line((CORE_PANIC, 62), &[]);
    let wanted = expected("E0080", "a/src/main.rs", 17, &[], &[]);
    let problems = run_a(&wanted, &[line]);
    assert!(
        problems[0].contains("`file: a/src/main.rs`"),
        "{problems:?}"
    );
}

#[test]
fn a_primary_span_in_the_case_wins_over_its_expansion_chain() {
    let line = expansion_line(("a/src/main.rs", 10), &[("a/src/main.rs", 99)]);
    // Reported where it is: line 10, not the call site's 99.
    let at_ten = expected("E0080", "a/src/main.rs", 10, &[], &["here"]);
    assert_eq!(
        run_a(&at_ten, std::slice::from_ref(&line)),
        Vec::<String>::new()
    );
    let at_call_site = expected("E0080", "a/src/main.rs", 99, &[], &[]);
    let problems = run_a(&at_call_site, std::slice::from_ref(&line));
    assert!(problems[0].contains("`line: 99`"), "{problems:?}");
}

#[test]
fn the_outermost_call_site_in_the_case_is_the_one_reported() {
    // Innermost first: line 5, then 7 in the case, with a call site in core between and one after.
    let line = expansion_line(
        (CORE_PANIC, 62),
        &[
            ("a/src/main.rs", 5),
            (CORE_PANIC, 1),
            ("a/src/main.rs", 7),
            (CORE_PANIC, 2),
        ],
    );
    let seven = expected("E0080", "a/src/main.rs", 7, &[], &["here"]);
    assert_eq!(
        run_a(&seven, std::slice::from_ref(&line)),
        Vec::<String>::new()
    );
    let five = expected("E0080", "a/src/main.rs", 5, &[], &[]);
    assert!(run_a(&five, &[line])[0].contains("`line: 5`"));
}

/// The `compiler-message` line of `cargo check --message-format json -p kernel-cpu-bad-value`
/// (m3.c2), trimmed from a real run on Windows. A macro's `compile_error!` has
/// `"code": null`, so the message text is all that names it, and its span is the `cpu = "off"` option
/// in the user's file. Cut: `package_id`, `manifest_path`, the target's paths, the rendered snippet and
/// the `text` array. Kept: the backslash `file_name`, the null code and the macro's own message.
const MACRO_ERROR_LINE: &str = r#"{"reason":"compiler-message","target":{"kind":["bin"],"crate_types":["bin"],"name":"kernel-cpu-bad-value"},"message":{"rendered":"error: [GUST-E0709] unknown `cpu` value; the only value is `\"unsupported\"`\n --> kernel-cpu-bad-value\\src\\main.rs:7:14\n","$message_type":"diagnostic","children":[],"level":"error","message":"[GUST-E0709] unknown `cpu` value; the only value is `\"unsupported\"`","spans":[{"byte_end":310,"byte_start":299,"column_end":25,"column_start":14,"expansion":null,"file_name":"kernel-cpu-bad-value\\src\\main.rs","is_primary":true,"label":null,"line_end":7,"line_start":7,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}],"code":null}}"#;

fn macro_expected() -> Expected {
    expected(
        "none",
        "kernel-cpu-bad-value/src/main.rs",
        7,
        &["[GUST-E0709] unknown `cpu` value; the only value is `\"unsupported\"`"],
        &[],
    )
}

fn run_macro(wanted: &Expected, line: &str) -> Vec<String> {
    judge(
        "kernel-cpu-bad-value",
        "kernel-cpu-bad-value",
        &fails(wanted.clone()),
        false,
        &cargo_stdout(&[line]),
        "",
    )
}

#[test]
fn code_none_names_an_error_without_a_rustc_code_by_its_message() {
    let diagnostics = parse_diagnostics(MACRO_ERROR_LINE);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, None);
    assert_eq!(
        run_macro(&macro_expected(), MACRO_ERROR_LINE),
        Vec::<String>::new()
    );
    // The message is a substring of the macro's text, so a shorter pin matches too.
    let mut shorter = macro_expected();
    shorter.messages = vec!["[GUST-E0709]".to_owned()];
    assert!(run_macro(&shorter, MACRO_ERROR_LINE).is_empty());
}

#[test]
fn code_none_still_pins_the_message_the_file_and_the_line() {
    // Another macro code, or other text, is a message gap: the message is what tells code-less errors apart.
    for wrong in ["[GUST-E0710] duplicate cpu option", "unknown `cpu` value!"] {
        let mut wanted = macro_expected();
        wanted.messages = vec![wrong.to_owned()];
        let problems = run_macro(&wanted, MACRO_ERROR_LINE);
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(
            problems[0].contains(&format!("`message: {wrong}`")),
            "{problems:?}"
        );
        assert!(
            problems[0].contains("at kernel-cpu-bad-value/src/main.rs:7)"),
            "{problems:?}"
        );
    }
    let mut line = macro_expected();
    line.line = 8;
    assert!(run_macro(&line, MACRO_ERROR_LINE)[0].contains("`line: 8`"));
    let mut file = macro_expected();
    file.file = "other/src/main.rs".to_owned();
    assert!(run_macro(&file, MACRO_ERROR_LINE)[0].contains("`file: other/src/main.rs`"));
}

#[test]
fn code_none_rejects_an_error_that_has_a_code_and_a_code_rejects_a_codeless_one() {
    let coded = [message_line(
        "error",
        Some("E0277"),
        "same text",
        &[("a.rs", 3, true, None)],
        &[],
    )];
    let none = expected("none", "a.rs", 3, &["same text"], &[]);
    let problems = run_a(&none, &coded);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`code: none`"), "{problems:?}");
    assert!(
        problems[0].contains("error[E0277] same text"),
        "{problems:?}"
    );
    // And the other way round: a code-less macro error does not satisfy `code: E0277`.
    let mut wanted = macro_expected();
    wanted.code = "E0277".to_owned();
    let problems = run_macro(&wanted, MACRO_ERROR_LINE);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`code: E0277`"), "{problems:?}");
}

#[test]
fn code_none_needs_a_message_and_is_spelled_in_lower_case() {
    assert_eq!(
        parse_expected("code: none\nfile: a.rs\nline: 2\nmessage: [GUST-E0709] x\n"),
        Ok(expected("none", "a.rs", 2, &["[GUST-E0709] x"], &[]))
    );
    // Without a message any error without a code would do; a label does not stand in for it.
    for text in [
        "code: none\nfile: a.rs\nline: 2\n",
        "code: none\nfile: a.rs\nline: 2\nlabel: x\n",
    ] {
        let error = parse_expected(text).unwrap_err();
        assert!(
            error.contains("`code: none` but no `message:` line"),
            "{error}"
        );
    }
    // The other required keys are still reported first; a real code needs no message.
    assert!(
        parse_expected("code: none\nmessage: x\n")
            .unwrap_err()
            .contains("no `file:` line")
    );
    assert!(parse_expected("code: E0277\nfile: a.rs\nline: 2\n").is_ok());
    // `None` is an ordinary code, which no diagnostic has: it never matches a code-less error.
    let wanted = parse_expected("code: None\nfile: a.rs\nline: 2\nmessage: x\n").unwrap();
    assert_eq!(wanted.code, "None");
    let codeless = [message_line(
        "error",
        None,
        "x",
        &[("a.rs", 2, true, None)],
        &[],
    )];
    assert!(run_a(&wanted, &codeless)[0].contains("`code: None`"));
}

#[test]
fn the_cpu_option_fixtures_pin_the_macro_messages_of_e0709_and_e0710() {
    let ui = workspace_root().unwrap().join(UI_DIR);
    let cases = discover_cases(&ui).unwrap();
    let find = |name: &str| cases.iter().find(|case| case.name == name).unwrap();
    assert_eq!(
        find("kernel-cpu-bad-value").expectation,
        fails(macro_expected())
    );
    assert_eq!(
        find("kernel-cpu-duplicate").expectation,
        fails(expected(
            "none",
            "kernel-cpu-duplicate/src/main.rs",
            6,
            &["[GUST-E0710] duplicate cpu option"],
            &[],
        ))
    );
}

/// A synthetic `compiler-message` of the cargo target `target` of the package whose directory is `consumer`.
fn sibling_line(target: &str, kind: &str, line: u32) -> String {
    message_line(
        "error",
        Some("E0277"),
        "`CpuDevice` cannot run the kernel `add`",
        &[(
            "consumer/src/lib.rs",
            line,
            true,
            Some("this device has no executable form of the kernel"),
        )],
        &[],
    )
    .replace(
        r#""target":{"kind":["bin"],"name":"a"}"#,
        &format!(r#""target":{{"kind":["{kind}"],"name":"{target}"}}"#),
    )
}

#[test]
fn a_diagnostic_on_a_sibling_target_is_selected_by_the_target_name() {
    // One package directory (`consumer`), two cargo targets: the library `consumer` and the binary `prod`.
    let wanted = expected(
        "E0277",
        "consumer/src/lib.rs",
        7,
        &["`CpuDevice` cannot run the kernel `add`"],
        &["no executable form"],
    );
    let on_bin = [sibling_line("prod", "bin", 7)];
    let stdout = cargo_stdout(&[&on_bin[0]]);
    let judged = |case: &str, target: &str, stdout: &str| {
        judge(case, target, &fails(wanted.clone()), false, stdout, "")
    };
    // Asking for the binary finds its error.
    assert_eq!(
        judged("consumer", "prod", &stdout),
        Vec::<String>::new(),
        "the error of target `prod` satisfies an expectation that names `prod`"
    );
    // The default of `check-ui`, target = case, sees no error of its own on a package with two targets.
    let problems = judged("consumer", "consumer", &stdout);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("printed no error-level compiler-message of its own"),
        "{problems:?}"
    );
    // A sibling target's error is neither this target's error nor a second one: the library's error does not
    // rescue the binary's wrong line, and does not make a correct binary error count twice.
    let on_lib = sibling_line("consumer", "lib", 7);
    let both = cargo_stdout(&[&on_lib, &on_bin[0]]);
    assert_eq!(judged("consumer", "prod", &both), Vec::<String>::new());
    assert_eq!(judged("consumer", "consumer", &both), Vec::<String>::new());
    let wrong_line = cargo_stdout(&[&on_lib, &sibling_line("prod", "bin", 99)]);
    let problems = judged("consumer", "prod", &wrong_line);
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(problems[0].contains("`line: 7`"), "{problems:?}");
}

// ---- build profiles (slice m4.b2): per-fixture environment and failures in `gust`'s build script ----

/// The stderr of `cargo check -p profile-unknown` under `GUST_PROFILE=bogus`, trimmed from a real run (the machine
/// paths are replaced by `<path>`): cargo's own report of the failed build script, then the script's stderr.
const BUILD_SCRIPT_STDERR: &str = "error: failed to run custom build command for `gust v0.1.0 (<path>)`\n\nCaused by:\n  process didn't exit successfully: `<path>\\build-script-build` (exit code: 1)\n  --- stderr\n  gust: unknown GUST_PROFILE `bogus`; use dev, native or web\n";

fn build_fails(texts: &[&str]) -> Expectation {
    Expectation::BuildFails(texts.iter().map(|text| (*text).to_owned()).collect())
}

fn judge_script(texts: &[&str], success: bool, stdout: &str, stderr: &str) -> Vec<String> {
    judge(
        "profile-unknown",
        "profile-unknown",
        &build_fails(texts),
        success,
        stdout,
        stderr,
    )
}

#[test]
fn expected_txt_with_build_error_lines_is_a_build_script_expectation() {
    assert_eq!(
        parse_expectation(
            "# a comment\nbuild-error: gust: unknown GUST_PROFILE `x`; use dev, native or web\n\nbuild-error: second: with a colon\n"
        ),
        Ok(build_fails(&[
            "gust: unknown GUST_PROFILE `x`; use dev, native or web",
            "second: with a colon"
        ]))
    );
    // CRLF and a byte order mark do not matter.
    assert_eq!(
        parse_expectation("\u{feff}build-error: a\r\n"),
        Ok(build_fails(&["a"]))
    );
    // Without it the file is the rustc keys, as before.
    assert_eq!(
        parse_expectation("code: E0277\nfile: a.rs\nline: 2\n"),
        Ok(Expectation::Fails(expected("E0277", "a.rs", 2, &[], &[])))
    );
}

#[test]
fn a_build_error_file_cannot_also_key_a_rustc_error_and_needs_a_value() {
    let rejected = |text: &str| parse_expectation(text).unwrap_err();
    assert!(rejected("build-error: a\ncode: E0080\n").contains("mixes `build-error:`"));
    assert!(rejected("build-error: a\nfile: a.rs\nline: 1\n").contains("mixes"));
    assert!(rejected("build-error:   \n").contains("`build-error` has no value"));
    assert!(rejected("build-error:\nbuild-error: a\n").contains("no value"));
    // A backslash could never match: cargo's stderr is compared with backslashes read as slashes.
    let error = rejected("build-error: path a\\b\n");
    assert!(error.contains("contains a backslash"), "{error}");
    assert!(error.contains("`build-error: path a\\b`"), "{error}");
    assert_eq!(
        parse_expectation("build-error: path a/b\n"),
        Ok(build_fails(&["path a/b"]))
    );
    // An empty file keeps its error, and `build-error` is no key of `parse_expected` (check-dual's files).
    assert!(rejected("").contains("lists no key"));
    assert!(
        parse_expected("build-error: a\n")
            .unwrap_err()
            .contains("unknown key `build-error`")
    );
}

#[test]
fn env_txt_lists_variables_in_order_and_keeps_values_as_written() {
    assert_eq!(
        parse_env("# why\n\nGUST_PROFILE=native\r\nOTHER_1=a=b c\n"),
        Ok(vec![
            ("GUST_PROFILE".to_owned(), "native".to_owned()),
            ("OTHER_1".to_owned(), "a=b c".to_owned()),
        ])
    );
    // An empty value is a value (the variable is set to the empty string); spaces after `=` are part of it.
    assert_eq!(
        parse_env("GUST_PROFILE=\n"),
        Ok(vec![("GUST_PROFILE".to_owned(), String::new())])
    );
    assert_eq!(
        parse_env("GUST_PROFILE= native\n"),
        Ok(vec![("GUST_PROFILE".to_owned(), " native".to_owned())])
    );
    // A byte order mark and spaces around the name do not matter.
    assert_eq!(
        parse_env("\u{feff} A =1\n"),
        Ok(vec![("A".to_owned(), "1".to_owned())])
    );
}

#[test]
fn env_txt_rejects_what_would_silently_do_nothing() {
    let rejected = |text: &str| parse_env(text).unwrap_err();
    for empty in ["", "\n\n", "# only a comment\n"] {
        assert!(rejected(empty).contains("sets no variable"), "{empty:?}");
    }
    assert!(rejected("GUST_PROFILE native\n").contains("`NAME=value`"));
    assert!(rejected("=native\n").contains("is not a variable name"));
    assert!(rejected("GUST PROFILE=native\n").contains("`GUST PROFILE` is not a variable name"));
    assert!(rejected("GUST-PROFILE=native\n").contains("is not a variable name"));
    assert!(
        rejected("GUST_PROFILE=native\nGUST_PROFILE=web\n")
            .contains("`GUST_PROFILE` appears twice")
    );
}

#[test]
fn a_fixture_sets_its_variables_after_the_workspace_profile() {
    let env = [
        ("GUST_PROFILE".to_owned(), "native".to_owned()),
        ("EXTRA".to_owned(), String::new()),
    ];
    let command = case_command(std::path::Path::new("ROOT"), "gpupod-missing", &env);
    let mut values = std::collections::BTreeMap::new();
    for (name, value) in command.get_envs() {
        values.insert(name.to_owned(), value.map(ToOwned::to_owned));
    }
    // The fixture's value wins over `dev`; an empty value stays set (it is not a removal).
    assert_eq!(
        values.get(std::ffi::OsStr::new("GUST_PROFILE")),
        Some(&Some(std::ffi::OsString::from("native")))
    );
    assert_eq!(
        values.get(std::ffi::OsStr::new("EXTRA")),
        Some(&Some(std::ffi::OsString::new()))
    );
    // Without an `env.txt` the workspace profile stays (see the test above).
    let plain = case_command(std::path::Path::new("ROOT"), "gpupod-missing", &[]);
    assert!(
        plain
            .get_envs()
            .any(|(name, value)| name == "GUST_PROFILE" && value == Some("dev".as_ref()))
    );
}

#[test]
fn discovery_reads_env_txt_beside_expected_txt() {
    let root = std::env::temp_dir().join(format!("gust-ui-env-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("a")).unwrap();
    fs::create_dir_all(root.join("b")).unwrap();
    for name in ["a", "b"] {
        fs::write(root.join(name).join("Cargo.toml"), "[package]\n").unwrap();
    }
    fs::write(root.join("a").join("env.txt"), "GUST_PROFILE=native\n").unwrap();
    let cases = discover_cases(&root).unwrap();
    assert_eq!(
        cases[0].env,
        vec![("GUST_PROFILE".to_owned(), "native".to_owned())]
    );
    assert!(cases[1].env.is_empty());
    assert_eq!(cases[0].expectation, Expectation::Compiles);
    // A broken env.txt is an error that names the fixture, not a case without variables.
    fs::write(root.join("b").join("env.txt"), "# nothing\n").unwrap();
    let error = discover_cases(&root).unwrap_err();
    let _ = fs::remove_dir_all(&root);
    assert!(
        error.contains("tests/ui/b") && error.contains("sets no variable"),
        "{error}"
    );
}

#[test]
fn a_build_script_failure_with_the_text_passes() {
    let texts = ["gust: unknown GUST_PROFILE `bogus`; use dev, native or web"];
    assert!(judge_script(&texts, false, "", BUILD_SCRIPT_STDERR).is_empty());
    // The text may be a part of a line.
    assert!(judge_script(&["unknown GUST_PROFILE"], false, "", BUILD_SCRIPT_STDERR).is_empty());
    // Several texts must all be there.
    assert!(
        judge_script(
            &["unknown GUST_PROFILE", "dev, native or web"],
            false,
            "",
            BUILD_SCRIPT_STDERR
        )
        .is_empty()
    );
}

#[test]
fn a_build_script_failure_without_the_text_or_that_compiled_is_a_problem() {
    // The text is missing, or differs by one character.
    for wrong in [
        "gust: unknown GUST_PROFILE `bogus`; use dev, native, web",
        "gust: unknown GUST_PROFILE `other`; use dev, native or web",
        "refuses feature slang-runtime",
    ] {
        let problems = judge_script(&[wrong], false, "", BUILD_SCRIPT_STDERR);
        assert_eq!(problems.len(), 2, "{wrong}: {problems:?}");
        assert!(
            problems[0].contains(&format!("does not contain `build-error: {wrong}`")),
            "{problems:?}"
        );
        assert!(problems[1].contains("cargo's output was:"), "{problems:?}");
        assert!(problems[1].contains("--- stderr"), "{problems:?}");
    }
    // One text found does not excuse another that is missing.
    let problems = judge_script(
        &["unknown GUST_PROFILE", "refuses feature"],
        false,
        "",
        BUILD_SCRIPT_STDERR,
    );
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("`build-error: refuses feature`"),
        "{problems:?}"
    );
    // A crate that compiles is a failure of the expectation, whatever the output says.
    let problems = judge_script(&["unknown GUST_PROFILE"], true, "", BUILD_SCRIPT_STDERR);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("must fail in a build script, but it compiled"),
        "{problems:?}"
    );
}

#[test]
fn a_failure_elsewhere_than_in_the_build_script_of_gust_never_satisfies_build_error() {
    let text = ["unknown GUST_PROFILE"];
    // Another package's build script, with the same words in its stderr.
    let other = BUILD_SCRIPT_STDERR.replace("`gust v0.1.0", "`other v0.1.0");
    let problems = judge_script(&text, false, "", &other);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("not in the build script of `gust`"),
        "{problems:?}"
    );
    // The words in a plain failure of cargo (no build script at all).
    let plain = "error: could not compile `x`: gust: unknown GUST_PROFILE `bogus`\n";
    assert_eq!(judge_script(&text, false, "", plain).len(), 2);
    // A rustc error of the crate itself means the build script did not fail first.
    let problems = judge_script(
        &text,
        false,
        &gpupod_stdout().replace("gpupod-missing", "profile-unknown"),
        BUILD_SCRIPT_STDERR,
    );
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("but rustc reported 1 error(s) of the crate itself"),
        "{problems:?}"
    );
    assert!(
        problems[0].contains("is not a GUST GPU POD type"),
        "{problems:?}"
    );
}

#[test]
fn the_profile_fixtures_pin_the_exact_texts_and_variables() {
    let ui = workspace_root().unwrap().join(UI_DIR);
    let cases = discover_cases(&ui).unwrap();
    let find = |name: &str| cases.iter().find(|case| case.name == name).unwrap();
    let variable = |value: &str| vec![("GUST_PROFILE".to_owned(), value.to_owned())];
    // Under `native`, the kernel without an artifact fails at its own line (`fn fill`, line 9 of the file).
    let native = find("profile-native-no-artifact");
    assert_eq!(native.env, variable("native"));
    assert_eq!(
        native.expectation,
        fails(expected(
            "E0080",
            "profile-native-no-artifact/src/main.rs",
            9,
            &[
                "evaluation panicked: gust: a strict profile (GUST_PROFILE native or web) needs a precompiled artifact for this kernel; run `cargo gust build`, or build with GUST_PROFILE=dev"
            ],
            &[],
        ))
    );
    // A strict profile with the artifact pair but no manifest.json fails at the kernel too, with the manifest text.
    let no_manifest = find("profile-native-no-manifest");
    assert_eq!(no_manifest.env, variable("native"));
    assert_eq!(
        no_manifest.expectation,
        fails(expected(
            "E0080",
            "profile-native-no-manifest/src/main.rs",
            9,
            &[
                "evaluation panicked: gust: a strict profile (GUST_PROFILE native or web) needs gust-artifacts/manifest.json in the kernel crate; run `cargo gust init` and `cargo gust build`, or build with GUST_PROFILE=dev"
            ],
            &[],
        ))
    );
    // With the pair and the manifest present the same strict build compiles.
    let complete = find("profile-native-artifact");
    assert_eq!(complete.env, variable("native"));
    assert_eq!(complete.expectation, Expectation::Compiles);
    // Under `dev` the very same file compiles.
    let dev = find("profile-dev-no-artifact");
    assert_eq!(dev.env, variable("dev"));
    assert_eq!(dev.expectation, Expectation::Compiles);
    let manifest =
        fs::read_to_string(ui.join("profile-dev-no-artifact").join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("path = \"../profile-native-no-artifact/src/main.rs\""),
        "{manifest}"
    );
    let source = fs::read_to_string(ui.join("profile-native-no-artifact/src/main.rs")).unwrap();
    assert_eq!(
        source.replace("\r\n", "\n").lines().nth(8),
        Some("    fn fill(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {")
    );
    // An unknown value, and an empty one, are refused by the build script with its exact text.
    assert_eq!(find("profile-unknown").env, variable("bogus"));
    assert_eq!(
        find("profile-unknown").expectation,
        build_fails(&["gust: unknown GUST_PROFILE `bogus`; use dev, native, web or cpu"])
    );
    assert_eq!(find("profile-empty").env, variable(""));
    assert_eq!(
        find("profile-empty").expectation,
        build_fails(&[
            "gust: GUST_PROFILE is set but empty; use dev, native, web or cpu (or unset it)",
        ])
    );
    // A strict profile with the feature on: the manifest turns it on, the build script refuses.
    assert_eq!(find("profile-native-bridge").env, variable("native"));
    assert_eq!(
        find("profile-native-bridge").expectation,
        build_fails(&[
            "gust: profile native refuses feature slang-runtime; a strict profile builds no run-time Slang bridge. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust` shows who enables it), or build with GUST_PROFILE=dev."
        ])
    );
    let manifest = fs::read_to_string(ui.join("profile-native-bridge").join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("features = [\"slang-runtime\"]"),
        "{manifest}"
    );
    // The map functions build under the cpu profile, where nothing names a GPU API (m7.a1); so do the two fixtures
    // that pin the argument-type errors, which would otherwise add E0433 for the missing `gust-wgpu`.
    assert_eq!(find("map-ok").env, variable("cpu"));
    assert_eq!(find("map-ok").expectation, Expectation::Compiles);
    assert_eq!(find("map-bool-for-u32").env, variable("cpu"));
    assert_eq!(find("map-u32-for-bool").env, variable("cpu"));
    // `par_for!` runs on the twin under the cpu profile too (m7.p), so its fixtures need no `gust-wgpu` either.
    for name in [
        "par-for-binding",
        "par-for-body-error",
        "par-for-bool-buffer",
        "par-for-form",
        "par-for-ok",
        "par-for-reserved-name",
        "par-for-three-wide",
    ] {
        assert_eq!(find(name).env, variable("cpu"), "{name}");
    }
    assert_eq!(find("par-for-ok").expectation, Expectation::Compiles);
    // No other fixture has variables of its own.
    let with_env: Vec<&str> = cases
        .iter()
        .filter(|case| !case.env.is_empty())
        .map(|case| case.name.as_str())
        .collect();
    assert_eq!(
        with_env,
        [
            "map-bool-for-u32",
            "map-ok",
            "map-u32-for-bool",
            "par-for-binding",
            "par-for-body-error",
            "par-for-bool-buffer",
            "par-for-form",
            "par-for-ok",
            "par-for-reserved-name",
            "par-for-three-wide",
            "profile-dev-no-artifact",
            "profile-empty",
            "profile-native-artifact",
            "profile-native-bridge",
            "profile-native-no-artifact",
            "profile-native-no-manifest",
            "profile-unknown",
            "typed-call-cpu-target"
        ]
    );
}

/// The fixture crates under tests/ui, by name. `discover_cases` takes whatever directories exist, so
/// deleting a fixture (with its `members` entry) would leave `check-ui` green; this list is the pin.
/// Adding or removing a fixture changes it deliberately, in the same change as the fixture.
const PINNED_FIXTURES: [&str; 83] = [
    "barrier-in-helper",
    "barrier-in-if",
    "barrier-in-nested-for",
    "barrier-in-while",
    "barrier-kernel-no-cpu-strategy",
    "barrier-ok",
    "break-in-barrier-loop",
    "continue-in-barrier-loop",
    "gpupod-missing",
    "gpupod-present",
    "kernel-cpu-bad-value",
    "kernel-cpu-duplicate",
    "kernel-cpu-lockstep",
    "kernel-missing",
    "labelled-break-in-barrier-kernel",
    "layout-outside-element",
    "layout-outside-field",
    "literal-let-ok",
    "literal-let-rejected",
    "map-bool-for-u32",
    "map-name-collision",
    "map-nested",
    "map-ok",
    "map-signature",
    "map-three-wide",
    "map-u32-for-bool",
    "par-for-binding",
    "par-for-body-error",
    "par-for-bool-buffer",
    "par-for-form",
    "par-for-ok",
    "par-for-reserved-name",
    "par-for-three-wide",
    "profile-dev-no-artifact",
    "profile-empty",
    "profile-native-artifact",
    "profile-native-bridge",
    "profile-native-no-artifact",
    "profile-native-no-manifest",
    "profile-unknown",
    "question-mark-in-barrier-kernel",
    "reserved-gust-prefix",
    "return-in-barrier-kernel",
    "shape-cpu-pin",
    "shape-grid-ok",
    "shape-not-a-name",
    "shape-pixel-atomic",
    "shape-pixel-barrier",
    "shape-pixel-conditional-write",
    "shape-pixel-mutable-invocation",
    "shape-pixel-no-output",
    "shape-pixel-ok",
    "shape-pixel-other-index",
    "shape-pixel-output-compound",
    "shape-pixel-output-read",
    "shape-pixel-return",
    "shape-pixel-second-alias",
    "shape-pixel-storage-texture",
    "shape-pixel-struct-output",
    "shape-pixel-workgroup-size",
    "shape-twice",
    "shape-unknown",
    "texture-as-value",
    "texture-bad-format",
    "texture-method",
    "texture-ok",
    "texture-sample-argument",
    "typed-call-buffer-for-sampler",
    "typed-call-buffer-for-texture",
    "typed-call-buffer-for-uniform",
    "typed-call-cpu-target",
    "typed-call-extent-for-count",
    "typed-call-kind-mismatch",
    "typed-call-mismatch",
    "typed-call-no-cpu-twin",
    "typed-call-no-run",
    "typed-call-ok",
    "typed-call-sampler-for-buffer",
    "typed-call-texture-for-buffer",
    "typed-call-texture-ok",
    "workgroup-over-16kib",
    "wrapping-div-rejected",
    "wrapping-ok",
];

#[test]
fn the_repository_fixture_set_is_pinned_by_name() {
    let ui = workspace_root().unwrap().join(UI_DIR);
    let found: Vec<String> = discover_cases(&ui)
        .unwrap()
        .into_iter()
        .map(|case| case.name)
        .collect();
    assert_eq!(found, PINNED_FIXTURES, "tests/ui fixture crates changed");
    // Each pinned fixture is also a member of the fixture workspace.
    let manifest = fs::read_to_string(ui.join("Cargo.toml")).unwrap();
    for name in PINNED_FIXTURES {
        assert!(manifest.contains(&format!("\"{name}\"")), "{name}");
    }
}
