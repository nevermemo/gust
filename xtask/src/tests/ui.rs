use std::fs;

use super::{
    Expectation, Expected, UI_DIR, cargo_args, discover_cases, judge, parse_diagnostics,
    parse_expected,
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

/// The fixture crates under tests/ui, by name. `discover_cases` takes whatever directories exist, so
/// deleting a fixture (with its `members` entry) would leave `check-ui` green; this list is the pin.
/// Adding or removing a fixture changes it deliberately, in the same change as the fixture.
const PINNED_FIXTURES: [&str; 11] = [
    "gpupod-missing",
    "gpupod-present",
    "kernel-cpu-bad-value",
    "kernel-cpu-duplicate",
    "kernel-missing",
    "layout-outside-element",
    "layout-outside-field",
    "typed-call-mismatch",
    "typed-call-no-cpu-twin",
    "typed-call-no-run",
    "typed-call-ok",
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
