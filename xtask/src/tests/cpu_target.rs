use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{
    CPU_PROFILE, CPU_TARGET_DIR, Call, Dirs, FINAL_LINE, FLIPS, FORBIDDEN, KERNELS, Profile,
    REFUSAL_FILE, REFUSAL_ITEM, SCAN_CONTROL, STEPS, all_calls, e_a_build, e_b_tree, e_c_build,
    e_d_check, e_e_check, e_f_check, e_g_check, e_h_build, e_i_test, executable_of, judge_run,
    match_problems, match_summary, parse_match_line, passed_tests, refusal_line, relative_to, scan,
    scan_problems, tree_problems, with_located_line,
};
use crate::{
    checks::{DUAL_RUSTFLAGS, DUAL_RUSTFLAGS_VARIABLE, RUSTFLAGS_VARIABLES},
    process::PROFILE_VARIABLE,
    suite::{Step, Suite},
    ui::{Expectation, parse_expectation, parse_expected},
    workspace::workspace_root,
};

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

fn dirs() -> Dirs {
    Dirs::under("T")
}

/// A file of the repository, with LF endings.
fn fixture_root_file(path: &str) -> String {
    let root = workspace_root().unwrap();
    fs::read_to_string(root.join(path))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
        .replace("\r\n", "\n")
}

fn fixture_file(name: &str) -> String {
    let path = workspace_root().unwrap().join(CPU_TARGET_DIR).join(name);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

// ---- the `matches` lines (E-a, E-c, E-h) ----

const GOOD: &str = "CPU target: matches (vector_add::add, 1000 elements, max ULP 0)
CPU target: matches (polynomial::evaluate, 1000 elements, max ULP 0)
CPU target: matches (atomic_counter::increment, 257 invocations, max ULP 0)
CPU target: all kernels match
";

#[test]
fn a_matches_line_has_a_kernel_a_count_with_its_unit_and_a_ulp() {
    let line = parse_match_line("CPU target: matches (vector_add::add, 1000 elements, max ULP 0)")
        .unwrap();
    assert_eq!(
        (
            line.kernel.as_str(),
            line.count,
            line.unit.as_str(),
            line.max_ulp
        ),
        ("vector_add::add", 1000, "elements", 0)
    );
    // The atomic counter has invocations, not elements (the m5.e1 review).
    let line = parse_match_line(
        "CPU target: matches (atomic_counter::increment, 257 invocations, max ULP 0)",
    )
    .unwrap();
    assert_eq!(
        (line.kernel.as_str(), line.count, line.unit.as_str()),
        ("atomic_counter::increment", 257, "invocations")
    );
    // Surrounding white space (a CRLF line end) does not matter.
    assert!(
        parse_match_line("CPU target: matches (polynomial::evaluate, 1 elements, max ULP 12)\r")
            .is_some()
    );
}

#[test]
fn a_malformed_matches_line_is_not_parsed() {
    for text in [
        "",
        "CPU target: all kernels match",
        "CPU twin: matches (3 elements, max ULP 0)",
        "CPU target: matches (add, elements, max ULP 0)",
        "CPU target: matches (add, 3 items, max ULP 0)",
        "CPU target: matches (add, -3 elements, max ULP 0)",
        "CPU target: matches (add, 3 elements, max ULP )",
        "CPU target: matches (add, 3 elements, max ULP x)",
        "CPU target: matches (add, 3 elements)",
        "CPU target: matches (add, 3 elements, max ULP 0, extra)",
        "CPU target: matches (, 3 elements, max ULP 0)",
        "CPU target: matches (two words, 3 elements, max ULP 0)",
        "CPU target: matches (add, 3 elements, max ULP 0",
    ] {
        assert_eq!(parse_match_line(text), None, "{text:?}");
    }
}

#[test]
fn the_output_of_a_good_run_has_no_problem() {
    assert_eq!(match_problems(GOOD), Vec::<String>::new());
    assert_eq!(
        match_problems(&GOOD.replace('\n', "\r\n")),
        Vec::<String>::new()
    );
    assert_eq!(
        match_summary(GOOD),
        "vector_add::add 1000 elements, max ULP 0; polynomial::evaluate 1000 elements, max ULP 0; atomic_counter::increment 257 invocations, max ULP 0"
    );
}

#[test]
fn a_run_without_a_kernel_names_that_kernel() {
    for kernel in KERNELS {
        let without: String = GOOD
            .lines()
            .filter(|line| !line.contains(kernel))
            .map(|line| format!("{line}\n"))
            .collect();
        assert_eq!(
            match_problems(&without),
            [format!("no `matches` line for the kernel `{kernel}`")],
            "{kernel}"
        );
    }
}

#[test]
fn a_run_that_stopped_early_names_every_missing_kernel_and_the_final_line() {
    // `app` exits on its first mismatch: nothing after the first kernel is printed.
    let problems =
        match_problems("CPU target: matches (vector_add::add, 1000 elements, max ULP 0)\n");
    assert_eq!(
        problems,
        [
            "no `matches` line for the kernel `polynomial::evaluate`".to_owned(),
            "no `matches` line for the kernel `atomic_counter::increment`".to_owned(),
            format!("the final line `{FINAL_LINE}` is missing"),
        ]
    );
    assert_eq!(match_problems("").len(), KERNELS.len() + 1);
}

#[test]
fn duplicates_unknown_kernels_empty_counts_and_a_repeated_final_line_are_problems() {
    let duplicate = format!("{GOOD}CPU target: matches (vector_add::add, 5 elements, max ULP 0)\n");
    assert_eq!(
        match_problems(&duplicate),
        ["2 `matches` lines for the kernel `vector_add::add`, expected one"]
    );
    let unknown = format!("{GOOD}CPU target: matches (other::kernel, 5 elements, max ULP 0)\n");
    assert_eq!(
        match_problems(&unknown),
        [
            "a `matches` line for the unknown kernel `other::kernel` (the fixture and the check disagree)"
        ]
    );
    let empty = GOOD.replace(
        "1000 elements, max ULP 0)\nCPU target: matches (poly",
        "0 elements, max ULP 0)\nCPU target: matches (poly",
    );
    assert_eq!(
        match_problems(&empty),
        ["the kernel `vector_add::add` matched 0 elements, which proves nothing"]
    );
    let twice = format!("{GOOD}{FINAL_LINE}\n");
    assert_eq!(
        match_problems(&twice),
        [format!("the final line `{FINAL_LINE}` occurs 2 times")]
    );
    let malformed =
        format!("{GOOD}CPU target: matches (vector_add::add, many elements, max ULP 0)\n");
    assert_eq!(
        match_problems(&malformed),
        [
            "a malformed `matches` line: `CPU target: matches (vector_add::add, many elements, max ULP 0)`"
        ]
    );
}

#[test]
fn other_text_in_the_output_is_not_a_problem() {
    let noisy = format!("some log line\nS5.2 vector_add elements=1 runs=5\n{GOOD}done\n");
    assert_eq!(match_problems(&noisy), Vec::<String>::new());
}

// ---- the byte scan (E-b) ----

#[test]
fn the_scan_finds_each_forbidden_string_in_any_letter_case() {
    for marker in FORBIDDEN {
        for text in [marker.to_owned(), marker.to_uppercase(), {
            let mut mixed = marker.to_owned();
            mixed[..1].make_ascii_uppercase();
            mixed
        }] {
            let mut bytes = b"\x00\x01 cpu target: matches \xff".to_vec();
            bytes.extend_from_slice(b"padding ");
            bytes.extend_from_slice(text.as_bytes());
            bytes.extend_from_slice(b" tail");
            let report = scan(&bytes);
            assert_eq!(report.control, 1, "{text}");
            let problems = scan_problems(&report);
            assert_eq!(
                problems,
                [format!(
                    "the release binary contains `{marker}` (1 time(s), any letter case)"
                )],
                "{text}"
            );
        }
    }
}

#[test]
fn a_clean_binary_with_its_control_string_passes() {
    let report = scan(b"abc CPU target: matches (x) def");
    assert_eq!(report.control, 1);
    assert_eq!(scan_problems(&report), Vec::<String>::new());
    // The control is found in any letter case too: the scan lowercases the binary, not the markers.
    assert_eq!(scan(b"CPU TARGET: MATCHES").control, 1);
    assert!(SCAN_CONTROL.bytes().all(|byte| !byte.is_ascii_uppercase()));
    assert!(
        FORBIDDEN
            .iter()
            .all(|marker| marker.bytes().all(|byte| !byte.is_ascii_uppercase()))
    );
}

#[test]
fn a_scan_that_cannot_see_its_control_is_blind_and_fails() {
    let report = scan(b"nothing readable here");
    assert_eq!(report.control, 0);
    let problems = scan_problems(&report);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].starts_with("the scan is blind"), "{problems:?}");
    // An empty binary is blind too, and a blind scan still reports a forbidden string.
    assert_eq!(scan_problems(&scan(b"")).len(), 1);
    assert_eq!(scan_problems(&scan(b"wgpu")).len(), 2);
}

#[test]
fn the_scan_counts_every_occurrence() {
    let report = scan(b"wgpu cpu target: matches WGPU xwgpux");
    assert_eq!(report.found[2], ("wgpu", 3));
    assert_eq!(report.found[0], ("slangc", 0));
}

// ---- cargo tree (E-b) ----

const TREE: &str = "app v0.0.0 (/x/tests/cpu-target/app)
|-- gust v0.1.0 (/x/crates/gust)
|   |-- gust-macros v0.1.0 (/x/crates/gust-macros)
|   `-- syn v2.0.1
`-- kernels v0.0.0 (/x/tests/cpu-target/kernels)
    `-- gust v0.1.0 (/x/crates/gust) (*)
";

#[test]
fn the_tree_of_app_has_no_gpu_runtime_and_no_bridge() {
    assert_eq!(tree_problems(TREE), Vec::<String>::new());
}

#[test]
fn a_tree_with_the_gpu_runtime_or_the_bridge_is_a_problem() {
    let with_wgpu = format!("{TREE}|-- gust-wgpu v0.1.0 (/x/crates/gust-wgpu)\n");
    assert_eq!(
        tree_problems(&with_wgpu),
        [
            "the GPU runtime or the bridge is in the tree: `|-- gust-wgpu v0.1.0 (/x/crates/gust-wgpu)`"
        ]
    );
    let with_wgpu_crate = format!("{TREE}`-- wgpu v30.0.1\n");
    assert_eq!(tree_problems(&with_wgpu_crate).len(), 1);
    let with_feature = TREE.replace(
        "`-- syn v2.0.1",
        "|   |-- gust feature \"slang-runtime\"\n|   `-- syn v2.0.1",
    );
    let problems = tree_problems(&with_feature);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("slang-runtime"));
}

#[test]
fn a_tree_full_of_the_gpu_runtime_shows_the_first_lines_and_the_count() {
    let mut tree = TREE.to_owned();
    for index in 0..10 {
        tree.push_str(&format!("`-- wgpu-part{index} v1.0.0\n"));
    }
    let problems = tree_problems(&tree);
    assert_eq!(problems.len(), 4, "{problems:?}");
    assert!(problems[0].ends_with("`-- wgpu-part0 v1.0.0`"));
    assert_eq!(
        problems[3],
        "... and 7 more lines of the tree name wgpu or slang-runtime"
    );
}

#[test]
fn ci_runs_check_cpu_target_once_in_a_job_of_its_own() {
    let ci = fixture_root_file(".github/workflows/ci.yml");
    // The job text: from its key to the next job key (two spaces of indentation, a name and a colon).
    let start = ci.find("\n  cpu-target:\n").expect(&ci);
    let rest = &ci[start + 1..];
    let job = &rest[..next_job(rest)];
    let name = job
        .find("      - name: Production CPU target\n")
        .expect(job);
    let run = job
        .find("        run: cargo --locked xtask check-cpu-target\n")
        .expect(job);
    assert!(name < run, "{job}");
    // The run line follows the name directly or after comments only, and it is the last step of the job.
    assert!(
        job[name..run]
            .lines()
            .skip(1)
            .all(|line| line.trim_start().starts_with('#')),
        "{job}"
    );
    assert!(!job[run + 1..].contains("\n      - "), "{job}");
    assert!(job.contains("    runs-on: ubuntu-latest\n"), "{job}");
    assert!(job.contains("      - uses: actions/checkout@"), "{job}");
    // One step in the whole workflow, and not in `unit`, which keeps its budget.
    assert_eq!(ci.matches("check-cpu-target").count(), 1, "{ci}");
    let unit = ci.split("\n  msrv:").next().unwrap();
    assert!(!unit.contains("cpu-target"), "{unit}");
}

/// The offset of the next job key (`\n  name:` at two spaces) in `text`, which starts with a job key.
fn next_job(text: &str) -> usize {
    let mut at = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index > 0
            && line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_end().ends_with(':')
        {
            return at;
        }
        at += line.len();
    }
    text.len()
}

// ---- cargo's JSON output (the binary a build wrote)

fn artifact(name: &str, kind: &str, executable: &str) -> String {
    format!(
        r#"{{"reason":"compiler-artifact","package_id":"path+file:///w#{name}@0.0.0","manifest_path":"/w/{name}/Cargo.toml","target":{{"kind":["{kind}"],"crate_types":["{kind}"],"name":"{name}","src_path":"/w/{name}/src/main.rs","edition":"2024","doc":true,"doctest":false,"test":true}},"profile":{{"opt_level":"3","debuginfo":0,"debug_assertions":false,"overflow_checks":false,"test":false}},"features":[],"filenames":["/t/{name}"],"executable":{executable},"fresh":false}}"#
    )
}

#[test]
fn the_executable_is_the_one_cargo_reports_for_the_binary() {
    let stdout = [
        r#"{"reason":"build-script-executed","package_id":"x","linked_libs":[],"linked_paths":[],"cfgs":[],"env":[],"out_dir":"/o"}"#.to_owned(),
        artifact("kernels", "lib", "null"),
        artifact("app", "bin", r#""/t/x86_64-unknown-linux-gnu/release/app""#),
        r#"{"reason":"build-finished","success":true}"#.to_owned(),
    ]
    .join("\n");
    assert_eq!(
        executable_of(&stdout, "app").unwrap(),
        PathBuf::from("/t/x86_64-unknown-linux-gnu/release/app")
    );
}

#[test]
fn the_windows_spelling_of_the_executable_is_read() {
    let stdout = artifact("app", "bin", r#""C:\\t\\release\\app.exe""#);
    assert_eq!(
        executable_of(&stdout, "app").unwrap(),
        PathBuf::from(r"C:\t\release\app.exe")
    );
}

#[test]
fn a_build_output_without_one_executable_for_the_binary_is_an_error() {
    // None: only a library, another binary, or an artifact with a null executable.
    for stdout in [
        String::new(),
        "Compiling app v0.0.0\nFinished\n".to_owned(),
        artifact("app", "lib", r#""/t/libapp.rlib""#),
        artifact("other", "bin", r#""/t/other""#),
        artifact("app", "bin", "null"),
        "{not json at all".to_owned(),
    ] {
        let error = executable_of(&stdout, "app").unwrap_err();
        assert!(error.contains("names no executable"), "{error}");
    }
    // Two: ambiguous.
    let two = [
        artifact("app", "bin", r#""/t/a/app""#),
        artifact("app", "bin", r#""/t/b/app""#),
    ]
    .join("\n");
    let error = executable_of(&two, "app").unwrap_err();
    assert!(error.contains("names 2 executables"), "{error}");
}

#[test]
fn the_builds_that_run_a_binary_print_cargos_json_and_the_others_do_not() {
    let d = dirs();
    for call in [e_a_build(&d), e_c_build(&d), e_h_build(&d, Profile::Cpu)] {
        let position = call
            .args
            .iter()
            .position(|arg| arg == "--message-format")
            .unwrap_or_else(|| panic!("{call:?}"));
        assert_eq!(call.args[position + 1], "json-render-diagnostics");
        assert!(call.args.iter().any(|arg| arg == "--bin"), "{call:?}");
    }
    // The tree and the test run read text.
    for call in [e_b_tree(&d), e_i_test(&d)] {
        assert!(
            !call.args.iter().any(|arg| arg == "--message-format"),
            "{call:?}"
        );
    }
}

#[test]
fn the_passed_tests_are_counted_from_the_result_lines() {
    assert_eq!(
        passed_tests("running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored\n"),
        Some(0)
    );
    assert_eq!(
        passed_tests(
            "test result: ok. 3 passed; 0 failed\nDoc-tests\ntest result: ok. 2 passed; 0 failed\n"
        ),
        Some(5)
    );
    assert_eq!(
        passed_tests("test result: FAILED. 1 passed; 1 failed\n"),
        None
    );
    assert_eq!(passed_tests(""), None);
}

#[test]
fn an_empty_or_unreadable_tree_cannot_pass() {
    assert_eq!(tree_problems("").len(), 3);
    let missing_kernels = TREE.replace("kernels v0.0.0", "other v0.0.0");
    assert_eq!(
        tree_problems(&missing_kernels),
        ["the tree has no `kernels v..` line, so it was not read correctly"]
    );
}

// ---- the refusal of gust-wgpu (E-e) ----

#[test]
fn the_refusal_item_is_found_on_its_own_line_and_only_once() {
    let source = format!("//! docs\n\nuse x;\n{REFUSAL_ITEM}\n\nstatic Y: u8 = 0;\n");
    assert_eq!(refusal_line(&source), Ok(4));
    // Indented, or with CRLF line ends, it still counts.
    let crlf = source.replace('\n', "\r\n");
    assert_eq!(refusal_line(&crlf), Ok(4));
    // A comment about it does not.
    let commented = format!("// {REFUSAL_ITEM}\n");
    assert!(
        refusal_line(&commented)
            .unwrap_err()
            .contains("has no line")
    );
    let twice = format!("{REFUSAL_ITEM}\n{REFUSAL_ITEM}\n");
    assert!(refusal_line(&twice).unwrap_err().contains("has 2 lines"));
    assert!(
        refusal_line("")
            .unwrap_err()
            .contains("does not refuse GUST_PROFILE=cpu")
    );
}

#[test]
fn gust_wgpu_really_has_the_refusal_item_once() {
    let root = workspace_root().unwrap();
    let source = fs::read_to_string(root.join(REFUSAL_FILE)).unwrap();
    assert!(refusal_line(&source).is_ok());
}

#[test]
fn the_located_line_replaces_line_auto_only() {
    let text = "code: E0080\nfile: f.rs\nline: auto\nmessage: m\n";
    assert_eq!(
        with_located_line(text, 81),
        "code: E0080\nfile: f.rs\nline: 81\nmessage: m\n"
    );
    // The line ending of the replaced line stays as it was.
    assert_eq!(
        with_located_line(&text.replace('\n', "\r\n"), 7),
        "code: E0080\r\nfile: f.rs\r\nline: 7\r\nmessage: m\r\n"
    );
    // A file with a number is not touched.
    let numbered = "code: E0277\nline: 14\n";
    assert_eq!(with_located_line(numbered, 81), numbered);
    // `line: auto` inside a comment is not the key.
    let comment = "# line: auto\ncode: E0080\n";
    assert_eq!(with_located_line(comment, 5), comment);
}

#[test]
fn the_expectation_of_e_e_is_the_one_error_of_the_refusal() {
    let text = with_located_line(&fixture_file("e-e-expected.txt"), 81);
    let wanted = parse_expected(&text).unwrap();
    assert_eq!(wanted.code, "E0080");
    assert_eq!(wanted.file, REFUSAL_FILE);
    assert_eq!(wanted.line, 81);
    // The text is the one of `gust::profile::NOT_A_GPU_TARGET`, read from the source so that they cannot drift.
    let profile =
        fs::read_to_string(workspace_root().unwrap().join("crates/gust/src/profile.rs")).unwrap();
    let message = wanted
        .messages
        .iter()
        .find(|message| message.starts_with("evaluation panicked: "))
        .unwrap();
    let sentence = message.strip_prefix("evaluation panicked: ").unwrap();
    assert!(
        profile.contains(&format!("\"{sentence}\"")),
        "profile.rs does not hold {sentence:?}"
    );
    assert!(sentence.contains("GUST_PROFILE=cpu"));
    assert!(matches!(
        parse_expectation(&text),
        Ok(Expectation::Fails(_))
    ));
}

#[test]
fn every_expectation_file_parses() {
    let d = parse_expected(&fixture_file("e-d-expected.txt")).unwrap();
    assert_eq!(
        (d.code.as_str(), d.file.as_str()),
        ("E0277", "app/src/bin/single_dispatch.rs")
    );
    let g = parse_expected(&fixture_file("e-g-expected.txt")).unwrap();
    assert_eq!(
        (g.code.as_str(), g.file.as_str()),
        ("E0277", "optout/src/main.rs")
    );
    assert!(matches!(
        parse_expectation(&fixture_file("e-f-expected.txt")),
        Ok(Expectation::BuildFails(_))
    ));
}

// ---- absolute paths (E-e) ----

#[test]
fn the_root_is_taken_off_the_front_of_a_path_in_every_spelling() {
    let escaped = r#"{"a":"D:\\build\\repo\\crates\\x\\lib.rs","b":"D:/build/repo/y.rs"}"#;
    assert_eq!(
        relative_to(escaped, r"D:\build\repo"),
        r#"{"a":"crates\\x\\lib.rs","b":"y.rs"}"#
    );
    // A trailing separator on the root, and a drive letter in the other case.
    assert_eq!(
        relative_to(escaped, r"d:\build\repo\"),
        r#"{"a":"crates\\x\\lib.rs","b":"y.rs"}"#
    );
    let unix = r#"{"a":"/ci/work/gust/crates/x/lib.rs"}"#;
    assert_eq!(
        relative_to(unix, "/ci/work/gust"),
        r#"{"a":"crates/x/lib.rs"}"#
    );
}

#[test]
fn a_path_outside_the_root_or_not_at_the_start_of_a_string_stays() {
    let text = r#"{"std":"C:\\rustc\\library\\core\\panic.rs","rendered":"--> D:\\build\\repo\\a.rs","sibling":"D:\\build\\repo2\\a.rs"}"#;
    let out = relative_to(text, r"D:\build\repo");
    // The sibling directory `repo2` does not start with `repo\`, so it stays.
    assert!(out.contains(r#""D:\\build\\repo2\\a.rs""#));
    assert!(out.contains(r#""C:\\rustc\\library\\core\\panic.rs""#));
    // Only a string value that starts with the root is rewritten.
    assert!(out.contains(r"--> D:\\build\\repo\\a.rs"));
}

/// The `compiler-message` line of an error of `target`, with the primary span at `file_json` (JSON text).
fn error_line(target: &str, code: &str, message: &str, file_json: &str, line: u64) -> String {
    format!(
        r#"{{"reason":"compiler-message","target":{{"kind":["lib"],"crate_types":["lib"],"name":"{target}"}},"message":{{"rendered":"error\n","children":[{{"children":[],"code":null,"level":"note","message":"inside `require_gpu_target`","rendered":null,"spans":[]}}],"code":{{"code":"{code}","explanation":null}},"level":"error","message":"{message}","spans":[{{"column_end":1,"column_start":1,"expansion":null,"file_name":"{file_json}","is_primary":true,"label":null,"line_end":{line},"line_start":{line}}}]}}}}"#
    )
}

const REFUSAL: &str = "evaluation panicked: gust-wgpu cannot be built under GUST_PROFILE=cpu: a CPU build contains one target";

fn e_e_text(line: usize) -> String {
    with_located_line(&fixture_file("e-e-expected.txt"), line)
}

#[test]
fn e_e_accepts_the_absolute_path_cargo_reports_for_the_dependency() {
    for (root, file) in [
        (
            r"D:\build\repo",
            r"D:\\build\\repo\\crates\\gust-wgpu\\src\\lib.rs",
        ),
        ("/ci/work/gust", "/ci/work/gust/crates/gust-wgpu/src/lib.rs"),
    ] {
        let stdout = error_line("gust_wgpu", "E0080", REFUSAL, file, 81);
        let result = judge_run(
            "e-e-expected.txt",
            &e_e_text(81),
            ("crates/gust-wgpu", "gust_wgpu"),
            root,
            (false, &stdout, ""),
        );
        assert_eq!(result, Ok(()), "{root}");
    }
}

#[test]
fn e_e_rejects_a_wrong_line_a_wrong_message_a_wrong_code_and_a_build_that_passed() {
    let file = "/r/crates/gust-wgpu/src/lib.rs";
    let run = |stdout: &str, success: bool, line: usize| {
        judge_run(
            "e-e-expected.txt",
            &e_e_text(line),
            ("crates/gust-wgpu", "gust_wgpu"),
            "/r",
            (success, stdout, ""),
        )
    };
    let good = error_line("gust_wgpu", "E0080", REFUSAL, file, 81);
    assert_eq!(run(&good, false, 81), Ok(()));
    // The item moved but the expectation was not located again.
    assert!(run(&good, false, 80).unwrap_err().contains("line: 80"));
    let wrong_message = error_line(
        "gust_wgpu",
        "E0080",
        "evaluation panicked: another",
        file,
        81,
    );
    assert!(
        run(&wrong_message, false, 81)
            .unwrap_err()
            .contains("message: evaluation panicked")
    );
    let wrong_code = error_line("gust_wgpu", "E0277", REFUSAL, file, 81);
    assert!(
        run(&wrong_code, false, 81)
            .unwrap_err()
            .contains("code: E0080")
    );
    // An error of another crate does not count (the check wants the error of `gust-wgpu` itself).
    let other = error_line("app_with_wgpu", "E0080", REFUSAL, file, 81);
    assert!(
        run(&other, false, 81)
            .unwrap_err()
            .contains("printed no error-level")
    );
    // The mutation "remove the const item": the build passes.
    assert!(
        run("", true, 1)
            .unwrap_err()
            .contains("must fail, but it compiled")
    );
    // Two errors are not the one error.
    let twice = format!("{good}\n{good}");
    assert!(run(&twice, false, 81).unwrap_err().contains("2 errors"));
}

// ---- the cargo calls ----

#[test]
fn the_calls_are_exactly_these() {
    let manifest = "tests/cpu-target/Cargo.toml";
    let fixture = |verb: &[&str], rest: &[&str], dir: Option<&str>| {
        let mut args = owned(verb);
        args.extend(owned(&[
            "--locked",
            "--color",
            "never",
            "--manifest-path",
            manifest,
        ]));
        args.extend(owned(rest));
        if let Some(dir) = dir {
            args.extend(owned(&["--target-dir", dir]));
        }
        args
    };
    let d = dirs();
    assert_eq!(
        (d.main.as_str(), d.dual.as_str(), d.flip.as_str()),
        ("T/cpu-target", "T/cpu-target-dual", "T/cpu-target-flip")
    );
    assert_eq!(
        e_a_build(&d),
        Call {
            args: fixture(
                &["build", "--release"],
                &[
                    "-p",
                    "app",
                    "--bin",
                    "app",
                    "--message-format",
                    "json-render-diagnostics"
                ],
                Some("T/cpu-target")
            ),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_b_tree(&d),
        Call {
            args: fixture(&["tree"], &["-e", "features", "-p", "app"], None),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_c_build(&d),
        Call {
            args: fixture(
                &["build"],
                &[
                    "-p",
                    "app",
                    "--bin",
                    "app",
                    "--message-format",
                    "json-render-diagnostics"
                ],
                Some("T/cpu-target-dual")
            ),
            profile: Profile::Cpu,
            cfg: true
        }
    );
    assert_eq!(
        e_d_check(&d),
        Call {
            args: fixture(
                &["check"],
                &[
                    "-p",
                    "app",
                    "--bin",
                    "single-dispatch",
                    "--message-format",
                    "json"
                ],
                Some("T/cpu-target")
            ),
            profile: Profile::Dev,
            cfg: false
        }
    );
    assert_eq!(
        e_e_check(&d),
        Call {
            args: fixture(
                &["check"],
                &["-p", "app-with-wgpu", "--message-format", "json"],
                Some("T/cpu-target")
            ),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_f_check(&d),
        Call {
            args: fixture(
                &["check"],
                &[
                    "-p",
                    "app",
                    "--features",
                    "gust/slang-runtime",
                    "--message-format",
                    "json"
                ],
                Some("T/cpu-target")
            ),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_g_check(&d),
        Call {
            args: fixture(
                &["check"],
                &["-p", "optout", "--message-format", "json"],
                Some("T/cpu-target")
            ),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_h_build(&d, Profile::Cpu),
        Call {
            args: fixture(
                &["build"],
                &[
                    "-p",
                    "app",
                    "--bin",
                    "app",
                    "--message-format",
                    "json-render-diagnostics"
                ],
                Some("T/cpu-target-flip")
            ),
            profile: Profile::Cpu,
            cfg: false
        }
    );
    assert_eq!(
        e_h_build(&d, Profile::Unset),
        Call {
            args: fixture(
                &["build"],
                &[
                    "-p",
                    "app",
                    "--bin",
                    "single-dispatch",
                    "--message-format",
                    "json"
                ],
                Some("T/cpu-target-flip")
            ),
            profile: Profile::Unset,
            cfg: false
        }
    );
    assert_eq!(
        e_i_test(&d),
        Call {
            args: fixture(&["test"], &["-p", "kernels"], Some("T/cpu-target")),
            profile: Profile::Cpu,
            cfg: false
        }
    );
}

#[test]
fn every_call_is_locked_and_builds_in_one_of_the_three_target_directories() {
    let calls = all_calls(&dirs());
    assert_eq!(calls.len(), 7 + FLIPS.len() + 1);
    for (id, call) in &calls {
        assert!(
            call.args.iter().any(|arg| arg == "--locked"),
            "{id}: {call:?}"
        );
        let directories: Vec<&String> = call
            .args
            .windows(2)
            .filter(|pair| pair[0] == "--target-dir")
            .map(|pair| &pair[1])
            .collect();
        match call.args[0].as_str() {
            "tree" => assert!(directories.is_empty(), "{id}"),
            _ => assert_eq!(directories.len(), 1, "{id}: {call:?}"),
        }
        assert!(call.args.iter().any(|arg| arg == "--manifest-path"), "{id}");
        assert!(!call.shown().contains("T/cpu"), "{id}: {}", call.shown());
    }
    // Only E-c carries the cfg, and it has a directory of its own, so its rustflags never invalidate the others.
    let with_cfg: Vec<&str> = calls
        .iter()
        .filter(|(_, call)| call.cfg)
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(with_cfg, ["E-c"]);
    let main = dirs().main;
    assert!(
        calls
            .iter()
            .filter(|(id, _)| *id == "E-h")
            .all(|(_, call)| call.args.contains(&dirs().flip))
    );
    assert!(
        calls
            .iter()
            .filter(|(id, _)| ["E-a", "E-d", "E-e", "E-f", "E-g", "E-i"].contains(id))
            .all(|(_, call)| call.args.contains(&main))
    );
}

#[test]
fn the_profile_flips_in_both_directions_starting_and_ending_in_different_states() {
    assert_eq!(FLIPS[0], Profile::Cpu);
    for pair in FLIPS.windows(2) {
        assert_ne!(pair[0], pair[1], "a flip that does not flip: {FLIPS:?}");
    }
    let to_unset = FLIPS
        .windows(2)
        .filter(|p| *p == [Profile::Cpu, Profile::Unset])
        .count();
    let to_cpu = FLIPS
        .windows(2)
        .filter(|p| *p == [Profile::Unset, Profile::Cpu])
        .count();
    assert!(to_unset >= 2 && to_cpu >= 2, "{FLIPS:?}");
    // Each state is the call of its profile: `cpu` builds `app`, the other state builds `single-dispatch`.
    for profile in FLIPS {
        let call = e_h_build(&dirs(), profile);
        assert_eq!(call.profile, profile);
        let builds_app = call.args.windows(2).any(|pair| pair == ["--bin", "app"]);
        assert_eq!(builds_app, profile == Profile::Cpu);
    }
}

fn env_of(call: &Call) -> Vec<(String, Option<String>)> {
    let mut env: Vec<(String, Option<String>)> = call
        .command(Path::new("."), "E-x")
        .get_envs()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.map(|value| value.to_string_lossy().into_owned()),
            )
        })
        .collect();
    env.sort();
    env
}

fn variable(call: &Call, name: &str) -> Option<Option<String>> {
    env_of(call)
        .into_iter()
        .find(|(candidate, _)| candidate == name)
        .map(|(_, value)| value)
}

#[test]
fn each_state_gets_its_own_profile_variable() {
    let d = dirs();
    // cpu: set. dev: the workspace's. unset: removed, which is what makes E-h's flip a change of the variable.
    assert_eq!(
        variable(&e_a_build(&d), PROFILE_VARIABLE),
        Some(Some(CPU_PROFILE.to_owned()))
    );
    assert_eq!(
        variable(&e_d_check(&d), PROFILE_VARIABLE),
        Some(Some("dev".to_owned()))
    );
    assert_eq!(
        variable(&e_h_build(&d, Profile::Unset), PROFILE_VARIABLE),
        Some(None)
    );
    assert_eq!(
        variable(&e_h_build(&d, Profile::Cpu), PROFILE_VARIABLE),
        Some(Some("cpu".to_owned()))
    );
}

#[test]
fn the_dual_cfg_reaches_e_c_only_and_no_call_inherits_rustflags() {
    let d = dirs();
    assert_eq!(
        variable(&e_c_build(&d), DUAL_RUSTFLAGS_VARIABLE),
        Some(Some(DUAL_RUSTFLAGS.to_owned()))
    );
    for (id, call) in all_calls(&d) {
        for name in RUSTFLAGS_VARIABLES {
            let value = variable(&call, name);
            if id == "E-c" && name == DUAL_RUSTFLAGS_VARIABLE {
                continue;
            }
            assert_eq!(
                value,
                Some(None),
                "{id}: {name} must be removed, not inherited"
            );
        }
    }
}

#[test]
fn a_call_is_shown_without_the_machine_s_target_directory() {
    let shown = e_c_build(&Dirs::under("/very/private/target")).shown();
    assert!(!shown.contains("/very/private"), "{shown}");
    assert!(shown.contains("<target>/cpu-target-dual"), "{shown}");
    assert!(
        shown.starts_with(
            "CARGO_ENCODED_RUSTFLAGS=\"--cfg gust_dual\" GUST_PROFILE=cpu cargo build"
        ),
        "{shown}"
    );
    assert!(
        e_h_build(&dirs(), Profile::Unset)
            .shown()
            .starts_with("(no GUST_PROFILE) cargo build")
    );
    // A Windows target directory with a trailing separator.
    let windows = Dirs::under(r"C:\t\target\");
    assert_eq!(windows.main, r"C:\t\target/cpu-target");
    assert!(!e_a_build(&windows).shown().contains(r"C:\t"));
}

// ---- the command and the suite ----

#[test]
fn the_steps_run_in_the_order_of_the_plan() {
    let ids: Vec<&str> = STEPS.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        [
            "E-a", "E-b", "E-c", "E-d", "E-e", "E-f", "E-g", "E-h", "E-i"
        ]
    );
}

#[test]
fn the_cpu_suite_is_the_one_command() {
    assert_eq!(Suite::Cpu.steps(), vec![Step::CheckCpuTarget]);
    assert_eq!(Suite::parse("cpu"), Ok(Suite::Cpu));
    assert!(Suite::names().contains(&"Cpu"));
    // The Unit suite does not contain it (it stays a tool-free tier of small steps); CI runs the command itself.
    assert!(!Suite::Unit.steps().contains(&Step::CheckCpuTarget));
    assert!(!Suite::Slangc.steps().contains(&Step::CheckCpuTarget));
    assert!(Suite::Cpu.covers().contains("GUST_PROFILE=cpu"));
    assert!(Suite::Cpu.not_covered().contains("GPU"));
}

#[test]
fn the_command_takes_no_arguments() {
    assert_eq!(
        super::check_cpu_target(&owned(&["--check-only"])),
        Err("check-cpu-target takes no arguments".to_owned())
    );
}

#[test]
fn the_fixture_has_every_file_the_calls_name() {
    let root: PathBuf = workspace_root().unwrap().join(CPU_TARGET_DIR);
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "e-d-expected.txt",
        "e-e-expected.txt",
        "e-f-expected.txt",
        "e-g-expected.txt",
        "app/Cargo.toml",
        "app/src/main.rs",
        "app/src/bin/single_dispatch.rs",
        "app-with-wgpu/src/main.rs",
        "optout/src/main.rs",
        "kernels/src/lib.rs",
    ] {
        assert!(root.join(name).is_file(), "{name}");
    }
    // The three kernels the check requires are the ones `app` prints (a rename of one fails here, not in a run).
    let app = fs::read_to_string(root.join("app/src/main.rs")).unwrap();
    for kernel in KERNELS {
        assert!(app.contains(&format!("{kernel}::dispatch")), "{kernel}");
        assert!(app.contains(&format!("matches ({kernel},")), "{kernel}");
    }
    assert!(app.contains(FINAL_LINE));
}
