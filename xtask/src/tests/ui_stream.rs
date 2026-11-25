use super::{PackageStream, fixture_of, shared_cause, split_by_package, stderr_of};
use crate::ui::{
    Case, Expectation, group_args, group_by_env, group_call, group_command, group_label,
    groups_call, judge_shared, manifest_sets_features, parse_expectation, parse_expected,
    shared_args, shared_command, shares_call,
};

const ROOT: &str = "C:/work/repo";

const LABEL: &str = "the shared `cargo check`";

fn manifest(case: &str) -> String {
    format!("{ROOT}/tests/ui/{case}/Cargo.toml")
}

/// A `compiler-message` line of the package `case` (target `target`), with one primary span at `line` of the
/// case's `src/main.rs` (relative to the fixture workspace, with backslashes as on Windows).
fn message(case: &str, target: &str, level: &str, code: &str, line: u32) -> String {
    format!(
        r#"{{"reason":"compiler-message","package_id":"path+file:///{ROOT}/tests/ui/{case}#0.0.0","manifest_path":"{}","target":{{"kind":["bin"],"name":"{target}"}},"message":{{"level":"{level}","message":"message of {case}","code":{{"code":"{code}"}},"spans":[{{"file_name":"{case}\\src\\main.rs","line_start":{line},"is_primary":true,"label":null}}],"children":[],"rendered":"rendered {case}\n"}}}}"#,
        manifest(case)
    )
}

fn artifact(case: &str) -> String {
    format!(
        r#"{{"reason":"compiler-artifact","package_id":"path+file:///{ROOT}/tests/ui/{case}#0.0.0","manifest_path":"{}","target":{{"kind":["bin"],"name":"{case}"}},"fresh":false}}"#,
        manifest(case)
    )
}

fn stream(lines: &[String]) -> String {
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

fn fails(case: &str, code: &str, line: u64) -> Case {
    let wanted = parse_expected(&format!(
        "code: {code}\nfile: {case}/src/main.rs\nline: {line}\nmessage: message of {case}\n"
    ))
    .unwrap();
    Case {
        name: case.to_owned(),
        expectation: Expectation::Fails(wanted),
        env: Vec::new(),
        features: false,
    }
}

fn compiles(case: &str) -> Case {
    Case {
        name: case.to_owned(),
        expectation: Expectation::Compiles,
        env: Vec::new(),
        features: false,
    }
}

#[test]
fn a_fixture_is_named_by_its_manifest_path_only() {
    assert_eq!(
        fixture_of("C:/work/repo/tests/ui/array-ok/Cargo.toml").as_deref(),
        Some("array-ok")
    );
    // Backslashes as on Windows, and a repository whose own path holds `tests/ui`.
    assert_eq!(
        fixture_of(r"C:\work\tests\ui\repo\tests\ui\array-ok\Cargo.toml").as_deref(),
        Some("array-ok")
    );
    for other in [
        "C:/work/repo/crates/gust/Cargo.toml",
        "C:/work/repo/tests/ui/Cargo.toml",
        "C:/work/repo/tests/ui/array-ok/inner/Cargo.toml",
        "C:/work/repo/tests/ui/array-ok/src/main.rs",
        "C:/work/repo/tests/reexport/outer/Cargo.toml",
        "",
    ] {
        assert_eq!(fixture_of(other), None, "{other}");
    }
}

#[test]
fn the_stream_is_cut_per_package_and_dependencies_belong_to_none() {
    let dependency = format!(
        r#"{{"reason":"compiler-message","manifest_path":"{ROOT}/crates/gust/Cargo.toml","target":{{"name":"gust"}},"message":{{"level":"error","message":"m","spans":[],"children":[],"rendered":"r"}}}}"#
    );
    let text = stream(&[
        message("a", "a", "error", "E1", 3),
        dependency,
        "plain text from cargo".to_owned(),
        r#"{"reason":"build-script-executed","package_id":"x"}"#.to_owned(),
        message("b", "b", "warning", "W1", 4),
        artifact("b"),
        r#"{"reason":"compiler-message","manifest_path":"#.to_owned(),
        // Parseable, but without a `manifest_path`: it belongs to no fixture.
        r#"{"reason":"compiler-message","message":{"level":"error","message":"m","spans":[],"children":[],"rendered":"r"}}"#.to_owned(),
        r#"{"reason":"build-finished","success":false}"#.to_owned(),
    ]);
    let packages = split_by_package(&text);
    assert_eq!(packages.keys().collect::<Vec<_>>(), ["a", "b"]);
    let a = &packages["a"];
    assert_eq!((a.artifacts, a.errors), (0, 1));
    assert_eq!(
        a.stdout,
        format!("{}\n", message("a", "a", "error", "E1", 3))
    );
    assert!(!a.compiled());
    let b = &packages["b"];
    assert_eq!((b.artifacts, b.errors), (1, 0));
    assert_eq!(
        b.stdout,
        format!("{}\n", message("b", "b", "warning", "W1", 4))
    );
    assert!(b.compiled());
}

#[test]
fn an_error_on_any_target_of_a_package_beats_its_artifact() {
    let text = stream(&[artifact("a"), message("a", "a-sibling", "error", "E1", 3)]);
    let package = &split_by_package(&text)["a"];
    assert_eq!((package.artifacts, package.errors), (1, 1));
    assert!(!package.compiled());
    assert!(!PackageStream::default().compiled());
}

#[test]
fn a_message_for_package_a_never_judges_fixture_b() {
    // `a` fails with E1 at line 3, as `a` and `b` both expect; `b` printed nothing but its artifact.
    let packages = split_by_package(&stream(&[
        message("a", "a", "error", "E1", 3),
        artifact("b"),
    ]));
    let (a, b) = (fails("a", "E1", 3), fails("b", "E1", 3));
    assert_eq!(
        judge_shared(&a, packages.get("a"), "", LABEL),
        Vec::<String>::new()
    );
    let problems = judge_shared(&b, packages.get("b"), "", LABEL);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0]
            .starts_with("b: has an expected.txt so `cargo check` must fail, but it compiled"),
        "{problems:?}"
    );
}

#[test]
fn a_wrong_expectation_is_named_by_fixture_and_the_others_still_pass() {
    let packages = split_by_package(&stream(&[
        message("a", "a", "error", "E1", 3),
        message("b", "b", "error", "E1", 3),
    ]));
    let wrong = fails("b", "E2", 3);
    let problems = judge_shared(&wrong, packages.get("b"), "", LABEL);
    assert!(
        problems[0].starts_with("b: the rustc error does not satisfy `code: E2` of expected.txt"),
        "{problems:?}"
    );
    assert!(judge_shared(&fails("a", "E1", 3), packages.get("a"), "", LABEL).is_empty());
}

#[test]
fn a_package_missing_from_the_stream_is_a_failure_never_a_pass() {
    let packages = split_by_package(&stream(&[artifact("other")]));
    let stderr = stderr_of(
        "error: could not compile `gone` (bin \"gone\") due to 1 previous error\nnoise about `else`\n",
        "gone",
    );
    for case in [compiles("gone"), fails("gone", "E1", 3)] {
        let problems = judge_shared(&case, packages.get("gone"), &stderr, LABEL);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].starts_with(
                "gone: the shared `cargo check` printed no compiler-message and no compiler-artifact for it, so it was not checked; cargo said:\n"
            ),
            "{problems:?}"
        );
        assert!(
            problems[0].contains("could not compile `gone`"),
            "{problems:?}"
        );
        assert!(!problems[0].contains("noise"), "{problems:?}");
    }
}

#[test]
fn a_fixture_that_must_compile_needs_an_artifact_and_no_error() {
    let packages = split_by_package(&stream(&[
        artifact("ok"),
        message("ok", "ok", "warning", "W1", 1),
        message("bad", "bad", "error", "E1", 2),
    ]));
    assert!(judge_shared(&compiles("ok"), packages.get("ok"), "", LABEL).is_empty());
    let problems = judge_shared(&compiles("bad"), packages.get("bad"), "", LABEL);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].starts_with(
            "bad: has no expected.txt so it must compile, but `cargo check` failed:\n"
        ),
        "{problems:?}"
    );
    assert!(problems[0].contains("rendered bad"), "{problems:?}");
}

#[test]
fn only_plain_fixtures_share_the_call() {
    assert!(shares_call(&compiles("a")));
    assert!(shares_call(&fails("a", "E1", 3)));
    let mut with_env = compiles("a");
    with_env.env = vec![("GUST_PROFILE".to_owned(), "native".to_owned())];
    assert!(!shares_call(&with_env));
    let build_error = Case {
        name: "a".to_owned(),
        expectation: parse_expectation("build-error: x\n").unwrap(),
        env: Vec::new(),
        features: false,
    };
    assert!(!shares_call(&build_error));
    let mut with_features = compiles("a");
    with_features.features = true;
    assert!(!shares_call(&with_features));
}

fn with_env(case: &str, variables: &[(&str, &str)]) -> Case {
    let mut case = compiles(case);
    case.env = variables
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    case
}

fn group_names(cases: &[Case]) -> Vec<Vec<String>> {
    let refs: Vec<&Case> = cases.iter().collect();
    group_by_env(&refs)
        .iter()
        .map(|group| group.iter().map(|case| case.name.clone()).collect())
        .collect()
}

#[test]
fn fixtures_of_the_same_env_and_no_features_share_a_group_in_name_order() {
    let cpu = [("GUST_PROFILE", "cpu")];
    let cases = [
        with_env("a", &cpu),
        with_env("b", &[("GUST_PROFILE", "native")]),
        with_env("c", &cpu),
        with_env("d", &cpu),
    ];
    assert_eq!(
        group_names(&cases),
        [vec!["a", "c", "d"], vec!["b"]],
        "groups come in the order of their first fixture"
    );
    assert_eq!(group_names(&cases), group_names(&cases), "stable");
}

#[test]
fn the_order_of_the_variables_does_not_split_a_group() {
    let cases = [
        with_env("a", &[("X", "1"), ("Y", "2")]),
        with_env("b", &[("Y", "2"), ("X", "1")]),
        with_env("c", &[("X", "1"), ("Y", "3")]),
        with_env("d", &[("X", "1")]),
    ];
    assert_eq!(group_names(&cases), [vec!["a", "b"], vec!["c"], vec!["d"]]);
}

#[test]
fn features_and_build_errors_stay_alone_even_with_the_same_env() {
    let cpu = [("GUST_PROFILE", "cpu")];
    let mut features = with_env("b", &cpu);
    features.features = true;
    let mut build_error = with_env("c", &cpu);
    build_error.expectation = parse_expectation("build-error: x\n").unwrap();
    let cases = [
        with_env("a", &cpu),
        features,
        build_error,
        with_env("d", &cpu),
    ];
    assert!(groups_call(&cases[0]) && !groups_call(&cases[1]) && !groups_call(&cases[2]));
    assert_eq!(group_names(&cases), [vec!["a", "d"], vec!["b"], vec!["c"]]);
}

#[test]
fn a_group_call_checks_its_packages_with_keep_going_and_the_group_env_wins_over_the_workspace_profile()
 {
    assert_eq!(
        group_args(&["a", "b"]),
        [
            "check",
            "--locked",
            "--keep-going",
            "--color",
            "never",
            "--manifest-path",
            "tests/ui/Cargo.toml",
            "-p",
            "a",
            "-p",
            "b",
            "--message-format",
            "json"
        ]
    );
    let env = [("GUST_PROFILE".to_owned(), "cpu".to_owned())];
    let command = group_command(std::path::Path::new("ROOT"), &["a", "b"], &env);
    let given: Vec<_> = command.get_args().map(ToOwned::to_owned).collect();
    let wanted: Vec<std::ffi::OsString> = group_args(&["a", "b"])
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(given, wanted);
    let profile: Vec<_> = command
        .get_envs()
        .filter(|(name, _)| *name == "GUST_PROFILE")
        .collect();
    assert_eq!(profile.len(), 1);
    assert_eq!(profile[0].1, Some(std::ffi::OsStr::new("cpu")));
}

#[test]
fn the_shared_call_is_one_keep_going_check_of_the_workspace_less_the_own_call_fixtures_under_the_dev_profile()
 {
    let args = shared_args(&["a", "b"]);
    assert_eq!(
        args,
        [
            "check",
            "--locked",
            "--keep-going",
            "--color",
            "never",
            "--manifest-path",
            "tests/ui/Cargo.toml",
            "--workspace",
            "--exclude",
            "a",
            "--exclude",
            "b",
            "--message-format",
            "json"
        ]
    );
    let command = shared_command(std::path::Path::new("ROOT"), &["a", "b"]);
    let given: Vec<_> = command.get_args().map(ToOwned::to_owned).collect();
    let wanted: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    assert_eq!(given, wanted);
    let profile: Vec<_> = command
        .get_envs()
        .filter(|(name, _)| *name == "GUST_PROFILE")
        .map(|(_, value)| value.map(ToOwned::to_owned))
        .collect();
    assert_eq!(profile, [Some(std::ffi::OsString::from("dev"))]);
}

#[test]
fn stderr_lines_are_kept_for_the_package_they_name_only() {
    let stderr = "failed to run custom build command for `a v0.0.0 (C:/x)`\nerror: could not compile `a` (bin \"a\")\nerror: could not compile `ab` (bin \"ab\")\n";
    assert_eq!(
        stderr_of(stderr, "a"),
        "failed to run custom build command for `a v0.0.0 (C:/x)`\nerror: could not compile `a` (bin \"a\")\n"
    );
    assert_eq!(stderr_of(stderr, "zzz").trim(), "");
}

#[test]
fn a_json_escaped_windows_manifest_path_names_its_fixture() {
    // As cargo prints it on Windows: the backslashes are escaped in the JSON text.
    let line = r#"{"reason":"compiler-artifact","manifest_path":"C:\\work\\repo\\tests\\ui\\a\\Cargo.toml","target":{"name":"a"}}"#;
    let packages = split_by_package(line);
    assert_eq!(packages.keys().collect::<Vec<_>>(), ["a"]);
    assert!(packages["a"].compiled());
}

#[test]
fn a_cause_that_names_no_fixture_is_carried_with_cargos_stderr() {
    // `gust` fails to compile: no fixture package prints anything, the dependency's error is no fixture's.
    let dependency = format!(
        r#"{{"reason":"compiler-message","manifest_path":"{ROOT}/crates/gust/Cargo.toml","target":{{"name":"gust"}},"message":{{"level":"error","message":"m","spans":[],"children":[],"rendered":"error: gust broke\n"}}}}"#
    );
    let stderr = "error: could not compile `gust-compute` (lib) due to 1 previous error\n";
    let cause = shared_cause(&stream(&[dependency, artifact("other")]), stderr, LABEL);
    assert!(
        cause.starts_with("tests/ui: the shared `cargo check` did not reach every fixture"),
        "{cause}"
    );
    assert!(cause.contains("    error: gust broke"), "{cause}");
    assert!(
        cause.contains("    error: could not compile `gust-compute`"),
        "{cause}"
    );
    // A fixture's own message is not repeated here.
    let own = shared_cause(&stream(&[message("a", "a", "error", "E1", 3)]), "", LABEL);
    assert!(!own.contains("rendered a"), "{own}");
}

#[test]
fn a_huge_cause_is_cut_to_200_lines_with_a_note() {
    let stderr: String = (0..250).map(|n| format!("line {n}\n")).collect();
    let cause = shared_cause("", &stderr, LABEL);
    assert!(cause.contains("    line 199\n"), "{cause}");
    assert!(!cause.contains("line 200"), "{cause}");
    assert!(
        cause.ends_with("    ... 50 more line(s) not shown"),
        "{cause}"
    );
}

#[test]
fn a_manifest_that_names_features_is_not_shared() {
    assert!(manifest_sets_features(
        "[dependencies]\ngust = { path = \"x\", features = [\"a\"] }\n"
    ));
    assert!(manifest_sets_features(
        "g = { path = \"x\", default-features = false }"
    ));
    assert!(!manifest_sets_features(
        "# features are not set here\n[dependencies]\ng = { path = \"x\" }\n"
    ));
}

#[test]
fn a_fixture_without_variables_or_with_features_only_never_joins_a_group() {
    let empty = compiles("a");
    let mut features = compiles("b");
    features.features = true;
    assert!(!groups_call(&empty), "no env.txt: the shared call takes it");
    assert!(
        !groups_call(&features),
        "features and no env.txt stay alone"
    );
    assert_eq!(group_names(&[empty, features]), [vec!["a"], vec!["b"]]);
}

#[test]
fn a_group_failure_names_its_group_and_not_the_shared_call() {
    let env = [
        ("GUST_PROFILE".to_owned(), "cpu".to_owned()),
        ("A".to_owned(), "1".to_owned()),
    ];
    let label = group_label(&env);
    assert_eq!(
        label,
        "the `cargo check` of the fixtures with A=1 GUST_PROFILE=cpu"
    );
    let packages = split_by_package("");
    let problems = judge_shared(&compiles("gone"), packages.get("gone"), "", &label);
    assert!(
        problems[0].starts_with(&format!("gone: {label} printed no")),
        "{problems:?}"
    );
    assert!(!problems[0].contains("shared"), "{problems:?}");
    let cause = shared_cause("", "", &label);
    assert!(
        cause.starts_with(&format!("tests/ui: {label} did not reach every fixture")),
        "{cause}"
    );
}

#[test]
fn a_group_call_carries_its_label_and_the_env_cargo_program() {
    let mut a = compiles("a");
    a.env = vec![
        ("GUST_PROFILE".to_owned(), "cpu".to_owned()),
        ("A".to_owned(), "1".to_owned()),
    ];
    let mut b = compiles("b");
    b.env = a.env.clone();
    let call = group_call(std::path::Path::new(ROOT), &[&a, &b]);
    assert_eq!(call.program, "env A=1 GUST_PROFILE=cpu cargo");
    assert_eq!(call.label, group_label(&a.env));
    assert_eq!(call.arguments, group_args(&["a", "b"]));
}
