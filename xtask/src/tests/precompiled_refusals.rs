use super::{super::artifacts::E4_EXPECTED, super::test_support::fixture_file, *};
use crate::workspace::workspace_root;

/// The refusal as cargo prints it when the build script of `gust` fails with `message`.
fn refusal_stderr(message: &str) -> String {
    format!(
        "error: failed to run custom build command for `gust-compute v0.1.0 (C:\\repo\\crates\\gust)`\n\nCaused by:\n  process didn't exit successfully\n  --- stderr\n  {message}\n"
    )
}

fn native_text() -> String {
    let text = fixture_file(E6_NATIVE_EXPECTED);
    text.lines()
        .find_map(|line| line.strip_prefix("build-error:"))
        .unwrap()
        .trim()
        .to_owned()
}

fn case(id: &str) -> super::E6Case {
    e6_cases().into_iter().find(|case| case.id == id).unwrap()
}

#[test]
fn e6_has_the_cases_of_the_plan_and_leaves_e6f_to_e4() {
    let ids: Vec<&str> = e6_cases().iter().map(|case| case.id).collect();
    assert_eq!(
        ids,
        [
            "E6a",
            "E6b",
            "E6c",
            "E6c-release",
            "E6d",
            "E6e",
            "E6g",
            "E6h",
            "E6i",
            "E6j"
        ]
    );
    // The refusals judge a message; the controls and the three dev-by-bridge cases compile.
    for case in e6_cases() {
        let refuses = matches!(case.outcome, E6Outcome::Refused { .. });
        assert_eq!(
            refuses,
            !matches!(case.id, "E6d" | "E6e" | "E6g" | "E6i" | "E6j"),
            "{}",
            case.id
        );
        let as_dev = matches!(case.outcome, E6Outcome::CompilesAsDev { .. });
        assert_eq!(
            as_dev,
            matches!(case.id, "E6g" | "E6i" | "E6j"),
            "{}",
            case.id
        );
        // Only E6i checks the whole workspace of the repository, only E6j has the dependency on cargo-gust.
        assert_eq!(case.workspace, case.id == "E6i", "{}", case.id);
        assert_eq!(case.tool_dependency, case.id == "E6j", "{}", case.id);
        // The cargo warning is required of E6j alone.
        if let E6Outcome::CompilesAsDev { ignored_tool, .. } = case.outcome {
            assert_eq!(ignored_tool, case.id == "E6j", "{}", case.id);
        }
    }
    assert_eq!(case("E6d").profile, E6Profile::Unset);
    assert!(case("E6d").release && !case("E6d").bridge_user);
    assert_eq!(case("E6e").profile, E6Profile::Dev);
    assert_eq!(case("E6e").features, Some("gust/slang-runtime"));
    assert!(case("E6c").bridge_user && case("E6c-release").bridge_user);
    assert!(!case("E6a").bridge_user && !case("E6b").bridge_user);
    // The bridge cases: release, no variable (E6g, E6i) or the explicit strict one (E6h), E6g and E6h on `app`.
    for id in ["E6g", "E6h", "E6i"] {
        assert!(case(id).release && !case(id).bridge_user, "{id}");
    }
    assert_eq!(case("E6g").profile, E6Profile::Unset);
    assert_eq!(case("E6h").profile, E6Profile::Native);
    assert_eq!(case("E6i").profile, E6Profile::Unset);
    assert_eq!(
        case("E6g").features,
        Some("gust/cargo-gust-bridge,gust/slang-runtime")
    );
    assert_eq!(case("E6h").features, case("E6g").features);
    assert_eq!(case("E6i").features, None);
    // E6j: a release build with no variable and no feature of its own, in a copy where app depends on cargo-gust.
    let j = case("E6j");
    assert!(j.release && j.features.is_none() && !j.bridge_user && j.in_copy());
    assert_eq!(j.profile, E6Profile::Unset);
}

#[test]
fn only_the_copies_of_e6c_and_e6j_run_without_locked() {
    for case in e6_cases() {
        let args = e6_args(&case, "M/Cargo.toml", "T");
        assert_eq!(
            args.iter().any(|arg| arg == "--locked"),
            !case.in_copy(),
            "{}: the fixture's own lock must never be regenerated, the copy's must be",
            case.id
        );
        assert_eq!(args.iter().any(|arg| arg == "--release"), case.release);
        assert_eq!(
            args.windows(2).any(|pair| pair == ["-p", "app"]),
            !case.workspace,
            "{}",
            case.id
        );
        assert_eq!(
            args.iter().any(|arg| arg == "--workspace"),
            case.workspace,
            "{}",
            case.id
        );
        assert_eq!(args.last().map(String::as_str), Some("T"));
        match case.features {
            Some(features) => assert!(
                args.windows(2)
                    .any(|pair| pair[0] == "--features" && pair[1] == features)
            ),
            None => assert!(!args.iter().any(|arg| arg == "--features")),
        }
    }
}

#[test]
fn the_e6_expectation_files_are_build_errors_and_differ_by_the_inferred_sentence() {
    let native = fixture_file(E6_NATIVE_EXPECTED);
    let inferred = fixture_file(E6_INFERRED_EXPECTED);
    for text in [&native, &inferred] {
        assert!(matches!(
            crate::ui::parse_expectation(text),
            Ok(crate::ui::Expectation::BuildFails(texts)) if texts.len() == 1
        ));
    }
    assert!(!native.contains(INFERRED_SENTENCE));
    assert!(inferred.contains(INFERRED_SENTENCE));
    assert!(inferred.contains(&native_text()));
}

#[test]
fn the_native_refusal_passes_and_a_changed_expectation_fails() {
    let message = native_text();
    let stderr = refusal_stderr(&message);
    let expected = fixture_file(E6_NATIVE_EXPECTED);
    let a = case("E6a");
    assert_eq!(
        e6_problems(&a, Some(&expected), false, "", &stderr),
        Ok(Vec::new())
    );
    // One character of the expectation is changed: the exact message no longer matches.
    let changed = expected.replace("refuses feature", "refuse feature");
    let problems = e6_problems(&a, Some(&changed), false, "", &stderr).unwrap();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("does not contain")),
        "{problems:?}"
    );
    // A build that compiles is no refusal.
    let problems = e6_problems(&a, Some(&expected), true, "", "").unwrap();
    assert!(
        problems.iter().any(|problem| problem.contains("compiled")),
        "{problems:?}"
    );
    // A failure that is not in `gust`'s build script is none either.
    let problems = e6_problems(
        &a,
        Some(&expected),
        false,
        "",
        &format!("error: nope\n{message}"),
    )
    .unwrap();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("not in the build script")),
        "{problems:?}"
    );
}

#[test]
fn the_explicit_refusal_must_not_carry_the_inferred_sentence_and_the_inferred_one_must() {
    let native = native_text();
    let inferred_message = format!("{native} {INFERRED_SENTENCE}");
    let native_file = fixture_file(E6_NATIVE_EXPECTED);
    let inferred_file = fixture_file(E6_INFERRED_EXPECTED);
    // The native text is a substring of the inferred message, so only the sentence check tells them apart.
    let problems = e6_problems(
        &case("E6a"),
        Some(&native_file),
        false,
        "",
        &refusal_stderr(&inferred_message),
    )
    .unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("must not have"), "{problems:?}");
    assert_eq!(
        e6_problems(
            &case("E6c-release"),
            Some(&inferred_file),
            false,
            "",
            &refusal_stderr(&inferred_message)
        ),
        Ok(Vec::new())
    );
    let problems = e6_problems(
        &case("E6c-release"),
        Some(&inferred_file),
        false,
        "",
        &refusal_stderr(&native),
    )
    .unwrap();
    assert!(problems.len() >= 2, "{problems:?}");
    assert!(
        problems.iter().any(|problem| problem.contains("lacks")),
        "{problems:?}"
    );
}

#[test]
fn rustc_keys_in_an_e6_expectation_are_refused() {
    let keys = fixture_file(E4_EXPECTED);
    let error = e6_problems(&case("E6a"), Some(&keys), false, "", "").unwrap_err();
    assert!(error.contains("build-error"), "{error}");
}

#[test]
fn the_positive_controls_must_compile_and_d_must_show_the_inference() {
    let d = case("E6d");
    let warning = format!("warning: gust-compute@0.1.0: {INFERRED_WARNING}\n");
    assert_eq!(e6_problems(&d, None, true, "", &warning), Ok(Vec::new()));
    let problems = e6_problems(&d, None, true, "", "").unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("did not print"), "{problems:?}");
    let problems = e6_problems(&d, None, false, "", &warning).unwrap();
    assert!(!problems.is_empty());
    let e = case("E6e");
    assert_eq!(e6_problems(&e, None, true, "", ""), Ok(Vec::new()));
    assert!(
        !e6_problems(&e, None, false, "", &refusal_stderr(&native_text()))
            .unwrap()
            .is_empty()
    );
}

fn bridge_warning_text() -> String {
    parse_warning_expectation(&fixture_file(E6_BRIDGE_EXPECTED))
        .unwrap()
        .build_script
        .remove(0)
}

fn cargo_warning_text() -> String {
    parse_warning_expectation(&fixture_file(E6_BRIDGE_EXPECTED))
        .unwrap()
        .cargo
        .remove(0)
}

#[test]
fn the_bridge_expectation_is_the_warning_of_the_build_script() {
    let wanted = parse_warning_expectation(&fixture_file(E6_BRIDGE_EXPECTED)).unwrap();
    assert_eq!(
        wanted.build_script,
        ["gust: cargo-gust-bridge treats a release build without GUST_PROFILE as dev"]
    );
    assert_eq!(
        wanted.cargo,
        ["ignoring invalid dependency `cargo-gust` which is missing a lib target"]
    );
    // The same text, with its `cargo:warning=` prefix, is what `crates/gust/build.rs` prints.
    let script =
        fs::read_to_string(workspace_root().unwrap().join("crates/gust/build.rs")).unwrap();
    assert!(script.contains(&format!("cargo:warning={}", wanted.build_script[0])));
    // The parser takes the two kinds of warning and nothing else; a `warning:` line is required.
    assert!(parse_warning_expectation("").is_err());
    assert!(parse_warning_expectation("# only a comment\n").is_err());
    assert!(parse_warning_expectation("warning:\n").is_err());
    assert!(parse_warning_expectation("build-error: x\n").is_err());
    assert!(parse_warning_expectation("cargo-warning: x\n").is_err());
    assert!(parse_warning_expectation("warning: a\ncargo-warning:\n").is_err());
}

#[test]
fn a_warning_counts_only_as_a_whole_line_after_cargos_prefix() {
    let text = "gust: cargo-gust-bridge treats a release build without GUST_PROFILE as dev";
    let line = format!("warning: gust-compute@0.1.0: {text}\n");
    assert!(has_build_script_warning(&line, text));
    assert!(has_build_script_warning(&format!("x\n{line}y\n"), text));
    assert!(has_build_script_warning(&line.replace('\n', "\r\n"), text));
    // A longer line, a shorter one, another package and another prefix do not count.
    assert!(!has_build_script_warning(
        &format!("warning: gust-compute@0.1.0: {text}X\n"),
        text
    ));
    assert!(!has_build_script_warning(
        &format!("warning: gust-compute@0.1.0: Y{text}\n"),
        text
    ));
    assert!(!has_build_script_warning(
        &format!("warning: gust-compute-wgpu@0.1.0: {text}\n"),
        text
    ));
    assert!(!has_build_script_warning(
        &format!("note: gust-compute@0.1.0: {text}\n"),
        text
    ));
    assert!(!has_build_script_warning(
        &format!("  warning: gust-compute@0.1.0: {text}\n"),
        text
    ));
    assert!(!has_build_script_warning(text, text));
    let cargo = "ignoring invalid dependency `cargo-gust` which is missing a lib target";
    assert!(has_cargo_warning(&format!("warning: {cargo}\n"), cargo));
    assert!(!has_cargo_warning(&format!("warning: {cargo} X\n"), cargo));
    assert!(!has_cargo_warning(
        &format!("warning: gust-compute@0.1.0: {cargo}\n"),
        cargo
    ));
    // Named after the package that declares the dependency, as cargo prints it (the path may hold parentheses).
    let named = format!("warning: app v0.0.0 (C:/x (y)/app) {cargo}\n");
    assert!(has_cargo_warning(&named, cargo));
    assert!(!has_cargo_warning(&named.replace('\n', " X\n"), cargo));
    assert!(!has_cargo_warning(
        &named.replace("app v", "other v"),
        cargo
    ));
}

#[test]
fn the_dev_by_bridge_cases_must_compile_print_the_warning_and_not_the_native_one() {
    let expected = fixture_file(E6_BRIDGE_EXPECTED);
    let warning = format!("warning: gust-compute@0.1.0: {}\n", bridge_warning_text());
    let ignored = format!("warning: {}\n", cargo_warning_text());
    for id in ["E6g", "E6i", "E6j"] {
        let bridged = case(id);
        // E6j also needs cargo's own warning about the dependency; the others do not look for it.
        let complete = if id == "E6j" {
            format!("{ignored}{warning}")
        } else {
            warning.clone()
        };
        assert_eq!(
            e6_problems(&bridged, Some(&expected), true, "", &complete),
            Ok(Vec::new()),
            "{id}"
        );
        // A silent change fails the case: no bridge warning at all.
        let silent = if id == "E6j" {
            ignored.clone()
        } else {
            String::new()
        };
        let problems = e6_problems(&bridged, Some(&expected), true, "", &silent).unwrap();
        assert_eq!(problems.len(), 1, "{id}: {problems:?}");
        assert!(problems[0].contains("did not print"), "{id}: {problems:?}");
        // A warning that only starts like the expected one is no match (the line must be the same).
        let longer = format!(
            "warning: gust-compute@0.1.0: {}X\n{silent}",
            bridge_warning_text()
        );
        let problems = e6_problems(&bridged, Some(&expected), true, "", &longer).unwrap();
        assert_eq!(problems.len(), 1, "{id}: {problems:?}");
        // The warning of the plain inference means the profile was native and not made dev.
        let native = format!("warning: gust-compute@0.1.0: {INFERRED_WARNING}\n{complete}");
        let problems = e6_problems(&bridged, Some(&expected), true, "", &native).unwrap();
        assert_eq!(problems.len(), 1, "{id}: {problems:?}");
        assert!(
            problems[0].contains("inferred as native"),
            "{id}: {problems:?}"
        );
        // A build that fails is no pass, with or without the warning.
        let problems = e6_problems(&bridged, Some(&expected), false, "", &complete).unwrap();
        assert!(!problems.is_empty(), "{id}");
        // A missing or malformed expectation is an error of the check itself.
        assert!(e6_problems(&bridged, None, true, "", &complete).is_err());
        assert!(e6_problems(&bridged, Some("build-error: x"), true, "", &complete).is_err());
    }
    // E6j without cargo's warning is not the dependency it pins, whatever else is printed.
    let problems = e6_problems(&case("E6j"), Some(&expected), true, "", &warning).unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`cargo-gust`"), "{problems:?}");
    // An expectation without a `cargo-warning:` line cannot serve E6j.
    let error = e6_problems(&case("E6j"), Some("warning: x"), true, "", &warning).unwrap_err();
    assert!(error.contains("cargo-warning"), "{error}");
}

#[test]
fn the_explicit_native_refusal_with_the_bridge_feature_is_the_one_of_e6a() {
    let h = case("E6h");
    let native_file = fixture_file(E6_NATIVE_EXPECTED);
    let message = native_text();
    assert_eq!(
        e6_problems(&h, Some(&native_file), false, "", &refusal_stderr(&message)),
        Ok(Vec::new())
    );
    // Accepted as dev (what E6g does) is the failure of E6h.
    let warning = format!("warning: gust-compute@0.1.0: {}\n", bridge_warning_text());
    let problems = e6_problems(&h, Some(&native_file), true, "", &warning).unwrap();
    assert!(
        problems.iter().any(|problem| problem.contains("compiled")),
        "{problems:?}"
    );
}

#[test]
fn the_bridge_user_copy_is_built_from_the_fixture_files() {
    let manifest = bridge_user_manifest("C:/repo");
    assert!(manifest.contains("path = \"C:/repo/crates/gust-wgpu\""));
    assert!(manifest.contains("features = [\"slang-runtime\"]"));
    let workspace = with_bridge_user_member(&fixture_file("Cargo.toml")).unwrap();
    assert!(workspace.contains("\"kernels\", \"app\", \"bridge-user\""));
    let app = with_bridge_user_dependency(&fixture_file("app/Cargo.toml")).unwrap();
    assert!(app.contains("bridge-user = { path = \"../bridge-user\" }"));
    assert!(with_bridge_user_member("members = []").is_err());
    assert!(with_bridge_user_dependency("[dependencies]").is_err());
}

#[test]
fn the_tool_dependency_of_e6j_is_a_normal_path_dependency_on_cargo_gust() {
    let app = with_tool_dependency(&fixture_file("app/Cargo.toml"), "C:/repo").unwrap();
    assert!(app.contains("cargo-gust = { path = \"C:/repo/crates/cargo-gust\" }"));
    // It sits in `[dependencies]`, beside `kernels`, and nowhere else.
    let dependencies = app.split("[dependencies]").nth(1).unwrap();
    let before_next_table = dependencies.split("\n[").next().unwrap();
    assert!(before_next_table.contains("cargo-gust = "));
    assert_eq!(app.matches("cargo-gust = ").count(), 1);
    assert!(with_tool_dependency("[dependencies]", "C:/repo").is_err());
}
