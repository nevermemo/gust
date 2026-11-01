use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use super::{
    DUAL_DIR, DUAL_RUSTFLAGS, DUAL_TESTS, DualKind, DualMode, DualStep, RUSTFLAGS_VARIABLES,
    dual_command, dual_dir_under, dual_problems, dual_steps, explain_check, parse_dual_args,
    refuse_preset_rustflags, rustflags_config_files, target_rustflags_variables, tests_not_ok,
};
use crate::{ui::parse_expected, workspace::workspace_root};

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

fn steps() -> Vec<DualStep> {
    dual_steps("TARGET")
}

fn step(id: &str) -> DualStep {
    steps().into_iter().find(|step| step.id == id).unwrap()
}

// ---- the refusal ----

/// A lookup that finds exactly the listed variables, with the given values.
fn environment(set: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
    let set: Vec<(String, OsString)> = set
        .iter()
        .map(|(name, value)| ((*name).to_owned(), OsString::from(value)))
        .collect();
    move |name| {
        set.iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value.clone())
    }
}

#[test]
fn a_clean_environment_is_not_refused() {
    assert_eq!(refuse_preset_rustflags(environment(&[])), Ok(()));
    // Unrelated variables, including look-alikes, do not count.
    assert_eq!(
        refuse_preset_rustflags(environment(&[
            ("RUSTDOCFLAGS", "-D warnings"),
            ("CARGO_BUILD_JOBS", "4"),
            ("CARGO_TARGET_DIR", "x"),
            ("RUSTC_WRAPPER", ""),
        ])),
        Ok(())
    );
}

#[test]
fn each_rustflags_variable_is_refused_by_name_even_when_empty() {
    assert_eq!(
        RUSTFLAGS_VARIABLES,
        [
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_RUSTFLAGS"
        ]
    );
    for name in RUSTFLAGS_VARIABLES {
        for value in ["-C target-cpu=native", "", " "] {
            let error = refuse_preset_rustflags(environment(&[(name, value)])).unwrap_err();
            assert!(error.contains(name), "{name}={value:?}: {error}");
            assert!(error.contains("refuses to run"), "{error}");
            assert!(error.contains("an empty value counts"), "{error}");
            // Only the variable that is set is named as set.
            for other in RUSTFLAGS_VARIABLES.iter().filter(|other| **other != name) {
                let named = error.split(" because ").nth(1).unwrap();
                let set_part = named.split(" in the environment").next().unwrap();
                // `RUSTFLAGS` is a substring of the other names: compare whole comma-separated items.
                assert!(
                    !set_part.split(", ").any(|item| item == *other),
                    "{name} alone must not name {other}: {error}"
                );
            }
        }
    }
}

#[test]
fn every_set_variable_is_named_when_several_are_set() {
    let error = refuse_preset_rustflags(environment(&[
        ("CARGO_BUILD_RUSTFLAGS", ""),
        ("RUSTFLAGS", "-Dwarnings"),
    ]))
    .unwrap_err();
    assert!(
        error.contains("RUSTFLAGS, CARGO_BUILD_RUSTFLAGS are set"),
        "{error}"
    );
    assert!(!error.contains("CARGO_ENCODED_RUSTFLAGS is set"), "{error}");
}

// ---- the child processes ----

/// What the spawned child process is given: its program, arguments, working directory and environment changes,
/// read from the real `Command` that `run_dual_step` runs (a removed variable is `(key, None)`).
#[allow(clippy::type_complexity)]
fn spawned(
    step: &DualStep,
) -> (
    OsString,
    Vec<OsString>,
    Option<PathBuf>,
    Vec<(OsString, Option<OsString>)>,
) {
    let command = dual_command(Path::new("ROOT"), step);
    let mut envs: Vec<(OsString, Option<OsString>)> = command
        .get_envs()
        .map(|(key, value)| (key.to_owned(), value.map(ToOwned::to_owned)))
        .collect();
    envs.sort();
    (
        command.get_program().to_owned(),
        command.get_args().map(ToOwned::to_owned).collect(),
        command.get_current_dir().map(ToOwned::to_owned),
        envs,
    )
}

#[test]
fn the_spawned_child_has_no_rustflags_variable_but_the_cfg_on_the_cfg_steps() {
    assert_eq!(DUAL_RUSTFLAGS, "--cfg\u{1f}gust_dual");
    assert_eq!(DUAL_RUSTFLAGS.as_bytes()[5], 0x1f);
    let cfg: Vec<(&str, bool)> = steps().iter().map(|step| (step.id, step.cfg)).collect();
    assert_eq!(
        cfg,
        [
            ("build", true),
            ("E1", false),
            ("E2-compile", true),
            ("E2-run", true),
            ("E3", false)
        ]
    );
    let os = |text: &str| OsString::from(text);
    // Every step, cfg or not, removes all three variables (absent, not inherited) ...
    let removed = |name: &str| (os(name), None);
    let without_cfg = vec![
        removed("CARGO_BUILD_RUSTFLAGS"),
        removed("CARGO_ENCODED_RUSTFLAGS"),
        removed("RUSTFLAGS"),
    ];
    // ... and the cfg steps then set the encoded one, exactly `--cfg` 0x1f `gust_dual`.
    let with_cfg = vec![
        removed("CARGO_BUILD_RUSTFLAGS"),
        (
            os("CARGO_ENCODED_RUSTFLAGS"),
            Some(os("--cfg\u{1f}gust_dual")),
        ),
        removed("RUSTFLAGS"),
    ];
    let program = std::env::var_os("CARGO").unwrap_or_else(|| os("cargo"));
    for step in steps() {
        let (child_program, args, directory, envs) = spawned(&step);
        assert_eq!(child_program, program, "{}", step.id);
        let wanted_args: Vec<OsString> = step.args.iter().map(|arg| os(arg)).collect();
        assert_eq!(args, wanted_args, "{}", step.id);
        assert_eq!(directory, Some(PathBuf::from("ROOT")), "{}", step.id);
        let expected = if step.cfg { &with_cfg } else { &without_cfg };
        assert_eq!(&envs, expected, "{}", step.id);
    }
}

#[test]
fn target_rustflags_variables_are_found_by_name_only() {
    let names = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
    let found = target_rustflags_variables(
        names(&[
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS",
            "cargo_target_aarch64_unknown_linux_gnu_rustflags",
            "CARGO_TARGET_DIR",
            "CARGO_TARGET__RUSTFLAGS",
            "CARGO_TARGET_X_RUSTDOCFLAGS",
            "CARGO_BUILD_RUSTFLAGS",
            "RUSTFLAGS",
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS",
        ])
        .into_iter(),
    );
    assert_eq!(
        found,
        [
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS"
        ]
    );
    assert!(target_rustflags_variables(names(&["PATH"]).into_iter()).is_empty());
}

#[test]
fn the_argument_lists_are_exactly_these() {
    let manifest = "tests/dual/Cargo.toml";
    let tail = ["--target-dir", "TARGET"];
    let expected: Vec<(&str, Vec<String>)> = vec![
        (
            "build",
            owned(&["build", "--locked", "--workspace", "--all-targets"]),
        ),
        (
            "E1",
            owned(&[
                "test",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
                "--message-format",
                "json",
            ]),
        ),
        (
            "E2-compile",
            owned(&[
                "test",
                "--locked",
                "--manifest-path",
                manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
            ]),
        ),
        (
            "E2-run",
            owned(&[
                "test",
                "--locked",
                "--manifest-path",
                manifest,
                "-p",
                "consumer",
                "--lib",
            ]),
        ),
        (
            "E3",
            owned(&[
                "check",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                manifest,
                "-p",
                "consumer",
                "--bin",
                "prod",
                "--message-format",
                "json",
            ]),
        ),
    ];
    let actual: Vec<(&str, Vec<String>)> = steps()
        .into_iter()
        .map(|step| {
            let mut args = step.args.clone();
            assert_eq!(args.split_off(args.len() - 2), tail, "{}", step.id);
            (step.id, args)
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn every_cargo_call_of_check_dual_is_locked() {
    for step in steps() {
        assert_eq!(
            step.args.iter().filter(|arg| *arg == "--locked").count(),
            1,
            "{}: {:?}",
            step.id,
            step.args
        );
        // The shown command line keeps the flag and hides the machine-specific directory.
        assert!(step.shown().contains("--locked"), "{}", step.id);
        assert!(!step.shown().contains("TARGET"), "{}", step.shown());
    }
}

#[test]
fn the_dual_build_has_its_own_target_directory_under_the_normal_one() {
    let root = Path::new("/repo");
    // Absolute: used as is; relative: read against the repository root, where the children run.
    assert_eq!(
        dual_dir_under(root, Path::new("/cache/target")),
        Path::new("/cache/target/dual")
    );
    assert_eq!(
        dual_dir_under(root, Path::new("out")),
        Path::new("/repo/out/dual")
    );
    assert_eq!(
        dual_dir_under(root, &root.join("target")),
        Path::new("/repo/target/dual")
    );
    // Every step passes that directory, last.
    for step in dual_steps("/cache/target/dual") {
        let last = &step.args[step.args.len() - 2..];
        assert_eq!(last, ["--target-dir", "/cache/target/dual"], "{}", step.id);
    }
}

#[test]
fn the_steps_run_in_the_order_build_e1_e2_e3_and_e2_is_judged_by_its_tests() {
    let ids: Vec<&str> = steps().iter().map(|step| step.id).collect();
    assert_eq!(ids, ["build", "E1", "E2-compile", "E2-run", "E3"]);
    assert_eq!(step("build").kind, DualKind::Build);
    assert_eq!(step("E2-compile").kind, DualKind::Compiles);
    assert_eq!(step("E2-run").kind, DualKind::Passes { tests: DUAL_TESTS });
    assert_eq!(
        DUAL_TESTS,
        [
            "tests::twin_runs",
            "tests::overflow_mode_is_the_dependents_own"
        ]
    );
    // E1 and E3 each name the file of their expectation and the target they judge.
    assert_eq!(
        step("E1").kind,
        DualKind::Fails {
            case: "consumer",
            target: "consumer",
            expected: "e1-expected.txt"
        }
    );
    assert_eq!(
        step("E3").kind,
        DualKind::Fails {
            case: "consumer",
            target: "prod",
            expected: "e3-expected.txt"
        }
    );
}

#[test]
fn only_check_only_or_no_argument_is_accepted() {
    assert_eq!(parse_dual_args(&[]), Ok(DualMode::Full));
    assert_eq!(
        parse_dual_args(&owned(&["--check-only"])),
        Ok(DualMode::CheckOnly)
    );
    for bad in [
        owned(&["--check"]),
        owned(&["check-only"]),
        owned(&["--check-only", "--check-only"]),
        owned(&["--check-only", "x"]),
    ] {
        assert!(parse_dual_args(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn e2_requires_each_test_to_be_reported_ok() {
    let ok = "running 2 tests\ntest tests::overflow_mode_is_the_dependents_own ... ok\ntest tests::twin_runs ... ok\n\ntest result: ok. 2 passed\n";
    assert!(tests_not_ok(ok, DUAL_TESTS).is_empty());
    // A run that found no test succeeds in cargo and must still fail here.
    assert_eq!(
        tests_not_ok("running 0 tests\n\ntest result: ok. 0 passed\n", DUAL_TESTS),
        DUAL_TESTS
    );
    // Colour codes that a forced colour output puts around `ok` (CI sets CARGO_TERM_COLOR=always) are ignored.
    let colored = "test tests::twin_runs ... \u{1b}[32mok\u{1b}[0m\ntest tests::overflow_mode_is_the_dependents_own ... \u{1b}[32mok\u{1b}[0m\n";
    assert!(tests_not_ok(colored, DUAL_TESTS).is_empty());
    assert_eq!(
        tests_not_ok(
            "test tests::twin_runs ... \u{1b}[31mFAILED\u{1b}[0m\n",
            &["tests::twin_runs"]
        ),
        ["tests::twin_runs"]
    );
    // Failed, ignored and filtered tests are not ok; a name that only contains another one does not count.
    let mixed = "test tests::twin_runs ... FAILED\ntest tests::overflow_mode_is_the_dependents_own ... ignored\n";
    assert_eq!(tests_not_ok(mixed, DUAL_TESTS), DUAL_TESTS);
    assert_eq!(
        tests_not_ok(
            "test other::tests::twin_runs ... ok\n",
            &["tests::twin_runs"]
        ),
        ["tests::twin_runs"]
    );
}

// ---- the rustflags note ----

#[test]
fn config_files_that_mention_rustflags_are_described_without_machine_paths() {
    let base = std::env::temp_dir().join(format!("gust-dual-config-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    let root = base.join("outer").join("repo");
    fs::create_dir_all(root.join(".cargo")).unwrap();
    fs::create_dir_all(base.join("outer").join(".cargo")).unwrap();
    fs::create_dir_all(base.join("home")).unwrap();
    // The repository's own config only has the alias (as the real one): no note.
    fs::write(
        root.join(".cargo/config.toml"),
        "[alias]\nxtask = \"run\"\n",
    )
    .unwrap();
    assert_eq!(
        rustflags_config_files(&root, Some(&base.join("home"))),
        Vec::<String>::new()
    );
    // A parent directory's config and the cargo home's both count; the key is matched case-insensitively.
    fs::write(
        base.join("outer/.cargo/config.toml"),
        "[build]\nrustflags = [\"-C\", \"target-cpu=native\"]\n",
    )
    .unwrap();
    fs::write(base.join("home/config.toml"), "[build]\nRUSTFLAGS = []\n").unwrap();
    let found = rustflags_config_files(&root, Some(&base.join("home")));
    // The workspace's own file too, in the old `config` spelling.
    fs::write(root.join(".cargo/config"), "target.x.rustflags = []\n").unwrap();
    let with_own = rustflags_config_files(&root, None);
    let _ = fs::remove_dir_all(&base);
    assert_eq!(
        found,
        [
            "1 level above the workspace: .cargo/config.toml",
            "CARGO_HOME: config.toml"
        ]
    );
    assert_eq!(
        with_own,
        [
            "the workspace: .cargo/config",
            "1 level above the workspace: .cargo/config.toml"
        ]
    );
    for line in found.iter().chain(&with_own) {
        assert!(!line.contains("gust-dual-config"), "{line}");
    }
}

#[test]
fn the_repository_has_no_rustflags_config_of_its_own() {
    // The note is for machines with such config; the repository itself must not add one (verified: only the alias).
    let root = workspace_root().unwrap();
    let text = fs::read_to_string(root.join(".cargo/config.toml")).unwrap();
    assert!(!text.to_ascii_lowercase().contains("rustflags"), "{text}");
}

// ---- the fixture workspace and its expectations ----

fn fixture(path: &str) -> String {
    let root = workspace_root().unwrap();
    fs::read_to_string(root.join(DUAL_DIR).join(path))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
}

#[test]
fn the_fixture_is_a_workspace_of_its_own_with_the_two_crates_and_a_lockfile() {
    let root = workspace_root().unwrap();
    let manifest = fixture("Cargo.toml");
    assert!(
        manifest.lines().any(|line| line.trim() == "[workspace]"),
        "{manifest}"
    );
    assert!(
        manifest.contains(r#"members = ["kernel-lib", "consumer"]"#),
        "{manifest}"
    );
    // The root workspace neither lists nor includes it, so `--workspace` never builds it.
    let root_manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(!root_manifest.contains("tests/dual"), "{root_manifest}");
    assert!(!root_manifest.contains("kernel-lib"), "{root_manifest}");
    // `--locked` needs the committed lockfile, which lists both crates.
    let lock = fixture("Cargo.lock");
    for name in ["kernel-lib", "consumer"] {
        assert!(lock.contains(&format!("name = \"{name}\"")), "{lock}");
    }
}

#[test]
fn both_crates_declare_the_cfg_and_the_production_binary_is_not_a_test_target() {
    for manifest in ["kernel-lib/Cargo.toml", "consumer/Cargo.toml"] {
        let text = fixture(manifest);
        assert!(
            text.contains("unexpected_cfgs = { level = \"warn\", check-cfg = ['cfg(gust_dual)'] }"),
            "{manifest}: {text}"
        );
        assert!(text.contains("[lints.rust]"), "{manifest}");
    }
    let consumer = fixture("consumer/Cargo.toml");
    assert!(consumer.contains("name = \"prod\""), "{consumer}");
    assert!(
        consumer.contains("path = \"src/bin/prod.rs\""),
        "{consumer}"
    );
    assert!(consumer.contains("test = false"), "{consumer}");
    assert!(consumer.contains("kernel-lib = { path = \"../kernel-lib\" }"));
}

#[test]
fn the_fixture_pins_the_overflow_mode_of_a_package_with_its_own_override() {
    let manifest = fixture("Cargo.toml");
    assert!(
        manifest.contains("[profile.dev.package.consumer]\noverflow-checks = false")
            || manifest.contains("[profile.dev.package.consumer]\r\noverflow-checks = false"),
        "{manifest}"
    );
    let source = fixture("consumer/src/lib.rs");
    assert!(source.contains("fn overflow_mode_is_the_dependents_own()"));
    assert!(source.contains("gust::cpu::overflow_checks!()"));
    assert!(source.contains("gust::cpu::overflow_checks()"));
    for name in DUAL_TESTS {
        let function = name.rsplit("::").next().unwrap();
        assert!(source.contains(&format!("fn {function}()")), "{name}");
    }
}

/// The expectation files, spelled out: editing a file alone fails this test.
#[test]
fn the_expectation_files_pin_one_e0277_on_the_dispatch_line_of_each_target() {
    let e1 = parse_expected(&fixture("e1-expected.txt")).unwrap();
    let e3 = parse_expected(&fixture("e3-expected.txt")).unwrap();
    let messages = [
        "`CpuDevice` cannot run the kernel `add`",
        "the trait `Run<add, 3>` is not implemented for `CpuDevice`",
    ];
    for (wanted, file, line) in [
        (&e1, "consumer/src/lib.rs", 23),
        (&e3, "consumer/src/bin/prod.rs", 13),
    ] {
        assert_eq!(wanted.code, "E0277");
        assert_eq!(wanted.file, file);
        assert_eq!(wanted.line, line);
        assert_eq!(wanted.messages, messages);
        assert_eq!(
            wanted.labels,
            ["this device has no executable form of the kernel"]
        );
        // The pinned line really is the dispatch call.
        let source = fixture(file);
        let text = source.lines().nth(line as usize - 1).unwrap_or_default();
        assert!(
            text.contains("kernels::add::dispatch(&device"),
            "{file}:{line}: {text}"
        );
    }
}

// ---- the matcher on captured cargo output ----

/// A `compiler-message` line like the real ones of `cargo test --message-format json` (trimmed to the fields the
/// matcher reads) for the cargo target `target`.
fn message(target: &str, level: &str, code: &str, file: &str, line: u32) -> String {
    let code = if code.is_empty() {
        "null".to_owned()
    } else {
        format!(r#"{{"code":"{code}","explanation":null}}"#)
    };
    format!(
        r#"{{"reason":"compiler-message","target":{{"kind":["lib"],"name":"{target}"}},"message":{{"rendered":"{level}: `CpuDevice` cannot run the kernel `add`\n","children":[{{"children":[],"code":null,"level":"help","message":"the trait `Run<add, 3>` is not implemented for `CpuDevice`","rendered":null,"spans":[]}}],"code":{code},"level":"{level}","message":"`CpuDevice` cannot run the kernel `add`","spans":[{{"file_name":"{}","is_primary":true,"label":"this device has no executable form of the kernel; GPU devices run every kernel that has a typed call","line_start":{line},"line_end":{line}}}]}}}}"#,
        file.replace('/', "\\\\")
    )
}

fn judged(id: &str, success: bool, lines: &[String]) -> Vec<String> {
    let root = workspace_root().unwrap();
    let stdout = lines.join("\n");
    dual_problems(&root, &step(id), success, &stdout, "").unwrap()
}

#[test]
fn e1_and_e3_accept_exactly_the_one_error_on_their_own_target() {
    let e1 = message("consumer", "error", "E0277", "consumer/src/lib.rs", 23);
    let e3 = message("prod", "error", "E0277", "consumer/src/bin/prod.rs", 13);
    assert_eq!(
        judged("E1", false, std::slice::from_ref(&e1)),
        Vec::<String>::new()
    );
    assert_eq!(
        judged("E3", false, std::slice::from_ref(&e3)),
        Vec::<String>::new()
    );
    // E3 judges the binary `prod`: the library's error beside it is a sibling target's, neither a second error
    // nor a stand-in, and the other way round for E1.
    assert_eq!(
        judged("E3", false, &[e1.clone(), e3.clone()]),
        Vec::<String>::new()
    );
    assert_eq!(
        judged("E1", false, &[e1.clone(), e3.clone()]),
        Vec::<String>::new()
    );
    assert!(!judged("E3", false, std::slice::from_ref(&e1)).is_empty());
    assert!(!judged("E1", false, std::slice::from_ref(&e3)).is_empty());
}

#[test]
fn e1_and_e3_reject_a_second_error_another_code_another_line_or_a_build_that_succeeds() {
    let e1 = message("consumer", "error", "E0277", "consumer/src/lib.rs", 23);
    let e3 = message("prod", "error", "E0277", "consumer/src/bin/prod.rs", 13);
    for (id, good) in [("E1", &e1), ("E3", &e3)] {
        // Two errors on the target: exactly-one is part of the contract.
        let second = good.replace("23", "30").replace("13", "20");
        let problems = judged(id, false, &[good.clone(), second]);
        assert!(
            problems[0].contains("expected.txt describes one error, but rustc reported 2 errors"),
            "{id}: {problems:?}"
        );
        // A different error code is not the missing-twin error.
        let other = good.replace("E0277", "E0599");
        let problems = judged(id, false, &[other]);
        assert!(
            problems[0].contains("does not satisfy `code: E0277`"),
            "{id}: {problems:?}"
        );
        // Another line (the error moved) is reported with the line that was expected.
        let moved = good
            .replace("\"line_start\":23", "\"line_start\":24")
            .replace("\"line_start\":13", "\"line_start\":14");
        let problems = judged(id, false, &[moved]);
        assert!(problems[0].contains("`line: "), "{id}: {problems:?}");
        // A different message is not the Run diagnostic.
        let reworded = good.replace("cannot run the kernel", "is broken for the kernel");
        assert!(!judged(id, false, &[reworded]).is_empty(), "{id}");
        // The call that must fail but succeeded is a problem, and so is one that failed without any error.
        let problems = judged(id, true, &[]);
        assert!(
            problems[0].contains("must fail, but it compiled"),
            "{id}: {problems:?}"
        );
        let problems = judged(id, false, &[]);
        assert!(
            problems[0].contains("printed no error-level compiler-message of its own"),
            "{id}: {problems:?}"
        );
    }
}

#[test]
fn a_step_without_an_expectation_file_has_no_problems_to_judge() {
    let root = workspace_root().unwrap();
    for id in ["build", "E2-compile", "E2-run"] {
        let error = dual_problems(&root, &step(id), false, "", "").unwrap_err();
        assert!(error.contains("has no expectation file"), "{error}");
    }
}

#[test]
fn check_dual_is_explained() {
    assert!(explain_check(Some("check-dual")).is_ok());
    assert!(explain_check(Some("all")).is_ok());
}
