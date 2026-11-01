use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use super::{
    DUAL_DIR, DUAL_RUSTFLAGS, DUAL_TESTS, DualKind, DualMode, DualStep, RUSTFLAGS_VARIABLES,
    dual_command, dual_dir_under, dual_problems, dual_run_steps, dual_steps, dual_steps_for,
    explain_check, parse_dual_args, parse_twin_match, refuse_preset_rustflags,
    rustflags_config_files, target_rustflags_variables, tests_not_ok, twin_run_problems,
};
use crate::{
    ui::parse_expected,
    verify::{VerifyMode, runs_check_dual},
    workspace::workspace_root,
};

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

/// Every step of the full mode: the `--check-only` steps, then the example runs.
fn steps() -> Vec<DualStep> {
    dual_steps_for(DualMode::Full, "TARGET")
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
            ("fmt", false),
            ("build", true),
            ("clippy", true),
            ("clippy-examples", true),
            ("E1", false),
            ("E2-compile", true),
            ("E2-run", true),
            ("E3", false),
            ("run-vector-add", true),
            ("run-polynomial", true),
            ("run-vector-add-no-cfg", false),
        ]
    );
    let os = |text: &str| OsString::from(text);
    // Every step, cfg or not, removes all three variables (absent, not inherited) ...
    let removed = |name: &str| (os(name), None);
    // ... and builds the workspace under the `dev` profile whatever the parent has (R4.2).
    let profile = (os("GUST_PROFILE"), Some(os("dev")));
    let without_cfg = vec![
        removed("CARGO_BUILD_RUSTFLAGS"),
        removed("CARGO_ENCODED_RUSTFLAGS"),
        profile.clone(),
        removed("RUSTFLAGS"),
    ];
    // ... and the cfg steps then set the encoded one, exactly `--cfg` 0x1f `gust_dual`.
    let with_cfg = vec![
        removed("CARGO_BUILD_RUSTFLAGS"),
        (
            os("CARGO_ENCODED_RUSTFLAGS"),
            Some(os("--cfg\u{1f}gust_dual")),
        ),
        profile,
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
        // `cargo fmt` builds nothing: no `--locked`, no target directory.
        (
            "fmt",
            owned(&["fmt", "--manifest-path", manifest, "--all", "--", "--check"]),
        ),
        (
            "build",
            owned(&["build", "--locked", "--workspace", "--all-targets"]),
        ),
        // The target directory goes before the `--` that starts the clippy arguments.
        (
            "clippy",
            owned(&[
                "clippy",
                "--locked",
                "--manifest-path",
                manifest,
                "--workspace",
                "--all-targets",
                "--target-dir",
                "TARGET",
                "--",
                "-D",
                "warnings",
            ]),
        ),
        // The same, for the two examples with a dual `main`, in the root workspace.
        (
            "clippy-examples",
            owned(&[
                "clippy",
                "--locked",
                "-p",
                "vector-add",
                "-p",
                "polynomial",
                "--all-targets",
                "--target-dir",
                "TARGET",
                "--",
                "-D",
                "warnings",
            ]),
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
    let actual: Vec<(&str, Vec<String>)> = dual_steps("TARGET")
        .into_iter()
        .map(|step| {
            let mut args = step.args.clone();
            // Where the directory sits depends on the step: last, before a `--` (clippy, listed whole above), or
            // nowhere (fmt).
            match step.id {
                "fmt" | "clippy" | "clippy-examples" => {}
                _ => assert_eq!(args.split_off(args.len() - 2), tail, "{}", step.id),
            }
            (step.id, args)
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn the_example_runs_are_exactly_these() {
    let tail = ["--target-dir", "TARGET"];
    let actual: Vec<(&str, Vec<String>, bool, DualKind)> = dual_run_steps("TARGET")
        .into_iter()
        .map(|step| {
            let mut args = step.args;
            assert_eq!(args.split_off(args.len() - 2), tail, "{}", step.id);
            (step.id, args, step.cfg, step.kind)
        })
        .collect();
    assert_eq!(
        actual,
        [
            (
                "run-vector-add",
                owned(&["run", "--locked", "-p", "vector-add"]),
                true,
                DualKind::Runs {
                    twin: true,
                    also: &[]
                }
            ),
            (
                "run-polynomial",
                owned(&["run", "--locked", "-p", "polynomial"]),
                true,
                DualKind::Runs {
                    twin: true,
                    also: &["CPU twin time: ", "debug profile, single-threaded"]
                }
            ),
            (
                "run-vector-add-no-cfg",
                owned(&["run", "--locked", "-p", "vector-add"]),
                false,
                DualKind::Runs {
                    twin: false,
                    also: &[]
                }
            ),
        ]
    );
}

#[test]
fn the_check_only_mode_has_no_example_run_and_the_full_mode_adds_them_last() {
    let ids = |mode| -> Vec<&'static str> {
        dual_steps_for(mode, "TARGET")
            .iter()
            .map(|step| step.id)
            .collect()
    };
    let check_only = ids(DualMode::CheckOnly);
    assert_eq!(
        check_only,
        [
            "fmt",
            "build",
            "clippy",
            "clippy-examples",
            "E1",
            "E2-compile",
            "E2-run",
            "E3",
        ]
    );
    let mut full = check_only.clone();
    full.extend(["run-vector-add", "run-polynomial", "run-vector-add-no-cfg"]);
    assert_eq!(ids(DualMode::Full), full);
    // CI's mode runs no example, so it needs no GPU: nothing in it is a `cargo run`.
    for step in dual_steps_for(DualMode::CheckOnly, "TARGET") {
        assert!(!matches!(step.kind, DualKind::Runs { .. }), "{}", step.id);
        assert_ne!(step.args[0], "run", "{}", step.id);
    }
}

#[test]
fn every_cargo_call_of_check_dual_is_locked() {
    for step in steps() {
        if step.id == "fmt" {
            // `cargo fmt` builds nothing and takes no `--locked`; it is the only step without it.
            assert_eq!(step.args[0], "fmt");
            assert!(!step.args.contains(&"--locked".to_owned()));
            continue;
        }
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
    // Every step that builds passes that directory exactly once (clippy before its `--`, the rest last); fmt builds
    // nothing.
    for step in dual_steps_for(DualMode::Full, "/cache/target/dual") {
        let pairs: Vec<&[String]> = step
            .args
            .windows(2)
            .filter(|pair| pair[0] == "--target-dir")
            .collect();
        if step.id == "fmt" {
            assert!(pairs.is_empty(), "{}", step.id);
            continue;
        }
        assert_eq!(pairs.len(), 1, "{}", step.id);
        assert_eq!(pairs[0][1], "/cache/target/dual", "{}", step.id);
        // The shown command line names a placeholder, never the directory itself.
        assert!(
            step.shown().contains("--target-dir <target>/dual"),
            "{}",
            step.id
        );
        assert!(!step.shown().contains("/cache"), "{}", step.shown());
    }
    // A clippy argument list keeps `-D warnings` after the `--` and the directory before it.
    for id in ["clippy", "clippy-examples"] {
        let clippy = step(id).shown();
        assert!(
            clippy.ends_with("--target-dir <target>/dual -- -D warnings"),
            "{clippy}"
        );
        assert!(step(id).cfg, "{id}");
    }
}

#[test]
fn the_steps_run_in_the_order_fmt_build_clippy_e1_e2_e3_and_e2_is_judged_by_its_tests() {
    let ids: Vec<&str> = dual_steps("TARGET").iter().map(|step| step.id).collect();
    assert_eq!(
        ids,
        [
            "fmt",
            "build",
            "clippy",
            "clippy-examples",
            "E1",
            "E2-compile",
            "E2-run",
            "E3",
        ]
    );
    assert_eq!(step("fmt").kind, DualKind::Build);
    assert_eq!(step("build").kind, DualKind::Build);
    assert_eq!(step("clippy").kind, DualKind::Build);
    assert_eq!(step("clippy-examples").kind, DualKind::Build);
    assert_eq!(step("E2-compile").kind, DualKind::Compiles);
    assert_eq!(step("E2-run").kind, DualKind::Passes { tests: DUAL_TESTS });
    assert_eq!(
        DUAL_TESTS,
        [
            "tests::twin_runs",
            "tests::overflow_mode_is_the_dependents_own",
            "twin_overflow::a_twin_follows_the_overflow_mode_of_the_package_that_defines_the_kernel",
            "twin_overflow::a_twin_defined_in_the_dependent_follows_the_dependents_own_mode",
            "vector_overflow::an_integer_vector_add_panics_only_where_gust_and_the_kernels_package_both_check"
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
    let ok = "running 5 tests\ntest tests::overflow_mode_is_the_dependents_own ... ok\ntest tests::twin_runs ... ok\ntest twin_overflow::a_twin_follows_the_overflow_mode_of_the_package_that_defines_the_kernel ... ok\ntest twin_overflow::a_twin_defined_in_the_dependent_follows_the_dependents_own_mode ... ok\ntest vector_overflow::an_integer_vector_add_panics_only_where_gust_and_the_kernels_package_both_check ... ok\n\ntest result: ok. 5 passed\n";
    assert!(tests_not_ok(ok, DUAL_TESTS).is_empty());
    // A run that found no test succeeds in cargo and must still fail here.
    assert_eq!(
        tests_not_ok("running 0 tests\n\ntest result: ok. 0 passed\n", DUAL_TESTS),
        DUAL_TESTS
    );
    // Colour codes that a forced colour output puts around `ok` (CI sets CARGO_TERM_COLOR=always) are ignored.
    let colored = "test tests::twin_runs ... \u{1b}[32mok\u{1b}[0m\ntest tests::overflow_mode_is_the_dependents_own ... \u{1b}[32mok\u{1b}[0m\ntest twin_overflow::a_twin_follows_the_overflow_mode_of_the_package_that_defines_the_kernel ... \u{1b}[32mok\u{1b}[0m\ntest twin_overflow::a_twin_defined_in_the_dependent_follows_the_dependents_own_mode ... \u{1b}[32mok\u{1b}[0m\ntest vector_overflow::an_integer_vector_add_panics_only_where_gust_and_the_kernels_package_both_check ... \u{1b}[32mok\u{1b}[0m\n";
    assert!(tests_not_ok(colored, DUAL_TESTS).is_empty());
    assert_eq!(
        tests_not_ok(
            "test tests::twin_runs ... \u{1b}[31mFAILED\u{1b}[0m\n",
            &["tests::twin_runs"]
        ),
        ["tests::twin_runs"]
    );
    // Each of the four is required: the two overflow-outcome tests do not pass vacuously through the older two.
    let older = "running 2 tests\ntest tests::overflow_mode_is_the_dependents_own ... ok\ntest tests::twin_runs ... ok\n\ntest result: ok. 2 passed\n";
    assert_eq!(tests_not_ok(older, DUAL_TESTS), &DUAL_TESTS[2..]);
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
    // The overflow-outcome tests observe what a twin does, not what a probe reports, and each kernel's twin is
    // compiled in the package that defines it.
    assert!(
        source.contains("#[cfg(all(test, gust_dual))]\nmod twin_overflow")
            || source.contains("#[cfg(all(test, gust_dual))]\r\nmod twin_overflow")
    );
    assert!(source.contains("\"attempt to add with overflow\""));
    assert!(source.contains("kernel_lib::kernels::add_one::dispatch("));
    assert!(source.contains("own::add_one::dispatch("));
    assert!(fixture("kernel-lib/src/lib.rs").contains("pub fn add_one("));
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
    for id in [
        "fmt",
        "build",
        "clippy",
        "clippy-examples",
        "E2-compile",
        "E2-run",
        "run-vector-add",
        "run-polynomial",
        "run-vector-add-no-cfg",
    ] {
        let error = dual_problems(&root, &step(id), false, "", "").unwrap_err();
        assert!(error.contains("has no expectation file"), "{error}");
    }
}

#[test]
fn check_dual_is_explained() {
    assert!(explain_check(Some("check-dual")).is_ok());
    assert!(explain_check(Some("all")).is_ok());
}

// ---- the example runs of the full mode ----

const MATCH: &str = "CPU twin: matches (262144 elements, max ULP 0)";
const TIME: &str =
    "CPU twin time: 1.5s (run_cpu: 0.5s; debug profile, single-threaded, not a benchmark)";

#[test]
fn the_twin_line_is_parsed_strictly() {
    assert_eq!(parse_twin_match(MATCH), Some((262_144, 0)));
    assert_eq!(
        parse_twin_match("CPU twin: matches (4 elements, max ULP 17)"),
        Some((4, 17))
    );
    for bad in [
        "",
        "CPU twin: matches",
        "CPU twin: matches (4 elements)",
        "CPU twin: matches (4 elements, max ULP )",
        "CPU twin: matches ( elements, max ULP 0)",
        "CPU twin: matches (-4 elements, max ULP 0)",
        "CPU twin: matches (4 elements, max ULP 0.5)",
        "CPU twin: matches (4 elements, max ULP 0) ",
        " CPU twin: matches (4 elements, max ULP 0)",
        "cpu twin: matches (4 elements, max ULP 0)",
        "CPU twin: mismatch (4 elements, max ULP 0)",
        "CPU twin: matches (4 element, max ULP 0)",
        "CPU twin: matches (99999999999999999999999 elements, max ULP 0)",
    ] {
        assert_eq!(parse_twin_match(bad), None, "{bad:?}");
    }
}

#[test]
fn a_dual_run_must_print_exactly_one_matching_line_with_elements() {
    let also = ["CPU twin time: ", "debug profile, single-threaded"];
    let output =
        format!("Polynomial benchmark\nelements: 262144\n{MATCH}\n{TIME}\nCPU is 3x faster\n");
    assert_eq!(
        twin_run_problems("run-polynomial", true, &also, &output),
        Vec::<String>::new()
    );
    // Windows line endings and surrounding blanks do not matter.
    let crlf = output.replace('\n', "\r\n");
    assert!(twin_run_problems("run-polynomial", true, &also, &crlf).is_empty());
    // No twin line: the cfg did not reach the example, and the message says so.
    let problems = twin_run_problems("run-vector-add", true, &[], "GPU result: [11.0]\n");
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("printed no `CPU twin: matches"),
        "{problems:?}"
    );
    assert!(problems[0].contains("run-vector-add"), "{problems:?}");
    // A twin line that does not parse, with the wrong word or a wrong shape, is no match.
    for bad in [
        "CPU twin: mismatch (4 elements, max ULP 0)",
        "CPU twin: matches (4 elements)",
        "CPU twin matches (4 elements, max ULP 0)",
    ] {
        let problems = twin_run_problems("run-vector-add", true, &[], bad);
        assert_eq!(problems.len(), 1, "{bad}: {problems:?}");
    }
    // Zero elements prove nothing; two lines are ambiguous.
    let zero = twin_run_problems(
        "run-vector-add",
        true,
        &[],
        "CPU twin: matches (0 elements, max ULP 0)",
    );
    assert!(zero[0].contains("0 elements"), "{zero:?}");
    let twice = format!("{MATCH}\n{MATCH}\n");
    let problems = twin_run_problems("run-vector-add", true, &[], &twice);
    assert!(
        problems[0].contains("2 `CPU twin: matches` lines"),
        "{problems:?}"
    );
    // Colour codes around the text are ignored like in E2's test names.
    let colored = format!("\u{1b}[32m{MATCH}\u{1b}[0m\n");
    assert!(twin_run_problems("run-vector-add", true, &[], &colored).is_empty());
}

#[test]
fn polynomial_must_also_print_the_labelled_twin_time() {
    let also = ["CPU twin time: ", "debug profile, single-threaded"];
    // Without the time line, and with a time line that lost its label, each is a problem of its own.
    let problems = twin_run_problems("run-polynomial", true, &also, MATCH);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("CPU twin time: "), "{problems:?}");
    let unlabelled = format!("{MATCH}\nCPU twin time: 1.5s (run_cpu: 0.5s)\n");
    let problems = twin_run_problems("run-polynomial", true, &also, &unlabelled);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("debug profile, single-threaded"),
        "{problems:?}"
    );
    // The label on a line that does not mention the twin does not count.
    let elsewhere = format!("{MATCH}\nCPU twin time: 1.5s\ndebug profile, single-threaded\n");
    assert_eq!(
        twin_run_problems("run-polynomial", true, &also, &elsewhere).len(),
        1
    );
}

#[test]
fn a_build_without_the_cfg_must_not_mention_the_twin_at_all() {
    let plain = "Rust -> Slang -> WGSL -> wgpu (plus SPIR-V export)\nGPU result: [11.0, 22.0]\n";
    assert!(twin_run_problems("run-vector-add-no-cfg", false, &[], plain).is_empty());
    assert!(twin_run_problems("run-vector-add-no-cfg", false, &[], "").is_empty());
    for leaked in [MATCH, TIME, "note: the CPU twin was skipped"] {
        let output = format!("{plain}{leaked}\n");
        let problems = twin_run_problems("run-vector-add-no-cfg", false, &[], &output);
        assert_eq!(problems.len(), 1, "{leaked}: {problems:?}");
        assert!(
            problems[0].contains("without --cfg gust_dual"),
            "{problems:?}"
        );
        assert!(problems[0].contains(leaked), "{problems:?}");
    }
}

#[test]
fn only_the_full_gate_runs_check_dual() {
    for mode in [
        VerifyMode::Smoke,
        VerifyMode::Fast,
        VerifyMode::Gpu,
        VerifyMode::Examples,
        VerifyMode::Artifacts,
    ] {
        assert!(!runs_check_dual(mode), "{mode:?}");
    }
    assert!(runs_check_dual(VerifyMode::Full));
}

// ---- the manifests, the examples and CI ----

fn repo_file(path: &str) -> String {
    let root = workspace_root().unwrap();
    fs::read_to_string(root.join(path))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
        .replace("\r\n", "\n")
}

#[test]
fn the_workspace_declares_the_cfg_and_every_member_that_can_write_it_opts_in() {
    let root = repo_file("Cargo.toml");
    assert!(
        root.contains(
            "[workspace.lints.rust]\nunexpected_cfgs = { level = \"warn\", check-cfg = ['cfg(gust_dual)', 'cfg(gust_profile, values(\"dev\", \"native\", \"web\"))'] }\n"
        ),
        "{root}"
    );
    let opts_in = |manifest: &str| manifest.contains("\n[lints]\nworkspace = true\n");
    // The members are read from the root `members` array, so a new member cannot be missed.
    let array = root
        .split_once("members = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(array, _)| array)
        .expect(&root);
    let members: Vec<&str> = array
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|member| *member != "crates/gust-macros")
        .collect();
    assert!(members.len() >= 11, "{members:?}");
    for must in [
        "crates/gust",
        "crates/gust-wgpu",
        "xtask",
        "examples/vector-add",
    ] {
        assert!(members.contains(&must), "{must} not in {members:?}");
    }
    for member in &members {
        let manifest = format!("{member}/Cargo.toml");
        assert!(
            opts_in(&repo_file(&manifest)),
            "{manifest} lacks [lints] workspace = true"
        );
    }
    // The proc-macro crate only emits the cfg as tokens into the user's crate and never evaluates it.
    assert!(!opts_in(&repo_file("crates/gust-macros/Cargo.toml")));
}

#[test]
fn both_example_mains_print_the_twin_line_only_under_the_cfg() {
    for (example, also) in [
        ("vector-add", None),
        ("polynomial", Some("CPU twin time: ")),
    ] {
        let source = repo_file(&format!("examples/{example}/src/app.rs"));
        // The one call from `main` is behind `#[cfg(gust_dual)]`, and so is the function that prints.
        assert!(
            source.contains("    #[cfg(gust_dual)]\n    report_twin("),
            "{example}"
        );
        assert!(
            source.contains("#[cfg(gust_dual)]\nfn report_twin("),
            "{example}"
        );
        assert_eq!(
            source
                .matches("\"CPU twin: matches ({} elements, max ULP {max_ulps})\"")
                .count(),
            1,
            "{example}"
        );
        if let Some(also) = also {
            assert!(source.contains(also), "{example}");
            assert!(source.contains("single-threaded"), "{example}");
        }
    }
}

#[test]
fn ci_runs_check_dual_check_only_in_the_unit_job() {
    let ci = repo_file(".github/workflows/ci.yml");
    let unit = ci.split("\n  msrv:").next().unwrap();
    let name = unit.find("      - name: Dual test build\n").expect(unit);
    let run = unit
        .find("        run: cargo xtask check-dual --check-only\n")
        .expect(unit);
    assert!(name < run, "{unit}");
    // The step is the last of the job ...
    let after = &unit[run + 1..];
    assert!(!after.contains("\n      - "), "{after}");
    // ... and its run line follows its name directly or after comments only.
    let between = &unit[name..run];
    assert!(
        between
            .lines()
            .skip(1)
            .all(|line| line.trim_start().starts_with('#')),
        "{between}"
    );
    // One step, in the unit job only, and never the GPU mode (the runner has none).
    assert_eq!(ci.matches("check-dual").count(), 1, "{ci}");
}
