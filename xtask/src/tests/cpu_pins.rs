use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{CPU_REQUIRED_TESTS, CPU_TESTS, check_cpu_output, cpu_test_package};
use crate::workspace::workspace_root;

// ---- the pins read the example sources ----

/// A test function of an example: its full libtest name, and whether it is `#[ignore]`d.
struct SourceTest {
    name: String,
    ignored: bool,
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

/// The `#[test]` functions of one source file, named `prefix` + the function name. A function counts when the
/// attribute lines and comments directly above it (any order, doc comments included) contain `#[test]`, and it may
/// carry a visibility or `async`/`unsafe` before `fn`. A miss fails closed: a required name that is not found fails
/// `every_required_test_exists_in_the_example_and_is_not_ignored`, and an unlisted test that is missed would also
/// have to dodge the `fn` form below.
fn parse_tests(prefix: &str, text: &str) -> Vec<SourceTest> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let mut tests = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let mut rest = *line;
        for qualifier in ["pub(crate) ", "pub(super) ", "pub ", "async ", "unsafe "] {
            rest = rest.strip_prefix(qualifier).unwrap_or(rest);
        }
        let Some(rest) = rest.strip_prefix("fn ") else {
            continue;
        };
        let attributes: Vec<&str> = lines[..at]
            .iter()
            .rev()
            .take_while(|above| above.starts_with("#[") || above.starts_with("//"))
            .copied()
            .collect();
        if !attributes.contains(&"#[test]") {
            continue;
        }
        let name = rest.split(['(', '<']).next().unwrap().trim();
        tests.push(SourceTest {
            name: format!("{prefix}{name}"),
            ignored: attributes
                .iter()
                .any(|attribute| attribute.starts_with("#[ignore")),
        });
    }
    tests
}

/// The `#[test]` functions of `examples/<package>/src`, named as libtest prints them. The convention this relies on
/// (and that the required names then check): the tests of `app.rs` (or any other module file) sit in a top-level
/// `mod tests`, so they are `<file>::tests::<fn>`, and those of `main.rs` and `lib.rs` (the crate root) are
/// `tests::<fn>`. A test elsewhere gets a wrong name here, so a required one is "not found" and fails.
fn source_tests(package: &str) -> Vec<SourceTest> {
    let root = workspace_root().unwrap();
    let src = root.join("examples").join(package).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    let mut tests = Vec::new();
    for file in files {
        let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
        let prefix = if stem == "main" || stem == "lib" {
            "tests::".to_owned()
        } else {
            format!("{stem}::tests::")
        };
        let text = fs::read_to_string(&file).unwrap();
        tests.extend(parse_tests(&prefix, &text));
    }
    tests
}

#[test]
fn the_source_scanner_sees_visibility_doc_comments_and_other_attributes_between_the_test_attribute_and_fn()
 {
    let text = "mod tests {\n    #[test]\n    #[ignore = \"gpu\"]\n    fn plain_ignored() {}\n\n    #[test]\n    /// A doc comment.\n    // A comment.\n    #[allow(clippy::x)]\n    pub fn twin_documented() {}\n\n    /// Docs above the attribute.\n    #[ignore]\n    #[test]\n    pub(crate) fn twin_ignored_first() {}\n\n    #[test]\n    async fn twin_async() {}\n\n    fn helper() {}\n\n    #[test]\n\n    fn blank_line_breaks_the_attribute_run() {}\n}\n";
    let found = parse_tests("tests::", text);
    let seen: Vec<(&str, bool)> = found
        .iter()
        .map(|test| (test.name.as_str(), test.ignored))
        .collect();
    assert_eq!(
        seen,
        [
            ("tests::plain_ignored", true),
            ("tests::twin_documented", false),
            ("tests::twin_ignored_first", true),
            ("tests::twin_async", false),
        ]
    );
}

/// Whether check-cpu must pin the test: a twin test, or the guard that the inputs stay inside the derivation domain.
fn must_be_pinned(name: &str) -> bool {
    let function = name.rsplit("::").next().unwrap();
    function.starts_with("twin_") || function.contains("derivation_domain")
}

#[test]
fn the_required_tests_are_listed_for_the_packages_of_cpu_tests_in_the_same_order() {
    let packages: Vec<&str> = CPU_TESTS
        .iter()
        .map(|args| cpu_test_package(args).unwrap())
        .collect();
    let required: Vec<&str> = CPU_REQUIRED_TESTS
        .iter()
        .map(|(package, _)| *package)
        .collect();
    assert_eq!(required, packages);
    for (package, tests) in CPU_REQUIRED_TESTS {
        assert!(
            tests
                .iter()
                .any(|name| name.rsplit("::").next().unwrap().starts_with("twin_")),
            "{package} pins no twin test"
        );
        let mut sorted = tests.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            tests.len(),
            "{package}: a name is listed twice"
        );
    }
}

#[test]
fn every_required_test_exists_in_the_example_and_is_not_ignored() {
    for (package, required) in CPU_REQUIRED_TESTS {
        let found = source_tests(package);
        for name in *required {
            let matches: Vec<&SourceTest> =
                found.iter().filter(|test| test.name == *name).collect();
            assert_eq!(matches.len(), 1, "{package}: `{name}` in the source");
            assert!(
                !matches[0].ignored,
                "{package}: `{name}` is #[ignore]d, so check-cpu would never run it"
            );
        }
    }
}

#[test]
fn every_non_ignored_twin_or_derivation_domain_test_of_an_example_is_required() {
    for (package, required) in CPU_REQUIRED_TESTS {
        for test in source_tests(package) {
            if test.ignored || !must_be_pinned(&test.name) {
                continue;
            }
            assert!(
                required.contains(&test.name.as_str()),
                "{package}: `{}` is a non-ignored twin or derivation-domain test and is not in CPU_REQUIRED_TESTS",
                test.name
            );
        }
    }
}

// ---- the judgement of one call's output ----

/// What `cargo test -p <package>` prints when every required test passes and an unrelated test ran too.
fn passing_output(package: &str) -> String {
    let (_, required) = CPU_REQUIRED_TESTS
        .iter()
        .find(|(name, _)| *name == package)
        .unwrap();
    let mut text = String::from("running tests\n");
    for name in *required {
        text.push_str(&format!("test {name} ... ok\n"));
    }
    text.push_str("test app::tests::demo_sizes_stay_within_the_default_workgroup_limit ... ok\n");
    text.push_str("\ntest result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n");
    text
}

#[test]
fn a_call_whose_required_tests_all_passed_is_accepted() {
    for (package, _) in CPU_REQUIRED_TESTS {
        assert_eq!(
            check_cpu_output(package, &passing_output(package)),
            Ok(()),
            "{package}"
        );
    }
}

#[test]
fn a_deleted_or_renamed_twin_test_fails_while_its_siblings_still_pass() {
    let output = passing_output("vector-add");
    let renamed = output.replace(
        "app::tests::twin_matches_host_reference",
        "app::tests::twin_matches_host_reference_renamed",
    );
    let error = check_cpu_output("vector-add", &renamed).unwrap_err();
    assert!(
        error.contains("`app::tests::twin_matches_host_reference`"),
        "{error}"
    );
    assert!(!error.contains("twin_checks_buffer_aliasing"), "{error}");
    let deleted = output
        .lines()
        .filter(|line| !line.contains("twin_checks_buffer_aliasing"))
        .collect::<Vec<_>>()
        .join("\n");
    let error = check_cpu_output("vector-add", &deleted).unwrap_err();
    assert!(
        error.contains("`app::tests::twin_checks_buffer_aliasing_like_the_gpu_backend`"),
        "{error}"
    );
    // Both missing: both are named.
    let error = check_cpu_output("vector-add", "test result: ok. 1 passed\n").unwrap_err();
    assert!(
        error.contains(
            "app::tests::twin_matches_host_reference`, `app::tests::twin_checks_buffer_aliasing"
        ),
        "{error}"
    );
}

#[test]
fn an_ignored_failed_or_filtered_required_test_fails() {
    let output = passing_output("atomic-counter");
    for state in ["ignored", "FAILED"] {
        let changed = output.replace(
            "tests::twin_counts_every_invocation ... ok",
            &format!("tests::twin_counts_every_invocation ... {state}"),
        );
        let error = check_cpu_output("atomic-counter", &changed).unwrap_err();
        assert!(
            error.contains("`tests::twin_counts_every_invocation`"),
            "{error}"
        );
    }
    // A test of another module with the same last segment does not count.
    let other = output.replace(
        "test tests::twin_counts_every_invocation",
        "test other::tests::twin_counts_every_invocation",
    );
    assert!(check_cpu_output("atomic-counter", &other).is_err());
}

#[test]
fn colour_codes_around_ok_are_ignored() {
    let colored = passing_output("polynomial").replace(" ... ok", " ... \u{1b}[32mok\u{1b}[0m");
    assert_eq!(check_cpu_output("polynomial", &colored), Ok(()));
}

#[test]
fn a_package_without_required_tests_is_an_error() {
    let error = check_cpu_output("not-an-example", "test x ... ok\n").unwrap_err();
    assert!(error.contains("CPU_REQUIRED_TESTS"), "{error}");
    assert!(error.contains("`not-an-example`"), "{error}");
}

#[test]
fn the_package_is_read_from_the_call() {
    assert_eq!(
        cpu_test_package(&["test", "--locked", "-p", "vector-add"]),
        Some("vector-add")
    );
    assert_eq!(cpu_test_package(&["test", "--locked"]), None);
    assert_eq!(cpu_test_package(&["test", "-p"]), None);
}
