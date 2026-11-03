use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
};

use super::{
    ADAPTER_PREFIX, BRIDGE_MARKERS, E4_EXPECTED, E6_INFERRED_EXPECTED, E6_NATIVE_EXPECTED,
    E6Outcome, E6Profile, E8_PACKAGES, Expansion, INFERRED_SENTENCE, INFERRED_WARNING,
    PRECOMPILED_DIR, RunReport, STRICT_PROFILE, StripReport, VERIFIED_PREFIX, artifact_fresh_flags,
    bridge_user_manifest, child_environment, consumer_manifest, count_occurrences,
    default_feature_problems, dependency_disables_defaults, e6_args, e6_cases, e6_problems,
    e7_steps, expansion_problem, files_written, gust_build_problems, identity_error, is_inside,
    is_slang_variable, judge_run, lock_problems, marker_counts, markers_missing, markers_present,
    mutate_wgsl, package_allows, package_files, package_problems, retarget_paths, section_version,
    slang_free_package_problems, slang_free_problems, snapshot_differences, strip_path,
    tree_problems, unused_patch_lines, with_bridge_user_dependency, with_bridge_user_member,
};
use crate::{process::PROFILE_VARIABLE, ui::parse_expected, workspace::workspace_root};

fn os(text: &str) -> OsString {
    OsString::from(text)
}

fn variables(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
    pairs
        .iter()
        .map(|(name, value)| (os(name), os(value)))
        .collect()
}

/// A fake `PATH` lookup: the directories named `slang`, `sdk/bin` and `tools` hold Slang.
fn has_slang(directory: &Path) -> bool {
    ["slang", "sdk/bin", "tools"]
        .iter()
        .any(|name| directory == Path::new(name))
}

fn path_of(directories: &[&str]) -> OsString {
    std::env::join_paths(directories).unwrap()
}

fn directories_of(path: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(path).collect()
}

#[test]
fn a_variable_names_slang_when_its_name_says_so_or_it_is_one_of_the_listed_ones() {
    for name in [
        "GUST_SLANG_REFLECT",
        "SLANG_SDK",
        "slang_dir",
        "SLANGC",
        "VULKAN_SDK",
        "vk_sdk_path",
        "GUST_EXPORT_DIR",
    ] {
        assert!(is_slang_variable(name), "{name}");
    }
    for name in [
        "PATH",
        "GUST_PROFILE",
        "CARGO",
        "VK_LAYER_PATH",
        "SystemRoot",
        "TMP",
    ] {
        assert!(!is_slang_variable(name), "{name}");
    }
}

#[test]
fn stripping_a_path_drops_the_directories_with_a_tool_and_keeps_the_order_of_the_rest() {
    let path = path_of(&["bin", "slang", "usr", "sdk/bin", "last"]);
    let (stripped, dropped) = strip_path(&path, &has_slang);
    assert_eq!(
        directories_of(&stripped),
        [PathBuf::from("bin"), "usr".into(), "last".into()]
    );
    assert_eq!(dropped, [PathBuf::from("slang"), "sdk/bin".into()]);

    let (same, none) = strip_path(&path_of(&["bin", "usr"]), &has_slang);
    assert_eq!(directories_of(&same), [PathBuf::from("bin"), "usr".into()]);
    assert!(none.is_empty());
}

#[test]
fn the_child_environment_has_no_slang_no_rustflags_and_the_strict_profile() {
    let mut parent = variables(&[
        ("Path", "unused"),
        ("GUST_SLANG_REFLECT", "/x/helper"),
        ("VULKAN_SDK", "/sdk"),
        ("GUST_PROFILE", "dev"),
        ("RUSTFLAGS", "-C debuginfo=2"),
        ("cargo_encoded_rustflags", "--cfg\u{1f}gust_dual"),
        ("CARGO_NET_OFFLINE", "true"),
        ("SystemRoot", "C:/Windows"),
    ]);
    parent[0].1 = path_of(&["bin", "slang", "usr"]);
    let mut report = StripReport::default();
    let child = child_environment(&parent, &has_slang, &mut report);

    let names: Vec<String> = child
        .iter()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        ["Path", "CARGO_NET_OFFLINE", "SystemRoot", PROFILE_VARIABLE]
    );
    assert_eq!(
        child.last().unwrap().1,
        os(STRICT_PROFILE),
        "the profile is the last entry and strict"
    );
    assert_eq!(
        directories_of(&child[0].1),
        [PathBuf::from("bin"), "usr".into()]
    );
    assert_eq!(report.variables, ["GUST_SLANG_REFLECT", "VULKAN_SDK"]);
    assert_eq!(report.directories, [PathBuf::from("slang")]);
    assert_eq!(
        slang_free_problems(&child, &has_slang),
        Vec::<String>::new()
    );
}

#[test]
fn the_slang_free_assertion_names_a_variable_a_directory_and_a_wrong_profile() {
    let clean = variables(&[("PATH", "bin"), (PROFILE_VARIABLE, STRICT_PROFILE)]);
    assert!(slang_free_problems(&clean, &has_slang).is_empty());

    let mut with_variable = clean.clone();
    with_variable.push((os("GUST_SLANG_REFLECT"), os("/helper")));
    let problems = slang_free_problems(&with_variable, &has_slang);
    assert_eq!(problems, ["the variable GUST_SLANG_REFLECT is set"]);

    let mut with_directory = clean;
    with_directory[0].1 = path_of(&["bin", "tools"]);
    let problems = slang_free_problems(&with_directory, &has_slang);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("tools") && problems[0].contains("holds a Slang tool"));

    let dev = variables(&[("PATH", "bin"), (PROFILE_VARIABLE, "dev")]);
    assert_eq!(
        slang_free_problems(&dev, &has_slang),
        ["GUST_PROFILE is not native"]
    );
    let unset = variables(&[("PATH", "bin")]);
    assert_eq!(
        slang_free_problems(&unset, &has_slang),
        ["GUST_PROFILE is not native"]
    );
}

#[test]
fn a_run_is_judged_by_its_exit_code_its_adapter_its_artifact_and_the_verified_line() {
    let good = format!(
        "precompiled vector-add\nkernel: vector_add::add\nartifact: Wgsl key=52e85c5f03f72272 bytes=476\n{ADAPTER_PREFIX}NVIDIA GeForce RTX 5090\n{VERIFIED_PREFIX}1000 elements\n"
    );
    assert_eq!(
        judge_run(true, &good, ""),
        Ok(RunReport {
            adapter: "NVIDIA GeForce RTX 5090".to_owned(),
            elements: "1000".to_owned()
        })
    );

    let problems = judge_run(false, &good, "error: boom").unwrap_err();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("error: boom"));

    for (broken, wanted) in [
        (good.replace(VERIFIED_PREFIX, "checked: "), "verified:"),
        (good.replace(ADAPTER_PREFIX, "device: "), "adapter:"),
        (
            good.replace("artifact: Wgsl", "artifact: Spirv"),
            "artifact: Wgsl",
        ),
        (good.replace("1000 elements", "0 elements"), "verified:"),
        (
            good.replace("adapter: NVIDIA GeForce RTX 5090", "adapter:   "),
            "adapter:",
        ),
    ] {
        let problems = judge_run(true, &broken, "").unwrap_err();
        assert!(
            problems.iter().any(|problem| problem.contains(wanted)),
            "{wanted}: {problems:?}"
        );
    }
    assert!(judge_run(true, "", "").unwrap_err().len() >= 3);
}

#[test]
fn markers_are_counted_without_overlap_and_an_empty_needle_never_matches() {
    assert_eq!(count_occurrences(b"abcabc", b"abc"), 2);
    assert_eq!(count_occurrences(b"aaaa", b"aa"), 2);
    assert_eq!(count_occurrences(b"ab", b"abc"), 0);
    assert_eq!(count_occurrences(b"", b"a"), 0);
    assert_eq!(count_occurrences(b"abc", b""), 0);
    assert_eq!(count_occurrences(b"xxabcxx", b"abc"), 1);
    assert_eq!(count_occurrences(b"abab\0abab", b"abab"), 2);

    let mut binary = vec![0_u8; 64];
    binary.extend_from_slice(b"..gust-slang-reflect..");
    binary.extend_from_slice(&[0; 8]);
    binary.extend_from_slice(b"gust-slang-reflect");
    let counts = marker_counts(&binary, &BRIDGE_MARKERS);
    assert_eq!(
        counts,
        [
            ("gust-slang-reflect", 2),
            ("kernel.slang", 0),
            ("kernel.artifact", 0)
        ]
    );
    // The clean scan reports what occurs, the control scan what does not.
    assert_eq!(markers_present(&counts), ["`gust-slang-reflect` x2"]);
    assert_eq!(
        markers_missing(&counts),
        ["`kernel.slang`", "`kernel.artifact`"]
    );
    let nothing = marker_counts(b"no markers here", &BRIDGE_MARKERS);
    assert!(markers_present(&nothing).is_empty());
    assert_eq!(markers_missing(&nothing).len(), BRIDGE_MARKERS.len());
}

#[test]
fn the_bridge_markers_are_the_strings_of_the_gated_code_and_not_slangc() {
    assert!(BRIDGE_MARKERS.contains(&"gust-slang-reflect"));
    // The ungated `Display` of `slang::Error::CompilerNotFound` spells `slangc`, and that of
    // `reflect::Error::HelperUnavailable` spells `GUST_SLANG_REFLECT`: both are in a binary without the bridge
    // (measured), so neither can be a marker.
    assert!(!BRIDGE_MARKERS.contains(&"slangc"));
    assert!(!BRIDGE_MARKERS.contains(&"GUST_SLANG_REFLECT"));
    for marker in BRIDGE_MARKERS {
        assert!(marker.len() >= 10, "{marker} is too short to be a marker");
    }
}

const TREE: &str = "app v0.0.0 (/w/tests/precompiled/app)\n├── gust v0.1.0 (/w/crates/gust)\n│   └── gust-macros v0.1.0 (/w/crates/gust-macros)\n├── gust-wgpu v0.1.0 (/w/crates/gust-wgpu)\n│   ├── gust v0.1.0 (/w/crates/gust) (*)\n│   └── wgpu v30.0.1\n└── kernels v0.1.0 (/w/tests/precompiled/kernels)\n    └── gust v0.1.0 (/w/crates/gust) (*)\n";

#[test]
fn the_tree_must_show_the_crates_and_not_the_feature() {
    assert_eq!(tree_problems(TREE), Vec::<String>::new());

    let on = TREE.replace(
        "│   └── wgpu v30.0.1\n",
        "│   ├── gust feature \"slang-runtime\"\n│   └── wgpu v30.0.1\n",
    );
    let problems = tree_problems(&on);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("the feature is on") && problems[0].contains("slang-runtime"));

    assert_eq!(tree_problems("").len(), 3, "an empty tree reads nothing");
    let no_wgpu = TREE.replace("gust-wgpu v0.1.0", "wgpu-gust v0.1.0");
    assert_eq!(tree_problems(&no_wgpu).len(), 1);
}

#[test]
fn a_dependency_disables_its_defaults_on_one_inline_line_without_the_bridge_feature() {
    let manifest = "[dependencies]\nkernels = { path = \"../kernels\" }\ngust = { path = \"../../../crates/gust\", default-features = false }\ngust-wgpu = { path = \"../../../crates/gust-wgpu\", default-features = false }\n";
    assert!(dependency_disables_defaults(manifest, "gust"));
    assert!(dependency_disables_defaults(manifest, "gust-wgpu"));
    assert!(!dependency_disables_defaults(manifest, "kernels"));
    assert!(!dependency_disables_defaults(manifest, "wgpu"));

    // Dropping the setting, spelling the table form, adding the feature, or a comment do not count.
    let dropped = manifest.replace(", default-features = false", "");
    assert!(!dependency_disables_defaults(&dropped, "gust"));
    let table = "[dependencies.gust]\npath = \"x\"\ndefault-features = false\n";
    assert!(!dependency_disables_defaults(table, "gust"));
    let featured = manifest.replace(
        "default-features = false }\ngust-wgpu",
        "default-features = false, features = [\"slang-runtime\"] }\ngust-wgpu",
    );
    assert!(!dependency_disables_defaults(&featured, "gust"));
    assert!(dependency_disables_defaults(&featured, "gust-wgpu"));
    let commented = "# gust = { default-features = false }\n";
    assert!(!dependency_disables_defaults(commented, "gust"));
    let spaced = "gust   =   { path = \"x\" ,  default-features=false }\n";
    assert!(dependency_disables_defaults(spaced, "gust"));
}

#[test]
fn the_mutation_changes_the_first_plus_only() {
    let wgsl = b"a_0[i] + b_0[i] + c";
    let mutated = mutate_wgsl(wgsl).unwrap();
    assert_eq!(mutated, b"a_0[i] - b_0[i] + c");
    assert_eq!(mutated.len(), wgsl.len());
    assert_ne!(mutated, wgsl);
    assert!(
        mutate_wgsl(b"no operator here")
            .unwrap_err()
            .contains("no `+`")
    );
}

const IDENTITY: &str = "error: dispatch failed: could not validate kernel reflection: native target, entry point, source or artifact identity mismatch: artifact hash (stored \"0123\", expected \"4567\")";

#[test]
fn the_identity_error_must_name_the_artifact_hash_alone_and_the_app_must_not_verify() {
    assert_eq!(
        identity_error(false, "precompiled vector-add\n", &format!("{IDENTITY}\n")).as_deref(),
        Ok(IDENTITY)
    );

    // The mutation did nothing: the app ran and verified.
    let verified = format!("{VERIFIED_PREFIX}1000 elements\n");
    let problems = identity_error(true, &verified, "").unwrap_err();
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems[0].contains("accepted"));
    assert!(problems[1].contains("verified"));
    assert!(problems[2].contains("no identity error"));

    // Failed, but with another error.
    let other = identity_error(false, "", "error: no GPU adapter: none\n").unwrap_err();
    assert_eq!(other.len(), 1);
    assert!(other[0].contains("no GPU adapter"));

    // More than the artifact hash differs: the mutation reached something else.
    let more = IDENTITY.replace(
        "artifact hash (stored \"0123\"",
        "artifact hash (stored \"0123\", expected \"4567\"); source hash (stored \"a\"",
    );
    let problems = identity_error(false, "", &more).unwrap_err();
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("source hash ("))
    );
}

#[test]
fn a_scratch_copy_points_its_path_dependencies_at_the_repository() {
    let manifest = "[dependencies]\nkernels = { path = \"../kernels\" }\ngust = { version = \"0.1.0\", path = \"../../../crates/gust\" }\n";
    let retargeted = retarget_paths(manifest, "C:/work space/repo");
    assert!(retargeted.contains("path = \"C:/work space/repo/crates/gust\""));
    assert!(retargeted.contains("kernels = { path = \"../kernels\" }"));
    assert!(!retargeted.contains("../../../"));
    assert_eq!(retarget_paths("[package]\n", "/r"), "[package]\n");
}

// The committed fixture, read from the repository: the guards that keep it what the checks assume.

fn fixture_file(relative: &str) -> String {
    let path = workspace_root()
        .unwrap()
        .join(PRECOMPILED_DIR)
        .join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{PRECOMPILED_DIR}/{relative}: {error}"))
}

#[test]
fn the_committed_manifests_keep_the_bridge_out_and_the_kernels_packageable() {
    let app = fixture_file("app/Cargo.toml");
    assert!(dependency_disables_defaults(&app, "gust"));
    assert!(dependency_disables_defaults(&app, "gust-wgpu"));
    assert!(!app.contains("slang-runtime\"]"));

    let kernels = fixture_file("kernels/Cargo.toml");
    assert!(dependency_disables_defaults(&kernels, "gust"));
    // R2-4: `gust` by version plus path, and no `publish = false` (`cargo package` refuses such a package).
    assert!(kernels.contains("version = \"0.1.0\", path = \"../../../crates/gust\""));
    assert!(
        !kernels
            .lines()
            .any(|line| !line.trim_start().starts_with('#') && line.contains("publish")),
        "no publish key"
    );

    let workspace = fixture_file("Cargo.toml");
    assert!(workspace.contains("members = [\"kernels\", \"app\"]"));
    assert!(
        !workspace.contains("[env]"),
        "the profile is set by the environment, never by a config file"
    );
}

#[test]
fn the_committed_e4_expectation_is_one_e0080_at_the_kernel_line() {
    let wanted = parse_expected(&fixture_file(E4_EXPECTED)).unwrap();
    assert_eq!(wanted.code, "E0080");
    assert_eq!(wanted.file, "kernels/src/lib.rs");
    let source = fixture_file("kernels/src/lib.rs");
    let line = usize::try_from(wanted.line).unwrap();
    assert!(
        source
            .lines()
            .nth(line - 1)
            .unwrap()
            .contains("pub fn add("),
        "line {line} of kernels/src/lib.rs is not the kernel's `fn`"
    );
    assert_eq!(wanted.messages.len(), 1);
    assert!(wanted.messages[0].contains("needs a precompiled artifact for this kernel"));
}

#[test]
fn the_committed_artifacts_are_the_ones_e3_mutates_and_e4_deletes() {
    let manifest = fixture_file("kernels/gust-artifacts/manifest.json");
    assert!(manifest.contains("\"schema\": 1") && manifest.contains("\"format\": \"wgsl\""));
    let key = "52e85c5f03f72272";
    let wgsl = fixture_file(&format!("kernels/gust-artifacts/{key}.wgsl"));
    assert!(
        mutate_wgsl(wgsl.as_bytes()).is_ok(),
        "E3 needs a `+` to change"
    );
    let reflection = fixture_file(&format!(
        "kernels/gust-artifacts/{key}.wgsl.reflection.json"
    ));
    assert!(reflection.contains("artifactHash"));
}

// ---------------------------------------------------------------------------------------------------------------
// E5 to E8

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

/// The refusal as cargo prints it when the build script of `gust` fails with `message`.
fn refusal_stderr(message: &str) -> String {
    format!(
        "error: failed to run custom build command for `gust v0.1.0 (C:\\repo\\crates\\gust)`\n\nCaused by:\n  process didn't exit successfully\n  --- stderr\n  {message}\n"
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
    assert_eq!(ids, ["E6a", "E6b", "E6c", "E6c-release", "E6d", "E6e"]);
    // The refusals judge a message; the controls compile.
    for case in e6_cases() {
        let refuses = matches!(case.outcome, E6Outcome::Refused { .. });
        assert_eq!(refuses, !matches!(case.id, "E6d" | "E6e"), "{}", case.id);
    }
    assert_eq!(case("E6d").profile, E6Profile::Unset);
    assert!(case("E6d").release && !case("E6d").bridge_user);
    assert_eq!(case("E6e").profile, E6Profile::Dev);
    assert_eq!(case("E6e").features, Some("gust/slang-runtime"));
    assert!(case("E6c").bridge_user && case("E6c-release").bridge_user);
    assert!(!case("E6a").bridge_user && !case("E6b").bridge_user);
}

#[test]
fn only_the_copy_of_e6c_runs_without_locked() {
    for case in e6_cases() {
        let args = e6_args(&case, "M/Cargo.toml", "T");
        assert_eq!(
            args.iter().any(|arg| arg == "--locked"),
            !case.bridge_user,
            "{}: the fixture's own lock must never be regenerated, the copy's must be",
            case.id
        );
        assert_eq!(args.iter().any(|arg| arg == "--release"), case.release);
        assert!(args.windows(2).any(|pair| pair == ["-p", "app"]));
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
    let warning = format!("warning: gust@0.1.0: {INFERRED_WARNING}\n");
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
fn e7_alternates_unset_and_changed_values_and_observes_one_artifact_each_time() {
    let steps = e7_steps();
    assert_eq!(steps[0].expect, Expansion::Any);
    assert_eq!(steps[1].expect, Expansion::Fresh);
    assert_eq!(steps[1].export, None);
    assert_eq!(
        (steps[2].export, steps[2].expect),
        (Some("a"), Expansion::Again)
    );
    assert_eq!(
        (steps[3].export, steps[3].expect),
        (Some("a"), Expansion::Fresh)
    );
    assert_eq!(
        (steps[4].export, steps[4].expect),
        (Some("b"), Expansion::Again)
    );
    assert_eq!((steps[5].export, steps[5].expect), (None, Expansion::Again));
    // A fresh step follows a step with the same value; an expanding step follows one with another value.
    for pair in steps.windows(2) {
        match pair[1].expect {
            Expansion::Fresh => assert_eq!(pair[0].export, pair[1].export),
            Expansion::Again => assert_ne!(pair[0].export, pair[1].export),
            Expansion::Any => {}
        }
    }
}

#[test]
fn the_expansion_is_read_from_the_fresh_flag_of_the_kernels_artifact() {
    let stdout = concat!(
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"gust\"},\"fresh\":false}\n",
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"kernels\"},\"fresh\":true}\n",
        "not json\n",
        "{\"reason\":\"build-finished\",\"success\":true}\n"
    );
    assert_eq!(artifact_fresh_flags(stdout, "kernels"), [true]);
    assert_eq!(artifact_fresh_flags(stdout, "gust"), [false]);
    assert!(artifact_fresh_flags(stdout, "app").is_empty());
    let steps = e7_steps();
    assert!(expansion_problem(&steps[1], &[true]).is_none());
    assert!(
        expansion_problem(&steps[1], &[false])
            .unwrap()
            .contains("compiled again")
    );
    assert!(expansion_problem(&steps[2], &[false]).is_none());
    assert!(
        expansion_problem(&steps[2], &[true])
            .unwrap()
            .contains("fresh")
    );
    assert!(expansion_problem(&steps[0], &[true]).is_none());
    assert!(expansion_problem(&steps[0], &[false]).is_none());
    // No message, or two, is a blind observation, not a pass.
    assert!(expansion_problem(&steps[2], &[]).unwrap().contains("blind"));
    assert!(
        expansion_problem(&steps[1], &[true, true])
            .unwrap()
            .contains("blind")
    );
}

#[test]
fn two_builds_differ_when_one_byte_or_one_file_does() {
    use std::collections::BTreeMap;
    let first: BTreeMap<String, Vec<u8>> =
        [("a".to_owned(), vec![1, 2]), ("b".to_owned(), vec![3])].into();
    assert!(snapshot_differences(&first, &first.clone()).is_empty());
    let mut changed = first.clone();
    changed.insert("a".to_owned(), vec![1, 9]);
    assert_eq!(snapshot_differences(&first, &changed).len(), 1);
    let mut extra = first.clone();
    extra.insert("c".to_owned(), Vec::new());
    assert!(snapshot_differences(&first, &extra)[0].contains("only in the second"));
    let mut fewer = first.clone();
    fewer.remove("b");
    assert!(snapshot_differences(&first, &fewer)[0].contains("only in the first"));
}

#[test]
fn a_gust_build_run_that_failed_or_pruned_is_a_problem() {
    assert!(gust_build_problems("run 1", true, "gust: wrote 4 files\n", "").is_empty());
    assert_eq!(gust_build_problems("run 1", false, "", "boom").len(), 1);
    let pruned = gust_build_problems("run 2", true, "gust: pruned 1 stale file\n", "");
    assert_eq!(pruned.len(), 1);
    assert!(pruned[0].contains("pruned"));
}

fn spec(name: &str) -> super::PackageSpec {
    E8_PACKAGES
        .into_iter()
        .find(|spec| spec.name == name)
        .unwrap()
}

#[test]
fn the_package_allow_list_names_a_stray_file() {
    let good = strings(&[
        "Cargo.toml",
        "Cargo.toml.orig",
        ".cargo_vcs_info.json",
        "Cargo.lock",
        "src/lib.rs",
        "src/a/b.rs",
        "README.md",
        "LICENSE-MIT",
    ]);
    assert!(package_problems(&spec("gust-macros"), &good, &[]).is_empty());
    for stray in [
        "tests/ui.rs",
        "examples/a.rs",
        "notes.txt",
        "src/../x",
        "benches/b.rs",
        "gust-artifacts/manifest.json",
        "build.rs",
        "docs/README.md",
    ] {
        let mut files = good.clone();
        files.push(stray.to_owned());
        let problems = package_problems(&spec("gust-macros"), &files, &[]);
        assert_eq!(problems.len(), 1, "{stray}: {problems:?}");
        assert!(problems[0].contains(stray), "{problems:?}");
    }
    // `gust` alone may list a build script.
    let mut with_script = good.clone();
    with_script.push("build.rs".to_owned());
    assert!(package_problems(&spec("gust"), &with_script, &[]).is_empty());
    assert!(!package_allows(&spec("gust-wgpu"), "gust-artifacts/a.wgsl"));
    assert!(package_allows(&spec("kernels"), "gust-artifacts/a.wgsl"));
}

#[test]
fn a_package_needs_its_manifest_and_library_and_the_kernels_package_every_artifact() {
    let problems = package_problems(&spec("gust"), &strings(&["Cargo.toml"]), &[]);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("src/lib.rs"));
    let committed = strings(&["manifest.json", "k.wgsl", "k.wgsl.reflection.json"]);
    let mut files = strings(&[
        "Cargo.toml",
        "src/lib.rs",
        "gust-artifacts/manifest.json",
        "gust-artifacts/k.wgsl",
    ]);
    let problems = package_problems(&spec("kernels"), &files, &committed);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("k.wgsl.reflection.json"));
    files.push("gust-artifacts/k.wgsl.reflection.json".to_owned());
    assert!(package_problems(&spec("kernels"), &files, &committed).is_empty());
    assert_eq!(
        package_files("Cargo.toml\r\nsrc\\lib.rs\r\n\r\n"),
        ["Cargo.toml", "src/lib.rs"]
    );
}

#[test]
fn a_kernels_package_with_slang_or_twin_text_is_refused() {
    let clean = vec![("src/lib.rs".to_owned(), "pub fn add() {}".to_owned())];
    assert!(slang_free_package_problems(&clean).is_empty());
    for (path, text) in [
        ("gust-artifacts/k.slang", ""),
        ("generated-wgpu/k.wgsl", ""),
        ("src/lib.rs", "use gust::cpu::CpuDevice;"),
        ("src/lib.rs", "fn cpu_add()"),
    ] {
        let problems = slang_free_package_problems(&[(path.to_owned(), text.to_owned())]);
        assert!(!problems.is_empty(), "{path}: {text}");
    }
}

fn root_manifest() -> String {
    let path = workspace_root().unwrap().join("Cargo.toml");
    fs::read_to_string(path).unwrap()
}

#[test]
fn the_versions_come_from_the_manifests_and_the_consumer_patches_all_four_packages() {
    let root = root_manifest();
    assert_eq!(
        section_version(&root, "workspace.package").as_deref(),
        Some("0.1.0")
    );
    assert_eq!(
        section_version(&fixture_file("kernels/Cargo.toml"), "package").as_deref(),
        Some("0.1.0")
    );
    assert_eq!(
        section_version("[a]\nversion = \"1\"\n[package]\nname = \"x\"\n", "package"),
        None
    );
    let versions = ["0.1.0", "0.1.0", "0.1.0", "0.0.0"].map(str::to_owned);
    let manifest = consumer_manifest(&versions);
    for name in ["gust-macros", "gust", "gust-wgpu", "kernels"] {
        assert!(
            manifest.contains(&format!("{name} = {{ path = \"../unpacked/")),
            "{name}"
        );
    }
    assert!(manifest.contains("gust = \"=0.1.0\"\ngust-wgpu = \"=0.1.0\"\nkernels = \"=0.0.0\"\n"));
    assert!(
        !manifest.contains("default-features"),
        "plain registry-style dependencies (D45)"
    );
    assert!(manifest.contains("[workspace]\n"));
    assert!(manifest.contains("[patch.crates-io]"));
}

#[test]
fn the_packaged_crates_carry_versions_on_their_path_dependencies() {
    let root = root_manifest();
    // The internal dependencies are exact (`=`) so that the four released crates can never be mixed (mr.a).
    let version = section_version(&root, "workspace.package").unwrap();
    for name in ["gust", "gust-macros", "gust-wgpu"] {
        let line = root
            .lines()
            .find(|line| line.trim_start().starts_with(&format!("{name} = ")))
            .unwrap();
        assert!(
            line.contains(&format!("version = \"={version}\"")) && line.contains("path = "),
            "{line}"
        );
        let manifest = fs::read_to_string(
            workspace_root()
                .unwrap()
                .join("crates")
                .join(name)
                .join("Cargo.toml"),
        )
        .unwrap();
        // `gust-wgpu` also keeps its `clippy.toml` out of the package (it only configures this workspace's lint).
        assert!(
            manifest.contains("exclude = [\"tests/\", \"examples/\", \"benches/\""),
            "{name}"
        );
    }
}

#[test]
fn a_lock_entry_with_a_source_or_a_missing_package_is_a_problem() {
    let entry = |name: &str, source: Option<&str>| {
        let mut text = format!("[[package]]\nname = \"{name}\"\nversion = \"0.1.0\"\n");
        if let Some(source) = source {
            text.push_str(&format!("source = \"{source}\"\n"));
        }
        text.push('\n');
        text
    };
    let registry = "registry+https://github.com/rust-lang/crates.io-index";
    let good: String = ["gust", "gust-macros", "gust-wgpu", "kernels"]
        .iter()
        .map(|name| entry(name, None))
        .chain([entry("wgpu", Some(registry))])
        .collect();
    assert!(
        lock_problems(&good).is_empty(),
        "{:?}",
        lock_problems(&good)
    );
    let sourced = good.replacen(
        "name = \"gust\"\nversion = \"0.1.0\"\n",
        &format!("name = \"gust\"\nversion = \"0.1.0\"\nsource = \"{registry}\"\n"),
        1,
    );
    let problems = lock_problems(&sourced);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`gust`") && problems[0].contains("registry"));
    let missing = good.replacen("name = \"kernels\"", "name = \"other\"", 1);
    assert!(
        lock_problems(&missing)
            .iter()
            .any(|problem| problem.contains("`kernels`"))
    );
    assert!(lock_problems("").len() >= 5, "an empty lock must not pass");
}

#[test]
fn an_unused_patch_is_found_in_cargos_warning() {
    let output = "warning: Patch `gust-macros v0.1.0 (C:\\x)` was not used in the crate graph.\nhelp: Check that the patched package version and available features are compatible\n";
    assert_eq!(unused_patch_lines(output).len(), 1);
    assert!(unused_patch_lines("Compiling gust v0.1.0\n").is_empty());
}

fn metadata(gust_features: &str, wgpu_features: &str, manifest: &str) -> String {
    format!(
        "{{\"packages\":[{{\"name\":\"gust\",\"manifest_path\":\"{manifest}/gust-0.1.0/Cargo.toml\",\"features\":{gust_features}}},{{\"name\":\"gust-wgpu\",\"manifest_path\":\"{manifest}/gust-wgpu-0.1.0/Cargo.toml\",\"features\":{wgpu_features}}}]}}"
    )
}

#[test]
fn a_default_feature_that_reaches_slang_runtime_is_found_directly_or_through_another_feature() {
    let ok = "{\"slang-runtime\":[],\"other\":[]}";
    let names = ["gust", "gust-wgpu"];
    assert_eq!(
        default_feature_problems(&metadata(ok, ok, "C:/u"), &names, "C:/u"),
        Ok(Vec::new())
    );
    let direct = "{\"default\":[\"slang-runtime\"],\"slang-runtime\":[]}";
    let problems = default_feature_problems(&metadata(ok, direct, "C:/u"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("gust-wgpu"));
    let through = "{\"default\":[\"bridge\"],\"bridge\":[\"gust/slang-runtime\"]}";
    let problems =
        default_feature_problems(&metadata(through, ok, "C:/u"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`bridge`"));
    // A package that is not the unpacked one (a registry copy) is a problem, and so is a missing package.
    let problems =
        default_feature_problems(&metadata(ok, ok, "C:/registry"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert_eq!(
        default_feature_problems("{\"packages\":[]}", &names, "C:/u")
            .unwrap()
            .len(),
        2
    );
    assert!(default_feature_problems("not json", &names, "C:/u").is_err());
}

#[test]
fn the_work_directory_must_lie_outside_the_repository() {
    assert!(is_inside(
        Path::new("C:/repo"),
        Path::new("C:/repo/target/x")
    ));
    assert!(is_inside(Path::new("C:/repo/"), Path::new("C:/repo")));
    assert!(!is_inside(Path::new("C:/repo"), Path::new("C:/repository")));
    assert!(!is_inside(Path::new("C:/repo"), Path::new("C:/tmp/repo")));
}

#[test]
fn the_fixture_app_has_the_observer_that_e5_needs() {
    let lib = fixture_file("app/src/lib.rs");
    assert!(lib.contains("#[cfg(gust_dual)]") && lib.contains("fn twin_smoke"));
    assert!(
        crate::checks::PRECOMPILED_TWIN_TESTS
            .iter()
            .all(|name| lib.contains(name.rsplit("::").next().unwrap()))
    );
}

#[test]
fn the_files_written_by_cargo_gust_are_read_from_its_summary_line() {
    let line = "gust: 1 kernel(s) (wgsl), Slang 2026.13.1: 3 file(s) written, 0 unchanged\n";
    assert_eq!(files_written(line), Some(3));
    assert_eq!(
        files_written("gust: x: 0 file(s) written, 3 unchanged"),
        Some(0)
    );
    assert_eq!(files_written("gust: compiling 1 kernel(s)\n"), None);
}
