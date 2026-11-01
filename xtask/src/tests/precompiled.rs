use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
};

use super::{
    ADAPTER_PREFIX, BRIDGE_MARKERS, E4_EXPECTED, PRECOMPILED_DIR, RunReport, STRICT_PROFILE,
    StripReport, VERIFIED_PREFIX, child_environment, count_occurrences,
    dependency_disables_defaults, identity_error, is_slang_variable, judge_run, marker_counts,
    markers_missing, markers_present, mutate_wgsl, retarget_paths, slang_free_problems, strip_path,
    tree_problems,
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
