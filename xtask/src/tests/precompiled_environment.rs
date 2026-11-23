use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

use super::*;

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
