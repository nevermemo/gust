use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::Command,
};

use super::{ToolFreeEnvironment, tool_free_command, tool_free_environment};

fn os(text: &str) -> OsString {
    OsString::from(text)
}

fn slang_folder() -> PathBuf {
    PathBuf::from("slang-sdk").join("bin")
}

fn other_folder() -> PathBuf {
    PathBuf::from("tools").join("bin")
}

fn parent(helper: bool) -> Vec<(OsString, OsString)> {
    let path = std::env::join_paths([slang_folder(), other_folder()]).unwrap();
    let mut parent = vec![
        (os("PATH"), path),
        (os("SLANG_SDK"), os("slang-sdk")),
        (os("GUST_SLANG_HOME"), os("cache")),
        (os("VULKAN_SDK"), os("vulkan")),
        (os("VK_ADD_DRIVER_FILES"), os("extra-driver")),
        (os("RUSTFLAGS"), os("-C debuginfo=0")),
        (os("GUST_PROFILE"), os("dev")),
        (os("HOME"), os("home")),
    ];
    if helper {
        parent.push((os("GUST_SLANG_REFLECT"), os("helper")));
    }
    parent
}

fn compute(helper: bool, hide_vulkan: bool) -> ToolFreeEnvironment {
    tool_free_environment(
        &parent(helper),
        &|folder: &Path| folder == slang_folder(),
        Path::new("empty-home"),
        hide_vulkan,
    )
}

fn value<'a>(environment: &'a ToolFreeEnvironment, name: &str) -> Option<&'a OsStr> {
    environment
        .set
        .iter()
        .find(|(set, _)| set == name)
        .map(|(_, value)| value.as_os_str())
}

fn names(unset: &[OsString]) -> Vec<String> {
    let mut names: Vec<String> = unset
        .iter()
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The variables that name Slang (the SDK, the home) and the Vulkan SDK are unset, and `PATH` loses the folder with a
/// Slang tool; the profile, the rustflags and the rest of the environment are not touched (they would change the build,
/// and the strict profile refuses `slang-runtime`). The helper is not unset but set, see the next test.
#[test]
fn the_slang_variables_are_unset_and_path_loses_its_slang_folders() {
    let environment = compute(true, false);
    assert_eq!(names(&environment.unset), ["SLANG_SDK", "VULKAN_SDK"]);
    assert_eq!(environment.dropped_folders, 1);
    assert_eq!(
        value(&environment, "PATH"),
        Some(std::env::join_paths([other_folder()]).unwrap().as_os_str())
    );
    assert_eq!(
        value(&environment, "GUST_SLANG_HOME"),
        Some(OsStr::new("empty-home"))
    );
    for left_alone in ["RUSTFLAGS", "GUST_PROFILE", "HOME"] {
        assert!(!environment.unset.iter().any(|name| name == left_alone));
        assert!(value(&environment, left_alone).is_none(), "{left_alone}");
    }
}

/// An unset `GUST_SLANG_REFLECT` would let the lookup fall back to the helper built in the checkout, which CI's runner
/// has not; an explicit path is returned as given, so the variable names a file below the empty home that does not
/// exist. `workspace_command` exports the repository's helper when the parent names none, so this holds either way.
#[test]
fn the_helper_names_a_missing_file_whether_or_not_the_parent_names_one() {
    for helper in [true, false] {
        let environment = compute(helper, false);
        assert!(
            !environment
                .unset
                .iter()
                .any(|name| name == "GUST_SLANG_REFLECT"),
            "parent names the helper: {helper}"
        );
        let expected = Path::new("empty-home")
            .join("no-helper")
            .join("gust-slang-reflect");
        assert_eq!(
            value(&environment, "GUST_SLANG_REFLECT"),
            Some(expected.as_os_str()),
            "parent names the helper: {helper}"
        );
    }
}

#[test]
fn the_vulkan_loader_finds_no_driver_only_when_asked() {
    let hidden = compute(false, true);
    let nowhere = Path::new("empty-home").join("no-vulkan-driver");
    for name in ["VK_DRIVER_FILES", "VK_ICD_FILENAMES"] {
        assert_eq!(value(&hidden, name), Some(nowhere.as_os_str()), "{name}");
    }
    // The loader adds the drivers of this variable whatever the two above say.
    assert!(
        names(&hidden.unset).contains(&"VK_ADD_DRIVER_FILES".to_owned()),
        "{:?}",
        hidden.unset
    );
    let visible = compute(false, false);
    for name in ["VK_DRIVER_FILES", "VK_ICD_FILENAMES"] {
        assert!(value(&visible, name).is_none(), "{name}");
    }
    assert!(!names(&visible.unset).contains(&"VK_ADD_DRIVER_FILES".to_owned()));
}

/// What a child will get: an unset variable is removed (`None`), a set one has its value, the others are not mentioned.
fn child_environment_of(command: &Command) -> Vec<(String, Option<String>)> {
    command
        .get_envs()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.map(|value| value.to_string_lossy().into_owned()),
            )
        })
        .collect()
}

fn lookup(envs: &[(String, Option<String>)], name: &str) -> Option<Option<String>> {
    envs.iter()
        .find(|(set, _)| set == name)
        .map(|(_, value)| value.clone())
}

#[test]
fn applied_to_a_command_the_changes_reach_its_environment() {
    let environment = compute(true, true);
    let mut command = Command::new("cargo");
    command.env("GUST_SLANG_REFLECT", "exported by workspace_command");
    environment.apply(&mut command);
    let envs = child_environment_of(&command);
    for unset in ["SLANG_SDK", "VULKAN_SDK", "VK_ADD_DRIVER_FILES"] {
        assert_eq!(lookup(&envs, unset), Some(None), "{unset} must be removed");
    }
    let helper = Path::new("empty-home")
        .join("no-helper")
        .join("gust-slang-reflect");
    assert_eq!(
        lookup(&envs, "GUST_SLANG_REFLECT"),
        Some(Some(helper.to_string_lossy().into_owned()))
    );
    assert_eq!(
        lookup(&envs, "GUST_SLANG_HOME"),
        Some(Some("empty-home".to_owned()))
    );
    assert!(lookup(&envs, "VK_DRIVER_FILES").is_some_and(|value| value.is_some()));
    assert!(lookup(&envs, "VK_ICD_FILENAMES").is_some_and(|value| value.is_some()));
    assert_eq!(lookup(&envs, "GUST_PROFILE"), None);
    assert_eq!(lookup(&envs, "RUSTFLAGS"), None);
}

/// The wiring of the step: the command that `run_counted_tool_free` runs is the workspace's cargo child (it has the
/// workspace profile) with the stripped environment applied, so the step stops being tool-free if either goes missing.
#[test]
fn the_command_of_the_step_is_the_workspace_cargo_child_with_the_stripped_environment() {
    let home = std::env::temp_dir().join("tool-free-wiring-test-home");
    let (command, _) = tool_free_command(&parent(true), &home, true);
    assert_eq!(command.get_program(), "cargo");
    let envs = child_environment_of(&command);
    assert_eq!(
        lookup(&envs, "GUST_PROFILE"),
        Some(Some("dev".to_owned())),
        "the workspace profile"
    );
    for unset in ["SLANG_SDK", "VULKAN_SDK", "VK_ADD_DRIVER_FILES"] {
        assert_eq!(lookup(&envs, unset), Some(None), "{unset} must be removed");
    }
    let helper = home.join("no-helper").join("gust-slang-reflect");
    assert_eq!(
        lookup(&envs, "GUST_SLANG_REFLECT"),
        Some(Some(helper.to_string_lossy().into_owned()))
    );
    assert!(helper.is_absolute(), "the lookup refuses a relative helper");
    assert!(!helper.exists());
    assert_eq!(
        lookup(&envs, "GUST_SLANG_HOME"),
        Some(Some(home.to_string_lossy().into_owned()))
    );
}
