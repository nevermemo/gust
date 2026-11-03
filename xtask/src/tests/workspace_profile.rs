use std::{
    ffi::{OsStr, OsString},
    process::Command,
};

use super::{PROFILE_VARIABLE, profile_override_note, set_workspace_profile, workspace_command};

/// The change a command makes to `GUST_PROFILE` in its child's environment: `Some(Some(value))` for a set,
/// `Some(None)` for a removal, `None` when the command leaves the variable alone.
fn profile_of(command: &Command) -> Option<Option<OsString>> {
    command
        .get_envs()
        .find(|(name, _)| *name == OsStr::new(PROFILE_VARIABLE))
        .map(|(_, value)| value.map(ToOwned::to_owned))
}

#[test]
fn a_cargo_child_gets_the_dev_profile() {
    let expected = Some(Some(OsString::from("dev")));
    assert_eq!(profile_of(&workspace_command("cargo")), expected);
    // `$CARGO` is usually a full path, with the platform's executable suffix.
    assert_eq!(profile_of(&workspace_command("cargo.exe")), expected);
    assert_eq!(profile_of(&workspace_command("tools/bin/cargo")), expected);
    assert_eq!(
        profile_of(&workspace_command("C:/tools/bin/cargo.exe")),
        expected
    );
}

#[test]
fn a_program_that_is_not_cargo_is_left_alone() {
    for program in [
        "git",
        "slangc",
        "spirv-val",
        "rustc",
        "cargo-fmt",
        "mycargo",
        "",
    ] {
        assert_eq!(profile_of(&workspace_command(program)), None, "{program:?}");
    }
}

#[test]
fn set_workspace_profile_sets_it_on_any_command() {
    let mut command = Command::new("anything");
    set_workspace_profile(&mut command);
    assert_eq!(profile_of(&command), Some(Some(OsString::from("dev"))));
    // A second call keeps the one value.
    set_workspace_profile(&mut command);
    assert_eq!(profile_of(&command), Some(Some(OsString::from("dev"))));
}

#[test]
fn only_a_different_preset_value_is_logged() {
    assert_eq!(profile_override_note(None), None);
    assert_eq!(profile_override_note(Some(OsStr::new("dev"))), None);
    assert_eq!(
        profile_override_note(Some(OsStr::new("native"))).as_deref(),
        Some(
            "xtask: GUST_PROFILE=native is set in the environment; workspace builds run under GUST_PROFILE=dev instead"
        )
    );
    // An empty or case-different value is not `dev`, and is overridden like any other.
    assert!(profile_override_note(Some(OsStr::new(""))).is_some());
    assert!(profile_override_note(Some(OsStr::new("DEV"))).is_some());
}

// ---- the `slang-runtime` feature in the manifests (D44) ----

/// The non-blank lines of the `[name]` table of a manifest, trimmed. Reads text, so it needs no TOML parser;
/// a manifest that spells the table another way fails the test, which is the safe direction.
fn table(manifest: &str, name: &str) -> Vec<String> {
    let root = crate::workspace::workspace_root().unwrap();
    let text = std::fs::read_to_string(root.join(manifest)).unwrap();
    let mut inside = false;
    let mut lines = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            inside = line == format!("[{name}]");
        } else if inside && !line.is_empty() && !line.starts_with('#') {
            lines.push(line.to_owned());
        }
    }
    lines
}

#[test]
fn slang_runtime_is_a_feature_of_gust_and_gust_wgpu_and_is_off_by_default() {
    assert_eq!(
        table("crates/gust/Cargo.toml", "features"),
        ["slang-runtime = []", "cargo-gust-bridge = []"]
    );
    assert_eq!(
        table("crates/gust-wgpu/Cargo.toml", "features"),
        [r#"slang-runtime = ["gust/slang-runtime"]"#]
    );
}

/// The whole text of a manifest, for the tables `table` cannot see (a `[dependencies.gust]` table of its own).
fn manifest_text(manifest: &str) -> String {
    let root = crate::workspace::workspace_root().unwrap();
    std::fs::read_to_string(root.join(manifest)).unwrap()
}

/// Whether `line` (trimmed) declares the dependency `name`, in any spelling of the key (`name = ..`, `name.key = ..`).
fn declares(line: &str, name: &str) -> bool {
    line.strip_prefix(name)
        .is_some_and(|rest| rest.starts_with([' ', '.', '=']))
}

#[test]
fn no_dependency_entry_turns_the_feature_on_for_every_consumer() {
    // A `features = ["slang-runtime"]` on the `gust` entry of gust-wgpu or of the workspace root would put the bridge
    // into every build; `-p gust -p gust-wgpu --no-default-features` would not notice, because both unify.
    for (manifest, table_name, names) in [
        ("crates/gust-wgpu/Cargo.toml", "dependencies", &["gust"][..]),
        (
            "Cargo.toml",
            "workspace.dependencies",
            &["gust", "gust-wgpu"][..],
        ),
    ] {
        let lines = table(manifest, table_name);
        for name in names {
            let entries: Vec<_> = lines.iter().filter(|line| declares(line, name)).collect();
            assert_eq!(
                entries.len(),
                1,
                "{manifest} [{table_name}] {name}: {entries:?}"
            );
            for entry in entries {
                assert!(
                    !entry.contains("slang-runtime") && !entry.contains("features"),
                    "{manifest} [{table_name}] {name}: {entry}"
                );
            }
        }
        // A table of its own for the entry would hide its features from `table`.
        let text = manifest_text(manifest);
        for name in names {
            assert!(
                !text.contains(&format!("[{table_name}.{name}]")),
                "{manifest}: {name}"
            );
        }
    }
}

#[test]
fn the_tests_of_both_crates_and_every_example_turn_the_feature_on() {
    assert_eq!(
        table("crates/gust/Cargo.toml", "dev-dependencies"),
        [r#"gust = { path = ".", features = ["slang-runtime"] }"#]
    );
    assert_eq!(
        table("crates/gust-wgpu/Cargo.toml", "dev-dependencies"),
        [r#"gust-wgpu = { path = ".", features = ["slang-runtime"] }"#]
    );
    for example in crate::workspace::EXAMPLES {
        let dependencies = table(&format!("examples/{example}/Cargo.toml"), "dependencies");
        assert!(
            dependencies.contains(
                &r#"gust-wgpu = { workspace = true, features = ["slang-runtime"] }"#.to_owned()
            ),
            "{example}: {dependencies:?}"
        );
    }
}
