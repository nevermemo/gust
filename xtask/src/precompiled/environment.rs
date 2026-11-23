//! The environment of the children of `check-precompiled` (E1): the parent's, without every variable and every `PATH`
//! directory that names Slang, and the checks that the result is Slang-free.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

use super::STRICT_PROFILE;
use crate::{checks::RUSTFLAGS_VARIABLES, process::PROFILE_VARIABLE};

/// The files that make a directory of `PATH` a place Slang can be launched from.
const SLANG_TOOLS: [&str; 4] = [
    "slangc",
    "slangc.exe",
    "gust-slang-reflect",
    "gust-slang-reflect.exe",
];

/// The variables, besides those with `SLANG` in the name, that E1 takes out of the children: the Vulkan SDK ships
/// `slangc`, and a stray export directory belongs to `cargo gust` alone (the macro would stay in export mode).
const STRIPPED_VARIABLES: [&str; 3] = ["VULKAN_SDK", "VK_SDK_PATH", "GUST_EXPORT_DIR"];

/// The variable names of the child that E1 must not contain.
pub(crate) fn is_slang_variable(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.contains("SLANG") || STRIPPED_VARIABLES.contains(&upper.as_str())
}

/// Whether `name` is a variable that cargo reads rustflags from (such a variable would change the builds).
fn is_rustflags_variable(name: &str) -> bool {
    RUSTFLAGS_VARIABLES
        .iter()
        .any(|variable| variable.eq_ignore_ascii_case(name))
}

/// A directory of `PATH` that holds a Slang tool.
pub(crate) fn directory_has_slang_tool(directory: &Path) -> bool {
    SLANG_TOOLS
        .iter()
        .any(|tool| directory.join(tool).is_file())
}

/// `path` without the directories for which `has_tool` is true, and those directories in their order.
pub(crate) fn strip_path(
    path: &OsStr,
    has_tool: &dyn Fn(&Path) -> bool,
) -> (OsString, Vec<PathBuf>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for directory in env::split_paths(path) {
        if has_tool(&directory) {
            dropped.push(directory);
        } else {
            kept.push(directory);
        }
    }
    // `join_paths` fails only for an entry that holds the separator, which `split_paths` never produces.
    let joined = env::join_paths(kept).unwrap_or_else(|_| path.to_owned());
    (joined, dropped)
}

/// The environment of a child: the parent's, without the Slang variables, the rustflags variables and the Slang
/// directories of `PATH`, and with `GUST_PROFILE=native`. `report` collects what was taken out.
pub(crate) fn child_environment(
    parent: &[(OsString, OsString)],
    has_tool: &dyn Fn(&Path) -> bool,
    report: &mut StripReport,
) -> Vec<(OsString, OsString)> {
    let mut child = Vec::new();
    for (name, value) in parent {
        let text = name.to_string_lossy();
        if is_slang_variable(&text) {
            report.variables.push(text.into_owned());
        } else if is_rustflags_variable(&text) || text.eq_ignore_ascii_case(PROFILE_VARIABLE) {
            // Set again below (the profile) or left to cargo's own sources (rustflags).
        } else if text.eq_ignore_ascii_case("PATH") {
            let (stripped, dropped) = strip_path(value, has_tool);
            report.directories.extend(dropped);
            child.push((name.clone(), stripped));
        } else {
            child.push((name.clone(), value.clone()));
        }
    }
    child.push((
        OsString::from(PROFILE_VARIABLE),
        OsString::from(STRICT_PROFILE),
    ));
    child
}

/// What E1 took out of the children's environment.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct StripReport {
    /// The variables that were removed (names only; the values may be paths).
    pub(crate) variables: Vec<String>,
    /// The `PATH` directories that were removed.
    pub(crate) directories: Vec<PathBuf>,
}

/// The problems of a child environment that is supposed to have no Slang: a variable that names Slang, a `PATH`
/// directory that holds a Slang tool, a missing `GUST_PROFILE=native`.
pub(crate) fn slang_free_problems(
    environment: &[(OsString, OsString)],
    has_tool: &dyn Fn(&Path) -> bool,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (name, value) in environment {
        let text = name.to_string_lossy();
        if is_slang_variable(&text) {
            problems.push(format!("the variable {text} is set"));
        }
        if text.eq_ignore_ascii_case("PATH") {
            for directory in env::split_paths(value).filter(|directory| has_tool(directory)) {
                problems.push(format!(
                    "the PATH directory {} holds a Slang tool",
                    directory.display()
                ));
            }
        }
    }
    let profile = environment.iter().find(|(name, _)| {
        name.to_string_lossy()
            .eq_ignore_ascii_case(PROFILE_VARIABLE)
    });
    if profile.is_none_or(|(_, value)| value != STRICT_PROFILE) {
        problems.push(format!("{PROFILE_VARIABLE} is not {STRICT_PROFILE}"));
    }
    problems
}

#[cfg(test)]
#[path = "../tests/precompiled_environment.rs"]
mod tests;
