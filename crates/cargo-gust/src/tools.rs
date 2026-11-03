//! The external tools the export uses: the native reflection helper (`gust-slang-reflect`) and `spirv-val`.
//!
//! Lookups take the `PATH` value as an argument, so a test never changes the process environment.

use std::{
    env::consts::EXE_SUFFIX,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
};

/// The first directory of `path_var` that holds a regular file called `file_name`.
pub(crate) fn find_on_path(file_name: &str, path_var: Option<&OsStr>) -> Option<PathBuf> {
    let path_var = path_var?;
    std::env::split_paths(path_var)
        .filter(|directory| !directory.as_os_str().is_empty())
        .map(|directory| directory.join(file_name))
        .find(|candidate| candidate.is_file())
}

/// The executable file name of a tool called `name` on this platform.
pub(crate) fn executable_name(name: &str) -> String {
    format!("{name}{EXE_SUFFIX}")
}

/// The helper executable to use, or `None` when it is not there. `helper` is what
/// `gust::reflect::native_compiler_path` returned: an explicit path (from `GUST_SLANG_REFLECT`, or the repository's
/// `target/slang-reflect`), or a bare file name to look up on `PATH`.
pub(crate) fn locate_executable(helper: &Path, path_var: Option<&OsStr>) -> Option<PathBuf> {
    let is_bare = helper
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty());
    if is_bare {
        helper
            .file_name()
            .and_then(OsStr::to_str)
            .and_then(|name| find_on_path(name, path_var))
    } else {
        helper.is_file().then(|| helper.to_path_buf())
    }
}

/// `spirv-val` on `PATH`, if any.
pub(crate) fn find_spirv_val(path_var: Option<&OsStr>) -> Option<PathBuf> {
    find_on_path(&executable_name("spirv-val"), path_var)
}

/// What to do when the helper is missing; printed by `doctor` and by the commands that need it.
pub(crate) fn helper_hint() -> &'static str {
    "build it with `cargo gust slang build-helper` (in the GUST repository `cargo xtask build-slang-reflect` does the same), or set GUST_SLANG_REFLECT to the helper executable"
}

/// Read the Slang build tag from the output of `gust-slang-reflect --version`:
/// `gust-slang-reflect schema=<1|2|3> slang=<tag>` (the highest reflection schema the helper writes).
pub(crate) fn parse_helper_version(output: &str) -> Result<String, String> {
    parse_helper_report(output).map(|(_, tag)| tag)
}

/// The schema and the Slang build tag of the output of `gust-slang-reflect --version`, with the refusals of
/// [`parse_helper_version`].
pub(crate) fn parse_helper_report(output: &str) -> Result<(String, String), String> {
    let line = output.lines().next().unwrap_or_default().trim();
    let mut words = line.split_whitespace();
    if words.next() != Some("gust-slang-reflect") {
        return Err(format!(
            "unexpected `--version` output from the helper: `{line}`"
        ));
    }
    let mut schema = None;
    let mut slang = None;
    for word in words {
        if let Some(value) = word.strip_prefix("schema=") {
            schema = Some(value);
        } else if let Some(value) = word.strip_prefix("slang=") {
            slang = Some(value);
        }
    }
    match (schema, slang) {
        (Some(schema @ ("1" | "2" | "3")), Some(tag)) if !tag.is_empty() => {
            Ok((schema.to_owned(), tag.to_owned()))
        }
        (Some(other), _) if !matches!(other, "1" | "2" | "3") => Err(format!(
            "the helper reports schema {other}; this tool reads schemas 1, 2 and 3"
        )),
        _ => Err(format!(
            "the helper's `--version` output has no schema and slang tag: `{line}`"
        )),
    }
}

/// Run `<helper> --version` and read the Slang build tag.
pub(crate) fn helper_slang_version(helper: &Path) -> Result<String, String> {
    parse_helper_version(&helper_version_output(helper)?)
}

/// Run `<helper> --version` and return what it printed on standard output.
pub(crate) fn helper_version_output(helper: &Path) -> Result<String, String> {
    let output = Command::new(helper)
        .arg("--version")
        .output()
        .map_err(|error| format!("cannot run the helper `{}`: {error}", helper.display()))?;
    if !output.status.success() {
        return Err(format!(
            "`{} --version` failed: {}",
            helper.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::Scratch;
    use std::fs;

    #[test]
    fn the_helper_version_line_gives_the_slang_tag() {
        assert_eq!(
            parse_helper_version("gust-slang-reflect schema=1 slang=2026.13.1-1-g84792eb15\n")
                .unwrap(),
            "2026.13.1-1-g84792eb15"
        );
        // The grown helpers (schemas 2 and 3) name the same tag.
        for schema in ["2", "3"] {
            assert_eq!(
                parse_helper_version(&format!(
                    "gust-slang-reflect schema={schema} slang=2026.13.1-1-g84792eb15\n"
                ))
                .unwrap(),
                "2026.13.1-1-g84792eb15"
            );
        }
        for (bad, expected) in [
            ("", "unexpected"),
            ("slangc 2026.13.1\n", "unexpected"),
            ("gust-slang-reflect schema=4 slang=x\n", "schema 4"),
            ("gust-slang-reflect slang=x\n", "no schema"),
            ("gust-slang-reflect schema=1 slang=\n", "no schema"),
            ("gust-slang-reflect schema=1\n", "no schema"),
        ] {
            let error = parse_helper_version(bad).unwrap_err();
            assert!(error.contains(expected), "{bad:?}: {error}");
        }
    }

    #[test]
    fn path_lookup_finds_the_first_directory_with_the_file_and_ignores_empty_entries() {
        let scratch = Scratch::new("path");
        let first = scratch.0.join("first");
        let second = scratch.0.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        fs::write(second.join("tool"), "x").unwrap();
        fs::create_dir(first.join("tool-dir")).unwrap();
        let joined = std::env::join_paths([first.clone(), second.clone()]).unwrap();
        assert_eq!(
            find_on_path("tool", Some(&joined)),
            Some(second.join("tool"))
        );
        // A directory with the name is not a tool.
        assert_eq!(find_on_path("tool-dir", Some(&joined)), None);
        assert_eq!(find_on_path("tool", None), None);
        assert_eq!(find_on_path("missing", Some(&joined)), None);
        assert_eq!(find_on_path("tool", Some(OsStr::new(""))), None);
    }

    #[test]
    fn an_explicit_helper_path_must_exist_and_a_bare_name_is_looked_up_on_path() {
        let scratch = Scratch::new("helper");
        let dir = scratch.0.join("bin");
        fs::create_dir_all(&dir).unwrap();
        let name = executable_name("gust-slang-reflect");
        fs::write(dir.join(&name), "x").unwrap();
        let joined = std::env::join_paths([dir.clone()]).unwrap();
        assert_eq!(
            locate_executable(Path::new(&name), Some(&joined)),
            Some(dir.join(&name))
        );
        assert_eq!(locate_executable(Path::new(&name), None), None);
        assert_eq!(
            locate_executable(&dir.join(&name), None),
            Some(dir.join(&name))
        );
        assert_eq!(locate_executable(&dir.join("nope"), Some(&joined)), None);
        assert_eq!(find_spirv_val(Some(&joined)), None);
        fs::write(dir.join(executable_name("spirv-val")), "x").unwrap();
        assert_eq!(
            find_spirv_val(Some(&joined)),
            Some(dir.join(executable_name("spirv-val")))
        );
    }
}
