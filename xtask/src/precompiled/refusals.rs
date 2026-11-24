//! E6a to E6j: the refusals of `gust`'s build script, judged on their exact messages, and the positive controls.

use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{
    CONTROL_PROFILE, PRECOMPILED_DIR,
    fixture::{Fixture, copy_fixture},
    profiles::e6_profile_values,
};
use crate::{
    process::run_capture_in,
    ui::{Expectation, judge, parse_expectation},
};

// ---------------------------------------------------------------------------------------------------------------
// E6

/// The expectation of the explicit `native` refusals: the `build-error:` text of `gust`'s build script, relative to
/// [`PRECOMPILED_DIR`].
const E6_NATIVE_EXPECTED: &str = "e6-native-expected.txt";

/// The same for the refusal of a release build without a variable: the text of the first plus the sentence that says
/// the profile was inferred.
const E6_INFERRED_EXPECTED: &str = "e6-inferred-expected.txt";

/// The expectation of the cases in which `cargo-gust-bridge` made the profile `dev`: the `warning:` lines that cargo
/// must print, relative to [`PRECOMPILED_DIR`].
const E6_BRIDGE_EXPECTED: &str = "e6-bridge-expected.txt";

/// The features of the bridge cases: the internal feature, and the bridge that a strict profile refuses.
const BRIDGE_FEATURES: &str = "gust/cargo-gust-bridge,gust/slang-runtime";

/// The sentence of the refusal that only an inferred profile has (`crates/gust/build.rs`).
const INFERRED_SENTENCE: &str =
    "This profile was inferred from a release build without GUST_PROFILE.";

/// What the build script prints for cargo when a release build had no `GUST_PROFILE` and was treated as `native`.
const INFERRED_WARNING: &str = "gust: release build without GUST_PROFILE treated as native";

/// Whether cargo's stderr `text` has the build-script warning `warning` of the package `gust-compute` as a whole line: cargo
/// prints it as `warning: gust-compute@<version>: <warning>`, so a line that continues after the words (`... as devX`) does not
/// count.
pub(crate) fn has_build_script_warning(text: &str, warning: &str) -> bool {
    text.lines().any(|line| {
        line.trim_end()
            .strip_prefix("warning: gust-compute@")
            .and_then(|rest| rest.split_once(": "))
            .is_some_and(|(version, printed)| {
                !version.is_empty() && !version.contains(' ') && printed == warning
            })
    })
}

/// Whether cargo's stderr `text` has the warning of cargo itself `warning` as a whole line: `warning: <warning>`, or,
/// for a warning about a dependency of the case's `app`, `warning: app v<version> (<path>) <warning>` (cargo names
/// the package that declares the dependency; the path is the machine's and is not matched).
pub(crate) fn has_cargo_warning(text: &str, warning: &str) -> bool {
    text.lines().any(|line| {
        line.trim_end()
            .strip_prefix("warning: ")
            .is_some_and(|rest| {
                rest == warning
                    || (rest.starts_with("app v") && rest.ends_with(&format!(") {warning}")))
            })
    })
}

/// The `GUST_PROFILE` of an E6 case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum E6Profile {
    Native,
    Dev,
    /// The variable is removed, so a release build infers its profile.
    Unset,
}

/// What an E6 case must do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum E6Outcome {
    /// The build script of `gust` refuses. `expected` holds the `build-error:` lines (relative to
    /// [`PRECOMPILED_DIR`]); `inferred` says whether the message must (or must not) carry [`INFERRED_SENTENCE`].
    Refused {
        expected: &'static str,
        inferred: bool,
    },
    /// The build compiles. With `warning`, cargo must also have printed that text (the inference was visible).
    Compiles { warning: Option<&'static str> },
    /// The build compiles with the run-time bridge in it because `cargo-gust-bridge` made the profile `dev`. `expected`
    /// holds the `warning:` lines (relative to [`PRECOMPILED_DIR`]) that cargo must print; the plain inference text
    /// ([`INFERRED_WARNING`], `native`) must not be there. With `ignored_tool`, the `cargo-warning:` lines of the file
    /// must be there too (cargo's own warning about a `cargo-gust` dependency, E6j).
    CompilesAsDev {
        expected: &'static str,
        ignored_tool: bool,
    },
}

/// One `cargo check -p app` of E6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct E6Case {
    pub(crate) id: &'static str,
    /// `--features`, if any.
    pub(crate) features: Option<&'static str>,
    pub(crate) release: bool,
    pub(crate) profile: E6Profile,
    /// E6c: built in a scratch copy in which `app` depends on a crate that enables `gust-wgpu/slang-runtime`; that
    /// copy's lockfile changes, so the case runs without `--locked` and never in the fixture.
    pub(crate) bridge_user: bool,
    /// E6i: checks the whole workspace of the repository (`--workspace`, the root manifest) instead of `-p app`.
    pub(crate) workspace: bool,
    /// E6j: built in a scratch copy in which `app` has `cargo-gust` as a normal dependency (a mistake: it has no
    /// library, so cargo ignores the dependency, but the features of the edge still unify). Like E6c, never `--locked`.
    pub(crate) tool_dependency: bool,
    pub(crate) outcome: E6Outcome,
}

impl E6Case {
    /// Whether the case builds in a scratch copy of the fixture (whose lockfile then changes).
    pub(crate) fn in_copy(&self) -> bool {
        self.bridge_user || self.tool_dependency
    }
}

/// The cases of E6, in order. E6f is E4 and is not repeated.
pub(crate) fn e6_cases() -> [E6Case; 10] {
    let native = E6Outcome::Refused {
        expected: E6_NATIVE_EXPECTED,
        inferred: false,
    };
    [
        E6Case {
            id: "E6a",
            features: Some("gust/slang-runtime"),
            release: false,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6b",
            features: Some("gust-wgpu/slang-runtime"),
            release: false,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6c",
            features: None,
            release: false,
            profile: E6Profile::Native,
            bridge_user: true,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6c-release",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: true,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Refused {
                expected: E6_INFERRED_EXPECTED,
                inferred: true,
            },
        },
        E6Case {
            id: "E6d",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Compiles {
                warning: Some(INFERRED_WARNING),
            },
        },
        E6Case {
            id: "E6e",
            features: Some("gust/slang-runtime"),
            release: true,
            profile: E6Profile::Dev,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::Compiles { warning: None },
        },
        E6Case {
            id: "E6g",
            features: Some(BRIDGE_FEATURES),
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: false,
            },
        },
        E6Case {
            id: "E6h",
            features: Some(BRIDGE_FEATURES),
            release: true,
            profile: E6Profile::Native,
            bridge_user: false,
            workspace: false,
            tool_dependency: false,
            outcome: native,
        },
        E6Case {
            id: "E6i",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: true,
            tool_dependency: false,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: false,
            },
        },
        E6Case {
            id: "E6j",
            features: None,
            release: true,
            profile: E6Profile::Unset,
            bridge_user: false,
            workspace: false,
            tool_dependency: true,
            outcome: E6Outcome::CompilesAsDev {
                expected: E6_BRIDGE_EXPECTED,
                ignored_tool: true,
            },
        },
    ]
}

/// The arguments of the `cargo check` of one E6 case.
pub(crate) fn e6_args(case: &E6Case, manifest: &str, target_dir: &str) -> Vec<String> {
    let mut args: Vec<String> = ["check"].iter().map(|arg| (*arg).to_owned()).collect();
    if !case.in_copy() {
        args.push("--locked".to_owned());
    }
    for arg in [
        "--color",
        "never",
        "--message-format",
        "json",
        "--manifest-path",
        manifest,
    ] {
        args.push(arg.to_owned());
    }
    if case.workspace {
        args.push("--workspace".to_owned());
    } else {
        args.push("-p".to_owned());
        args.push("app".to_owned());
    }
    if case.release {
        args.push("--release".to_owned());
    }
    if let Some(features) = case.features {
        args.push("--features".to_owned());
        args.push(features.to_owned());
    }
    args.push("--target-dir".to_owned());
    args.push(target_dir.to_owned());
    args
}

/// The problems of one E6 case for its outcome. `expectation` is the text of the case's expectation file (`None` for a
/// case that must compile). A refusal is judged on the exact message of the build script by [`judge`], never on the
/// exit status alone; the sentence about an inferred profile must be there for the inferred case and absent for the
/// explicit ones.
pub(crate) fn e6_problems(
    case: &E6Case,
    expectation: Option<&str>,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<String>, String> {
    let text = stderr.replace('\\', "/");
    match case.outcome {
        E6Outcome::Refused { expected, inferred } => {
            let source = expectation.ok_or_else(|| format!("{}: no expectation text", case.id))?;
            let wanted =
                parse_expectation(source).map_err(|error| format!("{expected}: {error}"))?;
            if !matches!(wanted, Expectation::BuildFails(_)) {
                return Err(format!(
                    "{expected} must hold `build-error:` lines (the refusal is a build-script failure), not rustc keys"
                ));
            }
            let mut problems = judge(case.id, "app", &wanted, success, stdout, stderr);
            if text.contains(INFERRED_SENTENCE) != inferred {
                problems.push(format!(
                    "{}: the refusal {} the sentence `{INFERRED_SENTENCE}`",
                    case.id,
                    if inferred {
                        "lacks"
                    } else {
                        "must not have, but has,"
                    }
                ));
            }
            Ok(problems)
        }
        E6Outcome::Compiles { warning } => {
            let mut problems = judge(
                case.id,
                "app",
                &Expectation::Compiles,
                success,
                stdout,
                stderr,
            );
            if let Some(warning) = warning
                && !has_build_script_warning(&text, warning)
            {
                problems.push(format!(
                    "{}: cargo did not print `{warning}`, so nothing shows the profile was inferred",
                    case.id
                ));
            }
            Ok(problems)
        }
        E6Outcome::CompilesAsDev {
            expected,
            ignored_tool,
        } => {
            let source = expectation.ok_or_else(|| format!("{}: no expectation text", case.id))?;
            let wanted = parse_warning_expectation(source)
                .map_err(|error| format!("{expected}: {error}"))?;
            if ignored_tool && wanted.cargo.is_empty() {
                return Err(format!(
                    "{expected}: {} needs a `cargo-warning:` line",
                    case.id
                ));
            }
            let mut problems = judge(
                case.id,
                if case.workspace {
                    "the workspace"
                } else {
                    "app"
                },
                &Expectation::Compiles,
                success,
                stdout,
                stderr,
            );
            for warning in &wanted.build_script {
                if !has_build_script_warning(&text, warning) {
                    problems.push(format!(
                        "{}: cargo did not print `{warning}`, so nothing shows that the bridge feature changed \
                         the profile",
                        case.id
                    ));
                }
            }
            if ignored_tool {
                for warning in &wanted.cargo {
                    if !has_cargo_warning(&text, warning) {
                        problems.push(format!(
                            "{}: cargo did not print `warning: {warning}`, so the dependency on `cargo-gust` is not \
                             the one this case pins",
                            case.id
                        ));
                    }
                }
            }
            if has_build_script_warning(&text, INFERRED_WARNING) {
                problems.push(format!(
                    "{}: cargo printed `{INFERRED_WARNING}`, so the profile was inferred as native and not made dev",
                    case.id
                ));
            }
            Ok(problems)
        }
    }
}

/// The lines of an expectation of the bridge cases: `warning:` is the text of the build script's `cargo:warning` that
/// cargo must print (as `warning: gust-compute@<version>: <text>`), `cargo-warning:` a warning of cargo itself (as
/// `warning: <text>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BridgeExpectation {
    pub(crate) build_script: Vec<String>,
    pub(crate) cargo: Vec<String>,
}

/// Parses an expectation of the bridge cases (blank lines and lines that start with `#` are ignored). Any other line,
/// or no `warning:` line, is an error.
pub(crate) fn parse_warning_expectation(text: &str) -> Result<BridgeExpectation, String> {
    let mut wanted = BridgeExpectation {
        build_script: Vec::new(),
        cargo: Vec::new(),
    };
    for raw in text.trim_start_matches('\u{feff}').lines() {
        let entry = raw.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        if let Some(value) = entry.strip_prefix("warning:").map(str::trim)
            && !value.is_empty()
        {
            wanted.build_script.push(value.to_owned());
        } else if let Some(value) = entry.strip_prefix("cargo-warning:").map(str::trim)
            && !value.is_empty()
        {
            wanted.cargo.push(value.to_owned());
        } else {
            return Err(format!(
                "`{entry}` is not a `warning: <text>` or `cargo-warning: <text>` line"
            ));
        }
    }
    if wanted.build_script.is_empty() {
        return Err("no `warning:` line".to_owned());
    }
    Ok(wanted)
}

/// The manifest of the crate that E6c adds: it depends on `gust-compute-wgpu` with the bridge feature. `root` is the
/// repository, with forward slashes.
pub(crate) fn bridge_user_manifest(root: &str) -> String {
    format!(
        "[package]\nname = \"bridge-user\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngust-wgpu = {{ package = \"gust-compute-wgpu\", path = \"{root}/crates/gust-wgpu\", features = [\"slang-runtime\"] }}\n"
    )
}

/// The fixture's workspace manifest with `bridge-user` as a third member.
pub(crate) fn with_bridge_user_member(workspace: &str) -> Result<String, String> {
    const MEMBERS: &str = "members = [\"kernels\", \"app\"]";
    if !workspace.contains(MEMBERS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/Cargo.toml no longer has the line `{MEMBERS}`; E6c cannot add its crate"
        ));
    }
    Ok(workspace.replace(MEMBERS, "members = [\"kernels\", \"app\", \"bridge-user\"]"))
}

/// `app`'s manifest with a normal dependency on `bridge-user`, after its dependency on `kernels`.
pub(crate) fn with_bridge_user_dependency(app: &str) -> Result<String, String> {
    const KERNELS: &str = "kernels = { path = \"../kernels\" }";
    if !app.contains(KERNELS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/app/Cargo.toml no longer has the line `{KERNELS}`; E6c cannot add its dependency"
        ));
    }
    Ok(app.replace(
        KERNELS,
        &format!("{KERNELS}\nbridge-user = {{ path = \"../bridge-user\" }}"),
    ))
}

/// `app`'s manifest with a normal path dependency on `cargo-gust`, after its dependency on `kernels`. `root` is the
/// repository, with forward slashes.
pub(crate) fn with_tool_dependency(app: &str, root: &str) -> Result<String, String> {
    const KERNELS: &str = "kernels = { path = \"../kernels\" }";
    if !app.contains(KERNELS) {
        return Err(format!(
            "{PRECOMPILED_DIR}/app/Cargo.toml no longer has the line `{KERNELS}`; E6j cannot add its dependency"
        ));
    }
    Ok(app.replace(
        KERNELS,
        &format!("{KERNELS}\ncargo-gust = {{ path = \"{root}/crates/cargo-gust\" }}"),
    ))
}

/// Makes `app` of the scratch copy `copy` of the fixture depend on `cargo-gust`.
fn add_tool_dependency(fixture: &Fixture, copy: &Path) -> Result<(), String> {
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    let app = copy.join("app").join("Cargo.toml");
    let text = fs::read_to_string(&app).map_err(|error| format!("{}: {error}", app.display()))?;
    fs::write(&app, with_tool_dependency(&text, &root)?)
        .map_err(|error| format!("{}: {error}", app.display()))
}

/// Adds the crate `bridge-user` to the scratch copy `copy` of the fixture and makes `app` depend on it.
fn add_bridge_user(fixture: &Fixture, copy: &Path) -> Result<(), String> {
    let root = fixture.root.to_string_lossy().replace('\\', "/");
    let write = |path: PathBuf, text: String| {
        fs::write(&path, text).map_err(|error| format!("{}: {error}", path.display()))
    };
    let read = |path: &Path| {
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
    };
    let directory = copy.join("bridge-user");
    fs::create_dir_all(directory.join("src"))
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    write(directory.join("Cargo.toml"), bridge_user_manifest(&root))?;
    write(
        directory.join("src").join("lib.rs"),
        "// Enables `gust-wgpu/slang-runtime` for E6c of `cargo xtask check-precompiled`; it has no code.\n".to_owned(),
    )?;
    let workspace = copy.join("Cargo.toml");
    write(
        workspace.clone(),
        with_bridge_user_member(&read(&workspace)?)?,
    )?;
    let app = copy.join("app").join("Cargo.toml");
    write(app.clone(), with_bridge_user_dependency(&read(&app)?)?)?;
    Ok(())
}

/// The target directory of `case`: one per (profile, release) pair, and one for the repository workspace, so that a
/// case finds the objects of the same case in the previous run instead of the build script re-running after every flip
/// of `GUST_PROFILE` (E3 and E4 stay in `scratch_target_dir`).
fn e6_target_dir(fixture: &Fixture, case: &E6Case) -> PathBuf {
    let profile = format!("{:?}", case.profile).to_lowercase();
    let mode = if case.release { "release" } else { "debug" };
    let scope = if case.workspace { "-workspace" } else { "" };
    fixture
        .scratch
        .join(format!("e6-target-{profile}-{mode}{scope}"))
}

/// E6a to E6j: the refusals of `gust`'s build script, their positive controls, and the three ways in which the internal
/// `cargo-gust-bridge` feature (D108) makes a release build without a variable `dev` (E6g, E6i, E6j; a release build
/// without `GUST_PROFILE` is otherwise strict). Returns the summary.
pub(super) fn e6_refusals(fixture: &Fixture) -> Result<String, String> {
    let fixture_manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let copy = fixture.scratch.join("e6c");
    copy_fixture(fixture, &copy)?;
    add_bridge_user(fixture, &copy)?;
    let copy_manifest = copy.join("Cargo.toml");
    let copy_manifest = copy_manifest
        .to_str()
        .ok_or("check-precompiled E6: the scratch path is not valid UTF-8")?;
    let tool_copy = fixture.scratch.join("e6j");
    copy_fixture(fixture, &tool_copy)?;
    add_tool_dependency(fixture, &tool_copy)?;
    let tool_manifest = tool_copy.join("Cargo.toml");
    let tool_manifest = tool_manifest
        .to_str()
        .ok_or("check-precompiled E6: the scratch path is not valid UTF-8")?;

    let mut problems = Vec::new();
    let mut lines = Vec::new();
    for case in e6_cases() {
        let manifest = if case.bridge_user {
            copy_manifest
        } else if case.tool_dependency {
            tool_manifest
        } else if case.workspace {
            "Cargo.toml"
        } else {
            fixture_manifest.as_str()
        };
        let case_target = e6_target_dir(fixture, &case);
        let target_dir = Fixture::target_dir_arg(&case_target)?;
        let args = e6_args(&case, manifest, target_dir);
        let arguments: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut command = match case.profile {
            E6Profile::Native => fixture.cargo(&arguments),
            E6Profile::Dev => fixture.cargo_under_profile(&arguments, CONTROL_PROFILE),
            E6Profile::Unset => fixture.cargo_without_profile(&arguments),
        };
        let output = run_capture_in(&mut command, "check-precompiled E6", "cargo")?;
        let expectation = match case.outcome {
            E6Outcome::Refused { expected, .. } | E6Outcome::CompilesAsDev { expected, .. } => {
                let path = fixture.root.join(PRECOMPILED_DIR).join(expected);
                Some(fs::read_to_string(&path).map_err(|error| {
                    format!("check-precompiled E6: {PRECOMPILED_DIR}/{expected}: {error}")
                })?)
            }
            E6Outcome::Compiles { .. } => None,
        };
        let found = e6_problems(
            &case,
            expectation.as_deref(),
            output.status.success(),
            &String::from_utf8_lossy(&output.stdout),
            &String::from_utf8_lossy(&output.stderr),
        )
        .map_err(|error| format!("check-precompiled E6: {error}"))?;
        let shown = match case.outcome {
            E6Outcome::Refused { .. } => "refused",
            E6Outcome::Compiles { .. } | E6Outcome::CompilesAsDev { .. } => "compiled",
        };
        println!(
            "check-precompiled E6 {}: {} (profile {:?}{}{}): {}",
            case.id,
            args.join(" ")
                .replace(copy_manifest, "<copy>/Cargo.toml")
                .replace(tool_manifest, "<tool-copy>/Cargo.toml")
                .replace(target_dir, "<scratch-target>"),
            case.profile,
            if case.release { ", release" } else { "" },
            if case.bridge_user {
                ", in a copy with bridge-user"
            } else if case.tool_dependency {
                ", in a copy where app depends on cargo-gust"
            } else if case.workspace {
                ", the whole workspace"
            } else {
                ""
            },
            if found.is_empty() {
                shown
            } else {
                "NOT AS EXPECTED"
            }
        );
        lines.push(format!("{} {shown}", case.id));
        problems.extend(found);
    }
    let profiles = e6_profile_values(fixture, &mut problems)?;
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E6 found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    Ok(format!(
        "passed: {}; each refusal was judged on the exact message of `gust`'s build script ({PRECOMPILED_DIR}/{E6_NATIVE_EXPECTED}, {PRECOMPILED_DIR}/{E6_INFERRED_EXPECTED}), E6d and E6e are the positive controls, E6g, E6i and E6j compiled as dev with the warning of {PRECOMPILED_DIR}/{E6_BRIDGE_EXPECTED}; the capability profiles (D138): {profiles}",
        lines.join(", ")
    ))
}

#[cfg(test)]
#[path = "../tests/precompiled_refusals.rs"]
mod tests;
