use std::{
    ffi::OsStr,
    io::{self, BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
    sync::Once,
};

use crate::test_exact::parse_result_line;

/// The variable that selects the profile `gust` is built under (D44, plan M4 R4.2).
pub(crate) const PROFILE_VARIABLE: &str = "GUST_PROFILE";

/// The profile the repository's own workspace builds under. The examples and the tests depend on the
/// `slang-runtime` feature, which a strict profile (`native`, `web`) refuses by design, and the workspace is not a
/// release artefact. So every cargo child of a workspace command gets this value, whatever the parent has.
pub(crate) const WORKSPACE_PROFILE: &str = "dev";

/// The exit code of a child for a message: its number, or `?` when it ended without one (a signal).
pub(crate) fn exit_code(status: std::process::ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "?".to_owned(), |code| code.to_string())
}

/// What is printed when the parent environment has a `GUST_PROFILE` other than [`WORKSPACE_PROFILE`]; `None` when
/// there is nothing to say. A preset strict value is overridden, not refused: unlike a preset rustflags variable it
/// cannot corrupt a measurement, so the override is only logged.
pub(crate) fn profile_override_note(preset: Option<&OsStr>) -> Option<String> {
    let preset = preset?;
    (preset != WORKSPACE_PROFILE).then(|| {
        format!(
            "xtask: {PROFILE_VARIABLE}={} is set in the environment; workspace builds run under {PROFILE_VARIABLE}={WORKSPACE_PROFILE} instead",
            preset.to_string_lossy()
        )
    })
}

/// The variable that names the native reflection helper (`gust::toolchain::HELPER_VARIABLE`, D109).
pub(crate) const HELPER_VARIABLE: &str = "GUST_SLANG_REFLECT";

/// The helper that `cargo xtask build-slang-reflect` writes by default:
/// `<root>/target/slang-reflect/gust-slang-reflect`.
pub(crate) fn repository_helper(root: &Path) -> std::path::PathBuf {
    root.join("target").join("slang-reflect").join(format!(
        "gust-slang-reflect{}",
        std::env::consts::EXE_SUFFIX
    ))
}

/// The value xtask exports as `GUST_SLANG_REFLECT` to its cargo children: the repository's own helper
/// ([`repository_helper`] of `root`) when the parent environment does not name one (`preset` is `None`) and that file
/// exists. A named helper, even a wrong one, is the parent's decision and is passed on untouched, and with no helper
/// built there is nothing to export (the children then follow `gust::toolchain`'s rules).
pub(crate) fn helper_export(preset: Option<&OsStr>, root: &Path) -> Option<std::path::PathBuf> {
    if preset.is_some() {
        return None;
    }
    let helper = repository_helper(root);
    helper.is_file().then_some(helper)
}

/// Gives `command`, a cargo child that builds workspace crates, `GUST_PROFILE=dev` and, when the repository's
/// reflection helper exists and the parent names none, `GUST_SLANG_REFLECT` ([`helper_export`]); logs once per process
/// when the profile overrides a different value of the parent.
pub(crate) fn set_workspace_profile(command: &mut Command) -> &mut Command {
    static NOTED: Once = Once::new();
    NOTED.call_once(|| {
        let preset = std::env::var_os(PROFILE_VARIABLE);
        if let Some(note) = profile_override_note(preset.as_deref()) {
            eprintln!("{note}");
        }
    });
    if let Ok(root) = crate::workspace::workspace_root()
        && let Some(helper) = helper_export(std::env::var_os(HELPER_VARIABLE).as_deref(), &root)
    {
        command.env(HELPER_VARIABLE, helper);
    }
    command.env(PROFILE_VARIABLE, WORKSPACE_PROFILE)
}

/// `Command::new(program)`, with [`set_workspace_profile`] applied when the program is cargo (by file stem, so a
/// full path from `$CARGO` counts). Any other program, such as `git` or `slangc`, is left alone.
pub(crate) fn workspace_command(program: impl AsRef<OsStr>) -> Command {
    let program = program.as_ref();
    let mut command = Command::new(program);
    if Path::new(program)
        .file_stem()
        .is_some_and(|stem| stem == "cargo")
    {
        set_workspace_profile(&mut command);
    }
    command
}

pub(crate) struct CheckRecord {
    pub(crate) program: String,
    pub(crate) arguments: Vec<String>,
    pub(crate) exit_code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

pub(crate) fn run_checked(
    root: &Path,
    checks: &mut Vec<CheckRecord>,
    program: &str,
    args: &[&str],
) -> Result<(), String> {
    let output = workspace_command(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not launch {program}: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    print!("{stdout}");
    eprint!("{stderr}");
    let exit_code = output.status.code();
    checks.push(CheckRecord {
        program: program.to_owned(),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        exit_code,
        stdout,
        stderr,
    });
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{} {} failed (exit {})",
            program,
            args.join(" "),
            exit_code.unwrap_or(-1)
        ))
    }
}

pub(crate) fn run_checked_owned(
    root: &Path,
    checks: &mut Vec<CheckRecord>,
    program: &str,
    args: &[String],
) -> Result<(), String> {
    let borrowed = args.iter().map(String::as_str).collect::<Vec<_>>();
    run_checked(root, checks, program, &borrowed)
}

pub(crate) fn run_direct(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    run_direct_args(
        root,
        program,
        &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
    )
}

pub(crate) fn run_direct_args(root: &Path, program: &str, args: &[String]) -> Result<(), String> {
    let status = workspace_command(program)
        .args(args)
        .current_dir(root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("could not launch {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{} {} failed (exit {})",
            program,
            args.join(" "),
            status.code().unwrap_or(-1)
        ))
    }
}

pub(crate) fn print_command(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not launch {program}: {error}"))?;
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{} {} failed (exit {})",
            program,
            args.join(" "),
            output.status.code().unwrap_or(-1)
        ))
    }
}

pub(crate) fn require_command(program: &str) -> Result<(), String> {
    if command_exists(program) {
        Ok(())
    } else {
        Err(format!(
            "required command `{program}` was not found on PATH"
        ))
    }
}

pub(crate) fn command_exists(program: &str) -> bool {
    Command::new(program)
        .arg(if cfg!(windows) { "/?" } else { "--version" })
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Prints the outcome of an in-process check and records it like a command run, so lint-style
/// checks appear in `.ai/VALIDATION.json` beside the cargo steps. `Ok` carries the summary,
/// `Err` one line per problem; the returned error is a one-line count.
pub(crate) fn report_check(
    checks: &mut Vec<CheckRecord>,
    subcommand: &str,
    outcome: Result<String, Vec<String>>,
) -> Result<(), String> {
    let (stdout, stderr, exit_code, result) = match outcome {
        Ok(summary) => (summary, String::new(), 0, Ok(())),
        Err(problems) => {
            let message = format!("{subcommand} found {} problem(s)", problems.len());
            (String::new(), problems.join("\n"), 1, Err(message))
        }
    };
    print!("{stdout}");
    if !stderr.is_empty() {
        eprintln!("{stderr}");
    }
    checks.push(CheckRecord {
        program: "cargo".to_owned(),
        arguments: vec!["xtask".to_owned(), subcommand.to_owned()],
        exit_code: Some(exit_code),
        stdout,
        stderr,
    });
    result
}

/// The totals of every libtest `test result:` line of one `cargo test` call: each test binary
/// (lib, bin, integration target) and the doc-tests print one line, and this sums them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TestTotals {
    pub(crate) passed: u64,
    pub(crate) failed: u64,
    pub(crate) ignored: u64,
    /// How many `test result:` lines were read.
    pub(crate) summaries: u64,
}

/// `text` without ANSI colour sequences (`ESC [ ... m`), which libtest prints around `ok` when colour is forced
/// (`CARGO_TERM_COLOR=always`, as CI sets it).
pub(crate) fn without_colors(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            for skipped in chars.by_ref() {
                if skipped.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

/// Sums the `test result: ok. N passed; M failed; K ignored; ...` lines of `output`. A
/// `test result:` line that cannot be read counts as nothing, so it can only make a gate stricter.
pub(crate) fn sum_test_results(output: &str) -> TestTotals {
    let mut totals = TestTotals::default();
    for line in without_colors(output).lines() {
        if let Some(counts) = parse_result_line(line) {
            totals.passed += u64::from(counts.passed);
            // A line that is not `ok` counts as a failure even when it names none.
            totals.failed += u64::from(counts.failed).max(u64::from(!counts.ok));
            totals.ignored += u64::from(counts.ignored);
            totals.summaries += 1;
        }
    }
    totals
}

/// Whether the cargo test arguments run the ignored tests as well (`-- --include-ignored`).
fn includes_ignored(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--include-ignored")
}

/// Fails when a `cargo test` call, which exits 0 with no test at all, ran nothing: the sum of its
/// passed tests must be above zero and nothing may have failed. A call with `--include-ignored`
/// must also report no ignored test (every ignored test ran, so an `ignored` count means the
/// harness did not run it).
pub(crate) fn require_tests_ran(args: &[String], totals: &TestTotals) -> Result<(), String> {
    let call = format!("cargo {}", args.join(" "));
    if totals.failed > 0 {
        return Err(format!("{call}: {} test(s) failed", totals.failed));
    }
    if totals.passed == 0 {
        return Err(format!(
            "{call}: ran 0 tests ({} test result line(s) read, none with a passed test); a gate that runs no test proves nothing, so a filter or an attribute that matches nothing fails",
            totals.summaries
        ));
    }
    if includes_ignored(args) && totals.ignored > 0 {
        return Err(format!(
            "{call}: {} test(s) reported ignored although --include-ignored was given",
            totals.ignored
        ));
    }
    Ok(())
}

/// Runs `cargo <args>` (a `cargo test` call) like [`run_direct_args`], streaming its output to the
/// console as it arrives, then requires that tests really ran ([`require_tests_ran`]).
pub(crate) fn run_cargo_test_counted(root: &Path, args: &[String]) -> Result<TestTotals, String> {
    run_cargo_test_captured(root, args).map(|(totals, _)| totals)
}

/// [`run_cargo_test_counted`], also returning the captured standard output, for a caller that requires named tests
/// to be reported `ok` ([`crate::checks::tests_not_ok`]).
pub(crate) fn run_cargo_test_captured(
    root: &Path,
    args: &[String],
) -> Result<(TestTotals, String), String> {
    let mut child = workspace_command("cargo")
        .args(args)
        .current_dir(root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("could not capture the cargo test output")?;
    let mut reader = BufReader::new(stdout);
    let mut captured = Vec::new();
    let mut line = Vec::new();
    let mut read_error = None;
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {
                // Echoing is best effort; the captured copy is what the guard reads.
                let _ = io::stdout().write_all(&line);
                captured.extend_from_slice(&line);
            }
            Err(error) => {
                read_error = Some(error);
                break;
            }
        }
    }
    if let Some(error) = read_error {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("could not read the cargo test output: {error}"));
    }
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for cargo: {error}"))?;
    if !status.success() {
        return Err(format!(
            "cargo {} failed (exit {})",
            args.join(" "),
            status.code().unwrap_or(-1)
        ));
    }
    let stdout = String::from_utf8_lossy(&captured).into_owned();
    let totals = sum_test_results(&stdout);
    require_tests_ran(args, &totals)?;
    Ok((totals, stdout))
}

/// [`run_cargo_test_counted`] for a borrowed argument list.
pub(crate) fn run_counted(root: &Path, args: &[&str]) -> Result<TestTotals, String> {
    run_cargo_test_counted(
        root,
        &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
    )
}

/// [`run_cargo_test_captured`] for a borrowed argument list.
pub(crate) fn run_counted_captured(
    root: &Path,
    args: &[&str],
) -> Result<(TestTotals, String), String> {
    run_cargo_test_captured(
        root,
        &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
    )
}

#[cfg(test)]
#[path = "tests/test_counts.rs"]
mod test_counts_tests;

#[cfg(test)]
#[path = "tests/workspace_profile.rs"]
mod workspace_profile_tests;

#[cfg(test)]
mod helper_export_tests {
    use std::{ffi::OsStr, fs};

    use super::{helper_export, repository_helper};

    fn scratch(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("gust-xtask-helper-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn the_repository_helper_is_exported_only_when_it_exists_and_the_parent_names_none() {
        let root = scratch("export");
        // Nothing built: nothing to export.
        assert_eq!(helper_export(None, &root), None);
        let helper = repository_helper(&root);
        fs::create_dir_all(helper.parent().unwrap()).unwrap();
        fs::write(&helper, "x").unwrap();
        assert_eq!(helper_export(None, &root), Some(helper.clone()));
        // The path is `<root>/target/slang-reflect/gust-slang-reflect` with the platform's suffix.
        assert_eq!(
            helper,
            root.join("target").join("slang-reflect").join(format!(
                "gust-slang-reflect{}",
                std::env::consts::EXE_SUFFIX
            ))
        );
        // A helper the parent names, even an empty or missing one, is the parent's decision.
        for preset in ["", "elsewhere"] {
            assert_eq!(
                helper_export(Some(OsStr::new(preset)), &root),
                None,
                "{preset:?}"
            );
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_directory_with_the_helper_name_is_not_a_helper() {
        let root = scratch("directory");
        fs::create_dir_all(repository_helper(&root)).unwrap();
        assert_eq!(helper_export(None, &root), None);
        let _ = fs::remove_dir_all(&root);
    }
}
