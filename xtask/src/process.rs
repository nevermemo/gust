use std::{
    io::{self, BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

use crate::test_exact::parse_result_line;

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
    let output = Command::new(program)
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
    let status = Command::new(program)
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
    let mut child = Command::new("cargo")
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
