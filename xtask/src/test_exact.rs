use crate::{process::workspace_command, workspace::workspace_root};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TestExactOptions {
    package: String,
    target: String,
    name: String,
    features: Option<String>,
}

impl TestExactOptions {
    /// `-p <package> --test <target> [--features <list>] <name>`
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut package = None;
        let mut target = None;
        let mut features = None;
        let mut name = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                flag @ ("-p" | "--package" | "--test" | "--features") => {
                    index += 1;
                    let value = args
                        .get(index)
                        .cloned()
                        .ok_or_else(|| format!("{flag} needs a value"))?;
                    match flag {
                        "-p" | "--package" => package = Some(value),
                        "--test" => target = Some(value),
                        _ => features = Some(value),
                    }
                }
                flag if flag.starts_with('-') => {
                    return Err(format!("unknown test-exact option `{flag}`"));
                }
                test => {
                    if name.replace(test.to_owned()).is_some() {
                        return Err("test-exact takes exactly one test name".into());
                    }
                }
            }
            index += 1;
        }
        Ok(Self {
            package: package.ok_or("test-exact needs -p <package>")?,
            target: target.ok_or("test-exact needs --test <target>")?,
            name: name.ok_or("test-exact needs a test name")?,
            features,
        })
    }

    /// `cargo test --locked -p <package> --test <target> [--features <list>]`
    fn base_args(&self) -> Vec<String> {
        let mut args = [
            "test",
            "--locked",
            "-p",
            self.package.as_str(),
            "--test",
            self.target.as_str(),
        ]
        .map(str::to_owned)
        .to_vec();
        if let Some(features) = &self.features {
            args.extend(["--features".to_owned(), features.clone()]);
        }
        args
    }
}

/// `cargo xtask test-exact`: runs one integration test by exact name, including an ignored
/// one, and fails unless libtest reports exactly one executed, passing test.
pub(crate) fn test_exact(options: &TestExactOptions) -> Result<(), String> {
    let root = workspace_root()?;
    let name = &options.name;

    let mut list = options.base_args();
    list.extend(["--".to_owned(), "--list".to_owned()]);
    let listing = workspace_command("cargo")
        .args(&list)
        .current_dir(&root)
        .output()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    if !listing.status.success() {
        eprint!("{}", String::from_utf8_lossy(&listing.stderr));
        return Err(format!(
            "could not list the tests of {} --test {}",
            options.package, options.target
        ));
    }
    let listed = count_listed(&String::from_utf8_lossy(&listing.stdout), name);
    if listed != 1 {
        return Err(format!(
            "exact test `{name}` is listed {listed} times in {} --test {}; it must be listed exactly once",
            options.package, options.target
        ));
    }

    let mut run = options.base_args();
    run.extend(
        [
            "--",
            name.as_str(),
            "--exact",
            "--include-ignored",
            "--nocapture",
        ]
        .map(str::to_owned),
    );
    let output = workspace_command("cargo")
        .args(&run)
        .current_dir(&root)
        .output()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    print!("{stdout}");
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "test `{name}` failed (exit {})",
            output.status.code().unwrap_or(-1)
        ));
    }
    check_single_pass(&stdout)?;
    println!(
        "test-exact passed: `{name}` ran once (1 passed; 0 failed; 0 ignored). This certifies execution only; inspect the test for real GPU assertions."
    );
    Ok(())
}

/// How many lines of `cargo test -- --list` output are exactly `<name>: test`.
pub(crate) fn count_listed(listing: &str, name: &str) -> usize {
    let wanted = format!("{name}: test");
    listing
        .lines()
        .filter(|line| line.trim_end() == wanted)
        .count()
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TestCounts {
    pub(crate) ok: bool,
    pub(crate) passed: u32,
    pub(crate) failed: u32,
    pub(crate) ignored: u32,
}

/// Parses `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; ...`.
pub(crate) fn parse_result_line(line: &str) -> Option<TestCounts> {
    let rest = line.trim_start().strip_prefix("test result: ")?;
    let (status, counts) = rest.split_once(". ")?;
    let (mut passed, mut failed, mut ignored) = (None, None, None);
    for part in counts.split(';') {
        let Some((number, label)) = part.trim().split_once(' ') else {
            continue;
        };
        let Ok(number) = number.parse::<u32>() else {
            continue;
        };
        match label {
            "passed" => passed = Some(number),
            "failed" => failed = Some(number),
            "ignored" => ignored = Some(number),
            _ => {}
        }
    }
    Some(TestCounts {
        ok: status == "ok",
        passed: passed?,
        failed: failed?,
        ignored: ignored?,
    })
}

/// Succeeds only when the output holds one libtest summary reporting one passed test and
/// nothing failed or ignored; zero executed tests (a filter that matched nothing) fail.
pub(crate) fn check_single_pass(output: &str) -> Result<(), String> {
    let summaries = output
        .lines()
        .filter(|line| line.trim_start().starts_with("test result: "))
        .collect::<Vec<_>>();
    let [summary] = summaries.as_slice() else {
        return Err(format!(
            "expected one `test result:` line, found {}",
            summaries.len()
        ));
    };
    let counts = parse_result_line(summary)
        .ok_or_else(|| format!("could not read the test summary `{}`", summary.trim()))?;
    if counts.ok && counts.passed == 1 && counts.failed == 0 && counts.ignored == 0 {
        Ok(())
    } else {
        Err(format!(
            "expected one executed passing test, got {} passed; {} failed; {} ignored{}",
            counts.passed,
            counts.failed,
            counts.ignored,
            if counts.ok { "" } else { " (result not ok)" }
        ))
    }
}

#[cfg(test)]
#[path = "tests/test_exact.rs"]
mod test_exact_tests;
