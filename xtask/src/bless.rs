use std::{fs, path::Path, process::Stdio};

use crate::{
    process::{print_command, workspace_command},
    validation::sha256_hex,
    workspace::workspace_root,
};

const FIXTURE_DIR: &str = "tests/fixtures";

/// `cargo xtask bless`: reruns the Slang golden tests with `BLESS_GOLDEN=1`, which rewrites a
/// fixture whose emitted text differs and fails that test with the line diff. Lists the changed
/// fixtures afterwards. Exits non-zero whenever a fixture was blessed, so a bless is never a
/// passing run; review the diff, then run `cargo xtask check-feature macro` without the variable.
pub(crate) fn bless() -> Result<(), String> {
    let root = workspace_root()?;
    let before = fixture_hashes(&root)?;
    println!(
        "Running the golden tests with BLESS_GOLDEN=1 (a changed golden fails its test by design)."
    );
    let status = workspace_command("cargo")
        .args(["test", "--locked", "-p", "gust-compute-macros", "golden"])
        .env("BLESS_GOLDEN", "1")
        .current_dir(&root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("could not launch cargo: {error}"))?;
    let changed = changed_fixtures(&before, &fixture_hashes(&root)?);
    // A failing `git diff --stat` must not hide the bless outcome: report both.
    let stat = if changed.is_empty() {
        Ok(())
    } else {
        println!("Changed fixtures:");
        print_command(&root, "git", &["diff", "--stat", "--", FIXTURE_DIR])
    };
    with_stat_result(outcome(status.success(), &changed), stat)
}

/// The bless outcome, with a failure of the diff listing appended instead of replacing it.
fn with_stat_result(outcome: Result<(), String>, stat: Result<(), String>) -> Result<(), String> {
    match (outcome, stat) {
        (outcome, Ok(())) => outcome,
        (Ok(()), Err(stat)) => Err(format!("could not list the changed fixtures: {stat}")),
        (Err(outcome), Err(stat)) => Err(format!(
            "{outcome}; additionally the diff listing failed: {stat}"
        )),
    }
}

/// `(file name, SHA-256)` of every `tests/fixtures/*.slang`, sorted by name.
fn fixture_hashes(root: &Path) -> Result<Vec<(String, String)>, String> {
    let directory = root.join(FIXTURE_DIR);
    let mut hashes = Vec::new();
    for entry in fs::read_dir(&directory).map_err(|error| format!("{FIXTURE_DIR}: {error}"))? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "slang")
        {
            let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            hashes.push((name, sha256_hex(&bytes)));
        }
    }
    hashes.sort();
    Ok(hashes)
}

/// Names whose hash differs between two snapshots, or that appear in only one.
fn changed_fixtures(before: &[(String, String)], after: &[(String, String)]) -> Vec<String> {
    let mut names: Vec<String> = before
        .iter()
        .chain(after)
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .filter(|name| {
            let find = |set: &[(String, String)]| {
                set.iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, hash)| hash.clone())
            };
            find(before) != find(after)
        })
        .collect()
}

/// Exit policy: success only when the tests passed and no fixture changed.
fn outcome(tests_passed: bool, changed: &[String]) -> Result<(), String> {
    match (tests_passed, changed.is_empty()) {
        (true, true) => {
            println!("No golden changed; nothing to bless.");
            Ok(())
        }
        (true, false) => Err(format!(
            "fixtures changed although the tests passed: {}; review `git diff {FIXTURE_DIR}`",
            changed.join(", ")
        )),
        (false, true) => Err(
            "golden tests failed without changing a fixture; this is a real failure, see the output above".to_owned(),
        ),
        (false, false) => Err(format!(
            "golden blessed: {} fixture(s) changed ({}); review the diff above, then run `cargo xtask check-feature macro` without BLESS_GOLDEN",
            changed.len(),
            changed.join(", ")
        )),
    }
}

#[cfg(test)]
#[path = "tests/bless.rs"]
mod bless_tests;
