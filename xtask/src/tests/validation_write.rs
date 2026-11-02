use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    AdapterRecord, ArtifactRecord, GitStamp, RunInfo, compact_validation_summary, json_string,
    validation_redaction_tests::HOME, write_validation_with_home,
};
use crate::{process::CheckRecord, verify::VerifyMode};

fn run(mode: VerifyMode, git: &GitStamp) -> RunInfo<'_> {
    RunInfo {
        mode,
        started: "2026-10-23T00:00:00Z",
        git,
        adapter: None,
        skipped: &[],
    }
}

/// A fresh root under the cargo target directory (`<target>/<profile>/deps/<test binary>`).
/// The counter keeps parallel tests apart when the clock gives them the same stamp.
fn temp_root() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(Path::to_path_buf))
        .unwrap_or_else(std::env::temp_dir);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let root = base.join("xtask-tests").join(format!(
        "validation-{}-{stamp}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(root.join(".ai")).unwrap();
    root
}

/// The spellings command output uses for a path: as given, backslashes, doubled
/// backslashes, the verbatim prefix, and forward slashes.
fn spellings(path: &str) -> Vec<String> {
    let backslash = path.replace('/', r"\");
    vec![
        path.to_owned(),
        backslash.replace('\\', r"\\"),
        format!(r"\\?\{backslash}"),
        backslash.replace('\\', "/"),
        backslash,
    ]
}

/// The text as the JSON file holds it: raw, and with JSON string escaping.
fn json_forms(text: &str) -> [String; 2] {
    let escaped = json_string(text);
    [
        text.to_ascii_lowercase(),
        escaped[1..escaped.len() - 1].to_ascii_lowercase(),
    ]
}

#[test]
fn written_record_contains_neither_the_root_nor_the_home() {
    let root = temp_root();
    let root_text = root.display().to_string();
    let leaky = |tail: &str| {
        let mut parts = spellings(&root_text);
        parts.extend(spellings(HOME));
        format!("{} {tail}", parts.join(" | "))
    };
    let checks = [CheckRecord {
        program: format!(r"{root_text}\tool.exe"),
        arguments: vec![leaky("arg one"), leaky("arg two")],
        exit_code: Some(1),
        stdout: leaky("stdout"),
        stderr: leaky("stderr"),
    }];
    let artifacts = [ArtifactRecord {
        file: leaky("artifact.spv"),
        sha256: "0".repeat(64),
    }];
    let failure = leaky("failed");

    write_validation_with_home(
        &root,
        HOME,
        &run(VerifyMode::Fast, &GitStamp::UNKNOWN),
        Some(&failure),
        &checks,
        &artifacts,
    )
    .unwrap();
    let json = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    let _ = fs::remove_dir_all(&root);

    let lowered = json.to_ascii_lowercase();
    for secret in spellings(&root_text).iter().chain(&spellings(HOME)) {
        for form in json_forms(secret) {
            assert!(
                !lowered.contains(&form),
                "VALIDATION.json leaks `{secret}`:\n{json}"
            );
        }
    }
    // The records survive, with placeholders where the paths were.
    assert!(json.contains(r#""program": "<workspace>"#), "{json}");
    assert!(json.contains(r#""passed": false"#), "{json}");
    assert!(json.contains(r#""exit_code": 1"#), "{json}");
    assert!(json.contains("<workspace>"), "{json}");
    assert!(json.contains("~"), "{json}");
    assert!(json.contains("artifact.spv"), "{json}");
    assert!(json.contains("failed"), "{json}");
}

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

fn write_record(git: &GitStamp) -> String {
    let root = temp_root();
    write_validation_with_home(&root, HOME, &run(VerifyMode::Fast, git), None, &[], &[]).unwrap();
    let json = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    let _ = fs::remove_dir_all(&root);
    json
}

#[test]
fn the_record_names_the_commit_and_whether_the_tree_was_dirty() {
    for dirty in [true, false] {
        let json = write_record(&GitStamp {
            commit: Some(COMMIT.to_owned()),
            dirty: Some(dirty),
        });
        assert!(
            json.contains(&format!("\"git_commit\": \"{COMMIT}\",\n")),
            "{json}"
        );
        assert!(
            json.contains(&format!("\"git_dirty\": {dirty},\n")),
            "{json}"
        );
        let summary = compact_validation_summary(&json);
        assert!(summary.starts_with("finished=20"), "{summary}");
        assert!(
            summary.ends_with(&format!(
                " mode=fast passed=true commit={COMMIT} dirty={dirty}"
            )),
            "{summary}"
        );
    }
}

#[test]
fn the_record_says_null_when_git_is_unavailable() {
    let json = write_record(&GitStamp::UNKNOWN);
    assert!(json.contains("\"git_commit\": null,\n"), "{json}");
    assert!(json.contains("\"git_dirty\": null,\n"), "{json}");
    assert!(
        compact_validation_summary(&json).ends_with("commit=null dirty=null"),
        "{json}"
    );
    // A record written before the stamp existed reads as unknown, not as clean.
    let old = "{\n  \"mode\": \"full\",\n  \"passed\": true\n}\n";
    assert!(compact_validation_summary(old).ends_with("commit=unknown dirty=unknown"));
}

#[test]
fn a_missing_git_program_gives_an_unknown_stamp() {
    let root = temp_root();
    let stamp = GitStamp::read_with(&root, "gust-test-no-such-git-program");
    let _ = fs::remove_dir_all(&root);
    assert_eq!(stamp, GitStamp::UNKNOWN);
}

fn git(root: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .args([
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
        .args(args)
        .current_dir(root)
        .output()
        .expect("git must be on PATH (lint needs it too)");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[test]
fn the_stamp_of_a_real_repository_follows_head_and_the_working_tree() {
    // A repository of its own, nested under the target directory.
    let root = temp_root();
    git(&root, &["init", "--quiet"]);
    // Before the first commit there is no HEAD to name.
    assert_eq!(GitStamp::read(&root).commit, None);
    fs::write(root.join("tracked.txt"), "one\n").unwrap();
    git(&root, &["add", "tracked.txt"]);
    git(&root, &["commit", "--quiet", "-m", "first"]);
    let head = git(&root, &["rev-parse", "HEAD"]);

    let clean = GitStamp::read(&root);
    assert_eq!(clean.commit.as_deref(), Some(head.as_str()));
    assert_eq!(clean.dirty, Some(false));

    // An untracked file, a modified tracked file and a staged change each make it dirty.
    fs::write(root.join("new.txt"), "x\n").unwrap();
    assert_eq!(GitStamp::read(&root).dirty, Some(true));
    fs::remove_file(root.join("new.txt")).unwrap();
    assert_eq!(GitStamp::read(&root).dirty, Some(false));
    fs::write(root.join("tracked.txt"), "two\n").unwrap();
    assert_eq!(GitStamp::read(&root).dirty, Some(true));
    git(&root, &["add", "tracked.txt"]);
    assert_eq!(GitStamp::read(&root).dirty, Some(true));
    git(&root, &["commit", "--quiet", "-m", "second"]);
    let second = GitStamp::read(&root);
    let _ = fs::remove_dir_all(&root);
    assert_ne!(second.commit.as_deref(), Some(head.as_str()));
    assert_eq!(second.dirty, Some(false));
}

#[test]
fn an_untracked_file_makes_the_tree_dirty_even_when_git_is_told_to_hide_untracked_files() {
    // `status.showUntrackedFiles=no` makes a plain `git status --porcelain` blind to new
    // files; the stamp must still say dirty.
    let root = temp_root();
    git(&root, &["init", "--quiet"]);
    fs::write(root.join("tracked.txt"), "one\n").unwrap();
    git(&root, &["add", "tracked.txt"]);
    git(&root, &["commit", "--quiet", "-m", "first"]);
    git(&root, &["config", "status.showUntrackedFiles", "no"]);
    assert_eq!(GitStamp::read(&root).dirty, Some(false));

    fs::write(root.join("untracked.txt"), "x\n").unwrap();
    // The premise of the test: with the setting in force, plain status shows nothing.
    assert_eq!(git(&root, &["status", "--porcelain"]), "");
    // An untracked file in a new directory is listed per file, not as a directory entry.
    fs::create_dir(root.join("nested")).unwrap();
    fs::write(root.join("nested/inner.txt"), "y\n").unwrap();
    let dirty = GitStamp::read(&root).dirty;
    let _ = fs::remove_dir_all(&root);
    assert_eq!(dirty, Some(true));
}

fn adapter_line() -> &'static str {
    r#"{"name":"NVIDIA GeForce RTX 5090","vendor":4318,"device":11011,"device_type":"DiscreteGpu","backend":"Vulkan","driver":"NVIDIA","driver_info":"581.80"}"#
}

fn adapter() -> AdapterRecord {
    AdapterRecord::parse(adapter_line()).unwrap()
}

#[test]
fn the_adapter_line_of_the_example_is_read_field_by_field() {
    let parsed = adapter();
    assert_eq!(
        parsed,
        AdapterRecord {
            name: "NVIDIA GeForce RTX 5090".to_owned(),
            vendor: "4318".to_owned(),
            device: "11011".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
            backend: "Vulkan".to_owned(),
            driver: "NVIDIA".to_owned(),
            driver_info: "581.80".to_owned(),
        }
    );
    // Blank lines around the one line are fine; JSON escapes are decoded.
    let escaped = adapter_line().replace("RTX 5090", r#"RTX \"5090\" \\ x"#);
    let with_blanks = format!("\r\n{escaped}\r\n\r\n");
    assert_eq!(
        AdapterRecord::parse(&with_blanks).unwrap().name,
        r#"NVIDIA GeForce RTX "5090" \ x"#
    );
    // Driver text may be empty (some drivers report none); the name may not.
    let no_driver = adapter_line().replace(r#""driver_info":"581.80""#, r#""driver_info":"""#);
    assert_eq!(AdapterRecord::parse(&no_driver).unwrap().driver_info, "");
}

#[test]
fn an_adapter_that_cannot_be_named_is_an_error_never_an_empty_record() {
    let rejected = [
        String::new(),
        "\n  \r\n".to_owned(),
        "no adapter".to_owned(),
        "[]".to_owned(),
        "{}".to_owned(),
        format!("{}\n{}", adapter_line(), adapter_line()),
        adapter_line().replace("NVIDIA GeForce RTX 5090", ""),
        adapter_line().replace("NVIDIA GeForce RTX 5090", "   "),
        adapter_line().replace("DiscreteGpu", ""),
        adapter_line().replace("Vulkan", ""),
        adapter_line().replace(r#""backend":"Vulkan","#, ""),
        adapter_line().replace(r#""driver":"NVIDIA","#, ""),
        adapter_line().replace("4318", r#""4318""#),
        adapter_line().replace("4318", "-1"),
        adapter_line().replace("4318", "4.5"),
        adapter_line().replace("11011", "null"),
        format!("{}x", adapter_line()),
    ];
    for text in rejected {
        let outcome = AdapterRecord::parse(&text);
        assert!(outcome.is_err(), "accepted {text:?}: {outcome:?}");
    }
    assert_eq!(
        AdapterRecord::parse("").unwrap_err(),
        "adapter_info must print exactly one JSON line on stdout (no adapter?)"
    );
    assert_eq!(
        AdapterRecord::parse(&adapter_line().replace("NVIDIA GeForce RTX 5090", "")).unwrap_err(),
        "adapter_info output has an empty `name`"
    );
}

#[test]
fn the_record_carries_the_adapter_or_says_null() {
    let root = temp_root();
    let git = GitStamp::UNKNOWN;
    let adapter = adapter();
    let named = RunInfo {
        adapter: Some(&adapter),
        ..run(VerifyMode::Full, &git)
    };
    write_validation_with_home(&root, HOME, &named, None, &[], &[]).unwrap();
    let with = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    write_validation_with_home(&root, HOME, &run(VerifyMode::Fast, &git), None, &[], &[]).unwrap();
    let without = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    let _ = fs::remove_dir_all(&root);

    assert!(
        with.contains(concat!(
            "  \"adapter\": {\n",
            "    \"name\": \"NVIDIA GeForce RTX 5090\",\n",
            "    \"vendor\": 4318,\n",
            "    \"device\": 11011,\n",
            "    \"device_type\": \"DiscreteGpu\",\n",
            "    \"backend\": \"Vulkan\",\n",
            "    \"driver\": \"NVIDIA\",\n",
            "    \"driver_info\": \"581.80\"\n",
            "  },\n  \"passed\": true,\n"
        )),
        "{with}"
    );
    assert!(
        without.contains("  \"adapter\": null,\n  \"passed\": true,\n"),
        "{without}"
    );
    // The summary line of `cargo xtask status` is unchanged by the new member.
    assert!(compact_validation_summary(&with).ends_with("commit=null dirty=null"));
}

/// What `adapter_info` printed on lavapipe in the M5 spike (WSL, Mesa 25.2.8): the software adapter of D52.
fn software_adapter_line() -> &'static str {
    r#"{"name":"llvmpipe (LLVM 20.1.2, 256 bits)","vendor":65541,"device":0,"device_type":"Cpu","backend":"Vulkan","driver":"llvmpipe","driver_info":"Mesa 25.2.8-0ubuntu0.24.04.4 (LLVM 20.1.2)"}"#
}

fn record_with_adapter(line: &str) -> String {
    let root = temp_root();
    let git = GitStamp::UNKNOWN;
    let adapter = AdapterRecord::parse(line).unwrap();
    let named = RunInfo {
        adapter: Some(&adapter),
        ..run(VerifyMode::Gpu, &git)
    };
    write_validation_with_home(&root, HOME, &named, None, &[], &[]).unwrap();
    let json = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    let _ = fs::remove_dir_all(&root);
    json
}

#[test]
fn a_software_adapter_reads_as_gpu_software_in_the_status_summary() {
    let json = record_with_adapter(software_adapter_line());
    // The lines the `gpu-software` workflow greps for (R1-6) are what the record writer produces.
    for line in [
        "    \"device_type\": \"Cpu\",\n",
        "    \"name\": \"llvmpipe (LLVM 20.1.2, 256 bits)\",\n",
        "    \"driver_info\": \"Mesa 25.2.8-0ubuntu0.24.04.4 (LLVM 20.1.2)\"\n",
    ] {
        assert!(json.contains(line), "missing {line:?} in\n{json}");
    }
    let summary = compact_validation_summary(&json);
    assert!(summary.starts_with("finished=20"), "{summary}");
    assert!(
        summary.ends_with(" mode=gpu passed=true commit=null dirty=null tier=Gpu (software)"),
        "{summary}"
    );
}

#[test]
fn hardware_missing_and_old_records_get_no_tier_word() {
    let hardware = record_with_adapter(adapter_line());
    assert!(
        hardware.contains("\"device_type\": \"DiscreteGpu\""),
        "{hardware}"
    );
    // Only the exact `Cpu` counts: a name that merely mentions it, or another device type, does not.
    let lookalikes = [
        adapter_line().replace("NVIDIA GeForce RTX 5090", "Cpu"),
        adapter_line().replace("DiscreteGpu", "IntegratedGpu"),
        adapter_line().replace("DiscreteGpu", "VirtualGpu"),
        adapter_line().replace("DiscreteGpu", "Other"),
        adapter_line().replace("DiscreteGpu", "cpu"),
    ];
    let mut summaries = vec![hardware];
    summaries.extend(lookalikes.iter().map(|line| record_with_adapter(line)));
    summaries.push("{\n  \"adapter\": null,\n  \"passed\": true\n}\n".to_owned());
    summaries.push("{\n  \"mode\": \"full\",\n  \"passed\": true\n}\n".to_owned());
    for json in summaries {
        let summary = compact_validation_summary(&json);
        assert!(!summary.contains("tier="), "{summary}\nfor\n{json}");
        assert!(summary.ends_with("dirty=null") || summary.ends_with("dirty=unknown"));
    }
}

#[test]
fn the_adapter_text_is_redacted_like_the_rest_of_the_record() {
    let root = temp_root();
    let root_text = root.display().to_string();
    let git = GitStamp::UNKNOWN;
    let mut leaky = adapter();
    leaky.driver_info = format!("{root_text}\\icd.json and {HOME}\\driver");
    let named = RunInfo {
        adapter: Some(&leaky),
        ..run(VerifyMode::Full, &git)
    };
    write_validation_with_home(&root, HOME, &named, None, &[], &[]).unwrap();
    let json = fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap();
    let _ = fs::remove_dir_all(&root);

    let lowered = json.to_ascii_lowercase();
    for secret in spellings(&root_text).iter().chain(&spellings(HOME)) {
        for form in json_forms(secret) {
            assert!(!lowered.contains(&form), "leaks `{secret}`:\n{json}");
        }
    }
    assert!(
        json.contains(r#""driver_info": "<workspace>\\icd.json and ~\\driver""#),
        "{json}"
    );
}

fn xtask_check(name: &str, exit_code: i32) -> CheckRecord {
    CheckRecord {
        program: "cargo".to_owned(),
        arguments: vec!["xtask".to_owned(), name.to_owned()],
        exit_code: Some(exit_code),
        stdout: String::new(),
        stderr: String::new(),
    }
}

/// The record of `check-full --skip-web` names the skipped check and keeps the browser in `untested`; a passed
/// `check-web` removes it from `untested`, and a failed one does not.
#[test]
fn the_record_names_skipped_checks_and_lists_the_browser_as_untested_unless_it_ran() {
    let root = temp_root();
    let git = GitStamp::UNKNOWN;
    let skipped = ["check-web".to_owned()];
    let write = |run: &RunInfo, checks: &[CheckRecord]| {
        write_validation_with_home(&root, HOME, run, None, checks, &[]).unwrap();
        fs::read_to_string(root.join(".ai/VALIDATION.json")).unwrap()
    };
    let precompiled = xtask_check("check-precompiled", 0);

    let skipping = RunInfo {
        skipped: &skipped,
        ..run(VerifyMode::Full, &git)
    };
    let skipped_record = write(&skipping, std::slice::from_ref(&precompiled));
    let complete = write(
        &run(VerifyMode::Full, &git),
        &[
            xtask_check("check-precompiled", 0),
            xtask_check("check-web", 0),
        ],
    );
    let failed_web = write(
        &run(VerifyMode::Full, &git),
        &[precompiled, xtask_check("check-web", 1)],
    );
    let _ = fs::remove_dir_all(&root);

    assert!(
        skipped_record.contains("  \"skipped_checks\": [\"check-web\"],\n"),
        "{skipped_record}"
    );
    assert!(skipped_record.contains("\"untested\": [\"browser WebGPU\", \"Metal\""));
    assert!(
        complete.contains("  \"skipped_checks\": [],\n"),
        "{complete}"
    );
    assert!(
        complete.contains("\"untested\": [\"Metal\", \"DXIL\""),
        "{complete}"
    );
    assert!(failed_web.contains("\"untested\": [\"browser WebGPU\", \"Metal\""));

    // `cargo xtask status` must not show a record that left a check out as a plain full one.
    let line = compact_validation_summary(&skipped_record);
    assert!(
        line.contains(" dirty=") && line.ends_with(" skipped=check-web"),
        "{line}"
    );
    assert!(line.contains(" mode=full passed="), "{line}");
    let line = compact_validation_summary(&complete);
    assert!(!line.contains("skipped"), "{line}");
    // A record from before the member existed keeps its line.
    let old = complete.replace("  \"skipped_checks\": [],\n", "");
    assert_eq!(compact_validation_summary(&old), line);
}
