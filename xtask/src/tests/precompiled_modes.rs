//! The modes of `check-precompiled`: `--prepare` and `--adapter-runs` split the full check, the stamp refuses stale
//! binaries, and the default is unchanged.

use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use super::{
    Mode, Options, STAMP_FILE, adapter_runs, environment::StripReport, fixture::Fixture,
    parse_args, source_state, stamp_text, stamped_binaries,
};

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|argument| (*argument).to_owned()).collect()
}

#[test]
fn the_default_is_the_full_check() {
    assert_eq!(parse_args(&[]), Ok(Options::default()));
    assert_eq!(Options::default().mode, Mode::Full);
    let with_directory = parse_args(&args(&["--work-dir", "x"])).unwrap();
    assert_eq!(with_directory.mode, Mode::Full);
    assert_eq!(with_directory.work_dir, Some(PathBuf::from("x")));
    for flag in ["--check-only", "--prepare", "--adapter-runs"] {
        assert_eq!(
            parse_args(&args(&[&format!("{flag}=1")])),
            Err(format!("`{flag}` takes no value"))
        );
    }
}

#[test]
fn each_flag_selects_one_mode_and_they_exclude_each_other() {
    for (flag, mode) in [
        ("--check-only", Mode::CheckOnly),
        ("--prepare", Mode::Prepare),
        ("--adapter-runs", Mode::AdapterRuns),
    ] {
        assert_eq!(parse_args(&args(&[flag])).unwrap().mode, mode);
        assert!(parse_args(&args(&[flag, flag])).is_err(), "{flag} twice");
    }
    assert!(parse_args(&args(&["--prepare", "--adapter-runs"])).is_err());
    assert!(parse_args(&args(&["--check-only", "--prepare"])).is_err());
    assert!(parse_args(&args(&["--adapter-runs", "--work-dir", "x"])).is_err());
    assert!(parse_args(&args(&["--prepare", "--work-dir", "x"])).is_ok());
}

#[test]
fn the_runs_and_e7b_are_each_in_exactly_one_of_prepare_and_adapter_runs_and_the_full_check() {
    type Part = fn(Mode) -> bool;
    let parts: [(&str, Part); 2] = [("runs", Mode::runs), ("slang", Mode::slang)];
    for (name, part) in parts {
        let halves = [Mode::Prepare, Mode::AdapterRuns].map(part);
        assert_eq!(
            halves.iter().filter(|runs| **runs).count(),
            1,
            "{name} must be in exactly one of --prepare and --adapter-runs"
        );
        assert!(part(Mode::Full), "{name} is part of the full check");
    }
    assert!(Mode::Prepare.slang() && !Mode::Prepare.runs());
    assert!(!Mode::AdapterRuns.slang() && Mode::AdapterRuns.runs());
    // `--check-only` is the build half without Slang: what CI runs.
    assert!(!Mode::CheckOnly.runs() && !Mode::CheckOnly.slang());
}

fn git(root: &Path, arguments: &[&str]) {
    let status = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(arguments)
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success(), "git {arguments:?}");
}

/// A scratch repository with one commit, in a directory of its own that the test deletes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("gust-precompiled-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), "one\n").unwrap();
        fs::write(root.join("blob.dat"), [0u8, 1, 2, 255]).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        Self(root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture_in(root: &Path) -> Fixture {
    // Inside `.git`, which the tree hash does not see: the stamp must not change the state it records.
    let target = root.join(".git").join("t");
    fs::create_dir_all(&target).unwrap();
    Fixture {
        root: root.to_path_buf(),
        target_dir: target.join("precompiled"),
        scratch: target.join("precompiled-scratch"),
        target,
        cargo: OsString::from("cargo"),
        environment: Vec::new(),
        report: StripReport::default(),
    }
}

/// What `--prepare` leaves: three stand-in binaries and the stamp of the current state.
fn prepared(scratch: &Scratch) -> Fixture {
    let fixture = fixture_in(&scratch.0);
    for (key, path) in stamped_binaries(&fixture) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, key).unwrap();
    }
    let (head, tree) = source_state(&scratch.0).unwrap();
    let stamp = stamp_text(&fixture, &head, &tree).unwrap();
    fs::write(fixture.target.join(STAMP_FILE), stamp).unwrap();
    fixture
}

#[test]
fn the_tree_hash_follows_tracked_untracked_and_binary_changes() {
    let scratch = Scratch::new("state");
    let root = &scratch.0;
    let (head, clean) = source_state(root).unwrap();
    assert_eq!(head.len(), 40);
    assert_eq!(
        source_state(root).unwrap().1,
        clean,
        "stable while nothing changes"
    );
    fs::write(root.join("new.txt"), "a\n").unwrap();
    let (_, with_new) = source_state(root).unwrap();
    assert_ne!(with_new, clean, "a new untracked file");
    fs::write(root.join("new.txt"), "b\n").unwrap();
    let (_, edited_new) = source_state(root).unwrap();
    assert_ne!(edited_new, with_new, "an edit of an untracked file");
    fs::remove_file(root.join("new.txt")).unwrap();
    fs::write(root.join("file.txt"), "two\n").unwrap();
    let (_, edited) = source_state(root).unwrap();
    assert_ne!(edited, clean, "a tracked edit");
    fs::write(root.join("blob.dat"), [0u8, 1, 3, 255]).unwrap();
    let (_, blob_a) = source_state(root).unwrap();
    fs::write(root.join("blob.dat"), [0u8, 1, 4, 255]).unwrap();
    assert_ne!(
        source_state(root).unwrap().1,
        blob_a,
        "two edits of a tracked binary file"
    );
}

#[test]
fn adapter_runs_refuses_without_a_stamp() {
    let scratch = Scratch::new("nostamp");
    let error = adapter_runs(&fixture_in(&scratch.0)).unwrap_err();
    assert!(
        error.contains("no stamp") && error.contains("--prepare"),
        "{error}"
    );
}

#[test]
fn adapter_runs_refuses_another_head_a_changed_tree_and_a_new_untracked_file() {
    let scratch = Scratch::new("stale");
    let fixture = prepared(&scratch);
    let (head, _) = source_state(&scratch.0).unwrap();

    fs::write(scratch.0.join("new.txt"), "x\n").unwrap();
    let error = adapter_runs(&fixture).unwrap_err();
    assert!(
        error.contains("tree was") && error.contains("stale") && error.contains("--prepare"),
        "a new untracked file: {error}"
    );
    fs::remove_file(scratch.0.join("new.txt")).unwrap();

    fs::write(scratch.0.join("file.txt"), "two\n").unwrap();
    let error = adapter_runs(&fixture).unwrap_err();
    assert!(error.contains("tree was"), "a tracked edit: {error}");

    git(&scratch.0, &["commit", "-q", "-a", "-m", "second"]);
    let (new_head, _) = source_state(&scratch.0).unwrap();
    let error = adapter_runs(&fixture).unwrap_err();
    assert!(
        error.contains(&format!("head was {head}")) && error.contains(&format!("is {new_head}")),
        "another HEAD: {error}"
    );
}

#[test]
fn adapter_runs_refuses_a_rebuilt_or_missing_binary_naming_it() {
    let scratch = Scratch::new("binaries");
    let fixture = prepared(&scratch);
    let [e1, e3, _] = stamped_binaries(&fixture);

    fs::write(&e3.1, "relinked").unwrap();
    let error = adapter_runs(&fixture).unwrap_err();
    assert!(
        error.contains(e3.0) && error.contains("stale") && !error.contains(e1.0),
        "{error}"
    );

    fs::remove_file(&e1.1).unwrap();
    let error = adapter_runs(&fixture).unwrap_err();
    assert!(
        error.contains("cannot read the binary") && error.contains("--prepare"),
        "{error}"
    );
}
