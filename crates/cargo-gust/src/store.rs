//! Writing `gust-artifacts/` (R4.3): staged, moved into place, manifest last, then pruned.
//!
//! [`commit`] never writes into the artifact directory while it is still deciding whether the export works:
//!
//! 1. Every file whose bytes differ from what is on disk is written into `gust-artifacts/.staging-<pid>/`. A failure
//!    here deletes the staging directory and leaves `gust-artifacts/` exactly as it was.
//! 2. The staged files are moved into place one by one, the manifest last, so the manifest never names a file that
//!    is not there. A failure here also deletes the staging directory; files that already moved stay (they are new
//!    keys or new bytes that the old manifest does not know, which `cargo gust check` reports until the next build).
//! 3. Files that are not part of the new set are pruned, and leftover staging directories (a crashed run: this process's own, and any other that is at least an hour
//!    old) removed. A younger one may belong to a concurrent run and is left alone; `check` reports it.
//!
//! Files whose bytes are already right are neither staged nor touched, so an unchanged export leaves every
//! modification time alone and cargo has nothing to rebuild.
//!
//! A test supplies a hook that fails at a chosen step; production passes a hook that never does.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use crate::manifest::MANIFEST_FILE;

/// The prefix of the staging directory names.
pub(crate) const STAGING_PREFIX: &str = ".staging-";

/// A file to put into `gust-artifacts/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutFile {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
}

/// A step of [`commit`], offered to the hook after it succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Event<'a> {
    /// A file was written into the staging directory.
    Staged(&'a str),
    /// A staged file was moved into place.
    Moved(&'a str),
    /// A file or directory that is not part of the new set was removed.
    Pruned(&'a str),
}

/// What [`commit`] did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Commit {
    /// Files that were written (new, or with different bytes), in the order they moved into place.
    pub(crate) written: Vec<String>,
    /// Files and staging directories that were removed, sorted.
    pub(crate) pruned: Vec<String>,
    /// Directories that are not staging directories; they are left alone and `check` reports them.
    pub(crate) left_alone: Vec<String>,
    /// How many files already had the right bytes.
    pub(crate) unchanged: usize,
}

/// The name of this process's staging directory.
pub(crate) fn staging_name() -> String {
    format!("{STAGING_PREFIX}{}", std::process::id())
}

/// Put `files` and then `manifest` into `directory` and prune everything else. See the module documentation.
pub(crate) fn commit(
    directory: &Path,
    files: &[OutFile],
    manifest: &OutFile,
    hook: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
) -> Result<Commit, String> {
    commit_at(directory, files, manifest, hook, SystemTime::now())
}

/// [`commit`] with the clock given, so a test can age the staging directories of other runs.
pub(crate) fn commit_at(
    directory: &Path,
    files: &[OutFile],
    manifest: &OutFile,
    hook: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    now: SystemTime,
) -> Result<Commit, String> {
    let created_directory = !directory.exists();
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create the artifact directory: {error}"))?;
    let mut ordered: Vec<&OutFile> = files.iter().collect();
    ordered.push(manifest);
    let mut result = Commit::default();
    let mut changed = Vec::new();
    for file in &ordered {
        check_name(&file.name)?;
        match fs::read(directory.join(&file.name)) {
            Ok(existing) if existing == file.bytes => result.unchanged += 1,
            _ => changed.push(*file),
        }
    }
    let staging = directory.join(staging_name());
    let staged = if changed.is_empty() {
        Ok(())
    } else {
        stage_and_move(directory, &staging, &changed, hook, &mut result)
    };
    if let Err(error) = staged {
        let _ = fs::remove_dir_all(&staging);
        if created_directory {
            // A first export that failed leaves no `gust-artifacts/` behind (this only removes an empty directory).
            let _ = fs::remove_dir(directory);
        }
        return Err(error);
    }
    let _ = fs::remove_dir_all(&staging);
    prune(directory, &ordered, hook, &mut result, now)?;
    Ok(result)
}

fn check_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') || name == ".." {
        return Err(format!("`{name}` is not a plain artifact file name"));
    }
    Ok(())
}

fn stage_and_move(
    directory: &Path,
    staging: &Path,
    changed: &[&OutFile],
    hook: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    result: &mut Commit,
) -> Result<(), String> {
    let _ = fs::remove_dir_all(staging);
    fs::create_dir(staging)
        .map_err(|error| format!("cannot create the staging directory: {error}"))?;
    for file in changed {
        fs::write(staging.join(&file.name), &file.bytes)
            .and_then(|()| hook(Event::Staged(&file.name)))
            .map_err(|error| format!("cannot stage `{}`: {error}", file.name))?;
    }
    for file in changed {
        fs::rename(staging.join(&file.name), directory.join(&file.name))
            .and_then(|()| hook(Event::Moved(&file.name)))
            .map_err(|error| format!("cannot move `{}` into place: {error}", file.name))?;
        result.written.push(file.name.clone());
    }
    Ok(())
}

fn prune(
    directory: &Path,
    keep: &[&OutFile],
    hook: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    result: &mut Commit,
    now: SystemTime,
) -> Result<(), String> {
    let mut found: Vec<(String, PathBuf, bool)> = Vec::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("cannot list the artifact directory: {error}"))?
    {
        let entry =
            entry.map_err(|error| format!("cannot list the artifact directory: {error}"))?;
        let is_dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
        found.push((
            entry.file_name().to_string_lossy().into_owned(),
            entry.path(),
            is_dir,
        ));
    }
    found.sort();
    for (name, path, is_dir) in found {
        if keep.iter().any(|file| file.name == name) {
            continue;
        }
        let removed = if is_dir && name.starts_with(STAGING_PREFIX) {
            if name != staging_name() && !is_stale(&path, now) {
                // Another run may be using it right now.
                result.left_alone.push(name);
                continue;
            }
            fs::remove_dir_all(&path)
        } else if is_dir {
            result.left_alone.push(name);
            continue;
        } else {
            fs::remove_file(&path)
        };
        removed
            .and_then(|()| hook(Event::Pruned(&name)))
            .map_err(|error| {
                format!("cannot remove `{name}` (the new set is in place): {error}")
            })?;
        result.pruned.push(name);
    }
    Ok(())
}

/// How old another run's staging directory must be before it counts as left over from a crash.
const STALE_AFTER: Duration = Duration::from_secs(60 * 60);

/// Whether the directory was last changed at least [`STALE_AFTER`] before `now`; an unreadable time counts as fresh.
fn is_stale(path: &Path, now: SystemTime) -> bool {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|age| age >= STALE_AFTER)
}

/// What is in an artifact directory, read for `check`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DirState {
    /// Regular files, sorted by name, with their bytes.
    pub(crate) files: Vec<(String, Vec<u8>)>,
    /// Sub-directories (a leftover `.staging-*` directory counts), sorted.
    pub(crate) dirs: Vec<String>,
    /// Whether the directory exists at all.
    pub(crate) exists: bool,
}

impl DirState {
    pub(crate) fn file(&self, name: &str) -> Option<&[u8]> {
        self.files
            .iter()
            .find(|(found, _)| found == name)
            .map(|(_, bytes)| bytes.as_slice())
    }
}

/// Read `directory`; a missing directory is an empty state with `exists == false`.
pub(crate) fn read_dir_state(directory: &Path) -> Result<DirState, String> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(DirState::default()),
        Err(error) => return Err(format!("cannot list the artifact directory: {error}")),
    };
    let mut state = DirState {
        exists: true,
        ..DirState::default()
    };
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("cannot list the artifact directory: {error}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            state.dirs.push(name);
        } else {
            let bytes =
                fs::read(entry.path()).map_err(|error| format!("cannot read `{name}`: {error}"))?;
            state.files.push((name, bytes));
        }
    }
    state.files.sort();
    state.dirs.sort();
    Ok(state)
}

/// Create `gust-artifacts/manifest.json` (an empty schema-1 manifest) unless it exists. Returns whether it was
/// created. The text comes from the caller so that this module does not depend on the manifest type.
pub(crate) fn create_manifest(directory: &Path, text: &str) -> Result<bool, String> {
    let path = directory.join(MANIFEST_FILE);
    if path.exists() {
        return Ok(false);
    }
    fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create the artifact directory: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("cannot write `{MANIFEST_FILE}`: {error}"))?;
    Ok(true)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scratch directory under the system temp directory, removed on drop.
    pub(crate) struct Scratch(pub(crate) PathBuf);

    impl Scratch {
        pub(crate) fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let serial = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cargo-gust test {} {serial} {name}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn file(name: &str, text: &str) -> OutFile {
        OutFile {
            name: name.to_owned(),
            bytes: text.as_bytes().to_vec(),
        }
    }

    fn no_fault(_: Event<'_>) -> io::Result<()> {
        Ok(())
    }

    /// The names and contents under `directory`, directories shown with a trailing `/`.
    fn listing(directory: &Path) -> Vec<(String, String)> {
        let mut found = Vec::new();
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().unwrap().is_dir() {
                found.push((format!("{name}/"), String::new()));
            } else {
                found.push((name, fs::read_to_string(entry.path()).unwrap()));
            }
        }
        found.sort();
        found
    }

    fn pair(name: &str, text: &str) -> (String, String) {
        (name.to_owned(), text.to_owned())
    }

    #[test]
    fn a_first_export_writes_the_files_and_the_manifest_last() {
        let scratch = Scratch::new("first");
        let dir = scratch.0.join("gust-artifacts");
        let mut moved = Vec::new();
        let result = commit(
            &dir,
            &[file("b.wgsl", "B"), file("a.wgsl", "A")],
            &file("manifest.json", "M"),
            &mut |event| {
                if let Event::Moved(name) = event {
                    moved.push(name.to_owned());
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(moved, ["b.wgsl", "a.wgsl", "manifest.json"]);
        assert_eq!(result.written, moved);
        assert!(result.pruned.is_empty() && result.left_alone.is_empty());
        assert_eq!(
            listing(&dir),
            [
                pair("a.wgsl", "A"),
                pair("b.wgsl", "B"),
                pair("manifest.json", "M")
            ]
        );
    }

    #[test]
    fn a_failure_while_staging_leaves_the_directory_exactly_as_it_was() {
        let scratch = Scratch::new("stage-fail");
        let dir = scratch.0.join("gust-artifacts");
        commit(
            &dir,
            &[file("old.wgsl", "OLD")],
            &file("manifest.json", "M1"),
            &mut no_fault,
        )
        .unwrap();
        let before = listing(&dir);
        let mut staged = 0;
        let error = commit(
            &dir,
            &[file("new1.wgsl", "N1"), file("new2.wgsl", "N2")],
            &file("manifest.json", "M2"),
            &mut |event| {
                if matches!(event, Event::Staged(_)) {
                    staged += 1;
                    if staged == 2 {
                        return Err(io::Error::other("injected"));
                    }
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            error.contains("cannot stage `new2.wgsl`") && error.contains("injected"),
            "{error}"
        );
        // Not one new file, not the new manifest, and no staging directory is left.
        assert_eq!(listing(&dir), before);
    }

    #[test]
    fn a_failure_while_moving_removes_the_staging_directory_and_keeps_the_old_manifest() {
        let scratch = Scratch::new("move-fail");
        let dir = scratch.0.join("gust-artifacts");
        commit(
            &dir,
            &[file("old.wgsl", "OLD")],
            &file("manifest.json", "M1"),
            &mut no_fault,
        )
        .unwrap();
        let error = commit(
            &dir,
            &[file("new.wgsl", "N")],
            &file("manifest.json", "M2"),
            &mut |event| {
                if event == Event::Moved("new.wgsl") {
                    Err(io::Error::other("injected"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(error.contains("cannot move `new.wgsl`"), "{error}");
        // The manifest moves last, so it never named the new file; the staging directory is gone and the stale
        // `old.wgsl` was not pruned (the failed run never reached the prune step).
        assert_eq!(
            listing(&dir),
            [
                pair("manifest.json", "M1"),
                pair("new.wgsl", "N"),
                pair("old.wgsl", "OLD")
            ]
        );
    }

    #[test]
    fn unchanged_files_are_not_staged_and_a_second_identical_commit_changes_nothing() {
        let scratch = Scratch::new("unchanged");
        let dir = scratch.0.join("gust-artifacts");
        let files = [file("a.wgsl", "A"), file("b.wgsl", "B")];
        let manifest = file("manifest.json", "M");
        commit(&dir, &files, &manifest, &mut no_fault).unwrap();
        let mut events = Vec::new();
        let again = commit(&dir, &files, &manifest, &mut |event| {
            events.push(format!("{event:?}"));
            Ok(())
        })
        .unwrap();
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(again.unchanged, 3);
        assert!(again.written.is_empty() && again.pruned.is_empty());
        // One changed file is the only thing staged and moved.
        let mut events = Vec::new();
        let third = commit(
            &dir,
            &[file("a.wgsl", "A2"), file("b.wgsl", "B")],
            &manifest,
            &mut |event| {
                events.push(format!("{event:?}"));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(events, [r#"Staged("a.wgsl")"#, r#"Moved("a.wgsl")"#]);
        assert_eq!(third.written, ["a.wgsl"]);
    }

    #[test]
    fn stale_files_strays_and_leftover_staging_directories_are_pruned() {
        let scratch = Scratch::new("prune");
        let dir = scratch.0.join("gust-artifacts");
        commit(
            &dir,
            &[file("stale.wgsl", "S"), file("keep.wgsl", "K")],
            &file("manifest.json", "M1"),
            &mut no_fault,
        )
        .unwrap();
        fs::write(dir.join("stray.txt"), "x").unwrap();
        // A temporary left by the macro's record writer is a stray file like any other.
        fs::write(dir.join(".0123.slang.1.0.tmp"), "x").unwrap();
        fs::create_dir(dir.join(".staging-99999")).unwrap();
        fs::write(dir.join(".staging-99999").join("half.wgsl"), "x").unwrap();
        fs::create_dir(dir.join("sub")).unwrap();
        let result = commit(
            &dir,
            &[file("keep.wgsl", "K")],
            &file("manifest.json", "M2"),
            &mut no_fault,
        )
        .unwrap();
        // A staging directory of another run may be in use right now: it is left alone while it is young.
        assert_eq!(
            result.pruned,
            [".0123.slang.1.0.tmp", "stale.wgsl", "stray.txt"]
        );
        assert_eq!(result.left_alone, [".staging-99999", "sub"]);
        assert_eq!(
            listing(&dir),
            [
                pair(".staging-99999/", ""),
                pair("keep.wgsl", "K"),
                pair("manifest.json", "M2"),
                pair("sub/", "")
            ]
        );
        // Two hours later it is a crash leftover and goes.
        let later = commit_at(
            &dir,
            &[file("keep.wgsl", "K")],
            &file("manifest.json", "M2"),
            &mut no_fault,
            SystemTime::now() + Duration::from_secs(2 * 60 * 60),
        )
        .unwrap();
        assert_eq!(later.pruned, [".staging-99999"]);
        assert_eq!(later.left_alone, ["sub"]);
        assert_eq!(
            listing(&dir),
            [
                pair("keep.wgsl", "K"),
                pair("manifest.json", "M2"),
                pair("sub/", "")
            ]
        );
    }

    #[test]
    fn this_runs_own_staging_directory_is_removed_whatever_its_age() {
        let scratch = Scratch::new("own-staging");
        let dir = scratch.0.join("gust-artifacts");
        fs::create_dir_all(dir.join(staging_name())).unwrap();
        let result = commit(
            &dir,
            &[file("a.wgsl", "A")],
            &file("manifest.json", "M"),
            &mut no_fault,
        )
        .unwrap();
        assert!(result.left_alone.is_empty());
        assert_eq!(
            listing(&dir),
            [pair("a.wgsl", "A"), pair("manifest.json", "M")]
        );
    }

    #[test]
    fn a_failed_first_export_does_not_leave_the_artifact_directory_behind() {
        let scratch = Scratch::new("first-fail");
        let dir = scratch.0.join("gust-artifacts");
        let error = commit(
            &dir,
            &[file("a.wgsl", "A")],
            &file("manifest.json", "M"),
            &mut |_| Err(io::Error::other("disk full")),
        )
        .unwrap_err();
        assert!(error.contains("disk full"), "{error}");
        assert!(!dir.exists());
        // A directory that existed before stays, empty.
        fs::create_dir_all(&dir).unwrap();
        commit(
            &dir,
            &[file("a.wgsl", "A")],
            &file("manifest.json", "M"),
            &mut |_| Err(io::Error::other("disk full")),
        )
        .unwrap_err();
        assert!(dir.is_dir());
    }

    #[test]
    fn names_that_could_leave_the_directory_are_refused_before_anything_is_written() {
        let scratch = Scratch::new("names");
        let dir = scratch.0.join("gust-artifacts");
        for name in ["../x", "a/b", "a\\b", "", ".hidden", ".."] {
            let error = commit(
                &dir,
                &[file(name, "x")],
                &file("manifest.json", "M"),
                &mut no_fault,
            )
            .unwrap_err();
            assert!(
                error.contains("not a plain artifact file name"),
                "{name}: {error}"
            );
        }
        assert_eq!(listing(&dir), Vec::<(String, String)>::new());
    }

    #[test]
    fn dir_state_lists_files_with_bytes_and_directories_and_tolerates_a_missing_directory() {
        let scratch = Scratch::new("state");
        assert_eq!(
            read_dir_state(&scratch.0.join("missing")).unwrap(),
            DirState::default()
        );
        let dir = scratch.0.join("d");
        fs::create_dir_all(dir.join(".staging-1")).unwrap();
        fs::write(dir.join("b"), b"2").unwrap();
        fs::write(dir.join("a"), b"1").unwrap();
        let state = read_dir_state(&dir).unwrap();
        assert!(state.exists);
        assert_eq!(state.file("a"), Some(&b"1"[..]));
        assert_eq!(state.file("c"), None);
        assert_eq!(state.dirs, [".staging-1"]);
        assert_eq!(state.files.len(), 2);
    }

    #[test]
    fn create_manifest_writes_once_and_never_overwrites() {
        let scratch = Scratch::new("init");
        let dir = scratch.0.join("gust-artifacts");
        assert!(create_manifest(&dir, "FIRST").unwrap());
        assert!(!create_manifest(&dir, "SECOND").unwrap());
        assert_eq!(
            fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap(),
            "FIRST"
        );
    }
}
