use super::*;
use crate::{
    export::{
        plan,
        tests::{FakeCompiler, other, sample},
    },
    testdata::SAMPLE_KEY,
};

fn ok_verify(_: &Record, _: Format, _: &[u8], _: &str) -> Result<String, String> {
    Ok("fake-slang-1".to_owned())
}

fn planned(records: Vec<Record>, formats: &[Format]) -> Planned {
    plan(records, formats, &FakeCompiler::new()).unwrap()
}

/// The directory a successful `build` of `planned` leaves.
fn state_of(planned: &Planned) -> DirState {
    let mut files: Vec<(String, Vec<u8>)> = planned
        .files
        .iter()
        .map(|file| (file.name.clone(), file.bytes.clone()))
        .collect();
    let manifest = planned.manifest_file();
    files.push((manifest.name, manifest.bytes));
    files.sort();
    DirState {
        files,
        dirs: Vec::new(),
        exists: true,
    }
}

fn check(records: &[Record], state: &DirState) -> Report {
    run(&Inputs {
        records,
        state,
        installed_slang: Some(Ok("fake-slang-1".to_owned())),
        verify: &ok_verify,
        recompiled: None,
    })
}

fn kernels() -> Vec<Record> {
    let mut records = vec![sample(), other("b")];
    records.sort_by(|a, b| a.key.cmp(&b.key));
    records
}

fn set_file(state: &mut DirState, name: &str, bytes: &[u8]) {
    let slot = state
        .files
        .iter_mut()
        .find(|(found, _)| found == name)
        .unwrap();
    slot.1 = bytes.to_vec();
}

fn remove_file(state: &mut DirState, name: &str) {
    state.files.retain(|(found, _)| found != name);
}

#[test]
fn a_directory_that_matches_the_sources_has_no_problems() {
    let records = kernels();
    let state = state_of(&planned(records.clone(), &[Format::Wgsl, Format::Spirv]));
    assert_eq!(check(&records, &state), Report::default());
}

#[test]
fn a_missing_directory_or_manifest_is_reported_once_per_kernel_and_for_the_directory() {
    let records = kernels();
    let missing = check(&records, &DirState::default());
    assert_eq!(missing.problems.len(), 3, "{:?}", missing.problems);
    assert!(missing.problems[0].starts_with("gust-artifacts/ does not exist"));
    let mut state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    remove_file(&mut state, MANIFEST_FILE);
    let report = check(&records, &state);
    assert!(
        report.problems[0].contains("manifest.json is missing"),
        "{:?}",
        report.problems
    );
    assert!(
        report
            .problems
            .iter()
            .any(|line| line.contains("no readable manifest.json"))
    );
    // Every artifact file is a stray without a manifest.
    assert_eq!(
        report
            .problems
            .iter()
            .filter(|line| line.contains("is not part of the manifest"))
            .count(),
        4
    );
}

#[test]
fn a_collected_key_without_a_manifest_entry_is_reported() {
    let state = state_of(&planned(vec![sample()], &[Format::Wgsl]));
    let report = check(&kernels(), &state);
    let new_key = other("b").key;
    assert!(
        report
            .problems
            .iter()
            .any(|line| line == &format!("kernel `demo::b` (key {new_key}) has no entry in manifest.json; run `cargo gust build`")),
        "{:?}",
        report.problems
    );
}

#[test]
fn an_orphan_manifest_entry_and_its_files_fail_the_check() {
    let state = state_of(&planned(kernels(), &[Format::Wgsl]));
    let report = check(&[sample()], &state);
    let orphan = other("b").key;
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert_eq!(
        report.problems[0],
        format!(
            "manifest entry {orphan} (`demo::b`) is not a collected kernel; `cargo gust build` prunes it"
        )
    );
}

#[test]
fn strays_directories_and_leftover_staging_are_reported() {
    let records = kernels();
    let mut state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    state.files.push(("notes.txt".to_owned(), b"x".to_vec()));
    state
        .files
        .push((format!("{SAMPLE_KEY}.wgsl.bak"), b"x".to_vec()));
    state.files.sort();
    state.dirs.push(".staging-4242".to_owned());
    state.dirs.push("sub".to_owned());
    let report = check(&records, &state);
    assert_eq!(
        report.problems,
        [
            format!("`{SAMPLE_KEY}.wgsl.bak` is not part of the manifest (a stray file or a stale artifact); `cargo gust build` prunes it"),
            "`notes.txt` is not part of the manifest (a stray file or a stale artifact); `cargo gust build` prunes it".to_owned(),
            "`.staging-4242/` is a leftover staging directory of an interrupted build; `cargo gust build` removes it".to_owned(),
            "`sub/` is a directory; the artifact directory holds files only".to_owned(),
        ]
    );
}

#[test]
fn a_changed_artifact_or_reflection_fails_on_its_sha256() {
    let records = kernels();
    let good = state_of(&planned(records.clone(), &[Format::Wgsl]));
    let mut state = good.clone();
    let name = format!("{SAMPLE_KEY}.wgsl");
    let mut bytes = state.file(&name).unwrap().to_vec();
    bytes[0] ^= 1;
    set_file(&mut state, &name, &bytes);
    let report = check(&records, &state);
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert!(
        report.problems[0].starts_with(&format!("`{name}`: SHA-256 "))
            && report.problems[0].contains("differs from the manifest's"),
        "{}",
        report.problems[0]
    );
    let mut state = good;
    let reflection = format!("{SAMPLE_KEY}.wgsl.reflection.json");
    set_file(&mut state, &reflection, b"{}");
    let report = check(&records, &state);
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert!(report.problems[0].starts_with(&format!("`{reflection}`: SHA-256 ")));
}

#[test]
fn a_listed_file_that_is_missing_is_reported_without_a_second_complaint_from_the_verifier() {
    let records = kernels();
    let mut state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    remove_file(&mut state, &format!("{SAMPLE_KEY}.wgsl"));
    let report = check(&records, &state);
    assert_eq!(
        report.problems,
        [format!(
            "`{SAMPLE_KEY}.wgsl` is listed in the manifest but missing; run `cargo gust build`"
        )]
    );
}

#[test]
fn a_manifest_that_is_not_canonical_is_reported_even_when_it_parses() {
    let records = kernels();
    let mut state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    let text = String::from_utf8(state.file(MANIFEST_FILE).unwrap().to_vec()).unwrap();
    set_file(
        &mut state,
        MANIFEST_FILE,
        text.replace('\n', "\r\n").as_bytes(),
    );
    let report = check(&records, &state);
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert!(
        report.problems[0].contains("not in canonical form")
            && report.problems[0].contains(".gitattributes")
    );
    set_file(&mut state, MANIFEST_FILE, b"not json");
    let report = check(&records, &state);
    assert!(
        report.problems[0].starts_with("manifest.json: invalid JSON"),
        "{:?}",
        report.problems
    );
}

#[test]
fn an_entry_that_disagrees_with_the_source_is_reported_field_by_field() {
    let records = vec![sample()];
    let good = planned(records.clone(), &[Format::Wgsl]);
    let mut manifest = good.manifest.clone();
    manifest.kernels[0].clean_sha256 = "0".repeat(64);
    manifest.kernels[0].module = "elsewhere".to_owned();
    manifest.kernels[0].formats.clear();
    let mut state = state_of(&good);
    set_file(&mut state, MANIFEST_FILE, manifest.render().as_bytes());
    let report = check(&records, &state);
    assert!(report.problems.iter().any(|line| line.contains("the manifest has module `elsewhere`, the source has `demo`")), "{:?}", report.problems);
    assert!(
        report.problems.iter().any(|line| line.contains(
            "clean-text SHA-256 0000000000000000 differs from the source's 5110bc65c7d5f23f"
        )),
        "{:?}",
        report.problems
    );
    assert!(
        report
            .problems
            .iter()
            .any(|line| line.contains("has no WGSL artifact")),
        "{:?}",
        report.problems
    );
}

#[test]
fn the_slang_build_must_match_the_installed_helper_unless_the_check_is_told_to_accept_a_mismatch() {
    let records = kernels();
    let state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    let run_with = |installed: Option<Result<String, String>>| {
        run(&Inputs {
            records: &records,
            state: &state,
            installed_slang: installed,
            verify: &ok_verify,
            recompiled: None,
        })
    };
    let mismatch = run_with(Some(Ok("fake-slang-2".to_owned())));
    assert_eq!(
        mismatch.problems,
        [
            "the manifest was built with Slang `fake-slang-1` but the installed helper reports `fake-slang-2` (pass --allow-slang-mismatch to accept this)"
        ]
    );
    assert_eq!(run_with(None), Report::default());
    let unreadable = run_with(Some(Err("no helper".to_owned())));
    assert_eq!(
        unreadable.problems,
        [
            "cannot read the installed Slang version: no helper (pass --allow-slang-mismatch to skip this check)"
        ]
    );
}

#[test]
fn a_stored_artifact_that_does_not_verify_or_was_compiled_by_another_slang_is_reported() {
    let records = vec![sample()];
    let state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    let failing = |_: &Record, _: Format, _: &[u8], _: &str| -> Result<String, String> {
        Err("invalid Slang reflection: source hash".to_owned())
    };
    let report = run(&Inputs {
        records: &records,
        state: &state,
        installed_slang: None,
        verify: &failing,
        recompiled: None,
    });
    assert_eq!(
        report.problems,
        [format!(
            "`{SAMPLE_KEY}.wgsl` (wgsl) does not verify against kernel `demo::run`: invalid Slang reflection: source hash"
        )]
    );
    let other_tag = |_: &Record, _: Format, _: &[u8], _: &str| -> Result<String, String> {
        Ok("fake-slang-9".to_owned())
    };
    let report = run(&Inputs {
        records: &records,
        state: &state,
        installed_slang: None,
        verify: &other_tag,
        recompiled: None,
    });
    assert_eq!(
        report.problems,
        [format!(
            "`{SAMPLE_KEY}.wgsl` was compiled by Slang `fake-slang-9` but the manifest says `fake-slang-1`"
        )]
    );
}

#[test]
fn spirv_exported_without_spirv_val_is_a_note_and_not_a_problem() {
    let records = vec![sample()];
    let good = planned(records.clone(), &[Format::Wgsl, Format::Spirv]);
    let mut manifest = good.manifest.clone();
    manifest.kernels[0].formats[1].validated = Some(false);
    let mut state = state_of(&good);
    set_file(&mut state, MANIFEST_FILE, manifest.render().as_bytes());
    let report = check(&records, &state);
    assert!(report.problems.is_empty(), "{:?}", report.problems);
    assert_eq!(
        report.notes,
        [format!(
            "`{SAMPLE_KEY}.spv` was exported without spirv-val (validated: false)"
        )]
    );
}

#[test]
fn a_recompile_that_differs_in_any_byte_fails_and_one_that_matches_passes() {
    let records = kernels();
    let built = planned(records.clone(), &[Format::Wgsl]);
    let state = state_of(&built);
    let recheck = |again: &Planned| {
        run(&Inputs {
            records: &records,
            state: &state,
            installed_slang: Some(Ok("fake-slang-1".to_owned())),
            verify: &ok_verify,
            recompiled: Some(again),
        })
    };
    assert_eq!(recheck(&built), Report::default());
    let mut changed = built.clone();
    let file = changed
        .files
        .iter_mut()
        .find(|file| file.name == format!("{SAMPLE_KEY}.wgsl"))
        .unwrap();
    file.bytes.push(b'!');
    let report = recheck(&changed);
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert!(report.problems[0].starts_with(&format!(
        "recompiling gives different bytes for `{SAMPLE_KEY}.wgsl`"
    )));
    let mut renamed = built.clone();
    renamed.manifest.slang = Some("fake-slang-2".to_owned());
    let report = recheck(&renamed);
    assert_eq!(
        report.problems,
        ["recompiling gives a different manifest.json; run `cargo gust build`"]
    );
    let mut extra = built.clone();
    extra.files.push(crate::store::OutFile {
        name: "ffffffffffffffff.wgsl".to_owned(),
        bytes: b"x".to_vec(),
    });
    let report = recheck(&extra);
    assert_eq!(
        report.problems,
        [
            "recompiling produces `ffffffffffffffff.wgsl`, which is not stored; run `cargo gust build`"
        ]
    );
}

/// A copy of the records whose `pairs` storage parameter has `layout: null` (the macro could not describe its type).
fn with_unchecked_pairs(records: &[Record]) -> Vec<Record> {
    let mut records = records.to_vec();
    for record in &mut records {
        if let Some(pairs) = record
            .parameters
            .iter_mut()
            .find(|parameter| parameter.name == "pairs")
        {
            pairs.layout = None;
        }
    }
    records
}

#[test]
fn unchecked_parameters_give_one_note_per_kernel_and_the_same_list_as_the_build_wrote_is_not_a_problem()
 {
    let records = with_unchecked_pairs(&kernels());
    let state = state_of(&planned(records.clone(), &[Format::Wgsl]));
    let report = check(&records, &state);
    assert!(report.problems.is_empty(), "{:?}", report.problems);
    assert_eq!(report.notes.len(), 2, "{:?}", report.notes);
    assert_eq!(
        report.notes[0],
        unchecked_note(&records[0].qualified_name(), &["pairs".to_owned()])
    );
    assert!(
        report.notes[0].contains("export_unchecked [pairs]"),
        "{}",
        report.notes[0]
    );
    // Kernels without unchecked parameters add no note.
    let plain = kernels();
    assert!(
        check(&plain, &state_of(&planned(plain.clone(), &[Format::Wgsl])))
            .notes
            .is_empty()
    );
}

#[test]
fn a_manifest_whose_unchecked_list_differs_from_the_source_is_a_problem_both_ways() {
    // The source now has an unchecked parameter that the manifest does not list.
    let plain = kernels();
    let state = state_of(&planned(plain.clone(), &[Format::Wgsl]));
    let report = check(&with_unchecked_pairs(&plain), &state);
    let mismatch: Vec<&String> = report
        .problems
        .iter()
        .filter(|line| line.contains("export_unchecked is none, the source's is [pairs]"))
        .collect();
    assert_eq!(mismatch.len(), 2, "{:?}", report.problems);
    // The manifest lists one that the source no longer has.
    let unchecked = with_unchecked_pairs(&plain);
    let state = state_of(&planned(unchecked, &[Format::Wgsl]));
    let report = check(&plain, &state);
    assert_eq!(
        report
            .problems
            .iter()
            .filter(|line| line.contains("export_unchecked is [pairs], the source's is none"))
            .count(),
        2,
        "{:?}",
        report.problems
    );
}

#[test]
fn a_recompile_without_spirv_val_says_so_instead_of_reporting_a_bare_manifest_difference() {
    let records = vec![sample()];
    let built = planned(records.clone(), &[Format::Wgsl, Format::Spirv]);
    assert_eq!(built.manifest.kernels[0].formats[1].validated, Some(true));
    let state = state_of(&built);
    let mut without = built.clone();
    without.manifest.kernels[0].formats[1].validated = Some(false);
    let report = run(&Inputs {
        records: &records,
        state: &state,
        installed_slang: Some(Ok("fake-slang-1".to_owned())),
        verify: &ok_verify,
        recompiled: Some(&without),
    });
    assert_eq!(report.problems.len(), 1, "{:?}", report.problems);
    assert!(
        report.problems[0].starts_with(&format!(
            "`{SAMPLE_KEY}.spv` was validated by spirv-val when it was built, but spirv-val is not on PATH now"
        )),
        "{:?}",
        report.problems
    );
}
