//! `cargo gust check`: is the committed `gust-artifacts/` directory the one a build of the current sources gives?
//!
//! The comparison is pure: the collected records, the directory contents, the installed Slang version and the verifier
//! come in as arguments, and the result is one line per problem. `check` is the CI freshness gate of a kernel crate.
//!
//! Problems it reports:
//!
//! - the manifest or the directory is missing, the manifest is unreadable, or it is not in canonical form (a
//!   hand edit, or a checkout that rewrote its line endings);
//! - a collected kernel has no manifest entry, or its entry names another module, name, entry point or clean-text
//!   SHA-256;
//! - a manifest entry is not a collected kernel (an orphan), or has no WGSL artifact;
//! - a manifest entry's `export_unchecked` list differs from the one the collected record gives (the storage and
//!   uniform parameters whose layout the macro could not write);
//! - a listed file is missing, or its SHA-256 differs from the manifest's;
//! - a stored artifact does not verify the way a program verifies it at run time
//!   (`gust::artifact::verify_artifact`), or was compiled by another Slang build than the manifest says;
//! - a file or directory is not part of the manifest (strays, a leftover `.staging-*` directory);
//! - the manifest's Slang build differs from the installed helper's, unless the check was told to accept that;
//! - with a recompile, any file or the manifest differs from what compiling the sources now gives (S3).
//!
//! `validated: false` on a SPIR-V entry is a note, not a problem: the module was exported without `spirv-val`. So is a kernel with
//! `export_unchecked` parameters: one note per kernel says that their descriptor-side D17 check was skipped at export.

use std::collections::BTreeSet;

use gust::Format;

use crate::{
    export::Planned,
    manifest::{MANIFEST_FILE, Manifest, format_name},
    record::Record,
    sha256::sha256_hex,
    store::{DirState, STAGING_PREFIX},
};

/// Verifies stored bytes and reflection JSON against a record; returns the Slang build tag on success.
pub(crate) type Verifier<'a> = &'a dyn Fn(&Record, Format, &[u8], &str) -> Result<String, String>;

/// What `check` compares.
pub(crate) struct Inputs<'a> {
    /// The collected kernels, sorted by key, without repeats.
    pub(crate) records: &'a [Record],
    pub(crate) state: &'a DirState,
    /// The Slang build tag of the installed helper, or the reason it is unknown. `None` skips the comparison
    /// (`--allow-slang-mismatch`).
    pub(crate) installed_slang: Option<Result<String, String>>,
    pub(crate) verify: Verifier<'a>,
    /// What compiling the sources now gives, for `--recompile`.
    pub(crate) recompiled: Option<&'a Planned>,
}

/// The outcome of a check.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Report {
    /// One line per problem; any problem fails the check.
    pub(crate) problems: Vec<String>,
    /// Things worth knowing that do not fail it.
    pub(crate) notes: Vec<String>,
}

/// `[a, b]`, or `none`.
fn list(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_owned()
    } else {
        format!("[{}]", names.join(", "))
    }
}

/// The one line `build` and `check` print for a kernel with unchecked parameters.
pub(crate) fn unchecked_note(kernel: &str, unchecked: &[String]) -> String {
    format!(
        "kernel `{kernel}` export_unchecked {}: the macro could not describe these element types, so their layout comes from the Slang reflection and the descriptor-side D17 check is skipped for them here; it runs at run time against the real GpuPod::LAYOUT",
        list(unchecked)
    )
}

fn short(hash: &str) -> &str {
    hash.get(..16).unwrap_or(hash)
}

/// Compare the collected kernels with the artifact directory.
pub(crate) fn run(inputs: &Inputs<'_>) -> Report {
    let mut report = Report::default();
    let state = inputs.state;
    let manifest = read_manifest(state, &mut report);
    let mut expected: BTreeSet<String> = BTreeSet::new();
    expected.insert(MANIFEST_FILE.to_owned());
    if let Some(manifest) = &manifest {
        compare_slang(manifest, inputs, &mut report);
        for record in inputs.records {
            match manifest.entry(&record.key) {
                None => report.problems.push(format!(
                    "kernel `{}` (key {}) has no entry in {MANIFEST_FILE}; run `cargo gust build`",
                    record.qualified_name(),
                    record.key
                )),
                Some(entry) => {
                    compare_entry(
                        record,
                        entry,
                        manifest.slang.as_deref(),
                        inputs,
                        &mut report,
                    );
                }
            }
        }
        for entry in &manifest.kernels {
            if !inputs.records.iter().any(|record| record.key == entry.key) {
                report.problems.push(format!(
                    "manifest entry {} (`{}`) is not a collected kernel; `cargo gust build` prunes it",
                    entry.key,
                    entry.qualified_name()
                ));
            }
            for format in &entry.formats {
                expected.insert(format.file.clone());
                expected.insert(format.reflection_file());
            }
        }
    } else {
        for record in inputs.records {
            report.problems.push(format!(
                "kernel `{}` (key {}) has no artifacts: there is no readable {MANIFEST_FILE}",
                record.qualified_name(),
                record.key
            ));
        }
    }
    for (name, _) in &state.files {
        // The routing driver's table (m8.r2) is not an artifact and is not a stray.
        if name == crate::routes::ROUTES_FILE {
            continue;
        }
        if !expected.contains(name) {
            report.problems.push(format!(
                "`{name}` is not part of the manifest (a stray file or a stale artifact); `cargo gust build` prunes it"
            ));
        }
    }
    for directory in &state.dirs {
        if directory.starts_with(STAGING_PREFIX) {
            report.problems.push(format!(
                "`{directory}/` is a leftover staging directory of an interrupted build; `cargo gust build` removes it"
            ));
        } else {
            report.problems.push(format!(
                "`{directory}/` is a directory; the artifact directory holds files only"
            ));
        }
    }
    if let Some(planned) = inputs.recompiled {
        compare_recompiled(planned, state, &mut report);
    }
    report
}

fn read_manifest(state: &DirState, report: &mut Report) -> Option<Manifest> {
    if !state.exists {
        report.problems.push(
            "gust-artifacts/ does not exist; run `cargo gust init` and `cargo gust build`"
                .to_owned(),
        );
        return None;
    }
    let Some(bytes) = state.file(MANIFEST_FILE) else {
        report.problems.push(format!(
            "gust-artifacts/{MANIFEST_FILE} is missing; run `cargo gust init` and `cargo gust build`"
        ));
        return None;
    };
    let Ok(text) = std::str::from_utf8(bytes) else {
        report
            .problems
            .push(format!("{MANIFEST_FILE} is not UTF-8"));
        return None;
    };
    match Manifest::parse(text) {
        Err(error) => {
            report.problems.push(format!("{MANIFEST_FILE}: {error}"));
            None
        }
        Ok(manifest) => {
            if manifest.render() != text {
                report.problems.push(format!(
                    "{MANIFEST_FILE} is not in canonical form (hand edited, or its line endings were rewritten: committed artifacts must be byte-exact, see .gitattributes); run `cargo gust build`"
                ));
            }
            Some(manifest)
        }
    }
}

fn compare_slang(manifest: &Manifest, inputs: &Inputs<'_>, report: &mut Report) {
    if manifest.slang.is_none() && !manifest.kernels.is_empty() {
        report.problems.push(format!(
            "{MANIFEST_FILE} has kernels but no `slang` build tag"
        ));
    }
    let Some(installed) = &inputs.installed_slang else {
        return;
    };
    match (installed, &manifest.slang) {
        (Err(error), _) => report.problems.push(format!(
            "cannot read the installed Slang version: {error} (pass --allow-slang-mismatch to skip this check)"
        )),
        (Ok(installed), Some(stored)) if installed != stored => report.problems.push(format!(
            "the manifest was built with Slang `{stored}` but the installed helper reports `{installed}` (pass --allow-slang-mismatch to accept this)"
        )),
        _ => {}
    }
}

fn compare_entry(
    record: &Record,
    entry: &crate::manifest::KernelEntry,
    slang: Option<&str>,
    inputs: &Inputs<'_>,
    report: &mut Report,
) {
    let name = record.qualified_name();
    for (what, stored, found) in [
        ("module", entry.module.as_str(), record.module.as_str()),
        ("name", entry.name.as_str(), record.name.as_str()),
        (
            "entry point",
            entry.entry_point.as_str(),
            record.entry_point.as_str(),
        ),
    ] {
        if stored != found {
            report.problems.push(format!(
                "kernel `{name}` (key {}): the manifest has {what} `{stored}`, the source has `{found}`",
                record.key
            ));
        }
    }
    if entry.shape != record.shape {
        report.problems.push(format!(
            "kernel `{name}` (key {}): the manifest has shape `{}`, the source has `{}`; run `cargo gust build`",
            record.key,
            entry.shape.name(),
            record.shape.name()
        ));
    }
    let clean = record.clean_sha256();
    if entry.clean_sha256 != clean {
        report.problems.push(format!(
            "kernel `{name}` (key {}): the manifest's clean-text SHA-256 {} differs from the source's {}",
            record.key,
            short(&entry.clean_sha256),
            short(&clean)
        ));
    }
    let unchecked = record.unchecked_parameters();
    if entry.export_unchecked != unchecked {
        report.problems.push(format!(
            "kernel `{name}` (key {}): the manifest's export_unchecked is {}, the source's is {}; run `cargo gust build`",
            record.key,
            list(&entry.export_unchecked),
            list(&unchecked)
        ));
    }
    if !unchecked.is_empty() {
        report.notes.push(unchecked_note(&name, &unchecked));
    }
    if !entry
        .formats
        .iter()
        .any(|format| format.format == Format::Wgsl)
    {
        report.problems.push(format!(
            "kernel `{name}` (key {}) has no WGSL artifact; the macro embeds `{}.wgsl`",
            record.key, record.key
        ));
    }
    for format in &entry.formats {
        let artifact = inputs.state.file(&format.file);
        let reflection_name = format.reflection_file();
        let reflection = inputs.state.file(&reflection_name);
        for (file, found, expected) in [
            (&format.file, artifact, &format.sha256),
            (&reflection_name, reflection, &format.reflection_sha256),
        ] {
            match found {
                None => report.problems.push(format!(
                    "`{file}` is listed in the manifest but missing; run `cargo gust build`"
                )),
                Some(bytes) => {
                    let actual = sha256_hex(bytes);
                    if actual != *expected {
                        report.problems.push(format!(
                            "`{file}`: SHA-256 {} differs from the manifest's {}",
                            short(&actual),
                            short(expected)
                        ));
                    }
                }
            }
        }
        if format.validated == Some(false) {
            report.notes.push(format!(
                "`{}` was exported without spirv-val (validated: false)",
                format.file
            ));
        }
        let (Some(artifact), Some(reflection)) = (artifact, reflection) else {
            continue;
        };
        let Ok(reflection) = std::str::from_utf8(reflection) else {
            report
                .problems
                .push(format!("`{reflection_name}` is not UTF-8"));
            continue;
        };
        match (inputs.verify)(record, format.format, artifact, reflection) {
            Err(error) => report.problems.push(format!(
                "`{}` ({}) does not verify against kernel `{name}`: {error}",
                format.file,
                format_name(format.format)
            )),
            Ok(tag) => {
                if let Some(stored) = slang
                    && stored != tag
                {
                    report.problems.push(format!(
                        "`{}` was compiled by Slang `{tag}` but the manifest says `{stored}`",
                        format.file
                    ));
                }
            }
        }
    }
}

fn compare_recompiled(planned: &Planned, state: &DirState, report: &mut Report) {
    for file in &planned.files {
        match state.file(&file.name) {
            None => report.problems.push(format!(
                "recompiling produces `{}`, which is not stored; run `cargo gust build`",
                file.name
            )),
            Some(stored) if stored != file.bytes => report.problems.push(format!(
                "recompiling gives different bytes for `{}` (stored SHA-256 {}, now {})",
                file.name,
                short(&sha256_hex(stored)),
                short(&sha256_hex(&file.bytes))
            )),
            Some(_) => {}
        }
    }
    let mut planned_manifest = planned.manifest.clone();
    let stored_manifest = state
        .file(MANIFEST_FILE)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| Manifest::parse(text).ok());
    if let Some(stored_manifest) = &stored_manifest {
        // A SPIR-V module that was validated when it was built, recompiled where `spirv-val` is missing: say so
        // instead of reporting a bare manifest difference.
        for entry in &mut planned_manifest.kernels {
            let Some(stored) = stored_manifest.entry(&entry.key) else {
                continue;
            };
            for format in &mut entry.formats {
                let was = stored
                    .formats
                    .iter()
                    .find(|other| other.format == format.format)
                    .and_then(|other| other.validated);
                if was == Some(true) && format.validated == Some(false) {
                    report.problems.push(format!(
                        "`{}` was validated by spirv-val when it was built, but spirv-val is not on PATH now, so the recompile cannot confirm it; install spirv-val (Vulkan SDK or spirv-tools)",
                        format.file
                    ));
                    format.validated = was;
                }
            }
        }
    }
    if state.file(MANIFEST_FILE) != Some(planned_manifest.render().as_bytes()) {
        report.problems.push(format!(
            "recompiling gives a different {MANIFEST_FILE}; run `cargo gust build`"
        ));
    }
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod tests;
