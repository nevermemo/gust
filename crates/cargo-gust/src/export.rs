//! Compiling and verifying the collected kernels, and planning the files of `gust-artifacts/`.
//!
//! [`plan`] turns the records into the complete set of files and the manifest in memory. It writes nothing: the
//! writing is [`crate::store::commit`], which only runs after every kernel and format compiled and verified, so a
//! failed export leaves the artifact directory untouched.
//!
//! The compiler sits behind the [`Compiler`] trait. Production uses [`HelperCompiler`], which runs the native Slang
//! helper on the clean text, cross-checks the layouts against the descriptor (D17) and then verifies the result the
//! way a program would at run time; tests use a fake that needs no helper.

use std::{fs, path::PathBuf, process::Command};

use gust::{
    Artifact, Format, KernelDescriptor, PodCrossCheck, artifact::verify_artifact,
    reflect::Reflection,
};

use crate::{
    manifest::{
        FormatEntry, KernelEntry, MANIFEST_FILE, Manifest, VariantEntry, artifact_file,
        format_name, format_rank, reflection_file,
    },
    record::Record,
    sha256::sha256_hex,
    store::OutFile,
};

/// What compiling one kernel in one format produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Compiled {
    /// The compiler output, verbatim.
    pub(crate) bytes: Vec<u8>,
    /// The helper's reflection JSON, verbatim.
    pub(crate) reflection_json: String,
    /// The Slang build tag the reflection JSON reports.
    pub(crate) compiler: String,
    /// `Some` for SPIR-V: whether `spirv-val` accepted the module (`false` means the tool was not available).
    pub(crate) validated: Option<bool>,
    /// Things the user should know that do not fail the export.
    pub(crate) warnings: Vec<String>,
}

/// Compiles one kernel in one format and verifies the result. A failure stops the whole export.
pub(crate) trait Compiler {
    fn compile(
        &self,
        record: &Record,
        kernel: &KernelDescriptor,
        format: Format,
    ) -> Result<Compiled, String>;
}

/// The files and manifest of one export, not yet written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Planned {
    /// Artifact and reflection files, sorted by name.
    pub(crate) files: Vec<OutFile>,
    pub(crate) manifest: Manifest,
    pub(crate) warnings: Vec<String>,
}

impl Planned {
    /// The manifest as the file `commit` writes last.
    pub(crate) fn manifest_file(&self) -> OutFile {
        OutFile {
            name: MANIFEST_FILE.to_owned(),
            bytes: self.manifest.render().into_bytes(),
        }
    }
}

/// WGSL always (it is what the macro embeds), plus what was asked for, each once, in manifest order.
pub(crate) fn formats_with_wgsl(requested: &[Format]) -> Vec<Format> {
    let mut formats = vec![Format::Wgsl];
    for format in requested {
        if !formats.contains(format) {
            formats.push(*format);
        }
    }
    formats.sort_by_key(|format| format_rank(*format));
    formats
}

/// Sort the records by key and collapse exact repeats of one kernel (the same kernel expanded twice, for example in a
/// library and a binary target, writes the same record twice).
///
/// Two kernels with one key are an error (R4.4): different clean text means the 64-bit FNV-1a keys collided, and
/// the same text under two names means two kernels share one entry point. Both would share one artifact file.
pub(crate) fn dedupe(mut records: Vec<Record>) -> Result<Vec<Record>, String> {
    records.sort_by(|a, b| a.key.cmp(&b.key));
    let mut kept: Vec<Record> = Vec::new();
    for record in records {
        let Some(last) = kept.last() else {
            kept.push(record);
            continue;
        };
        if last.key != record.key {
            kept.push(record);
            continue;
        }
        if last.slang != record.slang {
            return Err(format!(
                "kernels `{}` and `{}` have the same artifact key {} but different Slang text: the 64-bit FNV-1a keys collided (R4.4); change the text of one kernel",
                last.body_name(),
                record.body_name(),
                record.key
            ));
        }
        if last.module != record.module || last.name != record.name {
            return Err(format!(
                "kernels `{}` and `{}` have the same entry point `{}` and Slang text, so they would share the artifact key {}; rename one of them",
                last.body_name(),
                record.body_name(),
                record.entry_point,
                record.key
            ));
        }
    }
    Ok(kept)
}

/// The `variants` lists of the manifest, as `(key of the own body, bodies sorted by key)`.
///
/// Every body of a kernel is a record of its own (its entry point, key and files), and the bodies of one kernel share
/// its `module` and `name` (M8, D139): the records are grouped by that pair. A kernel is listed when one of its bodies
/// has facts of its own, a `variant` or a `requires` (`Record::has_variant_facts`); a group of plain bodies is not,
/// so a crate without variants keeps the manifest it had. A listed kernel needs exactly one own body (`variant` ""),
/// and each variant function appears once. The records may come in any order; they are those of one `cargo check`,
/// whatever the profile (the macro's export mode writes every body, so no second build is needed).
pub(crate) fn variant_lists(
    records: &[Record],
) -> Result<Vec<(String, Vec<VariantEntry>)>, String> {
    let mut groups: Vec<((&str, &str), Vec<&Record>)> = Vec::new();
    for record in records {
        let at = (record.module.as_str(), record.name.as_str());
        match groups.iter_mut().find(|(group, _)| *group == at) {
            Some((_, members)) => members.push(record),
            None => groups.push((at, vec![record])),
        }
    }
    let mut lists = Vec::new();
    for ((module, name), members) in groups {
        if !members.iter().any(|record| record.has_variant_facts()) {
            continue;
        }
        let kernel = format!("{module}::{name}");
        let own: Vec<&&Record> = members
            .iter()
            .filter(|record| record.variant.is_empty())
            .collect();
        let owner = match own.as_slice() {
            [only] => only.key.clone(),
            [] => {
                return Err(format!(
                    "kernel `{kernel}` has variants but no body of its own (a `#[gpu]` function named `{name}`) in the collected records; rebuild with the macros of this gust version"
                ));
            }
            [first, second, ..] => {
                return Err(format!(
                    "kernel `{kernel}` has two bodies of its own with different text (keys {} and {}); one `#[gpu]` function `{name}` per module",
                    first.key, second.key
                ));
            }
        };
        let mut bodies: Vec<VariantEntry> = Vec::new();
        for record in members {
            if !record.variant.is_empty()
                && bodies.iter().any(|body| body.variant == record.variant)
            {
                return Err(format!(
                    "kernel `{kernel}` has the variant `{}` twice (keys {} and {}); rename one of them",
                    record.variant,
                    bodies
                        .iter()
                        .find(|body| body.variant == record.variant)
                        .map_or("", |body| body.key.as_str()),
                    record.key
                ));
            }
            bodies.push(VariantEntry {
                key: record.key.clone(),
                requires: record.requires.clone(),
                variant: record.variant.clone(),
            });
        }
        bodies.sort_by(|a, b| a.key.cmp(&b.key));
        lists.push((owner, bodies));
    }
    lists.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(lists)
}

/// What `check` says about the `variants` lists: the stored manifest against the lists the collected records give
/// ([`variant_lists`]). Empty when they agree, or when the records themselves are inconsistent and the manifest cannot
/// be judged (the error is the problem then).
pub(crate) fn variant_problems(records: &[Record], manifest: &Manifest) -> Vec<String> {
    let expected = match variant_lists(records) {
        Ok(lists) => lists,
        Err(error) => return vec![error],
    };
    let mut problems = Vec::new();
    for (owner, bodies) in &expected {
        let Some(entry) = manifest.entry(owner) else {
            continue; // the missing entry is reported with the kernels
        };
        if entry.variants.is_empty() {
            problems.push(format!(
                "kernel `{}` (key {owner}) has {} variant body(ies) now but the manifest lists none; run `cargo gust build`",
                entry.qualified_name(),
                bodies.len()
            ));
            continue;
        }
        let mut stored = entry.variants.clone();
        stored.sort_by(|a, b| a.key.cmp(&b.key));
        if stored != *bodies {
            problems.push(format!(
                "kernel `{}` (key {owner}): the `variants` list of the manifest ({}) is not the one the sources give ({}); run `cargo gust build`",
                entry.qualified_name(),
                describe_bodies(&stored),
                describe_bodies(bodies)
            ));
        }
    }
    for entry in &manifest.kernels {
        if !entry.variants.is_empty() && !expected.iter().any(|(owner, _)| *owner == entry.key) {
            problems.push(format!(
                "manifest entry {} (`{}`) lists variants but the sources have none for it; `cargo gust build` rewrites it",
                entry.key,
                entry.qualified_name()
            ));
        }
    }
    problems
}

/// `key variant requires`, for the bodies of one kernel, as a short comparison text.
fn describe_bodies(bodies: &[VariantEntry]) -> String {
    bodies
        .iter()
        .map(|body| {
            let name = if body.variant.is_empty() {
                "own body"
            } else {
                body.variant.as_str()
            };
            let needs: Vec<&str> = body.requires.iter().map(|c| c.name()).collect();
            format!("{name} {} needs [{}]", body.key, needs.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Compile every record in every format and build the files and the manifest.
pub(crate) fn plan(
    records: Vec<Record>,
    formats: &[Format],
    compiler: &dyn Compiler,
) -> Result<Planned, String> {
    let records = dedupe(records)?;
    let mut files = Vec::new();
    let mut kernels = Vec::new();
    let mut warnings = Vec::new();
    let mut tag: Option<String> = None;
    for record in &records {
        let kernel = record.descriptor();
        let mut entries = Vec::new();
        for format in formats {
            let compiled = compiler
                .compile(record, &kernel, *format)
                .map_err(|error| {
                    format!("{} ({}): {error}", record.body_name(), format_name(*format))
                })?;
            match &tag {
                None => tag = Some(compiled.compiler.clone()),
                Some(first) if *first != compiled.compiler => {
                    return Err(format!(
                        "{} ({}): the helper reports Slang build `{}` but an earlier artifact says `{first}`; one export uses one compiler",
                        record.body_name(),
                        format_name(*format),
                        compiled.compiler
                    ));
                }
                Some(_) => {}
            }
            warnings.extend(compiled.warnings.iter().map(|warning| {
                format!(
                    "{} ({}): {warning}",
                    record.body_name(),
                    format_name(*format)
                )
            }));
            let file = artifact_file(&record.key, *format);
            entries.push(FormatEntry {
                format: *format,
                sha256: sha256_hex(&compiled.bytes),
                reflection_sha256: sha256_hex(compiled.reflection_json.as_bytes()),
                validated: compiled.validated,
                file: file.clone(),
            });
            files.push(OutFile {
                name: reflection_file(&file),
                bytes: compiled.reflection_json.into_bytes(),
            });
            files.push(OutFile {
                name: file,
                bytes: compiled.bytes,
            });
        }
        kernels.push(KernelEntry {
            key: record.key.clone(),
            module: record.module.clone(),
            name: record.name.clone(),
            entry_point: record.entry_point.clone(),
            shape: record.shape,
            clean_sha256: record.clean_sha256(),
            export_unchecked: record.unchecked_parameters(),
            formats: entries,
            variants: Vec::new(),
        });
    }
    for (owner, bodies) in variant_lists(&records)? {
        if let Some(entry) = kernels.iter_mut().find(|entry| entry.key == owner) {
            entry.variants = bodies;
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Planned {
        files,
        manifest: Manifest {
            slang: tag,
            kernels,
        },
        warnings,
    })
}

fn leak_bytes(bytes: &[u8]) -> &'static [u8] {
    Box::leak(bytes.to_vec().into_boxed_slice())
}

fn leak_str(text: &str) -> &'static str {
    Box::leak(text.to_owned().into_boxed_str())
}

/// Verify stored bytes and reflection JSON against the record exactly as a program does at run time
/// (`gust::artifact::verify_artifact`: identity, hashes, workgroup size and the complete layout cross-check). Returns
/// the Slang build tag the reflection JSON reports.
pub(crate) fn verify_stored(
    record: &Record,
    format: Format,
    bytes: &[u8],
    reflection_json: &str,
) -> Result<String, String> {
    let kernel = completed_descriptor(record, reflection_json)?;
    let artifact = Artifact {
        format,
        key: leak_str(&record.key),
        bytes: leak_bytes(bytes),
        reflection: leak_str(reflection_json),
    };
    verify_artifact(&kernel, &artifact)
        .map(|verified| verified.compiled.compiler_version)
        .map_err(|error| error.to_string())
}

/// The record's descriptor with every `layout: null` parameter filled from the reflection ([`Record::completed`]): a
/// storage element as `StorageV1` or `StorageV2` by its shape, a uniform as `Std140V1` (D80).
///
/// A parameter whose record does carry a layout keeps it, with the rules the record states, and the cross-check refuses
/// it when those rules do not fit the parameter (a constant buffer with storage rules, a storage buffer with
/// `std140_v1`). Nothing is accepted on the strength of the fill alone: the descriptor goes through the same
/// cross-check as any other, which holds a constant buffer to std140 and every layout to the reflection.
fn completed_descriptor(
    record: &Record,
    reflection_json: &str,
) -> Result<KernelDescriptor, String> {
    Ok(record.completed(reflection_json)?.descriptor())
}

/// D17 at export time: the helper's reflection against the record's descriptor, resource by resource.
///
/// The macro could not describe some element types (`layout: null`, the manifest's `export_unchecked`): a type from
/// outside the module or an alias, in a storage buffer or in a uniform. Their layout is filled in from the same
/// reflection first ([`Record::completed`]), so for those parameters only the binding, the group and the access are
/// compared with the descriptor and the layout half is skipped; the manifest says so. Every other parameter, uniforms
/// included, is checked in full, layout included: the macro's own numbers against the reflection's.
pub(crate) fn cross_check_completed(
    record: &Record,
    reflection: &Reflection,
    reflection_json: &str,
) -> Result<Vec<PodCrossCheck>, String> {
    let completed = completed_descriptor(record, reflection_json)?;
    reflection
        .cross_check_kernel(&completed)
        .map_err(|error| error.to_string())
}

/// The production compiler: the native helper, then D17's cross-check, then the run-time verification, and for SPIR-V
/// `spirv-val`.
pub(crate) struct HelperCompiler {
    pub(crate) helper: PathBuf,
    /// `spirv-val`, when it was found on `PATH`.
    pub(crate) spirv_val: Option<PathBuf>,
}

impl Compiler for HelperCompiler {
    fn compile(
        &self,
        record: &Record,
        kernel: &KernelDescriptor,
        format: Format,
    ) -> Result<Compiled, String> {
        let clean = gust::reflect::compile_reflected_clean(kernel, format.target(), &self.helper)
            .map_err(|error| error.to_string())?;
        // The cross-check fails with `Incomplete` when the evidence does not cover the full ABI, so a pass is a full pass.
        cross_check_completed(record, &clean.compiled.reflection, &clean.reflection_json)?;
        let compiler = verify_stored(
            record,
            format,
            &clean.compiled.artifact,
            &clean.reflection_json,
        )
        .map_err(|error| {
            format!("the fresh artifact does not verify as a program would verify it: {error}")
        })?;
        let validated = match format {
            Format::Wgsl => None,
            Format::Spirv => Some(self.validate_spirv(&clean.compiled.artifact)?),
            other => return Err(format!("unsupported artifact format {other:?}")),
        };
        Ok(Compiled {
            bytes: clean.compiled.artifact,
            reflection_json: clean.reflection_json,
            compiler,
            validated,
            warnings: Vec::new(),
        })
    }
}

impl HelperCompiler {
    /// `spirv-val --target-env vulkan1.2`. `Ok(false)` when the tool is not available, `Err` when it rejects the
    /// module.
    fn validate_spirv(&self, bytes: &[u8]) -> Result<bool, String> {
        let Some(tool) = &self.spirv_val else {
            return Ok(false);
        };
        let directory = gust::slang::TemporaryDirectory::create().map_err(|error| {
            format!("cannot create a temporary directory for spirv-val: {error}")
        })?;
        let module = directory.path().join("kernel.spv");
        fs::write(&module, bytes)
            .map_err(|error| format!("cannot write the module for spirv-val: {error}"))?;
        let output = Command::new(tool)
            .args(["--target-env", "vulkan1.2"])
            .arg(&module)
            .output()
            .map_err(|error| format!("cannot run spirv-val: {error}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            Err(format!(
                "spirv-val --target-env vulkan1.2 rejected the module ({}): {}{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::testdata::{SAMPLE_KEY, SAMPLE_RECORD_JSON, SAMPLE_SLANG};

    /// A compiler that returns canned, deterministic output and needs no helper.
    pub(crate) struct FakeCompiler {
        pub(crate) tag: &'static str,
        pub(crate) fail_on: Option<(&'static str, Format)>,
    }

    impl FakeCompiler {
        pub(crate) fn new() -> Self {
            Self {
                tag: "fake-slang-1",
                fail_on: None,
            }
        }
    }

    impl Compiler for FakeCompiler {
        fn compile(
            &self,
            record: &Record,
            _kernel: &KernelDescriptor,
            format: Format,
        ) -> Result<Compiled, String> {
            if self.fail_on == Some((record.entry_point.as_str(), format)) {
                return Err("injected compile failure".to_owned());
            }
            let name = format_name(format);
            Ok(Compiled {
                bytes: format!("{name} bytes of {}", record.entry_point).into_bytes(),
                reflection_json: format!(
                    "{{\"compiler\":\"{}\",\"entryPoint\":\"{}\"}}\n",
                    self.tag, record.entry_point
                ),
                compiler: self.tag.to_owned(),
                validated: (format == Format::Spirv).then_some(true),
                warnings: Vec::new(),
            })
        }
    }

    pub(crate) fn sample() -> Record {
        Record::parse(SAMPLE_RECORD_JSON, SAMPLE_SLANG).unwrap()
    }

    /// A second, different kernel: the sample renamed (the entry point is part of the key).
    pub(crate) fn other(name: &str) -> Record {
        let entry = format!("gpu_demo_{name}");
        let slang = SAMPLE_SLANG.replace("gpu_demo_run", &entry);
        let json = SAMPLE_RECORD_JSON
            .replace("gpu_demo_run", &entry)
            .replace("\"name\": \"run\"", &format!("\"name\": \"{name}\""))
            .replace(SAMPLE_KEY, &crate::record::key_of(&entry, &slang));
        Record::parse(&json, &slang).unwrap()
    }

    #[test]
    fn the_plan_holds_every_file_sorted_and_a_manifest_that_describes_them() {
        let planned = plan(
            vec![other("b"), sample()],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        let names: Vec<&str> = planned
            .files
            .iter()
            .map(|file| file.name.as_str())
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"022c52d675787198.wgsl"));
        assert!(names.contains(&"022c52d675787198.wgsl.reflection.json"));
        assert_eq!(planned.manifest.slang.as_deref(), Some("fake-slang-1"));
        let entry = planned.manifest.entry(SAMPLE_KEY).unwrap();
        assert_eq!(entry.qualified_name(), "demo::run");
        assert_eq!(entry.entry_point, "gpu_demo_run");
        assert_eq!(entry.clean_sha256, crate::testdata::SAMPLE_SLANG_SHA256);
        let wgsl = &entry.formats[0];
        let file = planned
            .files
            .iter()
            .find(|file| file.name == wgsl.file)
            .unwrap();
        assert_eq!(wgsl.sha256, sha256_hex(&file.bytes));
        let reflection = planned
            .files
            .iter()
            .find(|file| file.name == wgsl.reflection_file())
            .unwrap();
        assert_eq!(wgsl.reflection_sha256, sha256_hex(&reflection.bytes));
        assert_eq!(wgsl.validated, None);
        // Deterministic: the same inputs in another order give the same bytes.
        let again = plan(
            vec![sample(), other("b")],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        assert_eq!(again, planned);
        assert_eq!(again.manifest_file().bytes, planned.manifest_file().bytes);
    }

    #[test]
    fn spirv_is_listed_with_its_validation_flag() {
        let planned = plan(
            vec![sample()],
            &[Format::Wgsl, Format::Spirv],
            &FakeCompiler::new(),
        )
        .unwrap();
        let entry = planned.manifest.entry(SAMPLE_KEY).unwrap();
        assert_eq!(entry.formats.len(), 2);
        assert_eq!(entry.formats[1].file, "022c52d675787198.spv");
        assert_eq!(entry.formats[1].validated, Some(true));
        assert_eq!(planned.files.len(), 4);
    }

    #[test]
    fn wgsl_is_always_built_and_each_format_once() {
        assert_eq!(formats_with_wgsl(&[]), [Format::Wgsl]);
        assert_eq!(
            formats_with_wgsl(&[Format::Spirv]),
            [Format::Wgsl, Format::Spirv]
        );
        assert_eq!(
            formats_with_wgsl(&[Format::Spirv, Format::Wgsl, Format::Spirv]),
            [Format::Wgsl, Format::Spirv]
        );
    }

    #[test]
    fn two_kernels_with_one_key_and_different_text_are_an_error() {
        let mut clash = other("b");
        clash.key = SAMPLE_KEY.to_owned();
        let error = plan(vec![sample(), clash], &[Format::Wgsl], &FakeCompiler::new()).unwrap_err();
        assert!(
            error.contains("`demo::run` and `demo::b` have the same artifact key 022c52d675787198 but different Slang text"),
            "{error}"
        );
        assert!(error.contains("collided"), "{error}");
    }

    #[test]
    fn an_exact_repeat_of_one_kernel_is_collapsed_and_a_renamed_repeat_is_an_error() {
        let planned = plan(
            vec![sample(), sample()],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        assert_eq!(planned.manifest.kernels.len(), 1);
        assert_eq!(dedupe(vec![sample(), sample()]).unwrap().len(), 1);
        let mut renamed = sample();
        renamed.name = "other".to_owned();
        let error = dedupe(vec![sample(), renamed]).unwrap_err();
        assert!(error.contains("share the artifact key"), "{error}");
    }

    #[test]
    fn a_compile_failure_names_the_kernel_and_format_and_a_mixed_compiler_is_refused() {
        let fake = FakeCompiler {
            tag: "fake-slang-1",
            fail_on: Some(("gpu_demo_b", Format::Spirv)),
        };
        let error = plan(
            vec![sample(), other("b")],
            &[Format::Wgsl, Format::Spirv],
            &fake,
        )
        .unwrap_err();
        assert_eq!(error, "demo::b (spirv): injected compile failure");

        struct Alternating;
        impl Compiler for Alternating {
            fn compile(
                &self,
                record: &Record,
                kernel: &KernelDescriptor,
                format: Format,
            ) -> Result<Compiled, String> {
                let tag = if record.name == "run" {
                    "tag-a"
                } else {
                    "tag-b"
                };
                FakeCompiler { tag, fail_on: None }.compile(record, kernel, format)
            }
        }
        let error = plan(vec![sample(), other("b")], &[Format::Wgsl], &Alternating).unwrap_err();
        assert!(error.contains("one export uses one compiler"), "{error}");
        assert!(
            error.contains("`tag-b`") && error.contains("`tag-a`"),
            "{error}"
        );
    }

    // ---- variants (M8, D139) ----

    use gust::capability::Capability;

    /// A body of the kernel `demo::<kernel>`: its own (`function == kernel`) or a variant named `function`.
    fn body(kernel: &str, function: &str, requires: &[Capability]) -> Record {
        let mut record = other(function);
        record.name = kernel.to_owned();
        if function != kernel {
            record.variant = function.to_owned();
        }
        record.requires = requires.to_vec();
        record
    }

    fn kernel_with_a_variant() -> Vec<Record> {
        vec![
            body("sum", "sum_subgroups", &[Capability::Subgroups]),
            body("sum", "sum", &[]),
            other("plain"),
        ]
    }

    #[test]
    fn the_bodies_of_a_kernel_are_grouped_by_module_and_name_and_the_own_body_carries_the_list() {
        let planned = plan(
            kernel_with_a_variant(),
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        // One entry for every body, three files pairs for three bodies.
        assert_eq!(planned.manifest.kernels.len(), 3);
        assert_eq!(planned.files.len(), 6);
        assert_eq!(planned.manifest.schema(), crate::manifest::VARIANT_SCHEMA);
        let own = body("sum", "sum", &[]).key;
        let variant = body("sum", "sum_subgroups", &[Capability::Subgroups]).key;
        let with_list: Vec<&str> = planned
            .manifest
            .kernels
            .iter()
            .filter(|entry| !entry.variants.is_empty())
            .map(|entry| entry.key.as_str())
            .collect();
        assert_eq!(with_list, [own.as_str()]);
        let list = &planned.manifest.entry(&own).unwrap().variants;
        let mut keys = [own.clone(), variant.clone()];
        keys.sort();
        assert_eq!(
            list.iter().map(|body| body.key.clone()).collect::<Vec<_>>(),
            keys
        );
        let item = |key: &str| list.iter().find(|body| body.key == key).unwrap();
        assert_eq!(item(&own).variant, "");
        assert!(item(&own).requires.is_empty());
        assert_eq!(item(&variant).variant, "sum_subgroups");
        assert_eq!(item(&variant).requires, [Capability::Subgroups]);
        // Both bodies are entries of the kernel `demo::sum`; the variant's entry point is its function's.
        let entry = planned.manifest.entry(&variant).unwrap();
        assert_eq!(entry.qualified_name(), "demo::sum");
        assert_eq!(entry.entry_point, "gpu_demo_sum_subgroups");
        assert!(entry.variants.is_empty());
        // The manifest the plan builds is one the reader accepts, and the freshness check finds nothing.
        let text = planned.manifest.render();
        let parsed = crate::manifest::Manifest::parse(&text).unwrap();
        assert_eq!(parsed, planned.manifest);
        assert_eq!(
            variant_problems(&kernel_with_a_variant(), &parsed),
            Vec::<String>::new()
        );
    }

    #[test]
    fn the_order_of_the_records_does_not_change_the_manifest() {
        let forward = plan(
            kernel_with_a_variant(),
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        let mut reversed_records = kernel_with_a_variant();
        reversed_records.reverse();
        let reversed = plan(reversed_records, &[Format::Wgsl], &FakeCompiler::new()).unwrap();
        assert_eq!(forward.manifest.render(), reversed.manifest.render());
        assert_eq!(forward.files, reversed.files);
    }

    #[test]
    fn kernels_without_variant_facts_make_a_manifest_without_a_list() {
        // Two kernels, no variant and no requirement: schema 1, no `variants` key, as before M8.
        let planned = plan(
            vec![sample(), other("b")],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        assert_eq!(planned.manifest.schema(), crate::manifest::SCHEMA);
        assert!(!planned.manifest.render().contains("variants"));
        // A kernel whose own body is plain stays unlisted even with other kernels listed.
        let planned = plan(
            kernel_with_a_variant(),
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap();
        let plain = planned.manifest.entry(&other("plain").key).unwrap();
        assert!(plain.variants.is_empty());
    }

    #[test]
    fn an_own_body_that_needs_a_capability_is_listed_alone() {
        let records = vec![body("sum", "sum", &[Capability::Subgroups])];
        let planned = plan(records, &[Format::Wgsl], &FakeCompiler::new()).unwrap();
        let entry = &planned.manifest.kernels[0];
        assert_eq!(entry.variants.len(), 1);
        assert_eq!(entry.variants[0].key, entry.key);
        assert_eq!(entry.variants[0].variant, "");
        assert_eq!(entry.variants[0].requires, [Capability::Subgroups]);
        assert_eq!(planned.manifest.schema(), crate::manifest::VARIANT_SCHEMA);
    }

    #[test]
    fn a_kernel_with_variants_but_no_body_of_its_own_is_refused() {
        let error = plan(
            vec![body("sum", "sum_subgroups", &[Capability::Subgroups])],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap_err();
        assert!(
            error.starts_with("kernel `demo::sum` has variants but no body of its own"),
            "{error}"
        );
    }

    #[test]
    fn two_own_bodies_or_a_variant_twice_are_refused() {
        // Two own bodies with different text (the other one is renamed so that its key differs).
        let mut second_own = body("sum", "sum", &[]);
        second_own.slang.push_str("// another\n");
        second_own.key = crate::record::key_of(&second_own.entry_point, &second_own.slang);
        let error = plan(
            vec![
                body("sum", "sum", &[]),
                second_own,
                body("sum", "sum_subgroups", &[Capability::Subgroups]),
            ],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap_err();
        assert!(error.contains("has two bodies of its own"), "{error}");
        // One variant name on two bodies with different text.
        let mut again = body("sum", "sum_subgroups", &[Capability::Subgroups]);
        again.slang.push_str("// another\n");
        again.key = crate::record::key_of(&again.entry_point, &again.slang);
        let error = plan(
            vec![
                body("sum", "sum", &[]),
                body("sum", "sum_subgroups", &[Capability::Subgroups]),
                again,
            ],
            &[Format::Wgsl],
            &FakeCompiler::new(),
        )
        .unwrap_err();
        assert!(
            error.contains("has the variant `sum_subgroups` twice"),
            "{error}"
        );
    }

    #[test]
    fn a_compile_failure_names_the_body_by_its_function() {
        let fake = FakeCompiler {
            tag: "fake-slang-1",
            fail_on: Some(("gpu_demo_sum_subgroups", Format::Wgsl)),
        };
        let error = plan(kernel_with_a_variant(), &[Format::Wgsl], &fake).unwrap_err();
        assert_eq!(
            error,
            "demo::sum_subgroups (wgsl): injected compile failure"
        );
    }

    #[test]
    fn the_freshness_check_finds_a_stale_or_missing_list() {
        let records = kernel_with_a_variant();
        let planned = plan(records.clone(), &[Format::Wgsl], &FakeCompiler::new()).unwrap();
        let own = body("sum", "sum", &[]).key;
        // No list in the manifest where the sources now have variants.
        let mut stripped = planned.manifest.clone();
        for entry in &mut stripped.kernels {
            entry.variants.clear();
        }
        let problems = variant_problems(&records, &stripped);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].starts_with(&format!(
                "kernel `demo::sum` (key {own}) has 2 variant body(ies) now but the manifest lists none"
            )),
            "{problems:?}"
        );
        // A capability changed in the source.
        let mut changed = records.clone();
        changed[0].requires.clear();
        let problems = variant_problems(&changed, &planned.manifest);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].contains("is not the one the sources give"),
            "{problems:?}"
        );
        // A list for a kernel that has no variant any more.
        let problems = variant_problems(&[sample(), other("plain")], &planned.manifest);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("lists variants but the sources have none")),
            "{problems:?}"
        );
        // Inconsistent records are reported as they are.
        let problems = variant_problems(
            &[body("sum", "sum_subgroups", &[Capability::Subgroups])],
            &stripped,
        );
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].contains("has variants but no body of its own"),
            "{problems:?}"
        );
    }
}

#[cfg(test)]
#[path = "export_unchecked_tests.rs"]
mod unchecked_tests;

#[cfg(test)]
#[path = "export_uniform_tests.rs"]
mod uniform_tests;

#[cfg(test)]
#[path = "export_vector_tests.rs"]
mod vector_tests;
