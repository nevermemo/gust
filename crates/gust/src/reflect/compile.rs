use super::{
    check::check_stage,
    error::{Error, invalid},
    json::{Json, Parser},
    model::{CompiledReflection, Reflection, Stage},
    read::{read_schema_version, read_stage},
};
use crate::{
    KernelDescriptor,
    slang::{self, Target},
};
#[cfg(feature = "slang-runtime")]
use crate::{SourceOrigin, slang::TemporaryDirectory};
#[cfg(feature = "slang-runtime")]
use std::{fs, path::Path, path::PathBuf, process::Command};

/// The native helper that [`compile_reflected`] launches, by the rules of `crate::toolchain` (D109):
/// `GUST_SLANG_REFLECT`, else the managed helper `<root>/helper/<pin>/gust-slang-reflect` of the Slang cache, else the
/// repository checkout's `target/slang-reflect` (only when this crate is built from a GUST checkout), else the bare
/// executable name, which the launch looks up on `PATH`. There is no current-directory lookup.
///
/// This accessor never fails: when the variables are unusable (a relative or empty `GUST_SLANG_REFLECT`, a bad
/// `GUST_SLANG_HOME`) it returns `GUST_SLANG_REFLECT` as given, or the bare name, and [`compile_reflected`] is the
/// call that reports the problem.
#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
///
/// ```
/// use gust::reflect::native_compiler_path;
///
/// // The helper is looked up by the toolchain rules; this call only names it and never fails. Under the default
/// // rules it is a path to `gust-slang-reflect`, or that bare name.
/// let helper = native_compiler_path();
/// assert!(helper.file_name().is_some());
/// ```
pub fn native_compiler_path() -> PathBuf {
    crate::toolchain::helper_candidate_in(&crate::toolchain::Env::from_process())
}

#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
/// Compile and reflect the same linked program using the native Slang helper ([`native_compiler_path`]). An unusable
/// `GUST_SLANG_REFLECT` or `GUST_SLANG_HOME` is an error here, with the variable and its value in the text.
///
/// ```
/// use gust::gpu;
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn double(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use gust::reflect::compile_reflected;
///     use gust::slang::Target;
///
///     let compiled = compile_reflected(&ops::double::DESCRIPTOR, Target::Wgsl)?;
///     assert_eq!(compiled.entry_point, ops::double::DESCRIPTOR.entry_point);
///     assert_eq!(compiled.workgroup_size, [64, 1, 1]);
///     assert!(!compiled.artifact.is_empty());
///     Ok(())
/// }
/// ```
pub fn compile_reflected(
    kernel: &KernelDescriptor,
    target: Target,
) -> Result<CompiledReflection, Error> {
    let helper = crate::toolchain::helper_path()
        .map_err(|reason| Error::Compiler(slang::Error::Toolchain(reason)))?
        .unwrap_or_else(crate::toolchain::bare_helper_name);
    let mut compiled = compile_reflected_with_helper(kernel, target, helper)?;
    // The run-time compile is the one that hands the artifact to the device: the subgroup fix-up (M8, D140) makes
    // Slang's WGSL what the pinned naga takes. The artifact hash was checked against the helper's text above, and a
    // run-time compile stores nothing.
    compiled.artifact = slang::fix_up_wgsl(compiled.artifact, target);
    Ok(compiled)
}

/// [`compile_reflected`] with the native reflection helper at `helper` instead of the one the toolchain rules find.
/// The artifact is the helper's WGSL as it is: the subgroup fix-up of `gust::slang::wgsl` (M8, D140) has **not**
/// been applied to it, so a kernel that calls a subgroup operation is WGSL that the pinned `naga` refuses. Use
/// [`compile_reflected`], which applies the fix-up, to hand the artifact to a device.
/// Fails with [`Error::HelperUnavailable`] when the helper cannot be launched, with [`Error::Compiler`] when the
/// compile fails, and with the other variants of [`Error`] when its reflection output is malformed or does not belong
/// to the kernel.
#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
///
/// ```
/// use gust::gpu;
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn double(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use gust::reflect::{compile_reflected_with_helper, native_compiler_path};
///     use gust::slang::Target;
///
///     let compiled =
///         compile_reflected_with_helper(&ops::double::DESCRIPTOR, Target::Wgsl, native_compiler_path())?;
///     assert_eq!(compiled.reflection.resources.len(), 1);
///     Ok(())
/// }
/// ```
pub fn compile_reflected_with_helper(
    kernel: &KernelDescriptor,
    target: Target,
    helper: impl AsRef<Path>,
) -> Result<CompiledReflection, Error> {
    let directory = TemporaryDirectory::create()?;
    let source = directory.path().join("kernel.slang");
    let artifact = directory.path().join("kernel.artifact");
    let json = directory.path().join("reflection.json");
    // The helper reads the compile copy (`#line` directives, M1.3); its source
    // hash is of that text, which `from_output` checks against the same copy.
    let compiled_source =
        slang::write_compile_inputs(directory.path(), kernel.slang_source, &kernel.source)?;
    let mut command = helper_command(helper.as_ref(), kernel, target, &source, &artifact, &json);
    let output =
        slang::output_retrying_busy(&mut command).map_err(|source| Error::HelperUnavailable {
            helper: helper.as_ref().to_owned(),
            source,
        })?;
    if !output.status.success() {
        return Err(Error::Compiler(slang::Error::CompilationFailed {
            target,
            diagnostics: annotated_failure(
                kernel,
                format!(
                    "{}\n{}\ncompiler exit: {}\nreflection helper: {}{}",
                    String::from_utf8_lossy(&output.stderr),
                    String::from_utf8_lossy(&output.stdout),
                    output.status,
                    helper.as_ref().display(),
                    outdated_helper_note(&output),
                ),
            ),
        }));
    }
    CompiledReflection::from_output(
        kernel,
        &compiled_source,
        target,
        fs::read(artifact)?,
        &fs::read_to_string(json)?,
    )
}

/// The helper's command line: source, entry point, target, artifact path, reflection path, and the stage of the
/// kernel's shape. The stage argument is optional for the helper (mx.t0a: it defaults to `compute`); it is passed
/// every time (mx.k), so that a kernel that is not a compute kernel never relies on the default.
#[cfg(feature = "slang-runtime")]
pub(super) fn helper_command(
    helper: &Path,
    kernel: &KernelDescriptor,
    target: Target,
    source: &Path,
    artifact: &Path,
    json: &Path,
) -> Command {
    let mut command = Command::new(helper);
    command
        .arg(source)
        .arg(kernel.entry_point)
        .arg(target_name(target))
        .arg(artifact)
        .arg(json)
        .arg(kernel.shape.slang_stage());
    command
}

/// A line for the diagnostics of a failed helper run when the helper answered with its usage text and that text has no
/// stage argument: a helper from before reflection schema 3 (mx.t0a) takes five arguments and refuses the seventh
/// that [`helper_command`] always passes (mx.k). Empty for any other failure.
#[cfg(feature = "slang-runtime")]
pub(super) fn outdated_helper_note(output: &std::process::Output) -> &'static str {
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
    let (stderr, stdout) = (text(&output.stderr), text(&output.stdout));
    let usage = stderr.contains("usage: gust-slang-reflect")
        || stdout.contains("usage: gust-slang-reflect");
    let has_stage = stderr.contains("[compute|fragment]") || stdout.contains("[compute|fragment]");
    if usage && !has_stage {
        "\nthe reflection helper predates reflection schema 3 and does not take the stage argument; rebuild it \
         (`cargo gust slang build-helper --force`, or `cargo xtask build-slang-reflect --force` in the GUST repository)"
    } else {
        ""
    }
}

/// What [`compile_reflected_clean`] produced: the parsed result and the helper's JSON, verbatim.
#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
///
/// ```
/// use gust::gpu;
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn double(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use gust::reflect::{compile_reflected_clean, native_compiler_path};
///     use gust::slang::Target;
///
///     let clean =
///         compile_reflected_clean(&ops::double::DESCRIPTOR, Target::Wgsl, native_compiler_path())?;
///     assert!(!clean.compiled.artifact.is_empty());
///     assert!(!clean.reflection_json.is_empty());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct CleanCompilation {
    /// The artifact and layouts, checked by `from_output` against the clean text.
    pub compiled: CompiledReflection,
    /// The helper's reflection JSON, byte for byte, as an export stores it beside the artifact.
    pub reflection_json: String,
}

/// Compile and reflect the kernel's *clean* Slang text: `KernelDescriptor::slang_source` exactly, without the `#line`
/// directives of `slang::compile_copy`. The helper's `sourceHash` is then the hash that
/// [`crate::artifact::verify_artifact`] recomputes at run time, so the result can be stored as a precompiled artifact
/// (D45).
///
/// Only a failed compile touches the directive copy: the helper compiles it once more, so that the diagnostics name
/// Rust lines as `compile_reflected_with_helper` does.
#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
///
/// ```
/// use gust::gpu;
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn double(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use gust::reflect::{compile_reflected_clean, native_compiler_path};
///     use gust::slang::Target;
///
///     let clean =
///         compile_reflected_clean(&ops::double::DESCRIPTOR, Target::Wgsl, native_compiler_path())?;
///     assert!(clean.reflection_json.contains("schemaVersion"));
///     assert_eq!(clean.compiled.workgroup_size, [64, 1, 1]);
///     Ok(())
/// }
/// ```
pub fn compile_reflected_clean(
    kernel: &KernelDescriptor,
    target: Target,
    helper: impl AsRef<Path>,
) -> Result<CleanCompilation, Error> {
    let helper = helper.as_ref();
    compile_clean_with(kernel, target, helper, |source, artifact, json| {
        let mut command = helper_command(helper, kernel, target, source, artifact, json);
        slang::output_retrying_busy(&mut command)
    })
}

/// The clean compile with the helper run by `run(source, artifact, json)`; a test supplies a fake.
#[cfg(feature = "slang-runtime")]
pub(super) fn compile_clean_with(
    kernel: &KernelDescriptor,
    target: Target,
    helper: &Path,
    run: impl FnOnce(&Path, &Path, &Path) -> std::io::Result<std::process::Output>,
) -> Result<CleanCompilation, Error> {
    let directory = TemporaryDirectory::create()?;
    let source = directory.path().join("kernel.slang");
    let artifact = directory.path().join("kernel.artifact");
    let json = directory.path().join("reflection.json");
    let compiled_source = write_clean_input(directory.path(), kernel)?;
    let output = run(&source, &artifact, &json).map_err(|source| Error::HelperUnavailable {
        helper: helper.to_owned(),
        source,
    })?;
    if !output.status.success() {
        let clean_failure = Error::Compiler(slang::Error::CompilationFailed {
            target,
            diagnostics: format!(
                "{}\n{}\ncompiler exit: {}\nreflection helper: {}{}\noriginating Rust kernel: {}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout),
                output.status,
                helper.display(),
                outdated_helper_note(&output),
                kernel.qualified_name(),
            ),
        });
        // The lines above refer to the clean text. Rust lines come from the directive copy.
        return Err(
            match compile_reflected_with_helper(kernel, target, helper) {
                Err(annotated @ Error::Compiler(_)) => annotated,
                _ => clean_failure,
            },
        );
    }
    let reflection_json = fs::read_to_string(&json)?;
    let compiled = CompiledReflection::from_output(
        kernel,
        compiled_source,
        target,
        fs::read(&artifact)?,
        &reflection_json,
    )?;
    Ok(CleanCompilation {
        compiled,
        reflection_json,
    })
}

/// Write the clean text as `kernel.slang` into `directory` and return the text the compiler will read, which is
/// `kernel.slang_source` itself.
#[cfg(feature = "slang-runtime")]
fn write_clean_input<'a>(
    directory: &Path,
    kernel: &'a KernelDescriptor,
) -> Result<&'a str, std::io::Error> {
    fs::write(directory.join("kernel.slang"), kernel.slang_source)?;
    Ok(kernel.slang_source)
}

#[cfg(feature = "slang-runtime")]
/// Attribute a failed compile to its Rust origin, as `slang::compile` does. A
/// diagnostic whose error location cannot be mapped still names the kernel.
///
/// A descriptor with [`SourceOrigin::UNKNOWN`] (hand-written, or built outside a
/// macro expansion) gets exactly the text this path produced before the line
/// table existed: the kernel name only, with no construct line (DX19).
fn annotated_failure(kernel: &KernelDescriptor, diagnostics: String) -> String {
    if kernel.source == SourceOrigin::UNKNOWN {
        return format!(
            "{diagnostics}\noriginating Rust kernel: {}",
            kernel.qualified_name()
        );
    }
    let annotated = slang::annotate_kernel_diagnostics(kernel, &diagnostics);
    // A helper above the first kernel marker maps to a Rust line but to no kernel.
    if annotated.contains("\noriginating Rust kernel: ") {
        annotated
    } else {
        format!(
            "{annotated}\noriginating Rust kernel: {}",
            kernel.qualified_name()
        )
    }
}

pub(super) fn target_name(target: Target) -> &'static str {
    match target {
        Target::Wgsl => "wgsl",
        Target::Spirv => "spirv",
    }
}

pub(super) fn fingerprint(bytes: &[u8]) -> String {
    format!("{:016x}", crate::artifact::fnv1a64(&[bytes]))
}

impl CompiledReflection {
    /// `compiled_source` is the text the helper compiled, whose hash the helper
    /// reports as `sourceHash`: the kernel's Slang with the `#line` directives of
    /// [`slang::compile_copy`] for a compile at run time, the clean
    /// `KernelDescriptor::slang_source` for a precompiled artifact
    /// ([`crate::artifact::verify_artifact`]).
    pub(crate) fn from_output(
        kernel: &KernelDescriptor,
        compiled_source: &str,
        target: Target,
        artifact: Vec<u8>,
        json: &str,
    ) -> Result<Self, Error> {
        let root = Parser::parse(json)?;
        let schema = read_schema_version(&root)?;
        let compiler_version = root.get("compiler")?.string()?.to_owned();
        if compiler_version.trim().is_empty() {
            return Err(invalid("missing native compiler build identity"));
        }
        let source_hash = fingerprint(compiled_source.as_bytes());
        let artifact_hash = fingerprint(&artifact);
        let mut differs = Vec::new();
        for (what, key, expected) in [
            ("target", "target", target_name(target)),
            ("entry point", "entryPoint", kernel.entry_point),
            ("source hash", "sourceHash", source_hash.as_str()),
            ("artifact hash", "artifactHash", artifact_hash.as_str()),
        ] {
            let stored = root.get(key)?.string()?;
            if stored != expected {
                differs.push(format!("{what} (stored {stored:?}, expected {expected:?})"));
            }
        }
        if !differs.is_empty() {
            return Err(invalid(format!(
                "native target, entry point, source or artifact identity mismatch: {}",
                differs.join("; ")
            )));
        }
        // The stage is judged before the workgroup size: a compute document read for a pixel kernel (or the other way
        // round) is a stage mismatch, not a workgroup one.
        let stage = read_stage(&root, schema)?;
        check_stage(kernel.shape, stage)?;
        let workgroup_size = match stage {
            Stage::Compute => {
                let workgroup_size = root
                    .get("workgroupSize")?
                    .array()?
                    .iter()
                    .map(Json::u32)
                    .collect::<Result<Vec<_>, _>>()?;
                let workgroup_size: [u32; 3] = workgroup_size
                    .try_into()
                    .map_err(|_| invalid("expected three workgroup dimensions"))?;
                if workgroup_size.contains(&0) || workgroup_size != kernel.workgroup_size {
                    return Err(Error::Mismatch(
                        "compute workgroup dimensions differ".into(),
                    ));
                }
                workgroup_size
            }
            // A fragment entry point has no workgroup: the document has no `workgroupSize` and the field is zeros.
            Stage::Fragment => {
                if root.optional("workgroupSize")?.is_some() {
                    return Err(invalid("a fragment reflection has no workgroupSize"));
                }
                [0; 3]
            }
        };
        if artifact.is_empty() {
            return Err(invalid("empty native artifact"));
        }
        match target {
            Target::Wgsl => {
                std::str::from_utf8(&artifact).map_err(|_| invalid("native WGSL is not UTF-8"))?;
            }
            Target::Spirv => {
                slang::decode_spirv(&artifact).map_err(Error::Compiler)?;
            }
        }
        Ok(Self {
            artifact,
            reflection: Reflection::read_json(target, &root, true, schema)?,
            compiler_version,
            entry_point: kernel.entry_point.to_owned(),
            workgroup_size,
        })
    }
}

/// The `slangc` command of the CLI reflection path up to its output options. The matrix layout is named for the same
/// reason as in `slang::slangc_command`: the CLI and the native helper must agree on it (M6 spike V1). The stage is
/// the shape's, as it is there (mx.k).
#[cfg(feature = "slang-runtime")]
pub(super) fn slangc_reflection_command(
    slangc: &Path,
    source: &Path,
    target: Target,
    entry: &str,
    shape: crate::Shape,
) -> Command {
    let mut command = Command::new(slangc);
    command.arg(source).args([
        "-target",
        target_name(target),
        "-entry",
        entry,
        "-stage",
        shape.slang_stage(),
        "-matrix-layout-column-major",
    ]);
    if target == Target::Spirv {
        command.arg("-fvk-use-entrypoint-name");
    }
    command
}

#[cfg(feature = "slang-runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "slang-runtime")))]
/// Compile exactly this target and entry point, requesting Slang's JSON metadata.
///
/// ```
/// use gust::gpu;
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn double(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     use gust::reflect::{Stage, reflect};
///     use gust::slang::Target;
///
///     let reflection = reflect(&ops::double::DESCRIPTOR, Target::Wgsl)?;
///     assert_eq!(reflection.stage, Stage::Compute);
///     assert_eq!(reflection.resources.len(), 1);
///     assert_eq!(reflection.resources[0].name, "out");
///     Ok(())
/// }
/// ```
pub fn reflect(kernel: &KernelDescriptor, target: Target) -> Result<Reflection, Error> {
    let directory = TemporaryDirectory::create()?;
    let source = directory.path().join("kernel.slang");
    let json = directory.path().join("reflection.json");
    let extension = match target {
        Target::Wgsl => "wgsl",
        Target::Spirv => "spv",
    };
    slang::write_compile_inputs(directory.path(), kernel.slang_source, &kernel.source)?;
    let slangc = slang::pinned_slangc().map_err(Error::Compiler)?;
    let mut command =
        slangc_reflection_command(&slangc, &source, target, kernel.entry_point, kernel.shape);
    command
        .arg("-reflection-json")
        .arg(&json)
        .arg("-o")
        .arg(directory.path().join(format!("kernel.{extension}")));
    let output = slang::output_retrying_busy(&mut command).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::Compiler(slang::Error::CompilerNotFound)
        } else {
            e.into()
        }
    })?;
    if !output.status.success() {
        return Err(Error::Compiler(slang::Error::CompilationFailed {
            target,
            diagnostics: annotated_failure(
                kernel,
                format!(
                    "{}\n{}\ncompiler exit: {}",
                    String::from_utf8_lossy(&output.stderr),
                    String::from_utf8_lossy(&output.stdout),
                    output.status,
                ),
            ),
        }));
    }
    Reflection::from_json(target, &fs::read_to_string(json)?)
}
