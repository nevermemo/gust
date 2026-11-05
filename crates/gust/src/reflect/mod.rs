//! Slang reflection for the restricted storage and constant-buffer ABIs.
//! Native compilation returns the artifact and complete layout from one linked program.
//! CLI JSON remains an explicit partial-evidence compatibility path.
//! Only global structured-buffer resources (and, under schema 2, constant buffers) are supported. Entry-point
//! parameters must be the generated `SV_DispatchThreadID` system value, not resources/uniforms.
//!
//! # Schema versions of the native reflection JSON
//!
//! The reader accepts `schemaVersion` 1, 2 and 3. Schema 1 describes exactly what it always did: 32-bit scalars and
//! structs of them, in structured buffers. Schema 2 adds, beside those, the type kinds `vector`
//! (`scalarType`, `elementCount`) and `matrix` (`scalarType`, `rows`, `columns`, `matrixLayout`; its `stride` is the
//! column stride), the parameter type `constantBuffer` (`elementType`, `size`), and a mandatory top-level
//! `matrixLayout` that must equal `MATRIX_LAYOUT` (column-major). Every type carries `size`, `alignment` and
//! `stride`. A schema-1 document checked against a descriptor with a vector, a matrix or a uniform is refused with
//! the instruction to re-export it ([`Reflection::cross_check_kernel`]); nothing committed before schema 2 needs
//! re-exporting otherwise.
//!
//! Schema 3 (mx.t0b, decision D91) is schema 2's grammar (its `matrixLayout` included) plus, at the top level, a
//! `stage` (`compute` or `fragment`) and, as parameters, the type kinds `texture` (`shape`, `access`, `result`,
//! `format`) and `sampler`; a fragment document has `targets` (`index`, `type`, in `SV_Target` order) and no
//! `workgroupSize`. The helper writes it only when a 2-D texture, a sampler or a non-compute stage is present, so
//! every document of a compute kernel over buffers stays schema 1 or 2, byte for byte, and the reader reads those
//! exactly as before (a texture in a schema-1 or schema-2 document is refused with the old message). A document of
//! schema 1 or 2 describes a compute entry point: [`Reflection::stage`] is [`Stage::Compute`] for it.
//!
//! # Where the CLI gate would sit
//!
//! The plan's section 3.3 had a second witness for the facts the helper could not see: slangc's
//! `-reflection-json`, run on the same compilation, parsed into `CliFacts` and stored in the record. X-S2 ended in
//! outcome A (mx.t0a): the helper reports the texture format, the access and a fragment's targets itself, so the
//! helper's facts are the D17 witness and that gate is **not built**. `reflect` (the CLI path) stays what it was: a
//! compute-only partial-evidence path for tests, and an optional agreement check against the helper is later work.
//! Were the helper to stop seeing a fact, the gate belongs next to `CompiledReflection::from_output`, with its
//! comparison in [`Reflection::cross_check_kernel`] beside the texture checks.
//!
//! # The `slang-runtime` feature
//!
//! The functions that launch a compiler process (`reflect`, `compile_reflected`, `compile_reflected_with_helper`,
//! `native_compiler_path`) exist only with the `slang-runtime` feature, which is off by default (D44). Parsing
//! and checking reflection JSON (`Reflection::from_json`, `cross_check_*`, `CompiledReflection::from_output`) needs
//! no feature, so [`crate::artifact::verify_artifact`] works without a compiler.

use crate::{
    GpuPod, KernelDescriptor, ScalarKind, ShaderLayoutRules, Shape, TypeLayout, TypeLayoutKind,
    abi::{MATRIX_LAYOUT, MatrixLayout, TextureAccess, TextureDimension},
    slang::{self, Target},
    texture::{Format as TexelFormat, SampleClass},
};
#[cfg(feature = "slang-runtime")]
use crate::{SourceOrigin, slang::TemporaryDirectory};
use std::{collections::BTreeMap, error, fmt, path::PathBuf};
#[cfg(feature = "slang-runtime")]
use std::{fs, path::Path, process::Command};

#[derive(Debug)]
pub enum Error {
    Compiler(slang::Error),
    HelperUnavailable {
        helper: PathBuf,
        source: std::io::Error,
    },
    Invalid(String),
    Mismatch(String),
    Incomplete(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(e) => e.fmt(f),
            Self::HelperUnavailable { helper, source } => write!(
                f,
                "could not launch Slang reflection compiler {}: {source}; run `cargo xtask build-slang-reflect` or set GUST_SLANG_REFLECT to the helper executable",
                helper.display()
            ),
            Self::Invalid(s) => write!(f, "invalid Slang reflection: {s}"),
            Self::Mismatch(s) => write!(f, "POD reflection mismatch: {s}"),
            Self::Incomplete(s) => write!(f, "incomplete Slang layout evidence: {s}"),
        }
    }
}
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Compiler(e) => Some(e),
            Self::HelperUnavailable { source, .. } => Some(source),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Compiler(slang::Error::Io(e))
    }
}
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reflection {
    pub target: Target,
    /// The `schemaVersion` the document was read under: 1 for the CLI JSON path and for schema-1 native documents,
    /// 2 for schema-2 native documents, 3 for schema-3 native documents.
    pub schema_version: u32,
    /// The pipeline stage of the entry point: the document's `stage` under schema 3, [`Stage::Compute`] under 1 and
    /// 2 (which describe compute entry points only).
    pub stage: Stage,
    /// The structured and constant buffers, in document order. Textures and samplers are not in this list.
    pub resources: Vec<ReflectedResource>,
    /// Schema 3 only: the 2-D texture bindings, in document order.
    pub textures: Vec<ReflectedTexture>,
    /// Schema 3 only: the sampler bindings, in document order.
    pub samplers: Vec<ReflectedSampler>,
    /// Schema 3, fragment only: the colour outputs in `SV_Target` order (ascending, no repeated index).
    pub targets: Vec<ReflectedTarget>,
}
/// The pipeline stage of a reflected entry point (decision D89: it is the stage of the kernel's [`Shape`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `[shader("compute")]`: the stage of [`Shape::Grid`].
    Compute,
    /// `[shader("fragment")]`: the stage of [`Shape::Pixel`].
    Fragment,
}
impl Stage {
    /// The stage a kernel of `shape` has.
    pub const fn of(shape: Shape) -> Self {
        match shape {
            Shape::Grid => Self::Compute,
            Shape::Pixel => Self::Fragment,
        }
    }

    /// The spelling in the reflection JSON (`stage`) and on `slangc -stage`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Fragment => "fragment",
        }
    }
}
/// A 2-D texture binding of a schema-3 document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedTexture {
    pub name: String,
    pub binding: u32,
    pub space: u32,
    pub dimension: TextureDimension,
    pub access: TextureAccess,
    /// The result type: a 32-bit scalar or a 2- to 4-wide vector of one. The helper reports no byte layout for it
    /// (`size`, `alignment` and `stride` are `None`).
    pub result: ReflectedType,
    /// The texel format of a storage or write-only texture. `None` is the document's `unknown`: a sampled texture
    /// has no format in the shader (its concrete format is checked at submit against the bound texture).
    pub format: Option<TexelFormat>,
}
/// A sampler binding of a schema-3 document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedSampler {
    pub name: String,
    pub binding: u32,
    pub space: u32,
}
/// One colour output of a fragment entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedTarget {
    /// The `SV_Target` index.
    pub index: u32,
    /// A 32-bit scalar or a 2- to 4-wide vector of one.
    pub ty: ReflectedType,
}
/// What kind of shader resource a [`ReflectedResource`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectedResourceKind {
    StructuredBuffer,
    /// Schema 2 only. Always read-only, with no element count.
    ConstantBuffer,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedResource {
    pub name: String,
    pub kind: ReflectedResourceKind,
    pub access: crate::Access,
    pub binding: u32,
    pub space: u32,
    pub element: ReflectedType,
    /// None means the compiler JSON did not report a buffer element stride.
    pub element_stride: Option<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedType {
    pub name: Option<String>,
    pub size: Option<u32>,
    pub alignment: Option<u32>,
    pub stride: Option<u32>,
    pub kind: ReflectedTypeKind,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReflectedTypeKind {
    Scalar(ScalarKind),
    /// Schema 2 only.
    Vector {
        scalar: ScalarKind,
        count: u32,
    },
    /// Schema 2 only. `ReflectedType::stride` of a matrix is its column stride.
    Matrix {
        scalar: ScalarKind,
        rows: u32,
        cols: u32,
        layout: MatrixLayout,
    },
    Struct(Vec<ReflectedField>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedField {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub ty: ReflectedType,
}
/// Coverage includes every nested type, not just the outer element.
#[must_use = "Inspect coverage: a successful cross-check may still lack full ABI evidence"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PodCrossCheck {
    pub checked_fields: usize,
    pub aggregate_size_verified: bool,
    pub element_stride_verified: bool,
    pub alignment_verified: bool,
}
impl PodCrossCheck {
    pub fn is_full_abi_match(&self) -> bool {
        self.aggregate_size_verified && self.element_stride_verified && self.alignment_verified
    }
}

#[derive(Debug)]
pub struct CompiledReflection {
    pub artifact: Vec<u8>,
    pub reflection: Reflection,
    pub compiler_version: String,
    pub entry_point: String,
    /// The compute workgroup size. `[0, 0, 0]` for a fragment entry point, which has none (mx.t0b).
    pub workgroup_size: [u32; 3],
}

/// The native helper that [`compile_reflected`] launches, by the rules of `crate::toolchain` (D109):
/// `GUST_SLANG_REFLECT`, else the managed helper `<root>/helper/<pin>/gust-slang-reflect` of the Slang cache, else the
/// repository checkout's `target/slang-reflect` (only when this crate is built from a GUST checkout), else the bare
/// executable name, which the launch looks up on `PATH`. There is no current-directory lookup.
///
/// This accessor never fails: when the variables are unusable (a relative or empty `GUST_SLANG_REFLECT`, a bad
/// `GUST_SLANG_HOME`) it returns `GUST_SLANG_REFLECT` as given, or the bare name, and [`compile_reflected`] is the
/// call that reports the problem.
#[cfg(feature = "slang-runtime")]
pub fn native_compiler_path() -> PathBuf {
    crate::toolchain::helper_candidate_in(&crate::toolchain::Env::from_process())
}

#[cfg(feature = "slang-runtime")]
/// Compile and reflect the same linked program using the native Slang helper ([`native_compiler_path`]). An unusable
/// `GUST_SLANG_REFLECT` or `GUST_SLANG_HOME` is an error here, with the variable and its value in the text.
pub fn compile_reflected(
    kernel: &KernelDescriptor,
    target: Target,
) -> Result<CompiledReflection, Error> {
    let helper = crate::toolchain::helper_path()
        .map_err(|reason| Error::Compiler(slang::Error::Toolchain(reason)))?
        .unwrap_or_else(crate::toolchain::bare_helper_name);
    compile_reflected_with_helper(kernel, target, helper)
}

#[cfg(feature = "slang-runtime")]
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
fn helper_command(
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
fn outdated_helper_note(output: &std::process::Output) -> &'static str {
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
#[derive(Debug)]
pub struct CleanCompilation {
    /// The artifact and layouts, checked by `from_output` against the clean text.
    pub compiled: CompiledReflection,
    /// The helper's reflection JSON, byte for byte, as an export stores it beside the artifact.
    pub reflection_json: String,
}

/// Compile and reflect the kernel's *clean* Slang text: `KernelDescriptor::slang_source` exactly, without the `#line`
/// directives of [`slang::compile_copy`]. The helper's `sourceHash` is then the hash that
/// [`crate::artifact::verify_artifact`] recomputes at run time, so the result can be stored as a precompiled artifact
/// (D45).
///
/// Only a failed compile touches the directive copy: the helper compiles it once more, so that the diagnostics name
/// Rust lines as `compile_reflected_with_helper` does.
#[cfg(feature = "slang-runtime")]
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
fn compile_clean_with(
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

fn target_name(target: Target) -> &'static str {
    match target {
        Target::Wgsl => "wgsl",
        Target::Spirv => "spirv",
    }
}

fn fingerprint(bytes: &[u8]) -> String {
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
fn slangc_reflection_command(
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
/// Compile exactly this target and entry point, requesting Slang's JSON metadata.
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

impl Reflection {
    /// Parse global structured buffers and the generated dispatch-ID parameter.
    /// Entry-point resources and ordinary value/uniform parameters are unsupported.
    pub fn from_json(target: Target, source: &str) -> Result<Self, Error> {
        let root = Parser::parse(source)?;
        Self::read_json(target, &root, false, 1)
    }

    /// `schema` is the document's `schemaVersion` (1, 2 or 3): under 1 exactly the pre-schema-2 grammar is read, and
    /// the texture, sampler, `stage` and `targets` members exist from 3 on only.
    fn read_json(target: Target, root: &Json, complete: bool, schema: u32) -> Result<Self, Error> {
        let stage = read_stage(root, schema)?;
        if schema >= 2 {
            let text = root.get("matrixLayout")?.string()?;
            match MatrixLayout::parse(text) {
                Some(layout) if layout == MATRIX_LAYOUT => {}
                Some(layout) => {
                    return Err(Error::Mismatch(format!(
                        "matrixLayout: host {}, compiler {}",
                        MATRIX_LAYOUT.as_str(),
                        layout.as_str()
                    )));
                }
                None => return Err(invalid("unknown matrixLayout")),
            }
        }
        if let Some(entries) = root.optional("entryPoints")? {
            for entry in entries.array()? {
                if let Some(parameters) = entry.optional("parameters")? {
                    for parameter in parameters.array()? {
                        if parameter.optional("binding")?.is_some()
                            || parameter.optional("bindings")?.is_some()
                        {
                            return Err(invalid(
                                "entry-point resource/uniform bindings are unsupported",
                            ));
                        }
                        let ty = parameter.get("type")?;
                        if parameter
                            .optional("semanticName")?
                            .map(Json::string)
                            .transpose()?
                            != Some("SV_DISPATCHTHREADID")
                            || ty.get("kind")?.string()? != "vector"
                            || ty.get("elementCount")?.u32()? != 3
                            || ty.get("elementType")?.get("kind")?.string()? != "scalar"
                            || ty.get("elementType")?.get("scalarType")?.string()? != "uint32"
                        {
                            return Err(invalid(
                                "only dispatch-thread-ID entry-point parameters are supported",
                            ));
                        }
                    }
                }
            }
        }
        let mut resources: Vec<ReflectedResource> = Vec::new();
        let mut textures: Vec<ReflectedTexture> = Vec::new();
        let mut samplers: Vec<ReflectedSampler> = Vec::new();
        for value in root.get("parameters")?.array()? {
            let name = value.get("name")?.string()?.to_owned();
            let binding = value.get("binding")?;
            if binding.get("kind")?.string()? != "descriptorTableSlot" {
                return Err(invalid("unsupported resource binding kind"));
            }
            let index = binding.get("index")?.u32()?;
            let space = if complete {
                binding.get("space")?.u32()?
            } else {
                binding
                    .optional("space")?
                    .map(Json::u32)
                    .transpose()?
                    .unwrap_or(0)
            };
            let ty = value.get("type")?;
            // Schema 3 only: a texture or a sampler. Schemas 1 and 2 never evaluate this and read exactly what they
            // always did (a texture there is refused as "only structured buffers are supported").
            if schema >= 3 && matches!(ty.get("kind")?.string()?, "texture" | "sampler") {
                if complete && binding.get("count")?.u32()? != 1 {
                    return Err(invalid("binding arrays are unsupported"));
                }
                if let Some(count) = binding.optional("count")?
                    && count.u32()? != 1
                {
                    return Err(invalid("binding arrays are unsupported"));
                }
                if slot_taken(&resources, &textures, &samplers, &name, index, space) {
                    return Err(invalid("duplicate resource name or binding"));
                }
                if ty.get("kind")?.string()? == "sampler" {
                    samplers.push(ReflectedSampler {
                        name,
                        binding: index,
                        space,
                    });
                } else {
                    textures.push(read_texture(name, index, space, ty)?);
                }
                continue;
            }
            // Schema 2 only: a constant buffer. Schema 1 never evaluates this and reads exactly what it always did.
            let constant = schema >= 2 && ty.get("kind")?.string()? == "constantBuffer";
            if !constant
                && (ty.get("kind")?.string()? != "resource"
                    || ty.get("baseShape")?.string()? != "structuredBuffer")
            {
                return Err(invalid("only structured buffers are supported"));
            }
            let access = if constant {
                crate::Access::ReadOnly
            } else {
                match ty.optional("access")?.map(Json::string).transpose()? {
                    Some("read") => crate::Access::ReadOnly,
                    None if !complete => crate::Access::ReadOnly,
                    Some("readWrite") => crate::Access::ReadWrite,
                    _ => return Err(invalid("unsupported resource access")),
                }
            };
            if complete && binding.get("count")?.u32()? != 1 {
                return Err(invalid("binding arrays are unsupported"));
            }
            if let Some(count) = binding.optional("count")?
                && count.u32()? != 1
            {
                return Err(invalid("binding arrays are unsupported"));
            }
            if slot_taken(&resources, &textures, &samplers, &name, index, space) {
                return Err(invalid("duplicate resource name or binding"));
            }
            let (kind, element, element_stride) = if constant {
                // A constant buffer has no array stride: `size` is its element's size, rounded up by the layout rule.
                (
                    ReflectedResourceKind::ConstantBuffer,
                    read_type(ty.get("elementType")?, complete, schema)?,
                    if complete {
                        Some(ty.get("size")?.u32()?)
                    } else {
                        None
                    },
                )
            } else {
                (
                    ReflectedResourceKind::StructuredBuffer,
                    read_type(ty.get("resultType")?, complete, schema)?,
                    if complete {
                        Some(value.get("elementStride")?.u32()?)
                    } else {
                        None
                    },
                )
            };
            resources.push(ReflectedResource {
                name,
                kind,
                access,
                binding: index,
                space,
                element,
                element_stride,
            });
        }
        let targets = if schema >= 3 {
            read_targets(root, stage)?
        } else {
            Vec::new()
        };
        Ok(Self {
            target,
            schema_version: schema,
            stage,
            resources,
            textures,
            samplers,
            targets,
        })
    }

    /// Require complete layout evidence for every descriptor binding, with no extras: `StorageV1` and `StorageV2`
    /// storage buffers, and `Std140V1` uniforms (constant buffers, schema 2 only).
    ///
    /// A schema-1 document cannot describe a vector, a matrix or a uniform, so a descriptor that has one is refused
    /// with `reflection schema 1 cannot describe <kind>; re-export with cargo gust build`.
    pub fn cross_check_kernel(
        &self,
        kernel: &KernelDescriptor,
    ) -> Result<Vec<PodCrossCheck>, Error> {
        for (index, resource) in self.resources.iter().enumerate() {
            if self.resources[..index].iter().any(|previous| {
                previous.name == resource.name
                    || (previous.space, previous.binding) == (resource.space, resource.binding)
            }) {
                return Err(invalid("duplicate resource name or binding"));
            }
        }
        // The stage first: a compute reflection checked against a pixel descriptor (or the reverse) has nothing else
        // worth comparing (D91).
        check_stage(kernel.shape, self.stage)?;
        let mut reports = Vec::new();
        let mut bindings = BTreeMap::new();
        let (mut textures, mut samplers) = (0_usize, 0_usize);
        let mut outputs: Vec<(&'static str, TypeLayout)> = Vec::new();
        for (index, parameter) in kernel.parameters.iter().enumerate() {
            if kernel.parameters[..index]
                .iter()
                .any(|previous| previous.name == parameter.name)
            {
                return Err(Error::Mismatch(
                    "duplicate descriptor parameter name".into(),
                ));
            }
            if parameter.kind == crate::ParameterKind::Invocation {
                if parameter.binding.is_some()
                    || parameter.resource_layout.is_some()
                    || parameter.access != crate::Access::NotApplicable
                {
                    return Err(Error::Mismatch(
                        "invocation parameter has a resource contract".into(),
                    ));
                }
                continue;
            }
            if self.schema_version < 3
                && matches!(
                    parameter.kind,
                    crate::ParameterKind::Texture | crate::ParameterKind::Sampler
                )
            {
                return Err(Error::Mismatch(format!(
                    "reflection schema {} cannot describe {}; re-export with cargo gust build",
                    self.schema_version,
                    if parameter.kind == crate::ParameterKind::Texture {
                        "texture"
                    } else {
                        "sampler"
                    }
                )));
            }
            if self.schema_version < 2
                && let Some(what) = schema_one_gap(parameter)
            {
                return Err(Error::Mismatch(format!(
                    "reflection schema 1 cannot describe {what}; re-export with cargo gust build"
                )));
            }
            // Under `pixel` a render-target output (no binding; see `is_pixel_output`) is a colour target.
            if kernel.shape == Shape::Pixel && is_pixel_output(parameter) {
                let layout = parameter.resource_layout.ok_or_else(|| {
                    Error::Mismatch(format!(
                        "{}: missing descriptor element layout",
                        parameter.name
                    ))
                })?;
                outputs.push((parameter.name, layout.element));
                continue;
            }
            if matches!(
                parameter.kind,
                crate::ParameterKind::Texture | crate::ParameterKind::Sampler
            ) {
                let binding = parameter
                    .binding
                    .ok_or_else(|| Error::Mismatch("missing descriptor binding".into()))?;
                if bindings
                    .insert((binding.group, binding.binding), parameter.name)
                    .is_some()
                {
                    return Err(Error::Mismatch("duplicate descriptor binding".into()));
                }
                if parameter.kind == crate::ParameterKind::Texture {
                    self.check_texture(parameter, binding)?;
                    textures += 1;
                } else {
                    self.check_sampler(parameter, binding)?;
                    samplers += 1;
                }
                continue;
            }
            let expected_kind = match parameter.kind {
                crate::ParameterKind::Storage => ReflectedResourceKind::StructuredBuffer,
                crate::ParameterKind::Uniform => ReflectedResourceKind::ConstantBuffer,
                _ => {
                    return Err(Error::Mismatch(
                        "only storage and uniform descriptor resources are supported".into(),
                    ));
                }
            };
            let binding = parameter
                .binding
                .ok_or_else(|| Error::Mismatch("missing descriptor binding".into()))?;
            let layout = parameter
                .resource_layout
                .ok_or_else(|| Error::Mismatch("missing descriptor element layout".into()))?;
            if bindings
                .insert((binding.group, binding.binding), parameter.name)
                .is_some()
            {
                return Err(Error::Mismatch("duplicate descriptor binding".into()));
            }
            let rules_fit = matches!(
                (expected_kind, layout.rules),
                (
                    ReflectedResourceKind::StructuredBuffer,
                    ShaderLayoutRules::StorageV1 | ShaderLayoutRules::StorageV2,
                ) | (
                    ReflectedResourceKind::ConstantBuffer,
                    ShaderLayoutRules::Std140V1
                )
            );
            if !rules_fit {
                return Err(Error::Mismatch(format!(
                    "{}.rules: {:?} does not fit this parameter kind",
                    parameter.name, layout.rules
                )));
            }
            // A constant buffer has no array stride: its stride slot holds the element size.
            let expected_stride = if layout.rules == ShaderLayoutRules::Std140V1 {
                layout.element.size
            } else {
                layout.element.stride
            };
            if layout.element_stride != expected_stride {
                return Err(Error::Mismatch(format!(
                    "{}.stride: inconsistent host resource layout",
                    parameter.name
                )));
            }
            let resource = self
                .resources
                .iter()
                .find(|resource| resource.name == parameter.name)
                .ok_or_else(|| {
                    Error::Mismatch(format!("missing compiler binding {}", parameter.name))
                })?;
            if resource.kind != expected_kind {
                return Err(Error::Mismatch(format!(
                    "{}.kind: host {expected_kind:?}, compiler {:?}",
                    parameter.name, resource.kind
                )));
            }
            equal(
                binding.group,
                resource.space,
                &format!("{}.group", parameter.name),
            )?;
            equal(
                binding.binding,
                resource.binding,
                &format!("{}.binding", parameter.name),
            )?;
            if !matches!(
                parameter.access,
                crate::Access::ReadOnly | crate::Access::ReadWrite
            ) || parameter.access != resource.access
            {
                return Err(Error::Mismatch(format!(
                    "{}.access differs",
                    parameter.name
                )));
            }
            let report = cross_check_layout(layout.element, layout.rules, resource)?;
            if !report.is_full_abi_match() {
                return Err(Error::Incomplete(format!(
                    "{} requires all nested sizes, alignments and strides",
                    parameter.name
                )));
            }
            reports.push(report);
        }
        if reports.len() != self.resources.len()
            || textures != self.textures.len()
            || samplers != self.samplers.len()
        {
            return Err(Error::Mismatch(
                "unexpected compiler resource bindings".into(),
            ));
        }
        self.check_targets(&outputs)?;
        Ok(reports)
    }

    /// A texture parameter against the document: the descriptor's own consistency first (the layout is the texture
    /// rule, the class and format agree with the result type, the access with the parameter's), then slot, dimension
    /// and access. A **sampled** texture is held to the sample class and the width of its result type and to having no
    /// format (its concrete format is checked at submit against the bound texture); a **storage** or **write-only**
    /// texture to its exact result type and exact format, which must have the result's channel count. Each mismatch
    /// names `<parameter>.<field>`.
    fn check_texture(
        &self,
        parameter: &crate::ParameterDescriptor,
        binding: crate::ResourceBinding,
    ) -> Result<(), Error> {
        let name = parameter.name;
        let layout = parameter
            .resource_layout
            .ok_or_else(|| Error::Mismatch(format!("{name}: missing descriptor element layout")))?;
        let ShaderLayoutRules::TextureV1(texture) = layout.rules else {
            return Err(Error::Mismatch(format!(
                "{name}.rules: {:?} does not fit this parameter kind",
                layout.rules
            )));
        };
        // The descriptor's own coherence: these are host-side facts, so a disagreement is a malformed descriptor.
        let Some((scalar, count)) = result_shape(layout.element).filter(|(_, count)| *count != 3)
        else {
            return Err(Error::Mismatch(format!(
                "{name}.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector"
            )));
        };
        if layout.element_stride != layout.element.stride {
            return Err(Error::Mismatch(format!(
                "{name}.stride: inconsistent host resource layout"
            )));
        }
        if texture.class != class_of(scalar) {
            return Err(Error::Mismatch(format!(
                "{name}.class: host {}, result type {}",
                texture.class.name(),
                class_of(scalar).name()
            )));
        }
        let access_fits = match texture.access {
            TextureAccess::Sampled => parameter.access == crate::Access::ReadOnly,
            TextureAccess::ReadWrite | TextureAccess::WriteOnly => {
                parameter.access == crate::Access::ReadWrite
            }
        };
        if !access_fits {
            return Err(Error::Mismatch(format!("{name}.access differs")));
        }
        match (texture.access, texture.format) {
            (TextureAccess::Sampled, None) => {}
            (TextureAccess::Sampled, Some(format)) => {
                return Err(Error::Mismatch(format!(
                    "{name}.format: a sampled texture has no format in the shader, host {format}"
                )));
            }
            (_, None) => {
                return Err(Error::Mismatch(format!(
                    "{name}.format: a storage texture needs a format"
                )));
            }
            (access, Some(format)) => {
                if format.sample_class() != texture.class
                    || (access == TextureAccess::ReadWrite && !format.read_write())
                {
                    return Err(Error::Mismatch(format!(
                        "{name}.format: {format} does not fit class {} and access {}",
                        texture.class.name(),
                        access.name()
                    )));
                }
                if format.channels() != count {
                    return Err(Error::Mismatch(format!(
                        "{name}.format: {format} has channel count {}, the result type has width {count}",
                        format.channels()
                    )));
                }
            }
        }
        let reflected = self
            .textures
            .iter()
            .find(|candidate| candidate.name == name);
        let Some(reflected) = reflected else {
            return Err(self.missing(name, "texture"));
        };
        equal(binding.group, reflected.space, &format!("{name}.group"))?;
        equal(
            binding.binding,
            reflected.binding,
            &format!("{name}.binding"),
        )?;
        if texture.dimension != reflected.dimension {
            return Err(Error::Mismatch(format!(
                "{name}.dimension: host {}, compiler {}",
                texture.dimension.name(),
                reflected.dimension.name()
            )));
        }
        if texture.access != reflected.access {
            return Err(Error::Mismatch(format!(
                "{name}.access: host {}, compiler {}",
                texture.access.name(),
                reflected.access.name()
            )));
        }
        let Some((reflected_scalar, reflected_count)) = reflected_shape(&reflected.result.kind)
        else {
            return Err(Error::Mismatch(format!(
                "{name}.result: the compiler's result type is not a scalar or a vector"
            )));
        };
        if texture.access == TextureAccess::Sampled && texture.class != class_of(reflected_scalar) {
            return Err(Error::Mismatch(format!(
                "{name}.class: host {}, compiler {}",
                texture.class.name(),
                class_of(reflected_scalar).name()
            )));
        }
        // The result type for every texture (D91): a sampled one differs from its twin in width as well.
        if (scalar, count) != (reflected_scalar, reflected_count) {
            return Err(Error::Mismatch(format!(
                "{name}.result: host {}, compiler {}",
                shape_text(scalar, count),
                shape_text(reflected_scalar, reflected_count)
            )));
        }
        if texture.format != reflected.format {
            return Err(Error::Mismatch(format!(
                "{name}.format: host {}, compiler {}",
                format_text(texture.format),
                format_text(reflected.format)
            )));
        }
        Ok(())
    }

    /// A sampler parameter against the document: only its slot is compared (the options are bound at submit).
    fn check_sampler(
        &self,
        parameter: &crate::ParameterDescriptor,
        binding: crate::ResourceBinding,
    ) -> Result<(), Error> {
        let name = parameter.name;
        if parameter.resource_layout.is_some() || parameter.access != crate::Access::NotApplicable {
            return Err(Error::Mismatch(format!(
                "{name}: a sampler has no resource contract"
            )));
        }
        let Some(reflected) = self.samplers.iter().find(|sampler| sampler.name == name) else {
            return Err(self.missing(name, "sampler"));
        };
        equal(binding.group, reflected.space, &format!("{name}.group"))?;
        equal(
            binding.binding,
            reflected.binding,
            &format!("{name}.binding"),
        )
    }

    /// The error for a descriptor texture or sampler the document has under that name in no list of its own kind: a
    /// buffer or the other kind of binding under that name is a kind mismatch, nothing at all a missing binding.
    fn missing(&self, name: &str, host: &str) -> Error {
        let other = if self.resources.iter().any(|r| r.name == name) {
            Some("a buffer")
        } else if self.textures.iter().any(|t| t.name == name) {
            Some("a texture")
        } else if self.samplers.iter().any(|s| s.name == name) {
            Some("a sampler")
        } else {
            None
        };
        match other {
            Some(other) => Error::Mismatch(format!("{name}.kind: host {host}, compiler {other}")),
            None => Error::Mismatch(format!("missing compiler binding {name}")),
        }
    }

    /// The colour outputs of a pixel descriptor against the document's `targets`: the same count, the output `k`
    /// (in parameter order) at `SV_Target` index `k`, and the same scalar kind and width.
    fn check_targets(&self, outputs: &[(&'static str, TypeLayout)]) -> Result<(), Error> {
        if self.targets.len() != outputs.len() {
            return Err(Error::Mismatch(format!(
                "targets: host {} outputs, compiler {}",
                outputs.len(),
                self.targets.len()
            )));
        }
        for (position, ((name, element), target)) in outputs.iter().zip(&self.targets).enumerate() {
            let host_index = u32::try_from(position).unwrap_or(u32::MAX);
            equal(
                host_index,
                target.index,
                &format!("targets[{position}].index"),
            )?;
            let Some((scalar, count)) = result_shape(*element) else {
                return Err(Error::Mismatch(format!(
                    "{name}: an output is a 32-bit scalar or a 2- to 4-wide vector"
                )));
            };
            let Some((reflected_scalar, reflected_count)) = reflected_shape(&target.ty.kind) else {
                return Err(Error::Mismatch(format!(
                    "targets[{position}].type: the compiler's type is not a scalar or a vector"
                )));
            };
            if (scalar, count) != (reflected_scalar, reflected_count) {
                return Err(Error::Mismatch(format!(
                    "targets[{position}].type: host {}, compiler {}",
                    shape_text(scalar, count),
                    shape_text(reflected_scalar, reflected_count)
                )));
            }
        }
        Ok(())
    }
}

/// Whether a pixel descriptor's parameter is a render-target output: a storage parameter without a shader binding.
/// Its `SV_Target` index is its position among such parameters.
///
/// This is the **render-target model** (D93: an output is a colour target, not a resource), which D120 keeps only for
/// targets the host provides itself (Breeze). MX pixel outputs under D120 are fragment *storage writes*: their
/// parameters carry a `binding`, this function is false for them, and they are cross-checked as buffers, so the
/// document's `targets` must then be empty: the fragment of a `pixel` kernel returns nothing, and the never-written
/// `r8unorm` attachment of the pass is a host object the shader does not name (mx.p1a). Under D120 no kernel the macro
/// writes reaches the branch below, so this function is dead for MX until Breeze; the host (mx.p1b) lists it. The rule
/// is the descriptor-side one until the descriptor names a target explicitly (plan section 4.1's
/// `target: Some(k)`): then this function reads that field instead and nothing else changes. It does not look at
/// `access`: a read-only input that lost its binding counts as an output, and fails as a `targets` count mismatch.
fn is_pixel_output(parameter: &crate::ParameterDescriptor) -> bool {
    parameter.kind == crate::ParameterKind::Storage && parameter.binding.is_none()
}

/// The stage of the document against the kernel's shape. `Mismatch` naming `stage`.
fn check_stage(shape: Shape, stage: Stage) -> Result<(), Error> {
    let host = Stage::of(shape);
    if host == stage {
        Ok(())
    } else {
        Err(Error::Mismatch(format!(
            "stage: host {}, compiler {}",
            host.name(),
            stage.name()
        )))
    }
}

/// The document's `stage`: the member from schema 3 on, compute for the schemas that predate it.
fn read_stage(root: &Json, schema: u32) -> Result<Stage, Error> {
    if schema < 3 {
        return Ok(Stage::Compute);
    }
    match root.get("stage")?.string()? {
        "compute" => Ok(Stage::Compute),
        "fragment" => Ok(Stage::Fragment),
        _ => Err(invalid("unknown stage")),
    }
}

/// Whether a resource of the same name or the same `(space, binding)` is already read, in any of the three lists.
fn slot_taken(
    resources: &[ReflectedResource],
    textures: &[ReflectedTexture],
    samplers: &[ReflectedSampler],
    name: &str,
    binding: u32,
    space: u32,
) -> bool {
    resources
        .iter()
        .map(|r| (r.name.as_str(), r.binding, r.space))
        .chain(
            textures
                .iter()
                .map(|t| (t.name.as_str(), t.binding, t.space)),
        )
        .chain(
            samplers
                .iter()
                .map(|s| (s.name.as_str(), s.binding, s.space)),
        )
        .any(|(other, other_binding, other_space)| {
            other == name || (other_binding == binding && other_space == space)
        })
}

/// One `kind: "texture"` parameter of a schema-3 document.
fn read_texture(
    name: String,
    binding: u32,
    space: u32,
    ty: &Json,
) -> Result<ReflectedTexture, Error> {
    let dimension = match ty.get("shape")?.string()? {
        "2d" => TextureDimension::D2,
        _ => return Err(invalid("unsupported texture shape (only 2d)")),
    };
    let access = match ty.get("access")?.string()? {
        "read" => TextureAccess::Sampled,
        "readWrite" => TextureAccess::ReadWrite,
        "write" => TextureAccess::WriteOnly,
        _ => return Err(invalid("unsupported texture access")),
    };
    let result = read_type(ty.get("result")?, false, 3)?;
    if !matches!(
        result.kind,
        ReflectedTypeKind::Scalar(_) | ReflectedTypeKind::Vector { .. }
    ) {
        return Err(invalid(
            "unsupported texture result type (a scalar or a vector)",
        ));
    }
    let format = match ty.get("format")?.string()? {
        "unknown" => None,
        text => Some(
            TexelFormat::ALL
                .into_iter()
                .find(|format| format.slang_name() == text)
                .ok_or_else(|| invalid(format!("unsupported texture format `{text}`")))?,
        ),
    };
    Ok(ReflectedTexture {
        name,
        binding,
        space,
        dimension,
        access,
        result,
        format,
    })
}

/// The `targets` of a schema-3 document: present with a fragment stage (possibly empty), absent with a compute one.
fn read_targets(root: &Json, stage: Stage) -> Result<Vec<ReflectedTarget>, Error> {
    let Some(targets) = root.optional("targets")? else {
        return if stage == Stage::Fragment {
            Err(invalid("a fragment reflection needs targets"))
        } else {
            Ok(Vec::new())
        };
    };
    if stage == Stage::Compute {
        return Err(invalid("a compute reflection has no targets"));
    }
    let mut read: Vec<ReflectedTarget> = Vec::new();
    for target in targets.array()? {
        let index = target.get("index")?.u32()?;
        if read.last().is_some_and(|previous| previous.index >= index) {
            return Err(invalid(
                "targets must be in ascending SV_Target order without repeats",
            ));
        }
        let ty = read_type(target.get("type")?, false, 3)?;
        if !matches!(
            ty.kind,
            ReflectedTypeKind::Scalar(_) | ReflectedTypeKind::Vector { .. }
        ) {
            return Err(invalid("unsupported target type (a scalar or a vector)"));
        }
        read.push(ReflectedTarget { index, ty });
    }
    Ok(read)
}

/// The scalar kind and width (1 for a scalar) of a host type layout, if it is a scalar or a vector.
fn result_shape(layout: TypeLayout) -> Option<(ScalarKind, u32)> {
    match layout.kind {
        TypeLayoutKind::Scalar(scalar) => Some((scalar, 1)),
        TypeLayoutKind::Vector { scalar, count } => Some((scalar, count)),
        _ => None,
    }
}

/// The same for a reflected type.
fn reflected_shape(kind: &ReflectedTypeKind) -> Option<(ScalarKind, u32)> {
    match kind {
        ReflectedTypeKind::Scalar(scalar) => Some((*scalar, 1)),
        ReflectedTypeKind::Vector { scalar, count } => Some((*scalar, *count)),
        _ => None,
    }
}

/// The sample class of a result type's scalar.
const fn class_of(scalar: ScalarKind) -> SampleClass {
    match scalar {
        ScalarKind::F32 => SampleClass::Float,
        ScalarKind::U32 => SampleClass::Uint,
        ScalarKind::I32 => SampleClass::Sint,
    }
}

fn shape_text(scalar: ScalarKind, count: u32) -> String {
    let name = scalar_text(scalar);
    if count == 1 {
        name.to_owned()
    } else {
        format!("{name}x{count}")
    }
}

fn scalar_text(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::F32 => "f32",
        ScalarKind::I32 => "i32",
        ScalarKind::U32 => "u32",
    }
}

fn format_text(format: Option<TexelFormat>) -> &'static str {
    format.map_or("none", TexelFormat::name)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "line one\nline two\n";
    const LINES: &[crate::LineSpan] = &[crate::LineSpan {
        slang_start: 2,
        slang_end: 2,
        rust_line: 9,
    }];
    const KERNEL: KernelDescriptor = KernelDescriptor {
        module: "m",
        name: "k",
        entry_point: "entry",
        shape: crate::Shape::Grid,
        compatible_shapes: &[crate::Shape::Grid],
        workgroup_size: [1, 1, 1],
        parameters: &[],
        slang_source: SOURCE,
        source: SourceOrigin {
            file: "src/k.rs",
            lines: LINES,
        },
        artifacts: &[],
    };
    const ARTIFACT: &[u8] = b"artifact bytes";

    /// The CLI reflection path names the matrix layout (the native helper pins the same one).
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_cli_reflection_command_pins_the_column_major_matrix_layout() {
        for (target, spirv_flag) in [
            (Target::Wgsl, None),
            (Target::Spirv, Some("-fvk-use-entrypoint-name")),
        ] {
            for (shape, stage) in [
                (crate::Shape::Grid, "compute"),
                (crate::Shape::Pixel, "fragment"),
            ] {
                let command = slangc_reflection_command(
                    std::path::Path::new("slangc"),
                    std::path::Path::new("kernel.slang"),
                    target,
                    "entry",
                    shape,
                );
                let mut expected = vec![
                    "kernel.slang",
                    "-target",
                    target_name(target),
                    "-entry",
                    "entry",
                    "-stage",
                    stage,
                    "-matrix-layout-column-major",
                ];
                expected.extend(spirv_flag);
                let actual: Vec<String> = command
                    .get_args()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect();
                assert_eq!(actual, expected, "{target:?} {shape:?}");
            }
        }
    }

    /// The helper is told the stage explicitly, as its seventh argument (mx.t0a), whatever the shape: a helper that
    /// defaults to compute and a kernel that is not compute can no longer agree by accident.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_helper_is_given_the_stage_of_the_shape_explicitly() {
        for (shape, stage) in [
            (crate::Shape::Grid, "compute"),
            (crate::Shape::Pixel, "fragment"),
        ] {
            let kernel = KernelDescriptor { shape, ..KERNEL };
            let command = helper_command(
                Path::new("helper"),
                &kernel,
                Target::Spirv,
                Path::new("kernel.slang"),
                Path::new("kernel.artifact"),
                Path::new("reflection.json"),
            );
            let arguments: Vec<String> = command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect();
            assert_eq!(
                arguments,
                [
                    "kernel.slang",
                    "entry",
                    "spirv",
                    "kernel.artifact",
                    "reflection.json",
                    stage
                ],
                "{shape:?}"
            );
        }
    }

    /// A failed helper run that printed `stderr`.
    #[cfg(feature = "slang-runtime")]
    fn failure_output(stderr: &str) -> std::process::Output {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;
        #[cfg(unix)]
        let status = std::process::ExitStatus::from_raw(1 << 8);
        #[cfg(windows)]
        let status = std::process::ExitStatus::from_raw(1);
        std::process::Output {
            status,
            stdout: Vec::new(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    /// A helper from before schema 3 answers the seventh argument with its usage text, which has no stage: the error
    /// says the helper must be rebuilt (mx.k). Any other failure, and the usage text of a current helper, are not
    /// given the note.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn a_helper_that_refuses_the_stage_argument_is_reported_as_outdated() {
        let old = "usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json>, or <layout.slang>, or \
                   --version";
        let current = "usage: gust-slang-reflect <source> <entry> <wgsl|spirv> <artifact> <json> [compute|fragment], \
                       or <layout.slang>, or --version";
        assert!(
            outdated_helper_note(&failure_output(old)).contains("predates reflection schema 3")
        );
        assert_eq!(outdated_helper_note(&failure_output(current)), "");
        assert_eq!(
            outdated_helper_note(&failure_output("error: unsupported stage: x")),
            ""
        );
        let error = compile_clean_with(
            &KERNEL,
            Target::Wgsl,
            Path::new("fake-helper"),
            |_, _, _| Ok(failure_output(old)),
        )
        .unwrap_err();
        match error {
            Error::Compiler(slang::Error::CompilationFailed { diagnostics, .. }) => {
                assert!(
                    diagnostics.contains("predates reflection schema 3")
                        && diagnostics.contains("cargo xtask build-slang-reflect --force"),
                    "{diagnostics}"
                );
            }
            other => panic!("expected the helper's failure, got {other:?}"),
        }
    }

    /// A canned helper result (a real one cannot be forced to lie): only the
    /// identity fields, which `from_output` checks before anything else.
    fn canned(target: &str, entry: &str, source_hash: &str, artifact_hash: &str) -> String {
        format!(
            "{{\"schemaVersion\":1,\"compiler\":\"canned\",\"target\":\"{target}\",\"entryPoint\":\"{entry}\",\"sourceHash\":\"{source_hash}\",\"artifactHash\":\"{artifact_hash}\"}}"
        )
    }

    fn identity_error(compiled_source: &str, json: &str) -> bool {
        let result = CompiledReflection::from_output(
            &KERNEL,
            compiled_source,
            Target::Wgsl,
            ARTIFACT.to_vec(),
            json,
        );
        matches!(result, Err(Error::Invalid(message)) if message.contains("identity mismatch"))
    }

    #[test]
    fn helper_identity_is_checked_field_by_field_against_the_compiled_copy() {
        let copy = slang::compile_copy(SOURCE, &KERNEL.source);
        assert_ne!(copy, SOURCE, "the fixture must have directives");
        let source = fingerprint(copy.as_bytes());
        let artifact = fingerprint(ARTIFACT);
        // The control: every field right gets past the identity check (and fails
        // later, on the missing workgroup size).
        assert!(!identity_error(
            &copy,
            &canned("wgsl", "entry", &source, &artifact)
        ));
        for (what, json) in [
            ("target", canned("spirv", "entry", &source, &artifact)),
            ("entry point", canned("wgsl", "other", &source, &artifact)),
            (
                "stale source hash (the clean text)",
                canned("wgsl", "entry", &fingerprint(SOURCE.as_bytes()), &artifact),
            ),
            (
                "mismatched source hash",
                canned("wgsl", "entry", "0000000000000000", &artifact),
            ),
            (
                "mismatched artifact hash",
                canned("wgsl", "entry", &source, &fingerprint(b"other bytes")),
            ),
        ] {
            assert!(identity_error(&copy, &json), "{what}");
        }
    }

    /// What the real helper writes for a WGSL artifact whose bytes are `text`, computed from the text it was given.
    #[cfg(feature = "slang-runtime")]
    fn fake_helper_json(text: &str, source_hash_of: &str) -> String {
        format!(
            "{{\"schemaVersion\":1,\"compiler\":\"canned\",\"target\":\"wgsl\",\"entryPoint\":\"entry\",\"sourceHash\":\"{}\",\"artifactHash\":\"{}\",\"workgroupSize\":[1,1,1],\"parameters\":[]}}\n",
            fingerprint(source_hash_of.as_bytes()),
            fingerprint(text.as_bytes()),
        )
    }

    #[cfg(feature = "slang-runtime")]
    fn success_output() -> std::process::Output {
        std::process::Output {
            status: std::process::ExitStatus::default(),
            stdout: Vec::new(),
            stderr: Vec::new(),
        }
    }

    /// The clean compile hands the helper `kernel.slang_source` exactly, never the `#line` copy, and checks the
    /// helper's `sourceHash` against that text, so the stored artifact verifies at run time (D45).
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_clean_compile_gives_the_helper_the_clean_text_never_the_directive_copy() {
        let copy = slang::compile_copy(SOURCE, &KERNEL.source);
        assert_ne!(copy, SOURCE, "the fixture must have directives");
        let given = std::cell::RefCell::new(None);
        let written = std::cell::RefCell::new(String::new());
        let result = compile_clean_with(
            &KERNEL,
            Target::Wgsl,
            Path::new("fake-helper"),
            |source, artifact, json| {
                let text = fs::read_to_string(source).unwrap();
                fs::write(artifact, &text).unwrap();
                let report = fake_helper_json(&text, &text);
                fs::write(json, &report).unwrap();
                *written.borrow_mut() = report;
                *given.borrow_mut() = Some(text);
                Ok(success_output())
            },
        )
        .unwrap();
        assert_eq!(given.borrow().as_deref(), Some(SOURCE));
        assert_eq!(result.compiled.artifact, SOURCE.as_bytes());
        assert_eq!(result.reflection_json, *written.borrow());
        assert_eq!(result.compiled.compiler_version, "canned");
        assert!(
            result
                .reflection_json
                .contains(&fingerprint(SOURCE.as_bytes()))
        );
        assert!(
            !result
                .reflection_json
                .contains(&fingerprint(copy.as_bytes()))
        );
    }

    /// A helper that reports the hash of the directive copy is rejected: the identity is the clean text's.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_clean_compile_rejects_a_source_hash_of_the_directive_copy() {
        let copy = slang::compile_copy(SOURCE, &KERNEL.source);
        let error = compile_clean_with(
            &KERNEL,
            Target::Wgsl,
            Path::new("fake-helper"),
            |source, artifact, json| {
                let text = fs::read_to_string(source).unwrap();
                fs::write(artifact, &text).unwrap();
                fs::write(json, fake_helper_json(&text, &copy)).unwrap();
                Ok(success_output())
            },
        )
        .unwrap_err();
        match error {
            Error::Invalid(message) => assert!(message.contains("source hash"), "{message}"),
            other => panic!("expected an identity mismatch, got {other:?}"),
        }
    }

    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_clean_compile_without_a_helper_is_the_helper_unavailable_error() {
        let error = compile_reflected_clean(
            &KERNEL,
            Target::Wgsl,
            "no-such-helper-for-the-clean-compile-test",
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::HelperUnavailable { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn strict_json_syntax_and_unicode() {
        let json =
            Parser::parse(r#"{"name":"a\uD83D\uDE80é\n","unknown":[true,false,null,-1.25e+2]}"#)
                .unwrap();
        assert_eq!(json.get("name").unwrap().string().unwrap(), "a🚀é\n");
        for source in [
            "01",
            "-",
            "1.",
            "1e+",
            "[1,]",
            r#""\uD800""#,
            r#""\uDC00""#,
            r#""\uD800\u0041""#,
            r#""\q""#,
            "\"raw\nline\"",
            "{\"x\":0,\"x\":1}",
            "true false",
        ] {
            assert!(Parser::parse(source).is_err(), "{source}");
        }
        let deep = format!("{}0{}", "[".repeat(66), "]".repeat(66));
        assert!(Parser::parse(&deep).is_err());
        for source in ["4294967296", "-1", "1.0", "1e1"] {
            assert!(Parser::parse(source).unwrap().u32().is_err());
        }
    }

    // ---- schema 2: vectors, matrices and constant buffers (slice m6.a1) ----

    use crate::{
        Access, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout,
        artifact::{Artifact, Format, verify_artifact},
        math::{Mat2, Mat4, Vec2, Vec4},
    };

    const F32_LAYOUT: TypeLayout = <f32 as GpuPod>::LAYOUT;
    const VEC2: TypeLayout = <Vec2 as GpuPod>::LAYOUT;
    const VEC4: TypeLayout = <Vec4 as GpuPod>::LAYOUT;
    const MAT2: TypeLayout = <Mat2 as GpuPod>::LAYOUT;
    const MAT4: TypeLayout = <Mat4 as GpuPod>::LAYOUT;

    const fn member(name: &'static str, offset: u32, ty: TypeLayout) -> crate::StructFieldLayout {
        crate::StructFieldLayout { name, offset, ty }
    }

    /// `TypeLayout::structure` for fields built at run time (its slice must be `'static`).
    fn make_struct(
        name: &'static str,
        size: u32,
        alignment: u32,
        fields: &[crate::StructFieldLayout],
    ) -> TypeLayout {
        TypeLayout::structure(
            name,
            size,
            alignment,
            Box::leak(fields.to_vec().into_boxed_slice()),
        )
    }

    /// `{a, v2, b}` (storage): a0 v8 b16, size 24.
    const STORAGE_STRUCT: TypeLayout = TypeLayout::structure(
        "FV2F",
        24,
        8,
        &[
            member("a", 0, F32_LAYOUT),
            member("v", 8, VEC2),
            member("b", 16, F32_LAYOUT),
        ],
    );
    const E_LAYOUT: TypeLayout = TypeLayout::structure("E", 16, 16, &[member("a", 0, F32_LAYOUT)]);
    /// The review's `cb.json` constant buffer: v16 m32 e96 w112 b120, size 128.
    const CB_STRUCT: TypeLayout = TypeLayout::structure(
        "P",
        128,
        16,
        &[
            member("a", 0, F32_LAYOUT),
            member("v", 16, VEC4),
            member("m", 32, MAT4),
            member("e", 96, E_LAYOUT),
            member("w", 112, VEC2),
            member("b", 120, F32_LAYOUT),
        ],
    );

    fn scalar_name(kind: ScalarKind) -> &'static str {
        match kind {
            ScalarKind::F32 => "float32",
            ScalarKind::I32 => "int32",
            ScalarKind::U32 => "uint32",
        }
    }

    /// The body (without braces) of one type in the schema-2 shape that plan section 2.4 specifies for the helper.
    fn type_body(layout: TypeLayout) -> String {
        let stride = match layout.kind {
            TypeLayoutKind::Matrix { column_stride, .. } => column_stride,
            _ => layout.stride,
        };
        let head = format!(
            r#""size":{},"alignment":{},"stride":{stride},"name":"{}""#,
            layout.size, layout.alignment, layout.name
        );
        let tail = match layout.kind {
            TypeLayoutKind::Scalar(scalar) => {
                format!(r#""kind":"scalar","scalarType":"{}""#, scalar_name(scalar))
            }
            TypeLayoutKind::Vector { scalar, count } => format!(
                r#""kind":"vector","scalarType":"{}","elementCount":{count}"#,
                scalar_name(scalar)
            ),
            TypeLayoutKind::Matrix {
                scalar, rows, cols, ..
            } => format!(
                r#""kind":"matrix","scalarType":"{}","rows":{rows},"columns":{cols},"matrixLayout":"column""#,
                scalar_name(scalar)
            ),
            TypeLayoutKind::Struct(fields) => {
                let fields: Vec<String> = fields
                    .iter()
                    .map(|field| {
                        format!(
                            r#"{{"name":"{}","binding":{{"kind":"uniform","offset":{},"size":{}}},"type":{{{}}}}}"#,
                            field.name,
                            field.offset,
                            field.ty.size,
                            type_body(field.ty)
                        )
                    })
                    .collect();
                format!(r#""kind":"struct","fields":[{}]"#, fields.join(","))
            }
        };
        format!("{head},{tail}")
    }

    fn storage_json(name: &str, index: u32, layout: TypeLayout, access: &str) -> String {
        format!(
            r#"{{"name":"{name}","binding":{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}},"elementStride":{},"type":{{"kind":"resource","baseShape":"structuredBuffer","access":"{access}","resultType":{{{}}}}}}}"#,
            layout.stride,
            type_body(layout)
        )
    }

    fn uniform_json(name: &str, index: u32, layout: TypeLayout) -> String {
        format!(
            r#"{{"name":"{name}","binding":{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}},"type":{{"kind":"constantBuffer","size":{},"elementType":{{{}}}}}}}"#,
            layout.size,
            type_body(layout)
        )
    }

    /// A whole document whose identity fields match `SOURCE` and `ARTIFACT`.
    fn document(schema: u32, matrix_layout: Option<&str>, parameters: &[String]) -> String {
        let source = fingerprint(SOURCE.as_bytes());
        let artifact = fingerprint(ARTIFACT);
        let matrix = matrix_layout
            .map(|layout| format!(r#""matrixLayout":"{layout}","#))
            .unwrap_or_default();
        format!(
            r#"{{"schemaVersion":{schema},{matrix}"compiler":"canned","target":"wgsl","entryPoint":"entry","sourceHash":"{source}","artifactHash":"{artifact}","workgroupSize":[1,1,1],"parameters":[{}]}}"#,
            parameters.join(",")
        )
    }

    fn parameter(
        name: &'static str,
        kind: ParameterKind,
        access: Access,
        layout: ResourceLayout,
        index: u32,
    ) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "T",
            kind,
            access,
            resource_layout: Some(layout),
            binding: Some(ResourceBinding {
                group: 0,
                binding: index,
            }),
        }
    }

    fn kernel_with(parameters: Vec<ParameterDescriptor>) -> KernelDescriptor {
        KernelDescriptor {
            parameters: Box::leak(parameters.into_boxed_slice()),
            ..KERNEL
        }
    }

    fn read(json: &str, kernel: &KernelDescriptor) -> Result<CompiledReflection, Error> {
        CompiledReflection::from_output(kernel, SOURCE, Target::Wgsl, ARTIFACT.to_vec(), json)
    }

    /// Read `json`, which must parse, and return the message of the cross-check's `Mismatch`.
    fn mismatch_of(json: &str, kernel: &KernelDescriptor) -> String {
        let compiled = read(json, kernel).unwrap_or_else(|error| panic!("{json}\n{error}"));
        match compiled.reflection.cross_check_kernel(kernel) {
            Err(Error::Mismatch(message)) => message,
            other => panic!("expected a mismatch, got {other:?}"),
        }
    }

    /// Storage buffers of a struct with a vector and of `Mat4`, and a constant buffer: the three new things at once.
    fn full_kernel() -> KernelDescriptor {
        kernel_with(vec![
            parameter(
                "pts",
                ParameterKind::Storage,
                Access::ReadOnly,
                ResourceLayout::storage_v2(STORAGE_STRUCT),
                0,
            ),
            parameter(
                "mats",
                ParameterKind::Storage,
                Access::ReadWrite,
                ResourceLayout::storage_v2(MAT4),
                1,
            ),
            parameter(
                "cb",
                ParameterKind::Uniform,
                Access::ReadOnly,
                ResourceLayout::std140_v1(CB_STRUCT),
                2,
            ),
        ])
    }

    fn full_parameters() -> Vec<String> {
        vec![
            storage_json("pts", 0, STORAGE_STRUCT, "read"),
            storage_json("mats", 1, MAT4, "readWrite"),
            uniform_json("cb", 2, CB_STRUCT),
        ]
    }

    #[test]
    fn schema_two_reads_vectors_matrices_and_constant_buffers() {
        let kernel = full_kernel();
        let compiled = read(&document(2, Some("column"), &full_parameters()), &kernel).unwrap();
        let reflection = &compiled.reflection;
        assert_eq!(reflection.schema_version, 2);
        let kinds: Vec<_> = reflection.resources.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            [
                ReflectedResourceKind::StructuredBuffer,
                ReflectedResourceKind::StructuredBuffer,
                ReflectedResourceKind::ConstantBuffer,
            ]
        );
        let ReflectedTypeKind::Struct(fields) = &reflection.resources[0].element.kind else {
            panic!("struct expected")
        };
        assert_eq!(
            fields[1].ty.kind,
            ReflectedTypeKind::Vector {
                scalar: ScalarKind::F32,
                count: 2
            }
        );
        assert_eq!(fields[1].ty.size, Some(8));
        let mats = &reflection.resources[1];
        assert_eq!(
            mats.element.kind,
            ReflectedTypeKind::Matrix {
                scalar: ScalarKind::F32,
                rows: 4,
                cols: 4,
                layout: MatrixLayout::Column
            }
        );
        // The matrix `stride` is the column stride; the buffer's element stride is the whole matrix.
        assert_eq!(mats.element.stride, Some(16));
        assert_eq!(mats.element_stride, Some(64));
        let cb = &reflection.resources[2];
        assert_eq!((cb.binding, cb.space, cb.access), (2, 0, Access::ReadOnly));
        assert_eq!(cb.element_stride, Some(128));
        assert_eq!(cb.element.size, Some(128));
        let reports = reflection.cross_check_kernel(&kernel).unwrap();
        assert_eq!(reports.len(), 3);
        assert!(reports.iter().all(PodCrossCheck::is_full_abi_match));
        assert_eq!(reports[0].checked_fields, 3);
        // P has six fields and the nested E has one.
        assert_eq!(reports[2].checked_fields, 7);
    }

    #[test]
    fn schema_two_still_describes_scalar_descriptors() {
        let kernel = kernel_with(vec![parameter(
            "a",
            ParameterKind::Storage,
            Access::ReadOnly,
            ResourceLayout::storage_v1(F32_LAYOUT),
            0,
        )]);
        let json = document(
            2,
            Some("column"),
            &[storage_json("a", 0, F32_LAYOUT, "read")],
        );
        let compiled = read(&json, &kernel).unwrap();
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&kernel)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn matrix_layout_is_mandatory_in_schema_two_and_must_be_the_pin() {
        let kernel = full_kernel();
        let parameters = full_parameters();
        match read(&document(2, None, &parameters), &kernel) {
            Err(Error::Invalid(message)) => assert_eq!(message, "missing matrixLayout"),
            other => panic!("a schema-2 document without matrixLayout was accepted: {other:?}"),
        }
        match read(&document(2, Some("row"), &parameters), &kernel) {
            Err(Error::Mismatch(message)) => {
                assert_eq!(message, "matrixLayout: host column, compiler row")
            }
            other => panic!("a row-major document was accepted: {other:?}"),
        }
        for unknown in ["diagonal", "Column", ""] {
            match read(&document(2, Some(unknown), &parameters), &kernel) {
                Err(Error::Invalid(message)) => assert_eq!(message, "unknown matrixLayout"),
                other => panic!("{unknown:?} was accepted: {other:?}"),
            }
        }
        // Schema 1 has no such field and ignores one: it reads exactly what it read before.
        let scalar = kernel_with(Vec::new());
        assert!(read(&document(1, None, &[]), &scalar).is_ok());
        assert!(read(&document(1, Some("row"), &[]), &scalar).is_ok());
    }

    #[test]
    fn a_matrix_type_in_the_wrong_layout_is_a_mismatch() {
        let json = document(2, Some("column"), &full_parameters());
        let column = r#""rows":4,"columns":4,"matrixLayout":"column""#;
        assert_eq!(json.matches(column).count(), 2, "the Mat4 buffer and P.m");
        let row = json.replacen(column, r#""rows":4,"columns":4,"matrixLayout":"row""#, 1);
        assert_eq!(
            mismatch_of(&row, &full_kernel()),
            "mats.matrixLayout: host column, compiler row"
        );
        let unknown = json.replacen(column, r#""rows":4,"columns":4,"matrixLayout":"zigzag""#, 1);
        assert!(matches!(
            read(&unknown, &full_kernel()),
            Err(Error::Invalid(message)) if message == "unknown matrixLayout"
        ));
    }

    #[test]
    fn schema_one_documents_cannot_describe_new_kinds() {
        let scalar_only = document(1, None, &[storage_json("a", 0, F32_LAYOUT, "read")]);
        let scalar_kernel = kernel_with(vec![parameter(
            "a",
            ParameterKind::Storage,
            Access::ReadOnly,
            ResourceLayout::storage_v1(F32_LAYOUT),
            0,
        )]);
        // The control: a schema-1 document and a scalar descriptor.
        let compiled = read(&scalar_only, &scalar_kernel).unwrap();
        assert_eq!(compiled.reflection.schema_version, 1);
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&scalar_kernel)
                .unwrap()
                .len(),
            1
        );
        let storage_struct_with_matrix = make_struct(
            "WithMatrix",
            80,
            16,
            &[member("a", 0, F32_LAYOUT), member("m", 16, MAT4)],
        );
        let uniform_of_scalars = make_struct("U", 16, 16, &[member("a", 0, F32_LAYOUT)]);
        for (what, kind, layout) in [
            (
                "vector",
                ParameterKind::Storage,
                ResourceLayout::storage_v2(VEC4),
            ),
            (
                "matrix",
                ParameterKind::Storage,
                ResourceLayout::storage_v2(MAT2),
            ),
            (
                "vector",
                ParameterKind::Storage,
                ResourceLayout::storage_v2(STORAGE_STRUCT),
            ),
            (
                "matrix",
                ParameterKind::Storage,
                ResourceLayout::storage_v2(storage_struct_with_matrix),
            ),
            (
                "uniform",
                ParameterKind::Uniform,
                ResourceLayout::std140_v1(uniform_of_scalars),
            ),
            (
                "uniform",
                ParameterKind::Uniform,
                ResourceLayout::std140_v1(CB_STRUCT),
            ),
        ] {
            let kernel = kernel_with(vec![parameter("a", kind, Access::ReadOnly, layout, 0)]);
            assert_eq!(
                compiled
                    .reflection
                    .cross_check_kernel(&kernel)
                    .unwrap_err()
                    .to_string(),
                format!(
                    "POD reflection mismatch: reflection schema 1 cannot describe {what}; re-export with cargo gust build"
                ),
                "{what}"
            );
        }
        // A schema-1 document that itself contains a new kind is refused by the reader, with the old messages.
        let with_vector = document(1, None, &[storage_json("pts", 0, STORAGE_STRUCT, "read")]);
        assert!(matches!(
            read(&with_vector, &full_kernel()),
            Err(Error::Invalid(message))
                if message == "unsupported element type (only 32-bit scalars and structs)"
        ));
        let with_matrix = document(1, None, &[storage_json("mats", 0, MAT4, "read")]);
        assert!(matches!(
            read(&with_matrix, &full_kernel()),
            Err(Error::Invalid(message))
                if message == "unsupported element type (only 32-bit scalars and structs)"
        ));
        let with_constant_buffer = document(1, None, &[uniform_json("cb", 0, CB_STRUCT)]);
        assert!(matches!(
            read(&with_constant_buffer, &full_kernel()),
            Err(Error::Invalid(message)) if message == "only structured buffers are supported"
        ));
    }

    #[test]
    fn only_schema_versions_one_to_three_are_read() {
        let kernel = kernel_with(Vec::new());
        for version in ["0", "4", "4294967295"] {
            let json = document(1, None, &[]).replacen(
                r#""schemaVersion":1"#,
                &format!(r#""schemaVersion":{version}"#),
                1,
            );
            assert!(matches!(
                read(&json, &kernel),
                Err(Error::Invalid(message)) if message == "unsupported native reflection schema version"
            ));
        }
        let json =
            document(1, None, &[]).replacen(r#""schemaVersion":1"#, r#""schemaVersion":"2""#, 1);
        assert!(read(&json, &kernel).is_err());
        assert!(read(&document(1, None, &[]), &kernel).is_ok());
        assert!(read(&document(2, Some("column"), &[]), &kernel).is_ok());
    }

    #[test]
    fn schema_two_fields_are_compared_exactly() {
        let json = document(2, Some("column"), &full_parameters());
        let kernel = full_kernel();
        // The control: the unmodified document passes, so each failure below is the one edit.
        let compiled = read(&json, &kernel).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&kernel).is_ok());
        let cases: &[(&str, &str, &str)] = &[
            (
                r#""name":"v","binding":{"kind":"uniform","offset":8,"size":8}"#,
                r#""name":"v","binding":{"kind":"uniform","offset":4,"size":8}"#,
                "pts.v.offset: host 8, compiler 4",
            ),
            (
                r#""elementCount":2"#,
                r#""elementCount":4"#,
                "pts.v: type or field count differs",
            ),
            (
                r#""kind":"vector","scalarType":"float32","elementCount":2"#,
                r#""kind":"vector","scalarType":"int32","elementCount":2"#,
                "pts.v: type or field count differs",
            ),
            (
                r#""size":8,"alignment":8,"stride":8,"name":"Vec2""#,
                r#""size":16,"alignment":8,"stride":8,"name":"Vec2""#,
                "pts.v.size: host 8, compiler 16",
            ),
            (
                r#""rows":4,"columns":4"#,
                r#""rows":4,"columns":3"#,
                "mats: type or field count differs",
            ),
            (
                r#""kind":"matrix","scalarType":"float32""#,
                r#""kind":"matrix","scalarType":"int32""#,
                "mats: type or field count differs",
            ),
            (
                r#""size":64,"alignment":16,"stride":16,"name":"Mat4""#,
                r#""size":64,"alignment":16,"stride":64,"name":"Mat4""#,
                "mats.stride: host 16, compiler 64",
            ),
            (
                r#""size":64,"alignment":16,"stride":16,"name":"Mat4""#,
                r#""size":32,"alignment":16,"stride":16,"name":"Mat4""#,
                "mats.size: host 64, compiler 32",
            ),
            (
                r#""type":{"kind":"constantBuffer","size":128"#,
                r#""type":{"kind":"constantBuffer","size":124"#,
                "cb.stride: host 128, compiler 124",
            ),
            (
                r#""size":128,"alignment":16,"stride":128,"name":"P""#,
                r#""size":128,"alignment":8,"stride":128,"name":"P""#,
                "cb.alignment: host 16, compiler 8",
            ),
            (
                r#""size":128,"alignment":16,"stride":128,"name":"P""#,
                r#""size":128,"alignment":16,"stride":112,"name":"P""#,
                "cb.stride: host 128, compiler 112",
            ),
            (
                r#""name":"m","binding":{"kind":"uniform","offset":32,"size":64}"#,
                r#""name":"m","binding":{"kind":"uniform","offset":16,"size":64}"#,
                "cb.m.offset: host 32, compiler 16",
            ),
            (
                r#""name":"e","binding":{"kind":"uniform","offset":96,"size":16}"#,
                r#""name":"e","binding":{"kind":"uniform","offset":80,"size":16}"#,
                "cb.e.offset: host 96, compiler 80",
            ),
            (
                r#""access":"readWrite""#,
                r#""access":"read""#,
                "mats.access differs",
            ),
            (
                r#""index":2"#,
                r#""index":7"#,
                "cb.binding: host 2, compiler 7",
            ),
        ];
        for (find, replace, expected) in cases {
            assert!(json.contains(find), "the fixture lost `{find}`");
            let edited = json.replacen(find, replace, 1);
            assert_eq!(&mismatch_of(&edited, &kernel), expected, "{find}");
        }
    }

    #[test]
    fn rules_kinds_and_access_must_fit_the_parameter() {
        let json = document(2, Some("column"), &full_parameters());
        let storage_shaped = make_struct("E", 4, 4, &[member("a", 0, F32_LAYOUT)]);
        let cases: Vec<(usize, ParameterDescriptor, &str)> = vec![
            (
                2,
                parameter(
                    "cb",
                    ParameterKind::Uniform,
                    Access::ReadOnly,
                    ResourceLayout::storage_v2(CB_STRUCT),
                    2,
                ),
                "cb.rules: StorageV2 does not fit this parameter kind",
            ),
            (
                0,
                parameter(
                    "pts",
                    ParameterKind::Storage,
                    Access::ReadOnly,
                    ResourceLayout::std140_v1(STORAGE_STRUCT),
                    0,
                ),
                "pts.rules: Std140V1 does not fit this parameter kind",
            ),
            (
                2,
                parameter(
                    "cb",
                    ParameterKind::Uniform,
                    Access::ReadWrite,
                    ResourceLayout::std140_v1(CB_STRUCT),
                    2,
                ),
                "cb.access differs",
            ),
            (
                2,
                parameter(
                    "cb",
                    ParameterKind::Value,
                    Access::ReadOnly,
                    ResourceLayout::std140_v1(CB_STRUCT),
                    2,
                ),
                "only storage and uniform descriptor resources are supported",
            ),
            (
                2,
                parameter(
                    "cb",
                    ParameterKind::Uniform,
                    Access::ReadOnly,
                    ResourceLayout::std140_v1(storage_shaped),
                    2,
                ),
                "host layout is not Std140V1",
            ),
            (
                2,
                parameter(
                    "cb",
                    ParameterKind::Uniform,
                    Access::ReadOnly,
                    ResourceLayout {
                        rules: ShaderLayoutRules::Std140V1,
                        element: CB_STRUCT,
                        element_stride: 64,
                    },
                    2,
                ),
                "cb.stride: inconsistent host resource layout",
            ),
            (
                0,
                parameter(
                    "pts",
                    ParameterKind::Storage,
                    Access::ReadOnly,
                    ResourceLayout::storage_v1(VEC4),
                    0,
                ),
                "host layout is not StorageV1",
            ),
            (
                0,
                parameter(
                    "pts",
                    ParameterKind::Storage,
                    Access::ReadOnly,
                    ResourceLayout::storage_v2(CB_STRUCT),
                    0,
                ),
                "host layout is not StorageV2",
            ),
        ];
        for (index, replacement, expected) in cases {
            let mut parameters = full_kernel().parameters.to_vec();
            parameters[index] = replacement;
            assert_eq!(
                mismatch_of(&json, &kernel_with(parameters)),
                expected,
                "{expected}"
            );
        }
        // A constant buffer in the document for a storage parameter, and the reverse.
        let kernel = full_kernel();
        let as_buffer = document(
            2,
            Some("column"),
            &[
                storage_json("pts", 0, STORAGE_STRUCT, "read"),
                storage_json("mats", 1, MAT4, "readWrite"),
                storage_json("cb", 2, CB_STRUCT, "read"),
            ],
        );
        assert_eq!(
            mismatch_of(&as_buffer, &kernel),
            "cb.kind: host ConstantBuffer, compiler StructuredBuffer"
        );
        let as_uniform = document(
            2,
            Some("column"),
            &[
                storage_json("pts", 0, STORAGE_STRUCT, "read"),
                uniform_json("mats", 1, MAT4),
                uniform_json("cb", 2, CB_STRUCT),
            ],
        );
        assert_eq!(
            mismatch_of(&as_uniform, &kernel),
            "mats.kind: host StructuredBuffer, compiler ConstantBuffer"
        );
    }

    #[test]
    fn cross_check_pod_checks_the_buffer_math_types() {
        let parameters = [
            storage_json("v4", 0, VEC4, "read"),
            storage_json("v2", 1, VEC2, "read"),
            storage_json("m2", 2, MAT2, "read"),
            storage_json("m4", 3, MAT4, "read"),
            uniform_json("cb", 4, CB_STRUCT),
        ];
        let compiled = read(
            &document(2, Some("column"), &parameters),
            &kernel_with(Vec::new()),
        )
        .unwrap();
        let resources = &compiled.reflection.resources;
        assert!(
            cross_check_pod::<Vec4>(&resources[0])
                .unwrap()
                .is_full_abi_match()
        );
        assert!(
            cross_check_pod::<Vec2>(&resources[1])
                .unwrap()
                .is_full_abi_match()
        );
        assert!(
            cross_check_pod::<Mat2>(&resources[2])
                .unwrap()
                .is_full_abi_match()
        );
        assert!(
            cross_check_pod::<Mat4>(&resources[3])
                .unwrap()
                .is_full_abi_match()
        );
        // Width, scalar kind, matrix size and resource kind each fail.
        assert!(cross_check_pod::<Vec2>(&resources[0]).is_err());
        assert!(cross_check_pod::<crate::math::IVec4>(&resources[0]).is_err());
        assert!(cross_check_pod::<Mat4>(&resources[2]).is_err());
        assert!(cross_check_pod::<Mat2>(&resources[3]).is_err());
        match cross_check_pod::<Vec4>(&resources[4]) {
            Err(Error::Mismatch(message)) => {
                assert_eq!(
                    message,
                    "cb.kind: host StructuredBuffer, compiler ConstantBuffer"
                )
            }
            other => panic!("a constant buffer was read as storage: {other:?}"),
        }
    }

    #[test]
    fn vector_widths_and_matrix_dimensions_are_two_to_four() {
        let kernel = kernel_with(Vec::new());
        let vector = document(2, Some("column"), &[storage_json("v", 0, VEC4, "read")]);
        for bad in ["0", "1", "5"] {
            let json = vector.replacen(
                r#""elementCount":4"#,
                &format!(r#""elementCount":{bad}"#),
                1,
            );
            assert!(matches!(
                read(&json, &kernel),
                Err(Error::Invalid(message)) if message == format!("unsupported elementCount {bad}")
            ));
        }
        let matrix = document(2, Some("column"), &[storage_json("m", 0, MAT4, "read")]);
        let rows = matrix.replacen(r#""rows":4"#, r#""rows":5"#, 1);
        assert!(matches!(
            read(&rows, &kernel),
            Err(Error::Invalid(message)) if message == "unsupported rows 5"
        ));
        let columns = matrix.replacen(r#""columns":4"#, r#""columns":1"#, 1);
        assert!(matches!(
            read(&columns, &kernel),
            Err(Error::Invalid(message)) if message == "unsupported columns 1"
        ));
        // The same text is a different, still valid, document with the width in range.
        assert!(read(&matrix, &kernel).is_ok());
    }

    #[gust::gpu]
    mod vector_add {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn add(
            id: SV_DispatchThreadID,
            a: StructuredBuffer<float>,
            b: StructuredBuffer<float>,
            mut out: RWStructuredBuffer<float>,
        ) {
            let i = id.x;
            if i < out.len() {
                out[i] = a[i] + b[i];
            }
        }
    }

    /// Every `*.reflection.json` (WGSL and SPIR-V sets alike) under the workspace, build output, VCS data and hidden
    /// directories aside. It walks the file system, not the index, so untracked files are picked up too.
    fn committed_reflection_files(
        directory: &std::path::Path,
        found: &mut Vec<std::path::PathBuf>,
    ) {
        let entries = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
        for entry in entries {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = entry.file_type().unwrap();
            if file_type.is_dir() {
                if name.starts_with('.')
                    || matches!(name.as_str(), "target" | "node_modules" | "generated-wgpu")
                {
                    continue;
                }
                committed_reflection_files(&entry.path(), found);
            } else if file_type.is_file() && name.ends_with(".reflection.json") {
                found.push(entry.path());
            }
        }
    }

    /// Schema 1 byte for byte: every committed artifact set that exists still verifies, through the same
    /// `verify_artifact` that the run-time path uses, with the reader that now also knows schema 2. No re-export,
    /// no re-blessing (plan R1-A.2).
    #[test]
    fn every_committed_schema_one_artifact_set_still_verifies() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut found = Vec::new();
        committed_reflection_files(&workspace, &mut found);
        found.sort();
        let relative: Vec<String> = found
            .iter()
            .map(|path| {
                path.strip_prefix(&workspace)
                    .unwrap()
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .collect();
        // The sets committed when this slice landed; a vanished one is a failure. Below, the vector-add WGSL sets
        // (the only kernel with a descriptor in this module) are verified and cross-checked through
        // `verify_artifact`; every other set, SPIR-V ones included, is only parsed by the reader, because there is no
        // descriptor to cross-check it against here. The walk is of the file system, so untracked files count too.
        for expected in [
            "crates/gust-wgpu/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json",
            "crates/gust/tests/fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json",
            "tests/ui/profile-native-artifact/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json",
            "tests/ui/profile-native-no-manifest/gust-artifacts/52e85c5f03f72272.wgsl.reflection.json",
        ] {
            assert!(relative.iter().any(|path| path == expected), "{expected}");
        }
        for (path, label) in found.iter().zip(&relative) {
            let reflection = std::fs::read_to_string(path).unwrap();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let key = file.split('.').next().unwrap().to_owned();
            if key == "52e85c5f03f72272" && file.ends_with(".wgsl.reflection.json") {
                let bytes =
                    std::fs::read(path.with_file_name(file.trim_end_matches(".reflection.json")))
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                let artifact = Artifact {
                    format: Format::Wgsl,
                    key: "52e85c5f03f72272",
                    bytes: Box::leak(bytes.into_boxed_slice()),
                    reflection: Box::leak(reflection.clone().into_boxed_str()),
                };
                let verified = verify_artifact(&vector_add::add::DESCRIPTOR, &artifact)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(verified.compiled.reflection.schema_version, 1, "{label}");
                assert_eq!(verified.cross_checks.len(), 3, "{label}");
                assert!(
                    verified
                        .cross_checks
                        .iter()
                        .all(PodCrossCheck::is_full_abi_match)
                );
                assert!(!reflection.contains("matrixLayout"), "{label}");
            } else {
                // Parsed only: no descriptor here to cross-check against.
                let target = if file.ends_with(".spv.reflection.json") {
                    Target::Spirv
                } else {
                    Target::Wgsl
                };
                let root = Parser::parse(&reflection).unwrap();
                let schema = read_schema_version(&root).unwrap();
                Reflection::read_json(target, &root, true, schema)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
            }
        }
    }

    // ---- schema 3: textures, samplers, the stage and a fragment's targets (slice mx.t0b, D91) ----

    use crate::{TextureLayout, math::UVec2};

    const U32_LAYOUT: TypeLayout = <u32 as GpuPod>::LAYOUT;
    const UVEC2: TypeLayout = <UVec2 as GpuPod>::LAYOUT;

    /// A texture or target result type: a scalar for `count == 1`, a vector otherwise.
    fn result_json(scalar: &str, count: u32) -> String {
        if count == 1 {
            format!(r#"{{"kind":"scalar","scalarType":"{scalar}"}}"#)
        } else {
            format!(r#"{{"kind":"vector","scalarType":"{scalar}","elementCount":{count}}}"#)
        }
    }

    const F4: (&str, u32) = ("float32", 4);

    fn slot_json(index: u32) -> String {
        format!(r#"{{"kind":"descriptorTableSlot","count":1,"index":{index},"space":0}}"#)
    }

    fn texture_json(
        name: &str,
        index: u32,
        access: &str,
        result: (&str, u32),
        format: &str,
    ) -> String {
        format!(
            r#"{{"name":"{name}","binding":{},"type":{{"kind":"texture","shape":"2d","access":"{access}","result":{},"format":"{format}"}}}}"#,
            slot_json(index),
            result_json(result.0, result.1)
        )
    }

    fn sampler_json(name: &str, index: u32) -> String {
        format!(
            r#"{{"name":"{name}","binding":{},"type":{{"kind":"sampler"}}}}"#,
            slot_json(index)
        )
    }

    fn target_json(index: u32, result: (&str, u32)) -> String {
        format!(
            r#"{{"index":{index},"type":{}}}"#,
            result_json(result.0, result.1)
        )
    }

    /// A schema-3 document whose identity fields match `SOURCE` and `ARTIFACT`: a compute one carries a workgroup
    /// size, a fragment one `targets`.
    fn document3(stage: &str, parameters: &[String], targets: &[String]) -> String {
        let source = fingerprint(SOURCE.as_bytes());
        let artifact = fingerprint(ARTIFACT);
        let (workgroup, targets) = if stage == "compute" {
            (r#""workgroupSize":[1,1,1],"#.to_owned(), String::new())
        } else {
            (
                String::new(),
                format!(r#","targets":[{}]"#, targets.join(",")),
            )
        };
        format!(
            r#"{{"schemaVersion":3,"matrixLayout":"column","compiler":"canned","target":"wgsl","entryPoint":"entry","stage":"{stage}","sourceHash":"{source}","artifactHash":"{artifact}",{workgroup}"parameters":[{}]{targets}}}"#,
            parameters.join(",")
        )
    }

    const fn sampled(class: crate::texture::SampleClass) -> TextureLayout {
        TextureLayout {
            format: None,
            dimension: TextureDimension::D2,
            access: TextureAccess::Sampled,
            class,
        }
    }

    const fn storage(format: TexelFormat, access: TextureAccess) -> TextureLayout {
        TextureLayout {
            format: Some(format),
            dimension: TextureDimension::D2,
            access,
            class: format.sample_class(),
        }
    }

    fn texture_parameter(
        name: &'static str,
        access: Access,
        element: TypeLayout,
        texture: TextureLayout,
        index: u32,
    ) -> ParameterDescriptor {
        parameter(
            name,
            ParameterKind::Texture,
            access,
            ResourceLayout::texture_v1(element, texture),
            index,
        )
    }

    fn sampler_parameter(name: &'static str, index: u32) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "Sampler",
            kind: ParameterKind::Sampler,
            access: Access::NotApplicable,
            resource_layout: None,
            binding: Some(ResourceBinding {
                group: 0,
                binding: index,
            }),
        }
    }

    /// An output buffer of a pixel kernel: a storage parameter with no shader binding (a colour target).
    fn output_parameter(name: &'static str, element: TypeLayout) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "RWStructuredBuffer<T>",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v2(element)),
            binding: None,
        }
    }

    fn pixel_kernel(parameters: Vec<ParameterDescriptor>) -> KernelDescriptor {
        KernelDescriptor {
            shape: crate::Shape::Pixel,
            ..kernel_with(parameters)
        }
    }

    /// A compute kernel with every texture form and a sampler and a buffer: `tex` sampled `float4` at 0, `rw` an
    /// `r32f` read-write texture at 1, `wo` an `rg32ui` write-only texture at 2, `samp` at 3, `out` at 4.
    fn texture_kernel() -> KernelDescriptor {
        kernel_with(vec![
            texture_parameter(
                "tex",
                Access::ReadOnly,
                VEC4,
                sampled(SampleClass::Float),
                0,
            ),
            texture_parameter(
                "rw",
                Access::ReadWrite,
                F32_LAYOUT,
                storage(TexelFormat::R32Float, TextureAccess::ReadWrite),
                1,
            ),
            texture_parameter(
                "wo",
                Access::ReadWrite,
                UVEC2,
                storage(TexelFormat::Rg32Uint, TextureAccess::WriteOnly),
                2,
            ),
            sampler_parameter("samp", 3),
            parameter(
                "out",
                ParameterKind::Storage,
                Access::ReadWrite,
                ResourceLayout::storage_v1(F32_LAYOUT),
                4,
            ),
        ])
    }

    fn texture_parameters() -> Vec<String> {
        vec![
            texture_json("tex", 0, "read", F4, "unknown"),
            texture_json("rw", 1, "readWrite", ("float32", 1), "r32f"),
            texture_json("wo", 2, "write", ("uint32", 2), "rg32ui"),
            sampler_json("samp", 3),
            storage_json("out", 4, F32_LAYOUT, "readWrite"),
        ]
    }

    fn texture_document() -> String {
        document3("compute", &texture_parameters(), &[])
    }

    /// A fragment kernel: `tex` and `samp`, one `float4` output.
    fn fragment_kernel() -> KernelDescriptor {
        pixel_kernel(vec![
            texture_parameter(
                "tex",
                Access::ReadOnly,
                VEC4,
                sampled(SampleClass::Float),
                0,
            ),
            sampler_parameter("samp", 1),
            output_parameter("color", VEC4),
        ])
    }

    fn fragment_parameters() -> Vec<String> {
        vec![
            texture_json("tex", 0, "read", F4, "unknown"),
            sampler_json("samp", 1),
        ]
    }

    fn fragment_document() -> String {
        document3("fragment", &fragment_parameters(), &[target_json(0, F4)])
    }

    /// The message of the `Mismatch` that reading `json` for `kernel` gives, at the read or at the cross-check.
    fn any_mismatch(json: &str, kernel: &KernelDescriptor) -> String {
        match read(json, kernel) {
            Err(Error::Mismatch(message)) => message,
            Err(other) => panic!("expected a mismatch, got {other:?}\n{json}"),
            Ok(compiled) => match compiled.reflection.cross_check_kernel(kernel) {
                Err(Error::Mismatch(message)) => message,
                other => panic!("expected a mismatch, got {other:?}\n{json}"),
            },
        }
    }

    fn invalid_of(json: &str, kernel: &KernelDescriptor) -> String {
        match read(json, kernel) {
            Err(Error::Invalid(message)) => message,
            other => panic!("expected an invalid document, got {other:?}\n{json}"),
        }
    }

    #[test]
    fn schema_three_reads_textures_samplers_and_the_stage() {
        let kernel = texture_kernel();
        let compiled = read(&texture_document(), &kernel).unwrap();
        let reflection = &compiled.reflection;
        assert_eq!(reflection.schema_version, 3);
        assert_eq!(reflection.stage, Stage::Compute);
        assert_eq!(compiled.workgroup_size, [1, 1, 1]);
        assert!(reflection.targets.is_empty());
        let textures: Vec<_> = reflection
            .textures
            .iter()
            .map(|t| (t.name.as_str(), t.binding, t.space, t.access, t.format))
            .collect();
        assert_eq!(
            textures,
            [
                ("tex", 0, 0, TextureAccess::Sampled, None),
                (
                    "rw",
                    1,
                    0,
                    TextureAccess::ReadWrite,
                    Some(TexelFormat::R32Float)
                ),
                (
                    "wo",
                    2,
                    0,
                    TextureAccess::WriteOnly,
                    Some(TexelFormat::Rg32Uint)
                ),
            ]
        );
        assert!(
            reflection
                .textures
                .iter()
                .all(|t| t.dimension == TextureDimension::D2)
        );
        assert_eq!(
            reflection.textures[2].result.kind,
            ReflectedTypeKind::Vector {
                scalar: ScalarKind::U32,
                count: 2
            }
        );
        // The helper reports no byte layout for a texture result.
        assert_eq!(reflection.textures[2].result.size, None);
        let samplers: Vec<_> = reflection
            .samplers
            .iter()
            .map(|s| (s.name.as_str(), s.binding, s.space))
            .collect();
        assert_eq!(samplers, [("samp", 3, 0)]);
        // The buffer is in the buffer list alone, and only buffers have a layout report.
        assert_eq!(reflection.resources.len(), 1);
        let reports = reflection.cross_check_kernel(&kernel).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].is_full_abi_match());
    }

    #[test]
    fn a_fragment_document_has_targets_in_order_and_no_workgroup() {
        let kernel = fragment_kernel();
        let compiled = read(&fragment_document(), &kernel).unwrap();
        let reflection = &compiled.reflection;
        assert_eq!(reflection.stage, Stage::Fragment);
        assert_eq!(compiled.workgroup_size, [0, 0, 0]);
        assert_eq!(reflection.targets.len(), 1);
        assert_eq!(reflection.targets[0].index, 0);
        assert_eq!(
            reflection.targets[0].ty.kind,
            ReflectedTypeKind::Vector {
                scalar: ScalarKind::F32,
                count: 4
            }
        );
        assert!(reflection.cross_check_kernel(&kernel).unwrap().is_empty());
        // Two outputs, in parameter order, at indices 0 and 1.
        let two = pixel_kernel(vec![
            output_parameter("a", U32_LAYOUT),
            output_parameter("b", VEC2),
        ]);
        let document = document3(
            "fragment",
            &[],
            &[
                target_json(0, ("uint32", 1)),
                target_json(1, ("float32", 2)),
            ],
        );
        let compiled = read(&document, &two).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&two).is_ok());
    }

    #[test]
    fn schemas_one_and_two_are_compute_documents_without_textures() {
        let kernel = full_kernel();
        for (schema, matrix) in [(1_u32, None), (2, Some("column"))] {
            let parameters = if schema == 1 {
                vec![storage_json("a", 0, F32_LAYOUT, "read")]
            } else {
                full_parameters()
            };
            let kernel = if schema == 1 {
                kernel_with(vec![parameter(
                    "a",
                    ParameterKind::Storage,
                    Access::ReadOnly,
                    ResourceLayout::storage_v1(F32_LAYOUT),
                    0,
                )])
            } else {
                kernel
            };
            let compiled = read(&document(schema, matrix, &parameters), &kernel).unwrap();
            let reflection = &compiled.reflection;
            assert_eq!(reflection.stage, Stage::Compute);
            assert!(reflection.textures.is_empty() && reflection.samplers.is_empty());
            assert!(reflection.targets.is_empty());
        }
        // A texture in a document of schema 1 or 2 is refused with the message those schemas always gave.
        let texture = texture_json("tex", 0, "read", F4, "unknown");
        for (schema, matrix) in [(1_u32, None), (2, Some("column"))] {
            let json = document(schema, matrix, std::slice::from_ref(&texture));
            assert_eq!(
                invalid_of(&json, &kernel_with(Vec::new())),
                "only structured buffers are supported",
                "schema {schema}"
            );
        }
        // A schema-3 document with nothing new in it is still a schema-3 document.
        assert!(read(&document3("compute", &[], &[]), &kernel_with(Vec::new())).is_ok());
    }

    #[test]
    fn a_schema_one_or_two_reflection_cannot_describe_a_texture_or_a_sampler() {
        let compiled = read(&document(2, Some("column"), &[]), &kernel_with(Vec::new())).unwrap();
        let tex = texture_parameter(
            "tex",
            Access::ReadOnly,
            VEC4,
            sampled(SampleClass::Float),
            0,
        );
        for (what, parameter) in [("texture", tex), ("sampler", sampler_parameter("samp", 1))] {
            assert_eq!(
                compiled
                    .reflection
                    .cross_check_kernel(&kernel_with(vec![parameter]))
                    .unwrap_err()
                    .to_string(),
                format!(
                    "POD reflection mismatch: reflection schema 2 cannot describe {what}; re-export with cargo gust build"
                ),
                "{what}"
            );
        }
    }

    #[test]
    fn the_schema_three_grammar_is_strict() {
        let kernel = texture_kernel();
        let fragment = fragment_kernel();
        let control = texture_parameters();
        let with = |index: usize, parameter: String| {
            let mut parameters = control.clone();
            parameters[index] = parameter;
            document3("compute", &parameters, &[])
        };
        let cases: Vec<(&str, String, &str)> = vec![
            (
                "an unknown stage",
                texture_document().replacen(r#""stage":"compute""#, r#""stage":"geometry""#, 1),
                "unknown stage",
            ),
            (
                "a stage that is not text",
                texture_document().replacen(r#""stage":"compute""#, r#""stage":1"#, 1),
                "expected string",
            ),
            (
                "a missing stage",
                texture_document().replacen(r#""stage":"compute","#, "", 1),
                "missing stage",
            ),
            (
                "a texture of another shape",
                with(0, texture_json("tex", 0, "read", F4, "unknown").replace("2d", "3d")),
                "unsupported texture shape (only 2d)",
            ),
            (
                "a texture access of another name",
                with(0, texture_json("tex", 0, "sample", F4, "unknown")),
                "unsupported texture access",
            ),
            (
                "a format outside the dialect",
                with(1, texture_json("rw", 1, "readWrite", F4, "rgba8")),
                "unsupported texture format `rgba8`",
            ),
            (
                "a matrix result",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown").replace(
                        r#""kind":"vector","scalarType":"float32","elementCount":4"#,
                        r#""kind":"matrix","scalarType":"float32","rows":2,"columns":2,"matrixLayout":"column""#,
                    ),
                ),
                "unsupported texture result type (a scalar or a vector)",
            ),
            (
                "a texture result of five components",
                with(0, texture_json("tex", 0, "read", ("float32", 5), "unknown")),
                "unsupported elementCount 5",
            ),
            (
                "a texture and a buffer in one slot",
                with(4, storage_json("out", 1, F32_LAYOUT, "readWrite")),
                "duplicate resource name or binding",
            ),
            (
                "a sampler named like a texture",
                with(3, sampler_json("tex", 3)),
                "duplicate resource name or binding",
            ),
            (
                "a sampler in a texture's slot",
                with(3, sampler_json("samp", 0)),
                "duplicate resource name or binding",
            ),
            (
                "a binding array of textures",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown")
                        .replacen(r#""count":1"#, r#""count":2"#, 1),
                ),
                "binding arrays are unsupported",
            ),
            (
                "targets in a compute document",
                texture_document().replacen(
                    r#""parameters""#,
                    r#""targets":[],"parameters""#,
                    1,
                ),
                "a compute reflection has no targets",
            ),
        ];
        for (what, json, expected) in cases {
            assert_eq!(invalid_of(&json, &kernel), expected, "{what}");
        }
        let fragment_cases: Vec<(&str, String, &str)> = vec![
            (
                "a fragment with a workgroup size",
                fragment_document().replacen(
                    r#""parameters""#,
                    r#""workgroupSize":[1,1,1],"parameters""#,
                    1,
                ),
                "a fragment reflection has no workgroupSize",
            ),
            (
                "a fragment without targets",
                fragment_document().replacen(r#","targets":[{"index":0,"type":{"kind":"vector","scalarType":"float32","elementCount":4}}]"#, "", 1),
                "a fragment reflection needs targets",
            ),
            (
                "targets out of order",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[target_json(1, F4), target_json(0, F4)],
                ),
                "targets must be in ascending SV_Target order without repeats",
            ),
            (
                "a repeated target index",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[target_json(0, F4), target_json(0, F4)],
                ),
                "targets must be in ascending SV_Target order without repeats",
            ),
            (
                "a target that is a struct",
                document3(
                    "fragment",
                    &fragment_parameters(),
                    &[r#"{"index":0,"type":{"kind":"struct","fields":[{"name":"a","binding":{"kind":"uniform","offset":0,"size":4},"type":{"kind":"scalar","scalarType":"float32"}}]}}"#.to_owned()],
                ),
                "unsupported target type (a scalar or a vector)",
            ),
        ];
        for (what, json, expected) in fragment_cases {
            assert_eq!(invalid_of(&json, &fragment), expected, "{what}");
        }
    }

    // ---- the tamper cases (plan section 3.3): every one is a `Mismatch` that names the field ----

    #[test]
    fn the_untouched_schema_three_documents_pass_their_cross_checks() {
        let compute = texture_kernel();
        let compiled = read(&texture_document(), &compute).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&compute).is_ok());
        let fragment = fragment_kernel();
        let compiled = read(&fragment_document(), &fragment).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&fragment).is_ok());
    }

    #[test]
    fn a_stage_that_differs_from_the_shape_is_a_mismatch_naming_stage() {
        // A compute reflection against a pixel descriptor.
        let pixel = fragment_kernel();
        let compute_document = document3("compute", &fragment_parameters(), &[]);
        assert_eq!(
            any_mismatch(&compute_document, &pixel),
            "stage: host fragment, compiler compute"
        );
        // The same at the cross-check, which does not depend on the read having caught it: the reflection was read for
        // a grid kernel and is checked against a pixel one.
        let grid = kernel_with(Vec::new());
        let compiled = read(&document3("compute", &[], &[]), &grid).unwrap();
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&pixel)
                .unwrap_err()
                .to_string(),
            "POD reflection mismatch: stage: host fragment, compiler compute"
        );
        // A fragment reflection against a grid descriptor, at the read and at the cross-check.
        assert_eq!(
            any_mismatch(&fragment_document(), &texture_kernel()),
            "stage: host compute, compiler fragment"
        );
        let compiled = read(&fragment_document(), &fragment_kernel()).unwrap();
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&texture_kernel())
                .unwrap_err()
                .to_string(),
            "POD reflection mismatch: stage: host compute, compiler fragment"
        );
        // A schema-1 and a schema-2 reflection are compute reflections.
        let compiled = read(&document(2, Some("column"), &[]), &grid).unwrap();
        assert_eq!(
            compiled
                .reflection
                .cross_check_kernel(&pixel)
                .unwrap_err()
                .to_string(),
            "POD reflection mismatch: stage: host fragment, compiler compute"
        );
    }

    #[test]
    fn a_target_index_or_type_that_differs_is_a_mismatch_naming_the_target() {
        let kernel = fragment_kernel();
        let parameters = fragment_parameters();
        for (what, targets, expected) in [
            (
                "the index",
                vec![target_json(1, F4)],
                "targets[0].index: host 0, compiler 1",
            ),
            (
                "the scalar type",
                vec![target_json(0, ("uint32", 4))],
                "targets[0].type: host f32x4, compiler u32x4",
            ),
            (
                "the width",
                vec![target_json(0, ("float32", 2))],
                "targets[0].type: host f32x4, compiler f32x2",
            ),
            (
                "a scalar for a vector",
                vec![target_json(0, ("float32", 1))],
                "targets[0].type: host f32x4, compiler f32",
            ),
            (
                "an extra target",
                vec![target_json(0, F4), target_json(1, F4)],
                "targets: host 1 outputs, compiler 2",
            ),
            (
                "a missing target",
                Vec::new(),
                "targets: host 1 outputs, compiler 0",
            ),
        ] {
            let json = document3("fragment", &parameters, &targets);
            assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
        }
        // A gap in the indices (the helper reports `SV_Target0` and `SV_Target2`) is not outputs in parameter order.
        let two = pixel_kernel(vec![
            output_parameter("a", U32_LAYOUT),
            output_parameter("b", VEC4),
        ]);
        let json = document3(
            "fragment",
            &[],
            &[target_json(0, ("uint32", 1)), target_json(2, F4)],
        );
        assert_eq!(
            any_mismatch(&json, &two),
            "targets[1].index: host 1, compiler 2"
        );
        // Swapping the two outputs' order is a type mismatch at both positions' first.
        let swapped = pixel_kernel(vec![
            output_parameter("b", VEC4),
            output_parameter("a", U32_LAYOUT),
        ]);
        let json = document3(
            "fragment",
            &[],
            &[target_json(0, ("uint32", 1)), target_json(1, F4)],
        );
        assert_eq!(
            any_mismatch(&json, &swapped),
            "targets[0].type: host f32x4, compiler u32"
        );
    }

    #[test]
    fn a_texture_that_differs_is_a_mismatch_naming_the_field() {
        let kernel = texture_kernel();
        let control = texture_parameters();
        let with = |index: usize, parameter: String| {
            let mut parameters = control.clone();
            parameters[index] = parameter;
            document3("compute", &parameters, &[])
        };
        for (what, json, expected) in [
            (
                "access of a read-write texture",
                with(1, texture_json("rw", 1, "write", ("float32", 1), "r32f")),
                "rw.access: host readWrite, compiler write",
            ),
            (
                "access of a write-only texture",
                with(
                    2,
                    texture_json("wo", 2, "readWrite", ("uint32", 2), "rg32ui"),
                ),
                "wo.access: host write, compiler readWrite",
            ),
            (
                "access of a sampled texture",
                with(0, texture_json("tex", 0, "readWrite", F4, "unknown")),
                "tex.access: host read, compiler readWrite",
            ),
            (
                "format of a read-write texture",
                with(
                    1,
                    texture_json("rw", 1, "readWrite", ("float32", 1), "rgba32f"),
                ),
                "rw.format: host r32float, compiler rgba32float",
            ),
            (
                "format of a write-only texture",
                with(2, texture_json("wo", 2, "write", ("uint32", 2), "rg32i")),
                "wo.format: host rg32uint, compiler rg32sint",
            ),
            (
                "format of a storage texture reported as unknown",
                with(
                    1,
                    texture_json("rw", 1, "readWrite", ("float32", 1), "unknown"),
                ),
                "rw.format: host r32float, compiler none",
            ),
            (
                "a format on a sampled texture",
                with(0, texture_json("tex", 0, "read", F4, "r32f")),
                "tex.format: host none, compiler r32float",
            ),
            (
                "result type of a storage texture",
                with(
                    1,
                    texture_json("rw", 1, "readWrite", ("float32", 2), "r32f"),
                ),
                "rw.result: host f32, compiler f32x2",
            ),
            (
                "result scalar of a write-only texture",
                with(2, texture_json("wo", 2, "write", ("int32", 2), "rg32ui")),
                "wo.result: host u32x2, compiler i32x2",
            ),
            (
                "sample class of a sampled texture",
                with(0, texture_json("tex", 0, "read", ("uint32", 4), "unknown")),
                "tex.class: host f32, compiler u32",
            ),
            (
                "result width of a sampled texture",
                with(0, texture_json("tex", 0, "read", ("float32", 1), "unknown")),
                "tex.result: host f32x4, compiler f32",
            ),
            (
                "slot of a texture",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown").replacen(
                        r#""index":0"#,
                        r#""index":6"#,
                        1,
                    ),
                ),
                "tex.binding: host 0, compiler 6",
            ),
            (
                "space of a texture",
                with(
                    0,
                    texture_json("tex", 0, "read", F4, "unknown").replacen(
                        r#""space":0"#,
                        r#""space":1"#,
                        1,
                    ),
                ),
                "tex.group: host 0, compiler 1",
            ),
            (
                "a texture the document reports as a buffer",
                with(0, storage_json("tex", 0, F32_LAYOUT, "read")),
                "tex.kind: host texture, compiler a buffer",
            ),
            (
                "a texture the document reports as a sampler",
                with(0, sampler_json("tex", 0)),
                "tex.kind: host texture, compiler a sampler",
            ),
            (
                "a texture the document names otherwise",
                with(0, texture_json("tex2", 0, "read", F4, "unknown")),
                "missing compiler binding tex",
            ),
        ] {
            assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
        }
        // A texture the descriptor does not have.
        let mut parameters = texture_parameters();
        parameters.push(texture_json("extra", 7, "read", F4, "unknown"));
        assert_eq!(
            any_mismatch(&document3("compute", &parameters, &[]), &kernel),
            "unexpected compiler resource bindings"
        );
    }

    #[test]
    fn a_sampler_that_differs_is_a_mismatch_naming_the_field() {
        let kernel = texture_kernel();
        let control = texture_parameters();
        let with = |parameter: String| {
            let mut parameters = control.clone();
            parameters[3] = parameter;
            document3("compute", &parameters, &[])
        };
        for (what, json, expected) in [
            (
                "a moved slot",
                with(sampler_json("samp", 5)),
                "samp.binding: host 3, compiler 5",
            ),
            (
                "a moved space",
                with(sampler_json("samp", 3).replacen(r#""space":0"#, r#""space":2"#, 1)),
                "samp.group: host 0, compiler 2",
            ),
            (
                "a sampler the document reports as a texture",
                with(texture_json("samp", 3, "read", F4, "unknown")),
                "samp.kind: host sampler, compiler a texture",
            ),
            (
                "a sampler the document names otherwise",
                with(sampler_json("other", 3)),
                "missing compiler binding samp",
            ),
        ] {
            assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
        }
        let mut parameters = texture_parameters();
        parameters.push(sampler_json("extra", 7));
        assert_eq!(
            any_mismatch(&document3("compute", &parameters, &[]), &kernel),
            "unexpected compiler resource bindings"
        );
        // A descriptor with a sampler the document lacks.
        let mut parameters = texture_parameters();
        parameters.remove(3);
        assert_eq!(
            any_mismatch(&document3("compute", &parameters, &[]), &kernel),
            "missing compiler binding samp"
        );
    }

    #[test]
    fn a_descriptor_that_contradicts_itself_is_a_mismatch_even_against_its_own_document() {
        let document = texture_document();
        let kernel_with_tex = |texture: ParameterDescriptor| {
            let mut kernel = texture_kernel();
            let mut parameters = kernel.parameters.to_vec();
            parameters[0] = texture;
            kernel.parameters = Box::leak(parameters.into_boxed_slice());
            kernel
        };
        let kernel_with_rw = |texture: ParameterDescriptor| {
            let mut kernel = texture_kernel();
            let mut parameters = kernel.parameters.to_vec();
            parameters[1] = texture;
            kernel.parameters = Box::leak(parameters.into_boxed_slice());
            kernel
        };
        let rgba = storage(TexelFormat::Rgba32Float, TextureAccess::ReadWrite);
        let cases: Vec<(&str, KernelDescriptor, &str)> = vec![
            (
                "a class that is not the result type's",
                kernel_with_tex(texture_parameter(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    sampled(SampleClass::Uint),
                    0,
                )),
                "tex.class: host u32, result type f32",
            ),
            (
                "a format on a sampled texture",
                kernel_with_tex(texture_parameter(
                    "tex",
                    Access::ReadOnly,
                    VEC4,
                    TextureLayout {
                        format: Some(TexelFormat::Rgba32Float),
                        ..sampled(SampleClass::Float)
                    },
                    0,
                )),
                "tex.format: a sampled texture has no format in the shader, host rgba32float",
            ),
            (
                "a storage texture without a format",
                kernel_with_rw(texture_parameter(
                    "rw",
                    Access::ReadWrite,
                    VEC4,
                    TextureLayout {
                        format: None,
                        ..rgba
                    },
                    1,
                )),
                "rw.format: a storage texture needs a format",
            ),
            (
                "a read-write texture of a wide format",
                kernel_with_rw(texture_parameter("rw", Access::ReadWrite, VEC4, rgba, 1)),
                "rw.format: rgba32float does not fit class f32 and access readWrite",
            ),
            (
                "a format of another class",
                kernel_with_rw(texture_parameter(
                    "rw",
                    Access::ReadWrite,
                    VEC4,
                    TextureLayout {
                        format: Some(TexelFormat::Rgba32Uint),
                        access: TextureAccess::WriteOnly,
                        ..rgba
                    },
                    1,
                )),
                "rw.format: rgba32uint does not fit class f32 and access write",
            ),
            (
                "a sampled texture with write access",
                kernel_with_tex(texture_parameter(
                    "tex",
                    Access::ReadWrite,
                    VEC4,
                    sampled(SampleClass::Float),
                    0,
                )),
                "tex.access differs",
            ),
            (
                "a texture with a buffer rule",
                kernel_with_tex(parameter(
                    "tex",
                    ParameterKind::Texture,
                    Access::ReadOnly,
                    ResourceLayout::storage_v2(VEC4),
                    0,
                )),
                "tex.rules: StorageV2 does not fit this parameter kind",
            ),
            (
                "a texture result that is a matrix",
                kernel_with_tex(texture_parameter(
                    "tex",
                    Access::ReadOnly,
                    MAT4,
                    sampled(SampleClass::Float),
                    0,
                )),
                "tex.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector",
            ),
            (
                "a texture result that is three wide",
                kernel_with_tex(texture_parameter(
                    "tex",
                    Access::ReadOnly,
                    TypeLayout::vector("Vec3", ScalarKind::F32, 3),
                    sampled(SampleClass::Float),
                    0,
                )),
                "tex.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector",
            ),
            (
                "a format with more channels than the result type has",
                kernel_with_rw(texture_parameter(
                    "rw",
                    Access::ReadWrite,
                    UVEC2,
                    TextureLayout {
                        format: Some(TexelFormat::Rgba32Uint),
                        access: TextureAccess::WriteOnly,
                        ..storage(TexelFormat::Rgba32Uint, TextureAccess::WriteOnly)
                    },
                    1,
                )),
                "rw.format: rgba32uint has channel count 4, the result type has width 2",
            ),
            (
                "a format with fewer channels than the result type has",
                kernel_with_rw(texture_parameter(
                    "rw",
                    Access::ReadWrite,
                    VEC4,
                    storage(TexelFormat::R32Float, TextureAccess::ReadWrite),
                    1,
                )),
                "rw.format: r32float has channel count 1, the result type has width 4",
            ),
            (
                "a sampler with a layout",
                {
                    let mut kernel = texture_kernel();
                    let mut parameters = kernel.parameters.to_vec();
                    parameters[3].resource_layout = Some(ResourceLayout::storage_v1(F32_LAYOUT));
                    kernel.parameters = Box::leak(parameters.into_boxed_slice());
                    kernel
                },
                "samp: a sampler has no resource contract",
            ),
        ];
        for (what, kernel, expected) in cases {
            assert_eq!(any_mismatch(&document, &kernel), expected, "{what}");
        }
        // A buffer parameter that carries the texture rule.
        let buffer = kernel_with(vec![parameter(
            "out",
            ParameterKind::Storage,
            Access::ReadWrite,
            ResourceLayout::texture_v1(F32_LAYOUT, sampled(SampleClass::Float)),
            0,
        )]);
        let json = document3(
            "compute",
            &[storage_json("out", 0, F32_LAYOUT, "readWrite")],
            &[],
        );
        assert!(
            any_mismatch(&json, &buffer).starts_with("out.rules: TextureV1("),
            "{}",
            any_mismatch(&json, &buffer)
        );
        // Two descriptor bindings in one slot, across kinds.
        let clash = kernel_with(vec![
            texture_parameter(
                "tex",
                Access::ReadOnly,
                VEC4,
                sampled(SampleClass::Float),
                0,
            ),
            sampler_parameter("samp", 0),
        ]);
        assert_eq!(
            any_mismatch(
                &document3(
                    "compute",
                    &[
                        texture_json("tex", 0, "read", F4, "unknown"),
                        sampler_json("samp", 1)
                    ],
                    &[]
                ),
                &clash
            ),
            "duplicate descriptor binding"
        );
    }

    #[test]
    fn a_pixel_kernel_may_have_no_outputs_in_its_descriptor_only_if_the_document_has_none() {
        let no_outputs = pixel_kernel(vec![texture_parameter(
            "tex",
            Access::ReadOnly,
            VEC4,
            sampled(SampleClass::Float),
            0,
        )]);
        let json = document3(
            "fragment",
            &[texture_json("tex", 0, "read", F4, "unknown")],
            &[],
        );
        let compiled = read(&json, &no_outputs).unwrap();
        assert!(compiled.reflection.cross_check_kernel(&no_outputs).is_ok());
        assert_eq!(
            any_mismatch(&fragment_document(), &no_outputs),
            "unexpected compiler resource bindings"
        );
    }
}

/// The document's `schemaVersion`: 1, 2 or 3, nothing else.
fn read_schema_version(root: &Json) -> Result<u32, Error> {
    match root.get("schemaVersion")?.u32()? {
        version @ (1..=3) => Ok(version),
        _ => Err(invalid("unsupported native reflection schema version")),
    }
}

/// What a schema-1 document cannot describe about this parameter, if anything: a uniform, or a vector or matrix
/// anywhere in its element.
fn schema_one_gap(parameter: &crate::ParameterDescriptor) -> Option<&'static str> {
    fn layout_gap(layout: TypeLayout) -> Option<&'static str> {
        match layout.kind {
            TypeLayoutKind::Scalar(_) => None,
            TypeLayoutKind::Vector { .. } => Some("vector"),
            TypeLayoutKind::Matrix { .. } => Some("matrix"),
            TypeLayoutKind::Struct(fields) => fields.iter().find_map(|field| layout_gap(field.ty)),
        }
    }
    if parameter.kind == crate::ParameterKind::Uniform {
        return Some("uniform");
    }
    parameter
        .resource_layout
        .and_then(|layout| layout_gap(layout.element))
}

fn read_scalar(value: &Json) -> Result<ScalarKind, Error> {
    Ok(match value.get("scalarType")?.string()? {
        "float32" => ScalarKind::F32,
        "int32" => ScalarKind::I32,
        "uint32" => ScalarKind::U32,
        _ => return Err(invalid("unsupported scalar type")),
    })
}

/// A vector width or matrix dimension of the dialect: two to four.
fn read_dimension(value: &Json, key: &str) -> Result<u32, Error> {
    let n = value.get(key)?.u32()?;
    if (2..=4).contains(&n) {
        Ok(n)
    } else {
        Err(invalid(format!("unsupported {key} {n}")))
    }
}

/// `schema` is the document's `schemaVersion`; the `vector` and `matrix` kinds exist from 2 on.
fn read_type(value: &Json, complete: bool, schema: u32) -> Result<ReflectedType, Error> {
    let kind = match value.get("kind")?.string()? {
        "scalar" => ReflectedTypeKind::Scalar(read_scalar(value)?),
        "vector" if schema >= 2 => ReflectedTypeKind::Vector {
            scalar: read_scalar(value)?,
            count: read_dimension(value, "elementCount")?,
        },
        "matrix" if schema >= 2 => ReflectedTypeKind::Matrix {
            scalar: read_scalar(value)?,
            rows: read_dimension(value, "rows")?,
            cols: read_dimension(value, "columns")?,
            layout: MatrixLayout::parse(value.get("matrixLayout")?.string()?)
                .ok_or_else(|| invalid("unknown matrixLayout"))?,
        },
        "struct" => {
            let mut fields = Vec::new();
            for field in value.get("fields")?.array()? {
                let name = field.get("name")?.string()?.to_owned();
                if fields.iter().any(|f: &ReflectedField| f.name == name) {
                    return Err(invalid("duplicate struct field"));
                }
                let binding = field.get("binding")?;
                if binding.get("kind")?.string()? != "uniform" {
                    return Err(invalid("unsupported field layout category"));
                }
                fields.push(ReflectedField {
                    name,
                    offset: binding.get("offset")?.u32()?,
                    size: binding.get("size")?.u32()?,
                    ty: read_type(field.get("type")?, complete, schema)?,
                });
            }
            if fields.is_empty() {
                return Err(invalid("empty struct layout"));
            }
            ReflectedTypeKind::Struct(fields)
        }
        _ if schema >= 2 => {
            return Err(invalid(
                "unsupported element type (only 32-bit scalars, vectors, matrices and structs)",
            ));
        }
        _ => {
            return Err(invalid(
                "unsupported element type (only 32-bit scalars and structs)",
            ));
        }
    };
    Ok(ReflectedType {
        name: value
            .optional("name")?
            .map(Json::string)
            .transpose()?
            .map(str::to_owned),
        size: if complete {
            Some(value.get("size")?.u32()?)
        } else {
            None
        },
        alignment: if complete {
            Some(value.get("alignment")?.u32()?)
        } else {
            None
        },
        stride: if complete {
            Some(value.get("stride")?.u32()?)
        } else {
            None
        },
        kind,
    })
}

/// Check all available compiler field metadata against `T`, retaining explicit gaps.
pub fn cross_check_pod<T: GpuPod>(resource: &ReflectedResource) -> Result<PodCrossCheck, Error> {
    let host = T::LAYOUT;
    if !host.is_storage_v2()
        || host.size as usize != size_of::<T>()
        || host.alignment as usize != align_of::<T>()
    {
        return Err(Error::Mismatch(
            "host layout is not a valid storage-v2 Rust layout".into(),
        ));
    }
    cross_check_layout(host, ShaderLayoutRules::StorageV2, resource)
}

/// Check `host`, declared under `rules`, against one reflected resource.
fn cross_check_layout(
    host: TypeLayout,
    rules: ShaderLayoutRules,
    resource: &ReflectedResource,
) -> Result<PodCrossCheck, Error> {
    let (host_fits, expected_kind) = match rules {
        ShaderLayoutRules::StorageV1 => (
            host.is_storage_v1(),
            ReflectedResourceKind::StructuredBuffer,
        ),
        ShaderLayoutRules::StorageV2 => (
            host.is_storage_v2(),
            ReflectedResourceKind::StructuredBuffer,
        ),
        ShaderLayoutRules::Std140V1 => (host.is_std140_v1(), ReflectedResourceKind::ConstantBuffer),
        // A texture is checked by `Reflection::cross_check_kernel` against the document's texture list, never as a
        // buffer.
        ShaderLayoutRules::TextureV1(_) => {
            return Err(Error::Mismatch(format!("host layout is not {rules:?}")));
        }
    };
    if !host_fits {
        return Err(Error::Mismatch(format!("host layout is not {rules:?}")));
    }
    if resource.kind != expected_kind {
        return Err(Error::Mismatch(format!(
            "{}.kind: host {expected_kind:?}, compiler {:?}",
            resource.name, resource.kind
        )));
    }
    let mut report = PodCrossCheck {
        checked_fields: 0,
        aggregate_size_verified: true,
        element_stride_verified: resource.element_stride.is_some(),
        alignment_verified: true,
    };
    check_type(host, &resource.element, &resource.name, &mut report)?;
    if let Some(stride) = resource.element_stride {
        equal(host.stride, stride, &format!("{}.stride", resource.name))?;
    }
    Ok(report)
}
fn equal(host: u32, compiler: u32, path: &str) -> Result<(), Error> {
    if host == compiler {
        Ok(())
    } else {
        Err(Error::Mismatch(format!(
            "{path}: host {host}, compiler {compiler}"
        )))
    }
}
fn check_type(
    host: TypeLayout,
    compiler: &ReflectedType,
    path: &str,
    report: &mut PodCrossCheck,
) -> Result<(), Error> {
    if let Some(size) = compiler.size {
        equal(host.size, size, &format!("{path}.size"))?;
    } else {
        report.aggregate_size_verified = false;
    }
    if let Some(alignment) = compiler.alignment {
        equal(host.alignment, alignment, &format!("{path}.alignment"))?;
    } else {
        report.alignment_verified = false;
    }
    if let Some(stride) = compiler.stride {
        // The stride of a matrix is its column stride (schema 2); every other type reports its array stride.
        let host_stride = match host.kind {
            TypeLayoutKind::Matrix { column_stride, .. } => column_stride,
            _ => host.stride,
        };
        equal(host_stride, stride, &format!("{path}.stride"))?;
    } else {
        report.element_stride_verified = false;
    }
    match (host.kind, &compiler.kind) {
        (TypeLayoutKind::Scalar(a), ReflectedTypeKind::Scalar(b)) if a == *b => Ok(()),
        (
            TypeLayoutKind::Vector {
                scalar: a,
                count: n,
            },
            ReflectedTypeKind::Vector {
                scalar: b,
                count: m,
            },
        ) if a == *b && n == *m => Ok(()),
        (
            TypeLayoutKind::Matrix {
                scalar: a,
                rows,
                cols,
                ..
            },
            ReflectedTypeKind::Matrix {
                scalar: b,
                rows: compiler_rows,
                cols: compiler_cols,
                layout,
            },
        ) if a == *b && rows == *compiler_rows && cols == *compiler_cols => {
            if *layout == MATRIX_LAYOUT {
                Ok(())
            } else {
                Err(Error::Mismatch(format!(
                    "{path}.matrixLayout: host {}, compiler {}",
                    MATRIX_LAYOUT.as_str(),
                    layout.as_str()
                )))
            }
        }
        (TypeLayoutKind::Struct(a), ReflectedTypeKind::Struct(b)) if a.len() == b.len() => {
            for (host, compiler) in a.iter().zip(b) {
                if host.name != compiler.name {
                    return Err(Error::Mismatch(format!("{path}: field name/order differs")));
                }
                let path = format!("{path}.{}", host.name);
                equal(host.offset, compiler.offset, &format!("{path}.offset"))?;
                equal(host.ty.size, compiler.size, &format!("{path}.size"))?;
                report.checked_fields += 1;
                check_type(host.ty, &compiler.ty, &path, report)?;
            }
            Ok(())
        }
        _ => Err(Error::Mismatch(format!(
            "{path}: type or field count differs"
        ))),
    }
}

#[derive(Debug)]
enum Json {
    Object(BTreeMap<String, Json>),
    Array(Vec<Json>),
    String(String),
    Number(String),
    Other,
}
impl Json {
    fn optional(&self, key: &str) -> Result<Option<&Self>, Error> {
        match self {
            Self::Object(m) => Ok(m.get(key)),
            _ => Err(invalid("expected object")),
        }
    }
    fn get(&self, key: &str) -> Result<&Self, Error> {
        self.optional(key)?
            .ok_or_else(|| invalid(format!("missing {key}")))
    }
    fn array(&self) -> Result<&[Self], Error> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(invalid("expected array")),
        }
    }
    fn string(&self) -> Result<&str, Error> {
        match self {
            Self::String(s) => Ok(s),
            _ => Err(invalid("expected string")),
        }
    }
    fn u32(&self) -> Result<u32, Error> {
        match self {
            Self::Number(s) if s.bytes().all(|b| b.is_ascii_digit()) => {
                s.parse().map_err(|_| invalid("integer exceeds u32"))
            }
            _ => Err(invalid("expected unsigned integer")),
        }
    }
}
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Parser<'a> {
    fn parse(source: &'a str) -> Result<Json, Error> {
        let mut p = Self {
            bytes: source.as_bytes(),
            pos: 0,
        };
        let value = p.value(0)?;
        p.ws();
        if p.pos != p.bytes.len() {
            return Err(invalid("trailing JSON data"));
        }
        Ok(value)
    }
    fn ws(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b' ' | b'\r' | b'\n' | b'\t'))
        {
            self.pos += 1;
        }
    }
    fn take(&mut self, b: u8) -> bool {
        self.ws();
        if self.bytes.get(self.pos) == Some(&b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, b: u8) -> Result<(), Error> {
        if self.take(b) {
            Ok(())
        } else {
            Err(invalid(format!(
                "expected {} at byte {}",
                b as char, self.pos
            )))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Json, Error> {
        if depth > 64 {
            return Err(invalid("JSON nesting exceeds 64"));
        }
        self.ws();
        match self.bytes.get(self.pos).copied() {
            Some(b'{') => {
                self.pos += 1;
                let mut fields = BTreeMap::new();
                if !self.take(b'}') {
                    loop {
                        self.ws();
                        let key = self.string()?;
                        self.expect(b':')?;
                        let value = self.value(depth + 1)?;
                        if fields.insert(key, value).is_some() {
                            return Err(invalid("duplicate JSON key"));
                        }
                        if self.take(b'}') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Json::Object(fields))
            }
            Some(b'[') => {
                self.pos += 1;
                let mut values = Vec::new();
                if !self.take(b']') {
                    loop {
                        values.push(self.value(depth + 1)?);
                        if self.take(b']') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Ok(Json::Array(values))
            }
            Some(b'"') => self.string().map(Json::String),
            Some(b'-' | b'0'..=b'9') => self.number().map(Json::Number),
            _ => {
                for literal in [b"true".as_slice(), b"false", b"null"] {
                    if self.bytes[self.pos..].starts_with(literal) {
                        self.pos += literal.len();
                        return Ok(Json::Other);
                    }
                }
                Err(invalid(format!("invalid JSON at byte {}", self.pos)))
            }
        }
    }
    fn number(&mut self) -> Result<String, Error> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        if self.bytes.get(self.pos) == Some(&b'0') {
            self.pos += 1;
        } else {
            self.digits()?;
        }
        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b'e' | b'E'))
        {
            self.pos += 1;
            if self
                .bytes
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'+' | b'-'))
            {
                self.pos += 1;
            }
            self.digits()?;
        }
        Ok(String::from_utf8(self.bytes[start..self.pos].to_vec()).expect("ASCII JSON number"))
    }
    fn digits(&mut self) -> Result<(), Error> {
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if self.pos == start {
            Err(invalid("expected digit"))
        } else {
            Ok(())
        }
    }
    fn hex(&mut self) -> Result<u16, Error> {
        let mut n = 0;
        for _ in 0..4 {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| invalid("truncated Unicode escape"))?;
            self.pos += 1;
            n = n * 16
                + (b as char)
                    .to_digit(16)
                    .ok_or_else(|| invalid("invalid Unicode escape"))? as u16;
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut result = String::new();
        let mut start = self.pos;
        loop {
            let b = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| invalid("unterminated string"))?;
            if b == b'"' || b == b'\\' {
                result.push_str(
                    std::str::from_utf8(&self.bytes[start..self.pos])
                        .map_err(|_| invalid("invalid UTF-8"))?,
                );
                self.pos += 1;
                if b == b'"' {
                    return Ok(result);
                }
                let escape = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| invalid("truncated escape"))?;
                self.pos += 1;
                result.push(match escape {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let high = self.hex()?;
                        let cp = if (0xd800..=0xdbff).contains(&high) {
                            if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                return Err(invalid("missing low surrogate"));
                            }
                            self.pos += 2;
                            let low = self.hex()?;
                            if !(0xdc00..=0xdfff).contains(&low) {
                                return Err(invalid("invalid low surrogate"));
                            }
                            0x10000 + ((high as u32 - 0xd800) << 10) + (low as u32 - 0xdc00)
                        } else {
                            high as u32
                        };
                        char::from_u32(cp).ok_or_else(|| invalid("invalid Unicode scalar"))?
                    }
                    _ => return Err(invalid("invalid escape")),
                });
                start = self.pos;
            } else {
                if b < 0x20 {
                    return Err(invalid("control byte in string"));
                }
                self.pos += 1;
            }
        }
    }
}
