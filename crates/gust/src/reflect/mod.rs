//! Slang reflection for the restricted storage and constant-buffer ABIs.
//! Native compilation returns the artifact and complete layout from one linked program.
//! CLI JSON remains an explicit partial-evidence compatibility path.
//! Only global structured-buffer resources (and, under schema 2, constant buffers) are supported. Entry-point
//! parameters must be the generated `SV_DispatchThreadID` system value, not resources/uniforms.
//!
//! # Schema versions of the native reflection JSON
//!
//! The reader accepts `schemaVersion` 1 and 2. Schema 1 describes exactly what it always did: 32-bit scalars and
//! structs of them, in structured buffers. Schema 2 adds, beside those, the type kinds `vector`
//! (`scalarType`, `elementCount`) and `matrix` (`scalarType`, `rows`, `columns`, `matrixLayout`; its `stride` is the
//! column stride), the parameter type `constantBuffer` (`elementType`, `size`), and a mandatory top-level
//! `matrixLayout` that must equal `MATRIX_LAYOUT` (column-major). Every type carries `size`, `alignment` and
//! `stride`. A schema-1 document checked against a descriptor with a vector, a matrix or a uniform is refused with
//! the instruction to re-export it ([`Reflection::cross_check_kernel`]); nothing committed before schema 2 needs
//! re-exporting otherwise.
//!
//! # The `slang-runtime` feature
//!
//! The functions that launch a compiler process (`reflect`, `compile_reflected`, `compile_reflected_with_helper`,
//! `native_compiler_path`) exist only with the `slang-runtime` feature, which is off by default (D44). Parsing
//! and checking reflection JSON (`Reflection::from_json`, `cross_check_*`, `CompiledReflection::from_output`) needs
//! no feature, so [`crate::artifact::verify_artifact`] works without a compiler.

use crate::{
    GpuPod, KernelDescriptor, ScalarKind, ShaderLayoutRules, TypeLayout, TypeLayoutKind,
    abi::{MATRIX_LAYOUT, MatrixLayout},
    slang::{self, Target},
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
    /// 2 for schema-2 native documents.
    pub schema_version: u32,
    pub resources: Vec<ReflectedResource>,
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
    pub workgroup_size: [u32; 3],
}

#[cfg(feature = "slang-runtime")]
pub fn native_compiler_path() -> PathBuf {
    if let Some(path) = std::env::var_os("GUST_SLANG_REFLECT") {
        return path.into();
    }
    let executable = format!("gust-slang-reflect{}", std::env::consts::EXE_SUFFIX);
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/slang-reflect")
        .join(&executable);
    if local.is_file() {
        local
    } else {
        executable.into()
    }
}

#[cfg(feature = "slang-runtime")]
/// Compile and reflect the same linked program using the native Slang helper.
pub fn compile_reflected(
    kernel: &KernelDescriptor,
    target: Target,
) -> Result<CompiledReflection, Error> {
    compile_reflected_with_helper(kernel, target, native_compiler_path())
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
                    "{}\n{}\ncompiler exit: {}\nreflection helper: {}",
                    String::from_utf8_lossy(&output.stderr),
                    String::from_utf8_lossy(&output.stdout),
                    output.status,
                    helper.as_ref().display(),
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
                "{}\n{}\ncompiler exit: {}\nreflection helper: {}\noriginating Rust kernel: {}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout),
                output.status,
                helper.display(),
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
    source: &std::path::Path,
    target: Target,
    entry: &str,
    shape: crate::Shape,
) -> Command {
    let mut command = Command::new("slangc");
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
    let mut command = slangc_reflection_command(&source, target, kernel.entry_point, kernel.shape);
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

    /// `schema` is the document's `schemaVersion` (1 or 2): under 1 exactly the pre-schema-2 grammar is read.
    fn read_json(target: Target, root: &Json, complete: bool, schema: u32) -> Result<Self, Error> {
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
        let mut resources = Vec::new();
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
            if resources.iter().any(|r: &ReflectedResource| {
                r.name == name || (r.binding == index && r.space == space)
            }) {
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
        Ok(Self {
            target,
            schema_version: schema,
            resources,
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
        let mut reports = Vec::new();
        let mut bindings = BTreeMap::new();
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
            if self.schema_version < 2
                && let Some(what) = schema_one_gap(parameter)
            {
                return Err(Error::Mismatch(format!(
                    "reflection schema 1 cannot describe {what}; re-export with cargo gust build"
                )));
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
        if reports.len() != self.resources.len() {
            return Err(Error::Mismatch(
                "unexpected compiler resource bindings".into(),
            ));
        }
        Ok(reports)
    }
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
    fn only_schema_versions_one_and_two_are_read() {
        let kernel = kernel_with(Vec::new());
        for version in ["0", "3", "4294967295"] {
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
}

/// The document's `schemaVersion`: 1 or 2, nothing else.
fn read_schema_version(root: &Json) -> Result<u32, Error> {
    match root.get("schemaVersion")?.u32()? {
        version @ (1 | 2) => Ok(version),
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
    let host_fits = match rules {
        ShaderLayoutRules::StorageV1 => host.is_storage_v1(),
        ShaderLayoutRules::StorageV2 => host.is_storage_v2(),
        ShaderLayoutRules::Std140V1 => host.is_std140_v1(),
    };
    if !host_fits {
        return Err(Error::Mismatch(format!("host layout is not {rules:?}")));
    }
    let expected_kind = match rules {
        ShaderLayoutRules::StorageV1 | ShaderLayoutRules::StorageV2 => {
            ReflectedResourceKind::StructuredBuffer
        }
        ShaderLayoutRules::Std140V1 => ReflectedResourceKind::ConstantBuffer,
    };
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
