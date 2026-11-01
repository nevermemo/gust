//! Slang reflection for the restricted scalar/struct storage ABI.
//! Native compilation returns the artifact and complete layout from one linked program.
//! CLI JSON remains an explicit partial-evidence compatibility path.
//! Only global structured-buffer resources are supported. Entry-point parameters
//! must be the generated `SV_DispatchThreadID` system value, not resources/uniforms.

use crate::{
    GpuPod, KernelDescriptor, ScalarKind, SourceOrigin, TypeLayout, TypeLayoutKind,
    slang::{self, Target, TemporaryDirectory},
};
use std::{
    collections::BTreeMap,
    error, fmt, fs,
    path::{Path, PathBuf},
    process::Command,
};

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
    pub resources: Vec<ReflectedResource>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedResource {
    pub name: String,
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

/// Compile and reflect the same linked program using the native Slang helper.
pub fn compile_reflected(
    kernel: &KernelDescriptor,
    target: Target,
) -> Result<CompiledReflection, Error> {
    compile_reflected_with_helper(kernel, target, native_compiler_path())
}

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
    let output = Command::new(helper.as_ref())
        .arg(&source)
        .arg(kernel.entry_point)
        .arg(target_name(target))
        .arg(&artifact)
        .arg(&json)
        .output()
        .map_err(|source| Error::HelperUnavailable {
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
        if root.get("schemaVersion")?.u32()? != 1 {
            return Err(invalid("unsupported native reflection schema version"));
        }
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
            reflection: Reflection::read_json(target, &root, true)?,
            compiler_version,
            entry_point: kernel.entry_point.to_owned(),
            workgroup_size,
        })
    }
}

/// Compile exactly this target and entry point, requesting Slang's JSON metadata.
pub fn reflect(kernel: &KernelDescriptor, target: Target) -> Result<Reflection, Error> {
    let directory = TemporaryDirectory::create()?;
    let source = directory.path().join("kernel.slang");
    let json = directory.path().join("reflection.json");
    let (target_name, extension) = match target {
        Target::Wgsl => ("wgsl", "wgsl"),
        Target::Spirv => ("spirv", "spv"),
    };
    slang::write_compile_inputs(directory.path(), kernel.slang_source, &kernel.source)?;
    let mut command = Command::new("slangc");
    command.arg(source).args([
        "-target",
        target_name,
        "-entry",
        kernel.entry_point,
        "-stage",
        "compute",
    ]);
    if target == Target::Spirv {
        command.arg("-fvk-use-entrypoint-name");
    }
    let output = command
        .arg("-reflection-json")
        .arg(&json)
        .arg("-o")
        .arg(directory.path().join(format!("kernel.{extension}")))
        .output()
        .map_err(|e| {
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
        Self::read_json(target, &root, false)
    }

    fn read_json(target: Target, root: &Json, complete: bool) -> Result<Self, Error> {
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
            if ty.get("kind")?.string()? != "resource"
                || ty.get("baseShape")?.string()? != "structuredBuffer"
            {
                return Err(invalid("only structured buffers are supported"));
            }
            let access = match ty.optional("access")?.map(Json::string).transpose()? {
                Some("read") => crate::Access::ReadOnly,
                None if !complete => crate::Access::ReadOnly,
                Some("readWrite") => crate::Access::ReadWrite,
                _ => return Err(invalid("unsupported resource access")),
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
            resources.push(ReflectedResource {
                name,
                access,
                binding: index,
                space,
                element: read_type(ty.get("resultType")?, complete)?,
                element_stride: if complete {
                    Some(value.get("elementStride")?.u32()?)
                } else {
                    None
                },
            });
        }
        Ok(Self { target, resources })
    }

    /// Require complete StorageV1 evidence for every descriptor binding, with no extras.
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
            if parameter.kind != crate::ParameterKind::Storage {
                return Err(Error::Mismatch(
                    "only StorageV1 descriptor resources are supported".into(),
                ));
            }
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
            if layout.element_stride != layout.element.stride {
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
            let report = cross_check_layout(layout.element, resource)?;
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
}

fn read_type(value: &Json, complete: bool) -> Result<ReflectedType, Error> {
    let kind = match value.get("kind")?.string()? {
        "scalar" => ReflectedTypeKind::Scalar(match value.get("scalarType")?.string()? {
            "float32" => ScalarKind::F32,
            "int32" => ScalarKind::I32,
            "uint32" => ScalarKind::U32,
            _ => return Err(invalid("unsupported scalar type")),
        }),
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
                    ty: read_type(field.get("type")?, complete)?,
                });
            }
            if fields.is_empty() {
                return Err(invalid("empty struct layout"));
            }
            ReflectedTypeKind::Struct(fields)
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
    if !host.is_storage_v1()
        || host.size as usize != size_of::<T>()
        || host.alignment as usize != align_of::<T>()
    {
        return Err(Error::Mismatch(
            "host layout is not a valid storage-v1 Rust layout".into(),
        ));
    }
    cross_check_layout(host, resource)
}

fn cross_check_layout(
    host: TypeLayout,
    resource: &ReflectedResource,
) -> Result<PodCrossCheck, Error> {
    if !host.is_storage_v1() {
        return Err(Error::Mismatch("host layout is not StorageV1".into()));
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
        equal(host.stride, stride, &format!("{path}.stride"))?;
    } else {
        report.element_stride_verified = false;
    }
    match (host.kind, &compiler.kind) {
        (TypeLayoutKind::Scalar(a), ReflectedTypeKind::Scalar(b)) if a == *b => Ok(()),
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
