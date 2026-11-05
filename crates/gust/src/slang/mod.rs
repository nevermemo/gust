//! Runtime access to the external Slang compiler.
//!
//! The procedural macro emits portable Slang source. Host backends call the
//! compiler only when they need a concrete target such as SPIR-V or WGSL.
//!
//! # The `slang-runtime` feature
//!
//! Everything that launches `slangc` (`compile`, `compile_spirv`, `compile_wgsl`, `compile_source`, `probe`) and the
//! helpers only those use (the temporary directory, the diagnostic mapping) exist only with the `slang-runtime`
//! feature, which is off by default (D44). `Target`, `Error`, `decode_spirv` and `compile_copy` stay outside
//! the feature, because [`crate::artifact`] verifies a precompiled artifact without a compiler.

use std::{
    error, fmt, fs,
    path::{Path, PathBuf},
};
#[cfg(feature = "slang-runtime")]
use std::{
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(feature = "slang-runtime")]
use crate::Shape;
use crate::{KernelDescriptor, SourceOrigin};
#[cfg(feature = "slang-runtime")]
use source_map::{SourceMap, SourceMapSegment};

#[cfg(feature = "slang-runtime")]
mod source_map;

#[cfg(feature = "slang-runtime")]
static NEXT_INVOCATION: AtomicU64 = AtomicU64::new(1);

/// Every launch of `slangc` or of the reflection helper goes through the one retry helper of [`crate::toolchain`],
/// which tries again while the executable is busy (`ETXTBSY`).
#[cfg(feature = "slang-runtime")]
pub(crate) use crate::toolchain::output_retrying_busy;

/// The `slangc` that `-version` has already shown to be the pinned Slang (or a listed build), so a compile costs no
/// extra process after the first. Only successes are kept: a mismatch is checked again, which is what an install of
/// the pin in the meantime needs.
#[cfg(feature = "slang-runtime")]
static VERIFIED_SLANGC: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// The `slangc` to launch, and a proof that it is the pinned Slang (D72, D109): the managed one for the pin, else the
/// first on `PATH` ([`crate::toolchain::slangc_path`]), whose `-version` must be the pin or a listed validated build.
/// [`Error::CompilerNotFound`] when there is none; [`Error::SlangVersionMismatch`] naming both versions when it is
/// another Slang. The check runs once per `slangc` per process.
#[cfg(feature = "slang-runtime")]
pub(crate) fn pinned_slangc() -> Result<PathBuf, Error> {
    pinned_slangc_in(&crate::toolchain::Env::from_process())
}

/// [`pinned_slangc`] for a given environment, so a test names the `PATH` and the cache without touching the process's.
/// [`Error::Toolchain`] for an unusable `GUST_SLANG_HOME`.
#[cfg(feature = "slang-runtime")]
pub(crate) fn pinned_slangc_in(env: &crate::toolchain::Env) -> Result<PathBuf, Error> {
    let slangc = crate::toolchain::slangc_path_in(env)
        .map_err(Error::Toolchain)?
        .ok_or(Error::CompilerNotFound)?;
    verified_slangc(slangc)
}

/// The proof half of [`pinned_slangc_in`]: `slangc` itself when its `-version` is the pin or a listed build.
#[cfg(feature = "slang-runtime")]
fn verified_slangc(slangc: PathBuf) -> Result<PathBuf, Error> {
    use crate::toolchain;

    if VERIFIED_SLANGC
        .lock()
        .is_ok_and(|verified| verified.contains(&slangc))
    {
        return Ok(slangc);
    }
    pin_verdict(toolchain::check_slangc_version(&slangc))?;
    if let Ok(mut verified) = VERIFIED_SLANGC.lock() {
        verified.push(slangc.clone());
    }
    Ok(slangc)
}

/// [`crate::toolchain::check_slangc_version`]'s answer as this module's error.
#[cfg(feature = "slang-runtime")]
fn pin_verdict(verdict: Result<(), crate::toolchain::VersionError>) -> Result<(), Error> {
    use crate::toolchain::{SLANG_PIN, VersionError};

    match verdict {
        Ok(()) => Ok(()),
        Err(VersionError::Mismatch { slangc, found }) => Err(Error::SlangVersionMismatch {
            slangc,
            found,
            pinned: SLANG_PIN,
        }),
        Err(VersionError::Run(reason) | VersionError::Output(reason)) => {
            Err(Error::Toolchain(reason))
        }
    }
}

/// What the compile-time copy is called, and the name a `#line` directive returns
/// to for lines with no Rust origin.
pub(crate) const KERNEL_FILE: &str = "kernel.slang";
/// The clean text, written beside the copy when the copy carries directives.
#[cfg(feature = "slang-runtime")]
const EXPORT_FILE: &str = "kernel.export.slang";

/// A compile target of the Slang compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// SPIR-V, the binary shader format of Vulkan.
    Spirv,
    /// WGSL, the shader language of the WebGPU API.
    Wgsl,
}

#[cfg(feature = "slang-runtime")]
impl Target {
    const fn slang_name(self) -> &'static str {
        match self {
            Self::Spirv => "spirv",
            Self::Wgsl => "wgsl",
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Spirv => "spv",
            Self::Wgsl => "wgsl",
        }
    }
}

/// Every way running the Slang compiler can fail.
///
/// The enum is `#[non_exhaustive]`: a later release can add a variant without a breaking change, so a `match` in
/// another crate needs a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// `slangc` could not be launched: it is neither in the managed install nor on `PATH`.
    CompilerNotFound,
    /// Reading or writing a Slang artifact failed (the temporary directory, the source file, the compiler's output),
    /// `slangc` could not be launched for a reason other than not being found, or the compiler returned text that is
    /// not UTF-8.
    Io(std::io::Error),
    /// `slangc` ran and rejected the source. `diagnostics` is its output for the `target` that failed.
    CompilationFailed {
        /// The target that failed.
        target: Target,
        /// The compiler's output.
        diagnostics: String,
    },
    /// The compiled SPIR-V is not a whole number of 32-bit words. The number is its length in bytes.
    InvalidSpirvLength(usize),
    /// The compiled SPIR-V failed the structural check (its header or instruction stream). The text says which
    /// part.
    InvalidSpirv(&'static str),
    /// The toolchain cannot be located or asked: a bad `GUST_SLANG_HOME` or `GUST_SLANG_REFLECT`, or a `slangc` whose
    /// `-version` fails or prints no version. The text names the variable or the executable.
    Toolchain(String),
    /// The `slangc` that would run reports another Slang than the pinned one (D72). Both versions are named; the
    /// run-time bridge compiles with the pin only.
    SlangVersionMismatch {
        /// The executable that was asked.
        slangc: PathBuf,
        /// What its `-version` printed.
        found: String,
        /// The Slang version GUST is pinned to.
        pinned: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompilerNotFound => write!(
                formatter,
                "could not launch `slangc`; install the pinned Slang with `cargo gust slang install`, or make a slangc of it available on PATH"
            ),
            Self::Io(error) => write!(formatter, "Slang artifact I/O failed: {error}"),
            Self::Toolchain(reason) => write!(formatter, "Slang toolchain: {reason}"),
            Self::CompilationFailed {
                target,
                diagnostics,
            } => write!(
                formatter,
                "Slang failed to compile the {:?} target:\n{}",
                target,
                diagnostics.trim()
            ),
            Self::InvalidSpirvLength(bytes) => {
                write!(formatter, "Slang returned a {bytes}-byte SPIR-V artifact")
            }
            Self::InvalidSpirv(reason) => {
                write!(formatter, "Slang returned invalid SPIR-V: {reason}")
            }
            Self::SlangVersionMismatch {
                slangc,
                found,
                pinned,
            } => write!(
                formatter,
                "{} reports Slang `{found}`, not the pinned {pinned}; install the pin with \
                 `cargo gust slang install`",
                slangc.display()
            ),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Compile the Slang source of `kernel` to `target` with the pinned `slangc` and return the output bytes (WGSL text or
/// SPIR-V words as little-endian bytes). Needs the `slang-runtime` feature.
///
/// Fails with [`Error::CompilerNotFound`] or [`Error::SlangVersionMismatch`] when `slangc` is missing or is not the
/// pinned version, with [`Error::CompilationFailed`] when the compiler rejects the source (its diagnostics name the
/// Rust lines where the descriptor has a source origin), and with [`Error::Io`] when a file cannot be read or written.
#[cfg(feature = "slang-runtime")]
pub fn compile(kernel: &KernelDescriptor, target: Target) -> Result<Vec<u8>, Error> {
    compile_with_origin(
        kernel.slang_source,
        kernel.entry_point,
        &kernel.source,
        kernel.shape,
        target,
    )
}

/// Compile `kernel` to SPIR-V and decode the output to 32-bit words. Fails as [`compile`] does, and with
/// [`Error::InvalidSpirvLength`] or [`Error::InvalidSpirv`] when the output is not a whole number of words or fails the
/// structural check.
#[cfg(feature = "slang-runtime")]
pub fn compile_spirv(kernel: &KernelDescriptor) -> Result<Vec<u32>, Error> {
    let bytes = compile(kernel, Target::Spirv)?;
    decode_spirv(&bytes)
}

pub(crate) fn decode_spirv(bytes: &[u8]) -> Result<Vec<u32>, Error> {
    if !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(Error::InvalidSpirvLength(bytes.len()));
    }
    // The length check above leaves no remainder.
    let (chunks, remainder) = bytes.as_chunks::<{ size_of::<u32>() }>();
    debug_assert!(remainder.is_empty());
    let words: Vec<_> = chunks.iter().copied().map(u32::from_le_bytes).collect();
    crate::spirv::validate_structure(&words).map_err(Error::InvalidSpirv)?;
    Ok(words)
}

/// Compile `kernel` to WGSL and return the text. Fails as [`compile`] does; output that is not UTF-8 is an
/// [`Error::Io`].
#[cfg(feature = "slang-runtime")]
pub fn compile_wgsl(kernel: &KernelDescriptor) -> Result<String, Error> {
    let bytes = compile(kernel, Target::Wgsl)?;
    String::from_utf8(bytes)
        .map_err(|error| Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, error)))
}

/// Compile Slang text that has no descriptor: a `grid` (compute) entry point, as every kernel is today. A text of
/// another shape goes through [`compile_source_as`].
#[cfg(feature = "slang-runtime")]
pub fn compile_source(source: &str, entry: &str, target: Target) -> Result<Vec<u8>, Error> {
    compile_source_as(source, entry, Shape::Grid, target)
}

/// [`compile_source`] for an entry point of `shape`: slangc gets that shape's `-stage`.
#[cfg(feature = "slang-runtime")]
pub fn compile_source_as(
    source: &str,
    entry: &str,
    shape: Shape,
    target: Target,
) -> Result<Vec<u8>, Error> {
    compile_with_origin(source, entry, &SourceOrigin::UNKNOWN, shape, target)
}

#[cfg(feature = "slang-runtime")]
fn compile_with_origin(
    source: &str,
    entry: &str,
    origin: &SourceOrigin,
    shape: Shape,
    target: Target,
) -> Result<Vec<u8>, Error> {
    compile_in_directory(
        TemporaryDirectory::create()?,
        source,
        entry,
        origin,
        shape,
        target,
    )
}

#[cfg(feature = "slang-runtime")]
fn compile_in_directory(
    directory: TemporaryDirectory,
    source: &str,
    entry: &str,
    origin: &SourceOrigin,
    shape: Shape,
    target: Target,
) -> Result<Vec<u8>, Error> {
    // The guard removes the directory when this returns, on every path.
    compile_at(&directory.0, source, entry, origin, shape, target)
}

/// The `slangc` command up to the output options. The matrix layout is named, not left to the default: the command
/// line defaults to column-major today and the native API of the reflection helper to row-major (M6 spike V1), and
/// `abi::MATRIX_LAYOUT` is column-major, so both paths pin it (the helper through `defaultMatrixLayoutMode`). The
/// stage is the shape's, not a constant (mx.k): `compute` for a `grid` kernel.
#[cfg(feature = "slang-runtime")]
fn slangc_command(
    slangc: &Path,
    source: &Path,
    target: Target,
    entry: &str,
    shape: Shape,
) -> Command {
    let mut command = Command::new(slangc);
    command.arg(source).args([
        "-target",
        target.slang_name(),
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
/// Compile inside `directory`, which the caller owns and removes.
fn compile_at(
    directory: &Path,
    source: &str,
    entry: &str,
    origin: &SourceOrigin,
    shape: Shape,
    target: Target,
) -> Result<Vec<u8>, Error> {
    compile_at_in(
        &crate::toolchain::Env::from_process(),
        directory,
        source,
        entry,
        origin,
        shape,
        target,
    )
}

/// [`compile_at`] with the environment that names the `slangc` given, so a test proves the compile asks for the pinned
/// one ([`pinned_slangc_in`]) before it launches anything.
#[cfg(feature = "slang-runtime")]
fn compile_at_in(
    env: &crate::toolchain::Env,
    directory: &Path,
    source: &str,
    entry: &str,
    origin: &SourceOrigin,
    shape: Shape,
    target: Target,
) -> Result<Vec<u8>, Error> {
    let source_path = directory.join(KERNEL_FILE);
    let map_path = directory.join("kernel.map.json");
    let output_path = directory.join(format!("kernel.{}", target.extension()));
    let source_map = SourceMap::from_slang(source);
    let line_directives = write_compile_inputs(directory, source, origin)? != source;
    fs::write(&map_path, source_map.json(origin, line_directives))?;

    let slangc = pinned_slangc_in(env)?;
    let mut command = slangc_command(&slangc, &source_path, target, entry, shape);
    let result = output_retrying_busy(command.arg("-o").arg(&output_path));
    let output = match result {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::CompilerNotFound);
        }
        Err(error) => {
            return Err(Error::Io(error));
        }
    };
    if !output.status.success() {
        let diagnostics = format!(
            "{}\n{}\ncompiler exit: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout),
            output.status
        );
        let diagnostics = annotate_diagnostics(origin, source, &source_map, &diagnostics);
        return Err(Error::CompilationFailed {
            target,
            diagnostics,
        });
    }
    let artifact = fs::read(&output_path)?;
    Ok(artifact)
}

/// Characters a `#line` file name may contain: the set `gust-macros` restricts
/// every recorded path component to (ASCII letters and digits, `_ . + -`, and
/// non-ASCII characters with the Unicode Alphabetic property), plus the `/`
/// between components. slangc echoes a UTF-8 name back unchanged (probed with
/// two-, three- and four-byte characters). Anything else (a quote, a
/// backslash, a control character, a line break) could end the directive's
/// string or the directive itself, so a hand-written origin with such a path
/// gets no directives. The structure is the macro's as well (`is_valid_record` in
/// `gust-macros`): every `/`-separated component is non-empty and not made of
/// dots only, so a leading or trailing `/`, `//`, `.` and `..` are refused. The
/// macro never records such a path, so no macro-recorded file is affected. Keep
/// this in step with `is_valid_component` there.
fn directive_file_is_usable(file: &str) -> bool {
    let name = file.rsplit('/').next().unwrap_or(file);
    file.split('/').all(|component| {
        !component.is_empty()
            && !component.chars().all(|c| c == '.')
            && component.chars().all(|c| {
                c.is_ascii_alphanumeric()
                    || matches!(c, '_' | '.' | '+' | '-')
                    || (!c.is_ascii() && c.is_alphabetic())
            })
    })
        // A Rust file that looks like the compile copy would make its own
        // locations indistinguishable from the switch-back ones.
        && !name.eq_ignore_ascii_case(KERNEL_FILE)
}

/// The text the compiler reads: `source` with `#line` directives (M1.3).
///
/// Every Slang line that `origin` attributes to a Rust line gets
/// `#line <rust_line> "<file>"` in front of it, so a compiler error there names
/// the Rust file and line. The directive is repeated for each attributed line
/// because the lowering of one Rust statement often spans several Slang lines
/// (a `for` pair, a lowered `match`) that must all keep that statement's line
/// instead of counting up. A line without an origin that follows an attributed
/// one gets `#line <slang_line> "kernel.slang"` first, so the compiler returns to
/// Slang coordinates and the construct lookup of DX19 keeps working for it.
/// Lines before the first mapping are copied untouched.
///
/// Directives also land inside struct and helper bodies, because those lines are
/// attributed too (a field line carries its struct's line). A directive is not a
/// token, so this changes no meaning for the source the emitter produces
/// (tested: identical WGSL and SPIR-V, including a struct body, a helper, a lowered
/// `match` and struct literals). The claim covers emitter-produced source only; a
/// hand-written `slang_source` with a table gets directives wherever its table
/// says, such as inside a multi-line comment or a macro continuation, where they
/// could change meaning.
///
/// With no file, no table, or a file name that cannot sit in a directive, the
/// copy equals `source` byte for byte (hand-written descriptors, probes). The
/// copy is a compile-time input only: exports, `KernelDescriptor::slang_source`
/// and the cache identity never see it.
#[cfg_attr(
    not(any(test, feature = "slang-runtime")),
    expect(
        dead_code,
        reason = "only the run-time bridge and a unit test of `artifact` call it"
    )
)]
pub(crate) fn compile_copy(source: &str, origin: &SourceOrigin) -> String {
    if origin.lines.is_empty() || !directive_file_is_usable(origin.file) {
        return source.to_owned();
    }
    let mut copy = String::with_capacity(source.len() + source.len() / 4);
    let mut previous_attributed = false;
    for (index, text) in source.split_inclusive('\n').enumerate() {
        let Ok(number) = u32::try_from(index + 1) else {
            copy.push_str(text);
            continue;
        };
        // A record for Rust line 0 is malformed (lines are 1-based): unattributed.
        match origin.rust_line_at(number).filter(|line| *line > 0) {
            Some(rust_line) => {
                copy.push_str(&format!("#line {rust_line} \"{}\"\n", origin.file));
                previous_attributed = true;
            }
            None => {
                if previous_attributed {
                    copy.push_str(&format!("#line {number} \"{KERNEL_FILE}\"\n"));
                }
                previous_attributed = false;
            }
        }
        copy.push_str(text);
    }
    copy
}

#[cfg(feature = "slang-runtime")]
/// Write what the compiler reads into `directory` as `kernel.slang` and, when
/// that differs from `source`, the clean text beside it as `kernel.export.slang`,
/// so the directory still shows the readable artifact. Returns the copy; it
/// carries `#line` directives exactly when it differs from `source`.
pub(crate) fn write_compile_inputs(
    directory: &Path,
    source: &str,
    origin: &SourceOrigin,
) -> Result<String, std::io::Error> {
    let copy = compile_copy(source, origin);
    fs::write(directory.join(KERNEL_FILE), &copy)?;
    if copy != source {
        fs::write(directory.join(EXPORT_FILE), source)?;
    }
    Ok(copy)
}

#[cfg(feature = "slang-runtime")]
/// Append the originating Rust kernel and construct (DX19), and the Rust source
/// location when the descriptor carries a line table, to a Slang failure.
/// Shared by every path that compiles a kernel, so none reports less than the
/// others. A diagnostic with no usable error location is returned unchanged.
pub(crate) fn annotate_diagnostics(
    origin: &SourceOrigin,
    slang_source: &str,
    source_map: &SourceMap,
    diagnostics: &str,
) -> String {
    let located = locate(origin, slang_source, source_map, diagnostics);
    // The macro renames the few user names that slangc cannot take as `__gust_r_<name>` (M6 plan 2.5). The location
    // above was found on the original text. naga and wgpu messages are not touched: they name slangc's mangled
    // `_gust_r_<name>_N`.
    let mut annotated = strip_rename_prefix(diagnostics);
    if let Some(segment) = located.segment {
        annotated.push_str(&format!(
            "\noriginating Rust kernel: {}\noriginating Rust construct: {} #{}",
            segment.kernel, segment.construct, segment.ordinal
        ));
    }
    if let Some(rust_line) = located.rust_line {
        annotated.push_str(&format!(
            "\noriginating Rust source: {}:{rust_line}",
            origin.file
        ));
    }
    if located.rust_location {
        annotated.push_str("\n(column is in the generated Slang)");
    }
    annotated
}

#[cfg(feature = "slang-runtime")]
/// The prefix the macro puts in front of a renamed user identifier (`gust-macros`, `slang/collisions.rs`; E0901 denies
/// the `__gust_` family to users). `crates/gust/tests/diagnostics.rs` pins that the two agree.
const RENAME_PREFIX: &str = "__gust_r_";

#[cfg(feature = "slang-runtime")]
/// Gives a slangc message the Rust spelling of the names the macro renamed: deletes [`RENAME_PREFIX`] where an emitted
/// renamed name starts, that is at a word boundary (not after a letter, digit or underscore) and in front of a name.
/// The macro only ever writes the prefix at the start of a name, so this undoes exactly its renames. A user name that
/// merely contains the text, such as `a__gust_r_b` (E0901 only denies a name that starts with `__gust_`), is left
/// alone.
fn strip_rename_prefix(text: &str) -> String {
    let is_word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let bytes = text.as_bytes();
    let mut stripped = String::with_capacity(text.len());
    let mut copied = 0;
    let mut search = 0;
    while let Some(found) = text[search..].find(RENAME_PREFIX) {
        let start = search + found;
        let end = start + RENAME_PREFIX.len();
        let at_boundary = start == 0 || !is_word(bytes[start - 1]);
        let before_name = bytes
            .get(end)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_');
        if at_boundary && before_name {
            stripped.push_str(&text[copied..start]);
            copied = end;
        }
        search = end;
    }
    stripped.push_str(&text[copied..]);
    stripped
}

#[cfg(feature = "slang-runtime")]
/// Where a failure came from, as far as the descriptor can tell.
struct Located {
    /// The construct segment of the failing Slang line, if it has one.
    segment: Option<SourceMapSegment>,
    /// The Rust line, independent of the segment: a helper or struct above the
    /// first `// @rust kernel:` marker has a line record but no construct.
    rust_line: Option<u32>,
    /// The compiler printed the Rust location itself (a `#line` directive took
    /// effect), so its column is a Slang column.
    rust_location: bool,
}

#[cfg(feature = "slang-runtime")]
fn locate(
    origin: &SourceOrigin,
    slang_source: &str,
    source_map: &SourceMap,
    diagnostics: &str,
) -> Located {
    let none = Located {
        segment: None,
        rust_line: None,
        rust_location: false,
    };
    let line_count = slang_source.lines().count();
    match parse_diagnostic_location(diagnostics, origin.file) {
        Some(DiagnosticLocation::Slang(line)) if line <= line_count => Located {
            segment: source_map.origin_at(line),
            rust_line: u32::try_from(line)
                .ok()
                .and_then(|line| origin.rust_line_at(line))
                .filter(|_| !origin.file.is_empty()),
            rust_location: false,
        },
        Some(DiagnosticLocation::Rust(rust_line)) => Located {
            // Many Slang lines can come from one Rust line; the lowest is the
            // outermost construct that starts on it, deterministically.
            segment: first_slang_line_of(origin, rust_line)
                .map(|line| usize::try_from(line).unwrap_or(usize::MAX))
                .filter(|line| *line <= line_count)
                .and_then(|line| source_map.origin_at(line)),
            rust_line: Some(rust_line),
            rust_location: true,
        },
        _ => none,
    }
}

#[cfg(feature = "slang-runtime")]
/// The lowest Slang line of any record generated from `rust_line`.
fn first_slang_line_of(origin: &SourceOrigin, rust_line: u32) -> Option<u32> {
    origin
        .lines
        .iter()
        .filter(|span| span.rust_line == rust_line)
        .map(|span| span.slang_start)
        .min()
}

#[cfg(feature = "slang-runtime")]
/// [`annotate_diagnostics`] for a kernel that is compiled without the sidecar
/// map, such as the reflection helper path.
pub(crate) fn annotate_kernel_diagnostics(kernel: &KernelDescriptor, diagnostics: &str) -> String {
    let map = SourceMap::from_slang(kernel.slang_source);
    annotate_diagnostics(&kernel.source, kernel.slang_source, &map, diagnostics)
}

/// Map a `slangc` diagnostic to a generated construct segment. The sidecar
/// `kernel.map.json` is written for inspection and is never parsed back; the
/// in-memory map built from the same source does the lookup. A line outside
/// every construct segment maps to the nearest preceding kernel marker, so a
/// missing or unreadable segment degrades to kernel attribution. A location in
/// the Rust file (a `#line` directive took effect) maps through the lowest Slang
/// line generated from that Rust line.
#[cfg(test)]
fn map_slang_diagnostic(
    source_map: &SourceMap,
    origin: &SourceOrigin,
    slang_source: &str,
    diagnostic: &str,
) -> Option<SourceMapSegment> {
    locate(origin, slang_source, source_map, diagnostic).segment
}

#[cfg(feature = "slang-runtime")]
/// Where a compiler error points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiagnosticLocation {
    /// A line of the generated Slang (`kernel.slang`).
    Slang(usize),
    /// A line of the kernel's Rust file, reported through a `#line` directive.
    Rust(u32),
}

#[cfg(feature = "slang-runtime")]
/// Whether `path` names the compile copy or the kernel's Rust file. The Rust file
/// must match `rust_file` exactly: the compiler echoes the directive's string
/// verbatim, and the directive only ever holds `/` separators. Any other file
/// stays unattributed. The same eligibility guard as injection applies
/// ([`directive_file_is_usable`]): a recorded file that gets no directives, such as
/// one named `kernel.slang`, can never be the origin of a location, so its name
/// cannot turn a Slang location into a Rust one.
fn classify_location(path: &str, line: usize, rust_file: &str) -> Option<DiagnosticLocation> {
    if directive_file_is_usable(rust_file) && path == rust_file {
        u32::try_from(line).ok().map(DiagnosticLocation::Rust)
    } else if path.rsplit(['/', '\\']).next()? == KERNEL_FILE {
        Some(DiagnosticLocation::Slang(line))
    } else {
        None
    }
}

#[cfg(feature = "slang-runtime")]
/// Select an error in this compilation unit, not an earlier warning or a filename
/// that merely ends with `kernel.slang`. Both `(line)` and `(line,column)` occur
/// in compiler diagnostics. A location in `rust_file` (empty for none) is a Rust
/// location. Unrecognized locations remain unmapped.
fn parse_diagnostic_location(diagnostic: &str, rust_file: &str) -> Option<DiagnosticLocation> {
    let mut error_header = false;
    diagnostic.lines().find_map(|diagnostic| {
        let diagnostic = diagnostic.trim();
        // Current Slang prints an error header followed by a separate arrow
        // location. Track severity so an earlier warning cannot claim the error.
        if diagnostic.starts_with("error[") || diagnostic.starts_with("error:") {
            error_header = true;
        } else if ["warning[", "warning:", "note:", "help:"]
            .iter()
            .any(|prefix| diagnostic.starts_with(prefix))
        {
            error_header = false;
        }
        if error_header && let Some(location) = diagnostic.strip_prefix("--> ") {
            let (path_line, column) = location.rsplit_once(':')?;
            let (path, line) = path_line.rsplit_once(':')?;
            positive_coordinate(column)?;
            return classify_location(path, positive_coordinate(line)?, rust_file);
        }
        let (location, message) = diagnostic.trim().split_once("):")?;
        let message = message.trim_start();
        let message = message.strip_prefix("fatal ").unwrap_or(message);
        if !message.starts_with("error:") && !message.starts_with("error ") {
            return None;
        }
        let (path, position) = location.rsplit_once('(')?;
        let mut coordinates = position.split(',');
        let line = positive_coordinate(coordinates.next()?)?;
        if let Some(column) = coordinates.next() {
            positive_coordinate(column)?;
        }
        if coordinates.next().is_some() {
            return None;
        }
        classify_location(path, line, rust_file)
    })
}

#[cfg(feature = "slang-runtime")]
fn positive_coordinate(value: &str) -> Option<usize> {
    let value = value.trim();
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok().filter(|value| *value > 0)
}

/// A directory under the system temporary directory for one compiler run. It is removed with its contents when dropped.
#[cfg(feature = "slang-runtime")]
pub struct TemporaryDirectory(PathBuf);

#[cfg(feature = "slang-runtime")]
impl TemporaryDirectory {
    /// The directory's path.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Create a new, empty directory whose name is unique to this process. Fails with the I/O error of the creation, or
    /// with `AlreadyExists` when 128 names were taken.
    pub fn create() -> Result<Self, std::io::Error> {
        for _ in 0..128 {
            let id = NEXT_INVOCATION.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("gust-slang-{}-{id}", std::process::id()));
            if let Some(directory) = Self::try_create(path)? {
                return Ok(directory);
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not reserve a Slang temporary directory after 128 attempts",
        ))
    }

    fn try_create(path: PathBuf) -> Result<Option<Self>, std::io::Error> {
        match fs::create_dir(&path) {
            Ok(()) => Ok(Some(Self(path))),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
            Err(error) => Err(error),
        }
    }
}

#[cfg(feature = "slang-runtime")]
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        // Only this exclusively created child is owned; never remove a collided path.
        // Cleanup is best effort and runs on every return path, including I/O errors.
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Write the Slang source of `kernel` to `path`. Fails with [`Error::Io`] when the file cannot be written.
pub fn write_source(kernel: &KernelDescriptor, path: impl AsRef<Path>) -> Result<(), Error> {
    fs::write(path, kernel.slang_source).map_err(Error::Io)
}

#[cfg(feature = "slang-runtime")]
/// Machine-readable record of whether the installed Slang compiler can emit a
/// target. Capability claims must follow this evidence, not assumption.
#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct TargetProbe {
    /// The probed target.
    pub target: Target,
    /// Whether `slangc` compiled the known-good probe kernel to this target.
    pub supported: bool,
    /// Evidence: a success marker, or the compiler diagnostic on failure.
    pub detail: String,
}

#[cfg(feature = "slang-runtime")]
/// A minimal, known-good compute kernel used only to probe target capability.
const PROBE_KERNEL: &str = concat!(
    "[shader(\"compute\")]\n",
    "[numthreads(1, 1, 1)]\n",
    "void gpu_probe_entry(uint3 id : SV_DispatchThreadID) {}\n",
);

#[cfg(feature = "slang-runtime")]
/// Probe whether `slangc` can emit `target` by compiling a known-good kernel.
///
/// This is capability evidence, not a correctness test: it records whether the
/// target is supported and, if not, why, so downstream claims can cite it.
#[doc(hidden)]
pub fn probe(target: Target) -> TargetProbe {
    match compile_source(PROBE_KERNEL, "gpu_probe_entry", target) {
        Ok(_) => TargetProbe {
            target,
            supported: true,
            detail: "compiled".to_string(),
        },
        Err(error) => TargetProbe {
            target,
            supported: false,
            detail: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "slang-runtime")]
    use crate::toolchain::{BUSY_FIRST_BACKOFF, BUSY_TRIES, SLANG_PIN, retry_busy};
    #[cfg(feature = "slang-runtime")]
    use std::time::Duration;

    /// A `slangc` of another version is a typed error that names the executable, both versions and the install
    /// command (D72's follow-up); the pin's own text names the pin.
    #[test]
    fn a_version_mismatch_names_both_versions_and_the_install_command() {
        let error = Error::SlangVersionMismatch {
            slangc: PathBuf::from("/vulkan/Bin/slangc"),
            found: "2026.13.1".to_owned(),
            pinned: "2026.20",
        };
        let text = error.to_string();
        assert!(
            text.contains("/vulkan/Bin/slangc")
                && text.contains("Slang `2026.13.1`")
                && text.contains("not the pinned 2026.20")
                && text.contains("cargo gust slang install"),
            "{text}"
        );
        // None of the bridge markers of check-precompiled (E2) may be spelled by this ungated text.
        for marker in ["gust-slang-reflect", "kernel.slang", "kernel.artifact"] {
            assert!(!text.contains(marker), "{marker}");
        }
    }

    /// The verdict of the version check becomes the typed error: a pin passes, a mismatch carries the executable, what
    /// it printed and the pin, and a `slangc` that cannot answer is a toolchain error with the reason.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_version_verdict_maps_to_the_typed_error() {
        use crate::toolchain::VersionError;

        assert!(pin_verdict(Ok(())).is_ok());
        let mismatch = VersionError::Mismatch {
            slangc: PathBuf::from("/s/slangc"),
            found: "2026.13.1".to_owned(),
        };
        match pin_verdict(Err(mismatch)) {
            Err(Error::SlangVersionMismatch {
                slangc,
                found,
                pinned,
            }) => {
                assert_eq!(slangc, PathBuf::from("/s/slangc"));
                assert_eq!(found, "2026.13.1");
                assert_eq!(pinned, SLANG_PIN);
            }
            other => panic!("{other:?}"),
        }
        for unanswered in [
            VersionError::Run("cannot run `/s/slangc`: nope".to_owned()),
            VersionError::Output("unexpected `-version` output: `x`".to_owned()),
        ] {
            let text = unanswered.to_string();
            match pin_verdict(Err(unanswered)) {
                Err(Error::Toolchain(reason)) => assert_eq!(reason, text),
                other => panic!("{other:?}"),
            }
        }
    }

    /// A toolchain problem is not printed as an I/O failure, and its text keeps the reason.
    #[test]
    fn a_toolchain_error_names_its_reason_and_is_not_an_io_failure() {
        let text =
            Error::Toolchain("GUST_SLANG_HOME must be an absolute path".to_owned()).to_string();
        assert!(
            text.contains("GUST_SLANG_HOME must be an absolute path"),
            "{text}"
        );
        assert!(!text.contains("I/O"), "{text}");
        let missing = Error::CompilerNotFound.to_string();
        assert!(missing.contains("cargo gust slang install"), "{missing}");
        for marker in ["gust-slang-reflect", "kernel.slang", "kernel.artifact"] {
            assert!(
                !missing.contains(marker) && !text.contains(marker),
                "{marker}"
            );
        }
    }

    /// A fake `slangc` in `directory` whose `-version` prints `version` (a script, so it runs wherever the test does;
    /// the temporary directory must allow execution).
    #[cfg(feature = "slang-runtime")]
    fn fake_slangc(directory: &Path, version: &str) -> PathBuf {
        fs::create_dir_all(directory).unwrap();
        let (path, script) = if cfg!(windows) {
            (
                directory.join("slangc.cmd"),
                format!("@echo off\r\necho {version}\r\n"),
            )
        } else {
            (
                directory.join("slangc"),
                format!("#!/bin/sh\necho {version}\n"),
            )
        };
        fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// A `slangc` that prints another Slang is refused with both versions, and the refusal is not remembered (an
    /// install of the pin in the meantime is picked up); the pin passes.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn a_slangc_of_another_slang_is_refused_and_the_pin_passes() {
        let directory = TemporaryDirectory::create().unwrap();
        let other = fake_slangc(&directory.path().join("other"), "2026.13.1");
        for _ in 0..2 {
            match verified_slangc(other.clone()) {
                Err(Error::SlangVersionMismatch {
                    slangc,
                    found,
                    pinned,
                }) => {
                    assert_eq!(slangc, other);
                    assert_eq!(found, "2026.13.1");
                    assert_eq!(pinned, SLANG_PIN);
                }
                result => panic!("{result:?}"),
            }
        }
        let pin = fake_slangc(&directory.path().join("pin"), SLANG_PIN);
        assert_eq!(verified_slangc(pin.clone()).unwrap(), pin);
    }

    /// An unusable `GUST_SLANG_HOME` is a toolchain error naming the variable, not an I/O failure.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn an_unusable_slang_home_is_a_toolchain_error() {
        let env = crate::toolchain::Env {
            slang_home: Some("relative/dir".into()),
            ..Default::default()
        };
        match pinned_slangc_in(&env) {
            Err(Error::Toolchain(reason)) => {
                assert!(reason.contains("GUST_SLANG_HOME"), "{reason}")
            }
            result => panic!("{result:?}"),
        }
    }

    /// The compile asks for the pinned `slangc` before it launches anything: with only a `slangc` of another Slang on
    /// the `PATH` of the environment it is given, `compile_at_in` fails with the mismatch (a revert to a bare `slangc`
    /// would run the fake and fail otherwise). Unix only: the lookup needs a file called `slangc`, and on Windows that
    /// is an `.exe`, which a script cannot be.
    #[cfg(all(feature = "slang-runtime", unix))]
    #[test]
    fn the_compile_asks_for_the_pinned_slangc_before_it_launches_anything() {
        let directory = TemporaryDirectory::create().unwrap();
        let bin = directory.path().join("bin");
        fake_slangc(&bin, "2026.13.1");
        let env = crate::toolchain::Env {
            path: Some(bin.into_os_string()),
            ..Default::default()
        };
        assert!(matches!(
            pinned_slangc_in(&env),
            Err(Error::SlangVersionMismatch { ref found, .. }) if found == "2026.13.1"
        ));
        let work = directory.path().join("work");
        fs::create_dir_all(&work).unwrap();
        let result = compile_at_in(
            &env,
            &work,
            "[shader(\"compute\")] void main() {}",
            "main",
            &SourceOrigin::UNKNOWN,
            Shape::Grid,
            Target::Wgsl,
        );
        assert!(
            matches!(result, Err(Error::SlangVersionMismatch { ref found, .. }) if found == "2026.13.1"),
            "{result:?}"
        );
    }

    fn mapped_kernel(source: &str, diagnostic: &str) -> Option<String> {
        map_slang_diagnostic(
            &SourceMap::from_slang(source),
            &SourceOrigin::UNKNOWN,
            source,
            diagnostic,
        )
        .map(|origin| origin.kernel)
    }

    /// The compile path names the matrix layout (the reflection helper pins the same one natively).
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_compile_command_pins_the_column_major_matrix_layout() {
        for (target, spirv_flag) in [
            (Target::Wgsl, None),
            (Target::Spirv, Some("-fvk-use-entrypoint-name")),
        ] {
            for (shape, stage) in [(Shape::Grid, "compute"), (Shape::Pixel, "fragment")] {
                let command = slangc_command(
                    Path::new("slangc"),
                    Path::new("kernel.slang"),
                    target,
                    "gpu_m_k",
                    shape,
                );
                let mut expected = vec![
                    "kernel.slang",
                    "-target",
                    target.slang_name(),
                    "-entry",
                    "gpu_m_k",
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

    /// The stage is the shape's, end to end: a fragment entry point compiles as `pixel` to both targets, and the same
    /// text is refused as `grid` (a stage mismatch is slangc's error, which `-stage compute` used to make for every
    /// shape but one).
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn a_fragment_entry_point_compiles_under_the_pixel_shape_only() {
        const FRAGMENT: &str = concat!(
            "struct Out\n{\n    float4 t0 : SV_Target0;\n};\n\n",
            "[shader(\"fragment\")]\n",
            "Out gpu_k_run(float4 pos : SV_Position)\n",
            "{\n    Out o;\n    o.t0 = float4(pos.x, 0.0, 0.0, 1.0);\n    return o;\n}\n",
        );
        for target in [Target::Wgsl, Target::Spirv] {
            let bytes = compile_source_as(FRAGMENT, "gpu_k_run", Shape::Pixel, target)
                .unwrap_or_else(|error| panic!("{target:?}: {error}"));
            assert!(!bytes.is_empty(), "{target:?}");
            if target == Target::Wgsl {
                let text = String::from_utf8(bytes).unwrap();
                assert!(text.contains("@fragment"), "{text}");
            }
            assert!(
                compile_source(FRAGMENT, "gpu_k_run", target).is_err(),
                "{target:?}: the default (grid) stage must not accept a fragment entry point"
            );
        }
    }

    /// A busy executable is tried again with doubling waits, and the last busy error is returned after the bound.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn a_busy_executable_is_retried_a_bounded_number_of_times() {
        let busy = |text: String| std::io::Error::new(std::io::ErrorKind::ExecutableFileBusy, text);
        let mut calls = 0;
        let mut waits = Vec::new();
        let result: std::io::Result<()> = retry_busy(
            BUSY_TRIES,
            BUSY_FIRST_BACKOFF,
            |wait| waits.push(wait),
            || {
                calls += 1;
                Err(busy(format!("busy {calls}")))
            },
        );
        let error = result.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::ExecutableFileBusy);
        assert_eq!(
            error.to_string(),
            "busy 5",
            "the last busy error is the one returned"
        );
        assert_eq!(calls, 5);
        let millis: Vec<u128> = waits.iter().map(Duration::as_millis).collect();
        assert_eq!(millis, [10, 20, 40, 80]);

        // A launch that stops being busy is returned as it is, with no further wait.
        let mut calls = 0;
        let mut waited = 0;
        let result = retry_busy(
            BUSY_TRIES,
            BUSY_FIRST_BACKOFF,
            |_| waited += 1,
            || {
                calls += 1;
                if calls < 3 {
                    Err(busy(String::new()))
                } else {
                    Ok(calls)
                }
            },
        );
        assert_eq!((result.unwrap(), waited), (3, 2));
    }

    /// Only the busy error is retried: a missing executable, or a success, is returned at once.
    #[cfg(feature = "slang-runtime")]
    #[test]
    fn an_error_other_than_busy_is_not_retried() {
        let mut calls = 0;
        let result: std::io::Result<()> = retry_busy(
            BUSY_TRIES,
            BUSY_FIRST_BACKOFF,
            |_| panic!("no wait before the first result is final"),
            || {
                calls += 1;
                Err(std::io::ErrorKind::NotFound.into())
            },
        );
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::NotFound);
        assert_eq!(calls, 1);

        let mut calls = 0;
        let result = retry_busy(
            BUSY_TRIES,
            BUSY_FIRST_BACKOFF,
            |_| panic!("a success is not waited on"),
            || {
                calls += 1;
                Ok(())
            },
        );
        assert!(result.is_ok());
        assert_eq!(calls, 1);
    }

    #[cfg(feature = "slang-runtime")]
    #[test]
    fn the_rename_prefix_is_stripped_only_where_a_renamed_name_starts() {
        assert_eq!(
            strip_rename_prefix(
                "error: undefined identifier '__gust_r_new' in (__gust_r_class.__gust_r_This)"
            ),
            "error: undefined identifier 'new' in (class.This)"
        );
        // A user name that contains the text is not a renamed name: it keeps every character.
        assert_eq!(strip_rename_prefix("a__gust_r_b"), "a__gust_r_b");
        assert_eq!(
            strip_rename_prefix("1__gust_r_b x_ __gust_r_"),
            "1__gust_r_b x_ __gust_r_"
        );
        assert_eq!(strip_rename_prefix("no prefix here"), "no prefix here");
        // slangc's mangled form in naga and wgpu messages is not touched.
        assert_eq!(strip_rename_prefix("_gust_r_new_0"), "_gust_r_new_0");
    }

    #[test]
    fn never_claims_or_removes_an_existing_directory() {
        let owner = TemporaryDirectory::create().unwrap();
        let path = owner.0.clone();
        fs::write(path.join("sentinel"), "retain me").unwrap();
        assert!(
            TemporaryDirectory::try_create(path.clone())
                .unwrap()
                .is_none()
        );
        assert_eq!(
            fs::read_to_string(path.join("sentinel")).unwrap(),
            "retain me"
        );
        drop(owner);
        assert!(!path.exists());
    }

    #[test]
    fn compiler_failure_reports_diagnostics_and_cleans_owned_files() {
        let directory = TemporaryDirectory::create().unwrap();
        let path = directory.0.clone();
        let error = compile_in_directory(
            directory,
            "this is not Slang;",
            "missing",
            &SourceOrigin::UNKNOWN,
            Shape::Grid,
            Target::Wgsl,
        )
        .unwrap_err();
        assert!(matches!(error, Error::CompilationFailed { .. }), "{error}");
        assert!(error.to_string().contains("compiler exit:"));
        assert!(!path.exists());
    }

    #[test]
    fn early_io_failure_also_cleans_owned_files() {
        let directory = TemporaryDirectory::create().unwrap();
        let path = directory.0.clone();
        fs::create_dir(path.join("kernel.slang")).unwrap();
        assert!(matches!(
            compile_in_directory(
                directory,
                "",
                "missing",
                &SourceOrigin::UNKNOWN,
                Shape::Grid,
                Target::Wgsl,
            ),
            Err(Error::Io(_))
        ));
        assert!(!path.exists());
    }

    #[test]
    fn probe_records_wgsl_capability() {
        let probe = probe(Target::Wgsl);
        assert!(probe.supported, "WGSL should compile: {}", probe.detail);
    }

    #[test]
    fn probe_records_spirv_capability() {
        let probe = probe(Target::Spirv);
        assert!(probe.supported, "SPIR-V should compile: {}", probe.detail);
    }

    #[test]
    fn maps_slang_diagnostic_to_originating_rust_kernel() {
        let source = concat!(
            "// Generated by GUST\n",
            "// @rust kernel: vector_add::add\n",
            "[shader(\"compute\")]\n",
            "[numthreads(64, 1, 1)]\n",
            "void gpu_vector_add_add(uint3 id : SV_DispatchThreadID) {\n",
            "    int x = undefined_symbol;\n",
            "}\n",
        );
        let diagnostic = "kernel.slang(6): error: expected an expression\n";
        let origin = map_slang_diagnostic(
            &SourceMap::from_slang(source),
            &SourceOrigin::UNKNOWN,
            source,
            diagnostic,
        )
        .expect("should map");
        assert_eq!(origin.kernel, "vector_add::add");
        assert_eq!(origin.construct, "local");
    }

    /// Output shaped like the emitter's: a Result helper, the resource prologue
    /// and a lowered `match`. `scrutinee` and `arm` inject the failing text.
    fn match_kernel(signature: &str, scrutinee: &str, arm: &str) -> String {
        format!(
            concat!(
                "struct R {{ bool isOk; uint value; }};\n",
                "R pick() {{ R r; r.isOk = true; r.value = uint(1); return r; }}\n",
                "RWStructuredBuffer<uint> values;\n",
                "\n",
                "// @rust kernel: example::broken\n",
                "[shader(\"compute\")]\n",
                "[numthreads(1, 1, 1)]\n",
                "void broken({signature})\n",
                "{{\n",
                "    uint __gpu_len_values;\n",
                "    uint __gpu_stride_values;\n",
                "    values.GetDimensions(__gpu_len_values, __gpu_stride_values);\n",
                "    var total = uint(0);\n",
                "    // @gust construct: match #1\n",
                "    var __gust_match_1 = {scrutinee};\n",
                "    if (__gust_match_1.isOk)\n",
                "    {{\n",
                "        var h = __gust_match_1.value;\n",
                "        total = ({arm});\n",
                "    }}\n",
                "    else\n",
                "    {{\n",
                "        var e = __gust_match_1.value;\n",
                "        total = (total + e);\n",
                "    }}\n",
                "    values[id.x] = total;\n",
                "}}\n",
            ),
            signature = signature,
            scrutinee = scrutinee,
            arm = arm,
        )
    }

    const VALID_SIGNATURE: &str = "uint3 id : SV_DispatchThreadID";

    fn failure_diagnostics(source: &str, target: Target) -> String {
        let error = compile_source(source, "broken", target).unwrap_err();
        let Error::CompilationFailed { diagnostics, .. } = error else {
            panic!("expected actual compiler failure for {target:?}: {error}");
        };
        diagnostics
    }

    #[test]
    fn sidecar_map_file_is_written_before_slangc_runs_and_has_the_documented_schema() {
        let source = concat!(
            "RWStructuredBuffer<uint> values;\n",
            "\n",
            "// @rust kernel: example::ok\n",
            "[shader(\"compute\")]\n",
            "[numthreads(1, 1, 1)]\n",
            "void ok(uint3 id : SV_DispatchThreadID)\n",
            "{\n",
            "    uint __gpu_len_values;\n",
            "    uint __gpu_stride_values;\n",
            "    values.GetDimensions(__gpu_len_values, __gpu_stride_values);\n",
            "    var i = id.x;\n",
            "    if ((i < __gpu_len_values))\n",
            "    {\n",
            "        values[i] = uint(1);\n",
            "    }\n",
            "}\n",
        );
        let expected = concat!(
            "{\n  \"schema_version\": 2,\n  \"rust_file\": null,\n  \"line_directives\": false,\n  \"segments\": [\n",
            "    {\"slang_start\":3,\"slang_end\":16,\"kernel\":\"example::ok\",\"construct\":\"kernel\",\"ordinal\":0},\n",
            "    {\"slang_start\":8,\"slang_end\":10,\"kernel\":\"example::ok\",\"construct\":\"prologue\",\"ordinal\":0},\n",
            "    {\"slang_start\":11,\"slang_end\":11,\"kernel\":\"example::ok\",\"construct\":\"local\",\"ordinal\":0},\n",
            "    {\"slang_start\":12,\"slang_end\":15,\"kernel\":\"example::ok\",\"construct\":\"conditional\",\"ordinal\":1},\n",
            "    {\"slang_start\":14,\"slang_end\":14,\"kernel\":\"example::ok\",\"construct\":\"assignment\",\"ordinal\":2}\n",
            "  ]\n}\n",
        );
        // A successful compile and a failing one both leave the sidecar behind
        // until the owner removes the directory.
        let directory = TemporaryDirectory::create().unwrap();
        compile_at(
            directory.path(),
            source,
            "ok",
            &SourceOrigin::UNKNOWN,
            Shape::Grid,
            Target::Wgsl,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("kernel.map.json")).unwrap(),
            expected
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("kernel.slang")).unwrap(),
            source
        );
        // No directives, so there is no second copy of the same text.
        assert!(!directory.path().join("kernel.export.slang").exists());

        let failing = TemporaryDirectory::create().unwrap();
        let broken = match_kernel(VALID_SIGNATURE, "undefined_symbol", "h");
        assert!(
            compile_at(
                failing.path(),
                &broken,
                "broken",
                &SourceOrigin::UNKNOWN,
                Shape::Grid,
                Target::Wgsl
            )
            .is_err()
        );
        let sidecar = fs::read_to_string(failing.path().join("kernel.map.json")).unwrap();
        assert_eq!(
            sidecar,
            SourceMap::from_slang(&broken).json(&SourceOrigin::UNKNOWN, false)
        );
        assert!(sidecar.contains("\"construct\":\"match\""), "{sidecar}");
    }

    #[test]
    fn sidecar_map_segments_follow_nested_blocks() {
        let source = concat!(
            "// @rust kernel: example::run\n",
            "void run(uint3 id : SV_DispatchThreadID)\n",
            "{\n",
            "    var value = id.x;\n",
            "    if (value > 0)\n",
            "    {\n",
            "        value = value - 1;\n",
            "    }\n",
            "}\n",
        );
        let mut kinds: Vec<_> = SourceMap::from_slang(source)
            .segments()
            .iter()
            .map(|segment| (segment.construct, segment.start_line, segment.end_line))
            .collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            vec![
                ("assignment", 7, 7),
                ("conditional", 5, 8),
                ("kernel", 1, 9),
                ("local", 4, 4),
            ]
        );
    }

    #[test]
    fn failure_inside_a_lowered_match_reports_the_match_on_each_target() {
        let source = match_kernel(VALID_SIGNATURE, "undefined_symbol", "h");
        for target in [Target::Wgsl, Target::Spirv] {
            let diagnostics = failure_diagnostics(&source, target);
            assert!(diagnostics.contains("undefined_symbol"), "{diagnostics}");
            assert!(
                diagnostics.contains("originating Rust kernel: example::broken"),
                "{target:?}: {diagnostics}"
            );
            // `total` is local #0, so the match is construct #1.
            assert!(
                diagnostics.contains("originating Rust construct: match #1"),
                "{target:?}: {diagnostics}"
            );
        }
    }

    #[test]
    fn match_label_uses_the_emitted_marker_number_not_the_map_ordinal() {
        // The map would count this match as construct #1; the emitter numbered it 7.
        let source = match_kernel(VALID_SIGNATURE, "undefined_symbol", "h")
            .replace("match_1", "match_7")
            .replace("match #1", "match #7");
        let diagnostics = failure_diagnostics(&source, Target::Wgsl);
        assert!(diagnostics.contains("undefined_symbol"), "{diagnostics}");
        assert!(
            diagnostics.contains("originating Rust construct: match #7"),
            "{diagnostics}"
        );
    }

    #[test]
    fn failure_in_a_user_statement_inside_a_match_arm_reports_that_statement() {
        let source = match_kernel(VALID_SIGNATURE, "pick()", "total + undefined_symbol");
        let diagnostics = failure_diagnostics(&source, Target::Wgsl);
        assert!(diagnostics.contains("undefined_symbol"), "{diagnostics}");
        assert!(
            diagnostics.contains("originating Rust construct: assignment #2"),
            "{diagnostics}"
        );
    }

    #[test]
    fn failure_on_an_unsegmented_kernel_line_falls_back_to_the_kernel() {
        let source = match_kernel("unknown_type id : SV_DispatchThreadID", "pick()", "h");
        let diagnostics = failure_diagnostics(&source, Target::Wgsl);
        assert!(
            diagnostics.contains("originating Rust kernel: example::broken"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust construct: kernel #0"),
            "{diagnostics}"
        );
    }

    #[test]
    fn maps_slang_diagnostic_before_any_marker_returns_none() {
        let source = concat!(
            "// Generated by GUST\n",
            "[shader(\"compute\")]\n",
            "void gpu_entry(uint3 id : SV_DispatchThreadID) {\n",
            "    int x = undefined_symbol;\n",
            "}\n",
        );
        let diagnostic = "kernel.slang(3): error: expected an expression\n";
        assert!(mapped_kernel(source, diagnostic).is_none());
    }

    #[test]
    fn diagnostic_mapping_rejects_unrelated_files_and_invalid_lines() {
        let source = "// @rust kernel: example::first\nvoid first() {}\n";
        for diagnostic in [
            "otherkernel.slang(2): error: bad expression",
            "C:\\generated\\otherkernel.slang(2): error: bad expression",
            "kernel.slang(0): error: bad expression",
            "kernel.slang(3): error: bad expression",
            "kernel.slang(2,0): error: bad expression",
            "kernel.slang(2,1,4): error: bad expression",
        ] {
            assert_eq!(mapped_kernel(source, diagnostic), None, "{diagnostic}");
        }
    }

    #[test]
    fn diagnostic_mapping_accepts_absolute_paths_and_columns() {
        let source = "// @rust kernel: example::first\nvoid first() {}\n";
        for diagnostic in [
            "C:\\generated files\\kernel.slang(2): error 30015: bad expression",
            "/tmp/generated/kernel.slang(2,7): error: bad expression",
            "kernel.slang(2, 7): error: bad expression",
        ] {
            assert_eq!(
                mapped_kernel(source, diagnostic).as_deref(),
                Some("example::first"),
                "{diagnostic}"
            );
        }
    }

    #[test]
    fn diagnostic_mapping_prefers_error_over_warning_and_note() {
        let source = concat!(
            "// @rust kernel: example::first\nvoid first() {}\n",
            "// @rust kernel: example::second\nvoid second() {}\n",
        );
        let diagnostic = concat!(
            "kernel.slang(2): warning 1: first warning\n",
            "kernel.slang(2): note: first note\n",
            "kernel.slang(4,3): error 30015: actual failure\n",
        );
        assert_eq!(
            mapped_kernel(source, diagnostic).as_deref(),
            Some("example::second")
        );
    }

    #[test]
    fn diagnostic_mapping_reads_current_multiline_slang_format() {
        let source = "// @rust kernel: example::first\nvoid first() {}\n// @rust kernel: example::second\nvoid second() {}\n";
        let diagnostics = "warning[W1]: warning\n --> C:\\generated\\kernel.slang:2:1\nnote: extra context\nerror[E30015]: undefined identifier\n --> C:\\generated\\kernel.slang:4:7\n";
        assert_eq!(
            mapped_kernel(source, diagnostics).as_deref(),
            Some("example::second")
        );
        assert_eq!(
            mapped_kernel(
                source,
                &diagnostics.replace("kernel.slang", "otherkernel.slang")
            ),
            None
        );
    }

    #[test]
    fn compiler_failure_maps_originating_kernel_on_each_target() {
        let source = concat!(
            "// @rust kernel: example::broken\n",
            "[shader(\"compute\")]\n[numthreads(1, 1, 1)]\n",
            "void broken(uint3 id : SV_DispatchThreadID) {\n",
            "    int value = undefined_symbol;\n}\n",
        );
        for target in [Target::Wgsl, Target::Spirv] {
            let error = compile_source(source, "broken", target).unwrap_err();
            let Error::CompilationFailed { diagnostics, .. } = error else {
                panic!("expected actual compiler failure for {target:?}: {error}");
            };
            assert!(diagnostics.contains("undefined_symbol"), "{diagnostics}");
            assert!(
                diagnostics.contains("originating Rust kernel: example::broken"),
                "{target:?}: {diagnostics}"
            );
            assert!(
                diagnostics.contains("originating Rust construct: local #0"),
                "{target:?}: {diagnostics}"
            );
        }
    }

    /// Slang lines 1 to 8 of this source: marker, attributes, signature, `{`, three
    /// statements and `}`. The synthetic table below maps lines 6 and 7 only.
    const ANNOTATED_SOURCE: &str = concat!(
        "// @rust kernel: demo::k\n",               // 1
        "[shader(\"compute\")]\n",                  // 2
        "[numthreads(1, 1, 1)]\n",                  // 3
        "void k(uint3 id : SV_DispatchThreadID)\n", // 4
        "{\n",                                      // 5
        "    var a = uint(1);\n",                   // 6
        "    var b = (a + undefined_symbol);\n",    // 7
        "    var c = uint(3);\n",                   // 8
        "}\n",                                      // 9
    );
    const ANNOTATED_LINES: &[crate::LineSpan] = &[
        crate::LineSpan {
            slang_start: 6,
            slang_end: 6,
            rust_line: 21,
        },
        crate::LineSpan {
            slang_start: 7,
            slang_end: 7,
            rust_line: 22,
        },
    ];
    const ANNOTATED_ORIGIN: SourceOrigin = SourceOrigin {
        file: "crates/demo/src/lib.rs",
        lines: ANNOTATED_LINES,
    };

    fn annotate(origin: &SourceOrigin, diagnostics: &str) -> String {
        annotate_diagnostics(
            origin,
            ANNOTATED_SOURCE,
            &SourceMap::from_slang(ANNOTATED_SOURCE),
            diagnostics,
        )
    }

    #[test]
    fn annotation_appends_the_rust_source_line_of_the_innermost_record() {
        for diagnostic in [
            "kernel.slang(7): error: undefined identifier\n",
            "error[E30015]: undefined identifier\n --> C:\\tmp\\kernel.slang:7:17\n",
        ] {
            let annotated = annotate(&ANNOTATED_ORIGIN, diagnostic);
            assert!(annotated.starts_with(diagnostic), "{annotated}");
            assert!(
                annotated.ends_with(concat!(
                    "\noriginating Rust kernel: demo::k",
                    "\noriginating Rust construct: local #1",
                    "\noriginating Rust source: crates/demo/src/lib.rs:22",
                )),
                "{annotated}"
            );
        }
    }

    #[test]
    fn annotation_without_a_record_for_the_line_names_kernel_and_construct_only() {
        // Line 8 is inside the kernel but outside every record.
        let annotated = annotate(
            &ANNOTATED_ORIGIN,
            "kernel.slang(8): error: undefined identifier\n",
        );
        assert!(
            annotated.contains("originating Rust construct: local #2"),
            "{annotated}"
        );
        assert!(
            !annotated.contains("originating Rust source"),
            "{annotated}"
        );
        // A record outside this source's lines never matches a line it does not cover.
        let annotated = annotate(&ANNOTATED_ORIGIN, "kernel.slang(5): error: x\n");
        assert!(
            !annotated.contains("originating Rust source"),
            "{annotated}"
        );
    }

    #[test]
    fn unknown_origin_diagnostics_match_the_pre_line_table_text() {
        let diagnostic = "kernel.slang(7): error: undefined identifier\n";
        let annotated = annotate(&SourceOrigin::UNKNOWN, diagnostic);
        assert_eq!(
            annotated,
            format!(
                "{diagnostic}\noriginating Rust kernel: demo::k\noriginating Rust construct: local #1"
            )
        );
        // A table without a file is as good as none.
        let fileless = SourceOrigin {
            file: "",
            lines: ANNOTATED_LINES,
        };
        assert_eq!(annotate(&fileless, diagnostic), annotated);
    }

    #[test]
    fn a_helper_line_above_the_first_marker_gets_a_rust_line_but_no_kernel() {
        let source = concat!(
            "uint twice(uint x)\n",                 // 1
            "{\n",                                  // 2
            "    return (x + undefined_symbol);\n", // 3
            "}\n",                                  // 4
            "\n",                                   // 5
            "// @rust kernel: demo::k\n",           // 6
            "void k() {}\n",                        // 7
        );
        const LINES: &[crate::LineSpan] = &[
            crate::LineSpan {
                slang_start: 1,
                slang_end: 4,
                rust_line: 30,
            },
            crate::LineSpan {
                slang_start: 3,
                slang_end: 3,
                rust_line: 31,
            },
        ];
        let origin = SourceOrigin {
            file: "src/helpers.rs",
            lines: LINES,
        };
        let diagnostic = "kernel.slang(3): error: undefined identifier\n";
        let annotated =
            annotate_diagnostics(&origin, source, &SourceMap::from_slang(source), diagnostic);
        assert_eq!(
            annotated,
            format!("{diagnostic}\noriginating Rust source: src/helpers.rs:31")
        );
        // Without a table the same failure stays unattributed, as before.
        assert_eq!(
            annotate_diagnostics(
                &SourceOrigin::UNKNOWN,
                source,
                &SourceMap::from_slang(source),
                diagnostic
            ),
            diagnostic
        );
    }

    /// `ANNOTATED_SOURCE` with the directives the table asks for: Rust lines 21
    /// and 22 for Slang lines 6 and 7, then a return to Slang coordinates at 8.
    const ANNOTATED_COPY: &str = concat!(
        "// @rust kernel: demo::k\n",
        "[shader(\"compute\")]\n",
        "[numthreads(1, 1, 1)]\n",
        "void k(uint3 id : SV_DispatchThreadID)\n",
        "{\n",
        "#line 21 \"crates/demo/src/lib.rs\"\n",
        "    var a = uint(1);\n",
        "#line 22 \"crates/demo/src/lib.rs\"\n",
        "    var b = (a + undefined_symbol);\n",
        "#line 8 \"kernel.slang\"\n",
        "    var c = uint(3);\n",
        "}\n",
    );

    #[test]
    fn compile_copy_puts_a_directive_before_each_attributed_line_and_switches_back_once() {
        assert_eq!(
            compile_copy(ANNOTATED_SOURCE, &ANNOTATED_ORIGIN),
            ANNOTATED_COPY
        );
        // Without a trailing newline the last line is still copied exactly.
        let trimmed = ANNOTATED_SOURCE.trim_end_matches('\n');
        assert_eq!(
            compile_copy(trimmed, &ANNOTATED_ORIGIN),
            ANNOTATED_COPY.trim_end_matches('\n')
        );
    }

    #[test]
    fn compile_copy_keeps_every_line_of_a_multi_line_record_on_its_rust_line() {
        // Slang lines 2 to 5 lower one Rust line 40, and line 4 is a nested record
        // of Rust line 41. Both outer lines keep 40, so the compiler never counts on.
        const LINES: &[crate::LineSpan] = &[
            crate::LineSpan {
                slang_start: 2,
                slang_end: 5,
                rust_line: 40,
            },
            crate::LineSpan {
                slang_start: 4,
                slang_end: 4,
                rust_line: 41,
            },
            // Past the end of the source: never matched, never fatal.
            crate::LineSpan {
                slang_start: 9,
                slang_end: 12,
                rust_line: 50,
            },
        ];
        let origin = SourceOrigin {
            file: "src/lib.rs",
            lines: LINES,
        };
        let source = "a\nb\nc\nd\ne\nf\n";
        let expected = concat!(
            "a\n",
            "#line 40 \"src/lib.rs\"\nb\n",
            "#line 40 \"src/lib.rs\"\nc\n",
            "#line 41 \"src/lib.rs\"\nd\n",
            "#line 40 \"src/lib.rs\"\ne\n",
            "#line 6 \"kernel.slang\"\nf\n",
        );
        assert_eq!(compile_copy(source, &origin), expected);
    }

    #[test]
    fn compile_copy_is_the_source_when_the_origin_cannot_be_trusted() {
        const LINE: &[crate::LineSpan] = &[crate::LineSpan {
            slang_start: 1,
            slang_end: 2,
            rust_line: 7,
        }];
        const ZERO: &[crate::LineSpan] = &[crate::LineSpan {
            slang_start: 1,
            slang_end: 2,
            rust_line: 0,
        }];
        let source = "x\ny\n";
        let with = |file: &'static str, lines: &'static [crate::LineSpan]| {
            compile_copy(source, &SourceOrigin { file, lines })
        };
        assert_eq!(compile_copy(source, &SourceOrigin::UNKNOWN), source);
        assert_eq!(with("", LINE), source, "no file");
        assert_eq!(with("src/lib.rs", &[]), source, "no table");
        assert_eq!(with("src/lib.rs", ZERO), source, "Rust lines are 1-based");
        // A string that could end the directive or the line is never embedded.
        for file in [
            "src/\"lib.rs",
            "src\\lib.rs",
            "src/lib.rs\n#define X 1",
            "src/lib rs",
            "src/lib.rs\r",
            "C:/src/lib.rs",
            "src/kernel.slang",
            "Kernel.Slang",
            // Non-ASCII characters that are not Alphabetic: control, separators, a
            // decomposed accent (the macro's whitelist rejects the same).
            "src/\u{85}lib.rs",
            "src/lib\u{2028}.rs",
            "src/\u{a0}lib.rs",
            "src/a\u{301}.rs",
            // Bidi control and zero-width joiner (format characters, not Alphabetic).
            "src/\u{202e}lib.rs",
            "src/a\u{200d}b.rs",
            // The macro's structure: no empty or dot-only component.
            "/src/lib.rs",
            "src//lib.rs",
            "src/lib.rs/",
            "src/./lib.rs",
            "src/../lib.rs",
            "..",
            ".",
        ] {
            assert_eq!(with(file, LINE), source, "{file:?}");
        }
        // Still applied for the whitelisted characters.
        assert_ne!(with("crates/g-ust+1/src/_a.b.rs", LINE), source);
        // Non-ASCII letters are whitelisted as in the macro; slangc echoes them intact.
        assert_eq!(
            with("crates/\u{fc}/\u{65e5}\u{672c}/k.rs", LINE),
            "#line 7 \"crates/\u{fc}/\u{65e5}\u{672c}/k.rs\"\nx\n#line 7 \"crates/\u{fc}/\u{65e5}\u{672c}/k.rs\"\ny\n"
        );
        // A four-byte letter (U+1D49C) and a combining mark that is Alphabetic (U+0345).
        assert_eq!(
            with("crates/\u{1d49c}/\u{345}x/k.rs", LINE),
            "#line 7 \"crates/\u{1d49c}/\u{345}x/k.rs\"\nx\n#line 7 \"crates/\u{1d49c}/\u{345}x/k.rs\"\ny\n"
        );
        // Dots inside a component are fine.
        assert_ne!(with("src/a..b/.hidden.rs", LINE), source);
    }

    #[test]
    fn a_rust_location_in_a_non_ascii_path_is_read_back() {
        // The text slangc printed for a `#line` naming these UTF-8 files (probed with
        // slangc 1.4.357), as `parse_diagnostic_location` receives it.
        for rust in ["dir/\u{fc}/x.rs", "d\u{65e5}\u{672c}/\u{345}\u{1d49c}/x.rs"] {
            let diagnostic = format!(
                "error[E30015]: undefined identifier\n --> {rust}:3:13\n  |\n3 | int x = undefined_thing;\n"
            );
            assert_eq!(
                parse_diagnostic_location(&diagnostic, rust),
                Some(DiagnosticLocation::Rust(3)),
                "{rust}"
            );
            // Another file, even one with the same last component, is not ours.
            assert_eq!(parse_diagnostic_location(&diagnostic, "dir/x.rs"), None);
        }
        // A decomposed (NFD) accent, as macOS file systems store names, is refused by both
        // ends, so such a file gets no `#line` and no Rust location.
        assert!(!directive_file_is_usable("dir/u\u{308}/x.rs"));
    }

    #[test]
    fn diagnostic_location_reads_the_rust_file_only_when_it_is_the_kernels_own() {
        let rust = "crates/x/src/k.rs";
        for (diagnostic, expected) in [
            (
                "error[E30015]: bad\n --> crates/x/src/k.rs:42:17\n",
                Some(DiagnosticLocation::Rust(42)),
            ),
            (
                "crates/x/src/k.rs(42,3): error 30015: bad\n",
                Some(DiagnosticLocation::Rust(42)),
            ),
            (
                "crates/x/src/k.rs(42): error: bad\n",
                Some(DiagnosticLocation::Rust(42)),
            ),
            (
                "kernel.slang(6): error: bad\n",
                Some(DiagnosticLocation::Slang(6)),
            ),
            (
                "error[E1]: bad\n --> C:\\tmp\\kernel.slang:6:3\n",
                Some(DiagnosticLocation::Slang(6)),
            ),
            (
                "error[E1]: bad\n --> kernel.slang:6:3\n",
                Some(DiagnosticLocation::Slang(6)),
            ),
            // Another Rust file, a prefix or suffix of ours, or a different case.
            ("error[E1]: bad\n --> other.rs:42:17\n", None),
            ("error[E1]: bad\n --> src/crates/x/src/k.rs:42:17\n", None),
            ("error[E1]: bad\n --> crates/x/src/k.rs.bak:42:17\n", None),
            ("error[E1]: bad\n --> Crates/x/src/k.rs:42:17\n", None),
            ("error[E1]: bad\n --> .\\crates\\x\\src\\k.rs:42:17\n", None),
            // Warnings never claim the location; neither do bad coordinates.
            ("warning[W1]: careful\n --> crates/x/src/k.rs:42:17\n", None),
            ("error[E1]: bad\n --> crates/x/src/k.rs:0:17\n", None),
            ("error[E1]: bad\n --> crates/x/src/k.rs:42:0\n", None),
        ] {
            assert_eq!(
                parse_diagnostic_location(diagnostic, rust),
                expected,
                "{diagnostic}"
            );
        }
        // A recorded file that injection refuses never claims a location: a file
        // recorded as `kernel.slang` leaves `kernel.slang(6)` a Slang line.
        for refused in [
            "kernel.slang",
            "src/kernel.slang",
            "src/lib rs",
            "src\\lib.rs",
        ] {
            for diagnostic in [
                "kernel.slang(6): error: bad\n",
                "error[E1]: bad\n --> kernel.slang:6:3\n",
                "error[E1]: bad\n --> C:\\tmp\\kernel.slang:6:3\n",
            ] {
                assert_eq!(
                    parse_diagnostic_location(diagnostic, refused),
                    Some(DiagnosticLocation::Slang(6)),
                    "{refused:?}: {diagnostic}"
                );
            }
        }
        assert_eq!(
            parse_diagnostic_location("error[E1]: bad\n --> src/lib rs:6:3\n", "src/lib rs"),
            None
        );
        // With no Rust file recorded, a path that looks like one is just foreign.
        assert_eq!(
            parse_diagnostic_location("error[E1]: bad\n --> crates/x/src/k.rs:42:17\n", ""),
            None
        );
    }

    #[test]
    fn a_rust_location_names_the_construct_of_the_lowest_slang_line_of_that_rust_line() {
        // Rust line 22 is lowered to Slang lines 7 (the local `b`) and, in this
        // table, also to line 8: the lowest wins, and line 8 is not looked at.
        const LINES: &[crate::LineSpan] = &[
            crate::LineSpan {
                slang_start: 8,
                slang_end: 8,
                rust_line: 22,
            },
            crate::LineSpan {
                slang_start: 7,
                slang_end: 7,
                rust_line: 22,
            },
        ];
        let origin = SourceOrigin {
            file: "crates/demo/src/lib.rs",
            lines: LINES,
        };
        let annotated = annotate(
            &origin,
            "error[E30015]: undefined identifier\n --> crates/demo/src/lib.rs:22:17\n",
        );
        assert!(
            annotated.ends_with(concat!(
                "\noriginating Rust kernel: demo::k",
                "\noriginating Rust construct: local #1",
                "\noriginating Rust source: crates/demo/src/lib.rs:22",
                "\n(column is in the generated Slang)",
            )),
            "{annotated}"
        );
        // The Rust location is reported even when the table has no record for it.
        let annotated = annotate(
            &origin,
            "error[E1]: bad\n --> crates/demo/src/lib.rs:99:1\n",
        );
        assert!(
            annotated.ends_with(
                "\noriginating Rust source: crates/demo/src/lib.rs:99\n(column is in the generated Slang)"
            ) && !annotated.contains("originating Rust kernel"),
            "{annotated}"
        );
        // A Slang location carries no column note.
        let annotated = annotate(&ANNOTATED_ORIGIN, "kernel.slang(7): error: bad\n");
        assert!(!annotated.contains("column is in"), "{annotated}");
    }

    /// A failing compile of `ANNOTATED_COPY`-shaped text: `source` with one Slang
    /// line replaced, compiled with `ANNOTATED_ORIGIN`'s table.
    fn compile_failure_with_table(source: &str) -> String {
        let directory = TemporaryDirectory::create().unwrap();
        let error = compile_at(
            directory.path(),
            source,
            "k",
            &ANNOTATED_ORIGIN,
            Shape::Grid,
            Target::Wgsl,
        )
        .unwrap_err();
        let Error::CompilationFailed { diagnostics, .. } = error else {
            panic!("expected a compiler failure: {error}");
        };
        diagnostics
    }

    #[test]
    fn a_failure_on_an_attributed_line_is_reported_in_the_rust_file() {
        let diagnostics = compile_failure_with_table(ANNOTATED_SOURCE);
        assert!(
            diagnostics.contains(" --> crates/demo/src/lib.rs:22:"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust source: crates/demo/src/lib.rs:22\n"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust construct: local #1"),
            "{diagnostics}"
        );
    }

    #[test]
    fn a_failure_on_an_unattributed_line_after_attributed_ones_is_back_in_slang_coordinates() {
        // Only line 8 (no record) fails; the directive before it must have returned
        // the compiler to `kernel.slang`, not let it count on from Rust line 22.
        let source = ANNOTATED_SOURCE
            .replace("(a + undefined_symbol)", "a")
            .replace("uint(3)", "missing_value");
        let diagnostics = compile_failure_with_table(&source);
        assert!(diagnostics.contains("missing_value"), "{diagnostics}");
        assert!(diagnostics.contains("kernel.slang:8:"), "{diagnostics}");
        assert!(
            !diagnostics.contains("crates/demo/src/lib.rs"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust construct: local #2"),
            "{diagnostics}"
        );
        assert!(
            !diagnostics.contains("originating Rust source"),
            "{diagnostics}"
        );
    }

    #[test]
    fn the_clean_text_stays_beside_the_copy_only_when_the_copy_differs() {
        let clean = TemporaryDirectory::create().unwrap();
        let copy =
            write_compile_inputs(clean.path(), ANNOTATED_SOURCE, &SourceOrigin::UNKNOWN).unwrap();
        assert_eq!(copy, ANNOTATED_SOURCE);
        assert!(!clean.path().join("kernel.export.slang").exists());
        let annotated = TemporaryDirectory::create().unwrap();
        let copy =
            write_compile_inputs(annotated.path(), ANNOTATED_SOURCE, &ANNOTATED_ORIGIN).unwrap();
        assert_eq!(copy, ANNOTATED_COPY);
        assert_eq!(
            fs::read_to_string(annotated.path().join("kernel.slang")).unwrap(),
            ANNOTATED_COPY
        );
        assert_eq!(
            fs::read_to_string(annotated.path().join("kernel.export.slang")).unwrap(),
            ANNOTATED_SOURCE
        );
    }

    #[test]
    fn annotation_does_not_attribute_a_warning_or_a_foreign_file() {
        for diagnostic in [
            "warning[W1]: careful\n --> C:\\tmp\\kernel.slang:7:17\n",
            // The kernel's own Rust file is attributed (M1.3); any other is not.
            "error[E1]: bad\n --> crates/other/src/lib.rs:7:17\n",
            "no location at all\n",
        ] {
            assert_eq!(annotate(&ANNOTATED_ORIGIN, diagnostic), diagnostic);
        }
    }

    #[test]
    fn sidecar_written_by_a_compile_carries_the_rust_file_and_lines() {
        let source = concat!(
            "RWStructuredBuffer<uint> values;\n",        // 1
            "\n",                                        // 2
            "// @rust kernel: example::ok\n",            // 3
            "[shader(\"compute\")]\n",                   // 4
            "[numthreads(1, 1, 1)]\n",                   // 5
            "void ok(uint3 id : SV_DispatchThreadID)\n", // 6
            "{\n",                                       // 7
            "    uint __gpu_len_values;\n",              // 8
            "    uint __gpu_stride_values;\n",           // 9
            "    values.GetDimensions(__gpu_len_values, __gpu_stride_values);\n", // 10
            "    var i = id.x;\n",                       // 11
            "    values[i] = uint(1);\n",                // 12
            "}\n",                                       // 13
        );
        const LINES: &[crate::LineSpan] = &[
            crate::LineSpan {
                slang_start: 11,
                slang_end: 11,
                rust_line: 7,
            },
            crate::LineSpan {
                slang_start: 12,
                slang_end: 12,
                rust_line: 8,
            },
        ];
        let origin = SourceOrigin {
            file: "src/kernels.rs",
            lines: LINES,
        };
        let directory = TemporaryDirectory::create().unwrap();
        compile_at(
            directory.path(),
            source,
            "ok",
            &origin,
            Shape::Grid,
            Target::Wgsl,
        )
        .unwrap();
        let sidecar = fs::read_to_string(directory.path().join("kernel.map.json")).unwrap();
        assert!(
            sidecar.starts_with(
                "{\n  \"schema_version\": 2,\n  \"rust_file\": \"src/kernels.rs\",\n  \"line_directives\": true,\n  \"segments\": ["
            ),
            "{sidecar}"
        );
        assert!(
            sidecar.contains("\"construct\":\"local\",\"ordinal\":0,\"rust_line\":7}"),
            "{sidecar}"
        );
        assert!(
            sidecar.contains("\"construct\":\"assignment\",\"ordinal\":1,\"rust_line\":8}"),
            "{sidecar}"
        );
        // The prologue has no record, and neither does the kernel as a whole.
        assert!(
            sidecar.contains("\"construct\":\"prologue\",\"ordinal\":0}"),
            "{sidecar}"
        );
        // The compiler reads the copy with directives; the clean text stays beside it.
        assert_eq!(
            fs::read_to_string(directory.path().join("kernel.slang")).unwrap(),
            compile_copy(source, &origin)
        );
        assert_ne!(compile_copy(source, &origin), source);
        assert_eq!(
            fs::read_to_string(directory.path().join("kernel.export.slang")).unwrap(),
            source
        );
    }

    #[test]
    fn malformed_compiler_artifacts_are_rejected() {
        assert!(matches!(
            decode_spirv(&[0, 1, 2]),
            Err(Error::InvalidSpirvLength(3))
        ));
        assert!(matches!(decode_spirv(&[]), Err(Error::InvalidSpirv(_))));
        assert!(matches!(
            decode_spirv(&[0; 20]),
            Err(Error::InvalidSpirv(_))
        ));
        let truncated = crate::spirv::words_as_le_bytes(&[
            0x0723_0203,
            0x0001_0000,
            0,
            2,
            0,
            (3 << 16) | 14,
            0,
        ]);
        assert!(matches!(
            decode_spirv(&truncated),
            Err(Error::InvalidSpirv(
                "SPIR-V instruction extends beyond the module"
            ))
        ));
    }
}
