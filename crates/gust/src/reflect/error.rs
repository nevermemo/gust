use crate::slang;
use std::{error, fmt, path::PathBuf};

/// Every way reading or checking Slang reflection can fail.
///
/// The enum is `#[non_exhaustive]`: a later release can add a variant without a breaking change, so a `match` in
/// another crate needs a wildcard arm.
///
/// ```
/// use gust::reflect::Error;
///
/// let error = Error::Invalid("no `entryPoints` array".to_string());
/// assert_eq!(error.to_string(), "invalid Slang reflection: no `entryPoints` array");
/// // The enum is non-exhaustive: match it with a wildcard arm.
/// match error {
///     Error::Invalid(_) | Error::Mismatch(_) | Error::Incomplete(_) => {}
///     _ => {}
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The compiler run that produces the reflection failed; the inner error says how (including a plain I/O failure,
    /// which is reported as `slang::Error::Io`). A `CompilationFailed` inside it may carry the exit status of the
    /// reflection helper rather than `slangc`'s.
    Compiler(slang::Error),
    /// The reflection helper executable could not be launched. `helper` is the path that was tried and `source` the
    /// operating system's error.
    HelperUnavailable {
        /// The helper executable path that was tried.
        helper: PathBuf,
        /// The operating system's error from launching it.
        source: std::io::Error,
    },
    /// The reflection output is malformed, lacks a field GUST needs, uses a construct GUST does not support (for
    /// example a binding array or a texture that is not 2d), or does not belong to the kernel: its target, entry
    /// point, source or artifact identity differs. The text says which.
    Invalid(String),
    /// The compiler's reflection disagrees with the kernel descriptor or with the POD layout the macro assigned (D17).
    /// The text names the parameter or the property (for example the stage or the workgroup size) and the fact that
    /// differs.
    Mismatch(String),
    /// The reflection is well formed but does not carry all the layout evidence the check needs, for example nested
    /// sizes, alignments or strides. The text says which.
    Incomplete(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(e) => e.fmt(f),
            Self::HelperUnavailable { helper, source } => write!(
                f,
                "could not launch Slang reflection compiler {}: {source}; run `cargo gust slang build-helper` or set GUST_SLANG_REFLECT to the helper executable",
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
pub(super) fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
