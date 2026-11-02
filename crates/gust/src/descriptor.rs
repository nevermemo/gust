use core::fmt;

/// Marker implemented by the zero-sized host handle generated for each kernel.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GUST kernel handle",
    label = "pass the zero-sized handle that `#[gpu]` generates for a `#[kernel]` function; it implements `Kernel`"
)]
pub trait Kernel {
    const DESCRIPTOR: KernelDescriptor;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleDescriptor {
    pub name: &'static str,
    pub kernels: &'static [KernelDescriptor],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelDescriptor {
    pub module: &'static str,
    pub name: &'static str,
    /// Prefixed entry point used in generated Slang and target binaries.
    pub entry_point: &'static str,
    pub workgroup_size: [u32; 3],
    pub parameters: &'static [ParameterDescriptor],
    /// Slang source emitted directly from the annotated Rust syntax tree.
    /// Backends compile this source to their preferred target on demand.
    pub slang_source: &'static str,
    /// Where the kernel's Rust source lives and which Slang lines came from
    /// which Rust line. Diagnostic metadata only: it never affects the emitted
    /// Slang, the pipeline cache identity or the runtime ABI.
    ///
    /// API change (M1.1): this field is new. A hand-written
    /// `KernelDescriptor { .. }` literal must add `source: SourceOrigin::UNKNOWN`,
    /// and the derived `PartialEq`/`Eq` now compare it too, so two descriptors that
    /// differ only in source metadata are no longer equal.
    pub source: SourceOrigin,
    /// Precompiled artifacts of this kernel. Each is checked against the descriptor by
    /// [`verify_artifact`](crate::artifact::verify_artifact); the backend that uses them
    /// calls it before use (`gust-wgpu`, slice m4.e1). The macro embeds the pair `gust-artifacts/<key>.wgsl` and
    /// `<key>.wgsl.reflection.json` of the kernel crate when both exist, and emits an empty slice otherwise.
    /// Like `source`, a hand-written `KernelDescriptor { .. }` literal must add
    /// `artifacts: &[]`, and the derived `PartialEq`/`Eq` compare it.
    pub artifacts: &'static [crate::artifact::Artifact],
}

/// Rust source location of a kernel module and the line table that maps its
/// emitted Slang lines back to Rust lines (DX19).
///
/// The table is a line index over the generated text, not an IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceOrigin {
    /// Rust source path with `/` separators, never absolute: a relative path as
    /// rustc reports it but normalised (`./src/lib.rs` becomes `src/lib.rs`), an
    /// absolute one under the expanding crate's manifest directory made relative
    /// to it, any other absolute one reduced to its file name. Every component is
    /// restricted to letters, digits and `_ . + -`; a path that cannot be made to
    /// fit has no origin. Empty when no location is known.
    pub file: &'static str,
    /// In emission order, outer construct first, so the innermost record
    /// containing a Slang line is the last match.
    pub lines: &'static [LineSpan],
}

impl SourceOrigin {
    /// No known Rust location, as for hand-written descriptors and probes.
    pub const UNKNOWN: Self = Self {
        file: "",
        lines: &[],
    };

    /// The Rust line of the innermost record containing `slang_line`.
    pub(crate) fn rust_line_at(&self, slang_line: u32) -> Option<u32> {
        self.lines
            .iter()
            .rev()
            .find(|span| span.slang_start <= slang_line && slang_line <= span.slang_end)
            .map(|span| span.rust_line)
    }
}

/// An inclusive, 1-based Slang line range generated from one Rust line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineSpan {
    pub slang_start: u32,
    pub slang_end: u32,
    pub rust_line: u32,
}

impl KernelDescriptor {
    pub const fn qualified_name(&self) -> QualifiedName<'_> {
        QualifiedName(self)
    }
}

pub struct QualifiedName<'a>(&'a KernelDescriptor);

impl fmt::Display for QualifiedName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.0.module, self.0.name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterDescriptor {
    pub name: &'static str,
    pub rust_type: &'static str,
    pub kind: ParameterKind,
    pub access: Access,
    /// Host/GPU element ABI for resource parameters supported by the runtime.
    pub resource_layout: Option<crate::ResourceLayout>,
    /// Shader resource location. Invocation and plain value parameters do not
    /// occupy a resource binding.
    pub binding: Option<ResourceBinding>,
}

/// Backend-neutral shader resource address assigned from a kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterKind {
    Invocation,
    Storage,
    Uniform,
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    ReadWrite,
    NotApplicable,
}
