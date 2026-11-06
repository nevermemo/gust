use core::fmt;

/// Marker implemented by the zero-sized host handle generated for each kernel.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GUST kernel handle",
    label = "pass the zero-sized handle that `#[gpu]` generates for a `#[kernel]` function; it implements `Kernel`"
)]
pub trait Kernel {
    /// The descriptor of the kernel this handle stands for.
    const DESCRIPTOR: KernelDescriptor;
}

/// A `#[gpu]` module as the macro describes it: its name and its kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleDescriptor {
    /// The module's name.
    pub name: &'static str,
    /// The module's kernels.
    pub kernels: &'static [KernelDescriptor],
}

/// Everything the host knows about one kernel: where it lives, its Slang source, its workgroup size, its parameters,
/// its shape and its precompiled artifacts. The `#[gpu]` macro builds one per `#[kernel]` function; a hand-written
/// literal must set every field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelDescriptor {
    /// The name of the `#[gpu]` module that holds the kernel.
    pub module: &'static str,
    /// The kernel function's name.
    pub name: &'static str,
    /// Prefixed entry point used in generated Slang and target binaries.
    pub entry_point: &'static str,
    /// How the kernel's invocations are laid out, and so which Slang stage its entry point has (D63, D89). Every
    /// kernel the macro emits today is [`Shape::Grid`].
    ///
    /// API change (mx.k): this field is new. A hand-written `KernelDescriptor { .. }` literal must add
    /// `shape: Shape::Grid`, and the derived `PartialEq`/`Eq` compare it. It is not part of the content key: the
    /// stage is already in the Slang text (D45), so one body under two shapes has two keys.
    pub shape: Shape,
    /// Every shape the kernel can take, judged by the macro's conservative syntactic rules (decision D63, slice
    /// mx.c). `Shape::Grid` is always in it; `Shape::Pixel` is when nothing in the kernel rules a pixel out. It is
    /// information only: nothing reads it at run time, and it is not part of the content key. A descriptor rebuilt
    /// from an export record (`cargo-gust`) carries only the shape it was built as.
    ///
    /// API change (mx.c): this field is new. A hand-written `KernelDescriptor { .. }` literal must add
    /// `compatible_shapes: &[Shape::Grid]` (after `shape`), and the derived `PartialEq`/`Eq` compare it.
    pub compatible_shapes: &'static [Shape],
    /// The workgroup size `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// The kernel's parameters, in declaration order.
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
    /// calls it before use (`gust-compute-wgpu`, slice m4.e1). The macro embeds the pair `gust-artifacts/<key>.wgsl`
    /// and `<key>.wgsl.reflection.json` of the kernel crate when both exist, and emits an empty slice otherwise.
    /// Like `source`, a hand-written `KernelDescriptor { .. }` literal must add
    /// `artifacts: &[]`, and the derived `PartialEq`/`Eq` compare it.
    pub artifacts: &'static [crate::artifact::Artifact],
}

/// The width of a row of fragments of a `pixel` call, in elements (slice mx.p1a, Option S, decisions D120 and D127): the
/// index of a fragment is `y * PIXEL_ROW_WIDTH + x`. The macro bakes this number into the Slang of every `pixel` kernel
/// (`gust-compute-macros` keeps its own copy, since a proc-macro crate cannot depend on `gust-compute`; a test of this
/// crate compares the emitted text with this constant), and the host draws attachments of this width, so the two must
/// be the same number. It is WebGPU's baseline `maxTextureDimension2D`, which every device has.
pub const PIXEL_ROW_WIDTH: u32 = 8192;

/// The most elements one `pixel` call covers (Option S): `PIXEL_ROW_WIDTH` columns by 8192 rows, the baseline texture
/// size in both directions. A host refuses a larger call by name.
pub const PIXEL_MAX_ELEMENTS: u32 = 1 << 26;

/// The execution shape of a kernel: how its invocations are laid out, and with it the Slang stage of its entry point
/// (decisions D63 and D89).
///
/// `cpu` is a target (D51), never a shape. The enum is `#[non_exhaustive]` so that a later shape is not a breaking
/// change for a `match` outside this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Shape {
    /// One compute thread per element of a dispatch grid (`[shader("compute")]`). The shape of every kernel until
    /// the pixel shape exists, and the one a record or manifest without a `shape` means.
    Grid,
    /// One fragment invocation per element, drawn by a full-screen pass (`[shader("fragment")]`).
    Pixel,
}

impl Shape {
    /// The shape's name in a record and a manifest: `grid` or `pixel`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Pixel => "pixel",
        }
    }

    /// The shape named `name` (the inverse of [`Shape::name`]), or `None` for any other text.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "grid" => Some(Self::Grid),
            "pixel" => Some(Self::Pixel),
            _ => None,
        }
    }

    /// The Slang stage of the entry point: the argument of `slangc -stage` and the stage the reflection helper is
    /// told to expect.
    pub const fn slang_stage(self) -> &'static str {
        match self {
            Self::Grid => "compute",
            Self::Pixel => "fragment",
        }
    }
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
    /// The first Slang line of the range.
    pub slang_start: u32,
    /// The last Slang line of the range (inclusive).
    pub slang_end: u32,
    /// The 1-based Rust line that those Slang lines were generated from.
    pub rust_line: u32,
}

impl KernelDescriptor {
    /// The kernel's `module::name`, as a value that implements `Display`.
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

/// One kernel parameter: what it is, how the kernel uses it and where the shader binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterDescriptor {
    /// The parameter's name in the kernel.
    pub name: &'static str,
    /// The parameter's Rust type, as the token text of the source.
    pub rust_type: &'static str,
    /// What the parameter is, and so how the host binds it.
    pub kind: ParameterKind,
    /// How the kernel accesses the parameter.
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
    /// The bind group.
    pub group: u32,
    /// The binding index inside the group.
    pub binding: u32,
}

/// What a kernel parameter is, and so how the host binds it.
///
/// API change (mx.t1): the enum is `#[non_exhaustive]` and has two new variants, [`Texture`](Self::Texture) and
/// [`Sampler`](Self::Sampler). A `match` outside this crate must add a wildcard arm. Constructing a variant is
/// unchanged. Inside the workspace every such `match` already had one (`gust-compute-wgpu`'s `validate_kernel` and
/// `gust-compute`'s own `reflect`), so none of them changed, and both refuse the two new kinds. The `#[gpu]` macro
/// produces both kinds since mx.t3 (a `Texture2D<T>`, `StorageTexture2D<F>`, `WriteTexture2D<F>` or `Sampler` kernel
/// parameter), and `cargo-gust`'s record reader and the macro's artifact writer know the names `texture` and
/// `sampler`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParameterKind {
    /// The invocation id parameter: not a resource, not bound.
    Invocation,
    /// A storage buffer, bound through a [`Resource`](crate::Resource) of kind `kind::Storage`.
    Storage,
    /// A uniform (constant) buffer, bound through a resource of kind `kind::Uniform`.
    Uniform,
    /// A plain value parameter (not a resource).
    Value,
    /// A 2-D texture: sampled (`Texture2D<T>`), storage read-write (`StorageTexture2D<F>`) or storage write-only
    /// (`WriteTexture2D<F>`), bound through a resource of kind `kind::Texture` (decision D64). It has a shader
    /// binding, and the typed call counts it among its `N` resource arguments.
    Texture,
    /// A sampler (`Sampler`). It has a shader binding but is not a [`Resource`](crate::Resource): it is the call's own
    /// argument kind ([`SamplerArg`](crate::SamplerArg)), counted separately from the resources.
    Sampler,
}

/// How a kernel accesses a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// The kernel only reads it: a read-only storage buffer, a uniform, or a sampled texture.
    ReadOnly,
    /// The kernel may read and write it: a read-write storage buffer, or a storage texture (including a write-only
    /// one).
    ReadWrite,
    /// The notion does not apply: the parameter is an invocation id, a plain value or a sampler.
    NotApplicable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_has_a_name_that_reads_back_and_a_slang_stage() {
        for (shape, name, stage) in [
            (Shape::Grid, "grid", "compute"),
            (Shape::Pixel, "pixel", "fragment"),
        ] {
            assert_eq!(shape.name(), name);
            assert_eq!(Shape::from_name(name), Some(shape));
            assert_eq!(shape.slang_stage(), stage);
        }
        // Only the exact lower-case names read back: `cpu` is a target (D51), not a shape.
        for other in ["", "Grid", "PIXEL", "cpu", "compute", "grid "] {
            assert_eq!(Shape::from_name(other), None, "{other:?}");
        }
    }
}
