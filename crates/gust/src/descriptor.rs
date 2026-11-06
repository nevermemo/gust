use core::fmt;

use crate::capability::Capability;
use crate::target::Target;

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
/// its shape, its capabilities, its target and its precompiled artifacts. The `#[gpu]` macro builds one per `#[kernel]`
/// function.
///
/// API change (0.2.0, M8): the struct is `#[non_exhaustive]`, so a later field is not a breaking change again, and a
/// literal is no longer possible outside this crate. Build one with [`KernelDescriptor::new`] and the `with_*`
/// setters, each a `const fn`, so a descriptor can still be a `const`. The fields [`variant`](Self::variant),
/// [`requires`](Self::requires) and [`target`](Self::target) are new. Struct-update syntax
/// (`KernelDescriptor { shape, ..BASE }`) becomes `BASE.with_shape(shape)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    /// The derived `PartialEq`/`Eq` compare it. It is not part of the content key: the stage is already in the Slang
    /// text (D45), so one body under two shapes has two keys. [`KernelDescriptor::new`] sets [`Shape::Grid`];
    /// [`with_shape`](KernelDescriptor::with_shape) changes it.
    pub shape: Shape,
    /// Every shape the kernel can take, judged by the macro's conservative syntactic rules (decision D63, slice
    /// mx.c). `Shape::Grid` is always in it; `Shape::Pixel` is when nothing in the kernel rules a pixel out. It is
    /// information only: nothing reads it at run time, and it is not part of the content key. A descriptor rebuilt
    /// from an export record (`cargo-gust`) carries only the shape it was built as. The derived `PartialEq`/`Eq`
    /// compare it. [`KernelDescriptor::new`] sets `&[Shape::Grid]`.
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
    /// Slang, the pipeline cache identity or the runtime ABI. The derived `PartialEq`/`Eq` compare it too, so two
    /// descriptors that differ only in source metadata are not equal. [`KernelDescriptor::new`] sets
    /// [`SourceOrigin::UNKNOWN`].
    pub source: SourceOrigin,
    /// Precompiled artifacts of this kernel. Each is checked against the descriptor by
    /// [`verify_artifact`](crate::artifact::verify_artifact); the backend that uses them
    /// calls it before use (`gust-compute-wgpu`, slice m4.e1). The macro embeds the pair `gust-artifacts/<key>.wgsl`
    /// and `<key>.wgsl.reflection.json` of the kernel crate when both exist, and emits an empty slice otherwise.
    /// The derived `PartialEq`/`Eq` compare it. [`KernelDescriptor::new`] sets an empty slice.
    pub artifacts: &'static [crate::artifact::Artifact],
    /// The name of the variant whose body this descriptor describes, or `""` for the kernel's portable body (M8,
    /// D23). It is not part of the content key: different bodies have different Slang text and so different keys.
    pub variant: &'static str,
    /// The capabilities the selected body needs (M8, D23): empty for the portable body. A device that lacks one of
    /// them cannot run the kernel. The derived `PartialEq`/`Eq` compare it.
    pub requires: &'static [Capability],
    /// The target this kernel is built for in this build (M8, D65, D137): [`Target::Gpu`] for every kernel until
    /// routing exists. A GPU device refuses a kernel whose target is [`Target::Cpu`].
    pub target: Target,
}

/// The `#[gpu]` macro's way to build a [`KernelDescriptor`] with named fields (support surface, hidden): it calls
/// [`KernelDescriptor::__gust_descriptor`], so a field added to the struct is a compile error in the expansion until
/// the macro sets it, and the names keep adjacent string arguments from being swapped.
///
/// Every slice and the origin is first bound to a constant of its own. The descriptor is built by a function call, and
/// the `'static` borrow of an argument is promoted only when the whole array is a constant expression, which the layout
/// blocks inside a parameter are not; a constant's own initializer extends its temporaries instead.
#[doc(hidden)]
#[macro_export]
macro_rules! __gust_descriptor {
    (
        module: $module:expr,
        name: $name:expr,
        entry_point: $entry_point:expr,
        shape: $shape:expr,
        compatible_shapes: $compatible_shapes:expr,
        workgroup_size: $workgroup_size:expr,
        parameters: $parameters:expr,
        slang_source: $slang_source:expr,
        source: $source:expr,
        artifacts: $artifacts:expr,
        variant: $variant:expr,
        requires: $requires:expr,
        target: $target:expr $(,)?
    ) => {{
        const __GUST_COMPATIBLE_SHAPES: &[$crate::Shape] = $compatible_shapes;
        const __GUST_PARAMETERS: &[$crate::ParameterDescriptor] = $parameters;
        const __GUST_SOURCE: $crate::SourceOrigin = $source;
        const __GUST_ARTIFACTS: &[$crate::Artifact] = $artifacts;
        const __GUST_REQUIRES: &[$crate::capability::Capability] = $requires;
        $crate::KernelDescriptor::__gust_descriptor(
            $module,
            $name,
            $entry_point,
            $shape,
            __GUST_COMPATIBLE_SHAPES,
            $workgroup_size,
            __GUST_PARAMETERS,
            $slang_source,
            __GUST_SOURCE,
            __GUST_ARTIFACTS,
            $variant,
            __GUST_REQUIRES,
            $target,
        )
    }};
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
    /// A descriptor with the fields every kernel has and the defaults of the rest: [`Shape::Grid`] (and
    /// `compatible_shapes` of `&[Shape::Grid]`), [`SourceOrigin::UNKNOWN`], no artifacts, the portable variant (`""`),
    /// no required capability and [`Target::Gpu`]. The `with_*` setters change the rest.
    ///
    /// ```
    /// use gust::{KernelDescriptor, Shape};
    ///
    /// const DESCRIPTOR: KernelDescriptor =
    ///     KernelDescriptor::new("demo", "run", "gpu_demo_run", [64, 1, 1], &[], "// slang")
    ///         .with_shape(Shape::Pixel);
    /// assert_eq!(DESCRIPTOR.shape, Shape::Pixel);
    /// assert_eq!(DESCRIPTOR.workgroup_size, [64, 1, 1]);
    /// assert!(DESCRIPTOR.requires.is_empty());
    /// ```
    pub const fn new(
        module: &'static str,
        name: &'static str,
        entry_point: &'static str,
        workgroup_size: [u32; 3],
        parameters: &'static [ParameterDescriptor],
        slang_source: &'static str,
    ) -> Self {
        Self {
            module,
            name,
            entry_point,
            shape: Shape::Grid,
            compatible_shapes: &[Shape::Grid],
            workgroup_size,
            parameters,
            slang_source,
            source: SourceOrigin::UNKNOWN,
            artifacts: &[],
            variant: "",
            requires: &[],
            target: Target::Gpu,
        }
    }

    /// The descriptor with `module` as its [`module`](Self::module).
    pub const fn with_module(mut self, module: &'static str) -> Self {
        self.module = module;
        self
    }

    /// The descriptor with `name` as its [`name`](Self::name).
    pub const fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// The descriptor with `entry_point` as its [`entry_point`](Self::entry_point).
    pub const fn with_entry_point(mut self, entry_point: &'static str) -> Self {
        self.entry_point = entry_point;
        self
    }

    /// The descriptor with `workgroup_size` as its [`workgroup_size`](Self::workgroup_size).
    pub const fn with_workgroup_size(mut self, workgroup_size: [u32; 3]) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// The descriptor with `parameters` as its [`parameters`](Self::parameters).
    pub const fn with_parameters(mut self, parameters: &'static [ParameterDescriptor]) -> Self {
        self.parameters = parameters;
        self
    }

    /// The descriptor with `slang_source` as its [`slang_source`](Self::slang_source).
    pub const fn with_slang_source(mut self, slang_source: &'static str) -> Self {
        self.slang_source = slang_source;
        self
    }

    /// The descriptor with `shape` as its [`shape`](Self::shape).
    pub const fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// The descriptor with `compatible_shapes` as its [`compatible_shapes`](Self::compatible_shapes).
    pub const fn with_compatible_shapes(mut self, compatible_shapes: &'static [Shape]) -> Self {
        self.compatible_shapes = compatible_shapes;
        self
    }

    /// The descriptor with `source` as its [`source`](Self::source).
    pub const fn with_source(mut self, source: SourceOrigin) -> Self {
        self.source = source;
        self
    }

    /// The descriptor with `artifacts` as its [`artifacts`](Self::artifacts).
    pub const fn with_artifacts(mut self, artifacts: &'static [crate::artifact::Artifact]) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// The descriptor with `variant` as its [`variant`](Self::variant).
    pub const fn with_variant(mut self, variant: &'static str) -> Self {
        self.variant = variant;
        self
    }

    /// The descriptor with `requires` as its [`requires`](Self::requires).
    pub const fn with_requires(mut self, requires: &'static [Capability]) -> Self {
        self.requires = requires;
        self
    }

    /// The descriptor with `target` as its [`target`](Self::target).
    pub const fn with_target(mut self, target: Target) -> Self {
        self.target = target;
        self
    }

    /// The constructor the `#[gpu]` macro calls: every field at once, so that a field added later is a compile error
    /// in the macro's output until the macro sets it. Support surface: not covered by the stability promise.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub const fn __gust_descriptor(
        module: &'static str,
        name: &'static str,
        entry_point: &'static str,
        shape: Shape,
        compatible_shapes: &'static [Shape],
        workgroup_size: [u32; 3],
        parameters: &'static [ParameterDescriptor],
        slang_source: &'static str,
        source: SourceOrigin,
        artifacts: &'static [crate::artifact::Artifact],
        variant: &'static str,
        requires: &'static [Capability],
        target: Target,
    ) -> Self {
        Self {
            module,
            name,
            entry_point,
            shape,
            compatible_shapes,
            workgroup_size,
            parameters,
            slang_source,
            source,
            artifacts,
            variant,
            requires,
            target,
        }
    }

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

    const BASE: KernelDescriptor =
        KernelDescriptor::new("demo", "run", "gpu_demo_run", [64, 1, 1], &[], "// slang");

    #[test]
    fn the_constructor_sets_the_documented_defaults() {
        assert_eq!(BASE.module, "demo");
        assert_eq!(BASE.name, "run");
        assert_eq!(BASE.entry_point, "gpu_demo_run");
        assert_eq!(BASE.workgroup_size, [64, 1, 1]);
        assert!(BASE.parameters.is_empty());
        assert_eq!(BASE.slang_source, "// slang");
        assert_eq!(BASE.shape, Shape::Grid);
        assert_eq!(BASE.compatible_shapes, &[Shape::Grid]);
        assert_eq!(BASE.source, SourceOrigin::UNKNOWN);
        assert!(BASE.artifacts.is_empty());
        assert_eq!(BASE.variant, "");
        assert!(BASE.requires.is_empty());
        assert_eq!(BASE.target, Target::Gpu);
    }

    #[test]
    fn each_setter_changes_its_field_only() {
        let pixel = BASE.with_shape(Shape::Pixel);
        assert_eq!(
            pixel,
            KernelDescriptor {
                shape: Shape::Pixel,
                ..BASE
            }
        );
        let both = BASE.with_compatible_shapes(&[Shape::Grid, Shape::Pixel]);
        assert_eq!(both.compatible_shapes, &[Shape::Grid, Shape::Pixel]);
        let variant = BASE
            .with_variant("subgroups")
            .with_requires(&[Capability::Subgroups]);
        assert_eq!(variant.variant, "subgroups");
        assert_eq!(variant.requires, &[Capability::Subgroups]);
        assert_ne!(variant, BASE);
        let cpu = BASE.with_target(Target::Cpu);
        assert_eq!(cpu.target, Target::Cpu);
        let located = BASE.with_source(SourceOrigin {
            file: "src/lib.rs",
            lines: &[],
        });
        assert_eq!(located.source.file, "src/lib.rs");
        assert!(BASE.with_artifacts(&[]).artifacts.is_empty());
    }

    #[test]
    fn the_macro_constructor_and_the_builder_agree() {
        let built = KernelDescriptor::__gust_descriptor(
            "demo",
            "run",
            "gpu_demo_run",
            Shape::Pixel,
            &[Shape::Grid, Shape::Pixel],
            [8, 8, 1],
            &[],
            "// slang",
            SourceOrigin::UNKNOWN,
            &[],
            "wide",
            &[Capability::Subgroups],
            Target::Cpu,
        );
        let chained =
            KernelDescriptor::new("demo", "run", "gpu_demo_run", [8, 8, 1], &[], "// slang")
                .with_shape(Shape::Pixel)
                .with_compatible_shapes(&[Shape::Grid, Shape::Pixel])
                .with_variant("wide")
                .with_requires(&[Capability::Subgroups])
                .with_target(Target::Cpu);
        assert_eq!(built, chained);
    }

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
