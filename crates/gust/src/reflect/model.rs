use crate::{
    ScalarKind, Shape,
    abi::{MatrixLayout, TextureAccess, TextureDimension},
    slang::Target,
    texture::Format as TexelFormat,
};

/// What one reflection document reports about a compiled entry point: its target, its stage, its buffer, texture and
/// sampler bindings and its colour outputs. Read with [`Reflection::from_json`], and checked against a kernel
/// descriptor by [`Reflection::cross_check_kernel`].
///
/// ```
/// use gust::reflect::{Reflection, Stage};
/// use gust::slang::Target;
///
/// // A minimal document with one structured buffer of `u32`, at binding 7 of space 2.
/// let document = r#"{"parameters":[{"name":"a","binding":{"kind":"descriptorTableSlot","index":7,"space":2},"type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"uint32"}}}]}"#;
/// let reflection = Reflection::from_json(Target::Spirv, document)?;
/// // This grammar is schema 1: a compute entry point with buffers only.
/// assert_eq!(reflection.schema_version, 1);
/// assert_eq!(reflection.stage, Stage::Compute);
/// assert_eq!(reflection.resources.len(), 1);
/// assert!(reflection.textures.is_empty() && reflection.samplers.is_empty() && reflection.targets.is_empty());
/// assert_eq!((reflection.resources[0].binding, reflection.resources[0].space), (7, 2));
/// # Ok::<(), gust::reflect::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reflection {
    /// The compile target the document describes.
    pub target: Target,
    /// The `schemaVersion` the document was read under: 1 for the CLI JSON path and for schema-1 native documents,
    /// 2 for schema-2 native documents, 3 for schema-3 native documents, 4 for schema-4 native documents (arrays).
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
///
/// ```
/// use gust::Shape;
/// use gust::reflect::Stage;
///
/// assert_eq!(Stage::of(Shape::Grid), Stage::Compute);
/// assert_eq!(Stage::of(Shape::Pixel).name(), "fragment");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
///
/// ```
/// use gust::reflect::{ReflectedType, ReflectedTypeKind, ReflectedTexture};
/// use gust::texture::Format;
/// use gust::{ScalarKind, TextureAccess, TextureDimension};
///
/// // Only a schema-3 document carries a texture binding, and reading one takes the compiler, so this one is built by hand.
/// let texture = ReflectedTexture {
///     name: "image".to_string(),
///     binding: 0,
///     space: 0,
///     dimension: TextureDimension::D2,
///     access: TextureAccess::WriteOnly,
///     result: ReflectedType {
///         name: None,
///         size: None,
///         alignment: None,
///         stride: None,
///         kind: ReflectedTypeKind::Scalar(ScalarKind::F32),
///     },
///     format: Some(Format::R32Float),
/// };
/// assert_eq!(texture.format, Some(Format::R32Float));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedTexture {
    /// The texture's name in the shader.
    pub name: String,
    /// The binding index.
    pub binding: u32,
    /// The register space, which is the bind group.
    pub space: u32,
    /// The texture's dimensionality.
    pub dimension: TextureDimension,
    /// How the shader uses the texture.
    pub access: TextureAccess,
    /// The result type: a 32-bit scalar or a 2- to 4-wide vector of one. The helper reports no byte layout for it
    /// (`size`, `alignment` and `stride` are `None`).
    pub result: ReflectedType,
    /// The texel format of a storage or write-only texture. `None` is the document's `unknown`: a sampled texture
    /// has no format in the shader (its concrete format is checked at submit against the bound texture).
    pub format: Option<TexelFormat>,
}
/// A sampler binding of a schema-3 document.
///
/// ```
/// use gust::reflect::ReflectedSampler;
///
/// // Only a schema-3 document carries a sampler binding, and reading one takes the compiler, so this one is built by hand.
/// let sampler = ReflectedSampler { name: "linear".to_string(), binding: 1, space: 0 };
/// assert_eq!(sampler.binding, 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedSampler {
    /// The sampler's name in the shader.
    pub name: String,
    /// The binding index.
    pub binding: u32,
    /// The register space, which is the bind group.
    pub space: u32,
}
/// One colour output of a fragment entry point.
///
/// ```
/// use gust::ScalarKind;
/// use gust::reflect::{ReflectedTarget, ReflectedType, ReflectedTypeKind};
///
/// // Only a schema-3 fragment document carries colour outputs, and reading one takes the compiler, so this one is built by hand.
/// let target = ReflectedTarget {
///     index: 0,
///     ty: ReflectedType {
///         name: None,
///         size: None,
///         alignment: None,
///         stride: None,
///         kind: ReflectedTypeKind::Vector { scalar: ScalarKind::F32, count: 4 },
///     },
/// };
/// assert_eq!(target.index, 0);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedTarget {
    /// The `SV_Target` index.
    pub index: u32,
    /// A 32-bit scalar or a 2- to 4-wide vector of one.
    pub ty: ReflectedType,
}
/// What kind of shader resource a [`ReflectedResource`] is.
///
/// ```
/// use gust::reflect::ReflectedResourceKind;
///
/// // The enum is non-exhaustive: match it with a wildcard arm.
/// let kind = ReflectedResourceKind::StructuredBuffer;
/// let name = match kind {
///     ReflectedResourceKind::StructuredBuffer => "structured buffer",
///     ReflectedResourceKind::ConstantBuffer => "constant buffer",
///     _ => "other",
/// };
/// assert_eq!(name, "structured buffer");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReflectedResourceKind {
    /// A structured buffer (`StructuredBuffer<T>` or `RWStructuredBuffer<T>`).
    StructuredBuffer,
    /// Schema 2 only. Always read-only, with no element count.
    ConstantBuffer,
}
/// One buffer binding of an entry point, as the reflection document reports it.
///
/// ```
/// use gust::reflect::{Reflection, ReflectedResourceKind, ReflectedTypeKind};
/// use gust::slang::Target;
/// use gust::ScalarKind;
///
/// let document = r#"{"parameters":[{"name":"a","binding":{"kind":"descriptorTableSlot","index":7,"space":2},"type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"uint32"}}}]}"#;
/// let reflection = Reflection::from_json(Target::Spirv, document)?;
/// let resource = &reflection.resources[0];
/// assert_eq!(resource.name, "a");
/// assert_eq!(resource.kind, ReflectedResourceKind::StructuredBuffer);
/// assert_eq!(resource.element.kind, ReflectedTypeKind::Scalar(ScalarKind::U32));
/// // `from_json` (the CLI path) never reads layout sizes; the native path (`compile_reflected`) fills them in.
/// assert_eq!(resource.element_stride, None);
/// # Ok::<(), gust::reflect::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedResource {
    /// The buffer's name in the shader.
    pub name: String,
    /// Whether it is a structured buffer or a constant buffer.
    pub kind: ReflectedResourceKind,
    /// Whether the shader only reads the buffer or also writes it.
    pub access: crate::Access,
    /// The binding index.
    pub binding: u32,
    /// The register space, which is the bind group.
    pub space: u32,
    /// The element type of a structured buffer, or the contents of a constant buffer.
    pub element: ReflectedType,
    /// None means the compiler JSON did not report a buffer element stride.
    pub element_stride: Option<u32>,
}
/// A type as the reflection document reports it. A size, alignment or stride that is not known is `None`:
/// `Reflection::from_json` never reads them, and the native path (`compile_reflected`) fills them in.
///
/// ```
/// use gust::ScalarKind;
/// use gust::reflect::{Reflection, ReflectedTypeKind};
/// use gust::slang::Target;
///
/// let document = r#"{"parameters":[{"name":"a","binding":{"kind":"descriptorTableSlot","index":0,"space":0},"type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"float32"}}}]}"#;
/// let reflection = Reflection::from_json(Target::Spirv, document)?;
/// let ty = &reflection.resources[0].element;
/// assert_eq!(ty.kind, ReflectedTypeKind::Scalar(ScalarKind::F32));
/// // `from_json` leaves the layout sizes unread.
/// assert_eq!((ty.size, ty.alignment, ty.stride), (None, None, None));
/// # Ok::<(), gust::reflect::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedType {
    /// The type's name, when the document gives one.
    pub name: Option<String>,
    /// The size in bytes, when reported.
    pub size: Option<u32>,
    /// The alignment in bytes, when reported.
    pub alignment: Option<u32>,
    /// The array stride in bytes, when reported. For a matrix, the column stride.
    pub stride: Option<u32>,
    /// The structure of the type.
    pub kind: ReflectedTypeKind,
}
/// The structure of a [`ReflectedType`].
///
/// ```
/// use gust::ScalarKind;
/// use gust::reflect::ReflectedTypeKind;
///
/// let kind = ReflectedTypeKind::Vector { scalar: ScalarKind::F32, count: 3 };
/// // The enum is non-exhaustive: match it with a wildcard arm.
/// let components = match &kind {
///     ReflectedTypeKind::Scalar(_) => 1,
///     ReflectedTypeKind::Vector { count, .. } => *count,
///     _ => 0,
/// };
/// assert_eq!(components, 3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReflectedTypeKind {
    /// A 32-bit scalar.
    Scalar(ScalarKind),
    /// Schema 2 only.
    Vector {
        /// The kind of each component.
        scalar: ScalarKind,
        /// The number of components.
        count: u32,
    },
    /// Schema 2 only. `ReflectedType::stride` of a matrix is its column stride.
    Matrix {
        /// The kind of each element.
        scalar: ScalarKind,
        /// The number of rows.
        rows: u32,
        /// The number of columns.
        cols: u32,
        /// The matrix layout the document states.
        layout: MatrixLayout,
    },
    /// A struct: its fields in document order.
    Struct(Vec<ReflectedField>),
    /// Schema 4 only: a fixed-size array. The [`ReflectedType`] around it carries the size, alignment and stride of the
    /// whole array.
    Array {
        /// The element type; an array again for a nested array.
        element: Box<ReflectedType>,
        /// The number of elements, at least 1.
        count: u32,
        /// The bytes between two consecutive elements.
        element_stride: u32,
    },
}
/// One field of a reflected struct.
///
/// ```
/// use gust::ScalarKind;
/// use gust::reflect::{ReflectedField, ReflectedType, ReflectedTypeKind};
///
/// let field = ReflectedField {
///     name: "x".to_string(),
///     offset: 0,
///     size: 4,
///     ty: ReflectedType {
///         name: None,
///         size: Some(4),
///         alignment: Some(4),
///         stride: None,
///         kind: ReflectedTypeKind::Scalar(ScalarKind::U32),
///     },
/// };
/// // A struct kind holds its fields in document order.
/// let kind = ReflectedTypeKind::Struct(vec![field]);
/// assert!(matches!(kind, ReflectedTypeKind::Struct(ref fields) if fields.len() == 1));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedField {
    /// The field's name.
    pub name: String,
    /// The field's byte offset from the start of the struct.
    pub offset: u32,
    /// The field's size in bytes.
    pub size: u32,
    /// The field's type.
    pub ty: ReflectedType,
}
/// Coverage includes every nested type, not just the outer element.
///
/// ```
/// use gust::reflect::{ReflectedResource, ReflectedResourceKind, ReflectedType, ReflectedTypeKind};
/// use gust::{Access, ScalarKind};
///
/// let resource = ReflectedResource {
///     name: "out".to_string(),
///     kind: ReflectedResourceKind::StructuredBuffer,
///     access: Access::ReadWrite,
///     binding: 0,
///     space: 0,
///     element: ReflectedType {
///         name: Some("float".to_string()),
///         size: Some(4),
///         alignment: Some(4),
///         stride: Some(4),
///         kind: ReflectedTypeKind::Scalar(ScalarKind::F32),
///     },
///     element_stride: Some(4),
/// };
/// use gust::reflect::cross_check_pod;
///
/// let check = cross_check_pod::<f32>(&resource)?;
/// assert!(check.is_full_abi_match());
/// # Ok::<(), gust::reflect::Error>(())
/// ```
#[must_use = "Inspect coverage: a successful cross-check may still lack full ABI evidence"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PodCrossCheck {
    /// The number of struct fields compared, counting those of nested structs.
    pub checked_fields: usize,
    /// Whether every type in the element, nested ones included, reported a size and each matched the host layout.
    pub aggregate_size_verified: bool,
    /// Whether every type in the element, nested ones included, reported a stride and each matched the host layout.
    pub element_stride_verified: bool,
    /// Whether every type in the element, nested ones included, reported an alignment and each matched the host layout.
    pub alignment_verified: bool,
}
impl PodCrossCheck {
    /// Whether the size, the stride and the alignment were all reported and all matched.
    pub fn is_full_abi_match(&self) -> bool {
        self.aggregate_size_verified && self.element_stride_verified && self.alignment_verified
    }
}

/// The result of a native compile: the compiler output and the reflection of the same linked program.
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
///     // Needs the `slang-runtime` feature and the pinned Slang.
///     #[cfg(feature = "slang-runtime")]
///     {
///         use gust::reflect::{compile_reflected, Reflection};
///         use gust::slang::Target;
///
///         let compiled = compile_reflected(&ops::double::DESCRIPTOR, Target::Wgsl)?;
///         let reflection: &Reflection = &compiled.reflection;
///         assert_eq!(reflection.resources.len(), 1);
///         assert!(!compiled.compiler_version.is_empty());
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct CompiledReflection {
    /// The compiler output for the target, as bytes.
    pub artifact: Vec<u8>,
    /// The reflection of the program that produced `artifact`.
    pub reflection: Reflection,
    /// The version string of the Slang compiler that produced them.
    pub compiler_version: String,
    /// The entry point's name.
    pub entry_point: String,
    /// The compute workgroup size. `[0, 0, 0]` for a fragment entry point, which has none (mx.t0b).
    pub workgroup_size: [u32; 3],
}
