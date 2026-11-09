use core::{mem, slice};

/// Scalar formats that have an identical Rust and GPU storage representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarKind {
    /// A 32-bit unsigned integer (`uint`, Rust `u32`).
    U32,
    /// A 32-bit signed integer (`int`, Rust `i32`).
    I32,
    /// A 32-bit float (`float`, Rust `f32`).
    F32,
}

impl ScalarKind {
    /// Every kind this release has, in the order of the enum. The enum is `#[non_exhaustive]`, so a test outside this
    /// crate can walk this list to learn that a later release added a kind its `match` does not know.
    pub const ALL: &'static [Self] = &[Self::U32, Self::I32, Self::F32];
}

/// How a matrix is stored: the pin that the macro, the reflection reader and the reflection helper all cite.
///
/// Column-major means that each column is contiguous, so the column stride is the size of one column (padded to 8 or
/// 16 bytes) and a Rust matrix of vector columns has the same bytes (slice m6.a1, plan F5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Each column is contiguous (the layout GUST uses).
    Column,
    /// Each row is contiguous. Recorded so that a reflection document can name it; no GUST compile uses it.
    Row,
}

impl MatrixLayout {
    /// The spelling used in the reflection JSON (`matrixLayout`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Column => "column",
            Self::Row => "row",
        }
    }

    /// The inverse of [`MatrixLayout::as_str`].
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "column" => Some(Self::Column),
            "row" => Some(Self::Row),
            _ => None,
        }
    }
}

/// The one matrix layout mode of GUST: every Slang compile and every reflection document uses it.
pub const MATRIX_LAYOUT: MatrixLayout = MatrixLayout::Column;

/// Structural category for one host-shareable type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeLayoutKind {
    /// One 32-bit scalar.
    Scalar(ScalarKind),
    /// `count` scalars of one kind (`float2`, `uint4`, ...).
    Vector {
        /// The kind of each component.
        scalar: ScalarKind,
        /// The number of components, 2 to 4.
        count: u32,
    },
    /// A `rows` by `cols` matrix, column-major (`MATRIX_LAYOUT`): `cols` columns of `rows` scalars, each column
    /// starting `column_stride` bytes after the previous one.
    Matrix {
        /// The kind of each element.
        scalar: ScalarKind,
        /// The number of rows, 2 to 4.
        rows: u32,
        /// The number of columns, 2 to 4.
        cols: u32,
        /// The distance in bytes from the start of one column to the start of the next.
        column_stride: u32,
    },
    /// A struct: its fields in declaration order, each with its byte offset.
    Struct(&'static [StructFieldLayout]),
    /// A fixed-size array `[T; count]`: `count` elements of `element`, each `element.stride` bytes after the previous
    /// one. The enclosing [`TypeLayout`] has size `element.stride * count`, the element's alignment, and a stride equal
    /// to its size (m9.r2b, D159).
    Array {
        /// The layout of one element.
        element: &'static TypeLayout,
        /// The number of elements, at least 1.
        count: u32,
    },
}

/// Complete storage-buffer ABI for one element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeLayout {
    /// The type's name (a Rust type name such as `u32`, or the struct's own name). Descriptive: the byte layout is
    /// `size`, `alignment` and `kind`.
    pub name: &'static str,
    /// The size in bytes.
    pub size: u32,
    /// The alignment in bytes.
    pub alignment: u32,
    /// The distance in bytes between two consecutive elements of an array of this type. The constructors set it to
    /// `size`, except `structure`, which rounds `size` up to `alignment`.
    pub stride: u32,
    /// The structural category, with the scalar, vector, matrix or field data of the type.
    pub kind: TypeLayoutKind,
}

impl TypeLayout {
    /// Check that this description follows the supported four-byte scalar/struct ABI (`StorageV1`).
    ///
    /// Usable in `const` contexts, so a layout can be asserted at compile time. Vectors and matrices are not
    /// `StorageV1`; they are [`TypeLayout::is_storage_v2`].
    pub const fn is_storage_v1(self) -> bool {
        if self.size == 0 || self.alignment != 4 || self.stride != self.size {
            return false;
        }
        match self.kind {
            TypeLayoutKind::Scalar(_) => self.size == 4,
            TypeLayoutKind::Vector { .. } | TypeLayoutKind::Matrix { .. } => false,
            TypeLayoutKind::Struct(fields) => {
                let mut offset = 0_u32;
                let mut index = 0;
                while index < fields.len() {
                    let field = &fields[index];
                    if field.offset != offset || !field.ty.is_storage_v1() {
                        return false;
                    }
                    let Some(next) = offset.checked_add(field.ty.size) else {
                        return false;
                    };
                    offset = next;
                    index += 1;
                }
                offset == self.size
            }
            TypeLayoutKind::Array { element, count } => {
                // An array of StorageV1 elements is the elements back to back (alignment 4, so no padding).
                !matches!(element.kind, TypeLayoutKind::Array { .. })
                    && element.is_storage_v1()
                    && self.array_shape_is(element, count)
            }
        }
    }

    /// Whether this array description has the numbers its element and count imply: element stride equal to element
    /// size (the element rule, D159), at least one element, the element's alignment, and size `stride * count`.
    const fn array_shape_is(self, element: &TypeLayout, count: u32) -> bool {
        if count == 0 || element.stride != element.size || self.alignment != element.alignment {
            return false;
        }
        match element.stride.checked_mul(count) {
            Some(size) => self.size == size,
            None => false,
        }
    }

    /// Check that this description follows the `StorageV2` ABI: [`TypeLayout::is_storage_v1`]'s scalars, plus vectors
    /// of two (size and alignment 8) and four (16), square matrices of two (size 16, alignment 8) and four (size 64,
    /// alignment 16), and structs of these with every field at its offset rounded up to the field's alignment, the
    /// struct's alignment the largest field alignment, and its size (and stride) rounded up to that alignment. The
    /// description may therefore contain padding (before a field, and at the tail), but only where this rounding puts
    /// it; Rust's `#[repr(C)]` rule gives the same offsets for types that carry their alignment (plan F4, F5). A
    /// padded layout is not a [`GpuPod`] type: see the safety section there.
    ///
    /// Three-wide vectors, 3x3 and non-square matrices are not buffer types. Every `StorageV1` layout is `StorageV2`.
    pub const fn is_storage_v2(self) -> bool {
        self.follows_host_rules(false)
    }

    /// Check that this description follows the `Std140V1` ABI of a constant buffer: [`TypeLayout::is_storage_v2`]'s
    /// scalars and vectors, 4x4 matrices only (a 2x2 matrix has column stride 16 in a constant buffer, so no Rust
    /// type matches it), and structs whose alignment, and so size and stride, is rounded up to 16 (plan F7).
    pub const fn is_std140_v1(self) -> bool {
        self.follows_host_rules(true)
    }

    /// The layout rule shared by `StorageV2` (`std140 == false`) and `Std140V1`.
    const fn follows_host_rules(self, std140: bool) -> bool {
        if self.size == 0 || self.alignment == 0 || self.stride != self.size {
            return false;
        }
        match self.kind {
            TypeLayoutKind::Scalar(_) => self.size == 4 && self.alignment == 4,
            TypeLayoutKind::Vector { count, .. } => match count {
                2 => self.size == 8 && self.alignment == 8,
                4 => self.size == 16 && self.alignment == 16,
                _ => false,
            },
            TypeLayoutKind::Matrix {
                scalar,
                rows,
                cols,
                column_stride,
            } => {
                if !matches!(scalar, ScalarKind::F32) || rows != cols {
                    return false;
                }
                match rows {
                    2 => !std140 && column_stride == 8 && self.size == 16 && self.alignment == 8,
                    4 => column_stride == 16 && self.size == 64 && self.alignment == 16,
                    _ => false,
                }
            }
            TypeLayoutKind::Struct(fields) => {
                let mut offset = 0_u32;
                let mut largest = 0_u32;
                let mut index = 0;
                while index < fields.len() {
                    let field = &fields[index];
                    if !field.ty.follows_host_rules(std140) {
                        return false;
                    }
                    let Some(aligned) = checked_align_up(offset, field.ty.alignment) else {
                        return false;
                    };
                    if field.offset != aligned {
                        return false;
                    }
                    let Some(next) = aligned.checked_add(field.ty.size) else {
                        return false;
                    };
                    offset = next;
                    if field.ty.alignment > largest {
                        largest = field.ty.alignment;
                    }
                    index += 1;
                }
                let alignment = if std140 && largest < 16 { 16 } else { largest };
                if alignment == 0 || self.alignment != alignment {
                    return false;
                }
                match checked_align_up(offset, alignment) {
                    Some(size) => self.size == size,
                    None => false,
                }
            }
            // Arrays are storage-buffer types only (D159): a constant buffer rounds array strides up to 16, which no
            // Rust array matches (E0215). Elements are never arrays (E0217).
            TypeLayoutKind::Array { element, count } => {
                !std140
                    && !matches!(element.kind, TypeLayoutKind::Array { .. })
                    && element.follows_host_rules(false)
                    && self.array_shape_is(element, count)
            }
        }
    }

    /// The layout of one 32-bit scalar named `name`: size, alignment and stride are 4.
    pub const fn scalar(name: &'static str, kind: ScalarKind) -> Self {
        Self {
            name,
            size: 4,
            alignment: 4,
            stride: 4,
            kind: TypeLayoutKind::Scalar(kind),
        }
    }

    /// A vector of `count` (2 to 4) scalars. A two-wide vector has size and alignment 8, a four-wide one 16; the
    /// three-wide vector is described (size 12, alignment 16) but is not a buffer type ([`TypeLayout::is_storage_v2`]
    /// rejects it).
    pub const fn vector(name: &'static str, scalar: ScalarKind, count: u32) -> Self {
        assert!(
            count >= 2 && count <= 4,
            "a vector has two to four components"
        );
        let alignment = if count == 2 { 8 } else { 16 };
        let size = 4 * count;
        Self {
            name,
            size,
            alignment,
            stride: size,
            kind: TypeLayoutKind::Vector { scalar, count },
        }
    }

    /// A column-major `rows` by `cols` matrix (2 to 4 each): its columns are `rows`-wide vectors, so the column
    /// stride is 8 for two rows and 16 for three or four, the alignment is the column stride, and the size is
    /// `cols` column strides.
    pub const fn matrix(name: &'static str, scalar: ScalarKind, rows: u32, cols: u32) -> Self {
        assert!(
            rows >= 2 && rows <= 4 && cols >= 2 && cols <= 4,
            "a matrix has two to four rows and columns"
        );
        let column_stride = if rows == 2 { 8 } else { 16 };
        let size = column_stride * cols;
        Self {
            name,
            size,
            alignment: column_stride,
            stride: size,
            kind: TypeLayoutKind::Matrix {
                scalar,
                rows,
                cols,
                column_stride,
            },
        }
    }

    /// An array of `count` (at least 1) elements of `element`: size `element.stride * count`, the element's alignment,
    /// stride equal to the size, the numbers of the measured std430 arrays (spike P4b). Whether the description is a
    /// buffer type is for [`TypeLayout::is_storage_v2`] to say: it requires `element.stride == element.size` (the
    /// element rule of D159, which excludes `Vec3` and `Mat3`) and a non-array element.
    pub const fn array(element: &'static TypeLayout, count: u32) -> Self {
        assert!(count >= 1, "an array has at least one element");
        let Some(size) = element.stride.checked_mul(count) else {
            panic!("the array size overflows u32")
        };
        Self {
            name: "array",
            size,
            alignment: element.alignment,
            stride: size,
            kind: TypeLayoutKind::Array { element, count },
        }
    }

    /// The layout of a struct named `name` with the given `size`, `alignment` and `fields` (in declaration order). The
    /// stride is `size` rounded up to `alignment`.
    pub const fn structure(
        name: &'static str,
        size: u32,
        alignment: u32,
        fields: &'static [StructFieldLayout],
    ) -> Self {
        Self {
            name,
            size,
            alignment,
            stride: align_up(size, alignment),
            kind: TypeLayoutKind::Struct(fields),
        }
    }
}

/// One field in a macro-described host/GPU structure layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructFieldLayout {
    /// The field's name.
    pub name: &'static str,
    /// The field's byte offset from the start of the struct.
    pub offset: u32,
    /// The field's layout.
    pub ty: TypeLayout,
}

/// Layout rules recorded for a shader resource (not Slang compiler reflection).
///
/// API change (mx.t0b): the enum has a new variant, [`TextureV1`](Self::TextureV1), for the 2-D texture parameters of
/// decision D91. The enum is `#[non_exhaustive]`, so a `match` outside this crate needs a wildcard arm
/// (`gust-compute-wgpu`'s `generated_host.rs` has one). `TextureV1` has no byte layout of its own: the element is the
/// texel type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShaderLayoutRules {
    /// The v1 storage ABI: 32-bit scalars and recursively padding-free structs.
    StorageV1,
    /// The v2 storage ABI ([`TypeLayout::is_storage_v2`]): `StorageV1` plus 2- and 4-wide vectors and 2x2 and 4x4
    /// matrices, with the padding that alignment implies. The rules accept padded structs; a `GpuPod` type is only
    /// the padding-free subset of them.
    StorageV2,
    /// The constant-buffer ABI ([`TypeLayout::is_std140_v1`]): `StorageV2` with structs rounded up to 16 and no 2x2
    /// matrix.
    Std140V1,
    /// The texture rule (mx.t0b, decision D91): the facts of a 2-D texture binding that the reflection can witness,
    /// in a [`TextureLayout`]. The resource's `element` is the texture's result type (`float`, `uint4`, ...) and its
    /// `element_stride` the stride of that element. Only [`ResourceLayout::texture_v1`] builds it.
    TextureV1(TextureLayout),
}

/// The dimensionality of a texture binding. Only 2-D textures exist in the dialect (D62).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureDimension {
    /// A plain 2-D texture: no array, multisample or mip-chain variant.
    D2,
}

impl TextureDimension {
    /// The spelling used in the reflection JSON (`shape`): `2d`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::D2 => "2d",
        }
    }
}

/// How a kernel uses a texture binding, which is also how the reflection spells it (`access`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureAccess {
    /// A sampled `Texture2D<T>`: reflection access `read`.
    Sampled,
    /// A storage texture the kernel may read and write (`RWTexture2D<T>`): reflection access `readWrite`.
    ReadWrite,
    /// A storage texture the kernel only writes (`WTexture2D<T>`): reflection access `write`.
    WriteOnly,
}

impl TextureAccess {
    /// The spelling used in the reflection JSON (`access`): `read`, `readWrite` or `write`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sampled => "read",
            Self::ReadWrite => "readWrite",
            Self::WriteOnly => "write",
        }
    }

    /// Whether the access is a storage access, which carries a texel format in the shader.
    pub const fn is_storage(self) -> bool {
        !matches!(self, Self::Sampled)
    }
}

/// What the descriptor says about a texture binding, for the cross-check against the helper's reflection (D17, D91).
///
/// A **sampled** texture is compared by slot, dimension and sample class: its shader type has no format, so its
/// `format` is `None` and the concrete format is checked at submit against the bound texture. A **storage** or
/// **write-only** texture is compared by slot, access, result type and exact `format`, which is `Some`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureLayout {
    /// The exact texel format of a storage texture; `None` for a sampled one.
    pub format: Option<crate::texture::Format>,
    /// The dimensionality; always [`TextureDimension::D2`] today.
    pub dimension: TextureDimension,
    /// How the kernel uses the texture.
    pub access: TextureAccess,
    /// The class of value a read yields; for a sampled texture it comes from the result type, for a storage texture
    /// it is the format's.
    pub class: crate::texture::SampleClass,
}

/// Element ABI attached to one shader resource binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLayout {
    /// The ABI that `element` follows.
    pub rules: ShaderLayoutRules,
    /// The layout of one element (for a texture, the texel type as the shader sees it).
    pub element: TypeLayout,
    /// The distance in bytes between two consecutive elements in the buffer. For a constant buffer, which has no array,
    /// it is the element's size.
    pub element_stride: u32,
}

impl ResourceLayout {
    /// The layout of a storage buffer element under `StorageV1` ([`TypeLayout::is_storage_v1`]); the stride is the
    /// element's.
    pub const fn storage_v1(element: TypeLayout) -> Self {
        Self {
            rules: ShaderLayoutRules::StorageV1,
            element_stride: element.stride,
            element,
        }
    }

    /// The layout of a storage buffer element under `StorageV2` ([`TypeLayout::is_storage_v2`]); the stride is the
    /// element's.
    pub const fn storage_v2(element: TypeLayout) -> Self {
        Self {
            rules: ShaderLayoutRules::StorageV2,
            element_stride: element.stride,
            element,
        }
    }

    /// The layout of a constant buffer's element: it has no array stride, so the stride slot holds the element's
    /// size (which `Std140V1` rounds up to 16).
    pub const fn std140_v1(element: TypeLayout) -> Self {
        Self {
            rules: ShaderLayoutRules::Std140V1,
            element_stride: element.size,
            element,
        }
    }

    /// The layout of a 2-D texture binding (D64, D91): `element` is the texture's result type in the shader (a 32-bit
    /// scalar or a 2- to 4-wide vector of one), `texture` the facts the reflection witnesses. A sampled texture has
    /// `texture.format == None` and a storage one `Some(format)`.
    pub const fn texture_v1(element: TypeLayout, texture: TextureLayout) -> Self {
        Self {
            rules: ShaderLayoutRules::TextureV1(texture),
            element_stride: element.stride,
            element,
        }
    }
}

/// A bitwise-copyable Rust type whose representation exactly matches the GPU storage ABI, and which has no padding:
/// a `GpuPod` type is the padding-free subset of the layouts that [`TypeLayout::is_storage_v2`] accepts (which also
/// accepts padded structs such as `{ a: f32, v: Vec2, b: f32 }`). Padding bytes are uninitialized, and
/// [`pod_slice_as_bytes`] would read them, which is undefined behavior.
///
/// # Safety
///
/// The rule that binds an implementor is this one: no padding. Implementors must be `repr(C)` or `repr(transparent)`,
/// contain no padding or
/// invalid bit patterns, have a non-zero size, and make `LAYOUT` exactly match
/// Rust's size, alignment, field order, and offsets.
///
/// # Arrays
///
/// `[T; N]` implements `GpuPod` for `N` in 1..=32 and 64, 128, 256, 512, 1024, 2048 and 4096 (39 impls, no blanket
/// impl, so `[T; 0]` has none) when `T: GpuPod`, `T`'s std430 stride equals its Rust size, and `T` is not an array.
/// `Vec3` and `Mat3` are not `GpuPod`, so they cannot be elements; a hand-written element with a padded stride, or an
/// array element, fails when `LAYOUT` is used (a compile-time assertion).
///
/// ```
/// use gust::{GpuPod, TypeLayoutKind};
/// let layout = <[f32; 4] as GpuPod>::LAYOUT;
/// assert_eq!((layout.size, layout.alignment, layout.stride), (16, 4, 16));
/// assert!(matches!(layout.kind, TypeLayoutKind::Array { count: 4, .. }));
/// ```
///
/// A length outside the set has no impl:
///
/// ```compile_fail,E0277
/// fn needs<T: gust::GpuPod>() {}
/// needs::<[f32; 33]>();
/// ```
///
/// ```compile_fail,E0277
/// fn needs<T: gust::GpuPod>() {}
/// needs::<[f32; 0]>();
/// ```
///
/// A `Vec3` element is refused (its Rust stride is 12, the std430 stride 16):
///
/// ```compile_fail,E0277
/// fn needs<T: gust::GpuPod>() {}
/// needs::<[gust::prelude::Vec3; 2]>();
/// ```
///
/// An element whose layout stride differs from its Rust size is refused by the element rule (E0080 when `LAYOUT` is
/// used):
///
/// ```compile_fail,E0080
/// use gust::{GpuPod, ScalarKind, TypeLayout, TypeLayoutKind};
/// #[derive(Clone, Copy)]
/// #[repr(C)]
/// struct Padded([u32; 3]);
/// unsafe impl GpuPod for Padded {
///     const LAYOUT: TypeLayout = TypeLayout {
///         name: "Padded",
///         size: 12,
///         alignment: 4,
///         stride: 16,
///         kind: TypeLayoutKind::Scalar(ScalarKind::U32),
///     };
/// }
/// let _ = <[Padded; 2] as GpuPod>::LAYOUT;
/// ```
///
/// A nested array is refused (E0217 in the dialect):
///
/// ```compile_fail,E0080
/// let _ = <[[f32; 2]; 2] as gust::GpuPod>::LAYOUT;
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GUST GPU POD type",
    label = "declare a plain struct inside the same `#[gpu]` module so that the macro implements `GpuPod` for it; `u32`, `i32` and `f32` already implement it",
    note = "an `unsafe impl` written by hand must uphold the layout contract in the safety section of `GpuPod`",
    note = "arrays `[T; N]` implement it for N in 1..=32 and 64, 128, 256, 512, 1024, 2048, 4096 (not 0), when `T` does and its std430 stride equals its Rust size; `Vec3` and `Mat3` do not implement it, so they cannot be array elements"
)]
pub unsafe trait GpuPod: Copy + 'static {
    /// The layout of `Self`. It must match Rust's size, alignment, field order and offsets.
    const LAYOUT: TypeLayout;
}

unsafe impl GpuPod for u32 {
    const LAYOUT: TypeLayout = TypeLayout::scalar("u32", ScalarKind::U32);
}

unsafe impl GpuPod for i32 {
    const LAYOUT: TypeLayout = TypeLayout::scalar("i32", ScalarKind::I32);
}

unsafe impl GpuPod for f32 {
    const LAYOUT: TypeLayout = TypeLayout::scalar("f32", ScalarKind::F32);
}

/// The layout of `[T; N]` for the macro-generated impls below.
///
/// The element rule (D159) is enforced here, at compile time, when `LAYOUT` is used: the element's std430 stride must
/// equal its Rust size (so the Rust array is the GPU array byte for byte), its Rust alignment must equal its layout
/// alignment, and it must be a buffer type that is not itself an array (nested arrays are E0217). `Vec3` and `Mat3`
/// are not `GpuPod` at all; a hand-written `GpuPod` type with a padded stride trips this assertion. Written as an
/// associated const of a helper so that `&T::LAYOUT` is a `'static` reference.
struct ArrayOf<T>(core::marker::PhantomData<T>);

impl<T: GpuPod> ArrayOf<T> {
    const ELEMENT_OK: () = {
        let element = T::LAYOUT;
        assert!(
            element.stride as usize == mem::size_of::<T>()
                && element.size as usize == mem::size_of::<T>()
                && element.alignment as usize == mem::align_of::<T>()
                && element.is_storage_v2()
                && !matches!(element.kind, TypeLayoutKind::Array { .. }),
            "array element rule (D159): the element must be a non-array storage type whose std430 stride equals its Rust size (Vec3 and Mat3 do not qualify)"
        );
    };
    const ELEMENT: TypeLayout = {
        let () = Self::ELEMENT_OK;
        T::LAYOUT
    };
}

macro_rules! impl_gpu_pod_for_arrays {
    ($($n:literal)*) => {
        $(
            // SAFETY: `[T; N]` is `N` consecutive `T`s with no padding between them when `size_of::<T>()` is a multiple
            // of its alignment (always true in Rust), so it is padding-free when `T` is; `ArrayOf::ELEMENT_OK` makes
            // `LAYOUT` match: size `N * stride`, the element's alignment, element stride equal to the Rust size.
            unsafe impl<T: GpuPod> GpuPod for [T; $n] {
                const LAYOUT: TypeLayout = TypeLayout::array(&ArrayOf::<T>::ELEMENT, $n);
            }
        )*
    };
}

// No blanket impl: `[T; 0]` has none, so `GpuPod`'s non-zero-size contract is untouched (D159, 39 impls).
impl_gpu_pod_for_arrays!(
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    64 128 256 512 1024 2048 4096
);

/// View a POD slice as its storage-buffer bytes without allocating.
pub fn pod_slice_as_bytes<T: GpuPod>(values: &[T]) -> &[u8] {
    let byte_len = mem::size_of_val(values);
    // SAFETY: `GpuPod` guarantees every byte in every value is initialized and
    // accepts arbitrary bitwise copies. The returned slice shares `values`' lifetime.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }
}

const fn align_up(value: u32, alignment: u32) -> u32 {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

/// [`align_up`] for untrusted layout descriptions: `None` on a zero alignment or an overflow.
const fn checked_align_up(value: u32, alignment: u32) -> Option<u32> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{IVec2, IVec4, Mat2, Mat3, Mat4, UVec2, UVec4, Vec2, Vec3, Vec4};

    #[test]
    fn scalar_kind_all_lists_every_variant_once() {
        // Inside the defining crate `#[non_exhaustive]` does not apply, so this `match` has no wildcard arm and stops
        // compiling when a variant is added: add it to `ScalarKind::ALL` then.
        let index = |kind: ScalarKind| match kind {
            ScalarKind::U32 => 0,
            ScalarKind::I32 => 1,
            ScalarKind::F32 => 2,
        };
        assert_eq!(ScalarKind::ALL.len(), 3);
        for (position, kind) in ScalarKind::ALL.iter().enumerate() {
            assert_eq!(index(*kind), position);
        }
    }

    #[test]
    fn scalar_layouts_are_explicit() {
        assert_eq!(<f32 as GpuPod>::LAYOUT.size, 4);
        assert_eq!(<f32 as GpuPod>::LAYOUT.alignment, 4);
        assert_eq!(<f32 as GpuPod>::LAYOUT.stride, 4);
        assert_eq!(
            ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT).element_stride,
            4
        );
    }

    // Evaluated by the compiler: a regression of `is_storage_v1` to a non-const fn, or to a wrong answer on these
    // layouts, fails the build of the test module.
    const U32_FIELDS: &[StructFieldLayout] = &[
        StructFieldLayout {
            name: "a",
            offset: 0,
            ty: TypeLayout::scalar("u32", ScalarKind::U32),
        },
        StructFieldLayout {
            name: "b",
            offset: 4,
            ty: TypeLayout::scalar("u32", ScalarKind::U32),
        },
    ];
    const PADDED_FIELDS: &[StructFieldLayout] = &[
        StructFieldLayout {
            name: "a",
            offset: 0,
            ty: TypeLayout::scalar("u32", ScalarKind::U32),
        },
        StructFieldLayout {
            name: "b",
            offset: 8,
            ty: TypeLayout::scalar("u32", ScalarKind::U32),
        },
    ];
    const SCALAR_OK: bool = <u32 as GpuPod>::LAYOUT.is_storage_v1();
    const PAIR_OK: bool = TypeLayout::structure("Pair", 8, 4, U32_FIELDS).is_storage_v1();
    const NESTED_OK: bool = TypeLayout::structure(
        "Nested",
        16,
        4,
        &[
            StructFieldLayout {
                name: "first",
                offset: 0,
                ty: TypeLayout::structure("Pair", 8, 4, U32_FIELDS),
            },
            StructFieldLayout {
                name: "second",
                offset: 8,
                ty: TypeLayout::structure("Pair", 8, 4, U32_FIELDS),
            },
        ],
    )
    .is_storage_v1();
    const PADDED_OK: bool = TypeLayout::structure("Padded", 12, 4, PADDED_FIELDS).is_storage_v1();
    // Sizes add up to the total (12) but field `c` sits at 12 instead of 8: only the per-field offset check sees it.
    const LATE_PADDING_OK: bool = TypeLayout::structure(
        "Late",
        12,
        4,
        &[
            StructFieldLayout {
                name: "a",
                offset: 0,
                ty: TypeLayout::scalar("u32", ScalarKind::U32),
            },
            StructFieldLayout {
                name: "b",
                offset: 4,
                ty: TypeLayout::scalar("u32", ScalarKind::U32),
            },
            StructFieldLayout {
                name: "c",
                offset: 12,
                ty: TypeLayout::scalar("u32", ScalarKind::U32),
            },
        ],
    )
    .is_storage_v1();

    #[test]
    #[allow(clippy::assertions_on_constants)] // the constants are evaluated by the compiler on purpose
    fn storage_v1_is_decided_in_const_context() {
        assert!(SCALAR_OK);
        assert!(PAIR_OK);
        assert!(NESTED_OK);
        // Padded: field `b` at offset 8 instead of 4, and size 12 instead of 8.
        assert!(!PADDED_OK);
        // Only the per-field offset check rejects this one (the field sizes sum to the total).
        assert!(!LATE_PADDING_OK);
        // The const results agree with the run-time calls.
        assert_eq!(SCALAR_OK, <u32 as GpuPod>::LAYOUT.is_storage_v1());
        assert_eq!(
            PADDED_OK,
            TypeLayout::structure("Padded", 12, 4, PADDED_FIELDS).is_storage_v1()
        );
    }

    #[test]
    fn pod_bytes_preserve_scalar_bits() {
        let values = [1.0_f32, -2.5];
        let bytes = pod_slice_as_bytes(&values);
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[..4], &1.0_f32.to_ne_bytes());
        assert_eq!(&bytes[4..], &(-2.5_f32).to_ne_bytes());
    }

    // The layout model, from the numbers measured on slangc 2026.13.1 for both targets (plan F4, F5, F7;
    // `probe/layouts.json`, `probe/review/opus1/cb.json`).

    const F: TypeLayout = TypeLayout::scalar("f32", ScalarKind::F32);
    const V2: TypeLayout = TypeLayout::vector("Vec2", ScalarKind::F32, 2);
    const V4: TypeLayout = TypeLayout::vector("Vec4", ScalarKind::F32, 4);
    const M2: TypeLayout = TypeLayout::matrix("Mat2", ScalarKind::F32, 2, 2);
    const M4: TypeLayout = TypeLayout::matrix("Mat4", ScalarKind::F32, 4, 4);

    const fn field(name: &'static str, offset: u32, ty: TypeLayout) -> StructFieldLayout {
        StructFieldLayout { name, offset, ty }
    }

    fn triple(layout: TypeLayout) -> (u32, u32, u32) {
        (layout.size, layout.alignment, layout.stride)
    }

    /// `TypeLayout::structure` for fields built at run time (its slice must be `'static`).
    fn make_struct(
        name: &'static str,
        size: u32,
        alignment: u32,
        fields: &[StructFieldLayout],
    ) -> TypeLayout {
        TypeLayout::structure(
            name,
            size,
            alignment,
            Box::leak(fields.to_vec().into_boxed_slice()),
        )
    }

    #[test]
    fn vector_layouts_are_size_alignment_stride() {
        // F4: float2 8/8/8, float4 16/16/16; the integer forms are the same.
        assert_eq!(triple(V2), (8, 8, 8));
        assert_eq!(triple(V4), (16, 16, 16));
        assert_eq!(
            triple(TypeLayout::vector("IVec2", ScalarKind::I32, 2)),
            (8, 8, 8)
        );
        assert_eq!(
            triple(TypeLayout::vector("UVec4", ScalarKind::U32, 4)),
            (16, 16, 16)
        );
        assert!(V2.is_storage_v2() && V4.is_storage_v2());
        assert!(V2.is_std140_v1() && V4.is_std140_v1());
        // `float3` is size 12, alignment 16 in Slang and has no Rust match: not a buffer type under any rule.
        let v3 = TypeLayout::vector("Vec3", ScalarKind::F32, 3);
        assert_eq!(triple(v3), (12, 16, 12));
        assert!(!v3.is_storage_v2() && !v3.is_std140_v1() && !v3.is_storage_v1());
        // A vector is not StorageV1, whatever its numbers.
        assert!(!V2.is_storage_v1() && !V4.is_storage_v1());
        // A vector with the wrong numbers is refused.
        let mut wrong = V4;
        wrong.alignment = 4;
        assert!(!wrong.is_storage_v2());
        wrong = V2;
        wrong.size = 12;
        wrong.stride = 12;
        assert!(!wrong.is_storage_v2());
    }

    #[test]
    fn matrix_layouts_are_column_major_with_column_stride() {
        // F5: float2x2 column stride 8, size 16 (alignment 8); float4x4 column stride 16, size 64 (alignment 16).
        assert_eq!(triple(M2), (16, 8, 16));
        assert_eq!(triple(M4), (64, 16, 64));
        assert!(matches!(
            M2.kind,
            TypeLayoutKind::Matrix {
                rows: 2,
                cols: 2,
                column_stride: 8,
                ..
            }
        ));
        assert!(matches!(
            M4.kind,
            TypeLayoutKind::Matrix {
                rows: 4,
                cols: 4,
                column_stride: 16,
                ..
            }
        ));
        assert!(M2.is_storage_v2() && M4.is_storage_v2());
        assert!(!M2.is_storage_v1() && !M4.is_storage_v1());
        // `float3x3` (three padded columns) and the rectangular forms are not buffer types.
        let m3 = TypeLayout::matrix("Mat3", ScalarKind::F32, 3, 3);
        assert_eq!(triple(m3), (48, 16, 48));
        assert!(!m3.is_storage_v2() && !m3.is_std140_v1());
        let m43 = TypeLayout::matrix("float4x3", ScalarKind::F32, 4, 3);
        assert_eq!(m43.size, 48);
        assert!(!m43.is_storage_v2());
        // Only float matrices exist.
        assert!(!TypeLayout::matrix("Imat", ScalarKind::I32, 4, 4).is_storage_v2());
        // The column stride is part of the description: a Mat4 with a row-sized stride is wrong.
        let mut wrong = M4;
        wrong.kind = TypeLayoutKind::Matrix {
            scalar: ScalarKind::F32,
            rows: 4,
            cols: 4,
            column_stride: 8,
        };
        assert!(!wrong.is_storage_v2());
        // In a constant buffer the 2x2 matrix has column stride 16 (size 32), which no Rust type matches.
        assert!(!M2.is_std140_v1());
        assert!(M4.is_std140_v1());
    }

    #[test]
    fn matrix_layout_pin_is_column() {
        assert_eq!(MATRIX_LAYOUT, MatrixLayout::Column);
        assert_eq!(MATRIX_LAYOUT.as_str(), "column");
        assert_eq!(MatrixLayout::parse("column"), Some(MatrixLayout::Column));
        assert_eq!(MatrixLayout::parse("row"), Some(MatrixLayout::Row));
        assert_eq!(MatrixLayout::parse("Column"), None);
        assert_eq!(MatrixLayout::parse(""), None);
        assert_eq!(MatrixLayout::Row.as_str(), "row");
    }

    const F_V2_F: TypeLayout = TypeLayout::structure(
        "FV2F",
        24,
        8,
        &[field("a", 0, F), field("v", 8, V2), field("b", 16, F)],
    );
    const F_V4_F: TypeLayout = TypeLayout::structure(
        "FV4F",
        48,
        16,
        &[field("a", 0, F), field("v", 16, V4), field("b", 32, F)],
    );
    const V2_V2_F: TypeLayout = TypeLayout::structure(
        "V2V2F",
        24,
        8,
        &[field("v", 0, V2), field("w", 8, V2), field("f", 16, F)],
    );

    #[test]
    fn storage_struct_offsets_follow_field_alignment() {
        // `{a, v2, b}` -> a0 v8 b16, stride 24; `{a, v4, b}` -> a0 v16 b32, stride 48; `{v2, v2, f}` stride 24.
        assert_eq!(triple(F_V2_F), (24, 8, 24));
        assert_eq!(triple(F_V4_F), (48, 16, 48));
        assert_eq!(triple(V2_V2_F), (24, 8, 24));
        for layout in [F_V2_F, F_V4_F, V2_V2_F] {
            assert!(layout.is_storage_v2(), "{}", layout.name);
            assert!(!layout.is_storage_v1(), "{}", layout.name);
        }
        // The same fields placed as `#[repr(C)]` of unaligned types would (v2 at 4) are not the Slang layout.
        let packed = make_struct(
            "Packed",
            16,
            4,
            &[field("a", 0, F), field("v", 4, V2), field("b", 12, F)],
        );
        assert!(!packed.is_storage_v2());
        // A vector at the right offset but a size that was not rounded up to the alignment.
        let unrounded = make_struct(
            "Unrounded",
            20,
            8,
            &[field("a", 0, F), field("v", 8, V2), field("b", 16, F)],
        );
        assert_eq!(unrounded.stride, 24);
        assert!(!unrounded.is_storage_v2());
        // The struct's alignment must be the largest field alignment.
        let low = make_struct(
            "Low",
            24,
            4,
            &[field("a", 0, F), field("v", 8, V2), field("b", 16, F)],
        );
        assert!(!low.is_storage_v2());
        // A trailing scalar after a vec4 pads the struct to 32.
        let tail = make_struct("Tail", 32, 16, &[field("v", 0, V4), field("a", 16, F)]);
        assert!(tail.is_storage_v2());
        // A struct nested in a struct keeps its own alignment (here 8): `{a, F_V2_F}` puts the inner at 8.
        let nested = make_struct(
            "Nested",
            32,
            8,
            &[field("a", 0, F), field("inner", 8, F_V2_F)],
        );
        assert!(nested.is_storage_v2());
        let nested_early = make_struct(
            "NestedEarly",
            28,
            4,
            &[field("a", 0, F), field("inner", 4, F_V2_F)],
        );
        assert!(!nested_early.is_storage_v2());
        // A matrix field: `{a, m4}` -> m at 16.
        let with_matrix = make_struct(
            "WithMatrix",
            80,
            16,
            &[field("a", 0, F), field("m", 16, M4)],
        );
        assert!(with_matrix.is_storage_v2());
        // The empty struct and a field at a wild offset are refused.
        assert!(!TypeLayout::structure("Empty", 0, 4, &[]).is_storage_v2());
        let wild = make_struct("Wild", 8, 8, &[field("v", u32::MAX - 2, V2)]);
        assert!(!wild.is_storage_v2());
        // A nested layout that is not itself valid poisons the struct.
        let mut bad = V2;
        bad.alignment = 4;
        let outer = make_struct("Outer", 8, 4, &[field("v", 0, bad)]);
        assert!(!outer.is_storage_v2());
    }

    #[test]
    fn storage_v1_layouts_are_the_scalar_sub_case_of_v2() {
        for layout in [
            <u32 as GpuPod>::LAYOUT,
            <f32 as GpuPod>::LAYOUT,
            TypeLayout::structure("Pair", 8, 4, U32_FIELDS),
        ] {
            assert!(layout.is_storage_v1());
            assert!(layout.is_storage_v2());
        }
        // The v1 refusals stay refusals under v2 where the layout is padded wrongly.
        assert!(!TypeLayout::structure("Padded", 12, 4, PADDED_FIELDS).is_storage_v2());
    }

    // Std140: `{E{a}; c}`, from the measured `{E{a,b,c}; d}` case: the member after a nested struct starts at the
    // struct's size rounded up to 16.
    const E_A: TypeLayout = TypeLayout::structure("E", 16, 16, &[field("a", 0, F)]);
    const E_THEN_C: TypeLayout =
        TypeLayout::structure("EC", 32, 16, &[field("e", 0, E_A), field("c", 16, F)]);
    // The review's cb.json: `struct P { float a; float4 v; float4x4 m; E e; float2 w; float b; }`.
    const CB_P: TypeLayout = TypeLayout::structure(
        "P",
        128,
        16,
        &[
            field("a", 0, F),
            field("v", 16, V4),
            field("m", 32, M4),
            field("e", 96, E_A),
            field("w", 112, V2),
            field("b", 120, F),
        ],
    );

    #[test]
    fn std140_rounds_nested_structs_and_the_whole_struct_to_16() {
        assert_eq!(triple(E_A), (16, 16, 16));
        assert!(E_A.is_std140_v1());
        // `{E{a}; c}`: c at 16, size 32.
        assert_eq!(triple(E_THEN_C), (32, 16, 32));
        assert!(E_THEN_C.is_std140_v1());
        let expected: [(&str, u32); 6] = [
            ("a", 0),
            ("v", 16),
            ("m", 32),
            ("e", 96),
            ("w", 112),
            ("b", 120),
        ];
        let TypeLayoutKind::Struct(fields) = CB_P.kind else {
            panic!("struct")
        };
        for (field, (name, offset)) in fields.iter().zip(expected) {
            assert_eq!((field.name, field.offset), (name, offset));
        }
        // `{a; v4; m4; E; w2; b}` -> v16 m32 e96 w112 b120, size 128.
        assert_eq!(triple(CB_P), (128, 16, 128));
        assert!(CB_P.is_std140_v1());
        // The same description is not StorageV2: there a nested struct has its own small alignment.
        assert!(!E_A.is_storage_v2());
        assert!(!CB_P.is_storage_v2());
        // Std140 refusals: a nested struct with storage numbers (size 4), a total that was not rounded (124), a field
        // before its rounded offset, a struct whose alignment stayed 8, a 3-wide vector and a 2x2 matrix.
        let storage_e = make_struct("E", 4, 4, &[field("a", 0, F)]);
        assert!(!storage_e.is_std140_v1());
        let unrounded = make_struct(
            "P",
            124,
            16,
            &[
                field("a", 0, F),
                field("v", 16, V4),
                field("m", 32, M4),
                field("e", 96, E_A),
                field("w", 112, V2),
                field("b", 120, F),
            ],
        );
        assert!(!unrounded.is_std140_v1());
        let early = make_struct("EC", 32, 16, &[field("e", 0, E_A), field("c", 4, F)]);
        assert!(!early.is_std140_v1());
        let small = make_struct("S", 16, 8, &[field("v", 0, V2), field("w", 8, V2)]);
        assert!(small.is_storage_v2());
        assert!(!small.is_std140_v1());
        let with_m2 = make_struct("S", 32, 16, &[field("m", 0, M2)]);
        assert!(!with_m2.is_std140_v1());
        // A `vec2` then a scalar packs as in storage inside a rounded struct: `{v2, f}` -> f at 8, size 16.
        let packed = make_struct("S", 16, 16, &[field("v", 0, V2), field("f", 8, F)]);
        assert!(packed.is_std140_v1());
        // Scalars only: size 4 rounds to a 16-byte struct.
        let single = make_struct("S", 16, 16, &[field("f", 0, F)]);
        assert!(single.is_std140_v1());
        let single_storage = make_struct("S", 4, 4, &[field("f", 0, F)]);
        assert!(!single_storage.is_std140_v1());
    }

    #[test]
    fn resource_layouts_name_their_rules_and_strides() {
        let storage = ResourceLayout::storage_v2(F_V2_F);
        assert_eq!(storage.rules, ShaderLayoutRules::StorageV2);
        assert_eq!(storage.element_stride, 24);
        assert_eq!(
            ResourceLayout::storage_v1(F).rules,
            ShaderLayoutRules::StorageV1
        );
        let uniform = ResourceLayout::std140_v1(CB_P);
        assert_eq!(uniform.rules, ShaderLayoutRules::Std140V1);
        assert_eq!(uniform.element_stride, 128);
        assert_eq!(uniform.element, CB_P);
    }

    #[test]
    fn a_texture_layout_carries_the_facts_the_reflection_witnesses() {
        use crate::texture::{Format, SampleClass};
        let element = <crate::math::Vec4 as GpuPod>::LAYOUT;
        let sampled = TextureLayout {
            format: None,
            dimension: TextureDimension::D2,
            access: TextureAccess::Sampled,
            class: SampleClass::Float,
        };
        let layout = ResourceLayout::texture_v1(element, sampled);
        assert_eq!(layout.rules, ShaderLayoutRules::TextureV1(sampled));
        // The element is the result type and the stride its stride, as for a buffer.
        assert_eq!((layout.element, layout.element_stride), (element, 16));
        let storage = TextureLayout {
            format: Some(Format::R32Uint),
            access: TextureAccess::ReadWrite,
            class: SampleClass::Uint,
            ..sampled
        };
        assert_ne!(
            ResourceLayout::texture_v1(F, storage),
            ResourceLayout::texture_v1(F, sampled)
        );
        assert_ne!(
            ShaderLayoutRules::TextureV1(storage),
            ShaderLayoutRules::StorageV2
        );
        // The names are the reflection's.
        assert_eq!(TextureDimension::D2.name(), "2d");
        let names = [
            TextureAccess::Sampled,
            TextureAccess::ReadWrite,
            TextureAccess::WriteOnly,
        ]
        .map(TextureAccess::name);
        assert_eq!(names, ["read", "readWrite", "write"]);
        assert!(!TextureAccess::Sampled.is_storage());
        assert!(TextureAccess::ReadWrite.is_storage() && TextureAccess::WriteOnly.is_storage());
    }

    // `GpuPod` for the buffer-eligible math types: the model's numbers are Rust's.

    fn check_pod<T: GpuPod>(size: usize, alignment: usize) {
        let layout = T::LAYOUT;
        assert_eq!(
            (mem::size_of::<T>(), mem::align_of::<T>()),
            (size, alignment)
        );
        assert_eq!(layout.size as usize, size, "{}", layout.name);
        assert_eq!(layout.alignment as usize, alignment, "{}", layout.name);
        assert_eq!(layout.stride as usize, size, "{}", layout.name);
        assert!(layout.is_storage_v2(), "{}", layout.name);
        assert!(!layout.is_storage_v1(), "{}", layout.name);
    }

    #[test]
    fn math_types_are_gpu_pod_with_their_rust_layout() {
        check_pod::<Vec2>(8, 8);
        check_pod::<IVec2>(8, 8);
        check_pod::<UVec2>(8, 8);
        check_pod::<Vec4>(16, 16);
        check_pod::<IVec4>(16, 16);
        check_pod::<UVec4>(16, 16);
        check_pod::<Mat2>(16, 8);
        check_pod::<Mat4>(64, 16);
        assert_eq!(
            <Vec2 as GpuPod>::LAYOUT.kind,
            TypeLayoutKind::Vector {
                scalar: ScalarKind::F32,
                count: 2
            }
        );
        assert_eq!(
            <IVec4 as GpuPod>::LAYOUT.kind,
            TypeLayoutKind::Vector {
                scalar: ScalarKind::I32,
                count: 4
            }
        );
        assert_eq!(
            <UVec2 as GpuPod>::LAYOUT.kind,
            TypeLayoutKind::Vector {
                scalar: ScalarKind::U32,
                count: 2
            }
        );
        assert_eq!(
            <Mat4 as GpuPod>::LAYOUT.kind,
            TypeLayoutKind::Matrix {
                scalar: ScalarKind::F32,
                rows: 4,
                cols: 4,
                column_stride: 16
            }
        );
        // The 3-wide types are locals only: their Rust layouts do not match any buffer position.
        assert_eq!((mem::size_of::<Vec3>(), mem::align_of::<Vec3>()), (12, 4));
        assert_eq!((mem::size_of::<Mat3>(), mem::align_of::<Mat3>()), (36, 4));
    }

    #[test]
    fn pod_bytes_of_math_types_are_their_components_in_order() {
        let values = [Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 }];
        let bytes = pod_slice_as_bytes(&values);
        assert_eq!(bytes.len(), 16);
        let floats: Vec<f32> = bytes
            .chunks(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(floats, [1.0, 2.0, 3.0, 4.0]);
        let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let floats: Vec<f32> = pod_slice_as_bytes(&[m])
            .chunks(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        // Column-major: the first column is contiguous.
        assert_eq!(floats, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rust_repr_c_struct_of_math_types_matches_the_model() {
        // The measured storage structs, as Rust declares them: the offsets equal the model's.
        #[repr(C)]
        struct FV2F {
            a: f32,
            v: Vec2,
            b: f32,
        }
        #[repr(C)]
        struct FV4F {
            a: f32,
            v: Vec4,
            b: f32,
        }
        #[repr(C)]
        struct V2V2F {
            v: Vec2,
            w: Vec2,
            f: f32,
        }
        assert_eq!(
            (
                mem::offset_of!(FV2F, v),
                mem::offset_of!(FV2F, b),
                mem::size_of::<FV2F>()
            ),
            (8, 16, 24)
        );
        assert_eq!(
            (
                mem::offset_of!(FV4F, v),
                mem::offset_of!(FV4F, b),
                mem::size_of::<FV4F>()
            ),
            (16, 32, 48)
        );
        assert_eq!(
            (
                mem::offset_of!(V2V2F, w),
                mem::offset_of!(V2V2F, f),
                mem::size_of::<V2V2F>()
            ),
            (8, 16, 24)
        );
        assert_eq!(F_V2_F.size as usize, mem::size_of::<FV2F>());
        assert_eq!(F_V4_F.size as usize, mem::size_of::<FV4F>());
        assert_eq!(V2_V2_F.size as usize, mem::size_of::<V2V2F>());
    }

    // Arrays (m9.r2b, D159). The numbers are spike P4b's measured std430 layouts (naga Layouter = Slang reflection =
    // SPIR-V), written out by hand.

    const A_F4: TypeLayout = TypeLayout::array(&F, 4);
    const A_V2_3: TypeLayout = TypeLayout::array(&V2, 3);
    const A_M4_2: TypeLayout = TypeLayout::array(&M4, 2);
    const A_M2_2: TypeLayout = TypeLayout::array(&M2, 2);
    const A_V4_2: TypeLayout = TypeLayout::array(&V4, 2);
    const PAIR: TypeLayout = TypeLayout::structure("Pair", 8, 4, U32_FIELDS);
    const A_PAIR_3: TypeLayout = TypeLayout::array(&PAIR, 3);
    const Q: TypeLayout = TypeLayout::structure("Q", 8, 4, &[field("x", 0, F), field("y", 4, F)]);
    const A_Q_3: TypeLayout = TypeLayout::array(&Q, 3);
    // struct R { Q q[3]; uint n; }: size 28, alignment 4, q at 0, n at 24 (spike P4b).
    const R: TypeLayout = TypeLayout::structure(
        "R",
        28,
        4,
        &[
            field("q", 0, A_Q_3),
            field("n", 24, TypeLayout::scalar("u32", ScalarKind::U32)),
        ],
    );
    // struct S { float a[4]; float2 b[3]; }: a at 0 (16 bytes), b at 16 (24 bytes, alignment 8), size 40, alignment 8.
    const S: TypeLayout =
        TypeLayout::structure("S", 40, 8, &[field("a", 0, A_F4), field("b", 16, A_V2_3)]);

    #[test]
    fn array_layouts_are_the_measured_std430_numbers() {
        assert_eq!(triple(A_F4), (16, 4, 16));
        assert_eq!(triple(A_V2_3), (24, 8, 24));
        assert_eq!(triple(A_V4_2), (32, 16, 32));
        assert_eq!(triple(A_M2_2), (32, 8, 32));
        assert_eq!(triple(A_M4_2), (128, 16, 128));
        assert_eq!(triple(A_PAIR_3), (24, 4, 24));
        assert_eq!(triple(A_Q_3), (24, 4, 24));
        assert_eq!(A_F4.name, "array");
        assert_eq!(
            A_F4.kind,
            TypeLayoutKind::Array {
                element: &F,
                count: 4
            }
        );
        // A float array is StorageV1 too (alignment 4); one with a vector element is StorageV2 only.
        assert!(A_F4.is_storage_v1() && A_F4.is_storage_v2());
        assert!(A_PAIR_3.is_storage_v1() && A_PAIR_3.is_storage_v2());
        assert!(!A_V2_3.is_storage_v1());
        for layout in [A_V2_3, A_V4_2, A_M2_2, A_M4_2, A_Q_3] {
            assert!(layout.is_storage_v2(), "{layout:?}");
        }
        // Arrays are not constant-buffer types (E0215): a std140 array stride would be rounded up to 16.
        for layout in [A_F4, A_V2_3, A_V4_2, A_M4_2, A_PAIR_3, S, R] {
            assert!(!layout.is_std140_v1(), "{layout:?}");
        }
    }

    #[test]
    fn a_struct_with_array_fields_follows_the_storage_rules() {
        assert_eq!(triple(S), (40, 8, 40));
        assert!(S.is_storage_v2() && !S.is_storage_v1());
        assert_eq!(triple(R), (28, 4, 28));
        assert!(R.is_storage_v2() && R.is_storage_v1());
        // `b` pushed to 20 (not aligned to 8) or the struct sized wrongly is refused.
        let moved = make_struct("S", 48, 8, &[field("a", 0, A_F4), field("b", 20, A_V2_3)]);
        assert!(!moved.is_storage_v2());
        let packed = make_struct("S", 36, 8, &[field("a", 0, A_F4), field("b", 12, A_V2_3)]);
        assert!(!packed.is_storage_v2());
        // The same offsets in Rust.
        #[repr(C)]
        struct RustS {
            a: [f32; 4],
            b: [crate::math::Vec2; 3],
        }
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct RustQ {
            x: f32,
            y: f32,
        }
        #[repr(C)]
        struct RustR {
            q: [RustQ; 3],
            n: u32,
        }
        assert_eq!(
            (
                mem::offset_of!(RustS, b),
                mem::size_of::<RustS>(),
                mem::align_of::<RustS>()
            ),
            (16, 40, 8)
        );
        assert_eq!(
            (mem::offset_of!(RustR, n), mem::size_of::<RustR>()),
            (24, 28)
        );
    }

    #[test]
    fn array_descriptions_with_wrong_numbers_are_refused() {
        // Count, size, alignment, stride each wrong alone.
        let mut wrong = A_V2_3;
        wrong.size = 32;
        wrong.stride = 32;
        assert!(!wrong.is_storage_v2());
        wrong = A_V2_3;
        wrong.alignment = 4;
        assert!(!wrong.is_storage_v2());
        wrong = A_V2_3;
        wrong.stride = 32;
        assert!(!wrong.is_storage_v2());
        wrong = A_V2_3;
        wrong.kind = TypeLayoutKind::Array {
            element: &V2,
            count: 4,
        };
        assert!(!wrong.is_storage_v2());
        wrong.kind = TypeLayoutKind::Array {
            element: &V2,
            count: 0,
        };
        assert!(!wrong.is_storage_v2());
        // An element with a padded stride (a three-wide vector: Rust 12, std430 16) is not an array element, and an
        // array of arrays is refused (E0217).
        const V3: TypeLayout = TypeLayout::vector("Vec3", ScalarKind::F32, 3);
        assert_eq!(triple(TypeLayout::array(&V3, 2)), (24, 16, 24));
        assert!(!TypeLayout::array(&V3, 2).is_storage_v2());
        let nested = TypeLayout::array(&A_F4, 2);
        assert_eq!(triple(nested), (32, 4, 32));
        assert!(!nested.is_storage_v2() && !nested.is_storage_v1());
        // A padded struct stride: an element whose stride exceeds its size cannot be packed.
        const PADDED_EL: TypeLayout = TypeLayout {
            name: "P",
            size: 12,
            alignment: 8,
            stride: 16,
            kind: TypeLayoutKind::Struct(&[]),
        };
        let padded = TypeLayout::array(&PADDED_EL, 2);
        assert_eq!(padded.size, 32);
        assert!(!padded.is_storage_v2());
    }

    fn check_array_pod<T: GpuPod>(size: usize, alignment: usize, count: u32) {
        let layout = T::LAYOUT;
        assert_eq!(
            (mem::size_of::<T>(), mem::align_of::<T>()),
            (size, alignment)
        );
        assert_eq!(layout.size as usize, size);
        assert_eq!(layout.alignment as usize, alignment);
        assert_eq!(layout.stride as usize, size);
        assert!(layout.is_storage_v2());
        let TypeLayoutKind::Array { count: got, .. } = layout.kind else {
            panic!("an array expected, got {layout:?}")
        };
        assert_eq!(got, count);
    }

    #[test]
    fn array_gpu_pod_impls_match_rusts_layout() {
        check_array_pod::<[u32; 4]>(16, 4, 4);
        check_array_pod::<[Vec2; 3]>(24, 8, 3);
        check_array_pod::<[Vec4; 2]>(32, 16, 2);
        check_array_pod::<[Mat2; 2]>(32, 8, 2);
        check_array_pod::<[Mat4; 2]>(128, 16, 2);
        check_array_pod::<[f32; 1]>(4, 4, 1);
        check_array_pod::<[f32; 32]>(128, 4, 32);
        check_array_pod::<[f32; 4096]>(16384, 4, 4096);
        let TypeLayoutKind::Array { element, .. } = <[Mat4; 2] as GpuPod>::LAYOUT.kind else {
            panic!("an array expected")
        };
        assert_eq!(*element, M4);
        // Every length of the set has an impl whose count and size are N and 4N.
        macro_rules! every_length {
            ($($n:literal)*) => {$(
                check_array_pod::<[f32; $n]>(4 * $n, 4, $n);
            )*};
        }
        every_length!(
            1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
            64 128 256 512 1024 2048 4096
        );
        // The bytes are the elements back to back.
        let bytes = pod_slice_as_bytes(&[[1_u32, 2, 3, 4], [5, 6, 7, 8]]);
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[16..20], &5_u32.to_ne_bytes());
    }
}
