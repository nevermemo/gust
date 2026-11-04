//! Textures and samplers in the GUST dialect (decision D64, slice mx.t1).
//!
//! This module holds three things, none of which names a `wgpu` type (D02):
//!
//! - **The formats.** [`Format`] is the closed list of texel formats of the dialect and [`TextureFormat`] the sealed
//!   trait of its marker types ([`R32Uint`] to [`Rgba32Float`]), which tie a format to the Rust type of one texel.
//!   `gust-wgpu` maps [`Format`] to `wgpu::TextureFormat` in one place.
//! - **The host-side vocabulary.** [`TextureUse`] (what a texture may be used for), [`SamplerOptions`], [`Filter`] and
//!   [`Address`].
//! - **The kernel-side types.** [`Texture2D`], [`StorageTexture2D`], [`WriteTexture2D`] and [`Sampler`] are what a
//!   `#[gpu]` kernel's parameters are in the Rust shadow body, as `Storage` and `StorageMut` are for buffers. They are
//!   views over host memory: the CPU twin builds them from a [`CpuTexture2d`](crate::cpu::texture::CpuTexture2d) and
//!   runs the shadow body, so they carry the twin's semantics of `load`, `store` and `sample`, written in
//!   [`cpu::texture`](crate::cpu::texture) (the conventions) and in `docs/SEMANTICS.md`.
//!
//! | Kernel parameter | Access | Methods |
//! | --- | --- | --- |
//! | `Texture2D<T>` | sampled, read | `load`, `sample` (float `T`), `width`, `height` |
//! | `StorageTexture2D<F>` (`R32Uint`, `R32Sint`, `R32Float`) | storage read-write | `load`, `store`, `width`, `height` |
//! | `WriteTexture2D<F>` (any format of the list) | storage write-only | `store`, `width`, `height` |
//! | `Sampler` | none | the argument of `sample` |
//!
//! Coordinates are `u32` texel addresses and an out-of-range one panics on the twin ("index out of bounds", as
//! KM-11 does for a buffer); `sample` takes normalised `f32` coordinates. The twin's `load` and `store` are exact; its
//! `sample` is the formula of the WGSL specification and lies outside the exact contract (D22, D39).
#![warn(missing_docs)]

use core::fmt;
use core::marker::PhantomData;
use core::ops::BitOr;

use crate::GpuPod;
use crate::cpu::texture::{sample_2d, texel_index};
use crate::math::{IVec2, IVec4, UVec2, UVec4, Vec2, Vec4};

mod sealed {
    pub trait Sealed {}
}

/// The class of value a texture format yields when it is read: what a sampled `Texture2D<T>` parameter's `T` must
/// agree with (`f32` forms for [`Float`](Self::Float), `u32` for [`Uint`](Self::Uint), `i32` for [`Sint`](Self::Sint)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleClass {
    /// `f32` texels: WGSL `texture_2d<f32>`.
    Float,
    /// `u32` texels: WGSL `texture_2d<u32>`. Integers are never filtered.
    Uint,
    /// `i32` texels: WGSL `texture_2d<i32>`. Integers are never filtered.
    Sint,
}

impl SampleClass {
    /// The class's name: `f32`, `u32` or `i32`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float => "f32",
            Self::Uint => "u32",
            Self::Sint => "i32",
        }
    }
}

/// A texel format of the dialect: the nine 32-bit formats whose store is unconverted (decisions D62 and D64). The 8- and
/// 16-bit formats convert on store and are outside D22's exact contract, so they are not in the list.
///
/// The names follow WebGPU's format names, so `gust-wgpu` maps a variant to a `wgpu::TextureFormat` with one `match`
/// arm each. The enum is `#[non_exhaustive]`: a later format is not a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// One `u32` channel.
    R32Uint,
    /// One `i32` channel.
    R32Sint,
    /// One `f32` channel.
    R32Float,
    /// Two `u32` channels.
    Rg32Uint,
    /// Two `i32` channels.
    Rg32Sint,
    /// Two `f32` channels.
    Rg32Float,
    /// Four `u32` channels.
    Rgba32Uint,
    /// Four `i32` channels.
    Rgba32Sint,
    /// Four `f32` channels.
    Rgba32Float,
}

impl Format {
    /// Every format of the dialect, in the order of the enum.
    pub const ALL: [Self; 9] = [
        Self::R32Uint,
        Self::R32Sint,
        Self::R32Float,
        Self::Rg32Uint,
        Self::Rg32Sint,
        Self::Rg32Float,
        Self::Rgba32Uint,
        Self::Rgba32Sint,
        Self::Rgba32Float,
    ];

    /// WebGPU's name of the format, which is also its WGSL storage-texture format name: `r32uint`, `rg32float`, ...
    pub const fn name(self) -> &'static str {
        match self {
            Self::R32Uint => "r32uint",
            Self::R32Sint => "r32sint",
            Self::R32Float => "r32float",
            Self::Rg32Uint => "rg32uint",
            Self::Rg32Sint => "rg32sint",
            Self::Rg32Float => "rg32float",
            Self::Rgba32Uint => "rgba32uint",
            Self::Rgba32Sint => "rgba32sint",
            Self::Rgba32Float => "rgba32float",
        }
    }

    /// The format named `name` (the inverse of [`Format::name`]), or `None` for any other text.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.name() == name)
    }

    /// Slang's spelling of the format, the argument of `[format("..")]`: `r32ui`, `rg32f`, ...
    pub const fn slang_name(self) -> &'static str {
        match self {
            Self::R32Uint => "r32ui",
            Self::R32Sint => "r32i",
            Self::R32Float => "r32f",
            Self::Rg32Uint => "rg32ui",
            Self::Rg32Sint => "rg32i",
            Self::Rg32Float => "rg32f",
            Self::Rgba32Uint => "rgba32ui",
            Self::Rgba32Sint => "rgba32i",
            Self::Rgba32Float => "rgba32f",
        }
    }

    /// The number of channels: 1, 2 or 4.
    pub const fn channels(self) -> u32 {
        match self {
            Self::R32Uint | Self::R32Sint | Self::R32Float => 1,
            Self::Rg32Uint | Self::Rg32Sint | Self::Rg32Float => 2,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 4,
        }
    }

    /// The size of one texel in bytes: four per channel.
    pub const fn texel_bytes(self) -> u32 {
        self.channels() * 4
    }

    /// The class of value a read of the format yields.
    pub const fn sample_class(self) -> SampleClass {
        match self {
            Self::R32Uint | Self::Rg32Uint | Self::Rgba32Uint => SampleClass::Uint,
            Self::R32Sint | Self::Rg32Sint | Self::Rgba32Sint => SampleClass::Sint,
            Self::R32Float | Self::Rg32Float | Self::Rgba32Float => SampleClass::Float,
        }
    }

    /// Whether a storage texture of the format may be read and written (`read_write`): WebGPU's baseline allows it
    /// for the `r32` formats only (decision D120 keeps the wider forms write-only). Every format may be written.
    pub const fn read_write(self) -> bool {
        self.channels() == 1
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The marker type of one [`Format`]: it ties the format to the Rust type of one texel. Sealed: the nine markers of
/// this module ([`R32Uint`] to [`Rgba32Float`]) are the only implementors.
pub trait TextureFormat: sealed::Sealed + 'static {
    /// The Rust type of one texel (`u32`, `Vec2`, `UVec4`, ...). It is the `Element` of the texture as a typed-call
    /// resource, and so a `GpuPod` of a non-zero size.
    type Texel: GpuPod + Default + fmt::Debug;
    /// The format.
    const FORMAT: Format;
    /// Slang's spelling of the format ([`Format::slang_name`]).
    const SLANG: &'static str;
    /// The size of one texel in bytes; equal to `size_of::<Self::Texel>()` (a build-time assertion per marker).
    const BYTES: u32;
    /// Whether a storage texture of this format may be read and written ([`Format::read_write`]).
    const READ_WRITE: bool;
}

/// A format a kernel may read and write through a [`StorageTexture2D`]: `R32Uint`, `R32Sint` and `R32Float`. Sealed.
///
/// A wider format is write-only in the dialect (D120), so `StorageTexture2D<Rg32Float>` is a compile error and
/// `WriteTexture2D<Rg32Float>` is the spelling.
pub trait ReadWriteFormat: TextureFormat {}

macro_rules! texture_formats {
    ($($(#[$meta:meta])* $Marker:ident => $Texel:ty;)+) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        pub enum $Marker {}

        impl sealed::Sealed for $Marker {}

        impl TextureFormat for $Marker {
            type Texel = $Texel;
            const FORMAT: Format = Format::$Marker;
            const SLANG: &'static str = Format::$Marker.slang_name();
            const BYTES: u32 = Format::$Marker.texel_bytes();
            const READ_WRITE: bool = Format::$Marker.read_write();
        }

        // The bytes the format says and the bytes the Rust texel has are one number.
        const _: () = assert!(
            <$Marker as TextureFormat>::BYTES as usize == size_of::<$Texel>()
        );
    )+};
}

texture_formats! {
    /// Marker: [`Format::R32Uint`], texel `u32`.
    R32Uint => u32;
    /// Marker: [`Format::R32Sint`], texel `i32`.
    R32Sint => i32;
    /// Marker: [`Format::R32Float`], texel `f32`.
    R32Float => f32;
    /// Marker: [`Format::Rg32Uint`], texel [`UVec2`].
    Rg32Uint => UVec2;
    /// Marker: [`Format::Rg32Sint`], texel [`IVec2`].
    Rg32Sint => IVec2;
    /// Marker: [`Format::Rg32Float`], texel [`Vec2`].
    Rg32Float => Vec2;
    /// Marker: [`Format::Rgba32Uint`], texel [`UVec4`].
    Rgba32Uint => UVec4;
    /// Marker: [`Format::Rgba32Sint`], texel [`IVec4`].
    Rgba32Sint => IVec4;
    /// Marker: [`Format::Rgba32Float`], texel [`Vec4`].
    Rgba32Float => Vec4;
}

impl ReadWriteFormat for R32Uint {}
impl ReadWriteFormat for R32Sint {}
impl ReadWriteFormat for R32Float {}

/// A texel type that can be filtered: `f32`, [`Vec2`] and [`Vec4`] (the float formats). Sealed. `Texture2D<u32>` and
/// `Texture2D<i32>` have no `sample`: integer texels are never filtered.
pub trait Filterable: sealed::Sealed + Copy {
    /// `a + (b - a) * t` per channel, in `f32` (the interpolation of the twin's linear filter, written in
    /// [`cpu::texture`](crate::cpu::texture)).
    fn lerp(a: Self, b: Self, t: f32) -> Self;
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for Vec2 {}
impl sealed::Sealed for Vec4 {}

fn lerp1(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl Filterable for f32 {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        lerp1(a, b, t)
    }
}

impl Filterable for Vec2 {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec2::new(lerp1(a.x, b.x, t), lerp1(a.y, b.y, t))
    }
}

impl Filterable for Vec4 {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec4::new(
            lerp1(a.x, b.x, t),
            lerp1(a.y, b.y, t),
            lerp1(a.z, b.z, t),
            lerp1(a.w, b.w, t),
        )
    }
}

/// What a texture may be used for: a set of the flags [`SAMPLED`](Self::SAMPLED), [`STORAGE`](Self::STORAGE) and
/// [`TARGET`](Self::TARGET), combined with `|`. A host texture always can be copied from and to; the flags are the
/// uses a backend must enable when it creates the texture. A backend checks the use of a bound texture against the
/// kernel parameter at submit (`TextureUse` errors).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureUse(u8);

impl TextureUse {
    /// No use beyond the copies every texture has.
    pub const NONE: Self = Self(0);
    /// Read by a kernel as a sampled `Texture2D<T>`.
    pub const SAMPLED: Self = Self(1);
    /// Read and written by a kernel as a `StorageTexture2D<F>` or written as a `WriteTexture2D<F>`.
    pub const STORAGE: Self = Self(2);
    /// A render-target attachment (the pixel shape of Breeze; not used by a kernel in MX).
    pub const TARGET: Self = Self(4);

    /// Whether every flag of `other` is set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The flags as bits: `SAMPLED` is 1, `STORAGE` 2 and `TARGET` 4.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The union of two sets (the `|` operator as a `const fn`).
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for TextureUse {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl fmt::Debug for TextureUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("TextureUse(NONE)");
        }
        let mut separator = "TextureUse(";
        for (flag, name) in [
            (Self::SAMPLED, "SAMPLED"),
            (Self::STORAGE, "STORAGE"),
            (Self::TARGET, "TARGET"),
        ] {
            if self.contains(flag) {
                write!(f, "{separator}{name}")?;
                separator = " | ";
            }
        }
        f.write_str(")")
    }
}

/// How a sampler reads between texels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Filter {
    /// The texel the coordinate falls in. Exact on the twin; a GPU may pick the neighbour when the coordinate is
    /// within one ulp of a texel edge.
    #[default]
    Nearest,
    /// Bilinear interpolation of the four nearest texels. A 32-bit float texture is linearly filterable only on a
    /// device that enabled `float32-filterable` (decision D64); the twin does not depend on the device.
    Linear,
}

/// What a sampler does with a coordinate outside `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Address {
    /// The coordinate is clamped to `[0, 1]`: the edge texel repeats.
    #[default]
    ClampToEdge,
    /// The coordinate wraps: `u - floor(u)`.
    Repeat,
}

/// The state of a sampler: its [`Filter`] and its [`Address`] mode (the same mode on both axes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SamplerOptions {
    /// How the sampler reads between texels.
    pub filter: Filter,
    /// What it does outside `[0, 1]`.
    pub address: Address,
}

impl SamplerOptions {
    /// Options with the given filter and address mode.
    pub const fn new(filter: Filter, address: Address) -> Self {
        Self { filter, address }
    }
}

/// Rust shadow of Slang's `SamplerState`: the `Sampler` parameter of a kernel, the argument of
/// [`Texture2D::sample`]. It is `Copy`, carries its [`SamplerOptions`] and does nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sampler {
    options: SamplerOptions,
}

impl Sampler {
    /// A sampler with the given options.
    pub const fn new(options: SamplerOptions) -> Self {
        Self { options }
    }

    /// The sampler's options.
    pub const fn options(&self) -> SamplerOptions {
        self.options
    }
}

/// `width * height` as a texel count, or a panic when `texels` is not exactly that long.
fn check_view(len: usize, width: u32, height: u32) {
    assert!(
        u64::from(width) * u64::from(height) == len as u64,
        "a texture view of {width}x{height} texels needs {} texels, got {len}",
        u64::from(width) * u64::from(height)
    );
}

/// Rust shadow of Slang's `Texture2D<T>`: a sampled, read-only 2-D texture. `T` is `f32`, `u32`, `i32`, [`Vec2`] or
/// [`Vec4`] (and their integer forms), the value a read yields.
///
/// Row-major: texel `(x, y)` is `texels[y * width + x]`. [`load`](Self::load) reads one texel at integer
/// coordinates (level 0) and panics outside the texture; [`sample`](Self::sample) is the filtered read of a float
/// texture.
#[derive(Clone, Copy, Debug)]
pub struct Texture2D<'a, T> {
    texels: &'a [T],
    width: u32,
    height: u32,
}

impl<'a, T> Texture2D<'a, T> {
    /// A view over `texels`, row-major, `width * height` of them (a panic otherwise).
    pub fn new(texels: &'a [T], width: u32, height: u32) -> Self {
        check_view(texels.len(), width, height);
        Self {
            texels,
            width,
            height,
        }
    }

    /// The width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl<T: Copy> Texture2D<'_, T> {
    /// The texel at `(x, y)`, exact. Panics ("index out of bounds") when `x >= width` or `y >= height`, as an
    /// out-of-bounds buffer index does (KM-11); the GPU's outcome there is unspecified.
    pub fn load(&self, x: u32, y: u32) -> T {
        self.texels[texel_index(x, y, self.width, self.height)]
    }
}

impl<T: Filterable> Texture2D<'_, T> {
    /// The filtered read at normalised coordinates `(u, v)` of level 0 through `sampler`: the formula of the WGSL
    /// specification as written in [`cpu::texture`](crate::cpu::texture). It is outside the exact contract (D64): a GPU
    /// may differ in the weights' precision and in the clamping of the coordinate, and the difference is documented,
    /// not chased (D39).
    pub fn sample(&self, sampler: Sampler, u: f32, v: f32) -> T {
        sample_2d(self.texels, self.width, self.height, sampler.options, u, v)
    }
}

/// Rust shadow of a Slang read-write storage texture (`[format("..")] RWTexture2D<T>`): the
/// `StorageTexture2D<F>` parameter of a kernel, with `F` one of [`R32Uint`], [`R32Sint`] and [`R32Float`]
/// ([`ReadWriteFormat`]).
pub struct StorageTexture2D<'a, F: ReadWriteFormat> {
    texels: &'a mut [F::Texel],
    width: u32,
    height: u32,
    _format: PhantomData<fn() -> F>,
}

impl<'a, F: ReadWriteFormat> StorageTexture2D<'a, F> {
    /// A view over `texels`, row-major, `width * height` of them (a panic otherwise).
    pub fn new(texels: &'a mut [F::Texel], width: u32, height: u32) -> Self {
        check_view(texels.len(), width, height);
        Self {
            texels,
            width,
            height,
            _format: PhantomData,
        }
    }

    /// The width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The texel at `(x, y)`; panics ("index out of bounds") outside the texture.
    pub fn load(&self, x: u32, y: u32) -> F::Texel {
        self.texels[texel_index(x, y, self.width, self.height)]
    }

    /// Replace the texel at `(x, y)`; panics ("index out of bounds") outside the texture.
    pub fn store(&mut self, x: u32, y: u32, value: F::Texel) {
        self.texels[texel_index(x, y, self.width, self.height)] = value;
    }
}

impl<F: ReadWriteFormat> fmt::Debug for StorageTexture2D<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageTexture2D")
            .field("format", &F::FORMAT)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// Rust shadow of a Slang write-only storage texture (`[format("..")] WTexture2D<T>` with `.Store`): the
/// `WriteTexture2D<F>` parameter of a kernel, with `F` any format of the dialect. It has no `load`: a wider format
/// than the `r32` ones cannot be read and written (D120).
pub struct WriteTexture2D<'a, F: TextureFormat> {
    texels: &'a mut [F::Texel],
    width: u32,
    height: u32,
    _format: PhantomData<fn() -> F>,
}

impl<'a, F: TextureFormat> WriteTexture2D<'a, F> {
    /// A view over `texels`, row-major, `width * height` of them (a panic otherwise).
    pub fn new(texels: &'a mut [F::Texel], width: u32, height: u32) -> Self {
        check_view(texels.len(), width, height);
        Self {
            texels,
            width,
            height,
            _format: PhantomData,
        }
    }

    /// The width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Replace the texel at `(x, y)`; panics ("index out of bounds") outside the texture.
    pub fn store(&mut self, x: u32, y: u32, value: F::Texel) {
        self.texels[texel_index(x, y, self.width, self.height)] = value;
    }
}

impl<F: TextureFormat> fmt::Debug for WriteTexture2D<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteTexture2D")
            .field("format", &F::FORMAT)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_has_the_name_bytes_and_class_the_dialect_gives_it() {
        // (format, WebGPU name, Slang name, channels, class, read_write), written out by hand.
        let table = [
            (
                Format::R32Uint,
                "r32uint",
                "r32ui",
                1,
                SampleClass::Uint,
                true,
            ),
            (
                Format::R32Sint,
                "r32sint",
                "r32i",
                1,
                SampleClass::Sint,
                true,
            ),
            (
                Format::R32Float,
                "r32float",
                "r32f",
                1,
                SampleClass::Float,
                true,
            ),
            (
                Format::Rg32Uint,
                "rg32uint",
                "rg32ui",
                2,
                SampleClass::Uint,
                false,
            ),
            (
                Format::Rg32Sint,
                "rg32sint",
                "rg32i",
                2,
                SampleClass::Sint,
                false,
            ),
            (
                Format::Rg32Float,
                "rg32float",
                "rg32f",
                2,
                SampleClass::Float,
                false,
            ),
            (
                Format::Rgba32Uint,
                "rgba32uint",
                "rgba32ui",
                4,
                SampleClass::Uint,
                false,
            ),
            (
                Format::Rgba32Sint,
                "rgba32sint",
                "rgba32i",
                4,
                SampleClass::Sint,
                false,
            ),
            (
                Format::Rgba32Float,
                "rgba32float",
                "rgba32f",
                4,
                SampleClass::Float,
                false,
            ),
        ];
        assert_eq!(table.len(), Format::ALL.len());
        for (index, (format, name, slang, channels, class, read_write)) in
            table.into_iter().enumerate()
        {
            assert_eq!(Format::ALL[index], format);
            assert_eq!(format.name(), name);
            assert_eq!(format.to_string(), name);
            assert_eq!(Format::from_name(name), Some(format));
            assert_eq!(format.slang_name(), slang);
            assert_eq!(format.channels(), channels);
            assert_eq!(format.texel_bytes(), channels * 4);
            assert_eq!(format.sample_class(), class);
            assert_eq!(format.read_write(), read_write);
        }
        for other in ["", "R32Uint", "rgba8unorm", "r32uint "] {
            assert_eq!(Format::from_name(other), None, "{other:?}");
        }
        assert_eq!(SampleClass::Float.name(), "f32");
        assert_eq!(SampleClass::Uint.name(), "u32");
        assert_eq!(SampleClass::Sint.name(), "i32");
    }

    /// A marker agrees with its `Format` on every constant, and the texel type has the size the format says.
    fn marker<F: TextureFormat>(format: Format) {
        assert_eq!(F::FORMAT, format);
        assert_eq!(F::SLANG, format.slang_name());
        assert_eq!(F::BYTES, format.texel_bytes());
        assert_eq!(F::READ_WRITE, format.read_write());
        assert_eq!(size_of::<F::Texel>() as u32, F::BYTES);
        // The texel layout is the one of the storage ABI: the same type a buffer of these texels would have.
        assert_eq!(F::Texel::LAYOUT.size, F::BYTES);
    }

    #[test]
    fn the_nine_markers_agree_with_the_format_list() {
        marker::<R32Uint>(Format::R32Uint);
        marker::<R32Sint>(Format::R32Sint);
        marker::<R32Float>(Format::R32Float);
        marker::<Rg32Uint>(Format::Rg32Uint);
        marker::<Rg32Sint>(Format::Rg32Sint);
        marker::<Rg32Float>(Format::Rg32Float);
        marker::<Rgba32Uint>(Format::Rgba32Uint);
        marker::<Rgba32Sint>(Format::Rgba32Sint);
        marker::<Rgba32Float>(Format::Rgba32Float);
    }

    #[test]
    fn a_texture_use_is_a_set_of_flags() {
        let both = TextureUse::SAMPLED | TextureUse::STORAGE;
        assert!(both.contains(TextureUse::SAMPLED));
        assert!(both.contains(TextureUse::STORAGE));
        assert!(!both.contains(TextureUse::TARGET));
        assert!(both.contains(TextureUse::NONE));
        assert!(!TextureUse::SAMPLED.contains(both));
        assert_eq!(both.bits(), 3);
        assert!(TextureUse::NONE.is_empty());
        assert!(!both.is_empty());
        assert_eq!(TextureUse::default(), TextureUse::NONE);
        assert_eq!(format!("{both:?}"), "TextureUse(SAMPLED | STORAGE)");
        assert_eq!(format!("{:?}", TextureUse::NONE), "TextureUse(NONE)");
        assert_eq!(format!("{:?}", TextureUse::TARGET), "TextureUse(TARGET)");
    }

    #[test]
    fn sampler_options_default_to_nearest_and_clamp() {
        let defaults = SamplerOptions::default();
        assert_eq!(defaults.filter, Filter::Nearest);
        assert_eq!(defaults.address, Address::ClampToEdge);
        assert_eq!(
            SamplerOptions::new(Filter::Linear, Address::Repeat),
            SamplerOptions {
                filter: Filter::Linear,
                address: Address::Repeat
            }
        );
        let sampler = Sampler::new(SamplerOptions::new(Filter::Linear, Address::Repeat));
        assert_eq!(sampler.options().filter, Filter::Linear);
        assert_eq!(Sampler::default().options(), defaults);
    }

    #[test]
    fn a_sampled_view_loads_row_major_texels_and_reports_its_size() {
        // 3 wide, 2 high: texel (x, y) is 10 * y + x.
        let texels = [0u32, 1, 2, 10, 11, 12];
        let view = Texture2D::new(&texels, 3, 2);
        assert_eq!((view.width(), view.height()), (3, 2));
        assert_eq!(view.load(0, 0), 0);
        assert_eq!(view.load(2, 0), 2);
        assert_eq!(view.load(0, 1), 10);
        assert_eq!(view.load(2, 1), 12);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn a_load_past_the_width_panics_even_when_the_flat_index_exists() {
        // (3, 0) is flat index 3, which exists in a 3x2 texture; the row is what is out of range.
        let texels = [0u32, 1, 2, 10, 11, 12];
        let _ = Texture2D::new(&texels, 3, 2).load(3, 0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn a_load_past_the_height_panics() {
        let texels = [0u32, 1, 2, 10, 11, 12];
        let _ = Texture2D::new(&texels, 3, 2).load(0, 2);
    }

    #[test]
    #[should_panic(expected = "needs 6 texels, got 5")]
    fn a_view_of_the_wrong_length_is_refused() {
        let texels = [0u32; 5];
        let _ = Texture2D::new(&texels, 3, 2);
    }

    #[test]
    fn a_storage_view_stores_and_loads_and_a_write_view_only_stores() {
        let mut texels = [0.0f32; 4];
        {
            let mut storage = StorageTexture2D::<R32Float>::new(&mut texels, 2, 2);
            storage.store(1, 0, 2.5);
            storage.store(0, 1, -1.0);
            assert_eq!(storage.load(1, 0), 2.5);
            assert_eq!(storage.load(0, 0), 0.0);
            assert_eq!((storage.width(), storage.height()), (2, 2));
            assert!(format!("{storage:?}").contains("R32Float"));
        }
        assert_eq!(texels, [0.0, 2.5, -1.0, 0.0]);

        let mut wide = [Vec4::default(); 2];
        {
            let mut write = WriteTexture2D::<Rgba32Float>::new(&mut wide, 2, 1);
            write.store(1, 0, Vec4::new(1.0, 2.0, 3.0, 4.0));
            assert_eq!((write.width(), write.height()), (2, 1));
            assert!(format!("{write:?}").contains("Rgba32Float"));
        }
        assert_eq!(wide[0].w, 0.0);
        assert_eq!(
            (wide[1].x, wide[1].y, wide[1].z, wide[1].w),
            (1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn a_store_outside_the_texture_panics() {
        let mut texels = [0u32; 4];
        StorageTexture2D::<R32Uint>::new(&mut texels, 2, 2).store(2, 1, 7);
    }

    #[test]
    fn filterable_texels_interpolate_per_channel() {
        assert_eq!(<f32 as Filterable>::lerp(2.0, 6.0, 0.25), 3.0);
        let two = <Vec2 as Filterable>::lerp(Vec2::new(0.0, 10.0), Vec2::new(4.0, 20.0), 0.5);
        assert_eq!((two.x, two.y), (2.0, 15.0));
        let four = <Vec4 as Filterable>::lerp(
            Vec4::new(0.0, 1.0, 2.0, 3.0),
            Vec4::new(8.0, 5.0, 2.0, -1.0),
            0.5,
        );
        assert_eq!((four.x, four.y, four.z, four.w), (4.0, 3.0, 2.0, 1.0));
    }
}
