//! The CPU twin's texture carrier and the conventions of `load`, `store` and `sample` (decision D64, slice mx.t1).
//!
//! [`CpuTexture2d<F>`] is a typed 2-D array of texels (a `Vec`, no image crate: none establishes GPU sampling
//! semantics) that a [`CpuDevice`] hands out the way it hands out buffers, and [`CpuSampler`] is a sampler of the same
//! device. A texture is a [`Resource`] of kind [`kind::Texture`]; a sampler is a
//! [`SamplerArg`], not a resource. The kernel-side views ([`Texture2D`](crate::texture::Texture2D) and the others)
//! use the functions of this module, so a shadow body and a host call mean the same thing.
//!
//! # The conventions
//!
//! - **Texel addresses.** `(0, 0)` is the top-left texel, `x` runs right, `y` runs down, and texel `(x, y)` is at
//!   `texels[y * width + x]`. `load` and `store` are exact (inside D22's contract). An address outside the texture is
//!   a panic in a kernel body, "index out of bounds" like KM-11, and [`Error::TextureCoordinate`] from the carrier's
//!   own methods; the GPU's outcome there is unspecified (WGSL leaves a dynamic error open, section 5 of
//!   `docs/SEMANTICS.md`).
//! - **Normalised coordinates** of `sample`. `(0, 0)` is the top-left *corner* of texel `(0, 0)` and `(1, 1)` the
//!   bottom-right corner of the last texel (WebGPU's convention), so the centre of texel `(i, j)` is
//!   `((i + 0.5) / width, (j + 0.5) / height)`. The unnormalised position is `x = u * width`, `y = v * height`.
//! - **Addressing** is applied first. `ClampToEdge`: the coordinate is clamped to `[0, 1]` (NaN becomes 0). `Repeat`:
//!   `u - floor(u)`. For a linear read the mode is applied to the neighbouring *texel indices* as well: clamped to
//!   `[0, n - 1]`, or wrapped modulo `n`.
//! - **Nearest** reads texel `(min(floor(x), width - 1), min(floor(y), height - 1))`. Exact on the twin. A GPU may
//!   pick the neighbouring texel when `x` is within one ulp of an integer; that is documented, not chased (D39).
//! - **Linear** (bilinear): `x' = x - 0.5`, `i0 = floor(x')`, `fx = x' - i0` (the same for `y`), the four texels
//!   `(i0, j0)`, `(i0 + 1, j0)`, `(i0, j0 + 1)`, `(i0 + 1, j0 + 1)` after the address mode, and
//!   `lerp(lerp(t00, t10, fx), lerp(t01, t11, fx), fy)` in `f32`, in that written order, with
//!   `lerp(a, b, t) = a + (b - a) * t` per channel.
//! - **The error domain** of a GPU `sample` against this formula, per channel: `|gpu - twin| <= 2^-8 * (max - min of
//!   the four texels) + 2 * ulp(max |texel|)` ([`linear_error_bound`]; an assumed 8-bit precision of the weights,
//!   rounded to the nearest step, which no specification gives: SEMANTICS, section 4). A nearest read is compared
//!   exactly. NaN and infinite texels, and coordinates with `|u| > 2^24`, are outside the domain: the twin does not
//!   panic there, and what it returns is not specified.
//!
//! `sample` is outside D22's exact contract (a GPU may differ in the weights' precision and in the clamping of the
//! coordinate); the difference is documented in `docs/SEMANTICS.md` and never chased.
#![warn(missing_docs)]

use core::any::Any;
use core::cell::{Ref, RefCell, RefMut};
use core::fmt;
use core::marker::PhantomData;

use super::{CpuBinding, CpuData, CpuDevice, Error, require_kind};
use crate::texture::{
    Address, Filter, Filterable, Sampler, SamplerOptions, TextureFormat, TextureUse,
};
use crate::{Access, BindingKind, GpuPod, LengthRule, Resource, SamplerArg, SamplerDevice, kind};

// ---- the conventions, as functions the views and the carrier share ----

/// The flat index of texel `(x, y)` in a `width` by `height` texture, row-major. Panics ("index out of bounds") when
/// the address is outside the texture, even if the flat index would exist (`x >= width` with a row below).
pub(crate) fn texel_index(x: u32, y: u32, width: u32, height: u32) -> usize {
    assert!(
        x < width && y < height,
        "index out of bounds: the texel is ({x}, {y}) but the texture is {width}x{height}"
    );
    y as usize * width as usize + x as usize
}

/// The coordinate after the address mode: `[0, 1]` for a clamp (NaN becomes 0), `u - floor(u)` for a repeat.
fn address_coordinate(coordinate: f32, mode: Address) -> f32 {
    match mode {
        // `max` and `min` return the other operand for a NaN, so a NaN lands on 0 and never reaches an index;
        // `f32::clamp` would return the NaN.
        #[allow(clippy::manual_clamp)]
        Address::ClampToEdge => coordinate.max(0.0).min(1.0),
        Address::Repeat => coordinate - coordinate.floor(),
    }
}

/// The texel index along one axis of size `n` after the address mode: clamped to `[0, n - 1]` or wrapped modulo `n`.
fn address_index(index: i64, n: u32, mode: Address) -> usize {
    let n = i64::from(n);
    let wrapped = match mode {
        Address::ClampToEdge => index.clamp(0, n - 1),
        Address::Repeat => index.rem_euclid(n),
    };
    wrapped as usize
}

/// `min(floor(position), n - 1)` for the nearest filter; `position` is not negative after the address mode, and a NaN
/// (outside the domain) is texel 0.
fn nearest_index(position: f32, n: u32) -> usize {
    (position.floor() as i64).clamp(0, i64::from(n) - 1) as usize
}

/// The filtered read of a `width` by `height` row-major texture at normalised `(u, v)`: the conventions of the module
/// documentation. Panics on an empty texture.
pub(crate) fn sample_2d<T: Filterable>(
    texels: &[T],
    width: u32,
    height: u32,
    options: SamplerOptions,
    u: f32,
    v: f32,
) -> T {
    assert!(
        width > 0 && height > 0,
        "index out of bounds: cannot sample a texture of {width}x{height} texels"
    );
    let x = address_coordinate(u, options.address) * width as f32;
    let y = address_coordinate(v, options.address) * height as f32;
    let at = |i: usize, j: usize| texels[j * width as usize + i];
    match options.filter {
        Filter::Nearest => at(nearest_index(x, width), nearest_index(y, height)),
        Filter::Linear => {
            let (x, y) = (x - 0.5, y - 0.5);
            let (floor_x, floor_y) = (x.floor(), y.floor());
            let (fx, fy) = (x - floor_x, y - floor_y);
            let (i0, j0) = (floor_x as i64, floor_y as i64);
            let (i0, i1) = (
                address_index(i0, width, options.address),
                address_index(i0.saturating_add(1), width, options.address),
            );
            let (j0, j1) = (
                address_index(j0, height, options.address),
                address_index(j0.saturating_add(1), height, options.address),
            );
            let top = T::lerp(at(i0, j0), at(i1, j0), fx);
            let bottom = T::lerp(at(i0, j1), at(i1, j1), fx);
            T::lerp(top, bottom, fy)
        }
    }
}

/// The spacing of `f32` values at the magnitude of `x`: the gap from `|x|` to the next larger magnitude. For a
/// non-finite `x` it is infinite or NaN (outside the domain).
pub fn ulp(x: f32) -> f32 {
    let magnitude = x.abs();
    f32::from_bits(magnitude.to_bits().wrapping_add(1)) - magnitude
}

/// The error domain of a linear `sample`, for one channel: how far a GPU's result may be from the twin's
/// ([`SamplerOptions`] with [`Filter::Linear`]), given the channel of the four texels the read touches:
/// `2^-8 * (max - min) + 2 * ulp(max |texel|)`.
///
/// The first term is an assumed 8-bit precision of the weights, rounded to the nearest 1/256 step; no specification
/// gives it (the WebGPU and WGSL specifications state no number of bits, and Vulkan requires `subTexelPrecisionBits` of
/// at least 4 in core and at least 8 of Vulkan 1.4 and Roadmap 2022), so a sampler that quantises worse can leave the
/// bound (SEMANTICS, section 4; D52).
/// The second is the rounding of the interpolation itself. A nearest read has no tolerance: compare it exactly.
pub fn linear_error_bound(corners: [f32; 4]) -> f32 {
    let mut lowest = corners[0];
    let mut highest = corners[0];
    let mut largest = 0.0f32;
    for corner in corners {
        lowest = lowest.min(corner);
        highest = highest.max(corner);
        largest = largest.max(corner.abs());
    }
    (highest - lowest) * (1.0 / 256.0) + 2.0 * ulp(largest)
}

// ---- the carrier ----

/// The cell of a texture: its size, its uses and its texels, keyed by the texel type (a format is one texel type per
/// class, so the kernel's `Texture2D<f32>` finds the cell of an `R32Float` texture).
struct TextureCell<T: GpuPod> {
    width: u32,
    height: u32,
    uses: TextureUse,
    texels: RefCell<Vec<T>>,
}

impl<T: GpuPod> CpuData for TextureCell<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn current_len(&self) -> Option<usize> {
        self.texels.try_borrow().ok().map(|texels| texels.len())
    }

    fn texture_uses(&self) -> Option<TextureUse> {
        Some(self.uses)
    }
}

/// A 2-D texture of one [`CpuDevice`], holding texels of format `F` row-major.
///
/// The counterpart of `gust_wgpu::GpuTexture2d`: it is a [`Resource`] of kind [`kind::Texture`], so a call takes it
/// where a kernel has a `Texture2D`, `StorageTexture2D` or `WriteTexture2D` parameter and rejects it where the kernel
/// has a buffer. It has no element count and is bound with an independent length. It belongs to the device that
/// created it.
pub struct CpuTexture2d<F: TextureFormat> {
    owner: u32,
    cell: TextureCell<F::Texel>,
    _format: PhantomData<fn() -> F>,
}

impl<F: TextureFormat> CpuTexture2d<F> {
    /// The width in texels.
    pub fn width(&self) -> u32 {
        self.cell.width
    }

    /// The height in texels.
    pub fn height(&self) -> u32 {
        self.cell.height
    }

    /// The uses the texture was created for.
    pub fn uses(&self) -> TextureUse {
        self.cell.uses
    }

    /// A copy of the texels, row-major.
    ///
    /// # Panics
    ///
    /// If the texture is mutably borrowed, which only a running kernel call does; use it between calls.
    pub fn read(&self) -> Vec<F::Texel> {
        self.cell.texels.borrow().clone()
    }

    /// Replace the texels, row-major; `texels` must hold exactly `width * height` of them
    /// ([`Error::TextureLength`] otherwise, and the texture is untouched), and [`Error::Borrowed`] (naming `texels`)
    /// is returned when it is borrowed elsewhere.
    pub fn write(&self, texels: &[F::Texel]) -> Result<(), Error> {
        let mut cells = self
            .cell
            .texels
            .try_borrow_mut()
            .map_err(|_| Error::Borrowed {
                parameter: "texels",
            })?;
        if cells.len() != texels.len() {
            return Err(Error::TextureLength {
                expected: cells.len(),
                actual: texels.len(),
            });
        }
        cells.copy_from_slice(texels);
        Ok(())
    }

    /// The texel at `(x, y)`, exact; [`Error::TextureCoordinate`] outside the texture.
    pub fn load(&self, x: u32, y: u32) -> Result<F::Texel, Error> {
        let index = self.index(x, y)?;
        let texels = self.cell.texels.try_borrow().map_err(|_| Error::Borrowed {
            parameter: "texels",
        })?;
        Ok(texels[index])
    }

    /// Replace the texel at `(x, y)`; [`Error::TextureCoordinate`] outside the texture.
    pub fn store(&self, x: u32, y: u32, value: F::Texel) -> Result<(), Error> {
        let index = self.index(x, y)?;
        let mut texels = self
            .cell
            .texels
            .try_borrow_mut()
            .map_err(|_| Error::Borrowed {
                parameter: "texels",
            })?;
        texels[index] = value;
        Ok(())
    }

    fn index(&self, x: u32, y: u32) -> Result<usize, Error> {
        if x < self.cell.width && y < self.cell.height {
            Ok(y as usize * self.cell.width as usize + x as usize)
        } else {
            Err(Error::TextureCoordinate {
                x,
                y,
                width: self.cell.width,
                height: self.cell.height,
            })
        }
    }
}

impl<F: TextureFormat> fmt::Debug for CpuTexture2d<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuTexture2d")
            .field("format", &F::FORMAT)
            .field("owner", &self.owner)
            .field("width", &self.cell.width)
            .field("height", &self.cell.height)
            .field("uses", &self.cell.uses)
            .finish_non_exhaustive()
    }
}

impl<F: TextureFormat> Resource for CpuTexture2d<F> {
    type Element = F::Texel;
    type Device = CpuDevice;
    type Kind = kind::Texture;

    // Never borrows the cell. A texture has no element count, so the length rule is `Independent` whatever the
    // argument says.
    fn bind<'r>(&'r self, access: Access, _length: LengthRule) -> CpuBinding<'r> {
        CpuBinding {
            data: &self.cell,
            access,
            length: LengthRule::Independent,
            layout: F::Texel::LAYOUT,
            owner: self.owner,
            kind: BindingKind::Texture,
        }
    }
}

impl CpuDevice {
    /// A texture of this device, `width` by `height` texels of format `F`, every texel `F::Texel::default()` (zero),
    /// created for `uses`. [`Error::TextureSize`] when a side is zero or the texel count exceeds `u32::MAX`.
    pub fn texture_2d<F: TextureFormat>(
        &self,
        width: u32,
        height: u32,
        uses: TextureUse,
    ) -> Result<CpuTexture2d<F>, Error> {
        let count = u64::from(width) * u64::from(height);
        if width == 0 || height == 0 || count > u64::from(u32::MAX) {
            return Err(Error::TextureSize { width, height });
        }
        Ok(CpuTexture2d {
            owner: self.id,
            cell: TextureCell {
                width,
                height,
                uses,
                texels: RefCell::new(vec![F::Texel::default(); count as usize]),
            },
            _format: PhantomData,
        })
    }

    /// [`CpuTexture2d::write`], spelled as the device call `gust-compute-wgpu` has. [`Error::WrongDevice`] (naming
    /// `texture`) when the texture belongs to another device instance, as `gust-compute-wgpu` refuses a texture of
    /// another device, and [`Error::Borrowed`] (naming `texture`, as [`CpuDevice::read_texture`] does) when it is
    /// borrowed elsewhere; the other errors are those of [`CpuTexture2d::write`].
    pub fn write_texture<F: TextureFormat>(
        &self,
        texture: &CpuTexture2d<F>,
        texels: &[F::Texel],
    ) -> Result<(), Error> {
        self.require_own_texture(texture)?;
        texture.write(texels).map_err(|error| match error {
            // `CpuTexture2d::write` names its own argument; the device call names the resource, like the read.
            Error::Borrowed { .. } => Error::Borrowed {
                parameter: "texture",
            },
            other => other,
        })
    }

    /// [`CpuTexture2d::read`], spelled as the device call `gust-compute-wgpu` has, with the same `Result` shape.
    /// [`Error::WrongDevice`] (naming `texture`) when the texture belongs to another device instance, and
    /// [`Error::Borrowed`] (naming `texture`) when it is mutably borrowed, which only a running kernel call does;
    /// [`CpuTexture2d::read`] panics there instead.
    pub fn read_texture<F: TextureFormat>(
        &self,
        texture: &CpuTexture2d<F>,
    ) -> Result<Vec<F::Texel>, Error> {
        self.require_own_texture(texture)?;
        texture
            .cell
            .texels
            .try_borrow()
            .map(|texels| texels.clone())
            .map_err(|_| Error::Borrowed {
                parameter: "texture",
            })
    }

    /// [`Error::WrongDevice`] (naming `texture`) unless `texture` belongs to this device instance.
    fn require_own_texture<F: TextureFormat>(
        &self,
        texture: &CpuTexture2d<F>,
    ) -> Result<(), Error> {
        if texture.owner != self.id {
            return Err(Error::WrongDevice {
                parameter: "texture",
            });
        }
        Ok(())
    }

    /// A sampler of this device with the given options.
    pub fn sampler(&self, options: SamplerOptions) -> CpuSampler {
        CpuSampler {
            owner: self.id,
            options,
        }
    }
}

// ---- the sampler ----

/// A sampler of one [`CpuDevice`]: the [`SamplerArg`] a call passes for a kernel's `Sampler` parameter.
#[derive(Clone, Copy, Debug)]
pub struct CpuSampler {
    owner: u32,
    options: SamplerOptions,
}

impl CpuSampler {
    /// The sampler's options.
    pub fn options(&self) -> SamplerOptions {
        self.options
    }

    /// The filtered read of `texture` at normalised `(u, v)`: the conventions of the module documentation, the same
    /// as `Texture2D::sample` in a kernel. [`Error::WrongDevice`] (naming `texture`) when the texture belongs to
    /// another device instance, and [`Error::Borrowed`] when it is mutably borrowed.
    pub fn sample<F>(&self, texture: &CpuTexture2d<F>, u: f32, v: f32) -> Result<F::Texel, Error>
    where
        F: TextureFormat,
        F::Texel: Filterable,
    {
        if texture.owner != self.owner {
            return Err(Error::WrongDevice {
                parameter: "texture",
            });
        }
        let texels = texture
            .cell
            .texels
            .try_borrow()
            .map_err(|_| Error::Borrowed {
                parameter: "texture",
            })?;
        Ok(sample_2d(
            &texels,
            texture.cell.width,
            texture.cell.height,
            self.options,
            u,
            v,
        ))
    }
}

/// Erased view of one bound [`CpuSampler`].
#[derive(Clone, Copy, Debug)]
pub struct CpuSamplerBinding<'r> {
    options: SamplerOptions,
    owner: u32,
    _borrow: PhantomData<&'r CpuSampler>,
}

impl SamplerDevice for CpuDevice {
    type SamplerBinding<'r> = CpuSamplerBinding<'r>;
}

impl SamplerArg for CpuSampler {
    type Device = CpuDevice;

    fn bind(&self) -> CpuSamplerBinding<'_> {
        CpuSamplerBinding {
            options: self.options,
            owner: self.owner,
            _borrow: PhantomData,
        }
    }
}

// ---- macro support: what the emitted twin of a kernel with textures calls ----

/// A texture borrowed for reading: its size and its texels, row-major. Macro support.
#[doc(hidden)]
pub struct TextureRef<'b, T> {
    /// The width in texels.
    pub width: u32,
    /// The height in texels.
    pub height: u32,
    /// The texels, row-major.
    pub texels: Ref<'b, Vec<T>>,
}

/// A texture borrowed for writing: its size and its texels, row-major. Macro support.
#[doc(hidden)]
pub struct TextureMut<'b, T> {
    /// The width in texels.
    pub width: u32,
    /// The height in texels.
    pub height: u32,
    /// The texels, row-major.
    pub texels: RefMut<'b, Vec<T>>,
}

fn cell_of<'b, T: GpuPod>(
    binding: &CpuBinding<'b>,
    parameter: &'static str,
) -> Result<&'b TextureCell<T>, Error> {
    require_kind(binding, BindingKind::Texture, parameter)?;
    binding
        .data
        .as_any()
        .downcast_ref::<TextureCell<T>>()
        .ok_or(Error::ElementType { parameter })
}

/// Borrow the bound texture for reading. The texel type `T` is decided by `TypeId` ([`Error::ElementType`]), the
/// kind must be [`BindingKind::Texture`] ([`Error::BindingKind`], checked first), the access read-only
/// ([`Error::Access`]) and the texture not mutably borrowed ([`Error::Borrowed`]). Macro support.
#[doc(hidden)]
pub fn read<'b, T: GpuPod>(
    binding: CpuBinding<'b>,
    parameter: &'static str,
) -> Result<TextureRef<'b, T>, Error> {
    let cell = cell_of::<T>(&binding, parameter)?;
    if binding.access != Access::ReadOnly {
        return Err(Error::Access {
            parameter,
            expected: Access::ReadOnly,
            actual: binding.access,
        });
    }
    Ok(TextureRef {
        width: cell.width,
        height: cell.height,
        texels: cell
            .texels
            .try_borrow()
            .map_err(|_| Error::Borrowed { parameter })?,
    })
}

/// Borrow the bound texture for writing: the rules of [`read`], with a read-write binding required. Macro support.
#[doc(hidden)]
pub fn write<'b, T: GpuPod>(
    binding: CpuBinding<'b>,
    parameter: &'static str,
) -> Result<TextureMut<'b, T>, Error> {
    let cell = cell_of::<T>(&binding, parameter)?;
    if binding.access != Access::ReadWrite {
        return Err(Error::Access {
            parameter,
            expected: Access::ReadWrite,
            actual: binding.access,
        });
    }
    Ok(TextureMut {
        width: cell.width,
        height: cell.height,
        texels: cell
            .texels
            .try_borrow_mut()
            .map_err(|_| Error::Borrowed { parameter })?,
    })
}

/// The kernel-side [`Sampler`] of a bound sampler. Macro support.
#[doc(hidden)]
pub fn sampler(binding: CpuSamplerBinding<'_>) -> Sampler {
    Sampler::new(binding.options)
}

/// The device instance a bound sampler belongs to, as [`validate`](super::validate) reads it from the call's
/// [`SamplerSet`](crate::SamplerSet). Macro support.
#[doc(hidden)]
pub trait SamplerOwner {
    /// The id of the owning [`CpuDevice`].
    fn owner(&self) -> u32;
}

impl SamplerOwner for CpuSamplerBinding<'_> {
    fn owner(&self) -> u32 {
        self.owner
    }
}

// The binding type of a call without samplers: there is never a value of it.
impl SamplerOwner for core::convert::Infallible {
    fn owner(&self) -> u32 {
        match *self {}
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::cpu::{CpuJob, invocations};
    use crate::math::{Vec2, Vec4};
    use crate::texture::{R32Float, R32Sint, R32Uint, Rg32Float, Rg32Uint, SampleClass, Texture2D};
    use crate::{
        Arg, Call, Device, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind,
        ResourceBinding, ResourceLayout, Run, SampledCall, StorageMut, TextureAccess,
        TextureDimension, TextureLayout, UVec3,
    };

    const NEAREST: SamplerOptions = SamplerOptions::new(Filter::Nearest, Address::ClampToEdge);
    const NEAREST_REPEAT: SamplerOptions = SamplerOptions::new(Filter::Nearest, Address::Repeat);
    const LINEAR: SamplerOptions = SamplerOptions::new(Filter::Linear, Address::ClampToEdge);
    const LINEAR_REPEAT: SamplerOptions = SamplerOptions::new(Filter::Linear, Address::Repeat);

    /// A 4x1 strip, texel `i` holding `10 * (i + 1)`.
    const STRIP: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    /// A 2x2 texture, row-major: `(0,0)=0, (1,0)=4, (0,1)=8, (1,1)=12`.
    const SQUARE: [f32; 4] = [0.0, 4.0, 8.0, 12.0];

    fn strip(options: SamplerOptions, u: f32) -> f32 {
        sample_2d(&STRIP, 4, 1, options, u, 0.5)
    }

    fn square(options: SamplerOptions, u: f32, v: f32) -> f32 {
        sample_2d(&SQUARE, 2, 2, options, u, v)
    }

    // ---- texel addresses ----

    #[test]
    fn texel_zero_zero_is_the_top_left_and_rows_run_down() {
        // 3 wide, 2 high; texel (x, y) holds 10 * y + x.
        assert_eq!(texel_index(0, 0, 3, 2), 0);
        assert_eq!(texel_index(2, 0, 3, 2), 2);
        assert_eq!(texel_index(0, 1, 3, 2), 3);
        assert_eq!(texel_index(2, 1, 3, 2), 5);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the texel is (3, 0) but the texture is 3x2")]
    fn a_texel_address_past_the_width_is_out_of_bounds_even_if_the_flat_index_exists() {
        let _ = texel_index(3, 0, 3, 2);
    }

    // ---- nearest ----

    #[test]
    fn nearest_picks_the_texel_the_coordinate_falls_in() {
        // Width 4: texel i covers [i/4, (i+1)/4). u = 0.24 is x = 0.96, still texel 0; u = 0.25 is x = 1.0, texel 1.
        assert_eq!(strip(NEAREST, 0.0), 10.0);
        assert_eq!(strip(NEAREST, 0.24), 10.0);
        assert_eq!(strip(NEAREST, 0.25), 20.0);
        assert_eq!(strip(NEAREST, 0.74), 30.0);
        assert_eq!(strip(NEAREST, 0.75), 40.0);
        // The bottom-right corner is x = 4 = width: the last texel, not one past it.
        assert_eq!(strip(NEAREST, 1.0), 40.0);
    }

    #[test]
    fn nearest_follows_v_down_the_rows() {
        // Height 2: v = 0.49 is y = 0.98, row 0; v = 0.5 is y = 1.0, row 1.
        assert_eq!(square(NEAREST, 0.25, 0.49), 0.0);
        assert_eq!(square(NEAREST, 0.25, 0.5), 8.0);
        assert_eq!(square(NEAREST, 0.75, 0.49), 4.0);
        assert_eq!(square(NEAREST, 0.75, 1.0), 12.0);
    }

    #[test]
    fn clamp_to_edge_clamps_the_coordinate_to_zero_one() {
        assert_eq!(strip(NEAREST, -0.3), 10.0);
        assert_eq!(strip(NEAREST, 1.7), 40.0);
        assert_eq!(strip(NEAREST, f32::NEG_INFINITY), 10.0);
        assert_eq!(strip(NEAREST, f32::INFINITY), 40.0);
        // A NaN coordinate is outside the domain and does not panic; the twin reads texel 0.
        assert_eq!(strip(NEAREST, f32::NAN), 10.0);
    }

    #[test]
    fn repeat_wraps_with_u_minus_floor_u() {
        assert_eq!(strip(NEAREST_REPEAT, 0.0), 10.0);
        assert_eq!(strip(NEAREST_REPEAT, 1.25), 20.0); // 1.25 - 1 = 0.25
        assert_eq!(strip(NEAREST_REPEAT, -0.25), 40.0); // -0.25 - (-1) = 0.75
        assert_eq!(strip(NEAREST_REPEAT, 2.5), 30.0); // 2.5 - 2 = 0.5, x = 2.0
        assert_eq!(strip(NEAREST_REPEAT, -1.0), 10.0); // -1 - (-1) = 0
        // Exactly 1.0 wraps to 0: texel 0, where clamping would give the last texel.
        assert_eq!(strip(NEAREST_REPEAT, 1.0), 10.0);
    }

    #[test]
    fn repeat_outside_the_domain_does_not_panic() {
        // NaN, infinities and `|u| > 2^24` are outside the error domain: the twin returns something and never panics
        // (a NaN position lands on texel 0 for the nearest filter; for the linear filter the value is unspecified).
        for u in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            33_554_432.0,
            -33_554_433.0,
        ] {
            let _ = strip(NEAREST_REPEAT, u);
            let _ = strip(LINEAR_REPEAT, u);
            let _ = sample_2d(&SQUARE, 2, 2, LINEAR_REPEAT, u, u);
            let _ = sample_2d(&SQUARE, 2, 2, NEAREST_REPEAT, u, u);
        }
        assert_eq!(strip(NEAREST_REPEAT, f32::NAN), 10.0);
        // 2^25 is a whole number of periods: it wraps to 0.
        assert_eq!(strip(NEAREST_REPEAT, 33_554_432.0), 10.0);
    }

    // ---- linear ----

    #[test]
    fn linear_at_a_texel_centre_is_that_texel() {
        // The centre of texel i is at u = (i + 0.5) / 4: 0.125, 0.375, 0.625, 0.875.
        assert_eq!(strip(LINEAR, 0.125), 10.0);
        assert_eq!(strip(LINEAR, 0.375), 20.0);
        assert_eq!(strip(LINEAR, 0.625), 30.0);
        assert_eq!(strip(LINEAR, 0.875), 40.0);
    }

    #[test]
    fn linear_between_centres_interpolates_with_the_distance() {
        // u = 0.25 is x = 1.0, halfway between the centres of texels 0 and 1: (10 + 20) / 2.
        assert_eq!(strip(LINEAR, 0.25), 15.0);
        // u = 0.5 is halfway between texels 1 and 2.
        assert_eq!(strip(LINEAR, 0.5), 25.0);
        // u = 0.4375 is x = 1.75, a quarter of the way from texel 1 (20) to texel 2 (30): 20 + 10 * 0.25.
        assert_eq!(strip(LINEAR, 0.4375), 22.5);
    }

    #[test]
    fn linear_with_clamp_repeats_the_edge_texel_past_the_first_and_last_centre() {
        // x' = -0.5 at u = 0: texels -1 and 0 both clamp to texel 0.
        assert_eq!(strip(LINEAR, 0.0), 10.0);
        // x' = 3.5 at u = 1: texels 3 and 4 both clamp to texel 3.
        assert_eq!(strip(LINEAR, 1.0), 40.0);
    }

    #[test]
    fn linear_with_repeat_wraps_to_the_far_edge() {
        // u = 0: x' = -0.5, texels -1 (wraps to 3: 40) and 0 (10), halfway: 25.
        assert_eq!(strip(LINEAR_REPEAT, 0.0), 25.0);
        // u = 1 wraps to 0 before the filter, the same answer.
        assert_eq!(strip(LINEAR_REPEAT, 1.0), 25.0);
        // u = 0.875 is the centre of texel 3; u = 0.9375 is a quarter of the way to the wrapped texel 0:
        // 40 + (10 - 40) * 0.25 = 32.5.
        assert_eq!(strip(LINEAR_REPEAT, 0.9375), 32.5);
    }

    #[test]
    fn bilinear_weights_the_four_texels_and_matches_the_hand_computed_sum() {
        // The middle of the 2x2 texture: all four texels weigh 1/4: (0 + 4 + 8 + 12) / 4.
        assert_eq!(square(LINEAR, 0.5, 0.5), 6.0);
        // fx = 0.25 (u = 0.375), fy = 0.75 (v = 0.625). Weights: t00 0.75*0.25, t10 0.25*0.25, t01 0.75*0.75,
        // t11 0.25*0.75; 0*0.1875 + 4*0.0625 + 8*0.5625 + 12*0.1875 = 0 + 0.25 + 4.5 + 2.25 = 7.
        assert_eq!(square(LINEAR, 0.375, 0.625), 7.0);
        // The corner under Repeat touches all four texels with weight 1/4.
        assert_eq!(square(LINEAR_REPEAT, 0.0, 0.0), 6.0);
        // The corner under clamp touches only texel (0, 0).
        assert_eq!(square(LINEAR, 0.0, 0.0), 0.0);
    }

    #[test]
    fn vector_texels_interpolate_channel_by_channel() {
        let two = [Vec2::new(0.0, 10.0), Vec2::new(4.0, 20.0)];
        // 2x1, u = 0.5 is x = 1.0, halfway between the two centres.
        let half = sample_2d(&two, 2, 1, LINEAR, 0.5, 0.5);
        assert_eq!((half.x, half.y), (2.0, 15.0));
        let four = [Vec4::new(0.0, 1.0, 2.0, 3.0), Vec4::new(4.0, 5.0, 6.0, 7.0)];
        let half = sample_2d(&four, 2, 1, LINEAR, 0.5, 0.5);
        assert_eq!((half.x, half.y, half.z, half.w), (2.0, 3.0, 4.0, 5.0));
        let nearest = sample_2d(&four, 2, 1, NEAREST, 0.75, 0.5);
        assert_eq!(
            (nearest.x, nearest.y, nearest.z, nearest.w),
            (4.0, 5.0, 6.0, 7.0)
        );
    }

    #[test]
    #[should_panic(expected = "cannot sample a texture of 0x0 texels")]
    fn sampling_an_empty_texture_is_refused() {
        let _ = sample_2d::<f32>(&[], 0, 0, NEAREST, 0.5, 0.5);
    }

    // ---- the error domain ----

    #[test]
    fn ulp_is_the_gap_to_the_next_magnitude() {
        assert_eq!(ulp(1.0), 1.1920929e-7); // 2^-23
        assert_eq!(ulp(-1.0), 1.1920929e-7);
        assert_eq!(ulp(12.0), 9.536743e-7); // 12 is in [8, 16): 2^(3-23) = 2^-20
        assert_eq!(ulp(-4.0), 4.7683716e-7); // 4 is in [4, 8): 2^(2-23) = 2^-21
        assert_eq!(ulp(0.0), 1.0e-45); // the smallest subnormal, 2^-149
    }

    #[test]
    fn the_linear_error_bound_is_a_256th_of_the_range_plus_two_ulps_of_the_largest_texel() {
        // Range 12, so 12/256 = 0.046875 = 3 * 2^-6; largest magnitude 12, two ulps = 2 * 2^-20 = 2^-19.
        assert_eq!(linear_error_bound(SQUARE), 0.046_876_907);
        // A flat patch has no weight error: only the rounding term remains.
        assert_eq!(linear_error_bound([5.0; 4]), 2.0 * 4.7683716e-7);
        // The largest magnitude may be a negative texel: range 6, largest 4.
        assert_eq!(
            linear_error_bound([-4.0, 2.0, 0.0, 1.0]),
            6.0 / 256.0 + 2.0 * 4.7683716e-7
        );
    }

    // ---- the carrier ----

    #[test]
    fn a_texture_is_created_zeroed_and_round_trips_its_texels() {
        let device = CpuDevice::new();
        let texture = device
            .texture_2d::<R32Uint>(3, 2, TextureUse::SAMPLED | TextureUse::STORAGE)
            .unwrap();
        assert_eq!((texture.width(), texture.height()), (3, 2));
        assert_eq!(texture.uses(), TextureUse::SAMPLED | TextureUse::STORAGE);
        assert_eq!(texture.read(), [0; 6]);
        device.write_texture(&texture, &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(device.read_texture(&texture).unwrap(), [1, 2, 3, 4, 5, 6]);
        // Row-major: (2, 1) is the last one, (0, 1) the fourth.
        assert_eq!(texture.load(2, 1).unwrap(), 6);
        assert_eq!(texture.load(0, 1).unwrap(), 4);
        texture.store(1, 0, 99).unwrap();
        assert_eq!(texture.read(), [1, 99, 3, 4, 5, 6]);
        assert!(format!("{texture:?}").contains("R32Uint"));
    }

    // gust-wgpu refuses a texture of another device (`Error::ForeignPersistentBuffer`); the twin refuses it with
    // the variant CPU buffers use for a resource of another device instance.
    #[test]
    fn write_texture_refuses_a_texture_of_another_device_and_changes_nothing() {
        let (device, other) = (CpuDevice::new(), CpuDevice::new());
        let texture = other
            .texture_2d::<R32Uint>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        texture.write(&[7, 8]).unwrap();
        assert_eq!(
            device.write_texture(&texture, &[1, 2]).unwrap_err(),
            Error::WrongDevice {
                parameter: "texture"
            }
        );
        assert_eq!(texture.read(), [7, 8]);
        // The owner still can, and a wrong length stays its own error on the owner.
        other.write_texture(&texture, &[1, 2]).unwrap();
        assert_eq!(texture.read(), [1, 2]);
        assert_eq!(
            other.write_texture(&texture, &[1]).unwrap_err(),
            Error::TextureLength {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn write_texture_names_the_resource_when_the_texture_is_borrowed_like_read_texture_does() {
        let device = CpuDevice::new();
        let texture = device
            .texture_2d::<R32Uint>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        texture.write(&[7, 8]).unwrap();
        let held = texture.cell.texels.borrow_mut();
        let expected = Error::Borrowed {
            parameter: "texture",
        };
        assert_eq!(
            device.write_texture(&texture, &[1, 2]).unwrap_err(),
            expected
        );
        assert_eq!(device.read_texture(&texture).unwrap_err(), expected);
        drop(held);
        // The texture is untouched by the refused write, and the same call works once the borrow ends.
        assert_eq!(texture.read(), [7, 8]);
        device.write_texture(&texture, &[1, 2]).unwrap();
        assert_eq!(texture.read(), [1, 2]);
    }

    #[test]
    fn read_texture_refuses_a_texture_of_another_device_and_a_borrowed_one() {
        let (device, other) = (CpuDevice::new(), CpuDevice::new());
        let texture = other
            .texture_2d::<R32Uint>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        texture.write(&[7, 8]).unwrap();
        assert_eq!(
            device.read_texture(&texture).unwrap_err(),
            Error::WrongDevice {
                parameter: "texture"
            }
        );
        assert_eq!(other.read_texture(&texture).unwrap(), [7, 8]);
        // A mutable borrow (a running kernel call) is an error, not a panic, as the `Result` shape promises.
        let held = texture.cell.texels.borrow_mut();
        assert_eq!(
            other.read_texture(&texture).unwrap_err(),
            Error::Borrowed {
                parameter: "texture"
            }
        );
        drop(held);
        assert_eq!(other.read_texture(&texture).unwrap(), [7, 8]);
    }

    #[test]
    fn a_texture_refuses_a_zero_side_a_too_large_size_a_wrong_length_and_a_bad_address() {
        let device = CpuDevice::new();
        for (width, height) in [(0, 4), (4, 0), (0, 0), (u32::MAX, 2), (65536, 65536)] {
            assert_eq!(
                device
                    .texture_2d::<R32Float>(width, height, TextureUse::SAMPLED)
                    .unwrap_err(),
                Error::TextureSize { width, height }
            );
        }
        let texture = device
            .texture_2d::<R32Sint>(2, 2, TextureUse::STORAGE)
            .unwrap();
        assert_eq!(
            texture.write(&[1, 2, 3]).unwrap_err(),
            Error::TextureLength {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(
            texture.read(),
            [0; 4],
            "a refused write leaves the texture untouched"
        );
        let outside = |x, y| Error::TextureCoordinate {
            x,
            y,
            width: 2,
            height: 2,
        };
        assert_eq!(texture.load(2, 0).unwrap_err(), outside(2, 0));
        assert_eq!(texture.load(0, 2).unwrap_err(), outside(0, 2));
        assert_eq!(texture.store(2, 1, 5).unwrap_err(), outside(2, 1));
        assert!(
            texture
                .load(2, 0)
                .unwrap_err()
                .to_string()
                .contains("(2, 0) is outside the 2x2 texture")
        );
    }

    #[test]
    fn a_texture_that_is_borrowed_cannot_be_read_by_the_host() {
        let device = CpuDevice::new();
        let texture = device
            .texture_2d::<R32Float>(1, 1, TextureUse::SAMPLED)
            .unwrap();
        let held = texture.cell.texels.borrow_mut();
        assert_eq!(
            texture.write(&[1.0]).unwrap_err(),
            Error::Borrowed {
                parameter: "texels"
            }
        );
        assert_eq!(
            texture.load(0, 0).unwrap_err(),
            Error::Borrowed {
                parameter: "texels"
            }
        );
        drop(held);
        texture.write(&[1.0]).unwrap();
    }

    #[test]
    fn the_carrier_samples_like_the_free_function_and_checks_the_device() {
        let device = CpuDevice::new();
        let texture = device
            .texture_2d::<R32Float>(4, 1, TextureUse::SAMPLED)
            .unwrap();
        texture.write(&STRIP).unwrap();
        let linear = device.sampler(LINEAR);
        assert_eq!(linear.options(), LINEAR);
        assert_eq!(linear.sample(&texture, 0.4375, 0.5).unwrap(), 22.5);
        let nearest = device.sampler(NEAREST);
        assert_eq!(nearest.sample(&texture, 0.4375, 0.5).unwrap(), 20.0);

        let other = CpuDevice::new();
        assert_eq!(
            other
                .sampler(NEAREST)
                .sample(&texture, 0.5, 0.5)
                .unwrap_err(),
            Error::WrongDevice {
                parameter: "texture"
            }
        );

        let vectors = device
            .texture_2d::<Rg32Float>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        vectors
            .write(&[Vec2::new(0.0, 10.0), Vec2::new(4.0, 20.0)])
            .unwrap();
        let mid = linear.sample(&vectors, 0.5, 0.5).unwrap();
        assert_eq!((mid.x, mid.y), (2.0, 15.0));
    }

    // ---- the kernel-side views over the same texels ----

    #[test]
    fn a_kernel_side_view_samples_the_same_value_as_the_carrier() {
        let view = Texture2D::new(&STRIP, 4, 1);
        let sampler = Sampler::new(LINEAR);
        assert_eq!(view.sample(sampler, 0.4375, 0.5), 22.5);
        assert_eq!(view.load(3, 0), 40.0);
    }

    // ---- a hand-written kernel with a texture and a sampler, through the typed call and the twin ----

    /// `blur(id, tex: Texture2D<f32>, samp: Sampler, out: RWStructuredBuffer<float>)`: `out[i]` is the sample at the
    /// centre of output element `i` of a row of `out.len()` elements.
    struct Blur;

    /// The layout of the sampled `f32` texture `tex`: what the macro writes for a `Texture2D<f32>` parameter (mx.t3),
    /// the texel layout under the texture rule, so a D17 cross-check of this kernel compares real texture facts.
    const fn texel_layout() -> ResourceLayout {
        ResourceLayout::texture_v1(
            <f32 as GpuPod>::LAYOUT,
            TextureLayout {
                format: None,
                dimension: TextureDimension::D2,
                access: TextureAccess::Sampled,
                class: SampleClass::Float,
            },
        )
    }

    impl Kernel for Blur {
        const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
            "texture_tests",
            "blur",
            "texture_tests_blur",
            [64, 1, 1],
            &[
                ParameterDescriptor {
                    name: "id",
                    rust_type: "SV_DispatchThreadID",
                    kind: ParameterKind::Invocation,
                    access: Access::NotApplicable,
                    resource_layout: None,
                    binding: None,
                },
                ParameterDescriptor {
                    name: "tex",
                    rust_type: "Texture2D<float>",
                    kind: ParameterKind::Texture,
                    access: Access::ReadOnly,
                    resource_layout: Some(texel_layout()),
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 0,
                    }),
                },
                ParameterDescriptor {
                    name: "samp",
                    rust_type: "Sampler",
                    kind: ParameterKind::Sampler,
                    access: Access::NotApplicable,
                    resource_layout: None,
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 1,
                    }),
                },
                ParameterDescriptor {
                    name: "out",
                    rust_type: "float",
                    kind: ParameterKind::Storage,
                    access: Access::ReadWrite,
                    resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 2,
                    }),
                },
            ],
            "",
        );
    }

    fn blur_body(id: UVec3, tex: Texture2D<'_, f32>, samp: Sampler, mut out: StorageMut<'_, f32>) {
        let i = id.x;
        if i < out.len() {
            out[i] = tex.sample(samp, (i as f32 + 0.5) / out.len() as f32, 0.5);
        }
    }

    impl Blur {
        /// The `call` the macro will generate: `Kind = Texture` for the texture, `SamplerArg` for the sampler.
        fn call<'r, R0, S1, R2>(
            element_count: u32,
            tex: impl Into<Arg<'r, R0>>,
            samp: &'r S1,
            out: impl Into<Arg<'r, R2>>,
        ) -> SampledCall<'r, R0::Device, Self, 2, 1>
        where
            R0: Resource<Element = f32, Kind = kind::Texture> + 'r,
            S1: SamplerArg<Device = R0::Device>,
            R2: Resource<Element = f32, Device = R0::Device, Kind = kind::Storage> + 'r,
            R0::Device: SamplerDevice,
        {
            Call::new_with_samplers(
                element_count,
                [
                    tex.into().bind(Access::ReadOnly),
                    out.into().bind(Access::ReadWrite),
                ],
                [samp.bind()],
            )
        }
    }

    impl Run<Blur, 2, [CpuSamplerBinding<'_>; 1]> for CpuDevice {
        fn submit_call(
            &self,
            call: &Call<'_, Self, Blur, 2, [CpuSamplerBinding<'_>; 1]>,
        ) -> Result<CpuJob, Error> {
            // `validate` checks the sampler bindings too: this impl cannot skip that.
            crate::cpu::validate(self, call)?;
            let [tex, out] = *call.bindings();
            let [samp] = *call.samplers();
            let tex = read::<f32>(tex, "tex")?;
            let mut out = crate::cpu::write::<f32>(out, "out")?;
            for id in invocations(call.kernel(), call.extent()) {
                blur_body(
                    id,
                    Texture2D::new(&tex.texels, tex.width, tex.height),
                    sampler(samp),
                    StorageMut::new(&mut out),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    /// A texture of the strip, a sampler and an output of `n` elements on one device.
    fn blur_fixture(
        options: SamplerOptions,
        n: usize,
    ) -> (
        CpuDevice,
        CpuTexture2d<R32Float>,
        CpuSampler,
        crate::cpu::CpuBuffer<f32>,
    ) {
        let device = CpuDevice::new();
        let texture = device
            .texture_2d::<R32Float>(4, 1, TextureUse::SAMPLED)
            .unwrap();
        texture.write(&STRIP).unwrap();
        let sampler = device.sampler(options);
        let out = device.buffer(&vec![0.0f32; n]);
        (device, texture, sampler, out)
    }

    #[test]
    fn a_kernel_with_a_texture_and_a_sampler_runs_through_the_typed_call_on_the_twin() {
        // Eight outputs at u = 1/16, 3/16, ..., 15/16 of the 4-texel strip, linear: x = u * 4 = 0.25, 0.75, ... 3.75,
        // x' = x - 0.5 = -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25. Clamped, hand-worked: the first reads
        // texels -1 and 0 (both 10); then 10 + 10 * 0.25, 10 + 10 * 0.75, 20 + 10 * 0.25, 20 + 10 * 0.75,
        // 30 + 10 * 0.25, 30 + 10 * 0.75, and the last reads texels 3 and 4 (both 40).
        let (device, texture, sampler, out) = blur_fixture(LINEAR, 8);
        let call = Blur::call(8, &texture, &sampler, &out);
        device.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), [10.0, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 40.0]);
        assert_eq!(call.kernel().sampler_parameter_count(), 1);
        assert_eq!(call.kernel().texture_parameter_count(), 1);
        assert_eq!(call.samplers().len(), 1);
        assert!(format!("{call:?}").contains("samplers"));
    }

    #[test]
    fn the_same_call_with_a_nearest_sampler_reads_whole_texels() {
        let (device, texture, sampler, out) = blur_fixture(NEAREST, 4);
        device
            .dispatch_call(&Blur::call(4, &texture, &sampler, &out))
            .unwrap();
        assert_eq!(out.read(), STRIP);
    }

    #[test]
    fn a_texture_without_the_sampled_use_is_refused_before_anything_runs() {
        let device = CpuDevice::new();
        let storage_only = device
            .texture_2d::<R32Float>(4, 1, TextureUse::STORAGE)
            .unwrap();
        let sampler = device.sampler(NEAREST);
        let out = device.buffer(&[7.0f32; 4]);
        let error = device
            .dispatch_call(&Blur::call(4, &storage_only, &sampler, &out))
            .unwrap_err();
        assert_eq!(
            error,
            Error::TextureUse {
                parameter: "tex",
                required: TextureUse::SAMPLED,
                actual: TextureUse::STORAGE
            }
        );
        assert!(
            error
                .to_string()
                .contains("`tex` needs a texture created with")
        );
        assert_eq!(
            out.read(),
            [7.0; 4],
            "a refused call leaves the output untouched"
        );
    }

    #[test]
    fn a_texture_or_sampler_of_another_device_instance_is_refused() {
        let (device, texture, sampler, out) = blur_fixture(NEAREST, 4);
        let stranger = CpuDevice::new();
        let foreign_texture = stranger
            .texture_2d::<R32Float>(4, 1, TextureUse::SAMPLED)
            .unwrap();
        assert_eq!(
            device
                .dispatch_call(&Blur::call(4, &foreign_texture, &sampler, &out))
                .unwrap_err(),
            Error::WrongDevice { parameter: "tex" }
        );
        let foreign_sampler = stranger.sampler(NEAREST);
        assert_eq!(
            device
                .dispatch_call(&Blur::call(4, &texture, &foreign_sampler, &out))
                .unwrap_err(),
            Error::WrongDevice { parameter: "samp" }
        );
    }

    #[test]
    fn the_twin_finds_a_texture_by_its_texel_type_and_checks_kind_and_access() {
        let device = CpuDevice::new();
        let uint = device
            .texture_2d::<R32Uint>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        let float = device
            .texture_2d::<R32Float>(2, 1, TextureUse::SAMPLED)
            .unwrap();
        let buffer = device.buffer(&[1.0f32, 2.0]);
        // A texture bound as a float texel type is found only for the float texture.
        assert!(read::<f32>(float.bind(Access::ReadOnly, LengthRule::Shared), "t").is_ok());
        assert_eq!(
            read::<f32>(uint.bind(Access::ReadOnly, LengthRule::Shared), "t")
                .err()
                .unwrap(),
            Error::ElementType { parameter: "t" }
        );
        // A buffer is not a texture, and a read-only binding is not writable.
        assert_eq!(
            read::<f32>(buffer.bind(Access::ReadOnly, LengthRule::Shared), "t")
                .err()
                .unwrap(),
            Error::BindingKind {
                parameter: "t",
                expected: BindingKind::Texture,
                actual: BindingKind::Storage
            }
        );
        assert_eq!(
            write::<f32>(float.bind(Access::ReadOnly, LengthRule::Shared), "t")
                .err()
                .unwrap(),
            Error::Access {
                parameter: "t",
                expected: Access::ReadWrite,
                actual: Access::ReadOnly
            }
        );
        let mut borrowed =
            write::<f32>(float.bind(Access::ReadWrite, LengthRule::Shared), "t").unwrap();
        assert_eq!((borrowed.width, borrowed.height), (2, 1));
        borrowed.texels[1] = 5.0;
        drop(borrowed);
        assert_eq!(float.read(), [0.0, 5.0]);
    }

    #[test]
    fn a_buffer_in_a_texture_parameter_and_a_texture_in_a_buffer_parameter_are_refused_at_submit() {
        let (device, texture, sampler, out) = blur_fixture(NEAREST, 4);
        // The typed call rules these out at compile time; a hand-built call is refused by the backend. Bind the
        // buffer where the texture goes (the call's first binding), and the texture where the buffer goes.
        let wrong = Call::<CpuDevice, Blur, 2, _>::new_with_samplers(
            4,
            [
                out.bind(Access::ReadOnly, LengthRule::Shared),
                texture.bind(Access::ReadWrite, LengthRule::Shared),
            ],
            [sampler.bind()],
        );
        assert_eq!(
            device.dispatch_call(&wrong).unwrap_err(),
            Error::BindingKind {
                parameter: "tex",
                expected: BindingKind::Texture,
                actual: BindingKind::Storage
            }
        );
    }
    #[test]
    fn a_texture_of_another_texel_type_is_refused_for_an_f32_texture_parameter() {
        // `Rg32Uint` texels are `UVec2`, not the `f32` the kernel's `Texture2D<float>` reads. The typed call's
        // `Element = f32` bound rejects it at compile time; a hand-built call is refused by the backend.
        let (device, _, sampler, out) = blur_fixture(NEAREST, 4);
        let wide = device
            .texture_2d::<Rg32Uint>(4, 1, TextureUse::SAMPLED)
            .unwrap();
        let call = Call::<CpuDevice, Blur, 2, _>::new_with_samplers(
            4,
            [
                wide.bind(Access::ReadOnly, LengthRule::Independent),
                out.bind(Access::ReadWrite, LengthRule::Shared),
            ],
            [sampler.bind()],
        );
        assert_eq!(
            device.dispatch_call(&call).unwrap_err(),
            Error::ElementType { parameter: "tex" }
        );
    }

    #[test]
    fn validate_alone_refuses_a_sampler_of_another_device_instance() {
        // The check lives in `cpu::validate`, so a `Run` impl cannot skip it; here nothing but `validate` runs.
        let (device, texture, _, out) = blur_fixture(NEAREST, 4);
        let stranger = CpuDevice::new();
        let foreign = stranger.sampler(NEAREST);
        let call = Blur::call(4, &texture, &foreign, &out);
        assert_eq!(
            crate::cpu::validate(&device, &call).unwrap_err(),
            Error::WrongDevice { parameter: "samp" }
        );
    }

    /// `Blur::dispatch` as the macro will emit it for a kernel with a sampler: generic over the three arguments, the
    /// device projected from the texture, and the `Run` bound spelled with the device's `SamplerBinding`.
    fn blur_dispatch<'r, R0, S1, R2>(
        device: &R0::Device,
        element_count: u32,
        tex: impl Into<Arg<'r, R0>>,
        samp: &'r S1,
        out: impl Into<Arg<'r, R2>>,
    ) -> Result<(), <R0::Device as Device>::Error>
    where
        R0: Resource<Element = f32, Kind = kind::Texture> + 'r,
        S1: SamplerArg<Device = R0::Device>,
        R2: Resource<Element = f32, Device = R0::Device, Kind = kind::Storage> + 'r,
        R0::Device:
            SamplerDevice + Run<Blur, 2, [<R0::Device as SamplerDevice>::SamplerBinding<'r>; 1]>,
    {
        <R0::Device as Run<Blur, 2, [<R0::Device as SamplerDevice>::SamplerBinding<'r>; 1]>>::dispatch_call(
            device,
            &Blur::call(element_count, tex, samp, out),
        )
    }

    /// `Blur::submit`, likewise: the job type is the device's.
    fn blur_submit<'d, 'r, R0, S1, R2>(
        device: &'d R0::Device,
        element_count: u32,
        tex: impl Into<Arg<'r, R0>>,
        samp: &'r S1,
        out: impl Into<Arg<'r, R2>>,
    ) -> Result<<R0::Device as Device>::Job<'d>, <R0::Device as Device>::Error>
    where
        R0: Resource<Element = f32, Kind = kind::Texture> + 'r,
        S1: SamplerArg<Device = R0::Device>,
        R2: Resource<Element = f32, Device = R0::Device, Kind = kind::Storage> + 'r,
        R0::Device:
            SamplerDevice + Run<Blur, 2, [<R0::Device as SamplerDevice>::SamplerBinding<'r>; 1]>,
    {
        <R0::Device as Run<Blur, 2, [<R0::Device as SamplerDevice>::SamplerBinding<'r>; 1]>>::submit_call(
            device,
            &Blur::call(element_count, tex, samp, out),
        )
    }

    #[test]
    fn the_generic_dispatch_the_macro_will_emit_compiles_and_runs_on_the_twin() {
        let (device, texture, sampler, out) = blur_fixture(LINEAR, 8);
        blur_dispatch(&device, 8, &texture, &sampler, &out).unwrap();
        assert_eq!(out.read(), [10.0, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 40.0]);
        let (device, texture, sampler, out) = blur_fixture(NEAREST, 4);
        let job = blur_submit(&device, 4, &texture, &sampler, &out).unwrap();
        crate::Job::wait(&job).unwrap();
        assert_eq!(out.read(), STRIP);
        // The generic path reports the same refusals as the concrete one.
        let stranger = CpuDevice::new();
        let foreign = stranger.sampler(NEAREST);
        assert_eq!(
            blur_dispatch(&device, 4, &texture, &foreign, &out).unwrap_err(),
            Error::WrongDevice { parameter: "samp" }
        );
    }
}
