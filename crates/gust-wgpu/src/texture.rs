//! Textures and samplers on the GPU (decision D64, slice mx.t2).
//!
//! [`GpuTexture2d`] is a 2-D texture of one [`Format`] with the uses its creator asked for; [`GpuSampler`] is a
//! sampler. Both belong to one [`HeadlessDevice`]. The texture is a [`Resource`] of kind `kind::Texture`, so a typed
//! call takes it where a kernel declares a `Texture2D`, `StorageTexture2D` or `WriteTexture2D` parameter; the sampler
//! is not a resource (it holds no elements) but the call's own argument kind, [`SamplerArg`].
//!
//! # The mapping to wgpu
//!
//! [`wgpu_format`] is the one place that turns a `gust::texture::Format` into a `wgpu::TextureFormat`. The size of a
//! texel and the sample type of a format are wgpu's own (`block_copy_size`, `sample_type`), not a second table here.
//!
//! # The two layout variants
//!
//! A float texture of the dialect is a 32-bit float one, and WebGPU filters those only on a device that enabled
//! `float32-filterable`. The bind-group layout of a kernel is therefore built per device in one of two variants, from
//! the features the device actually has ([`HeadlessDevice::float32_filterable`]):
//!
//! | | with the feature | without it |
//! | --- | --- | --- |
//! | sampled float texture | `Float { filterable: true }` | `Float { filterable: false }` |
//! | sampler | `Filtering` | `NonFiltering` |
//!
//! Sampled `u32` and `i32` textures are `Uint` and `Sint` in both. At submit a [`Filter::Linear`] sampler under the
//! variant without the feature is [`Error::TextureFilter`]; a [`Filter::Nearest`] one binds under either.
//!
//! # Storage textures
//!
//! A storage texture is bound `WriteOnly` when the kernel only writes it (`WriteTexture2D`, any of the nine formats)
//! and `ReadWrite` only for a `StorageTexture2D` of an `r32` format. wgpu refuses `ReadWrite` for any wider format
//! (decision D120), so the layout entry follows the descriptor's [`TextureAccess`], never the parameter's `Access`
//! alone (both a write-only and a read-write storage texture are `Access::ReadWrite` there).

use std::marker::PhantomData;

use gust::texture::{
    Address, Filter, Format, SampleClass, SamplerOptions, TextureFormat, TextureUse,
};
use gust::{
    Access, BindingKind, GpuPod, LengthRule, ParameterDescriptor, Resource, SamplerArg,
    SamplerDevice, ShaderLayoutRules, TextureAccess, TextureLayout, TypeLayout, TypeLayoutKind,
    pod_slice_as_bytes,
};

use crate::{
    BufferBinding, Completion, Error, GpuBufferAccess, HeadlessDevice, MappingReady,
    PendingMapping, Raw, pod_vec_from_bytes,
};

/// The name of the device feature a linear filter of a 32-bit float texture needs, as an error spells it.
pub(crate) const FLOAT32_FILTERABLE_NAME: &str = "float32-filterable";

/// The `wgpu` texture format of `format`: the one place the dialect's [`Format`] meets a wgpu type. `None` for a
/// format this backend does not know (`Format` is `#[non_exhaustive]`).
pub(crate) fn wgpu_format(format: Format) -> Option<wgpu::TextureFormat> {
    Some(match format {
        Format::R32Uint => wgpu::TextureFormat::R32Uint,
        Format::R32Sint => wgpu::TextureFormat::R32Sint,
        Format::R32Float => wgpu::TextureFormat::R32Float,
        Format::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        Format::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
        Format::Rg32Float => wgpu::TextureFormat::Rg32Float,
        Format::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        Format::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
        Format::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        _ => return None,
    })
}

/// [`wgpu_format`] or the error of a format this backend does not map.
pub(crate) fn mapped_format(what: &str, format: Format) -> Result<wgpu::TextureFormat, Error> {
    wgpu_format(format).ok_or_else(|| Error::TextureFormat {
        parameter: what.to_owned(),
        expected: "a texture format this backend maps".to_owned(),
        actual: format,
    })
}

/// A format of the class: the one whose wgpu sample type stands for every format of it. The nine formats of the
/// dialect agree within a class (all float ones are 32-bit).
fn class_format(class: SampleClass) -> Option<wgpu::TextureFormat> {
    match class {
        SampleClass::Float => Some(wgpu::TextureFormat::R32Float),
        SampleClass::Uint => Some(wgpu::TextureFormat::R32Uint),
        SampleClass::Sint => Some(wgpu::TextureFormat::R32Sint),
        _ => None,
    }
}

/// The uses a format allows on a device with `features`, as the dialect's flags.
fn allowed_uses(format: wgpu::TextureFormat, features: wgpu::Features) -> TextureUse {
    let allowed = format.guaranteed_format_features(features).allowed_usages;
    let mut uses = TextureUse::NONE;
    if allowed.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        uses = uses | TextureUse::SAMPLED;
    }
    if allowed.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        uses = uses | TextureUse::STORAGE;
    }
    if allowed.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        uses = uses | TextureUse::TARGET;
    }
    uses
}

/// The wgpu usages of a [`TextureUse`]; every texture can be copied from and to.
fn wgpu_usages(uses: TextureUse) -> wgpu::TextureUsages {
    let mut usages = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
    if uses.contains(TextureUse::SAMPLED) {
        usages |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if uses.contains(TextureUse::STORAGE) {
        usages |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if uses.contains(TextureUse::TARGET) {
        usages |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    usages
}

/// A 2-D texture of format `F`, owned by one [`HeadlessDevice`].
///
/// Create it with [`HeadlessDevice::create_texture_2d`], fill it with [`HeadlessDevice::write_texture`] and read it
/// back with `HeadlessDevice::read_texture`. It is a [`Resource`] of kind `kind::Texture`: a typed call takes it
/// where a kernel declares a texture parameter, and the use it was created with ([`TextureUse`]) must fit that
/// parameter (a sampled one needs `SAMPLED`, a storage one `STORAGE`), checked at submit ([`Error::TextureUse`]).
pub struct GpuTexture2d<F: TextureFormat> {
    raw: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    uses: TextureUse,
    device_id: u64,
    format: PhantomData<fn() -> F>,
}

impl<F: TextureFormat> GpuTexture2d<F> {
    /// The width in texels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The height in texels.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The uses the texture was created with.
    pub const fn uses(&self) -> TextureUse {
        self.uses
    }

    /// The format of the texture.
    pub const fn format(&self) -> Format {
        F::FORMAT
    }

    /// The identity of the device that created the texture ([`HeadlessDevice::id`]).
    pub const fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Expose the underlying texture for advanced wgpu code.
    pub fn as_wgpu_texture(&self) -> &wgpu::Texture {
        &self.raw
    }
}

/// A sampler, owned by one [`HeadlessDevice`]: the argument of a typed call for a kernel's `Sampler` parameter.
///
/// It is a [`SamplerArg`], not a [`Resource`]. A [`Filter::Linear`] sampler can be created on any device but bound
/// only on one that enabled `float32-filterable` ([`HeadlessDevice::float32_filterable`]).
pub struct GpuSampler {
    raw: wgpu::Sampler,
    options: SamplerOptions,
    device_id: u64,
}

impl GpuSampler {
    /// The options the sampler was created with.
    pub const fn options(&self) -> SamplerOptions {
        self.options
    }

    /// The identity of the device that created the sampler ([`HeadlessDevice::id`]).
    pub const fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Expose the underlying sampler for advanced wgpu code.
    pub fn as_wgpu_sampler(&self) -> &wgpu::Sampler {
        &self.raw
    }
}

/// The erased binding of a [`GpuSampler`] for one dispatch ([`SamplerDevice::SamplerBinding`]).
#[derive(Clone, Copy)]
pub struct GpuSamplerBinding<'a> {
    raw: &'a wgpu::Sampler,
    options: SamplerOptions,
    device_id: u64,
}

impl<'a> GpuSamplerBinding<'a> {
    /// The binding of `sampler`.
    pub fn new(sampler: &'a GpuSampler) -> Self {
        Self {
            raw: &sampler.raw,
            options: sampler.options,
            device_id: sampler.device_id,
        }
    }

    /// The options of the bound sampler.
    pub const fn options(&self) -> SamplerOptions {
        self.options
    }

    pub(crate) fn resource(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Sampler(self.raw)
    }
}

impl SamplerDevice for HeadlessDevice {
    type SamplerBinding<'resources> = GpuSamplerBinding<'resources>;
}

impl SamplerArg for GpuSampler {
    type Device = HeadlessDevice;

    fn bind(&self) -> GpuSamplerBinding<'_> {
        GpuSamplerBinding::new(self)
    }
}

/// What a texture binding carries beside the buffer-shaped fields of a [`BufferBinding`].
#[derive(Clone, Copy)]
pub(crate) struct TextureBinding<'a> {
    pub(crate) view: &'a wgpu::TextureView,
    pub(crate) format: Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) uses: TextureUse,
}

impl<'a> BufferBinding<'a> {
    /// A binding of `texture`: `ReadOnly` for a sampled parameter, `ReadWrite` for a storage one.
    fn new_texture<F: TextureFormat>(texture: &'a GpuTexture2d<F>, access: Access) -> Self {
        Self {
            raw: Raw::Texture(&texture.raw),
            len: 1,
            layout: <F::Texel as GpuPod>::LAYOUT,
            buffer_access: if access == Access::ReadOnly {
                GpuBufferAccess::ReadOnly
            } else {
                GpuBufferAccess::ReadWrite
            },
            access,
            device_id: texture.device_id,
            independent_length: true,
            indirect: false,
            kind: BindingKind::Texture,
            bytes: 0,
            texture: Some(TextureBinding {
                view: &texture.view,
                format: F::FORMAT,
                width: texture.width,
                height: texture.height,
                uses: texture.uses,
            }),
        }
    }
}

/// A texture is a typed resource of [`HeadlessDevice`] of kind `kind::Texture`: a generated call accepts it only for a
/// texture parameter, and a buffer where a texture is expected does not compile. The element is the texel type, the
/// length rule is ignored (a texture has a width and a height and no element count), and the access is `ReadOnly` for
/// a sampled parameter and `ReadWrite` for a storage one.
impl<F: TextureFormat> Resource for GpuTexture2d<F> {
    type Element = F::Texel;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Texture;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        _length: LengthRule,
    ) -> BufferBinding<'resources> {
        BufferBinding::new_texture(self, access)
    }
}

/// The texel counts and the sampler count of a kernel's parameters, for the per-stage limits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TextureCounts {
    pub(crate) sampled: usize,
    pub(crate) storage: usize,
    pub(crate) samplers: usize,
}

impl TextureCounts {
    /// A kernel with no texture and no sampler.
    pub(crate) const NONE: Self = Self {
        sampled: 0,
        storage: 0,
        samplers: 0,
    };

    /// The counts of `resources`: sampled textures, storage textures (read-write and write-only) and samplers.
    pub(crate) fn of(resources: &[&ParameterDescriptor]) -> Self {
        let mut counts = Self::NONE;
        for parameter in resources {
            match parameter.kind {
                gust::ParameterKind::Texture => match texture_layout(parameter).map(|t| t.access) {
                    Some(TextureAccess::Sampled) => counts.sampled += 1,
                    Some(_) => counts.storage += 1,
                    None => {}
                },
                gust::ParameterKind::Sampler => counts.samplers += 1,
                _ => {}
            }
        }
        counts
    }
}

/// The texture facts of a parameter's resource layout, when it has them.
pub(crate) fn texture_layout(parameter: &ParameterDescriptor) -> Option<TextureLayout> {
    match parameter.resource_layout?.rules {
        ShaderLayoutRules::TextureV1(texture) => Some(texture),
        _ => None,
    }
}

/// The scalar class and width of a texture result type: a 32-bit scalar or a 2- or 4-wide vector of one.
fn result_shape(layout: TypeLayout) -> Option<(SampleClass, u32)> {
    let (scalar, count) = match layout.kind {
        TypeLayoutKind::Scalar(scalar) => (scalar, 1),
        TypeLayoutKind::Vector { scalar, count } => (scalar, count),
        _ => return None,
    };
    let class = match scalar {
        gust::ScalarKind::F32 => SampleClass::Float,
        gust::ScalarKind::U32 => SampleClass::Uint,
        gust::ScalarKind::I32 => SampleClass::Sint,
        _ => return None,
    };
    (matches!(count, 1 | 2 | 4)).then_some((class, count))
}

/// Whether the descriptor of a texture parameter is coherent: the facts [`ShaderLayoutRules::TextureV1`] carries agree
/// with each other and with the parameter's access. (The match against the compiler's reflection is `gust`'s, D17.)
pub(crate) fn texture_parameter_is_valid(parameter: &ParameterDescriptor) -> bool {
    let Some(layout) = parameter.resource_layout else {
        return false;
    };
    let ShaderLayoutRules::TextureV1(texture) = layout.rules else {
        return false;
    };
    let Some((class, count)) = result_shape(layout.element) else {
        return false;
    };
    if layout.element_stride != layout.element.stride
        || texture.class != class
        || texture.dimension != gust::TextureDimension::D2
    {
        return false;
    }
    match (texture.access, texture.format) {
        (TextureAccess::Sampled, None) => parameter.access == Access::ReadOnly,
        (access @ (TextureAccess::ReadWrite | TextureAccess::WriteOnly), Some(format)) => {
            parameter.access == Access::ReadWrite
                && format.sample_class() == texture.class
                && format.channels() == count
                && (access == TextureAccess::WriteOnly || format.read_write())
        }
        _ => false,
    }
}

/// Whether the descriptor of a sampler parameter is coherent: it has no resource contract.
pub(crate) fn sampler_parameter_is_valid(parameter: &ParameterDescriptor) -> bool {
    parameter.resource_layout.is_none() && parameter.access == Access::NotApplicable
}

impl HeadlessDevice {
    /// Whether this device enabled the `float32-filterable` feature, which a linear filter of a 32-bit float texture
    /// needs. It reports the enabled state, not what the adapter offers: a device opened with
    /// [`FeaturesPolicy::Default`](crate::FeaturesPolicy::Default) says `false` on an adapter that has the feature.
    pub fn float32_filterable(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
    }

    /// The `wgpu` sample type of a sampled texture parameter of `class` under this device's layout variant (see the
    /// module documentation): wgpu's own `sample_type` for a format of the class, on the device's enabled features.
    pub(crate) fn sampled_sample_type(
        &self,
        parameter: &ParameterDescriptor,
        class: SampleClass,
    ) -> Result<wgpu::TextureSampleType, Error> {
        class_format(class)
            .and_then(|format| format.sample_type(None, Some(self.device.features())))
            .ok_or_else(|| Error::InvalidResourceLayout {
                parameter: parameter.name.to_owned(),
                actual: parameter.resource_layout.map(Box::new),
            })
    }

    /// The layout entry type of a texture or sampler parameter, under this device's layout variant.
    pub(crate) fn texture_binding_type(
        &self,
        parameter: &ParameterDescriptor,
    ) -> Result<wgpu::BindingType, Error> {
        if parameter.kind == gust::ParameterKind::Sampler {
            return Ok(wgpu::BindingType::Sampler(if self.float32_filterable() {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }));
        }
        let texture = texture_layout(parameter).ok_or_else(|| Error::InvalidResourceLayout {
            parameter: parameter.name.to_owned(),
            actual: parameter.resource_layout.map(Box::new),
        })?;
        Ok(match (texture.access, texture.format) {
            (TextureAccess::Sampled, _) => wgpu::BindingType::Texture {
                sample_type: self.sampled_sample_type(parameter, texture.class)?,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            (access, Some(format)) => wgpu::BindingType::StorageTexture {
                // Write-only stays write-only: wgpu refuses `ReadWrite` outside the `r32` formats (D120).
                access: if access == TextureAccess::ReadWrite {
                    wgpu::StorageTextureAccess::ReadWrite
                } else {
                    wgpu::StorageTextureAccess::WriteOnly
                },
                format: mapped_format(parameter.name, format)?,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            (_, None) => {
                return Err(Error::InvalidResourceLayout {
                    parameter: parameter.name.to_owned(),
                    actual: parameter.resource_layout.map(Box::new),
                });
            }
        })
    }

    /// Create a 2-D texture of format `F`, `width` by `height` texels, for the given uses.
    ///
    /// The texture can always be written ([`write_texture`](Self::write_texture)) and read back
    /// (`read_texture`); `uses` adds what a kernel may do with it: `SAMPLED` for a
    /// `Texture2D` parameter, `STORAGE` for a `StorageTexture2D` or `WriteTexture2D` one, `TARGET` for a render
    /// attachment. Its contents start as zeros.
    ///
    /// # Errors
    ///
    /// - [`Error::TextureSize`] for a zero width or height.
    /// - [`Error::DeviceLimit`] naming `max_texture_dimension_2d` for a side above the device's limit.
    /// - [`Error::TextureUse`] when the format does not allow one of `uses` on this device.
    /// - [`Error::OutOfMemory`] when the adapter has no memory for it (native).
    pub fn create_texture_2d<F: TextureFormat>(
        &self,
        width: u32,
        height: u32,
        uses: TextureUse,
    ) -> Result<GpuTexture2d<F>, Error> {
        let label = "GUST texture";
        if width == 0 || height == 0 {
            return Err(Error::TextureSize {
                width,
                height,
                reason: "a texture needs at least one texel along each axis".to_owned(),
            });
        }
        let maximum = self.limits.max_texture_dimension_2d;
        for side in [width, height] {
            if side > maximum {
                return Err(Error::DeviceLimit {
                    kernel: format!("create_texture_2d {width}x{height}"),
                    limit: "max_texture_dimension_2d",
                    requested: u64::from(side),
                    maximum: u64::from(maximum),
                });
            }
        }
        let format = mapped_format("create_texture_2d", F::FORMAT)?;
        let allowed = allowed_uses(format, self.device.features());
        if !allowed.contains(uses) {
            return Err(Error::TextureUse {
                parameter: "create_texture_2d".to_owned(),
                required: uses,
                actual: allowed,
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        let bytes = u64::from(width) * u64::from(height) * u64::from(F::BYTES);
        #[cfg(not(target_arch = "wasm32"))]
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let raw = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu_usages(uses),
            view_formats: &[],
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_scope(label, bytes, scope.pop())?;
        let view = raw.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(GpuTexture2d {
            raw,
            view,
            width,
            height,
            uses,
            device_id: self.id,
            format: PhantomData,
        })
    }

    /// Fail with [`Error::ForeignPersistentBuffer`] unless `texture` belongs to this device.
    fn validate_texture_device<F: TextureFormat>(
        &self,
        texture: &GpuTexture2d<F>,
    ) -> Result<(), Error> {
        if texture.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        Ok(())
    }

    /// Replace the whole contents of `texture` with `texels`, row-major: texel `(x, y)` is
    /// `texels[y * width + x]`, so `texels` holds exactly `width * height` of them
    /// ([`Error::TextureSize`] otherwise).
    ///
    /// The write is a `Queue::write_texture`: ordered after every earlier submission and before the next one, so a
    /// kernel submitted afterwards reads what was written.
    pub fn write_texture<F: TextureFormat>(
        &self,
        texture: &GpuTexture2d<F>,
        texels: &[F::Texel],
    ) -> Result<(), Error> {
        self.validate_texture_device(texture)?;
        let expected = u64::from(texture.width) * u64::from(texture.height);
        if texels.len() as u64 != expected {
            return Err(Error::TextureSize {
                width: texture.width,
                height: texture.height,
                reason: format!("expected {expected} texels, got {}", texels.len()),
            });
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pod_slice_as_bytes(texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture.width * F::BYTES),
                rows_per_image: Some(texture.height),
            },
            wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Copy `texture` back to host memory, row-major (the layout of [`write_texture`](Self::write_texture)).
    ///
    /// Blocks the calling thread until the copy has completed; not available on `wasm32`, where
    /// [`read_texture_async`](Self::read_texture_async) is the readback. The copy goes through a staging buffer
    /// whose rows are padded to 256 bytes (WebGPU's rule for a texture-to-buffer copy); the padding is removed here.
    ///
    /// # Errors
    ///
    /// - [`Error::ForeignPersistentBuffer`] for a texture of another device.
    /// - [`Error::DeviceLimit`] naming `max_buffer_size` when the padded staging buffer (`padded row bytes * height`)
    ///   is larger than the device's `max_buffer_size`: the readback is one copy, not row bands, so a very large
    ///   texture (rgba32f at 8192 x 8192 is 1 GiB) cannot be read back on a device with the default 256 MiB limit.
    /// - [`Error::OutOfMemory`] when the adapter has no memory for the staging buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_texture<F: TextureFormat>(
        &self,
        texture: &GpuTexture2d<F>,
    ) -> Result<Vec<F::Texel>, Error> {
        let readback = self.start_texture_readback(texture)?;
        readback.wait(&self.device)?;
        crate::block_on_mapping(readback.finish(&self.device))
    }

    /// `read_texture` without blocking the caller. On native the future completes only while
    /// something polls the device; each poll of it polls the device once (see
    /// [`read_typed_buffer_async`](Self::read_typed_buffer_async), which behaves the same way).
    ///
    /// The errors are those of `read_texture`, the `max_buffer_size` one included.
    pub async fn read_texture_async<F: TextureFormat>(
        &self,
        texture: &GpuTexture2d<F>,
    ) -> Result<Vec<F::Texel>, Error> {
        self.start_texture_readback(texture)?
            .finish(&self.device)
            .await
    }

    /// Copy `texture` into a mappable staging buffer, submit the copy and request the mapping.
    fn start_texture_readback<F: TextureFormat>(
        &self,
        texture: &GpuTexture2d<F>,
    ) -> Result<TextureReadback<F::Texel>, Error> {
        self.validate_texture_device(texture)?;
        let row_bytes = texture.width * F::BYTES;
        let padded_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = u64::from(padded_row) * u64::from(texture.height);
        let label = "GUST texture readback";
        let readback = self.allocate(
            label,
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST texture readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(texture.height),
                },
            },
            wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let mapped = Completion::new();
        let callback = mapped.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = callback.complete(result.map_err(|error| error.to_string()));
        });
        Ok(TextureReadback {
            mapping: PendingMapping { readback, mapped },
            height: texture.height as usize,
            row_bytes: row_bytes as usize,
            padded_row: padded_row as usize,
            texel: PhantomData,
        })
    }

    /// Create a sampler with the given options (the same address mode on both axes).
    ///
    /// A [`Filter::Linear`] sampler is created on any device and bound only on one that enabled
    /// `float32-filterable` ([`float32_filterable`](Self::float32_filterable)); binding it elsewhere is
    /// [`Error::TextureFilter`] at submit. A [`Filter::Nearest`] sampler binds on every device.
    pub fn create_sampler(&self, options: SamplerOptions) -> Result<GpuSampler, Error> {
        // `Filter` and `Address` are `#[non_exhaustive]`: a later variant this backend has no mapping for is refused.
        let unknown = || Error::SamplerOptions {
            filter: options.filter,
            address: options.address,
        };
        let filter = match options.filter {
            Filter::Nearest => wgpu::FilterMode::Nearest,
            Filter::Linear => wgpu::FilterMode::Linear,
            _ => return Err(unknown()),
        };
        let address = match options.address {
            Address::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            Address::Repeat => wgpu::AddressMode::Repeat,
            _ => return Err(unknown()),
        };
        let raw = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("GUST sampler"),
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Ok(GpuSampler {
            raw,
            options,
            device_id: self.id,
        })
    }
}

/// A texture readback after its copy was submitted and its mapping requested.
struct TextureReadback<T> {
    mapping: PendingMapping,
    height: usize,
    /// The bytes of one row of texels.
    row_bytes: usize,
    /// The bytes of one row in the staging buffer: `row_bytes` rounded up to 256.
    padded_row: usize,
    texel: PhantomData<fn() -> T>,
}

impl<T: GpuPod> TextureReadback<T> {
    /// Block until the device has finished the submitted work (see `TypedReadback::wait`).
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, device: &wgpu::Device) -> Result<(), Error> {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;
        Ok(())
    }

    /// Await the mapping, copy the rows out without their padding and unmap.
    async fn finish(self, device: &wgpu::Device) -> Result<Vec<T>, Error> {
        let Self {
            mapping: PendingMapping { readback, mapped },
            height,
            row_bytes,
            padded_row,
            ..
        } = self;
        MappingReady { device, mapped }.await?;
        let view = readback
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let mut packed = Vec::with_capacity(row_bytes * height);
        for row in 0..height {
            let start = row * padded_row;
            packed.extend_from_slice(&view[start..start + row_bytes]);
        }
        drop(view);
        readback.unmap();
        pod_vec_from_bytes(&packed)
    }
}

/// The submit-time checks of one texture binding against its parameter, in this order: the format (a storage parameter
/// needs its exact format, a sampled one a format of its sample class: [`Error::TextureFormat`]), the device, the
/// access the binding was made with, then the use the texture was created with ([`Error::TextureUse`]: a sampled
/// parameter needs `SAMPLED`, a storage one `STORAGE`).
pub(crate) fn validate_texture_binding(
    device_id: u64,
    parameter: &ParameterDescriptor,
    binding: &BufferBinding<'_>,
    texture: TextureBinding<'_>,
) -> Result<(), Error> {
    let layout =
        texture_layout(parameter).expect("validate_kernel keeps coherent texture parameters");
    match layout.format {
        Some(format) if texture.format != format => {
            return Err(Error::TextureFormat {
                parameter: parameter.name.to_owned(),
                expected: format!("a {format} texture"),
                actual: texture.format,
            });
        }
        None => {
            // A sampled parameter reads one texel type, as the typed call and the twin do: the class and the
            // channel count must both match (`Texture2D<float4>` on an `r32float` texture is refused, not read as
            // `(r, 0, 0, 1)`).
            let channels = parameter
                .resource_layout
                .and_then(|layout| result_shape(layout.element))
                .map(|(_, channels)| channels);
            if texture.format.sample_class() != layout.class
                || channels.is_some_and(|channels| texture.format.channels() != channels)
            {
                return Err(Error::TextureFormat {
                    parameter: parameter.name.to_owned(),
                    expected: format!(
                        "a texture with {} {} texel channel(s)",
                        channels.unwrap_or(1),
                        layout.class.name()
                    ),
                    actual: texture.format,
                });
            }
        }
        _ => {}
    }
    if binding.device_id != device_id {
        return Err(Error::ForeignPersistentBuffer);
    }
    if parameter.access != binding.access {
        return Err(Error::BindingAccess {
            parameter: parameter.name.to_owned(),
            expected: parameter.access,
            actual: binding.access,
        });
    }
    let required = if layout.access == TextureAccess::Sampled {
        TextureUse::SAMPLED
    } else {
        TextureUse::STORAGE
    };
    if !texture.uses.contains(required) {
        return Err(Error::TextureUse {
            parameter: parameter.name.to_owned(),
            required,
            actual: texture.uses,
        });
    }
    Ok(())
}

/// The submit-time checks of one sampler binding: its device, and its filter against the device's layout variant. A
/// [`Filter::Linear`] sampler needs `float32-filterable` ([`Error::TextureFilter`] without it); a
/// [`Filter::Nearest`] one binds under either variant.
pub(crate) fn validate_sampler_binding(
    device_id: u64,
    float32_filterable: bool,
    parameter: &ParameterDescriptor,
    sampler: &GpuSamplerBinding<'_>,
) -> Result<(), Error> {
    if sampler.device_id != device_id {
        return Err(Error::ForeignPersistentBuffer);
    }
    if sampler.options.filter != Filter::Nearest && !float32_filterable {
        return Err(Error::TextureFilter {
            parameter: parameter.name.to_owned(),
            filter: sampler.options.filter,
            feature: FLOAT32_FILTERABLE_NAME,
        });
    }
    Ok(())
}

/// The dimension check of a launch: every bound texture against `max_texture_dimension_2d` of the limits in force.
/// A texture is created within the device's own limit, so this fires only under lowered limits
/// ([`HeadlessDevice::with_limits`]).
pub(crate) fn check_texture_dimensions(
    limits: &wgpu::Limits,
    kernel: &gust::KernelDescriptor,
    bindings: &[BufferBinding<'_>],
) -> Result<(), Error> {
    let maximum = u64::from(limits.max_texture_dimension_2d);
    for texture in bindings.iter().filter_map(|binding| binding.texture) {
        for side in [texture.width, texture.height] {
            if u64::from(side) > maximum {
                return Err(Error::DeviceLimit {
                    kernel: kernel.qualified_name().to_string(),
                    limit: "max_texture_dimension_2d",
                    requested: u64::from(side),
                    maximum,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use gust::texture::{
        R32Float, R32Sint, R32Uint, Rg32Float, Rg32Sint, Rg32Uint, Rgba32Float, Rgba32Sint,
        Rgba32Uint,
    };

    use super::*;

    /// The size and the sample type the dialect says for a format are wgpu's own (the unit test of plan R1).
    #[test]
    fn the_dialect_agrees_with_wgpu_on_every_format() {
        for format in Format::ALL {
            let mapped = wgpu_format(format).expect("every format of the dialect is mapped");
            assert_eq!(
                mapped.block_copy_size(None),
                Some(format.texel_bytes()),
                "{format}"
            );
            let class = match mapped.sample_type(None, None) {
                Some(wgpu::TextureSampleType::Float { .. }) => SampleClass::Float,
                Some(wgpu::TextureSampleType::Uint) => SampleClass::Uint,
                Some(wgpu::TextureSampleType::Sint) => SampleClass::Sint,
                other => panic!("{format}: unexpected sample type {other:?}"),
            };
            assert_eq!(class, format.sample_class(), "{format}");
            // Every format of the dialect is a plain colour format with one block per texel.
            assert_eq!(mapped.block_dimensions(), (1, 1), "{format}");
        }
    }

    fn bytes<F: TextureFormat>() {
        let mapped = wgpu_format(F::FORMAT).expect("mapped");
        assert_eq!(mapped.block_copy_size(None), Some(F::BYTES));
    }

    #[test]
    fn every_marker_has_the_block_size_wgpu_gives_its_format() {
        bytes::<R32Uint>();
        bytes::<R32Sint>();
        bytes::<R32Float>();
        bytes::<Rg32Uint>();
        bytes::<Rg32Sint>();
        bytes::<Rg32Float>();
        bytes::<Rgba32Uint>();
        bytes::<Rgba32Sint>();
        bytes::<Rgba32Float>();
    }

    #[test]
    fn the_class_formats_carry_the_sample_type_of_their_class() {
        let features = wgpu::Features::empty();
        for (class, expected) in [
            (
                SampleClass::Float,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            (SampleClass::Uint, wgpu::TextureSampleType::Uint),
            (SampleClass::Sint, wgpu::TextureSampleType::Sint),
        ] {
            let format = class_format(class).expect("a class format");
            assert_eq!(format.sample_type(None, Some(features)), Some(expected));
        }
        assert_eq!(
            class_format(SampleClass::Float).and_then(
                |format| format.sample_type(None, Some(wgpu::Features::FLOAT32_FILTERABLE))
            ),
            Some(wgpu::TextureSampleType::Float { filterable: true })
        );
    }

    #[test]
    fn texture_uses_map_to_wgpu_usages_with_the_copies_always_on() {
        let copies = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        assert_eq!(wgpu_usages(TextureUse::NONE), copies);
        assert_eq!(
            wgpu_usages(TextureUse::SAMPLED),
            copies | wgpu::TextureUsages::TEXTURE_BINDING
        );
        assert_eq!(
            wgpu_usages(TextureUse::STORAGE),
            copies | wgpu::TextureUsages::STORAGE_BINDING
        );
        assert_eq!(
            wgpu_usages(TextureUse::TARGET),
            copies | wgpu::TextureUsages::RENDER_ATTACHMENT
        );
    }

    #[test]
    fn every_format_allows_sampling_and_storage_on_a_device_without_extra_features() {
        for format in Format::ALL {
            let mapped = wgpu_format(format).expect("mapped");
            let allowed = allowed_uses(mapped, wgpu::Features::empty());
            assert!(
                allowed.contains(TextureUse::SAMPLED | TextureUse::STORAGE),
                "{format}"
            );
        }
    }
}
