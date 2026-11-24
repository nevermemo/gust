use crate::texture::TextureBinding;
use crate::{GpuBuffer, GpuBufferAccess, GpuUniform, same_buffer};
use gust::{Access, BindingKind, GpuPod};

/// One persistent buffer bound to an `f32` storage parameter.
///
/// Deprecated with `F32BufferDispatch`; [`BufferBinding`] binds a buffer of any element type:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu_module;
/// use gust_wgpu::{F32BufferBinding, GpuBufferAccess, HeadlessDevice};
///
/// #[gpu_module]
/// mod vector_add {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn add(
///         id: SV_DispatchThreadID,
///         a: StructuredBuffer<float>,
///         b: StructuredBuffer<float>,
///         mut out: RWStructuredBuffer<float>,
///     ) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = a[i] + b[i];
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let a = device.create_f32_buffer("a", &[1.0, 2.0], GpuBufferAccess::ReadOnly);
///     let b = device.create_f32_buffer("b", &[10.0, 20.0], GpuBufferAccess::ReadOnly);
///     let out = device.create_f32_buffer("out", &[0.0, 0.0], GpuBufferAccess::ReadWrite);
///     let bindings = [
///         F32BufferBinding::ReadOnly(&a),
///         F32BufferBinding::ReadOnly(&b),
///         F32BufferBinding::ReadWrite(&out),
///     ];
///     device.dispatch_f32_buffers(&vector_add::add::DESCRIPTOR, 2, &bindings)?;
///     assert_eq!(device.read_f32_buffer(&out)?, [11.0, 22.0]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy)]
#[deprecated(since = "0.2.0", note = "use `BufferBinding`")]
pub enum F32BufferBinding<'a> {
    /// The buffer is bound read-only.
    ReadOnly(&'a GpuBuffer<f32>),
    /// The buffer is bound read-write.
    ReadWrite(&'a GpuBuffer<f32>),
}

#[allow(deprecated)]
impl<'a> F32BufferBinding<'a> {
    pub(crate) fn buffer(self) -> &'a GpuBuffer<f32> {
        match self {
            Self::ReadOnly(buffer) | Self::ReadWrite(buffer) => buffer,
        }
    }

    pub(crate) fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}

/// A checked view of a typed allocation for one heterogeneous storage binding.
/// Construct through `read_only` or `read_write`; the original element ABI is retained.
///
/// A small settings buffer, which has its own length, opts out of the length rule with `independent_length`:
///
/// ```
/// use gust::gpu_module;
/// use gust_wgpu::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice};
///
/// #[gpu_module]
/// mod scale {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn by(
///         id: SV_DispatchThreadID,
///         factor: StructuredBuffer<float>,
///         x: StructuredBuffer<float>,
///         mut out: RWStructuredBuffer<float>,
///     ) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = factor[0] * x[i];
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let factor = device.create_typed_buffer("factor", &[3.0f32], GpuBufferAccess::ReadOnly);
///     let x = device.create_typed_buffer("x", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadOnly);
///     let out = device.create_typed_buffer("out", &[0.0f32; 3], GpuBufferAccess::ReadWrite);
///     let bindings = [
///         BufferBinding::read_only(&factor).independent_length(),
///         BufferBinding::read_only(&x),
///         BufferBinding::read_write(&out),
///     ];
///     device.dispatch_batch(&[BufferDispatch::new(&scale::by::DESCRIPTOR, 3, &bindings)])?;
///     assert_eq!(device.read_typed_buffer(&out)?, [3.0, 6.0, 9.0]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy)]
pub struct BufferBinding<'a> {
    /// What the binding stands for in the aliasing rule and the staged graph's hazard check: its buffer, or its texture.
    pub(crate) raw: Raw<'a>,
    pub(crate) len: usize,
    pub(crate) layout: gust::TypeLayout,
    pub(crate) buffer_access: GpuBufferAccess,
    pub(crate) access: Access,
    pub(crate) device_id: u64,
    pub(crate) independent_length: bool,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) indirect: bool,
    /// Storage buffer or uniform; checked against the parameter's kind before anything else.
    pub(crate) kind: BindingKind,
    /// The bytes the bind group exposes, from offset 0: the logical length (at least one element, as for an empty
    /// buffer), which is the whole allocation unless the buffer is pooled and its class is larger.
    pub(crate) bytes: u64,
    /// A texture binding's own facts (`raw` is then the texture itself); `None` for a buffer.
    pub(crate) texture: Option<TextureBinding<'a>>,
}

/// The identity of what a [`BufferBinding`] binds, for the aliasing rule of a call and the staged graph's hazard
/// check (slice mx.p1b: this replaces the 4-byte identity buffer a texture carried since mx.t2). Two identities are
/// equal when they are the same wgpu allocation; a buffer is never a texture.
#[derive(Clone, Copy)]
pub(crate) enum Raw<'a> {
    Buffer(&'a wgpu::Buffer),
    Texture(&'a wgpu::Texture),
}

impl PartialEq for Raw<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Buffer(first), Self::Buffer(second)) => same_buffer(first, second),
            (Self::Texture(first), Self::Texture(second)) => first == second,
            _ => false,
        }
    }
}

impl Eq for Raw<'_> {}

/// Hashes what [`PartialEq`] compares (the wgpu allocation), so equal identities hash alike.
impl std::hash::Hash for Raw<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Buffer(buffer) => {
                0_u8.hash(state);
                buffer.hash(state);
            }
            Self::Texture(texture) => {
                1_u8.hash(state);
                texture.hash(state);
            }
        }
    }
}

impl<'a> BufferBinding<'a> {
    /// Bind `buffer` read-only.
    pub fn read_only<T: GpuPod>(buffer: &'a GpuBuffer<T>) -> Self {
        Self::new(buffer, Access::ReadOnly)
    }

    /// Bind `buffer` read-write.
    pub fn read_write<T: GpuPod>(buffer: &'a GpuBuffer<T>) -> Self {
        Self::new(buffer, Access::ReadWrite)
    }

    /// Allow this buffer's length to differ from the dispatch element count.
    ///
    /// The default rule requires every binding to hold exactly `element_count`
    /// elements so one `if i < out.len()` guard covers all of them. Opting out is
    /// for small settings buffers and reductions whose output is shorter than the
    /// input. The kernel must then guard every access to this buffer with its own
    /// `.len()`; the runtime cannot prove that. An empty buffer is rejected when the
    /// dispatch has work, because the GPU would observe a one-element placeholder.
    pub const fn independent_length(mut self) -> Self {
        self.independent_length = true;
        self
    }

    pub(crate) fn new<T: GpuPod>(buffer: &'a GpuBuffer<T>, access: Access) -> Self {
        Self {
            raw: Raw::Buffer(&buffer.raw),
            len: buffer.len,
            layout: buffer.layout,
            buffer_access: buffer.access,
            access,
            device_id: buffer.device_id,
            independent_length: false,
            indirect: buffer.indirect,
            kind: BindingKind::Storage,
            bytes: buffer.logical_bytes(),
            texture: None,
        }
    }

    /// A uniform binding: one element, and independent of the dispatch's element count by construction.
    /// `access` is `ReadOnly` for a kernel's parameter and `ReadWrite` for the copy node of a staged graph.
    pub(crate) fn new_uniform<T: GpuPod>(uniform: &'a GpuUniform<T>, access: Access) -> Self {
        Self {
            raw: Raw::Buffer(&uniform.raw),
            len: 1,
            layout: uniform.layout,
            buffer_access: GpuBufferAccess::ReadOnly,
            access,
            device_id: uniform.device_id,
            independent_length: true,
            indirect: false,
            kind: BindingKind::Uniform,
            bytes: uniform.raw.size(),
            texture: None,
        }
    }

    /// The bytes and kind the launch limits check this binding with: what the bind group exposes.
    pub(crate) fn limit_bytes(&self) -> (u64, BindingKind) {
        (self.bytes, self.kind)
    }

    /// The binding resource of the bind group: `bytes` from offset 0, so a pooled buffer larger than its logical
    /// length shows the kernel (`arrayLength`, `.len()`) the logical length only.
    pub(crate) fn resource(&self) -> wgpu::BindingResource<'a> {
        if let Some(texture) = self.texture {
            return wgpu::BindingResource::TextureView(texture.view);
        }
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer(),
            offset: 0,
            size: std::num::NonZeroU64::new(self.bytes),
        })
    }

    /// The wgpu buffer behind a buffer or uniform binding (a texture binding has none: it never reaches a copy, a
    /// readback or an indirect dispatch, which take buffers only).
    pub(crate) fn buffer(&self) -> &'a wgpu::Buffer {
        match self.raw {
            Raw::Buffer(buffer) => buffer,
            Raw::Texture(_) => unreachable!("a texture binding is never used as a buffer"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn byte_len(&self) -> u64 {
        self.len as u64 * u64::from(self.layout.stride)
    }
}

#[allow(deprecated)]
impl<'a> From<F32BufferBinding<'a>> for BufferBinding<'a> {
    fn from(binding: F32BufferBinding<'a>) -> Self {
        Self::new(binding.buffer(), binding.access())
    }
}
