#[cfg(doc)]
use crate::BufferBinding;
#[cfg(not(target_arch = "wasm32"))]
use gust::Access;

/// Host data for one reflected `Storage<f32>` or `StorageMut<f32>` parameter.
///
/// Deprecated with the v1 `dispatch_f32`, which uploads the data on every call and reads every read-write binding
/// back; new code creates a buffer once and binds it with [`BufferBinding`]:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu_module; use gust_wgpu::{F32Binding, HeadlessDevice};
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
///     let bindings = [
///         F32Binding::ReadOnly(&[1.0, 2.0]),
///         F32Binding::ReadOnly(&[10.0, 20.0]),
///         F32Binding::ReadWrite(&[0.0, 0.0]),
///     ];
///     let output = device.dispatch_f32(&vector_add::add::DESCRIPTOR, 2, &bindings)?;
///     assert_eq!(output.writable_buffers, [vec![11.0, 22.0]]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug)]
#[deprecated(
    since = "0.2.0",
    note = "use a `GpuBuffer<f32>` from `create_typed_buffer` with `BufferBinding`"
)]
pub enum F32Binding<'a> {
    /// A read-only storage buffer with these contents.
    ReadOnly(&'a [f32]),
    /// A read-write storage buffer initialised with these contents. Its final contents come back in the dispatch
    /// output.
    ReadWrite(&'a [f32]),
}

#[allow(deprecated)]
impl<'a> F32Binding<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn values(self) -> &'a [f32] {
        match self {
            Self::ReadOnly(values) | Self::ReadWrite(values) => values,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}

/// Host data for one reflected `Storage<u32>` or `StorageMut<u32>` parameter.
///
/// Deprecated with the v1 `dispatch_u32`; new code binds a `GpuBuffer<u32>` with [`BufferBinding`]:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu_module; use gust_wgpu::{HeadlessDevice, U32Binding};
///
/// #[gpu_module]
/// mod counting {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn add_index(id: SV_DispatchThreadID, mut data: RWStructuredBuffer<uint>) {
///         let i = id.x;
///         if i < data.len() {
///             data[i] = data[i] + i;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let output = device.dispatch_u32(&counting::add_index::DESCRIPTOR, 3, &[U32Binding::ReadWrite(&[10, 10, 10])])?;
///     assert_eq!(output.writable_buffers, [vec![10, 11, 12]]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug)]
#[deprecated(
    since = "0.2.0",
    note = "use a `GpuBuffer<u32>` from `create_typed_buffer` with `BufferBinding`"
)]
pub enum U32Binding<'a> {
    /// A read-only storage buffer with these contents.
    ReadOnly(&'a [u32]),
    /// A read-write storage buffer initialised with these contents. Its final contents come back in the dispatch
    /// output.
    ReadWrite(&'a [u32]),
}

#[allow(deprecated)]
impl<'a> U32Binding<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn values(self) -> &'a [u32] {
        match self {
            Self::ReadOnly(values) | Self::ReadWrite(values) => values,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}
