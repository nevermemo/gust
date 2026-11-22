#[cfg(doc)]
#[allow(deprecated)]
use crate::{F32Binding, U32Binding};
/// Result of a completed dispatch. Writable buffers are returned in binding order.
///
/// `dispatch_f32` returns one (see [`F32Binding`]); it is deprecated in favour of `read_typed_buffer`:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu_module;
/// use gust_wgpu::{F32Binding, HeadlessDevice};
///
/// #[gpu_module]
/// mod double {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let output = device.dispatch_f32(&double::run::DESCRIPTOR, 2, &[F32Binding::ReadWrite(&[1.0, 2.0])])?;
///     // One entry per read-write binding, in binding order.
///     assert_eq!(output.writable_buffers, vec![vec![2.0, 4.0]]);
///     Ok(())
/// }
/// ```
#[derive(Debug, PartialEq)]
#[deprecated(
    since = "0.2.0",
    note = "read the buffers back with `read_typed_buffer`"
)]
pub struct DispatchOutput {
    /// The final contents of the writable buffers, in binding order.
    pub writable_buffers: Vec<Vec<f32>>,
}

/// Result of a completed `u32` dispatch. Writable buffers are returned in binding order.
///
/// `dispatch_u32` returns one (see [`U32Binding`]); it is deprecated in favour of `read_typed_buffer`:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu_module;
/// use gust_wgpu::{HeadlessDevice, U32Binding};
///
/// #[gpu_module]
/// mod bump {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(id: SV_DispatchThreadID, mut data: RWStructuredBuffer<uint>) {
///         let i = id.x;
///         if i < data.len() {
///             data[i] = data[i] + 1;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let output = device.dispatch_u32(&bump::run::DESCRIPTOR, 3, &[U32Binding::ReadWrite(&[1, 2, 3])])?;
///     assert_eq!(output.writable_buffers, vec![vec![2, 3, 4]]);
///     Ok(())
/// }
/// ```
#[derive(Debug, PartialEq)]
#[deprecated(
    since = "0.2.0",
    note = "read the buffers back with `read_typed_buffer`"
)]
pub struct U32DispatchOutput {
    /// The final contents of the writable buffers, in binding order.
    pub writable_buffers: Vec<Vec<u32>>,
}
