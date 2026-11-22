#[cfg(doc)]
use crate::{GpuBuffer, HeadlessDevice};
#[cfg(doc)]
use gust::GpuPod;
use std::marker::PhantomData;

/// A device-owned uniform: one `T` in a constant buffer, bound to a `Uniform` parameter of a kernel.
///
/// `T` is a padding-free [`GpuPod`] type with a `Std140V1` layout ([`HeadlessDevice::create_uniform`] refuses any
/// other and names the fix). It is a [`gust::Resource`] of kind [`gust::kind::Uniform`], so a typed call takes it
/// only where the kernel declares a uniform, and a [`GpuBuffer`] only where it declares a storage buffer; swapping
/// them does not compile. A uniform is always read-only for the kernel and has length one whatever the dispatch
/// element count.
///
/// Update it with [`HeadlessDevice::write_uniform`] (a `Queue::write_buffer`: ordered after every earlier submission
/// and before the next one, so every dispatch of one later submission, a `submit_batch` or a graph, sees the last
/// value written) or, between the dispatches of one submission, with `StagedGraph::write_uniform`, which encodes a
/// copy in the command stream.
///
/// ```
/// use gust::gpu_module; use gust_wgpu::{GpuBufferAccess, GpuUniform, HeadlessDevice};
///
/// #[gpu_module]
/// mod scale {
///     /// A uniform is a struct with no implicit padding and a size that is a multiple of 16 bytes.
///     pub struct Params {
///         pub factor: f32,
///         pub pad0: f32,
///         pub pad1: f32,
///         pub pad2: f32,
///     }
///
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn apply(id: SV_DispatchThreadID, p: Uniform<Params>, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * p.factor;
///         }
///     }
/// }
///
/// fn params(factor: f32) -> scale::Params {
///     scale::Params { factor, pad0: 0.0, pad1: 0.0, pad2: 0.0 }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0], GpuBufferAccess::ReadWrite);
///     let uniform: GpuUniform<scale::Params> = device.create_uniform("params", &params(3.0))?;
///     assert_eq!(uniform.layout().size, 16);
///
///     scale::apply::dispatch(&device, 2, &uniform, &out)?;
///     device.write_uniform(&uniform, &params(10.0))?;
///     scale::apply::dispatch(&device, 2, &uniform, &out)?;
///     assert_eq!(device.read_typed_buffer(&out)?, [30.0, 60.0]);
///     Ok(())
/// }
/// ```
pub struct GpuUniform<T> {
    pub(crate) raw: wgpu::Buffer,
    pub(crate) layout: gust::TypeLayout,
    pub(crate) device_id: u64,
    pub(crate) marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for GpuUniform<T> {
    /// The owning device; never the raw wgpu handle.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuUniform")
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl<T> GpuUniform<T> {
    /// The layout of the uniform's value.
    pub const fn layout(&self) -> gust::TypeLayout {
        self.layout
    }

    /// Expose the underlying allocation for advanced wgpu code.
    pub fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.raw
    }
}
