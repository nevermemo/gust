#[cfg(doc)]
use crate::Error;
#[allow(deprecated)]
use crate::{BufferBinding, F32BufferBinding, GpuSamplerBinding};
use gust::{Extent, KernelDescriptor};

/// One persistent dispatch whose bindings can have different POD element types.
///
/// The extent is `element_count` along x and, along y and z, the two axes fixed by the constructor (slice m7.d):
/// [`BufferDispatch::extent`] reads it and [`BufferDispatch::set_extent`] replaces it. There is one source of truth,
/// so `dispatch.element_count = n` on a dispatch from [`BufferDispatch::new`] dispatches `n` invocations, as it always
/// did. The staged graph runs one-dimensional dispatches only (a node with a y or z extent above 1 is refused with
/// [`Error::Extent`] when the graph runs), and reads `element_count`.
///
/// ```
/// use gust::{Extent, gpu};
/// use gust_wgpu::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice};
///
/// #[gpu]
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
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadWrite);
///     let bindings = [BufferBinding::read_write(&out)];
///     let mut dispatch = BufferDispatch::new(&double::run::DESCRIPTOR, 3, &bindings);
///     assert_eq!(dispatch.extent(), Extent::linear(3));
///
///     // Batches run in order inside one submission.
///     device.dispatch_batch(&[dispatch, dispatch])?;
///     assert_eq!(device.read_typed_buffer(&out)?, [4.0, 8.0, 12.0]);
///
///     dispatch.set_extent(Extent::linear(1));
///     assert_eq!(dispatch.element_count, 1);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy)]
pub struct BufferDispatch<'a> {
    /// The kernel to run.
    pub kernel: &'a KernelDescriptor,
    /// The x axis of the extent; the whole extent of a one-dimensional dispatch.
    pub element_count: u32,
    /// The y and z axes of the extent.
    extent_yz: [u32; 2],
    /// The kernel's resource arguments, in parameter order.
    pub bindings: &'a [BufferBinding<'a>],
    /// The sampler bindings of a kernel with `Sampler` parameters, in declaration order; empty for a kernel with none.
    /// Set with [`BufferDispatch::with_samplers`].
    pub(crate) samplers: &'a [GpuSamplerBinding<'a>],
}

impl std::fmt::Debug for BufferDispatch<'_> {
    /// The kernel's name, the extent and the number of bindings.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferDispatch")
            .field("kernel", &self.kernel.name)
            .field("element_count", &self.element_count)
            .field("extent_yz", &self.extent_yz)
            .field("bindings", &self.bindings.len())
            .field("samplers", &self.samplers.len())
            .finish_non_exhaustive()
    }
}

impl<'a> BufferDispatch<'a> {
    /// A one-dimensional dispatch of `element_count` invocations: the extent `element_count x 1 x 1`.
    pub const fn new(
        kernel: &'a KernelDescriptor,
        element_count: u32,
        bindings: &'a [BufferBinding<'a>],
    ) -> Self {
        Self::new_extent(kernel, Extent::linear(element_count), bindings)
    }

    /// A dispatch over an [`Extent`]: `x * y * z` invocations before every axis is rounded up to whole workgroups.
    pub const fn new_extent(
        kernel: &'a KernelDescriptor,
        extent: Extent,
        bindings: &'a [BufferBinding<'a>],
    ) -> Self {
        Self {
            kernel,
            element_count: extent.x,
            extent_yz: [extent.y, extent.z],
            bindings,
            samplers: &[],
        }
    }

    /// The same dispatch with `samplers` as the bindings of the kernel's `Sampler` parameters, one per parameter in
    /// declaration order (slice mx.t2). A kernel with a sampler parameter and no sampler binding is
    /// [`Error::BindingCount`] at submit.
    #[must_use]
    pub const fn with_samplers(mut self, samplers: &'a [GpuSamplerBinding<'a>]) -> Self {
        self.samplers = samplers;
        self
    }

    /// The sampler bindings of this dispatch.
    pub const fn samplers(&self) -> &'a [GpuSamplerBinding<'a>] {
        self.samplers
    }

    /// The extent the batch functions dispatch: `element_count` by the two other axes.
    pub const fn extent(&self) -> Extent {
        Extent::new(self.element_count, self.extent_yz[0], self.extent_yz[1])
    }

    /// Replaces the whole extent, `element_count` with it.
    pub const fn set_extent(&mut self, extent: Extent) {
        self.element_count = extent.x;
        self.extent_yz = [extent.y, extent.z];
    }
}

/// One persistent-buffer dispatch to encode as part of a batch.
///
/// Deprecated; [`BufferDispatch`] takes bindings of any element type:
///
/// ```
/// # #![allow(deprecated)]
/// use gust::gpu;
/// use gust_wgpu::{F32BufferBinding, F32BufferDispatch, GpuBufferAccess, HeadlessDevice};
///
/// #[gpu]
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
///     let out = device.create_f32_buffer("out", &[1.0, 2.0], GpuBufferAccess::ReadWrite);
///     let bindings = [F32BufferBinding::ReadWrite(&out)];
///     let dispatch = F32BufferDispatch::new(&double::run::DESCRIPTOR, 2, &bindings);
///     device.dispatch_f32_batch(&[dispatch, dispatch])?;
///     assert_eq!(device.read_f32_buffer(&out)?, [4.0, 8.0]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy)]
#[deprecated(since = "0.2.0", note = "use `BufferDispatch`")]
#[allow(deprecated)]
pub struct F32BufferDispatch<'a> {
    /// The kernel to run.
    pub kernel: &'a KernelDescriptor,
    /// The element count: the extent of this one-dimensional dispatch.
    pub element_count: u32,
    /// The kernel's buffer arguments, in parameter order.
    pub bindings: &'a [F32BufferBinding<'a>],
}

#[allow(deprecated)]
impl<'a> F32BufferDispatch<'a> {
    /// A dispatch of `kernel` over `element_count` elements with `bindings`.
    pub const fn new(
        kernel: &'a KernelDescriptor,
        element_count: u32,
        bindings: &'a [F32BufferBinding<'a>],
    ) -> Self {
        Self {
            kernel,
            element_count,
            bindings,
        }
    }
}
