use crate::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferView, GpuSamplerBinding, GpuUniform,
    HeadlessDevice, WgpuJob,
};
use gust::target::GpuKernel;
use gust::{Access, Call, GpuPod, LengthRule, Resource, Run, TypedDevice};

impl gust::Device for HeadlessDevice {
    type Error = Error;
    type Dispatch<'resources> = BufferDispatch<'resources>;
    type Job<'device>
        = WgpuJob<'device>
    where
        Self: 'device;

    fn submit<'device, 'resources>(
        &'device self,
        dispatches: &[Self::Dispatch<'resources>],
    ) -> Result<Self::Job<'device>, Self::Error> {
        self.submit_batch(dispatches)
    }
}

/// Whether two handles are the same wgpu buffer. `wgpu::Buffer` equality compares the underlying
/// allocation, not the address of the handle, so a cloned handle is the same buffer. Buffer identity
/// in the typed-call alias rule and in the staged graph (hazard check and edge derivation) is this comparison.
pub(crate) fn same_buffer(first: &wgpu::Buffer, second: &wgpu::Buffer) -> bool {
    first == second
}

impl TypedDevice for HeadlessDevice {
    type Binding<'resources> = BufferBinding<'resources>;
}

/// One blanket implementation for every kernel type that this build runs on a GPU (the hidden marker
/// `gust::target::GpuKernel`, D137, D145: the macro implements it for the kernels it builds for the GPU target, and a
/// hand-written `Kernel` adds the line `impl gust::target::GpuKernel for MyKernel {}`). A kernel routed to the CPU has
/// no GPU form, so naming this device for it does not compile, and the message names the kernel. A uniform parameter
/// takes a [`GpuUniform`]; a parameter of
/// any other non-storage kind (`Value`) is rejected with `Error::UnsupportedParameter`. The
/// call goes through `submit_batch`, so
/// `validate_persistent_dispatch` checks each binding against `call.kernel()` (access, layout, device
/// instance, length, aliasing), then the launch limits and the pipeline cache take over, exactly as for
/// `dispatch_buffers`. A hand-built `Call` is therefore never trusted.
impl<K: GpuKernel, const N: usize> Run<K, N> for HeadlessDevice {
    fn submit_call<'device>(
        &'device self,
        call: &Call<'_, Self, K, N>,
    ) -> Result<Self::Job<'device>, Self::Error> {
        self.submit_batch(&[BufferDispatch::from(call)])
    }
}

impl<'call, 'resources: 'call, K: GpuKernel, const N: usize>
    From<&'call Call<'resources, HeadlessDevice, K, N>> for BufferDispatch<'call>
{
    /// Lets typed calls of different kernels share one `submit_batch`/`dispatch_batch`:
    /// `device.dispatch_batch(&[(&first).into(), (&second).into()])`. The call must outlive the batch.
    /// `Call` is invariant over the lifetime of the buffers it borrows, so the two lifetimes stay separate:
    /// the dispatch borrows the call only for `'call`.
    fn from(call: &'call Call<'resources, HeadlessDevice, K, N>) -> Self {
        Self::new_extent(call.kernel(), call.extent(), call.bindings())
    }
}

/// The calls of kernels with sampler parameters (decision D64, slice mx.t2): `Run<K, N, [GpuSamplerBinding; S]>`, beside
/// `Run<K, N>`. The sampler bindings travel in the call (`SampledCall`) and are checked at submit with the others.
impl<'sampler, K: GpuKernel, const N: usize, const S: usize>
    Run<K, N, [GpuSamplerBinding<'sampler>; S]> for HeadlessDevice
{
    fn submit_call<'device>(
        &'device self,
        call: &Call<'_, Self, K, N, [GpuSamplerBinding<'sampler>; S]>,
    ) -> Result<Self::Job<'device>, Self::Error> {
        self.submit_batch(&[BufferDispatch::from(call)])
    }
}

impl<'call, 'resources: 'call, 'sampler: 'call, K: GpuKernel, const N: usize, const S: usize>
    From<&'call Call<'resources, HeadlessDevice, K, N, [GpuSamplerBinding<'sampler>; S]>>
    for BufferDispatch<'call>
{
    /// The dispatch of a call that carries sampler bindings (the other `From<&Call>` impl is the one without).
    fn from(
        call: &'call Call<'resources, HeadlessDevice, K, N, [GpuSamplerBinding<'sampler>; S]>,
    ) -> Self {
        Self::new_extent(call.kernel(), call.extent(), call.bindings())
            .with_samplers(call.samplers())
    }
}

/// A persistent buffer is a typed resource of [`HeadlessDevice`]. In a generated `call` its element
/// type is checked at compile time: a `GpuBuffer<u32>` where the kernel wants `f32` does not compile
/// (rustc error E0271, reproduced in a scratch crate, 2026-10-24).
///
/// ```compile_fail
/// # use gust::*;
/// # use gust_wgpu::GpuBuffer;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, R0::Device, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// #     R1: Resource<Element = f32, Device = R0::Device> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// fn wrong(a: &GpuBuffer<f32>, b: &GpuBuffer<u32>) {
///     let _ = call(1, a, b);
/// }
/// ```
///
/// The control differs in exactly one place (the element type of `b`):
///
/// ```
/// # use gust::*;
/// # use gust_wgpu::GpuBuffer;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, R0::Device, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// #     R1: Resource<Element = f32, Device = R0::Device> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// fn right(a: &GpuBuffer<f32>, b: &GpuBuffer<f32>) {
///     let _ = call(1, a, b);
/// }
/// ```
impl<T: GpuPod> Resource for GpuBuffer<T> {
    type Element = T;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Storage;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        length: LengthRule,
    ) -> BufferBinding<'resources> {
        let binding = BufferBinding::new(self, access);
        match length {
            LengthRule::Independent => binding.independent_length(),
            // `Shared` and any rule added later: the strictest rule (exactly `element_count`
            // elements) is the safe reading of one this crate does not know.
            _ => binding,
        }
    }
}

/// A [`GpuBufferView`] is a typed resource of [`HeadlessDevice`] wherever a `GpuBuffer<U>` is: its element type is `U`,
/// checked at compile time like a buffer's (a `u32` view where the kernel wants `f32` does not compile), and its
/// binding carries the view's length, so the shared length rule applies to `len()` of the view.
///
/// ```compile_fail,E0271
/// # use gust::*;
/// # use gust_wgpu::GpuBuffer;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadWrite)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0>(n: u32, a: impl Into<Arg<'r, R0>>) -> Call<'r, R0::Device, Probe, 1>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadWrite)])
/// # }
/// fn wrong(words: &GpuBuffer<f32>) -> Result<(), gust_wgpu::Error> {
///     let as_words = words.view_as::<u32>()?;
///     let _ = call(1, &as_words);
///     Ok(())
/// }
/// ```
///
/// The control differs in exactly one place (the view's element type):
///
/// ```
/// # use gust::*;
/// # use gust_wgpu::GpuBuffer;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadWrite)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0>(n: u32, a: impl Into<Arg<'r, R0>>) -> Call<'r, R0::Device, Probe, 1>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadWrite)])
/// # }
/// fn right(words: &GpuBuffer<f32>) -> Result<(), gust_wgpu::Error> {
///     let as_floats = words.view_as::<f32>()?;
///     let _ = call(1, &as_floats);
///     Ok(())
/// }
/// ```
impl<U: GpuPod> Resource for GpuBufferView<'_, U> {
    type Element = U;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Storage;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        length: LengthRule,
    ) -> BufferBinding<'resources> {
        let binding = BufferBinding::new_view(self, access);
        match length {
            LengthRule::Independent => binding.independent_length(),
            // `Shared` and any rule added later: the strictest rule, as for a `GpuBuffer`.
            _ => binding,
        }
    }
}

/// A uniform is a typed resource of [`HeadlessDevice`] of kind [`gust::kind::Uniform`]. In a generated `call` it is accepted
/// only for a uniform parameter: it is read-only, one element long whatever the dispatch's element count (the length
/// rule is ignored), and a `GpuUniform` where a storage buffer is expected does not compile.
///
/// ```compile_fail,E0271
/// # use gust::*;
/// # use gust_wgpu::GpuUniform;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadOnly)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0>(n: u32, a: impl Into<Arg<'r, R0>>) -> Call<'r, R0::Device, Probe, 1>
/// # where
/// #     R0: Resource<Element = f32, Kind = kind::Storage> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly)])
/// # }
/// fn wrong(a: &GpuUniform<f32>) {
///     let _ = call(1, a);
/// }
/// ```
///
/// The control differs in exactly one place (a buffer instead of a uniform):
///
/// ```
/// # use gust::*;
/// # use gust_wgpu::GpuBuffer;
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "probe", "m_probe", [1, 1, 1], &[p("a", Access::ReadOnly)], "");
/// # }
/// # impl gust::target::GpuKernel for Probe {}
/// # fn call<'r, R0>(n: u32, a: impl Into<Arg<'r, R0>>) -> Call<'r, R0::Device, Probe, 1>
/// # where
/// #     R0: Resource<Element = f32, Kind = kind::Storage> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly)])
/// # }
/// fn right(a: &GpuBuffer<f32>) {
///     let _ = call(1, a);
/// }
/// ```
impl<T: GpuPod> Resource for GpuUniform<T> {
    type Element = T;
    type Device = HeadlessDevice;
    type Kind = gust::kind::Uniform;

    fn bind<'resources>(
        &'resources self,
        access: Access,
        _length: LengthRule,
    ) -> BufferBinding<'resources> {
        BufferBinding::new_uniform(self, access)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuBufferAccess, HeadlessDevice};

    #[test]
    fn same_buffer_compares_the_allocation_not_the_handle_address() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let first =
            device.create_typed_buffer("identity first", &[0.0], GpuBufferAccess::ReadWrite);
        let second =
            device.create_typed_buffer("identity second", &[0.0], GpuBufferAccess::ReadWrite);
        let cloned = first.raw.clone();
        assert!(
            !std::ptr::eq(&first.raw, &cloned),
            "the clone is another handle, at another address"
        );
        assert!(
            same_buffer(&first.raw, &cloned),
            "a cloned handle is the same buffer"
        );
        assert!(same_buffer(&first.raw, &first.raw));
        assert!(!same_buffer(&first.raw, &second.raw));
    }
}
