//! Headless Vulkan/wgpu execution for GUST kernel descriptors.
//!
//! Storage buffers support 32-bit scalars and nested GPU POD structs. A uniform parameter is bound to a
//! [`GpuUniform`]: one padding-free `Std140V1` value in a constant buffer (see there).
//! It reflects the descriptor's group-0 bindings, uploads the supplied data,
//! takes the WGSL of a precompiled artifact or compiles descriptor Slang source to WGSL, dispatches it, and reads
//! writable buffers back. A persistent typed-buffer path separates allocation, upload, dispatch, and
//! readback so allocations can remain resident across repeated work.
//!
//! # Where a kernel's shader comes from
//!
//! A kernel is dispatched in this order of preference (D45):
//!
//! 1. The first WGSL artifact of `KernelDescriptor::artifacts`. It is checked with `gust::artifact::verify_artifact`
//!    (D17: identity, content hashes, reflection against the descriptor's layouts) before a shader module is created.
//!    A check that fails is an error (`Error::Reflection`), never a reason to compile the Slang instead.
//! 2. With the `slang-runtime` feature (off by default, D44, which enables `gust/slang-runtime`): compile the
//!    descriptor's Slang at run time through the native helper.
//! 3. Otherwise `Error::NoArtifact`: the crate holds no run-time Slang bridge. `cargo gust build` writes the
//!    artifact.
//!
//! The backend-neutral typed surface of `gust` is implemented here: `GpuBuffer<T>` is a
//! `gust::Resource`, and `HeadlessDevice` runs every `gust::Call` (`gust::Run`). A call is
//! validated like `dispatch_buffers`: each binding is checked against `call.kernel()` at submit,
//! and one buffer may be bound twice only read-only (`Error::AliasedBinding`).
//!
//! # Blocking and async readback
//!
//! `read_typed_buffer_async` and `HeadlessDevice::new_async` are the async core (R4.9); a std-only `Completion` future
//! sits above wgpu's `map_async` callback. On native the blocking functions (`new`, `read_typed_buffer`) keep their
//! names and wrap the async functions (`new` with `pollster::block_on`); a readback wrapper waits with `poll(Wait)`
//! first and then waits for the mapping's own completion (it parks until the callback wakes it, polls the device
//! again every 10 ms, and gives up with `Error::Map` after 30 s). On native the callback runs inside a device poll,
//! and the poll that runs it may be another thread's, so one poll and one check would be a race. They do not exist on
//! `wasm32`.
//!
//! # The `wasm32` surface
//!
//! In the browser (`wasm32-unknown-unknown`, wgpu's `webgpu` backend) nothing may block or read a clock, so the
//! crate offers only what can run there: `new_async`, the buffer, upload and typed `submit` functions, and
//! `read_typed_buffer_async`. These are compiled out on `wasm32`: the blocking readbacks, the `_timed` functions
//! (`std::time::Instant` panics in the browser), `dispatch_f32`/`dispatch_u32`, the blocking `dispatch_batch`,
//! `dispatch_buffers`, `dispatch_f32_batch` and `dispatch_f32_buffers`, the timestamp readback (one-shot channels), the
//! `graph` module and `HeadlessDevice::pool_read`. A call to one of them fails to compile.
//!
//! A [`WgpuJob`] cannot block there either: its `status` reports `Pending` until wgpu's `on_submitted_work_done`
//! callback has run and `Complete` after it, and `wait` and `wait_timeout` return [`Error::WouldBlock`]. The methods
//! that wait through the shared traits are not compiled out, because `gust::cpu::CpuDevice` runs them fine on
//! `wasm32`. On this device the generated `dispatch` and `dispatch_extent`, `gust::Run::dispatch_call` and
//! `gust::Device::dispatch` submit their work, which will run, and then return that error at run time. Use `submit`,
//! then an awaited readback to synchronise.
//!
//! `clippy.toml` bans `std::thread::sleep` and `std::time::Instant::now` for the browser; native code may call them,
//! hence the crate-level allow below.
//!
//! # Stability
//!
//! This crate is pre-1.0 and follows SemVer for `0.x`: a breaking change bumps the minor version (`0.1` to `0.2`),
//! and a patch release (`0.1.0` to `0.1.1`) only adds or fixes. The four crates of a release (`gust-compute`, the `#[gpu]`
//! macro crate, this crate and `cargo-gust`) share one version and depend on each other with exact (`=`) pins: take
//! them at the same version, and a minor bump of one is a minor bump of all.
//!
//! What a minor bump may change: any public item that is not support surface, including a function's signature and
//! the public fields of the plain record types (`BufferDispatch`, `DeviceOptions`, `PoolStats` and the like).
//! Every public enum is `#[non_exhaustive]` ([`Error`], `AdapterChoice`, `LimitsPolicy` and `FeaturesPolicy` among
//! them) except `TransferDirection`, `GpuBufferAccess` and the binding enums (`F32Binding`, `U32Binding` and
//! `F32BufferBinding`), which are closed sets: a new variant of the others is not a breaking change and a `match`
//! outside this crate needs a wildcard arm; the variants themselves are not, so adding a field to a struct variant is a
//! breaking change and comes only in a minor bump. A `GraphReport` is `#[non_exhaustive]` and only this crate builds
//! one.
//!
//! # Support surface
//!
//! `render_wgpu_source` is public because the repository's export command and tests call it. It is `#[doc(hidden)]`;
//! it has no stability promise and a patch release may change it. The artifact and reflection formats that this
//! crate reads are those of `gust` and keep its promise: a reader accepts every schema version that an earlier release
//! wrote.

#![warn(missing_docs)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};
#[cfg(not(target_arch = "wasm32"))]
use std::{sync::mpsc, time::Instant};

use gust::{
    Access, BindingKind, Call, Extent, GpuPod, JobStatus, KernelDescriptor, LengthRule,
    ParameterDescriptor, ParameterKind, Resource, ResourceLayout, Run, Shape, TypedDevice,
    capability::Capability,
    pod_slice_as_bytes,
    target::{GpuKernel, Target},
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

mod cache;
mod completion;
mod device;
mod error;
mod generated_host;
#[cfg(not(target_arch = "wasm32"))]
pub mod graph;
mod map_host;
pub mod pool;
mod render;
mod texture;
mod vec;
pub use cache::PipelineCacheStats;
use cache::{CachedKernel, KernelCacheKey, Pipeline, PipelineCache};
use completion::Completion;
#[cfg(not(target_arch = "wasm32"))]
use completion::blocking::{block_on_mapping, recv_mapping};
pub use device::{
    AdapterChoice, DeviceHandle, DeviceOptions, FeaturesPolicy, LimitsPolicy, default_device,
    default_device_async, set_default_options,
};
pub use error::Error;
#[doc(hidden)]
pub use generated_host::render_wgpu_source;
#[cfg(not(target_arch = "wasm32"))]
pub use graph::{GraphOutput, GraphReport, NodeId, StagedGraph, TransferDirection, TransferRecord};
pub use pool::{BufferPool, GpuPool, GrowthRecord, PoolStats};
pub use texture::{GpuSampler, GpuSamplerBinding, GpuTexture2d};
use texture::{TextureBinding, TextureCounts};
pub use vec::GpuVec;

// A build holds one target (D21): this crate is the GPU runtime, so a build under `GUST_PROFILE=cpu` stops here with
// a compile error (E0080) that names the profile. Under `dev`, `native` and `web` this does nothing.
const _: () = ::gust::profile::require_gpu_target();

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// wgpu's default `max_compute_workgroups_per_dimension`: the most workgroups one dispatch may
/// launch along one axis on a device opened with the default limits, which is what
/// `HeadlessDevice` requests. A single-dimension dispatch of a kernel with workgroup width `w`
/// therefore covers at most `w * DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION` elements; more
/// is an `Error::DeviceLimit` (a multi-dimensional extent spreads it over y and z). Pinned to
/// `wgpu::Limits::default()` by a unit test and to the opened device by the `limits` test.
pub const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Whether `layout` has no padding byte anywhere: a struct's field sizes add up to its size (recursively), a matrix's
/// columns are packed. [`gust::TypeLayout::is_storage_v2`] accepts padded structs, which are not `GpuPod` types:
/// `pod_slice_as_bytes` would read their uninitialised padding.
const fn padding_free(layout: gust::TypeLayout) -> bool {
    match layout.kind {
        gust::TypeLayoutKind::Scalar(_) | gust::TypeLayoutKind::Vector { .. } => true,
        gust::TypeLayoutKind::Matrix {
            rows,
            cols,
            column_stride,
            ..
        } => column_stride == rows * 4 && column_stride * cols == layout.size,
        gust::TypeLayoutKind::Struct(fields) => {
            let mut sum = 0_u32;
            let mut index = 0;
            while index < fields.len() {
                let field = fields[index].ty;
                let Some(next) = sum.checked_add(field.size) else {
                    return false;
                };
                if !padding_free(field) {
                    return false;
                }
                sum = next;
                index += 1;
            }
            sum == layout.size
        }
        // A kind this release does not know cannot be shown padding-free, so it is not a `GpuPod` layout.
        _ => false,
    }
}

/// Panic unless `T`'s `GpuPod::LAYOUT` is StorageV2 (StorageV1 is its scalar sub-case), has no padding, and agrees
/// with Rust's size and alignment. The message keeps the StorageV1 wording that tests pin.
/// Every typed allocation (`create_typed_buffer`, pool buffers) runs it first, so a hand-written
/// `GpuPod` that lies, or that describes padding, is refused at creation instead of failing later in wgpu or at
/// dispatch (or uploading uninitialised bytes).
pub(crate) fn assert_pod_matches_storage_layout<T: GpuPod>() {
    assert!(
        T::LAYOUT.is_storage_v2()
            && padding_free(T::LAYOUT)
            && T::LAYOUT.size as usize == size_of::<T>()
            && T::LAYOUT.alignment as usize == align_of::<T>(),
        "GpuPod implementation must match the Rust StorageV1 representation \
         (StorageV2 for vectors and matrices, without padding)"
    );
}

/// Whether `T` can be a uniform: a `GpuPod` type whose layout is `Std140V1`, agrees with Rust's size and alignment, and
/// has no padding byte. The padding rule is the one that keeps `pod_slice_as_bytes` sound: a `Std140V1` struct is
/// rounded up to 16 bytes, so a struct whose fields do not already fill a multiple of 16 has trailing padding that
/// Rust leaves uninitialised, and the upload would read it. The remedy is in the message: spell the padding out as
/// fields. (Copying the value into a zeroed buffer and skipping the gaps would also be sound; it is not done, so that
/// the bytes on the GPU are exactly the bytes of the Rust value.)
pub(crate) fn check_uniform_type<T: GpuPod>() -> Result<(), Error> {
    let layout = T::LAYOUT;
    let refuse = |reason: &'static str| {
        Err(Error::InvalidUniformType {
            type_name: layout.name,
            reason,
        })
    };
    if !layout.is_std140_v1() {
        return refuse(
            "its layout is not Std140V1 (a constant buffer takes scalars, 2- and 4-wide vectors, 4x4 matrices and \
             structs of them, each struct rounded up to 16 bytes; 3-wide vectors and 2x2 matrices do not qualify)",
        );
    }
    if layout.size as usize != size_of::<T>() || layout.alignment as usize != align_of::<T>() {
        return refuse(
            "its Std140V1 size or alignment differs from the Rust type's; a struct is rounded up to 16 bytes in a \
             constant buffer, so add explicit padding fields (and `repr(C, align(16))` if needed) until the Rust size \
             is the Std140V1 size",
        );
    }
    if !padding_free(layout) {
        return refuse(
            "it has padding bytes, which Rust leaves uninitialised; add explicit padding fields so that the fields \
             fill the whole struct (a Std140V1 struct is rounded up to 16 bytes)",
        );
    }
    Ok(())
}

/// Whether `layout` is exactly `wgpu::util::DispatchIndirectArgs`: three `u32`s.
fn is_indirect_args_layout(layout: gust::TypeLayout) -> bool {
    use gust::{ScalarKind, TypeLayoutKind};
    layout.size == 12
        && layout.stride == 12
        && matches!(layout.kind, TypeLayoutKind::Struct(fields) if fields.len() == 3
            && fields.iter().all(|field| matches!(field.ty.kind, TypeLayoutKind::Scalar(ScalarKind::U32))))
}

/// Host data for one reflected `Storage<f32>` or `StorageMut<f32>` parameter.
#[derive(Clone, Copy, Debug)]
pub enum F32Binding<'a> {
    /// A read-only storage buffer with these contents.
    ReadOnly(&'a [f32]),
    /// A read-write storage buffer initialised with these contents. Its final contents come back in the dispatch
    /// output.
    ReadWrite(&'a [f32]),
}

impl<'a> F32Binding<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    fn values(self) -> &'a [f32] {
        match self {
            Self::ReadOnly(values) | Self::ReadWrite(values) => values,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}

/// Host data for one reflected `Storage<u32>` or `StorageMut<u32>` parameter.
#[derive(Clone, Copy, Debug)]
pub enum U32Binding<'a> {
    /// A read-only storage buffer with these contents.
    ReadOnly(&'a [u32]),
    /// A read-write storage buffer initialised with these contents. Its final contents come back in the dispatch
    /// output.
    ReadWrite(&'a [u32]),
}

impl<'a> U32Binding<'a> {
    #[cfg(not(target_arch = "wasm32"))]
    fn values(self) -> &'a [u32] {
        match self {
            Self::ReadOnly(values) | Self::ReadWrite(values) => values,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}

/// Logical shader access assigned to a persistent GPU buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBufferAccess {
    /// Kernels only read the buffer, and the host cannot read it back.
    ReadOnly,
    /// Kernels may write the buffer, and the host can read it back.
    ReadWrite,
}

impl GpuBufferAccess {
    fn allows(self, access: Access) -> bool {
        matches!(
            (self, access),
            (Self::ReadOnly, Access::ReadOnly)
                | (Self::ReadWrite, Access::ReadOnly | Access::ReadWrite)
        )
    }
}

/// A device-owned, element-typed allocation that remains resident across dispatches.
pub struct GpuBuffer<T> {
    raw: wgpu::Buffer,
    len: usize,
    access: GpuBufferAccess,
    layout: gust::TypeLayout,
    device_id: u64,
    /// Created with `INDIRECT` usage by `create_indirect_buffer`.
    indirect: bool,
    /// The pool lease of a pooled buffer (`raw` may be larger than `len` elements; bindings use the logical
    /// length). Dropping the buffer returns the allocation to its device's pool.
    #[allow(dead_code)] // held for its `Drop`, which returns the allocation
    lease: Option<pool::Lease>,
    marker: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    /// The number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no element.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The access the buffer was created with.
    pub const fn access(&self) -> GpuBufferAccess {
        self.access
    }

    /// The layout of one element.
    pub const fn layout(&self) -> gust::TypeLayout {
        self.layout
    }

    /// Expose the underlying allocation for generated or advanced wgpu code.
    ///
    /// A pooled buffer (a [`GpuVec`]'s) is the full size class, not `len()` elements: its `arrayLength` and the bytes
    /// past the logical length are not the vector's (they may hold the previous owner's data). Bind it with
    /// `wgpu::BufferBinding { offset: 0, size: Some(len_bytes), .. }`; `as_entire_binding()` shows a kernel the whole
    /// class.
    ///
    /// The buffer returns to the pool when its owner drops, and the pool clears it for its next owner in a submission
    /// of its own. Commands you recorded in an encoder of your own and submit after the owner dropped may therefore
    /// run after that clear (or after the next owner's writes). Submit them before dropping the owner.
    pub fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.raw
    }
}

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
pub struct GpuUniform<T> {
    raw: wgpu::Buffer,
    layout: gust::TypeLayout,
    device_id: u64,
    marker: PhantomData<T>,
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

/// One persistent buffer bound to an `f32` storage parameter.
#[derive(Clone, Copy)]
pub enum F32BufferBinding<'a> {
    /// The buffer is bound read-only.
    ReadOnly(&'a GpuBuffer<f32>),
    /// The buffer is bound read-write.
    ReadWrite(&'a GpuBuffer<f32>),
}

impl<'a> F32BufferBinding<'a> {
    fn buffer(self) -> &'a GpuBuffer<f32> {
        match self {
            Self::ReadOnly(buffer) | Self::ReadWrite(buffer) => buffer,
        }
    }

    fn access(self) -> Access {
        match self {
            Self::ReadOnly(_) => Access::ReadOnly,
            Self::ReadWrite(_) => Access::ReadWrite,
        }
    }
}

/// A checked view of a typed allocation for one heterogeneous storage binding.
/// Construct through `read_only` or `read_write`; the original element ABI is retained.
#[derive(Clone, Copy)]
pub struct BufferBinding<'a> {
    /// What the binding stands for in the aliasing rule and the staged graph's hazard check: its buffer, or its texture.
    raw: Raw<'a>,
    len: usize,
    layout: gust::TypeLayout,
    buffer_access: GpuBufferAccess,
    access: Access,
    device_id: u64,
    independent_length: bool,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    indirect: bool,
    /// Storage buffer or uniform; checked against the parameter's kind before anything else.
    kind: BindingKind,
    /// The bytes the bind group exposes, from offset 0: the logical length (at least one element, as for an empty
    /// buffer), which is the whole allocation unless the buffer is pooled and its class is larger.
    bytes: u64,
    /// A texture binding's own facts (`raw` is then the texture itself); `None` for a buffer.
    texture: Option<TextureBinding<'a>>,
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

    fn new<T: GpuPod>(buffer: &'a GpuBuffer<T>, access: Access) -> Self {
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
            bytes: logical_bytes(buffer),
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
    fn resource(&self) -> wgpu::BindingResource<'a> {
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
    fn buffer(&self) -> &'a wgpu::Buffer {
        match self.raw {
            Raw::Buffer(buffer) => buffer,
            Raw::Texture(_) => unreachable!("a texture binding is never used as a buffer"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn byte_len(&self) -> u64 {
        self.len as u64 * u64::from(self.layout.stride)
    }
}

/// The bytes a binding of `buffer` exposes: `len` elements (one element for an empty buffer, whose allocation holds
/// a placeholder), capped by the allocation. For every buffer that is not pooled this is the whole allocation, which
/// is what `as_entire_binding` bound before; for a pooled buffer it is the logical length, not the class size.
fn logical_bytes<T>(buffer: &GpuBuffer<T>) -> u64 {
    let wanted = (buffer.len.max(1) as u64).saturating_mul(u64::from(buffer.layout.stride));
    wanted.min(buffer.raw.size())
}

impl<'a> From<F32BufferBinding<'a>> for BufferBinding<'a> {
    fn from(binding: F32BufferBinding<'a>) -> Self {
        Self::new(binding.buffer(), binding.access())
    }
}

/// One persistent dispatch whose bindings can have different POD element types.
///
/// The extent is `element_count` along x and, along y and z, the two axes fixed by the constructor (slice m7.d):
/// [`BufferDispatch::extent`] reads it and [`BufferDispatch::set_extent`] replaces it. There is one source of truth,
/// so `dispatch.element_count = n` on a dispatch from [`BufferDispatch::new`] dispatches `n` invocations, as it always
/// did. The staged graph runs one-dimensional dispatches only (a node with a y or z extent above 1 is refused with
/// [`Error::Extent`] when the graph runs), and reads `element_count`.
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
    samplers: &'a [GpuSamplerBinding<'a>],
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
#[derive(Clone, Copy)]
pub struct F32BufferDispatch<'a> {
    /// The kernel to run.
    pub kernel: &'a KernelDescriptor,
    /// The element count: the extent of this one-dimensional dispatch.
    pub element_count: u32,
    /// The kernel's buffer arguments, in parameter order.
    pub bindings: &'a [F32BufferBinding<'a>],
}

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

/// Result of a completed dispatch. Writable buffers are returned in binding order.
#[derive(Debug, PartialEq)]
pub struct DispatchOutput {
    /// The final contents of the writable buffers, in binding order.
    pub writable_buffers: Vec<Vec<f32>>,
}

/// Result of a completed `u32` dispatch. Writable buffers are returned in binding order.
#[derive(Debug, PartialEq)]
pub struct U32DispatchOutput {
    /// The final contents of the writable buffers, in binding order.
    pub writable_buffers: Vec<Vec<u32>>,
}

/// Host-observed time for an explicit transfer boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferTiming {
    /// Number of application bytes transferred.
    pub bytes: u64,
    /// Wall-clock time including encoding, submission, synchronization, and host copies.
    pub elapsed: Duration,
}

/// Host and device measurements for one synchronous persistent submission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchTiming {
    /// Wall-clock time including bind-group/command encoding, submission, and synchronization.
    pub host_elapsed: Duration,
    /// Hardware timestamp duration around the submitted compute work, when supported.
    pub gpu_elapsed: Option<Duration>,
}

/// Fence for one submitted wgpu command buffer.
///
/// # On `wasm32`
///
/// The browser cannot block, so the job reports completion without polling and refuses to wait:
///
/// - [`status`](WgpuJob::status) (and [`is_complete`](WgpuJob::is_complete)) is `Pending` until wgpu's
///   `Queue::on_submitted_work_done` callback for this submission has run, and `Complete` after it. A job that
///   submitted nothing (an empty batch) is `Complete`. The callback runs on the browser's event loop, so a status
///   read in the same synchronous run of code as the submission is `Pending`.
/// - [`wait`](WgpuJob::wait) and [`wait_timeout`](WgpuJob::wait_timeout) always return [`Error::WouldBlock`], never
///   `Ok`, so a caller cannot take a finished GPU for granted. Any work the job submitted has been submitted and will
///   run; only the wait is refused. Synchronise with an awaited readback
///   ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)), which resolves when the work and the copy
///   are done.
///
/// Every blocking convenience built on `wait` fails the same way on `wasm32`, after it has submitted the work: the
/// generated `dispatch` and `dispatch_extent` of a kernel, `gust::Run::dispatch_call` and `gust::Device::dispatch`
/// return that error at run time (use `submit` and an awaited readback there), and the `HeadlessDevice` helpers that
/// wait are compiled out.
#[must_use = "GPU work is asynchronous; poll, await a readback or wait on the returned job when completion matters"]
pub struct WgpuJob<'device> {
    // Only the native `poll_with_timeout` reads these two.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    device: &'device wgpu::Device,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    submission: Option<wgpu::SubmissionIndex>,
    /// Set by the `on_submitted_work_done` callback of this job's submission (`true` from the start for a job that
    /// submitted nothing). An `Arc<AtomicBool>` is `Send + Sync` and the callback must be `Send`; the field exists on
    /// `wasm32` only, so the auto traits of `WgpuJob` on native are what they were.
    #[cfg(target_arch = "wasm32")]
    done: Arc<std::sync::atomic::AtomicBool>,
}

impl WgpuJob<'_> {
    /// Poll the exact submission once without blocking.
    ///
    /// On `wasm32` this reads the flag that `Queue::on_submitted_work_done` sets: `Pending` until the callback has
    /// run, then `Complete`; a job with no submission is `Complete`. A lost device also reads `Complete` there (the
    /// WebGPU specification runs the completion steps when the device is lost, and wgpu calls back when the promise
    /// rejects), where native returns `Err(Error::Poll)`.
    pub fn status(&self) -> Result<JobStatus, Error> {
        #[cfg(target_arch = "wasm32")]
        {
            Ok(if self.done.load(Ordering::Acquire) {
                JobStatus::Complete
            } else {
                JobStatus::Pending
            })
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(Some(Duration::ZERO))
    }

    /// Whether the submission has completed, from one non-blocking poll ([`WgpuJob::status`]).
    pub fn is_complete(&self) -> Result<bool, Error> {
        self.status().map(JobStatus::is_complete)
    }

    /// Wait for at most `timeout`.
    ///
    /// # On `wasm32`
    ///
    /// Always returns [`Error::WouldBlock`]: the browser cannot block, and `timeout` is ignored. Use
    /// [`status`](WgpuJob::status) or an awaited readback
    /// ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)).
    pub fn wait_timeout(&self, timeout: Duration) -> Result<JobStatus, Error> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = timeout;
            Err(Error::WouldBlock("wait_timeout"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(Some(timeout))
    }

    /// Wait indefinitely for this submission, without waiting for later work.
    ///
    /// # On `wasm32` this returns an error
    ///
    /// The browser cannot block, so on `wasm32` this always returns [`Error::WouldBlock`], never `Ok`, even for a job
    /// that has already completed or submitted nothing. Any work the job submitted will run; only the wait is
    /// refused. The synchronisation on the web is an awaited readback
    /// ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)), which resolves only when the work and the
    /// copy are done; [`status`](WgpuJob::status) reports completion without waiting.
    pub fn wait(&self) -> Result<(), Error> {
        #[cfg(target_arch = "wasm32")]
        {
            Err(Error::WouldBlock("wait"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(None).map(|_| ())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_with_timeout(&self, timeout: Option<Duration>) -> Result<JobStatus, Error> {
        let Some(submission) = &self.submission else {
            return Ok(JobStatus::Complete);
        };
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout,
        }) {
            Ok(status) if status.wait_finished() => Ok(JobStatus::Complete),
            Ok(_) | Err(wgpu::PollError::Timeout) => Ok(JobStatus::Pending),
            Err(error) => Err(Error::Poll(error.to_string())),
        }
    }
}

impl gust::Job for WgpuJob<'_> {
    type Error = Error;

    fn status(&self) -> Result<JobStatus, Self::Error> {
        Self::status(self)
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<JobStatus, Self::Error> {
        Self::wait_timeout(self, timeout)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Self::wait(self)
    }
}

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

/// What one prepared dispatch launches: compute workgroups for a `grid` kernel, a full-screen pass over `elements`
/// fragments for a `pixel` kernel (slice mx.p1b).
#[derive(Clone, Copy)]
enum Launch {
    /// The workgroups along x, y and z; `[count, 1, 1]` for a one-dimensional dispatch.
    Compute { workgroups: [u32; 3] },
    /// The element count `n` of a `pixel` call: `render::PixelPlan::new(n)` says how it is drawn.
    Pixel { elements: u32 },
}

struct PreparedDispatch {
    cached_kernel: Arc<CachedKernel>,
    bind_group: wgpu::BindGroup,
    launch: Launch,
}

/// The query slots a timed pass writes: the first pass of a batch writes `begin`, the last writes `end`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct PassTimestamps<'a> {
    query_set: &'a wgpu::QuerySet,
    begin: Option<u32>,
    end: Option<u32>,
}

impl PreparedDispatch {
    /// The workgroups along x, which is all the staged graph reports (it runs one-dimensional dispatches only); 0 for a
    /// `pixel` launch, which has none.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn workgroup_count(&self) -> u32 {
        match self.launch {
            Launch::Compute { workgroups } => workgroups[0],
            Launch::Pixel { .. } => 0,
        }
    }

    /// Encode this dispatch into `encoder` as one pass of its own: a compute pass for `grid`, a render pass for `pixel`
    /// (a pass per call, because one call's output may be the next call's input and a pass is one usage scope).
    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        timestamps: Option<PassTimestamps<'_>>,
    ) -> Result<(), Error> {
        match (&self.cached_kernel.pipeline, self.launch) {
            (Pipeline::Compute(pipeline), Launch::Compute { workgroups }) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: timestamps.map(|stamps| wgpu::ComputePassTimestampWrites {
                        query_set: stamps.query_set,
                        beginning_of_pass_write_index: stamps.begin,
                        end_of_pass_write_index: stamps.end,
                    }),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                Ok(())
            }
            (Pipeline::Render(pipeline), Launch::Pixel { elements }) => render::encode_pixel_pass(
                device,
                encoder,
                pipeline,
                &self.bind_group,
                elements,
                label,
                timestamps,
            ),
            _ => unreachable!("a kernel's shape fixes both its pipeline and its launch"),
        }
    }
}

const TIMESTAMP_RESULT_BYTES: u64 = 2 * size_of::<u64>() as u64;

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct TimestampProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
}

impl TimestampProfiler {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GUST dispatch timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST timestamp resolve"),
            size: TIMESTAMP_RESULT_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST timestamp readback"),
            size: TIMESTAMP_RESULT_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
        }
    }
}

/// Adapter requests made by `HeadlessDevice::new` before `NoAdapter` is returned.
const ADAPTER_ATTEMPTS: u32 = 3;
/// Pauses after the first and second failed adapter request.
const ADAPTER_BACKOFF: [Duration; 2] = [Duration::from_millis(100), Duration::from_millis(400)];

/// Run `request` up to `attempts` times, pausing `backoff[n]` through `pause` after the
/// `n`th failure. Only `Error::NoAdapter` is retried; any other result returns at once.
/// The final `NoAdapter` keeps its message and gains the attempt count. When a request
/// succeeds after at least one failed attempt, one line goes to `log`, so the transient
/// failure stays visible instead of being absorbed by the retry.
///
/// Both the request and the pause are futures, so the same loop serves the browser, where
/// nothing may block. `pause` is awaited between attempts; on native it sleeps the thread and
/// returns a ready future (`pause_between_attempts`).
async fn retry_no_adapter<
    T,
    Request: Future<Output = Result<T, Error>>,
    Pause: Future<Output = ()>,
>(
    attempts: u32,
    backoff: &[Duration],
    mut pause: impl FnMut(Duration) -> Pause,
    mut log: impl FnMut(&str),
    mut request: impl FnMut() -> Request,
) -> Result<T, Error> {
    let mut attempt = 1;
    let mut first_failure: Option<String> = None;
    loop {
        match request().await {
            Err(Error::NoAdapter(message)) if attempt >= attempts => {
                return Err(Error::NoAdapter(format!(
                    "{message} (after {attempt} attempts)"
                )));
            }
            Err(Error::NoAdapter(message)) => {
                first_failure.get_or_insert(message);
                if let Some(delay) = backoff.get(attempt as usize - 1) {
                    pause(*delay).await;
                }
                attempt += 1;
            }
            other => {
                if let (Ok(_), Some(message)) = (&other, &first_failure) {
                    log(&format!(
                        "gust-compute-wgpu: adapter request succeeded on attempt {attempt} of {attempts} after a transient NoAdapter failure: {message}"
                    ));
                }
                return other;
            }
        }
    }
}

/// The pause between two adapter requests. Native sleeps the calling thread and returns a ready future.
#[cfg(not(target_arch = "wasm32"))]
fn pause_between_attempts(delay: Duration) -> std::future::Ready<()> {
    std::thread::sleep(delay);
    std::future::ready(())
}

/// The pause between two adapter requests. A browser thread cannot sleep (`std::thread::sleep` panics on
/// `wasm32-unknown-unknown`), so the next request follows at once.
#[cfg(target_arch = "wasm32")]
fn pause_between_attempts(_delay: Duration) -> std::future::Ready<()> {
    std::future::ready(())
}

/// A staging buffer whose `map_async` has been requested, with the completion its callback fills.
struct PendingMapping {
    readback: wgpu::Buffer,
    mapped: Completion<Result<(), String>>,
}

/// A typed readback after its copy was submitted and its mapping requested. An empty buffer has no mapping.
struct TypedReadback<T> {
    mapping: Option<PendingMapping>,
    element: PhantomData<fn() -> T>,
}

impl<T: GpuPod> TypedReadback<T> {
    /// Block until the device has finished the submitted work. This does not mean this mapping's callback has run:
    /// another thread's poll may have collected it and not fired it yet. The native blocking wrappers call it before
    /// they wait for the completion in `finish` (`block_on_mapping`), so that the wait is short; the completion is
    /// what they rely on.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, device: &wgpu::Device) -> Result<(), Error> {
        if self.mapping.is_some() {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| Error::Poll(error.to_string()))?;
        }
        Ok(())
    }

    /// Await the mapping (`MappingReady`), copy the bytes out and unmap.
    async fn finish(self, device: &wgpu::Device) -> Result<Vec<T>, Error> {
        let Some(PendingMapping { readback, mapped }) = self.mapping else {
            return Ok(Vec::new());
        };
        MappingReady { device, mapped }.await?;
        let view = readback
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let values = pod_vec_from_bytes(&view)?;
        drop(view);
        readback.unmap();
        Ok(values)
    }
}

/// Resolves when the `map_async` callback has delivered its result. Every poll polls the device once without
/// waiting (`PollType::Poll`), which is what runs the callback on native; it never waits, because the browser must not
/// block. It stores the task's waker in the completion and never wakes itself.
struct MappingReady<'device> {
    device: &'device wgpu::Device,
    mapped: Completion<Result<(), String>>,
}

impl Future for MappingReady<'_> {
    type Output = Result<(), Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::Poll(error.to_string())));
        }
        Pin::new(&mut self.mapped)
            .poll(cx)
            .map(|result| result.map_err(Error::Map))
    }
}

/// A logical device selected without creating a window or presentation surface.
pub struct HeadlessDevice {
    id: u64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    pipeline_cache: Mutex<PipelineCache>,
    timestamp_profiler: Option<Mutex<TimestampProfiler>>,
    /// Limits every launch check reads: the opened device's, or lower ones installed
    /// by `with_limits`.
    limits: wgpu::Limits,
    /// Whether the adapter lets a fragment shader write a storage buffer (`DownlevelFlags::FRAGMENT_WRITABLE_STORAGE`),
    /// which every `pixel` kernel does (D120). A `pixel` call on an adapter without it is `Error::ShapeUnsupported`.
    fragment_storage: bool,
    /// The capabilities the device was opened with (`DeviceOptions::capabilities`); a kernel's `requires` must be a
    /// subset (`check_kernel_capabilities`).
    capabilities: &'static [Capability],
    /// The adapter's subgroup size range, when it offers subgroup operations (`HeadlessDevice::subgroup_sizes`).
    subgroup_sizes: Option<(u32, u32)>,
    /// Recycles the allocations of [`GpuVec`] and other pooled buffers.
    pool: BufferPool,
}

impl HeadlessDevice {
    /// Select a high-performance Vulkan adapter and open a logical device.
    ///
    /// A missing adapter is retried a bounded number of times with a short backoff,
    /// because the Vulkan loader can fail transiently when many devices are opened at
    /// once. After the last attempt the `NoAdapter` error is returned; there is no skip and
    /// no runtime fallback to the CPU twin. The adapter is whatever wgpu offers for Vulkan with a
    /// high-performance preference: a software Vulkan adapter (lavapipe, llvmpipe) is
    /// accepted like a hardware one, because the adapter type is not checked.
    ///
    /// This blocks the calling thread (`pollster::block_on` over [`new_async`](Self::new_async)); it is
    /// not available on `wasm32`, where the browser cannot be blocked.
    ///
    /// The device requests wgpu's default limits, the portable ones that a browser always gets, and no optional
    /// features. [`DeviceOptions::default()`](DeviceOptions) differs on native: it raises
    /// `max_buffer_size`, the storage binding size and the workgroups per dimension to the adapter's own
    /// ([`LimitsPolicy::Adapter`]) and asks for `float32-filterable` when the adapter offers it, and so does
    /// [`default_device`]. Use [`with_options`](Self::with_options) to get those on a device of your own.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, Error> {
        pollster::block_on(Self::new_async(wgpu::Backends::VULKAN))
    }

    /// Select a high-performance adapter among `backends` and open a logical device, with the retry of
    /// [`new_async_with`](Self::new_async_with).
    ///
    /// On native the retry pauses by sleeping the calling thread, which blocks an executor thread that awaits
    /// this; call it from a thread that may block, or use `new`. In the browser there is no pause between
    /// attempts. Opening the device completes without anyone polling it: wgpu drives the adapter and device
    /// requests itself, unlike the readbacks.
    pub async fn new_async(backends: wgpu::Backends) -> Result<Self, Error> {
        Self::new_async_with(DeviceOptions {
            backends,
            adapter: AdapterChoice::HighPerformance,
            limits: LimitsPolicy::Default,
            features: FeaturesPolicy::Default,
            capabilities: &[],
        })
        .await
    }

    /// Open the device `options` describe, with the adapter retry of
    #[cfg_attr(not(target_arch = "wasm32"), doc = "[`new`](Self::new).")]
    #[cfg_attr(target_arch = "wasm32", doc = "`new`.")]
    /// Unlike `new`, which is
    /// fixed to Vulkan, the high-performance adapter, wgpu's default limits and no optional feature, the options
    /// choose the backends, the adapter, whether the adapter's own buffer limits are requested and whether
    /// `float32-filterable` is enabled where the adapter offers it, so that a linear sampler can be bound
    /// ([`float32_filterable`](Self::float32_filterable) reports the result; see [`DeviceOptions`]).
    pub async fn new_async_with(options: DeviceOptions) -> Result<Self, Error> {
        let DeviceOptions {
            backends,
            adapter,
            limits,
            features,
            capabilities,
        } = options;
        let adapter = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            pause_between_attempts,
            |line| eprintln!("{line}"),
            || Self::request_adapter(backends, &adapter),
        )
        .await?;
        Self::open(adapter, limits, features, capabilities).await
    }

    async fn open(
        adapter: wgpu::Adapter,
        limits_policy: LimitsPolicy,
        features_policy: FeaturesPolicy,
        capabilities: &'static [Capability],
    ) -> Result<Self, Error> {
        let adapter_info = adapter.get_info();
        // A capability the adapter lacks ends the open here (D23's startup failure), before wgpu is asked for anything.
        let capability_features = device::required_capability_features(
            capabilities,
            adapter.features(),
            &adapter_info.name,
        )?;
        let subgroup_sizes = device::adapter_subgroup_sizes(adapter.features(), &adapter_info);
        let timestamp_queries = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamp_queries {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GUST headless device"),
                required_features: required_features
                    | (adapter.features() & wgpu::Features::MEMORY_DECORATION_COHERENT)
                    | features_policy.required_features(&adapter)
                    | capability_features,
                required_limits: limits_policy.required_limits(&adapter),
                ..Default::default()
            })
            .await
            .map_err(|error| Error::RequestDevice(error.to_string()))?;
        let timestamp_profiler =
            timestamp_queries.then(|| Mutex::new(TimestampProfiler::new(&device)));
        let limits = device.limits();
        let fragment_storage = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE);

        Ok(Self {
            id: NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed),
            device,
            queue,
            adapter_info,
            pipeline_cache: Mutex::new(PipelineCache::default()),
            timestamp_profiler,
            limits,
            fragment_storage,
            capabilities,
            subgroup_sizes,
            pool: BufferPool::new(),
        })
    }

    /// The wgpu description of the adapter that the device was opened on.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// The capabilities this device was opened with (`DeviceOptions::capabilities`), as asked: a kernel whose
    /// `requires` names a capability not in this set is refused with [`Error::MissingCapability`].
    pub fn capabilities(&self) -> &'static [Capability] {
        self.capabilities
    }

    /// The smallest and the largest subgroup size the adapter reports (`AdapterInfo`), or `None` when the adapter does
    /// not offer subgroup operations.
    ///
    /// This is the adapter's range, not a promise of one size: the lane count a shader sees can differ between the
    /// dispatches of one device (spike S3: an AMD Radeon 890M on DX12 gave 32 lanes in one kernel and 64 in the
    /// others), so a kernel reads its lane count at run time and never assumes one value from this method. It is
    /// `Some` whether or not the device was opened with [`Capability::Subgroups`].
    pub fn subgroup_sizes(&self) -> Option<(u32, u32)> {
        self.subgroup_sizes
    }

    /// The limits launches are checked against: those the device was opened with
    /// (wgpu's defaults for `new`/`new_async`; the adapter's own buffer and dispatch limits under
    /// [`LimitsPolicy::Adapter`]), or the lower ones from `with_limits`.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Check launches against `limits` instead of the opened device's, for emulating a
    /// weaker target and for tests. The limits may only be lowered: a value that is
    /// better than the one the device was opened with is rejected with
    /// `Error::LimitRaised` and never clamped, because wgpu would keep enforcing the
    /// device's own. The replacement is compared with the opened device, not with
    /// limits installed earlier, so a lowered limit can be raised back up to it. The
    /// pipeline cache is kept; the checks run before it is consulted. A rejected call
    /// consumes the device, like any failing builder step.
    ///
    /// Only these fields are honoured by the launch checks:
    /// `max_compute_workgroup_size_x`, `max_compute_workgroup_size_y`,
    /// `max_compute_workgroup_size_z`, `max_compute_invocations_per_workgroup`,
    /// `max_compute_workgroups_per_dimension` (for x, y and z alike), `max_storage_buffers_per_shader_stage`,
    /// `max_storage_buffer_binding_size`, `max_uniform_buffers_per_shader_stage`,
    /// `max_uniform_buffer_binding_size`, `max_sampled_textures_per_shader_stage`,
    /// `max_storage_textures_per_shader_stage`, `max_samplers_per_shader_stage`, `max_texture_dimension_2d` (the
    /// sides of every texture a launch binds, and of `create_texture_2d`) and `max_bindings_per_bind_group`
    /// (every binding index of a kernel must be below it). Every other field is accepted if it is not
    /// raised, but has no effect. `max_buffer_size` is read by the fallible buffer functions
    /// ([`try_create_typed_buffer`](Self::try_create_typed_buffer),
    /// [`create_zeroed_buffer`](Self::create_zeroed_buffer),
    /// [`GpuVec`]) and by nothing else; `create_typed_buffer` never checks it.
    pub fn with_limits(mut self, limits: wgpu::Limits) -> Result<Self, Error> {
        let mut raised = None;
        limits.check_limits_with_fail_fn(
            &self.device.limits(),
            true,
            |limit, requested, device| {
                raised = Some(Error::LimitRaised {
                    limit,
                    requested,
                    device,
                });
            },
        );
        if let Some(error) = raised {
            return Err(error);
        }
        self.limits = limits;
        Ok(self)
    }

    /// Whether the device has wgpu's `MEMORY_DECORATION_COHERENT` feature, coherent storage memory.
    pub fn supports_coherent_storage(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::MEMORY_DECORATION_COHERENT)
    }

    /// Whether this device can measure compute passes with hardware timestamp queries.
    pub const fn supports_gpu_timestamps(&self) -> bool {
        self.timestamp_profiler.is_some()
    }

    /// Return hit/miss counts for cached shader modules, layouts, and pipelines.
    pub fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        let cache = self.lock_pipeline_cache();
        PipelineCacheStats {
            entries: cache.kernels.len(),
            hits: cache.hits,
            misses: cache.misses,
        }
    }

    /// Drop every compiled kernel owned by this device and reset its counters.
    pub fn clear_pipeline_cache(&self) {
        *self.lock_pipeline_cache() = PipelineCache::default();
    }

    /// Allocate and initialize a persistent storage buffer for any GPU POD type.
    ///
    /// The signature is infallible, so GUST does not check `max_buffer_size` here; an
    /// allocation above it is wgpu's to reject (a panic). Launch limits (workgroup shape, workgroup
    /// count, storage buffer count and binding size) are checked at dispatch time. Use
    /// [`try_create_typed_buffer`](Self::try_create_typed_buffer) or
    /// [`create_zeroed_buffer`](Self::create_zeroed_buffer) for a request that may be too large.
    pub fn create_typed_buffer<T: GpuPod>(
        &self,
        label: &str,
        values: &[T],
        access: GpuBufferAccess,
    ) -> GpuBuffer<T> {
        assert_pod_matches_storage_layout::<T>();
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if access == GpuBufferAccess::ReadWrite {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        let raw = if values.is_empty() {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(T::LAYOUT.stride),
                usage,
                mapped_at_creation: false,
            })
        } else {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: pod_slice_as_bytes(values),
                usage,
            })
        };
        GpuBuffer {
            raw,
            len: values.len(),
            access,
            layout: T::LAYOUT,
            device_id: self.id,
            indirect: false,
            lease: None,
            marker: PhantomData,
        }
    }

    /// [`HeadlessDevice::create_typed_buffer`] for `f32` values.
    pub fn create_f32_buffer(
        &self,
        label: &str,
        values: &[f32],
        access: GpuBufferAccess,
    ) -> GpuBuffer<f32> {
        self.create_typed_buffer(label, values, access)
    }

    /// Allocate a uniform (a constant buffer of one `T`) holding `value`.
    ///
    /// `T` must be a padding-free `Std140V1` type with Rust's size and alignment: scalars, 2- and 4-wide vectors, 4x4
    /// matrices and structs of them whose fields fill a multiple of 16 bytes. Anything else is an
    /// `Error::InvalidUniformType` whose message names the fix (spell the padding out as fields), because a padded
    /// struct would upload uninitialised bytes. The buffer is created with `UNIFORM | COPY_DST | COPY_SRC`.
    pub fn create_uniform<T: GpuPod>(
        &self,
        label: &str,
        value: &T,
    ) -> Result<GpuUniform<T>, Error> {
        check_uniform_type::<T>()?;
        let raw = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: pod_slice_as_bytes(std::slice::from_ref(value)),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        Ok(GpuUniform {
            raw,
            layout: T::LAYOUT,
            device_id: self.id,
            marker: PhantomData,
        })
    }

    /// Replace the value of a uniform with `Queue::write_buffer`: the write is ordered after every earlier submission
    /// and before the next one, so work already submitted sees the old value. Every dispatch of one later submission
    /// (a `submit_batch` or a graph) sees the last value written, whatever the order they were recorded in; use
    /// `StagedGraph::write_uniform`, which encodes a copy in the command stream, when an update must fall between two
    /// dispatches of one submission.
    pub fn write_uniform<T: GpuPod>(
        &self,
        uniform: &GpuUniform<T>,
        value: &T,
    ) -> Result<(), Error> {
        if uniform.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        self.queue.write_buffer(
            &uniform.raw,
            0,
            pod_slice_as_bytes(std::slice::from_ref(value)),
        );
        Ok(())
    }

    /// Replace the complete contents of an existing persistent POD buffer.
    pub fn write_typed_buffer<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
        values: &[T],
    ) -> Result<(), Error> {
        self.validate_buffer_device(buffer)?;
        if buffer.len != values.len() {
            return Err(Error::PersistentBufferLength {
                expected: buffer.len,
                actual: values.len(),
            });
        }
        if !values.is_empty() {
            self.queue
                .write_buffer(&buffer.raw, 0, pod_slice_as_bytes(values));
        }
        Ok(())
    }

    /// [`HeadlessDevice::write_typed_buffer`] for an `f32` buffer.
    pub fn write_f32_buffer(&self, buffer: &GpuBuffer<f32>, values: &[f32]) -> Result<(), Error> {
        self.write_typed_buffer(buffer, values)
    }

    /// Replace a persistent POD buffer and wait until the upload has completed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_typed_buffer_timed<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
        values: &[T],
    ) -> Result<TransferTiming, Error> {
        let started = Instant::now();
        self.write_typed_buffer(buffer, values)?;
        if !values.is_empty() {
            let submission = self.queue.submit([]);
            self.wait_for_submission(submission)?;
        }
        Ok(TransferTiming {
            bytes: values.len() as u64 * size_of::<T>() as u64,
            elapsed: started.elapsed(),
        })
    }

    /// [`HeadlessDevice::write_typed_buffer_timed`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_f32_buffer_timed(
        &self,
        buffer: &GpuBuffer<f32>,
        values: &[f32],
    ) -> Result<TransferTiming, Error> {
        self.write_typed_buffer_timed(buffer, values)
    }

    /// Copy a persistent read-write POD buffer back to host memory.
    ///
    /// Blocks the calling thread until the copy has completed; not available on `wasm32`, where
    /// [`read_typed_buffer_async`](Self::read_typed_buffer_async) is the readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_typed_buffer<T: GpuPod>(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>, Error> {
        self.read_typed_buffer_timed(buffer)
            .map(|(values, _)| values)
    }

    /// [`HeadlessDevice::read_typed_buffer`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_f32_buffer(&self, buffer: &GpuBuffer<f32>) -> Result<Vec<f32>, Error> {
        self.read_typed_buffer(buffer)
    }

    /// Copy a persistent read-write POD buffer back to host memory without blocking the caller.
    ///
    /// The returned future requests the copy and the mapping when it is first polled. Each poll then polls the
    /// device once (`wgpu::PollType::Poll`, never `Wait`) and stores the task's waker for the `map_async` callback
    /// to wake.
    ///
    /// # Native: the future completes only while something polls the device
    ///
    /// wgpu runs the `map_async` callback inside a device poll, and on native nothing polls the device in the
    /// background. This future never wakes itself, so awaiting it on an executor that re-polls only after a wake
    /// hangs, unless another thread keeps polling the device. The blocking
    #[cfg_attr(
        not(target_arch = "wasm32"),
        doc = "[`read_typed_buffer`](Self::read_typed_buffer)"
    )]
    #[cfg_attr(target_arch = "wasm32", doc = "`read_typed_buffer`")]
    /// does it correctly: it starts the copy, waits with `poll(Wait)`, then waits on the finished future itself, because
    /// another thread's device poll can run this mapping's callback after this thread's `poll(Wait)` returned
    /// (bounded, then `Error::Map`; `completion::blocking`). On `wasm32` the browser runs the callback from its event
    /// loop, which wakes the task, so a plain `.await` suffices there (measured in Chrome by `cargo xtask check-web`).
    ///
    /// ```no_run
    /// # use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
    /// let device = HeadlessDevice::new()?;
    /// let buffer = device.create_typed_buffer("values", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadWrite);
    /// // On native this is the way to read a buffer back; it polls the device for you.
    /// let values = device.read_typed_buffer(&buffer)?;
    /// assert_eq!(values, [1.0, 2.0, 3.0]);
    /// // A caller who drives the async function on its own executor must keep polling the future (each poll
    /// // polls the device once) or the device itself, as `read_typed_buffer` does.
    /// # Ok::<(), gust_wgpu::Error>(())
    /// ```
    pub async fn read_typed_buffer_async<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<Vec<T>, Error> {
        self.start_typed_readback(buffer)?
            .finish(&self.device)
            .await
    }

    /// Copy a persistent read-write POD buffer back and report its complete host time.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_typed_buffer_timed<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<(Vec<T>, TransferTiming), Error> {
        let started = Instant::now();
        let readback = self.start_typed_readback(buffer)?;
        // The callback runs inside a device poll, but not necessarily this thread's: another thread's poll collects
        // the callbacks of every finished submission, so `poll(Wait)` can return before this mapping's callback has
        // run. The wait is therefore on the completion itself, which the callback signals from whichever thread
        // runs it, for at most `MAPPING_WAIT` (then `Error::Map`).
        readback.wait(&self.device)?;
        let values = block_on_mapping(readback.finish(&self.device))?;
        Ok((
            values,
            TransferTiming {
                bytes: buffer.len as u64 * size_of::<T>() as u64,
                elapsed: started.elapsed(),
            },
        ))
    }

    /// Validate `buffer`, copy it into a mappable staging buffer, submit the copy and request the mapping. The
    /// returned readback completes once a device poll has run the `map_async` callback.
    fn start_typed_readback<T: GpuPod>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<TypedReadback<T>, Error> {
        self.validate_buffer_device(buffer)?;
        if buffer.access != GpuBufferAccess::ReadWrite {
            return Err(Error::PersistentBufferNotReadable);
        }
        if buffer.is_empty() {
            return Ok(TypedReadback {
                mapping: None,
                element: PhantomData,
            });
        }

        let byte_size = buffer.len as u64 * size_of::<T>() as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST persistent buffer readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST persistent buffer readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&buffer.raw, 0, &readback, 0, byte_size);
        self.queue.submit([encoder.finish()]);

        let mapped = Completion::new();
        let callback = mapped.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            // The callback is `FnOnce`, so it runs at most once; a late or repeated completion would be rejected by
            // the `Completion` anyway.
            let _ = callback.complete(result.map_err(|error| error.to_string()));
        });
        Ok(TypedReadback {
            mapping: Some(PendingMapping { readback, mapped }),
            element: PhantomData,
        })
    }

    /// [`HeadlessDevice::read_typed_buffer_timed`] for an `f32` buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_f32_buffer_timed(
        &self,
        buffer: &GpuBuffer<f32>,
    ) -> Result<(Vec<f32>, TransferTiming), Error> {
        self.read_typed_buffer_timed(buffer)
    }

    /// Compatibility wrapper for an all-f32 persistent dispatch.
    pub fn submit_f32_buffers<'device>(
        &'device self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        self.submit_f32_batch(&[F32BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Dispatch `kernel` over `element_count` elements with `bindings` and wait until it has completed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_f32_buffers(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<(), Error> {
        self.submit_f32_buffers(kernel, element_count, bindings)?
            .wait()
    }

    /// [`HeadlessDevice::dispatch_f32_buffers`], returning the host timing of the dispatch.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_f32_buffers_timed(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32BufferBinding<'_>],
    ) -> Result<DispatchTiming, Error> {
        self.dispatch_f32_batch_timed(&[F32BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Submit `dispatches` as one command buffer and return its job without waiting.
    pub fn submit_f32_batch<'device>(
        &'device self,
        dispatches: &[F32BufferDispatch<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        with_f32_dispatches(dispatches, |typed| self.submit_batch(typed))
    }

    /// Submit `dispatches` as one command buffer and wait until it has completed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_f32_batch(&self, dispatches: &[F32BufferDispatch<'_>]) -> Result<(), Error> {
        self.submit_f32_batch(dispatches)?.wait()
    }

    /// [`HeadlessDevice::dispatch_f32_batch`], returning the host timing of the batch.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_f32_batch_timed(
        &self,
        dispatches: &[F32BufferDispatch<'_>],
    ) -> Result<DispatchTiming, Error> {
        with_f32_dispatches(dispatches, |typed| self.dispatch_batch_timed(typed))
    }

    /// Submit one persistent-buffer dispatch without waiting for completion.
    pub fn submit_buffers<'device>(
        &'device self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        self.submit_batch(&[BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Dispatch using already-resident typed buffers, without upload or readback.
    ///
    /// This convenience wrapper calls [`Self::submit_buffers`] and waits on
    /// its exact submission fence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_buffers(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<(), Error> {
        self.submit_buffers(kernel, element_count, bindings)?.wait()
    }

    /// Dispatch using persistent buffers and report host time plus an optional GPU timestamp.
    ///
    /// `gpu_elapsed` covers only the compute pass. It is `None` when the
    /// adapter does not expose timestamp queries or when `element_count` is zero.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_buffers_timed(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[BufferBinding<'_>],
    ) -> Result<DispatchTiming, Error> {
        self.dispatch_batch_timed(&[BufferDispatch::new(kernel, element_count, bindings)])
    }

    /// Encode and submit a batch, returning before its GPU work completes.
    ///
    /// Every request is validated before any pipeline, bind group, or command
    /// encoding work begins. Empty requests are valid and do not create GPU work. Encoding a `pixel` call creates its
    /// attachment, so such a call can still fail with [`Error::OutOfMemory`] named `GUST pixel attachment`; nothing
    /// is submitted then.
    pub fn submit_batch<'device>(
        &'device self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<WgpuJob<'device>, Error> {
        let prepared = self.prepare_extent_batch(dispatches)?;
        if prepared.is_empty() {
            return Ok(WgpuJob {
                device: &self.device,
                submission: None,
                #[cfg(target_arch = "wasm32")]
                done: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST typed batch encoder"),
            });
        for dispatch in &prepared {
            dispatch.encode(&self.device, &mut encoder, "GUST typed batched pass", None)?;
        }
        let submission = self.queue.submit([encoder.finish()]);
        // The browser cannot poll, so the job learns of completion from the queue (`WgpuJob::status`).
        #[cfg(target_arch = "wasm32")]
        let done = {
            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = Arc::clone(&done);
            self.queue
                .on_submitted_work_done(move || flag.store(true, Ordering::Release));
            done
        };
        Ok(WgpuJob {
            device: &self.device,
            submission: Some(submission),
            #[cfg(target_arch = "wasm32")]
            done,
        })
    }

    /// Submit a batch and synchronously wait for its exact submission fence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_batch(&self, dispatches: &[BufferDispatch<'_>]) -> Result<(), Error> {
        self.submit_batch(dispatches)?.wait()
    }

    /// Submit a batch once and measure its total host and GPU time.
    ///
    /// `gpu_elapsed` brackets the entire compute pass containing every
    /// non-empty dispatch. Divide it by the batch length when the dispatches
    /// represent equivalent repeated work.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_batch_timed(
        &self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<DispatchTiming, Error> {
        let started = Instant::now();
        let prepared = self.prepare_extent_batch(dispatches)?;
        if prepared.is_empty() {
            return Ok(DispatchTiming {
                host_elapsed: started.elapsed(),
                gpu_elapsed: None,
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST profiled typed batch encoder"),
            });

        let Some(timestamp_profiler) = &self.timestamp_profiler else {
            for dispatch in &prepared {
                dispatch.encode(
                    &self.device,
                    &mut encoder,
                    "GUST profiled typed batched pass",
                    None,
                )?;
            }
            let submission = self.queue.submit([encoder.finish()]);
            self.wait_for_submission(submission)?;
            return Ok(DispatchTiming {
                host_elapsed: started.elapsed(),
                gpu_elapsed: None,
            });
        };

        let profiler = timestamp_profiler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let last_dispatch = prepared.len() - 1;
        for (index, dispatch) in prepared.iter().enumerate() {
            let timestamps = (index == 0 || index == last_dispatch).then_some(PassTimestamps {
                query_set: &profiler.query_set,
                begin: (index == 0).then_some(0),
                end: (index == last_dispatch).then_some(1),
            });
            dispatch.encode(
                &self.device,
                &mut encoder,
                "GUST profiled typed batched pass",
                timestamps,
            )?;
        }
        encoder.resolve_query_set(&profiler.query_set, 0..2, &profiler.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &profiler.resolve_buffer,
            0,
            &profiler.readback_buffer,
            0,
            TIMESTAMP_RESULT_BYTES,
        );
        let submission = self.queue.submit([encoder.finish()]);
        let gpu_elapsed = self.read_timestamp_duration(&profiler, submission)?;
        Ok(DispatchTiming {
            host_elapsed: started.elapsed(),
            gpu_elapsed: Some(gpu_elapsed),
        })
    }

    /// Dispatch an executable kernel over `element_count` one-dimensional invocations.
    ///
    /// This v1 slice accepts one host binding for each storage parameter and reads
    /// every read-write binding back after the dispatch completes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_f32(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[F32Binding<'_>],
    ) -> Result<DispatchOutput, Error> {
        let resources = validate_dispatch(kernel, element_count, bindings)?;
        if element_count == 0 {
            return Ok(DispatchOutput {
                writable_buffers: bindings
                    .iter()
                    .copied()
                    .filter_map(|binding| match binding {
                        F32Binding::ReadWrite(values) => Some(values.to_vec()),
                        F32Binding::ReadOnly(_) => None,
                    })
                    .collect(),
            });
        }

        let workgroup_count = self.check_launch(
            kernel,
            element_count,
            &resources,
            bindings
                .iter()
                .map(|binding| (size_of_val(binding.values()) as u64, BindingKind::Storage)),
        )?;
        let label = kernel.qualified_name().to_string();
        let cached_kernel = self.cached_kernel(kernel, &resources)?;

        let gpu_buffers = bindings
            .iter()
            .copied()
            .enumerate()
            .map(|(index, binding)| {
                let bytes = f32_as_bytes(binding.values());
                let mut usage = wgpu::BufferUsages::STORAGE;
                if binding.access() == Access::ReadWrite {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(&format!("{label} binding {index}")),
                    contents: bytes,
                    usage,
                })
            })
            .collect::<Vec<_>>();
        let bind_group_entries = resources
            .iter()
            .zip(&gpu_buffers)
            .map(|(parameter, buffer)| wgpu::BindGroupEntry {
                binding: parameter
                    .binding
                    .expect("validated resource binding")
                    .binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout: &cached_kernel.bind_group_layout,
            entries: &bind_group_entries,
        });

        let byte_size = u64::from(element_count) * size_of::<f32>() as u64;
        let readback_buffers = bindings
            .iter()
            .copied()
            .map(|binding| match binding {
                F32Binding::ReadOnly(_) => None,
                F32Binding::ReadWrite(_) => {
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label} readback")),
                        size: byte_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }))
                }
            })
            .collect::<Vec<_>>();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label} command encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} compute pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(cached_kernel.pipeline.compute());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        for (gpu_buffer, readback) in gpu_buffers.iter().zip(&readback_buffers) {
            if let Some(readback) = readback {
                encoder.copy_buffer_to_buffer(gpu_buffer, 0, readback, 0, byte_size);
            }
        }
        self.queue.submit([encoder.finish()]);

        let mut completions = Vec::new();
        for readback in readback_buffers.iter().flatten() {
            let (sender, receiver) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
            completions.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;

        let mut writable_buffers = Vec::with_capacity(completions.len());
        for (readback, completion) in readback_buffers.iter().flatten().zip(completions) {
            recv_mapping(&completion)?;
            let view = readback
                .get_mapped_range(..)
                .map_err(|error| Error::Map(error.to_string()))?;
            let values = view
                .as_chunks::<{ size_of::<f32>() }>()
                .0
                .iter()
                .map(|chunk| f32::from_ne_bytes(*chunk))
                .collect();
            drop(view);
            readback.unmap();
            writable_buffers.push(values);
        }

        Ok(DispatchOutput { writable_buffers })
    }

    /// Dispatch an executable kernel over `element_count` one-dimensional invocations
    /// using `u32` storage buffers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch_u32(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        bindings: &[U32Binding<'_>],
    ) -> Result<U32DispatchOutput, Error> {
        let resources = validate_u32_dispatch(kernel, element_count, bindings)?;
        if element_count == 0 {
            return Ok(U32DispatchOutput {
                writable_buffers: bindings
                    .iter()
                    .copied()
                    .filter_map(|binding| match binding {
                        U32Binding::ReadWrite(values) => Some(values.to_vec()),
                        U32Binding::ReadOnly(_) => None,
                    })
                    .collect(),
            });
        }

        let workgroup_count = self.check_launch(
            kernel,
            element_count,
            &resources,
            bindings
                .iter()
                .map(|binding| (size_of_val(binding.values()) as u64, BindingKind::Storage)),
        )?;
        let label = kernel.qualified_name().to_string();
        let cached_kernel = self.cached_kernel(kernel, &resources)?;

        let gpu_buffers = bindings
            .iter()
            .copied()
            .enumerate()
            .map(|(index, binding)| {
                let bytes = pod_slice_as_bytes(binding.values());
                let mut usage = wgpu::BufferUsages::STORAGE;
                if binding.access() == Access::ReadWrite {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(&format!("{label} binding {index}")),
                    contents: bytes,
                    usage,
                })
            })
            .collect::<Vec<_>>();
        let bind_group_entries = resources
            .iter()
            .zip(&gpu_buffers)
            .map(|(parameter, buffer)| wgpu::BindGroupEntry {
                binding: parameter
                    .binding
                    .expect("validated resource binding")
                    .binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout: &cached_kernel.bind_group_layout,
            entries: &bind_group_entries,
        });

        let byte_size = u64::from(element_count) * size_of::<u32>() as u64;
        let readback_buffers = bindings
            .iter()
            .copied()
            .map(|binding| match binding {
                U32Binding::ReadOnly(_) => None,
                U32Binding::ReadWrite(_) => {
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{label} readback")),
                        size: byte_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }))
                }
            })
            .collect::<Vec<_>>();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label} command encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} compute pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(cached_kernel.pipeline.compute());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        for (gpu_buffer, readback) in gpu_buffers.iter().zip(&readback_buffers) {
            if let Some(readback) = readback {
                encoder.copy_buffer_to_buffer(gpu_buffer, 0, readback, 0, byte_size);
            }
        }
        self.queue.submit([encoder.finish()]);

        let mut completions = Vec::new();
        for readback in readback_buffers.iter().flatten() {
            let (sender, receiver) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
            completions.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;

        let mut writable_buffers = Vec::with_capacity(completions.len());
        for (readback, completion) in readback_buffers.iter().flatten().zip(completions) {
            recv_mapping(&completion)?;
            let view = readback
                .get_mapped_range(..)
                .map_err(|error| Error::Map(error.to_string()))?;
            let values = view
                .as_chunks::<{ size_of::<u32>() }>()
                .0
                .iter()
                .map(|chunk| u32::from_ne_bytes(*chunk))
                .collect();
            drop(view);
            readback.unmap();
            writable_buffers.push(values);
        }

        Ok(U32DispatchOutput { writable_buffers })
    }

    /// The staged graph's preparation (`graph.rs` reads `workgroup_count`, the x count): one-dimensional dispatches
    /// only. A dispatch with a y or z extent other than 1 is refused with [`Error::Extent`] instead of being encoded
    /// as its x count alone, which would silently run a fraction of it.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn prepare_batch(
        &self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<Vec<PreparedDispatch>, Error> {
        if let Some(dispatch) = dispatches
            .iter()
            .find(|dispatch| dispatch.extent().y != 1 || dispatch.extent().z != 1)
        {
            // A `pixel` kernel has no extent launch at all: the refusal is its own, not the graph's.
            render::check_pixel_extent(dispatch.kernel, dispatch.extent())?;
            return Err(Error::Extent {
                extent: dispatch.extent().to_array().map(u64::from),
                reason: "a staged graph runs one-dimensional dispatches only (use dispatch_batch)",
            });
        }
        self.prepare_dispatches(dispatches, false)
    }

    /// The batch functions' preparation: every dispatch over its whole [`Extent`].
    fn prepare_extent_batch(
        &self,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<Vec<PreparedDispatch>, Error> {
        self.prepare_dispatches(dispatches, false)
    }

    /// Prepare one indirect dispatch: every check of `prepare_batch` except the workgroup count,
    /// which only the GPU reads, from the arguments buffer. `dispatch.extent()` must be a
    /// nonzero placeholder; the prepared `workgroup_count` is meaningless (0) and never encoded.
    #[cfg(not(target_arch = "wasm32"))]
    fn prepare_indirect(
        &self,
        dispatch: &BufferDispatch<'_>,
    ) -> Result<Vec<PreparedDispatch>, Error> {
        self.prepare_dispatches(std::slice::from_ref(dispatch), true)
    }

    fn prepare_dispatches(
        &self,
        dispatches: &[BufferDispatch<'_>],
        indirect: bool,
    ) -> Result<Vec<PreparedDispatch>, Error> {
        let validated = dispatches
            .iter()
            .map(|dispatch| {
                let resources = validate_dispatch_extent(
                    self.id,
                    self.float32_filterable(),
                    dispatch.kernel,
                    dispatch.extent(),
                    dispatch.bindings,
                    dispatch.samplers,
                )?;
                let binding_bytes = dispatch.bindings.iter().map(BufferBinding::limit_bytes);
                let workgroups = if indirect {
                    self.check_indirect_launch(dispatch.kernel, &resources, binding_bytes)?;
                    [0; 3]
                } else {
                    self.check_launch_extent(
                        dispatch.kernel,
                        dispatch.extent(),
                        &resources,
                        binding_bytes,
                    )?
                };
                if indirect || !dispatch.extent().is_empty() {
                    texture::check_texture_dimensions(
                        &self.limits,
                        dispatch.kernel,
                        dispatch.bindings,
                    )?;
                }
                Ok((resources, workgroups))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut prepared = Vec::with_capacity(dispatches.len());
        for (dispatch, (resources, workgroups)) in dispatches.iter().zip(validated) {
            if dispatch.extent().is_empty() {
                continue;
            }
            let cached_kernel = self.cached_kernel(dispatch.kernel, &resources)?;
            let label = dispatch.kernel.qualified_name().to_string();
            // The bound parameters in declaration order; a sampler parameter takes the next sampler binding and every
            // other one the next resource binding (`validate_dispatch_extent` counted both).
            let mut resource_bindings = dispatch.bindings.iter();
            let mut sampler_bindings = dispatch.samplers.iter();
            let bind_group_entries = resources
                .iter()
                .map(|parameter| wgpu::BindGroupEntry {
                    binding: parameter
                        .binding
                        .expect("validated resource binding")
                        .binding,
                    resource: if parameter.kind == ParameterKind::Sampler {
                        sampler_bindings
                            .next()
                            .expect("validated sampler binding")
                            .resource()
                    } else {
                        resource_bindings
                            .next()
                            .expect("validated resource binding")
                            .resource()
                    },
                })
                .collect::<Vec<_>>();
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{label} batched bind group")),
                layout: &cached_kernel.bind_group_layout,
                entries: &bind_group_entries,
            });
            let launch = if dispatch.kernel.shape == Shape::Pixel {
                Launch::Pixel {
                    elements: dispatch.extent().x,
                }
            } else {
                Launch::Compute { workgroups }
            };
            prepared.push(PreparedDispatch {
                cached_kernel,
                bind_group,
                launch,
            });
        }
        Ok(prepared)
    }

    /// Check one non-empty launch against the stored limits and return its workgroup
    /// count. Runs after `validate_*` and before the pipeline-cache lookup, so no
    /// compilation, cache hit or wgpu object can precede a limit error. A launch of
    /// zero elements does no GPU work and is not checked.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn check_launch(
        &self,
        kernel: &KernelDescriptor,
        element_count: u32,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<u32, Error> {
        self.check_launch_extent(
            kernel,
            Extent::linear(element_count),
            resources,
            binding_bytes,
        )
        .map(|workgroups| workgroups[0])
    }

    /// `check_launch` over an [`Extent`]: the workgroups along x, y and z. A launch with a zero axis does no GPU
    /// work, is not checked and has `[0, 0, 0]` workgroups.
    fn check_launch_extent(
        &self,
        kernel: &KernelDescriptor,
        extent: Extent,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<[u32; 3], Error> {
        if extent.is_empty() {
            return Ok([0; 3]);
        }
        self.check_launch_limits(kernel, Some(extent), resources, binding_bytes)
    }

    /// `check_launch` for an indirect dispatch: every check except the workgroup count, which the
    /// host never sees (the GPU reads it from the arguments buffer), so a lowered
    /// `max_compute_workgroups_per_dimension` never rejects an indirect launch.
    fn check_indirect_launch(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<(), Error> {
        render::refuse_indirect(kernel)?;
        self.check_launch_limits(kernel, None, resources, binding_bytes)
            .map(|_| ())
    }

    /// The limit checks in their fixed order. `extent` is `None` for an indirect launch,
    /// which skips the workgroup counts and returns zeros.
    fn check_launch_limits(
        &self,
        kernel: &KernelDescriptor,
        extent: Option<Extent>,
        resources: &[&ParameterDescriptor],
        binding_bytes: impl IntoIterator<Item = (u64, BindingKind)>,
    ) -> Result<[u32; 3], Error> {
        let count = |kind: ParameterKind| {
            resources
                .iter()
                .filter(|parameter| parameter.kind == kind)
                .count()
        };
        check_kernel_limits(
            &self.limits,
            kernel,
            count(ParameterKind::Storage),
            count(ParameterKind::Uniform),
            TextureCounts::of(resources),
        )?;
        check_kernel_capabilities(
            self.capabilities,
            self.subgroup_sizes,
            &self.adapter_info.name,
            kernel,
        )?;
        check_binding_indices(&self.limits, kernel, resources)?;
        let workgroups = match (kernel.shape, extent) {
            // A `pixel` launch has no workgroups: the cap and the attachment are its limits (the adapter's ability to
            // write storage from a fragment first, since without it there is no pipeline to build).
            (Shape::Pixel, Some(extent)) => {
                render::require_fragment_storage(kernel, self.fragment_storage)?;
                render::check_pixel_launch(&self.limits, kernel, extent.x)?;
                [0; 3]
            }
            (_, Some(extent)) => workgroup_counts(&self.limits, kernel, extent)?,
            (_, None) => [0; 3],
        };
        for (bytes, kind) in binding_bytes {
            check_binding_size(&self.limits, kernel, bytes, kind)?;
        }
        Ok(workgroups)
    }

    fn cached_kernel(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
    ) -> Result<Arc<CachedKernel>, Error> {
        let key = KernelCacheKey::new(kernel);
        {
            let mut cache = self.lock_pipeline_cache();
            if let Some(cached) = cache.kernels.get(&key).cloned() {
                cache.hits += 1;
                return Ok(cached);
            }
        }

        let compiled = Arc::new(self.compile_kernel(kernel, resources)?);
        let mut cache = self.lock_pipeline_cache();
        if let Some(cached) = cache.kernels.get(&key).cloned() {
            cache.hits += 1;
            return Ok(cached);
        }
        cache.misses += 1;
        cache.kernels.insert(key, Arc::clone(&compiled));
        Ok(compiled)
    }

    /// Compile and reflect `kernel` through the native helper, then cross-check it against the descriptor.
    #[cfg(feature = "slang-runtime")]
    fn compile_at_run_time(
        kernel: &KernelDescriptor,
    ) -> Result<gust::reflect::CompiledReflection, Error> {
        let compiled = gust::reflect::compile_reflected(kernel, gust::slang::Target::Wgsl)
            .map_err(Error::Reflection)?;
        let _ = compiled
            .reflection
            .cross_check_kernel(kernel)
            .map_err(Error::Reflection)?;
        Ok(compiled)
    }

    /// Without `slang-runtime` this crate holds no run-time Slang bridge: the only kernels it can run are the ones whose
    /// precompiled artifact it verifies. `select_shader_source` never sends a kernel here in this build (it
    /// answers `NoArtifact` first); the same error is returned in case a caller does.
    #[cfg(not(feature = "slang-runtime"))]
    fn compile_at_run_time(
        kernel: &KernelDescriptor,
    ) -> Result<gust::reflect::CompiledReflection, Error> {
        Err(no_artifact(kernel))
    }

    fn compile_kernel(
        &self,
        kernel: &KernelDescriptor,
        resources: &[&ParameterDescriptor],
    ) -> Result<CachedKernel, Error> {
        let pixel = match kernel.shape {
            Shape::Grid => false,
            Shape::Pixel => {
                // Without fragment-stage storage writes there is no pipeline to build (D120).
                render::require_fragment_storage(kernel, self.fragment_storage)?;
                true
            }
            // `Shape` is `#[non_exhaustive]`: a shape this backend does not know has no pipeline either.
            _ => return Err(render::unknown_shape(kernel)),
        };
        let compiled = match select_shader_source(kernel, cfg!(feature = "slang-runtime"))? {
            // A verification failure is the answer: there is no fallback to the run-time compiler.
            ShaderSource::Precompiled(artifact) => {
                gust::artifact::verify_artifact(kernel, artifact)
                    .map_err(Error::Reflection)?
                    .compiled
            }
            ShaderSource::RunTime => Self::compile_at_run_time(kernel)?,
        };
        let wgsl = std::str::from_utf8(&compiled.artifact)
            .expect("WGSL validated by the native compiler bridge");
        let label = kernel.qualified_name().to_string();
        // A texture or sampler entry follows the device's layout variant (`texture.rs`); a buffer entry is as ever.
        let layout_entries = resources
            .iter()
            .map(|parameter| {
                let ty = match parameter.kind {
                    ParameterKind::Texture | ParameterKind::Sampler => {
                        self.texture_binding_type(parameter)?
                    }
                    _ => wgpu::BindingType::Buffer {
                        ty: if parameter.kind == ParameterKind::Uniform {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: parameter.access == Access::ReadOnly,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                };
                Ok(wgpu::BindGroupLayoutEntry {
                    binding: parameter
                        .binding
                        .expect("validated resource binding")
                        .binding,
                    visibility: if pixel {
                        wgpu::ShaderStages::FRAGMENT
                    } else {
                        wgpu::ShaderStages::COMPUTE
                    },
                    ty,
                    count: None,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label} bind group layout")),
                    entries: &layout_entries,
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} pipeline layout")),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{label} Slang-generated WGSL")),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = if pixel {
            let vertex = self.fullscreen_module();
            Pipeline::Render(render::build_pipeline(
                &self.device,
                &label,
                &pipeline_layout,
                &vertex,
                &shader,
                kernel.entry_point,
            ))
        } else {
            Pipeline::Compute(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{label} pipeline")),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(kernel.entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ))
        };
        Ok(CachedKernel {
            _shader: shader,
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            pipeline,
        })
    }

    /// The vertex module every `pixel` pipeline of this device shares, created on first use.
    fn fullscreen_module(&self) -> wgpu::ShaderModule {
        let mut cache = self.lock_pipeline_cache();
        cache
            .fullscreen
            .get_or_insert_with(|| render::fullscreen_module(&self.device))
            .clone()
    }

    fn lock_pipeline_cache(&self) -> MutexGuard<'_, PipelineCache> {
        self.pipeline_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn validate_buffer_device<T>(&self, buffer: &GpuBuffer<T>) -> Result<(), Error> {
        if buffer.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait_for_submission(&self, submission: wgpu::SubmissionIndex) -> Result<(), Error> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| Error::Poll(error.to_string()))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_timestamp_duration(
        &self,
        profiler: &TimestampProfiler,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Duration, Error> {
        let (sender, receiver) = mpsc::channel();
        profiler
            .readback_buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
        self.wait_for_submission(submission)?;
        recv_mapping(&receiver)?;
        let view = profiler
            .readback_buffer
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let mut timestamps = view
            .as_chunks::<{ size_of::<u64>() }>()
            .0
            .iter()
            .map(|chunk| u64::from_ne_bytes(*chunk));
        let beginning = timestamps.next().expect("beginning timestamp");
        let end = timestamps.next().expect("end timestamp");
        drop(view);
        profiler.readback_buffer.unmap();
        let nanoseconds =
            end.wrapping_sub(beginning) as f64 * f64::from(self.queue.get_timestamp_period());
        Ok(Duration::from_secs_f64(nanoseconds / 1_000_000_000.0))
    }
}

fn with_f32_dispatches<R>(
    dispatches: &[F32BufferDispatch<'_>],
    run: impl FnOnce(&[BufferDispatch<'_>]) -> R,
) -> R {
    let bindings = dispatches
        .iter()
        .map(|dispatch| {
            dispatch
                .bindings
                .iter()
                .copied()
                .map(BufferBinding::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let typed = dispatches
        .iter()
        .zip(&bindings)
        .map(|(dispatch, bindings)| {
            BufferDispatch::new(dispatch.kernel, dispatch.element_count, bindings)
        })
        .collect::<Vec<_>>();
    run(&typed)
}

fn device_limit(
    kernel: &KernelDescriptor,
    limit: &'static str,
    requested: u64,
    maximum: u64,
) -> Error {
    Error::DeviceLimit {
        kernel: kernel.qualified_name().to_string(),
        limit,
        requested,
        maximum,
    }
}

/// Limits a pipeline for `kernel` must respect: each workgroup axis against its own limit (x, then y, then z), the
/// product of the three (a checked product: [`Error::Extent`] when it overflows) against
/// `max_compute_invocations_per_workgroup`, then the buffer counts, then the texture and sampler counts
/// (`max_sampled_textures_per_shader_stage`, `max_storage_textures_per_shader_stage` and
/// `max_samplers_per_shader_stage`; the texture dimension is `texture::check_texture_dimensions`'s).
fn check_kernel_limits(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    storage_buffers: usize,
    uniform_buffers: usize,
    textures: TextureCounts,
) -> Result<(), Error> {
    // A `pixel` kernel has no workgroup: the three workgroup limits are a compute stage's.
    let compute = kernel.shape != Shape::Pixel;
    let size = kernel.workgroup_size.map(u64::from);
    if compute {
        for (limit, requested, maximum) in [
            (
                "max_compute_workgroup_size_x",
                size[0],
                u64::from(limits.max_compute_workgroup_size_x),
            ),
            (
                "max_compute_workgroup_size_y",
                size[1],
                u64::from(limits.max_compute_workgroup_size_y),
            ),
            (
                "max_compute_workgroup_size_z",
                size[2],
                u64::from(limits.max_compute_workgroup_size_z),
            ),
        ] {
            if requested > maximum {
                return Err(device_limit(kernel, limit, requested, maximum));
            }
        }
    }
    let invocations = if compute {
        size[0]
            .checked_mul(size[1])
            .and_then(|plane| plane.checked_mul(size[2]))
            .ok_or(Error::Extent {
                extent: size,
                reason: "the workgroup size multiplies to more invocations than a u64 holds",
            })?
    } else {
        0
    };
    for (limit, requested, maximum) in [
        (
            "max_compute_invocations_per_workgroup",
            invocations,
            u64::from(limits.max_compute_invocations_per_workgroup),
        ),
        (
            "max_storage_buffers_per_shader_stage",
            storage_buffers as u64,
            u64::from(limits.max_storage_buffers_per_shader_stage),
        ),
        (
            "max_uniform_buffers_per_shader_stage",
            uniform_buffers as u64,
            u64::from(limits.max_uniform_buffers_per_shader_stage),
        ),
        (
            "max_sampled_textures_per_shader_stage",
            textures.sampled as u64,
            u64::from(limits.max_sampled_textures_per_shader_stage),
        ),
        (
            "max_storage_textures_per_shader_stage",
            textures.storage as u64,
            u64::from(limits.max_storage_textures_per_shader_stage),
        ),
        (
            "max_samplers_per_shader_stage",
            textures.samplers as u64,
            u64::from(limits.max_samplers_per_shader_stage),
        ),
    ] {
        if requested > maximum {
            return Err(device_limit(kernel, limit, requested, maximum));
        }
    }
    Ok(())
}

/// What a kernel's `requires` asks of a device (M8, D142), checked beside [`check_kernel_limits`]: every capability in
/// `kernel.requires` is one the device was opened with (`capabilities`), else [`Error::MissingCapability`] naming the
/// kernel and `adapter`; and a kernel that requires subgroups has a workgroup of a whole number of the largest subgroups
/// the adapter reports (`subgroup_sizes`), else [`Error::SubgroupSize`]. Only the largest size is checked, and the
/// macro already requires 64 invocations at a time (E0751): the lane count of one dispatch is the shader's to read.
fn check_kernel_capabilities(
    capabilities: &[Capability],
    subgroup_sizes: Option<(u32, u32)>,
    adapter: &str,
    kernel: &KernelDescriptor,
) -> Result<(), Error> {
    for &required in kernel.requires {
        if !capabilities.contains(&required) {
            return Err(Error::MissingCapability {
                capability: required,
                adapter: adapter.to_owned(),
                kernel: Some(kernel.qualified_name().to_string()),
            });
        }
    }
    if kernel.requires.contains(&Capability::Subgroups)
        && let Some((_, maximum)) = subgroup_sizes
        && maximum != 0
    {
        let invocations = kernel.workgroup_size.iter().fold(1_u64, |product, &axis| {
            product.saturating_mul(u64::from(axis))
        });
        if invocations % u64::from(maximum) != 0 {
            return Err(Error::SubgroupSize {
                kernel: kernel.qualified_name().to_string(),
                invocations,
                subgroup_max: maximum,
            });
        }
    }
    Ok(())
}

/// Every binding index must be below `max_bindings_per_bind_group`: wgpu rejects the bind group layout
/// (an uncaptured-error panic) when an index is equal to or above it, however few bindings there are.
/// `requested` is the highest index plus one, so it compares with the maximum like the other limits.
fn check_binding_indices(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    resources: &[&ParameterDescriptor],
) -> Result<(), Error> {
    let maximum = u64::from(limits.max_bindings_per_bind_group);
    let requested = resources
        .iter()
        .filter_map(|parameter| parameter.binding)
        .map(|binding| u64::from(binding.binding) + 1)
        .max()
        .unwrap_or(0);
    if requested > maximum {
        return Err(device_limit(
            kernel,
            "max_bindings_per_bind_group",
            requested,
            maximum,
        ));
    }
    Ok(())
}

/// The limit the workgroups along y name in an [`Error::DeviceLimit`]; the one along x keeps the unsuffixed name of the
/// one-dimensional dispatch, the one along z is `max_compute_workgroups_per_dimension (z)`.
const WORKGROUPS_Y_LIMIT: &str = "max_compute_workgroups_per_dimension (y)";
const WORKGROUPS_Z_LIMIT: &str = "max_compute_workgroups_per_dimension (z)";

/// The name of the invocation cap in an [`Error::DeviceLimit`]; the CPU twin reports the same name
/// (`gust::cpu::Error::DeviceLimit`).
const INVOCATION_CAP_LIMIT: &str = "global_invocation_id";

/// Workgroups along x, y and z needed for `extent`, each within `max_compute_workgroups_per_dimension` (the error
/// names the axis: `max_compute_workgroups_per_dimension` for x, with ` (y)` and ` (z)` for the others), and the launch
/// within the invocation cap: every axis rounded up to whole workgroups, multiplied, at most 2^32
/// ([`Extent::MAX_INVOCATIONS`]), the rule of the CPU twin. A kernel without a zero workgroup axis
/// (`validate_kernel`) and a non-empty extent are assumed.
fn workgroup_counts(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    extent: Extent,
) -> Result<[u32; 3], Error> {
    let size = kernel.workgroup_size;
    let maximum = limits.max_compute_workgroups_per_dimension;
    let counts = [
        extent.x.div_ceil(size[0]),
        extent.y.div_ceil(size[1]),
        extent.z.div_ceil(size[2]),
    ];
    for (limit, count) in [
        ("max_compute_workgroups_per_dimension", counts[0]),
        (WORKGROUPS_Y_LIMIT, counts[1]),
        (WORKGROUPS_Z_LIMIT, counts[2]),
    ] {
        if count > maximum {
            return Err(device_limit(
                kernel,
                limit,
                u64::from(count),
                u64::from(maximum),
            ));
        }
    }
    // `rounded_total` is in `u128`: three rounded axes are at most 2^96, so it cannot overflow. The error carries a
    // `u64`, so a launch beyond it saturates (as the CPU twin's does).
    let invocations = extent
        .rounded_total(size)
        .expect("a workgroup size without a zero axis");
    if invocations > u128::from(Extent::MAX_INVOCATIONS) {
        return Err(device_limit(
            kernel,
            INVOCATION_CAP_LIMIT,
            u64::try_from(invocations).unwrap_or(u64::MAX),
            Extent::MAX_INVOCATIONS,
        ));
    }
    Ok(counts)
}

/// One binding of `bytes` bytes against `max_storage_buffer_binding_size`, or `max_uniform_buffer_binding_size` for a
/// uniform.
fn check_binding_size(
    limits: &wgpu::Limits,
    kernel: &KernelDescriptor,
    bytes: u64,
    kind: BindingKind,
) -> Result<(), Error> {
    // A texture has no byte size in a binding: its limit is the dimension (`texture::check_texture_dimensions`).
    if kind == BindingKind::Texture {
        return Ok(());
    }
    let (limit, maximum) = if kind == BindingKind::Uniform {
        (
            "max_uniform_buffer_binding_size",
            limits.max_uniform_buffer_binding_size,
        )
    } else {
        (
            "max_storage_buffer_binding_size",
            limits.max_storage_buffer_binding_size,
        )
    };
    if bytes > maximum {
        return Err(device_limit(kernel, limit, bytes, maximum));
    }
    Ok(())
}

/// The resource layout a typed buffer of element layout `element` presents: `StorageV1` when the element is a scalar
/// or a struct of scalars (the macro describes those parameters as `StorageV1`), else `StorageV2` (vectors, matrices).
fn buffer_resource_layout(element: gust::TypeLayout) -> ResourceLayout {
    if element.is_storage_v1() {
        ResourceLayout::storage_v1(element)
    } else {
        ResourceLayout::storage_v2(element)
    }
}

/// The resource layout a binding of `kind` and element layout `element` presents: a storage buffer's
/// (`buffer_resource_layout`), or `Std140V1` for a uniform.
fn binding_resource_layout(kind: BindingKind, element: gust::TypeLayout) -> ResourceLayout {
    if kind == BindingKind::Uniform {
        ResourceLayout::std140_v1(element)
    } else {
        buffer_resource_layout(element)
    }
}

fn validate_binding_layout(
    parameter: &ParameterDescriptor,
    actual: ResourceLayout,
) -> Result<(), Error> {
    let expected = parameter
        .resource_layout
        .expect("validated resource layout");
    if expected != actual {
        return Err(Error::UnsupportedResourceLayout {
            parameter: parameter.name.to_owned(),
            expected: Box::new(expected),
            actual: Some(Box::new(actual)),
        });
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_dispatch(
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[F32Binding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    render::require_grid(kernel, render::ONE_SHOT_REASON)?;
    let resources = validate_kernel(kernel)?;
    if resources.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resources.len(),
            actual: bindings.len(),
        });
    }
    for (parameter, binding) in resources.iter().copied().zip(bindings.iter().copied()) {
        validate_binding_layout(
            parameter,
            ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT),
        )?;
        if parameter.access != binding.access() {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access(),
            });
        }
        if binding.values().len() != element_count as usize {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected: element_count as usize,
                actual: binding.values().len(),
            });
        }
    }
    Ok(resources)
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_u32_dispatch(
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[U32Binding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    render::require_grid(kernel, render::ONE_SHOT_REASON)?;
    let resources = validate_kernel(kernel)?;
    if resources.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resources.len(),
            actual: bindings.len(),
        });
    }
    for (parameter, binding) in resources.iter().copied().zip(bindings.iter().copied()) {
        validate_binding_layout(
            parameter,
            ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT),
        )?;
        if parameter.access != binding.access() {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access(),
            });
        }
        if binding.values().len() != element_count as usize {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected: element_count as usize,
                actual: binding.values().len(),
            });
        }
    }
    Ok(resources)
}

/// The binding checks of a one-dimensional dispatch of `element_count` invocations of a buffer-only kernel: what the
/// staged graph validates a node with. A kernel with a texture or sampler parameter is refused
/// ([`Error::UnsupportedParameter`]): the graph tracks hazards by buffer and does not learn textures in MX.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn validate_persistent_dispatch(
    device_id: u64,
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[BufferBinding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    let resources = validate_kernel(kernel)?;
    if let Some(parameter) = resources.iter().find(|parameter| {
        matches!(
            parameter.kind,
            ParameterKind::Texture | ParameterKind::Sampler
        )
    }) {
        return Err(Error::UnsupportedParameter {
            name: parameter.name.to_owned(),
            kind: parameter.kind,
        });
    }
    validate_dispatch_extent(
        device_id,
        false,
        kernel,
        Extent::linear(element_count),
        bindings,
        &[],
    )
}

/// The binding checks of a persistent dispatch over `extent`: the kernel, then per binding the kind, layout, device,
/// access and length, then aliasing. A shared binding holds exactly `x * y * z` elements (the checked product;
/// [`Error::Extent`] when it overflows a `u64`, before any binding is looked at), an independent one only has to be
/// non-empty when the extent is. The launch limits are `check_launch_extent`'s.
///
/// A texture binding has no length rule. It is checked for its format (a storage parameter's exact format, a sampled
/// one's sample class: [`Error::TextureFormat`]), its device, its access and its use ([`Error::TextureUse`]). A sampler
/// binding is checked for its device and its filter against the device's layout variant (`float32_filterable`:
/// [`Error::TextureFilter`]). `samplers` are the bindings of the kernel's sampler parameters in declaration order.
fn validate_dispatch_extent(
    device_id: u64,
    float32_filterable: bool,
    kernel: &KernelDescriptor,
    extent: Extent,
    bindings: &[BufferBinding<'_>],
    samplers: &[GpuSamplerBinding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    let resources = validate_kernel(kernel)?;
    // A `pixel` call is one-dimensional (the fragment index is `y * 8192 + x` of the attachment, not a call's y).
    render::check_pixel_extent(kernel, extent)?;
    let total = extent.checked_total().ok_or(Error::Extent {
        extent: extent.to_array().map(u64::from),
        reason: "x * y * z overflows a u64",
    })?;
    // A product above `usize::MAX` is longer than any buffer: it saturates, so the length check fails for all of them.
    let expected = usize::try_from(total).unwrap_or(usize::MAX);
    // The resource parameters (buffers and textures) pair with `bindings`, the sampler parameters with `samplers`.
    let (sampler_parameters, resource_parameters): (
        Vec<&ParameterDescriptor>,
        Vec<&ParameterDescriptor>,
    ) = resources
        .iter()
        .copied()
        .partition(|parameter| parameter.kind == ParameterKind::Sampler);
    if resource_parameters.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resource_parameters.len(),
            actual: bindings.len(),
        });
    }
    if sampler_parameters.len() != samplers.len() {
        return Err(Error::BindingCount {
            expected: sampler_parameters.len(),
            actual: samplers.len(),
        });
    }
    for (parameter, binding) in resource_parameters
        .iter()
        .copied()
        .zip(bindings.iter().copied())
    {
        let buffer = binding;
        let expected_kind =
            BindingKind::of(parameter.kind).expect("validate_kernel keeps only bound parameters");
        if buffer.kind != expected_kind {
            return Err(Error::BindingKind {
                parameter: parameter.name.to_owned(),
                expected: expected_kind,
                actual: buffer.kind,
            });
        }
        if let Some(texture) = binding.texture {
            texture::validate_texture_binding(device_id, parameter, &binding, texture)?;
            continue;
        }
        validate_binding_layout(
            parameter,
            binding_resource_layout(buffer.kind, buffer.layout),
        )?;
        if buffer.device_id != device_id {
            return Err(Error::ForeignPersistentBuffer);
        }
        if parameter.access != binding.access {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access,
            });
        }
        if !buffer.buffer_access.allows(binding.access) {
            return Err(Error::PersistentBufferAccess {
                parameter: parameter.name.to_owned(),
                buffer: buffer.buffer_access,
                binding: binding.access,
            });
        }
        // An output a `pixel` kernel writes in place must have exactly `n` elements (D127): there is no clamp-free way
        // to bound a store to a shorter one, so the twin refuses it too (`cpu::Error::PixelIndependentOutput`).
        if kernel.shape == Shape::Pixel
            && parameter.kind == ParameterKind::Storage
            && parameter.access == Access::ReadWrite
            && binding.independent_length
        {
            return Err(Error::ShapeIndependentOutput {
                kernel: kernel.qualified_name().to_string(),
                parameter: parameter.name.to_owned(),
            });
        }
        if binding.independent_length {
            if buffer.len == 0 && total != 0 {
                return Err(Error::IndependentLengthEmpty {
                    parameter: parameter.name.to_owned(),
                });
            }
        } else if buffer.len != expected {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected,
                actual: buffer.len,
            });
        }
    }
    for (parameter, sampler) in sampler_parameters.iter().copied().zip(samplers) {
        texture::validate_sampler_binding(device_id, float32_filterable, parameter, sampler)?;
    }
    // One resource may be bound twice only when both bindings are read-only. The access of each binding
    // equals its parameter's access here, and the check names both parameters in declaration order. A texture
    // is compared by its own identity (`Raw`): a storage texture bound twice is refused, a sampled one twice is
    // accepted (wgpu allows two read-only uses of one texture), and a texture never equals a buffer.
    for (index, (first_parameter, first)) in resource_parameters.iter().zip(bindings).enumerate() {
        for (second_parameter, second) in resource_parameters.iter().zip(bindings).skip(index + 1) {
            if first.raw == second.raw
                && (first.access == Access::ReadWrite || second.access == Access::ReadWrite)
            {
                return Err(Error::AliasedBinding {
                    kernel: kernel.qualified_name().to_string(),
                    first: first_parameter.name.to_owned(),
                    second: second_parameter.name.to_owned(),
                });
            }
        }
    }
    Ok(resources)
}

pub(crate) fn validate_kernel(
    kernel: &KernelDescriptor,
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    // A kernel built for the CPU has no GPU form (D137). The typed entry points refuse it at compile time; a kernel given
    // by its descriptor is refused here, first, on every path, whatever its extent.
    if kernel.target != Target::Gpu {
        return Err(Error::KernelTarget {
            kernel: kernel.qualified_name().to_string(),
        });
    }
    // `grid` runs as a compute pass and `pixel` as a render pass (slice mx.p1b); a shape added later has no path yet.
    if !matches!(kernel.shape, Shape::Grid | Shape::Pixel) {
        return Err(render::unknown_shape(kernel));
    }
    if kernel.slang_source.trim().is_empty() {
        return Err(Error::MissingSlangSource(
            kernel.qualified_name().to_string(),
        ));
    }
    // Workgroups of one, two or three dimensions (slice m7.d); an axis of 0 is no workgroup. The device limits of each
    // axis and of their product are `check_kernel_limits`'s.
    if kernel.workgroup_size.contains(&0) {
        return Err(Error::UnsupportedWorkgroupSize(kernel.workgroup_size));
    }

    let resources = kernel
        .parameters
        .iter()
        .filter(|parameter| parameter.binding.is_some())
        .collect::<Vec<_>>();
    for parameter in resources.iter().copied() {
        // The rule is a function of the element (`binding_resource_layout`): a `Vec4` labelled StorageV1, or a storage
        // parameter under the Std140V1 rule, is refused here and not only later at the persistent dispatch. A uniform
        // is a constant buffer: read-only, and `Std140V1` with an element that satisfies it.
        let layout_ok = match parameter.kind {
            ParameterKind::Storage => parameter.resource_layout.is_some_and(|layout| {
                layout.element.is_storage_v2()
                    && layout == binding_resource_layout(BindingKind::Storage, layout.element)
            }),
            ParameterKind::Uniform => {
                if parameter.access != Access::ReadOnly {
                    return Err(Error::BindingAccess {
                        parameter: parameter.name.to_owned(),
                        expected: Access::ReadOnly,
                        actual: parameter.access,
                    });
                }
                parameter.resource_layout.is_some_and(|layout| {
                    layout.element.is_std140_v1()
                        && layout == binding_resource_layout(BindingKind::Uniform, layout.element)
                })
            }
            // A texture's contract is the `TextureV1` rule (D91) and a sampler has none: `texture.rs` judges both.
            ParameterKind::Texture => texture::texture_parameter_is_valid(parameter),
            ParameterKind::Sampler => texture::sampler_parameter_is_valid(parameter),
            _ => {
                return Err(Error::UnsupportedParameter {
                    name: parameter.name.to_owned(),
                    kind: parameter.kind,
                });
            }
        };
        if !layout_ok {
            return Err(Error::InvalidResourceLayout {
                parameter: parameter.name.to_owned(),
                actual: parameter.resource_layout.map(Box::new),
            });
        }
        let resource_binding = parameter.binding.expect("filtered resource binding");
        if resource_binding.group != 0 {
            return Err(Error::UnsupportedBindGroup(resource_binding.group));
        }
    }
    Ok(resources)
}

#[cfg(not(target_arch = "wasm32"))]
fn f32_as_bytes(values: &[f32]) -> &[u8] {
    pod_slice_as_bytes(values)
}

fn pod_vec_from_bytes<T: GpuPod>(bytes: &[u8]) -> Result<Vec<T>, Error> {
    let element_size = size_of::<T>();
    if element_size == 0 || !bytes.len().is_multiple_of(element_size) {
        return Err(Error::Map(format!(
            "{} mapped bytes cannot be decoded as `{}` elements",
            bytes.len(),
            T::LAYOUT.name
        )));
    }
    let len = bytes.len() / element_size;
    let mut values = Vec::<T>::with_capacity(len);
    // SAFETY: `values` owns enough correctly aligned storage for `len` values.
    // `GpuPod` guarantees all copied bit patterns are valid and contain no padding.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        values.set_len(len);
    }
    Ok(values)
}

/// Where the WGSL of a kernel comes from.
#[derive(Clone, Copy, Debug)]
enum ShaderSource {
    /// The kernel's embedded WGSL artifact, to be verified before use.
    Precompiled(&'static gust::Artifact),
    /// The run-time compiler (the `slang-runtime` feature).
    RunTime,
}

/// The shader source policy (D45): a kernel with a WGSL artifact uses it, whether or not the run-time compiler is
/// available; a kernel without one is compiled at run time only when `run_time_compiler` says this build holds the
/// compiler, and is otherwise `Error::NoArtifact`. `run_time_compiler` is `cfg!(feature = "slang-runtime")` in
/// production and an argument so that a test can reach both builds from the one it runs in.
fn select_shader_source(
    kernel: &KernelDescriptor,
    run_time_compiler: bool,
) -> Result<ShaderSource, Error> {
    if let Some(artifact) = cache::preferred_artifact(kernel) {
        return Ok(ShaderSource::Precompiled(artifact));
    }
    if run_time_compiler {
        return Ok(ShaderSource::RunTime);
    }
    Err(no_artifact(kernel))
}

/// The error of a kernel that has no WGSL artifact in a build without the run-time compiler.
fn no_artifact(kernel: &KernelDescriptor) -> Error {
    Error::NoArtifact {
        kernel: kernel.qualified_name().to_string(),
        profile: gust::profile::NAME,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::ResourceBinding;

    const WGSL_ARTIFACT: gust::Artifact = gust::Artifact {
        format: gust::Format::Wgsl,
        key: "0123456789abcdef",
        bytes: b"",
        reflection: "",
    };

    #[test]
    fn a_kernel_with_a_wgsl_artifact_uses_it_with_or_without_the_run_time_compiler() {
        let kernel = DESCRIPTOR.with_artifacts(&[WGSL_ARTIFACT]);
        for run_time_compiler in [false, true] {
            let Ok(ShaderSource::Precompiled(artifact)) =
                select_shader_source(&kernel, run_time_compiler)
            else {
                panic!("a WGSL artifact is always preferred");
            };
            assert_eq!(artifact, &WGSL_ARTIFACT);
        }
    }

    #[test]
    fn a_kernel_without_an_artifact_compiles_at_run_time_only_where_the_compiler_exists() {
        assert!(DESCRIPTOR.artifacts.is_empty());
        assert!(matches!(
            select_shader_source(&DESCRIPTOR, true),
            Ok(ShaderSource::RunTime)
        ));
    }

    #[test]
    fn a_kernel_without_an_artifact_in_a_build_without_the_compiler_is_no_artifact() {
        // The `slang-runtime` feature is always on in tests (the self dev-dependency), so this is the one place the
        // off-state policy runs: through the argument that stands for the feature.
        let spirv_only = DESCRIPTOR.with_artifacts(&[gust::Artifact {
            format: gust::Format::Spirv,
            ..WGSL_ARTIFACT
        }]);
        for kernel in [&DESCRIPTOR, &spirv_only] {
            let error = select_shader_source(kernel, false)
                .expect_err("no WGSL artifact and no compiler is an error, not a fallback");
            let Error::NoArtifact {
                kernel: name,
                profile,
            } = &error
            else {
                panic!("{error:?}");
            };
            assert_eq!(name, &kernel.qualified_name().to_string());
            assert_eq!(*profile, gust::profile::NAME);
            let message = error.to_string();
            assert!(message.contains(&format!("`{name}`")), "{message}");
            assert!(message.contains("cargo gust build"), "{message}");
            assert!(message.contains("slang-runtime"), "{message}");
        }
    }

    const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
        name: "out",
        rust_type: "StorageMut<f32>",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    }];

    const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
        "test",
        "kernel",
        "gpu_test_kernel",
        [64, 1, 1],
        PARAMETERS,
        r#"
            [[vk::binding(0, 0)]] RWStructuredBuffer<float> out;
            [shader("compute")]
            [numthreads(64, 1, 1)]
            void gpu_test_kernel(uint3 id : SV_DispatchThreadID) { if (id.x < 1) out[id.x] = 0.0; }
        "#,
    );

    fn no_adapter() -> Error {
        Error::NoAdapter("loader failed".to_string())
    }

    /// The retry loop with synchronous closures, driven to completion: these tests pin the loop's decisions, not its
    /// futures.
    fn retry_no_adapter<T>(
        attempts: u32,
        backoff: &[Duration],
        mut pause: impl FnMut(Duration),
        log: impl FnMut(&str),
        mut request: impl FnMut() -> Result<T, Error>,
    ) -> Result<T, Error> {
        pollster::block_on(super::retry_no_adapter(
            attempts,
            backoff,
            |delay| {
                pause(delay);
                std::future::ready(())
            },
            log,
            || std::future::ready(request()),
        ))
    }

    /// Pending on its first poll (waking itself), ready on the next.
    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                return Poll::Ready(());
            }
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    #[test]
    fn adapter_retry_awaits_requests_and_pauses_that_are_not_ready() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let result = pollster::block_on(super::retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| {
                pauses.push(delay);
                YieldOnce(false)
            },
            |_| {},
            || {
                calls += 1;
                let outcome = if calls < ADAPTER_ATTEMPTS {
                    Err(no_adapter())
                } else {
                    Ok(calls)
                };
                async move {
                    YieldOnce(false).await;
                    outcome
                }
            },
        ));
        assert!(matches!(result, Ok(n) if n == ADAPTER_ATTEMPTS));
        assert_eq!(pauses, ADAPTER_BACKOFF);
    }

    #[test]
    fn adapter_retry_succeeds_on_first_attempt_without_pausing() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                Ok(7)
            },
        );
        assert!(matches!(result, Ok(7)));
        assert_eq!(calls, 1);
        assert!(pauses.is_empty());
        assert!(logs.is_empty(), "no retry happened, so nothing is logged");
    }

    #[test]
    fn adapter_backoff_has_one_pause_between_each_pair_of_attempts() {
        assert_eq!(ADAPTER_BACKOFF.len() as u32, ADAPTER_ATTEMPTS - 1);
    }

    #[test]
    fn adapter_retry_succeeds_on_a_later_attempt() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                if calls < ADAPTER_ATTEMPTS {
                    Err(no_adapter())
                } else {
                    Ok(calls)
                }
            },
        );
        assert!(matches!(result, Ok(n) if n == ADAPTER_ATTEMPTS));
        assert_eq!(calls, ADAPTER_ATTEMPTS);
        assert_eq!(pauses, ADAPTER_BACKOFF);
        assert_eq!(
            pauses.iter().sum::<Duration>(),
            ADAPTER_BACKOFF.iter().sum::<Duration>()
        );
        // The transient failure stays visible: exactly one line, naming the attempt
        // that succeeded and the loader message of the failed one.
        assert_eq!(
            logs,
            [format!(
                "gust-compute-wgpu: adapter request succeeded on attempt {ADAPTER_ATTEMPTS} of {ADAPTER_ATTEMPTS} after a transient NoAdapter failure: loader failed"
            )]
        );
    }

    #[test]
    fn adapter_retry_fails_after_the_limit_with_the_attempt_count() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result: Result<(), Error> = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                Err(no_adapter())
            },
        );
        assert_eq!(calls, ADAPTER_ATTEMPTS);
        assert_eq!(
            pauses.iter().sum::<Duration>(),
            ADAPTER_BACKOFF.iter().sum::<Duration>()
        );
        // A final failure is reported by the error itself, not by the retry log.
        assert!(logs.is_empty());
        match result {
            Err(Error::NoAdapter(message)) => {
                assert_eq!(
                    message,
                    format!("loader failed (after {ADAPTER_ATTEMPTS} attempts)")
                );
            }
            other => panic!("expected NoAdapter after the limit, got {other:?}"),
        }
    }

    #[test]
    fn adapter_retry_returns_other_errors_immediately() {
        let mut calls = 0;
        let mut pauses = 0;
        let result: Result<(), Error> = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |_| pauses += 1,
            |_| panic!("an immediate non-adapter error is not a retry"),
            || {
                calls += 1;
                Err(Error::RequestDevice("lost".to_string()))
            },
        );
        assert!(matches!(result, Err(Error::RequestDevice(_))));
        assert_eq!(calls, 1);
        assert_eq!(pauses, 0);
    }

    #[test]
    fn pipeline_cache_identity_includes_entrypoint() {
        let other = DESCRIPTOR.with_entry_point("gpu_other_entry");
        assert_ne!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&other)
        );
        assert_eq!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&DESCRIPTOR)
        );
    }

    #[test]
    fn pipeline_cache_identity_ignores_the_diagnostic_source_origin() {
        // The line table is diagnostic metadata only (DX19): two descriptors that
        // differ only in `source` must share one cached pipeline.
        const LINES: &[gust::LineSpan] = &[gust::LineSpan {
            slang_start: 3,
            slang_end: 3,
            rust_line: 9,
        }];
        let located = DESCRIPTOR.with_source(gust::SourceOrigin {
            file: "crates/demo/src/lib.rs",
            lines: LINES,
        });
        assert_ne!(located.source, DESCRIPTOR.source);
        assert_eq!(
            KernelCacheKey::new(&DESCRIPTOR),
            KernelCacheKey::new(&located)
        );
    }

    #[test]
    fn shared_source_entrypoints_execute_distinct_pipelines() {
        const SOURCE: &str = r#"
            [[vk::binding(0, 0)]] RWStructuredBuffer<float> out;
            [shader("compute")] [numthreads(64, 1, 1)]
            void first(uint3 id : SV_DispatchThreadID) { if (id.x == 0) out[0] = 11.0; }
            [shader("compute")] [numthreads(64, 1, 1)]
            void second(uint3 id : SV_DispatchThreadID) { if (id.x == 0) out[0] = 22.0; }
        "#;
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let buffer =
            device.create_f32_buffer("entrypoint identity", &[0.0], GpuBufferAccess::ReadWrite);
        for (entry, expected) in [
            ("first", 11.0),
            ("second", 22.0),
            ("first", 11.0),
            ("second", 22.0),
        ] {
            let descriptor = DESCRIPTOR.with_entry_point(entry).with_slang_source(SOURCE);
            device
                .dispatch_buffers(&descriptor, 1, &[BufferBinding::read_write(&buffer)])
                .unwrap();
            assert_eq!(device.read_f32_buffer(&buffer).unwrap(), [expected]);
        }
        let stats = device.pipeline_cache_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (2, 2, 2));
    }

    #[test]
    fn same_buffer_compares_the_allocation_not_the_handle_address() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let first = device.create_f32_buffer("identity first", &[0.0], GpuBufferAccess::ReadWrite);
        let second =
            device.create_f32_buffer("identity second", &[0.0], GpuBufferAccess::ReadWrite);
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

    #[test]
    fn rejects_incorrect_binding_count_before_gpu_work() {
        assert!(matches!(
            validate_dispatch(&DESCRIPTOR, 0, &[]),
            Err(Error::BindingCount {
                expected: 1,
                actual: 0,
            })
        ));
    }

    #[test]
    fn rejects_resource_layout_that_does_not_match_f32_backend() {
        let parameters = Box::leak(Box::new([ParameterDescriptor {
            resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
            ..PARAMETERS[0]
        }]));
        let descriptor = DESCRIPTOR.with_parameters(parameters);
        assert!(matches!(
            validate_dispatch(&descriptor, 0, &[F32Binding::ReadWrite(&[])]),
            Err(Error::UnsupportedResourceLayout { parameter, .. }) if parameter == "out"
        ));
    }

    #[test]
    fn renders_descriptor_specific_wgpu_source() {
        let source = render_wgpu_source(&DESCRIPTOR).expect("source should render");
        assert!(source.contains("pub struct TestKernelPipeline"));
        assert!(source.contains("impl TestKernelPipeline"));
        assert!(source.contains("pub fn encode("));
        assert!(source.contains("binding: 0,"));
        assert!(source.contains("read_only: false"));
        assert!(source.contains("gust::slang::compile_wgsl"));
        assert!(source.contains("gust::artifact::verify_artifact(descriptor, artifact)"));
        assert!(source.contains("artifact.format == gust::Format::Wgsl"));
        assert!(source.contains("out: f32 [storage-v1; size=4, align=4, stride=4]"));
        assert!(source.contains("pass.dispatch_workgroups(element_count.div_ceil(64), 1, 1);"));
        // A kernel with a one-dimensional workgroup takes the element count and nothing else (m7.d).
        assert!(!source.contains("extent"), "{source}");
    }

    #[test]
    fn a_kernel_with_a_two_or_three_dimensional_workgroup_renders_a_host_that_takes_an_extent() {
        for (size, dispatch) in [
            (
                [16, 16, 1],
                "pass.dispatch_workgroups(extent[0].div_ceil(16), extent[1].div_ceil(16), extent[2].div_ceil(1));",
            ),
            (
                [4, 4, 4],
                "pass.dispatch_workgroups(extent[0].div_ceil(4), extent[1].div_ceil(4), extent[2].div_ceil(4));",
            ),
            (
                [8, 1, 2],
                "pass.dispatch_workgroups(extent[0].div_ceil(8), extent[1].div_ceil(1), extent[2].div_ceil(2));",
            ),
        ] {
            let descriptor = DESCRIPTOR.with_workgroup_size(size);
            let source = render_wgpu_source(&descriptor).expect("source should render");
            assert!(
                source.contains("        extent: [u32; 3],"),
                "{size:?}: {source}"
            );
            assert!(source.contains(dispatch), "{size:?}: {source}");
            assert!(!source.contains("element_count"), "{size:?}: {source}");
        }
        // A zero axis is refused, as for every dispatch.
        let flat = DESCRIPTOR.with_workgroup_size([16, 0, 1]);
        assert!(matches!(
            render_wgpu_source(&flat),
            Err(Error::UnsupportedWorkgroupSize([16, 0, 1]))
        ));
    }

    #[test]
    fn the_workgroup_constant_is_wgpus_default() {
        assert_eq!(
            DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
            wgpu::Limits::default().max_compute_workgroups_per_dimension
        );
    }

    /// A hand-written `GpuPod` whose layout claims alignment 8 for a four-byte value.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingAlignment(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingAlignment {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingAlignment",
            size: 4,
            alignment: 8,
            stride: 4,
            kind: gust::TypeLayoutKind::Scalar(gust::ScalarKind::U32),
        };
    }

    /// A hand-written `GpuPod` whose layout is a valid StorageV1 struct of two `u32`s (size 8)
    /// while the Rust type is four bytes.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingSize(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingSize {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingSize",
            size: 8,
            alignment: 4,
            stride: 8,
            kind: gust::TypeLayoutKind::Struct(&[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 4,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
            ]),
        };
    }

    /// Right size (4) and alignment (4), but the only field sits at offset 2, so the layout is not
    /// StorageV1: only the `is_storage_v1()` clause of the assertion can refuse it.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LyingOffset(u32);

    // SAFETY: deliberately dishonest, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for LyingOffset {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout {
            name: "LyingOffset",
            size: 4,
            alignment: 4,
            stride: 4,
            kind: gust::TypeLayoutKind::Struct(&[gust::StructFieldLayout {
                name: "a",
                offset: 2,
                ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
            }]),
        };
    }

    #[test]
    fn the_layout_assertion_accepts_honest_types() {
        assert_pod_matches_storage_layout::<u32>();
        assert_pod_matches_storage_layout::<f32>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_layout_that_is_not_storage_v1() {
        assert_pod_matches_storage_layout::<LyingAlignment>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_bad_offset_with_the_right_size_and_alignment() {
        assert_eq!(LyingOffset::LAYOUT.size as usize, size_of::<LyingOffset>());
        assert_eq!(
            LyingOffset::LAYOUT.alignment as usize,
            align_of::<LyingOffset>()
        );
        assert!(!LyingOffset::LAYOUT.is_storage_v1());
        assert_pod_matches_storage_layout::<LyingOffset>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_size_that_differs_from_rust() {
        assert!(LyingSize::LAYOUT.is_storage_v1());
        assert_pod_matches_storage_layout::<LyingSize>();
    }

    /// An honest hand-written `GpuPod` over vectors: `Vec4` (16, align 16) then two `Vec2` at 16 and 24, size 32, no
    /// padding.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct HonestBody {
        a: gust::math::Vec4,
        b: gust::math::Vec2,
        c: gust::math::Vec2,
    }

    // SAFETY: `repr(C)` over three `repr(C)` vector types, no padding (16 + 8 + 8 = 32), offsets 0, 16, 24.
    unsafe impl GpuPod for HonestBody {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "HonestBody",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 16,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "c",
                    offset: 24,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
            ],
        );
    }

    /// A hand-written `GpuPod` that honestly describes padding: a `u32` then a `Vec2` (align 8) at offset 8, so
    /// bytes 4 to 7 are padding. Size 16 and alignment 8 are what Rust reports and what `is_storage_v2` accepts.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct PaddedPair {
        a: u32,
        b: gust::math::Vec2,
    }

    // SAFETY: deliberately breaks the no-padding rule, to exercise the runtime assertion; it is never uploaded.
    unsafe impl GpuPod for PaddedPair {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "PaddedPair",
            16,
            8,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: gust::TypeLayout::scalar("u32", gust::ScalarKind::U32),
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 8,
                    ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
                },
            ],
        );
    }

    #[test]
    fn the_layout_assertion_accepts_vectors_matrices_and_a_padding_free_struct_of_vectors() {
        assert_pod_matches_storage_layout::<gust::math::Vec2>();
        assert_pod_matches_storage_layout::<gust::math::Vec4>();
        assert_pod_matches_storage_layout::<gust::math::Mat2>();
        assert_pod_matches_storage_layout::<gust::math::Mat4>();
        assert_pod_matches_storage_layout::<HonestBody>();
    }

    #[test]
    #[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
    fn the_layout_assertion_refuses_a_struct_with_padding_even_when_the_layout_is_storage_v2() {
        assert!(PaddedPair::LAYOUT.is_storage_v2());
        assert_eq!(PaddedPair::LAYOUT.size as usize, size_of::<PaddedPair>());
        assert_eq!(
            PaddedPair::LAYOUT.alignment as usize,
            align_of::<PaddedPair>()
        );
        assert!(!padding_free(PaddedPair::LAYOUT));
        assert_pod_matches_storage_layout::<PaddedPair>();
    }

    #[test]
    fn the_resource_rule_follows_the_element() {
        use gust::math::{Mat2, Mat4, Vec2, Vec4};
        let scalar_struct = gust::TypeLayout::structure(
            "Pair",
            8,
            4,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <u32 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 4,
                    ty: <f32 as GpuPod>::LAYOUT,
                },
            ],
        );
        for element in [
            <f32 as GpuPod>::LAYOUT,
            <u32 as GpuPod>::LAYOUT,
            scalar_struct,
        ] {
            assert_eq!(
                buffer_resource_layout(element),
                ResourceLayout::storage_v1(element),
                "{}",
                element.name
            );
        }
        for element in [
            <Vec2 as GpuPod>::LAYOUT,
            <Vec4 as GpuPod>::LAYOUT,
            <Mat2 as GpuPod>::LAYOUT,
            <Mat4 as GpuPod>::LAYOUT,
            <HonestBody as GpuPod>::LAYOUT,
        ] {
            assert_eq!(
                buffer_resource_layout(element),
                ResourceLayout::storage_v2(element),
                "{}",
                element.name
            );
        }
    }

    /// `DESCRIPTOR` with its one storage parameter relabelled.
    fn descriptor_with(resource_layout: ResourceLayout) -> KernelDescriptor {
        let parameters = Box::leak(Box::new([ParameterDescriptor {
            resource_layout: Some(resource_layout),
            ..PARAMETERS[0]
        }]));
        DESCRIPTOR.with_parameters(parameters)
    }

    /// Slice mx.p1b: a `pixel` kernel is valid (it runs as a render pass); the callers that only know compute passes
    /// refuse it with a typed error before any compile or pipeline (no device is involved), not a wgpu panic.
    #[test]
    fn a_pixel_kernel_is_valid_and_the_compute_only_callers_refuse_it_by_name() {
        let pixel = descriptor_with(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT))
            .with_shape(Shape::Pixel)
            .with_compatible_shapes(&[Shape::Pixel]);
        assert!(validate_kernel(&pixel).is_ok());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let error = validate_dispatch(&pixel, 0, &[]).unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::ShapeUnsupported {
                        shape: Shape::Pixel,
                        ..
                    }
                ),
                "{error:?}"
            );
            let text = error.to_string();
            assert!(
                text.contains("Pixel") && text.contains("one-shot"),
                "{text}"
            );
            assert!(matches!(
                validate_u32_dispatch(&pixel, 0, &[]).unwrap_err(),
                Error::ShapeUnsupported { .. }
            ));
        }
        assert!(
            validate_kernel(&descriptor_with(ResourceLayout::storage_v1(
                <f32 as GpuPod>::LAYOUT
            )))
            .is_ok()
        );
    }

    #[test]
    fn a_pixel_call_over_more_than_one_dimension_is_refused_before_any_binding_is_looked_at() {
        let pixel = descriptor_with(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT))
            .with_shape(Shape::Pixel)
            .with_compatible_shapes(&[Shape::Pixel]);
        for extent in [Extent::new(4, 2, 1), Extent::new(4, 1, 2)] {
            let error = validate_dispatch_extent(0, false, &pixel, extent, &[], &[]).unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::ShapeUnsupported {
                        shape: Shape::Pixel,
                        ..
                    }
                ),
                "{extent:?}: {error:?}"
            );
        }
    }

    #[test]
    fn raw_compares_the_resource_and_a_buffer_never_equals_a_texture() {
        let device = HeadlessDevice::new().expect("a GPU adapter");
        let first = device.create_f32_buffer("raw first", &[0.0], GpuBufferAccess::ReadWrite);
        let second = device.create_f32_buffer("raw second", &[0.0], GpuBufferAccess::ReadWrite);
        let texture = device
            .create_texture_2d::<gust::texture::R32Float>(2, 2, gust::texture::TextureUse::SAMPLED)
            .expect("a texture");
        let other = device
            .create_texture_2d::<gust::texture::R32Float>(2, 2, gust::texture::TextureUse::SAMPLED)
            .expect("a texture");
        assert!(Raw::Buffer(&first.raw) == Raw::Buffer(&first.raw));
        assert!(Raw::Buffer(&first.raw) != Raw::Buffer(&second.raw));
        assert!(Raw::Texture(texture.as_wgpu_texture()) == Raw::Texture(texture.as_wgpu_texture()));
        assert!(Raw::Texture(texture.as_wgpu_texture()) != Raw::Texture(other.as_wgpu_texture()));
        assert!(Raw::Buffer(&first.raw) != Raw::Texture(texture.as_wgpu_texture()));
    }

    #[test]
    fn a_kernel_is_valid_only_with_the_rule_that_its_element_implies() {
        use gust::math::{Mat4, Vec4};
        for (label, valid) in [
            (ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT), true),
            (ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT), true),
            (ResourceLayout::storage_v2(<Mat4 as GpuPod>::LAYOUT), true),
            // A V2 claim on a scalar, a V1 claim on a vector and a constant-buffer rule on a storage parameter.
            (ResourceLayout::storage_v2(<f32 as GpuPod>::LAYOUT), false),
            (ResourceLayout::storage_v1(<Vec4 as GpuPod>::LAYOUT), false),
            (ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT), false),
            (ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT), false),
        ] {
            let result = validate_kernel(&descriptor_with(label));
            if valid {
                assert!(result.is_ok(), "{label:?}");
            } else {
                assert!(
                    matches!(result, Err(Error::InvalidResourceLayout { .. })),
                    "{label:?}"
                );
            }
        }
    }

    #[test]
    fn a_vector_buffer_does_not_match_the_f32_backend() {
        let descriptor = descriptor_with(ResourceLayout::storage_v2(
            <gust::math::Vec4 as GpuPod>::LAYOUT,
        ));
        assert!(matches!(
            validate_dispatch(&descriptor, 0, &[F32Binding::ReadWrite(&[])]),
            Err(Error::UnsupportedResourceLayout { parameter, .. }) if parameter == "out"
        ));
    }

    #[test]
    fn the_generated_host_text_names_the_storage_v2_rule_of_a_vector_buffer() {
        let descriptor = descriptor_with(ResourceLayout::storage_v2(
            <gust::math::Vec4 as GpuPod>::LAYOUT,
        ));
        let source = render_wgpu_source(&descriptor).expect("source should render");
        assert!(
            source.contains("out: Vec4 [storage-v2; size=16, align=16, stride=16]"),
            "{source}"
        );
    }

    /// `DESCRIPTOR` with its one parameter replaced by a uniform of the given access and layout.
    fn uniform_descriptor(access: Access, resource_layout: ResourceLayout) -> KernelDescriptor {
        let parameters = Box::leak(Box::new([ParameterDescriptor {
            name: "factor",
            rust_type: "Uniform<Vec4>",
            kind: ParameterKind::Uniform,
            access,
            resource_layout: Some(resource_layout),
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
        }]));
        DESCRIPTOR.with_parameters(parameters)
    }

    #[test]
    fn a_uniform_parameter_is_valid_only_as_a_read_only_std140_v1_constant_buffer() {
        use gust::math::{Mat2, Mat4, Vec4};
        for (label, valid) in [
            (ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT), true),
            (ResourceLayout::std140_v1(<Mat4 as GpuPod>::LAYOUT), true),
            (ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT), true),
            // The storage rules on a uniform, even for an element that is valid under both.
            (ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT), false),
            (ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT), false),
            // A constant buffer has no 2x2 matrix.
            (ResourceLayout::std140_v1(<Mat2 as GpuPod>::LAYOUT), false),
        ] {
            let result = validate_kernel(&uniform_descriptor(Access::ReadOnly, label));
            if valid {
                assert!(result.is_ok(), "{label:?}");
            } else {
                assert!(
                    matches!(result, Err(Error::InvalidResourceLayout { .. })),
                    "{label:?}"
                );
            }
        }
        assert!(matches!(
            validate_kernel(&uniform_descriptor(
                Access::ReadWrite,
                ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT)
            )),
            Err(Error::BindingAccess {
                parameter,
                expected: Access::ReadOnly,
                actual: Access::ReadWrite,
            }) if parameter == "factor"
        ));
    }

    #[test]
    fn a_parameter_that_is_neither_storage_nor_uniform_is_still_unsupported() {
        let parameters = Box::leak(Box::new([ParameterDescriptor {
            kind: ParameterKind::Value,
            ..PARAMETERS[0]
        }]));
        let descriptor = DESCRIPTOR.with_parameters(parameters);
        assert!(matches!(
            validate_kernel(&descriptor),
            Err(Error::UnsupportedParameter { name, kind: ParameterKind::Value }) if name == "out"
        ));
    }

    #[test]
    fn the_generated_host_text_names_the_std140_rule_and_the_uniform_binding_of_a_uniform() {
        let descriptor = uniform_descriptor(
            Access::ReadOnly,
            ResourceLayout::std140_v1(<gust::math::Vec4 as GpuPod>::LAYOUT),
        );
        let source = render_wgpu_source(&descriptor).expect("source should render");
        assert!(
            source.contains("factor: Vec4 [std140-v1; size=16, align=16, stride=16]"),
            "{source}"
        );
        assert!(
            source.contains("ty: wgpu::BufferBindingType::Uniform,"),
            "{source}"
        );
        assert!(!source.contains("BufferBindingType::Storage"), "{source}");
    }

    /// A padding-free `Std140V1` struct: `Vec4` then three `f32`s and one `f32` of explicit padding, size 32.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct FullStd140 {
        a: gust::math::Vec4,
        b: [f32; 4],
    }

    // SAFETY: `repr(C)`, a `Vec4` (16 bytes, align 16) then four `f32`s at 16, 20, 24 and 28: no padding, size 32.
    unsafe impl GpuPod for FullStd140 {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "FullStd140",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b0",
                    offset: 16,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b1",
                    offset: 20,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b2",
                    offset: 24,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
                gust::StructFieldLayout {
                    name: "b3",
                    offset: 28,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
            ],
        );
    }

    /// An honest `Std140V1` layout and honest Rust size and alignment, but padded: a `Vec4` then one `f32`, so bytes 20
    /// to 31 are padding.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct PaddedStd140 {
        a: gust::math::Vec4,
        b: f32,
    }

    // SAFETY: deliberately breaks the no-padding rule, to exercise the uniform check; it is never uploaded.
    unsafe impl GpuPod for PaddedStd140 {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "PaddedStd140",
            32,
            16,
            &[
                gust::StructFieldLayout {
                    name: "a",
                    offset: 0,
                    ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
                },
                gust::StructFieldLayout {
                    name: "b",
                    offset: 16,
                    ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
                },
            ],
        );
    }

    /// A struct of one `f32`: its `Std140V1` size is 16 but Rust's is 4.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct OneFloat {
        a: f32,
    }

    // SAFETY: deliberately dishonest about the size, to exercise the uniform check; it is never uploaded.
    unsafe impl GpuPod for OneFloat {
        const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
            "OneFloat",
            16,
            16,
            &[gust::StructFieldLayout {
                name: "a",
                offset: 0,
                ty: gust::TypeLayout::scalar("f32", gust::ScalarKind::F32),
            }],
        );
    }

    fn uniform_refusal<T: GpuPod>() -> (&'static str, &'static str) {
        match check_uniform_type::<T>() {
            Err(Error::InvalidUniformType { type_name, reason }) => (type_name, reason),
            other => panic!("expected an InvalidUniformType error, got {other:?}"),
        }
    }

    #[test]
    fn a_uniform_type_is_a_padding_free_std140_v1_type_with_rusts_size_and_alignment() {
        use gust::math::{Mat4, Vec2, Vec4};
        assert!(check_uniform_type::<f32>().is_ok());
        assert!(check_uniform_type::<Vec2>().is_ok());
        assert!(check_uniform_type::<Vec4>().is_ok());
        assert!(check_uniform_type::<Mat4>().is_ok());
        assert!(check_uniform_type::<FullStd140>().is_ok());
        assert!(check_uniform_type::<HonestBody>().is_ok());
    }

    #[test]
    fn a_padded_uniform_type_is_refused_with_the_fix_in_the_message() {
        let (name, reason) = uniform_refusal::<PaddedStd140>();
        assert_eq!(name, "PaddedStd140");
        assert!(reason.contains("padding bytes"), "{reason}");
        assert!(reason.contains("explicit padding fields"), "{reason}");
        let text = check_uniform_type::<PaddedStd140>()
            .unwrap_err()
            .to_string();
        assert!(
            text.starts_with("`PaddedStd140` cannot be a uniform: "),
            "{text}"
        );
    }

    #[test]
    fn a_uniform_type_whose_layout_is_not_std140_v1_or_not_rusts_is_refused() {
        use gust::math::Mat2;
        // StorageV2 only: a 2x2 matrix, a 4-aligned struct of scalars (not rounded to 16), a padded storage struct.
        for (name, reason) in [
            uniform_refusal::<Mat2>(),
            uniform_refusal::<LyingSize>(),
            uniform_refusal::<PaddedPair>(),
        ] {
            assert!(reason.contains("not Std140V1"), "{name}: {reason}");
        }
        let (name, reason) = uniform_refusal::<OneFloat>();
        assert_eq!(name, "OneFloat");
        assert!(reason.contains("size or alignment"), "{reason}");
    }

    /// A kernel that requires subgroups, with a workgroup of `size` invocations along x.
    fn subgroup_kernel(size: u32) -> KernelDescriptor {
        DESCRIPTOR
            .with_workgroup_size([size, 1, 1])
            .with_requires(&[Capability::Subgroups])
    }

    #[test]
    fn a_requires_capability_the_device_was_not_opened_with_is_missing_capability() {
        let kernel = subgroup_kernel(64);
        // A device opened without the capability refuses the kernel, naming the kernel and the adapter.
        let error = check_kernel_capabilities(&[], Some((32, 32)), "Fake Adapter", &kernel)
            .expect_err("a subgroup kernel on a device without the capability");
        match &error {
            Error::MissingCapability {
                capability,
                adapter,
                kernel: Some(name),
            } => {
                assert_eq!(*capability, Capability::Subgroups);
                assert_eq!(adapter, "Fake Adapter");
                assert_eq!(name, &kernel.qualified_name().to_string());
            }
            other => panic!("expected MissingCapability with a kernel, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` requires the `subgroups` capability, which this device (adapter `Fake Adapter`) was not opened with; open it with `DeviceOptions::with_capabilities`"
        );
        // With the capability the same kernel passes, and a kernel with no `requires` never needs one.
        assert!(
            check_kernel_capabilities(
                &[Capability::Subgroups],
                Some((32, 32)),
                "Fake Adapter",
                &kernel
            )
            .is_ok()
        );
        assert!(check_kernel_capabilities(&[], None, "Fake Adapter", &DESCRIPTOR).is_ok());
    }

    #[test]
    fn a_subgroup_kernel_needs_whole_subgroups_of_the_adapters_largest_size() {
        let have = [Capability::Subgroups];
        // 64 invocations are whole subgroups of 32 and of 64, and so are 128; 96 are not whole subgroups of 64.
        for (size, range, whole) in [
            (64, (32, 32), true),
            (64, (32, 64), true),
            (128, (32, 64), true),
            (96, (32, 32), true),
            (96, (32, 64), false),
            (48, (32, 32), false),
            (64, (8, 8), true),
            (32, (32, 64), false),
        ] {
            let result = check_kernel_capabilities(&have, Some(range), "A", &subgroup_kernel(size));
            assert_eq!(result.is_ok(), whole, "{size} invocations, range {range:?}");
        }
        let error = check_kernel_capabilities(&have, Some((32, 64)), "A", &subgroup_kernel(96))
            .unwrap_err();
        match &error {
            Error::SubgroupSize {
                kernel,
                invocations,
                subgroup_max,
            } => {
                assert_eq!(kernel, "test::kernel");
                assert_eq!((*invocations, *subgroup_max), (96, 64));
            }
            other => panic!("expected SubgroupSize, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` runs 96 invocations per workgroup, which is not a multiple of 64, the largest subgroup size this adapter reports; a subgroup kernel needs full subgroups"
        );
        // The invocation count is the product of the axes, not the x size.
        let tile = DESCRIPTOR
            .with_workgroup_size([8, 8, 1])
            .with_requires(&[Capability::Subgroups]);
        assert!(check_kernel_capabilities(&have, Some((32, 64)), "A", &tile).is_ok());
        // A kernel without `requires` is never judged by subgroup sizes, and an adapter without a range cannot judge.
        assert!(
            check_kernel_capabilities(
                &have,
                Some((32, 64)),
                "A",
                &DESCRIPTOR.with_workgroup_size([48, 1, 1])
            )
            .is_ok()
        );
        assert!(check_kernel_capabilities(&have, None, "A", &subgroup_kernel(48)).is_ok());
    }

    #[test]
    fn a_kernel_built_for_the_cpu_is_refused_before_anything_else() {
        assert!(validate_kernel(&DESCRIPTOR).is_ok());
        let cpu = DESCRIPTOR.with_target(Target::Cpu);
        let error = validate_kernel(&cpu).expect_err("a CPU-target kernel on a GPU device");
        match &error {
            Error::KernelTarget { kernel } => assert_eq!(kernel, "test::kernel"),
            other => panic!("expected KernelTarget, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` is built for the CPU target and cannot run on a GPU device; run it on a `CpuDevice`, or route its tag to `gpu`"
        );
        // First: a CPU kernel with an empty source is still the target's error, not the source's.
        let empty = cpu.with_slang_source("");
        assert!(matches!(
            validate_kernel(&empty),
            Err(Error::KernelTarget { .. })
        ));
    }

    #[test]
    fn uniform_limits_are_checked_under_their_own_names() {
        let limits = wgpu::Limits {
            max_uniform_buffer_binding_size: 64,
            max_uniform_buffers_per_shader_stage: 2,
            ..wgpu::Limits::default()
        };
        assert!(check_binding_size(&limits, &DESCRIPTOR, 64, BindingKind::Uniform).is_ok());
        assert!(matches!(
            check_binding_size(&limits, &DESCRIPTOR, 65, BindingKind::Uniform),
            Err(Error::DeviceLimit {
                limit: "max_uniform_buffer_binding_size",
                requested: 65,
                maximum: 64,
                ..
            })
        ));
        // The same bytes as a storage binding are within the (much larger) storage limit.
        assert!(check_binding_size(&limits, &DESCRIPTOR, 65, BindingKind::Storage).is_ok());
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 1, 2, TextureCounts::NONE).is_ok());
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 3, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_uniform_buffers_per_shader_stage",
                requested: 3,
                maximum: 2,
                ..
            })
        ));
        // A uniform does not count against the storage buffer limit.
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 8, 2, TextureCounts::NONE).is_ok());
    }

    #[test]
    fn binding_size_is_checked_against_the_stored_limit() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 16,
            ..wgpu::Limits::default()
        };
        assert!(check_binding_size(&limits, &DESCRIPTOR, 16, BindingKind::Storage).is_ok());
        assert!(matches!(
            check_binding_size(&limits, &DESCRIPTOR, 17, BindingKind::Storage),
            Err(Error::DeviceLimit {
                kernel,
                limit: "max_storage_buffer_binding_size",
                requested: 17,
                maximum: 16,
            }) if kernel == "test::kernel"
        ));
    }

    #[test]
    fn workgroup_count_rounds_up_and_stops_at_the_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        let count =
            |elements: u32| workgroup_counts(&limits, &DESCRIPTOR, Extent::linear(elements));
        // An empty extent never reaches the count (the dispatch is skipped first); its x count is 0.
        assert_eq!(count(0).unwrap(), [0, 1, 1]);
        assert_eq!(count(1).unwrap(), [1, 1, 1]);
        assert_eq!(count(192).unwrap(), [3, 1, 1]);
        assert!(matches!(
            count(193),
            Err(Error::DeviceLimit {
                limit: "max_compute_workgroups_per_dimension",
                requested: 4,
                maximum: 3,
                ..
            })
        ));
    }

    /// A 16 x 16 kernel for the multi-dimensional unit tests below.
    const TILE: KernelDescriptor = DESCRIPTOR.with_workgroup_size([16, 16, 1]);

    #[test]
    fn workgroup_counts_round_every_axis_and_name_the_axis_that_is_over_the_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        let counts = |x, y, z| workgroup_counts(&limits, &TILE, Extent::new(x, y, z));
        assert_eq!(counts(15, 17, 1).unwrap(), [1, 2, 1]);
        assert_eq!(counts(48, 48, 3).unwrap(), [3, 3, 3]);
        let over = |result: Result<[u32; 3], Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => (limit, requested, maximum),
            other => panic!("expected a DeviceLimit, got {other:?}"),
        };
        assert_eq!(
            over(counts(49, 16, 1)),
            ("max_compute_workgroups_per_dimension", 4, 3)
        );
        assert_eq!(
            over(counts(16, 49, 1)),
            ("max_compute_workgroups_per_dimension (y)", 4, 3)
        );
        assert_eq!(
            over(counts(16, 16, 4)),
            ("max_compute_workgroups_per_dimension (z)", 4, 3)
        );
        // The first axis over the limit is the one named: x before y before z.
        assert_eq!(
            over(counts(49, 49, 4)),
            ("max_compute_workgroups_per_dimension", 4, 3)
        );
    }

    #[test]
    fn the_invocation_cap_is_the_rounded_product_at_most_2_pow_32() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: u32::MAX,
            ..wgpu::Limits::default()
        };
        let counts = |x, y, z| workgroup_counts(&limits, &TILE, Extent::new(x, y, z));
        let cap = |result: Result<[u32; 3], Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => Some((limit, requested, maximum)),
            Ok(_) => None,
            other => panic!("expected a DeviceLimit or Ok, got {other:?}"),
        };
        // Exactly 2^32 after rounding is accepted (not "below"), even when the raw product is smaller.
        assert_eq!(counts(65_536, 65_536, 1).unwrap(), [4096, 4096, 1]);
        assert_eq!(counts(65_536, 65_535, 1).unwrap(), [4096, 4096, 1]);
        // A raw product below 2^32 (65537 * 65535) that rounds up above it is refused.
        assert_eq!(
            cap(counts(65_537, 65_535, 1)),
            Some(("global_invocation_id", 65_552 * 65_536, 1 << 32))
        );
        assert_eq!(
            cap(counts(65_537, 65_536, 1)),
            Some(("global_invocation_id", 65_552 * 65_536, 1 << 32))
        );
        // The rounded product is far beyond a u64 here, and saturates in the error.
        assert_eq!(
            cap(counts(u32::MAX, u32::MAX, u32::MAX)),
            Some(("global_invocation_id", u64::MAX, 1 << 32))
        );
    }

    #[test]
    fn kernel_limits_name_the_first_exceeded_limit() {
        let mut limits = wgpu::Limits::default();
        assert!(check_kernel_limits(&limits, &DESCRIPTOR, 8, 0, TextureCounts::NONE).is_ok());
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 9, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 9,
                maximum: 8,
                ..
            })
        ));
        limits.max_compute_invocations_per_workgroup = 32;
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_compute_invocations_per_workgroup",
                requested: 64,
                maximum: 32,
                ..
            })
        ));
        limits.max_compute_workgroup_size_x = 32;
        assert!(matches!(
            check_kernel_limits(&limits, &DESCRIPTOR, 1, 0, TextureCounts::NONE),
            Err(Error::DeviceLimit {
                limit: "max_compute_workgroup_size_x",
                requested: 64,
                maximum: 32,
                ..
            })
        ));
    }

    #[test]
    fn kernel_limits_check_the_y_and_z_sizes_and_the_checked_product() {
        let limits = wgpu::Limits::default();
        let kernel = |size| DESCRIPTOR.with_workgroup_size(size);
        let check = |limits: &wgpu::Limits, size| {
            check_kernel_limits(limits, &kernel(size), 1, 0, TextureCounts::NONE)
        };
        let named = |result: Result<(), Error>| match result {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => (limit, requested, maximum),
            other => panic!("expected a DeviceLimit, got {other:?}"),
        };
        assert!(check(&limits, [16, 16, 1]).is_ok());
        assert!(check(&limits, [1, 1, 64]).is_ok());
        assert_eq!(
            named(check(&limits, [1, 257, 1])),
            ("max_compute_workgroup_size_y", 257, 256)
        );
        assert_eq!(
            named(check(&limits, [1, 1, 65])),
            ("max_compute_workgroup_size_z", 65, 64)
        );
        // 16 * 16 * 2 = 512 invocations: every axis is within its limit and the product is not.
        assert_eq!(
            named(check(&limits, [16, 16, 2])),
            ("max_compute_invocations_per_workgroup", 512, 256)
        );
        // The axes are checked in the order x, y, z, before the product.
        assert_eq!(
            named(check(&limits, [257, 257, 65])),
            ("max_compute_workgroup_size_x", 257, 256)
        );
        assert_eq!(
            named(check(&limits, [1, 257, 65])),
            ("max_compute_workgroup_size_y", 257, 256)
        );
        // A product beyond a u64 is an `Error::Extent`, not a wrapped number.
        let wide = wgpu::Limits {
            max_compute_workgroup_size_x: u32::MAX,
            max_compute_workgroup_size_y: u32::MAX,
            max_compute_workgroup_size_z: u32::MAX,
            ..limits
        };
        assert!(matches!(
            check(&wide, [u32::MAX, u32::MAX, u32::MAX]),
            Err(Error::Extent { extent, .. }) if extent == [u64::from(u32::MAX); 3]
        ));
    }

    #[test]
    fn device_limit_error_text_names_kernel_limit_and_remedy() {
        let error = device_limit(
            &DESCRIPTOR,
            "max_compute_workgroups_per_dimension",
            65536,
            65535,
        );
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` needs 65536 for `max_compute_workgroups_per_dimension`, the limit is 65535; \
             split the work into several dispatches, or spread it over y and z with a multi-dimensional extent"
        );
        let error = device_limit(&DESCRIPTOR, "max_storage_buffer_binding_size", 20, 16);
        assert_eq!(
            error.to_string(),
            "kernel `test::kernel` needs 20 for `max_storage_buffer_binding_size`, the limit is 16"
        );
        // The y and z counts name their axis and get a remedy of their own; neither mentions a roadmap.
        for axis in ["y", "z"] {
            let limit: &'static str = Box::leak(
                format!("max_compute_workgroups_per_dimension ({axis})").into_boxed_str(),
            );
            let text = device_limit(&DESCRIPTOR, limit, 4, 3).to_string();
            assert_eq!(
                text,
                format!(
                    "kernel `test::kernel` needs 4 for `{limit}`, the limit is 3; split the work into several \
                     dispatches, or use fewer invocations along that axis"
                )
            );
            assert!(!text.contains("roadmap"), "{text}");
        }
        // A workgroup size with a zero axis says so; it does not claim the backend is one-dimensional.
        assert_eq!(
            Error::UnsupportedWorkgroupSize([16, 0, 1]).to_string(),
            "workgroup size [16, 0, 1] has a zero axis; every axis must be at least 1"
        );
    }
}
