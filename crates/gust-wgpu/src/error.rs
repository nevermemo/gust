use std::{error, fmt};

use gust::{
    Access, BindingKind, ParameterKind, ResourceLayout, Shape, capability::Capability,
    target::Target,
};

use crate::GpuBufferAccess;

/// Label `Error::AliasedBinding` carries as `second` when the buffer is the arguments buffer of an
/// indirect dispatch in a staged graph. It stands where a parameter name would.
pub(crate) const INDIRECT_ARGUMENTS: &str = "<indirect dispatch arguments>";

/// Every way a `gust-compute-wgpu` call can refuse or fail: opening a device, validating a kernel and its bindings, building
/// a graph, and reading results back. A launch or a graph is validated before it is encoded, so a validation error
/// means none of its work was submitted.
///
/// The enum is `#[non_exhaustive]`: a later release can add a variant without a breaking change, so a `match` in
/// another crate needs a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// No adapter could be used. The text is the reason: wgpu's error, no adapter that matches the requested
    /// `AdapterChoice` among those the backends offer, or the default device not being open yet. Opening a device
    /// retries this variant, and only this one.
    NoAdapter(String),
    /// An adapter was found but wgpu would not open a device on it with the requested features and limits. The text
    /// is wgpu's error.
    RequestDevice(String),
    /// The kernel descriptor carries no generated Slang source, so there is nothing to compile or to validate. The
    /// text is the kernel's qualified name.
    MissingSlangSource(String),
    /// A kernel's Slang could not be compiled, reflected or verified; the inner error says how. Three paths return it:
    /// the run-time compile and reflect (`slang-runtime`; a `slangc` failure arrives as
    /// `reflect::Error::Compiler`), the cross-check of the reflection against the descriptor (D17), and the
    /// verification of a precompiled artifact.
    Reflection(gust::reflect::Error),
    /// The kernel has no precompiled WGSL artifact and this build cannot compile one at run time: the
    /// `slang-runtime` feature of `gust-compute-wgpu` is off (D44), and with it the only compiler this crate could start.
    /// `profile` is the build profile of `gust` (D44). Remedy: run `cargo gust build` in the kernel's crate and
    /// rebuild, so that the artifact is embedded; or, in a `dev` build, enable the `slang-runtime` feature.
    NoArtifact {
        /// The kernel's qualified name.
        kernel: String,
        /// The build profile of `gust`.
        profile: &'static str,
    },
    /// A kernel's workgroup size has a zero axis; every axis of `workgroup_size` must be at least 1. (The size may
    /// have y and z above 1; whether the device allows it is a [`Error::DeviceLimit`].)
    UnsupportedWorkgroupSize([u32; 3]),
    /// The kernel's execution shape cannot be used the way this call asks (plan §4.3 spells it `UnsupportedShape`).
    /// `grid` runs as compute passes and `pixel` as full-screen fragment passes (slice mx.p1b); a `pixel` kernel is
    /// refused, by `reason`, for an extent call (a pixel call is one-dimensional), an indirect dispatch (the element
    /// count must be known on the host), the v1 `dispatch_f32`/`dispatch_u32` family, the generated host snippet, and
    /// on an adapter whose fragment stage cannot write storage buffers. Reported before any pipeline or wgpu object is
    /// created.
    ShapeUnsupported {
        /// The kernel's qualified name.
        kernel: String,
        /// The kernel's shape.
        shape: Shape,
        /// Why the kernel's shape cannot run here.
        reason: &'static str,
    },
    /// An output of a `pixel` kernel (a read-write storage parameter) was bound with an independent length. A `pixel`
    /// output holds exactly the call's element count (`LengthRule::Shared`). Remedy: drop `Independent`.
    ShapeIndependentOutput {
        /// The kernel's qualified name.
        kernel: String,
        /// The output parameter that was bound with an independent length.
        parameter: String,
    },
    /// A parameter has a resource kind this call cannot bind, for example a `Value` parameter, or a texture or sampler
    /// in a staged graph, which tracks hazards by buffer only.
    UnsupportedParameter {
        /// The parameter's name.
        name: String,
        /// The kind that this call cannot bind.
        kind: ParameterKind,
    },
    /// The kernel's own parameter declares no resource layout, or one that does not fit its kind (`StorageV1` or
    /// `StorageV2` for a storage buffer, `Std140V1` for a uniform, `TextureV1` for a texture or sampler, D91).
    /// Reported when the kernel is validated, and when a texture layout is built for a parameter that has no sample
    /// type on this device or no storage format.
    InvalidResourceLayout {
        /// The parameter's name.
        parameter: String,
        /// The layout the parameter declares, or `None` when it declares none.
        actual: Option<Box<ResourceLayout>>,
    },
    /// A resource of one kind was bound to a parameter of the other: a storage buffer to a uniform parameter, or a
    /// uniform to a storage one. The typed `call` rules this out at compile time; a hand-built `Call` is caught here.
    BindingKind {
        /// The parameter's name.
        parameter: String,
        /// The kind of resource the parameter takes.
        expected: BindingKind,
        /// The kind of the resource that was bound.
        actual: BindingKind,
    },
    /// `HeadlessDevice::create_uniform` refused the element type: it is not a padding-free `Std140V1` type. `reason`
    /// names what is wrong and the fix.
    InvalidUniformType {
        /// The name of the rejected element type.
        type_name: &'static str,
        /// What is wrong with the type, and the fix.
        reason: &'static str,
    },
    /// The resource bound to a parameter has another layout than the parameter declares (`expected`); `actual` is the
    /// layout of what was bound (always `Some` today).
    UnsupportedResourceLayout {
        /// The parameter's name.
        parameter: String,
        /// The layout the parameter declares.
        expected: Box<ResourceLayout>,
        /// The layout of the resource that was bound.
        actual: Option<Box<ResourceLayout>>,
    },
    /// A kernel parameter sits in a bind group other than 0, the only one this backend supports. The number is the
    /// group.
    UnsupportedBindGroup(u32),
    /// The call binds another number of resources (or of samplers) than the kernel declares.
    BindingCount {
        /// The number of resources (or samplers) the kernel declares.
        expected: usize,
        /// The number the call binds.
        actual: usize,
    },
    /// A parameter was bound with another access than it declares, or a kernel declares a uniform parameter that is
    /// not read-only (`expected` is then read-only and no binding is involved).
    BindingAccess {
        /// The parameter's name.
        parameter: String,
        /// The access the parameter declares.
        expected: Access,
        /// The access it was bound with.
        actual: Access,
    },
    /// A buffer holds another number of elements than the parameter expects (its length rule is shared, not
    /// independent).
    BufferLength {
        /// The parameter's name.
        parameter: String,
        /// The element count the parameter expects.
        expected: usize,
        /// The element count of the buffer that was bound.
        actual: usize,
    },
    /// A resource (buffer, uniform, pool, texture or sampler) is used with a device other than the one that created
    /// it.
    ForeignPersistentBuffer,
    /// One buffer is bound to two parameters of one dispatch and at least one of them is
    /// read-write. Reported before any wgpu object is created or anything is written. Binding the
    /// same buffer twice is accepted only when both bindings are read-only. `first` precedes
    /// `second` in declaration order. Remedy: bind the buffer once, or use separate buffers.
    ///
    /// In a staged graph, an indirect dispatch node whose arguments buffer is also bound read-write
    /// reports that parameter as `first` and the label `<indirect dispatch arguments>` as `second`
    /// (wgpu does not allow the GPU to read a buffer as indirect arguments while the same dispatch
    /// writes it). Binding the arguments buffer read-only is accepted.
    AliasedBinding {
        /// The kernel's qualified name.
        kernel: String,
        /// The first of the two parameters, in declaration order.
        first: String,
        /// The second of the two parameters, or the indirect-arguments label in a staged graph.
        second: String,
    },
    /// A persistent buffer is bound with an access its own access (`buffer`) does not allow, for example a buffer
    /// created read-only bound as read-write.
    PersistentBufferAccess {
        /// The parameter's name.
        parameter: String,
        /// The access the buffer was created with.
        buffer: GpuBufferAccess,
        /// The access the parameter asks for.
        binding: Access,
    },
    /// A host slice written to a persistent buffer, or a graph upload to one, does not match the buffer's element
    /// count.
    PersistentBufferLength {
        /// The buffer's element count.
        expected: usize,
        /// The element count of the host slice or graph upload.
        actual: usize,
    },
    /// A readback was asked of a persistent buffer that was not created with read-write access.
    PersistentBufferNotReadable,
    /// A parameter was bound with an independent length but the buffer is empty while the call's extent is not; the GPU
    /// would see a one-element placeholder.
    IndependentLengthEmpty {
        /// The parameter's name.
        parameter: String,
    },
    /// A graph node depends on a node that is not an earlier node of the same graph.
    GraphDependencyOrder {
        /// The index of the node.
        node: usize,
        /// The index of the node it depends on, which is not an earlier one.
        dependency: usize,
    },
    /// A graph node touches a buffer that an earlier node (`producer`, the earliest such node) also touches, with at
    /// least one of the two accesses read-write, without depending on it directly or through other nodes. Two reads
    /// never conflict.
    GraphMissingDependency {
        /// The index of the node.
        node: usize,
        /// The index of the earlier node that touches the same buffer.
        producer: usize,
    },
    /// A graph readback was asked for a node that is not a readback node of that execution.
    GraphUnknownReadback(usize),
    /// A graph readback holds one element type and another was requested. Both are element type names.
    GraphReadbackType {
        /// The name of the element type the readback holds.
        expected: &'static str,
        /// The name of the element type that was requested.
        actual: &'static str,
    },
    /// An indirect-arguments buffer is not a non-empty struct of exactly three `u32` fields. `create_indirect_buffer`
    /// returns it for another element type, and a graph returns it for an indirect dispatch whose arguments buffer was
    /// not made by `create_indirect_buffer` or is empty. The text is the name of the struct that was given.
    IndirectArgsLayout(&'static str),
    /// A parameter of an indirect dispatch is not bound with an independent length and a non-empty buffer.
    IndirectBindingLength {
        /// The parameter's name.
        parameter: String,
    },
    /// A pool was asked to truncate to more elements (`requested`) than its logical length (`len`).
    PoolTruncateGrows {
        /// The pool's logical length.
        len: usize,
        /// The length that was asked for.
        requested: usize,
    },
    /// A launch needs more than a limit of the device (or of the limits installed with
    /// `HeadlessDevice::with_limits`) allows. A single dispatch reports it before any
    /// pipeline is compiled, looked up in the cache or encoded. A multi-node graph
    /// validates every node, in order, before anything is compiled, cached or encoded
    /// (the limit pass is part of `validate`), so the error belongs to the first node
    /// that fails and no node, earlier ones included, has compiled anything, created
    /// an encoder or submitted work. An indirect dispatch is never
    /// checked against `max_compute_workgroups_per_dimension`, because the host does
    /// not see its workgroup count; every other limit GUST checks applies to it.
    ///
    /// `kernel` is the qualified name of the kernel; for a limit met outside a launch it names the operation
    /// (`create_texture_2d 5x1`, a buffer label).
    DeviceLimit {
        /// The kernel's qualified name, or the operation that met the limit.
        kernel: String,
        /// The name of the limit, as wgpu spells it.
        limit: &'static str,
        /// The value the launch needs.
        requested: u64,
        /// The value the device allows.
        maximum: u64,
    },
    /// `HeadlessDevice::with_limits` may only lower limits; `limit` was raised above the
    /// value the device was opened with.
    LimitRaised {
        /// The name of the limit, as wgpu spells it.
        limit: &'static str,
        /// The value that was asked for.
        requested: u64,
        /// The value the device was opened with.
        device: u64,
    },
    /// Waiting for the device failed: the poll returned an error, the device was lost, or the wait did not finish. Also
    /// returned when an allocation's error scope caught a wgpu error other than out-of-memory, or did not resolve. The
    /// text is the reason.
    Poll(String),
    /// A readback failed: the buffer could not be mapped (including a mapping that did not complete within the bounded
    /// wait), or the mapped bytes cannot be decoded as the requested element type. The text is the reason.
    Map(String),
    /// A blocking wait was asked for on `wasm32`, where the browser cannot block: `WgpuJob::wait` and
    /// `WgpuJob::wait_timeout` always return this there, and so do the calls built on them (the generated `dispatch`
    /// and `dispatch_extent` of a kernel, `gust::Run::dispatch_call`, `gust::Device::dispatch`). The text is the
    /// `WgpuJob` method that refused (`wait` or `wait_timeout`). Any work the job submitted has been submitted and will
    /// run; only the wait is refused. To synchronise, await a readback with `HeadlessDevice::read_typed_buffer_async`
    /// (it resolves when the work and the copy are done), or read `WgpuJob::status` without waiting. Never returned on
    /// native targets.
    WouldBlock(&'static str),
    /// A buffer or texture allocation failed inside an out-of-memory error scope: the request was within the
    /// device's limits (`max_buffer_size` for a buffer, `max_texture_dimension_2d` for a texture), but the adapter had
    /// no memory for it. `label` names the buffer or texture (a `pixel` call's attachment is
    /// `GUST pixel attachment`) and `bytes` is the size asked for. Where a device cannot report this through its
    /// scope and is lost instead, S7.6 outcome B maps it to [`Error::Poll`].
    OutOfMemory {
        /// The label of the buffer or texture.
        label: String,
        /// The size that was asked for, in bytes.
        bytes: u64,
    },
    /// Two resident buffers of one operation belong to different devices (compared by `HeadlessDevice::id`).
    MixedDevices {
        /// The id of the device the operation runs on.
        first: u64,
        /// The id of the device the other buffer belongs to.
        second: u64,
    },
    /// A dispatch extent is unusable: a product overflowed, so no launch can be described (the extent's `x * y * z`, or
    /// the kernel's workgroup size multiplied out), or the dispatch is of a kind that supports one-dimensional extents
    /// only (a staged graph node with a y or z extent above 1). `reason` names the failing step.
    Extent {
        /// The extent as `[x, y, z]`, or the kernel's workgroup size when that overflowed (`reason` says which).
        extent: [u64; 3],
        /// The step that failed.
        reason: &'static str,
    },
    /// `set_default_options` was called after the default device had been opened successfully: the options can no
    /// longer change what it was opened with. (After a failed open they can still be set, and the next call retries.)
    DefaultDeviceOpened,
    /// A texture is bound (or created) for a use it does not have: a sampled parameter needs `TextureUse::SAMPLED`,
    /// a storage one `TextureUse::STORAGE`. `required` is what the parameter needs and `actual` what the texture
    /// offers (for `create_texture_2d`: what the caller asked for, and what the format allows on the device).
    TextureUse {
        /// The parameter's name, or the operation that met the use.
        parameter: String,
        /// The use the parameter needs.
        required: gust::TextureUse,
        /// The uses the texture offers.
        actual: gust::TextureUse,
    },
    /// A texture of the wrong format is bound: a storage parameter needs its exact format, a sampled one a format of
    /// its sample class. `expected` says which.
    TextureFormat {
        /// The parameter's name, or the operation that met the format.
        parameter: String,
        /// What the parameter needs, in words.
        expected: String,
        /// The format of the texture.
        actual: gust::texture::Format,
    },
    /// A sampler that filters linearly is bound on a device without the `float32-filterable` feature, which a linear
    /// filter of a 32-bit float texture needs. `feature` names what is missing. Remedy: open the device with
    /// `FeaturesPolicy::Adapter` on an adapter that offers it, or bind a `Filter::Nearest` sampler.
    TextureFilter {
        /// The parameter's name.
        parameter: String,
        /// The filter of the sampler.
        filter: gust::texture::Filter,
        /// The device feature that is missing.
        feature: &'static str,
    },
    /// A texture's size is unusable: a zero side, or a host slice that does not hold `width * height` texels.
    TextureSize {
        /// The width in texels.
        width: u32,
        /// The height in texels.
        height: u32,
        /// What is wrong with the size.
        reason: String,
    },
    /// `create_sampler` was given a filter or an address mode this backend has no mapping for (both enums are
    /// `#[non_exhaustive]`, so a newer `gust` can name one).
    SamplerOptions {
        /// The filter that was given.
        filter: gust::texture::Filter,
        /// The address mode that was given.
        address: gust::texture::Address,
    },
    /// A device capability is missing (M8, D23, D142). With `kernel: None` it is a refusal at open: the device options
    /// request `capability` (`DeviceOptions::with_capabilities`) and the adapter, or the browser, does not offer it, so
    /// no device exists. With `kernel: Some(..)` it is a refusal at launch: the kernel's `requires` names `capability`
    /// and this device was not opened with it. `adapter` is the adapter's name.
    MissingCapability {
        /// The capability that is missing.
        capability: Capability,
        /// The name of the adapter the device is, or would have been, opened on.
        adapter: String,
        /// The kernel that requires the capability, or `None` when the refusal is the device's own.
        kernel: Option<String>,
    },
    /// A kernel built for another target (`KernelDescriptor::target` is not `Target::Gpu`, today `Target::Cpu`) reached a
    /// GPU device. The typed entry points refuse this at compile time (the `GpuKernel` bound); this is the run-time
    /// refusal for a kernel passed by its descriptor, and nothing is submitted.
    KernelTarget {
        /// The kernel's qualified name.
        kernel: String,
        /// The target the descriptor is built for.
        target: Target,
    },
    /// A kernel that requires subgroups has a workgroup whose invocation count is not a multiple of the largest subgroup
    /// size wgpu reports for the adapter, so some subgroup would run partly empty. Only full subgroups are supported
    /// (D140).
    SubgroupSize {
        /// The kernel's qualified name.
        kernel: String,
        /// The invocations in one workgroup (the product of the workgroup size).
        invocations: u64,
        /// The largest subgroup size wgpu reports for the adapter (`AdapterInfo::subgroup_max_size`).
        subgroup_max: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(error) => {
                write!(formatter, "no Vulkan compute adapter is available: {error}")
            }
            Self::RequestDevice(error) => {
                write!(formatter, "could not open the GPU device: {error}")
            }
            Self::MissingSlangSource(kernel) => {
                write!(formatter, "kernel `{kernel}` has no generated Slang source")
            }
            Self::Reflection(error) => {
                write!(formatter, "could not validate kernel reflection: {error}")
            }
            Self::NoArtifact { kernel, profile } => write!(
                formatter,
                "kernel `{kernel}` has no precompiled WGSL artifact, and this build of gust-compute-wgpu (profile `{profile}`) was built without the `slang-runtime` feature, so it cannot compile the kernel at run time; run `cargo gust build` in the crate that defines the kernel and rebuild, or enable the `slang-runtime` feature in a dev build",
            ),
            Self::ShapeUnsupported {
                kernel,
                shape,
                reason,
            } => write!(
                formatter,
                "kernel `{kernel}` has the shape {shape:?}, which cannot be used here: {reason}"
            ),
            Self::ShapeIndependentOutput { kernel, parameter } => write!(
                formatter,
                "output `{parameter}` of the pixel kernel `{kernel}` was bound with an independent length; a pixel output holds exactly the call's element count (drop `Independent`)"
            ),
            Self::UnsupportedWorkgroupSize(size) => {
                write!(
                    formatter,
                    "workgroup size {size:?} has a zero axis; every axis must be at least 1"
                )
            }
            Self::UnsupportedParameter { name, kind } => {
                write!(
                    formatter,
                    "parameter `{name}` has unsupported resource kind {kind:?}"
                )
            }
            Self::InvalidResourceLayout { parameter, actual } => write!(
                formatter,
                "parameter `{parameter}` requires a valid resource layout for its kind (StorageV1 or StorageV2 for a storage buffer, Std140V1 for a uniform), got {actual:?}"
            ),
            Self::BindingKind {
                parameter,
                expected,
                actual,
            } => write!(
                formatter,
                "parameter `{parameter}` expects a {expected:?} binding, got a {actual:?} one",
            ),
            Self::InvalidUniformType { type_name, reason } => {
                write!(formatter, "`{type_name}` cannot be a uniform: {reason}",)
            }
            Self::UnsupportedResourceLayout {
                parameter,
                expected,
                actual,
            } => write!(
                formatter,
                "parameter `{parameter}` has resource layout {actual:?}, but the f32 backend requires {expected:?}",
            ),
            Self::UnsupportedBindGroup(group) => {
                write!(
                    formatter,
                    "only bind group 0 is supported, got group {group}"
                )
            }
            Self::BindingCount { expected, actual } => {
                write!(
                    formatter,
                    "kernel expects {expected} resource bindings, got {actual}"
                )
            }
            Self::BindingAccess {
                parameter,
                expected,
                actual,
            } => write!(
                formatter,
                "parameter `{parameter}` expects {expected:?} access, got {actual:?}",
            ),
            Self::BufferLength {
                parameter,
                expected,
                actual,
            } => write!(
                formatter,
                "parameter `{parameter}` expects {expected} elements, got {actual}",
            ),
            Self::ForeignPersistentBuffer => {
                write!(
                    formatter,
                    "persistent buffer belongs to a different GPU device"
                )
            }
            Self::AliasedBinding {
                kernel,
                first,
                second,
            } if second == INDIRECT_ARGUMENTS => write!(
                formatter,
                "the dispatch-arguments buffer of kernel `{kernel}` is also bound read-write by parameter `{first}`, which wgpu does not allow while the GPU reads it as indirect arguments (bind `{first}` read-only, or use a separate arguments buffer)",
            ),
            Self::AliasedBinding {
                kernel,
                first,
                second,
            } => write!(
                formatter,
                "parameters `{first}` and `{second}` of kernel `{kernel}` are the same buffer and at least one of them is written; a buffer may be bound twice only read-only (bind it once, or use separate buffers)",
            ),
            Self::PersistentBufferAccess {
                parameter,
                buffer,
                binding,
            } => write!(
                formatter,
                "parameter `{parameter}` was bound as {binding:?}, which the {buffer:?} persistent buffer does not allow",
            ),
            Self::PersistentBufferLength { expected, actual } => write!(
                formatter,
                "persistent buffer expects {expected} elements, got {actual}",
            ),
            Self::PersistentBufferNotReadable => write!(
                formatter,
                "readback requires a persistent buffer created with read-write access",
            ),
            Self::IndependentLengthEmpty { parameter } => write!(
                formatter,
                "parameter `{parameter}` was bound with an independent length but the buffer is empty; the GPU would observe a one-element placeholder",
            ),
            Self::GraphDependencyOrder { node, dependency } => write!(
                formatter,
                "graph node {node} depends on node {dependency}, which is not an earlier node of the same graph",
            ),
            Self::GraphMissingDependency { node, producer } => write!(
                formatter,
                "graph node {node} touches a buffer last written or read by node {producer} without declaring a dependency on it",
            ),
            Self::GraphUnknownReadback(node) => write!(
                formatter,
                "graph node {node} is not a readback node of this execution",
            ),
            Self::GraphReadbackType { expected, actual } => write!(
                formatter,
                "graph readback holds `{expected}` elements, but `{actual}` was requested",
            ),
            Self::IndirectArgsLayout(name) => write!(
                formatter,
                "indirect dispatch arguments must come from `create_indirect_buffer` with a non-empty struct of exactly three `u32` fields (x, y, z workgroups); got `{name}`",
            ),
            Self::IndirectBindingLength { parameter } => write!(
                formatter,
                "parameter `{parameter}` of an indirect dispatch must be bound with `independent_length()` and a non-empty buffer; the element count is decided on the GPU",
            ),
            Self::PoolTruncateGrows { len, requested } => write!(
                formatter,
                "pool truncate to {requested} exceeds the logical length {len}; use push or reserve to grow",
            ),
            Self::DeviceLimit {
                kernel,
                limit,
                requested,
                maximum,
            } => {
                // A buffer allocation reports its label in `kernel` (the pre-check of the fallible buffer functions).
                let what = if *limit == "max_buffer_size" {
                    "buffer"
                } else {
                    "kernel"
                };
                write!(
                    formatter,
                    "{what} `{kernel}` needs {requested} for `{limit}`, the limit is {maximum}"
                )?;
                // The remedy of a workgroup count over the limit: x names no axis, y and z end in `(y)` and `(z)`.
                if *limit == "max_compute_workgroups_per_dimension" {
                    write!(
                        formatter,
                        "; split the work into several dispatches, or spread it over y and z with a \
                         multi-dimensional extent"
                    )?;
                } else if limit.starts_with("max_compute_workgroups_per_dimension (") {
                    write!(
                        formatter,
                        "; split the work into several dispatches, or use fewer invocations along that axis"
                    )?;
                }
                Ok(())
            }
            Self::LimitRaised {
                limit,
                requested,
                device,
            } => write!(
                formatter,
                "`with_limits` may only lower limits: `{limit}` is {requested}, which is better than the {device} the device was opened with",
            ),
            Self::Poll(error) => write!(formatter, "GPU synchronization failed: {error}"),
            Self::Map(error) => write!(formatter, "GPU readback failed: {error}"),
            Self::WouldBlock(what) => write!(
                formatter,
                "`{what}` would block, which the browser cannot do; any work submitted will run: await a \
                 readback (`read_typed_buffer_async`) to synchronise, or read the job's `status`",
            ),
            Self::OutOfMemory { label, bytes } => write!(
                formatter,
                "the adapter is out of memory allocating `{label}` ({bytes} bytes)",
            ),
            Self::MixedDevices { first, second } => write!(
                formatter,
                "resident buffers of devices {first} and {second} cannot be used together; every resident argument of one operation must share one device",
            ),
            Self::Extent { extent, reason } => {
                write!(
                    formatter,
                    "dispatch extent {extent:?} is unusable: {reason}"
                )
            }
            Self::DefaultDeviceOpened => write!(
                formatter,
                "the default device is already open; call `set_default_options` before the first `default_device()`",
            ),
            Self::TextureUse {
                parameter,
                required,
                actual,
            } => write!(
                formatter,
                "`{parameter}` needs the texture use {required:?}, and the texture offers {actual:?}",
            ),
            Self::TextureFormat {
                parameter,
                expected,
                actual,
            } => write!(
                formatter,
                "`{parameter}` needs {expected}, got a {actual} texture",
            ),
            Self::TextureFilter {
                parameter,
                filter,
                feature,
            } => write!(
                formatter,
                "`{parameter}` filters with {filter:?}, which needs the `{feature}` device feature; open the device with `FeaturesPolicy::Adapter` on an adapter that offers it, or bind a Nearest sampler",
            ),
            Self::TextureSize {
                width,
                height,
                reason,
            } => write!(formatter, "texture of {width}x{height} texels: {reason}"),
            Self::SamplerOptions { filter, address } => write!(
                formatter,
                "this backend maps no sampler with filter {filter:?} and address mode {address:?}",
            ),
            Self::MissingCapability {
                capability,
                adapter,
                kernel: None,
            } => write!(
                formatter,
                "the adapter `{adapter}` does not offer the `{}` capability that the device options request",
                capability.name(),
            ),
            Self::MissingCapability {
                capability,
                adapter,
                kernel: Some(kernel),
            } => write!(
                formatter,
                "kernel `{kernel}` requires the `{}` capability, which this device (adapter `{adapter}`) was not opened with; open it with `DeviceOptions::with_capabilities`",
                capability.name(),
            ),
            Self::KernelTarget { kernel, target } => write!(
                formatter,
                "kernel `{kernel}` is built for the `{}` target and cannot run on a GPU device; run it on a `CpuDevice`, or route its tag to `gpu`",
                target.name(),
            ),
            Self::SubgroupSize {
                kernel,
                invocations,
                subgroup_max,
            } => write!(
                formatter,
                "kernel `{kernel}` runs {invocations} invocations per workgroup, which is not a multiple of {subgroup_max}, the largest subgroup size wgpu reports for this adapter; a subgroup kernel needs full subgroups",
            ),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Reflection(error) => Some(error),
            _ => None,
        }
    }
}
