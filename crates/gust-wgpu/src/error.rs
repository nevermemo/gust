use std::{error, fmt};

use gust::{Access, BindingKind, ParameterKind, ResourceLayout};

use crate::GpuBufferAccess;

/// Label `Error::AliasedBinding` carries as `second` when the buffer is the arguments buffer of an
/// indirect dispatch in a staged graph. It stands where a parameter name would.
pub(crate) const INDIRECT_ARGUMENTS: &str = "<indirect dispatch arguments>";

#[derive(Debug)]
pub enum Error {
    NoAdapter(String),
    RequestDevice(String),
    MissingSlangSource(String),
    Slang(gust::slang::Error),
    Reflection(gust::reflect::Error),
    /// The kernel has no precompiled WGSL artifact and this build cannot compile one at run time: the
    /// `slang-runtime` feature of `gust-wgpu` is off (D44), and with it the only compiler this crate could start.
    /// `profile` is the build profile of `gust` (D44). Remedy: run `cargo gust build` in the kernel's crate and
    /// rebuild, so that the artifact is embedded; or, in a `dev` build, enable the `slang-runtime` feature.
    NoArtifact {
        kernel: String,
        profile: &'static str,
    },
    /// A kernel's workgroup size has a zero axis; every axis of `workgroup_size` must be at least 1. (The size may
    /// have y and z above 1; whether the device allows it is a [`Error::DeviceLimit`].)
    UnsupportedWorkgroupSize([u32; 3]),
    UnsupportedParameter {
        name: String,
        kind: ParameterKind,
    },
    InvalidResourceLayout {
        parameter: String,
        actual: Option<Box<ResourceLayout>>,
    },
    /// A resource of one kind was bound to a parameter of the other: a storage buffer to a uniform parameter, or a
    /// uniform to a storage one. The typed `call` rules this out at compile time; a hand-built `Call` is caught here.
    BindingKind {
        parameter: String,
        expected: BindingKind,
        actual: BindingKind,
    },
    /// `HeadlessDevice::create_uniform` refused the element type: it is not a padding-free `Std140V1` type. `reason`
    /// names what is wrong and the fix.
    InvalidUniformType {
        type_name: &'static str,
        reason: &'static str,
    },
    UnsupportedResourceLayout {
        parameter: String,
        expected: Box<ResourceLayout>,
        actual: Option<Box<ResourceLayout>>,
    },
    UnsupportedBindGroup(u32),
    BindingCount {
        expected: usize,
        actual: usize,
    },
    BindingAccess {
        parameter: String,
        expected: Access,
        actual: Access,
    },
    BufferLength {
        parameter: String,
        expected: usize,
        actual: usize,
    },
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
        kernel: String,
        first: String,
        second: String,
    },
    PersistentBufferAccess {
        parameter: String,
        buffer: GpuBufferAccess,
        binding: Access,
    },
    PersistentBufferLength {
        expected: usize,
        actual: usize,
    },
    PersistentBufferNotReadable,
    IndependentLengthEmpty {
        parameter: String,
    },
    GraphDependencyOrder {
        node: usize,
        dependency: usize,
    },
    GraphMissingDependency {
        node: usize,
        producer: usize,
    },
    GraphUnknownReadback(usize),
    GraphReadbackType {
        expected: &'static str,
        actual: &'static str,
    },
    IndirectArgsLayout(&'static str),
    IndirectBindingLength {
        parameter: String,
    },
    PoolTruncateGrows {
        len: usize,
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
    DeviceLimit {
        kernel: String,
        limit: &'static str,
        requested: u64,
        maximum: u64,
    },
    /// `HeadlessDevice::with_limits` may only lower limits; `limit` was raised above the
    /// value the device was opened with.
    LimitRaised {
        limit: &'static str,
        requested: u64,
        device: u64,
    },
    Poll(String),
    Map(String),
    /// A buffer allocation failed inside an out-of-memory error scope: the request was within `max_buffer_size`, but
    /// the adapter had no memory for it. `label` names the buffer and `bytes` is the size asked for. Where a device
    /// cannot report this through its scope and is lost instead, S7.6 outcome B maps it to [`Error::Poll`].
    OutOfMemory {
        label: String,
        bytes: u64,
    },
    /// Two resident buffers of one operation belong to different devices (compared by `HeadlessDevice::id`).
    MixedDevices {
        first: u64,
        second: u64,
    },
    /// A dispatch extent is unusable: a product overflowed, so no launch can be described (the extent's `x * y * z`, or
    /// the kernel's workgroup size multiplied out), or the dispatch is of a kind that supports one-dimensional extents
    /// only (a staged graph node with a y or z extent above 1). `reason` names the failing step.
    Extent {
        extent: [u64; 3],
        reason: &'static str,
    },
    /// `set_default_options` was called after the default device had been opened successfully: the options can no
    /// longer change what it was opened with. (After a failed open they can still be set, and the next call retries.)
    DefaultDeviceOpened,
    /// A texture is bound (or created) for a use it does not have: a sampled parameter needs `TextureUse::SAMPLED`,
    /// a storage one `TextureUse::STORAGE`. `required` is what the parameter needs and `actual` what the texture
    /// offers (for `create_texture_2d`: what the caller asked for, and what the format allows on the device).
    TextureUse {
        parameter: String,
        required: gust::TextureUse,
        actual: gust::TextureUse,
    },
    /// A texture of the wrong format is bound: a storage parameter needs its exact format, a sampled one a format of
    /// its sample class. `expected` says which.
    TextureFormat {
        parameter: String,
        expected: String,
        actual: gust::texture::Format,
    },
    /// A sampler that filters linearly is bound on a device without the `float32-filterable` feature, which a linear
    /// filter of a 32-bit float texture needs. `feature` names what is missing. Remedy: open the device with
    /// `FeaturesPolicy::Adapter` on an adapter that offers it, or bind a `Filter::Nearest` sampler.
    TextureFilter {
        parameter: String,
        filter: gust::texture::Filter,
        feature: &'static str,
    },
    /// A texture's size is unusable: a zero side, or a host slice that does not hold `width * height` texels.
    TextureSize {
        width: u32,
        height: u32,
        reason: String,
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
            Self::Slang(error) => write!(formatter, "could not compile kernel with Slang: {error}"),
            Self::Reflection(error) => {
                write!(formatter, "could not validate kernel reflection: {error}")
            }
            Self::NoArtifact { kernel, profile } => write!(
                formatter,
                "kernel `{kernel}` has no precompiled WGSL artifact, and this build of gust-wgpu (profile `{profile}`) was built without the `slang-runtime` feature, so it cannot compile the kernel at run time; run `cargo gust build` in the crate that defines the kernel and rebuild, or enable the `slang-runtime` feature in a dev build",
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
            Self::OutOfMemory { label, bytes } => write!(
                formatter,
                "the adapter is out of memory allocating buffer `{label}` ({bytes} bytes)",
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
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Slang(error) => Some(error),
            Self::Reflection(error) => Some(error),
            _ => None,
        }
    }
}
