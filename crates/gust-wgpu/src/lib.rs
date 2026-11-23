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
//! (`std::time::Instant` panics in the browser), `dispatch_f32`/`dispatch_u32` (deprecated), the blocking `dispatch_batch`,
//! `dispatch_buffers`, `dispatch_f32_batch` and `dispatch_f32_buffers` (both deprecated), the timestamp readback (one-shot channels), the
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
//! This crate is pre-1.0 and follows SemVer for `0.x`: a breaking change bumps the minor version (`0.1` to `0.2`), and
//! a patch release (`0.1.0` to `0.1.1`) only adds or fixes. The four crates of a release (`gust-compute`, the
//! `#[gpu_module]` macro crate, this crate and `cargo-gust`) share one version and depend on each other with exact
//! (`=`) pins: take them at the same version, and a minor bump of one is a minor bump of all.
//!
//! What a minor bump may change: any public item that is not support surface, including a function's signature and
//! the public fields of the plain record types (`BufferDispatch`, `DeviceOptions` and the like).
//! Every public enum is `#[non_exhaustive]` ([`Error`], `AdapterChoice`, `LimitsPolicy` and `FeaturesPolicy` among
//! them) except `TransferDirection`, `GpuBufferAccess` and the binding enums (`F32Binding`, `U32Binding` and
//! `F32BufferBinding`, all deprecated), which are closed sets: a new variant of the others is not a breaking change and a `match`
//! outside this crate needs a wildcard arm; the variants themselves are not, so adding a field to a struct variant is a
//! breaking change and comes only in a minor bump. The statistics and timing records (`PipelineCacheStats`, `PoolStats`,
//! `GrowthRecord`, `TransferRecord`, `TransferTiming`, `DispatchTiming` and `RouteEvent`) and `GraphReport` are
//! `#[non_exhaustive]` and only this crate builds one: read their fields, and a later release may add one.
//!
//! # Support surface
//!
//! `render_wgpu_source` is public because the repository's export command and tests call it. It is `#[doc(hidden)]`;
//! it has no stability promise and a patch release may change it. The artifact and reflection formats that this
//! crate reads are those of `gust` and keep its promise: a reader accepts every schema version that an earlier release
//! wrote.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

mod buffers;
mod cache;
mod check;
mod compile;
mod completion;
mod device;
mod dispatch;
mod encode;
mod error;
mod generated_host;
#[cfg(not(target_arch = "wasm32"))]
pub mod graph;
#[doc = include_str!("guide/index.md")]
pub mod guide;
mod map_host;
// The SPIR-V route (m9.rt2, D172): native Vulkan only, and only with the feature.
#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
mod passthrough;
#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "spirv-route", not(target_arch = "wasm32"))))
)]
pub use passthrough::{Route, RouteEvent};

/// Whether this build has the opt-in SPIR-V route (the Cargo feature `spirv-route`, m9.rt2): the kernels are then emitted with
/// the guards naga adds on the WGSL route, and a device with `PASSTHROUGH_SHADERS` builds its pipelines from slangc's SPIR-V.
/// A test that pins emitted text or a driver's behaviour reads it to pick the expectation of the route.
pub const SPIRV_ROUTE: bool = cfg!(all(feature = "spirv-route", not(target_arch = "wasm32")));
pub mod pool;
mod render;
// `GpuBackend for HeadlessDevice` reads buffers back with `read_typed_buffer`, which does not exist on wasm32, where a
// call cannot block either, so the routed handle has no GPU backend there (`RoutedDevice<NoGpu>` does).
mod headless;
pub mod inspect;
mod readback;
#[cfg(not(target_arch = "wasm32"))]
mod routed;
mod texture;
mod typed_call;
mod vec;
pub use buffers::*;
pub use cache::PipelineCacheStats;
pub(crate) use check::*;
pub(crate) use compile::*;
use completion::Completion;
#[cfg(not(target_arch = "wasm32"))]
use completion::blocking::block_on_mapping;
pub use device::{
    AdapterChoice, DeviceHandle, DeviceOptions, FeaturesPolicy, LimitsPolicy, default_device,
    default_device_async, set_default_device, set_default_options,
};
pub use dispatch::*;
pub use error::Error;
#[doc(hidden)]
pub use generated_host::render_wgpu_source;
#[cfg(not(target_arch = "wasm32"))]
pub use graph::{GraphOutput, GraphReport, NodeId, StagedGraph, TransferDirection, TransferRecord};
pub use headless::*;
pub use pool::{BufferPool, GpuPool, GrowthRecord, PoolStats};
pub(crate) use readback::*;
pub use texture::{GpuSampler, GpuSamplerBinding, GpuTexture2d};
pub(crate) use typed_call::*;
pub use vec::GpuVec;
/// The `wgpu` this crate is built against, so that a renderer names the same copy, with the same features, as the
/// handles it shares with GUST (see [`HeadlessDevice::adopt`]).
pub use wgpu;

// A build holds one target (D21): this crate is the GPU runtime, so a build under `GUST_PROFILE=cpu` stops here with
// a compile error (E0080) that names the profile. Under `dev`, `native` and `web` this does nothing.
const _: () = ::gust::profile::require_gpu_target();

/// wgpu's default `max_compute_workgroups_per_dimension`: the most workgroups one dispatch may
/// launch along one axis on a device opened with the default limits, which is what
/// `HeadlessDevice` requests. A single-dimension dispatch of a kernel with workgroup width `w`
/// therefore covers at most `w * DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION` elements; more
/// is an `Error::DeviceLimit` (a multi-dimensional extent spreads it over y and z). Pinned to
/// `wgpu::Limits::default()` by a unit test and to the opened device by the `limits` test.
pub const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
