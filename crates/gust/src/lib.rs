//! Public contract for the Rust-to-Slang GUST compiler.
//!
//! The `#[gpu]` macro lives in a separate proc-macro crate. This crate owns the
//! small, intentionally Slang-shaped vocabulary used by annotated modules,
//! generated Slang descriptors, and host runtime interfaces.
//!
//! # Unsupported source must fail at the Rust boundary
//!
//! A compute entry point cannot return a value:
//! ```compile_fail
//! #[gust::gpu]
//! mod invalid {
//!     #[kernel]
//!     fn run(id: SV_DispatchThreadID) -> uint { id.x }
//! }
//! ```
//! Conditional attributes inside the shader cannot silently lose their meaning:
//! ```compile_fail
//! #[gust::gpu]
//! mod invalid {
//!     #[kernel]
//!     fn run(id: SV_DispatchThreadID) {
//!         #[cfg(any())]
//!         let excluded = 1u32;
//!     }
//! }
//! ```
//! rustc still checks shadow arithmetic after macro validation:
//! ```compile_fail
//! #[gust::gpu]
//! mod invalid {
//!     #[kernel]
//!     fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
//!         out[id.x] = true + 1u32;
//!     }
//! }
//! ```
//! Types and casts outside the 32-bit scalar subset are rejected by the macro,
//! even though rustc would accept them in the shadow body:
//! ```compile_fail
//! #[gust::gpu]
//! mod invalid {
//!     #[kernel]
//!     fn run(id: SV_DispatchThreadID) {
//!         let wide: f64 = id.x as f64;
//!     }
//! }
//! ```
//!
//! # Layout checks run at compile time
//!
//! `TypeLayout::is_storage_v1` is a `const fn`, so a layout can be asserted while compiling. A padded layout fails
//! the build (the field `b` sits at offset 8, not at 4):
//! ```compile_fail
//! use gust::{ScalarKind, StructFieldLayout, TypeLayout};
//!
//! const FIELDS: &[StructFieldLayout] = &[
//!     StructFieldLayout { name: "a", offset: 0, ty: TypeLayout::scalar("u32", ScalarKind::U32) },
//!     StructFieldLayout { name: "b", offset: 8, ty: TypeLayout::scalar("u32", ScalarKind::U32) },
//! ];
//! const _: () = assert!(TypeLayout::structure("P", 12, 4, FIELDS).is_storage_v1());
//! ```
//! The same assertion on the padding-free layout compiles (the control for the case above):
//! ```
//! use gust::{ScalarKind, StructFieldLayout, TypeLayout};
//!
//! const FIELDS: &[StructFieldLayout] = &[
//!     StructFieldLayout { name: "a", offset: 0, ty: TypeLayout::scalar("u32", ScalarKind::U32) },
//!     StructFieldLayout { name: "b", offset: 4, ty: TypeLayout::scalar("u32", ScalarKind::U32) },
//! ];
//! const _: () = assert!(TypeLayout::structure("P", 8, 4, FIELDS).is_storage_v1());
//! ```
//!
//! # Supported source
//!
//! Named struct construction is lowered field-aware: Slang has no field-name
//! initializers, so the translator emits the portable construct-then-assign form,
//! assigning fields by name in source order. This preserves field identity even
//! when the literal lists fields in a different order than the declaration:
//! ```
//! #[gust::gpu]
//! mod valid {
//!     struct Pair { first: uint, second: uint }
//!     #[kernel]
//!     fn run(id: SV_DispatchThreadID) {
//!         let pair = Pair { second: 1u32, first: 2u32 };
//!     }
//! }
//! ```
//!
//! # The CPU runtime: the `cpu` target and the twin
//!
//! [`cpu`] runs a kernel's Rust shadow body on the CPU (`CpuDevice`, `CpuBuffer`). It is the runtime of the `cpu`
//! target (D21, D51) and the engine of the CPU twin (D22). The per-kernel impl of the typed calls that the `#[gpu]`
//! macro emits for it is the twin under `cfg(test)` of the crate that defines the kernel or under `--cfg gust_dual`,
//! unless the kernel has `#[kernel(cpu = "unsupported")]`; under `GUST_PROFILE=cpu` the macro emits the same impl in
//! every build and that build is the `cpu` target (see "Build profiles"). Nothing of it is re-exported at the crate
//! root, except the hidden macro behind `cpu::overflow_checks!`.
//!
//! # Build profiles
//!
//! The environment variable `GUST_PROFILE` chooses the profile of a build (D44): `dev`, `native`, `web` or `cpu`.
//! `gust`'s build script reads it; any other value, including an empty one, fails the build and names the four.
//! `native` and `web` are strict: every `#[gpu]` kernel must have a precompiled artifact (see [`artifact`]), or the
//! build fails with a compile error at the kernel, and the `slang-runtime` feature is refused, so the build contains
//! no run-time Slang compiler. `dev` has neither rule. `profile::NAME` is the effective profile (support surface, see below).
//!
//! `cpu` is the production CPU target (D21): the `#[gpu]` macro emits the kernel's `Run` impl for
//! [`cpu::CpuDevice`] in every build, without the test and `gust_dual` gate, uses no
//! artifact, and the `slang-runtime` feature is refused, so the build holds no run-time Slang bridge. `gust-compute-wgpu`
//! evaluates `profile::require_gpu_target`, which fails under it, so a crate graph that reaches the GPU runtime does
//! not build. It is never inferred: a release build without the variable stays `native` (`web` on `wasm32`). The Slang
//! text that a kernel's descriptor keeps in a `cpu` build is metadata (the identity of the kernel, what `export` and
//! diagnostics show), not a second target: nothing in the build can turn it into a pipeline. A panic in a kernel body
//! follows the build's panic strategy: it unwinds out of `dispatch` and `submit` under `panic = "unwind"` and aborts
//! the process under `panic = "abort"`; nothing catches it.
//!
//! A release build is strict even without the variable: when `GUST_PROFILE` is not set and Cargo builds `gust` with
//! `PROFILE=release`, the profile is `web` for `wasm32` and `native` for every other target, and the build prints
//! `warning: gust-compute@<version>: gust: release build without GUST_PROFILE treated as native` (or `web`; cargo shows a build
//! script's warning only for path and workspace packages, or with `-vv`, so a registry or git dependency stays
//! silent). So `cargo build --release` of any crate with `#[gpu]` kernels needs artifacts and no bridge, except where
//! the internal `cargo-gust-bridge` feature (D108) is in the graph (the `cargo-gust` install, a workspace that has
//! `cargo-gust` as a member, or a mistaken `cargo-gust` dependency): there the profile is `dev`, with a warning that
//! only path and workspace builds show. A release build that wants the bridge, such as
//! `cargo run --release` of an example that enables `slang-runtime`, sets `GUST_PROFILE=dev` explicitly:
//!
//! ```sh
//! GUST_PROFILE=dev cargo run --release -p polynomial
//! ```
//!
//! Set the variable per command, in the shell, in CI, or per workspace in `.cargo/config.toml` under `[env]`. Changing
//! it rebuilds `gust` and everything that depends on it. The repository's own `cargo xtask` commands run their cargo
//! children under `GUST_PROFILE=dev`, whatever the parent environment holds (a different preset value is logged and
//! overridden).
//!
//! # Stability
//!
//! `gust-compute` is pre-1.0 and follows SemVer for `0.x`: a breaking change bumps the minor version (`0.1` to `0.2`), and a
//! patch release (`0.1.0` to `0.1.1`) only adds or fixes. The four crates of a release (this crate, the `#[gpu]` macro
//! crate, the wgpu backend and `cargo-gust`) share one version and depend on each other with exact (`=`) pins, because
//! the macro emits items that only the same release of this crate has: take them at the same version, and a minor bump
//! of one is a minor bump of all.
//!
//! What a minor bump may change: any item of this crate that is not marked as support surface, including a function's
//! signature, the fields of the descriptor and layout types (a hand-written `KernelDescriptor` literal then needs the
//! new field), the variants of an enum that is not `#[non_exhaustive]`, and the Slang that the macro emits. The error
//! enums, [`Shape`], [`ParameterKind`], and the enums that the roadmap will grow (`ScalarKind`, `TypeLayoutKind`,
//! `TextureAccess`, `ShaderLayoutRules`, `JobStatus`, `reflect::Stage`, `reflect::ReflectedResourceKind`,
//! `reflect::ReflectedTypeKind`, `slang::Target` and `artifact::Format`) are `#[non_exhaustive]`, so a new variant is
//! not a breaking change and a `match` outside this crate needs a wildcard arm. The formats of the files that
//! `cargo gust` writes ([`artifact`] manifests, records and reflection documents) carry a schema version, and a reader accepts every
//! schema version that an earlier release wrote.
//!
//! # Support surface
//!
//! These items are public because macro-generated code, `cargo-gust` or the repository's own tests reach them, and are
//! `#[doc(hidden)]`; they have no stability promise and a patch release may change them:
//!
//! - the modules `profile`, `spirv` and `toolchain` (the last one with the `slang-runtime` feature), and
//!   `slang::probe` with its `slang::TargetProbe`;
//! - in `cpu`: `validate`, `read`, `write`, `read_uniform`, `invocations`, `FromGlobalId`, `CpuJob::complete`,
//!   `__no_hook`, the macro `__gust_overflow_checks` behind `overflow_checks!`, and the module `parallel`;
//! - in `cpu::texture`: `TextureRef`, `TextureMut`, `read`, `write`, `sampler` and `SamplerOwner`;
//! - in `map`: `MapElement`, `OwnIndex`, `element_count` and `resource`;
//! - the `*_parameter_count` methods of `KernelDescriptor`;
//! - every `__gust_*` name, the `descriptor;` form of `par_for!` (D103), and the `cargo-gust-bridge` feature.
//!
//! The `check-*` commands and the evidence tiers of `docs/development/VALIDATION.md` belong to the repository, not to
//! this crate.

#![warn(missing_docs)]

// The `#[gpu]` expansion names `::gust::..`; this lets the crate's own unit tests expand it too.
#[cfg(test)]
extern crate self as gust;

mod abi;
pub mod artifact;
pub mod cpu;
mod descriptor;
mod dispatch;
mod job;
pub mod map;
pub mod math;
#[doc(hidden)]
pub mod profile;
pub mod reflect;
mod runtime;
pub mod slang;
#[doc(hidden)]
pub mod spirv;
pub mod texture;
#[cfg(feature = "slang-runtime")]
pub mod toolchain;

pub use abi::{
    GpuPod, MATRIX_LAYOUT, MatrixLayout, ResourceLayout, ScalarKind, ShaderLayoutRules,
    StructFieldLayout, TextureAccess, TextureDimension, TextureLayout, TypeLayout, TypeLayoutKind,
    pod_slice_as_bytes,
};
pub use artifact::{Artifact, Format};
pub use descriptor::{
    Access, Kernel, KernelDescriptor, LineSpan, ModuleDescriptor, PIXEL_MAX_ELEMENTS,
    PIXEL_ROW_WIDTH, ParameterDescriptor, ParameterKind, ResourceBinding, Shape, SourceOrigin,
};
pub use dispatch::{
    Arg, BindingKind, Call, Extent, Independent, LengthRule, Resource, Run, SampledCall,
    SamplerArg, SamplerDevice, SamplerSet, TypedDevice, kind,
};
pub use gust_macros::{gpu, par_for};
pub use job::{Device, Job, JobStatus};
pub use reflect::{
    PodCrossCheck, ReflectedField, ReflectedResource, ReflectedType, ReflectedTypeKind, Reflection,
    cross_check_pod,
};
pub use runtime::{
    ConstantBuffer, GroupId, Invocation, LocalId, RWStructuredBuffer, SV_DispatchThreadID,
    SV_GroupID, SV_GroupThreadID, Storage, StorageMut, StructuredBuffer, UVec3, Uniform, Workgroup,
    atomic_add, atomic_compare_exchange, atomic_exchange, atomic_max, atomic_min, float, int, uint,
    workgroup_barrier,
};
pub use texture::{
    Sampler, SamplerOptions, StorageTexture2D, Texture2D, TextureUse, WriteTexture2D,
};

/// Names automatically imported into every `#[gpu]` module.
pub mod prelude {
    pub use crate::math::{IVec2, IVec4, Mat2, Mat3, Mat4, UVec2, UVec4, Vec2, Vec3, Vec4};
    pub use crate::texture::{
        R32Float, R32Sint, R32Uint, Rg32Float, Rg32Sint, Rg32Uint, Rgba32Float, Rgba32Sint,
        Rgba32Uint, Sampler, StorageTexture2D, Texture2D, WriteTexture2D,
    };
    pub use crate::{
        ConstantBuffer, GroupId, Invocation, LocalId, RWStructuredBuffer, SV_DispatchThreadID,
        SV_GroupID, SV_GroupThreadID, Storage, StorageMut, StructuredBuffer, UVec3, Uniform,
        Workgroup, atomic_add, atomic_compare_exchange, atomic_exchange, atomic_max, atomic_min,
        float, int, uint, workgroup_barrier,
    };
}
