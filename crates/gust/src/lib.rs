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
//! # CPU twin
//!
//! [`cpu`] runs a kernel's Rust shadow body on the CPU (`CpuDevice`, `CpuBuffer`); the per-kernel twin of the typed
//! calls that the `#[gpu]` macro emits uses it. The macro emits the twin under `cfg(test)` of the crate that defines
//! the kernel or under `--cfg gust_dual`, unless the kernel has a uniform parameter or `#[kernel(cpu =
//! "unsupported")]`. Nothing of it is re-exported at the crate root, except the hidden macro behind
//! `cpu::overflow_checks!`.

// The `#[gpu]` expansion names `::gust::..`; this lets the crate's own unit tests expand it too.
#[cfg(test)]
extern crate self as gust;

mod abi;
pub mod artifact;
pub mod cpu;
mod descriptor;
mod dispatch;
mod job;
pub mod reflect;
mod runtime;
pub mod slang;
pub mod spirv;

pub use abi::{
    GpuPod, ResourceLayout, ScalarKind, ShaderLayoutRules, StructFieldLayout, TypeLayout,
    TypeLayoutKind, pod_slice_as_bytes,
};
pub use artifact::{Artifact, Format};
pub use descriptor::{
    Access, Kernel, KernelDescriptor, LineSpan, ModuleDescriptor, ParameterDescriptor,
    ParameterKind, ResourceBinding, SourceOrigin,
};
pub use dispatch::{Arg, Call, Independent, LengthRule, Resource, Run, TypedDevice};
pub use gust_macros::gpu;
pub use job::{Device, Job, JobStatus};
pub use reflect::{
    PodCrossCheck, ReflectedField, ReflectedResource, ReflectedType, ReflectedTypeKind, Reflection,
    cross_check_pod,
};
pub use runtime::{
    Invocation, RWStructuredBuffer, SV_DispatchThreadID, Storage, StorageMut, StructuredBuffer,
    UVec3, Uniform, atomic_add, atomic_compare_exchange, atomic_exchange, atomic_max, atomic_min,
    float, int, uint,
};

/// Names automatically imported into every `#[gpu]` module.
pub mod prelude {
    pub use crate::{
        Invocation, RWStructuredBuffer, SV_DispatchThreadID, Storage, StorageMut, StructuredBuffer,
        UVec3, Uniform, atomic_add, atomic_compare_exchange, atomic_exchange, atomic_max,
        atomic_min, float, int, uint,
    };
}
