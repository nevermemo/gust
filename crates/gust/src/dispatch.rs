//! Backend-neutral typed dispatch surface (roadmap M2, slice m2.a).
//!
//! A `#[gpu]` kernel handle (a [`Kernel`]) gets generated `call`, `submit` and `dispatch` functions, a kernel with a
//! `Uniform<T>`/`ConstantBuffer<T>` parameter included (slice m6.u2; the host types for the uniform are here and in
//! the backends); `call` turns
//! typed buffer references into a [`Call`]: the kernel, the element count and one erased [`TypedDevice::Binding`] per
//! storage or uniform parameter. This module holds only what that generated code and the backends share:
//!
//! - [`TypedDevice`]: extension of [`Device`] that names the erased, `Copy` binding of one device. [`Device`] itself is
//!   unchanged, so existing backends compile as before.
//! - [`Run<K, N>`]: the executable hook. A device runs kernel `K` (with `N` resource parameters) when it implements
//!   `Run<K, N>`. A GPU backend implements it once for every `K` (a blanket impl); a CPU backend implements it per
//!   kernel, next to the body it executes. `K: Kernel` exposes only a descriptor, so generic code cannot reach a body
//!   through `K` itself.
//! - [`Resource`]: a typed buffer of one device. Through a *generated* `call`, its `Element` and `Device` associated
//!   types make a buffer of the wrong element type, or of another device *type*, a compile error. `Call::new` and
//!   `Resource::bind` are public, so hand-built calls are not covered by that check; see the contract below.
//! - [`Arg`], [`Independent`], [`LengthRule`]: one argument of a call and the call-site opt-in that lets a
//!   buffer's length differ from the element count.
//! - [`Call`]: a plain, reusable value: what the generated `call` returns, what [`Run::submit_call`] consumes and what
//!   the staged graph (m2.e) records. Its descriptor is taken from `K` and its parameter count from `N`; it cannot be
//!   built from a free-form descriptor.
//!
//! # Contract for every backend
//!
//! - **The backend checks every binding against `call.kernel()` at submit**, and does not trust the binding or the
//!   generated code: the binding's kind ([`BindingKind`]) is the parameter's, the binding's access equals the
//!   descriptor's access for that parameter, its element layout equals
//!   the parameter's layout, it belongs to this device *instance* (two devices of one type share a `Binding` type), and
//!   its length obeys the rule below, measured at submit and not when the call was built. A hand-built `Call` can have
//!   the right count and still the wrong order, access, element or device.
//! - One buffer bound twice in one call is accepted when both bindings are [`Access::ReadOnly`]; when either side is
//!   [`Access::ReadWrite`] the backend rejects the call at run time with an error that names both parameters, before
//!   it writes anything (S0 decision A; `gust-wgpu` implements this in m2.c).
//! - `LengthRule::Shared` bindings must hold exactly the call's [`Extent::checked_total`] elements (`x * y * z`; the
//!   element count of a one-dimensional call); `LengthRule::Independent` ones only need to be non-empty when the
//!   extent is non-empty (the rules of the existing runtime checks). A uniform holds one value and has no element
//!   count: it is `Independent` whatever the argument says, and never empty.
//! - An extent with a zero axis runs nothing and is `Ok` (after the binding checks, before any limit check). A launch
//!   whose three axes, each rounded up to whole workgroups, multiply to more than 2^32 invocations
//!   ([`Extent::MAX_INVOCATIONS`]) is refused by every backend, and a launch of exactly 2^32 is accepted by every
//!   backend: the GPU backend adds its device limits (workgroup size and workgroups per dimension) on top.
//!
//! # What the tests in this file are
//!
//! The `tests` module exercises this surface with two devices. The recording mock mirrors a GPU backend: one blanket
//! `Run` impl, it records the call and never runs a kernel body (D21/D22). The CPU side is the real
//! [`crate::cpu::CpuDevice`] (M3): the tests implement `Run` for it per hand-written test kernel, in the shape the
//! `#[gpu]` macro emits for real kernels (slice m3.c), and run *hand-written test bodies* through the same `Call`.
//! The CPU backend implements the backend contract above; its binding identity and access are read through
//! `gust::cpu` only (the `CpuBinding` is opaque here), so this module adds no backend-neutral accessor for them.

use core::fmt;
use core::marker::PhantomData;

use crate::{Access, Device, GpuPod, Kernel, KernelDescriptor, ParameterKind};

/// What a binding of a typed call binds: a storage buffer or a uniform (constant) buffer.
///
/// Every backend binding carries its kind, and the backend checks it against the kind of the descriptor parameter it
/// is bound to ([`BindingKind::of`]) before anything else, so a hand-built [`Call`] that puts a storage buffer in a
/// uniform parameter is rejected at run time. The typed surface rules that out at compile time through
/// [`Resource::Kind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindingKind {
    /// A storage buffer: [`ParameterKind::Storage`].
    Storage,
    /// A uniform (constant) buffer holding exactly one value: [`ParameterKind::Uniform`].
    Uniform,
}

impl BindingKind {
    /// The binding kind a parameter of `kind` takes; `None` for a parameter that is not bound as a resource.
    pub const fn of(kind: ParameterKind) -> Option<Self> {
        match kind {
            ParameterKind::Storage => Some(Self::Storage),
            ParameterKind::Uniform => Some(Self::Uniform),
            ParameterKind::Invocation | ParameterKind::Value => None,
        }
    }
}

/// The marker types of [`Resource::Kind`].
///
/// They are named by path (`gust::kind::Storage`, `gust::kind::Uniform`) because `gust::Storage` and `gust::Uniform`
/// are the kernel-side parameter types. The trait is sealed: the two kinds are the only ones a binding can have.
pub mod kind {
    use super::BindingKind;

    mod sealed {
        pub trait Sealed {}
    }

    /// The kind of a [`Resource`](super::Resource): [`Storage`] or [`Uniform`].
    pub trait ResourceKind: sealed::Sealed + 'static {
        /// The [`BindingKind`] a binding of this resource carries.
        const BINDING: BindingKind;
    }

    /// Marker: the resource is a storage buffer (many elements, every length rule applies).
    #[derive(Clone, Copy, Debug)]
    pub enum Storage {}

    /// Marker: the resource is a uniform buffer (one value, always `Independent` for the length rule).
    #[derive(Clone, Copy, Debug)]
    pub enum Uniform {}

    impl sealed::Sealed for Storage {}
    impl sealed::Sealed for Uniform {}

    impl ResourceKind for Storage {
        const BINDING: BindingKind = BindingKind::Storage;
    }

    impl ResourceKind for Uniform {
        const BINDING: BindingKind = BindingKind::Uniform;
    }
}

/// Extension of [`Device`] for typed calls (implemented by `gust-wgpu` in m2.c).
///
/// [`Device`] is not modified: the binding type lives here so a backend that has no typed surface yet keeps compiling.
pub trait TypedDevice: Device {
    /// Erased, backend-specific, `Copy` view of one bound resource for one dispatch.
    type Binding<'r>: Copy;
}

/// A device that can run kernel `K`, which has `N` storage and uniform parameters.
///
/// GPU backends implement this for every `K`; a CPU backend implements it per kernel, next to the body it executes
/// (M3). Generated `submit` and `dispatch` functions require the device of their first buffer to implement
/// `Run<Self, N>`, so a kernel without an executable form on a device fails to compile for that device.
///
/// The returned job is tied to the device (`'d`), not to the borrowed buffers (`'r`), as for [`Device::submit`]: a CPU
/// backend must finish the work inside `submit_call` (or hold owned handles), because the buffers may be released
/// when the call returns. Backends check every binding against `call.kernel()` (see the module documentation).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot run the kernel `{K}`",
    label = "this device has no executable form of the kernel; GPU devices run every kernel that has a typed call, CPU twins only those emitted for them",
    note = "a kernel runs on `CpuDevice` in a test build of its crate, under `--cfg gust_dual`, and in every build under `GUST_PROFILE=cpu`; it never does when it is `#[kernel(cpu = \"unsupported\")]`, so if this build is `GUST_PROFILE=cpu` the kernel opted out of the CPU"
)]
pub trait Run<K: Kernel, const N: usize>: TypedDevice {
    /// Start the call and return a job to wait on.
    fn submit_call<'d>(&'d self, call: &Call<'_, Self, K, N>)
    -> Result<Self::Job<'d>, Self::Error>;

    /// Start the call and wait for it to finish.
    fn dispatch_call(&self, call: &Call<'_, Self, K, N>) -> Result<(), Self::Error> {
        use crate::Job as _;
        self.submit_call(call)?.wait()
    }
}

/// How a binding's length relates to the call's extent (its element count for a one-dimensional call).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LengthRule {
    /// The buffer must hold exactly the call's [`Extent::checked_total`] elements, `x * y * z` (`element_count` for a
    /// one-dimensional call).
    Shared,
    /// The buffer's length is independent of the extent (a settings buffer, a counter).
    Independent,
}

/// The invocations of a call along the three dispatch axes (slice m7.d): `x * y * z` invocations before rounding each
/// axis up to whole workgroups of the kernel's `workgroup_size`.
///
/// The one-dimensional forms of the typed API (`call`, `submit`, `dispatch`) mean [`Extent::linear`]: `y` and `z`
/// are 1.
/// `call_extent`, `submit_extent` and `dispatch_extent` take an `Extent` instead. Every backend reads it the same way:
///
/// - an axis of 0 makes the call empty: nothing runs, nothing is limit-checked, the result is `Ok`;
/// - each axis is rounded up to whole workgroups, `ceil(axis / size) * size`, and the three rounded axes multiplied are
///   the launch's invocations, which must be at most 2^32 ([`Extent::MAX_INVOCATIONS`]);
/// - a [`LengthRule::Shared`] binding holds exactly [`Extent::checked_total`] elements, the product of the three axes
///   before rounding.
///
/// The global invocation id is `(x, y, z)` with every axis below its rounded extent, so `LocalId` and `GroupId`
/// (`id % size`, `id / size` per axis) stay exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent {
    /// Invocations along x.
    pub x: u32,
    /// Invocations along y.
    pub y: u32,
    /// Invocations along z.
    pub z: u32,
}

impl Extent {
    /// The most invocations of one launch, counted after rounding every axis up to whole workgroups: 2^32. It is the
    /// range of a `u32` global id along a single axis, and both backends enforce it the same way (at most, not below).
    pub const MAX_INVOCATIONS: u64 = 1 << 32;

    /// An extent of `x` invocations along x and `y`, `z`.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// `count` invocations along x only: what the one-dimensional `call` means.
    pub const fn linear(count: u32) -> Self {
        Self {
            x: count,
            y: 1,
            z: 1,
        }
    }

    /// Whether some axis is 0, so that the call runs nothing.
    pub const fn is_empty(&self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }

    /// `x * y * z`, or `None` when the product does not fit a `u64` (an extent far beyond any launch: the cap is 2^32
    /// invocations).
    pub const fn checked_total(&self) -> Option<u64> {
        // `u32 * u32` always fits a `u64`; only the second product can overflow.
        (self.x as u64 * self.y as u64).checked_mul(self.z as u64)
    }

    /// The axes as `[x, y, z]`.
    pub const fn to_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    /// The three axes rounded up to whole workgroups of `size`, `ceil(axis / size) * size` each (at most 2^32, so a
    /// `u64`). `None` when an axis of `size` is 0.
    pub const fn rounded_axes(&self, size: [u32; 3]) -> Option<[u64; 3]> {
        if size[0] == 0 || size[1] == 0 || size[2] == 0 {
            return None;
        }
        Some([
            self.x.div_ceil(size[0]) as u64 * size[0] as u64,
            self.y.div_ceil(size[1]) as u64 * size[1] as u64,
            self.z.div_ceil(size[2]) as u64 * size[2] as u64,
        ])
    }

    /// The invocations of a launch of this extent with workgroups of `size`: the [`rounded_axes`](Self::rounded_axes)
    /// multiplied. `None` when an axis of `size` is 0. Computed in `u128`, so it cannot overflow: three rounded axes
    /// are at most 2^96. A launch is valid for a backend only when this is at most [`Extent::MAX_INVOCATIONS`].
    pub const fn rounded_total(&self, size: [u32; 3]) -> Option<u128> {
        match self.rounded_axes(size) {
            Some([x, y, z]) => Some(x as u128 * y as u128 * z as u128),
            None => None,
        }
    }
}

impl From<u32> for Extent {
    /// `count` along x: [`Extent::linear`].
    fn from(count: u32) -> Self {
        Self::linear(count)
    }
}

impl From<[u32; 3]> for Extent {
    fn from([x, y, z]: [u32; 3]) -> Self {
        Self::new(x, y, z)
    }
}

impl From<(u32, u32, u32)> for Extent {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

/// A typed buffer of one backend.
///
/// In a *generated* call function, `Element` and `Device` are constrained, so the element type and the device type
/// of every argument are checked at compile time. That is all the type system knows: two devices of one type
/// are the same `Device` here, and access is not part of the buffer type (the call binds it). `Call::new` and
/// [`Resource::bind`] are public, so a hand-built call is unchecked; the backend checks access, layout, device
/// instance and length against the kernel descriptor at submit.
///
/// The buffer's length must not change through `&self`: [`Resource::bind`] may record it when the call is built and a
/// [`Call`] can be submitted later and repeatedly. A resource whose length can change through `&self` (interior
/// mutability) must let the backend measure it again at submit. GPU buffers grow only through `&mut`.
///
/// A buffer of the wrong element type does not compile:
///
/// ```compile_fail
/// # use gust::*;
/// # use core::{marker::PhantomData, time::Duration};
/// # struct Dev<const ID: u8>;
/// # struct Done;
/// # impl Job for Done {
/// #     type Error = ();
/// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
/// # }
/// # impl<const ID: u8> Device for Dev<ID> {
/// #     type Error = ();
/// #     type Dispatch<'r> = ();
/// #     type Job<'d> = Done where Self: 'd;
/// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
/// # }
/// # impl<const ID: u8> TypedDevice for Dev<ID> { type Binding<'r> = (); }
/// # struct Buf<const ID: u8, T>(PhantomData<T>);
/// # impl<const ID: u8, T: GpuPod> Resource for Buf<ID, T> {
/// #     type Element = T;
/// #     type Device = Dev<ID>;
/// #     type Kind = gust::kind::Storage;
/// #     fn bind<'r>(&'r self, _: Access, _: LengthRule) {}
/// # }
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
/// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
/// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
/// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
/// #     };
/// # }
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, R0::Device, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// #     R1: Resource<Element = f32, Device = R0::Device> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// let a = Buf::<1, u32>(PhantomData);
/// let b = Buf::<1, f32>(PhantomData);
/// let _ = call(1, &a, &b);
/// ```
///
/// A buffer of another device type does not compile:
///
/// ```compile_fail
/// # use gust::*;
/// # use core::{marker::PhantomData, time::Duration};
/// # struct Dev<const ID: u8>;
/// # struct Done;
/// # impl Job for Done {
/// #     type Error = ();
/// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
/// # }
/// # impl<const ID: u8> Device for Dev<ID> {
/// #     type Error = ();
/// #     type Dispatch<'r> = ();
/// #     type Job<'d> = Done where Self: 'd;
/// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
/// # }
/// # impl<const ID: u8> TypedDevice for Dev<ID> { type Binding<'r> = (); }
/// # struct Buf<const ID: u8, T>(PhantomData<T>);
/// # impl<const ID: u8, T: GpuPod> Resource for Buf<ID, T> {
/// #     type Element = T;
/// #     type Device = Dev<ID>;
/// #     type Kind = gust::kind::Storage;
/// #     fn bind<'r>(&'r self, _: Access, _: LengthRule) {}
/// # }
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
/// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
/// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
/// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
/// #     };
/// # }
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, R0::Device, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// #     R1: Resource<Element = f32, Device = R0::Device> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// let a = Buf::<1, f32>(PhantomData);
/// let b = Buf::<2, f32>(PhantomData);
/// let _ = call(1, &a, &b);
/// ```
///
/// The control for both cases differs from each in exactly one place (the element type, respectively the device):
///
/// ```
/// # use gust::*;
/// # use core::{marker::PhantomData, time::Duration};
/// # struct Dev<const ID: u8>;
/// # struct Done;
/// # impl Job for Done {
/// #     type Error = ();
/// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
/// # }
/// # impl<const ID: u8> Device for Dev<ID> {
/// #     type Error = ();
/// #     type Dispatch<'r> = ();
/// #     type Job<'d> = Done where Self: 'd;
/// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
/// # }
/// # impl<const ID: u8> TypedDevice for Dev<ID> { type Binding<'r> = (); }
/// # struct Buf<const ID: u8, T>(PhantomData<T>);
/// # impl<const ID: u8, T: GpuPod> Resource for Buf<ID, T> {
/// #     type Element = T;
/// #     type Device = Dev<ID>;
/// #     type Kind = gust::kind::Storage;
/// #     fn bind<'r>(&'r self, _: Access, _: LengthRule) {}
/// # }
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
/// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
/// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
/// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
/// #     };
/// # }
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, R0::Device, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32> + 'r,
/// #     R1: Resource<Element = f32, Device = R0::Device> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// let a = Buf::<1, f32>(PhantomData);
/// let b = Buf::<1, f32>(PhantomData);
/// let built = call(1, &a, &b);
/// assert_eq!(built.element_count(), 1);
/// assert_eq!(built.kernel().name, "probe");
/// ```
///
/// A uniform where a storage buffer is expected does not compile ([`Resource::Kind`]; a generated call names
/// `Kind = gust::kind::Storage` for a buffer parameter and `Kind = gust::kind::Uniform` for a uniform one; the error
/// code is documentation only, because stable rustdoc accepts a `compile_fail` code and does not check it, which a
/// wrong code in a scratch run confirmed):
///
/// ```compile_fail,E0271
/// # use gust::*;
/// # use core::{marker::PhantomData, time::Duration};
/// # struct Dev;
/// # struct Done;
/// # impl Job for Done {
/// #     type Error = ();
/// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
/// # }
/// # impl Device for Dev {
/// #     type Error = ();
/// #     type Dispatch<'r> = ();
/// #     type Job<'d> = Done where Self: 'd;
/// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
/// # }
/// # impl TypedDevice for Dev { type Binding<'r> = (); }
/// # struct Buf<K, T>(PhantomData<(K, T)>);
/// # impl<K: gust::kind::ResourceKind, T: GpuPod> Resource for Buf<K, T> {
/// #     type Element = T;
/// #     type Device = Dev;
/// #     type Kind = K;
/// #     fn bind<'r>(&'r self, _: Access, _: LengthRule) {}
/// # }
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
/// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
/// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
/// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
/// #     };
/// # }
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, Dev, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32, Device = Dev, Kind = gust::kind::Storage> + 'r,
/// #     R1: Resource<Element = f32, Device = Dev, Kind = gust::kind::Storage> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// let a = Buf::<gust::kind::Storage, f32>(PhantomData);
/// let b = Buf::<gust::kind::Uniform, f32>(PhantomData);
/// let _ = call(1, &a, &b);
/// ```
///
/// The control differs in exactly one place (the kind of `b`):
///
/// ```
/// # use gust::*;
/// # use core::{marker::PhantomData, time::Duration};
/// # struct Dev;
/// # struct Done;
/// # impl Job for Done {
/// #     type Error = ();
/// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
/// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
/// # }
/// # impl Device for Dev {
/// #     type Error = ();
/// #     type Dispatch<'r> = ();
/// #     type Job<'d> = Done where Self: 'd;
/// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
/// # }
/// # impl TypedDevice for Dev { type Binding<'r> = (); }
/// # struct Buf<K, T>(PhantomData<(K, T)>);
/// # impl<K: gust::kind::ResourceKind, T: GpuPod> Resource for Buf<K, T> {
/// #     type Element = T;
/// #     type Device = Dev;
/// #     type Kind = K;
/// #     fn bind<'r>(&'r self, _: Access, _: LengthRule) {}
/// # }
/// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
/// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
/// # }
/// # struct Probe;
/// # impl Kernel for Probe {
/// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
/// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
/// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
/// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
/// #     };
/// # }
/// # fn call<'r, R0, R1>(n: u32, a: impl Into<Arg<'r, R0>>, b: impl Into<Arg<'r, R1>>) -> Call<'r, Dev, Probe, 2>
/// # where
/// #     R0: Resource<Element = f32, Device = Dev, Kind = gust::kind::Storage> + 'r,
/// #     R1: Resource<Element = f32, Device = Dev, Kind = gust::kind::Storage> + 'r,
/// # {
/// #     Call::new(n, [a.into().bind(Access::ReadOnly), b.into().bind(Access::ReadWrite)])
/// # }
/// let a = Buf::<gust::kind::Storage, f32>(PhantomData);
/// let b = Buf::<gust::kind::Storage, f32>(PhantomData);
/// let built = call(1, &a, &b);
/// assert_eq!(built.kernel().name, "probe");
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a GUST buffer of a typed device",
    label = "typed calls take the buffer type of a GUST backend; wrap or allocate the data through that backend"
)]
pub trait Resource {
    /// The element type the kernel parameter must have. `GpuPod` gives backends the element layout to check.
    type Element: GpuPod;
    /// The device type this buffer belongs to; every argument of one generated call must name the same device type.
    type Device: TypedDevice;
    /// [`kind::Storage`] for a buffer of many elements, [`kind::Uniform`] for a uniform buffer holding one value.
    /// A generated call names the kind of each parameter, so a uniform where a buffer is expected, or the other way
    /// round, is a compile error. The binding carries the same kind ([`BindingKind`]) and the backend checks it again
    /// at submit, for a hand-built [`Call`].
    type Kind: kind::ResourceKind;

    /// The erased binding of this buffer for one dispatch.
    fn bind<'r>(
        &'r self,
        access: Access,
        length: LengthRule,
    ) -> <Self::Device as TypedDevice>::Binding<'r>;
}

/// One argument of a typed call: a borrowed resource plus its length rule.
#[derive(Debug)]
pub struct Arg<'r, R: Resource> {
    /// The borrowed buffer.
    pub resource: &'r R,
    /// How its length relates to the element count.
    pub length: LengthRule,
}

impl<'r, R: Resource> From<&'r R> for Arg<'r, R> {
    fn from(resource: &'r R) -> Self {
        Self {
            resource,
            length: LengthRule::Shared,
        }
    }
}

/// Call-site opt-in: this buffer's length may differ from the element count.
///
/// `kernel::call(n, &a, Independent(&settings), &out)` marks only `settings`.
#[derive(Debug)]
pub struct Independent<T>(pub T);

impl<'r, R: Resource> From<Independent<&'r R>> for Arg<'r, R> {
    fn from(independent: Independent<&'r R>) -> Self {
        Self {
            resource: independent.0,
            length: LengthRule::Independent,
        }
    }
}

impl<'r, R: Resource> Arg<'r, R> {
    /// Bind the buffer with the access the kernel parameter declares.
    #[must_use = "the binding is only useful as an element of a `Call`"]
    pub fn bind(self, access: Access) -> <R::Device as TypedDevice>::Binding<'r> {
        self.resource.bind(access, self.length)
    }
}

/// The descriptor and parameter count a `Call<_, K, N>` carries, derived from `K` and `N` alone.
struct Described<K, const N: usize>(PhantomData<fn() -> K>);

impl<K: Kernel, const N: usize> Described<K, N> {
    /// `&K::DESCRIPTOR` as a `'static` reference. A `const` item's borrow is extended to `'static`, so no generic
    /// caller needs constant promotion of an associated constant (plan F12).
    const DESCRIPTOR: &'static KernelDescriptor = &K::DESCRIPTOR;

    /// Evaluated whenever `Call::new` is instantiated: the kernel's storage plus uniform parameters must number `N`.
    const BOUND_PARAMETERS_MATCH: () = assert!(
        K::DESCRIPTOR.storage_parameter_count() + K::DESCRIPTOR.uniform_parameter_count() == N,
        "Call<_, K, N>: N must equal the number of storage plus uniform parameters of K::DESCRIPTOR"
    );
}

impl KernelDescriptor {
    /// Number of `Storage` parameters (a typed call binds the uniform parameters too).
    ///
    /// Not part of the stable surface: `#[gpu]` expansions use it in a constant assertion that the generated
    /// typed call takes as many arguments as the descriptor has storage and uniform parameters.
    #[doc(hidden)]
    pub const fn storage_parameter_count(&self) -> usize {
        let mut index = 0;
        let mut count = 0;
        while index < self.parameters.len() {
            if matches!(self.parameters[index].kind, ParameterKind::Storage) {
                count += 1;
            }
            index += 1;
        }
        count
    }

    /// Number of `Uniform` parameters (the other ones a typed call binds).
    ///
    /// Not part of the stable surface, like [`KernelDescriptor::storage_parameter_count`].
    #[doc(hidden)]
    pub const fn uniform_parameter_count(&self) -> usize {
        let mut index = 0;
        let mut count = 0;
        while index < self.parameters.len() {
            if matches!(self.parameters[index].kind, ParameterKind::Uniform) {
                count += 1;
            }
            index += 1;
        }
        count
    }
}

/// A typed kernel call: the kernel, the extent and one binding per storage or uniform parameter.
///
/// It is a plain value (build once, submit as often as needed) and what the generated `call` returns, what
/// [`Run::submit_call`] consumes and what the staged graph records. The descriptor is `K::DESCRIPTOR` and `N` is
/// the kernel's number of storage plus uniform parameters: [`Call::new`] takes neither as an argument. A call made by
/// [`Call::new`] covers `element_count` invocations along x; one made by [`Call::new_extent`] covers an [`Extent`].
#[must_use = "a `Call` does nothing until a device runs it"]
pub struct Call<'r, D: TypedDevice + ?Sized, K: Kernel, const N: usize> {
    kernel: &'static KernelDescriptor,
    extent: Extent,
    bindings: [D::Binding<'r>; N],
    _kernel: PhantomData<fn() -> K>,
}

impl<'r, D: TypedDevice + ?Sized, K: Kernel, const N: usize> Call<'r, D, K, N> {
    /// Build the call of kernel `K`: `bindings` are its storage and uniform parameters in declaration order.
    ///
    /// The descriptor is `K::DESCRIPTOR`, so the public metadata cannot disagree with the kernel type, and `N` must
    /// equal the number of `Storage` plus `Uniform` parameters of that descriptor. A mismatch is a compile error
    /// when the call is instantiated (a failed constant assertion). That is after monomorphisation: `cargo check`,
    /// clippy and rust-analyzer do not report it, and it never fires for a `call` nothing uses. This is a backstop;
    /// the `#[gpu]` macro also emits a non-generic `const _: () = assert!(..)` per kernel. The count is the only
    /// property of `bindings` that is checked here: order, access, element and device are the generated code's
    /// and the backend's business.
    ///
    /// ```compile_fail
    /// # use gust::*;
    /// # use core::time::Duration;
    /// # struct Dev;
    /// # struct Done;
    /// # impl Job for Done {
    /// #     type Error = ();
    /// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
    /// # }
    /// # impl Device for Dev {
    /// #     type Error = ();
    /// #     type Dispatch<'r> = ();
    /// #     type Job<'d> = Done where Self: 'd;
    /// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
    /// # }
    /// # impl TypedDevice for Dev { type Binding<'r> = (); }
    /// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
    /// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
    /// # }
    /// # struct Probe;
    /// # impl Kernel for Probe {
    /// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
    /// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
    /// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
    /// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
    /// #     };
    /// # }
    /// # fn build<const N: usize>() -> Call<'static, Dev, Probe, N> {
    /// #     Call::new(1, [(); N])
    /// # }
    /// // `Probe` has two storage parameters, the call claims three.
    /// let _ = build::<3>();
    /// ```
    ///
    /// The control differs in exactly one place (the count):
    ///
    /// ```
    /// # use gust::*;
    /// # use core::time::Duration;
    /// # struct Dev;
    /// # struct Done;
    /// # impl Job for Done {
    /// #     type Error = ();
    /// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
    /// # }
    /// # impl Device for Dev {
    /// #     type Error = ();
    /// #     type Dispatch<'r> = ();
    /// #     type Job<'d> = Done where Self: 'd;
    /// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
    /// # }
    /// # impl TypedDevice for Dev { type Binding<'r> = (); }
    /// # const fn p(name: &'static str, access: Access) -> ParameterDescriptor {
    /// #     ParameterDescriptor { name, rust_type: "f32", kind: ParameterKind::Storage, access, resource_layout: None, binding: None }
    /// # }
    /// # struct Probe;
    /// # impl Kernel for Probe {
    /// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
    /// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
    /// #         parameters: &[p("a", Access::ReadOnly), p("b", Access::ReadWrite)],
    /// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
    /// #     };
    /// # }
    /// # fn build<const N: usize>() -> Call<'static, Dev, Probe, N> {
    /// #     Call::new(1, [(); N])
    /// # }
    /// let call = build::<2>();
    /// assert_eq!(call.kernel().name, "probe");
    /// ```
    ///
    /// A uniform parameter counts: a kernel with one storage and one uniform parameter takes `N = 2`, and `N = 1`
    /// (the storage count alone) does not compile.
    ///
    /// ```compile_fail,E0080
    /// # use gust::*;
    /// # use core::time::Duration;
    /// # struct Dev;
    /// # struct Done;
    /// # impl Job for Done {
    /// #     type Error = ();
    /// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
    /// # }
    /// # impl Device for Dev {
    /// #     type Error = ();
    /// #     type Dispatch<'r> = ();
    /// #     type Job<'d> = Done where Self: 'd;
    /// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
    /// # }
    /// # impl TypedDevice for Dev { type Binding<'r> = (); }
    /// # const fn p(name: &'static str, kind: ParameterKind) -> ParameterDescriptor {
    /// #     ParameterDescriptor { name, rust_type: "f32", kind, access: Access::ReadOnly, resource_layout: None, binding: None }
    /// # }
    /// # struct Probe;
    /// # impl Kernel for Probe {
    /// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
    /// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
    /// #         parameters: &[p("a", ParameterKind::Storage), p("u", ParameterKind::Uniform)],
    /// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
    /// #     };
    /// # }
    /// # fn build<const N: usize>() -> Call<'static, Dev, Probe, N> {
    /// #     Call::new(1, [(); N])
    /// # }
    /// let _ = build::<1>();
    /// ```
    ///
    /// The control differs in exactly one place (the count):
    ///
    /// ```
    /// # use gust::*;
    /// # use core::time::Duration;
    /// # struct Dev;
    /// # struct Done;
    /// # impl Job for Done {
    /// #     type Error = ();
    /// #     fn status(&self) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait_timeout(&self, _: Duration) -> Result<JobStatus, ()> { Ok(JobStatus::Complete) }
    /// #     fn wait(&self) -> Result<(), ()> { Ok(()) }
    /// # }
    /// # impl Device for Dev {
    /// #     type Error = ();
    /// #     type Dispatch<'r> = ();
    /// #     type Job<'d> = Done where Self: 'd;
    /// #     fn submit<'d, 'r>(&'d self, _: &[()]) -> Result<Done, ()> { Ok(Done) }
    /// # }
    /// # impl TypedDevice for Dev { type Binding<'r> = (); }
    /// # const fn p(name: &'static str, kind: ParameterKind) -> ParameterDescriptor {
    /// #     ParameterDescriptor { name, rust_type: "f32", kind, access: Access::ReadOnly, resource_layout: None, binding: None }
    /// # }
    /// # struct Probe;
    /// # impl Kernel for Probe {
    /// #     const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
    /// #         module: "m", name: "probe", entry_point: "m_probe", workgroup_size: [1, 1, 1],
    /// #         parameters: &[p("a", ParameterKind::Storage), p("u", ParameterKind::Uniform)],
    /// #         slang_source: "", source: SourceOrigin::UNKNOWN, artifacts: &[],
    /// #     };
    /// # }
    /// # fn build<const N: usize>() -> Call<'static, Dev, Probe, N> {
    /// #     Call::new(1, [(); N])
    /// # }
    /// let call = build::<2>();
    /// assert_eq!(call.kernel().storage_parameter_count(), 1);
    /// assert_eq!(call.kernel().uniform_parameter_count(), 1);
    /// ```
    pub const fn new(element_count: u32, bindings: [D::Binding<'r>; N]) -> Self {
        Self::new_extent(Extent::linear(element_count), bindings)
    }

    /// [`Call::new`] over an [`Extent`] instead of an element count (slice m7.d): `Call::new(n, ..)` is
    /// `Call::new_extent(Extent::linear(n), ..)`.
    pub const fn new_extent(extent: Extent, bindings: [D::Binding<'r>; N]) -> Self {
        let () = Described::<K, N>::BOUND_PARAMETERS_MATCH;
        Self {
            kernel: Described::<K, N>::DESCRIPTOR,
            extent,
            bindings,
            _kernel: PhantomData,
        }
    }

    /// The descriptor of `K`.
    pub const fn kernel(&self) -> &'static KernelDescriptor {
        self.kernel
    }

    /// The invocations along x (before rounding up to whole workgroups): the element count of a one-dimensional
    /// call. For a call made with [`Call::new_extent`] this is the x axis only; [`Call::extent`] is the whole. Do not
    /// size a buffer from it for a multi-dimensional call: the length of a shared buffer is
    /// `call.extent().checked_total()`, the product of the three axes.
    pub const fn element_count(&self) -> u32 {
        self.extent.x
    }

    /// The invocations the call covers along the three axes (before rounding up to whole workgroups).
    pub const fn extent(&self) -> Extent {
        self.extent
    }

    /// One binding per storage or uniform parameter, in declaration order.
    pub const fn bindings(&self) -> &[D::Binding<'r>; N] {
        &self.bindings
    }
}

impl<'r, D: TypedDevice + ?Sized, K: Kernel, const N: usize> fmt::Debug for Call<'r, D, K, N>
where
    D::Binding<'r>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Call")
            .field("kernel", &self.kernel.name)
            .field("extent", &self.extent)
            .field("bindings", &self.bindings)
            .finish()
    }
}

#[cfg(test)]
#[forbid(unsafe_code)]
mod tests {
    use core::cell::RefCell;
    use core::marker::PhantomData;
    use core::time::Duration;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{
        Invocation, Job, JobStatus, ParameterDescriptor, ResourceBinding, ResourceLayout,
        SourceOrigin, Storage, StorageMut, TypeLayout,
        cpu::{self, CpuDevice, CpuJob, Error as CpuError, FromGlobalId},
    };

    // ---- hand-written kernels, in the exact form the macro emits (m2.d1) ----

    const fn invocation() -> ParameterDescriptor {
        ParameterDescriptor {
            name: "id",
            rust_type: "SV_DispatchThreadID",
            kind: ParameterKind::Invocation,
            access: Access::NotApplicable,
            resource_layout: None,
            binding: None,
        }
    }

    const fn storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "float",
            kind: ParameterKind::Storage,
            access,
            resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
            binding: Some(ResourceBinding { group: 0, binding }),
        }
    }

    const fn descriptor(
        name: &'static str,
        parameters: &'static [ParameterDescriptor],
    ) -> KernelDescriptor {
        KernelDescriptor {
            module: "typed_tests",
            name,
            entry_point: name,
            workgroup_size: [64, 1, 1],
            parameters,
            slang_source: "",
            source: SourceOrigin::UNKNOWN,
            artifacts: &[],
        }
    }

    /// `scale(id, a: RO, settings: RO, out: RW)`; the invocation parameter is not a binding.
    struct Scale;
    impl Kernel for Scale {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "scale",
            &[
                invocation(),
                storage("a", Access::ReadOnly, 0),
                storage("settings", Access::ReadOnly, 1),
                storage("out", Access::ReadWrite, 2),
            ],
        );
    }

    /// `sum2(id, a: RO, b: RO, out: RW)`: the RO/RO alias case; same shape as `Scale` with another name.
    struct Sum2;
    impl Kernel for Sum2 {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "sum2",
            &[
                invocation(),
                storage("a", Access::ReadOnly, 0),
                storage("b", Access::ReadOnly, 1),
                storage("out", Access::ReadWrite, 2),
            ],
        );
    }

    /// `swap2(id, x: RW, y: RW)`: the RW/RW alias case.
    struct Swap2;
    impl Kernel for Swap2 {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "swap2",
            &[
                invocation(),
                storage("x", Access::ReadWrite, 0),
                storage("y", Access::ReadWrite, 1),
            ],
        );
    }

    /// `shade(id, a: RO, factor: Uniform<f32>, out: RW)`: two storage parameters and one uniform, so `N` is 3.
    struct Shade;
    impl Kernel for Shade {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "shade",
            &[
                invocation(),
                storage("a", Access::ReadOnly, 0),
                ParameterDescriptor {
                    name: "factor",
                    rust_type: "float",
                    kind: ParameterKind::Uniform,
                    access: Access::ReadOnly,
                    resource_layout: Some(ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT)),
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 1,
                    }),
                },
                storage("out", Access::ReadWrite, 2),
            ],
        );
    }

    /// A kernel without storage parameters.
    struct Nothing;
    impl Kernel for Nothing {
        const DESCRIPTOR: KernelDescriptor = descriptor("nothing", &[invocation()]);
    }

    impl Scale {
        fn call<'r, R0, R1, R2>(
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            settings: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Call<'r, R0::Device, Self, 3>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R2: Resource<Element = f32, Device = R0::Device> + 'r,
        {
            Call::new(
                __gust_element_count,
                [
                    a.into().bind(Access::ReadOnly),
                    settings.into().bind(Access::ReadOnly),
                    out.into().bind(Access::ReadWrite),
                ],
            )
        }

        fn submit<'d, 'r, R0, R1, R2>(
            __gust_device: &'d R0::Device,
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            settings: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Result<<R0::Device as Device>::Job<'d>, <R0::Device as Device>::Error>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R2: Resource<Element = f32, Device = R0::Device> + 'r,
            R0::Device: Run<Self, 3>,
        {
            __gust_device.submit_call(&Self::call(__gust_element_count, a, settings, out))
        }

        fn dispatch<'r, R0, R1, R2>(
            __gust_device: &R0::Device,
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            settings: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Result<(), <R0::Device as Device>::Error>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R2: Resource<Element = f32, Device = R0::Device> + 'r,
            R0::Device: Run<Self, 3>,
        {
            __gust_device.dispatch_call(&Self::call(__gust_element_count, a, settings, out))
        }
    }

    impl Sum2 {
        fn call<'r, R0, R1, R2>(
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            b: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Call<'r, R0::Device, Self, 3>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R2: Resource<Element = f32, Device = R0::Device> + 'r,
        {
            Call::new(
                __gust_element_count,
                [
                    a.into().bind(Access::ReadOnly),
                    b.into().bind(Access::ReadOnly),
                    out.into().bind(Access::ReadWrite),
                ],
            )
        }

        fn dispatch<'r, R0, R1, R2>(
            __gust_device: &R0::Device,
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            b: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Result<(), <R0::Device as Device>::Error>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R2: Resource<Element = f32, Device = R0::Device> + 'r,
            R0::Device: Run<Self, 3>,
        {
            __gust_device.dispatch_call(&Self::call(__gust_element_count, a, b, out))
        }
    }

    impl Swap2 {
        fn call<'r, R0, R1>(
            __gust_element_count: u32,
            x: impl Into<Arg<'r, R0>>,
            y: impl Into<Arg<'r, R1>>,
        ) -> Call<'r, R0::Device, Self, 2>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
        {
            Call::new(
                __gust_element_count,
                [
                    x.into().bind(Access::ReadWrite),
                    y.into().bind(Access::ReadWrite),
                ],
            )
        }

        fn dispatch<'r, R0, R1>(
            __gust_device: &R0::Device,
            __gust_element_count: u32,
            x: impl Into<Arg<'r, R0>>,
            y: impl Into<Arg<'r, R1>>,
        ) -> Result<(), <R0::Device as Device>::Error>
        where
            R0: Resource<Element = f32> + 'r,
            R1: Resource<Element = f32, Device = R0::Device> + 'r,
            R0::Device: Run<Self, 2>,
        {
            __gust_device.dispatch_call(&Self::call(__gust_element_count, x, y))
        }
    }

    impl Shade {
        /// The `call` the macro will generate: `Kind = Storage` for the buffers, `Kind = Uniform` for the uniform.
        fn call<'r, R0, R1, R2>(
            __gust_element_count: u32,
            a: impl Into<Arg<'r, R0>>,
            factor: impl Into<Arg<'r, R1>>,
            out: impl Into<Arg<'r, R2>>,
        ) -> Call<'r, R0::Device, Self, 3>
        where
            R0: Resource<Element = f32, Kind = kind::Storage> + 'r,
            R1: Resource<Element = f32, Device = R0::Device, Kind = kind::Uniform> + 'r,
            R2: Resource<Element = f32, Device = R0::Device, Kind = kind::Storage> + 'r,
        {
            Call::new(
                __gust_element_count,
                [
                    a.into().bind(Access::ReadOnly),
                    factor.into().bind(Access::ReadOnly),
                    out.into().bind(Access::ReadWrite),
                ],
            )
        }
    }

    impl Nothing {
        fn call<'r, D: TypedDevice>(__gust_element_count: u32) -> Call<'r, D, Self, 0> {
            Call::new(__gust_element_count, [])
        }
    }

    // Test bodies (not `#[gpu]` shadow bodies): the same shape and guards as GPU kernels.

    fn scale_body(
        id: Invocation,
        a: Storage<'_, f32>,
        settings: Storage<'_, f32>,
        mut out: StorageMut<'_, f32>,
    ) {
        let x = id.global_id().x;
        if x < out.len() {
            out[x] = a[x] * settings[0];
        }
    }

    fn sum2_body(
        id: Invocation,
        a: Storage<'_, f32>,
        b: Storage<'_, f32>,
        mut out: StorageMut<'_, f32>,
    ) {
        let x = id.global_id().x;
        if x < out.len() {
            out[x] = a[x] + b[x];
        }
    }

    #[allow(clippy::manual_swap)] // a shadow-shaped body: the shape is the point
    fn swap2_body(id: Invocation, mut x: StorageMut<'_, f32>, mut y: StorageMut<'_, f32>) {
        let i = id.global_id().x;
        if i < x.len() && i < y.len() {
            let t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    }

    // ---- shared mock plumbing ----

    struct Done<E>(PhantomData<E>);

    impl<E> Done<E> {
        fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<E> Job for Done<E> {
        type Error = E;

        fn status(&self) -> Result<JobStatus, E> {
            Ok(JobStatus::Complete)
        }

        fn wait_timeout(&self, _timeout: Duration) -> Result<JobStatus, E> {
            Ok(JobStatus::Complete)
        }

        fn wait(&self) -> Result<(), E> {
            Ok(())
        }
    }

    // ---- recording mock: GPU-shaped, one blanket `Run` impl, never executes a body ----

    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

    struct RecBuffer<T> {
        id: usize,
        values: Vec<T>,
    }

    impl<T> RecBuffer<T> {
        fn new(values: Vec<T>) -> Self {
            Self {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                values,
            }
        }
    }

    /// Mirrors `BufferBinding<'a>` of `gust-wgpu`: `Copy` and borrowing the buffer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RecBinding<'r> {
        id: usize,
        len: usize,
        layout: TypeLayout,
        access: Access,
        length: LengthRule,
        kind: BindingKind,
        _buffer: PhantomData<&'r ()>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Recorded {
        kernel: &'static str,
        element_count: u32,
        bindings: Vec<(usize, usize, Access, LengthRule)>,
        layouts: Vec<TypeLayout>,
        kinds: Vec<BindingKind>,
    }

    #[derive(Default)]
    struct RecordingDevice {
        calls: RefCell<Vec<Recorded>>,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum RecError {}

    impl Device for RecordingDevice {
        type Error = RecError;
        type Dispatch<'resources> = ();
        type Job<'device> = Done<RecError>;

        fn submit(&self, _dispatches: &[()]) -> Result<Done<RecError>, RecError> {
            Ok(Done::new())
        }
    }

    impl TypedDevice for RecordingDevice {
        type Binding<'r> = RecBinding<'r>;
    }

    impl<T: GpuPod> Resource for RecBuffer<T> {
        type Element = T;
        type Device = RecordingDevice;
        type Kind = kind::Storage;

        fn bind<'r>(&'r self, access: Access, length: LengthRule) -> RecBinding<'r> {
            RecBinding {
                id: self.id,
                len: self.values.len(),
                layout: T::LAYOUT,
                access,
                length,
                kind: BindingKind::Storage,
                _buffer: PhantomData,
            }
        }
    }

    /// A uniform of the recording device: one value, `Independent` whatever the argument says.
    struct RecUniform<T> {
        id: usize,
        _value: T,
    }

    impl<T> RecUniform<T> {
        fn new(value: T) -> Self {
            Self {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                _value: value,
            }
        }
    }

    impl<T: GpuPod> Resource for RecUniform<T> {
        type Element = T;
        type Device = RecordingDevice;
        type Kind = kind::Uniform;

        fn bind<'r>(&'r self, access: Access, _length: LengthRule) -> RecBinding<'r> {
            RecBinding {
                id: self.id,
                len: 1,
                layout: T::LAYOUT,
                access,
                length: LengthRule::Independent,
                kind: BindingKind::Uniform,
                _buffer: PhantomData,
            }
        }
    }

    impl<K: Kernel, const N: usize> Run<K, N> for RecordingDevice {
        fn submit_call(&self, call: &Call<'_, Self, K, N>) -> Result<Done<RecError>, RecError> {
            self.calls.borrow_mut().push(Recorded {
                kernel: call.kernel().name,
                element_count: call.element_count(),
                bindings: call
                    .bindings()
                    .iter()
                    .map(|b| (b.id, b.len, b.access, b.length))
                    .collect(),
                layouts: call.bindings().iter().map(|b| b.layout).collect(),
                kinds: call.bindings().iter().map(|b| b.kind).collect(),
            });
            Ok(Done::new())
        }
    }

    // ---- CPU side: the real `gust::cpu` backend, with `Run` implemented per test kernel in the macro's shape ----
    //
    // The work finishes inside `submit_call` (the job is not tied to the buffers' lifetime). The backend contract of the
    // module documentation (validation, aliasing, lengths) lives in `cpu::validate`.

    impl Run<Scale, 3> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Scale, 3>) -> Result<CpuJob, CpuError> {
            cpu::validate(self, call)?;
            let [a, settings, out] = *call.bindings();
            let a = cpu::read::<f32>(a, "a")?;
            let settings = cpu::read::<f32>(settings, "settings")?;
            let mut out = cpu::write::<f32>(out, "out")?;
            for id in cpu::invocations(call.kernel(), call.extent()) {
                scale_body(
                    <Invocation as FromGlobalId>::from_global_id(id),
                    Storage::new(&a),
                    Storage::new(&settings),
                    StorageMut::new(&mut out),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    impl Run<Sum2, 3> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Sum2, 3>) -> Result<CpuJob, CpuError> {
            cpu::validate(self, call)?;
            let [a, b, out] = *call.bindings();
            let a = cpu::read::<f32>(a, "a")?;
            let b = cpu::read::<f32>(b, "b")?;
            let mut out = cpu::write::<f32>(out, "out")?;
            for id in cpu::invocations(call.kernel(), call.extent()) {
                sum2_body(
                    <Invocation as FromGlobalId>::from_global_id(id),
                    Storage::new(&a),
                    Storage::new(&b),
                    StorageMut::new(&mut out),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    impl Run<Swap2, 2> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Swap2, 2>) -> Result<CpuJob, CpuError> {
            cpu::validate(self, call)?;
            let [x, y] = *call.bindings();
            let mut x = cpu::write::<f32>(x, "x")?;
            let mut y = cpu::write::<f32>(y, "y")?;
            for id in cpu::invocations(call.kernel(), call.extent()) {
                swap2_body(
                    <Invocation as FromGlobalId>::from_global_id(id),
                    StorageMut::new(&mut x),
                    StorageMut::new(&mut y),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    // ---- tests ----

    // Execution and writeback through the typed call: 5 elements under a 64-wide workgroup, so the guard runs.
    #[test]
    fn cpu_executes_the_body_and_writes_back() -> Result<(), CpuError> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32; 5]);
        Scale::dispatch(&cpu, 5, &a, Independent(&settings), &out)?;
        assert_eq!(out.read(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        Ok(())
    }

    #[test]
    fn cpu_enforces_the_shared_length_rule() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32; 4]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32; 5]);
        assert_eq!(
            Scale::dispatch(&cpu, 4, &a, Independent(&settings), &out),
            Err(CpuError::BufferLength {
                parameter: "out",
                expected: 4,
                actual: 5
            })
        );
        // Without `Independent`, the one-element settings buffer is a length error too.
        assert_eq!(
            Scale::dispatch(&cpu, 4, &a, &settings, &cpu.buffer(&[0.0f32; 4])),
            Err(CpuError::BufferLength {
                parameter: "settings",
                expected: 4,
                actual: 1
            })
        );
        // An empty independent buffer is rejected while there is work to do.
        let empty = cpu.buffer::<f32>(&[]);
        assert_eq!(
            Scale::dispatch(&cpu, 4, &a, Independent(&empty), &cpu.buffer(&[0.0f32; 4])),
            Err(CpuError::IndependentLengthEmpty {
                parameter: "settings"
            })
        );
    }

    // RO/RO: accepted.
    #[test]
    fn alias_read_read_is_accepted() -> Result<(), CpuError> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0]);
        let out = cpu.buffer(&[0.0f32; 3]);
        Sum2::dispatch(&cpu, 3, &a, &a, &out)?;
        assert_eq!(out.read(), vec![2.0, 4.0, 6.0]);
        Ok(())
    }

    // RO/RW: compiles, rejected at run time by buffer identity with both parameters named, buffer untouched.
    #[test]
    fn alias_read_write_is_rejected_at_run_time() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0]);
        let settings = cpu.buffer(&[2.0f32]);
        assert_eq!(
            Scale::dispatch(&cpu, 3, &a, Independent(&settings), &a),
            Err(CpuError::Aliased {
                kernel: "typed_tests::scale".to_string(),
                first: "a",
                second: "out"
            })
        );
        assert_eq!(a.read(), vec![1.0, 2.0, 3.0]);
    }

    // RW/RW: compiles, rejected at run time, buffer untouched; two distinct buffers swap.
    #[test]
    fn alias_write_write_is_rejected_at_run_time() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0]);
        assert_eq!(
            Swap2::dispatch(&cpu, 2, &a, &a),
            Err(CpuError::Aliased {
                kernel: "typed_tests::swap2".to_string(),
                first: "x",
                second: "y"
            })
        );
        assert_eq!(a.read(), vec![1.0, 2.0]);
        let b = cpu.buffer(&[9.0f32, 8.0]);
        Swap2::dispatch(&cpu, 2, &a, &b).unwrap();
        assert_eq!((a.read(), b.read()), (vec![9.0, 8.0], vec![1.0, 2.0]));
    }

    // Aliasing is decided by identity before any borrow, and an unrelated outside borrow is a
    // different error, not an alias.
    #[test]
    fn alias_is_identity_and_an_outside_borrow_is_not_an_alias() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0]);
        let b = cpu.buffer(&[3.0f32, 4.0]);
        let settings = cpu.buffer(&[1.0f32]);
        // The read-write parameter first or last: both are aliases of `a`.
        assert_eq!(
            Swap2::dispatch(&cpu, 2, &b, &b),
            Err(CpuError::Aliased {
                kernel: "typed_tests::swap2".to_string(),
                first: "x",
                second: "y"
            })
        );
        assert_eq!(
            Scale::dispatch(&cpu, 2, &a, Independent(&settings), &a),
            Err(CpuError::Aliased {
                kernel: "typed_tests::scale".to_string(),
                first: "a",
                second: "out"
            })
        );
        // Outside borrows of distinct buffers.
        {
            let _held = b.cells().borrow();
            assert_eq!(
                Sum2::dispatch(&cpu, 2, &a, &a, &b),
                Err(CpuError::Borrowed { parameter: "out" })
            );
        }
        {
            let _held = a.cells().borrow_mut();
            assert_eq!(
                Sum2::dispatch(&cpu, 2, &a, &a, &b),
                Err(CpuError::Borrowed { parameter: "a" })
            );
        }
        assert_eq!(b.read(), vec![3.0, 4.0], "no rejected call wrote anything");
    }

    // A hand-built `Call` is not covered by the compile-time checks: the backend checks every binding against
    // `call.kernel()`.
    #[test]
    fn hand_built_calls_are_checked_against_the_descriptor() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32]);
        let shared = LengthRule::Shared;
        let independent = LengthRule::Independent;
        let (ro, rw) = (Access::ReadOnly, Access::ReadWrite);

        // The right call, built by hand, runs.
        let call = Call::<CpuDevice, Scale, 3>::new(
            1,
            [
                a.bind(ro, shared),
                settings.bind(ro, independent),
                out.bind(rw, shared),
            ],
        );
        cpu.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), vec![2.0]);

        // Wrong access on the first parameter.
        let call = Call::<CpuDevice, Scale, 3>::new(
            1,
            [
                a.bind(rw, shared),
                settings.bind(ro, independent),
                out.bind(rw, shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(CpuError::Access {
                parameter: "a",
                expected: ro,
                actual: rw
            })
        );

        // Wrong order: the output buffer first (right count, right elements).
        let call = Call::<CpuDevice, Scale, 3>::new(
            1,
            [
                out.bind(rw, shared),
                settings.bind(ro, independent),
                a.bind(ro, shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(CpuError::Access {
                parameter: "a",
                expected: ro,
                actual: rw
            })
        );

        // Wrong element: a u32 buffer in the f32 parameter `a`.
        let integers = cpu.buffer(&[7u32]);
        let call = Call::<CpuDevice, Scale, 3>::new(
            1,
            [
                integers.bind(ro, shared),
                settings.bind(ro, independent),
                out.bind(rw, shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(CpuError::ElementType { parameter: "a" })
        );

        // Wrong device instance: same device type, another instance.
        let foreign = CpuDevice::new().buffer(&[0.0f32]);
        let call = Call::<CpuDevice, Scale, 3>::new(
            1,
            [
                a.bind(ro, shared),
                settings.bind(ro, independent),
                foreign.bind(rw, shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(CpuError::WrongDevice { parameter: "out" })
        );
        assert_eq!(foreign.read(), vec![0.0]);
        assert_eq!(out.read(), vec![2.0], "rejected calls wrote nothing");
    }

    // A reused `Call` carries no stale length: the backend measures the buffer at submit.
    #[test]
    fn a_reused_call_is_checked_against_the_current_length() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 1.0]);
        let settings = cpu.buffer(&[3.0f32]);
        let out = cpu.buffer(&[0.0f32; 2]);
        let call = Scale::call(2, &a, Independent(&settings), &out);
        cpu.dispatch_call(&call).unwrap();
        out.cells().borrow_mut().push(0.0);
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(CpuError::BufferLength {
                parameter: "out",
                expected: 2,
                actual: 3
            })
        );
    }

    // The same source line of `Scale::dispatch` as the CPU test, on the GPU-shaped device: records access, length
    // rule, element layout and lengths, executes nothing.
    #[test]
    fn recording_device_records_and_never_executes() {
        let rec = RecordingDevice::default();
        let a = RecBuffer::new(vec![1.0f32, 2.0, 3.0, 4.0]);
        let settings = RecBuffer::new(vec![2.0f32]);
        let out = RecBuffer::new(vec![0.0f32; 4]);
        Scale::dispatch(&rec, 4, &a, Independent(&settings), &out).unwrap();
        assert_eq!(
            *rec.calls.borrow(),
            vec![Recorded {
                kernel: "scale",
                element_count: 4,
                bindings: vec![
                    (a.id, 4, Access::ReadOnly, LengthRule::Shared),
                    (settings.id, 1, Access::ReadOnly, LengthRule::Independent),
                    (out.id, 4, Access::ReadWrite, LengthRule::Shared),
                ],
                layouts: vec![<f32 as GpuPod>::LAYOUT; 3],
                kinds: vec![BindingKind::Storage; 3],
            }]
        );
        assert_eq!(
            out.values,
            vec![0.0; 4],
            "the recording device must not execute"
        );
    }

    // Buffers of three scopes compile (the compile is the proof); `submit` returns a job to wait on.
    #[test]
    fn buffers_of_different_scopes_and_submit() -> Result<(), CpuError> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[3.0f32, 4.0]);
        {
            let settings = cpu.buffer(&[10.0f32]);
            {
                let out = cpu.buffer(&[0.0f32; 2]);
                let job = Scale::submit(&cpu, 2, &a, Independent(&settings), &out)?;
                job.wait()?;
                assert_eq!(out.read(), vec![30.0, 40.0]);
            }
        }
        Ok(())
    }

    // A `Call` is a plain value: build once, dispatch twice (what m2.e records in the graph).
    #[test]
    fn a_call_is_a_reusable_value() -> Result<(), CpuError> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 1.0]);
        let settings = cpu.buffer(&[3.0f32]);
        let out = cpu.buffer(&[0.0f32; 2]);
        let call = Scale::call(2, &a, Independent(&settings), &out);
        assert_eq!(call.kernel().name, "scale");
        assert_eq!(call.bindings().len(), 3);
        cpu.dispatch_call(&call)?;
        cpu.dispatch_call(&call)?;
        assert_eq!(out.read(), vec![3.0, 3.0]);

        let rec = RecordingDevice::default();
        let a = RecBuffer::new(vec![1.0f32]);
        let out = RecBuffer::new(vec![0.0f32]);
        let call = Scale::call(1, &a, Independent(&a), &out);
        rec.dispatch_call(&call).unwrap();
        rec.dispatch_call(&call).unwrap();
        let calls = rec.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], calls[1]);
        Ok(())
    }

    // Explicit `Arg` values take the same path as `&buffer` (reflexive `From`).
    #[test]
    fn explicit_args_take_the_same_path() -> Result<(), CpuError> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[2.0f32]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32]);
        Scale::dispatch(
            &cpu,
            1,
            Arg::from(&a),
            Arg::from(Independent(&settings)),
            Arg::from(&out),
        )?;
        assert_eq!(out.read(), vec![4.0]);
        Ok(())
    }

    // The Independent wrapper maps to the right rule, the plain reference to Shared.
    #[test]
    fn arg_conversions_set_the_length_rule() {
        let buffer = CpuDevice::new().buffer(&[0.0f32]);
        assert_eq!(Arg::from(&buffer).length, LengthRule::Shared);
        assert_eq!(
            Arg::from(Independent(&buffer)).length,
            LengthRule::Independent
        );
    }

    // `Call::new` takes its descriptor from `K`: two kernels of the same shape do not get each other's descriptor.
    #[test]
    fn call_carries_the_descriptor_of_its_kernel() {
        let buffer = RecBuffer::new(vec![0.0f32]);
        let scale = Scale::call(1, &buffer, &buffer, &buffer);
        let sum2 = Sum2::call(1, &buffer, &buffer, &buffer);
        assert_eq!(*scale.kernel(), Scale::DESCRIPTOR);
        assert_eq!(*sum2.kernel(), Sum2::DESCRIPTOR);
        assert_eq!(scale.kernel().name, "scale");
        assert_eq!(sum2.kernel().name, "sum2");
        assert_eq!(
            scale.bindings().len(),
            3,
            "the invocation parameter is not a binding"
        );
    }

    // N is the parameter count (storage and uniform): zero for a kernel without resource parameters.
    #[test]
    fn a_kernel_without_storage_parameters_builds_an_empty_call() {
        let call = Nothing::call::<RecordingDevice>(7);
        assert_eq!(call.kernel().name, "nothing");
        assert_eq!(call.element_count(), 7);
        assert!(call.bindings().is_empty());
        let rec = RecordingDevice::default();
        rec.dispatch_call(&call).unwrap();
        assert_eq!(rec.calls.borrow()[0].bindings, vec![]);
    }

    #[test]
    fn storage_parameter_count_counts_only_storage_kind() {
        assert_eq!(Scale::DESCRIPTOR.storage_parameter_count(), 3);
        assert_eq!(Swap2::DESCRIPTOR.storage_parameter_count(), 2);
        assert_eq!(Nothing::DESCRIPTOR.storage_parameter_count(), 0);
        assert_eq!(Shade::DESCRIPTOR.storage_parameter_count(), 2);
    }

    #[test]
    fn uniform_parameter_count_counts_only_uniform_kind() {
        assert_eq!(Shade::DESCRIPTOR.uniform_parameter_count(), 1);
        assert_eq!(Scale::DESCRIPTOR.uniform_parameter_count(), 0);
        assert_eq!(Nothing::DESCRIPTOR.uniform_parameter_count(), 0);
    }

    #[test]
    fn a_binding_kind_follows_the_parameter_kind() {
        assert_eq!(
            BindingKind::of(ParameterKind::Storage),
            Some(BindingKind::Storage)
        );
        assert_eq!(
            BindingKind::of(ParameterKind::Uniform),
            Some(BindingKind::Uniform)
        );
        assert_eq!(BindingKind::of(ParameterKind::Invocation), None);
        assert_eq!(BindingKind::of(ParameterKind::Value), None);
        assert_eq!(
            <kind::Storage as kind::ResourceKind>::BINDING,
            BindingKind::Storage
        );
        assert_eq!(
            <kind::Uniform as kind::ResourceKind>::BINDING,
            BindingKind::Uniform
        );
    }

    // `N` counts storage plus uniform parameters: three bindings for `Shade`, in parameter order, each with its kind.
    // The uniform is passed as a plain reference (a `Shared` argument) and still records `Independent`.
    #[test]
    fn a_call_counts_storage_plus_uniform_parameters_and_records_the_kinds() {
        let rec = RecordingDevice::default();
        let a = RecBuffer::new(vec![1.0f32, 2.0, 3.0, 4.0]);
        let factor = RecUniform::new(0.5f32);
        let out = RecBuffer::new(vec![0.0f32; 4]);
        let call = Shade::call(4, &a, &factor, &out);
        assert_eq!(call.bindings().len(), 3);
        assert_eq!(call.kernel().name, "shade");
        rec.dispatch_call(&call).unwrap();
        assert_eq!(
            *rec.calls.borrow(),
            vec![Recorded {
                kernel: "shade",
                element_count: 4,
                bindings: vec![
                    (a.id, 4, Access::ReadOnly, LengthRule::Shared),
                    (factor.id, 1, Access::ReadOnly, LengthRule::Independent),
                    (out.id, 4, Access::ReadWrite, LengthRule::Shared),
                ],
                layouts: vec![<f32 as GpuPod>::LAYOUT; 3],
                kinds: vec![
                    BindingKind::Storage,
                    BindingKind::Uniform,
                    BindingKind::Storage
                ],
            }]
        );
    }

    #[test]
    fn call_arg_and_independent_are_debug() {
        let buffer = RecBuffer::new(vec![0.0f32]);
        let call = Nothing::call::<RecordingDevice>(7);
        assert_eq!(
            format!("{call:?}"),
            "Call { kernel: \"nothing\", extent: Extent { x: 7, y: 1, z: 1 }, bindings: [] }"
        );
        assert!(format!("{:?}", Independent(&buffer.values)).starts_with("Independent("));
    }

    // ---- extents (slice m7.d) ----

    #[test]
    fn a_one_dimensional_call_is_the_linear_extent() {
        let call = Nothing::call::<RecordingDevice>(7);
        assert_eq!(call.extent(), Extent::linear(7));
        assert_eq!(call.extent(), Extent::new(7, 1, 1));
        assert_eq!(call.element_count(), 7);
        let from_extent = Call::<RecordingDevice, Nothing, 0>::new_extent(Extent::new(7, 1, 1), []);
        assert_eq!(format!("{call:?}"), format!("{from_extent:?}"));
        assert_eq!(Extent::from(7), Extent::linear(7));
        assert_eq!(Extent::from([2, 3, 4]), Extent::new(2, 3, 4));
        assert_eq!(Extent::from((2, 3, 4)), Extent::new(2, 3, 4));
        assert_eq!(Extent::new(2, 3, 4).to_array(), [2, 3, 4]);
    }

    #[test]
    fn a_multi_dimensional_call_keeps_its_extent_and_the_recording_device_sees_it() {
        let call = Call::<RecordingDevice, Nothing, 0>::new_extent(Extent::new(15, 17, 2), []);
        assert_eq!(call.extent(), Extent::new(15, 17, 2));
        // `element_count` is the x axis only; the extent is the whole.
        assert_eq!(call.element_count(), 15);
        RecordingDevice::default().dispatch_call(&call).unwrap();
    }

    #[test]
    fn the_checked_total_is_the_product_and_none_when_it_overflows() {
        assert_eq!(Extent::new(15, 17, 1).checked_total(), Some(255));
        assert_eq!(Extent::new(0, u32::MAX, u32::MAX).checked_total(), Some(0));
        assert_eq!(
            Extent::new(u32::MAX, u32::MAX, 1).checked_total(),
            Some(u64::from(u32::MAX) * u64::from(u32::MAX))
        );
        assert_eq!(
            Extent::new(u32::MAX, u32::MAX, u32::MAX).checked_total(),
            None
        );
        assert!(Extent::new(1, 0, 1).is_empty());
        assert!(!Extent::new(1, 1, 1).is_empty());
    }

    #[test]
    fn rounding_is_per_axis_and_the_total_cannot_overflow() {
        let extent = Extent::new(15, 17, 1);
        assert_eq!(extent.rounded_axes([16, 16, 1]), Some([16, 32, 1]));
        assert_eq!(extent.rounded_total([16, 16, 1]), Some(16 * 32));
        assert_eq!(extent.rounded_axes([0, 16, 1]), None);
        assert_eq!(extent.rounded_total([16, 1, 0]), None);
        // The raw product of this one is below 2^32, the rounded one is above (the boundary the backends share).
        let edge = Extent::new(65_537, 65_535, 1);
        assert_eq!(edge.checked_total(), Some((1 << 32) - 1));
        assert_eq!(edge.rounded_total([16, 16, 1]), Some(65_552 * 65_536));
        assert_eq!(
            Extent::new(u32::MAX, u32::MAX, u32::MAX).rounded_total([1, 1, 1]),
            Some(u128::from(u32::MAX).pow(3))
        );
        assert_eq!(Extent::MAX_INVOCATIONS, 1 << 32);
    }
}
