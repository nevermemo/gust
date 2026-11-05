//! The CPU runtime: it runs a `#[gpu]` kernel's Rust shadow body on the CPU. It is the runtime of the `cpu` target
//! (D21, D51) and the engine of the CPU twin (D22).
//!
//! For each kernel with a typed call the macro emits an `impl Run<Kernel, N> for CpuDevice` that calls the shadow body
//! once per invocation, and two kinds of build contain it:
//!
//! - **The `cpu` target** (`GUST_PROFILE=cpu`, D51): the impl is emitted in every build, uses no artifact, and the
//!   build holds no GPU runtime. This is a production build.
//! - **The twin** (D22): in a test build (`cfg(test)` of the crate that defines the kernel) or under `--cfg gust_dual`
//!   the impl is emitted beside the GPU path, and tests compare the GPU against it.
//!
//! `#[kernel(cpu = "unsupported")]` opts a kernel out of both. The tests of this module and of `dispatch` write the same
//! impl by hand for their own kernels. The module holds everything such an impl needs, so the emitted code contains no
//! downcast: `validate`, `read`, `write`, `invocations` and `FromGlobalId` (all `#[doc(hidden)]` macro support). In
//! both builds the kernel body is the Rust shadow body as written, executed one invocation after another; the one
//! exception is the generated map kernel of the `cpu` target (`#[gpu]` on a free function), which splits its output
//! over threads (`cpu::parallel`, macro support).
//!
//! A [`CpuDevice`] hands out [`CpuBuffer`]s; the typed `call`/`dispatch` functions of a kernel take them like the
//! buffers of any other backend:
//!
//! ```text
//! let device = gust::cpu::CpuDevice::new();
//! let input = device.buffer(&[1.0f32, 2.0, 3.0]);
//! let output = device.buffer(&[0.0f32; 3]);
//! my_kernel::scale::dispatch(&device, 3, &input, &output)?;
//! assert_eq!(output.read(), [2.0, 4.0, 6.0]);
//! ```
//!
//! Everything is safe Rust (`RefCell` borrows only). The work finishes inside `submit_call`, so the job is already
//! complete when it is returned.
//!
//! # Where the twin and the GPU differ
//!
//! The twin executes the shadow body with Rust semantics, compiled in the crate that defines the kernel: integer
//! overflow panics when that crate (the one that contains the body) has `overflow-checks` and wraps when it has not
//! (the macro `overflow_checks!()` tells which for the crate where it is written; the function [`overflow_checks()`] answers for `gust` itself),
//! division by zero panics, and floating-point results are rustc's, which the GPU may round differently.
//! Edge cases are documented limitations, never matched (D39); see `docs/SEMANTICS.md` for the contract and the
//! known-mismatch ledger.
#![forbid(unsafe_code)]

use core::any::Any;
use core::cell::{Ref, RefCell, RefMut};
use core::convert::Infallible;
use core::fmt;
use core::hint::black_box;
use core::time::Duration;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::texture::TextureUse;
use crate::{
    Access, BindingKind, Call, Device, Extent, GpuPod, Invocation, Job, JobStatus, Kernel,
    KernelDescriptor, LengthRule, ParameterKind, Resource, ResourceLayout, SamplerSet, TypeLayout,
    TypeLayoutKind, TypedDevice, UVec3, kind,
};

static NEXT_DEVICE: AtomicU32 = AtomicU32::new(1);

/// A CPU device instance; a [`CpuBuffer`] belongs to the instance that created it.
///
/// Two instances are different devices even though they share a type: a call that mixes their buffers is rejected
/// with [`Error::WrongDevice`]. Instance numbers come from a process-wide counter and wrap after 2^32 - 1 devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuDevice {
    id: u32,
}

impl CpuDevice {
    /// A new device instance with a fresh identity.
    pub fn new() -> Self {
        Self {
            id: NEXT_DEVICE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// A buffer of this device holding a copy of `values`.
    pub fn buffer<T: GpuPod>(&self, values: &[T]) -> CpuBuffer<T> {
        CpuBuffer {
            owner: self.id,
            cells: RefCell::new(values.to_vec()),
        }
    }

    /// A buffer of this device that takes `values` over without a copy (the zeroed output of a map call).
    pub(crate) fn buffer_from_vec<T: GpuPod>(&self, values: Vec<T>) -> CpuBuffer<T> {
        CpuBuffer {
            owner: self.id,
            cells: RefCell::new(values),
        }
    }

    /// A uniform of this device holding `value`.
    ///
    /// `T` follows the rule of `gust_wgpu::HeadlessDevice::create_uniform`: a padding-free `Std140V1` type with
    /// Rust's size and alignment. Anything else is [`Error::InvalidUniformType`] with the same wording, so a program
    /// that the GPU refuses is refused here too. The descriptor's `Std140V1` layout of the parameter is compared
    /// with `T::LAYOUT` at submit, as for a buffer.
    pub fn uniform<T: GpuPod>(&self, value: T) -> Result<CpuUniform<T>, Error> {
        if let Some(reason) = uniform_type_refusal(T::LAYOUT, size_of::<T>(), align_of::<T>()) {
            return Err(Error::InvalidUniformType {
                type_name: T::LAYOUT.name,
                reason,
            });
        }
        Ok(CpuUniform {
            owner: self.id,
            cell: UniformCell(RefCell::new(value)),
        })
    }
}

/// Whether `layout` has no padding byte anywhere (the rule of `gust_wgpu`'s `padding_free`).
const fn padding_free(layout: TypeLayout) -> bool {
    match layout.kind {
        TypeLayoutKind::Scalar(_) | TypeLayoutKind::Vector { .. } => true,
        TypeLayoutKind::Matrix {
            rows,
            cols,
            column_stride,
            ..
        } => column_stride == rows * 4 && column_stride * cols == layout.size,
        TypeLayoutKind::Struct(fields) => {
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
    }
}

/// Why a type of `layout`, `rust_size` and `rust_align` cannot be a uniform, or `None` when it can. The wording is
/// `gust_wgpu::check_uniform_type`'s, which the GPU backend applies in `create_uniform`.
fn uniform_type_refusal(
    layout: TypeLayout,
    rust_size: usize,
    rust_align: usize,
) -> Option<&'static str> {
    if !layout.is_std140_v1() {
        return Some(
            "its layout is not Std140V1 (a constant buffer takes scalars, 2- and 4-wide vectors, 4x4 matrices and \
             structs of them, each struct rounded up to 16 bytes; 3-wide vectors and 2x2 matrices do not qualify)",
        );
    }
    if layout.size as usize != rust_size || layout.alignment as usize != rust_align {
        return Some(
            "its Std140V1 size or alignment differs from the Rust type's; a struct is rounded up to 16 bytes in a \
             constant buffer, so add explicit padding fields (and `repr(C, align(16))` if needed) until the Rust size \
             is the Std140V1 size",
        );
    }
    if !padding_free(layout) {
        return Some(
            "it has padding bytes, which Rust leaves uninitialised; add explicit padding fields so that the fields \
             fill the whole struct (a Std140V1 struct is rounded up to 16 bytes)",
        );
    }
    None
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// What the erased binding points at: the cell, its element type and its current length.
trait CpuData {
    fn as_any(&self) -> &dyn Any;
    /// The length right now (a uniform holds one value, a texture `width * height` texels); `None` while the cell is
    /// mutably borrowed outside this call.
    fn current_len(&self) -> Option<usize>;
    /// The uses a texture was created with; `None` for anything that is not a texture.
    fn texture_uses(&self) -> Option<TextureUse> {
        None
    }
}

/// The cell of a [`CpuUniform`]: its own type, so that it cannot be mistaken for the cell of a buffer.
struct UniformCell<T>(RefCell<T>);

impl<T: GpuPod> CpuData for UniformCell<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn current_len(&self) -> Option<usize> {
        self.0.try_borrow().ok().map(|_| 1)
    }
}

impl<T: GpuPod> CpuData for RefCell<Vec<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn current_len(&self) -> Option<usize> {
        self.try_borrow().ok().map(|values| values.len())
    }
}

/// A typed buffer of one [`CpuDevice`].
#[derive(Debug)]
pub struct CpuBuffer<T: GpuPod> {
    owner: u32,
    cells: RefCell<Vec<T>>,
}

impl<T: GpuPod> CpuBuffer<T> {
    /// A copy of the current contents.
    ///
    /// # Panics
    ///
    /// If the buffer is mutably borrowed, which only a running kernel call does; use it between calls.
    pub fn read(&self) -> Vec<T> {
        self.cells.borrow().clone()
    }

    /// Replace the contents; the length must not change ([`Error::BufferLength`] otherwise, naming `values`). The
    /// buffer is untouched on an error, and [`Error::Borrowed`] is returned when it is borrowed elsewhere.
    pub fn write(&self, values: &[T]) -> Result<(), Error> {
        let mut cells = self.cells.try_borrow_mut().map_err(|_| Error::Borrowed {
            parameter: "values",
        })?;
        if cells.len() != values.len() {
            return Err(Error::BufferLength {
                parameter: "values",
                expected: cells.len(),
                actual: values.len(),
            });
        }
        cells.copy_from_slice(values);
        Ok(())
    }

    /// The number of elements.
    ///
    /// # Panics
    ///
    /// If the buffer is mutably borrowed, as for [`CpuBuffer::read`].
    pub fn len(&self) -> usize {
        self.cells.borrow().len()
    }

    /// Whether the buffer holds no element.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The cell itself, for tests that borrow it from outside a call.
    #[cfg(test)]
    pub(crate) fn cells(&self) -> &RefCell<Vec<T>> {
        &self.cells
    }
}

/// A typed uniform of one [`CpuDevice`]: one value that a kernel reads by copy.
///
/// The counterpart of `gust_wgpu::GpuUniform`: it is a [`Resource`] of kind [`kind::Uniform`], so a call takes it where
/// a kernel has a `Uniform<T>` parameter and rejects it where the kernel has a buffer. It has no element count, so it
/// is bound with an independent length whatever the argument says.
pub struct CpuUniform<T: GpuPod> {
    owner: u32,
    cell: UniformCell<T>,
}

impl<T: GpuPod> CpuUniform<T> {
    /// A copy of the current value.
    ///
    /// The cell is borrowed mutably only inside [`write`](Self::write), which returns before anything else can run,
    /// and a kernel call copies the value under a shared borrow, so this does not panic.
    pub fn read(&self) -> T {
        *self.cell.0.borrow()
    }

    /// Replace the value. [`Error::Borrowed`] (naming `value`) when it is borrowed elsewhere; untouched then.
    pub fn write(&self, value: T) -> Result<(), Error> {
        *self
            .cell
            .0
            .try_borrow_mut()
            .map_err(|_| Error::Borrowed { parameter: "value" })? = value;
        Ok(())
    }

    /// The cell itself, for tests that borrow it from outside a call.
    #[cfg(test)]
    pub(crate) fn cell(&self) -> &RefCell<T> {
        &self.cell.0
    }
}

impl<T: GpuPod + fmt::Debug> fmt::Debug for CpuUniform<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuUniform")
            .field("owner", &self.owner)
            .field("value", &self.cell.0)
            .finish()
    }
}

/// Erased view of one bound [`CpuBuffer`] or [`CpuUniform`]. It records no length: the backend measures it at submit.
#[derive(Clone, Copy)]
pub struct CpuBinding<'r> {
    data: &'r dyn CpuData,
    access: Access,
    length: LengthRule,
    layout: TypeLayout,
    owner: u32,
    kind: BindingKind,
}

impl fmt::Debug for CpuBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBinding")
            .field("kind", &self.kind)
            .field("access", &self.access)
            .field("length", &self.length)
            .field("layout", &self.layout)
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

/// A completed CPU submission: CPU work finishes inside `submit_call`, so there is nothing left to wait for.
#[derive(Clone, Copy, Debug)]
pub struct CpuJob {
    _complete: (),
}

impl CpuJob {
    /// A job that has already finished. Macro support.
    #[doc(hidden)]
    pub const fn complete() -> Self {
        Self { _complete: () }
    }
}

impl Job for CpuJob {
    type Error = Error;

    fn status(&self) -> Result<JobStatus, Error> {
        Ok(JobStatus::Complete)
    }

    fn wait_timeout(&self, _timeout: Duration) -> Result<JobStatus, Error> {
        Ok(JobStatus::Complete)
    }

    fn wait(&self) -> Result<(), Error> {
        Ok(())
    }
}

/// Why a CPU call was rejected. The wording follows `gust_wgpu::Error` where the cases are shared.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The binding's kind (storage buffer, uniform or texture) is not the one the descriptor declares for the parameter.
    BindingKind {
        /// The kernel parameter.
        parameter: &'static str,
        /// The kind the descriptor declares.
        expected: BindingKind,
        /// The kind the binding carries.
        actual: BindingKind,
    },
    /// A uniform parameter's resource layout is not `Std140V1` of an element that satisfies it; the counterpart of
    /// `gust_wgpu::Error::InvalidResourceLayout` for a uniform.
    InvalidResourceLayout {
        /// The kernel parameter.
        parameter: &'static str,
    },
    /// [`CpuDevice::uniform`] refused the type: it is not a padding-free `Std140V1` type. `reason` names what is
    /// wrong and the fix; the wording is `gust_wgpu::Error::InvalidUniformType`'s.
    InvalidUniformType {
        /// The type's layout name.
        type_name: &'static str,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// The binding's element layout (or type) is not the parameter's.
    ElementType {
        /// The kernel parameter.
        parameter: &'static str,
    },
    /// The binding's access is not the one the descriptor declares for the parameter.
    Access {
        /// The kernel parameter.
        parameter: &'static str,
        /// The access the descriptor declares.
        expected: Access,
        /// The access the binding carries.
        actual: Access,
    },
    /// The buffer, texture or sampler belongs to another device instance.
    WrongDevice {
        /// The kernel parameter.
        parameter: &'static str,
    },
    /// Two parameters are the same buffer and one of them is read-write.
    Aliased {
        /// The kernel, as `module::name`.
        kernel: String,
        /// The earlier parameter.
        first: &'static str,
        /// The later parameter.
        second: &'static str,
    },
    /// The buffer is borrowed by someone outside this call (not an alias within the call).
    Borrowed {
        /// The kernel parameter, or `values` for [`CpuBuffer::write`].
        parameter: &'static str,
    },
    /// A shared-length buffer does not hold exactly `element_count` elements.
    BufferLength {
        /// The kernel parameter, or `values` for [`CpuBuffer::write`].
        parameter: &'static str,
        /// The length required.
        expected: usize,
        /// The length found.
        actual: usize,
    },
    /// An independent-length buffer is empty while there is work to do.
    IndependentLengthEmpty {
        /// The kernel parameter.
        parameter: &'static str,
    },
    /// A workgroup size with a zero axis. The twin runs workgroups of any non-zero size in one, two or three
    /// dimensions (slice m7.d).
    UnsupportedWorkgroupSize {
        /// The kernel's workgroup size.
        size: [u32; 3],
    },
    /// A launch needs more than a limit of the CPU twin allows; the counterpart of `gust_wgpu::Error::DeviceLimit`.
    /// The only limit is `global_invocation_id` (a `u32` per axis): the extent with every axis rounded up to whole
    /// workgroups must not multiply to more than 2^32 invocations ([`crate::Extent::MAX_INVOCATIONS`]), or ids would
    /// wrap and re-run real elements. A width that is not a power of two can exceed it below `u32::MAX` elements
    /// (192 wide, `u32::MAX` elements: 4294967424). A launch with a zero axis is not checked.
    DeviceLimit {
        /// The kernel, as `module::name`.
        kernel: String,
        /// The limit that is exceeded.
        limit: &'static str,
        /// What the launch needs.
        requested: u64,
        /// What the twin allows.
        maximum: u64,
    },
    /// `GUST_CPU_THREADS` is set to something other than a positive integer (slice m7.c). The parallel map reports
    /// the value instead of falling back to a default, so a typo cannot silently change the thread count.
    InvalidThreads {
        /// The value found (non-Unicode text is shown lossily).
        value: String,
    },
    /// The operating system could not start a worker thread for the parallel map (slice m7.c). Nothing is returned;
    /// set `GUST_CPU_THREADS=1` to run the map on the calling thread.
    ThreadSpawn {
        /// The operating system's message.
        message: String,
    },
    /// A texture parameter is bound to a texture that was not created for the use the kernel makes of it (slice mx.t1):
    /// a sampled parameter needs [`TextureUse::SAMPLED`], a storage one [`TextureUse::STORAGE`]. The counterpart of
    /// `gust_wgpu::Error::TextureUse`.
    TextureUse {
        /// The kernel parameter.
        parameter: &'static str,
        /// The use the parameter needs.
        required: TextureUse,
        /// The uses the texture has.
        actual: TextureUse,
    },
    /// [`CpuDevice::texture_2d`] was asked for a texture with a zero side, or one with more than `u32::MAX` texels.
    TextureSize {
        /// The width asked for.
        width: u32,
        /// The height asked for.
        height: u32,
    },
    /// [`CpuTexture2d::write`](texture::CpuTexture2d::write) got a slice that is not `width * height` texels long.
    TextureLength {
        /// The number of texels the texture has.
        expected: usize,
        /// The number of texels given.
        actual: usize,
    },
    /// A texel address outside the texture, from [`CpuTexture2d::load`](texture::CpuTexture2d::load) or `store`. A
    /// kernel's own `load` and `store` panic instead (KM-11).
    TextureCoordinate {
        /// The x asked for.
        x: u32,
        /// The y asked for.
        y: u32,
        /// The texture's width.
        width: u32,
        /// The texture's height.
        height: u32,
    },
    /// A `pixel` kernel was called over an extent with `y` or `z` other than 1 (slice mx.p1b, mx.p1a review N3). A
    /// `pixel` call is one-dimensional, so the same program fails on every backend: this is the counterpart of
    /// `gust_wgpu::Error::ShapeUnsupported` for an extent call.
    PixelExtent {
        /// The kernel, as `module::name`.
        kernel: String,
        /// The extent asked for.
        extent: [u32; 3],
    },
    /// A `pixel` kernel's output (a read-write storage parameter) was bound with an independent length. A `pixel`
    /// output is `Shared`, so its length is the call's element count; the counterpart of
    /// `gust_wgpu::Error::ShapeIndependentOutput`.
    PixelIndependentOutput {
        /// The kernel, as `module::name`.
        kernel: String,
        /// The output parameter.
        parameter: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BindingKind {
                parameter,
                expected,
                actual,
            } => write!(
                f,
                "parameter `{parameter}` expects a {expected:?} binding, got a {actual:?} one"
            ),
            Self::InvalidResourceLayout { parameter } => write!(
                f,
                "parameter `{parameter}` requires a valid Std140V1 resource layout for a uniform"
            ),
            Self::InvalidUniformType { type_name, reason } => {
                write!(f, "`{type_name}` cannot be a uniform: {reason}")
            }
            Self::ElementType { parameter } => write!(
                f,
                "parameter `{parameter}` is bound to a buffer of another element type"
            ),
            Self::Access {
                parameter,
                expected,
                actual,
            } => write!(
                f,
                "parameter `{parameter}` expects {expected:?} access, got {actual:?}"
            ),
            Self::WrongDevice { parameter } => write!(
                f,
                "parameter `{parameter}` is bound to a buffer of another CPU device instance"
            ),
            Self::Aliased {
                kernel,
                first,
                second,
            } => write!(
                f,
                "parameters `{first}` and `{second}` of kernel `{kernel}` are the same buffer and at least one of them is written; a buffer may be bound twice only read-only (bind it once, or use separate buffers)"
            ),
            Self::Borrowed { parameter } => write!(
                f,
                "the buffer bound to `{parameter}` is borrowed outside this call"
            ),
            Self::BufferLength {
                parameter,
                expected,
                actual,
            } => write!(
                f,
                "parameter `{parameter}` expects {expected} elements, got {actual}"
            ),
            Self::IndependentLengthEmpty { parameter } => write!(
                f,
                "parameter `{parameter}` was bound with an independent length but the buffer is empty"
            ),
            Self::UnsupportedWorkgroupSize { size } => write!(
                f,
                "workgroup size {size:?} is not supported by the CPU twin (every axis must be non-zero)"
            ),
            Self::DeviceLimit {
                kernel,
                limit,
                requested,
                maximum,
            } => write!(
                f,
                "kernel `{kernel}` needs {requested} for `{limit}`, the limit is {maximum}; split the work into several dispatches"
            ),
            Self::InvalidThreads { value } => write!(
                f,
                "the environment variable `GUST_CPU_THREADS` must be a positive integer, got `{value}`"
            ),
            Self::ThreadSpawn { message } => write!(
                f,
                "could not start a worker thread for the parallel map ({message}); set GUST_CPU_THREADS=1 to run on the calling thread"
            ),
            Self::TextureUse {
                parameter,
                required,
                actual,
            } => write!(
                f,
                "parameter `{parameter}` needs a texture created with {required:?}, got one created with {actual:?}"
            ),
            Self::TextureSize { width, height } => write!(
                f,
                "a texture of {width}x{height} texels is not allowed: both sides must be non-zero and the texel count at most u32::MAX"
            ),
            Self::TextureLength { expected, actual } => write!(
                f,
                "the texture holds {expected} texels, {actual} were given"
            ),
            Self::TextureCoordinate {
                x,
                y,
                width,
                height,
            } => write!(
                f,
                "the texel ({x}, {y}) is outside the {width}x{height} texture"
            ),
            Self::PixelExtent { kernel, extent } => write!(
                f,
                "kernel `{kernel}` has the shape Pixel, which runs one-dimensional calls only; the extent {extent:?} has a y or z other than 1 (use `call`, not `call_extent`)"
            ),
            Self::PixelIndependentOutput { kernel, parameter } => write!(
                f,
                "output `{parameter}` of the pixel kernel `{kernel}` was bound with an independent length; a pixel output has exactly the call's element count (drop `Independent`)"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl Device for CpuDevice {
    type Error = Error;
    /// There is no erased CPU dispatch in M3 (a CPU kernel runs through [`Run`](crate::Run) only), so this type
    /// cannot be constructed and `submit` can only receive an empty batch.
    type Dispatch<'r> = Infallible;
    type Job<'d> = CpuJob;

    fn submit(&self, _dispatches: &[Infallible]) -> Result<CpuJob, Error> {
        Ok(CpuJob::complete())
    }
}

impl TypedDevice for CpuDevice {
    type Binding<'r> = CpuBinding<'r>;
}

impl<T: GpuPod> Resource for CpuBuffer<T> {
    type Element = T;
    type Device = CpuDevice;
    type Kind = kind::Storage;

    // Never borrows the cell, so it cannot panic while the buffer is borrowed.
    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> CpuBinding<'r> {
        CpuBinding {
            data: &self.cells,
            access,
            length,
            layout: T::LAYOUT,
            owner: self.owner,
            kind: BindingKind::Storage,
        }
    }
}

impl<T: GpuPod> Resource for CpuUniform<T> {
    type Element = T;
    type Device = CpuDevice;
    type Kind = kind::Uniform;

    // Never borrows the cell. A uniform has no element count, so the length rule is `Independent` whatever the
    // argument says (the rule would otherwise demand `element_count` elements of a single value).
    fn bind<'r>(&'r self, access: Access, _length: LengthRule) -> CpuBinding<'r> {
        CpuBinding {
            data: &self.cell,
            access,
            length: LengthRule::Independent,
            layout: T::LAYOUT,
            owner: self.owner,
            kind: BindingKind::Uniform,
        }
    }
}

/// The id range of a launch: `global_invocation_id` is a `u32` per axis, so at most 2^32 invocations
/// ([`Extent::MAX_INVOCATIONS`], the one rule of both backends).
const MAX_INVOCATIONS: u64 = Extent::MAX_INVOCATIONS;

/// Invocations of a launch: every axis of `extent` rounded up to whole workgroups of `size`, multiplied
/// ([`Extent::rounded_total`], in `u128`, so it cannot overflow). `size` has no zero axis: [`validate`] refuses one,
/// and [`invocations`] panics on it.
fn launch_invocations(size: [u32; 3], extent: Extent) -> u128 {
    extent
        .rounded_total(size)
        .expect("a workgroup size without a zero axis")
}

/// The backend contract once for every kernel: every binding against `call.kernel()`, then aliasing by identity, then
/// the launch limit. Nothing is borrowed for writing before this returns `Ok`, so a rejected call leaves every buffer
/// untouched.
///
/// The check order follows `gust_wgpu`'s `validate_persistent_dispatch` (workgroup shape, then per binding the
/// element layout, device, access and length, then aliasing in declaration order, then the launch limit), so a call
/// that is wrong in two ways reports the same cause on both backends. One difference: `Borrowed` (a buffer that is
/// borrowed outside the call; a GPU buffer cannot be) is reported after access and before length.
///
/// The call's sampler bindings (the `X` of a [`SampledCall`](crate::SampledCall); none for `()`) are checked here too,
/// after the resource bindings: a sampler of another device instance is [`Error::WrongDevice`] naming the kernel's
/// sampler parameter. A `Run` impl that validates through this function cannot skip that check.
///
/// Macro support.
#[doc(hidden)]
pub fn validate<K: Kernel, const N: usize, X>(
    device: &CpuDevice,
    call: &Call<'_, CpuDevice, K, N, X>,
) -> Result<(), Error>
where
    X: SamplerSet,
    X::Binding: texture::SamplerOwner,
{
    let kernel = call.kernel();
    let size = kernel.workgroup_size;
    if size.contains(&0) {
        return Err(Error::UnsupportedWorkgroupSize { size });
    }
    let extent = call.extent();
    // A `pixel` call is one-dimensional (Option S, slice mx.p1b): the refusal of `gust_wgpu`'s `validate_dispatch_extent`.
    let pixel = kernel.shape == crate::Shape::Pixel;
    if pixel && (extent.y != 1 || extent.z != 1) {
        return Err(Error::PixelExtent {
            kernel: kernel.qualified_name().to_string(),
            extent: extent.to_array(),
        });
    }
    let parameters: Vec<_> = kernel
        .parameters
        .iter()
        .filter(|parameter| BindingKind::of(parameter.kind).is_some())
        .collect();
    // `Run<K, N>` makes a mismatch unreachable (`Call::new` asserts `N` equals the storage, uniform and texture
    // parameter count at compile time); a hand-built `Call` cannot differ either. The assertion documents that the zip
    // below is total. A sampler parameter is not one of them (`BindingKind::of` is `None` for it); the sampler
    // bindings are checked after the loop.
    debug_assert_eq!(parameters.len(), N);
    // A shared buffer holds exactly `x * y * z` elements. A product that overflows a `u64` or a `usize` is larger than
    // any buffer, so it saturates: the length check then fails for every buffer, as it must.
    let expected = extent
        .checked_total()
        .and_then(|total| usize::try_from(total).ok())
        .unwrap_or(usize::MAX);
    for (parameter, binding) in parameters.iter().zip(call.bindings().iter().copied()) {
        let name = parameter.name;
        // The kind comes first: the layout rule of a uniform differs from a buffer's, so everything after it assumes
        // the right kind of binding.
        if let Some(wanted) = BindingKind::of(parameter.kind)
            && binding.kind != wanted
        {
            return Err(Error::BindingKind {
                parameter: name,
                expected: wanted,
                actual: binding.kind,
            });
        }
        // A uniform is a constant buffer: its rule is `Std140V1` of an element that satisfies it (the check of
        // gust-wgpu's `validate_kernel`).
        if parameter.kind == ParameterKind::Uniform
            && !parameter.resource_layout.is_some_and(|layout| {
                layout.element.is_std140_v1() && layout == ResourceLayout::std140_v1(layout.element)
            })
        {
            return Err(Error::InvalidResourceLayout { parameter: name });
        }
        if Some(binding.layout) != parameter.resource_layout.map(|layout| layout.element) {
            return Err(Error::ElementType { parameter: name });
        }
        if binding.owner != device.id {
            return Err(Error::WrongDevice { parameter: name });
        }
        if binding.access != parameter.access {
            return Err(Error::Access {
                parameter: name,
                expected: parameter.access,
                actual: binding.access,
            });
        }
        // A `pixel` output is `Shared`: its length is the call's element count (the refusal of `gust_wgpu`).
        if pixel
            && parameter.kind == ParameterKind::Storage
            && parameter.access == Access::ReadWrite
            && binding.length == LengthRule::Independent
        {
            return Err(Error::PixelIndependentOutput {
                kernel: kernel.qualified_name().to_string(),
                parameter: name,
            });
        }
        // A texture must have been created for the use the kernel makes of it: a read-only parameter is a sampled
        // texture, a read-write one a storage texture (the descriptor's access is all a texture parameter has until
        // the texture layout of mx.t0b names the write-only form). A texture parameter with no access is malformed.
        if parameter.kind == ParameterKind::Texture {
            let required = match parameter.access {
                Access::ReadOnly => TextureUse::SAMPLED,
                Access::ReadWrite => TextureUse::STORAGE,
                Access::NotApplicable => {
                    return Err(Error::InvalidResourceLayout { parameter: name });
                }
            };
            let actual = binding.data.texture_uses().unwrap_or(TextureUse::NONE);
            if !actual.contains(required) {
                return Err(Error::TextureUse {
                    parameter: name,
                    required,
                    actual,
                });
            }
        }
        let len = binding
            .data
            .current_len()
            .ok_or(Error::Borrowed { parameter: name })?;
        // A uniform has no element count; its one value is never empty, so only the borrow above applies. A texture
        // has a width and a height instead, and no length rule either.
        match binding.length {
            _ if matches!(binding.kind, BindingKind::Uniform | BindingKind::Texture) => {}
            LengthRule::Shared if len != expected => {
                return Err(Error::BufferLength {
                    parameter: name,
                    expected,
                    actual: len,
                });
            }
            LengthRule::Independent if len == 0 && expected != 0 => {
                return Err(Error::IndependentLengthEmpty { parameter: name });
            }
            _ => {}
        }
    }
    // Each sampler binding belongs to this device instance, named by the kernel's sampler parameter in order.
    let sampler_parameters = kernel
        .parameters
        .iter()
        .filter(|parameter| parameter.kind == ParameterKind::Sampler);
    for (parameter, binding) in sampler_parameters.zip(call.samplers().sampler_bindings()) {
        if texture::SamplerOwner::owner(binding) != device.id {
            return Err(Error::WrongDevice {
                parameter: parameter.name,
            });
        }
    }
    // One buffer may be bound twice only when both bindings are read-only; the pair is named in declaration order.
    for (index, (first_parameter, first)) in parameters.iter().zip(call.bindings()).enumerate() {
        for (second_parameter, second) in parameters.iter().zip(call.bindings()).skip(index + 1) {
            let written = first.access == Access::ReadWrite || second.access == Access::ReadWrite;
            if written && core::ptr::addr_eq(first.data, second.data) {
                return Err(Error::Aliased {
                    kernel: kernel.qualified_name().to_string(),
                    first: first_parameter.name,
                    second: second_parameter.name,
                });
            }
        }
    }
    // The launch limit, after validation and not for an empty launch, as `check_launch` does on the GPU. The GPU
    // limits are `max_compute_workgroups_per_dimension` (65535 by default) per axis and the same invocation cap as
    // here ([`Extent::MAX_INVOCATIONS`], at most and rounded up to whole workgroups); the twin's only limit is
    // that cap.
    if !extent.is_empty() {
        // A `pixel` call covers at most `PIXEL_MAX_ELEMENTS` elements on every backend (Option S): the limit of
        // `gust_wgpu`'s `check_launch_limits`, named the same. Its workgroup size is `[1, 1, 1]` and means nothing.
        if pixel && extent.x > crate::PIXEL_MAX_ELEMENTS {
            return Err(Error::DeviceLimit {
                kernel: kernel.qualified_name().to_string(),
                limit: "pixel_max_elements",
                requested: u64::from(extent.x),
                maximum: u64::from(crate::PIXEL_MAX_ELEMENTS),
            });
        }
        let requested = launch_invocations(size, extent);
        if requested > u128::from(MAX_INVOCATIONS) {
            return Err(Error::DeviceLimit {
                kernel: kernel.qualified_name().to_string(),
                limit: "global_invocation_id",
                // Three rounded axes are below 2^96; the error carries a `u64`, so a launch beyond it saturates.
                requested: u64::try_from(requested).unwrap_or(u64::MAX),
                maximum: MAX_INVOCATIONS,
            });
        }
    }
    Ok(())
}

/// Borrow the bound buffer for reading as `Vec<T>`.
///
/// The element type is decided by `TypeId`, not by layout: two distinct types with one layout are different
/// element types here ([`Error::ElementType`]). The binding's access must be read-only ([`Error::Access`] otherwise,
/// checked after the element type): [`validate`] has already compared the binding's access with the descriptor's, so
/// this catches a call site that acquires a parameter in the wrong mode. This and [`write`] hold the only downcast of
/// the backend. Macro support.
#[doc(hidden)]
pub fn read<'b, T: GpuPod>(
    binding: CpuBinding<'b>,
    parameter: &'static str,
) -> Result<Ref<'b, Vec<T>>, Error> {
    require_kind(&binding, BindingKind::Storage, parameter)?;
    let cells = binding
        .data
        .as_any()
        .downcast_ref::<RefCell<Vec<T>>>()
        .ok_or(Error::ElementType { parameter })?;
    if binding.access != Access::ReadOnly {
        return Err(Error::Access {
            parameter,
            expected: Access::ReadOnly,
            actual: binding.access,
        });
    }
    cells
        .try_borrow()
        .map_err(|_| Error::Borrowed { parameter })
}

/// Borrow the bound buffer for writing as `Vec<T>`; the same rules as [`read`], with a read-write binding required.
/// Macro support.
#[doc(hidden)]
pub fn write<'b, T: GpuPod>(
    binding: CpuBinding<'b>,
    parameter: &'static str,
) -> Result<RefMut<'b, Vec<T>>, Error> {
    require_kind(&binding, BindingKind::Storage, parameter)?;
    let cells = binding
        .data
        .as_any()
        .downcast_ref::<RefCell<Vec<T>>>()
        .ok_or(Error::ElementType { parameter })?;
    if binding.access != Access::ReadWrite {
        return Err(Error::Access {
            parameter,
            expected: Access::ReadWrite,
            actual: binding.access,
        });
    }
    cells
        .try_borrow_mut()
        .map_err(|_| Error::Borrowed { parameter })
}

/// Read the bound uniform by copy.
///
/// The kind must be [`BindingKind::Uniform`] ([`Error::BindingKind`] otherwise, checked first), the value type is
/// decided by `TypeId` like [`read`] ([`Error::ElementType`]), the binding's access must be read-only
/// ([`Error::Access`]), and the uniform must not be mutably borrowed ([`Error::Borrowed`]). The shadow body receives
/// the copy as `Uniform(value)`, so it cannot write through it. Macro support.
#[doc(hidden)]
pub fn read_uniform<T: GpuPod>(
    binding: CpuBinding<'_>,
    parameter: &'static str,
) -> Result<T, Error> {
    require_kind(&binding, BindingKind::Uniform, parameter)?;
    let cell = binding
        .data
        .as_any()
        .downcast_ref::<UniformCell<T>>()
        .ok_or(Error::ElementType { parameter })?;
    if binding.access != Access::ReadOnly {
        return Err(Error::Access {
            parameter,
            expected: Access::ReadOnly,
            actual: binding.access,
        });
    }
    cell.0
        .try_borrow()
        .map(|value| *value)
        .map_err(|_| Error::Borrowed { parameter })
}

/// The binding must be of kind `wanted`.
fn require_kind(
    binding: &CpuBinding<'_>,
    wanted: BindingKind,
    parameter: &'static str,
) -> Result<(), Error> {
    if binding.kind == wanted {
        Ok(())
    } else {
        Err(Error::BindingKind {
            parameter,
            expected: wanted,
            actual: binding.kind,
        })
    }
}

/// Global invocation ids for whole workgroups, like the GPU: every axis of `extent` rounded up to a multiple of the
/// workgroup size along that axis, so shadow bodies must guard with `.len()` (or the extent) as GPU bodies do. The ids
/// run over `z`, then `y`, then `x` (x fastest), each axis from 0 below its rounded extent; the id of an invocation is
/// its `(x, y, z)`, so `LocalId` (`id % size`) and `GroupId` (`id / size`) are exact in two and three dimensions.
/// Macro support.
///
/// [`validate`] rejects a launch whose rounded-up total exceeds 2^32 ([`Error::DeviceLimit`]); this function is not
/// reachable with such a launch from a validated call. A direct caller who skips `validate` gets the first 2^32 ids
/// (the range is capped, a debug build also asserts), never a wrapped id that would re-run real elements.
#[doc(hidden)]
pub fn invocations(
    kernel: &KernelDescriptor,
    extent: Extent,
) -> impl Iterator<Item = UVec3> + use<> {
    let total = launch_invocations(kernel.workgroup_size, extent);
    debug_assert!(
        total <= u128::from(MAX_INVOCATIONS),
        "{total} invocations exceed the u32 id range; `validate` rejects this launch"
    );
    let [x, y, z] = extent
        .rounded_axes(kernel.workgroup_size)
        .expect("a workgroup size without a zero axis");
    // `u64` so that rounding up near `u32::MAX` cannot overflow, and the cap keeps every id below 2^32 (a width that
    // is not a power of two, 192, can round `u32::MAX` elements up past it, and `x as u32` would wrap).
    let total = u64::try_from(total.min(u128::from(MAX_INVOCATIONS))).unwrap_or(MAX_INVOCATIONS);
    if y == 1 && z == 1 {
        // One dimension, and the D51 CPU target: a plain range, no per-id work beyond the cast.
        Invocations::Linear(0..total)
    } else {
        Invocations::Grid {
            at: [0; 3],
            size: [x, y],
            remaining: total,
        }
    }
}

/// The ids of [`invocations`]: a plain range for a launch of one row, else a counter that carries from `x` into `y`
/// and `z` without dividing.
enum Invocations {
    Linear(std::ops::Range<u64>),
    Grid {
        /// The next id, `x` fastest.
        at: [u64; 3],
        /// The rounded extent along `x` and `y`; `z` ends the walk through `remaining`.
        size: [u64; 2],
        remaining: u64,
    },
}

impl Iterator for Invocations {
    type Item = UVec3;

    fn next(&mut self) -> Option<UVec3> {
        match self {
            Self::Linear(range) => range.next().map(|x| UVec3 {
                x: x as u32,
                y: 0,
                z: 0,
            }),
            Self::Grid {
                at,
                size,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                let id = UVec3 {
                    x: at[0] as u32,
                    y: at[1] as u32,
                    z: at[2] as u32,
                };
                at[0] += 1;
                if at[0] == size[0] {
                    at[0] = 0;
                    at[1] += 1;
                    if at[1] == size[1] {
                        at[1] = 0;
                        at[2] += 1;
                    }
                }
                Some(id)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = match self {
            Self::Linear(range) => range.end - range.start,
            Self::Grid { remaining, .. } => *remaining,
        };
        match usize::try_from(left) {
            Ok(left) => (left, Some(left)),
            Err(_) => (usize::MAX, None),
        }
    }
}

/// Builds the invocation parameter of a shadow body from the global invocation id. Macro support.
#[doc(hidden)]
pub trait FromGlobalId {
    /// The value of the kernel's invocation parameter for the invocation `id`.
    fn from_global_id(id: UVec3) -> Self;
}

impl FromGlobalId for UVec3 {
    fn from_global_id(id: UVec3) -> Self {
        id
    }
}

impl FromGlobalId for Invocation {
    fn from_global_id(id: UVec3) -> Self {
        Invocation::new(id)
    }
}

/// Distance between two `f32` values in units in the last place: the number of representable values between them
/// plus one, counted in representation order. `-0.0` and `0.0` are the same point (distance 0), values on opposite
/// sides of zero are counted through zero, and the distance is `u64::MAX` if either value is NaN.
///
/// The comparison assertions of the twin tests use it instead of per-example copies.
pub fn ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    // Map the bit pattern to a signed key that grows with the value: negative floats count down from zero.
    fn key(x: f32) -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    }
    key(a).abs_diff(key(b))
}

/// Whether integer `+ - *` and negation panic on overflow in a given crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OverflowChecks {
    /// Overflow panics (`overflow-checks` is on, the default of the dev profile).
    Checked,
    /// Overflow wraps modulo 2^bits of the type (`overflow-checks` is off, the default of the release profile).
    Wrapping,
}

/// How many times the cache of [`overflow_checks()`] ran the probe in this test process.
#[cfg(test)]
static OVERFLOW_PROBES: AtomicU32 = AtomicU32::new(0);

fn probe_overflow_checks() -> OverflowChecks {
    let probe = std::panic::catch_unwind(|| black_box(u32::MAX) + black_box(1));
    if probe.is_err() {
        OverflowChecks::Checked
    } else {
        OverflowChecks::Wrapping
    }
}

/// The overflow mode of the `gust` crate itself, detected at run time (D38).
///
/// Rust decides overflow behaviour per crate when the crate is compiled, so this function answers for the crate
/// that contains it, `gust`. A twin is compiled in the crate that defines its kernel, and twin tests must key their
/// expectations on that crate, the one that contains the shadow body: use the macro [`overflow_checks!`](crate::cpu::overflow_checks!)
/// there. The function and the macro agree whenever one profile covers the whole dependency graph (the default);
/// they differ under a `[profile.<name>.package.gust]` (or `package.<your-crate>`) override of `overflow-checks`.
///
/// The first call runs `u32::MAX + 1` under `std::panic::catch_unwind` and caches the result for the process
/// (`OnceLock`).
///
/// - In a checked build the first call prints one panic message ("attempt to add with overflow") to stderr, through
///   the panic hook that is installed at that moment. Call it early (before a test that asserts on stderr) if that
///   matters.
/// - It needs the default `panic = "unwind"`; with `panic = "abort"` a checked build aborts on the first call.
///
/// In this workspace, `cargo test` (dev profile) is `Checked` and `cargo test --release` is `Wrapping`, because
/// the root `Cargo.toml` sets no `overflow-checks`; that is a fact about these profiles, not a general rule.
pub fn overflow_checks() -> OverflowChecks {
    static MODE: OnceLock<OverflowChecks> = OnceLock::new();
    *MODE.get_or_init(|| {
        #[cfg(test)]
        OVERFLOW_PROBES.fetch_add(1, Ordering::Relaxed);
        probe_overflow_checks()
    })
}

/// The overflow mode of the crate where this macro is written, as an [`OverflowChecks`] (D38).
///
/// This is the form twin tests use: the twin runs the shadow body compiled in the crate that defines the kernel, so
/// integer overflow in it panics exactly when that crate is built with `overflow-checks` (a dependent crate's own
/// setting does not reach the twin of a dependency's kernel). The macro expands the probe
/// (`u32::MAX + 1` under `std::panic::catch_unwind`) in the calling crate, so it reports that crate's mode, also
/// under a per-package `overflow-checks` override that makes the function [`overflow_checks()`](crate::cpu::overflow_checks())
/// answer for `gust` instead. The result is cached once per call site (a `static OnceLock` in the expansion); two
/// call sites probe twice. The notes of the function apply: the first probe of a checked build prints one panic
/// message to stderr, and `panic = "abort"` aborts there.
///
/// ```
/// use gust::cpu::{OverflowChecks, overflow_checks};
///
/// let mode = overflow_checks!();
/// assert_eq!(mode == OverflowChecks::Checked, cfg!(debug_assertions));
/// ```
///
/// (The doctest assertion holds in this workspace's dev and release profiles, not in general.)
#[doc(hidden)]
#[macro_export]
macro_rules! __gust_overflow_checks {
    () => {
        $crate::__gust_overflow_checks!(@probed $crate::cpu::__no_hook)
    };
    // Internal arm: `$on_probe` runs once per call site, inside the cache, and the tests count probes with it. It is
    // reachable by users (the macro is exported) but is not a supported spelling.
    (@probed $on_probe:expr) => {{
        static __GUST_OVERFLOW_MODE: ::std::sync::OnceLock<$crate::cpu::OverflowChecks> =
            ::std::sync::OnceLock::new();
        *__GUST_OVERFLOW_MODE.get_or_init(|| {
            ($on_probe)();
            // The closure is written here, so rustc compiles its `+` under the calling crate's setting.
            let __gust_probe = ::std::panic::catch_unwind(|| {
                ::std::hint::black_box(::core::primitive::u32::MAX)
                    + ::std::hint::black_box(1_u32)
            });
            if __gust_probe.is_err() {
                $crate::cpu::OverflowChecks::Checked
            } else {
                $crate::cpu::OverflowChecks::Wrapping
            }
        })
    }};
}

#[doc(inline)]
pub use crate::__gust_overflow_checks as overflow_checks;

/// Macro support: the probe hook of the default expansion does nothing.
#[doc(hidden)]
pub fn __no_hook() {}

#[doc(hidden)]
pub mod parallel;

pub mod texture;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Arg, Independent, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout, Run,
        SV_DispatchThreadID, SourceOrigin, Storage, StorageMut, Uniform, math::Vec4,
    };

    // ---- two hand-written kernels in the form the macro emits (this module is the macro's specification) ----

    const fn invocation(rust_type: &'static str) -> ParameterDescriptor {
        ParameterDescriptor {
            name: "id",
            rust_type,
            kind: ParameterKind::Invocation,
            access: Access::NotApplicable,
            resource_layout: None,
            binding: None,
        }
    }

    const fn storage(
        name: &'static str,
        access: Access,
        binding: u32,
        element: TypeLayout,
    ) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "float",
            kind: ParameterKind::Storage,
            access,
            resource_layout: Some(ResourceLayout::storage_v1(element)),
            binding: Some(ResourceBinding { group: 0, binding }),
        }
    }

    const fn descriptor(
        name: &'static str,
        workgroup_size: [u32; 3],
        parameters: &'static [ParameterDescriptor],
    ) -> KernelDescriptor {
        KernelDescriptor {
            module: "cpu_tests",
            name,
            entry_point: name,
            shape: crate::Shape::Grid,
            compatible_shapes: &[crate::Shape::Grid],
            workgroup_size,
            parameters,
            slang_source: "",
            source: SourceOrigin::UNKNOWN,
            artifacts: &[],
        }
    }

    const F32: TypeLayout = <f32 as GpuPod>::LAYOUT;

    /// `scale(id: Invocation, a: RO, settings: RO, out: RW)`, workgroup 64.
    struct Scale;
    impl Kernel for Scale {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "scale",
            [64, 1, 1],
            &[
                invocation("Invocation"),
                storage("a", Access::ReadOnly, 0, F32),
                storage("settings", Access::ReadOnly, 1, F32),
                storage("out", Access::ReadWrite, 2, F32),
            ],
        );
    }

    /// `swap(id: SV_DispatchThreadID, x: RW, y: RW)`, workgroup 4: the `UVec3` spelling of the invocation.
    struct Swap;
    impl Kernel for Swap {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "swap",
            [4, 1, 1],
            &[
                invocation("SV_DispatchThreadID"),
                storage("x", Access::ReadWrite, 0, F32),
                storage("y", Access::ReadWrite, 1, F32),
            ],
        );
    }

    const fn uniform(name: &'static str, binding: u32, element: TypeLayout) -> ParameterDescriptor {
        ParameterDescriptor {
            name,
            rust_type: "float",
            kind: ParameterKind::Uniform,
            access: Access::ReadOnly,
            resource_layout: Some(ResourceLayout::std140_v1(element)),
            binding: Some(ResourceBinding { group: 0, binding }),
        }
    }

    /// `shade(id: Invocation, a: RO, factor: Uniform<f32>, out: RW)`, workgroup 64: two storage parameters and one
    /// uniform, so `N` is 3.
    struct Shade;
    impl Kernel for Shade {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "shade",
            [64, 1, 1],
            &[
                invocation("Invocation"),
                storage("a", Access::ReadOnly, 0, F32),
                uniform("factor", 1, F32),
                storage("out", Access::ReadWrite, 2, F32),
            ],
        );
    }

    /// `tint(id: Invocation, color: Uniform<Vec4>, out: RW)`: the uniform is a vector, the buffer holds `f32`.
    struct Tint;
    impl Kernel for Tint {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "tint",
            [64, 1, 1],
            &[
                invocation("Invocation"),
                uniform("color", 0, <Vec4 as GpuPod>::LAYOUT),
                storage("out", Access::ReadWrite, 1, F32),
            ],
        );
    }

    /// `tint` with its uniform labelled `StorageV2`: valid for the element but not for a constant buffer.
    struct TintStorageRule;
    impl Kernel for TintStorageRule {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "tint_storage_rule",
            [64, 1, 1],
            &[
                invocation("Invocation"),
                ParameterDescriptor {
                    resource_layout: Some(ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT)),
                    ..uniform("color", 0, <Vec4 as GpuPod>::LAYOUT)
                },
                storage("out", Access::ReadWrite, 1, F32),
            ],
        );
    }

    fn shade_body(
        id: Invocation,
        a: Storage<'_, f32>,
        factor: Uniform<f32>,
        mut out: StorageMut<'_, f32>,
    ) {
        let x = id.global_id().x;
        if x < out.len() {
            out[x] = a[x] * *factor;
        }
    }

    fn tint_body(id: Invocation, color: Uniform<Vec4>, mut out: StorageMut<'_, f32>) {
        let x = id.global_id().x;
        if x < out.len() {
            out[x] = color.x + color.w;
        }
    }

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

    #[allow(clippy::manual_swap)] // a shadow-shaped body: the shape is the point
    fn swap_body(id: SV_DispatchThreadID, mut x: StorageMut<'_, f32>, mut y: StorageMut<'_, f32>) {
        let i = id.x;
        if i < x.len() && i < y.len() {
            let t = x[i];
            x[i] = y[i];
            y[i] = t;
        }
    }

    // The exact shape the macro emits (S1, plan section 4): validate, destructure, acquire, loop, complete.
    impl Run<Scale, 3> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Scale, 3>) -> Result<CpuJob, Error> {
            validate(self, call)?;
            let [a, settings, out] = *call.bindings();
            let a = read::<f32>(a, "a")?;
            let settings = read::<f32>(settings, "settings")?;
            let mut out = write::<f32>(out, "out")?;
            for id in invocations(call.kernel(), call.extent()) {
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

    // The shape the macro will emit for a uniform (slice m6.u2): `read_uniform` by copy, wrapped in `Uniform`.
    impl Run<Shade, 3> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Shade, 3>) -> Result<CpuJob, Error> {
            validate(self, call)?;
            let [a, factor, out] = *call.bindings();
            let a = read::<f32>(a, "a")?;
            let factor = Uniform(read_uniform::<f32>(factor, "factor")?);
            let mut out = write::<f32>(out, "out")?;
            for id in invocations(call.kernel(), call.extent()) {
                shade_body(
                    <Invocation as FromGlobalId>::from_global_id(id),
                    Storage::new(&a),
                    factor,
                    StorageMut::new(&mut out),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    impl Run<Tint, 2> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Tint, 2>) -> Result<CpuJob, Error> {
            validate(self, call)?;
            let [color, out] = *call.bindings();
            let color = Uniform(read_uniform::<Vec4>(color, "color")?);
            let mut out = write::<f32>(out, "out")?;
            for id in invocations(call.kernel(), call.extent()) {
                tint_body(
                    <Invocation as FromGlobalId>::from_global_id(id),
                    color,
                    StorageMut::new(&mut out),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    impl Run<Swap, 2> for CpuDevice {
        fn submit_call(&self, call: &Call<'_, Self, Swap, 2>) -> Result<CpuJob, Error> {
            validate(self, call)?;
            let [x, y] = *call.bindings();
            let mut x = write::<f32>(x, "x")?;
            let mut y = write::<f32>(y, "y")?;
            for id in invocations(call.kernel(), call.extent()) {
                swap_body(
                    <SV_DispatchThreadID as FromGlobalId>::from_global_id(id),
                    StorageMut::new(&mut x),
                    StorageMut::new(&mut y),
                );
            }
            Ok(CpuJob::complete())
        }
    }

    type F32Buffer = CpuBuffer<f32>;

    fn scale_call<'r>(
        count: u32,
        a: impl Into<Arg<'r, F32Buffer>>,
        settings: impl Into<Arg<'r, F32Buffer>>,
        out: impl Into<Arg<'r, F32Buffer>>,
    ) -> Call<'r, CpuDevice, Scale, 3> {
        Call::new(
            count,
            [
                a.into().bind(Access::ReadOnly),
                settings.into().bind(Access::ReadOnly),
                out.into().bind(Access::ReadWrite),
            ],
        )
    }

    /// The call the macro will generate for `Shade`: a uniform takes `Resource<Kind = Uniform>`.
    fn shade_call<'r, U>(
        count: u32,
        a: impl Into<Arg<'r, F32Buffer>>,
        factor: impl Into<Arg<'r, U>>,
        out: impl Into<Arg<'r, F32Buffer>>,
    ) -> Call<'r, CpuDevice, Shade, 3>
    where
        U: Resource<Element = f32, Device = CpuDevice, Kind = kind::Uniform> + 'r,
    {
        Call::new(
            count,
            [
                a.into().bind(Access::ReadOnly),
                factor.into().bind(Access::ReadOnly),
                out.into().bind(Access::ReadWrite),
            ],
        )
    }

    fn swap_call<'r>(
        count: u32,
        x: impl Into<Arg<'r, F32Buffer>>,
        y: impl Into<Arg<'r, F32Buffer>>,
    ) -> Call<'r, CpuDevice, Swap, 2> {
        Call::new(
            count,
            [
                x.into().bind(Access::ReadWrite),
                y.into().bind(Access::ReadWrite),
            ],
        )
    }

    // ---- execution and writeback ----

    // 5 elements under a 64-wide workgroup: 59 invocations are past the end and the body's guard must hold them.
    #[test]
    fn execution_writes_back_through_the_typed_call() -> Result<(), Error> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[-1.0f32; 5]);
        let call = scale_call(5, &a, Independent(&settings), &out);
        cpu.dispatch_call(&call)?;
        assert_eq!(out.read(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(a.read(), vec![1.0, 2.0, 3.0, 4.0, 5.0], "inputs unchanged");
        Ok(())
    }

    // Two read-write parameters through the `UVec3` spelling of the invocation parameter.
    #[test]
    fn execution_with_the_uvec3_invocation_spelling() -> Result<(), Error> {
        let cpu = CpuDevice::new();
        let x = cpu.buffer(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = cpu.buffer(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
        cpu.dispatch_call(&swap_call(6, &x, &y))?;
        assert_eq!(x.read(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(y.read(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn an_empty_call_runs_nothing() -> Result<(), Error> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer::<f32>(&[]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer::<f32>(&[]);
        cpu.dispatch_call(&scale_call(0, &a, Independent(&settings), &out))?;
        assert!(out.is_empty());
        Ok(())
    }

    // ---- the length rules ----

    #[test]
    fn the_length_rules_are_enforced_at_submit() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32; 4]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32; 5]);
        assert_eq!(
            cpu.dispatch_call(&scale_call(4, &a, Independent(&settings), &out)),
            Err(Error::BufferLength {
                parameter: "out",
                expected: 4,
                actual: 5
            })
        );
        // Without `Independent`, the one-element settings buffer is a length error too.
        let out = cpu.buffer(&[0.0f32; 4]);
        assert_eq!(
            cpu.dispatch_call(&scale_call(4, &a, &settings, &out)),
            Err(Error::BufferLength {
                parameter: "settings",
                expected: 4,
                actual: 1
            })
        );
        // An empty independent buffer is rejected while there is work to do.
        let empty = cpu.buffer::<f32>(&[]);
        assert_eq!(
            cpu.dispatch_call(&scale_call(4, &a, Independent(&empty), &out)),
            Err(Error::IndependentLengthEmpty {
                parameter: "settings"
            })
        );
        assert_eq!(out.read(), vec![0.0; 4], "rejected calls wrote nothing");
    }

    // A reused `Call` carries no stale length: the backend measures the buffer at submit.
    #[test]
    fn a_reused_call_is_measured_at_submit() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 1.0]);
        let settings = cpu.buffer(&[3.0f32]);
        let out = cpu.buffer(&[0.0f32; 2]);
        let call = scale_call(2, &a, Independent(&settings), &out);
        cpu.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), vec![3.0, 3.0]);
        out.cells().borrow_mut().push(0.0);
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(Error::BufferLength {
                parameter: "out",
                expected: 2,
                actual: 3
            })
        );
    }

    // ---- aliasing ----

    #[test]
    fn alias_read_read_is_accepted() -> Result<(), Error> {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0]);
        let out = cpu.buffer(&[0.0f32; 3]);
        // `a` is both the input and the (independent-length) settings buffer: `out[i] = a[i] * a[0]`.
        cpu.dispatch_call(&scale_call(3, &a, Independent(&a), &out))?;
        assert_eq!(out.read(), vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn alias_read_write_is_rejected_with_both_names_and_the_buffer_untouched() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0]);
        let settings = cpu.buffer(&[2.0f32]);
        assert_eq!(
            cpu.dispatch_call(&scale_call(3, &a, Independent(&settings), &a)),
            Err(Error::Aliased {
                kernel: "cpu_tests::scale".to_string(),
                first: "a",
                second: "out"
            })
        );
        assert_eq!(a.read(), vec![1.0, 2.0, 3.0]);
        // All three parameters are one buffer: reported at the first pair that has a writer, in declaration order.
        let b = cpu.buffer(&[4.0f32, 5.0, 6.0]);
        assert_eq!(
            cpu.dispatch_call(&scale_call(3, &b, Independent(&b), &b)),
            Err(Error::Aliased {
                kernel: "cpu_tests::scale".to_string(),
                first: "a",
                second: "out"
            })
        );
        assert_eq!(b.read(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn alias_write_write_is_rejected_and_two_distinct_buffers_swap() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0]);
        assert_eq!(
            cpu.dispatch_call(&swap_call(2, &a, &a)),
            Err(Error::Aliased {
                kernel: "cpu_tests::swap".to_string(),
                first: "x",
                second: "y"
            })
        );
        assert_eq!(a.read(), vec![1.0, 2.0]);
        let b = cpu.buffer(&[9.0f32, 8.0]);
        cpu.dispatch_call(&swap_call(2, &a, &b)).unwrap();
        assert_eq!((a.read(), b.read()), (vec![9.0, 8.0], vec![1.0, 2.0]));
    }

    // An outside borrow is `Borrowed`, not `Aliased`, and no rejected call wrote anything.
    #[test]
    fn an_outside_borrow_is_borrowed_and_not_an_alias() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0]);
        let b = cpu.buffer(&[3.0f32, 4.0]);
        let settings = cpu.buffer(&[1.0f32]);
        {
            let _held = b.cells().borrow();
            assert_eq!(
                cpu.dispatch_call(&scale_call(2, &a, Independent(&settings), &b)),
                Err(Error::Borrowed { parameter: "out" })
            );
        }
        {
            let _held = a.cells().borrow_mut();
            assert_eq!(
                cpu.dispatch_call(&scale_call(2, &a, Independent(&settings), &b)),
                Err(Error::Borrowed { parameter: "a" })
            );
            assert_eq!(
                a.write(&[0.0, 0.0]),
                Err(Error::Borrowed {
                    parameter: "values"
                })
            );
        }
        assert_eq!(b.read(), vec![3.0, 4.0], "no rejected call wrote anything");
        assert_eq!(a.read(), vec![1.0, 2.0]);
    }

    // ---- hand-built calls are checked against the descriptor ----

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
            Err(Error::Access {
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
            Err(Error::Access {
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
            Err(Error::ElementType { parameter: "a" })
        );

        // Wrong device instance: same device type, another instance.
        let other = CpuDevice::new();
        assert_ne!(cpu, other);
        let foreign = other.buffer(&[0.0f32]);
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
            Err(Error::WrongDevice { parameter: "out" })
        );
        assert_eq!(foreign.read(), vec![0.0]);
        assert_eq!(out.read(), vec![2.0], "rejected calls wrote nothing");
    }

    // ---- the element type is decided by `TypeId`, not by layout ----

    mod first {
        #[gust::gpu]
        pub mod pods {
            pub struct Pair {
                pub left: uint,
                pub right: uint,
            }
        }
    }

    mod second {
        #[gust::gpu]
        pub mod pods {
            pub struct Pair {
                pub left: uint,
                pub right: uint,
            }
        }
    }

    /// `pairs(id, p: RO)` over `first::pods::Pair`.
    struct Pairs;
    impl Kernel for Pairs {
        const DESCRIPTOR: KernelDescriptor = descriptor(
            "pairs",
            [1, 1, 1],
            &[
                invocation("Invocation"),
                storage(
                    "p",
                    Access::ReadOnly,
                    0,
                    <first::pods::Pair as GpuPod>::LAYOUT,
                ),
            ],
        );
    }

    #[test]
    fn read_decides_the_element_type_by_type_identity_not_by_layout() {
        use first::pods::Pair as First;
        use second::pods::Pair as Second;
        // Two distinct types with the same name and the same layout.
        assert_eq!(<First as GpuPod>::LAYOUT, <Second as GpuPod>::LAYOUT);
        assert_ne!(
            core::any::TypeId::of::<First>(),
            core::any::TypeId::of::<Second>()
        );

        let cpu = CpuDevice::new();
        let firsts = cpu.buffer(&[First { left: 1, right: 2 }]);
        let seconds = cpu.buffer(&[Second { left: 3, right: 4 }]);

        // `validate` compares layouts: a buffer of the other type passes it ...
        let call = Call::<CpuDevice, Pairs, 1>::new(
            1,
            [seconds.bind(Access::ReadOnly, LengthRule::Shared)],
        );
        assert_eq!(validate(&cpu, &call), Ok(()));
        // ... and `read` refuses it, naming the parameter; the right type is read.
        let [binding] = *call.bindings();
        assert_eq!(
            read::<First>(binding, "p").map(|values| values.len()),
            Err(Error::ElementType { parameter: "p" })
        );
        assert_eq!(
            write::<First>(binding, "p").map(|values| values.len()),
            Err(Error::ElementType { parameter: "p" })
        );
        let right = read::<Second>(binding, "p").unwrap();
        assert_eq!((right[0].left, right[0].right), (3, 4));
        drop(right);
        let first_binding = firsts.bind(Access::ReadOnly, LengthRule::Shared);
        let right = read::<First>(first_binding, "p").unwrap();
        assert_eq!((right[0].left, right[0].right), (1, 2));
    }

    #[test]
    fn read_and_write_refuse_a_binding_of_the_other_access() {
        let cpu = CpuDevice::new();
        let buffer = cpu.buffer(&[1.0f32, 2.0]);
        let read_only = buffer.bind(Access::ReadOnly, LengthRule::Shared);
        let read_write = buffer.bind(Access::ReadWrite, LengthRule::Shared);

        assert_eq!(
            write::<f32>(read_only, "x").map(|values| values.len()),
            Err(Error::Access {
                parameter: "x",
                expected: Access::ReadWrite,
                actual: Access::ReadOnly,
            })
        );
        assert_eq!(
            read::<f32>(read_write, "y").map(|values| values.len()),
            Err(Error::Access {
                parameter: "y",
                expected: Access::ReadOnly,
                actual: Access::ReadWrite,
            })
        );
        // The refused calls borrowed nothing; the right modes work.
        assert_eq!(read::<f32>(read_only, "x").unwrap().len(), 2);
        write::<f32>(read_write, "y").unwrap()[0] = 5.0;
        assert_eq!(buffer.read(), [5.0, 2.0]);
    }

    // ---- invocations ----

    #[test]
    fn invocations_cover_whole_workgroups() {
        let kernel = &Scale::DESCRIPTOR;
        for (count, expected) in [(0u32, 0usize), (1, 64), (63, 64), (64, 64), (65, 128)] {
            let ids: Vec<UVec3> = invocations(kernel, Extent::linear(count)).collect();
            assert_eq!(ids.len(), expected, "element count {count}");
            for (index, id) in ids.iter().enumerate() {
                assert_eq!(
                    *id,
                    UVec3 {
                        x: index as u32,
                        y: 0,
                        z: 0
                    }
                );
            }
        }
        // A workgroup of width 1 has no partial group, width 4 rounds up.
        let narrow = descriptor("narrow", [1, 1, 1], &[]);
        assert_eq!(invocations(&narrow, Extent::linear(5)).count(), 5);
        assert_eq!(invocations(&Swap::DESCRIPTOR, Extent::linear(5)).count(), 8);
    }

    // Two and three dimensions (slice m7.d): every axis rounds up to whole workgroups, the ids run z, then y, then x
    // (x fastest), and each id is its own `(x, y, z)`, so `id % size` and `id / size` are the local and group ids.
    #[test]
    fn invocations_cover_whole_workgroups_in_two_and_three_dimensions() {
        let tile = descriptor("tile", [4, 2, 1], &[]);
        let ids: Vec<UVec3> = invocations(&tile, Extent::new(5, 3, 1)).collect();
        // x rounds 5 up to 8, y rounds 3 up to 4, z stays 1: 32 invocations.
        assert_eq!(ids.len(), 32);
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(
                *id,
                UVec3 {
                    x: (index % 8) as u32,
                    y: (index / 8) as u32,
                    z: 0
                },
                "invocation {index}"
            );
        }
        let block = descriptor("block", [2, 2, 2], &[]);
        let ids: Vec<UVec3> = invocations(&block, Extent::new(3, 1, 3)).collect();
        // 4 * 2 * 4 = 32; z is the slowest axis.
        assert_eq!(ids.len(), 32);
        assert_eq!(ids[0], UVec3 { x: 0, y: 0, z: 0 });
        assert_eq!(ids[1], UVec3 { x: 1, y: 0, z: 0 });
        assert_eq!(ids[4], UVec3 { x: 0, y: 1, z: 0 });
        assert_eq!(ids[8], UVec3 { x: 0, y: 0, z: 1 });
        assert_eq!(ids[31], UVec3 { x: 3, y: 1, z: 3 });
        // The ids of a 15 x 17 launch of 16 x 16 workgroups are the 16 x 32 rectangle; the local and group ids
        // derived from them are exact.
        let grid = descriptor("grid", [16, 16, 1], &[]);
        let ids: Vec<UVec3> = invocations(&grid, Extent::new(15, 17, 1)).collect();
        assert_eq!(ids.len(), 16 * 32);
        for id in &ids {
            let (local, group) = ((id.x % 16, id.y % 16), (id.x / 16, id.y / 16));
            assert_eq!(id.x, group.0 * 16 + local.0);
            assert_eq!(id.y, group.1 * 16 + local.1);
            assert!(group.0 < 1 && group.1 < 2, "{id:?}");
        }
        // A zero axis runs nothing, whichever it is.
        for extent in [
            Extent::new(0, 5, 5),
            Extent::new(5, 0, 5),
            Extent::new(5, 5, 0),
        ] {
            assert_eq!(invocations(&block, extent).count(), 0, "{extent:?}");
        }
    }

    // Rounding up near `u32::MAX` does not overflow: 2^32 invocations, the last id is `u32::MAX`.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn invocations_near_the_u32_limit_do_not_overflow() {
        let total = 1usize << 32;
        assert_eq!(
            invocations(&Scale::DESCRIPTOR, Extent::linear(u32::MAX)).size_hint(),
            (total, Some(total))
        );
        // Two dimensions reach the same total: 65536 * 65536 (the ids themselves are not walked: 2^32 of them).
        let grid = descriptor("grid", [16, 16, 1], &[]);
        assert_eq!(
            invocations(&grid, Extent::new(65_536, 65_536, 1)).size_hint(),
            (total, Some(total))
        );
    }

    #[test]
    fn a_workgroup_size_with_a_zero_axis_is_rejected_before_anything_runs() {
        struct Wide;
        impl Kernel for Wide {
            const DESCRIPTOR: KernelDescriptor = descriptor("wide", [8, 0, 1], &[]);
        }
        struct Deep;
        impl Kernel for Deep {
            const DESCRIPTOR: KernelDescriptor = descriptor("deep", [8, 1, 0], &[]);
        }
        struct Empty;
        impl Kernel for Empty {
            const DESCRIPTOR: KernelDescriptor = descriptor("empty", [0, 1, 1], &[]);
        }
        let cpu = CpuDevice::new();
        assert_eq!(
            validate(&cpu, &Call::<CpuDevice, Wide, 0>::new(1, [])),
            Err(Error::UnsupportedWorkgroupSize { size: [8, 0, 1] })
        );
        assert_eq!(
            validate(&cpu, &Call::<CpuDevice, Deep, 0>::new(1, [])),
            Err(Error::UnsupportedWorkgroupSize { size: [8, 1, 0] })
        );
        assert_eq!(
            validate(&cpu, &Call::<CpuDevice, Empty, 0>::new(1, [])),
            Err(Error::UnsupportedWorkgroupSize { size: [0, 1, 1] })
        );
        assert_eq!(
            validate(
                &cpu,
                &Call::<CpuDevice, Swap, 2>::new(
                    0,
                    [
                        cpu.buffer::<f32>(&[])
                            .bind(Access::ReadWrite, LengthRule::Shared),
                        cpu.buffer::<f32>(&[])
                            .bind(Access::ReadWrite, LengthRule::Shared),
                    ]
                )
            ),
            Ok(())
        );
    }

    #[test]
    fn from_global_id_builds_both_spellings() {
        let id = UVec3 { x: 7, y: 0, z: 0 };
        assert_eq!(<UVec3 as FromGlobalId>::from_global_id(id), id);
        assert_eq!(
            <Invocation as FromGlobalId>::from_global_id(id),
            Invocation::new(id)
        );
        assert_eq!(
            <Invocation as FromGlobalId>::from_global_id(id)
                .global_id()
                .x,
            7
        );
    }

    // ---- devices, buffers, jobs, errors ----

    #[test]
    fn buffers_copy_their_input_and_write_keeps_the_length() {
        let cpu = CpuDevice::new();
        let source = [1u32, 2, 3];
        let buffer = cpu.buffer(&source);
        assert_eq!((buffer.len(), buffer.is_empty()), (3, false));
        buffer.write(&[4, 5, 6]).unwrap();
        assert_eq!(buffer.read(), vec![4, 5, 6]);
        assert_eq!(source, [1, 2, 3], "the buffer owns a copy");
        assert_eq!(
            buffer.write(&[1, 2]),
            Err(Error::BufferLength {
                parameter: "values",
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            buffer.read(),
            vec![4, 5, 6],
            "a rejected write changes nothing"
        );
        assert!(cpu.buffer::<u32>(&[]).is_empty());
    }

    #[test]
    fn device_instances_are_distinct_and_copies_are_the_same_device() {
        let (a, b) = (CpuDevice::new(), CpuDevice::new());
        assert_ne!(a, b);
        assert_eq!(a, a);
        assert_ne!(CpuDevice::default(), a);
    }

    #[test]
    fn the_job_is_complete_and_an_empty_batch_submits() {
        let cpu = CpuDevice::new();
        let job = cpu.submit(&[]).unwrap();
        assert_eq!(job.status(), Ok(JobStatus::Complete));
        assert_eq!(job.is_complete(), Ok(true));
        assert_eq!(job.wait_timeout(Duration::ZERO), Ok(JobStatus::Complete));
        assert_eq!(job.wait(), Ok(()));
        assert_eq!(Device::dispatch(&cpu, &[]), Ok(()));
        let a = cpu.buffer(&[1.0f32]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32]);
        let job = cpu
            .submit_call(&scale_call(1, &a, Independent(&settings), &out))
            .unwrap();
        assert_eq!(job.is_complete(), Ok(true));
        assert_eq!(
            out.read(),
            vec![2.0],
            "the work is done when submit returns"
        );
    }

    #[test]
    fn errors_are_std_errors_with_the_documented_wording() {
        fn is_std_error<E: std::error::Error + Send + Sync + 'static>(_: &E) {}
        let error = Error::ElementType { parameter: "a" };
        is_std_error(&error);
        let boxed: Box<dyn std::error::Error> = Box::new(error);
        assert_eq!(
            boxed.to_string(),
            "parameter `a` is bound to a buffer of another element type"
        );
        let messages = [
            (
                Error::Access {
                    parameter: "a",
                    expected: Access::ReadOnly,
                    actual: Access::ReadWrite,
                },
                "parameter `a` expects ReadOnly access, got ReadWrite",
            ),
            (
                Error::WrongDevice { parameter: "out" },
                "parameter `out` is bound to a buffer of another CPU device instance",
            ),
            (
                Error::Aliased {
                    kernel: "cpu_tests::scale".to_string(),
                    first: "a",
                    second: "out",
                },
                "parameters `a` and `out` of kernel `cpu_tests::scale` are the same buffer and at least one of them is written; a buffer may be bound twice only read-only (bind it once, or use separate buffers)",
            ),
            (
                Error::Borrowed { parameter: "out" },
                "the buffer bound to `out` is borrowed outside this call",
            ),
            (
                Error::BufferLength {
                    parameter: "out",
                    expected: 4,
                    actual: 5,
                },
                "parameter `out` expects 4 elements, got 5",
            ),
            (
                Error::IndependentLengthEmpty {
                    parameter: "settings",
                },
                "parameter `settings` was bound with an independent length but the buffer is empty",
            ),
            (
                Error::UnsupportedWorkgroupSize { size: [8, 0, 1] },
                "workgroup size [8, 0, 1] is not supported by the CPU twin (every axis must be non-zero)",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(error.to_string(), expected);
        }
    }

    // ---- ulps ----

    #[test]
    fn ulps_counts_representable_steps() {
        let up = |x: f32| f32::from_bits(x.to_bits() + 1);
        let down = |x: f32| f32::from_bits(x.to_bits() - 1);
        assert_eq!(ulps(1.0, 1.0), 0);
        assert_eq!(ulps(1.0, up(1.0)), 1);
        assert_eq!(ulps(up(1.0), 1.0), 1, "symmetric");
        assert_eq!(ulps(1.0, down(1.0)), 1);
        assert_eq!(ulps(1.0, 2.0), 0x0080_0000);
        // Negative values count the same way.
        assert_eq!(ulps(-1.0, -up(1.0)), 1);
        assert_eq!(ulps(-1.0, -2.0), 0x0080_0000);
        // Opposite signs are counted through zero, and the two zeros are one point.
        let tiny = f32::from_bits(1);
        assert_eq!(ulps(-tiny, tiny), 2);
        assert_eq!(ulps(-tiny, 0.0), 1);
        assert_eq!(ulps(0.0, tiny), 1);
        assert_eq!(ulps(-0.0, 0.0), 0);
        assert_eq!(ulps(-1.0, 1.0), 2 * 0x3F80_0000);
        // Extremes, infinities and NaN.
        assert_eq!(ulps(f32::MAX, f32::INFINITY), 1);
        assert_eq!(ulps(f32::NEG_INFINITY, f32::INFINITY), 2 * 0x7F80_0000);
        assert_eq!(ulps(f32::NAN, 1.0), u64::MAX);
        assert_eq!(ulps(1.0, f32::NAN), u64::MAX);
        assert_eq!(ulps(f32::NAN, f32::NAN), u64::MAX);
    }

    // ---- overflow_checks ----

    // A fact about this workspace's profiles, not a general rule: the root `Cargo.toml` sets no `overflow-checks`,
    // so `cargo test` (dev) is checked and `cargo test --release` wraps. CI runs the release half through
    // `cargo test -p gust --release --lib cpu`.
    #[test]
    fn overflow_checks_agrees_with_the_workspace_profile() {
        let expected = if cfg!(debug_assertions) {
            OverflowChecks::Checked
        } else {
            OverflowChecks::Wrapping
        };
        assert_eq!(overflow_checks(), expected);
        // The detection itself, run directly, gives the same answer as the cached one.
        assert_eq!(probe_overflow_checks(), expected);
    }

    // However many threads and calls, the probe runs once per process.
    #[test]
    fn overflow_checks_is_computed_once_and_cached() {
        let first = overflow_checks();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| assert_eq!(overflow_checks(), first));
            }
        });
        assert_eq!(overflow_checks(), first);
        // The counter only moves inside the cache, so it is exactly 1 however the tests interleave.
        assert_eq!(OVERFLOW_PROBES.load(Ordering::Relaxed), 1);
    }

    // ---- overflow_checks! (the probe that expands in the calling crate) ----

    // The test build of this crate is the calling crate here, so the macro, the function and the profile agree (a
    // workspace-profile fact, as above). That the macro answers for the *caller* when the caller's profile differs
    // from `gust`'s cannot be shown inside this crate: that proof is m3.f1's cross-crate fixture (`tests/dual`).
    #[test]
    fn the_overflow_macro_agrees_with_the_function_and_the_workspace_profile() {
        let expected = if cfg!(debug_assertions) {
            OverflowChecks::Checked
        } else {
            OverflowChecks::Wrapping
        };
        assert_eq!(overflow_checks!(), expected);
        assert_eq!(overflow_checks!(), overflow_checks());
    }

    static SITE_A_PROBES: AtomicU32 = AtomicU32::new(0);
    static SITE_B_PROBES: AtomicU32 = AtomicU32::new(0);

    fn count_site_a() {
        SITE_A_PROBES.fetch_add(1, Ordering::Relaxed);
    }

    fn count_site_b() {
        SITE_B_PROBES.fetch_add(1, Ordering::Relaxed);
    }

    fn site_a() -> OverflowChecks {
        crate::__gust_overflow_checks!(@probed count_site_a)
    }

    fn site_b() -> OverflowChecks {
        crate::__gust_overflow_checks!(@probed count_site_b)
    }

    // Each call site caches its own result: however many calls and threads, one probe per site, and the sites do not
    // share a cache.
    #[test]
    fn each_overflow_macro_call_site_is_cached_once() {
        let first = site_a();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| assert_eq!(site_a(), first));
            }
        });
        assert_eq!(site_a(), first);
        assert_eq!(SITE_A_PROBES.load(Ordering::Relaxed), 1);
        assert_eq!(
            SITE_B_PROBES.load(Ordering::Relaxed),
            0,
            "site B has not run"
        );
        assert_eq!(site_b(), first);
        assert_eq!(site_b(), first);
        assert_eq!(SITE_B_PROBES.load(Ordering::Relaxed), 1);
        assert_eq!(SITE_A_PROBES.load(Ordering::Relaxed), 1);
    }

    // ---- the launch limit (ids are `u32`) ----

    struct Wide192;
    impl Kernel for Wide192 {
        const DESCRIPTOR: KernelDescriptor = descriptor("wide192", [192, 1, 1], &[]);
    }

    struct Wide64;
    impl Kernel for Wide64 {
        const DESCRIPTOR: KernelDescriptor = descriptor("wide64", [64, 1, 1], &[]);
    }

    // The GPU path rejects a launch over its workgroup limit with `DeviceLimit` (`workgroup_count` in gust-wgpu,
    // after validation, not for an empty launch). The twin's limit is the `u32` id range: a width that is not a power
    // of two rounds `u32::MAX` elements up past 2^32 (192 * 22369622 = 4294967424), and the ids would wrap.
    #[test]
    fn a_launch_beyond_the_id_range_is_rejected() {
        let cpu = CpuDevice::new();
        let over = |count: u32| validate(&cpu, &Call::<CpuDevice, Wide192, 0>::new(count, []));
        let too_large = || Error::DeviceLimit {
            kernel: "cpu_tests::wide192".to_string(),
            limit: "global_invocation_id",
            requested: 4_294_967_424,
            maximum: 1 << 32,
        };
        assert_eq!(over(u32::MAX), Err(too_large()));
        // The boundary: 22369621 whole workgroups of 192 (4294967232) fit, one more element needs another.
        assert_eq!(over(4_294_967_232), Ok(()));
        assert_eq!(over(4_294_967_233), Err(too_large()));
        // An empty launch is not checked, and 64 divides 2^32: `u32::MAX` elements round up to exactly 2^32.
        assert_eq!(over(0), Ok(()));
        assert_eq!(
            validate(&cpu, &Call::<CpuDevice, Wide64, 0>::new(u32::MAX, [])),
            Ok(())
        );
        assert_eq!(
            too_large().to_string(),
            "kernel `cpu_tests::wide192` needs 4294967424 for `global_invocation_id`, the limit is 4294967296; split the work into several dispatches"
        );
    }

    /// `fill(out: RW)` of the `pixel` shape (slice mx.p1b): the twin's refusals are the ones `gust_wgpu` makes.
    struct PixelFill;
    impl Kernel for PixelFill {
        const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
            shape: crate::Shape::Pixel,
            compatible_shapes: &[crate::Shape::Pixel],
            workgroup_size: [1, 1, 1],
            ..descriptor(
                "pixel_fill",
                [1, 1, 1],
                &[storage("out", Access::ReadWrite, 0, F32)],
            )
        };
    }

    #[test]
    fn a_pixel_call_is_one_dimensional_one_shared_output_and_at_most_the_cap() {
        let cpu = CpuDevice::new();
        let out = cpu.buffer(&[0.0f32; 4]);
        let call = |extent: Extent| {
            Call::<CpuDevice, PixelFill, 1>::new_extent(
                extent,
                [Arg::<F32Buffer>::from(&out).bind(Access::ReadWrite)],
            )
        };
        assert_eq!(validate(&cpu, &call(Extent::linear(4))), Ok(()));
        // A y or z other than 1 is refused before any binding is looked at, however small the buffer.
        for extent in [[4, 2, 1], [4, 1, 2], [0, 2, 1]] {
            let error = validate(&cpu, &call(Extent::new(extent[0], extent[1], extent[2])));
            assert_eq!(
                error,
                Err(Error::PixelExtent {
                    kernel: "cpu_tests::pixel_fill".to_string(),
                    extent,
                })
            );
        }
        assert_eq!(
            Error::PixelExtent {
                kernel: "cpu_tests::pixel_fill".to_string(),
                extent: [4, 2, 1],
            }
            .to_string(),
            "kernel `cpu_tests::pixel_fill` has the shape Pixel, which runs one-dimensional calls only; the extent [4, 2, 1] has a y or z other than 1 (use `call`, not `call_extent`)"
        );
        // An independent read-write output is refused (the same refusal as `gust_wgpu`'s `ShapeIndependentOutput`).
        let independent = Call::<CpuDevice, PixelFill, 1>::new(
            4,
            [Arg::<F32Buffer>::from(Independent(&out)).bind(Access::ReadWrite)],
        );
        assert_eq!(
            validate(&cpu, &independent),
            Err(Error::PixelIndependentOutput {
                kernel: "cpu_tests::pixel_fill".to_string(),
                parameter: "out",
            })
        );
        // The same kernel as a grid kernel takes both, which is what makes the refusal the shape's.
        struct GridFill;
        impl Kernel for GridFill {
            const DESCRIPTOR: KernelDescriptor = descriptor(
                "grid_fill",
                [1, 1, 1],
                &[storage("out", Access::ReadWrite, 0, F32)],
            );
        }
        assert_eq!(
            validate(
                &cpu,
                &Call::<CpuDevice, GridFill, 1>::new(
                    4,
                    [Arg::<F32Buffer>::from(Independent(&out)).bind(Access::ReadWrite)]
                )
            ),
            Ok(())
        );
        assert_eq!(
            validate(
                &cpu,
                &Call::<CpuDevice, GridFill, 1>::new_extent(
                    Extent::new(2, 2, 1),
                    [Arg::<F32Buffer>::from(&out).bind(Access::ReadWrite)]
                )
            ),
            Ok(())
        );
    }

    #[test]
    fn a_pixel_call_beyond_the_cap_is_the_named_device_limit_and_an_empty_one_is_not_checked() {
        let cpu = CpuDevice::new();
        let cap = crate::PIXEL_MAX_ELEMENTS;
        let over = |count: u32| validate(&cpu, &pixel_empty(count));
        fn pixel_empty(count: u32) -> Call<'static, CpuDevice, PixelNoBindings, 0> {
            Call::new(count, [])
        }
        struct PixelNoBindings;
        impl Kernel for PixelNoBindings {
            const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
                shape: crate::Shape::Pixel,
                compatible_shapes: &[crate::Shape::Pixel],
                ..descriptor("pixel_none", [1, 1, 1], &[])
            };
        }
        assert_eq!(over(cap), Ok(()));
        assert_eq!(over(0), Ok(()));
        assert_eq!(
            over(cap + 1),
            Err(Error::DeviceLimit {
                kernel: "cpu_tests::pixel_none".to_string(),
                limit: "pixel_max_elements",
                requested: u64::from(cap) + 1,
                maximum: u64::from(cap),
            })
        );
        assert_eq!(
            over(u32::MAX),
            Err(Error::DeviceLimit {
                kernel: "cpu_tests::pixel_none".to_string(),
                limit: "pixel_max_elements",
                requested: u64::from(u32::MAX),
                maximum: u64::from(cap),
            })
        );
    }

    // The cap in two and three dimensions (slice m7.d): the product of the three rounded axes, at most 2^32. It is not
    // the raw product (65537 * 65535 is below 2^32 and still over) and it is not "below" (exactly 2^32 is accepted).
    // The GPU backend applies the same rule (`crates/gust-wgpu/tests/limits.rs`, which compares both).
    #[test]
    fn the_cap_is_the_rounded_up_product_of_all_axes_at_most_2_pow_32() {
        struct Tile16;
        impl Kernel for Tile16 {
            const DESCRIPTOR: KernelDescriptor = descriptor("tile16", [16, 16, 1], &[]);
        }
        struct Block4;
        impl Kernel for Block4 {
            const DESCRIPTOR: KernelDescriptor = descriptor("block4", [4, 4, 4], &[]);
        }
        let cpu = CpuDevice::new();
        let tile = |x, y, z| {
            validate(
                &cpu,
                &Call::<CpuDevice, Tile16, 0>::new_extent(Extent::new(x, y, z), []),
            )
        };
        let block = |x, y, z| {
            validate(
                &cpu,
                &Call::<CpuDevice, Block4, 0>::new_extent(Extent::new(x, y, z), []),
            )
        };
        let over = |kernel: &str, requested: u64| Error::DeviceLimit {
            kernel: format!("cpu_tests::{kernel}"),
            limit: "global_invocation_id",
            requested,
            maximum: 1 << 32,
        };
        // Exactly 2^32 after rounding: accepted, including a raw product that is smaller (65536 * 65535 rounds up).
        assert_eq!(tile(65_536, 65_536, 1), Ok(()));
        assert_eq!(tile(65_536, 65_535, 1), Ok(()));
        assert_eq!(block(2048, 2048, 1024), Ok(()));
        // The raw product 65537 * 65535 = 2^32 - 1 is below the cap, the rounded one (65552 * 65536) is above it.
        assert_eq!(
            tile(65_537, 65_535, 1),
            Err(over("tile16", 65_552 * 65_536))
        );
        assert_eq!(
            tile(65_537, 65_536, 1),
            Err(over("tile16", 65_552 * 65_536))
        );
        assert_eq!(
            block(2048, 2048, 1025),
            Err(over("block4", 2048 * 2048 * 1028))
        );
        // A one-dimensional extent follows the same rule through a 2D kernel: y rounds 1 up to 16, so 2^28 elements
        // along x are 2^32 invocations and one more element needs another workgroup column.
        assert_eq!(tile(1 << 28, 1, 1), Ok(()));
        assert_eq!(
            tile((1 << 28) + 1, 1, 1),
            Err(over("tile16", ((1 << 28) + 16) * 16))
        );
        // A launch beyond a `u64` saturates in the error, and an empty one is never checked.
        assert_eq!(
            tile(u32::MAX, u32::MAX, u32::MAX),
            Err(over("tile16", u64::MAX))
        );
        assert_eq!(tile(0, u32::MAX, u32::MAX), Ok(()));
        assert_eq!(tile(u32::MAX, 0, u32::MAX), Ok(()));
        assert_eq!(tile(u32::MAX, u32::MAX, 0), Ok(()));
    }

    // A shared buffer holds `x * y * z` elements (an `Independent` one only has to be non-empty).
    #[test]
    fn a_shared_buffer_holds_the_product_of_the_extent() {
        struct Fill;
        impl Kernel for Fill {
            const DESCRIPTOR: KernelDescriptor = descriptor(
                "fill",
                [4, 4, 1],
                &[storage("out", Access::ReadWrite, 0, F32)],
            );
        }
        let cpu = CpuDevice::new();
        let out = cpu.buffer(&[0.0f32; 12]);
        let run = |extent: Extent, length: LengthRule| {
            validate(
                &cpu,
                &Call::<CpuDevice, Fill, 1>::new_extent(
                    extent,
                    [out.bind(Access::ReadWrite, length)],
                ),
            )
        };
        let shared = LengthRule::Shared;
        assert_eq!(run(Extent::new(3, 4, 1), shared), Ok(()));
        assert_eq!(run(Extent::new(2, 2, 3), shared), Ok(()));
        assert_eq!(
            run(Extent::new(4, 4, 1), shared),
            Err(Error::BufferLength {
                parameter: "out",
                expected: 16,
                actual: 12
            })
        );
        // A zero axis makes the product 0, so a non-empty shared buffer is wrong, as for a count of 0.
        assert_eq!(
            run(Extent::new(0, 4, 3), shared),
            Err(Error::BufferLength {
                parameter: "out",
                expected: 0,
                actual: 12
            })
        );
        assert_eq!(run(Extent::new(4, 4, 4), LengthRule::Independent), Ok(()));
        // A product too large for any buffer saturates, never wraps to a size that fits.
        assert_eq!(
            run(Extent::new(u32::MAX, u32::MAX, u32::MAX), shared),
            Err(Error::BufferLength {
                parameter: "out",
                expected: usize::MAX,
                actual: 12
            })
        );
    }

    // `invocations` cannot wrap an id even when `validate` was skipped: a debug build asserts, a release build is
    // capped at 2^32 ids (the first 2^32, never ids that restart from 0).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "exceed the u32 id range")]
    fn invocations_assert_beyond_the_id_range_in_debug() {
        let _ = invocations(&Wide192::DESCRIPTOR, Extent::linear(u32::MAX));
    }

    #[test]
    #[cfg(all(not(debug_assertions), target_pointer_width = "64"))]
    fn invocations_are_capped_at_the_id_range_in_release() {
        let total = 1usize << 32;
        assert_eq!(
            invocations(&Wide192::DESCRIPTOR, Extent::linear(u32::MAX)).size_hint(),
            (total, Some(total))
        );
    }

    // ---- check order and borrows ----

    // Wrong in two ways: the cause is the one `validate_persistent_dispatch` of gust-wgpu reports first (element
    // layout, device, access, length, then aliasing; the launch limit last).
    #[test]
    fn a_call_wrong_in_two_ways_reports_the_cause_gust_wgpu_reports_first() {
        let cpu = CpuDevice::new();
        let other = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32; 2]);
        let settings = cpu.buffer(&[2.0f32]);
        let out = cpu.buffer(&[0.0f32; 2]);
        let (ro, rw) = (Access::ReadOnly, Access::ReadWrite);
        let (shared, independent) = (LengthRule::Shared, LengthRule::Independent);
        fn run_scale<'r>(
            cpu: &CpuDevice,
            bindings: [CpuBinding<'r>; 3],
            count: u32,
        ) -> Result<(), Error> {
            cpu.dispatch_call(&Call::<CpuDevice, Scale, 3>::new(count, bindings))
        }
        let settings = settings.bind(ro, independent);
        macro_rules! run {
            ($first:expr, $last:expr, $count:expr) => {
                run_scale(&cpu, [$first, settings, $last], $count)
            };
        }

        // Element layout before device: a `u32` buffer of another device in `a`.
        let foreign_integers = other.buffer(&[7u32; 2]);
        assert_eq!(
            run!(foreign_integers.bind(ro, shared), out.bind(rw, shared), 2),
            Err(Error::ElementType { parameter: "a" })
        );
        // Device before access: a foreign `f32` buffer bound read-write in `a`.
        let foreign = other.buffer(&[7.0f32; 2]);
        assert_eq!(
            run!(foreign.bind(rw, shared), out.bind(rw, shared), 2),
            Err(Error::WrongDevice { parameter: "a" })
        );
        // Access before length: `a` read-write and of the wrong length.
        assert_eq!(
            run!(a.bind(rw, shared), out.bind(rw, shared), 3),
            Err(Error::Access {
                parameter: "a",
                expected: ro,
                actual: rw
            })
        );
        // Length before aliasing: `out` is `a` and the count is wrong.
        assert_eq!(
            run!(a.bind(ro, shared), a.bind(rw, shared), 3),
            Err(Error::BufferLength {
                parameter: "a",
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(a.read(), vec![1.0, 1.0], "no rejected call wrote anything");
    }

    #[gust::gpu]
    mod order_kernel {
        // Parameter 0 is read-write, so an alias on parameters 0 and 1 is an alias with a writer.
        #[kernel]
        pub fn probe(
            id: SV_DispatchThreadID,
            mut a: RWStructuredBuffer<float>,
            b: StructuredBuffer<float>,
            c: StructuredBuffer<float>,
        ) {
            let i = id.x;
            if i < a.len() {
                a[i] = b[i] + c[i];
            }
        }
    }

    // The per-binding checks (including the length of the last parameter) run before the alias pass, as in gust-wgpu's
    // `validate_persistent_dispatch`: the alias is on parameters 0 and 1 and the length error on parameter 2. An alias
    // check inline per binding would find the alias at parameter 1 first and report `Aliased`.
    #[test]
    fn the_alias_pass_runs_after_every_binding_check() {
        let cpu = CpuDevice::new();
        let shared_ab = cpu.buffer(&[1.0f32; 2]);
        let short_c = cpu.buffer(&[1.0f32; 3]);
        let call = order_kernel::probe::call(2, &shared_ab, &shared_ab, &short_c);
        assert_eq!(
            validate(&cpu, &call),
            Err(Error::BufferLength {
                parameter: "c",
                expected: 2,
                actual: 3
            })
        );
        // With the length fixed the alias is what remains, named with the kernel in declaration order.
        let c = cpu.buffer(&[1.0f32; 2]);
        let call = order_kernel::probe::call(2, &shared_ab, &shared_ab, &c);
        assert_eq!(
            validate(&cpu, &call),
            Err(Error::Aliased {
                kernel: "order_kernel::probe".to_string(),
                first: "a",
                second: "b"
            })
        );
        assert_eq!(shared_ab.read(), vec![1.0, 1.0]);
    }

    // The other half of the borrow rule: a shared `Ref` held outside blocks `write`, which reports `Borrowed` and
    // leaves the buffer as it was.
    #[test]
    fn write_while_a_shared_borrow_is_held_is_borrowed_and_changes_nothing() {
        let cpu = CpuDevice::new();
        let buffer = cpu.buffer(&[1.0f32, 2.0]);
        {
            let _held = buffer.cells().borrow();
            assert_eq!(
                buffer.write(&[5.0, 6.0]),
                Err(Error::Borrowed {
                    parameter: "values"
                })
            );
        }
        assert_eq!(buffer.read(), vec![1.0, 2.0]);
        buffer.write(&[5.0, 6.0]).unwrap();
        assert_eq!(buffer.read(), vec![5.0, 6.0]);
    }

    // ---- uniforms (slice m6.u1) ----

    // The shadow body reads the value by copy; an update between two dispatches is seen by the second one only.
    #[test]
    fn a_uniform_is_read_by_copy_and_an_update_is_seen_by_the_next_dispatch() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32, 2.0, 3.0]);
        let out = cpu.buffer(&[0.0f32; 3]);
        let factor = cpu.uniform(10.0f32).unwrap();
        let call = shade_call(3, &a, &factor, &out);
        cpu.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), vec![10.0, 20.0, 30.0]);
        factor.write(0.5).unwrap();
        assert_eq!(factor.read(), 0.5);
        cpu.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), vec![0.5, 1.0, 1.5]);
    }

    // A uniform has no element count: whatever length rule the argument names, one value serves every element.
    #[test]
    fn a_uniform_is_independent_of_the_element_count_whatever_the_argument_says() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32; 5]);
        let out = cpu.buffer(&[0.0f32; 5]);
        let factor = cpu.uniform(2.0f32).unwrap();
        let shared = Arg::from(&factor);
        assert_eq!(
            shared.length,
            LengthRule::Shared,
            "the argument says Shared"
        );
        assert_eq!(
            shared.bind(Access::ReadOnly).length,
            LengthRule::Independent
        );
        cpu.dispatch_call(&shade_call(5, &a, &factor, &out))
            .unwrap();
        assert_eq!(out.read(), vec![2.0; 5]);
        cpu.dispatch_call(&shade_call(5, &a, Independent(&factor), &out))
            .unwrap();
        // A launch of no elements is fine too, and a uniform is never "empty".
        let (none_in, none_out) = (cpu.buffer::<f32>(&[]), cpu.buffer::<f32>(&[]));
        cpu.dispatch_call(&shade_call(0, &none_in, &factor, &none_out))
            .unwrap();
    }

    // A vector uniform: the element layout of the binding must be the descriptor's (Vec4, not f32).
    #[test]
    fn a_vector_uniform_is_checked_against_the_descriptor_layout() {
        let cpu = CpuDevice::new();
        let out = cpu.buffer(&[0.0f32; 2]);
        let color = cpu.uniform(Vec4::new(1.0, 2.0, 3.0, 4.0)).unwrap();
        let call = Call::<CpuDevice, Tint, 2>::new(
            2,
            [
                color.bind(Access::ReadOnly, LengthRule::Shared),
                out.bind(Access::ReadWrite, LengthRule::Shared),
            ],
        );
        cpu.dispatch_call(&call).unwrap();
        assert_eq!(out.read(), vec![5.0, 5.0]);
        let wrong = cpu.uniform(7.0f32).unwrap();
        let call = Call::<CpuDevice, Tint, 2>::new(
            2,
            [
                wrong.bind(Access::ReadOnly, LengthRule::Shared),
                out.bind(Access::ReadWrite, LengthRule::Shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(Error::ElementType { parameter: "color" })
        );
    }

    // The twin refuses the uniform types the GPU refuses (`create_uniform`), with the same wording. `unsafe_code` is
    // forbidden in this module, so the padded case is a layout: no safe type has padding and a Std140V1 layout.
    #[test]
    fn a_uniform_type_must_be_a_padding_free_std140_v1_type() {
        use crate::math::{Mat2, Vec2};
        let cpu = CpuDevice::new();
        assert!(cpu.uniform(1.0f32).is_ok());
        assert!(cpu.uniform(Vec4::splat(1.0)).is_ok());
        // A 2x2 matrix is a storage type with no constant-buffer layout.
        let matrix = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        match cpu.uniform(matrix) {
            Err(Error::InvalidUniformType { type_name, reason }) => {
                assert_eq!(type_name, <Mat2 as GpuPod>::LAYOUT.name);
                assert!(reason.contains("not Std140V1"), "{reason}");
            }
            other => panic!("expected InvalidUniformType, got {:?}", other.map(|_| ())),
        }
        // A padded layout with honest Std140V1 numbers: a `Vec4` then one `f32`, bytes 20 to 31 are padding.
        const PADDED: TypeLayout = TypeLayout::structure(
            "Padded",
            32,
            16,
            &[
                crate::StructFieldLayout {
                    name: "v",
                    offset: 0,
                    ty: <Vec4 as GpuPod>::LAYOUT,
                },
                crate::StructFieldLayout {
                    name: "a",
                    offset: 16,
                    ty: <f32 as GpuPod>::LAYOUT,
                },
            ],
        );
        assert!(PADDED.is_std140_v1());
        let reason = uniform_type_refusal(PADDED, 32, 16).expect("padding is refused");
        assert!(reason.contains("padding bytes"), "{reason}");
        assert!(reason.contains("explicit padding fields"), "{reason}");
        let size = uniform_type_refusal(PADDED, 20, 4).expect("a size that differs is refused");
        assert!(size.contains("size or alignment"), "{size}");
        assert_eq!(uniform_type_refusal(<Vec4 as GpuPod>::LAYOUT, 16, 16), None);
        let error = Error::InvalidUniformType {
            type_name: "Padded",
            reason,
        };
        assert!(
            error
                .to_string()
                .starts_with("`Padded` cannot be a uniform: it has padding bytes"),
            "{error}"
        );
    }

    // The parameter's rule must be Std140V1 too: a uniform labelled with the storage rule is refused by `validate`,
    // as gust-wgpu's `validate_kernel` refuses it, before the element comparison.
    #[test]
    fn a_uniform_parameter_with_the_storage_rule_is_refused_by_the_twin() {
        let cpu = CpuDevice::new();
        let out = cpu.buffer(&[0.0f32; 2]);
        let color = cpu.uniform(Vec4::splat(1.0)).unwrap();
        let call = Call::<CpuDevice, TintStorageRule, 2>::new(
            2,
            [
                color.bind(Access::ReadOnly, LengthRule::Shared),
                out.bind(Access::ReadWrite, LengthRule::Shared),
            ],
        );
        assert_eq!(
            validate(&cpu, &call),
            Err(Error::InvalidResourceLayout { parameter: "color" })
        );
        // The control: the same call under the right rule validates.
        let call = Call::<CpuDevice, Tint, 2>::new(
            2,
            [
                color.bind(Access::ReadOnly, LengthRule::Shared),
                out.bind(Access::ReadWrite, LengthRule::Shared),
            ],
        );
        assert_eq!(validate(&cpu, &call), Ok(()));
    }

    // A hand-built `Call` can put a buffer where the kernel has a uniform and the other way round: the kind is the
    // first thing `validate` compares, and it names the parameter and both kinds.
    #[test]
    fn a_hand_built_call_with_the_wrong_kind_is_rejected_first() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32]);
        let out = cpu.buffer(&[0.0f32]);
        let one = cpu.buffer(&[3.0f32]);
        let factor = cpu.uniform(3.0f32).unwrap();
        let (ro, rw, shared) = (Access::ReadOnly, Access::ReadWrite, LengthRule::Shared);
        // A buffer in the uniform parameter (same element, same length, same access).
        let call = Call::<CpuDevice, Shade, 3>::new(
            1,
            [
                a.bind(ro, shared),
                one.bind(ro, shared),
                out.bind(rw, shared),
            ],
        );
        assert_eq!(
            cpu.dispatch_call(&call),
            Err(Error::BindingKind {
                parameter: "factor",
                expected: BindingKind::Uniform,
                actual: BindingKind::Storage,
            })
        );
        // A uniform in the first storage parameter.
        let call = Call::<CpuDevice, Shade, 3>::new(
            1,
            [
                factor.bind(ro, shared),
                factor.bind(ro, shared),
                out.bind(rw, shared),
            ],
        );
        let rejected = cpu.dispatch_call(&call).unwrap_err();
        assert_eq!(
            rejected,
            Error::BindingKind {
                parameter: "a",
                expected: BindingKind::Storage,
                actual: BindingKind::Uniform,
            }
        );
        assert_eq!(
            rejected.to_string(),
            "parameter `a` expects a Storage binding, got a Uniform one"
        );
        assert_eq!(out.read(), vec![0.0], "rejected calls wrote nothing");
    }

    // The macro-support readers repeat the kind and access checks for a call site that acquires a parameter wrongly.
    #[test]
    fn read_uniform_checks_kind_type_access_and_borrow() {
        let cpu = CpuDevice::new();
        let factor = cpu.uniform(3.0f32).unwrap();
        let buffer = cpu.buffer(&[3.0f32]);
        let uniform_binding = |access| factor.bind(access, LengthRule::Independent);
        assert_eq!(
            read_uniform::<f32>(uniform_binding(Access::ReadOnly), "f"),
            Ok(3.0)
        );
        assert_eq!(
            read_uniform::<u32>(uniform_binding(Access::ReadOnly), "f"),
            Err(Error::ElementType { parameter: "f" })
        );
        assert_eq!(
            read_uniform::<f32>(uniform_binding(Access::ReadWrite), "f"),
            Err(Error::Access {
                parameter: "f",
                expected: Access::ReadOnly,
                actual: Access::ReadWrite,
            })
        );
        assert_eq!(
            read_uniform::<f32>(buffer.bind(Access::ReadOnly, LengthRule::Shared), "f"),
            Err(Error::BindingKind {
                parameter: "f",
                expected: BindingKind::Uniform,
                actual: BindingKind::Storage,
            })
        );
        // The storage readers refuse a uniform with the same error instead of an element-type one.
        let wrong_kind = Error::BindingKind {
            parameter: "p",
            expected: BindingKind::Storage,
            actual: BindingKind::Uniform,
        };
        assert_eq!(
            read::<f32>(uniform_binding(Access::ReadOnly), "p").err(),
            Some(wrong_kind.clone())
        );
        assert_eq!(
            write::<f32>(uniform_binding(Access::ReadWrite), "p").err(),
            Some(wrong_kind)
        );
        {
            let _held = factor.cell().borrow_mut();
            assert_eq!(
                read_uniform::<f32>(uniform_binding(Access::ReadOnly), "f"),
                Err(Error::Borrowed { parameter: "f" })
            );
            assert_eq!(
                factor.write(1.0),
                Err(Error::Borrowed { parameter: "value" })
            );
        }
        assert_eq!(factor.read(), 3.0, "the rejected write changed nothing");
    }

    // An outside borrow of the uniform is the same `Borrowed` error as for a buffer, and nothing was written.
    #[test]
    fn a_borrowed_uniform_rejects_the_call_before_any_write() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32]);
        let out = cpu.buffer(&[9.0f32]);
        let factor = cpu.uniform(2.0f32).unwrap();
        {
            let _held = factor.cell().borrow_mut();
            assert_eq!(
                cpu.dispatch_call(&shade_call(1, &a, &factor, &out)),
                Err(Error::Borrowed {
                    parameter: "factor"
                })
            );
        }
        assert_eq!(out.read(), vec![9.0]);
    }

    // A uniform of another device instance is the same `WrongDevice` error as for a buffer.
    #[test]
    fn a_uniform_of_another_device_instance_is_rejected() {
        let cpu = CpuDevice::new();
        let a = cpu.buffer(&[1.0f32]);
        let out = cpu.buffer(&[9.0f32]);
        let foreign = CpuDevice::new().uniform(2.0f32).unwrap();
        assert_eq!(
            cpu.dispatch_call(&shade_call(1, &a, &foreign, &out)),
            Err(Error::WrongDevice {
                parameter: "factor"
            })
        );
        assert_eq!(out.read(), vec![9.0]);
    }
}
