//! A device handle that follows the route: [`RoutedDevice`] (m9.h1, D65, D137).
//!
//! In 0.2.0 a kernel call names its device: `CpuDevice` for a CPU-routed kernel, the GPU runtime's device for a
//! GPU-routed one, so flipping a kernel's route in `gust.toml` changes every call site of that kernel. A
//! [`RoutedDevice`] is one type whose [`Run`] impl runs each kernel on the target the build resolved for it, so the call
//! site (`kernel::dispatch(&device, n, &a, &b)`) is the same source under either route. The explicit-device calls
//! (`CpuDevice`, `HeadlessDevice`) are unchanged and stay available beside it.
//!
//! # Use
//!
//! ```text
//! let device = RoutedDevice::with_gpu(HeadlessDevice::new()?); // or RoutedDevice::<NoGpu>::cpu_only()
//! let input = device.buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
//! let out = device.buffer(&[0.0f32; 4])?;
//! physics::scale::dispatch(&device, 4, &input, &out)?; // GPU or CPU, as the build routed `physics`
//! assert_eq!(out.read(&device)?, [2.0, 4.0, 6.0, 8.0]);
//! ```
//!
//! # How it stays static and layered
//!
//! - The kernel names its target in the type system through [`Routed::Side`] ([`OnCpu`] or [`OnGpu`]). The `#[gpu]`
//!   macro emits one `impl Routed` line per kernel next to the `GpuKernel` marker; a hand-written kernel writes it
//!   itself. One associated type, not two marker bounds, because two blanket impls bounded on two different markers
//!   overlap in rustc's coherence check (E0119). Nothing is looked up at run time and nothing is `dyn` in the dispatch.
//! - `gust` does not depend on the GPU runtime: [`GpuBackend`] is the small surface the handle needs of a GPU device,
//!   and `gust-wgpu` implements it for `HeadlessDevice`. [`NoGpu`] is the backend of a build without one.
//! - A GPU-routed kernel called on a handle that has no GPU is [`Error::NoGpu`], never a panic, and nothing is written.
//!   A buffer made on another handle is refused by the checks that already exist on each side: on the CPU side
//!   `gust::cpu::Error::WrongDevice` ([`Error::Cpu`]), on the GPU side `gust_wgpu::Error::ForeignPersistentBuffer`
//!   ([`Error::Gpu`]).
//!
//! # Limits (read these before relying on it)
//!
//! - **Storage buffers only.** A kernel with a uniform, a texture or a sampler parameter has no `Run` impl on the
//!   handle, so its generated call does not compile there; call such a kernel on an explicit device.
//! - **Calls are synchronous.** Every call finishes inside `submit_call` (the GPU side waits), so the returned job is
//!   already complete and there is no GPU overlap through the handle.
//! - **Whole-buffer copies when a call changes side.** A [`Buffer`] holds a host copy and, when the handle has a GPU,
//!   a device copy. A call that runs on a side that does not hold the current contents first copies the whole buffer
//!   there. Nothing is partial and nothing is overlapped; this is the correctness-first rule, not a fast one (D35's
//!   transfer insertion is the real answer, and the cost of a transfer through the handle is not measured).
//! - **The residency rule.** A fresh buffer is valid on both sides. A call that binds a buffer for writing makes the
//!   side that ran it the only valid one; [`Buffer::write`] makes the host the only valid one; [`Buffer::read`] copies
//!   back from the device first when the device holds the contents. A buffer made on a handle without a GPU has no
//!   device copy, so a GPU-routed kernel on it is [`Error::NoGpu`].
//! - `Error::Gpu` boxes the backend's error, so the backend's own variants are reached through `source()`.

use std::cell::Cell;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::time::Duration;

use crate::cpu::{CpuBinding, CpuBuffer, CpuDevice};
use crate::target::GpuKernel;
use crate::{
    Access, Call, Device as DeviceTrait, GpuPod, Job as JobTrait, JobStatus, Kernel, LengthRule,
    Resource, Run, TypedDevice, kind,
};

/// Which target a kernel is built for, as a type. Implemented by [`OnCpu`] and [`OnGpu`] only.
pub trait Side: sealed::Sealed {}

/// The kernel runs on the CPU in this build.
#[derive(Clone, Copy, Debug)]
pub struct OnCpu;
/// The kernel runs on a GPU in this build.
#[derive(Clone, Copy, Debug)]
pub struct OnGpu;

impl Side for OnCpu {}
impl Side for OnGpu {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::OnCpu {}
    impl Sealed for super::OnGpu {}
}

/// A kernel whose target is a compile-time fact: `type Side = OnGpu` for a kernel this build runs on a GPU (it also
/// implements `GpuKernel`), `type Side = OnCpu` for a CPU-routed one. The `#[gpu]` macro emits this beside the
/// `GpuKernel` marker, so every kernel it builds has it; a hand-written kernel writes it.
///
/// A kernel without it does not run on a [`RoutedDevice`]:
///
/// ```compile_fail,E0277
/// use gust::cpu::{CpuDevice, CpuJob, Error};
/// use gust::routed::{NoGpu, RoutedDevice};
/// use gust::{Call, Kernel, KernelDescriptor, Run};
///
/// struct Hand;
/// impl Kernel for Hand {
///     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "hand", "hand", [1, 1, 1], &[], "");
/// }
/// impl Run<Hand, 0> for CpuDevice {
///     fn submit_call(&self, _call: &Call<'_, Self, Hand, 0>) -> Result<CpuJob, Error> {
///         Ok(CpuJob::complete())
///     }
/// }
/// fn runs_on_the_handle<K: Kernel>()
/// where
///     RoutedDevice<NoGpu>: Run<K, 0>,
/// {
/// }
/// runs_on_the_handle::<Hand>(); // no `impl Routed for Hand`
/// ```
///
/// The same type with the line added does run (the control of the example above):
///
/// ```
/// use gust::cpu::{CpuDevice, CpuJob, Error};
/// use gust::routed::{NoGpu, OnCpu, Routed, RoutedDevice};
/// use gust::{Call, Kernel, KernelDescriptor, Run};
///
/// struct Hand;
/// impl Kernel for Hand {
///     const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new("m", "hand", "hand", [1, 1, 1], &[], "");
/// }
/// impl Run<Hand, 0> for CpuDevice {
///     fn submit_call(&self, _call: &Call<'_, Self, Hand, 0>) -> Result<CpuJob, Error> {
///         Ok(CpuJob::complete())
///     }
/// }
/// impl Routed for Hand {
///     type Side = OnCpu;
/// }
/// fn runs_on_the_handle<K: Kernel>()
/// where
///     RoutedDevice<NoGpu>: Run<K, 0>,
/// {
/// }
/// runs_on_the_handle::<Hand>();
/// ```
#[diagnostic::on_unimplemented(
    message = "the kernel `{Self}` does not say which target it runs on, so `gust::routed::RoutedDevice` cannot run it",
    note = "a `#[gpu]` kernel gets `Routed` from the macro; a hand-written `Kernel` adds `impl gust::routed::Routed for {Self} {{ type Side = gust::routed::OnGpu; }}` (or `OnCpu`)"
)]
pub trait Routed: Kernel {
    /// [`OnGpu`] or [`OnCpu`].
    type Side: Side;
}

/// What the handle needs of a GPU device. `gust-wgpu` implements it for `HeadlessDevice`.
pub trait GpuBackend:
    DeviceTrait<Error: std::error::Error + Send + Sync + 'static> + TypedDevice + Sized + 'static
{
    /// A buffer of this device, a storage resource of this device type.
    type Buffer<T: GpuPod>: Resource<Element = T, Device = Self, Kind = kind::Storage>;

    /// A buffer holding a copy of `values`.
    fn create_buffer<T: GpuPod>(
        &self,
        values: &[T],
    ) -> Result<Self::Buffer<T>, <Self as DeviceTrait>::Error>;
    /// Replace the contents of `buffer` (same length).
    fn write_buffer<T: GpuPod>(
        &self,
        buffer: &Self::Buffer<T>,
        values: &[T],
    ) -> Result<(), <Self as DeviceTrait>::Error>;
    /// A copy of the contents of `buffer`.
    fn read_buffer<T: GpuPod>(
        &self,
        buffer: &Self::Buffer<T>,
    ) -> Result<Vec<T>, <Self as DeviceTrait>::Error>;
}

/// The backend of a handle that has no GPU runtime: a GPU-routed kernel called on it is [`Error::NoGpu`].
#[derive(Clone, Copy, Debug)]
pub struct NoGpu;

/// The buffer type of [`NoGpu`]; it cannot be constructed.
#[derive(Debug)]
pub struct NoGpuBuffer<T>(Infallible, PhantomData<T>);

impl DeviceTrait for NoGpu {
    type Error = Infallible;
    type Dispatch<'r> = Infallible;
    type Job<'d> = NoGpuJob;

    fn submit(&self, _dispatches: &[Infallible]) -> Result<NoGpuJob, Infallible> {
        Ok(NoGpuJob)
    }
}

/// The job type of [`NoGpu`].
#[derive(Clone, Copy, Debug)]
pub struct NoGpuJob;

impl JobTrait for NoGpuJob {
    type Error = Infallible;
    fn status(&self) -> Result<JobStatus, Infallible> {
        Ok(JobStatus::Complete)
    }
    fn wait_timeout(&self, _timeout: Duration) -> Result<JobStatus, Infallible> {
        Ok(JobStatus::Complete)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}

impl TypedDevice for NoGpu {
    type Binding<'r> = ();
}

impl<T: GpuPod> Resource for NoGpuBuffer<T> {
    type Element = T;
    type Device = NoGpu;
    type Kind = kind::Storage;
    fn bind(&self, _access: Access, _length: LengthRule) {
        match self.0 {}
    }
}

impl<K: Kernel, const N: usize> Run<K, N> for NoGpu {
    fn submit_call(&self, _call: &Call<'_, Self, K, N>) -> Result<NoGpuJob, Infallible> {
        // The handle never reaches a backend it does not hold (`RoutedDevice::gpu` is `None`), so this is not run.
        Ok(NoGpuJob)
    }
}

impl GpuBackend for NoGpu {
    type Buffer<T: GpuPod> = NoGpuBuffer<T>;
    fn create_buffer<T: GpuPod>(&self, _values: &[T]) -> Result<NoGpuBuffer<T>, Infallible> {
        unreachable!("a handle without a GPU never asks its backend for a buffer")
    }
    fn write_buffer<T: GpuPod>(
        &self,
        buffer: &NoGpuBuffer<T>,
        _values: &[T],
    ) -> Result<(), Infallible> {
        match buffer.0 {}
    }
    fn read_buffer<T: GpuPod>(&self, buffer: &NoGpuBuffer<T>) -> Result<Vec<T>, Infallible> {
        match buffer.0 {}
    }
}

/// Why a call or a transfer on the handle failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The kernel is built for a GPU and this handle has none (it was made with [`RoutedDevice::cpu_only`], or its buffer
    /// was made before a GPU existed).
    NoGpu {
        /// The kernel's qualified name.
        kernel: String,
    },
    /// The CPU side refused or failed.
    Cpu(crate::cpu::Error),
    /// The GPU side refused or failed.
    Gpu(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGpu { kernel } => write!(
                f,
                "the kernel `{kernel}` runs on a GPU in this build, and this device handle has no GPU"
            ),
            Self::Cpu(error) => write!(f, "cpu: {error}"),
            Self::Gpu(error) => write!(f, "gpu: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoGpu { .. } => None,
            Self::Cpu(error) => Some(error),
            Self::Gpu(error) => Some(error.as_ref()),
        }
    }
}

fn gpu_error<G: GpuBackend>(error: <G as DeviceTrait>::Error) -> Error {
    Error::Gpu(Box::new(error))
}

/// A finished submission (the handle's calls complete inside `submit_call`).
#[derive(Clone, Copy, Debug)]
pub struct Done;

impl JobTrait for Done {
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

/// The device handle: a [`CpuDevice`] and, optionally, a GPU device `G`.
pub struct RoutedDevice<G: GpuBackend> {
    cpu: CpuDevice,
    gpu: Option<G>,
}

impl<G: GpuBackend> RoutedDevice<G> {
    /// A handle that has a GPU and the CPU.
    pub fn with_gpu(gpu: G) -> Self {
        Self {
            cpu: CpuDevice::new(),
            gpu: Some(gpu),
        }
    }

    /// A handle without a GPU: only CPU-routed kernels run on it.
    pub fn cpu_only() -> Self {
        Self {
            cpu: CpuDevice::new(),
            gpu: None,
        }
    }

    /// The GPU device, if the handle has one.
    pub fn gpu(&self) -> Option<&G> {
        self.gpu.as_ref()
    }

    /// The CPU device of the handle.
    pub fn cpu(&self) -> &CpuDevice {
        &self.cpu
    }

    /// A buffer holding a copy of `values`: on the host and, when the handle has a GPU, on the device.
    pub fn buffer<T: GpuPod>(&self, values: &[T]) -> Result<Buffer<T, G>, Error> {
        let gpu = match &self.gpu {
            Some(gpu) => Some(gpu.create_buffer(values).map_err(gpu_error::<G>)?),
            None => None,
        };
        Ok(Buffer {
            cpu: self.cpu.buffer(values),
            gpu,
            valid: Cell::new(Valid::Both),
        })
    }
}

impl<G: GpuBackend> DeviceTrait for RoutedDevice<G> {
    type Error = Error;
    type Dispatch<'r> = Infallible;
    type Job<'d>
        = Done
    where
        Self: 'd;

    fn submit(&self, _dispatches: &[Infallible]) -> Result<Done, Error> {
        Ok(Done)
    }
}

impl<G: GpuBackend> TypedDevice for RoutedDevice<G> {
    type Binding<'r> = Binding<'r, G>;
}

/// Where a buffer's current contents live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Valid {
    Both,
    Cpu,
    Gpu,
}

/// A typed storage buffer of a [`RoutedDevice`].
pub struct Buffer<T: GpuPod, G: GpuBackend> {
    cpu: CpuBuffer<T>,
    gpu: Option<G::Buffer<T>>,
    valid: Cell<Valid>,
}

impl<T: GpuPod, G: GpuBackend> Buffer<T, G> {
    /// A copy of the current contents (read back from the GPU first when the GPU holds them).
    pub fn read(&self, device: &RoutedDevice<G>) -> Result<Vec<T>, Error> {
        self.make_cpu_valid(device)?;
        Ok(self.cpu.read())
    }

    /// Replace the contents; the length must not change.
    pub fn write(&self, values: &[T]) -> Result<(), Error> {
        self.cpu.write(values).map_err(Error::Cpu)?;
        self.valid.set(Valid::Cpu);
        Ok(())
    }

    fn make_cpu_valid(&self, device: &RoutedDevice<G>) -> Result<(), Error> {
        if self.valid.get() == Valid::Gpu {
            let (Some(gpu), Some(buffer)) = (device.gpu.as_ref(), self.gpu.as_ref()) else {
                return Err(no_gpu());
            };
            let values = gpu.read_buffer(buffer).map_err(gpu_error::<G>)?;
            self.cpu.write(&values).map_err(Error::Cpu)?;
            self.valid.set(Valid::Both);
        }
        Ok(())
    }

    fn make_gpu_valid(&self, device: &RoutedDevice<G>) -> Result<(), Error> {
        if self.valid.get() == Valid::Cpu {
            let (Some(gpu), Some(buffer)) = (device.gpu.as_ref(), self.gpu.as_ref()) else {
                return Err(no_gpu());
            };
            gpu.write_buffer(buffer, &self.cpu.read())
                .map_err(gpu_error::<G>)?;
            self.valid.set(Valid::Both);
        }
        Ok(())
    }
}

fn no_gpu() -> Error {
    Error::NoGpu {
        kernel: String::from("(a buffer transfer)"),
    }
}

/// The object-safe face of a buffer that the call needs: move it to a side, and say that a side wrote it.
trait Mirror<G: GpuBackend> {
    fn to_cpu(&self, device: &RoutedDevice<G>) -> Result<(), Error>;
    fn to_gpu(&self, device: &RoutedDevice<G>) -> Result<(), Error>;
    fn wrote(&self, side: Valid);
}

impl<T: GpuPod, G: GpuBackend> Mirror<G> for Buffer<T, G> {
    fn to_cpu(&self, device: &RoutedDevice<G>) -> Result<(), Error> {
        self.make_cpu_valid(device)
    }
    fn to_gpu(&self, device: &RoutedDevice<G>) -> Result<(), Error> {
        self.make_gpu_valid(device)
    }
    fn wrote(&self, side: Valid) {
        self.valid.set(side);
    }
}

/// The erased view of one bound [`Buffer`]: both sides' bindings and the means to move the buffer between them.
pub struct Binding<'r, G: GpuBackend> {
    cpu: CpuBinding<'r>,
    gpu: Option<G::Binding<'r>>,
    mirror: &'r dyn Mirror<G>,
    access: Access,
}

impl<G: GpuBackend> Clone for Binding<'_, G> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<G: GpuBackend> Copy for Binding<'_, G> {}

impl<T: GpuPod, G: GpuBackend> Resource for Buffer<T, G> {
    type Element = T;
    type Device = RoutedDevice<G>;
    type Kind = kind::Storage;

    fn bind<'r>(&'r self, access: Access, length: LengthRule) -> Binding<'r, G> {
        Binding {
            cpu: self.cpu.bind(access, length),
            gpu: self.gpu.as_ref().map(|buffer| buffer.bind(access, length)),
            mirror: self,
            access,
        }
    }
}

/// Runs a call on one side. Implemented for [`OnCpu`] and [`OnGpu`]; [`RoutedDevice`]'s `Run` impl picks the impl by
/// `K::Side`, so the choice is a type-level fact and nothing is looked up at run time.
pub trait SideRun<K: Kernel, const N: usize, G: GpuBackend> {
    /// Run `call` on this side of `device`.
    fn run(device: &RoutedDevice<G>, call: &Call<'_, RoutedDevice<G>, K, N>) -> Result<(), Error>;
}

fn written(access: Access) -> bool {
    !matches!(access, Access::ReadOnly | Access::NotApplicable)
}

impl<K: Kernel, const N: usize, G: GpuBackend> SideRun<K, N, G> for OnCpu
where
    CpuDevice: Run<K, N>,
{
    fn run(device: &RoutedDevice<G>, call: &Call<'_, RoutedDevice<G>, K, N>) -> Result<(), Error> {
        for binding in call.bindings() {
            binding.mirror.to_cpu(device)?;
        }
        let cpu_call = Call::<CpuDevice, K, N>::new_extent(
            call.extent(),
            std::array::from_fn(|index| call.bindings()[index].cpu),
        );
        device.cpu.dispatch_call(&cpu_call).map_err(Error::Cpu)?;
        for binding in call.bindings() {
            if written(binding.access) {
                binding.mirror.wrote(Valid::Cpu);
            }
        }
        Ok(())
    }
}

impl<K: GpuKernel, const N: usize, G: GpuBackend + Run<K, N>> SideRun<K, N, G> for OnGpu {
    fn run(device: &RoutedDevice<G>, call: &Call<'_, RoutedDevice<G>, K, N>) -> Result<(), Error> {
        let Some(gpu) = device.gpu.as_ref() else {
            return Err(Error::NoGpu {
                kernel: call.kernel().qualified_name().to_string(),
            });
        };
        for binding in call.bindings() {
            binding.mirror.to_gpu(device)?;
        }
        let mut bindings = Vec::with_capacity(N);
        for binding in call.bindings() {
            bindings.push(binding.gpu.ok_or_else(|| Error::NoGpu {
                kernel: call.kernel().qualified_name().to_string(),
            })?);
        }
        let bindings: [G::Binding<'_>; N] = bindings
            .try_into()
            .unwrap_or_else(|_| unreachable!("one binding per parameter"));
        let gpu_call = Call::<G, K, N>::new_extent(call.extent(), bindings);
        gpu.dispatch_call(&gpu_call).map_err(gpu_error::<G>)?;
        for binding in call.bindings() {
            if written(binding.access) {
                binding.mirror.wrote(Valid::Gpu);
            }
        }
        Ok(())
    }
}

impl<K: Routed, const N: usize, G: GpuBackend> Run<K, N> for RoutedDevice<G>
where
    K::Side: SideRun<K, N, G>,
{
    fn submit_call(&self, call: &Call<'_, Self, K, N>) -> Result<Done, Error> {
        <K::Side as SideRun<K, N, G>>::run(self, call)?;
        Ok(Done)
    }
}
