//! Capture one GPU call and replay it on the CPU twin (stage G2 of the debugging design, slice g2.cap).
//!
//! A wrong GPU result can be studied by reading buffers back, but not by stepping through the kernel. Capture copies
//! the inputs of one call, as they are just before the dispatch, into [`CpuDevice`] buffers; the replay then runs
//! chosen invocations of the kernel's CPU twin over those copies, where a debugger can break inside the kernel body.
//!
//! The typed `capture` function is generated per kernel, under the twin's `cfg` (a test build of the crate that defines
//! the kernel, or `--cfg gust_dual`), because only the macro knows each parameter's element type:
//!
//! ```text
//! let captured = update::step::capture(&gpu, &call)?;   // before the dispatch
//! captured.run_element([4711, 0, 0])?;                  // one invocation on the twin
//! captured.run_until([4711, 0, 0])?;                    // the twin's launch up to and including 4711
//! captured.run_all()?;                                  // the twin's whole launch
//! let out = captured.buffer::<float>("out").unwrap().read();
//! ```
//!
//! # What a replay is
//!
//! A replay reproduces the twin's order, never the GPU's schedule (the GPU has no order): the twin's launch visits
//! every invocation of the extent rounded up to whole workgroups, ascending, one at a time, on the captured copies
//! (never on the GPU buffers). [`Captured::run_all`] is exactly that launch, padding included, so it equals the twin's
//! own call of the kernel; for a race-free kernel without atomics it also equals the GPU's result. For a kernel with
//! atomics, or whose result depends on the order of invocations, it is one member of the allowed set
//! (`docs/SEMANTICS.md`, section 6 and the domain paragraph of section 3), not the history the GPU had. The twin also
//! has the twin's semantics: Rust's integer overflow and division rules and its own floating-point rounding, which the
//! GPU can differ from (documented limitations, D39).
//!
//! The replay runs the twin's body, which for a kernel with variants is the portable variant, not necessarily the body
//! the GPU ran.
//!
//! Every run starts from the captured values: the read-write buffers are restored first, so running one element twice,
//! or after `run_until`, gives the same result each time. Inputs (read-only buffers and uniforms) are not restored, so
//! an edit of one through [`Captured::buffer`] or [`Captured::uniform`] ("what if this input were different?") stays
//! for every later run.
//!
//! A kernel with `cpu = "unsupported"`, or with no body that runs without a capability, has no twin: its `capture`
//! returns [`Error::NoTwin`]. Textures and samplers are not captured yet ([`Error::Unsupported`]). A call the GPU would
//! refuse (the same buffer bound twice with one writer, a binding of another element layout, access or length) is
//! refused by `capture` with the device's own error ([`Error::Call`]).

use core::any::Any;
use core::fmt;
use core::marker::PhantomData;

use crate::cpu::{self, CpuBinding, CpuBuffer, CpuDevice, CpuUniform};
use crate::{
    Access, Extent, GpuPod, Kernel, KernelDescriptor, LengthRule, Resource, TypedDevice, UVec3,
};

/// The values of one captured storage buffer or uniform and the length rule it was bound with.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferCopy<T> {
    /// The elements as they were when the copy was made: one element for a uniform.
    pub values: Vec<T>,
    /// How the binding's length related to the call's extent.
    pub length: LengthRule,
}

/// A [`TypedDevice`] whose bound buffers can be copied to the host for a capture.
///
/// `gust_wgpu::HeadlessDevice` implements it; a buffer that cannot be read back is the device's own error, and
/// [`hint`](Self::hint) says how to fix it.
pub trait CaptureDevice: TypedDevice {
    /// Check the call (`kernel` over `extent` with `bindings`) the way the device checks a dispatch, with the device's
    /// own validation: the same error the dispatch of this call would give. The generated `capture` calls it first, so
    /// a call the device refuses is never captured.
    fn check_call(
        &self,
        kernel: &KernelDescriptor,
        extent: Extent,
        bindings: &[Self::Binding<'_>],
    ) -> Result<(), Self::Error>;

    /// Copy what `binding` binds (a storage buffer or a uniform of element type `T`) to the host: a uniform gives one
    /// element. The binding is never a texture. The copy sees every submission made before it.
    fn capture_buffer<T: GpuPod>(
        &self,
        binding: &Self::Binding<'_>,
    ) -> Result<BufferCopy<T>, Self::Error>;

    /// How to fix `error`, when the device knows: appended to the message of [`Error::Read`].
    fn hint(error: &Self::Error) -> Option<&'static str> {
        let _ = error;
        None
    }
}

/// Why a capture failed. `E` is the device's own error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error<E> {
    /// The kernel has no twin to replay on: `#[kernel(cpu = "unsupported")]`, or no body that runs without a
    /// capability.
    NoTwin {
        /// The kernel.
        kernel: &'static str,
    },
    /// The kernel has a parameter this release does not capture (a texture or a sampler).
    Unsupported {
        /// The kernel.
        kernel: &'static str,
        /// The first parameter that is not captured.
        parameter: &'static str,
        /// What the parameter is.
        what: &'static str,
    },
    /// The device refused the call itself, with the error a dispatch of it gives (see [`CaptureDevice::check_call`]).
    Call(E),
    /// The device could not copy a bound buffer.
    Read {
        /// The kernel parameter.
        parameter: &'static str,
        /// The device's error.
        error: E,
        /// How to fix it, when the device knows.
        hint: Option<&'static str>,
    },
    /// The copy could not become a twin buffer (a uniform type the twin refuses).
    Twin {
        /// The kernel parameter.
        parameter: &'static str,
        /// The twin's error.
        error: cpu::Error,
    },
}

impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTwin { kernel } => write!(
                formatter,
                "cannot capture `{kernel}`: no twin (no CPU body to replay: `cpu = \"unsupported\"`, or no \
                 body that runs without a capability)"
            ),
            Self::Unsupported {
                kernel,
                parameter,
                what,
            } => write!(
                formatter,
                "cannot capture `{kernel}`: parameter `{parameter}` is {what}, which capture does not support yet"
            ),
            Self::Call(error) => {
                write!(formatter, "cannot capture: the call is not valid: {error}")
            }
            Self::Read {
                parameter,
                error,
                hint,
            } => {
                write!(formatter, "cannot capture parameter `{parameter}`: {error}")?;
                match hint {
                    Some(hint) => write!(formatter, "; hint: {hint}"),
                    None => Ok(()),
                }
            }
            Self::Twin { parameter, error } => write!(
                formatter,
                "cannot capture parameter `{parameter}` for the twin: {error}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Call(error) | Self::Read { error, .. } => Some(error),
            Self::Twin { error, .. } => Some(error),
            Self::NoTwin { .. } | Self::Unsupported { .. } => None,
        }
    }
}

/// Why a replay failed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayError {
    /// The id lies outside the call's extent on some axis (or the extent is empty). Nothing ran.
    OutOfExtent {
        /// The id asked for.
        id: [u32; 3],
        /// The extent of the captured call.
        extent: Extent,
    },
    /// The twin refused the captured call.
    Twin(cpu::Error),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfExtent { id, extent } => write!(
                formatter,
                "invocation id {id:?} is outside the captured extent [{}, {}, {}]",
                extent.x, extent.y, extent.z
            ),
            Self::Twin(error) => write!(formatter, "the twin refused the captured call: {error}"),
        }
    }
}

impl std::error::Error for ReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Twin(error) => Some(error),
            Self::OutOfExtent { .. } => None,
        }
    }
}

impl From<cpu::Error> for ReplayError {
    fn from(error: cpu::Error) -> Self {
        Self::Twin(error)
    }
}

/// The type-erased contents of one slot.
trait SlotData {
    /// Put the captured values back into the working copy; a no-op for an input, which a run cannot change.
    fn restore(&self) -> Result<(), cpu::Error>;
    fn as_any(&self) -> &dyn Any;
}

struct StorageSlot<T: GpuPod> {
    work: CpuBuffer<T>,
    /// The values a run restores: kept for a read-write buffer only.
    captured: Option<Vec<T>>,
    length: LengthRule,
}

impl<T: GpuPod> SlotData for StorageSlot<T> {
    fn restore(&self) -> Result<(), cpu::Error> {
        match &self.captured {
            Some(captured) => self.work.write(captured),
            None => Ok(()),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct UniformSlot<T: GpuPod> {
    work: CpuUniform<T>,
}

impl<T: GpuPod> SlotData for UniformSlot<T> {
    fn restore(&self) -> Result<(), cpu::Error> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One captured parameter: the working copy on the twin's device and the values it is restored from. Macro support.
#[doc(hidden)]
pub struct Slot {
    name: &'static str,
    data: Box<dyn SlotData>,
}

/// The generated replay of one kernel: validate the rebuilt call, acquire the slots and run the twin's body for each
/// id. Macro support.
#[doc(hidden)]
pub type ReplayFn =
    fn(&CpuDevice, Extent, &[Slot], &mut dyn Iterator<Item = UVec3>) -> Result<(), cpu::Error>;

/// Copy the storage buffer `binding` (bound with `access`) into a slot of `cpu`. Macro support.
#[doc(hidden)]
pub fn capture_storage<D: CaptureDevice, T: GpuPod>(
    device: &D,
    binding: &D::Binding<'_>,
    parameter: &'static str,
    access: Access,
    cpu: &CpuDevice,
) -> Result<Slot, Error<D::Error>> {
    let BufferCopy { values, length } = read_buffer::<D, T>(device, binding, parameter)?;
    Ok(Slot {
        name: parameter,
        data: Box::new(StorageSlot {
            work: cpu.buffer(&values),
            captured: (access == Access::ReadWrite).then_some(values),
            length,
        }),
    })
}

/// Copy the uniform `binding` binds into a slot of `cpu`. Macro support.
#[doc(hidden)]
pub fn capture_uniform<D: CaptureDevice, T: GpuPod>(
    device: &D,
    binding: &D::Binding<'_>,
    parameter: &'static str,
    cpu: &CpuDevice,
) -> Result<Slot, Error<D::Error>> {
    let BufferCopy { values, .. } = read_buffer::<D, T>(device, binding, parameter)?;
    // `CaptureDevice::capture_buffer` gives one element for a uniform (its documented contract; an implementation
    // that breaks it panics here).
    let value = *values
        .first()
        .expect("a uniform binding holds exactly one element");
    let work = cpu
        .uniform(value)
        .map_err(|error| Error::Twin { parameter, error })?;
    Ok(Slot {
        name: parameter,
        data: Box::new(UniformSlot { work }),
    })
}

fn read_buffer<D: CaptureDevice, T: GpuPod>(
    device: &D,
    binding: &D::Binding<'_>,
    parameter: &'static str,
) -> Result<BufferCopy<T>, Error<D::Error>> {
    device
        .capture_buffer::<T>(binding)
        .map_err(|error| Error::Read {
            parameter,
            hint: D::hint(&error),
            error,
        })
}

/// The twin binding of a storage slot of element type `T`. Macro support.
#[doc(hidden)]
pub fn bind_storage<T: GpuPod>(slot: &Slot, access: Access) -> Result<CpuBinding<'_>, cpu::Error> {
    let data =
        slot.data
            .as_any()
            .downcast_ref::<StorageSlot<T>>()
            .ok_or(cpu::Error::ElementType {
                parameter: slot.name,
            })?;
    Ok(data.work.bind(access, data.length))
}

/// The twin binding of a uniform slot of element type `T`. Macro support.
#[doc(hidden)]
pub fn bind_uniform<T: GpuPod>(slot: &Slot) -> Result<CpuBinding<'_>, cpu::Error> {
    let data =
        slot.data
            .as_any()
            .downcast_ref::<UniformSlot<T>>()
            .ok_or(cpu::Error::ElementType {
                parameter: slot.name,
            })?;
    Ok(data.work.bind(Access::ReadOnly, LengthRule::Independent))
}

/// One GPU call captured for replay on the CPU twin. Made by the generated `capture` function of a kernel; see the
/// [module](self) for what a replay is and is not.
///
/// It owns copies of the call's buffers and uniforms on a private [`CpuDevice`], and the call's extent.
pub struct Captured<K: Kernel> {
    device: CpuDevice,
    extent: Extent,
    slots: Vec<Slot>,
    replay: ReplayFn,
    kernel: PhantomData<fn() -> K>,
}

impl<K: Kernel> fmt::Debug for Captured<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Captured")
            .field("kernel", &K::DESCRIPTOR.name)
            .field("extent", &self.extent)
            .field(
                "parameters",
                &self.slots.iter().map(|slot| slot.name).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl<K: Kernel> Captured<K> {
    /// Assemble a capture. Macro support.
    #[doc(hidden)]
    pub fn new(device: CpuDevice, extent: Extent, slots: Vec<Slot>, replay: ReplayFn) -> Self {
        Self {
            device,
            extent,
            slots,
            replay,
            kernel: PhantomData,
        }
    }

    /// The extent of the captured call.
    pub fn extent(&self) -> Extent {
        self.extent
    }

    /// The working copy of the storage buffer parameter `name`: what the replays read and write. `None` when there is
    /// no such buffer parameter or its element type is not `T`. It holds the captured values until a run changes it,
    /// and the result of the last run after one.
    pub fn buffer<T: GpuPod>(&self, name: &str) -> Option<&CpuBuffer<T>> {
        self.slot(name)?
            .data
            .as_any()
            .downcast_ref::<StorageSlot<T>>()
            .map(|slot| &slot.work)
    }

    /// The working copy of the uniform parameter `name`; `None` when there is none or its type is not `T`.
    pub fn uniform<T: GpuPod>(&self, name: &str) -> Option<&CpuUniform<T>> {
        self.slot(name)?
            .data
            .as_any()
            .downcast_ref::<UniformSlot<T>>()
            .map(|slot| &slot.work)
    }

    fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|slot| slot.name == name)
    }

    /// Run exactly one invocation, `id`, on the twin, over the captured outputs and the current inputs. Only the
    /// buffers that invocation writes change; everything else keeps its value. A panic in the kernel body unwinds
    /// through this call, which is where a debugger breaks.
    ///
    /// An id outside the call's extent is [`ReplayError::OutOfExtent`], and nothing runs. The ids of the padding of the
    /// launch (at or past the extent on some axis) are outside it too: only [`run_until`](Self::run_until) (before its
    /// target) and [`run_all`](Self::run_all) run them.
    pub fn run_element(&self, id: [u32; 3]) -> Result<(), ReplayError> {
        self.check(id)?;
        self.run(&mut core::iter::once(UVec3 {
            x: id[0],
            y: id[1],
            z: id[2],
        }))
    }

    /// Run the twin's launch from its first invocation up to and including `id`, in the twin's order (ascending, `x`
    /// fastest, then `y`, then `z`, one at a time), over the captured outputs and the current inputs. The launch covers
    /// every axis rounded up to whole workgroups, so the ids before `id` can include padding ids (at or past the extent
    /// on some axis); the kernel body runs for them as the twin runs it, and a body that does not guard them can write
    /// there. With atomics or an order-dependent result the outcome is one member of the allowed set, not the history
    /// the GPU had.
    ///
    /// An id outside the call's extent is [`ReplayError::OutOfExtent`], and nothing runs. To run the whole launch, use
    /// [`run_all`](Self::run_all).
    pub fn run_until(&self, id: [u32; 3]) -> Result<(), ReplayError> {
        self.check(id)?;
        let target = UVec3 {
            x: id[0],
            y: id[1],
            z: id[2],
        };
        let mut reached = false;
        let mut ids = self.launch()?.take_while(|current| {
            if reached {
                return false;
            }
            reached = *current == target;
            true
        });
        self.run(&mut ids)
    }

    /// Run the twin's whole launch, padding ids included, over the captured outputs and the current inputs: exactly
    /// what the twin does for the call. For a race-free kernel without atomics this equals the GPU's result; with
    /// atomics or an order-dependent result it is one member of the allowed set, not the history the GPU had.
    pub fn run_all(&self) -> Result<(), ReplayError> {
        let mut ids = self.launch()?;
        self.run(&mut ids)
    }

    /// The twin's launch: every invocation id of the extent rounded up to whole workgroups.
    fn launch(&self) -> Result<impl Iterator<Item = UVec3>, ReplayError> {
        let size = K::DESCRIPTOR.workgroup_size;
        if size.contains(&0) {
            return Err(cpu::Error::UnsupportedWorkgroupSize { size }.into());
        }
        Ok(cpu::invocations(&K::DESCRIPTOR, self.extent))
    }

    fn check(&self, id: [u32; 3]) -> Result<(), ReplayError> {
        let extent = self.extent;
        if id[0] < extent.x && id[1] < extent.y && id[2] < extent.z {
            Ok(())
        } else {
            Err(ReplayError::OutOfExtent { id, extent })
        }
    }

    fn run(&self, ids: &mut dyn Iterator<Item = UVec3>) -> Result<(), ReplayError> {
        for slot in &self.slots {
            slot.data.restore()?;
        }
        (self.replay)(&self.device, self.extent, &self.slots, ids)?;
        Ok(())
    }
}
