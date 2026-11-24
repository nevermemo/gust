//! Stream compaction of a `GpuVec` of flags: [`count`], [`indices`] and the gathers [`gather_u32`], [`gather_i32`] and
//! [`gather_f32`] (plan section 5.4).
//!
//! Nothing here reads a value back to the host: the count of a compaction stays on the device as a one-element
//! `GpuVec`, ready for whatever consumes it next (the T13 rule). Because the host never learns the count, no result can
//! be sized to it; the output lengths below follow from that.
//!
//! * The **flag rule**: a flag is set when it is not zero, so the flags may be any `u32`s, not only 0 and 1. A first
//!   step turns them into a 0/1 mask, and the count and the scan see only the mask. The mask is why the sum of the
//!   count is exact: every partial is at most `n`, inside the domain of plain `+` for every `n` the binding limit
//!   admits (`n <= 2^29`, section 5.1).
//! * [`count`] is the number of set flags: one element, on the device (`reduce::sum_on_device` of the mask).
//! * [`indices`] returns `(indices, count)`. `indices` is `flags.len()` long (the most a compaction can produce); its
//!   first `count` elements are the positions of the set flags in increasing order, and **every element from `count`
//!   on is `0`** (the buffer is created zeroed and only the first `count` slots are written). That zero tail is the
//!   promise of this immediate function alone: the recording form ([`record`](crate::record)) writes the same first
//!   `count` elements into a buffer of the caller's, and past `count` its contents are unspecified once a list has run
//!   more than once. The tail is not a sentinel: `0` is also a valid index, so a consumer must use the count, which is
//!   what the gathers do. The indices
//!   come from the exclusive scan of the mask: output slot `scan[i]` belongs to element `i` alone, so the scatter
//!   `indices[scan[i]] = i` has one writer per slot.
//! * The **gathers** compute `out[j] = values[indices[j]]` for `j < count[0]`. The output is `indices.len()` long,
//!   which is `flags.len()` when `indices` came from [`indices`], and is zero from `count` on, like the indices. The
//!   bound is guarded in the kernel: a slot at or past `count`, and an index that is not below `values.len()`, stay
//!   `0`. To use only the dense prefix, hand `count` (still on the device) to the next kernel.
//!
//! An empty input is not an error: the count is `[0]`, `indices` is empty, and a gather of empty indices is empty.
//! Inputs past the device's limits (the storage binding limit for any buffer) are `Error::Device(DeviceLimit)` before
//! any buffer is taken or dispatch is made. A launch past the workgroup limit is split by the host (D211, D208) and is
//! no error.
//!
//! # A gather of structs
//!
//! There is no generic `gather<T>`: generic GPU functions are not part of the dialect before M9's traits (plan section
//! 5.4). For a struct, write the gather kernel yourself, against the `indices` and `count` of [`indices`]. Copy the
//! block-strided body of `gather_u32` in `tests/fixtures/compact.rs` and change the element type: launch
//! `schedule::Pass::over(n).partials()` invocations, not one per element (that is refused past 65 535
//! workgroups, 16 776 960 elements at 256 lanes), and keep the `source < values.len()` check before the read (an
//! index past the values is then skipped, not read out of bounds). `src/tests/struct_gather.rs` holds a worked,
//! tested example (`out[j] = particles[indices[j]]` for a struct of a `u32` and an `f32`).

#[cfg(not(target_arch = "wasm32"))]
use gust::GpuPod;
use gust::Independent;
#[cfg(not(target_arch = "wasm32"))]
use gust_wgpu::{BufferDispatch, HeadlessDevice};
use gust_wgpu::{DeviceHandle, GpuVec};

use crate::{
    Error, Variant, compact_kernels, limits,
    record::{Operand, Sink, Stage},
    reduce::{self, Op, check_len, non_empty},
    scan,
    schedule::Pass,
    variant::{Operation, shipped},
};

/// The name the limit errors carry.
const KERNEL: &str = "gust_algo::compact";

/// The number of set (non-zero) flags, as a one-element `GpuVec<u32>` on the device of `flags`: no readback. An empty
/// input counts `0`.
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the device's limits.
#[cfg(not(target_arch = "wasm32"))]
pub fn count(flags: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    if flags.is_empty() {
        return Ok(GpuVec::zeroed_on(flags.device(), 1)?);
    }
    reduce::sum_on_device(&mask_of(flags)?)
}

/// The positions of the set (non-zero) flags and their number: `(indices, count)`, both on the device of `flags`, no
/// readback. `indices` is `flags.len()` long; its first `count` elements are the increasing positions of the set flags
/// and the rest are `0` (see the module documentation; the recording form does not promise the zero tail). `count` has
/// one element.
///
/// # Errors
///
/// As for [`count`].
#[cfg(not(target_arch = "wasm32"))]
pub fn indices(flags: &GpuVec<u32>) -> Result<(GpuVec<u32>, GpuVec<u32>), Error> {
    let device = flags.device();
    if flags.is_empty() {
        return Ok((GpuVec::zeroed_on(device, 0)?, GpuVec::zeroed_on(device, 1)?));
    }
    let mask = mask_of(flags)?;
    let positions = scan::exclusive(&mask)?;
    let total = reduce::sum_on_device(&mask)?;
    let indices = GpuVec::<u32>::zeroed_on(device, flags.len())?;
    let mut sink = Sink::new();
    plan_scatter(&mut sink, flags.len(), &mask, &positions, &indices);
    sink.dispatch(device)?;
    Ok((indices, total))
}

/// The 0/1 mask of `flags` (1 where the flag is not zero), after the limit checks.
#[cfg(not(target_arch = "wasm32"))]
fn mask_of(flags: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    let device = flags.device();
    limits::check(device, KERNEL, flags.len(), size_of::<u32>())?;
    // Every element is written, so the zero fill is not needed for correctness; `GpuVec` has no public constructor
    // without it (see `reduce`).
    let mask = GpuVec::<u32>::zeroed_on(device, flags.len())?;
    let mut sink = Sink::new();
    plan_mask(&mut sink, flags.len(), flags, &mask);
    sink.dispatch(device)?;
    Ok(mask)
}

/// Append the dispatch that turns the `len` flags of `flags` into the 0/1 `mask`.
fn plan_mask<'s, F: Operand<u32>>(
    sink: &mut Sink<'s>,
    len: u32,
    flags: &'s F,
    mask: &'s GpuVec<u32>,
) {
    sink.push(&compact_kernels::flags_to_mask_u32::call(
        Pass::over(len).partials(),
        Independent(flags),
        Independent(mask),
    ));
}

/// Append the dispatch that writes `indices[positions[i]] = i` for every set element `i` of the `len`-element `mask`.
fn plan_scatter<'s, O: Operand<u32>>(
    sink: &mut Sink<'s>,
    len: u32,
    mask: &'s GpuVec<u32>,
    positions: &'s GpuVec<u32>,
    indices: &'s O,
) {
    sink.push(&compact_kernels::scatter_indices_u32::call(
        Pass::over(len).partials(),
        Independent(mask),
        Independent(positions),
        Independent(indices),
    ));
}

/// The dispatches of one `count` of `len` flags, made once and appended over any operands of that length (see
/// [`Sink`]): the mask and the reduction of the mask, owned, with no borrow of the flags or the result and no
/// lifetime.
///
/// **A plan must outlive the GPU work of every dispatch appended from it**: dropping it returns its mask and partials
/// to the device's pool, and a new owner may then write them while those dispatches still run. The plan may be
/// appended more than once; the appended runs share the scratch, so two of them must not be in flight at the same
/// time. The count of no flags is `0` ([`Op::identity`]); an empty input has no plan ([`Error::Empty`]).
pub struct CountPlan {
    mask: GpuVec<u32>,
    total: reduce::Parts<u32>,
    len: u32,
}

impl CountPlan {
    /// The plan of the count of `len` flags on `device`. `variant` overrides the shipped one of the sum of the mask.
    ///
    /// # Errors
    ///
    /// `Error::Empty` for `len == 0`, `Error::Device` for a length past the device's storage binding limit or a
    /// failed allocation.
    pub fn new(device: &DeviceHandle, variant: Option<Variant>, len: u32) -> Result<Self, Error> {
        non_empty(len)?;
        limits::check(device, KERNEL, len, size_of::<u32>())?;
        let mask = GpuVec::<u32>::zeroed_on(device, len)?;
        let total = reduce::Parts::new(device, variant.unwrap_or(shipped(Operation::Sum)), len)?;
        Ok(Self { mask, total, len })
    }

    /// The number of flags the plan was made for.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// A plan is never for an empty input, so this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Append the dispatches of the count of `flags` into the one-element `count` to `sink`: the mask, then the sum.
    ///
    /// # Errors
    ///
    /// `Error::Length` when `flags` does not hold the plan's length or `count` does not hold one element; nothing is
    /// appended then.
    pub fn append<'s, F: Operand<u32>, C: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        flags: &'s F,
        count: &'s C,
    ) -> Result<(), Error> {
        check_len(self.len, flags.operand_len())?;
        check_len(1, count.operand_len())?;
        plan_mask(sink, self.len, flags, &self.mask);
        self.total.plan(Op::Sum, sink, &self.mask, count);
        Ok(())
    }
}

/// The dispatches of one `indices` of `len` flags, made once and appended over any operands of that length: the mask,
/// its exclusive scan, its sum and the scatter, in the order of the immediate functions, with the same ownership,
/// lifetime rule and empty-input rule as [`CountPlan`].
///
/// A run writes `count` whole and the first `count` elements of `indices`; **past `count` the contents of `indices`
/// are unspecified once the plan has run more than once** (a buffer made with `GpuVec::zeroed_on` shows the zero
/// tail of the immediate function on its first run). A consumer uses the count.
pub struct CompactPlan {
    counted: CountPlan,
    scan: scan::Parts,
    positions: GpuVec<u32>,
}

impl CompactPlan {
    /// The plan of the compaction of `len` flags on `device`. `variant` overrides the shipped one of the scan and the
    /// sum.
    ///
    /// # Errors
    ///
    /// As for [`CountPlan::new`].
    pub fn new(device: &DeviceHandle, variant: Option<Variant>, len: u32) -> Result<Self, Error> {
        let counted = CountPlan::new(device, variant, len)?;
        let scan = scan::Parts::new(
            device,
            variant.unwrap_or(shipped(Operation::Exclusive)),
            len,
        )?
        .expect("the mask is not empty");
        let positions = GpuVec::<u32>::zeroed_on(device, len)?;
        Ok(Self {
            counted,
            scan,
            positions,
        })
    }

    /// The number of flags the plan was made for.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.counted.len
    }

    /// A plan is never for an empty input, so this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Append the dispatches of the compaction of `flags` into `indices` and the one-element `count` to `sink`: the
    /// mask, the exclusive scan of the mask, the sum of the mask and the scatter.
    ///
    /// # Errors
    ///
    /// `Error::Length` when `flags` or `indices` does not hold the plan's length or `count` does not hold one
    /// element; nothing is appended then.
    pub fn append<'s, F: Operand<u32>, I: Operand<u32>, C: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        flags: &'s F,
        indices: &'s I,
        count: &'s C,
    ) -> Result<(), Error> {
        let len = self.counted.len;
        check_len(len, flags.operand_len())?;
        check_len(len, indices.operand_len())?;
        check_len(1, count.operand_len())?;
        let mask = &self.counted.mask;
        plan_mask(sink, len, flags, mask);
        self.scan.plan(sink, mask, &self.positions, true);
        self.counted.total.plan(Op::Sum, sink, mask, count);
        plan_scatter(sink, len, mask, &self.positions, indices);
        Ok(())
    }
}

/// A recorded `count` of non-empty flags.
pub(crate) struct Count<'a> {
    plan: CountPlan,
    flags: &'a GpuVec<u32>,
    count: &'a GpuVec<u32>,
}

impl<'a> Count<'a> {
    /// `Error::Length` for a `count` that is not one element, `Error::Empty` for empty `flags`.
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Option<Variant>,
        flags: &'a GpuVec<u32>,
        count: &'a GpuVec<u32>,
    ) -> Result<Self, Error> {
        check_len(1, count.len())?;
        let plan = CountPlan::new(device, variant, flags.len())?;
        Ok(Self { plan, flags, count })
    }
}

impl Stage for Count<'_> {
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>) {
        self.plan
            .append(sink, self.flags, self.count)
            .expect("the operands were checked when the plan was made");
    }
}

/// A recorded `indices` of non-empty flags.
pub(crate) struct Compact<'a> {
    plan: CompactPlan,
    flags: &'a GpuVec<u32>,
    indices: &'a GpuVec<u32>,
    count: &'a GpuVec<u32>,
}

impl<'a> Compact<'a> {
    /// `Error::Length` for an `indices` that is not as long as `flags` or a `count` that is not one element,
    /// `Error::Empty` for empty `flags`.
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Option<Variant>,
        flags: &'a GpuVec<u32>,
        indices: &'a GpuVec<u32>,
        count: &'a GpuVec<u32>,
    ) -> Result<Self, Error> {
        check_len(1, count.len())?;
        check_len(flags.len(), indices.len())?;
        let plan = CompactPlan::new(device, variant, flags.len())?;
        Ok(Self {
            plan,
            flags,
            indices,
            count,
        })
    }
}

impl Stage for Compact<'_> {
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>) {
        self.plan
            .append(sink, self.flags, self.indices, self.count)
            .expect("the operands were checked when the plan was made");
    }
}

/// The shared frame of the gathers: the count and device checks, the limit checks, the zeroed output of
/// `indices.len()` elements, which `run` fills (it is given the device, the invocation count and the output).
#[cfg(not(target_arch = "wasm32"))]
fn gather_with<T: GpuPod>(
    values: &GpuVec<T>,
    indices: &GpuVec<u32>,
    count: &GpuVec<u32>,
    run: impl FnOnce(&HeadlessDevice, u32, &GpuVec<T>) -> Result<(), gust_wgpu::Error>,
) -> Result<GpuVec<T>, Error> {
    let device = indices.device();
    if count.len() != 1 {
        return Err(Error::Count(count.len()));
    }
    values.check_device(device)?;
    count.check_device(device)?;
    let pass = Pass::over(indices.len());
    limits::check(device, KERNEL, indices.len(), size_of::<u32>())?;
    limits::check(device, KERNEL, values.len(), size_of::<T>())?;
    let out = GpuVec::<T>::zeroed_on(device, indices.len())?;
    // With nothing to read from or to write, the zeroed output is the result, and no empty binding is made.
    if !indices.is_empty() && !values.is_empty() {
        run(device, pass.partials(), &out)?;
    }
    Ok(out)
}

macro_rules! gather {
    ($(#[$meta:meta])* $name:ident, $ty:ty) => {
        $(#[$meta])*
        ///
        /// `out[j] = values[indices[j]]` for `j < count[0]`; `out` is `indices.len()` long and `0` from `count[0]` on,
        /// and also where an index is not below `values.len()` (see the module documentation). `count` must hold
        /// exactly one element, normally the `count` returned by [`indices`]; nothing is read back.
        ///
        /// # Errors
        ///
        /// `Error::Count` if `count` does not hold exactly one element; `Error::Device` for a device failure, a buffer
        /// of another device, or a buffer past the device's limits.
        #[cfg(not(target_arch = "wasm32"))]
        pub fn $name(
            values: &GpuVec<$ty>,
            indices: &GpuVec<u32>,
            count: &GpuVec<u32>,
        ) -> Result<GpuVec<$ty>, Error> {
            gather_with(values, indices, count, |device, invocations, out| {
                let call = compact_kernels::$name::call(
                    invocations,
                    Independent(values),
                    Independent(indices),
                    Independent(count),
                    Independent(out),
                );
                crate::timing::dispatch_batch_in_scope(device, &[BufferDispatch::from(&call)])
            })
        }
    };
}

gather!(
    /// Gather `u32` values by index.
    gather_u32,
    u32
);
gather!(
    /// Gather `i32` values by index.
    gather_i32,
    i32
);
gather!(
    /// Gather `f32` values by index; the bits are copied, so the result is exact for finite values (WGSL does not
    /// promise that NaN payloads or subnormals survive a copy).
    gather_f32,
    f32
);
