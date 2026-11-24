//! `inclusive` and `exclusive` prefix sums of a [`GpuVec`] of `u32` (plan section 5.3).
//!
//! Which kernels a scan runs is [`crate::variant::shipped`] (v1 for both, plan section 5.5). A scan is one
//! `dispatch_batch`, with one pooled `GpuVec` per level of totals (returned to the device's buffer pool on drop); the
//! result is a new `GpuVec` for `inclusive` and `exclusive`, the caller's own for the `_into` forms. Every write goes to the invocation's own element or chunk, or
//! to a cell of workgroup memory behind barriers: no atomics, no race.
//!
//! - v1, the shipped form ([`TileSchedule`]): a workgroup of 256 lanes owns a span of 4096 elements and scans it in 16
//!   tiles of 256 with eight rounds in workgroup memory (coalesced loads and stores); the span totals are scanned the
//!   same way until at most 4096 are left, one workgroup scans those in place, and the carries go back down, one
//!   add-back per level. One dispatch up to 4096 elements, 3 up to 2^24, 5 above that (`2k + 1` for `k` levels of
//!   totals). `exclusive` is the exclusive form of the level-0 tile scan: no extra pass.
//! - v0, the fallback ([`ScanSchedule`]): the input is cut into chunks of 256 elements, one invocation per chunk, and
//!   the chunk totals are scanned the same way until at most 256 are left; the prefixes then go back down, one add-back
//!   per level: `2k + 1` dispatches, 3 up to 65 536 elements, 5 up to 2^24 and 7 above that. `exclusive` is `inclusive`
//!   shifted inside the final add-back, which also reads the input: no extra pass. A v1 call on a device
//!   with too little workgroup memory runs v0 (a binding past the storage limit is refused by v0 as well); a level 0
//!   past the workgroup limit is split by the host (D211, D208).
//!
//! The contract (section 5.1): the sum is plain `+`, so a scan is exact and in the domain only while no inclusive
//! prefix, the total included, passes `u32::MAX` (the kernels wrap on the GPU, the debug twin panics, as every `+` of
//! the dialect). The scan of an empty input is an empty vector. An input larger than the device's storage binding
//! limit is `Error::Device(DeviceLimit)` before any buffer is taken or dispatch is made (no chunking in M7, section 8
//! C3); a level 0 that needs more workgroups than the device allows is split by the host (D211).
//!
//! `inclusive_into` and `exclusive_into` (m11.fric1b) write into a buffer of the caller's, of the input's length; the
//! allocating forms are wrappers around the same code.

use gust::Independent;
use gust_wgpu::{DeviceHandle, GpuVec};

#[cfg(not(target_arch = "wasm32"))]
use crate::variant::{Operation, shipped};
use crate::{
    Error, limits,
    record::{Operand, Sink, Stage},
    reduce::{check_len, non_empty},
    scan_kernels,
    schedule::{LANES, ScanSchedule, TileSchedule},
    variant::Variant,
};

/// The inclusive prefix sums: `out[i] = input[0] + ... + input[i]`, a new vector on the device of `input`.
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the device's limits.
#[cfg(not(target_arch = "wasm32"))]
pub fn inclusive(input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    inclusive_with(shipped(Operation::Inclusive), input)
}

/// [`inclusive`] with the kernels of `variant` (v1 falls back to v0 for an input its launch cannot take).
///
/// # Errors
///
/// As for [`inclusive`].
#[cfg(not(target_arch = "wasm32"))]
pub fn inclusive_with(variant: Variant, input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    scan_new(variant, input, false)
}

/// The inclusive prefix sums of `input`, written into the caller's `out` (m11.fric1b): `out[i] = input[0] + ... +
/// input[i]`. `out` must hold exactly as many elements as `input` and belong to the same device; every element of it is
/// written, so the buffer can be reused for scan after scan without clearing it, and a caller that scans every step
/// allocates nothing for the result. The kernels, the v1/v0 choice, the storage-limit refusal and the host split of a
/// level 0 past the workgroup limit (D211) are those of [`inclusive`]; this is the one code path of both.
///
/// `out` must not be `input` (the same buffer bound to a read and a write parameter is refused by the dispatch).
///
/// # Errors
///
/// - `Error::Length { expected, actual }` when `out` holds another number of elements than `input` (nothing is
///   dispatched).
/// - `Error::Device(gust_wgpu::Error::AliasedBinding { .. })` when `out` is `input` (nothing is dispatched).
/// - An `out` of another device is not checked up front: the dispatch refuses it, as `Error::Device` (a buffer of
///   another device is reported by `gust_wgpu` as foreign, `Error::ForeignPersistentBuffer`), before anything runs.
///
/// The last two refusals come from the dispatch, so an empty `input` (and `out`), which dispatches nothing, is `Ok`
/// even when `out` is `input` or belongs to another device; the length rule applies to it as to any input.
/// - Otherwise as for [`inclusive`].
#[cfg(not(target_arch = "wasm32"))]
pub fn inclusive_into(input: &GpuVec<u32>, out: &GpuVec<u32>) -> Result<(), Error> {
    inclusive_into_with(shipped(Operation::Inclusive), input, out)
}

/// [`inclusive_into`] with the kernels of `variant` (v1 falls back to v0 for an input its launch cannot take).
///
/// # Errors
///
/// As for [`inclusive_into`].
#[cfg(not(target_arch = "wasm32"))]
pub fn inclusive_into_with(
    variant: Variant,
    input: &GpuVec<u32>,
    out: &GpuVec<u32>,
) -> Result<(), Error> {
    scan(variant, input, out, false)
}

/// The exclusive prefix sums: `out[0] = 0` and `out[i] = input[0] + ... + input[i - 1]`, a new vector on the device of
/// `input`.
///
/// # Errors
///
/// As for [`inclusive`].
#[cfg(not(target_arch = "wasm32"))]
pub fn exclusive(input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    exclusive_with(shipped(Operation::Exclusive), input)
}

/// [`exclusive`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`inclusive`].
#[cfg(not(target_arch = "wasm32"))]
pub fn exclusive_with(variant: Variant, input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    scan_new(variant, input, true)
}

/// The exclusive prefix sums of `input`, written into the caller's `out` (m11.fric1b): `out[0] = 0` and `out[i] =
/// input[0] + ... + input[i - 1]`. The length rule, the reuse of `out`, the variants, the limits and the split of D211
/// are those of [`inclusive_into`].
///
/// # Errors
///
/// As for [`inclusive_into`].
#[cfg(not(target_arch = "wasm32"))]
pub fn exclusive_into(input: &GpuVec<u32>, out: &GpuVec<u32>) -> Result<(), Error> {
    exclusive_into_with(shipped(Operation::Exclusive), input, out)
}

/// [`exclusive_into`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`inclusive_into`].
#[cfg(not(target_arch = "wasm32"))]
pub fn exclusive_into_with(
    variant: Variant,
    input: &GpuVec<u32>,
    out: &GpuVec<u32>,
) -> Result<(), Error> {
    scan(variant, input, out, true)
}

/// A scan into a new vector: the storage limit is checked before the result is allocated (as it always was), then
/// [`scan`] writes into it.
#[cfg(not(target_arch = "wasm32"))]
fn scan_new(variant: Variant, input: &GpuVec<u32>, exclusive: bool) -> Result<GpuVec<u32>, Error> {
    let device = input.device();
    limits::check(device, "gust_algo::scan", input.len(), size_of::<u32>())?;
    let out = GpuVec::<u32>::zeroed_on(device, input.len())?;
    scan(variant, input, &out, exclusive)?;
    Ok(out)
}

/// One scan: the length rule, the limit checks, the pooled buffers, one `dispatch_batch` in the order level-0 scan,
/// scans of totals, top scan, add-backs into totals from the top down, final add-back.
#[cfg(not(target_arch = "wasm32"))]
fn scan(
    variant: Variant,
    input: &GpuVec<u32>,
    out: &GpuVec<u32>,
    exclusive: bool,
) -> Result<(), Error> {
    check_len(input.len(), out.len())?;
    let Some(parts) = Parts::new(input.device(), variant, input.len())? else {
        return Ok(());
    };
    let mut sink = Sink::new();
    parts.plan(&mut sink, input, out, exclusive);
    sink.dispatch(input.device())?;
    Ok(())
}

/// The schedule of one scan and the pooled levels of totals it needs besides its input and result.
pub(crate) struct Parts {
    plan: Plan,
    sums: Vec<GpuVec<u32>>,
}

/// The v1 tile scan, or the v0 chunk scan.
enum Plan {
    Tiles(TileSchedule),
    Chunks(ScanSchedule),
}

impl Parts {
    /// The schedule of a scan of `input` and the pooled totals: v1 when `variant` asks for it and the device admits
    /// it, else v0, which checks the storage binding. `None` for an empty input: nothing to dispatch.
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Variant,
        len: u32,
    ) -> Result<Option<Self>, Error> {
        let take = |lengths: &[u32]| {
            lengths
                .iter()
                .map(|len| GpuVec::<u32>::zeroed_on(device, *len))
                .collect::<Result<Vec<_>, _>>()
        };
        if variant == Variant::V1
            && let Some(schedule) = TileSchedule::new(len)
            && limits::admits(device, len, size_of::<u32>())
        {
            let sums = take(schedule.sums())?;
            return Ok(Some(Self {
                plan: Plan::Tiles(schedule),
                sums,
            }));
        }
        let Some(schedule) = ScanSchedule::new(len) else {
            return Ok(None);
        };
        limits::check(device, "gust_algo::scan", len, size_of::<u32>())?;
        let sums = take(schedule.sums())?;
        Ok(Some(Self {
            plan: Plan::Chunks(schedule),
            sums,
        }))
    }

    /// Append the dispatches of the scan of `input` into `out` (as long as `input`).
    pub(crate) fn plan<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
        exclusive: bool,
    ) {
        let start = sink.len();
        match &self.plan {
            Plan::Tiles(schedule) => {
                self.plan_tiles(sink, input, out, schedule.sums(), exclusive);
                debug_assert_eq!(sink.len() - start, schedule.dispatches());
            }
            Plan::Chunks(schedule) => {
                self.plan_chunks(sink, input, out, schedule.sums(), exclusive);
                debug_assert_eq!(sink.len() - start, schedule.dispatches());
            }
        }
    }

    /// The v0 order: level-0 scan, scans of totals, top scan, add-backs into totals from the top down, final add-back.
    fn plan_chunks<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
        lengths: &[u32],
        exclusive: bool,
    ) {
        let sums = &self.sums;
        let levels = sums.len();
        sink.push(&scan_kernels::scan_chunks_u32::call(
            lengths[0],
            Independent(input),
            Independent(out),
            &sums[0],
        ));
        for level in 1..levels {
            sink.push(&scan_kernels::scan_sums_u32::call(
                lengths[level],
                Independent(&sums[level - 1]),
                &sums[level],
            ));
        }
        sink.push(&scan_kernels::scan_top_u32::call(
            1,
            Independent(&sums[levels - 1]),
        ));
        for level in (1..levels).rev() {
            sink.push(&scan_kernels::add_back_sums_u32::call(
                lengths[level - 1],
                Independent(&sums[level]),
                &sums[level - 1],
            ));
        }
        if exclusive {
            sink.push(&scan_kernels::add_back_exclusive_u32::call(
                lengths[0],
                &sums[0],
                Independent(input),
                Independent(out),
            ));
        } else {
            sink.push(&scan_kernels::add_back_inclusive_u32::call(
                lengths[0],
                &sums[0],
                Independent(out),
            ));
        }
    }

    /// The v1 order: the level-0 tile scan, tile scans of the totals, the top scan, add-backs into the totals from
    /// the top down, the final add-back. A single span is one dispatch.
    fn plan_tiles<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
        lengths: &[u32],
        exclusive: bool,
    ) {
        let sums = &self.sums;
        let levels = sums.len();
        // A workgroup is 256 invocations: the dispatch of `w` workgroups is `w * 256` invocations.
        let invocations = |workgroups: u32| workgroups * LANES;

        if exclusive {
            sink.push(&scan_kernels::scan_tiles_exclusive_u32::call(
                invocations(lengths[0]),
                Independent(input),
                Independent(out),
                Independent(&sums[0]),
            ));
        } else {
            sink.push(&scan_kernels::scan_tiles_inclusive_u32::call(
                invocations(lengths[0]),
                Independent(input),
                Independent(out),
                Independent(&sums[0]),
            ));
        }
        if lengths[0] > 1 {
            for level in 1..levels {
                sink.push(&scan_kernels::scan_tiles_sums_u32::call(
                    invocations(lengths[level]),
                    Independent(&sums[level - 1]),
                    Independent(&sums[level]),
                ));
            }
            sink.push(&scan_kernels::scan_tiles_top_u32::call(
                invocations(1),
                Independent(&sums[levels - 1]),
            ));
            for level in (1..levels).rev() {
                sink.push(&scan_kernels::add_carry_u32::call(
                    invocations(lengths[level]),
                    Independent(&sums[level]),
                    Independent(&sums[level - 1]),
                ));
            }
            sink.push(&scan_kernels::add_carry_u32::call(
                invocations(lengths[0]),
                Independent(&sums[0]),
                Independent(out),
            ));
        }
    }
}

/// The dispatches of one scan of `len` elements, made once and appended over any operands of that length (see
/// [`Sink`]): the schedule and the pooled levels of totals, owned, with no borrow of an input or a result and no
/// lifetime.
///
/// **A plan must outlive the GPU work of every dispatch appended from it**: dropping it returns its totals to the
/// device's pool, and a new owner may then write them while those dispatches still run. The plan may be appended
/// more than once; the appended runs share the totals, so two of them must not be in flight at the same time.
///
/// An empty input has no plan ([`Error::Empty`]; a scan of nothing has nothing to dispatch and nothing to leave
/// stale, and [`Recording`](crate::record::Recording) records nothing for it).
pub struct ScanPlan {
    parts: Parts,
    exclusive: bool,
    len: u32,
}

impl ScanPlan {
    /// The plan of the inclusive (`exclusive == false`) or exclusive scan over `len` elements on `device`, with the
    /// kernels of `variant` (v1 falls back to v0 for an input its launch cannot take).
    ///
    /// # Errors
    ///
    /// `Error::Empty` for `len == 0`, `Error::Device` for a length past the device's storage binding limit or a
    /// failed allocation.
    pub fn new(
        device: &DeviceHandle,
        variant: Variant,
        len: u32,
        exclusive: bool,
    ) -> Result<Self, Error> {
        non_empty(len)?;
        let parts = Parts::new(device, variant, len)?.expect("the input is not empty");
        Ok(Self {
            parts,
            exclusive,
            len,
        })
    }

    /// The number of elements of the input the plan was made for.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// A plan is never for an empty input, so this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Append the dispatches of the scan of `input` into `out` to `sink`.
    ///
    /// # Errors
    ///
    /// `Error::Length` when `input` or `out` does not hold the plan's length; nothing is appended then. An operand
    /// of another device, or an `out` that is `input`, is refused later, by the dispatch.
    pub fn append<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
    ) -> Result<(), Error> {
        check_len(self.len, input.operand_len())?;
        check_len(self.len, out.operand_len())?;
        self.parts.plan(sink, input, out, self.exclusive);
        Ok(())
    }
}

/// A recorded scan of a non-empty `input` into `out`.
pub(crate) struct Scan<'a> {
    plan: ScanPlan,
    input: &'a GpuVec<u32>,
    out: &'a GpuVec<u32>,
}

impl<'a> Scan<'a> {
    /// `Error::Length` when `out` and `input` differ in length; `None` for an empty input (nothing to record).
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Variant,
        input: &'a GpuVec<u32>,
        out: &'a GpuVec<u32>,
        exclusive: bool,
    ) -> Result<Option<Self>, Error> {
        check_len(input.len(), out.len())?;
        if input.is_empty() {
            return Ok(None);
        }
        let plan = ScanPlan::new(device, variant, input.len(), exclusive)?;
        Ok(Some(Self { plan, input, out }))
    }
}

impl Stage for Scan<'_> {
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>) {
        self.plan
            .append(sink, self.input, self.out)
            .expect("the operands were checked when the plan was made");
    }
}
