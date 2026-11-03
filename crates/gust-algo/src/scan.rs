//! `inclusive` and `exclusive` prefix sums of a [`GpuVec`] of `u32` (plan section 5.3).
//!
//! Which kernels a scan runs is [`crate::variant::shipped`] (v1 for both, plan section 5.5). A scan is one
//! `dispatch_batch`, with one pooled `GpuVec` per level of totals and one for the result, all taken from the device's
//! buffer pool (the totals are returned to it on drop). Every write goes to the invocation's own element or chunk, or
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
//!   shifted inside the final add-back, which also reads the input: no extra pass. A v1 call whose level 0 would pass
//!   the device's workgroup limit (more than 65 535 spans, 268 million elements) runs v0.
//!
//! The contract (section 5.1): the sum is plain `+`, so a scan is exact and in the domain only while no inclusive
//! prefix, the total included, passes `u32::MAX` (the kernels wrap on the GPU, the debug twin panics, as every `+` of
//! the dialect). The scan of an empty input is an empty vector. An input larger than the device's storage binding
//! limit, or one whose level 0 needs more workgroups than the device allows, is `Error::Device(DeviceLimit)` before
//! any buffer is taken or dispatch is made (no chunking in M7, section 8 C3).

use gust::Independent;
use gust_wgpu::{BufferDispatch, GpuVec};

use crate::{
    Error, limits, scan_kernels,
    schedule::{LANES, ScanSchedule, TileSchedule},
    variant::{Operation, Variant, shipped},
};

/// The inclusive prefix sums: `out[i] = input[0] + ... + input[i]`, a new vector on the device of `input`.
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the device's limits.
pub fn inclusive(input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    inclusive_with(shipped(Operation::Inclusive), input)
}

/// [`inclusive`] with the kernels of `variant` (v1 falls back to v0 for an input its launch cannot take).
///
/// # Errors
///
/// As for [`inclusive`].
pub fn inclusive_with(variant: Variant, input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    scan(variant, input, false)
}

/// The exclusive prefix sums: `out[0] = 0` and `out[i] = input[0] + ... + input[i - 1]`, a new vector on the device of
/// `input`.
///
/// # Errors
///
/// As for [`inclusive`].
pub fn exclusive(input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    exclusive_with(shipped(Operation::Exclusive), input)
}

/// [`exclusive`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`inclusive`].
pub fn exclusive_with(variant: Variant, input: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    scan(variant, input, true)
}

/// One scan: the limit checks, the pooled buffers, one `dispatch_batch` in the order level-0 scan, scans of totals, top
/// scan, add-backs into totals from the top down, final add-back.
fn scan(variant: Variant, input: &GpuVec<u32>, exclusive: bool) -> Result<GpuVec<u32>, Error> {
    if variant == Variant::V1
        && let Some(schedule) = TileSchedule::new(input.len())
        && limits::admits(
            input.device(),
            schedule.workgroups(),
            input.len(),
            size_of::<u32>(),
        )
    {
        return scan_tiles(input, &schedule, exclusive);
    }
    let device = input.device();
    let Some(schedule) = ScanSchedule::new(input.len()) else {
        return Ok(GpuVec::zeroed_on(device, 0)?);
    };
    limits::check(
        device,
        "gust_algo::scan",
        schedule.workgroups(),
        input.len(),
        size_of::<u32>(),
    )?;
    // Every element and every total is written before it is read, so the zero fill is not needed for correctness;
    // `GpuVec` has no public constructor without it (see `reduce`).
    let out = GpuVec::<u32>::zeroed_on(device, input.len())?;
    let lengths = schedule.sums();
    let sums = lengths
        .iter()
        .map(|len| GpuVec::<u32>::zeroed_on(device, *len))
        .collect::<Result<Vec<_>, _>>()?;
    let levels = sums.len();

    let first = scan_kernels::scan_chunks_u32::call(
        lengths[0],
        Independent(input),
        Independent(&out),
        &sums[0],
    );
    let upward: Vec<_> = (1..levels)
        .map(|level| {
            scan_kernels::scan_sums_u32::call(
                lengths[level],
                Independent(&sums[level - 1]),
                &sums[level],
            )
        })
        .collect();
    let top = scan_kernels::scan_top_u32::call(1, Independent(&sums[levels - 1]));
    let downward: Vec<_> = (1..levels)
        .rev()
        .map(|level| {
            scan_kernels::add_back_sums_u32::call(
                lengths[level - 1],
                Independent(&sums[level]),
                &sums[level - 1],
            )
        })
        .collect();
    let (shifted, plain);
    let mut batch = vec![BufferDispatch::from(&first)];
    batch.extend(upward.iter().map(BufferDispatch::from));
    batch.push(BufferDispatch::from(&top));
    batch.extend(downward.iter().map(BufferDispatch::from));
    if exclusive {
        shifted = scan_kernels::add_back_exclusive_u32::call(
            lengths[0],
            &sums[0],
            Independent(input),
            Independent(&out),
        );
        batch.push(BufferDispatch::from(&shifted));
    } else {
        plain = scan_kernels::add_back_inclusive_u32::call(lengths[0], &sums[0], Independent(&out));
        batch.push(BufferDispatch::from(&plain));
    }
    debug_assert_eq!(batch.len(), schedule.dispatches());
    device.dispatch_batch(&batch)?;
    Ok(out)
}

/// One v1 scan of a non-empty input whose launch fits the device: the level-0 tile scan, tile scans of the totals, the
/// top scan, add-backs into the totals from the top down, the final add-back. A single span is one dispatch.
fn scan_tiles(
    input: &GpuVec<u32>,
    schedule: &TileSchedule,
    exclusive: bool,
) -> Result<GpuVec<u32>, Error> {
    let device = input.device();
    let out = GpuVec::<u32>::zeroed_on(device, input.len())?;
    let lengths = schedule.sums();
    let sums = lengths
        .iter()
        .map(|len| GpuVec::<u32>::zeroed_on(device, *len))
        .collect::<Result<Vec<_>, _>>()?;
    let levels = sums.len();
    // A workgroup is 256 invocations: the dispatch of `w` workgroups is `w * 256` invocations.
    let invocations = |workgroups: u32| workgroups * LANES;

    let first_inclusive;
    let first_exclusive;
    let first = if exclusive {
        first_exclusive = scan_kernels::scan_tiles_exclusive_u32::call(
            invocations(lengths[0]),
            Independent(input),
            Independent(&out),
            Independent(&sums[0]),
        );
        BufferDispatch::from(&first_exclusive)
    } else {
        first_inclusive = scan_kernels::scan_tiles_inclusive_u32::call(
            invocations(lengths[0]),
            Independent(input),
            Independent(&out),
            Independent(&sums[0]),
        );
        BufferDispatch::from(&first_inclusive)
    };
    let mut batch = vec![first];
    if lengths[0] > 1 {
        let upward: Vec<_> = (1..levels)
            .map(|level| {
                scan_kernels::scan_tiles_sums_u32::call(
                    invocations(lengths[level]),
                    Independent(&sums[level - 1]),
                    Independent(&sums[level]),
                )
            })
            .collect();
        let top =
            scan_kernels::scan_tiles_top_u32::call(invocations(1), Independent(&sums[levels - 1]));
        let downward: Vec<_> = (1..levels)
            .rev()
            .map(|level| {
                scan_kernels::add_carry_u32::call(
                    invocations(lengths[level]),
                    Independent(&sums[level]),
                    Independent(&sums[level - 1]),
                )
            })
            .collect();
        let last = scan_kernels::add_carry_u32::call(
            invocations(lengths[0]),
            Independent(&sums[0]),
            Independent(&out),
        );
        batch.extend(upward.iter().map(BufferDispatch::from));
        batch.push(BufferDispatch::from(&top));
        batch.extend(downward.iter().map(BufferDispatch::from));
        batch.push(BufferDispatch::from(&last));
        debug_assert_eq!(batch.len(), schedule.dispatches());
        device.dispatch_batch(&batch)?;
    } else {
        debug_assert_eq!(batch.len(), schedule.dispatches());
        device.dispatch_batch(&batch)?;
    }
    Ok(out)
}
