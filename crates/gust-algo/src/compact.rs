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
//!   on is `0`** (the buffer is created zeroed and only the first `count` slots are written). The tail is not a
//!   sentinel: `0` is also a valid index, so a consumer must use the count, which is what the gathers do. The indices
//!   come from the exclusive scan of the mask: output slot `scan[i]` belongs to element `i` alone, so the scatter
//!   `indices[scan[i]] = i` has one writer per slot.
//! * The **gathers** compute `out[j] = values[indices[j]]` for `j < count[0]`. The output is `indices.len()` long,
//!   which is `flags.len()` when `indices` came from [`indices`], and is zero from `count` on, like the indices. The
//!   bound is guarded in the kernel: a slot at or past `count`, and an index that is not below `values.len()`, stay
//!   `0`. To use only the dense prefix, hand `count` (still on the device) to the next kernel.
//!
//! An empty input is not an error: the count is `[0]`, `indices` is empty, and a gather of empty indices is empty.
//! Inputs past the device's limits (the storage binding limit for any buffer, the workgroup limit for the first
//! dispatch) are `Error::Device(DeviceLimit)` before any buffer is taken or dispatch is made.
//!
//! # A gather of structs
//!
//! There is no generic `gather<T>`: generic GPU functions are not part of the dialect before M9's traits (plan section
//! 5.4). For a struct, write the one-line kernel yourself, against the `indices` and `count` of [`indices`]; the block
//! shape of `tests/fixtures/compact.rs` is the one to copy, and `src/tests/struct_gather.rs` holds a worked, tested
//! example (`out[j] = particles[indices[j]]` for a struct of a `u32` and an `f32`).

use gust::{GpuPod, Independent};
use gust_wgpu::{BufferDispatch, GpuVec, HeadlessDevice};

use crate::{Error, compact_kernels, limits, reduce, scan, schedule::Pass};

/// The name the limit errors carry.
const KERNEL: &str = "gust_algo::compact";

/// The number of set (non-zero) flags, as a one-element `GpuVec<u32>` on the device of `flags`: no readback. An empty
/// input counts `0`.
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the device's limits.
pub fn count(flags: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    if flags.is_empty() {
        return Ok(GpuVec::zeroed_on(flags.device(), 1)?);
    }
    reduce::sum_on_device(&mask_of(flags)?)
}

/// The positions of the set (non-zero) flags and their number: `(indices, count)`, both on the device of `flags`, no
/// readback. `indices` is `flags.len()` long; its first `count` elements are the increasing positions of the set flags
/// and the rest are `0` (see the module documentation). `count` has one element.
///
/// # Errors
///
/// As for [`count`].
pub fn indices(flags: &GpuVec<u32>) -> Result<(GpuVec<u32>, GpuVec<u32>), Error> {
    let device = flags.device();
    if flags.is_empty() {
        return Ok((GpuVec::zeroed_on(device, 0)?, GpuVec::zeroed_on(device, 1)?));
    }
    let mask = mask_of(flags)?;
    let positions = scan::exclusive(&mask)?;
    let total = reduce::sum_on_device(&mask)?;
    let indices = GpuVec::<u32>::zeroed_on(device, flags.len())?;
    let call = compact_kernels::scatter_indices_u32::call(
        Pass::over(flags.len()).partials(),
        Independent(&mask),
        Independent(&positions),
        Independent(&indices),
    );
    device.dispatch_batch(&[BufferDispatch::from(&call)])?;
    Ok((indices, total))
}

/// The 0/1 mask of `flags` (1 where the flag is not zero), after the limit checks.
fn mask_of(flags: &GpuVec<u32>) -> Result<GpuVec<u32>, Error> {
    let device = flags.device();
    let pass = Pass::over(flags.len());
    limits::check(
        device,
        KERNEL,
        pass.workgroups,
        flags.len(),
        size_of::<u32>(),
    )?;
    // Every element is written, so the zero fill is not needed for correctness; `GpuVec` has no public constructor
    // without it (see `reduce`).
    let mask = GpuVec::<u32>::zeroed_on(device, flags.len())?;
    let call = compact_kernels::flags_to_mask_u32::call(
        pass.partials(),
        Independent(flags),
        Independent(&mask),
    );
    device.dispatch_batch(&[BufferDispatch::from(&call)])?;
    Ok(mask)
}

/// The shared frame of the gathers: the count and device checks, the limit checks, the zeroed output of
/// `indices.len()` elements, which `run` fills (it is given the device, the invocation count and the output).
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
    limits::check(
        device,
        KERNEL,
        pass.workgroups,
        indices.len(),
        size_of::<u32>(),
    )?;
    limits::check(device, KERNEL, 0, values.len(), size_of::<T>())?;
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
                device.dispatch_batch(&[BufferDispatch::from(&call)])
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
    /// Gather `f32` values by index; the bits are copied, so the result is exact.
    gather_f32,
    f32
);
