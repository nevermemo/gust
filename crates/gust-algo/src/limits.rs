//! The launch limits `reduce`, `scan` and `compact` check before they take a buffer (plan section 8, C3: no chunking
//! past one storage binding).
//!
//! The workgroup count is not checked here any more (m11.algolift, D211): a first dispatch past
//! `max_compute_workgroups_per_dimension` is split by the host into pieces (D208, "Launch size" in
//! `docs/SEMANTICS.md`), because every kernel of this crate is a `#[gpu]` module kernel whose descriptor carries a split variant and every
//! launch goes through the batch dispatch of the host. What still refuses is the storage half below; a build without
//! run-time compilation refuses a launch that needs splitting when it is made, until D209.

use gust_wgpu::HeadlessDevice;

/// An input past the storage binding limit is a `DeviceLimit` error before a buffer is taken or anything is compiled
/// or dispatched (the backend would refuse the dispatch too, but only after the intermediate buffers were allocated).
/// The name is the one the backend reports. `len` is the length of the longest binding. The workgroup count of the
/// launch is no part of this check (D211, D208): the host splits a launch past the workgroup limit.
pub(crate) fn check(
    device: &HeadlessDevice,
    kernel: &str,
    len: u32,
    element_bytes: usize,
) -> Result<(), gust_wgpu::Error> {
    let limits = device.limits();
    let bytes = u64::from(len) * element_bytes as u64;
    let binding = limits.max_storage_buffer_binding_size;
    if bytes > binding {
        return Err(device_limit(
            kernel,
            "max_storage_buffer_binding_size",
            bytes,
            binding,
        ));
    }
    Ok(())
}

/// The workgroup memory the v1 kernels declare, in bytes: the largest of them is the scan tile kernels, three arrays of
/// 256 `u32` (the reduce blocks use one array of at most 256 four-byte elements).
pub(crate) const V1_SHARED_BYTES: u32 = 3 * 256 * 4;

/// Whether [`check`] would accept the same arguments and the device offers the workgroup memory of the v1 kernels
/// ([`V1_SHARED_BYTES`]): the test of a v1 schedule, which runs only inside those limits and otherwise leaves the call
/// (and its `DeviceLimit` error) to v0, which uses no workgroup memory (a binding past the storage limit is refused by
/// v0 as well). The number of workgroups does not decide (D211): a v1 launch past the workgroup limit runs, split by
/// the host (D208).
pub(crate) fn admits(device: &HeadlessDevice, len: u32, element_bytes: usize) -> bool {
    device.limits().max_compute_workgroup_storage_size >= V1_SHARED_BYTES
        && check(device, "", len, element_bytes).is_ok()
}

fn device_limit(
    kernel: &str,
    limit: &'static str,
    requested: u64,
    maximum: u64,
) -> gust_wgpu::Error {
    gust_wgpu::Error::DeviceLimit {
        kernel: kernel.to_owned(),
        limit,
        requested,
        maximum,
    }
}
