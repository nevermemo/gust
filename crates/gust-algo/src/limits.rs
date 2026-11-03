//! The launch limits `reduce` and `scan` check before they take a buffer (plan section 8, C3: no chunking in M7).

use gust_wgpu::HeadlessDevice;

/// An input past the storage binding limit, or a first dispatch past the workgroup limit, is a `DeviceLimit` error
/// before a buffer is taken or anything is compiled or dispatched (the backend would refuse the dispatch too, but only
/// after the intermediate buffers were allocated). The names are the ones the backend reports. `workgroups` is the
/// largest workgroup count of any dispatch of the call, `len` the length of the longest binding.
pub(crate) fn check(
    device: &HeadlessDevice,
    kernel: &str,
    workgroups: u32,
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
    let ceiling = u64::from(limits.max_compute_workgroups_per_dimension);
    if u64::from(workgroups) > ceiling {
        return Err(device_limit(
            kernel,
            "max_compute_workgroups_per_dimension",
            u64::from(workgroups),
            ceiling,
        ));
    }
    Ok(())
}

/// The workgroup memory the v1 kernels declare, in bytes: the largest of them is the scan tile kernels, three arrays of
/// 256 `u32` (the reduce blocks use one array of at most 256 four-byte elements).
pub(crate) const V1_SHARED_BYTES: u32 = 3 * 256 * 4;

/// Whether [`check`] would accept the same arguments and the device offers the workgroup memory of the v1 kernels
/// ([`V1_SHARED_BYTES`]): the test of a v1 schedule, which runs only inside the limits and otherwise leaves the call
/// (and its `DeviceLimit` error) to v0, whose launch is the smaller one and uses no workgroup memory.
pub(crate) fn admits(
    device: &HeadlessDevice,
    workgroups: u32,
    len: u32,
    element_bytes: usize,
) -> bool {
    device.limits().max_compute_workgroup_storage_size >= V1_SHARED_BYTES
        && check(device, "", workgroups, len, element_bytes).is_ok()
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
