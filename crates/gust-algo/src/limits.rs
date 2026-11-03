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
