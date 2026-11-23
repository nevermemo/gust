//! Command encoding shared by the staged graph: the ordered upload copy, the readback staging copy and the indirect
//! dispatch pass. Each helper records into a caller's encoder and never submits. The staging buffers of an upload and
//! of a readback come from the device's buffer pool (`HeadlessDevice::pooled_staging`); an upload's lease must stay
//! alive until the submission has finished, a readback's until it has been mapped and read. No merging of passes
//! happens here.

#[cfg(not(target_arch = "wasm32"))]
use crate::{
    BufferBinding, Error, HeadlessDevice, PassTimestamps, dispatch::PreparedDispatch, pool::Lease,
};

/// Copy `bytes` into `target` as an encoded buffer copy, so it stays ordered among the passes around it. The bytes
/// are written into a pooled staging buffer with `Queue::write_buffer`, which takes effect before the next submission
/// whatever it holds; that is early enough here, because only the copy recorded from the staging buffer reads it, and
/// that copy keeps its place among the passes. The staging lease must outlive the submission. An empty upload records
/// nothing and returns `None`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_upload(
    device: &HeadlessDevice,
    encoder: &mut wgpu::CommandEncoder,
    target: &BufferBinding<'_>,
    bytes: &[u8],
    label: &str,
) -> Result<Option<Lease>, Error> {
    if bytes.is_empty() {
        return Ok(None);
    }
    let staging = device.pooled_staging(
        label,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        bytes.len() as u64,
    )?;
    device.wgpu_queue().write_buffer(staging.buffer(), 0, bytes);
    encoder.copy_buffer_to_buffer(staging.buffer(), 0, target.buffer(), 0, bytes.len() as u64);
    Ok(Some(staging))
}

/// Copy `source` into a pooled mappable staging buffer holding at least `source.byte_len()` bytes (the logical length
/// is what the copy and the later mapping cover, not the pooled allocation's size). An empty source records nothing
/// and returns `None`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_readback_copy(
    device: &HeadlessDevice,
    encoder: &mut wgpu::CommandEncoder,
    source: &BufferBinding<'_>,
    label: &str,
) -> Result<Option<Lease>, Error> {
    let byte_size = source.byte_len();
    if byte_size == 0 {
        return Ok(None);
    }
    let staging = device.pooled_staging(
        label,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        byte_size,
    )?;
    encoder.copy_buffer_to_buffer(source.buffer(), 0, staging.buffer(), 0, byte_size);
    Ok(Some(staging))
}

/// Record one indirect compute pass for `prepared`, its workgroup counts read from `args` on the GPU.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_indirect_pass(
    encoder: &mut wgpu::CommandEncoder,
    prepared: &PreparedDispatch,
    args: &BufferBinding<'_>,
    label: &str,
    timestamps: Option<PassTimestamps<'_>>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: timestamps.map(|stamps| wgpu::ComputePassTimestampWrites {
            query_set: stamps.query_set,
            beginning_of_pass_write_index: stamps.begin,
            end_of_pass_write_index: stamps.end,
        }),
    });
    pass.set_pipeline(prepared.cached_kernel.pipeline.compute());
    pass.set_bind_group(0, &prepared.bind_group, &[]);
    pass.dispatch_workgroups_indirect(args.buffer(), 0);
}
