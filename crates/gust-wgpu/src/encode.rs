//! Command encoding shared by the staged graph: the ordered upload copy, the readback staging copy and the indirect
//! dispatch pass. Each helper records into a caller's encoder and never submits. An upload's staging buffer is a fresh
//! one that must stay alive until the submission has finished; a readback's staging buffer comes from the device's
//! buffer pool (native, up to the ceiling; `HeadlessDevice::pooled_staging`) and must stay alive until it has been mapped and read. No merging of passes happens here.

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::BufferBinding;
#[cfg(not(target_arch = "wasm32"))]
use crate::{Error, HeadlessDevice, PassTimestamps, dispatch::PreparedDispatch, pool::Lease};

/// Copy `bytes` into `target` as an encoded buffer copy, so it stays ordered among the passes around it. One staging
/// buffer is created per upload; it must outlive the submission. An empty upload records nothing and returns `None`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn encode_upload(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    target: &BufferBinding<'_>,
    bytes: &[u8],
    label: &str,
) -> Option<wgpu::Buffer> {
    let staging = stage_upload(device, bytes, label)?;
    record_upload_copy(encoder, &staging, target.buffer(), bytes.len() as u64);
    Some(staging)
}

/// The first step of an upload: a fresh staging buffer that holds `bytes`. An empty upload makes none. A caller that
/// records the copy later (a batch frame, which collects its copies while it is written) calls this, then
/// [`record_upload_copy`].
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn stage_upload(
    device: &wgpu::Device,
    bytes: &[u8],
    label: &str,
) -> Option<wgpu::Buffer> {
    if bytes.is_empty() {
        return None;
    }
    Some(device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    }))
}

/// The second step of an upload: record the copy of `bytes` bytes of `staging` into `target`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn record_upload_copy(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    target: &wgpu::Buffer,
    bytes: u64,
) {
    encoder.copy_buffer_to_buffer(staging, 0, target, 0, bytes);
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

/// Copy `source` into a fresh mappable staging buffer of `source.byte_len()` bytes: the wasm32 form of the pooled
/// read-back copy, whose pool is native only. An empty source records nothing and returns `None`.
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub(crate) fn encode_readback_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &BufferBinding<'_>,
    label: &str,
) -> Option<wgpu::Buffer> {
    let byte_size = source.byte_len();
    (byte_size > 0).then(|| {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source.buffer(), 0, &staging, 0, byte_size);
        staging
    })
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
