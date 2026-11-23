//! Command encoding shared by the staged graph: the ordered upload copy, the readback staging copy and the indirect
//! dispatch pass. Each helper records into a caller's encoder and returns what the caller must keep alive until the
//! submission has finished. Moved unchanged out of `graph.rs`; no pooling and no merging of passes happens here.

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{BufferBinding, PassTimestamps, dispatch::PreparedDispatch};

/// Copy `bytes` into `target` as an encoded buffer copy, so it stays ordered among the passes around it. One staging
/// buffer is created per upload; it must outlive the submission. An empty upload records nothing and returns `None`.
pub(crate) fn encode_upload(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    target: &BufferBinding<'_>,
    bytes: &[u8],
    label: &str,
) -> Option<wgpu::Buffer> {
    if bytes.is_empty() {
        return None;
    }
    let staging = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&staging, 0, target.buffer(), 0, bytes.len() as u64);
    Some(staging)
}

/// Copy `source` into a fresh mappable staging buffer. An empty source records nothing and returns `None`.
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
