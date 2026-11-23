//! Batches recorded into a caller's command encoder (m14.encb; Gpu tier): `encode_batch` and `EncodedJob::submitted`,
//! the per-device submission counter, and the one-submission dispatch plus read-back (`submit_batch_read`,
//! `encode_batch_read`). The kernels are hand-written descriptors; every expectation comes from a host loop or from
//! the `dispatch_batch` path, which is the older way to run the same work.

mod common;

use common::{device_handle, device_logged, drive};

use gust::{
    Access, GpuPod, JobStatus, KernelDescriptor, ParameterDescriptor, ParameterKind,
    ResourceBinding, ResourceLayout,
};
use gust_wgpu::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, GpuVec, HeadlessDevice, wgpu,
};

const fn storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "uint",
        kind: ParameterKind::Storage,
        access,
        resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding { group: 0, binding }),
    }
}

/// `add_one(src: RO, dst: RW)`: `dst[i] = src[i] + 1` for `i < dst.len()`.
const ADD_ONE: KernelDescriptor = KernelDescriptor::new(
    "encode_batch_tests",
    "add_one",
    "encode_batch_add_one",
    [64, 1, 1],
    &[
        storage("src", Access::ReadOnly, 0),
        storage("dst", Access::ReadWrite, 1),
    ],
    r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void encode_batch_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1u; }
}
"#,
);

/// `write_even(dst: RW)`: `dst[i] = 7` for even `i < dst.len()`, and touches no odd element: an odd element keeps what
/// the buffer held before the call, so a stale buffer shows.
const WRITE_EVEN: KernelDescriptor = KernelDescriptor::new(
    "encode_batch_tests",
    "write_even",
    "encode_batch_write_even",
    [64, 1, 1],
    &[storage("dst", Access::ReadWrite, 0)],
    r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void encode_batch_write_even(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n && (id.x & 1u) == 0u) { dst[id.x] = 7u; }
}
"#,
);

const N: u32 = 1000;

fn ramp(len: u32) -> Vec<u32> {
    (0..len)
        .map(|index| index.wrapping_mul(2_654_435_761))
        .collect()
}

fn plus(values: &[u32], by: u32) -> Vec<u32> {
    values.iter().map(|value| value.wrapping_add(by)).collect()
}

fn buffer(device: &HeadlessDevice, name: &str, values: &[u32]) -> GpuBuffer<u32> {
    device.create_typed_buffer(name, values, GpuBufferAccess::ReadWrite)
}

fn zeros(device: &HeadlessDevice, name: &str) -> GpuBuffer<u32> {
    buffer(device, name, &vec![0; N as usize])
}

fn read(device: &HeadlessDevice, buffer: &GpuBuffer<u32>) -> Vec<u32> {
    device.read_typed_buffer(buffer).unwrap()
}

fn add_one_bindings<'a>(
    src: &'a GpuBuffer<u32>,
    dst: &'a GpuBuffer<u32>,
) -> [BufferBinding<'a>; 2] {
    [
        BufferBinding::read_only(src),
        BufferBinding::read_write(dst),
    ]
}

fn new_encoder(device: &HeadlessDevice) -> wgpu::CommandEncoder {
    device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
}

/// A pooled buffer of `N` elements holding stale values, handed back to the pool, so the next pooled buffer of the same
/// class is a recycled one.
fn leave_stale_buffer_in_pool(handle: &gust_wgpu::DeviceHandle) {
    let stale = GpuVec::from_slice_on(handle, &vec![100; N as usize]).unwrap();
    drop(stale);
}

/// Test 1, the G6 shape: GUST batch A, a caller copy of its own, GUST batch B, one caller encoder, one `queue.submit`;
/// the results equal the `dispatch_batch` path and a host loop.
#[test]
fn a_caller_copy_between_two_gust_batches_runs_in_one_submission() {
    let device = device_logged("encode_batch G6");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let (b, c, d) = (
        zeros(&device, "b"),
        zeros(&device, "c"),
        zeros(&device, "d"),
    );
    let (first, second) = (add_one_bindings(&a, &b), add_one_bindings(&c, &d));

    let mut encoder = new_encoder(&device);
    let counted = device.gust_submissions();
    let job_a = device
        .encode_batch(&mut encoder, &[BufferDispatch::new(&ADD_ONE, N, &first)])
        .unwrap();
    // The caller's own work between the two batches: b (batch A's output) into c (batch B's input).
    encoder.copy_buffer_to_buffer(
        b.as_wgpu_buffer(),
        0,
        c.as_wgpu_buffer(),
        0,
        4 * u64::from(N),
    );
    let job_b = device
        .encode_batch(&mut encoder, &[BufferDispatch::new(&ADD_ONE, N, &second)])
        .unwrap();
    let index = device.wgpu_queue().submit([encoder.finish()]);
    let (job_a, job_b) = (job_a.submitted(&index), job_b.submitted(&index));
    job_a.wait().unwrap();
    job_b.wait().unwrap();
    assert_eq!(job_a.status().unwrap(), JobStatus::Complete);
    assert_eq!(
        device.gust_submissions(),
        counted,
        "two encoded batches and the caller's submission are none of GUST's submissions"
    );

    // The older path over the same work: dispatch_batch with the copy replaced by binding b as the second input.
    let (b2, d2) = (zeros(&device, "b2"), zeros(&device, "d2"));
    let (first2, second2) = (add_one_bindings(&a, &b2), add_one_bindings(&b2, &d2));
    device
        .dispatch_batch(&[
            BufferDispatch::new(&ADD_ONE, N, &first2),
            BufferDispatch::new(&ADD_ONE, N, &second2),
        ])
        .unwrap();

    assert_eq!(read(&device, &b), plus(&input, 1));
    assert_eq!(
        read(&device, &c),
        plus(&input, 1),
        "the caller's copy ran after batch A"
    );
    assert_eq!(
        read(&device, &d),
        plus(&input, 2),
        "batch B ran after the caller's copy"
    );
    assert_eq!(read(&device, &d), read(&device, &d2));
}

/// Test 2: the encode path makes 0 GUST submissions, `submit_batch` exactly 1, an empty batch 0; and the listed calls
/// that submit on their own account count one each.
#[test]
fn the_counter_counts_gust_submissions_only() {
    let device = device_logged("encode_batch counter");
    let handle = device_handle();
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let (b, d) = (zeros(&device, "b"), zeros(&device, "d"));
    let (first, second) = (add_one_bindings(&a, &b), add_one_bindings(&b, &d));
    let batch = [
        BufferDispatch::new(&ADD_ONE, N, &first),
        BufferDispatch::new(&ADD_ONE, N, &second),
    ];
    let delta = |device: &HeadlessDevice, before: u64| device.gust_submissions() - before;

    // The encode path, two dispatches: 0, also after the caller's own submission.
    let before = device.gust_submissions();
    let mut encoder = new_encoder(&device);
    let encoded = device.encode_batch(&mut encoder, &batch).unwrap();
    assert_eq!(delta(&device, before), 0);
    let index = device.wgpu_queue().submit([encoder.finish()]);
    encoded.submitted(&index).wait().unwrap();
    assert_eq!(delta(&device, before), 0);
    assert_eq!(read(&device, &d), plus(&input, 2));
    assert_eq!(
        delta(&device, before),
        1,
        "only the read-back above is GUST's"
    );

    // submit_batch: exactly 1 for two dispatches.
    let before = device.gust_submissions();
    device.submit_batch(&batch).unwrap().wait().unwrap();
    assert_eq!(delta(&device, before), 1);

    // An empty batch and a batch of empty dispatches record nothing and make no submission.
    let before = device.gust_submissions();
    let empty = device.submit_batch(&[]).unwrap();
    assert_eq!(empty.status().unwrap(), JobStatus::Complete);
    let (no_src, no_dst) = (
        buffer(&device, "no_src", &[]),
        buffer(&device, "no_dst", &[]),
    );
    let no_bindings = add_one_bindings(&no_src, &no_dst);
    let none = [BufferDispatch::new(&ADD_ONE, 0, &no_bindings)];
    let nothing = device.submit_batch(&none).unwrap();
    assert_eq!(nothing.status().unwrap(), JobStatus::Complete);
    assert_eq!(delta(&device, before), 0);

    // Calls that submit on their own account: one each.
    let before = device.gust_submissions();
    device.dispatch_buffers(&ADD_ONE, N, &first).unwrap();
    assert_eq!(delta(&device, before), 1, "dispatch_buffers");
    let before = device.gust_submissions();
    device.dispatch_batch_timed(&batch).unwrap();
    assert_eq!(delta(&device, before), 1, "dispatch_batch_timed");
    let before = device.gust_submissions();
    device.write_typed_buffer_timed(&a, &input).unwrap();
    assert_eq!(delta(&device, before), 1, "write_typed_buffer_timed");
    let before = device.gust_submissions();
    read(&device, &a);
    assert_eq!(delta(&device, before), 1, "read_typed_buffer");
    let vec = GpuVec::from_slice_on(&handle, &input).unwrap();
    let before = handle.gust_submissions();
    assert_eq!(vec.to_vec().unwrap(), input);
    assert_eq!(delta(&handle, before), 1, "GpuVec::to_vec");
    // A recycled zeroed buffer: the clear is a submission of its own.
    drop(vec);
    let before = handle.gust_submissions();
    let zeroed = GpuVec::<u32>::zeroed_on(&handle, N).unwrap();
    assert_eq!(
        delta(&handle, before),
        1,
        "a recycled zeroed buffer's clear"
    );
    drop(zeroed);
}

/// The wrapper is the only submit call of the crate (any receiver, any path): a site that bypasses it would make the
/// counter wrong.
#[test]
fn no_source_file_submits_except_through_the_counting_wrapper() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let needles = [[".sub", "mit("].concat(), ["::sub", "mit("].concat()];
    let mut sites = Vec::new();
    let mut folders = vec![root];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(folder).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                folders.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                // Comment lines (doc examples) are not submissions; rustfmt may break a call chain across lines.
                let code: String = text
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("//"))
                    .collect();
                let squeezed: String = code.split_whitespace().collect();
                for _ in needles
                    .iter()
                    .flat_map(|needle| squeezed.matches(needle.as_str()))
                {
                    sites.push(path.file_name().unwrap().to_string_lossy().into_owned());
                }
            }
        }
    }
    assert_eq!(
        sites,
        ["accessors.rs"],
        "every `.submit(` call of gust-wgpu goes through `HeadlessDevice::submit_counted`, which is in accessors.rs"
    );
}

/// Test 7: a batch whose second dispatch is refused at preparation leaves the encoder untouched: nothing of the first
/// dispatch is recorded, so submitting the encoder changes no buffer, and the encoder is still usable.
#[test]
fn a_batch_refused_at_preparation_leaves_the_callers_encoder_untouched() {
    let device = device_logged("encode_batch refusal");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let (b, d) = (zeros(&device, "b"), zeros(&device, "d"));
    let short = buffer(&device, "short", &[1, 2, 3]);
    let first = add_one_bindings(&a, &b);
    // The second dispatch binds a 3-element destination for N elements: refused with `BufferLength`.
    let refused = add_one_bindings(&a, &short);
    let mut encoder = new_encoder(&device);
    let counted = device.gust_submissions();
    let error = device
        .encode_batch(
            &mut encoder,
            &[
                BufferDispatch::new(&ADD_ONE, N, &first),
                BufferDispatch::new(&ADD_ONE, N, &refused),
            ],
        )
        .expect_err("the second dispatch is refused");
    match error {
        Error::BufferLength {
            expected, actual, ..
        } => assert_eq!((expected, actual), (N as usize, 3)),
        other => panic!("expected BufferLength, got {other:?}"),
    }
    // The same encoder takes a good batch afterwards, and the refused batch left no pass behind it.
    let good = add_one_bindings(&a, &d);
    let encoded = device
        .encode_batch(&mut encoder, &[BufferDispatch::new(&ADD_ONE, N, &good)])
        .unwrap();
    let index = device.wgpu_queue().submit([encoder.finish()]);
    encoded.submitted(&index).wait().unwrap();
    assert_eq!(
        read(&device, &b),
        vec![0; N as usize],
        "the refused batch's first dispatch was never recorded"
    );
    assert_eq!(read(&device, &d), plus(&input, 1));
    assert_eq!(
        device.gust_submissions() - counted,
        2,
        "the two read-backs only"
    );
}

/// An empty batch records nothing: its job is complete at once, whatever index it is given.
#[test]
fn an_empty_encoded_batch_has_a_job_that_is_complete_at_once() {
    let device = device_logged("encode_batch empty");
    let mut encoder = new_encoder(&device);
    let encoded = device.encode_batch(&mut encoder, &[]).unwrap();
    let index = device.wgpu_queue().submit([encoder.finish()]);
    let job = encoded.submitted(&index);
    assert_eq!(job.status().unwrap(), JobStatus::Complete);
    job.wait().unwrap();
}

/// The ordering paragraph of `encode_batch`: a `queue.write_buffer` queued after the batch was recorded and before the
/// caller's submission is applied before the encoder's commands, so the batch sees it.
#[test]
fn a_write_buffer_before_the_submission_is_seen_by_a_batch_recorded_earlier() {
    let device = device_logged("encode_batch write order");
    let old = ramp(N);
    let new: Vec<u32> = old.iter().map(|value| value ^ 0xA5A5_A5A5).collect();
    let src = buffer(&device, "src", &old);
    let dst = zeros(&device, "dst");
    let bindings = add_one_bindings(&src, &dst);
    let mut encoder = new_encoder(&device);
    let encoded = device
        .encode_batch(&mut encoder, &[BufferDispatch::new(&ADD_ONE, N, &bindings)])
        .unwrap();
    device
        .wgpu_queue()
        .write_buffer(src.as_wgpu_buffer(), 0, gust::pod_slice_as_bytes(&new));
    let index = device.wgpu_queue().submit([encoder.finish()]);
    encoded.submitted(&index).wait().unwrap();
    assert_eq!(read(&device, &dst), plus(&new, 1));
}

/// What `submit_batch_read` must produce for `WRITE_EVEN` over a recycled buffer that held `100` everywhere: the
/// batch's 7 at even positions and the clear's 0 at odd ones.
fn even_sevens() -> Vec<u32> {
    (0..N)
        .map(|index| if index % 2 == 0 { 7 } else { 0 })
        .collect()
}

/// Item 6: a typed submit plus a read-back of a recycled, zeroed output is 3 submissions on the old path and 1 on
/// `submit_batch_read`, with the same values.
#[test]
fn submit_batch_read_is_one_submission_where_the_old_path_is_three() {
    let handle = device_handle();
    let device: &HeadlessDevice = &handle;
    eprintln!("encode_batch read adapter: {}", device.adapter_info().name);

    leave_stale_buffer_in_pool(&handle);
    let before = device.gust_submissions();
    let out = GpuVec::<u32>::zeroed_on(&handle, N).unwrap();
    let bindings = [BufferBinding::read_write(out.as_buffer())];
    device
        .submit_batch(&[BufferDispatch::new(&WRITE_EVEN, N, &bindings)])
        .unwrap()
        .wait()
        .unwrap();
    let old = out.to_vec().unwrap();
    assert_eq!(
        device.gust_submissions() - before,
        3,
        "clear, dispatch, read-back"
    );
    assert_eq!(old, even_sevens());
    drop(out);

    leave_stale_buffer_in_pool(&handle);
    let before = device.gust_submissions();
    let out = device.uncleared_buffer::<u32>(N as usize).unwrap();
    assert_eq!(
        device.gust_submissions() - before,
        0,
        "uncleared_buffer submits nothing"
    );
    let bindings = [BufferBinding::read_write(&out)];
    let pending = device
        .submit_batch_read(
            &[BufferDispatch::new(&WRITE_EVEN, N, &bindings)],
            &bindings,
            &out,
        )
        .unwrap();
    assert_eq!(
        device.gust_submissions() - before,
        1,
        "clear, dispatch and copy in one"
    );
    assert_eq!(pending.wait(device).unwrap(), old);
    assert_eq!(
        device.gust_submissions() - before,
        1,
        "waiting submits nothing"
    );
}

/// The same call awaited instead of blocked on, and an error that submits nothing.
#[test]
fn submit_batch_read_is_awaitable_and_refuses_before_recording() {
    let handle = device_handle();
    let device: &HeadlessDevice = &handle;
    eprintln!(
        "encode_batch read async adapter: {}",
        device.adapter_info().name
    );

    leave_stale_buffer_in_pool(&handle);
    let out = device.uncleared_buffer::<u32>(N as usize).unwrap();
    let bindings = [BufferBinding::read_write(&out)];
    let before = device.gust_submissions();
    let pending = device
        .submit_batch_read(
            &[BufferDispatch::new(&WRITE_EVEN, N, &bindings)],
            &bindings,
            &out,
        )
        .unwrap();
    assert_eq!(drive(pending.finish(device)).unwrap(), even_sevens());
    assert_eq!(device.gust_submissions() - before, 1);

    // A read-only buffer cannot be cleared, and a clear that is refused makes no submission and no read-back.
    let read_only = device.create_typed_buffer("ro", &[1u32, 2, 3], GpuBufferAccess::ReadOnly);
    let before = device.gust_submissions();
    let refused = device.submit_batch_read(&[], &[BufferBinding::read_only(&read_only)], &out);
    assert!(
        matches!(refused, Err(Error::PersistentBufferAccess { .. })),
        "got {refused:?}"
    );
    // A read of a read-only buffer is refused as any read-back of one is.
    let refused = device.submit_batch_read(&[], &[], &read_only);
    assert!(
        matches!(refused, Err(Error::PersistentBufferNotReadable)),
        "got {refused:?}"
    );
    assert_eq!(device.gust_submissions(), before);
}

/// Nothing to record (no work, no clear, an empty buffer to read) makes no submission and reads back nothing.
#[test]
fn submit_batch_read_with_nothing_to_record_submits_nothing() {
    let handle = device_handle();
    let device: &HeadlessDevice = &handle;
    let empty = device.uncleared_buffer::<u32>(0).unwrap();
    let before = device.gust_submissions();
    let pending = device.submit_batch_read(&[], &[], &empty).unwrap();
    assert!(pending.wait(device).unwrap().is_empty());
    assert_eq!(device.gust_submissions(), before);
}

/// The encode form of item 6: clear, batch and copy in the caller's encoder next to a copy of the caller's own; the
/// caller's one submission carries all of it; GUST makes none.
#[test]
fn encode_batch_read_records_into_the_callers_encoder_and_submits_nothing() {
    let handle = device_handle();
    let device: &HeadlessDevice = &handle;
    eprintln!(
        "encode_batch read encode adapter: {}",
        device.adapter_info().name
    );

    let input = ramp(N);
    let src = buffer(device, "src", &input);
    let mirror = zeros(device, "mirror");
    leave_stale_buffer_in_pool(&handle);
    let out = device.uncleared_buffer::<u32>(N as usize).unwrap();
    let bindings = [BufferBinding::read_write(&out)];

    let before = device.gust_submissions();
    let mut encoder = new_encoder(device);
    // The caller's own copy goes first, then GUST's clear, batch and read-back copy.
    encoder.copy_buffer_to_buffer(
        src.as_wgpu_buffer(),
        0,
        mirror.as_wgpu_buffer(),
        0,
        4 * u64::from(N),
    );
    let encoded = device
        .encode_batch_read(
            &mut encoder,
            &[BufferDispatch::new(&WRITE_EVEN, N, &bindings)],
            &bindings,
            &out,
        )
        .unwrap();
    let index = device.wgpu_queue().submit([encoder.finish()]);
    let pending = encoded.submitted(&index);
    assert_eq!(pending.wait(device).unwrap(), even_sevens());
    assert_eq!(
        device.gust_submissions(),
        before,
        "the caller's one submission is not GUST's"
    );
    assert_eq!(read(device, &mirror), input);
}

/// A clear binding or a read buffer of another device is refused before anything is recorded or submitted.
#[test]
fn submit_batch_read_refuses_a_buffer_of_another_device_before_recording() {
    let (handle, other) = (device_handle(), device_handle());
    let device: &HeadlessDevice = &handle;
    let own = device.uncleared_buffer::<u32>(4).unwrap();
    let foreign = other.uncleared_buffer::<u32>(4).unwrap();
    let before = device.gust_submissions();
    let refused = device.submit_batch_read(&[], &[BufferBinding::read_write(&foreign)], &own);
    assert!(
        matches!(refused, Err(Error::ForeignPersistentBuffer)),
        "got {refused:?}"
    );
    let refused = device.submit_batch_read(&[], &[], &foreign);
    assert!(
        matches!(refused, Err(Error::ForeignPersistentBuffer)),
        "got {refused:?}"
    );
    assert_eq!(device.gust_submissions(), before);
}

/// A dropped `EncodedRead` retires its staging buffer: the caller's encoder still copies into it, so it must not go
/// back to the pool. The encoder still submits, and the next read-back of the same class is a pool miss, not a hit
/// on the retired buffer.
#[test]
fn a_dropped_encoded_read_retires_its_staging_buffer() {
    let device = device_logged("encode_batch dropped read");
    let input = ramp(N);
    let source = buffer(&device, "source", &input);
    let before = device.buffer_pool_stats();

    let mut encoder = new_encoder(&device);
    let encoded = device
        .encode_batch_read(&mut encoder, &[], &[], &source)
        .unwrap();
    drop(encoded);
    let after_drop = device.buffer_pool_stats();
    assert_eq!(
        after_drop.retired,
        before.retired + 1,
        "the buffer is retired"
    );
    assert_eq!(after_drop.returned, before.returned, "and not returned");
    assert_eq!(after_drop.free_buffers, 0, "so it is not idle");
    assert_eq!(
        after_drop.outstanding, before.outstanding,
        "and not outstanding"
    );

    // The commands that still copy into the retired buffer run without a validation error.
    device.wgpu_queue().submit([encoder.finish()]);
    device
        .wgpu_device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    // The same class again is a fresh allocation: the retired buffer is not in the pool.
    let values = device
        .submit_batch_read(&[], &[], &source)
        .unwrap()
        .wait(&device)
        .unwrap();
    assert_eq!(values, input);
    let end = device.buffer_pool_stats();
    assert_eq!(end.misses, after_drop.misses + 1, "a miss");
    assert_eq!(end.hits, after_drop.hits, "not a hit");
}

/// `PendingRead::finish` and `wait` refuse a device other than the one that made the read-back.
#[test]
fn a_pending_read_refuses_another_device() {
    let device = device_logged("encode_batch pending read");
    let other = device_logged("encode_batch other device");
    let source = buffer(&device, "source", &ramp(N));

    let pending = device.submit_batch_read(&[], &[], &source).unwrap();
    let refused = pending.wait(&other);
    assert!(
        matches!(refused, Err(Error::ForeignPersistentBuffer)),
        "got {refused:?}"
    );

    let pending = device.submit_batch_read(&[], &[], &source).unwrap();
    let refused = drive(pending.finish(&other));
    assert!(
        matches!(refused, Err(Error::ForeignPersistentBuffer)),
        "got {refused:?}"
    );

    // The right device still reads it.
    let pending = device.submit_batch_read(&[], &[], &source).unwrap();
    assert_eq!(pending.wait(&device).unwrap(), ramp(N));
}
