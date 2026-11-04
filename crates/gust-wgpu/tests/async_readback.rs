//! The completion model, natively (roadmap M4, m4.g1, R4.9): `read_typed_buffer_async` awaited through a minimal
//! executor, and the blocking readback on a device that is still busy.
//!
//! On native nothing polls the device in the background, so the async future completes only while something does. The
//! executor below re-polls the future on a timer and ignores wakes; every poll of the future polls the device once
//! (`wgpu::PollType::Poll`). The blocking wrapper waits with `poll(Wait)` and then waits for the same future with a
//! parking waiter that polls the device again every 10 ms (`completion::blocking`).
//!
//! The slow kernel keeps the device busy for tens of milliseconds, so the readback queued behind it is normally still
//! pending when the first poll runs; the async test retries up to three times and requires one attempt with two or more
//! polls. A readback that returned before the mapping completed fails the value check (a hang is caught by a
//! timeout). Expectations come from a host loop, not from a CPU twin. The blocking test checks values on a busy device;
//! it does not prove the `poll(Wait)`, which only shortens the wait since the parking waiter polls by itself.
//!
//! The Gpu tests drive the future with a no-op waker, so they do not exercise the waker path inside the readback; the
//! `Completion` unit tests in `src/completion.rs` cover waking, including from another thread.
//!
//! The step is not affine on purpose: an affine recurrence (`x * a + c`) is folded into a closed form by the shader
//! compiler and ran in 2.5 ms for 2^25 iterations, too short to keep the device busy.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    pin::pin,
    sync::mpsc,
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, SourceOrigin,
};
use gust_wgpu::{BufferBinding, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice};

const COUNT: usize = 70;
/// Elements of the busy buffer; the host loop below runs once per element and iteration.
const BUSY_COUNT: usize = 8;
/// Iterations of the step each thread runs (about 40 ms on the development GPU).
const STEPS: u32 = 1 << 23;
const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "state",
    rust_type: "uint",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

// The shader loop bound below (8388608u) is `STEPS`; the assertion ties them.
const _: () = assert!(STEPS == 8_388_608);

/// `state[i]` after `STEPS` applications of `step` (below) to its seed.
const SLOW: KernelDescriptor = KernelDescriptor {
    module: "async_readback_tests",
    name: "slow_steps",
    entry_point: "async_readback_slow",
    shape: gust::Shape::Grid,
    compatible_shapes: &[gust::Shape::Grid],
    workgroup_size: [64, 1, 1],
    parameters: PARAMETERS,
    slang_source: r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> state;
[shader("compute")] [numthreads(64, 1, 1)]
void async_readback_slow(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; state.GetDimensions(n, stride);
    if (id.x < n)
    {
        uint x = state[id.x];
        for (uint i = 0; i < 8388608u; i++) { x = x * 1664525u + 1013904223u; x ^= x >> 13; }
        state[id.x] = x;
    }
}
"#,
    source: SourceOrigin::UNKNOWN,
    artifacts: &[],
};

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// The host statement of the kernel's loop body (wrapping u32 arithmetic).
fn step(x: u32) -> u32 {
    let x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x ^ (x >> 13)
}

fn seeds() -> Vec<u32> {
    (0..BUSY_COUNT as u32).map(|index| index * 7 + 1).collect()
}

fn expected() -> Vec<u32> {
    seeds()
        .into_iter()
        .map(|seed| (0..STEPS).fold(seed, |x, _| step(x)))
        .collect()
}

/// Poll `future` every millisecond until it is ready, returning its output and how often it was polled. Each poll of
/// a `read_*_async` future polls the device once; the deadline turns a future that never completes into a failure.
fn drive<F: Future>(future: F) -> (F::Output, u32) {
    // A waker that does nothing: this executor does not wait for wakes.
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut polls = 0;
    loop {
        polls += 1;
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return (output, polls);
        }
        assert!(
            Instant::now() < deadline,
            "the readback future was still pending after {polls} polls"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// Run `work` on a thread and fail, instead of hanging, when it does not finish in time.
fn within<T: Send + 'static>(seconds: u64, work: impl FnOnce() -> T + Send + 'static) -> T {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(work());
    });
    receiver
        .recv_timeout(Duration::from_secs(seconds))
        .expect("the readback did not finish in time")
}

fn busy_buffer(device: &HeadlessDevice) -> GpuBuffer<u32> {
    device.create_typed_buffer("state", &seeds(), GpuBufferAccess::ReadWrite)
}

#[test]
fn an_async_readback_returns_the_uploaded_values() {
    let device = device();
    let values: Vec<f32> = (0..COUNT).map(|index| index as f32 + 0.5).collect();
    let buffer = device.create_typed_buffer("values", &values, GpuBufferAccess::ReadWrite);
    let (read, polls) = drive(device.read_typed_buffer_async(&buffer));
    assert_eq!(read.unwrap(), values);
    assert!(polls >= 1);
    assert_eq!(device.read_typed_buffer(&buffer).unwrap(), values);
}

#[test]
fn async_readback_validates_before_waiting() {
    let device = device();
    let empty = device.create_typed_buffer::<f32>("empty", &[], GpuBufferAccess::ReadWrite);
    let (read, polls) = drive(device.read_typed_buffer_async(&empty));
    assert_eq!(read.unwrap(), Vec::<f32>::new());
    assert_eq!(polls, 1, "an empty buffer has nothing to wait for");

    let read_only = device.create_typed_buffer("input", &[1.0f32], GpuBufferAccess::ReadOnly);
    let (read, _) = drive(device.read_typed_buffer_async(&read_only));
    assert!(matches!(read, Err(Error::PersistentBufferNotReadable)));

    let other = HeadlessDevice::new().unwrap();
    let foreign = other.create_typed_buffer("foreign", &[1.0f32], GpuBufferAccess::ReadWrite);
    let (read, _) = drive(device.read_typed_buffer_async(&foreign));
    assert!(matches!(read, Err(Error::ForeignPersistentBuffer)));
}

/// One dispatch on a scratch buffer, waited for, so the pipeline is compiled and cached before a timed test submits it.
fn warm(device: &HeadlessDevice) {
    let scratch =
        device.create_typed_buffer("warm", &[1u32; BUSY_COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &SLOW,
            BUSY_COUNT as u32,
            &[BufferBinding::read_write(&scratch)],
        )
        .unwrap();
}

#[test]
fn an_async_readback_queued_behind_a_busy_kernel_waits_for_the_mapping() {
    let device = device();
    warm(&device);
    // The first poll of a readback queued behind a 40 ms kernel is normally pending, but a test thread that is
    // descheduled for the length of the kernel polls late and sees it ready at once (1 poll, seen once in a Gpu gate round
    // on a loaded machine). Three attempts must show the pending-then-ready path at least once.
    let mut polls_per_attempt = Vec::new();
    for _ in 0..3 {
        let buffer = busy_buffer(&device);
        // Not waited on: the readback below is queued behind the kernel on the same queue.
        let job = device
            .submit_buffers(
                &SLOW,
                BUSY_COUNT as u32,
                &[BufferBinding::read_write(&buffer)],
            )
            .unwrap();
        let started = Instant::now();
        let (read, polls) = drive(device.read_typed_buffer_async(&buffer));
        eprintln!(
            "async readback behind the slow kernel: {polls} polls, {:?}",
            started.elapsed()
        );
        assert_eq!(read.unwrap(), expected());
        job.wait().unwrap();
        polls_per_attempt.push(polls);
        if polls >= 2 {
            break;
        }
    }
    // A software adapter (lavapipe) runs the kernel on the polling thread, so the first poll returns only when the
    // kernel is done and the pending path cannot be observed (an adapter property, measured in CI: 1 poll in 3 of 3
    // attempts, about 136 ms each). Only this assertion is skipped there; the values above are checked on every
    // adapter. Hardware adapters keep it. Recorded as a software-tier limitation in docs/development/VALIDATION.md
    // (D52): lavapipe does not prove the pending path.
    let info = device.adapter_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        eprintln!(
            "pending-path assertion skipped: adapter {:?} is a software adapter (device_type Cpu) whose first poll \
             blocks until the kernel finishes; polls per attempt: {polls_per_attempt:?}",
            info.name
        );
        return;
    }
    eprintln!(
        "pending-path assertion runs: adapter {:?} (device_type {:?})",
        info.name, info.device_type
    );
    assert!(
        polls_per_attempt.iter().any(|polls| *polls >= 2),
        "no attempt saw the readback pending on its first poll; polls per attempt: {polls_per_attempt:?}"
    );
}

#[test]
fn the_blocking_readback_returns_the_values_of_a_busy_device() {
    // The readback is queued behind a kernel that is still running, so the values are right only if the wrapper waits
    // for the copy and the mapping. (`poll(Wait)` makes that wait short; the waiter would also finish without it, so
    // this test does not discriminate the wait itself.) The thread and its timeout turn a hang into a failure.
    let values = within(60, || {
        let device = device();
        warm(&device);
        let buffer = busy_buffer(&device);
        let _job = device
            .submit_buffers(
                &SLOW,
                BUSY_COUNT as u32,
                &[BufferBinding::read_write(&buffer)],
            )
            .unwrap();
        device.read_typed_buffer(&buffer)
    });
    assert_eq!(values.unwrap(), expected());
}
