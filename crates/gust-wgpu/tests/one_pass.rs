//! The plain `grid` dispatches of a batch are recorded in one compute pass (slice m14.onepass; Gpu tier).
//!
//! Two entries are covered: `dispatch_batch` over a list of dispatches, and the frame path of a `gpu_batch!` batch
//! (`BatchDevice::frame_stages`). A pass count is read from `batch_passes_recorded`, the number of compute passes (and
//! render passes) a batch recorded on this thread; a counted stage with the indirect launch records none of its own
//! (its dispatches join the run) and two under `force_pass_per_dispatch`. Every result is
//! held to a host loop, and the batch results also to the CPU twin, so "one pass" cannot hide "wrong bytes": the chains
//! read what the dispatch before them wrote, so a missing ordering between dispatches of one pass shows as wrong data.
//!
//! The per-pass form is a process-wide switch (`force_pass_per_dispatch`), so every test here takes one lock.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};

use common::{device_logged, drive};

use gust::batch::{
    Batch, BatchDevice, CountedLaunch, D1, Error, FrameTiming, Stage, set_counted_launch,
};
use gust::cpu::CpuDevice;
use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, gpu_batch, gpu_task,
};
use gust_wgpu::{
    BufferBinding, BufferDispatch, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    batch_passes_recorded, force_pass_per_dispatch, wgpu,
};

/// Large enough that the dispatches of a chain span many workgroups, so an ordering between them matters.
const N: u32 = 1 << 20;

static LOCK: Mutex<()> = Mutex::new(());

/// Holds the lock and puts back the pass form it found when it drops.
struct Form {
    _guard: MutexGuard<'static, ()>,
    /// The setting the process had when the lock was taken, put back when it drops.
    before: bool,
}

impl Form {
    /// The lock with the pass form as the process has it, never set by a test.
    fn as_found() -> Self {
        let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = force_pass_per_dispatch(false);
        force_pass_per_dispatch(before);
        Self {
            _guard: guard,
            before,
        }
    }

    fn take(per_pass: bool) -> Self {
        let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = force_pass_per_dispatch(per_pass);
        Self {
            _guard: guard,
            before,
        }
    }
}

impl Drop for Form {
    fn drop(&mut self) {
        force_pass_per_dispatch(self.before);
        set_counted_launch(None);
    }
}

/// The passes `work` recorded on this thread.
fn passes_of<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let before = batch_passes_recorded();
    let value = work();
    (value, batch_passes_recorded() - before)
}

// ---- dispatch_batch ----

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
    "one_pass_tests",
    "add_one",
    "one_pass_add_one",
    [64, 1, 1],
    &[
        storage("src", Access::ReadOnly, 0),
        storage("dst", Access::ReadWrite, 1),
    ],
    r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void one_pass_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1u; }
}
"#,
);

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

fn bindings<'a>(src: &'a GpuBuffer<u32>, dst: &'a GpuBuffer<u32>) -> [BufferBinding<'a>; 2] {
    [
        BufferBinding::read_only(src),
        BufferBinding::read_write(dst),
    ]
}

/// A chain of four `add_one` calls: a to b, b to c, c to d, d to e. Each reads what the call before it wrote.
fn chain_in_dispatch_batch(per_pass: bool) -> (Vec<Vec<u32>>, u64) {
    let _form = Form::take(per_pass);
    let device = device_logged("one_pass dispatch_batch chain");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let rest: Vec<GpuBuffer<u32>> = (0..4)
        .map(|index| buffer(&device, &format!("t{index}"), &vec![0; N as usize]))
        .collect();
    let wires = [
        bindings(&a, &rest[0]),
        bindings(&rest[0], &rest[1]),
        bindings(&rest[1], &rest[2]),
        bindings(&rest[2], &rest[3]),
    ];
    let batch: Vec<BufferDispatch<'_>> = wires
        .iter()
        .map(|wire| BufferDispatch::new(&ADD_ONE, N, wire))
        .collect();
    let ((), passes) = passes_of(|| device.dispatch_batch(&batch).unwrap());
    let results = rest
        .iter()
        .map(|buffer| device.read_typed_buffer(buffer).unwrap())
        .collect();
    (results, passes)
}

fn assert_chain(results: &[Vec<u32>]) {
    let input = ramp(N);
    for (step, got) in results.iter().enumerate() {
        assert_eq!(*got, plus(&input, step as u32 + 1), "link {step}");
    }
}

/// Four dependent grid calls in one `dispatch_batch`: one pass, each call sees the stores of the one before it.
/// (Pass count: 1.)
#[test]
fn a_dependent_chain_of_grid_calls_is_one_pass_and_equals_the_host() {
    let (results, passes) = chain_in_dispatch_batch(false);
    assert_eq!(passes, 1, "four grid calls in one pass");
    assert_chain(&results);
}

/// The forced per-pass form gives the same bytes in a pass per call. (Pass count: 4.)
#[test]
fn the_forced_per_pass_form_gives_the_same_bytes_in_a_pass_per_call() {
    let (results, passes) = chain_in_dispatch_batch(true);
    assert_eq!(passes, 4, "a pass per call when forced");
    assert_chain(&results);
}

/// The second call writes the buffer the first one read from, the fourth again: write after read inside one pass.
/// (Pass count: 1.)
#[test]
fn a_call_overwriting_what_an_earlier_call_read_is_ordered_after_it() {
    let _form = Form::take(false);
    let device = device_logged("one_pass ping pong");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let b = buffer(&device, "b", &vec![0; N as usize]);
    let (forward, back) = (bindings(&a, &b), bindings(&b, &a));
    let batch = [
        BufferDispatch::new(&ADD_ONE, N, &forward),
        BufferDispatch::new(&ADD_ONE, N, &back),
        BufferDispatch::new(&ADD_ONE, N, &forward),
        BufferDispatch::new(&ADD_ONE, N, &back),
    ];
    let ((), passes) = passes_of(|| device.dispatch_batch(&batch).unwrap());
    assert_eq!(passes, 1);
    assert_eq!(device.read_typed_buffer(&a).unwrap(), plus(&input, 4));
    assert_eq!(device.read_typed_buffer(&b).unwrap(), plus(&input, 3));
}

/// A timed batch of grid calls is one pass with one GPU time; where the adapter has no timestamps there is none.
/// (Pass count: 1.)
#[test]
fn a_timed_batch_of_grid_calls_is_one_pass() {
    let _form = Form::take(false);
    let device = device_logged("one_pass timed");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let (b, c) = (
        buffer(&device, "b", &vec![0; N as usize]),
        buffer(&device, "c", &vec![0; N as usize]),
    );
    let (first, second) = (bindings(&a, &b), bindings(&b, &c));
    let batch = [
        BufferDispatch::new(&ADD_ONE, N, &first),
        BufferDispatch::new(&ADD_ONE, N, &second),
    ];
    let (timing, passes) = passes_of(|| device.dispatch_batch_timed(&batch).unwrap());
    assert_eq!(passes, 1);
    if let Some(gpu) = timing.gpu_elapsed {
        assert!(gpu > std::time::Duration::ZERO);
    }
    assert_eq!(device.read_typed_buffer(&c).unwrap(), plus(&input, 2));
}

// ---- the frame path ----

#[derive(Clone, Copy)]
struct Six {
    x: Stage<u32, D1>,
    y: [Stage<u32, D1>; 6],
}

fn build_gpu(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Result<(Batch<HeadlessDevice>, Six), Error> {
    gpu_batch!("one-pass-six", on(device); in_flight(1); |b| {
        b.frame_timing(level);
        let x = b.input::<u32>(N);
        let y0 = gpu_task!(N, |i| -> u32 { x[i] + 1u32 });
        let y1 = gpu_task!(N, |i| -> u32 { y0[i] * 3u32 });
        let y2 = gpu_task!(N, |i| -> u32 { y1[i] + 5u32 });
        let y3 = gpu_task!(N, |i| -> u32 { y2[i] * 7u32 });
        let y4 = gpu_task!(N, |i| -> u32 { y3[i] + 11u32 });
        let y5 = gpu_task!(N, |i| -> u32 { y4[i] * 2u32 });
        Six { x, y: [y0, y1, y2, y3, y4, y5] }
    })
}

fn build_cpu(device: &Arc<CpuDevice>) -> Result<(Batch<CpuDevice>, Six), Error> {
    gpu_batch!("one-pass-six", on(device); in_flight(1); |b| {
        let x = b.input::<u32>(N);
        let y0 = gpu_task!(N, |i| -> u32 { x[i] + 1u32 });
        let y1 = gpu_task!(N, |i| -> u32 { y0[i] * 3u32 });
        let y2 = gpu_task!(N, |i| -> u32 { y1[i] + 5u32 });
        let y3 = gpu_task!(N, |i| -> u32 { y2[i] * 7u32 });
        let y4 = gpu_task!(N, |i| -> u32 { y3[i] + 11u32 });
        let y5 = gpu_task!(N, |i| -> u32 { y4[i] * 2u32 });
        Six { x, y: [y0, y1, y2, y3, y4, y5] }
    })
}

/// The host model of the six stages.
fn six_model(x: &[u32]) -> Vec<u32> {
    x.iter()
        .map(|&v| ((((v + 1) * 3 + 5) * 7) + 11) * 2)
        .collect()
}

fn small(seed: u32) -> Vec<u32> {
    small_of(N, seed)
}

fn small_of(len: u32, seed: u32) -> Vec<u32> {
    (0..len).map(|i| (i * 7 + seed) % 1_000).collect()
}

/// The six-stage batch at `level`, run by `submit()` or by `encode()` into the caller's encoder: the last stage's rows
/// and the passes recorded by the frame.
fn run_six(level: FrameTiming, per_pass: bool, encode: bool) -> (Vec<u32>, u64, usize) {
    let _form = Form::take(per_pass);
    run_six_as_set(level, encode)
}

/// [`run_six`] with the pass form left as the caller set it (or as the process has it).
fn run_six_as_set(level: FrameTiming, encode: bool) -> (Vec<u32>, u64, usize) {
    let device = Arc::new(device_logged("one_pass six stages"));
    let (mut batch, h) = build_gpu(&device, level).unwrap();
    let x = small(5);
    let mut frame = drive(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    let ticket = frame.ticket(&h.y[5]).unwrap();
    let time = frame.time();
    let (rows, passes) = if encode {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let (encoded, passes) = passes_of(|| frame.encode(&mut encoder).unwrap());
        let index = device.wgpu_queue().submit([encoder.finish()]);
        let _reported = encoded.submitted(&index);
        (drive(ticket).unwrap(), passes)
    } else {
        let (_submitted, passes) = passes_of(|| frame.submit().unwrap());
        (drive(ticket).unwrap(), passes)
    };
    let times = drive(time).unwrap();
    if level != FrameTiming::Off && device.supports_gpu_timestamps() {
        let times = times.as_ref().expect("a timed frame has a time");
        assert!(
            times.span > std::time::Duration::ZERO,
            "{level:?}: the span is above zero"
        );
    }
    let entries = times.map_or(0, |times| times.passes.len());
    assert_eq!(rows, six_model(&x), "{level:?}: rows against the host");
    (rows, passes, entries)
}

/// Six plain stages, each reading the one before: one pass untimed and at `Frame` (no per-stage entries). (Pass
/// counts: 1, 1.)
#[test]
fn six_plain_stages_are_one_pass_untimed_and_at_frame() {
    let twin = {
        let device = Arc::new(CpuDevice::new());
        let (mut batch, h) = build_cpu(&device).unwrap();
        let x = small(5);
        let mut frame = drive(batch.begin()).unwrap();
        frame.write(&h.x, &x).unwrap();
        let ticket = frame.ticket(&h.y[5]).unwrap();
        let _submitted = frame.submit().unwrap();
        drive(ticket).unwrap()
    };
    let (rows, passes, entries) = run_six(FrameTiming::Off, false, false);
    assert_eq!((passes, entries), (1, 0), "Off");
    assert_eq!(rows, twin, "against the CPU twin");
    let (_, passes, entries) = run_six(FrameTiming::Frame, false, false);
    assert_eq!(
        passes, 1,
        "a Frame-timed frame has the passes of an untimed one"
    );
    assert_eq!(entries, 0, "Frame has no per-stage entries");
}

/// At `Passes` a device with stamps inside passes records the one pass of the untimed frame; one without them keeps a
/// pass per stage. Either way, one entry per stage. (Pass count: 1, or 6 without the feature.)
#[test]
fn six_plain_stages_at_passes_are_one_pass_with_stamps_inside_or_a_pass_each_without() {
    let probe = device_logged("one_pass passes probe");
    if !probe.supports_gpu_timestamps() {
        println!(
            "SKIP: the adapter has no timestamp queries, so level Passes has no entries to count"
        );
        return;
    }
    let inside = probe
        .wgpu_device()
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    let (_, passes, entries) = run_six(FrameTiming::Passes, false, false);
    if inside {
        assert_eq!(
            passes, 1,
            "stamps inside the pass: the untimed frame's pass"
        );
    } else {
        assert_eq!(passes, 6, "no stamps inside passes: a pass per stage");
    }
    assert_eq!(entries, 6, "one entry per stage");
}

#[test]
fn six_plain_stages_are_a_pass_each_when_forced_and_give_the_same_rows() {
    let (rows, passes, _) = run_six(FrameTiming::Off, true, false);
    assert_eq!(passes, 6);
    let (merged, ..) = run_six(FrameTiming::Off, false, false);
    assert_eq!(rows, merged);
    let (rows, passes, _) = run_six(FrameTiming::Frame, true, false);
    assert_eq!(passes, 6, "forced, at Frame");
    assert_eq!(rows, merged);
}

/// `encode` into the caller's encoder records the same passes as `submit`. (Pass counts: 1, 6.)
#[test]
fn encode_into_a_callers_encoder_records_the_same_passes() {
    let (_, passes, _) = run_six(FrameTiming::Off, false, true);
    assert_eq!(passes, 1);
    let (_, passes, _) = run_six(FrameTiming::Off, true, true);
    assert_eq!(passes, 6);
    let (_, passes, _) = run_six(FrameTiming::Frame, false, true);
    assert_eq!(passes, 1);
}

// ---- a counted stage ----

const CAP: u32 = 1000;

#[derive(Clone, Copy)]
struct Counted {
    n: Stage<u32, D1>,
    x: Stage<u32, D1>,
    d: Stage<u32, D1>,
}

fn build_counted_gpu(
    device: &Arc<HeadlessDevice>,
) -> Result<(Batch<HeadlessDevice>, Counted), Error> {
    gpu_batch!("one-pass-counted", on(device); in_flight(1); |b| {
        let n = b.input::<u32>(1u32);
        let x = b.input::<u32>(CAP);
        let a = gpu_task!(CAP, |i| -> u32 { x[i] + 1u32 });
        let bb = gpu_task!(CAP, |i| -> u32 { a[i] + 1u32 });
        let c = gpu_task!(count(n); CAP, |i| -> u32 { bb[i] * 3u32 });
        let d = gpu_task!(CAP, |i| -> u32 { c[i] + 5u32 });
        Counted { n, x, d }
    })
}

fn build_counted_cpu(device: &Arc<CpuDevice>) -> Result<(Batch<CpuDevice>, Counted), Error> {
    gpu_batch!("one-pass-counted", on(device); in_flight(1); |b| {
        let n = b.input::<u32>(1u32);
        let x = b.input::<u32>(CAP);
        let a = gpu_task!(CAP, |i| -> u32 { x[i] + 1u32 });
        let bb = gpu_task!(CAP, |i| -> u32 { a[i] + 1u32 });
        let c = gpu_task!(count(n); CAP, |i| -> u32 { bb[i] * 3u32 });
        let d = gpu_task!(CAP, |i| -> u32 { c[i] + 5u32 });
        Counted { n, x, d }
    })
}

/// Two frames (counts 300 then 100) of plain, plain, counted, plain, with the pass form `forced` or not: the rows, the
/// passes of each frame and the indirect launches the probe saw.
fn run_counted(launch: CountedLaunch, forced: bool) -> (Vec<Vec<u32>>, Vec<u64>, u64) {
    let _form = Form::take(forced);
    set_counted_launch(Some(launch));
    let device = Arc::new(device_logged("one_pass counted"));
    HeadlessDevice::counted_probe(true);
    let (mut batch, h) = build_counted_gpu(&device).unwrap();
    let x = small_of(CAP, 3);
    let (mut rows, mut counts) = (Vec::new(), Vec::new());
    for (frame_number, n) in [300u32, 100].into_iter().enumerate() {
        let mut frame = drive(batch.begin()).unwrap();
        frame.write(&h.n, &[n]).unwrap();
        if frame_number == 0 {
            frame.write(&h.x, &x).unwrap();
        }
        let ticket = frame.ticket(&h.d).unwrap();
        let (_submitted, passes) = passes_of(|| frame.submit().unwrap());
        rows.push(drive(ticket).unwrap());
        counts.push(passes);
    }
    let indirect = HeadlessDevice::counted_probe_launches();
    HeadlessDevice::counted_probe(false);
    (rows, counts, indirect)
}

fn counted_twin() -> Vec<Vec<u32>> {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_counted_cpu(&device).unwrap();
    let x = small_of(CAP, 3);
    [300u32, 100]
        .into_iter()
        .enumerate()
        .map(|(frame_number, n)| {
            let mut frame = drive(batch.begin()).unwrap();
            frame.write(&h.n, &[n]).unwrap();
            if frame_number == 0 {
                frame.write(&h.x, &x).unwrap();
            }
            let ticket = frame.ticket(&h.d).unwrap();
            let _submitted = frame.submit().unwrap();
            drive(ticket).unwrap()
        })
        .collect()
}

/// Plain, plain, indirect counted, plain: the counted stage's args dispatch and indirect dispatch join the run, so the
/// frame is one pass; under `force_pass_per_dispatch` (the old form) every dispatch is a pass, the counted stage's two
/// among them. The rows are the CPU twin's and the host model's in both. (Pass counts: 1; forced 1 + 1 + 2 + 1 = 5.)
#[test]
fn an_indirect_counted_stage_joins_the_run() {
    let (rows, counts, indirect) = run_counted(CountedLaunch::Indirect, false);
    assert_eq!(counts, [1, 1], "one pass a frame");
    assert_eq!(indirect, 2, "one indirect launch a frame");
    assert_eq!(rows, counted_twin(), "against the CPU twin");
    let x = small_of(CAP, 3);
    let first: Vec<u32> = (0..CAP as usize)
        .map(|i| if i < 300 { (x[i] + 2) * 3 + 5 } else { 5 })
        .collect();
    assert_eq!(rows[0], first, "the first frame against the host");

    let (forced_rows, forced_counts, forced_indirect) = run_counted(CountedLaunch::Indirect, true);
    assert_eq!(forced_counts, [5, 5], "forced: a pass per dispatch");
    assert_eq!(forced_indirect, 2, "forced: one indirect launch a frame");
    assert_eq!(forced_rows, rows, "forced and default give the same rows");
}

/// The bounded launch is one plain kernel: the whole frame is one pass. (Pass count: 1.)
#[test]
fn a_bounded_counted_stage_joins_the_run() {
    let (rows, counts, indirect) = run_counted(CountedLaunch::Bounded, false);
    assert_eq!(counts, [1, 1]);
    assert_eq!(indirect, 0);
    assert_eq!(rows, counted_twin());
}

// ---- library ops ----

/// Three prepared stages, the middle one a list of two dispatches as a library op is (`prepare_list`): recorded
/// through the frame hook they are one pass and the last reads what the first two wrote. (Pass count: 1.)
#[test]
fn a_library_op_between_plain_stages_joins_the_run() {
    let _form = Form::take(false);
    let device = device_logged("one_pass library op");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let wires: Vec<GpuBuffer<u32>> = (0..4)
        .map(|index| buffer(&device, &format!("w{index}"), &vec![0; N as usize]))
        .collect();
    let (w0, w1, w2, w3) = (
        bindings(&a, &wires[0]),
        bindings(&wires[0], &wires[1]),
        bindings(&wires[1], &wires[2]),
        bindings(&wires[2], &wires[3]),
    );
    let first = device
        .prepare_list((), &[BufferDispatch::new(&ADD_ONE, N, &w0)])
        .unwrap();
    let op = device
        .prepare_list(
            (),
            &[
                BufferDispatch::new(&ADD_ONE, N, &w1),
                BufferDispatch::new(&ADD_ONE, N, &w2),
            ],
        )
        .unwrap();
    let last = device
        .prepare_list((), &[BufferDispatch::new(&ADD_ONE, N, &w3)])
        .unwrap();
    let mut encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let ((), passes) = passes_of(|| {
        device
            .frame_stages(&mut encoder, &[&first, &op, &last])
            .unwrap();
    });
    let index = device.wgpu_queue().submit([encoder.finish()]);
    device
        .wgpu_device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let _ = index;
    assert_eq!(passes, 1, "four dispatches of three stages in one pass");
    assert_eq!(
        device.read_typed_buffer(&wires[3]).unwrap(),
        plus(&input, 4)
    );
}

// ---- the default ----

/// Nothing sets the form: batches are recorded in one pass by default (the per-pass form is off until asked for).
/// The frame path reads the same switch and is checked under the same default. (Pass counts: 1, 1.)
#[test]
fn one_pass_is_the_default_form() {
    let form = Form::as_found();
    // The frame path first: a six-stage frame, nothing set by the test, is one pass.
    let (_, frame_passes, _) = run_six_as_set(FrameTiming::Off, false);
    assert_eq!(frame_passes, 1, "a frame is one pass by default");
    assert!(!form.before, "the per-pass form is off by default");
    let device = device_logged("one_pass default");
    let input = ramp(N);
    let a = buffer(&device, "a", &input);
    let (b, c) = (
        buffer(&device, "b", &vec![0; N as usize]),
        buffer(&device, "c", &vec![0; N as usize]),
    );
    let (first, second) = (bindings(&a, &b), bindings(&b, &c));
    let batch = [
        BufferDispatch::new(&ADD_ONE, N, &first),
        BufferDispatch::new(&ADD_ONE, N, &second),
    ];
    let ((), passes) = passes_of(|| device.dispatch_batch(&batch).unwrap());
    assert_eq!(passes, 1);
    assert_eq!(device.read_typed_buffer(&c).unwrap(), plus(&input, 2));
    drop(form);
}
