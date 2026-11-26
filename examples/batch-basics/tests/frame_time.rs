//! The frame timing hook (m14.pfft): `FrameTiming` at the build of a batch, `FrameRun::time`, `FrameTime` and
//! `Batch::frame_timing`, on the CPU twin and on the GPU.
//!
//! The checks C1 to C4 of the slice's contract are here as assertions: presence, order and the 8x scaling of a timed
//! frame (C1), the entries of level `Passes` (C2), the two forms and the one-submission rule (C3), and "off costs
//! nothing" shown structurally with the counters of the hook, never with a stopwatch (C4), plus the path of a device
//! without timestamp queries. The batch is a tiny stage of fixed size first and a heavy stage that scales with the
//! element count second, so a span that ends at the first pass instead of the last does not scale and fails C1.
//!
//! The GPU tests are `#[ignore]`d like the example's other GPU tests: they run in full verification, on the adapter the
//! environment selects, and print its name. The counters of the hook are process-wide, so every GPU test holds
//! [`serial`].

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use batch_basics::app::block_on;
use gust::batch::{
    Batch, BatchDevice, D1, Error, FrameTime, FrameTimes, FrameTiming, Param, Stage, Submitted,
    Ticket,
};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, wgpu};

/// Elements of the scan stage of the `Passes` batch.
const SCAN_LEN: u32 = 1000;
/// Elements of the heavy stage of the `Passes` batch.
const HEAVY: u32 = 1 << 18;

struct Handles {
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

/// The batch: a tiny stage, then `y = f(x)` where `f` is a counted loop per element (compute-bound, so the time of the
/// frame scales with the element count).
macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(
            device: &Arc<$device>,
            n: u32,
            level: FrameTiming,
        ) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("timed", on(device); in_flight($depth); |b| {
                b.frame_timing(level);
                let x = b.input::<u32>(n);
                let _tiny = gpu_task!(64u32, |i| -> u32 { i * 3u32 });
                let y = gpu_task!(n, |i| -> u32 {
                    let mut v = x[i] % 65521u32;
                    for _k in 0u32..4000u32 {
                        v = (v * 3u32 + 1u32) % 65521u32;
                    }
                    v
                });
                Handles { x, y }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice, 1);
define_build!(build_gpu, HeadlessDevice, 1);
define_build!(build_gpu2, HeadlessDevice, 2);
define_build!(build_gpu4, HeadlessDevice, 4);

struct PassHandles {
    size: Param<u32>,
    values: Stage<u32, D1>,
    x: Stage<u32, D1>,
}

/// A tiny stage, a scan over a resizable input (it has no pass at length zero), then the heavy stage.
fn build_passes(
    device: &Arc<HeadlessDevice>,
) -> Result<(Batch<HeadlessDevice>, PassHandles), Error> {
    gpu_batch!("passes", on(device); in_flight(1); |b| {
        b.frame_timing(FrameTiming::Passes);
        let size = b.param::<u32>(SCAN_LEN);
        let values = b.input::<u32>(&size);
        let x = b.input::<u32>(HEAVY);
        let _tiny = gpu_task!(64u32, |i| -> u32 { i * 3u32 });
        let _inclusive = gust::scan(&values);
        let _heavy = gpu_task!(HEAVY, |i| -> u32 {
            let mut v = x[i] % 65521u32;
            for _k in 0u32..1000u32 {
                v = (v * 3u32 + 1u32) % 65521u32;
            }
            v
        });
        PassHandles { size, values, x }
    })
}

fn data(n: u32, seed: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 7 + seed) % 100_000).collect()
}

static SERIAL: Mutex<()> = Mutex::new(());

/// The GPU tests of this file one at a time: the counters of the hook are the process's.
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn private_device() -> Arc<HeadlessDevice> {
    let device = HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    });
    println!("frame time adapter: {}", device.adapter_info().name);
    if std::env::var_os("GUST_PFFT_NO_TIMESTAMPS").is_some() {
        // The route to the `None` branch on any machine: the same adapter, opened without timestamp queries.
        let bare = bare_on(&device.adapter_info().name);
        println!("frame time: timestamp queries hidden by GUST_PFFT_NO_TIMESTAMPS");
        return bare;
    }
    Arc::new(device)
}

/// The adapter called `name`, opened without any feature: it has no timestamp queries.
fn bare_on(name: &str) -> Arc<HeadlessDevice> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .find(|adapter| adapter.get_info().name == name)
        .expect("the adapter the tests run on is among the Vulkan adapters");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("a device without timestamp queries"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("a bare device");
    let bare = Arc::new(HeadlessDevice::adopt(&adapter, device, queue, &[]).expect("adopts"));
    assert!(!bare.supports_gpu_timestamps());
    bare
}

/// What every timed check asserts on an adapter without timestamp queries (lavapipe, or a device opened without the
/// feature): no batch can time frames, every `time()` is `Ok(None)`, nothing is made or recorded and the values of the
/// frames are those of an untimed batch. The timed checks did not run.
fn none_path(device: &Arc<HeadlessDevice>, what: &str) {
    println!(
        "{what}: this adapter has no timestamp queries; the timed checks did not run, the None path is asserted"
    );
    let before = device.frame_timing_counters();
    let n = 1 << 12;
    let x = data(n, 13);
    let (mut plain, hp) = build_gpu(device, n, FrameTiming::Off).unwrap();
    let (ticket, _none, _s) = frame(&mut plain, &hp, Some(&x), false);
    let reference = block_on(ticket).unwrap();
    for level in [FrameTiming::Frame, FrameTiming::Passes] {
        let (mut batch, h) = build_gpu(device, n, level).unwrap();
        assert!(
            batch.frame_timing().is_none(),
            "{level:?}: no timestamp queries"
        );
        let (ticket, time, _s) = frame(&mut batch, &h, Some(&x), true);
        assert_eq!(
            block_on(ticket).unwrap(),
            reference,
            "{level:?}: values unchanged"
        );
        assert!(matches!(block_on(time.unwrap()), Ok(None)), "{level:?}");
        let dropped = {
            let mut run = block_on(batch.begin()).unwrap();
            run.time()
        };
        assert!(matches!(block_on(dropped), Err(Error::NotSubmitted { .. })));
    }
    assert_eq!(
        device.frame_timing_counters(),
        before,
        "nothing was made or recorded"
    );
}

/// One poll with a waker that does nothing: the way a caller without a runtime reads a ticket, and the only way these
/// tests read a frame time (nothing on the path waits).
fn poll_now<F: Future + Unpin>(future: &mut F) -> Option<F::Output> {
    let mut context = Context::from_waker(Waker::noop());
    match Pin::new(future).poll(&mut context) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

/// One frame of `batch`: the first frame writes `x`; a ticket on `y` when asked; the time when asked.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    x: Option<&[u32]>,
    time: bool,
) -> (Ticket<u32>, Option<FrameTime>, Submitted<D>)
where
    D::Buffer<u32>: 'static,
{
    let mut run = block_on(batch.begin()).unwrap();
    if let Some(x) = x {
        run.write(&h.x, x).unwrap();
    }
    let ticket = run.ticket(&h.y).unwrap();
    let time = time.then(|| run.time());
    (ticket, time, run.submit().unwrap())
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values[values.len() / 2]
}

/// The span every frame time of `times` resolves to, polled until all are in (20 s at most).
fn spans_of(mut times: Vec<FrameTime>) -> Vec<Duration> {
    let started = Instant::now();
    let mut spans = Vec::new();
    while !times.is_empty() {
        let mut waiting = Vec::new();
        for mut time in times {
            match poll_now(&mut time) {
                Some(result) => spans.push(result.unwrap().expect("a timed frame has a time").span),
                None => waiting.push(time),
            }
        }
        times = waiting;
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{} frame times were still pending",
            times.len()
        );
        std::thread::yield_now();
    }
    spans
}

// ---- the CPU twin: no data, frames unchanged ----

#[test]
fn twin_frame_time_is_none_and_the_frames_are_unchanged() {
    let device = Arc::new(CpuDevice::new());
    let x = data(256, 1);
    let (mut plain, hp) = build_cpu(&device, 256, FrameTiming::Off).unwrap();
    let (mut asked, ha) = build_cpu(&device, 256, FrameTiming::Passes).unwrap();
    assert!(plain.frame_timing().is_none());
    assert!(
        asked.frame_timing().is_none(),
        "the twin has no timestamps: None, not an error"
    );
    let (reference, none, _s) = frame(&mut plain, &hp, Some(&x), false);
    assert!(none.is_none());
    let reference = block_on(reference).unwrap();
    let (ticket, time, submitted) = frame(&mut asked, &ha, Some(&x), true);
    assert_eq!(block_on(ticket).unwrap(), reference);
    let time = time.unwrap();
    assert!(
        matches!(block_on(time), Ok(None)),
        "a frame of the twin has no time"
    );
    // The same measurement is reachable from the submitted frame, and again.
    assert!(matches!(
        block_on(
            submitted
                .frame_time()
                .expect("the frame asked for its time")
        ),
        Ok(None)
    ));
    // A frame that is dropped resolves like its tickets.
    let doomed = {
        let mut run = block_on(asked.begin()).unwrap();
        run.time()
    };
    assert!(matches!(block_on(doomed), Err(Error::NotSubmitted { .. })));
}

// ---- C1: presence, order and the scaling of the span ----

/// The element count at which one frame of the heavy batch takes at least this long on the device: the checks are
/// sized by measurement, and say so.
const MIN_SPAN: Duration = Duration::from_millis(2);

fn calibrate(device: &Arc<HeadlessDevice>) -> u32 {
    let mut n = 1_u32 << 16;
    loop {
        let (mut batch, h) = build_gpu(device, n, FrameTiming::Frame).unwrap();
        let x = data(n, 3);
        let (_ticket, time, _s) = frame(&mut batch, &h, Some(&x), true);
        // A second frame, after the first one has been through the pipelines.
        drop(time);
        let (_ticket, time, _s) = frame(&mut batch, &h, None, true);
        let span = spans_of(vec![time.unwrap()])[0];
        if span >= MIN_SPAN || n >= 1 << 22 {
            println!("calibrated: {n} elements, one frame {span:?}");
            return n;
        }
        n *= 2;
    }
}

/// 30 frames of each of two batches, alternately, with `depth` frames in flight: every frame is timed, every time
/// resolves, and the median of the 8x frames is at least twice that of the 1x frames.
macro_rules! define_c1 {
    ($check:ident, $build:ident, $depth:expr) => {
        fn $check(device: &Arc<HeadlessDevice>, n: u32) {
            let (mut small, hs) = $build(device, n, FrameTiming::Frame).unwrap();
            let (mut big, hb) = $build(device, 8 * n, FrameTiming::Frame).unwrap();
            let info = small.frame_timing().expect("this adapter has timestamps");
            assert_eq!(info.level, FrameTiming::Frame);
            assert!(info.tick > Duration::ZERO);
            println!("depth {}: tick {:?}", $depth, info.tick);
            let (xs, xb) = (data(n, 1), data(8 * n, 2));
            let (mut small_times, mut big_times) = (Vec::new(), Vec::new());
            let (mut small_spans, mut big_spans) = (Vec::new(), Vec::new());
            for k in 0..30 {
                let (_t, time, _s) = frame(&mut small, &hs, (k == 0).then_some(&xs[..]), true);
                small_times.push(time.unwrap());
                let (_t, time, _s) = frame(&mut big, &hb, (k == 0).then_some(&xb[..]), true);
                big_times.push(time.unwrap());
                // Read what has arrived: nothing here waits, and the polls drive the device.
                for (times, spans) in [
                    (&mut small_times, &mut small_spans),
                    (&mut big_times, &mut big_spans),
                ] {
                    let mut waiting = Vec::new();
                    for mut time in times.drain(..) {
                        match poll_now(&mut time) {
                            Some(result) => spans.push(result.unwrap().unwrap().span),
                            None => waiting.push(time),
                        }
                    }
                    *times = waiting;
                }
            }
            small_spans.extend(spans_of(small_times));
            big_spans.extend(spans_of(big_times));
            assert_eq!(small_spans.len(), 30, "one time per frame");
            assert_eq!(big_spans.len(), 30, "one time per frame");
            assert!(
                small_spans
                    .iter()
                    .chain(&big_spans)
                    .all(|s| *s > Duration::ZERO)
            );
            let (ms, mb) = (median(small_spans), median(big_spans));
            println!("depth {}: median span 1x {ms:?}, 8x {mb:?}", $depth);
            assert!(
                ms >= Duration::from_millis(1),
                "the 1x frames are too short to order ({ms:?}); the calibration found {n} elements"
            );
            assert!(
                mb >= 2 * ms,
                "the 8x frames ({mb:?}) must take at least twice the 1x frames ({ms:?})"
            );
        }
    };
}

define_c1!(c1_depth2, build_gpu2, 2);
define_c1!(c1_depth4, build_gpu4, 4);

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_c1_a_timed_frame_has_its_own_ordered_span() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "C1");
        return;
    }
    let n = calibrate(&device);
    c1_depth2(&device, n);
    c1_depth4(&device, n);
}

// ---- C2: level Passes ----

fn expected_names(batch: &Batch<HeadlessDevice>, without: &[&str]) -> Vec<String> {
    batch
        .stages()
        .into_iter()
        .map(|stage| stage.name)
        .filter(|name| !without.contains(&name.as_str()))
        .collect()
}

fn passes_of(time: FrameTime) -> FrameTimes {
    block_on(time).unwrap().expect("a timed frame has a time")
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_c2_level_passes_gives_one_entry_per_stage_that_has_a_pass() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "C2");
        return;
    }
    let (mut batch, h) = build_passes(&device).unwrap();
    let info = batch.frame_timing().expect("this adapter has timestamps");
    assert_eq!(info.level, FrameTiming::Passes);
    let scan = batch.stages()[1].name.clone();
    let (values, x) = (data(SCAN_LEN, 1), data(HEAVY, 2));
    let check = |times: &FrameTimes, names: &[String]| {
        let listed: Vec<&str> = times
            .passes
            .iter()
            .map(|pass| {
                assert_eq!(pass.stages.len(), 1, "today a pass covers one stage");
                pass.stages[0].as_str()
            })
            .collect();
        assert_eq!(listed, names, "every stage with a dispatch once, in order");
        assert!(times.span > Duration::ZERO);
        let mut sum = Duration::ZERO;
        for pass in &times.passes {
            assert!(
                pass.duration <= times.span,
                "{pass:?} against {:?}",
                times.span
            );
            sum += pass.duration;
        }
        let slack =
            times.span.mul_f64(1.02) + info.tick * u32::try_from(times.passes.len()).unwrap();
        assert!(
            sum <= slack,
            "entries sum to {sum:?}, span {:?}",
            times.span
        );
    };

    // Every stage has a pass.
    let mut run = block_on(batch.begin()).unwrap();
    run.write(&h.values, &values).unwrap();
    run.write(&h.x, &x).unwrap();
    let time = run.time();
    let _s = run.submit().unwrap();
    let times = passes_of(time);
    check(&times, &expected_names(&batch, &[]));
    println!("passes: {times:?}");

    // The scan over zero elements has none: no entry for it.
    let mut run = block_on(batch.begin()).unwrap();
    run.set(&h.size, 0).unwrap();
    let time = run.time();
    let _s = run.submit().unwrap();
    let times = passes_of(time);
    check(&times, &expected_names(&batch, &[scan.as_str()]));
    assert_eq!(times.passes.len(), 2);
}

// ---- C3: both forms, one submission per frame, a frame never submitted ----

fn gpu_encoder(device: &HeadlessDevice) -> wgpu::CommandEncoder {
    device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
}

fn submit_caller(device: &HeadlessDevice, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
    device.wgpu_queue().submit([encoder.finish()])
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_c3_submit_and_encode_forms_resolve_with_one_submission_per_frame() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "C3");
        return;
    }
    let n = 1 << 12;
    let x = data(n, 5);
    for level in [FrameTiming::Off, FrameTiming::Frame, FrameTiming::Passes] {
        let (mut batch, h) = build_gpu(&device, n, level).unwrap();
        let timed = level != FrameTiming::Off;
        assert_eq!(batch.frame_timing().is_some(), timed, "{level:?}");

        // `submit`: exactly one GUST submission, with or without a time.
        for ask in [false, true] {
            let before = device.gust_submissions();
            let (ticket, time, _s) = frame(&mut batch, &h, Some(&x), ask);
            assert_eq!(
                device.gust_submissions() - before,
                1,
                "{level:?}, time asked: {ask}"
            );
            let _ = block_on(ticket).unwrap();
            if let Some(time) = time {
                let got = block_on(time).unwrap();
                assert_eq!(
                    got.is_some(),
                    timed,
                    "{level:?}: a time exactly when the batch is timed"
                );
            }
        }

        // `encode` and `submitted`: the caller's one submission carries the frame; GUST makes none.
        let mut encoder = gpu_encoder(&device);
        let mut run = block_on(batch.begin()).unwrap();
        run.write(&h.x, &x).unwrap();
        let ticket = run.ticket(&h.y).unwrap();
        let time = run.time();
        let before = device.gust_submissions();
        let encoded = run.encode(&mut encoder).unwrap();
        let index = submit_caller(&device, encoder);
        let _s = encoded.submitted(&index);
        assert_eq!(
            device.gust_submissions(),
            before,
            "{level:?}: the caller submits"
        );
        let _ = block_on(ticket).unwrap();
        assert_eq!(
            block_on(time).unwrap().is_some(),
            timed,
            "{level:?}: encode form"
        );

        // A dropped `Encoded` (and its encoder): nothing ran, the time says so like a ticket.
        let mut encoder = gpu_encoder(&device);
        let mut run = block_on(batch.begin()).unwrap();
        let time = run.time();
        drop(run.encode(&mut encoder).unwrap());
        drop(encoder);
        assert!(
            matches!(block_on(time), Err(Error::NotSubmitted { .. })),
            "{level:?}"
        );

        // An `Encoded` reported with the submission of something else: the epoch word says the copy never ran.
        let unrelated = submit_caller(&device, gpu_encoder(&device));
        let mut encoder = gpu_encoder(&device);
        let mut run = block_on(batch.begin()).unwrap();
        let time = run.time();
        let encoded = run.encode(&mut encoder).unwrap();
        let _s = encoded.submitted(&unrelated);
        drop(encoder);
        let got = block_on(time);
        if timed {
            assert!(matches!(got, Err(Error::NotSubmitted { .. })), "{level:?}");
        } else {
            // A batch that records no timestamps has no copy whose epoch could say so: no data, not an error.
            assert!(matches!(got, Ok(None)), "{level:?}");
        }

        // A frame dropped before `submit`.
        let time = {
            let mut run = block_on(batch.begin()).unwrap();
            run.time()
        };
        assert!(
            matches!(block_on(time), Err(Error::NotSubmitted { .. })),
            "{level:?}"
        );
    }
}

// ---- C4: off costs nothing, shown with the counters of the hook ----

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_c4_what_is_not_asked_for_costs_nothing() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "C4");
        return;
    }
    let n = 1 << 12;
    let x = data(n, 9);
    let counters = || device.frame_timing_counters();

    // A batch built `Off` makes no timing object, whatever a frame asks.
    let before = counters();
    let (mut off, ho) = build_gpu(&device, n, FrameTiming::Off).unwrap();
    assert_eq!(
        counters().0,
        before.0,
        "a batch built Off made a timing object"
    );
    assert!(off.frame_timing().is_none());
    let (ticket, time, _s) = frame(&mut off, &ho, Some(&x), true);
    let reference = block_on(ticket).unwrap();
    assert!(matches!(block_on(time.unwrap()), Ok(None)));
    assert_eq!(
        counters(),
        before,
        "an Off batch recorded a stamp or a resolve"
    );

    // A timed batch makes its objects once, at the build.
    let before = counters();
    let (mut frames_only, hf) = build_gpu(&device, n, FrameTiming::Frame).unwrap();
    let (mut passes, hp) = build_gpu(&device, n, FrameTiming::Passes).unwrap();
    assert_eq!(
        counters().0,
        before.0 + 2,
        "one set of objects per timed batch"
    );
    let built = counters();

    // A frame that does not ask records no write, no resolve and no copy, and has the same values.
    for (batch, h) in [(&mut frames_only, &hf), (&mut passes, &hp)] {
        let (ticket, none, _s) = frame(batch, h, Some(&x), false);
        assert!(none.is_none());
        assert_eq!(
            block_on(ticket).unwrap(),
            reference,
            "a frame that did not ask"
        );
    }
    assert_eq!(
        counters(),
        built,
        "a frame that did not ask recorded timing work"
    );

    // A frame that asks: one begin and one end at level Frame (two stages, the first begins and the last ends), a
    // begin and an end per stage at level Passes, and one resolve either way; the values do not change.
    let (ticket, time, _s) = frame(&mut frames_only, &hf, None, true);
    assert_eq!(block_on(ticket).unwrap(), reference, "timed at level Frame");
    assert!(block_on(time.unwrap()).unwrap().is_some());
    let after = counters();
    assert_eq!(
        (after.1 - built.1, after.2 - built.2),
        (2, 1),
        "level Frame"
    );
    let (ticket, time, _s) = frame(&mut passes, &hp, None, true);
    assert_eq!(
        block_on(ticket).unwrap(),
        reference,
        "timed at level Passes"
    );
    assert!(block_on(time.unwrap()).unwrap().is_some());
    let end = counters();
    assert_eq!(
        (end.1 - after.1, end.2 - after.2),
        (4, 1),
        "level Passes: two stages"
    );
    assert_eq!(end.0, after.0, "no object is made by a frame");
}

// ---- no timestamp feature: None, frames unchanged ----

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_device_without_timestamp_queries_gives_none_and_the_same_frames() {
    let _serial = serial();
    let normal = private_device();
    let name = normal.adapter_info().name.clone();
    let bare = bare_on(&name);
    let before = bare.frame_timing_counters();

    let n = 1 << 12;
    let x = data(n, 11);
    let (mut on_bare, hb) = build_gpu(&bare, n, FrameTiming::Passes).unwrap();
    assert!(
        on_bare.frame_timing().is_none(),
        "no timestamp queries: the batch cannot time frames"
    );
    assert_eq!(bare.frame_timing_counters(), before, "nothing was made");
    let (ticket, time, _s) = frame(&mut on_bare, &hb, Some(&x), true);
    let bare_values = block_on(ticket).unwrap();
    assert!(matches!(block_on(time.unwrap()), Ok(None)));
    assert_eq!(bare.frame_timing_counters(), before, "nothing was recorded");

    let (mut plain, hn) = build_gpu(&normal, n, FrameTiming::Off).unwrap();
    let (ticket, _none, _s) = frame(&mut plain, &hn, Some(&x), false);
    assert_eq!(
        bare_values,
        block_on(ticket).unwrap(),
        "the frames are unchanged"
    );
}

// ---- the pool of read-back buffers (S1, S2) ----

fn wait_idle(device: &HeadlessDevice) {
    device
        .wgpu_device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_unread_and_never_run_frames_do_not_use_up_the_read_back_pool() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "pool");
        return;
    }
    let n = 1 << 12;
    let x = data(n, 21);
    // in_flight 2: the pool holds 8 buffers. 60 frames whose times are dropped unread (some whose copy ran, some
    // reported with the submission of something else so the epoch word says it never ran, some by `submit`) would
    // use it up three times over if a buffer were lost each time.
    let (mut batch, h) = build_gpu2(&device, n, FrameTiming::Frame).unwrap();
    let before = device.frame_timing_counters();
    for k in 0..60 {
        let mut run = block_on(batch.begin()).unwrap();
        if k == 0 {
            run.write(&h.x, &x).unwrap();
        }
        let _ticket = run.ticket(&h.y).unwrap();
        let time = run.time();
        match k % 3 {
            0 => {
                drop(run.submit().unwrap());
            }
            1 => {
                let mut encoder = gpu_encoder(&device);
                let encoded = run.encode(&mut encoder).unwrap();
                let index = submit_caller(&device, encoder);
                drop(encoded.submitted(&index));
            }
            _ => {
                let unrelated = submit_caller(&device, gpu_encoder(&device));
                let mut encoder = gpu_encoder(&device);
                let encoded = run.encode(&mut encoder).unwrap();
                drop(encoded.submitted(&unrelated));
                drop(encoder);
            }
        }
        drop(time);
        if k % 8 == 7 {
            wait_idle(&device);
        }
    }
    wait_idle(&device);
    let after = device.frame_timing_counters();
    assert!(
        after.3 > before.3,
        "the frames that were never read retired their buffers"
    );
    // The pool made replacements: later frames are still timed, in both forms.
    let (_t, time, _s) = frame(&mut batch, &h, None, true);
    assert!(block_on(time.unwrap()).unwrap().is_some(), "submit form");
    let mut encoder = gpu_encoder(&device);
    let mut run = block_on(batch.begin()).unwrap();
    let time = run.time();
    let encoded = run.encode(&mut encoder).unwrap();
    let index = submit_caller(&device, encoder);
    let _s = encoded.submitted(&index);
    assert!(block_on(time).unwrap().is_some(), "encode form");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_dropped_encoded_retires_its_read_back_buffer_and_a_frame_that_was_not_encoded_does_not() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "retire");
        return;
    }
    let n = 1 << 12;
    let x = data(n, 22);
    let (mut batch, h) = build_gpu(&device, n, FrameTiming::Frame).unwrap();
    let retired = || device.frame_timing_counters().3;

    // Dropped before it recorded anything: the buffer goes back to the pool.
    let before = retired();
    let time = {
        let mut run = block_on(batch.begin()).unwrap();
        run.time()
    };
    assert!(matches!(block_on(time), Err(Error::NotSubmitted { .. })));
    assert_eq!(
        retired(),
        before,
        "nothing was recorded: nothing is retired"
    );

    // Recorded into the caller's encoder, then `Encoded` dropped: the copy may still run, so the buffer is retired,
    // as a ticket's staging buffer is. The caller's encoder is submitted afterwards anyway.
    let before = retired();
    let mut encoder = gpu_encoder(&device);
    let mut run = block_on(batch.begin()).unwrap();
    run.write(&h.x, &x).unwrap();
    let time = run.time();
    drop(run.encode(&mut encoder).unwrap());
    assert_eq!(retired(), before + 1, "a recorded copy retires its buffer");
    let _ = submit_caller(&device, encoder);
    assert!(matches!(block_on(time), Err(Error::NotSubmitted { .. })));
    wait_idle(&device);

    // The frames after it are timed and exact.
    let (ticket, time, _s) = frame(&mut batch, &h, Some(&x), true);
    let _ = block_on(ticket).unwrap();
    assert!(block_on(time.unwrap()).unwrap().is_some());
}

// ---- S3: frames in flight on one batch, each its own value ----

struct Shaped {
    n: Param<u32>,
    x: Stage<u32, D1>,
}

macro_rules! define_shaped {
    ($build:ident, $depth:tt) => {
        fn $build(device: &Arc<HeadlessDevice>, n: u32) -> Result<(Batch<HeadlessDevice>, Shaped), Error> {
            gpu_batch!("shaped", on(device); in_flight($depth); |b| {
                b.frame_timing(FrameTiming::Frame);
                let n = b.param::<u32>(n);
                let x = b.input::<u32>(&n);
                let _tiny = gpu_task!(64u32, |i| -> u32 { i * 3u32 });
                let _y = gpu_task!(&n, |i| -> u32 {
                    let mut v = x[i] % 65521u32;
                    for _k in 0u32..4000u32 {
                        v = (v * 3u32 + 1u32) % 65521u32;
                    }
                    v
                });
                Shaped { n, x }
            })
        }
    };
}

define_shaped!(build_shaped2, 2);
define_shaped!(build_shaped4, 4);

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_frames_in_flight_on_one_batch_each_read_their_own_span() {
    let _serial = serial();
    let device = private_device();
    if !device.supports_gpu_timestamps() {
        none_path(&device, "S3");
        return;
    }
    let n = calibrate(&device);
    for (depth, build) in [
        (2, build_shaped2 as fn(&Arc<HeadlessDevice>, u32) -> _),
        (4, build_shaped4),
    ] {
        let (mut batch, h) = build(&device, n).unwrap();
        let (small, big) = (data(n, 4), data(8 * n, 5));
        let mut times = Vec::new();
        // One query set and one resolve buffer serve every frame of the batch: the frames alternate 1x and 8x, and
        // all of them are in flight before any time is read.
        for k in 0..24 {
            let mut run = block_on(batch.begin()).unwrap();
            if k % 2 == 0 {
                run.set(&h.n, n).unwrap();
                run.write(&h.x, &small).unwrap();
            } else {
                run.set(&h.n, 8 * n).unwrap();
                run.write(&h.x, &big).unwrap();
            }
            times.push(run.time());
            drop(run.submit().unwrap());
        }
        let spans: Vec<Duration> = times
            .into_iter()
            .map(|time| {
                block_on(time)
                    .unwrap()
                    .expect("a timed frame has a time")
                    .span
            })
            .collect();
        let class = |parity: usize| -> Vec<Duration> {
            spans.iter().copied().skip(parity).step_by(2).collect()
        };
        let (smalls, bigs) = (class(0), class(1));
        println!(
            "depth {depth}: 1x spans {:?}..{:?}, 8x spans {:?}..{:?}",
            smalls.iter().min().unwrap(),
            smalls.iter().max().unwrap(),
            bigs.iter().min().unwrap(),
            bigs.iter().max().unwrap()
        );
        assert!(
            smalls.iter().max() < bigs.iter().min(),
            "depth {depth}: every 8x frame must have taken longer than every 1x frame: {spans:?}"
        );
    }
}
