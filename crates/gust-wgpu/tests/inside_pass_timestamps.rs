//! A frame timed at level `Passes` is the frame that runs untimed, with the stage borders stamped inside the passes
//! (slice m14.onepassts; Gpu tier).
//!
//! Where the device has `TIMESTAMP_QUERY_INSIDE_PASSES` (GUST asks for it where the adapter offers it) a timed frame
//! records the same passes as the untimed one, one time per stage; a device opened without the feature (here through the
//! adopt path) keeps a pass per stage and still returns one time per stage. The pass count is `batch_passes_recorded`,
//! read around `submit`; the results are held to a host model, so a stamp cannot hide a wrong byte. Each test prints
//! the adapter's name and the per-stage times it saw (run with `--show-output`).

mod common;

use std::sync::{Arc, Mutex, MutexGuard};

use common::{device_logged, drive};

use gust::batch::{
    Batch, CountedLaunch, D1, Error, FrameTimes, FrameTiming, Stage, set_counted_launch,
};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{
    DeviceOptions, HeadlessDevice, batch_passes_recorded, force_pass_per_dispatch, wgpu,
};

/// Large enough that a stage spans many workgroups.
const N: u32 = 1 << 18;
/// The capacity of the counted stage.
const CAP: u32 = 1000;
/// Frames of one batch a test runs: the first one meets cold pipelines, the last one's times are printed.
const FRAMES: usize = 3;

/// The counted launch and the pass form are process-wide settings, so every test of this file takes this lock.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The counted launch a test set, put back when it drops (the test holds [`serial`] meanwhile).
struct Launch;

impl Launch {
    fn set(launch: CountedLaunch) -> Self {
        set_counted_launch(Some(launch));
        Self
    }
}

impl Drop for Launch {
    fn drop(&mut self) {
        set_counted_launch(None);
    }
}

/// The pass form `force_pass_per_dispatch` had before the test set it, put back when it drops.
struct Forced(bool);

impl Forced {
    fn on() -> Self {
        Self(force_pass_per_dispatch(true))
    }
}

impl Drop for Forced {
    fn drop(&mut self) {
        force_pass_per_dispatch(self.0);
    }
}

fn has_inside_passes(device: &HeadlessDevice) -> bool {
    device
        .wgpu_device()
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
}

fn small(len: u32, seed: u32) -> Vec<u32> {
    (0..len).map(|i| (i * 7 + seed) % 1_000).collect()
}

// ---- three plain stages ----

#[derive(Clone, Copy)]
struct Three {
    x: Stage<u32, D1>,
    out: Stage<u32, D1>,
}

fn build_three(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Result<(Batch<HeadlessDevice>, Three), Error> {
    gpu_batch!("inside-pass-three", on(device); in_flight(1); |b| {
        b.frame_timing(level);
        let x = b.input::<u32>(N);
        let y0 = gpu_task!(N, |i| -> u32 { x[i] + 1u32 });
        let y1 = gpu_task!(N, |i| -> u32 { y0[i] * 3u32 });
        let out = gpu_task!(N, |i| -> u32 { y1[i] + 5u32 });
        Three { x, out }
    })
}

fn three_model(x: &[u32]) -> Vec<u32> {
    x.iter().map(|&v| (v + 1) * 3 + 5).collect()
}

/// What `FRAMES` frames of the three-stage batch at `level` recorded: for each, the rows, the passes and the times.
fn run_three(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Vec<(Vec<u32>, u64, Option<FrameTimes>)> {
    let (mut batch, h) = build_three(device, level).unwrap();
    let x = small(N, 5);
    (0..FRAMES)
        .map(|_| {
            let mut frame = drive(batch.begin()).unwrap();
            frame.write(&h.x, &x).unwrap();
            let ticket = frame.ticket(&h.out).unwrap();
            let time = frame.time();
            let before = batch_passes_recorded();
            let _submitted = frame.submit().unwrap();
            let passes = batch_passes_recorded() - before;
            let rows = drive(ticket).unwrap();
            assert_eq!(rows, three_model(&x), "{level:?}: rows against the host");
            (rows, passes, drive(time).unwrap())
        })
        .collect()
}

fn print_times(tag: &str, device: &HeadlessDevice, times: &FrameTimes) {
    let stages: Vec<String> = times
        .passes
        .iter()
        .map(|pass| format!("{}={:?}", pass.stages.join("+"), pass.duration))
        .collect();
    println!(
        "{tag}: adapter {} span {:?} stages [{}]",
        device.adapter_info().name,
        times.span,
        stages.join(", ")
    );
}

/// On a device with the feature a three-stage plain frame at `Passes` is the one pass of the untimed frame, with one
/// finite time per stage that together fit inside the span, and the rows of the untimed frame. (Pass counts: 1, 1.)
#[test]
fn three_plain_stages_at_passes_keep_the_untimed_pass_and_time_each_stage() {
    let _serial = serial();
    let device = Arc::new(device_logged("inside-pass three plain"));
    if !device.supports_gpu_timestamps() || !has_inside_passes(&device) {
        println!("SKIP: the adapter has no timestamp queries inside passes");
        return;
    }
    let untimed = run_three(&device, FrameTiming::Off);
    let timed = run_three(&device, FrameTiming::Passes);
    for ((rows, passes, none), (timed_rows, timed_passes, times)) in untimed.iter().zip(&timed) {
        assert!(none.is_none(), "an Off batch gives no time");
        assert_eq!(passes, &1, "three plain stages untimed are one pass");
        assert_eq!(
            timed_passes, passes,
            "the timed frame has the untimed frame's passes"
        );
        assert_eq!(
            timed_rows, rows,
            "the timed frame's rows are the untimed frame's"
        );
        let times = times.as_ref().expect("a timed frame has a time");
        assert_eq!(times.passes.len(), 3, "one time per stage");
        for pass in &times.passes {
            assert_eq!(pass.stages.len(), 1, "one stage an entry");
            assert!(
                pass.duration > std::time::Duration::ZERO,
                "a stage's end stamp is after its begin stamp: {:?}",
                pass.stages
            );
        }
        let sum: std::time::Duration = times.passes.iter().map(|pass| pass.duration).sum();
        assert!(
            sum <= times.span,
            "the stage intervals are disjoint and inside the span: {sum:?} against {:?}",
            times.span
        );
        assert!(
            sum * 2 >= times.span,
            "the stages cover most of the span, so no stamp collapsed: {sum:?} against {:?}",
            times.span
        );
        assert!(
            times.span > std::time::Duration::ZERO,
            "the span is above zero"
        );
        // The three stages do the same work, so none may be a sliver of the span (a swapped or collapsed stamp).
        for pass in &times.passes {
            assert!(
                pass.duration * 8 >= times.span,
                "stage {:?} is under an eighth of the span {:?}: {:?}",
                pass.stages,
                times.span,
                pass.duration
            );
        }
    }
    let last = timed.last().and_then(|frame| frame.2.as_ref()).unwrap();
    print_times("inside-pass three plain (Passes)", &device, last);
}

// ---- a plain run with an indirect counted stage ----

#[derive(Clone, Copy)]
struct Mixed {
    n: Stage<u32, D1>,
    x: Stage<u32, D1>,
    d: Stage<u32, D1>,
}

fn build_mixed(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Result<(Batch<HeadlessDevice>, Mixed), Error> {
    gpu_batch!("inside-pass-mixed", on(device); in_flight(1); |b| {
        b.frame_timing(level);
        let n = b.input::<u32>(1u32);
        let x = b.input::<u32>(CAP);
        let a = gpu_task!(CAP, |i| -> u32 { x[i] + 1u32 });
        let bb = gpu_task!(CAP, |i| -> u32 { a[i] + 1u32 });
        let c = gpu_task!(count(n); CAP, |i| -> u32 { bb[i] * 3u32 });
        let d = gpu_task!(CAP, |i| -> u32 { c[i] + 5u32 });
        Mixed { n, x, d }
    })
}

/// The host model of plain, plain, counted, plain over `count` live elements.
fn mixed_model(x: &[u32], count: u32) -> Vec<u32> {
    (0..CAP as usize)
        .map(|i| {
            if (i as u32) < count {
                (x[i] + 2) * 3 + 5
            } else {
                5
            }
        })
        .collect()
}

/// Two frames of the mixed batch (counts 100 then 300, so no element holds a value an earlier frame left): the rows, the
/// passes and the times of each.
fn run_mixed(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Vec<(Vec<u32>, u64, Option<FrameTimes>)> {
    let (mut batch, h) = build_mixed(device, level).unwrap();
    let x = small(CAP, 3);
    [100u32, 300]
        .into_iter()
        .enumerate()
        .map(|(number, count)| {
            let mut frame = drive(batch.begin()).unwrap();
            frame.write(&h.n, &[count]).unwrap();
            if number == 0 {
                frame.write(&h.x, &x).unwrap();
            }
            let ticket = frame.ticket(&h.d).unwrap();
            let time = frame.time();
            let before = batch_passes_recorded();
            let _submitted = frame.submit().unwrap();
            let passes = batch_passes_recorded() - before;
            let rows = drive(ticket).unwrap();
            assert_eq!(
                rows,
                mixed_model(&x, count),
                "{level:?}: rows against the host"
            );
            (rows, passes, drive(time).unwrap())
        })
        .collect()
}

/// Plain, plain, indirect counted, plain at `Passes`: the run before the counted stage is one pass, the counted stage
/// keeps its two, the stage after it one, as untimed; each of the four stages has a time. (Pass counts: 1 + 2 + 1 = 4.)
#[test]
fn a_plain_run_with_an_indirect_counted_stage_is_timed_per_stage_in_the_untimed_passes() {
    let _serial = serial();
    let _launch = Launch::set(CountedLaunch::Indirect);
    let device = Arc::new(device_logged("inside-pass mixed"));
    if !device.supports_gpu_timestamps() || !has_inside_passes(&device) {
        println!("SKIP: the adapter has no timestamp queries inside passes");
        return;
    }
    let untimed = run_mixed(&device, FrameTiming::Off);
    let timed = run_mixed(&device, FrameTiming::Passes);
    for ((rows, passes, _), (timed_rows, timed_passes, times)) in untimed.iter().zip(&timed) {
        assert_eq!(passes, &4, "1 + 2 + 1 passes untimed");
        assert_eq!(timed_passes, &4, "the timed frame has the same passes");
        assert_eq!(timed_rows, rows);
        let times = times.as_ref().expect("a timed frame has a time");
        assert_eq!(
            times.passes.len(),
            4,
            "one time per stage, the counted one included"
        );
        for index in [0, 1, 3] {
            assert!(
                times.passes[index].duration > std::time::Duration::ZERO,
                "the plain stage {:?} has a time above zero",
                times.passes[index].stages
            );
        }
        let sum: std::time::Duration = times.passes.iter().map(|pass| pass.duration).sum();
        assert!(sum <= times.span, "{sum:?} against {:?}", times.span);
        // No `sum * 2 >= span` lower bound here, unlike the three plain stages: the counted stage's two passes each
        // begin and end a pass, and those boundaries sit between stamps, so the stage times can cover under half the
        // span on some adapters.
    }
    let last = timed.last().and_then(|frame| frame.2.as_ref()).unwrap();
    print_times("inside-pass plain+counted (Passes)", &device, last);
}

// ---- a device opened without the feature ----

/// A device a caller opened with GUST's request minus the stamps inside passes, adopted.
fn adopted_without_inside_passes() -> Arc<HeadlessDevice> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a Vulkan adapter is required; this test must not silently skip");
    let options = DeviceOptions::default().with_backends(wgpu::Backends::VULKAN);
    let request = options.device_request(&adapter).unwrap();
    let features = request.features - wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("without stamps inside passes"),
        required_features: features,
        required_limits: request.limits,
        ..Default::default()
    }))
    .expect("the adapter opens a device");
    let adopted = HeadlessDevice::adopt(&adapter, device, queue, options.capabilities).unwrap();
    Arc::new(adopted)
}

/// Without the feature the frame at `Passes` is a pass per stage, with one time per stage, and the rows are right.
/// (Pass counts: 1 untimed, 3 timed.)
#[test]
fn a_device_without_stamps_inside_passes_keeps_a_pass_per_stage_and_times_each() {
    let _serial = serial();
    let device = adopted_without_inside_passes();
    eprintln!(
        "inside-pass fallback adapter: {}",
        device.adapter_info().name
    );
    assert!(
        !has_inside_passes(&device),
        "the device was opened without the feature"
    );
    if !device.supports_gpu_timestamps() {
        println!("SKIP: the adapter has no timestamp queries");
        return;
    }
    let untimed = run_three(&device, FrameTiming::Off);
    let timed = run_three(&device, FrameTiming::Passes);
    for ((rows, passes, _), (timed_rows, timed_passes, times)) in untimed.iter().zip(&timed) {
        assert_eq!(passes, &1, "untimed: one pass");
        assert_eq!(timed_passes, &3, "without the feature: a pass per stage");
        assert_eq!(timed_rows, rows);
        let times = times.as_ref().expect("a timed frame has a time");
        assert_eq!(times.passes.len(), 3, "one time per stage");
    }
    let last = timed.last().and_then(|frame| frame.2.as_ref()).unwrap();
    print_times("inside-pass fallback (Passes)", &device, last);
}

/// A device with the feature, while `force_pass_per_dispatch` is on, keeps a pass per stage at `Passes` as the untimed
/// frame does, and still returns one time per stage. (Pass counts: 3 untimed, 3 timed.)
#[test]
fn forcing_a_pass_per_dispatch_keeps_a_pass_per_stage_at_passes_and_times_each() {
    let _serial = serial();
    let _forced = Forced::on();
    let device = Arc::new(device_logged("inside-pass forced"));
    if !device.supports_gpu_timestamps() || !has_inside_passes(&device) {
        println!(
            "SKIP: the adapter has no timestamp queries inside passes, so the force gate is not reached"
        );
        return;
    }
    let untimed = run_three(&device, FrameTiming::Off);
    let timed = run_three(&device, FrameTiming::Passes);
    for ((rows, passes, _), (timed_rows, timed_passes, times)) in untimed.iter().zip(&timed) {
        assert_eq!(passes, &3, "forced, untimed: a pass per stage");
        assert_eq!(timed_passes, &3, "forced, at Passes: a pass per stage");
        assert_eq!(timed_rows, rows);
        let times = times.as_ref().expect("a timed frame has a time");
        assert_eq!(times.passes.len(), 3, "one time per stage");
    }
    let last = timed.last().and_then(|frame| frame.2.as_ref()).unwrap();
    print_times("inside-pass forced (Passes)", &device, last);
}
