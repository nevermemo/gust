//! The device's timestamp objects are made at the first timed call, not at open (m14.lazyts; Gpu tier).
//!
//! The seam is `HeadlessDevice::timestamp_profiler_made`: how often this device made its profiler (the query set and
//! two buffers behind `dispatch_batch_timed` and `execute_graph_timed`). A batch frame's timing objects belong to the
//! batch, never to the profiler, and are made when a batch with a timing level is built; the process-wide counters
//! of `frame_timing_counters` show it. On an adapter without timestamp queries nothing is ever made and every timed
//! call reports no device time; the tests say so and assert that path instead.

mod common;

use std::sync::Arc;

use common::{device_logged, drive};
use gust::batch::{Batch, D1, Error, FrameTiming, Stage};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};

const N: usize = 256;

#[gust::gpu_module]
mod bump {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut dst: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < dst.len() {
            dst[i] = src[i] + 1u32;
        }
    }
}

/// What `timestamp_profiler_made` must say once the profiler was asked for: one where the device has timestamp
/// queries, none where it has not.
fn made_once(device: &HeadlessDevice) -> u32 {
    u32::from(device.supports_gpu_timestamps())
}

struct Rig {
    src: gust_wgpu::GpuBuffer<u32>,
    dst: gust_wgpu::GpuBuffer<u32>,
}

impl Rig {
    fn new(device: &HeadlessDevice) -> Self {
        let values: Vec<u32> = (0..N as u32).collect();
        Self {
            src: device.create_typed_buffer("src", &values, GpuBufferAccess::ReadWrite),
            dst: device.create_typed_buffer("dst", &[0_u32; N], GpuBufferAccess::ReadWrite),
        }
    }

    fn check(&self, device: &HeadlessDevice) {
        let expected: Vec<u32> = (1..=N as u32).collect();
        assert_eq!(device.read_typed_buffer(&self.dst).unwrap(), expected);
    }
}

/// The report of a one-node graph over the rig, run timed or untimed.
fn graph_report(device: &HeadlessDevice, rig: &Rig, timed: bool) -> gust_wgpu::GraphReport {
    let call = bump::run::call(N as u32, &rig.src, &rig.dst);
    let mut graph = StagedGraph::new();
    graph.run(&call);
    let result = if timed {
        device.execute_graph_timed(&graph)
    } else {
        device.execute_graph(&graph)
    };
    result.unwrap().report
}

#[test]
fn an_untimed_device_makes_no_timestamp_objects() {
    let device = device_logged("lazy untimed");
    assert_eq!(device.timestamp_profiler_made(), 0, "not at open");
    // Asking what the device supports makes nothing.
    let supports = device.supports_gpu_timestamps();
    println!("lazy timestamps: supports_gpu_timestamps={supports}");
    assert_eq!(device.timestamp_profiler_made(), 0, "not by asking");

    let rig = Rig::new(&device);
    let call = bump::run::call(N as u32, &rig.src, &rig.dst);
    device.dispatch_batch(&[(&call).into()]).unwrap();
    rig.check(&device);
    assert_eq!(graph_report(&device, &rig, false).gpu_elapsed, None);
    assert_eq!(
        device.timestamp_profiler_made(),
        0,
        "untimed dispatches and graphs make none"
    );
}

#[test]
fn the_first_timed_call_makes_the_timestamp_objects_once() {
    // Each entry point first on a device of its own, then every entry point again on it.
    for first in 0..2 {
        let device = device_logged("lazy first timed call");
        let rig = Rig::new(&device);
        let timed_batch = |device: &HeadlessDevice| {
            let call = bump::run::call(N as u32, &rig.src, &rig.dst);
            let timing = device.dispatch_batch_timed(&[(&call).into()]).unwrap();
            assert_eq!(
                timing.gpu_elapsed.is_some(),
                device.supports_gpu_timestamps(),
                "a timed batch's device time"
            );
            if let Some(elapsed) = timing.gpu_elapsed {
                assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
            }
        };
        let timed_graph = |device: &HeadlessDevice| {
            let report = graph_report(device, &rig, true);
            assert_eq!(
                report.gpu_elapsed.is_some(),
                device.supports_gpu_timestamps(),
                "a timed graph's device time"
            );
            if let Some(elapsed) = report.gpu_elapsed {
                assert!(elapsed < std::time::Duration::from_secs(10), "{elapsed:?}");
            }
        };
        assert_eq!(device.timestamp_profiler_made(), 0);
        if first == 0 {
            timed_batch(&device);
        } else {
            timed_graph(&device);
        }
        assert_eq!(
            device.timestamp_profiler_made(),
            made_once(&device),
            "made by the first timed call"
        );
        for _ in 0..3 {
            timed_batch(&device);
            timed_graph(&device);
        }
        assert_eq!(
            device.timestamp_profiler_made(),
            made_once(&device),
            "made once, not once per timed call"
        );
        rig.check(&device);
    }
}

struct Handles {
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

fn build(
    device: &Arc<HeadlessDevice>,
    level: FrameTiming,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("lazy frame", on(device); in_flight(1); |b| {
        b.frame_timing(level);
        let x = b.input::<u32>(N as u32);
        let y = gpu_task!(N as u32, |i| -> u32 { x[i] + 1u32 });
        Handles { x, y }
    })
}

#[test]
fn a_timed_frame_still_gets_its_time_and_never_touches_the_profiler() {
    let device = Arc::new(device_logged("lazy frame"));
    let values: Vec<u32> = (0..N as u32).collect();
    let expected: Vec<u32> = (1..=N as u32).collect();
    let before = device.frame_timing_counters();

    // A batch that does not ask for timing makes no timing objects.
    let (mut plain, handles) = build(&device, FrameTiming::Off).unwrap();
    let mut run = drive(plain.begin()).unwrap();
    run.write(&handles.x, &values).unwrap();
    let ticket = run.ticket(&handles.y).unwrap();
    let _submitted = run.submit().unwrap();
    assert_eq!(drive(ticket).unwrap(), expected);
    assert_eq!(device.frame_timing_counters(), before, "Off makes nothing");

    // A batch that asks makes them when it is built, once, and its frames return a time on a device that can.
    let (mut batch, handles) = build(&device, FrameTiming::Frame).unwrap();
    let after_build = device.frame_timing_counters();
    let supports = device.supports_gpu_timestamps();
    assert_eq!(after_build.0, before.0 + u64::from(supports));
    for _ in 0..3 {
        let mut run = drive(batch.begin()).unwrap();
        run.write(&handles.x, &values).unwrap();
        let ticket = run.ticket(&handles.y).unwrap();
        let time = run.time();
        let _submitted = run.submit().unwrap();
        assert_eq!(drive(ticket).unwrap(), expected);
        let time = drive(time).unwrap();
        assert_eq!(time.is_some(), supports, "a frame's device time");
        if let Some(time) = time {
            assert!(time.span < std::time::Duration::from_secs(10), "{time:?}");
        }
    }
    assert_eq!(
        device.frame_timing_counters().0,
        after_build.0,
        "the frames made no more timing objects"
    );
    assert_eq!(
        device.timestamp_profiler_made(),
        0,
        "frame timing never makes the device's profiler"
    );
}
