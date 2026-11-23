//! The host-side cost of executing an unchanged N-node `StagedGraph`, written with GUST's public API only.
//!
//! ```text
//! GUST_PROFILE=dev cargo run --release --manifest-path bench/Cargo.toml -p graph-overhead
//! ```
//!
//! A chain of N one-kernel nodes (node `i` reads buffer `i` and writes buffer `i + 1`), built once and executed over
//! and over. Three shapes, each at 10, 20 and 40 nodes:
//!
//! - `zero`: every element count is 0, so the graph is validated and submitted but nothing is prepared or encoded;
//!   this is the floor that validation raises.
//! - `step`: 128 elements per node, no transfer: validation, preparation, encoding, submission and the wait.
//! - `frame`: `step` plus an upload into the first buffer and a readback of the last, the shape of a frame.
//!
//! Per row: the median of three runs of the mean wall time of one `execute_graph` call, and the same for the CPU time
//! of the calling thread (Windows only: `GetThreadTimes`, whose tick is coarse, so a run is thousands of calls). The
//! program asserts no time and no threshold; it is a one-machine measurement, never a speed claim.

use std::time::Instant;

use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice, StagedGraph};

#[gust::gpu]
mod ops {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn step(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<uint>,
        mut dst: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < dst.len() && i < src.len() {
            dst[i] = src[i] + 1u32;
        }
    }
}

const WARM_UP: usize = 200;
const RUNS: usize = 3;
const CALLS_PER_RUN: usize = 2000;
const ELEMENTS: usize = 128;

#[cfg(windows)]
fn thread_cpu_ns() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn GetThreadTimes(
            thread: isize,
            created: *mut FileTime,
            exited: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let ticks = |time: &FileTime| (u64::from(time.high) << 32 | u64::from(time.low)) * 100;
    let (mut created, mut exited, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // SAFETY: the pseudo handle of the calling thread and four live out-structures of the documented layout.
    let ok = unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            &raw mut created,
            &raw mut exited,
            &raw mut kernel,
            &raw mut user,
        )
    };
    assert_ne!(ok, 0, "GetThreadTimes failed");
    ticks(&kernel) + ticks(&user)
}

#[cfg(not(windows))]
fn thread_cpu_ns() -> u64 {
    0
}

fn median(mut values: [f64; RUNS]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[RUNS / 2]
}

/// Median over the runs of (mean wall us, mean thread CPU us) of one call of `execute`.
fn measure(mut execute: impl FnMut()) -> (f64, f64) {
    for _ in 0..WARM_UP {
        execute();
    }
    let mut wall = [0.0; RUNS];
    let mut cpu = [0.0; RUNS];
    for run in 0..RUNS {
        let cpu_before = thread_cpu_ns();
        let started = Instant::now();
        for _ in 0..CALLS_PER_RUN {
            execute();
        }
        wall[run] = started.elapsed().as_secs_f64() * 1e6 / CALLS_PER_RUN as f64;
        cpu[run] = (thread_cpu_ns() - cpu_before) as f64 / 1e3 / CALLS_PER_RUN as f64;
    }
    (median(wall), median(cpu))
}

fn buffers(device: &HeadlessDevice, count: usize, len: usize) -> Vec<GpuBuffer<u32>> {
    (0..count)
        .map(|index| {
            device.create_typed_buffer(
                &format!("chain {index}"),
                &vec![0u32; len],
                GpuBufferAccess::ReadWrite,
            )
        })
        .collect()
}

fn row(name: &str, nodes: usize, (wall, cpu): (f64, f64)) {
    println!("{name:<6} {nodes:>3} nodes   wall {wall:>8.1} us   thread cpu {cpu:>8.1} us");
}

fn main() {
    let device = HeadlessDevice::new().expect("an adapter");
    let info = device.adapter_info();
    println!(
        "adapter: {} ({:?}); {WARM_UP} warm-up calls, median of {RUNS} runs of {CALLS_PER_RUN} calls",
        info.name, info.backend
    );
    for nodes in [10, 20, 40] {
        // zero floor: empty buffers, count 0.
        let empty = buffers(&device, nodes + 1, 0);
        let calls: Vec<_> = (0..nodes)
            .map(|index| ops::step::call(0, &empty[index], &empty[index + 1]))
            .collect();
        let mut graph = StagedGraph::new();
        for call in &calls {
            graph.run(call);
        }
        row(
            "zero",
            nodes,
            measure(|| {
                device.execute_graph(&graph).expect("zero graph");
            }),
        );

        // step: 128 elements per node, no transfer.
        let chain = buffers(&device, nodes + 1, ELEMENTS);
        let calls: Vec<_> = (0..nodes)
            .map(|index| ops::step::call(ELEMENTS as u32, &chain[index], &chain[index + 1]))
            .collect();
        let mut graph = StagedGraph::new();
        for call in &calls {
            graph.run(call);
        }
        row(
            "step",
            nodes,
            measure(|| {
                let output = device.execute_graph(&graph).expect("step graph");
                assert_eq!(output.report.dispatches, nodes);
            }),
        );

        // frame: the same chain with an upload in front and a readback behind.
        let input = vec![7u32; ELEMENTS];
        let mut graph = StagedGraph::new();
        graph.write(&chain[0], &input);
        for call in &calls {
            graph.run(call);
        }
        let read = graph.read(&chain[nodes]);
        let mut checked = false;
        row(
            "frame",
            nodes,
            measure(|| {
                let output = device.execute_graph(&graph).expect("frame graph");
                if !checked {
                    let values = output.readback::<u32>(read).expect("readback");
                    assert_eq!(values, vec![7 + nodes as u32; ELEMENTS]);
                    checked = true;
                }
            }),
        );
    }
}
