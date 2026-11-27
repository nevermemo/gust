//! Registered check s2sc.m (m14.s2sc): a scatter stage of a batch against the same body written as a `#[gpu_module]`
//! kernel, on struct rows and a permutation of places. A probe, skipped unless `GUST_S2SC_PROBE=1`: it prints a table
//! and asserts only that the two results are equal before it times anything (a measurement of a wrong answer is worth
//! nothing). Run it on one adapter at a time, outside a quiet window:
//!
//! `GUST_S2SC_PROBE=1 cargo test -p batch-basics --release --test scatter_measure -- --nocapture`
//!
//! The two cells are `one` (one target, `out[dst[i]] = rule(rows[i])`) and `two` (the same row and a second target,
//! `flags_out[dst[i]] = rows[i].flags + 1`), over 2^20 and 2^22 rows. The stage side is an uncounted scatter stage of a
//! batch frame (`begin`, `submit`, wait; the inputs are written once before timing, as the frames of a batch keep their
//! buffers); the kernel side is the typed `dispatch` of a module kernel with the same fields in a row of the module's
//! own (D279 keeps `GpuStruct` out of modules), which returns when the work is done. Each round builds the two variants
//! in the other order (the first variant created on the Radeon 890M can be up to 3x slower than the second), and the
//! median is the middle frame time of [`FRAMES`] frames after [`WARMUP`] warm-up frames, over
//! [`ROUNDS`] rounds.
//!
//! Pre-registered rule, written before the first run: the ratio stage over kernel inside 0.90 to 1.10 on every cell
//! means there is nothing to say; outside it, the cause is explained before any sentence about cost goes into a
//! document, and no speed claim is made either way. The launch cost of a counted stage (D281) is not part of this
//! ratio.

// A scatter stage has the value `()`, and its `let` names the stage for the twin's panic message.
#![allow(clippy::let_unit_value)]

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, D1, Error, Stage};
use gust::{gpu_batch, gpu_module, gpu_task};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

const WARMUP: usize = 5;
const FRAMES: usize = 21;
const ROUNDS: usize = 3;
const SIZES: [u32; 2] = [1 << 20, 1 << 22];

/// The element of the stage side.
#[derive(Clone, Copy, Debug, PartialEq, gust::GpuStruct)]
#[repr(C)]
struct Row {
    life: f32,
    flags: u32,
    tag: u32,
}

#[gpu_module]
mod kernels {
    #[derive(Debug, PartialEq)]
    pub struct Row {
        pub life: f32,
        pub flags: u32,
        pub tag: u32,
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn one(
        id: SV_DispatchThreadID,
        rows: StructuredBuffer<Row>,
        dst: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<Row>,
    ) {
        let i = id.x;
        if i < rows.len() {
            let r = rows[i];
            out[dst[i]] = Row {
                life: r.life * 0.5,
                flags: r.flags + 1u32,
                tag: r.tag,
            };
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn two(
        id: SV_DispatchThreadID,
        rows: StructuredBuffer<Row>,
        dst: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<Row>,
        mut flags_out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < rows.len() {
            let r = rows[i];
            out[dst[i]] = Row {
                life: r.life * 0.5,
                flags: r.flags + 1u32,
                tag: r.tag,
            };
            flags_out[dst[i]] = r.flags + 1u32;
        }
    }
}

#[derive(Clone, Copy)]
struct Handles {
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
    flags_out: Stage<u32, D1>,
}

fn build_one(
    device: &Arc<HeadlessDevice>,
    n: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("scatter_one", on(device); in_flight(1); |b| {
        let rows = b.input::<Row>(n);
        let dst = b.input::<u32>(n);
        let out = b.input::<Row>(n);
        let flags_out = b.input::<u32>(1u32);
        let _scatter = gpu_task!(n, scatter(&mut out), |i| {
            let r = rows[i];
            out[dst[i]] = Row { life: r.life * 0.5, flags: r.flags + 1u32, tag: r.tag };
        });
        Handles { rows, dst, out, flags_out }
    })
}

fn build_two(
    device: &Arc<HeadlessDevice>,
    n: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("scatter_two", on(device); in_flight(1); |b| {
        let rows = b.input::<Row>(n);
        let dst = b.input::<u32>(n);
        let out = b.input::<Row>(n);
        let flags_out = b.input::<u32>(n);
        let _scatter = gpu_task!(n, scatter(&mut out, &mut flags_out), |i| {
            let r = rows[i];
            out[dst[i]] = Row { life: r.life * 0.5, flags: r.flags + 1u32, tag: r.tag };
            flags_out[dst[i]] = r.flags + 1u32;
        });
        Handles { rows, dst, out, flags_out }
    })
}

fn rows_of(n: u32) -> Vec<Row> {
    (0..n)
        .map(|i| Row {
            life: i as f32 * 0.25,
            flags: i * 3,
            tag: 1000 + i,
        })
        .collect()
}

/// A permutation of `0 .. n` for a power-of-two `n` (7 is odd).
fn places_of(n: u32) -> Vec<u32> {
    (0..n)
        .map(|i| i.wrapping_mul(7).wrapping_add(3) & (n - 1))
        .collect()
}

/// The host model of the target rows.
fn model(rows: &[Row], dst: &[u32]) -> (Vec<Row>, Vec<u32>) {
    let mut out = vec![
        Row {
            life: 0.0,
            flags: 0,
            tag: 0
        };
        rows.len()
    ];
    let mut flags = vec![0u32; rows.len()];
    for (r, d) in rows.iter().zip(dst) {
        out[*d as usize] = Row {
            life: r.life * 0.5,
            flags: r.flags + 1,
            tag: r.tag,
        };
        flags[*d as usize] = r.flags + 1;
    }
    (out, flags)
}

/// The median of the two middle values.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Writes the inputs once, reads the targets back after one frame and compares them with the model; returns the
/// median frame time in microseconds of the frames after it.
fn stage_time(
    batch: &mut Batch<HeadlessDevice>,
    h: &Handles,
    rows: &[Row],
    dst: &[u32],
    want: &(Vec<Row>, Vec<u32>),
    two: bool,
) -> f64 {
    let (out, flags) = block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, rows)?;
        frame.write(&h.dst, dst)?;
        let out = frame.ticket(&h.out)?;
        let flags = if two {
            Some(frame.ticket(&h.flags_out)?)
        } else {
            None
        };
        let _submitted = frame.submit()?;
        let out = out.await?;
        let flags = match flags {
            Some(ticket) => ticket.await?,
            None => Vec::new(),
        };
        Ok::<_, Error>((out, flags))
    })
    .unwrap();
    assert_eq!(out, want.0, "the stage's rows differ from the model");
    if two {
        assert_eq!(
            flags, want.1,
            "the stage's second target differs from the model"
        );
    }
    let mut times = Vec::new();
    for round in 0..WARMUP + FRAMES {
        let started = Instant::now();
        let frame = block_on(batch.begin()).unwrap();
        let submitted = frame.submit().unwrap();
        submitted.wait().unwrap();
        if round >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    median(times)
}

/// The same for the module kernel.
fn kernel_time(
    device: &HeadlessDevice,
    rows: &[Row],
    dst: &[u32],
    want: &(Vec<Row>, Vec<u32>),
    two: bool,
) -> f64 {
    let n = rows.len();
    let module_rows: Vec<kernels::Row> = rows
        .iter()
        .map(|r| kernels::Row {
            life: r.life,
            flags: r.flags,
            tag: r.tag,
        })
        .collect();
    let blank: Vec<kernels::Row> = (0..n)
        .map(|_| kernels::Row {
            life: 0.0,
            flags: 0,
            tag: 0,
        })
        .collect();
    let source =
        device.create_typed_buffer("scatter rows", &module_rows, GpuBufferAccess::ReadOnly);
    let places = device.create_typed_buffer("scatter dst", dst, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("scatter out", &blank, GpuBufferAccess::ReadWrite);
    let flags =
        device.create_typed_buffer("scatter flags", &vec![0u32; n], GpuBufferAccess::ReadWrite);
    let run = || {
        if two {
            kernels::two::dispatch(device, n as u32, &source, &places, &out, &flags).unwrap();
        } else {
            kernels::one::dispatch(device, n as u32, &source, &places, &out).unwrap();
        }
    };
    run();
    let got = device.read_typed_buffer(&out).unwrap();
    let got: Vec<Row> = got
        .iter()
        .map(|r| Row {
            life: r.life,
            flags: r.flags,
            tag: r.tag,
        })
        .collect();
    assert_eq!(got, want.0, "the kernel's rows differ from the model");
    if two {
        assert_eq!(
            device.read_typed_buffer(&flags).unwrap(),
            want.1,
            "the kernel's second target differs from the model"
        );
    }
    let mut times = Vec::new();
    for round in 0..WARMUP + FRAMES {
        let started = Instant::now();
        run();
        if round >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    median(times)
}

#[test]
fn gpu_s2sc_m_scatter_stage_against_module_kernel() {
    // A probe, not a gate test: it runs only when `GUST_S2SC_PROBE=1` and is no `#[ignore]` for the pin scan to list.
    if std::env::var("GUST_S2SC_PROBE").as_deref() != Ok("1") {
        return;
    }
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    println!("adapter: {}", device.adapter_info().name);
    assert_eq!(
        std::mem::size_of::<Row>(),
        std::mem::size_of::<kernels::Row>(),
        "the two rows must have one size"
    );
    println!("cell,rows,stage_us,kernel_us,ratio_stage_over_kernel");
    for two in [false, true] {
        for n in SIZES {
            let rows = rows_of(n);
            let dst = places_of(n);
            let want = model(&rows, &dst);
            let mut stage = Vec::new();
            let mut kernel = Vec::new();
            for round in 0..ROUNDS {
                let stage_first = round % 2 == 0;
                let mut run_stage = || {
                    let (mut batch, h) = if two {
                        build_two(&device, n).unwrap()
                    } else {
                        build_one(&device, n).unwrap()
                    };
                    stage.push(stage_time(&mut batch, &h, &rows, &dst, &want, two));
                };
                let mut run_kernel = || kernel.push(kernel_time(&device, &rows, &dst, &want, two));
                if stage_first {
                    run_stage();
                    run_kernel();
                } else {
                    run_kernel();
                    run_stage();
                }
            }
            let (stage, kernel) = (median(stage), median(kernel));
            println!(
                "{},{n},{stage:.1},{kernel:.1},{:.3}",
                if two { "two" } else { "one" },
                stage / kernel
            );
        }
    }
}
