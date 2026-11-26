//! A struct stored as the element of a batch's buffers (m14.s2s, D279): `b.input::<Row>(..)` with `Row` a
//! `#[derive(GpuStruct)]` struct, a stage that reads a field, a stage whose result is a whole `Row`, and an in-place
//! stage that stores a field, a compound and a whole row, and a read through a 2-D view.
//!
//! The host writes and reads `Row` values, the device holds the struct itself (`StructuredBuffer<Row>`): there is no
//! decoder and no word. Every value is small and exactly representable, so a result equals the host reference bit for
//! bit. The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU tests and
//! run in full verification on the adapter the environment selects.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, D2, Error, Param, Stage};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod rows {
    /// Three words of two kinds, so that a field read at the wrong offset reads the wrong kind.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub life: f32,
        pub flags: u32,
        pub tag: u32,
    }
}

use rows::Row;

const N: u32 = 48;
const W: u32 = 8;
const H: u32 = 6;

#[derive(Clone, Copy)]
struct Handles {
    rows: Stage<Row, D1>,
    life: Stage<f32, D1>,
    next: Stage<Row, D1>,
    work: Stage<Row, D1>,
    whole: Stage<Row, D1>,
    grid: Stage<Row, D2>,
    cell: Stage<u32, D2>,
}

macro_rules! define_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("struct_element", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let life = gpu_task!(N, |i| -> f32 { rows[i].life });
                let next = gpu_task!(N, |i| -> Row {
                    Row { life: rows[i].life * 0.5, flags: rows[i].flags + 1u32, tag: rows[i].tag }
                });
                let work = gpu_task!(N, |i| -> Row { rows[i] });
                let _step = gpu_task!(&mut work, |i| {
                    work[i].life += 1.0;
                    work[i].flags += 2u32;
                    work[i].tag = i;
                });
                let whole = gpu_task!(N, |i| -> Row { rows[i] });
                let _swap = gpu_task!(&mut whole, |i| {
                    whole[i] = Row {
                        life: whole[i].life * 2.0,
                        flags: whole[i].tag,
                        tag: whole[i].flags,
                    };
                });
                let grid = b.input::<Row>((W, H));
                let cell = gpu_task!((W, H), |x, y| -> u32 { grid[(x, y)].flags + x * 1000u32 + y });
                Handles { rows, life, next, work, whole, grid, cell }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice);
define_build!(build_gpu, HeadlessDevice);

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

fn rows_of(seed: u32) -> Vec<Row> {
    (0..N)
        .map(|i| Row {
            life: (i * 3 + seed) as f32 * 0.25,
            flags: i * 7 + seed,
            tag: 1000 + i,
        })
        .collect()
}

type Results = (Vec<f32>, Vec<Row>, Vec<Row>, Vec<Row>, Vec<u32>);

/// One frame: the rows and the grid written, the five results read.
fn frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, input: &[Row], grid: &[Row]) -> Results
where
    D::Buffer<u32>: 'static,
    D::Buffer<f32>: 'static,
    D::Buffer<Row>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, input)?;
        frame.write(&h.grid, grid)?;
        let life = frame.ticket(&h.life)?;
        let next = frame.ticket(&h.next)?;
        let work = frame.ticket(&h.work)?;
        let whole = frame.ticket(&h.whole)?;
        let cell = frame.ticket(&h.cell)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((
            life.await?,
            next.await?,
            work.await?,
            whole.await?,
            cell.await?,
        ))
    })
    .unwrap()
}

/// Field reads, a whole-row result, and an in-place field store, compound and index store, each equal to the host
/// reference over 16 frames (the buffers are reused, so a stale row would show).
fn results_equal_the_host<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
    D::Buffer<f32>: 'static,
    D::Buffer<Row>: 'static,
{
    for seed in 0..16 {
        let input = rows_of(seed);
        let grid = rows_of(seed + 1);
        let (life, next, work, whole, cell) = frame(&mut batch, h, &input, &grid);
        assert_eq!(
            life,
            input.iter().map(|r| r.life).collect::<Vec<_>>(),
            "a field read, frame {seed}"
        );
        assert_eq!(
            next,
            input
                .iter()
                .map(|r| Row {
                    life: r.life * 0.5,
                    flags: r.flags + 1,
                    tag: r.tag
                })
                .collect::<Vec<_>>(),
            "a whole row as the result, frame {seed}"
        );
        assert_eq!(
            work,
            input
                .iter()
                .enumerate()
                .map(|(i, r)| Row {
                    life: r.life + 1.0,
                    flags: r.flags + 2,
                    tag: i as u32
                })
                .collect::<Vec<_>>(),
            "in-place field store and compound, frame {seed}"
        );
        assert_eq!(
            whole,
            input
                .iter()
                .map(|r| Row {
                    life: r.life * 2.0,
                    flags: r.tag,
                    tag: r.flags
                })
                .collect::<Vec<_>>(),
            "an in-place whole-element store, frame {seed}"
        );
        assert_eq!(
            cell,
            grid.iter()
                .enumerate()
                .map(|(k, r)| r.flags + (k as u32 % W) * 1000 + k as u32 / W)
                .collect::<Vec<_>>(),
            "a field read through a 2-D view, frame {seed}"
        );
    }
}

#[test]
fn twin_a_struct_element_is_read_stored_and_updated_in_place_like_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    results_equal_the_host(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_struct_element_is_read_stored_and_updated_in_place_like_the_host() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    results_equal_the_host(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_struct_element_equals_the_cpu_twin() {
    let gpu = private_device();
    let cpu = Arc::new(CpuDevice::new());
    println!("adapter: {}", gpu.adapter_info().name);
    let (mut on_gpu, gpu_h) = build_gpu(&gpu).unwrap();
    let (mut on_cpu, cpu_h) = build_cpu(&cpu).unwrap();
    for seed in 0..3 {
        let input = rows_of(seed);
        assert_eq!(
            frame(&mut on_gpu, &gpu_h, &input, &input),
            frame(&mut on_cpu, &cpu_h, &input, &input),
            "seed {seed}"
        );
    }
}

// ---- a counted stage over a struct target (check 4) ----

/// What every row of the target holds before a frame; a row the counted stage does not reach keeps it, in every byte.
const SENTINEL: Row = Row {
    life: 2.5,
    flags: 999_999,
    tag: 888_888,
};
const CAP: u32 = 1000;

#[derive(Clone, Copy)]
struct Counted {
    n: Stage<u32, D1>,
    src: Stage<u32, D1>,
    rows: Stage<Row, D1>,
}

macro_rules! define_counted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Counted), Error> {
            gpu_batch!("struct_counted", on(device); in_flight(1); |b| {
                let n = b.input::<u32>(1u32);
                let src = b.input::<u32>(CAP);
                let rows = b.input::<Row>(CAP);
                let _fill = gpu_task!(count(n); &mut rows, |i| {
                    rows[i].flags = src[i] + 7u32;
                    rows[i].tag = i;
                });
                Counted { n, src, rows }
            })
        }
    };
}

define_counted!(counted_cpu, CpuDevice);
define_counted!(counted_gpu, HeadlessDevice);

/// Counts at the edges (zero, one, one below the capacity, the capacity, above it, the largest) over a sentinel-filled
/// target: the rows below the count are written field by field, every other row keeps the sentinel in every byte.
fn counted_rows_from_n_keep_the_sentinel<D: BatchDevice>(mut batch: Batch<D>, h: &Counted)
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    for n in [0, 1, CAP - 1, CAP, CAP + 1, u32::MAX, 0] {
        let src: Vec<u32> = (0..CAP).map(|i| i * 3 + n % 11).collect();
        let got = block_on(async {
            let mut frame = batch.begin().await?;
            frame.write(&h.n, &[n])?;
            frame.write(&h.src, &src)?;
            frame.write(&h.rows, &[SENTINEL; CAP as usize])?;
            let rows = frame.ticket(&h.rows)?;
            let _submitted = frame.submit()?;
            rows.await
        })
        .unwrap();
        let reached = n.min(CAP) as usize;
        let want: Vec<Row> = (0..CAP as usize)
            .map(|i| {
                if i < reached {
                    Row {
                        life: SENTINEL.life,
                        flags: src[i] + 7,
                        tag: i as u32,
                    }
                } else {
                    SENTINEL
                }
            })
            .collect();
        assert_eq!(got, want, "count {n}");
    }
}

#[test]
fn twin_counted_rows_from_n_keep_the_sentinel() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = counted_cpu(&device).unwrap();
    counted_rows_from_n_keep_the_sentinel(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counted_rows_from_n_keep_the_sentinel() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = counted_gpu(&device).unwrap();
    counted_rows_from_n_keep_the_sentinel(batch, &h);
}

// ---- a resize of a struct buffer (check 6) ----

#[derive(Clone, Copy)]
struct Resized {
    size: Param<(u32, u32)>,
    rows: Stage<Row, D1>,
    copy: Stage<Row, D1>,
}

macro_rules! define_resized {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Resized), Error> {
            gpu_batch!("struct_resize", on(device); in_flight(1); |b| {
                let size = b.param::<(u32, u32)>((4, 2));
                let count = size.count();
                let rows = b.input::<Row>(&count);
                let copy = gpu_task!(&count, |i| -> Row { rows[i] });
                Resized { size, rows, copy }
            })
        }
    };
}

define_resized!(resized_cpu, CpuDevice);
define_resized!(resized_gpu, HeadlessDevice);

/// A grow and a shrink between frames: the rows after a resize are zero in every byte (nothing is carried over), and the
/// same buffer holds its rows across frames at one size.
fn a_resize_zero_fills_the_rows<D: BatchDevice>(mut batch: Batch<D>, h: &Resized)
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    let mut run = |size: Option<(u32, u32)>, input: Option<&[Row]>| {
        block_on(async {
            let mut frame = batch.begin().await?;
            if let Some(size) = size {
                frame.set(&h.size, size)?;
            }
            if let Some(input) = input {
                frame.write(&h.rows, input)?;
            }
            let copy = frame.ticket(&h.copy)?;
            let _submitted = frame.submit()?;
            copy.await
        })
        .unwrap()
    };
    let small = rows_of(1)[..8].to_vec();
    assert_eq!(run(None, Some(&small)), small, "8 rows");
    assert_eq!(run(None, None), small, "the rows stay across frames");
    assert_eq!(
        run(Some((8, 3)), None),
        vec![
            Row {
                life: 0.0,
                flags: 0,
                tag: 0
            };
            24
        ],
        "a grown buffer is zero in every byte"
    );
    let grown = rows_of(2)[..24].to_vec();
    assert_eq!(run(None, Some(&grown)), grown, "24 rows");
    assert_eq!(
        run(Some((2, 2)), None),
        vec![
            Row {
                life: 0.0,
                flags: 0,
                tag: 0
            };
            4
        ],
        "a shrunk buffer is zero in every byte"
    );
}

#[test]
fn twin_a_resize_zero_fills_the_rows() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = resized_cpu(&device).unwrap();
    a_resize_zero_fills_the_rows(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_resize_zero_fills_the_rows() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = resized_gpu(&device).unwrap();
    a_resize_zero_fills_the_rows(batch, &h);
}
