//! A struct stored as the element of a batch's buffers (m14.s2s, D279): `b.input::<Row>(..)` with `Row` a
//! `#[derive(GpuStruct)]` struct, a stage that reads a field, a stage whose result is a whole `Row`, and an in-place
//! stage that stores a field, a compound and a whole row.
//!
//! The host writes and reads `Row` values, the device holds the struct itself (`StructuredBuffer<Row>`): there is no
//! decoder and no word. Every value is small and exactly representable, so a result equals the host reference bit for
//! bit. The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU tests and
//! run in full verification on the adapter the environment selects (Radeon 890M only; RTX 5090 pending).

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Error, Stage};
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

#[derive(Clone, Copy)]
struct Handles {
    rows: Stage<Row, D1>,
    life: Stage<f32, D1>,
    next: Stage<Row, D1>,
    work: Stage<Row, D1>,
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
                Handles { rows, life, next, work }
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

type Results = (Vec<f32>, Vec<Row>, Vec<Row>);

/// One frame: the rows written, the three results read.
fn frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, input: &[Row]) -> Results
where
    D::Buffer<u32>: 'static,
    D::Buffer<f32>: 'static,
    D::Buffer<Row>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, input)?;
        let life = frame.ticket(&h.life)?;
        let next = frame.ticket(&h.next)?;
        let work = frame.ticket(&h.work)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((life.await?, next.await?, work.await?))
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
        let (life, next, work) = frame(&mut batch, h, &input);
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
            frame(&mut on_gpu, &gpu_h, &input),
            frame(&mut on_cpu, &cpu_h, &input),
            "seed {seed}"
        );
    }
}
