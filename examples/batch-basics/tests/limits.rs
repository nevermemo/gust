//! A batch stage over the device's storage buffer limit (m14.lim-b, D280): the batch refuses it when it is built, naming
//! the stage, its count, the limit and both cures; on a device opened with enough buffers the same batch builds and runs.
//!
//! The stage `out` reads nine inputs and writes its result: ten storage buffers in one shader stage. (A counted stage
//! counts its count buffer among them; its args kernel binds the count and the args buffer, two more of its own.) The
//! `twin_` test runs on the CPU twin, which has no such limit; the
//! `gpu_` tests are `#[ignore]`d like the example's other GPU tests and run in full verification on the adapter the
//! environment selects, and each must see a device that offers at least ten.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Error, Stage};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{DeviceOptions, HeadlessDevice};

/// The rows of every column.
const LEN: u32 = 777;
/// The frames each batch runs.
const FRAMES: u32 = 16;
/// The storage buffers of the stage: nine inputs and its result.
const BUFFERS: u64 = 10;

#[derive(Clone, Copy)]
struct Handles {
    inputs: [Stage<u32, D1>; 9],
    out: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("wide", on(device); in_flight(1); |b| {
                let a0 = b.input::<u32>(LEN);
                let a1 = b.input::<u32>(LEN);
                let a2 = b.input::<u32>(LEN);
                let a3 = b.input::<u32>(LEN);
                let a4 = b.input::<u32>(LEN);
                let a5 = b.input::<u32>(LEN);
                let a6 = b.input::<u32>(LEN);
                let a7 = b.input::<u32>(LEN);
                let a8 = b.input::<u32>(LEN);
                let out = gpu_task!(LEN, |i| -> u32 {
                    a0[i] + a1[i] * 2u32 + a2[i] * 3u32 + a3[i] * 4u32 + a4[i] * 5u32 + a5[i] * 6u32
                        + a6[i] * 7u32 + a7[i] * 8u32 + a8[i] * 9u32
                });
                Handles { inputs: [a0, a1, a2, a3, a4, a5, a6, a7, a8], out }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice);
define_build!(build_gpu, HeadlessDevice);

/// Column `k` of frame `frame`.
fn column(k: u32, frame: u32) -> Vec<u32> {
    (0..LEN)
        .map(|i| (i * (k + 3) + frame * 11) % 1000)
        .collect()
}

/// The host model of the stage over frame `frame`.
fn model(frame: u32) -> Vec<u32> {
    let columns: Vec<Vec<u32>> = (0..9).map(|k| column(k, frame)).collect();
    (0..LEN as usize)
        .map(|i| (0..9).map(|k| columns[k][i] * (k as u32 + 1)).sum())
        .collect()
}

fn frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, number: u32) -> Vec<u32>
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        for (k, input) in h.inputs.iter().enumerate() {
            frame.write(input, &column(k as u32, number))?;
        }
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        out.await
    })
    .unwrap()
}

fn device_with(buffers: u32) -> Arc<HeadlessDevice> {
    Arc::new(
        HeadlessDevice::with_options(DeviceOptions::default().with_storage_buffers(buffers))
            .unwrap_or_else(|error| {
                panic!("a GPU adapter offering {buffers} storage buffers is required: {error}")
            }),
    )
}

/// The device limit the build refused with: the batch and stage it names, and the limit's own numbers.
fn refusal(error: &Error, limit_device: u64) -> String {
    let Error::StageDevice {
        batch,
        stage,
        source,
    } = error
    else {
        panic!("a stage the device refuses at build is `StageDevice`: {error:?}")
    };
    assert_eq!((batch.as_str(), stage.as_str()), ("wide", "out"));
    match source.downcast_ref::<gust_wgpu::Error>() {
        Some(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => assert_eq!(
            (*limit, *requested, *maximum),
            (
                "max_storage_buffers_per_shader_stage",
                BUFFERS,
                limit_device
            )
        ),
        other => panic!("the source of the refusal is a `DeviceLimit`: {other:?}"),
    }
    error.to_string()
}

#[test]
fn twin_the_ten_column_stage_matches_the_model() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu(&device).unwrap();
    for number in 0..FRAMES {
        assert_eq!(
            frame(&mut batch, &h, number),
            model(number),
            "frame {number}"
        );
    }
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_default_device_refuses_the_stage_at_build_with_both_cures() {
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    let Err(error) = build_gpu(&device) else {
        panic!("a ten-buffer stage must not build on a device opened with eight")
    };
    let message = refusal(&error, 8);
    for part in [
        "stage `out`",
        "batch `wide`",
        "needs 10",
        "the limit is 8",
        "max_storage_buffers_per_shader_stage",
        "#[derive(GpuStruct)]",
        "DeviceOptions::with_storage_buffers",
    ] {
        assert!(message.contains(part), "`{part}` missing from: {message}");
    }
    // Refused at build: nothing was submitted and no pipeline was made for the stage.
    assert_eq!(device.gust_submissions(), 0);
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_device_one_short_refuses_the_stage_with_its_own_limit() {
    let device = device_with(9);
    let Err(error) = build_gpu(&device) else {
        panic!("a ten-buffer stage must not build on a device opened with nine")
    };
    let message = refusal(&error, 9);
    assert!(message.contains("the limit is 9"), "{message}");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_raised_device_builds_and_runs_sixteen_frames_equal_to_the_twin() {
    let device = device_with(10);
    let (mut batch, h) = build_gpu(&device).unwrap();
    let twin = Arc::new(CpuDevice::new());
    let (mut twin_batch, th) = build_cpu(&twin).unwrap();
    for number in 0..FRAMES {
        let got = frame(&mut batch, &h, number);
        assert_eq!(got, model(number), "frame {number}: GPU against the model");
        assert_eq!(
            got,
            frame(&mut twin_batch, &th, number),
            "frame {number}: GPU against the twin"
        );
    }
}

// ---- a counted stage and a library op name their stage too ----

/// The batch, stage and the storage buffer limit numbers (`requested`, `maximum`) of a `StageDevice` refusal.
fn stage_refusal(error: &Error) -> (String, String, u64, u64) {
    let Error::StageDevice {
        batch,
        stage,
        source,
    } = error
    else {
        panic!("a stage the device refuses at build is `StageDevice`: {error:?}")
    };
    match source.downcast_ref::<gust_wgpu::Error>() {
        Some(gust_wgpu::Error::DeviceLimit {
            limit: "max_storage_buffers_per_shader_stage",
            requested,
            maximum,
            ..
        }) => (batch.clone(), stage.clone(), *requested, *maximum),
        other => panic!("the source of the refusal is a storage buffer `DeviceLimit`: {other:?}"),
    }
}

/// A counted stage over nine inputs: its count buffer, the nine inputs and its result are eleven storage buffers of the
/// counted kernel (the args kernel binds the count and the args buffer, two).
fn build_counted(
    device: &Arc<HeadlessDevice>,
) -> Result<(Batch<HeadlessDevice>, (Stage<u32, D1>, Stage<u32, D1>)), Error> {
    gpu_batch!("wide-counted", on(device); in_flight(1); |b| {
        let n = b.input::<u32>(1u32);
        let a0 = b.input::<u32>(LEN);
        let a1 = b.input::<u32>(LEN);
        let a2 = b.input::<u32>(LEN);
        let a3 = b.input::<u32>(LEN);
        let a4 = b.input::<u32>(LEN);
        let a5 = b.input::<u32>(LEN);
        let a6 = b.input::<u32>(LEN);
        let a7 = b.input::<u32>(LEN);
        let a8 = b.input::<u32>(LEN);
        let out = gpu_task!(count(n); LEN, |i| -> u32 {
            a0[i] + a1[i] * 2u32 + a2[i] * 3u32 + a3[i] * 4u32 + a4[i] * 5u32 + a5[i] * 6u32
                + a6[i] * 7u32 + a7[i] * 8u32 + a8[i] * 9u32
        });
        (n, out)
    })
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_counted_stage_over_the_limit_is_refused_at_build_by_name() {
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    let Err(error) = build_counted(&device) else {
        panic!("an eleven-buffer counted stage must not build on a device opened with eight")
    };
    assert_eq!(
        stage_refusal(&error),
        ("wide-counted".to_owned(), "out".to_owned(), 11, 8)
    );
    assert!(
        error.to_string().contains("with_storage_buffers"),
        "{error}"
    );
    // Refused before the args kernel's pipeline was made (and before anything was submitted): the limit of the counted
    // kernel is checked first.
    assert_eq!(device.gust_submissions(), 0);
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

/// A stage built over zero rows skips the device's limit checks (nothing is launched); when a resize gives it rows the
/// batch prepares it again, and that refusal names the stage as the one at build does.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_stage_built_empty_and_resized_over_the_limit_is_refused_by_name() {
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    let (mut batch, (len, _out)) = gpu_batch!("wide-resized", on(&device); in_flight(1); |b| {
        let len = b.param::<u32>(0);
        let a0 = b.input::<u32>(&len);
        let a1 = b.input::<u32>(&len);
        let a2 = b.input::<u32>(&len);
        let a3 = b.input::<u32>(&len);
        let a4 = b.input::<u32>(&len);
        let a5 = b.input::<u32>(&len);
        let a6 = b.input::<u32>(&len);
        let a7 = b.input::<u32>(&len);
        let a8 = b.input::<u32>(&len);
        let out = gpu_task!(&len, |i| -> u32 {
            a0[i] + a1[i] + a2[i] + a3[i] + a4[i] + a5[i] + a6[i] + a7[i] + a8[i]
        });
        (len, out)
    })
    .expect("a stage over zero rows builds on a device of eight");
    let error = block_on(async {
        let mut frame = batch.begin().await?;
        frame.set(&len, LEN)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>(())
    })
    .expect_err("the resize gives the stage rows, ten buffers over the limit");
    let Error::StageDevice {
        batch: batch_name,
        stage,
        source,
    } = &error
    else {
        panic!("a stage the device refuses after a resize is `StageDevice`: {error:?}")
    };
    assert_eq!(
        (batch_name.as_str(), stage.as_str()),
        ("wide-resized", "out")
    );
    assert!(
        matches!(
            source.downcast_ref::<gust_wgpu::Error>(),
            Some(gust_wgpu::Error::DeviceLimit {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 10,
                maximum: 8,
                ..
            })
        ),
        "{source:?}"
    );
}

/// A library op has no input wide enough to pass 8 buffers (its kernels bind a few), so the limit is lowered instead:
/// `with_limits` may only lower, and a device whose limit is one buffer refuses `gust::sum`, whose kernels bind two or
/// more, when the batch is built.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_library_op_over_the_limit_is_refused_at_build_by_name() {
    let opened = HeadlessDevice::new().expect("a GPU adapter is required");
    let mut limits = opened.limits().clone();
    limits.max_storage_buffers_per_shader_stage = 1;
    let device = Arc::new(
        opened
            .with_limits(limits)
            .expect("a lowered limit is accepted"),
    );
    let Err(error) = gpu_batch!("op-limit", on(&device); in_flight(1); |b| {
        let values = b.input::<u32>(LEN);
        let total = gust::sum(&values);
        (values, total)
    }) else {
        panic!("a library op must not build on a device of one storage buffer")
    };
    let (batch, stage, requested, maximum) = stage_refusal(&error);
    assert_eq!(
        (batch.as_str(), stage.as_str(), maximum),
        ("op-limit", "total", 1)
    );
    assert!(requested >= 2, "{requested}");
}
