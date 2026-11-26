// A batch stage over the default storage buffer limit in the precompiled consumer (m14.lim-b, D280, check 6): a stage
// that reads nine inputs and writes its result binds ten storage buffers. Its kernel is the committed WGSL artifact of
// `app/gust-artifacts/` (`cargo gust build -p app`), so this build has no Slang for it. On a default device (eight) the
// batch is refused when it is built, with the typed `DeviceLimit` and not a wgpu validation error; on a device opened with
// ten the same batch builds, runs from the artifact and equals the host model.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use gust::batch::{Batch, D1, Error, Stage};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{DeviceOptions, HeadlessDevice};

/// The rows of every column: more than one workgroup and not a multiple of it.
const ROWS: u32 = 1000;

/// What `cargo xtask check-precompiled` may read on the stdout of a passing run.
const REFUSED: &str = "wide: a ten-buffer stage from its precompiled artifact is refused at build on a default device with DeviceLimit (10 against 8)";

/// The weights of the nine inputs in the stage's sum.
const WEIGHTS: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

type Handles = ([Stage<u32, D1>; 9], Stage<u32, D1>);

/// The batch: nine inputs and the stage that sums them, weighted; one kernel, whatever the device.
fn build(device: &Arc<HeadlessDevice>) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("wide", on(device); in_flight(1); |b| {
        let a0 = b.input::<u32>(ROWS);
        let a1 = b.input::<u32>(ROWS);
        let a2 = b.input::<u32>(ROWS);
        let a3 = b.input::<u32>(ROWS);
        let a4 = b.input::<u32>(ROWS);
        let a5 = b.input::<u32>(ROWS);
        let a6 = b.input::<u32>(ROWS);
        let a7 = b.input::<u32>(ROWS);
        let a8 = b.input::<u32>(ROWS);
        let out = gpu_task!(ROWS, |i| -> u32 {
            a0[i] + a1[i] * 2u32 + a2[i] * 3u32 + a3[i] * 4u32 + a4[i] * 5u32 + a5[i] * 6u32
                + a6[i] * 7u32 + a7[i] * 8u32 + a8[i] * 9u32
        });
        ([a0, a1, a2, a3, a4, a5, a6, a7, a8], out)
    })
}

/// Column `k` of the one frame.
fn column(k: u32) -> Vec<u32> {
    (0..ROWS).map(|i| (i * (k + 3) + 5) % 1000).collect()
}

pub fn run() -> Result<(), String> {
    let default = Arc::new(
        HeadlessDevice::new().map_err(|error| format!("wide: no default device: {error}"))?,
    );
    match build(&default) {
        Err(Error::StageDevice {
            batch,
            stage,
            source,
        }) if (batch.as_str(), stage.as_str()) != ("wide", "out") => {
            return Err(format!(
                "wide: the refusal names batch `{batch}` and stage `{stage}` ({source}), not `wide` and `out`"
            ));
        }
        Err(Error::StageDevice { source, .. }) => match source.downcast_ref::<gust_wgpu::Error>() {
            Some(gust_wgpu::Error::DeviceLimit {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 10,
                maximum: 8,
                ..
            }) => println!("{REFUSED}"),
            other => {
                return Err(format!(
                    "wide: the default device refused the stage with {other:?}, not DeviceLimit 10 against 8"
                ));
            }
        },
        Err(other) => {
            return Err(format!(
                "wide: the default device refused the stage with {other}, not a stage device error"
            ));
        }
        Ok(_) => return Err("wide: a ten-buffer stage was built on a device of eight".to_owned()),
    }
    drop(default);

    let raised = Arc::new(
        HeadlessDevice::with_options(DeviceOptions::default().with_storage_buffers(10))
            .map_err(|error| format!("wide: no device with ten storage buffers: {error}"))?,
    );
    let (mut batch, (inputs, out)) = build(&raised)
        .map_err(|error| format!("wide: build on a raised device failed: {error}"))?;
    let columns: Vec<Vec<u32>> = (0..9).map(column).collect();
    let got = block_on(async {
        let mut frame = batch.begin().await?;
        for (input, values) in inputs.iter().zip(&columns) {
            frame.write(input, values)?;
        }
        let ticket = frame.ticket(&out)?;
        let _submitted = frame.submit()?;
        ticket.await
    })
    .map_err(|error| format!("wide: frame failed: {error}"))?;
    let expected: Vec<u32> = (0..ROWS as usize)
        .map(|i| columns.iter().zip(WEIGHTS).map(|(c, w)| c[i] * w).sum())
        .collect();
    if got != expected {
        let first = got
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        return Err(format!(
            "wide: the result differs from the host model at row {first}"
        ));
    }
    println!(
        "wide: a ten-buffer stage ran from its precompiled artifact on a device opened with ten storage buffers and equals the host model for {ROWS} rows"
    );
    Ok(())
}

/// Drives a future on the calling thread with a waker that does nothing, for at most a minute.
fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let started = std::time::Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "a future was still pending after a minute"
        );
        std::thread::yield_now();
    }
}
