//! `gpu_batch!`: a sealed batch of three stages, run frame after frame with one frame in flight (m14.s2a2).
//!
//! The block declares everything a frame needs, once:
//!
//! - inputs `a` (`N` floats) and `bias` (`WIDTH` integers), and the parameter `gain`;
//! - **scale** (1-D): `a[i] * gain + 0.25`;
//! - **grid** (2-D): `bias[x] + y * 100`, row-major, `WIDTH` by `HEIGHT`;
//! - **halve** (in place): `scaled[i] = scaled[i] * 0.5 + 1.0`, on the result of **scale**.
//!
//! A frame writes the inputs, sets `gain`, reserves two tickets (the in-place result and the grid) and submits: on the
//! GPU device that is one queue submission whatever the number of stages and tickets. Every value is exactly
//! representable in `f32` and every operation on it is exact, so a result equals the host reference bit for bit,
//! whatever the rounding mode of the device. The same macro-generated code runs on the CPU twin (`CpuDevice`, test
//! builds) and on the GPU (`HeadlessDevice`).

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use gust::batch::{Batch, D1, D2, Error, Param, Stage};
#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::gpu_batch;
use gust_wgpu::{DeviceHandle, HeadlessDevice, default_device};

/// Elements of the 1-D stages.
pub const N: u32 = 1000;
/// Columns and rows of the 2-D stage.
pub const WIDTH: u32 = 37;
pub const HEIGHT: u32 = 19;

/// What the block returns: the handles a frame uses.
#[derive(Clone, Copy, Debug)]
pub struct Handles {
    pub gain: Param<f32>,
    pub a: Stage<f32, D1>,
    pub bias: Stage<u32, D1>,
    pub scaled: Stage<f32, D1>,
    pub grid: Stage<u32, D2>,
}

/// The inputs of a frame.
#[derive(Clone, Debug)]
pub struct Inputs {
    pub a: Vec<f32>,
    pub bias: Vec<u32>,
}

/// The two results of a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Outputs {
    pub scaled: Vec<f32>,
    pub grid: Vec<u32>,
}

/// The inputs: small integers, so every product and sum below is exact in `f32`.
pub fn inputs() -> Inputs {
    Inputs {
        a: (0..N).map(|i| i as f32).collect(),
        bias: (0..WIDTH).map(|x| x * 3).collect(),
    }
}

/// The host reference, written from the rules of the stages.
pub fn reference(inputs: &Inputs, gain: f32) -> Outputs {
    let scaled = inputs
        .a
        .iter()
        .map(|a| (a * gain + 0.25) * 0.5 + 1.0)
        .collect();
    let mut grid = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            grid.push(inputs.bias[x as usize] + y * 100);
        }
    }
    Outputs { scaled, grid }
}

/// The longest `block_on` waits: a future that is not ready after this long is a bug, and the process panics instead
/// of spinning (a ticket that never resolves must fail a test, not hold a GPU).
const DEADLINE: std::time::Duration = std::time::Duration::from_secs(20);

/// Drives a future on this thread with a no-op waker (a poll polls the device once and never waits), for at most
/// [`DEADLINE`].
///
/// # Panics
///
/// When the future is still pending after [`DEADLINE`].
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let started = std::time::Instant::now();
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                assert!(
                    started.elapsed() < DEADLINE,
                    "a future was still pending after {DEADLINE:?}"
                );
                std::thread::yield_now();
            }
        }
    }
}

/// Polls a future once with a no-op waker: its value when it is ready now, `None` when it is not (never waits, so a
/// ticket that should have resolved and did not fails a test instead of hanging it).
pub fn ready_now<F: Future>(future: F) -> Option<F::Output> {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

/// Defines the functions that build the batch for a device of the given type and run frames of it.
macro_rules! define_batch {
    ($build:ident, $frame:ident, $blocking:ident, $device_type:ty) => {
        /// The batch and its handles.
        pub fn $build(device: &Arc<$device_type>) -> Result<(Batch<$device_type>, Handles), Error> {
            gpu_batch!("basics", on(device); in_flight(1); |b| {
                let gain = b.param::<f32>(3.0);
                let a = b.input::<f32>(N);
                let bias = b.input::<u32>(WIDTH);
                let scaled = gust::gpu_task!(N, |i| -> f32 { a[i] * gain.get() + 0.25 });
                let grid = gust::gpu_task!((WIDTH, HEIGHT), |x, y| -> u32 { bias[x] + y * 100u32 });
                let _halve = gust::gpu_task!(&mut scaled, |i| { scaled[i] = scaled[i] * 0.5 + 1.0; });
                Handles { gain, a, bias, scaled, grid }
            })
        }

        /// One frame: the writes and the `set` that are given, two tickets, one submission.
        pub fn $frame(
            batch: &mut Batch<$device_type>,
            handles: &Handles,
            inputs: Option<&Inputs>,
            gain: Option<f32>,
        ) -> Result<Outputs, Error> {
            block_on(async {
                let mut frame = batch.begin().await?;
                if let Some(inputs) = inputs {
                    frame.write(&handles.a, &inputs.a)?;
                    frame.write(&handles.bias, &inputs.bias)?;
                }
                if let Some(gain) = gain {
                    frame.set(&handles.gain, gain)?;
                }
                let scaled = frame.ticket(&handles.scaled)?;
                let grid = frame.ticket(&handles.grid)?;
                let _submitted = frame.submit()?;
                Ok(Outputs {
                    scaled: scaled.await?,
                    grid: grid.await?,
                })
            })
        }

        /// The blocking form: a frame of the data as it stands, then a blocking read of each result.
        pub fn $blocking(batch: &mut Batch<$device_type>, handles: &Handles) -> Result<Outputs, Error> {
            let done = batch.dispatch()?;
            Ok(Outputs {
                scaled: done.read(&handles.scaled)?,
                grid: done.read(&handles.grid)?,
            })
        }
    };
}

#[cfg(any(test, gust_dual))]
define_batch!(build_cpu, frame_cpu, blocking_cpu, CpuDevice);
define_batch!(build_gpu, frame_gpu, blocking_gpu, HeadlessDevice);

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle: DeviceHandle = default_device()?;
    let info = handle.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let inputs = inputs();
    let (mut batch, handles) = build_gpu(&handle)?;
    let got = frame_gpu(&mut batch, &handles, Some(&inputs), None)?;
    if got != reference(&inputs, 3.0) {
        return Err("the first frame differs from the host reference".into());
    }
    // The inputs stay on the device: the next frame writes nothing and sets only the gain.
    let got = frame_gpu(&mut batch, &handles, None, Some(5.0))?;
    if got != reference(&inputs, 5.0) {
        return Err("the second frame differs from the host reference".into());
    }
    let got = blocking_gpu(&mut batch, &handles)?;
    if got != reference(&inputs, 5.0) {
        return Err("the blocking frame differs from the host reference".into());
    }
    println!(
        "gpu: scale and halve {N} values, grid {WIDTH} x {HEIGHT}; three frames, equal to the host"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::gpu_task;

    fn cpu() -> Arc<CpuDevice> {
        Arc::new(CpuDevice::new())
    }

    #[test]
    fn twin_matches_the_host_over_three_frames() {
        let inputs = inputs();
        let device = cpu();
        let (mut batch, handles) = build_cpu(&device).unwrap();
        let first = frame_cpu(&mut batch, &handles, Some(&inputs), None).unwrap();
        assert_eq!(first, reference(&inputs, 3.0));
        // The data persists: a frame that writes nothing and sets the gain only.
        let second = frame_cpu(&mut batch, &handles, None, Some(5.0)).unwrap();
        assert_eq!(second, reference(&inputs, 5.0));
        // An unchanged frame is the same frame.
        let third = frame_cpu(&mut batch, &handles, None, None).unwrap();
        assert_eq!(third, second);
    }

    #[test]
    fn twin_the_blocking_form_reads_the_frame_it_ran() {
        let inputs = inputs();
        let device = cpu();
        let (mut batch, handles) = build_cpu(&device).unwrap();
        frame_cpu(&mut batch, &handles, Some(&inputs), None).unwrap();
        assert_eq!(
            blocking_cpu(&mut batch, &handles).unwrap(),
            reference(&inputs, 3.0)
        );
    }

    #[test]
    fn twin_the_batch_reports_its_stages_in_order() {
        let device = cpu();
        let (batch, _) = build_cpu(&device).unwrap();
        let names: Vec<String> = batch
            .stages()
            .iter()
            .map(|stage| stage.name.clone())
            .collect();
        assert_eq!(names, ["scaled", "grid", "_halve"]);
        let dims: Vec<[u32; 2]> = batch.stages().iter().map(|stage| stage.dims).collect();
        assert_eq!(dims, [[N, 1], [WIDTH, HEIGHT], [N, 1]]);
    }

    #[test]
    fn twin_a_dropped_frame_fails_its_tickets_and_the_next_frame_is_exact() {
        let inputs = inputs();
        let device = cpu();
        let (mut batch, handles) = build_cpu(&device).unwrap();
        let ticket = {
            let mut frame = block_on(batch.begin()).unwrap();
            frame.write(&handles.a, &inputs.a).unwrap();
            frame.ticket(&handles.scaled).unwrap()
        };
        assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));
        let got = frame_cpu(&mut batch, &handles, Some(&inputs), None).unwrap();
        assert_eq!(got, reference(&inputs, 3.0));
    }

    /// After a frame that ran and was read, a dropped frame's ticket on the same handle is ready at once with
    /// `NotSubmitted` (no bytes, whatever an earlier read left behind). Polled once, so a ticket that stayed pending
    /// fails the test and does not hang it.
    #[test]
    fn twin_a_dropped_frame_after_a_read_resolves_not_submitted_at_once() {
        let inputs = inputs();
        let device = cpu();
        let (mut batch, handles) = build_cpu(&device).unwrap();
        let first = frame_cpu(&mut batch, &handles, Some(&inputs), Some(3.0)).unwrap();
        assert_eq!(first, reference(&inputs, 3.0));
        let ticket = {
            let mut frame = block_on(batch.begin()).unwrap();
            frame.ticket(&handles.scaled).unwrap()
        };
        match ready_now(ticket) {
            Some(Err(Error::NotSubmitted { .. })) => {}
            other => panic!("expected an immediate Err(NotSubmitted), got {other:?}"),
        }
        let again = frame_cpu(&mut batch, &handles, None, None).unwrap();
        assert_eq!(again, first);
    }

    #[test]
    fn twin_the_report_counts_the_buffers_and_their_bytes() {
        let device = cpu();
        let (batch, _) = build_cpu(&device).unwrap();
        let report = batch.report();
        assert_eq!(report.stages, 3);
        assert_eq!(report.buffers, 4);
        let words = u64::from(N) * 2 + u64::from(WIDTH) + u64::from(WIDTH * HEIGHT);
        assert_eq!(report.bytes, words * 4);
        let names: Vec<&str> = report
            .per_buffer
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, ["a", "bias", "scaled", "grid"]);
        let bytes: Vec<u64> = report.per_buffer.iter().map(|(_, bytes)| *bytes).collect();
        assert_eq!(
            bytes,
            [
                u64::from(N) * 4,
                u64::from(WIDTH) * 4,
                u64::from(N) * 4,
                u64::from(WIDTH * HEIGHT) * 4
            ]
        );
    }

    #[test]
    fn twin_a_frame_before_the_last_is_over_is_refused() {
        let device = cpu();
        let (mut batch, _) = build_cpu(&device).unwrap();
        let _live = block_on(batch.begin()).unwrap();
        assert!(matches!(
            block_on(batch.begin()),
            Err(Error::FrameLive { .. })
        ));
    }

    #[test]
    fn twin_a_builder_kept_beyond_its_block_is_sealed() {
        let device = cpu();
        let (_batch, (_a, kept)) = gpu_batch!("sealed", on(&device); |b| {
            let a = b.input::<f32>(N);
            let kept = b.clone();
            (a, kept)
        })
        .unwrap();
        let _late = kept.input::<f32, _>(N);
        assert!(matches!(kept.check(), Err(Error::Sealed { .. })));
    }

    #[test]
    fn twin_a_batch_started_inside_a_build_is_nested() {
        let device = cpu();
        let (_outer, inner) = gpu_batch!("outer", on(&device); |b| {
            let _a = b.input::<f32>(N);
            gpu_batch!("inner", on(&device); |c| {
                let _x = c.input::<f32>(N);
            })
        })
        .unwrap();
        assert!(matches!(inner, Err(Error::NestedBuild { .. })));
    }

    #[test]
    fn twin_a_handle_of_another_batch_is_foreign() {
        let device = cpu();
        let (_one, handles) = build_cpu(&device).unwrap();
        let foreign = handles.a;
        let result = gpu_batch!("two", on(&device); |b| {
            let own = b.input::<f32>(N);
            let _ = gpu_task!(N, [foreign: Stage<f32>], |i| -> f32 { foreign[i] });
            own
        });
        assert!(matches!(result, Err(Error::Foreign { .. })), "{result:?}");
    }

    #[test]
    fn twin_a_stage_reading_what_it_writes_is_a_hazard() {
        let device = cpu();
        let result = gpu_batch!("hazard", on(&device); |b| {
            let a = b.input::<f32>(N);
            let alias = a;
            let _h = gpu_task!(&mut a, [alias: Stage<f32>], |i| { a[i] = alias[i] + 1.0; });
            a
        });
        let Err(Error::Hazard { stage, buffer, .. }) = result else {
            panic!("a hazard was expected");
        };
        assert_eq!((stage.as_str(), buffer.as_str()), ("_h", "a"));
    }

    #[test]
    fn twin_a_write_of_another_length_and_a_new_shape_are_mismatches() {
        let device = cpu();
        let (mut batch, (size, pixels, doubled)) = gpu_batch!("shape", on(&device); |b| {
            let size = b.param::<(u32, u32)>((4, 2));
            let pixels = b.input::<u32>(&size);
            let doubled = gpu_task!(&size, |x, y| -> u32 { pixels[(x, y)] * 2u32 });
            (size, pixels, doubled)
        })
        .unwrap();
        let mut frame = block_on(batch.begin()).unwrap();
        assert!(matches!(
            frame.write(&pixels, &[1; 7]),
            Err(Error::ShapeMismatch { .. })
        ));
        frame.write(&pixels, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        // A shape can change between frames (slice s2b), but not under a buffer the frame has already written.
        assert!(matches!(
            frame.set(&size, (8, 1)),
            Err(Error::ShapeMismatch { .. })
        ));
        // The same shape again is not a change.
        frame.set(&size, (4, 2)).unwrap();
        let ticket = frame.ticket(&doubled).unwrap();
        let _submitted = frame.submit().unwrap();
        assert_eq!(block_on(ticket).unwrap(), [2, 4, 6, 8, 10, 12, 14, 16]);
    }

    #[test]
    fn twin_a_handle_the_block_did_not_return_is_not_an_output() {
        let device = cpu();
        let mut leaked = None;
        let (mut batch, (a, shown)) = gpu_batch!("output", on(&device); |b| {
            let a = b.input::<f32>(4u32);
            let hidden = gpu_task!(4u32, |i| -> f32 { a[i] + 1.0 });
            let shown = gpu_task!(4u32, |i| -> f32 { hidden[i] * 2.0 });
            leaked = Some(hidden);
            (a, shown)
        })
        .unwrap();
        let hidden = leaked.unwrap();
        let mut frame = block_on(batch.begin()).unwrap();
        assert!(matches!(
            frame.ticket(&hidden),
            Err(Error::NotOutput { .. })
        ));
        frame.write(&a, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let ticket = frame.ticket(&shown).unwrap();
        let _submitted = frame.submit().unwrap();
        assert_eq!(block_on(ticket).unwrap(), [4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn twin_a_write_to_a_stage_is_not_an_input() {
        let device = cpu();
        let (mut batch, handles) = build_cpu(&device).unwrap();
        let mut frame = block_on(batch.begin()).unwrap();
        let data = vec![0.0_f32; N as usize];
        assert!(matches!(
            frame.write(&handles.scaled, &data),
            Err(Error::NotInput { .. })
        ));
    }

    /// A device of this test's own, so that no other test's work moves its submission counter.
    fn private_device() -> Arc<HeadlessDevice> {
        Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        }))
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_a_frame_is_one_submission_at_submit_and_none_before() {
        let inputs = inputs();
        let device = private_device();
        let (mut batch, handles) = build_gpu(&device).unwrap();
        for round in 0..3 {
            let before = device.gust_submissions();
            let mut frame = block_on(batch.begin()).unwrap();
            frame.write(&handles.a, &inputs.a).unwrap();
            frame.write(&handles.bias, &inputs.bias).unwrap();
            frame.set(&handles.gain, 3.0).unwrap();
            let scaled = frame.ticket(&handles.scaled).unwrap();
            let grid = frame.ticket(&handles.grid).unwrap();
            assert_eq!(
                device.gust_submissions() - before,
                0,
                "round {round}: before submit"
            );
            let _submitted = frame.submit().unwrap();
            assert_eq!(
                device.gust_submissions() - before,
                1,
                "round {round}: at submit"
            );
            let got = Outputs {
                scaled: block_on(scaled).unwrap(),
                grid: block_on(grid).unwrap(),
            };
            assert_eq!(got, reference(&inputs, 3.0), "round {round}");
            assert_eq!(
                device.gust_submissions() - before,
                1,
                "round {round}: after the read-back"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_a_dropped_frame_is_not_submitted_and_the_next_one_is_exact() {
        let inputs = inputs();
        let device = private_device();
        let (mut batch, handles) = build_gpu(&device).unwrap();
        let before = device.gust_submissions();
        let ticket = {
            let mut frame = block_on(batch.begin()).unwrap();
            frame.write(&handles.a, &inputs.a).unwrap();
            frame.write(&handles.bias, &inputs.bias).unwrap();
            frame.ticket(&handles.scaled).unwrap()
        };
        assert!(
            matches!(ready_now(ticket), Some(Err(Error::NotSubmitted { .. }))),
            "the ticket of a dropped frame is ready at once with NotSubmitted"
        );
        assert_eq!(
            device.gust_submissions() - before,
            0,
            "a dropped frame made no submission"
        );
        let got = frame_gpu(&mut batch, &handles, Some(&inputs), Some(3.0)).unwrap();
        assert_eq!(got, reference(&inputs, 3.0));
        assert_eq!(device.gust_submissions() - before, 1);
        // The staging buffer of that read is back in the pool holding real values. A dropped frame with a ticket on
        // the same handle resolves at once (polled once: a pending ticket fails the test and does not hang it),
        // makes no submission and hands out none of those bytes.
        let ticket = {
            let mut frame = block_on(batch.begin()).unwrap();
            frame.ticket(&handles.scaled).unwrap()
        };
        match ready_now(ticket) {
            Some(Err(Error::NotSubmitted { .. })) => {}
            other => panic!("expected an immediate Err(NotSubmitted), got {other:?}"),
        }
        assert_eq!(
            device.gust_submissions() - before,
            1,
            "the dropped frame made no submission"
        );
        let again = frame_gpu(&mut batch, &handles, None, None).unwrap();
        assert_eq!(again, got);
        assert_eq!(device.gust_submissions() - before, 2);
    }

    /// One frame that writes what it is given, sets the gain and reads every handle the block returned (the two
    /// inputs and the two results).
    macro_rules! frame_everything {
        ($batch:expr, $handles:expr, $inputs:expr, $gain:expr) => {
            block_on(async {
                let mut frame = $batch.begin().await?;
                if let Some(inputs) = $inputs {
                    frame.write(&$handles.a, &inputs.a)?;
                    frame.write(&$handles.bias, &inputs.bias)?;
                }
                frame.set(&$handles.gain, $gain)?;
                let a = frame.ticket(&$handles.a)?;
                let bias = frame.ticket(&$handles.bias)?;
                let scaled = frame.ticket(&$handles.scaled)?;
                let grid = frame.ticket(&$handles.grid)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>((a.await?, bias.await?, scaled.await?, grid.await?))
            })
        };
    }

    /// Plan 2.4: one batch, built on both devices, reports the same stages and memory and gives the same value in
    /// every handle, frame after frame.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_and_the_cpu_twin_agree_on_every_output() {
        let inputs = inputs();
        let cpu_device = cpu();
        let gpu_device = private_device();
        let (mut on_cpu, cpu_handles) = build_cpu(&cpu_device).unwrap();
        let (mut on_gpu, gpu_handles) = build_gpu(&gpu_device).unwrap();
        assert_eq!(on_cpu.stages(), on_gpu.stages());
        assert_eq!(on_cpu.report(), on_gpu.report());
        for (round, (write, gain)) in [(true, 3.0_f32), (false, 5.0), (false, 5.0)]
            .into_iter()
            .enumerate()
        {
            let given = write.then_some(&inputs);
            let from_cpu = frame_everything!(on_cpu, cpu_handles, given, gain).unwrap();
            let from_gpu = frame_everything!(on_gpu, gpu_handles, given, gain).unwrap();
            assert_eq!(from_cpu, from_gpu, "round {round}");
            assert_eq!(from_cpu.0, inputs.a, "round {round}: the input a");
            assert_eq!(from_cpu.1, inputs.bias, "round {round}: the input bias");
            let want = reference(&inputs, gain);
            assert_eq!((from_cpu.2, from_cpu.3), (want.scaled, want.grid));
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        let inputs = inputs();
        let (mut batch, handles) = build_gpu(&handle).unwrap();
        let first = frame_gpu(&mut batch, &handles, Some(&inputs), None).unwrap();
        assert_eq!(first, reference(&inputs, 3.0));
        let second = frame_gpu(&mut batch, &handles, None, Some(5.0)).unwrap();
        assert_eq!(second, reference(&inputs, 5.0));
        let blocking = blocking_gpu(&mut batch, &handles).unwrap();
        assert_eq!(blocking, second);
        println!("gpu on {}: three frames equal to the host", info.name);
    }
}
