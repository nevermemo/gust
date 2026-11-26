// A counted batch stage in the precompiled consumer, a binary of its own beside `app` (`app`'s `main.rs` is also the
// consumer of E8, which has no `gust-artifacts/` of its own) (m14.s2i, D278): both kernels of the stage (the counted one and the
// one-invocation args kernel that writes its indirect grid) are the committed WGSL artifacts of `app/gust-artifacts/`
// (`cargo gust build -p app`), so this binary needs no Slang for them either. The stage runs under the indirect launch
// that a native device takes, and the frames below show the clamp and the untouched rows.

use std::sync::Arc;

use gust::batch::Error;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

/// The rows of the stage's capacity: more than one workgroup and not a multiple of it.
const CAPACITY: u32 = 1000;

/// Runs three frames with the counts 300, 100 and 5000 (above the capacity), and compares each frame's result with the
/// host model, where the rows from the count on keep what they held.
fn check(device: &Arc<HeadlessDevice>) -> Result<(), String> {
    let (mut batch, (count, x, z)) = gpu_batch!("counted", on(device); in_flight(1); |b| {
        let count = b.input::<u32>(1u32);
        let x = b.input::<u32>(CAPACITY);
        let z = gpu_task!(count(count); CAPACITY, |i| -> u32 { x[i] * 3u32 + 1u32 });
        (count, x, z)
    })
    .map_err(|error| format!("counted build failed: {error}"))?;
    let mut model = vec![0u32; CAPACITY as usize];
    for (frame, n) in [300u32, 100, 5000].into_iter().enumerate() {
        let input: Vec<u32> = (0..CAPACITY).map(|i| i + 10 * frame as u32).collect();
        let got = block_on(async {
            let mut f = batch.begin().await?;
            f.write(&count, &[n])?;
            f.write(&x, &input)?;
            let ticket = f.ticket(&z)?;
            let _submitted = f.submit()?;
            Ok::<_, Error>(ticket.await?)
        })
        .map_err(|error| format!("counted frame {frame} failed: {error}"))?;
        for (row, value) in model.iter_mut().zip(&input).take(n as usize) {
            *row = value * 3 + 1;
        }
        if got != model {
            let first = got.iter().zip(&model).position(|(a, b)| a != b).unwrap_or(0);
            return Err(format!(
                "counted frame {frame} (n = {n}) differs from the model at row {first}"
            ));
        }
    }
    println!(
        "counted: a counted stage ran from its precompiled artifacts, three counts (one above the capacity) equal the host model for {CAPACITY} rows"
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

fn main() -> std::process::ExitCode {
    let device = match HeadlessDevice::new() {
        Ok(device) => Arc::new(device),
        Err(error) => {
            eprintln!("error: GPU initialization failed: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!("adapter: {}", device.adapter_info().name);
    match check(&device) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
