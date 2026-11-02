//! Blocking readbacks from many threads on one device (m7.rb).
//!
//! wgpu runs a `map_async` callback inside a device poll, and a poll runs the callbacks of every submission that has
//! finished, not only its own. When thread B's poll collects thread A's callback, A's `poll(Wait)` returns while the
//! callback has not run yet. A blocking readback that polled once and then checked once reported `Error::Map` ("mapping
//! not complete after poll(Wait)") in that window. The readbacks now wait on the completion itself, so these threads
//! must see no error and no wrong value.
//!
//! The regression test runs 8 threads (`GUST_CONCURRENCY_THREADS` overrides) for 4 seconds
//! (`GUST_CONCURRENCY_SECONDS`, at most 20) on one shared device. Each iteration does a typed-buffer readback and a
//! pooled `GpuVec` readback and compares the values with what was uploaded.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::disallowed_methods)]

use std::{
    env,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use gust_wgpu::{DeviceHandle, Error, GpuBufferAccess, GpuVec, HeadlessDevice};

const DEFAULT_THREADS: usize = 8;
const DEFAULT_SECONDS: u64 = 4;
const MAX_SECONDS: u64 = 20;

/// What one thread saw.
#[derive(Clone, Debug, Default)]
struct Tally {
    iterations: u64,
    /// `Error::Map`, the failure this test exists for.
    map_errors: u64,
    other_errors: u64,
    wrong_values: u64,
    first_error: Option<String>,
}

impl Tally {
    fn record(&mut self, error: Error) {
        match error {
            Error::Map(_) => self.map_errors += 1,
            _ => self.other_errors += 1,
        }
        self.first_error.get_or_insert_with(|| error.to_string());
    }

    fn add(&mut self, other: &Self) {
        self.iterations += other.iterations;
        self.map_errors += other.map_errors;
        self.other_errors += other.other_errors;
        self.wrong_values += other.wrong_values;
        if self.first_error.is_none() {
            self.first_error.clone_from(&other.first_error);
        }
    }
}

fn setting<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn device() -> DeviceHandle {
    Arc::new(
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip"),
    )
}

/// The values thread `thread` uploads in iteration `iteration`; the length changes so that the pool serves several
/// size classes.
fn values(thread: usize, iteration: u64) -> Vec<u32> {
    let len = 1 + (iteration as usize * 37 + thread * 11) % 700;
    (0..len as u32)
        .map(|index| {
            index ^ ((thread as u32) << 24) ^ (iteration as u32).wrapping_mul(2_654_435_761)
        })
        .collect()
}

/// One iteration: a typed-buffer readback, then a pooled readback.
fn round(device: &DeviceHandle, thread: usize, iteration: u64, tally: &mut Tally) {
    let expected = values(thread, iteration);
    let buffer = device.create_typed_buffer("concurrency", &expected, GpuBufferAccess::ReadWrite);
    match device.read_typed_buffer(&buffer) {
        Ok(read) if read == expected => {}
        Ok(_) => tally.wrong_values += 1,
        Err(error) => tally.record(error),
    }
    match GpuVec::from_slice_on(device, &expected).and_then(|vec| vec.to_vec()) {
        Ok(read) if read == expected => {}
        Ok(_) => tally.wrong_values += 1,
        Err(error) => tally.record(error),
    }
    tally.iterations += 1;
}

/// `threads` threads reading back in a loop on `device` for `seconds`; returns the sum of their tallies.
fn hammer(device: &DeviceHandle, threads: usize, seconds: u64) -> Tally {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let workers: Vec<_> = (0..threads)
        .map(|thread_index| {
            let device = Arc::clone(device);
            thread::spawn(move || {
                let mut tally = Tally::default();
                let mut iteration = 0;
                while Instant::now() < deadline {
                    round(&device, thread_index, iteration, &mut tally);
                    iteration += 1;
                }
                tally
            })
        })
        .collect();
    let mut total = Tally::default();
    for worker in workers {
        total.add(&worker.join().expect("a worker thread panicked"));
    }
    total
}

#[test]
fn blocking_readbacks_from_many_threads_on_one_device_never_miss_their_mapping() {
    let threads = setting("GUST_CONCURRENCY_THREADS", DEFAULT_THREADS).max(2);
    let seconds = setting("GUST_CONCURRENCY_SECONDS", DEFAULT_SECONDS).clamp(1, MAX_SECONDS);
    let device = device();
    let total = hammer(&device, threads, seconds);
    eprintln!(
        "concurrency: {threads} threads, {seconds} s: {} iterations, {} Error::Map, {} other errors, {} wrong values",
        total.iterations, total.map_errors, total.other_errors, total.wrong_values
    );
    assert!(
        total.iterations >= threads as u64,
        "every thread should finish an iteration"
    );
    assert_eq!(
        (total.map_errors, total.other_errors, total.wrong_values),
        (0, 0, 0),
        "first error: {:?}",
        total.first_error
    );
}
