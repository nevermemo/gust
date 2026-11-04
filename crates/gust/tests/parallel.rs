//! The parallel CPU map runner (M7, slice m7.c; Unit tier, no adapter, no slangc, no GPU).
//!
//! `gust::cpu::parallel::run_map_chunked` is what the generated `cpu` map calls. These tests drive it with chunk bodies
//! of the generated shape (`chunk[i - lo]` is the only write, `cur` records the element) and check the three promises
//! of the plan's section 6: the result is the one a single thread gives, bit for bit; the panic of the lowest failing
//! index is the one that comes out, with nothing returned; and `GUST_CPU_THREADS` is a positive integer or an error.
//! The thread count rule is tested through functions that take the value, never through the process environment.

use std::any::Any;
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

use gust::cpu::{
    CpuDevice, Error,
    parallel::{
        ELEMENTS_PER_THREAD, bounded_threads, input, run_map_chunked as try_run_map_chunked,
        run_map_into_vec as try_run_map_into_vec, threads_from,
    },
};
use gust::map::MapHost;
use gust::{Access, LengthRule, Resource};

/// A deterministic float vector with values of every magnitude, so rounding differences would show.
fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n as u32)
        .map(|i| {
            let x = i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8;
            x as f32 / 16_777_216.0 - 0.25
        })
        .collect()
}

/// The element function: a few dependent float operations, order of evaluation fixed by the source.
fn scalar(a: f32, b: f32) -> f32 {
    let mut acc = b;
    for step in 0..8 {
        acc = acc * a + step as f32 * 0.125;
    }
    acc
}

/// The generated chunk body's shape with `f(i, a[i], b[i])` as the user's function: whole-length reads at the global
/// index, one write at the own index, `cur` set before each element.
fn run(a: &[f32], b: &[f32], threads: usize, f: impl Fn(u32, f32, f32) -> f32 + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    run_map_chunked(&mut out, threads, |lo, hi, chunk, cur| {
        for i in lo..hi {
            cur.set(i);
            chunk[(i - lo) as usize] = f(i, a[i as usize], b[i as usize]);
        }
    });
    out
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn available() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

// ---------------------------------------------------------------------------------------------------------------
// The result is the one a single thread gives.
// ---------------------------------------------------------------------------------------------------------------

/// Lengths: empty, one, small, a prime, around the per-thread element count (below, at, above), and lengths that
/// threads 2, 4 and the default cannot divide evenly.
fn lengths() -> Vec<usize> {
    vec![
        0,
        1,
        2,
        3,
        7,
        997,
        65_536,
        ELEMENTS_PER_THREAD - 1,
        ELEMENTS_PER_THREAD,
        ELEMENTS_PER_THREAD + 1,
        2 * ELEMENTS_PER_THREAD + 5,
        1_000_003,
    ]
}

#[test]
fn every_thread_count_gives_the_bits_of_one_thread() {
    let default = available();
    for n in lengths() {
        let (a, b) = (values(n, 12_345), values(n, 777));
        let single = run(&a, &b, 1, |_, a, b| scalar(a, b));
        assert_eq!(single.len(), n);
        // More threads than elements is allowed too (the runner never makes an empty piece); only for small `n`.
        let more = if n <= 1000 {
            vec![n.max(1), n + 5]
        } else {
            vec![]
        };
        for threads in [1, 2, 3, 4, 7, default].into_iter().chain(more) {
            let chunked = run(&a, &b, threads, |_, a, b| scalar(a, b));
            assert_eq!(
                bits(&chunked),
                bits(&single),
                "n = {n}, threads = {threads}"
            );
        }
    }
}

#[test]
fn a_broadcast_scalar_is_the_slice_of_one_element_in_every_chunk() {
    // The generated body reads `input[if input.len() == 1 { 0 } else { i }]`; the runner only shares the slices.
    let n = 100_003;
    let a = values(n, 1);
    let scale = [1.5f32];
    let mut out = vec![0.0f32; n];
    run_map_chunked(&mut out, 4, |lo, hi, chunk, cur| {
        for i in lo..hi {
            cur.set(i);
            let s = scale[if scale.len() == 1 { 0 } else { i as usize }];
            chunk[(i - lo) as usize] = a[i as usize] * s;
        }
    });
    let expected: Vec<f32> = a.iter().map(|value| value * 1.5).collect();
    assert_eq!(bits(&out), bits(&expected));
}

#[test]
fn the_only_write_is_the_own_element_and_every_element_is_written_once() {
    // Counts the visits of each element: a chunk boundary error would visit one twice or skip one.
    for (n, threads) in [
        (1_000_003usize, 4usize),
        (999_983, 7),
        (10, 3),
        (5, 5),
        (2, 8),
    ] {
        let mut out = vec![0u32; n];
        run_map_chunked(&mut out, threads, |lo, hi, chunk, cur| {
            assert_eq!(chunk.len(), (hi - lo) as usize);
            for i in lo..hi {
                cur.set(i);
                chunk[(i - lo) as usize] += i + 1;
            }
        });
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, value)| *value == i as u32 + 1),
            "n = {n}, threads = {threads}"
        );
    }
}

#[test]
fn run_map_into_vec_takes_the_zeroed_output_of_the_device() {
    let device = CpuDevice::new();
    for n in [0u32, 1, 9, 300_001] {
        let a = values(n as usize, 5);
        let zeroed = device.zeroed::<f32>(n).unwrap();
        let out = run_map_into_vec(zeroed, 4, |lo, hi, chunk, cur| {
            for i in lo..hi {
                cur.set(i);
                chunk[(i - lo) as usize] = a[i as usize] * 2.0;
            }
        });
        let expected: Vec<f32> = a.iter().map(|value| value * 2.0).collect();
        assert_eq!(bits(&out), bits(&expected), "n = {n}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Panics: the lowest failing index, no output.
// ---------------------------------------------------------------------------------------------------------------

fn text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else {
        "<non-text payload>".to_owned()
    }
}

/// Runs the map with a function that panics at the indexes in `bad` (message `bad index {i}`); the lowest of them
/// sleeps first, so a higher chunk finishes and panics before it. Returns the panic text, or `None` when the call
/// returned a value.
fn panic_of(n: usize, threads: usize, bad: &[u32]) -> Option<String> {
    let (a, b) = (values(n, 3), values(n, 4));
    let lowest = bad.iter().copied().min().unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        run(&a, &b, threads, |i, a, b| {
            if bad.contains(&i) {
                if i == lowest {
                    std::thread::sleep(Duration::from_millis(5));
                }
                panic!("bad index {i}");
            }
            scalar(a, b)
        })
    }));
    result.err().map(|payload| text(&*payload))
}

#[test]
fn panics_at_777_and_9_report_9_for_every_thread_count_with_no_output() {
    for threads in [1, 2, 4, 8, available()] {
        for repetition in 0..20 {
            // n = 1000 with four threads: 9 is in chunk 0 and 777 in chunk 3.
            let panic = panic_of(1000, threads, &[777, 9]);
            assert_eq!(
                panic.as_deref(),
                Some("bad index 9"),
                "threads = {threads}, repetition = {repetition}"
            );
        }
    }
}

#[test]
fn the_payload_is_the_one_of_the_lowest_index_unchanged() {
    struct Marker(u32);
    let (a, b) = (values(1000, 3), values(1000, 4));
    let mut out = vec![0.0f32; 1000];
    let result = catch_unwind(AssertUnwindSafe(|| {
        run_map_chunked(&mut out, 4, |lo, hi, chunk, cur| {
            for i in lo..hi {
                cur.set(i);
                if i == 9 || i == 777 {
                    std::panic::panic_any(Marker(i));
                }
                chunk[(i - lo) as usize] = scalar(a[i as usize], b[i as usize]);
            }
        });
    }));
    let payload = result.expect_err("the call panics");
    assert_eq!(
        payload.downcast_ref::<Marker>().map(|marker| marker.0),
        Some(9)
    );
}

#[test]
fn the_first_of_two_panics_in_one_chunk_is_reported() {
    for threads in [1, 2, 4] {
        // 300 and 310 lie in one chunk for every count above (n = 1000), and a later chunk panics too.
        for _ in 0..20 {
            let panic = panic_of(1000, threads, &[310, 300, 900]);
            assert_eq!(
                panic.as_deref(),
                Some("bad index 300"),
                "threads = {threads}"
            );
        }
    }
}

#[test]
fn a_clean_call_after_a_panic_equals_one_thread() {
    let (a, b) = (values(10_007, 8), values(10_007, 9));
    for threads in [2, 4] {
        assert!(panic_of(10_007, threads, &[9_000, 12]).is_some());
        let single = run(&a, &b, 1, |_, a, b| scalar(a, b));
        let chunked = run(&a, &b, threads, |_, a, b| scalar(a, b));
        assert_eq!(bits(&chunked), bits(&single), "threads = {threads}");
    }
}

#[test]
fn a_single_thread_lets_the_panic_unwind_unchanged() {
    assert_eq!(panic_of(50, 1, &[40, 7]).as_deref(), Some("bad index 7"));
}

// ---------------------------------------------------------------------------------------------------------------
// The thread count.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn gust_cpu_threads_is_a_positive_integer_or_an_error() {
    assert_eq!(threads_from(None, 24), Ok(24));
    assert_eq!(threads_from(None, 0), Ok(1));
    assert_eq!(threads_from(Some(OsStr::new("1")), 24), Ok(1));
    assert_eq!(threads_from(Some(OsStr::new("4")), 24), Ok(4));
    assert_eq!(threads_from(Some(OsStr::new("64")), 2), Ok(64));
    assert_eq!(threads_from(Some(OsStr::new("007")), 2), Ok(7));
    for junk in [
        "0",
        "00",
        "",
        " 4",
        "4 ",
        "+4",
        "-1",
        "4x",
        "x4",
        "four",
        "1.5",
        "1e3",
        "0x10",
        "99999999999999999999999999",
        "\u{663}",
    ] {
        let error = threads_from(Some(OsStr::new(junk)), 24).unwrap_err();
        assert_eq!(
            error,
            Error::InvalidThreads {
                value: junk.to_owned()
            }
        );
        let message = error.to_string();
        assert!(message.contains("GUST_CPU_THREADS"), "{message}");
        assert!(message.contains(&format!("`{junk}`")), "{message}");
    }
}

#[cfg(unix)]
#[test]
fn a_value_that_is_not_unicode_is_an_error_naming_it() {
    use std::os::unix::ffi::OsStrExt;
    let value = OsStr::from_bytes(&[0x34, 0xff]);
    let error = threads_from(Some(value), 24).unwrap_err();
    assert!(matches!(error, Error::InvalidThreads { .. }));
    assert!(error.to_string().contains("GUST_CPU_THREADS"));
}

#[test]
fn the_thread_count_is_bounded_by_the_input() {
    let f = ELEMENTS_PER_THREAD;
    assert_eq!(bounded_threads(24, 0), 1);
    assert_eq!(bounded_threads(24, 100), 1);
    assert_eq!(bounded_threads(24, f - 1), 1);
    assert_eq!(bounded_threads(24, f), 1);
    assert_eq!(bounded_threads(24, 2 * f - 1), 1);
    assert_eq!(bounded_threads(24, 2 * f), 2);
    assert_eq!(bounded_threads(24, 4 * f + 3), 4);
    assert_eq!(bounded_threads(24, 3 * f + 1), 3);
    assert_eq!(bounded_threads(24, 100 * f), 24);
    assert_eq!(bounded_threads(4, 100 * f), 4);
    assert_eq!(bounded_threads(1, 100 * f), 1);
    assert_eq!(bounded_threads(0, 100 * f), 1);
}

// ---------------------------------------------------------------------------------------------------------------
// The inputs the generated map borrows.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn an_input_of_another_device_is_refused_and_a_borrowed_one_is_reported() {
    let (device, other) = (CpuDevice::new(), CpuDevice::new());
    let buffer = device.buffer(&[1.0f32, 2.0, 3.0]);
    let borrowed = input::<f32, _>(&device, &buffer, "a").unwrap();
    assert_eq!(*borrowed, [1.0, 2.0, 3.0]);
    drop(borrowed);
    assert_eq!(
        input::<f32, _>(&other, &buffer, "a").unwrap_err(),
        Error::WrongDevice { parameter: "a" }
    );
    // A buffer that something else is writing: the binding of a running call, here taken by hand.
    let writing =
        gust::cpu::write::<f32>(buffer.bind(Access::ReadWrite, LengthRule::Independent), "a")
            .unwrap();
    assert_eq!(
        input::<f32, _>(&device, &buffer, "x").unwrap_err(),
        Error::Borrowed { parameter: "x" }
    );
    drop(writing);
    assert!(input::<f32, _>(&device, &buffer, "x").is_ok());
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers: the runner returns a `Result` (a failed thread spawn); the tests below expect the spawns to work.
// ---------------------------------------------------------------------------------------------------------------

fn run_map_chunked<R, C>(out: &mut [R], threads: usize, chunk: C)
where
    R: Send,
    C: Fn(u32, u32, &mut [R], &std::cell::Cell<u32>) + Sync,
{
    try_run_map_chunked(out, threads, chunk).expect("worker threads start");
}

fn run_map_into_vec<R, C>(out: gust::cpu::CpuBuffer<R>, threads: usize, chunk: C) -> Vec<R>
where
    R: gust::GpuPod + Send,
    C: Fn(u32, u32, &mut [R], &std::cell::Cell<u32>) + Sync,
{
    try_run_map_into_vec(out, threads, chunk).expect("worker threads start")
}
