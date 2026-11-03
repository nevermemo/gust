//! v0 against v1 of the reductions and scans (plan section 5.5, slice m7.r4): the timing behind the shipping rule.
//!
//! `cargo run --release -p gust-algo --example ab -- <n> [reps]` times, for one input length, every operation of
//! [`gust_algo::variant::OPERATIONS`] on the adapter's own limits, calling `reduce::*_with` and `scan::*_with` of both
//! variants alternately (the order flips every repetition), after warm-up calls that compile the pipelines. One
//! repetition is one public call end to end: the batch with its fence wait and, for a reduction, the readback of the
//! one-element result. It prints one `RESULT` line per operation and element type: the medians and minima in
//! microseconds of both variants and their ratio. v1 is faster by the rule of the plan when `v0_median / v1_median`
//! is at least 1.10 at every length. A fresh device per group of operations keeps the video memory bounded (an input
//! of 1e8 elements is 400 MB, a scan output another 400 MB).

use std::{env, process, sync::Arc, time::Instant};

use gust_algo::{
    Variant,
    reduce::{self, Element},
    scan,
};
use gust_wgpu::{DeviceOptions, GpuVec, HeadlessDevice};

const WARMUP: usize = 5;

fn device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::with_options(DeviceOptions::default()).expect("a Vulkan adapter"))
}

/// A deterministic hash of the index, small enough that no sum or prefix of 1e8 values leaves `u32`.
fn small(index: usize) -> u32 {
    ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60) as u32
}

struct Samples {
    v0: Vec<f64>,
    v1: Vec<f64>,
}

fn micros<R>(call: impl FnOnce() -> R) -> (f64, R) {
    let start = Instant::now();
    let result = call();
    (start.elapsed().as_secs_f64() * 1e6, result)
}

/// Warm up both variants, then alternate them `reps` times; the first call of each also compares the answers.
fn bench<R: PartialEq + std::fmt::Debug>(
    label: &str,
    n: usize,
    reps: usize,
    mut call: impl FnMut(Variant) -> R,
    exact: bool,
) {
    let (mut first0, mut first1) = (None, None);
    for _ in 0..WARMUP {
        first0 = Some(call(Variant::V0));
        first1 = Some(call(Variant::V1));
    }
    if exact {
        assert_eq!(first0, first1, "{label}: v0 and v1 differ");
    }
    let mut samples = Samples {
        v0: Vec::with_capacity(reps),
        v1: Vec::with_capacity(reps),
    };
    for rep in 0..reps {
        let order = if rep % 2 == 0 {
            [Variant::V0, Variant::V1]
        } else {
            [Variant::V1, Variant::V0]
        };
        for variant in order {
            let (time, _) = micros(|| call(variant));
            match variant {
                Variant::V0 => samples.v0.push(time),
                Variant::V1 => samples.v1.push(time),
            }
        }
    }
    let stat = |values: &mut Vec<f64>| {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (values[values.len() / 2], values[0])
    };
    let (median0, min0) = stat(&mut samples.v0);
    let (median1, min1) = stat(&mut samples.v1);
    println!(
        "RESULT op={label} n={n} reps={reps} v0_median_us={median0:.1} v1_median_us={median1:.1} \
         v0_min_us={min0:.1} v1_min_us={min1:.1} ratio_median={:.3} ratio_min={:.3}",
        median0 / median1,
        min0 / min1
    );
}

fn reductions<T: Element + std::fmt::Debug>(name: &str, values: &[T], reps: usize, exact: bool) {
    let device = device();
    let n = values.len();
    let input = GpuVec::from_slice_on(&device, values).unwrap();
    bench(
        &format!("sum_{name}"),
        n,
        reps,
        |variant| reduce::sum_with(variant, &input).unwrap(),
        exact,
    );
    bench(
        &format!("min_{name}"),
        n,
        reps,
        |variant| reduce::min_with(variant, &input).unwrap(),
        true,
    );
    bench(
        &format!("max_{name}"),
        n,
        reps,
        |variant| reduce::max_with(variant, &input).unwrap(),
        true,
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(n) = args.next().and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("usage: ab <n> [reps]");
        process::exit(2);
    };
    let reps = args
        .next()
        .and_then(|text| text.parse::<usize>().ok())
        .unwrap_or(if n <= 1_000_000 {
            200
        } else if n <= 10_000_000 {
            100
        } else {
            30
        });
    {
        let device = device();
        println!("ADAPTER {:?}", device.adapter_info());
        println!(
            "LIMITS {:?}",
            device.limits().max_storage_buffer_binding_size
        );
    }
    // f32: the sums differ in the last bits between the variants (different groupings), so they are not compared.
    let floats: Vec<f32> = (0..n).map(|i| small(i) as f32 / 16.0).collect();
    reductions("f32", &floats, reps, false);
    drop(floats);
    let ints: Vec<i32> = (0..n).map(|i| small(i) as i32 - 8).collect();
    reductions("i32", &ints, reps, true);
    drop(ints);
    let words: Vec<u32> = (0..n).map(small).collect();
    reductions("u32", &words, reps, true);
    {
        let device = device();
        let input = GpuVec::from_slice_on(&device, &words).unwrap();
        bench(
            "sum_wrapping_u32",
            n,
            reps,
            |variant| reduce::sum_wrapping_with(variant, &input).unwrap(),
            true,
        );
    }
    {
        let device = device();
        let input = GpuVec::from_slice_on(&device, &words).unwrap();
        // The scan returns its output without a readback; the call ends with the fence wait of the batch. The last
        // element is read once per variant, outside the timing, to check the answer.
        bench(
            "inclusive_u32",
            n,
            reps,
            |variant| scan::inclusive_with(variant, &input).unwrap().len(),
            true,
        );
        bench(
            "exclusive_u32",
            n,
            reps,
            |variant| scan::exclusive_with(variant, &input).unwrap().len(),
            true,
        );
        let expected = words.iter().map(|w| u64::from(*w)).sum::<u64>();
        for variant in [Variant::V0, Variant::V1] {
            let out = scan::inclusive_with(variant, &input)
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(
                u64::from(out[n - 1]),
                expected,
                "{variant:?} inclusive total"
            );
            let out = scan::exclusive_with(variant, &input)
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(
                u64::from(out[n - 1]) + u64::from(words[n - 1]),
                expected,
                "{variant:?} exclusive total"
            );
        }
        println!("CHECK scan totals equal the host sum for both variants");
    }
}
