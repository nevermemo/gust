//! v1 against the subgroup form of the reduction's block step (slice m8.e, plan section 3 and Opus 14): the timing
//! behind the rule "the subgroup form is a variant only where faster at every size on both adapters".
//!
//! `GUST_PROFILE=dev+subgroups cargo run --release -p gust-algo --example sg_ab -- <n> [reps]` with `SG_ADAPTER=<part of
//! the adapter name>` (for example `5090` or `890M`; the adapter is picked by name among the Vulkan adapters, and the
//! first line printed is the adapter that opened). It times the `u32` sum of `n` elements two ways: v1, the shipped
//! shared-memory form (`reduce::sum_with(Variant::V1, ..)`), and `evidence::sum_u32`, which under this profile runs the
//! subgroup body. One call is one sum end to end, as in the m7.r4 `ab` example: the batch with its fence wait and the
//! read-back of the one-element result. A *run* is the median of `reps` consecutive calls of one form; there are three
//! runs of each form, interleaved (v1, subgroup, v1, subgroup, v1, subgroup), after warm-up pairs that compile the
//! pipelines. The first call of each form is compared with the host sum, so a wrong answer cannot be timed. It prints
//! one `RUN` line per run and one `RESULT` line with the three run times of both forms and the rule's verdict: the
//! subgroup form is faster only when its slowest run is below v1's fastest.

use std::{env, process, sync::Arc, time::Instant};

use gust_algo::{Variant, evidence, reduce};
use gust_wgpu::{AdapterChoice, DeviceOptions, GpuVec, HeadlessDevice};

const WARMUP: usize = 5;
const RUNS: usize = 3;

/// A deterministic hash of the index, small enough that no sum of 1e8 values leaves `u32` (as the m7.r4 example).
fn small(index: usize) -> u32 {
    ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60) as u32
}

fn micros<R>(call: impl FnOnce() -> R) -> (f64, R) {
    let start = Instant::now();
    let result = call();
    (start.elapsed().as_secs_f64() * 1e6, result)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(n) = args.next().and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("usage: SG_ADAPTER=<name part> sg_ab <n> [reps]");
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
    let Ok(name) = env::var("SG_ADAPTER") else {
        eprintln!(
            "SG_ADAPTER must name the adapter (a part of its name, for example 5090 or 890M)"
        );
        process::exit(2);
    };
    let device = Arc::new(
        HeadlessDevice::with_options(
            DeviceOptions::default().with_adapter(AdapterChoice::NameContains(name)),
        )
        .expect("the named Vulkan adapter, opened with this build's capabilities"),
    );
    println!("ADAPTER {:?}", device.adapter_info());
    println!(
        "DEVICE capabilities={:?} subgroup_sizes={:?} storage_binding={}",
        device.capabilities(),
        device.subgroup_sizes(),
        device.limits().max_storage_buffer_binding_size
    );
    let selected = evidence::block_sum_kernels::block_sum::DESCRIPTOR.requires;
    println!("BODY requires={selected:?} n={n} reps={reps} runs={RUNS}");
    let words: Vec<u32> = (0..n).map(small).collect();
    let expected = words.iter().map(|w| u64::from(*w)).sum::<u64>();
    assert!(
        expected <= u64::from(u32::MAX),
        "the input must stay inside u32"
    );
    let input = GpuVec::from_slice_on(&device, &words).unwrap();
    drop(words);

    let v1 = || reduce::sum_with(Variant::V1, &input).unwrap();
    let subgroup = || evidence::sum_u32(&input).unwrap();
    for _ in 0..WARMUP {
        assert_eq!(u64::from(v1()), expected, "v1");
        assert_eq!(u64::from(subgroup()), expected, "subgroup form");
    }
    let (mut v1_runs, mut sg_runs) = (Vec::new(), Vec::new());
    for run in 1..=RUNS {
        for (label, call, runs) in [
            ("v1", &v1 as &dyn Fn() -> u32, &mut v1_runs),
            ("subgroup", &subgroup as &dyn Fn() -> u32, &mut sg_runs),
        ] {
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let (time, value) = micros(call);
                assert_eq!(u64::from(value), expected, "{label}");
                times.push(time);
            }
            let (typical, fastest) = (
                median(&times),
                times.iter().copied().fold(f64::MAX, f64::min),
            );
            println!("RUN form={label} run={run} median_us={typical:.1} min_us={fastest:.1}");
            runs.push(typical);
        }
    }
    let slowest_sg = sg_runs.iter().copied().fold(0.0, f64::max);
    let fastest_v1 = v1_runs.iter().copied().fold(f64::MAX, f64::min);
    let times = |runs: &[f64]| {
        runs.iter()
            .map(|time| format!("{time:.1}"))
            .collect::<Vec<_>>()
            .join("/")
    };
    println!(
        "RESULT n={n} reps={reps} v1_us={} v1_median_us={:.1} sg_us={} sg_median_us={:.1} \
         ratio_median={:.3} faster={}",
        times(&v1_runs),
        median(&v1_runs),
        times(&sg_runs),
        median(&sg_runs),
        median(&v1_runs) / median(&sg_runs),
        slowest_sg < fastest_v1
    );
}
