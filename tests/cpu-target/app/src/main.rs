// The production CPU target, end to end (steps E-a to E-c, E-h of the plan): the kernels of `kernels` run on
// `gust::cpu::CpuDevice` in an ordinary build, and every result is compared with a host reference written here, apart
// from the kernels. Each kernel prints `CPU target: matches (<kernel>, <n> elements, max ULP <k>)`; any mismatch prints
// the error on stderr and exits with 1, without the final line.
//
// `app --bench` runs the sequential-premise measurement (S5.2) in the same binary: the vector-add and polynomial
// kernels at 2^21 elements through `dispatch` on `CpuDevice`, against a hand-written loop over the same data, five
// runs each, median.

use std::{hint::black_box, process::ExitCode, time::Instant};

use gust::{
    Independent,
    cpu::{CpuDevice, ulps},
    math::Vec4,
};
use kernels::{atomic_counter, polynomial, vector_add, vector_ops};

/// More than one workgroup (64 and 128 wide) and not a multiple of either, so the kernels' bounds checks matter.
const ELEMENTS: usize = 1000;
/// The atomic counter's invocations: more than one value of any workgroup size, and a prime.
const INVOCATIONS: u32 = 257;
/// The size of the S5.2 measurement.
const BENCH_ELEMENTS: usize = 1 << 21;
/// Runs per measurement; the median is reported.
const RUNS: usize = 5;

fn main() -> ExitCode {
    let bench = std::env::args().any(|argument| argument == "--bench");
    match run(bench) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(bench: bool) -> Result<(), String> {
    check_vector_add()?;
    check_polynomial()?;
    check_atomic_counter()?;
    check_vector_ops()?;
    println!("CPU target: all kernels match");
    if bench {
        bench_vector_add()?;
        bench_polynomial()?;
    }
    Ok(())
}

/// Small integers and halves: every sum is exact in `f32`, so the reference is exact.
fn vector_add_inputs(count: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..count).map(|i| i as f32).collect();
    let b = (0..count).map(|i| 2.0 * i as f32 + 0.5).collect();
    (a, b)
}

/// `v` in [-1, 1) on a grid of 4096 steps, like `examples/polynomial`.
fn polynomial_inputs(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (index % 4096) as f32 / 2048.0 - 1.0)
        .collect()
}

/// The polynomial in the written order of the kernel, in `f32`: the CPU target evaluates exactly this (SEMANTICS
/// section 4, the twin column), so the reference is bit-exact.
fn polynomial_f32(value: f32) -> f32 {
    let squared = value * value;
    let cubed = squared * value;
    let curved = cubed - (2.0 * squared);
    let linear = 0.5 * value;
    (curved + linear) + 1.0
}

/// The polynomial in `f64`, with the sum of the magnitudes of its terms: the scale of its rounding error.
fn polynomial_f64(value: f32) -> (f64, f64) {
    let v = f64::from(value);
    let terms = [v * v * v, -2.0 * v * v, 0.5 * v, 1.0];
    (
        terms.iter().sum(),
        terms.iter().map(|term| term.abs()).sum(),
    )
}

fn max_ulp(actual: &[f32], expected: &[f32]) -> Result<u64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{} results for {} expected",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| ulps(*actual, *expected))
        .max()
        .unwrap_or(0))
}

fn check_vector_add() -> Result<(), String> {
    let (a, b) = vector_add_inputs(ELEMENTS);
    let expected: Vec<f32> = a.iter().zip(&b).map(|(left, right)| left + right).collect();
    let device = CpuDevice::new();
    let (a, b) = (device.buffer(&a), device.buffer(&b));
    let out = device.buffer(&vec![0.0f32; ELEMENTS]);
    vector_add::add::dispatch(&device, ELEMENTS as u32, &a, &b, &out)
        .map_err(|error| format!("vector_add::add: {error:?}"))?;
    let worst = max_ulp(&out.read(), &expected)?;
    if worst != 0 {
        return Err(format!(
            "vector_add::add differs by {worst} ULP; it is exact"
        ));
    }
    println!("CPU target: matches (vector_add::add, {ELEMENTS} elements, max ULP {worst})");
    Ok(())
}

fn check_polynomial() -> Result<(), String> {
    let input = polynomial_inputs(ELEMENTS);
    let expected: Vec<f32> = input.iter().map(|value| polynomial_f32(*value)).collect();
    let device = CpuDevice::new();
    let buffer = device.buffer(&input);
    let out = device.buffer(&vec![0.0f32; ELEMENTS]);
    polynomial::evaluate::dispatch(&device, ELEMENTS as u32, &buffer, &out)
        .map_err(|error| format!("polynomial::evaluate: {error:?}"))?;
    let actual = out.read();
    let worst = max_ulp(&actual, &expected)?;
    if worst != 0 {
        return Err(format!(
            "polynomial::evaluate differs from the written-order reference by {worst} ULP; the CPU target is exact"
        ));
    }
    // An independent second reference: the polynomial in f64. The result must lie within 48 u of the scale of its
    // terms (u = 2^-24), the bound derived for this expression.
    let unit = f64::from(f32::EPSILON) / 2.0;
    for (value, result) in input.iter().zip(&actual) {
        let (exact, scale) = polynomial_f64(*value);
        let error = (f64::from(*result) - exact).abs();
        if error > 48.0 * unit * scale {
            return Err(format!(
                "polynomial::evaluate({value}) = {result}, {error:e} from the f64 value {exact}"
            ));
        }
    }
    println!("CPU target: matches (polynomial::evaluate, {ELEMENTS} elements, max ULP {worst})");
    Ok(())
}

fn check_atomic_counter() -> Result<(), String> {
    let device = CpuDevice::new();
    // The counter has one element, not `INVOCATIONS`: `Independent` is the call-site opt-in for that.
    let counter = device.buffer(&[0u32]);
    atomic_counter::increment::dispatch(&device, INVOCATIONS, Independent(&counter))
        .map_err(|error| format!("atomic_counter::increment: {error:?}"))?;
    let result = counter.read();
    if result != [INVOCATIONS] {
        return Err(format!(
            "atomic_counter::increment left {result:?}, expected [{INVOCATIONS}]"
        ));
    }
    println!(
        "CPU target: matches (atomic_counter::increment, {INVOCATIONS} invocations, max ULP 0)"
    );
    Ok(())
}

/// The lanes of element `index` of the vector kernel's inputs: small integers and halves, so the kernel's sums and
/// products are exact in `f32` and the comparison below is exact.
fn vector_lanes(index: usize, shift: usize) -> [f32; 4] {
    let i = index + shift;
    [
        (i % 7) as f32,
        (i % 5) as f32 - 2.0,
        (i % 3) as f32 * 0.5,
        (i % 11) as f32,
    ]
}

/// The vector kernel `vector_ops::blend` of `kernels` (M6), in the production CPU build, against a reference written
/// lane by lane on plain arrays: `blended = (a + reverse(b)) * 2 + 0.5` and `dot = sum of a * b` in lane order.
///
/// The line is not a `CPU target: matches (..)` line: `xtask/src/cpu_target.rs` (`KERNELS`) requires exactly one of
/// those per listed kernel and rejects a kernel it does not list, and that file is outside the slice that added this
/// kernel. A mismatch fails the run (exit 1, no final line) all the same.
fn check_vector_ops() -> Result<(), String> {
    let a: Vec<[f32; 4]> = (0..ELEMENTS).map(|index| vector_lanes(index, 0)).collect();
    let b: Vec<[f32; 4]> = (0..ELEMENTS).map(|index| vector_lanes(index, 3)).collect();
    let to_vec4 = |lanes: &[f32; 4]| Vec4::new(lanes[0], lanes[1], lanes[2], lanes[3]);
    let device = CpuDevice::new();
    let a_buffer = device.buffer(&a.iter().map(to_vec4).collect::<Vec<_>>());
    let b_buffer = device.buffer(&b.iter().map(to_vec4).collect::<Vec<_>>());
    let blended = device.buffer(&vec![Vec4::splat(0.0); ELEMENTS]);
    let dots = device.buffer(&vec![0.0f32; ELEMENTS]);
    vector_ops::blend::dispatch(
        &device,
        ELEMENTS as u32,
        &a_buffer,
        &b_buffer,
        &blended,
        &dots,
    )
    .map_err(|error| format!("vector_ops::blend: {error:?}"))?;
    let (blended, dots) = (blended.read(), dots.read());
    for index in 0..ELEMENTS {
        let (a, b) = (a[index], b[index]);
        let want: [f32; 4] = std::array::from_fn(|lane| (a[lane] + b[3 - lane]) * 2.0 + 0.5);
        let want_dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        let got = blended[index];
        if [got.x, got.y, got.z, got.w] != want || dots[index] != want_dot {
            return Err(format!(
                "vector_ops::blend element {index}: got {:?} and dot {}, expected {want:?} and dot {want_dot}",
                [got.x, got.y, got.z, got.w],
                dots[index]
            ));
        }
    }
    println!("CPU target: vector kernel exact (vector_ops::blend, {ELEMENTS} elements)");
    Ok(())
}

/// The median of `RUNS` timings, in milliseconds.
fn median_ms(mut run_once: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        run_once()?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[RUNS / 2])
}

fn report(kernel: &str, dispatch_ms: f64, hand_ms: f64) {
    println!(
        "S5.2 {kernel} elements={BENCH_ELEMENTS} runs={RUNS} dispatch_ms={dispatch_ms:.3} \
         hand_loop_ms={hand_ms:.3} ratio={:.2}",
        dispatch_ms / hand_ms
    );
}

fn bench_vector_add() -> Result<(), String> {
    let (a, b) = vector_add_inputs(BENCH_ELEMENTS);
    let device = CpuDevice::new();
    let (a_buffer, b_buffer) = (device.buffer(&a), device.buffer(&b));
    let out_buffer = device.buffer(&vec![0.0f32; BENCH_ELEMENTS]);
    let dispatch_ms = median_ms(|| {
        vector_add::add::dispatch(
            &device,
            BENCH_ELEMENTS as u32,
            &a_buffer,
            &b_buffer,
            &out_buffer,
        )
        .map_err(|error| format!("vector_add::add: {error:?}"))
    })?;
    let mut hand = vec![0.0f32; BENCH_ELEMENTS];
    let hand_ms = median_ms(|| {
        let (a, b) = (black_box(&a), black_box(&b));
        for index in 0..BENCH_ELEMENTS {
            hand[index] = a[index] + b[index];
        }
        black_box(&hand);
        Ok(())
    })?;
    if out_buffer.read() != hand {
        return Err("the bench results of vector_add differ".to_owned());
    }
    report("vector_add", dispatch_ms, hand_ms);
    Ok(())
}

fn bench_polynomial() -> Result<(), String> {
    let input = polynomial_inputs(BENCH_ELEMENTS);
    let device = CpuDevice::new();
    let in_buffer = device.buffer(&input);
    let out_buffer = device.buffer(&vec![0.0f32; BENCH_ELEMENTS]);
    let dispatch_ms = median_ms(|| {
        polynomial::evaluate::dispatch(&device, BENCH_ELEMENTS as u32, &in_buffer, &out_buffer)
            .map_err(|error| format!("polynomial::evaluate: {error:?}"))
    })?;
    let mut hand = vec![0.0f32; BENCH_ELEMENTS];
    let hand_ms = median_ms(|| {
        let input = black_box(&input);
        for index in 0..BENCH_ELEMENTS {
            hand[index] = polynomial_f32(input[index]);
        }
        black_box(&hand);
        Ok(())
    })?;
    if out_buffer.read() != hand {
        return Err("the bench results of polynomial differ".to_owned());
    }
    report("polynomial", dispatch_ms, hand_ms);
    Ok(())
}
