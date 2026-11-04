// The element-wise map calls of `#[gust::gpu]` on a function (M7, D60) in the production CPU target: `map` and `map_on`
// run on `gust::cpu::CpuDevice` with no GPU API, in parallel chunks of the output (slice m7.c, `GUST_CPU_THREADS`), and
// the CPU-only chunk body `__gust_map_cpu_chunk` runs one range of elements. Built under `GUST_PROFILE=cpu` only (see
// `build.rs`).

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use gust::{
    cpu::CpuDevice,
    map::{Broadcast, Error},
};

#[gust::gpu]
pub fn scale_add(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

#[gust::gpu]
pub fn pick(flag: bool, x: u32, y: u32) -> u32 {
    if flag { x } else { y }
}

#[gust::gpu]
pub fn quotient(a: u32, b: u32) -> u32 {
    a / b
}

#[gust::gpu]
pub fn is_multiple(a: u32, b: u32) -> bool {
    a % b == 0
}

/// More than one workgroup (256 wide) and not a multiple of it.
const ELEMENTS: usize = 1000;

/// Long enough for several threads at the default bound of the parallel map (one thread per 2^18 elements, so four
/// threads here on a machine that has them) and not divisible by the thread count.
const MANY: usize = 1_048_583;

/// Runs the map calls and compares them with host references. Small integers, so every float result is exact.
pub fn run() -> Result<(), String> {
    let x: Vec<f32> = (0..ELEMENTS).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..ELEMENTS).map(|i| (3 * i % 17) as f32 - 8.0).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 2.0 * x + y).collect();
    let flags: Vec<bool> = (0..ELEMENTS).map(|i| i % 3 == 0).collect();
    let left: Vec<u32> = (0..ELEMENTS as u32).collect();

    let mapped = scale_add::map(Broadcast(2.0), &x, &y).map_err(|error| format!("scale_add::map: {error}"))?;
    if mapped != expected {
        return Err("scale_add::map differs from the host reference".to_owned());
    }
    let device = CpuDevice::new();
    let on = scale_add::map_on(&device, &x, Broadcast(1.0), &y).map_err(|error| format!("scale_add::map_on: {error}"))?;
    if on != x.iter().zip(&y).map(|(a, y)| a + y).collect::<Vec<_>>() {
        return Err("scale_add::map_on differs from the host reference".to_owned());
    }
    let picked = pick::map(&flags, &left, Broadcast(7)).map_err(|error| format!("pick::map: {error}"))?;
    let picked_expected: Vec<u32> = flags.iter().zip(&left).map(|(flag, x)| if *flag { *x } else { 7 }).collect();
    if picked != picked_expected {
        return Err("pick::map differs from the host reference".to_owned());
    }
    // The chunk body: elements 300 to 700, the broadcast scalar as a slice of one.
    let mut chunk = vec![0.0f32; 400];
    let current = Cell::new(0);
    scale_add::__gust_map_cpu_chunk(300, 700, &[2.0], &x, &y, &mut chunk, &current);
    if chunk != expected[300..700] {
        return Err("scale_add::__gust_map_cpu_chunk differs from the host reference".to_owned());
    }
    if current.get() != 699 {
        return Err(format!("the chunk body left element {} as the current one, not 699", current.get()));
    }
    // A mismatched length is an error before any element is computed.
    match scale_add::map(Broadcast(2.0), &x, &y[..10]) {
        Err(Error::LengthMismatch {
            parameter: "y",
            expected: ELEMENTS,
            found: 10,
        }) => {}
        other => return Err(format!("scale_add::map with a short argument returned {other:?}")),
    }
    parallel()?;
    println!("CPU target: map calls match (scale_add, pick, {ELEMENTS} elements)");
    Ok(())
}

/// The parallel path (a long input, several chunks on a machine with the threads), the `bool` result, the lowest
/// failing index and a clean call after a panic, each against a host reference.
fn parallel() -> Result<(), String> {
    let x: Vec<f32> = (0..MANY).map(|i| (i % 1021) as f32).collect();
    let y: Vec<f32> = (0..MANY).map(|i| (5 * i % 29) as f32 - 14.0).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 3.0 * x + y).collect();
    let mapped = scale_add::map(Broadcast(3.0), &x, &y).map_err(|error| format!("parallel scale_add::map: {error}"))?;
    if mapped != expected {
        return Err("the parallel scale_add::map differs from the host reference".to_owned());
    }
    let numbers: Vec<u32> = (0..MANY as u32).map(|i| i.wrapping_mul(2_654_435_761) >> 8).collect();
    let divisors: Vec<u32> = (0..MANY as u32).map(|i| i % 7 + 1).collect();
    let multiples = is_multiple::map(&numbers, &divisors)
        .map_err(|error| format!("parallel is_multiple::map: {error}"))?;
    if !multiples.iter().zip(numbers.iter().zip(&divisors)).all(|(got, (a, b))| *got == (a % b == 0)) {
        return Err("the parallel is_multiple::map differs from the host reference".to_owned());
    }
    // Zero divisors at 9 and 777 (and one far into the input): the call panics, nothing is returned, and the next
    // call is clean. The hook is silenced so that the expected panics do not clutter the output.
    let mut zeros = divisors.clone();
    zeros[9] = 0;
    zeros[777] = 0;
    zeros[MANY - 3] = 0;
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panicked = catch_unwind(AssertUnwindSafe(|| quotient::map(&numbers, &zeros)));
    std::panic::set_hook(hook);
    match panicked {
        Err(payload) => {
            let text = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            if !text.contains("divide by zero") {
                return Err(format!("the panic of quotient::map is not the division by zero: {text:?}"));
            }
        }
        Ok(_) => return Err("quotient::map with zero divisors returned a value".to_owned()),
    }
    let quotients = quotient::map(&numbers, &divisors).map_err(|error| format!("quotient::map after a panic: {error}"))?;
    if !quotients.iter().zip(numbers.iter().zip(&divisors)).all(|(got, (a, b))| *got == a / b) {
        return Err("quotient::map after a panic differs from the host reference".to_owned());
    }
    Ok(())
}
