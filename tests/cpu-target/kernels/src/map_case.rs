// The element-wise map calls of `#[gust::gpu]` on a function (M7, D60) in the production CPU target: `map` and `map_on`
// run on `gust::cpu::CpuDevice` with no GPU API, and the CPU-only chunk body `__gust_map_cpu_chunk` (the hook of the
// parallel CPU map, slice m7.c) runs one range of elements. Built under `GUST_PROFILE=cpu` only (see `build.rs`).

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

/// More than one workgroup (256 wide) and not a multiple of it.
const ELEMENTS: usize = 1000;

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
    scale_add::__gust_map_cpu_chunk(300, 700, &[2.0], &x, &y, &mut chunk);
    if chunk != expected[300..700] {
        return Err("scale_add::__gust_map_cpu_chunk differs from the host reference".to_owned());
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
    println!("CPU target: map calls match (scale_add, pick, {ELEMENTS} elements)");
    Ok(())
}
