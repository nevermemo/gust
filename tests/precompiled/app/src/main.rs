// The consumer of the precompiled fixture (D45): it dispatches `kernels::vector_add::add` on the GPU and checks the
// result against a host reference. It has no Slang: the shader is the committed WGSL artifact, verified against the
// descriptor (D17) before a pipeline exists. A failed check ends the program with the error on stderr and exit code 1,
// without the `verified:` line that `cargo xtask check-precompiled` looks for.

use std::process::ExitCode;

use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
use kernels::vector_add;

/// More than one workgroup (64) and not a multiple of it, so the kernel's bounds check matters.
const LENGTH: usize = 1000;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    // Small integers and halves: every sum is exact in f32, so the comparison below is exact.
    let a: Vec<f32> = (0..LENGTH).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..LENGTH).map(|i| 2.0 * i as f32 + 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(left, right)| left + right).collect();
    let seed = vec![0.0_f32; LENGTH];

    let descriptor = vector_add::add::DESCRIPTOR;
    println!("precompiled vector-add");
    println!("kernel: {}", descriptor.qualified_name());
    for artifact in descriptor.artifacts {
        println!(
            "artifact: {:?} key={} bytes={}",
            artifact.format,
            artifact.key,
            artifact.bytes.len()
        );
    }

    let device = HeadlessDevice::new().map_err(|error| format!("no GPU adapter: {error}"))?;
    println!("adapter: {}", device.adapter_info().name);
    let a = device.create_f32_buffer("precompiled a", &a, GpuBufferAccess::ReadOnly);
    let b = device.create_f32_buffer("precompiled b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_f32_buffer("precompiled out", &seed, GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(&device, LENGTH as u32, &a, &b, &out)
        .map_err(|error| format!("dispatch failed: {error}"))?;
    let result = device
        .read_f32_buffer(&out)
        .map_err(|error| format!("read-back failed: {error}"))?;

    if result != expected {
        let first = result
            .iter()
            .zip(&expected)
            .position(|(got, want)| got != want)
            .unwrap_or(0);
        return Err(format!(
            "the GPU result differs from the host reference at index {first} (got {:?}, expected {:?})",
            result.get(first),
            expected.get(first)
        ));
    }
    println!("verified: GPU result equals the host reference for {LENGTH} elements");
    Ok(())
}
