// The consumer of the precompiled fixture (D45): it dispatches `kernels::vector_add::add` on the GPU and checks the
// result against a host reference. It has no Slang: the shader is the committed WGSL artifact, verified against the
// descriptor (D17) before a pipeline exists. A failed check ends the program with the error on stderr and exit code 1,
// without the `verified:` line that `cargo xtask check-precompiled` looks for.

use std::process::ExitCode;

use gust_wgpu::{Error, GpuBufferAccess, HeadlessDevice};
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

/// The message for a failed `HeadlessDevice::new`. Only the no-adapter variant says "no GPU adapter": CI runs this
/// binary with no Vulkan driver and requires exactly that prefix, so any other initialisation failure (a refused
/// device request, say) must not be able to satisfy it.
fn initialization_message(error: &Error) -> String {
    match error {
        Error::NoAdapter(_) => format!("no GPU adapter: {error}"),
        _ => format!("GPU initialization failed: {error}"),
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

    let device = HeadlessDevice::new().map_err(|error| initialization_message(&error))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_no_adapter_error_is_reported_as_no_gpu_adapter() {
        let none = initialization_message(&Error::NoAdapter("no driver".to_owned()));
        assert_eq!(
            none,
            "no GPU adapter: no Vulkan compute adapter is available: no driver"
        );
        let request = initialization_message(&Error::RequestDevice("device lost".to_owned()));
        assert_eq!(
            request,
            "GPU initialization failed: could not open the GPU device: device lost"
        );
        let poll = initialization_message(&Error::Poll("timeout".to_owned()));
        assert!(poll.starts_with("GPU initialization failed: "), "{poll}");
        // The CI step greps for this prefix on the start of the line `error: ` adds.
        assert!(!request.contains("no GPU adapter") && !poll.contains("no GPU adapter"));
    }
}
