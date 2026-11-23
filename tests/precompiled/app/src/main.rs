// The consumer of the precompiled fixture (D45): it dispatches `kernels::vector_add::add` on the GPU and checks the
// result against a host reference. It has no Slang: the shader is the committed WGSL artifact, verified against the
// descriptor (D17) before a pipeline exists. A failed check ends the program with the error on stderr and exit code 1,
// without the `verified:` line that `cargo xtask check-precompiled` looks for. It then launches the same kernel past a
// lowered workgroup limit, which the host runs as pieces of the kernel's split variant from the split's own committed
// artifact (D208, D209), and prints the `split:` line that the check also requires.

use std::process::ExitCode;

use gust::{Access, Extent, KernelDescriptor, LengthRule, Resource};
use gust_wgpu::{BufferDispatch, Error, GpuBufferAccess, HeadlessDevice};
use kernels::vector_add;

/// What `cargo xtask check-precompiled` requires before the limit that refused the launch.
const SPLIT_REFUSED_PREFIX: &str = "split fallback: no split artifact and no run-time compiler, so the launch past the limit is refused with DeviceLimit on ";

/// The split variant of the kernel without its precompiled artifact, and the kernel with that split.
const SPLIT_WITHOUT_ARTIFACT: KernelDescriptor = match vector_add::add::DESCRIPTOR.split {
    Some(split) => split.with_artifacts(&[]),
    None => panic!("the kernel has no split variant"),
};
const WITHOUT_SPLIT_ARTIFACT: KernelDescriptor =
    vector_add::add::DESCRIPTOR.with_split(Some(&SPLIT_WITHOUT_ARTIFACT));

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
    let a_buffer = device.create_typed_buffer("precompiled a", &a, GpuBufferAccess::ReadOnly);
    let b_buffer = device.create_typed_buffer("precompiled b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("precompiled out", &seed, GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(&device, LENGTH as u32, &a_buffer, &b_buffer, &out)
        .map_err(|error| format!("dispatch failed: {error}"))?;
    let result = device
        .read_typed_buffer(&out)
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
    split_launch(device, &a, &b, &expected, &seed)
}

/// The same launch on a device whose per-dimension workgroup limit is lowered below the launch (D208): the host cuts it
/// into pieces of the kernel's split variant. This binary has no run-time compiler, so the pieces can only run from the
/// split's own precompiled artifact (D209); a build that lacks it fails the launch with the `DeviceLimit` refusal.
fn split_launch(
    device: HeadlessDevice,
    a: &[f32],
    b: &[f32],
    expected: &[f32],
    seed: &[f32],
) -> Result<(), String> {
    let descriptor = vector_add::add::DESCRIPTOR;
    let split = descriptor
        .split
        .ok_or_else(|| "the kernel has no split variant".to_owned())?;
    for artifact in split.artifacts {
        println!(
            "split artifact: {:?} key={} bytes={}",
            artifact.format,
            artifact.key,
            artifact.bytes.len()
        );
    }
    if split.artifacts.is_empty() {
        return Err("the split variant carries no precompiled artifact".to_owned());
    }
    // 64 invocations per workgroup: `LENGTH` is 16 workgroups, so a limit of 4 makes four pieces.
    let workgroups = LENGTH.div_ceil(descriptor.workgroup_size[0] as usize) as u32;
    let limit = 4;
    let mut limits = device.limits().clone();
    limits.max_compute_workgroups_per_dimension = limit;
    let device = device
        .with_limits(limits)
        .map_err(|error| format!("could not lower the workgroup limit: {error}"))?;
    println!("split: {workgroups} workgroups against a limit of {limit}");
    let a = device.create_typed_buffer("split a", a, GpuBufferAccess::ReadOnly);
    let b = device.create_typed_buffer("split b", b, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("split out", seed, GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(&device, LENGTH as u32, &a, &b, &out)
        .map_err(|error| format!("split dispatch failed: {error}"))?;
    let result = device
        .read_typed_buffer(&out)
        .map_err(|error| format!("split read-back failed: {error}"))?;
    if result != expected {
        let first = result
            .iter()
            .zip(expected)
            .position(|(got, want)| got != want)
            .unwrap_or(0);
        return Err(format!(
            "the split launch differs from the host reference at index {first} (got {:?}, expected {:?})",
            result.get(first),
            expected.get(first)
        ));
    }
    println!(
        "split: a launch past a lowered workgroup limit ran as pieces of the split artifact and equals the host reference for {LENGTH} elements"
    );

    // The fallback: the same launch of the same kernel whose split variant has no precompiled artifact. This binary has
    // no run-time compiler either, so the host cannot run the pieces and answers with the launch's own refusal (D208).
    let bindings = [
        a.bind(Access::ReadOnly, LengthRule::Shared),
        b.bind(Access::ReadOnly, LengthRule::Shared),
        out.bind(Access::ReadWrite, LengthRule::Shared),
    ];
    let launch = BufferDispatch::new_extent(
        &WITHOUT_SPLIT_ARTIFACT,
        Extent::linear(LENGTH as u32),
        &bindings,
    );
    match device.dispatch_batch(&[launch]) {
        Err(Error::DeviceLimit { limit, .. }) => println!("{SPLIT_REFUSED_PREFIX}{limit}"),
        Err(other) => {
            return Err(format!(
                "a split launch without a split artifact or a compiler failed with {other}, not the DeviceLimit refusal"
            ));
        }
        Ok(_) => {
            return Err(
                "a split launch without a split artifact or a compiler ran: the pieces came from nowhere".to_owned(),
            );
        }
    }
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
