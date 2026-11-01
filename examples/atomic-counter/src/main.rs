use std::path::PathBuf;

use gust::{Independent, gpu, slang, spirv};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice, render_wgpu_source};

#[gpu]
mod atomic_counter {
    /// T10 vertical slice: every thread increments the same shared counter.
    /// The final value must equal the number of threads that executed the increment.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn increment(_id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
        atomic_add(&mut counter[0], 1u32);
    }
}

const INVOCATIONS: u32 = 257;

pub fn main() {
    let descriptor = atomic_counter::increment::DESCRIPTOR;

    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let counter =
        device.create_typed_buffer("atomic counter", &[0u32; 1], GpuBufferAccess::ReadWrite);
    // The counter has one element, not `INVOCATIONS`: `Independent` is the call-site opt-in for that.
    atomic_counter::increment::dispatch(&device, INVOCATIONS, Independent(&counter))
        .expect("atomic_add should execute on the GPU");
    let gpu_result = device
        .read_typed_buffer(&counter)
        .expect("atomic counter readback should succeed");
    let host_reference = INVOCATIONS;
    let artifact_directory = export_artifacts().expect("export generated shader artifacts");

    println!("Rust -> Slang -> WGSL -> wgpu (plus SPIR-V export)");
    println!("kernel: {}", descriptor.qualified_name());
    println!("adapter: {}", device.adapter_info().name);
    println!("invocations: {INVOCATIONS}");
    println!("GPU result: {gpu_result:?}");
    println!("host reference: {host_reference}");
    println!("generated artifacts: {}", artifact_directory.display());

    assert_eq!(
        gpu_result,
        vec![host_reference],
        "atomic_add under contention must produce exactly {INVOCATIONS}"
    );
    println!("PASS: GPU result matches host reference ({host_reference})");
}

fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    let descriptor = &atomic_counter::increment::DESCRIPTOR;
    std::fs::write(
        directory.join("atomic_counter__increment.slang"),
        descriptor.slang_source,
    )?;
    std::fs::write(
        directory.join("atomic_counter__increment.wgsl"),
        slang::compile_wgsl(descriptor)?,
    )?;
    std::fs::write(
        directory.join("atomic_counter__increment.spv"),
        spirv::words_as_le_bytes(&slang::compile_spirv(descriptor)?),
    )?;
    std::fs::write(
        directory.join("atomic_counter__increment.rs"),
        render_wgpu_source(descriptor)?,
    )?;
    Ok(directory.canonicalize().unwrap_or(directory))
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;

    /// The typed `dispatch` on the CPU twin, which the macro emits under `cfg(test)`: no GPU involved. The twin runs
    /// the invocations one after another, so each `atomic_add` is a plain read-modify-write.
    fn increment_on_cpu(initial: u32, invocations: u32) -> Result<u32, CpuError> {
        let device = CpuDevice::new();
        let counter = device.buffer(&[initial]);
        atomic_counter::increment::dispatch(&device, invocations, Independent(&counter))?;
        Ok(counter.read()[0])
    }

    fn increment_on_gpu(device: &HeadlessDevice, invocations: u32) -> u32 {
        let counter =
            device.create_typed_buffer("atomic counter", &[0u32; 1], GpuBufferAccess::ReadWrite);
        atomic_counter::increment::dispatch(device, invocations, Independent(&counter))
            .expect("atomic_add should execute on the GPU");
        device
            .read_typed_buffer(&counter)
            .expect("atomic counter readback should succeed")[0]
    }

    #[test]
    fn twin_counts_every_invocation() {
        // The host reference is the invocation count (plus the start value). Zero, one, a power of two, the demo's
        // 257 and counts that are not multiples of anything in the kernel.
        for invocations in [0, 1, 2, 64, 257, 1000] {
            assert_eq!(
                increment_on_cpu(0, invocations).expect("twin dispatch"),
                invocations,
                "{invocations} invocations from 0"
            );
            assert_eq!(
                increment_on_cpu(5, invocations).expect("twin dispatch"),
                5 + invocations,
                "{invocations} invocations from 5"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu() {
        let device = match HeadlessDevice::new() {
            Ok(device) => device,
            Err(error) => panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            ),
        };
        // `atomic_add` of 1 commutes, so both targets are exact whatever order the invocations run in.
        for invocations in [1, 64, INVOCATIONS, 1000] {
            let reference = invocations;
            let gpu = increment_on_gpu(&device, invocations);
            assert_eq!(gpu, reference, "GPU versus reference, {invocations}");
            let twin = increment_on_cpu(0, invocations).expect("twin dispatch");
            assert_eq!(twin, reference, "twin versus reference, {invocations}");
            assert_eq!(twin, gpu, "twin versus GPU, {invocations}");
        }
    }

    #[test]
    fn invocation_count_stays_within_the_default_workgroup_limit() {
        let groups = INVOCATIONS.div_ceil(atomic_counter::increment::DESCRIPTOR.workgroup_size[0]);
        assert!(
            groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
            "{INVOCATIONS} invocations need {groups} workgroups"
        );
    }
}
