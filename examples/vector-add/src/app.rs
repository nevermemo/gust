use std::path::PathBuf;

#[cfg(any(test, gust_dual))]
use gust::cpu::{CpuDevice, Error as CpuError};
#[cfg(test)]
use gust::{Access, ResourceBinding};
use gust::{gpu, slang, spirv};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice, render_wgpu_source};

#[gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}

pub fn main() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let expected = a
        .iter()
        .zip(b)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    let seed = [0.0; 4];
    let descriptor = vector_add::add::DESCRIPTOR;

    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let result = add_on_gpu(&device, &a, &b, &seed)
        .expect("Rust-to-Slang vector add should execute on the GPU");

    let artifact_directory = export_artifacts().expect("export generated shader artifacts");
    println!("Rust -> Slang -> WGSL -> wgpu (plus SPIR-V export)");
    println!("kernel: {}", descriptor.qualified_name());
    println!("adapter: {}", device.adapter_info().name);
    println!("GPU result: {result:?}");
    println!("generated artifacts: {}", artifact_directory.display());
    println!("\nGenerated Slang:\n{}", descriptor.slang_source);

    assert_eq!(result, expected);

    // Only the dual test build (`cargo xtask check-dual`, `--cfg gust_dual`, D26) has the twin in a binary.
    #[cfg(gust_dual)]
    report_twin(&a, &b, &seed, &expected, &result);
}

/// The same inputs through the CPU twin, checked against the host reference (to the last bit: the sums are exact)
/// and against the GPU result, which `cargo xtask check-dual` requires `main` to print.
#[cfg(gust_dual)]
fn report_twin(a: &[f32], b: &[f32], seed: &[f32], expected: &[f32], gpu: &[f32]) {
    let twin = add_on_cpu(a, b, seed).expect("the CPU twin should dispatch");
    let max_ulps = twin
        .iter()
        .zip(expected)
        .map(|(twin, expected)| gust::cpu::ulps(*twin, *expected))
        .max()
        .unwrap_or(0);
    assert_eq!(max_ulps, 0, "CPU twin versus the host reference");
    assert_eq!(twin, gpu, "CPU twin versus the GPU result");
    println!(
        "CPU twin: matches ({} elements, max ULP {max_ulps})",
        twin.len()
    );
}

/// The same typed `dispatch` on the CPU twin, which the macro emits under `cfg(test)` and `--cfg gust_dual`: no GPU
/// involved.
#[cfg(any(test, gust_dual))]
fn add_on_cpu(a: &[f32], b: &[f32], seed: &[f32]) -> Result<Vec<f32>, CpuError> {
    let device = CpuDevice::new();
    let a = device.buffer(a);
    let b = device.buffer(b);
    let out = device.buffer(seed);
    vector_add::add::dispatch(&device, seed.len() as u32, &a, &b, &out)?;
    Ok(out.read())
}

/// Uploads the inputs, runs the generated typed `dispatch` (the element types and the device of the
/// three buffers are checked at compile time) and reads the written buffer back.
fn add_on_gpu(
    device: &HeadlessDevice,
    a: &[f32],
    b: &[f32],
    seed: &[f32],
) -> Result<Vec<f32>, gust_wgpu::Error> {
    let a = device.create_f32_buffer("vector add a", a, GpuBufferAccess::ReadOnly);
    let b = device.create_f32_buffer("vector add b", b, GpuBufferAccess::ReadOnly);
    let out = device.create_f32_buffer("vector add out", seed, GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(device, seed.len() as u32, &a, &b, &out)?;
    device.read_f32_buffer(&out)
}

fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    let descriptor = &vector_add::add::DESCRIPTOR;
    std::fs::write(
        directory.join("vector_add__add.slang"),
        descriptor.slang_source,
    )?;
    std::fs::write(
        directory.join("vector_add__add.wgsl"),
        slang::compile_wgsl(descriptor)?,
    )?;
    std::fs::write(
        directory.join("vector_add__add.spv"),
        spirv::words_as_le_bytes(&slang::compile_spirv(descriptor)?),
    )?;
    std::fs::write(
        directory.join("vector_add__add.rs"),
        render_wgpu_source(descriptor)?,
    )?;
    Ok(directory.canonicalize().unwrap_or(directory))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host reference: what the kernel computes for every element.
    fn host_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(left, right)| left + right).collect()
    }

    /// Inputs whose sums are exact in `f32`, so every comparison below is exact.
    fn inputs(count: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..count).map(|k| k as f32 * 0.5 - 7.0).collect();
        let b = (0..count).map(|k| 100.0 - k as f32 * 0.25).collect();
        (a, b)
    }

    #[test]
    fn twin_matches_host_reference() {
        // Zero, one, a partial workgroup, a whole one (64) and counts that end mid-workgroup.
        for count in [0, 1, 3, 4, 63, 64, 65, 200] {
            let (a, b) = inputs(count);
            let twin = add_on_cpu(&a, &b, &vec![-1.0; count]).expect("twin dispatch");
            assert_eq!(twin, host_reference(&a, &b), "{count} elements");
        }
    }

    #[test]
    fn twin_checks_buffer_aliasing_like_the_gpu_backend() {
        let device = CpuDevice::new();
        let a = device.buffer(&[1.0f32, 2.0, 3.0]);
        let b = device.buffer(&[10.0f32, 20.0, 30.0]);
        let out = device.buffer(&[0.0f32; 3]);
        // One buffer bound read-only twice is fine.
        vector_add::add::dispatch(&device, 3, &a, &a, &out).expect("read-only alias");
        assert_eq!(out.read(), [2.0, 4.0, 6.0]);
        // A read-only parameter and the read-write one on the same buffer are rejected, and nothing is written.
        let error = vector_add::add::dispatch(&device, 3, &a, &b, &a).unwrap_err();
        assert_eq!(
            error,
            CpuError::Aliased {
                kernel: "vector_add::add".to_owned(),
                first: "a",
                second: "out",
            }
        );
        assert_eq!(a.read(), [1.0, 2.0, 3.0]);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        for count in [4, 65, 200] {
            let (a, b) = inputs(count);
            let seed = vec![-1.0; count];
            let reference = host_reference(&a, &b);
            let twin = add_on_cpu(&a, &b, &seed).expect("twin dispatch");
            let gpu = add_on_gpu(&device, &a, &b, &seed).expect("GPU dispatch");
            assert_eq!(twin, reference, "{count} elements: twin versus reference");
            // A single IEEE addition of exactly representable sums: both targets are exact.
            assert_eq!(twin, gpu, "{count} elements: twin versus GPU");
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn descriptor_is_slang_first_and_reflects_bindings() {
        let descriptor = vector_add::add::DESCRIPTOR;
        assert!(
            descriptor
                .slang_source
                .contains("StructuredBuffer<float> a")
        );
        assert!(
            descriptor
                .slang_source
                .contains("RWStructuredBuffer<float> out")
        );
        assert!(descriptor.slang_source.contains("SV_DispatchThreadID"));
        assert_eq!(descriptor.entry_point, "gpu_vector_add_add");
        assert_eq!(descriptor.parameters[0].binding, None);
        assert_eq!(descriptor.parameters[1].access, Access::ReadOnly);
        assert_eq!(
            descriptor.parameters[1].binding,
            Some(ResourceBinding {
                group: 0,
                binding: 0,
            })
        );
        assert_eq!(
            descriptor.parameters[3].binding,
            Some(ResourceBinding {
                group: 0,
                binding: 2,
            })
        );
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn slang_compiles_to_readable_wgsl() {
        let descriptor = vector_add::add::DESCRIPTOR;
        let wgsl = slang::compile_wgsl(&descriptor).expect("Slang WGSL compilation");
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@binding(2) @group(0)"));
        assert!(wgsl.contains("a_0[i_0] + b_0[i_0]"));
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn rust_to_slang_executes_on_headless_wgpu() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let a = [2.0, 4.0, 8.0];
        let b = [3.0, 5.0, 7.0];
        let seed = [0.0; 3];
        let result = add_on_gpu(&device, &a, &b, &seed).expect("dispatch through Slang");
        assert_eq!(result, [5.0, 9.0, 15.0]);
    }
}
