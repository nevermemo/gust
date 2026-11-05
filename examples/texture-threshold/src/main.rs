use std::path::PathBuf;

use gust::{
    Extent, Independent, gpu, slang, spirv,
    texture::{R32Float, R32Uint, TextureUse},
};
use gust_wgpu::{GpuBufferAccess, GpuTexture2d, HeadlessDevice, render_wgpu_source};

#[gpu]
mod threshold {
    /// Marks the texels of a storage texture that are brighter than one half.
    ///
    /// Each bright texel adds one to `bright[0]` and writes `1` to the same address of `mask`; every other texel writes
    /// `0`. The comparison is strict, so a texel of exactly one half is not bright. The dispatch is a whole number of
    /// workgroups and may be larger than the texture, so the kernel guards against the texture's own size.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn mark(
        id: SV_DispatchThreadID,
        image: StorageTexture2D<R32Float>,
        mut bright: RWStructuredBuffer<u32>,
        mut mask: WriteTexture2D<R32Uint>,
    ) {
        if id.x < image.width() && id.y < image.height() {
            let value = image.load(id.x, id.y);
            if value > 0.5f32 {
                atomic_add(&mut bright[0], 1u32);
                mask.store(id.x, id.y, 1u32);
            } else {
                mask.store(id.x, id.y, 0u32);
            }
        }
    }
}

/// A texture that is not a multiple of the workgroup (8 by 8) in either direction.
const WIDTH: u32 = 37;
const HEIGHT: u32 = 21;

/// The texel `(x, y)` of the input: sixteenths from 0 to 1, so a texel of exactly one half (8/16) occurs and the
/// strict comparison is exercised. Every value is exact in `f32`.
fn input_texel(x: u32, y: u32) -> f32 {
    ((x * 7 + y * 13) % 17) as f32 / 16.0
}

fn input_texels() -> Vec<f32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| input_texel(x, y)))
        .collect()
}

/// The host reference: the mask row-major and the number of bright texels, from a plain loop over the input.
fn host_reference(texels: &[f32]) -> (Vec<u32>, u32) {
    let mask: Vec<u32> = texels.iter().map(|&value| u32::from(value > 0.5)).collect();
    let count = mask.iter().sum();
    (mask, count)
}

pub fn main() {
    let descriptor = threshold::mark::DESCRIPTOR;
    let texels = input_texels();
    let (reference_mask, reference_count) = host_reference(&texels);

    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let (gpu_mask, gpu_count) = run_on_gpu(&device, &texels);
    let artifact_directory = export_artifacts().expect("export generated shader artifacts");

    println!("Rust -> Slang -> WGSL -> wgpu (plus SPIR-V export)");
    println!("kernel: {}", descriptor.qualified_name());
    println!("adapter: {}", device.adapter_info().name);
    println!("texture: {WIDTH} x {HEIGHT} texels (r32float in, r32uint mask out)");
    println!("GPU bright texels: {gpu_count}");
    println!("host reference: {reference_count}");
    println!("generated artifacts: {}", artifact_directory.display());

    assert_eq!(gpu_count, reference_count, "the bright count");
    assert_eq!(gpu_mask, reference_mask, "the mask texture");
    println!(
        "PASS: GPU mask and count match the host reference ({reference_count} of {} texels)",
        texels.len()
    );
}

/// `mark` on the GPU: the mask texture (row-major) and the counter.
fn run_on_gpu(device: &HeadlessDevice, texels: &[f32]) -> (Vec<u32>, u32) {
    let image: GpuTexture2d<R32Float> = device
        .create_texture_2d(WIDTH, HEIGHT, TextureUse::STORAGE)
        .expect("the input texture");
    device
        .write_texture(&image, texels)
        .expect("upload the input texture");
    let mask: GpuTexture2d<R32Uint> = device
        .create_texture_2d(WIDTH, HEIGHT, TextureUse::STORAGE)
        .expect("the mask texture");
    // The counter has one element, not one per invocation: `Independent` is the call-site opt-in for that.
    let bright = device.create_typed_buffer("bright", &[0u32; 1], GpuBufferAccess::ReadWrite);
    threshold::mark::dispatch_extent(
        device,
        Extent::new(WIDTH, HEIGHT, 1),
        &image,
        Independent(&bright),
        &mask,
    )
    .expect("the threshold kernel should execute on the GPU");
    let count = device
        .read_typed_buffer(&bright)
        .expect("counter readback should succeed")[0];
    let mask = device
        .read_texture(&mask)
        .expect("mask readback should succeed");
    (mask, count)
}

fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    let descriptor = &threshold::mark::DESCRIPTOR;
    std::fs::write(
        directory.join("threshold__mark.slang"),
        descriptor.slang_source,
    )?;
    std::fs::write(
        directory.join("threshold__mark.wgsl"),
        slang::compile_wgsl(descriptor)?,
    )?;
    std::fs::write(
        directory.join("threshold__mark.spv"),
        spirv::words_as_le_bytes(&slang::compile_spirv(descriptor)?),
    )?;
    std::fs::write(
        directory.join("threshold__mark.rs"),
        render_wgpu_source(descriptor)?,
    )?;
    Ok(directory.canonicalize().unwrap_or(directory))
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;

    /// `mark` on the CPU twin, which the macro emits under `cfg(test)`: no GPU involved. The mask is read back
    /// row-major and the counter is the single element.
    fn mark_on_cpu(texels: &[f32]) -> Result<(Vec<u32>, u32), CpuError> {
        let device = CpuDevice::new();
        let image = device.texture_2d::<R32Float>(WIDTH, HEIGHT, TextureUse::STORAGE)?;
        image.write(texels)?;
        let mask = device.texture_2d::<R32Uint>(WIDTH, HEIGHT, TextureUse::STORAGE)?;
        let bright = device.buffer(&[0u32]);
        threshold::mark::dispatch_extent(
            &device,
            Extent::new(WIDTH, HEIGHT, 1),
            &image,
            Independent(&bright),
            &mask,
        )?;
        Ok((mask.read(), bright.read()[0]))
    }

    #[test]
    fn twin_matches_host_reference() {
        let texels = input_texels();
        let (mask, count) = mark_on_cpu(&texels).expect("twin dispatch");
        let (reference_mask, reference_count) = host_reference(&texels);
        // The input has texels of exactly one half, texels above and below it: the reference is not trivial.
        assert!(texels.contains(&0.5));
        assert!(reference_count > 0 && (reference_count as usize) < texels.len());
        assert_eq!(count, reference_count, "the bright count");
        assert_eq!(mask, reference_mask, "the mask texture");
    }

    /// Everything above, on the GPU too: the twin, the GPU and the host loop agree texel for texel.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu() {
        let device = match HeadlessDevice::new() {
            Ok(device) => device,
            Err(error) => panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            ),
        };
        let texels = input_texels();
        let reference = host_reference(&texels);
        let gpu = run_on_gpu(&device, &texels);
        let twin = mark_on_cpu(&texels).expect("twin dispatch");
        assert_eq!(gpu, reference, "GPU versus reference");
        assert_eq!(twin, reference, "twin versus reference");
        assert_eq!(twin, gpu, "twin versus GPU");
    }
}
