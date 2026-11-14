// ANCHOR: imports
use gust::{
    Extent, gpu,
    texture::{Address, Filter, R32Float, SamplerOptions, TextureUse},
};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu]
mod threshold {
    /// A `grid` kernel on a storage texture: every texel brighter than one half becomes 1.0 and every other texel 0.0,
    /// in place. The dispatch is a whole number of 8 by 8 workgroups and so may be larger than the texture; the kernel
    /// guards against the texture's own size, which it reads from the texture like a buffer's length.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn binarize(id: SV_DispatchThreadID, mut image: StorageTexture2D<R32Float>) {
        if id.x < image.width() && id.y < image.height() {
            let value = image.load(id.x, id.y);
            if value > 0.5f32 {
                image.store(id.x, id.y, 1.0f32);
            } else {
                image.store(id.x, id.y, 0.0f32);
            }
        }
    }

    /// The same decision as a `pixel` kernel: the body runs once for each element `0..n` of a one-dimensional call
    /// (as a fragment shader; there is no workgroup size, no guard and no padding invocation). The element `i` is the
    /// texel `(i % width, i / width)`, read through a sampler at the centre of that texel, and the answer is the one
    /// output `flags[i]`, written on every path.
    #[kernel(shape = pixel)]
    pub fn flag(
        id: SV_DispatchThreadID,
        image: Texture2D<f32>,
        nearest: Sampler,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        let width = image.width();
        let u = ((i % width) as f32 + 0.5f32) / (width as f32);
        let v = ((i / width) as f32 + 0.5f32) / (image.height() as f32);
        if image.sample(nearest, u, v) > 0.5f32 {
            flags[i] = 1u32;
        } else {
            flags[i] = 0u32;
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: host
/// The headline image: 37 by 21 texels, which is not a multiple of the 8 by 8 workgroup in either direction.
const WIDTH: u32 = 37;
const HEIGHT: u32 = 21;

/// The texel `(x, y)` of the input: sixteenths from 0 to 1, so texels of exactly one half (8/16) occur and the strict
/// comparison is exercised. Every value is exact in `f32`.
fn texel(x: u32, y: u32) -> f32 {
    ((x * 7 + y * 13) % 17) as f32 / 16.0
}

/// The input image, row-major.
fn image(width: u32, height: u32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| texel(x, y)))
        .collect()
}

/// The independent host reference: a plain loop over the texels, which never goes through a kernel (D26, D42). One
/// flag per texel, row-major.
fn host_flags(texels: &[f32]) -> Vec<u32> {
    texels.iter().map(|value| u32::from(*value > 0.5)).collect()
}
// ANCHOR_END: host

// ANCHOR: main
/// Both kernels on the GPU for an image of `width` by `height` texels: the flags of the `pixel` kernel (through a
/// sampled texture and a nearest sampler) and the texels of the `grid` kernel (a storage texture, in place).
fn run(device: &HeadlessDevice, width: u32, height: u32) -> (Vec<u32>, Vec<f32>) {
    let texels = image(width, height);
    let count = texels.len();

    let sampled = device
        .create_texture_2d::<R32Float>(width, height, TextureUse::SAMPLED)
        .expect("the sampled texture");
    device
        .write_texture(&sampled, &texels)
        .expect("upload the sampled texture");
    let nearest = device
        .create_sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge))
        .expect("the sampler");
    let flags =
        device.create_typed_buffer("flags", &vec![u32::MAX; count], GpuBufferAccess::ReadWrite);
    // A `pixel` call has an element count, not an extent; its output is as long as the count.
    let elements = u32::try_from(count).expect("a small image");
    threshold::flag::dispatch(device, elements, &sampled, &nearest, &flags)
        .expect("pixel dispatch");

    let stored = device
        .create_texture_2d::<R32Float>(width, height, TextureUse::STORAGE)
        .expect("the storage texture");
    device
        .write_texture(&stored, &texels)
        .expect("upload the storage texture");
    threshold::binarize::dispatch_extent(device, Extent::new(width, height, 1), &stored)
        .expect("grid dispatch");

    (
        device.read_typed_buffer(&flags).expect("read the flags"),
        device.read_texture(&stored).expect("read the texture"),
    )
}

fn main() {
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    // The headline size, then a wider image whose 10 000 texels pass the 8192-wide rows of a `pixel` call, so the
    // call ends in a partial second row (and 200 by 50 ends in partial workgroups of the `grid` kernel too).
    let (by_pixel, by_grid) = run(&device, WIDTH, HEIGHT);
    let want = host_flags(&image(WIDTH, HEIGHT));
    assert_eq!(by_pixel, want, "the pixel kernel is exact");
    assert_eq!(
        by_grid,
        want.iter().map(|flag| *flag as f32).collect::<Vec<_>>(),
        "the grid kernel is exact"
    );
    let (wide_pixel, wide_grid) = run(&device, 200, 50);
    let wide = host_flags(&image(200, 50));
    assert_eq!(wide_pixel, wide, "the pixel kernel is exact across rows");
    assert_eq!(
        wide_grid,
        wide.iter().map(|flag| *flag as f32).collect::<Vec<_>>(),
        "the grid kernel is exact at 200 by 50"
    );

    let bright_pixel: u32 = by_pixel.iter().sum();
    let bright_grid = by_grid.iter().filter(|value| **value == 1.0).count();
    println!(
        "image-threshold: {WIDTH} x {HEIGHT} texels, {bright_pixel} bright by the pixel kernel, {bright_grid} by the grid kernel"
    );
    println!("adapter: {}", device.adapter_info().name);
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::{
        Extent,
        cpu::CpuDevice,
        texture::{Address, Filter, R32Float, SamplerOptions, TextureUse},
    };

    use super::{host_flags, image, threshold};

    /// The `pixel` kernel on the CPU twin (the macro emits the typed `dispatch` for `CpuDevice` under `cfg(test)`).
    fn pixel_on_twin(width: u32, height: u32) -> Vec<u32> {
        let device = CpuDevice::new();
        let sampled = device
            .texture_2d::<R32Float>(width, height, TextureUse::SAMPLED)
            .expect("a sampled texture");
        sampled.write(&image(width, height)).expect("upload");
        let nearest = device.sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge));
        let count = (width * height) as usize;
        let flags = device.buffer(&vec![u32::MAX; count]);
        threshold::flag::dispatch(&device, width * height, &sampled, &nearest, &flags)
            .expect("the twin dispatches");
        flags.read()
    }

    /// The `grid` kernel on the CPU twin, over a dispatch rounded up to whole 8 by 8 workgroups by the call itself.
    fn grid_on_twin(width: u32, height: u32) -> Vec<f32> {
        let device = CpuDevice::new();
        let stored = device
            .texture_2d::<R32Float>(width, height, TextureUse::STORAGE)
            .expect("a storage texture");
        stored.write(&image(width, height)).expect("upload");
        threshold::binarize::dispatch_extent(&device, Extent::new(width, height, 1), &stored)
            .expect("the twin dispatches");
        stored.read()
    }

    /// Image sizes below, at and above the 8 by 8 workgroup, ones that are not a multiple of it in either direction
    /// (so the last row and column of workgroups are partial), a single texel, and 100 by 100, whose 10 000 texels
    /// are more than one 8192-wide row of a `pixel` call.
    const SIZES: [(u32, u32); 8] = [
        (1, 1),
        (7, 3),
        (8, 8),
        (9, 9),
        (37, 21),
        (64, 33),
        (100, 100),
        (200, 50),
    ];

    #[test]
    fn the_twin_matches_hand_computed_values() {
        // Row 0 of the image is `(7 x % 17) / 16`: 0, 7, 14, 4, 11, 1 and 8 sixteenths for x = 0 to 6. Those above one
        // half are 14 and 11; the 8 of x = 6 is exactly one half, which is not bright (the comparison is strict).
        assert_eq!(pixel_on_twin(7, 1), [0, 0, 1, 0, 1, 0, 0]);
        assert_eq!(grid_on_twin(7, 1), [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(image(7, 1).contains(&0.5));
    }

    #[test]
    fn the_pixel_twin_matches_the_host_reference_exactly() {
        for (width, height) in SIZES {
            assert_eq!(
                pixel_on_twin(width, height),
                host_flags(&image(width, height)),
                "{width} x {height}"
            );
        }
    }

    /// Exact: every texel is a multiple of 1/16, so the comparison with one half has no rounding to tolerate, and a
    /// nearest read at a texel centre is that texel (the centre is never near a texel edge, so KM-14 does not apply).
    #[test]
    fn the_grid_twin_matches_the_host_reference_exactly() {
        for (width, height) in SIZES {
            let want: Vec<f32> = host_flags(&image(width, height))
                .iter()
                .map(|flag| *flag as f32)
                .collect();
            assert_eq!(grid_on_twin(width, height), want, "{width} x {height}");
        }
    }

    #[test]
    fn the_image_is_not_trivial() {
        // A guard on the example: bright and dark texels both occur.
        let flags = host_flags(&image(37, 21));
        let bright: u32 = flags.iter().sum();
        assert!(bright > 0 && (bright as usize) < flags.len());
    }
}
// ANCHOR_END: tests
