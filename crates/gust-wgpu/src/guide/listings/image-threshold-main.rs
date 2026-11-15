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
