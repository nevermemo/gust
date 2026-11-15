# GPU concepts in Rust, running half

This is the host half of [chapter 5][`gust::guide::gpu_concepts`]: the programs that run the kernels of that chapter's
shapes, textures, workgroup memory and subgroup variants on a device. Each listing is a complete program, a packaged
copy of an entry of the repository's `gallery/` directory, compiled by this crate's doctests (`no_run`: running one
needs a Vulkan compute adapter) and kept identical to its entry by `cargo xtask check-gallery`. The conventions of
calls, buffers and `Independent` are those of [chapter 4][`crate::guide::running_kernels`].

## Textures and the two shapes (G5)

`create_texture_2d::<F>(width, height, use)` makes a `GpuTexture2d<F>`; `TextureUse::SAMPLED` is for a `Texture2D`
parameter and `TextureUse::STORAGE` for a storage texture. `write_texture` and `read_texture` move texels, and
`create_sampler(SamplerOptions::new(filter, address))` makes the `GpuSampler` a `Sampler` parameter takes. A `grid`
kernel over a texture launches with `dispatch_extent(device, Extent::new(width, height, 1), ..)`; a `pixel` kernel is a
one-dimensional call over an element count, and its output buffer is as long as the count. The program runs both on the
same image, 37 by 21 texels (not a multiple of the 8 by 8 workgroup), and a wider one, and compares each with a host
loop:

```rust,no_run
