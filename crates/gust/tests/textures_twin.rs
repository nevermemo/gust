//! The textures fixture on the CPU twin (roadmap MX, slice mx.t3; Unit tier, no adapter, no slangc).
//!
//! `tests/fixtures/textures.rs` is the module of the golden `textures.slang`. Including it here compiles its expansion
//! with rustc, which the string tests of the macro cannot do: the typed calls with their `Kind` and sampler bounds, the
//! shadow `Texture2D` / `StorageTexture2D` / `WriteTexture2D` views and the twin that runs the body. Each kernel runs
//! on the twin and every texel is compared with a value worked by hand below (all of them are exactly representable,
//! so the comparison is exact). The twin is never GPU evidence.
//!
//! Every dispatch is larger than the texture it writes (the fixture's kernels guard against the texture's size), which
//! is what a dispatch of whole workgroups over a texture of any size does.

include!("../../../tests/fixtures/textures.rs");

use gust::Extent;
use gust::cpu::CpuDevice;
use gust::math::{IVec2, UVec4, Vec2};
use gust::texture::{
    Address, Filter, R32Float, R32Uint, Rg32Float, Rg32Sint, Rg32Uint, Rgba32Float, Rgba32Uint,
    SamplerOptions, TextureUse,
};

/// A storage texture read and written in place (`invert`): `value = 1 - value` on a 3 by 2 texture, dispatched over
/// 4 by 4 invocations.
///
/// `1 - 0.25 = 0.75`, `1 - 0.5 = 0.5`, `1 - 1 = 0`, `1 - 0 = 1`, `1 - 2 = -1`, `1 - (-1) = 2`.
#[test]
fn a_storage_texture_is_loaded_and_stored_in_place() {
    let device = CpuDevice::new();
    let image = device
        .texture_2d::<R32Float>(3, 2, TextureUse::STORAGE)
        .unwrap();
    image.write(&[0.25, 0.5, 1.0, 0.0, 2.0, -1.0]).unwrap();
    textures::invert::dispatch_extent(&device, Extent::new(4, 4, 1), &image)
        .expect("the twin of `invert` refused the call");
    assert_eq!(image.read(), [0.75, 0.5, 0.0, 1.0, -1.0, 2.0]);
}

/// A write-only texture of each of three formats (`write_uints`), 3 by 2, dispatched over 4 by 4 invocations: the
/// texel `(x, y)` of `one` is `x`, of `two` is `(x, y)` and of `four` is `(x, y, 3, 4)`, row-major.
#[test]
fn a_write_only_texture_is_stored_texel_by_texel() {
    let device = CpuDevice::new();
    let one = device
        .texture_2d::<R32Uint>(3, 2, TextureUse::STORAGE)
        .unwrap();
    let two = device
        .texture_2d::<Rg32Uint>(3, 2, TextureUse::STORAGE)
        .unwrap();
    let four = device
        .texture_2d::<Rgba32Uint>(3, 2, TextureUse::STORAGE)
        .unwrap();
    textures::write_uints::dispatch_extent(&device, Extent::new(4, 4, 1), &one, &two, &four)
        .expect("the twin of `write_uints` refused the call");
    assert_eq!(one.read(), [0, 1, 2, 0, 1, 2]);
    let two: Vec<[u32; 2]> = two.read().iter().map(|t| [t.x, t.y]).collect();
    assert_eq!(two, [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],]);
    let four: Vec<[u32; 4]> = four.read().iter().map(|t| [t.x, t.y, t.z, t.w]).collect();
    assert_eq!(
        four,
        [
            [0, 0, 3, 4],
            [1, 0, 3, 4],
            [2, 0, 3, 4],
            [0, 1, 3, 4],
            [1, 1, 3, 4],
            [2, 1, 3, 4],
        ]
    );
}

/// One nearest and one linear sample, and two integer loads (`gather`), on a 2 by 1 output. The sampled textures are
/// 3 texels wide and 1 tall, so the output texel `x` samples at `u = (x + 0.5) / 2` (`0.25`, `0.75`) and `v = 0.5`,
/// which is the position `u * 3` (`0.75`, `2.25`) along the texture; `v * 1 = 0.5` is the centre of the only row.
///
/// - `scalar` (nearest sampler) is `[10, 20, 30]`: the texel the position falls in, `floor(0.75) = 0` and
///   `floor(2.25) = 2`: `10` and `30`.
/// - `pair` (linear sampler) is `[(0, 0), (4, 8), (8, 16)]`; texel `j` has its centre at `j + 0.5`, so the position
///   `p` mixes texels `p - 0.5` down and up. `p = 0.75`: `0.25` of the way from texel 0 to texel 1,
///   `(0.25 * 4, 0.25 * 8) = (1, 2)`. `p = 2.25`: `0.75` of the way from texel 1 to texel 2,
///   `(4 + 0.75 * 4, 8 + 0.75 * 8) = (7, 14)`.
/// - `counts` is `[(1, 0, 0, 2), (3, 0, 0, 4)]`, so `c.x + c.w` is `3` and `7`; `offsets` is `[(5, 2), (1, 4)]`, so
///   `d.x - d.y` is `3` and `-3`.
///
/// The output is `(a + b.x, b.y, c.x + c.w, d.x - d.y)`: `(10 + 1, 2, 3, 3)` and `(30 + 7, 14, 7, -3)`.
#[test]
fn a_nearest_sample_a_linear_sample_and_integer_loads_match_the_hand_values() {
    let device = CpuDevice::new();
    let sampled = TextureUse::SAMPLED;
    let scalar = device.texture_2d::<R32Float>(3, 1, sampled).unwrap();
    scalar.write(&[10.0, 20.0, 30.0]).unwrap();
    let pair = device.texture_2d::<Rg32Float>(3, 1, sampled).unwrap();
    pair.write(&[
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 8.0),
        Vec2::new(8.0, 16.0),
    ])
    .unwrap();
    let counts = device.texture_2d::<Rgba32Uint>(2, 1, sampled).unwrap();
    counts
        .write(&[UVec4::new(1, 0, 0, 2), UVec4::new(3, 0, 0, 4)])
        .unwrap();
    let offsets = device.texture_2d::<Rg32Sint>(2, 1, sampled).unwrap();
    offsets
        .write(&[IVec2::new(5, 2), IVec2::new(1, 4)])
        .unwrap();
    let gathered = device
        .texture_2d::<Rgba32Float>(2, 1, TextureUse::STORAGE)
        .unwrap();
    let nearest = device.sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge));
    let bilinear = device.sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge));
    textures::gather::dispatch_extent(
        &device,
        Extent::new(4, 2, 1),
        &scalar,
        &pair,
        &counts,
        &offsets,
        &nearest,
        &bilinear,
        &gathered,
    )
    .expect("the twin of `gather` refused the call");
    let texels = gathered.read();
    let texels: Vec<[f32; 4]> = texels.iter().map(|t| [t.x, t.y, t.z, t.w]).collect();
    assert_eq!(texels, [[11.0, 2.0, 3.0, 3.0], [37.0, 14.0, 7.0, -3.0]]);
}

/// The filter belongs to the sampler, not to the kernel: the same `gather` with the two samplers swapped reads
/// `scalar` linearly and `pair` by the nearest texel. `scalar` at `p = 0.75`: `10 + 0.25 * 10 = 12.5`; at `p = 2.25`:
/// `20 + 0.75 * 10 = 27.5`. `pair` by the nearest texel: `(0, 0)` and `(8, 16)`. The output is
/// `(12.5 + 0, 0, 3, 3)` and `(27.5 + 8, 16, 7, -3)`.
#[test]
fn swapping_the_samplers_swaps_the_filters() {
    let device = CpuDevice::new();
    let sampled = TextureUse::SAMPLED;
    let scalar = device.texture_2d::<R32Float>(3, 1, sampled).unwrap();
    scalar.write(&[10.0, 20.0, 30.0]).unwrap();
    let pair = device.texture_2d::<Rg32Float>(3, 1, sampled).unwrap();
    pair.write(&[
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 8.0),
        Vec2::new(8.0, 16.0),
    ])
    .unwrap();
    let counts = device.texture_2d::<Rgba32Uint>(2, 1, sampled).unwrap();
    counts
        .write(&[UVec4::new(1, 0, 0, 2), UVec4::new(3, 0, 0, 4)])
        .unwrap();
    let offsets = device.texture_2d::<Rg32Sint>(2, 1, sampled).unwrap();
    offsets
        .write(&[IVec2::new(5, 2), IVec2::new(1, 4)])
        .unwrap();
    let gathered = device
        .texture_2d::<Rgba32Float>(2, 1, TextureUse::STORAGE)
        .unwrap();
    let nearest = device.sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge));
    let bilinear = device.sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge));
    // The kernel's `nearest` parameter now gets the linear sampler, and its `bilinear` the nearest one.
    textures::gather::dispatch_extent(
        &device,
        Extent::new(4, 2, 1),
        &scalar,
        &pair,
        &counts,
        &offsets,
        &bilinear,
        &nearest,
        &gathered,
    )
    .expect("the twin of `gather` refused the call");
    let texels = gathered.read();
    let texels: Vec<[f32; 4]> = texels.iter().map(|t| [t.x, t.y, t.z, t.w]).collect();
    assert_eq!(texels, [[12.5, 0.0, 3.0, 3.0], [35.5, 16.0, 7.0, -3.0]]);
}
