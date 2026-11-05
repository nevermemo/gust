//! The pixel fixture on the CPU twin and against its own descriptors (roadmap MX, slice mx.p1a; Unit tier, no adapter,
//! no slangc).
//!
//! `tests/fixtures/pixel.rs` is the module of the golden `pixel.slang`. Including it here compiles its expansion with
//! rustc, which the string tests of the macro cannot do: the `shape = pixel` pin, the one-dimensional typed calls
//! (a `pixel` call has a count, not an extent) and the twin that runs the body once per element, as the `Linear`
//! harness does. Every output is compared with a value worked by hand. The twin is never GPU evidence: no fragment
//! ran on a device in this slice (that is mx.p1b).

include!("../../../tests/fixtures/pixel.rs");

use gust::cpu::CpuDevice;
use gust::math::{Vec2, Vec4};
use gust::texture::{Address, Filter, Rgba32Float, SamplerOptions, TextureUse};
use gust::{ParameterKind, Shape};

/// The word in `[shader("..")]` of a descriptor's Slang source.
fn stage_word(source: &str) -> &str {
    let after = source
        .split_once("[shader(\"")
        .expect("the source has a shader attribute")
        .1;
    after.split_once("\")]").expect("the attribute closes").0
}

/// The stage string the macro writes into the Slang text is the one `Shape::slang_stage()` answers, which is what the
/// compile paths pass to `slangc -stage` and the reflection helper: the two cannot drift apart unnoticed.
#[test]
fn the_stage_in_the_slang_text_is_the_stage_of_the_shape() {
    let kernels = pixel::MODULE_DESCRIPTOR.kernels;
    assert_eq!(kernels.len(), 3);
    for kernel in kernels {
        assert_eq!(kernel.shape, Shape::Pixel, "{}", kernel.name);
        assert_eq!(
            stage_word(kernel.slang_source),
            kernel.shape.slang_stage(),
            "{}",
            kernel.name
        );
        assert_eq!(Shape::Pixel.slang_stage(), "fragment");
        assert!(
            kernel.slang_source.contains("SV_Position"),
            "{}: a fragment entry reads its position",
            kernel.name
        );
        assert!(
            !kernel.slang_source.contains("SV_Target"),
            "{}",
            kernel.name
        );
        // The descriptor of a fragment carries a workgroup size of one: the field exists for every kernel, and a
        // fragment has no workgroup (the host draws one fragment per element).
        assert_eq!(kernel.workgroup_size, [1, 1, 1], "{}", kernel.name);
        assert!(kernel.compatible_shapes.contains(&Shape::Pixel));
    }
    // A grid kernel of the same crate says compute; the two words are not the same string.
    assert_eq!(Shape::Grid.slang_stage(), "compute");
}

/// An output is an ordinary read-write storage parameter with a binding: a `pixel` kernel's outputs are fragment
/// storage writes (D120), described like a `grid` kernel's.
#[test]
fn the_outputs_of_a_pixel_kernel_are_storage_parameters_with_bindings() {
    let shade = &pixel::MODULE_DESCRIPTOR.kernels[0];
    assert_eq!(shade.name, "shade");
    let storage: Vec<_> = shade
        .parameters
        .iter()
        .filter(|parameter| parameter.kind == ParameterKind::Storage)
        .collect();
    assert_eq!(storage.len(), 3);
    let slots: Vec<u32> = storage
        .iter()
        .map(|parameter| {
            parameter
                .binding
                .expect("a storage parameter has a slot")
                .binding
        })
        .collect();
    assert_eq!(slots, [0, 1, 2]);
}

/// `shade`: `scaled = 2 * v + 1` and `flags = (v > 0.5)`. For `v = 0, 0.25, 0.5, 0.75, 2`: scaled `1, 1.5, 2, 2.5, 5`
/// and flags `0, 0, 0, 1, 1` (`0.5 > 0.5` is false).
#[test]
fn shade_writes_both_outputs_of_every_element() {
    let device = CpuDevice::new();
    let input = device.buffer(&[0.0f32, 0.25, 0.5, 0.75, 2.0]);
    let scaled = device.buffer(&[f32::NAN; 5]);
    let flags = device.buffer(&[7u32; 5]);
    pixel::shade::dispatch(&device, 5, &input, &scaled, &flags)
        .expect("the twin of `shade` refused the call");
    assert_eq!(scaled.read(), [1.0, 1.5, 2.0, 2.5, 5.0]);
    assert_eq!(flags.read(), [0, 0, 0, 1, 1]);
}

/// `pack`: `sums = a + 1`, `wide = (a.x, a.y, 0, 1)`, `steps = b + 1`, for `a = (1, 2), (3, 4)` and `b = 5, -7`.
#[test]
fn pack_writes_vector_and_signed_outputs() {
    let device = CpuDevice::new();
    let a = device.buffer(&[Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)]);
    let b = device.buffer(&[5i32, -7]);
    let sums = device.buffer(&[Vec2::new(f32::NAN, f32::NAN); 2]);
    let wide = device.buffer(&[Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN); 2]);
    let steps = device.buffer(&[0i32; 2]);
    pixel::pack::dispatch(&device, 2, &a, &b, &sums, &wide, &steps)
        .expect("the twin of `pack` refused the call");
    let sums: Vec<[f32; 2]> = sums.read().iter().map(|v| [v.x, v.y]).collect();
    assert_eq!(sums, [[2.0, 3.0], [4.0, 5.0]]);
    let wide: Vec<[f32; 4]> = wide.read().iter().map(|v| [v.x, v.y, v.z, v.w]).collect();
    assert_eq!(wide, [[1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]);
    assert_eq!(steps.read(), [6, -6]);
}

/// `row`: the pixel `x` samples a 16 by 1 texture at `u = (x + 0.5) / 16`, the centre of texel `x`, so the bilinear
/// sample is exactly the texel `(x, 2x, 3, 1)`; the texture's width, 16, is added to every component:
/// `(x + 16, 2x + 16, 19, 17)`.
#[test]
fn row_samples_the_texture_at_the_pixels_own_column_and_reads_its_width() {
    let device = CpuDevice::new();
    let image = device
        .texture_2d::<Rgba32Float>(16, 1, TextureUse::SAMPLED)
        .unwrap();
    let texels: Vec<Vec4> = (0..16)
        .map(|j| Vec4::new(j as f32, 2.0 * j as f32, 3.0, 1.0))
        .collect();
    image.write(&texels).unwrap();
    let bilinear = device.sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge));
    let colours = device.buffer(&[Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN); 4]);
    pixel::row::dispatch(&device, 4, &image, &bilinear, &colours)
        .expect("the twin of `row` refused the call");
    let colours: Vec<[f32; 4]> = colours
        .read()
        .iter()
        .map(|v| [v.x, v.y, v.z, v.w])
        .collect();
    assert_eq!(
        colours,
        [
            [16.0, 16.0, 19.0, 17.0],
            [17.0, 18.0, 19.0, 17.0],
            [18.0, 20.0, 19.0, 17.0],
            [19.0, 22.0, 19.0, 17.0],
        ]
    );
}
