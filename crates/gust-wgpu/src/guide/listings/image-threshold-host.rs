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
