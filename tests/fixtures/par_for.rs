// `par_for!` calls (slice m7.p, plan section 3.1). The including module names the device type `Dev`
// (`HeadlessDevice` for the GPU run, `CpuDevice` for the twin run), so one text serves both backends.
//
// Every call runs `on(device)`, the form that names the device; without it `par_for!` opens the default device. The
// bodies are the dialect, so each is also a golden test of the Slang the macro makes for it
// (`crates/gust-macros/src/tests/par_for.rs`, `tests/fixtures/par_for.slang`).

/// One dimension: `out[i] = a[i] * b[i]` over `out.len()` indexes, not a multiple of the 256-wide workgroup in tests.
pub fn product_1d(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    device: &Dev,
) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] {
        out[i] = a[i] * b[i];
    })
}

/// The same loop with the workgroup size overridden: 64 invocations along x.
pub fn product_64(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    device: &Dev,
) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(workgroup_size(64, 1, 1); on(device); i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] {
        out[i] = a[i] * b[i];
    })
}

/// A prefix of a buffer: the extent is shorter than `out`, so only the bounds guard keeps the invocations of the last
/// workgroup from writing the elements past `n`.
pub fn fill_prefix(out: &mut [u32], n: u32, device: &Dev) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); i in 0..n, [out: &mut [u32]] {
        out[i] = i + 1u32;
    })
}

/// Two dimensions: a row-major image scaled by one half, `w` x `h` cells in 16 x 16 workgroups.
pub fn halve_2d(
    img: &[f32],
    out: &mut [f32],
    w: u32,
    h: u32,
    device: &Dev,
) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); (x, y) in (0..w, 0..h), [img: &[f32], out: &mut [f32], w: u32] {
        out[y * w + x] = img[y * w + x] * 0.5;
    })
}

/// Two dimensions over a buffer that is larger than the extent (rows of `stride` cells): the guard on each axis.
pub fn tag_2d(
    out: &mut [u32],
    w: u32,
    h: u32,
    stride: u32,
    device: &Dev,
) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); (x, y) in (0..w, 0..h), [out: &mut [u32], stride: u32] {
        out[y * stride + x] = y * 1000u32 + x + 1u32;
    })
}

/// Three dimensions: one packed coordinate per cell, in 8 x 8 x 4 workgroups.
pub fn index_3d(
    out: &mut [u32],
    w: u32,
    h: u32,
    d: u32,
    device: &Dev,
) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); (x, y, z) in (0..w, 0..h, 0..d), [out: &mut [u32], w: u32, h: u32] {
        out[(z * h + y) * w + x] = z * 1000000u32 + y * 1000u32 + x;
    })
}

/// Scalars of every kind in the extent record: `f32`, `bool` (a `u32` word on the device) and `i32`.
pub fn blend(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    t: f32,
    invert: bool,
    bias: i32,
    device: &Dev,
) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [
        a: &[f32], b: &[f32], out: &mut [f32], t: f32, invert: bool, bias: i32
    ] {
        let mixed = a[i] * (1.0 - t) + b[i] * t + bias as f32;
        if invert {
            out[i] = 0.0 - mixed;
        } else {
            out[i] = mixed;
        }
    })
}

/// The extent is a `u64`: `TryInto<u32>` converts it, and one past `u32::MAX` is an `Err` before any device work.
pub fn fill_u64(out: &mut [u32], n: u64, device: &Dev) -> Result<(), gust::map::Error> {
    gust::par_for!(on(device); i in 0..n, [out: &mut [u32]] {
        out[i] = 7u32;
    })
}

/// A one-dimensional loop in a workgroup taller than the loop: 64 x 4 x 1. The invocations of rows 1 to 3 have `x`
/// in range, so only the guard on the unused axes keeps each element from being bumped four times.
pub fn bump_tall(out: &mut [u32], device: &Dev) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(workgroup_size(64, 4, 1); on(device); i in 0..n, [out: &mut [u32]] {
        out[i] += 1u32;
    })
}

/// A two-dimensional loop in a workgroup with a third axis: 8 x 8 x 2, so the guard on `z` is what runs each cell once.
pub fn bump_deep(out: &mut [u32], w: u32, h: u32, device: &Dev) -> Result<(), gust::map::Error> {
    gust::par_for!(workgroup_size(8, 8, 2); on(device); (x, y) in (0..w, 0..h), [out: &mut [u32], w: u32] {
        out[y * w + x] += 1u32;
    })
}
