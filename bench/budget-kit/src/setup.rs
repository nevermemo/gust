//! Shared setup: the plan (steps, sizes), the host data and the CPU references every row checks against.

use gust::{Extent, math::Vec4};

// ---- the plan ----

/// Steps counted per cell, and steps run first and not counted.
pub(crate) const STEPS: usize = 30;
pub(crate) const WARMUP: usize = 5;
/// One frame at 60 Hz, in milliseconds.
pub(crate) const FRAME_MS: f64 = 1000.0 / 60.0;
// The width of the invocation grid the kernels index through: the module-level constant of the `#[gpu]` module.
pub(crate) use crate::kernels::kit::ROW;

/// Particle and field sizes (the counts are exact; "1M" is 2^20 and "4M" is 2^22).
pub(crate) const W1_SIZES: [u32; 3] = [1 << 20, 1 << 22, 1 << 24];
pub(crate) const W2_SIDES: [u32; 3] = [512, 1024, 2048];
pub(crate) const W3_SIZES: [u32; 3] = [100_000, 1 << 20, 1 << 22];
pub(crate) const W4_SIZES: [u32; 3] = [100_000, 1 << 20, 1 << 22];
pub(crate) const W5_SIZES: [u32; 3] = [1 << 20, 1 << 22, 1 << 24];
pub(crate) const W6_SIZES: [u32; 3] = [1, 4096, 1 << 20];
/// The field side W3 samples, and the grid side W4 bins into.
pub(crate) const W3_FIELD_SIDE: u32 = 1024;
pub(crate) const W4_GRID_SIDE: u32 = 256;
/// Steps of W2 run before its result is compared with the CPU.
pub(crate) const W2_CHECKED_STEPS: usize = 3;

pub(crate) fn extent(n: u32) -> Extent {
    Extent::new(ROW, n.div_ceil(ROW), 1)
}

pub(crate) fn dbg<E: std::fmt::Debug>(error: E) -> String {
    format!("{error:?}")
}

// ---- host data ----

pub(crate) fn mix(i: u32) -> u32 {
    let mut h = i.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_507);
    h ^ (h >> 13)
}

/// A value in [0, 1), exact in `f32` (24 bits).
pub(crate) fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

pub(crate) fn field_init(i: u32) -> Vec4 {
    Vec4::new(
        unit(mix(i * 4)),
        unit(mix(i * 4 + 1)),
        unit(mix(i * 4 + 2)),
        unit(mix(i * 4 + 3)),
    )
}

/// A particle with its position in [0, side) and its velocity in [-1, 1).
pub(crate) fn particle_init(i: u32, side: u32) -> Vec4 {
    let s = side as f32;
    Vec4::new(
        unit(mix(i * 4)) * s,
        unit(mix(i * 4 + 1)) * s,
        unit(mix(i * 4 + 2)) * 2.0 - 1.0,
        unit(mix(i * 4 + 3)) * 2.0 - 1.0,
    )
}

pub(crate) fn lanes(v: Vec4) -> [f32; 4] {
    [v.x, v.y, v.z, v.w]
}

// ---- CPU references ----

/// The expression of `w2_diffuse`, in the same order, on the host.
pub(crate) fn cpu_diffuse(input: &[Vec4], side: u32) -> Vec<Vec4> {
    let at = |x: u32, y: u32| lanes(input[(y * side + x) as usize]);
    (0..side * side)
        .map(|i| {
            let (x, y) = (i % side, i / side);
            let c = at(x, y);
            let l = at((x + side - 1) % side, y);
            let r = at((x + 1) % side, y);
            let u = at(x, (y + side - 1) % side);
            let d = at(x, (y + 1) % side);
            let lane = |k: usize| (c[k] * 0.5 + (l[k] + r[k] + u[k] + d[k]) * 0.125) * 0.998;
            Vec4::new(lane(0), lane(1), lane(2), lane(3))
        })
        .collect()
}

/// The expression of `w3_step`, on the host, for one particle.
pub(crate) fn cpu_particle(q: Vec4, field: &[Vec4], side: u32) -> Vec4 {
    let s = side as f32;
    let cx = (q.x as u32).min(side - 1);
    let cy = (q.y as u32).min(side - 1);
    let f = field[(cy * side + cx) as usize];
    let vx = q.z + (f.x - 0.5) * 0.05;
    let vy = q.w + (f.y - 0.5) * 0.05;
    let mut x = q.x + vx;
    let mut y = q.y + vy;
    if x >= s {
        x -= s;
    }
    if x < 0.0 {
        x += s;
    }
    if y >= s {
        y -= s;
    }
    if y < 0.0 {
        y += s;
    }
    Vec4::new(x, y, vx, vy)
}

/// The per-cell counts of particles, on the host.
pub(crate) fn cpu_counts(parts: &[Vec4], side: u32) -> Vec<u32> {
    let mut counts = vec![0u32; (side * side) as usize];
    for q in parts {
        let cx = (q.x as u32).min(side - 1);
        let cy = (q.y as u32).min(side - 1);
        counts[(cy * side + cx) as usize] += 1;
    }
    counts
}

/// The bin of a position on a `bins x bins` grid over a field of side `field`: the host's form of `bin_of` (the same
/// scale, exact for powers of two). With `bins == field` it is the cell of `cpu_counts`.
pub(crate) fn bin_of_host(q: &Vec4, bins: u32, field: u32) -> usize {
    let scale = bins as f32 / field as f32;
    let cx = ((q.x * scale) as u32).min(bins - 1);
    let cy = ((q.y * scale) as u32).min(bins - 1);
    (cy * bins + cx) as usize
}

/// The per-bin counts of particles on a `bins x bins` grid over a field of side `field`, on the host.
pub(crate) fn cpu_bin_counts(parts: &[Vec4], bins: u32, field: u32) -> Vec<u32> {
    let mut counts = vec![0u32; (bins * bins) as usize];
    for q in parts {
        counts[bin_of_host(q, bins, field)] += 1;
    }
    counts
}

pub(crate) fn cpu_exclusive(counts: &[u32]) -> Vec<u32> {
    let mut sum = 0u32;
    counts
        .iter()
        .map(|c| {
            let before = sum;
            sum += c;
            before
        })
        .collect()
}
