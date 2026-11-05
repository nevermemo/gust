//! Control case for the `missing_docs` lint (slice mr.b2): a crate with `#![deny(missing_docs)]` compiles over a
//! `#[gust::gpu]` function, whose expansion adds a module of the same name with the map calls, and over a `par_for!`
//! call. Built under `GUST_PROFILE=cpu` (env.txt), where the generated calls run on `gust::cpu::CpuDevice` and nothing
//! names a GPU API; `missing-docs-mod-ok` covers the module forms in the dev profile. This crate must compile.
#![deny(missing_docs)]

use gust::{cpu::CpuDevice, math::Vec2};

/// `a * x + y` for every element.
#[gust::gpu]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

/// Shifts a vector element by a scalar.
#[gust::gpu]
pub fn shift(v: Vec2, s: f32) -> Vec2 {
    Vec2::new(v.x + s, v.y + s)
}

fn scale(values: &[f32], out: &mut [f32], by: f32, device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [values: &[f32], out: &mut [f32], by: f32] {
        out[i] = values[i] * by;
    })
}

fn main() {
    let device = CpuDevice::new();
    let mut out = [0.0_f32; 3];
    scale(&[1.0, 2.0, 3.0], &mut out, 2.0, &device).unwrap();
}
