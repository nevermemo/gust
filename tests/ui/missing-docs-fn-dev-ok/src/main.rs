//! Control case for the `missing_docs` lint (slice mr.b2): a crate with `#![deny(missing_docs)]` compiles over a
//! `#[gust::gpu]` function in the dev profile (no env.txt). Its expansion adds the module of the same name with
//! `map_gpu`, `map_gpu_on`, `map_async` and the generic `map_on`, which name `gust_wgpu` (D99), hence the direct
//! dependency. The cpu profile is `missing-docs-fn-ok`. This crate must compile.
#![deny(missing_docs)]

use gust::math::Vec2;

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

fn main() {}
