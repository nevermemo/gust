// Control case for `#[gust::gpu]` on a function (D60): the accepted signatures expand and compile. Built under
// `GUST_PROFILE=cpu` (env.txt), where the generated `map` and `map_on` run on `gust::cpu::CpuDevice` and nothing names
// a GPU API. A scalar, `bool`, an integer, a vector and a matrix are all elements, and a 3-wide value is fine inside
// the body. The functions stay plain Rust.

use gust::{
    map::Broadcast,
    math::{Mat4, Vec2, Vec3},
};

#[gust::gpu]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

#[gust::gpu]
pub fn both(flag: bool, n: u32) -> bool {
    n % 2u32 == 0u32 && flag
}

#[gust::gpu]
pub fn lift(v: Vec2, s: f32) -> Vec2 {
    let w = Vec3::new(v.x, v.y, s);
    Vec2::new(w.x + w.z, w.y)
}

#[gust::gpu]
pub fn same(m: Mat4) -> Mat4 {
    m
}

fn main() {
    let sums = saxpy::map(Broadcast(2.0), &[1.0f32, 2.0], &[0.5f32, 0.5]).unwrap();
    assert_eq!(sums, [2.5, 4.5]);
    assert_eq!(both::map(Broadcast(true), &[1u32, 2]).unwrap(), [false, true]);
    assert_eq!(saxpy(1.0, 2.0, 3.0), 5.0);
    let _ = (lift::map(&[Vec2::new(1.0, 2.0)], Broadcast(0.5)), same::DESCRIPTOR);
}
