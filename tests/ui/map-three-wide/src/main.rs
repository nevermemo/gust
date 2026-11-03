// The arguments and results of a map call are buffer elements, and a 3-wide vector has stride 16 in a buffer but size
// 12 in Rust: GUST-E0204. The same type is fine inside the body (map-ok). The error points at the type.

use gust::math::Vec3;

#[gust::gpu]
pub fn length_squared(v: Vec3) -> f32 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

fn main() {}
