// A `#[gpu_fn]` function in a crate with `limits = "portable"` is checked against the portable floor too (m11.floor):
// the generated map kernel binds one storage buffer per parameter and one for the result, so this one has 9 and
// GUST-E0746 points at the function's signature. expected.txt pins the code and the signature line.

#[gust::gpu_fn]
pub fn many(p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32) -> f32 {
    p0 + p7
}

fn main() {}
