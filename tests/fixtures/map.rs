// m7.a1: the map functions of the map tests. `tests/fixtures/map.slang` is the Slang of the kernel the generator makes
// for each, in this order; `crates/gust-wgpu/tests/map.rs` includes this file, so each is a real `#[gpu]` function
// there (`saxpy::map(..)`, `saxpy::map_gpu(..)`, ...). Plain `//` comments only: the file is `include!`d.

// Floats, with a broadcast scalar in the tests: `a` is `Broadcast(2.0)`.
#[gust::gpu]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

// Signed integers and a branch.
#[gust::gpu]
pub fn clamp_signed(x: i32, low: i32, high: i32) -> i32 {
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

// `bool` in and out: 0/1 words in the buffers, `Vec<bool>` on the host.
#[gust::gpu]
pub fn even_and(n: u32, flag: bool) -> bool {
    (n & 1u32) == 0u32 && flag
}

// A vector element: `Vec2` is a buffer element (8 bytes, alignment 8).
#[gust::gpu]
pub fn mirror(v: Vec2, s: f32) -> Vec2 {
    Vec2::new(v.x * s, -v.y * s)
}

// One round of the kernel's grid-stride loop covers 16 776 960 elements; the tests run more than that through it.
#[gust::gpu]
pub fn add_one(x: u32) -> u32 {
    x + 1u32
}
