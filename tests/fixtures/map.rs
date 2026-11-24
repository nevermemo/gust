// m7.a1: the map functions of the map tests. `tests/fixtures/map.slang` is the Slang of the kernel the generator makes
// for each, in this order; `crates/gust-wgpu/tests/map.rs` includes this file, so each is a real `#[gpu_fn]` function
// there (`saxpy::map(..)`, `saxpy::map_gpu(..)`, ...). Plain `//` comments only: the file is `include!`d.

// Floats, with a broadcast scalar in the tests: `a` is `Broadcast(2.0)`.
#[gust::gpu_fn]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 {
    a * x + y
}

// Signed integers and a branch.
#[gust::gpu_fn]
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
#[gust::gpu_fn]
pub fn even_and(n: u32, flag: bool) -> bool {
    (n & 1u32) == 0u32 && flag
}

// A vector element: `Vec2` is a buffer element (8 bytes, alignment 8).
#[gust::gpu_fn]
pub fn mirror(v: Vec2, s: f32) -> Vec2 {
    Vec2::new(v.x * s, -v.y * s)
}

// One round of the kernel's grid-stride loop covers 16 776 960 elements; the tests run more than that through it.
#[gust::gpu_fn]
pub fn add_one(x: u32) -> u32 {
    x + 1u32
}

// m14.mapw1: a user struct is a map's element when the function is written inside the `#[gpu_module]` module that
// declares it (`records::scale_item::map(..)`). The struct has a vector field, so its stride is the 16 bytes of
// `Vec2` (8, alignment 8) + `u32` + `f32`.
#[gust::gpu_module]
pub mod records {
    pub struct Item {
        pub pos: Vec2,
        pub id: u32,
        pub weight: f32,
    }

    // A struct in and a struct out, with a broadcast scalar beside it.
    #[gust::gpu_fn]
    pub fn scale_item(item: Item, k: f32) -> Item {
        let mut scaled = item;
        scaled.pos = Vec2::new(item.pos.x * k, item.pos.y + k);
        scaled.id = item.id + 1u32;
        scaled.weight = item.weight * k;
        scaled
    }

    // A struct in and a scalar out.
    #[gust::gpu_fn]
    pub fn item_total(item: Item) -> f32 {
        item.pos.x + item.pos.y + item.weight + item.id as f32
    }
}

// m14.mapw2: structs whose fields are 16-aligned or are structs themselves. A struct may not have implicit padding
// (GUST-E0208), so each scalar before a wide field is followed by explicit fill words. `Frame`: `id` at 0, `fill_a` at
// 4, `dir` (`Vec4`) at 16, `tag` at 32, `fill_b` at 36, `basis` (`Mat4`) at 48, size 112. `Slot` nests it: `flag` at 0,
// `fill_a` at 4, `frame` at 16, `weight` at 128, `fill_b` at 132, size 144.
#[gust::gpu_module]
pub mod wide {
    pub struct Frame {
        pub id: u32,
        pub fill_a: [u32; 3],
        pub dir: Vec4,
        pub tag: u32,
        pub fill_b: [u32; 3],
        pub basis: Mat4,
    }

    pub struct Slot {
        pub flag: u32,
        pub fill_a: [u32; 3],
        pub frame: Frame,
        pub weight: f32,
        pub fill_b: [u32; 3],
    }

    // A struct with a `Vec4` and a `Mat4` field, in and out.
    #[gust::gpu_fn]
    pub fn frame_mix(frame: Frame, k: f32) -> Frame {
        let mut mixed = frame;
        mixed.id = frame.id + 1u32;
        mixed.fill_a = frame.fill_b;
        mixed.dir = frame.basis.mul_vec4(frame.dir) * k;
        mixed.tag = frame.tag + frame.id;
        mixed.fill_b = frame.fill_a;
        mixed.basis = frame.basis.transpose();
        mixed
    }

    // A struct with a struct field (itself holding a `Vec4` and a `Mat4`), in and out.
    #[gust::gpu_fn]
    pub fn slot_step(slot: Slot, k: f32) -> Slot {
        let mut stepped = slot;
        stepped.flag = slot.flag + 1u32;
        stepped.fill_a = slot.fill_b;
        stepped.frame.id = slot.frame.tag;
        stepped.frame.fill_a = slot.frame.fill_b;
        stepped.frame.dir = slot.frame.basis.mul_vec4(slot.frame.dir) + Vec4::splat(k);
        stepped.frame.tag = slot.frame.id;
        stepped.frame.fill_b = slot.frame.fill_a;
        stepped.frame.basis = slot.frame.basis.transpose();
        stepped.weight = slot.weight * k;
        stepped.fill_b = slot.fill_a;
        stepped
    }
}
