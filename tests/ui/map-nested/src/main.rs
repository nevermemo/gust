// `#[gpu]` on a function generates a module beside the function, which a `#[gpu]` module has no place for:
// GUST-E0716. The error points at the attribute.

#[gust::gpu]
mod shaders {
    #[gust::gpu]
    pub fn add_one(x: f32) -> f32 {
        x + 1.0
    }
}

fn main() {}
