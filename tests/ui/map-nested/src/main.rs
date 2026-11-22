// `#[gpu_fn]` on a function generates a module beside the function, which a `#[gpu_module]` module has no place for:
// GUST-E0716. The error points at the attribute.

#[gust::gpu_module]
mod shaders {
    #[gust::gpu_fn]
    pub fn add_one(x: f32) -> f32 {
        x + 1.0
    }
}

fn main() {}
