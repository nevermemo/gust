// `#[gpu_module]` on a function inside a `#[gpu_module]` module is GUST-E0716 (`#[gpu_fn]` there is allowed since
// m14.mapw1, so it is a map over the module's structs). The error points at the attribute.

#[gust::gpu_module]
mod shaders {
    #[gust::gpu_module]
    pub fn add_one(x: f32) -> f32 {
        x + 1.0
    }
}

fn main() {}
