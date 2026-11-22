// `#[gpu_fn]` goes on a free scalar function. Written on a module it is GUST-E0702, and the message names
// `#[gpu_module]`, the attribute that fits a module.

#[gust::gpu_fn]
mod shaders {
    pub fn f(x: f32) -> f32 {
        x + 1.0
    }
}

fn main() {}
