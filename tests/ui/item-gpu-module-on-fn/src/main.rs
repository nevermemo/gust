// `#[gpu_module]` goes on an inline module. Written on a function it is GUST-E0702, and the message names `#[gpu_fn]`,
// the attribute that fits a free scalar function.

#[gust::gpu_module]
pub fn f(x: f32) -> f32 {
    x + 1.0
}

fn main() {}
