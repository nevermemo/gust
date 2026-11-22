// `#[gpu_module]` goes on a module. Written on a function it is GUST-E0702, and the message names `#[gpu_fn]`. The
// function's own `#[kernel]` is dropped, so this is the one error and not a second one from rustc.

#[gust::gpu_module]
#[kernel]
pub fn f(x: f32) -> f32 {
    x + 1.0
}

fn main() {}
