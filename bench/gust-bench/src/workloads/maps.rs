//! The map forms (`#[gpu_fn] fn`, D60) of W1 and W2, for the production CPU build only: under `GUST_PROFILE=cpu` the map
//! generator emits a parallel chunked run (`gust::cpu::parallel`, D119), the one parallel path of the `cpu` target.
//! The same arithmetic as the kernels of `vector_add` and `saxpy`. Not in the GPU builds, where the maps would only add
//! kernels nothing runs. W3 has no map form here: its coefficients are a runtime array, a map takes scalars.

use gust::gpu_fn;

#[allow(dead_code)]
#[gpu_fn]
pub fn add_map(a: f32, b: f32) -> f32 {
    a + b
}

#[allow(dead_code)]
#[gpu_fn]
pub fn saxpy_map(x: f32, y: f32, scale: f32) -> f32 {
    scale * x + y
}
