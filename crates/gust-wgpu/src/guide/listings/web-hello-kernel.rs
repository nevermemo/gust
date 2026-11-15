/// The element-wise sum of two vectors. `#[gpu]` on a plain scalar function keeps the function as written and adds a
/// module of the same name with the kernel and the calls that run it (`map`, `map_async`, ..).
#[gpu]
pub fn add(a: f32, b: f32) -> f32 {
    a + b
}
