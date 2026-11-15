/// How many elements the page adds: more than one workgroup (64) and not a multiple of it.
pub const LENGTH: usize = 1000;

/// The inputs of `n` elements: small integers, so that every sum is exact in `f32` on any target.
pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..n).map(|i| (i % 97) as f32).collect();
    let b = (0..n).map(|i| (i % 13) as f32 * 8.0).collect();
    (a, b)
}

/// The independent host reference: plain Rust over slices, which never goes through the kernel (D26, D42).
pub fn host_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
