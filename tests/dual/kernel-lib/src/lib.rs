// The kernel crate of the cross-crate fixture. The `#[gust::gpu]` module is in the vector-add shape. The CPU twin
// of `add` is emitted into THIS crate under `cfg(any(test, gust_dual))` (D26), so a dependent crate can run it only
// when `kernel-lib` itself is compiled with `--cfg gust_dual`: its own `cfg(test)` does not reach a dependency.

#[gust::gpu]
pub mod kernels {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}
