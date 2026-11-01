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

    // An integer kernel whose `+` overflows for the input `u32::MAX`: `out[i] = input[i] + 1`. Its twin is compiled
    // in THIS crate, so it follows this package's `overflow-checks` (tests/dual/Cargo.toml overrides only the
    // consumer's).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add_one(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = input[i] + 1u32;
        }
    }
}

/// The overflow mode of THIS package (`kernel-lib`): the macro expands here, so it answers for the crate that holds
/// the twin of `kernels::add_one`, whatever the dependent's own mode is.
pub fn overflow_mode() -> gust::cpu::OverflowChecks {
    gust::cpu::overflow_checks!()
}
