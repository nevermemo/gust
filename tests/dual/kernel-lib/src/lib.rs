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

    // The same overflow through a vector operator (m6.a2): `UVec2 + UVec2` is `impl Add for UVec2` in `gust::math`.
    // The twin is compiled in THIS crate (consumer/src/lib.rs, module `vector_overflow`, says what it does).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add_one_uvec2(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            let value = UVec2 {
                x: input[i],
                y: 0u32,
            };
            let one = UVec2 { x: 1u32, y: 0u32 };
            let sum = value + one;
            out[i] = sum.x;
        }
    }
}

/// The overflow mode of THIS package (`kernel-lib`): the macro expands here, so it answers for the crate that holds
/// the twin of `kernels::add_one`, whatever the dependent's own mode is.
pub fn overflow_mode() -> gust::cpu::OverflowChecks {
    gust::cpu::overflow_checks!()
}

// A routed module (m8.r1, D65 to D67): its target is chosen at build time. The tag `physics` is declared in this
// package's Cargo.toml. Routed to the CPU (`GUST_ROUTE__kernel_lib__physics=cpu`, which `cargo gust` sets from the
// app's `gust.toml` in m8.r2), the module is a CPU body in any build, with no `cfg(gust_dual)`: `consumer`'s `prod-routed`
// binary dispatches it on `CpuDevice` (check-dual E4). Unrouted it is a GPU kernel and the same call is E0277.
#[gust::gpu(route = "physics")]
pub mod routed {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = input[i] * 2.0;
        }
    }
}
