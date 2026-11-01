// The kernel crate of the CPU-target fixture. Built under `GUST_PROFILE=cpu` the macro emits, for each kernel that has
// a typed call, `impl Run<kernel, N> for gust::cpu::CpuDevice` in every build (no `cfg(test)`, no `--cfg gust_dual`),
// so a dependent crate can dispatch on the CPU in a production binary. Built under any other profile the impl is
// absent and the dispatch is the E0277 that `app/src/bin/single_dispatch.rs` expects (e-d-expected.txt).

#[gust::gpu]
pub mod vector_add {
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

// The expression of `examples/polynomial`, so its derived bound applies: v^3 - 2 v^2 + 0.5 v + 1, written order.
#[gust::gpu]
pub mod polynomial {
    #[kernel(workgroup_size(128, 1, 1))]
    pub fn evaluate(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let value = input[i];
            let squared = value * value;
            let cubed = squared * value;
            let curved = cubed - (2.0 * squared);
            let linear = 0.5 * value;
            out[i] = (curved + linear) + 1.0;
        }
    }
}

// Every invocation increments the same counter: the result is the number of invocations that executed the add.
#[gust::gpu]
pub mod atomic_counter {
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn increment(_id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
        atomic_add(&mut counter[0], 1u32);
    }
}

// The kernel that opted out of the CPU: it has no `Run` impl for `CpuDevice` under any profile, `cpu` included.
#[gust::gpu]
pub mod opt_out {
    #[kernel(cpu = "unsupported")]
    pub fn negate(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = -input[i];
        }
    }
}
