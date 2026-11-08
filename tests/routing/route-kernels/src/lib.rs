//! The kernels of the routing fixture (m8.r2). One module per tag; the target of each is chosen at build time (D65 to D67)
//! by the variable `cargo gust` sets from an app's `gust.toml`, so the same source is a GPU kernel or a CPU body.

/// Tag `physics` (targets gpu and cpu, default gpu).
#[gust::gpu(route = "physics")]
pub mod physics {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}

/// Tag `physics.broadphase`, a child of `physics`: it follows an entry or a default of `physics` unless it has its own.
#[gust::gpu(route = "physics.broadphase")]
pub mod broadphase {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn pairs(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] + 1.0;
        }
    }
}

/// Tag `fusion` (no `targets`, default cpu).
#[gust::gpu(route = "fusion")]
pub mod fusion {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn blend(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 0.5;
        }
    }
}
