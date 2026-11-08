// Step E-j (m8.r1, D65 to D67): a routed module in a production CPU build. The tag `physics` of this crate's Cargo.toml
// is routed to the CPU by `GUST_ROUTE__routed__physics=cpu`, so `scale` is a CPU body whatever `GUST_PROFILE` is
// (E-j builds it under the strict profile `native`, with no artifact and no `cargo gust build`): `Run` for `CpuDevice` is
// ungated and no artifact is required. Without the route the module is a GPU kernel and the same crate fails under
// `native` with the missing-artifact error that `e-j-expected.txt` pins (the line of `pub fn scale` below). The module
// lives in the library so that this is the crate's only error: the binary, which runs the kernel on `CpuDevice` and so
// would add a second error for a GPU kernel, is not compiled once the library has failed.

#[gust::gpu(route = "physics")]
pub mod kernels {
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
