// The kernel crate of the precompiled fixture: the vector-add shape of `examples/vector-add`. `cargo gust build`
// compiled it ahead of time; the result is committed in `../gust-artifacts/`, and a build under `GUST_PROFILE=native`
// embeds it and needs no Slang. Editing the kernel without running `cargo gust build` again fails the identity check.

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
