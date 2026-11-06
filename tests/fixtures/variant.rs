#[gust::gpu]
mod variant {
    // Variants of a kernel (M8, m8.b1, D139). `double` is the kernel and its portable body; `double_subgroups` is the
    // body a device with subgroups gets instead. Both are plain bodies here: no subgroup operation exists in the
    // dialect until m8.s, so the two compute the same thing and the fixture pins what the macro does with the pair,
    // not what a wave could do. The two functions have one signature, token for token, and a variant takes the
    // workgroup size (and the shape and the CPU form) of its kernel: it has no options of its own.
    //
    // Every body is validated and lowered in every build and has its own entry point (`gpu_variant_double` and
    // `gpu_variant_double_subgroups`), so its own content key and its own export record. A build selects one: the
    // body with the largest `requires` that the profile's capabilities contain.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn double(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x < out.len() {
            out[id.x] = data[id.x] * 2u32;
        }
    }

    #[variant(of = double, requires(subgroups))]
    pub fn double_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x < out.len() {
            out[id.x] = data[id.x] + data[id.x];
        }
    }
}
