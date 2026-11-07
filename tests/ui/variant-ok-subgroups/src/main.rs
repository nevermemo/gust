// `variant-ok`, built under `GUST_PROFILE=dev+subgroups` (env.txt), so that rustc compiles the SELECTED variant: the
// descriptor with `requires: &[Capability::Subgroups]`, the variant's entry point and Slang, and a kernel's own body
// that needs a capability. Both kernels are covered by the default profiles: `sum` by its portable own body and
// `prod` by its `requires()` variant, which is also the CPU form of `prod`.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = sum, requires(subgroups))]
    pub fn sum_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
    pub fn prod(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x] * 2u32;
    }

    #[variant(of = prod, requires())]
    pub fn prod_portable(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x] * 2u32;
    }
}

// What the build selected is in the descriptors, and rustc checked them.
const _: () = {
    assert!(reduce::sum::DESCRIPTOR.requires.len() == 1);
    assert!(reduce::prod::DESCRIPTOR.requires.len() == 1);
};

fn main() {}
