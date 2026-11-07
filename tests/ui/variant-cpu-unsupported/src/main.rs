// A kernel with no CPU form, in a crate whose `profiles` list names `cpu` (GUST-E0744). Without the list the default
// list's CPU entry is met by the `cpu = "unsupported"` the kernel wrote; only a list the crate wrote can be contradicted.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
