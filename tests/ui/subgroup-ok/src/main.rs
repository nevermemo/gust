// Control case for the subgroup operations (m8.s): a portable kernel and a `requires(subgroups)` variant that combines
// the invocations of a subgroup at the accepted positions, a statement of the body and a statement of a top-level
// `for` with two literal bounds. Built under `GUST_PROFILE=dev+subgroups` (env.txt) so that rustc compiles the
// SELECTED variant: its descriptor with `requires: &[Capability::Subgroups]` and its Slang. This crate must compile.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x < data.len() {
            out[id.x] = data[id.x];
        }
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let mut total = subgroup_add(x);
        for step in 0u32..2u32 {
            total += subgroup_max(x + step);
        }
        let first = subgroup_elect();
        let ballot = subgroup_ballot(x > 3u32);
        let width = subgroup_size();
        if id.x < data.len() {
            out[id.x] = 0u32;
            if first {
                out[id.x] = total + ballot.x + width;
            }
        }
    }
}

// What the build selected is in the descriptor, and rustc checked it.
const _: () = assert!(scan::totals::DESCRIPTOR.requires.len() == 1);

fn main() {}
