// A route whose `targets` list is empty (D138, m8.x1). The macro reads the crate's Cargo.toml when it expands a
// `#[gpu]` module and reports GUST-E0753 at the module's name, before it looks at the route; without this check every
// route of the tag would be a GUST-E0749 that names the empty set. Omit the key for the default.

#[gust::gpu(route = "physics")]
mod kernels {
    #[kernel]
    fn fill(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = id.x;
    }
}

fn main() {}
