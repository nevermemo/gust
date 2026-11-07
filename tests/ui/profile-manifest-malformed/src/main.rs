// A malformed `[package.metadata.gust]` table (D138). The macro reads the crate's Cargo.toml when it expands a
// `#[gpu]` module and reports GUST-E0753 at the module's name; the message carries the line and column in
// Cargo.toml, which no span of rustc can point into. The one value of `limits` is "portable".

#[gust::gpu]
mod kernels {
    #[kernel]
    fn fill(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        out[id.x] = id.x;
    }
}

fn main() {}
