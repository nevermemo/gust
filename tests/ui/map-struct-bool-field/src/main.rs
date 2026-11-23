// A `bool` is not a field of a struct a map takes: it has no layout a Rust type shares with the GPU's, so rustc reports
// that `bool` is not a `GpuPod` (the module expansion asks for the layout of each field). A flag is a `u32` that is 0
// or 1. The map's own `bool` parameters and results are different: they travel as 0/1 words.

#[gust::gpu_module]
mod shaders {
    pub struct Flagged {
        pub on: bool,
        pub n: u32,
    }

    #[gust::gpu_fn]
    pub fn count(item: Flagged) -> u32 {
        item.n
    }
}

fn main() {}
