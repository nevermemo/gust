// The chapters of `gust_wgpu::guide`, one empty module each (the semver rule: the guide modules hold nothing but chapter
// modules, which hold nothing; they exist for their documentation). Chapter 10 (benchmarks) has no listings and no Rust
// code block (its two `sh` blocks are not tested by rustdoc), so it adds no doctest.
//
// The listings are packaged copies of regions of the gallery entries (`check-gallery` keeps them identical to the
// regions). The programs of chapters 4 and 5 are compiled by this crate's doctests against the real backend, and so is
// the host half of chapter 9's page (its kernel and reference, in one `rust` block with an empty `main`, which the
// doctest also runs). Two listings are plain-text blocks and not doctests: chapter 6's `routed` program (a route tag
// must be declared in the crate that uses it, GUST-E0748), which `check-gallery` builds as the gallery entry, and the
// browser-side function of chapter 9 (it exists on `wasm32` only). An `ignore` block would count as an ignored test,
// which the Gpu suite refuses. The first program of chapter 2 of `gust::guide` is compiled by the private module below,
// which exists only in `cargo test --doc`.
#[cfg(doctest)]
#[doc = "```no_run"]
#[doc = include_str!("listings/hello-gust-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-main.rs")]
#[doc = "```"]
mod first_program {}

#[doc = include_str!("04-running-kernels-1.md")]
#[doc = include_str!("listings/saxpy-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/saxpy-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/saxpy-params.rs")]
#[doc = ""]
#[doc = include_str!("listings/saxpy-main.rs")]
#[doc = include_str!("04-running-kernels-2.md")]
#[doc = include_str!("listings/particles-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-setup.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-energy.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-total.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-host.rs")]
#[doc = ""]
#[doc = include_str!("listings/particles-main.rs")]
#[doc = include_str!("04-running-kernels-3.md")]
pub mod running_kernels {}

#[doc = include_str!("05-gpu-concepts-running-1.md")]
#[doc = include_str!("listings/image-threshold-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/image-threshold-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/image-threshold-host.rs")]
#[doc = ""]
#[doc = include_str!("listings/image-threshold-main.rs")]
#[doc = include_str!("05-gpu-concepts-running-2.md")]
#[doc = include_str!("listings/reduce-scan-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/reduce-scan-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/reduce-scan-host.rs")]
#[doc = ""]
#[doc = include_str!("listings/reduce-scan-main.rs")]
#[doc = include_str!("05-gpu-concepts-running-3.md")]
#[doc = include_str!("listings/variants-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/variants-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/variants-host.rs")]
#[doc = ""]
#[doc = include_str!("listings/variants-main.rs")]
#[doc = include_str!("05-gpu-concepts-running-4.md")]
pub mod gpu_concepts_running {}

#[doc = include_str!("06-profiles-artifacts-routing-1.md")]
#[doc = include_str!("listings/routed-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/routed-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/routed-host.rs")]
#[doc = ""]
#[doc = include_str!("listings/routed-main.rs")]
#[doc = include_str!("06-profiles-artifacts-routing-2.md")]
pub mod profiles_artifacts_routing {}

#[doc = include_str!("09-browser-1.md")]
#[doc = include_str!("listings/web-hello-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/web-hello-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/web-hello-reference.rs")]
#[doc = "# fn main() {}"]
#[doc = include_str!("09-browser-2.md")]
#[doc = include_str!("listings/web-hello-map-async.rs")]
#[doc = include_str!("09-browser-3.md")]
pub mod browser {}

#[doc = include_str!("10-benchmarks.md")]
pub mod benchmarks {}
