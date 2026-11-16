// The chapters of `gust::guide`, one empty module each (the semver rule: the guide modules hold nothing but chapter
// modules, which hold nothing; they exist for their documentation). Chapters 4, 6, 9 and 10, and the running halves of
// chapter 5, live in `gust_wgpu::guide` (the index says so for each).
//
// The links to `gust_wgpu::guide` are built here, from this crate's version, because a Markdown file cannot read
// `CARGO_PKG_VERSION`: the index ends with them (inline links; a link definition in this inner doc was accepted by
// stable rustdoc but not by nightly with `--cfg docsrs`), and each chapter defines the ones it uses for its own page.
// The versions of the two crates are locked together, so a link names the sibling guide of the same release (it works
// once that release is published and its docs have built).
#![doc = concat!(
    "\n\nThe sibling guide: [`gust_wgpu::guide`](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/index.html). Its chapters: [4, running kernels](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/running_kernels/index.html), [5, GPU concepts, running half](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/gpu_concepts_running/index.html), [6, build profiles, artifacts and routing](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/profiles_artifacts_routing/index.html), [9, the browser](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/browser/index.html) and [10, benchmarks](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/benchmarks/index.html).\n"
)]

#[doc = include_str!("01-why-gust.md")]
pub mod why_gust {}

#[doc = include_str!("02-install-a.md")]
#[doc = include_str!("listings/hello-gust-kernel.rs")]
#[doc = include_str!("02-install-b.md")]
#[doc = include_str!("listings/hello-gust-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-main.rs")]
#[doc = include_str!("02-install-c.md")]
#[doc = concat!(
    "\n\n[`gust_wgpu::guide`]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/index.html\n"
)]
pub mod install_and_first_program {}

#[doc = include_str!("03-authoring-model-1.md")]
#[doc = include_str!("listings/saxpy-kernel.rs")]
#[doc = include_str!("03-authoring-model-2.md")]
#[doc = include_str!("listings/histogram-kernel.rs")]
#[doc = include_str!("03-authoring-model-3.md")]
#[doc = concat!(
    "\n\n[ch4]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/running_kernels/index.html\n"
)]
pub mod authoring_model {}

#[doc = include_str!("05-gpu-concepts-1.md")]
#[doc = include_str!("listings/image-threshold-kernel.rs")]
#[doc = include_str!("05-gpu-concepts-2.md")]
#[doc = include_str!("listings/reduce-scan-kernel.rs")]
#[doc = include_str!("05-gpu-concepts-3.md")]
#[doc = include_str!("listings/variants-kernel.rs")]
#[doc = include_str!("05-gpu-concepts-4.md")]
#[doc = concat!(
    "\n\n[ch4]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/running_kernels/index.html\n"
)]
#[doc = concat!(
    "\n\n[ch5-running]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/gpu_concepts_running/index.html\n"
)]
#[doc = concat!(
    "\n\n[ch6]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/profiles_artifacts_routing/index.html\n"
)]
pub mod gpu_concepts {}

#[doc = include_str!("07-semantics-and-the-twin.md")]
#[doc = concat!(
    "\n\n[ch6]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/profiles_artifacts_routing/index.html\n"
)]
pub mod semantics_and_the_twin {}

#[doc = include_str!("08-m9-language-1.md")]
#[doc = include_str!("listings/state-machine-kernel.rs")]
#[doc = include_str!("08-m9-language-2.md")]
#[doc = concat!(
    "\n\n[ch4]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/running_kernels/index.html\n"
)]
pub mod m9_language {}

#[doc = include_str!("11-stability-and-platforms.md")]
#[doc = concat!(
    "\n\n[ch9]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/browser/index.html\n"
)]
pub mod stability_and_platforms {}

#[doc = include_str!("12-internals-for-contributors.md")]
#[doc = concat!(
    "\n\n[ch9]: https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/browser/index.html\n"
)]
pub mod internals_for_contributors {}
