// The chapters of `gust::guide`, one empty module each (the semver rule: the guide modules hold nothing but chapter
// modules, which hold nothing; they exist for their documentation). Chapters 3 to 12 are listed in the index as not yet
// written and have no module until they have text.
//
// The link to `gust_wgpu::guide` is built here, from this crate's version, because a Markdown file cannot read
// `CARGO_PKG_VERSION`: the index ends with it (an inline link; a link definition in this inner doc was accepted by
// stable rustdoc but not by nightly with `--cfg docsrs`), and chapter 2 defines it for its own page. The versions of
// the two crates are locked together, so the link names the sibling guide of the same release (it works once that
// release is published and its docs have built).
#![doc = concat!(
    "\n\nThe sibling guide: [`gust_wgpu::guide`](https://docs.rs/gust-compute-wgpu/",
    env!("CARGO_PKG_VERSION"),
    "/gust_wgpu/guide/index.html).\n"
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
