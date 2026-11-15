// Compile check of the first program of `gust::guide` (chapter 2): its three listings together, against the real
// backend. This crate packages its own copies of them (`check-gallery` keeps them identical to the gallery). The module
// is private and exists only in `cargo test --doc`, so no item reaches the documentation or the API.
#[cfg(doctest)]
#[doc = "```no_run"]
#[doc = include_str!("listings/hello-gust-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-kernel.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-main.rs")]
#[doc = "```"]
mod first_program {}
