# gust-compute-macros

The proc macros of GUST: `#[gpu]` on a Rust module (or function) and `par_for!`. They translate the Rust syntax tree
directly to Slang and emit a kernel descriptor, with no intermediate shader IR.

This crate is an implementation detail of `gust-compute`, which re-exports the macros. Depend on `gust-compute`, not on
this crate. It is released at the same version as `gust-compute` and only works with exactly that version.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0: expect the API to change between 0.x releases.

```sh
cargo add gust-compute
```

The tests of this crate read files of the repository, so they run from a checkout, not from this package.

Documentation: <https://docs.rs/gust-compute-macros>. The project, its platforms and its status:
<https://github.com/nevermemo/gust/blob/main/README.md>.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
