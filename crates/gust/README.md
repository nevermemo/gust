# gust-compute

The core library of GUST: write a GPU kernel as a Rust module, and a proc macro translates the Rust syntax tree
directly to Slang. This crate holds the typed surface (kernel descriptors, typed buffers, `Call` and `Run`), the
`#[gpu]` attribute, the precompiled-artifact check, and the optional run-time Slang bridge.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0: expect the API to change between 0.x releases.

```sh
cargo add gust-compute
cargo add gust-compute-wgpu   # the headless wgpu backend that runs the kernels
```

The package is `gust-compute`; the library is named `gust`, so code writes `use gust::...` and `#[gust::gpu]`.

- The `slang-runtime` feature (off by default) compiles a kernel at run time by launching `slangc` and the native
  reflection helper. A production build leaves it off and ships the artifacts that `cargo gust build` writes.
- `cargo install cargo-gust --locked --version "~0.1"` installs the tool that writes those artifacts; `cargo gust
  slang install` then installs the pinned Slang release.
- The tests of this crate read files of the repository, so they run from a checkout, not from this package.

A complete first program (the manifest with a `gust-dev` feature, `cargo run`, and the release build with `cargo gust
build`) is in the root README: <https://github.com/nevermemo/gust/blob/main/README.md#install-from-cratesio>.
Minimum supported Rust: 1.90.

Documentation: <https://docs.rs/gust-compute>. The project, its platforms and its status:
<https://github.com/nevermemo/gust/blob/main/README.md>.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
