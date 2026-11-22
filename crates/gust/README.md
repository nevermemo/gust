# gust-compute

The core library of GUST: write a GPU kernel as a Rust module, and a proc macro translates the Rust syntax tree
directly to Slang. This crate holds the typed surface (kernel descriptors, typed buffers, `Call` and `Run`), the
`#[gpu_module]` attribute, the precompiled-artifact check, and the optional run-time Slang bridge.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0 and follows SemVer for 0.x: a break bumps the
minor version of all four packages, a patch only adds or fixes, so expect the API to change between 0.x minors.

What this crate promises (the full text is the root README's section
[Stability](https://github.com/nevermemo/gust/blob/v0.2.0/README.md#stability)): every item that is not `#[doc(hidden)]`;
the `#[gpu_module]` language, meaning the accepted constructs and their lowering contract, and the `GUST-E....` diagnostic
codes (a code keeps its meaning, a message may be reworded); the file formats that `cargo gust` writes and this crate
checks; `gust.toml` and the `[package.metadata.gust]` table; the features `slang-runtime` and `spirv-route` by name; and
the seven environment variables `GUST_PROFILE`, `GUST_SLANG_HOME`, `GUST_SLANG_REFLECT`, `GUST_ROUTE__*`,
`GUST_CPU_THREADS`, `GUST_EXPORT_DIR` and `GUST_ROUTED_PROFILE` (the last two are set by `cargo gust` for the macros;
do not set them yourself). It does not promise the hidden support surface (`gust::profile`, `gust::spirv`,
`gust::slang::probe`, the `__gust_*` names), the Slang text of a kernel, any other `GUST_*` variable, or the chapters of
the `gust::guide` module, which is documentation only and holds no items. Raising the minimum Rust version is a minor
release, and so is a Slang pin bump that changes an artifact's content key.

```sh
cargo add gust-compute
cargo add gust-compute-wgpu   # the headless wgpu backend that runs the kernels
```

The package is `gust-compute`; the library is named `gust`, so code writes `use gust::...` and `#[gust::gpu_module]`.

- The `slang-runtime` feature (off by default) compiles a kernel at run time by launching `slangc` and the native
  reflection helper. A production build leaves it off and ships the artifacts that `cargo gust build` writes.
- `cargo install cargo-gust --locked --version "~0.2"` installs the tool that writes those artifacts; `cargo gust
  slang install` then installs the pinned Slang release.
- The tests of this crate read files of the repository, so they run from a checkout, not from this package.

A complete first program (the manifest with a `gust-dev` feature, `cargo run`, and the release build with `cargo gust
build`) is in the root README: <https://github.com/nevermemo/gust/blob/v0.2.0/README.md#install-from-cratesio>.
Minimum supported Rust: 1.90.

Documentation: <https://docs.rs/gust-compute>. The project, its platforms and its status:
<https://github.com/nevermemo/gust/blob/v0.2.0/README.md>.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
