# gust-compute-wgpu

The headless wgpu backend of GUST: it runs the kernels of `gust-compute` on a Vulkan adapter (and, on wasm32, on
WebGPU) with typed buffers, a pipeline cache, ordered batches and async readback. Every pipeline is checked against the
layout that Slang reports before it is created.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0: expect the API to change between 0.x releases.

```sh
cargo add gust-compute
cargo add gust-compute-wgpu
```

The package is `gust-compute-wgpu`; the library is named `gust_wgpu`, so code writes `use gust_wgpu::...`.

- The `slang-runtime` feature (off by default) compiles a kernel at run time through the native reflection helper. A
  production build leaves it off and runs the artifacts that `cargo gust build` writes.
- The SPIR-V that GUST exports is validated, but the runtime executes the WGSL.
- The tests of this crate read files of the repository, so they run from a checkout, not from this package.

A complete first program (the manifest with a `gust-dev` feature, `cargo run`, and the release build with `cargo gust
build`) is in the root README: <https://github.com/nevermemo/gust/blob/main/README.md#install-from-cratesio>.
Minimum supported Rust: 1.90. Platforms: Vulkan on Windows and Linux, WebGPU in the browser; no macOS or Metal.

Documentation: <https://docs.rs/gust-compute-wgpu>. The project, its platforms and its status:
<https://github.com/nevermemo/gust/blob/main/README.md>.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
