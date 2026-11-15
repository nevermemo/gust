# gust-compute-wgpu

The headless wgpu backend of GUST: it runs the kernels of `gust-compute` on a Vulkan adapter (and, on wasm32, on
WebGPU) with typed buffers, a pipeline cache, ordered batches and async readback. Every pipeline is checked against the
layout that Slang reports before it is created.

GUST is an experimental Rust to Slang compute subsystem. It is pre-1.0 and follows SemVer for 0.x: a break bumps the
minor version of all four packages, a patch only adds or fixes, so expect the API to change between 0.x minors.

What this crate promises (the full text is the root README's section
[Stability](https://github.com/nevermemo/gust/blob/v0.2.0/README.md#stability)): every item that is not `#[doc(hidden)]`
(`HeadlessDevice`, the buffer, pool, job and staged-graph types, the route types behind `spirv-route`), and the features
`slang-runtime` and `spirv-route` by name (`spirv-route` stays opt-in and Vulkan only; its behaviour may change in a
minor release, its name does not). It does not promise `render_wgpu_source`, which is hidden support surface, nor the
chapters of the `gust_wgpu::guide` module, which is documentation only and holds no items. Platforms: Vulkan on Windows
and Linux, WebGPU in Chrome; nothing else is claimed. Known issue: on an NVIDIA GeForce RTX 5090 (Vulkan, driver 610.88)
one array pattern computes wrong values on the default WGSL route; the root README's
[Platforms](https://github.com/nevermemo/gust/blob/v0.2.0/README.md#platforms) has the pattern, the
opt-in cure and what was not measured.

```sh
cargo add gust-compute
cargo add gust-compute-wgpu
```

The package is `gust-compute-wgpu`; the library is named `gust_wgpu`, so code writes `use gust_wgpu::...`.

- The `slang-runtime` feature (off by default) compiles a kernel at run time through the native reflection helper. A
  production build leaves it off and runs the artifacts that `cargo gust build` writes.
- The SPIR-V that GUST exports is validated, but the runtime executes the WGSL by default. The opt-in `spirv-route`
  feature (Vulkan only, run-time compilation only) runs slangc's own SPIR-V where it can, and falls back to WGSL
  (precompiled artifacts, other backends), recording the reason in `route_events`.
- The tests of this crate read files of the repository, so they run from a checkout, not from this package.

A complete first program (the manifest with a `gust-dev` feature, `cargo run`, and the release build with `cargo gust
build`) is in the root README: <https://github.com/nevermemo/gust/blob/v0.2.0/README.md#install-from-cratesio>.
Minimum supported Rust: 1.90. Platforms: Vulkan on Windows and Linux, WebGPU in Chrome; no macOS or Metal.

Documentation: <https://docs.rs/gust-compute-wgpu>. The project, its platforms and its status:
<https://github.com/nevermemo/gust/blob/v0.2.0/README.md>.

Licensed under either of MIT or Apache-2.0 at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
