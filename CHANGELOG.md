# Changelog

All notable changes to the published packages `gust-compute`, `gust-compute-macros`, `gust-compute-wgpu` and
`cargo-gust` are listed here. The four packages always share one version. GUST follows SemVer for 0.x: a change that
breaks existing code bumps the minor version, and the promise is described in the
[README](README.md#stability).

## Unreleased (0.2.0)

The workspace version is `0.2.0-dev`. Work for 0.2.0 lands on `main`; the 0.1.0 line continues on the branch
`release/0.1`. This list is what the milestone has merged so far.

### Added

- Capabilities in the build profile. `GUST_PROFILE` takes `<target>(+<capability>)*`, and `subgroups` is the first
  capability (`dev+subgroups`); `gust::capability::enabled()` and `cfg(gust_capability = "..")` read it. A malformed
  `[package.metadata.gust]` table is `GUST-E0753`.
- Variants. `#[variant(of = kernel, requires(..))]` gives a kernel a second body for devices that have a capability, and
  the macro checks that the declared profiles are all covered (`GUST-E0740` to `GUST-E0744`, `GUST-E0750`). `cargo
  gust` reads and writes artifact record schema 4 and manifest schema 3 for them; kernels without variants keep the
  bytes they had.
- Subgroup operations (21, as shadow functions such as `subgroup_add`, `subgroup_ballot`, `subgroup_shuffle`), accepted
  only in a body that requires `subgroups` and only where every invocation reaches them (`GUST-E0745`, `GUST-E0747`,
  `GUST-E0751`). The WGSL that Slang writes for them is repaired for the pinned `naga` by a text fix-up, in the
  run-time compiler, the reflected bridge and `cargo gust build`.
- `gust-compute-wgpu` opens its devices with the capabilities of the build, refuses a kernel whose `requires` the device
  does not meet before launch (`Error::MissingCapability`, `Error::KernelTarget`, `Error::SubgroupSize`), and reports
  the adapter's subgroup size range.
- An opt-in portable floor: `limits = "portable"` in `[package.metadata.gust]` checks every kernel against
  `gust::limits::PORTABLE`, WebGPU's default limits (`GUST-E0746`).
- New public modules of `gust-compute`: `gust::target` (`Target`) and `gust::routes` (`variable_name`,
  `VARIABLE_PREFIX`, the spelling of the build-time route variables).

### Changed, and may break code written for 0.1.0

- `KernelDescriptor` is `#[non_exhaustive]` and has new fields (`variant`, `requires`, `target`). Build one with
  `KernelDescriptor::new` and the `with_*` setters instead of a struct literal. `cargo semver-checks` lists this one.
- The generic entry points of `gust-compute-wgpu` and the runs of `StagedGraph` require the `GpuKernel` marker.
  `#[gpu]` kernels get the marker from the macro, so they need no change. A hand-written `Kernel` impl that runs on
  `HeadlessDevice` adds `impl gust::target::GpuKernel for MyKernel {}`; without it the call is E0277 with a message
  that says so. The semver tool does not list this change (the marker is `#[doc(hidden)]`).
- `gust-compute-macros` depends on `toml` 1 (no default features; `std` and `parse`) to read `[package.metadata.gust]`.
- The pinned Slang is 2026.21 (`cargo gust slang install` fetches it); the 0.1.0 line stays on 2026.20.

## 0.1.0 (not yet published)

The first public release of GUST, an experimental compiler from a restricted subset of Rust to Slang for compute
kernels. Nothing in this entry changes an earlier release: there is none. The release is deferred; the packages are not
on crates.io and there is no `v0.1.0` tag yet. The branch `release/0.1` holds the state that will be published.

### What you can do

- Write a kernel as a Rust module marked `#[gpu]`. Rustc checks it as ordinary Rust, and the macro translates the
  syntax tree directly to readable Slang. Constructs outside the supported subset are rejected with a stable
  diagnostic code, `GUST-E....`, listed in [DIAGNOSTICS](docs/DIAGNOSTICS.md).
- The supported subset covers scalars, nested plain structs, vectors and matrices, `Option` and `Result` values,
  bounded `for` loops, uniform parameters, atomics, workgroup memory with barriers, and one-, two- and
  three-dimensional dispatch. [README](README.md#current-subset) has the list.
- Call a kernel with a typed `call`, `submit` or `dispatch` that checks element types, resource counts, access,
  layouts, aliasing and device limits. `#[gpu]` on a free function of scalars or vectors generates an element-wise
  map, and `par_for!` writes a loop as a kernel.
- Run kernels on a Vulkan compute adapter, or on WebGPU in the browser from `wasm32-unknown-unknown`, with typed
  buffers, a pipeline cache, ordered batches (`StagedGraph`), device-resident pools and vectors (`GpuPool`, `GpuVec`)
  and asynchronous readback. Every pipeline is checked against the layout that Slang reports before it is created.
- Run the same kernels as plain Rust: a kernel has a CPU twin in test and dual builds unless it opts out, and a
  build with `GUST_PROFILE=cpu` is a production CPU target with no GPU runtime.
- Compile ahead of time. `cargo gust build` compiles the kernels with Slang, verifies their layouts and writes the
  `gust-artifacts/` directory, which you commit; `cargo gust check` fails when it is stale. A release build embeds the
  artifacts and needs neither Slang, nor the native helper, nor `cargo-gust`.
- Compile at run time during development, through a feature of your own crate that turns on `slang-runtime`.
- Manage the toolchain with `cargo-gust`: `init`, `build`, `check`, `doctor`, and `slang install`, `locate` and
  `build-helper` for the pinned Slang release (2026.20, downloaded and checked by SHA-256, never shipped in the
  crates).

### Execution shapes and textures

`pixel` kernels (`#[kernel(shape = pixel)]`: one fragment per element, outputs written as storage buffers), the `Shape`
type and the 2-D texture and sampler types are part of this release and of the 0.x promise, so a minor release may
change them. Their evidence is narrower than the rest: a GUST `pixel` or texture kernel has run on one hardware
adapter, the RTX 5090 (Vulkan), and on the lavapipe software adapter, and on no other hardware. Headless Chrome on that
NVIDIA GPU has run a `pixel` kernel and no texture kernel. README, "Stability", says the same.

### Platforms and requirements

- Vulkan on Windows and Linux, WebGPU in the browser. macOS and Metal are not supported, and DX12 is not claimed.
- Rust 1.90 or newer.
- A C++ compiler (MSVC on Windows, g++ or clang++ elsewhere) for `cargo gust slang build-helper`, needed only for the
  run-time compile path and for authors who export artifacts.

### Known limits

- The error bound of a linear texture `sample` assumes a sampler with 8-bit filter weights, rounded to the nearest
  step; no specification gives that (WebGPU states no number; Vulkan requires `subTexelPrecisionBits` of 4 in core and
  of 8 for Vulkan 1.4 or Roadmap 2022) and the evidence is one adapter, so the bound can be exceeded, and its test fail,
  on a device that reports fewer than 8 bits (SEMANTICS, section 4).
- In the browser (`wasm32`) nothing blocks: a job's `wait`, and so a kernel's `dispatch`, returns `Error::WouldBlock`
  (any work it submitted will run). Submit, then await a readback (`read_typed_buffer_async`, `GpuVec::to_vec_async`).
- SPIR-V is exported and validated with `spirv-val` when it is on `PATH`, but the runtime executes the WGSL.
  Running the exported SPIR-V is planned for a 0.1.x release.
- A crate that depends directly on both `gust-compute` and the unrelated crate `gust` fails with rustc error
  E0464, because both libraries are called `gust`. Renaming a dependency needs an `extern crate ... as gust;` line.
- The packaged crates are not test-runnable: their tests read files of the repository. Run the tests from a checkout.
- Kernels that use workgroup memory or barriers have no CPU twin yet: they must say `cpu = "unsupported"`, and a
  `dispatch` on them does not compile under the `cpu` profile.
- The algorithms crate of the repository (`gust-algo`: reduce, scan, compact) and the examples are not published.
