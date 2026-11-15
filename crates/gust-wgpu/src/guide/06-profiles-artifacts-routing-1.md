# Build profiles, artifacts and routing

A kernel reaches the GPU as Slang text compiled to a shader. When that happens, and what a build is allowed to contain,
is chosen by the build profile; `cargo gust` is the tool that does the compiling ahead of time and checks the result;
and a `gust.toml` can send a kernel to the CPU or the GPU for a whole build. This chapter is the summary of those three;
the [README](https://github.com/nevermemo/gust/blob/main/README.md) sections "Try it", "Install from crates.io" and
"Routing kernels to the CPU or the GPU at build time" are the reference.

## Profiles

The environment variable `GUST_PROFILE` is `dev`, `native`, `web` or `cpu`; the build script of `gust-compute` reads it,
and any other value, including an empty one, fails the build.

| Profile | What the build is |
| --- | --- |
| `dev` | Kernels compile at run time through Slang (the `slang-runtime` feature); neither strict rule applies. For development and tests. |
| `native` | Strict: every `#[gpu]` kernel needs a precompiled artifact or the build fails at the kernel, and the run-time Slang bridge is refused. |
| `web` | The same strictness for the browser (`wasm32`). |
| `cpu` | The production CPU target: no Slang, no GPU runtime. See below. |

With `GUST_PROFILE` unset, `cargo build --release` of a crate with `#[gpu]` kernels is `native` (`web` on `wasm32`) and
the build script prints a warning that says so; cargo shows a build script's warning only for path and workspace
packages, or with `-vv`, so a crate that depends on `gust-compute` from crates.io does not see it. The CPU is never
inferred. A profile may name a capability after its target, `<target>(+<capability>)*`, as in `native+subgroups`; the
one capability of this release is `subgroups`, and `cpu` takes none. A crate says what it builds for under
`[package.metadata.gust]` in its `Cargo.toml`: `profiles` (`["native", "web", "cpu"]` when absent), `limits =
"portable"` to check every kernel against WebGPU's default limits (GUST-E0746), and `routes`.

## Artifacts and `cargo gust`

`cargo gust build` compiles every kernel with Slang, verifies the layouts against the reflection of the compiler, and
writes `gust-artifacts/` (WGSL always, SPIR-V with `--format spirv`, checked with `spirv-val` when that is on `PATH`).
Commit the directory. A strict build embeds the artifacts and needs no Slang at all; `cargo gust check` fails when the
directory no longer matches the sources, which is how a CI job catches a stale artifact. The first build of a crate
runs `cargo clean -p` for it once and says so. The commands:

| Command | Does |
| --- | --- |
| `cargo gust slang install` | Downloads the pinned Slang, checks its SHA-256, unpacks it into `<CARGO_HOME>/gust/slang` (or `GUST_SLANG_HOME` when set) |
| `cargo gust slang build-helper` | Compiles the native reflection helper against that Slang (needs a C++ compiler) |
| `cargo gust doctor` | Names whatever is still missing |
| `cargo gust build` | Writes `gust-artifacts/` |
| `cargo gust check` | Fails if `gust-artifacts/` is stale |
| `cargo gust run`, `cargo gust test` | `cargo run` and `cargo test` with the routes of a `gust.toml` applied |

The headless backend feeds the Slang-generated WGSL to wgpu; the SPIR-V is exported and validated but not what runs,
unless you opt in to the `spirv-route` feature of `gust-compute-wgpu` (Vulkan only, run-time compile only), which has
its own documented result differences.

## The CPU target

`GUST_PROFILE=cpu cargo build --release` makes the CPU the one target of the build. The macro emits each kernel's `Run`
implementation for `gust::cpu::CpuDevice` and uses no artifact; the binary holds no Slang bridge and no GPU runtime, and
this crate fails to compile under it (a compile error that names the profile). Kernels run on one thread in ascending
order, except the calls of an element-wise map, which run in parallel chunks of the output. A kernel marked
`cpu = "unsupported"` (workgroup memory, barriers, [chapter 5][`gust::guide::gpu_concepts`]) has no CPU form, and a
`dispatch` on it does not compile. The CPU target is the twin's contract made production: where it may differ from the
GPU, SEMANTICS lists the limitation and the project does not chase a match (D39).

## Routing (G8)

Routing sends a kernel to the CPU or the GPU for a whole build, chosen by the application and not by the kernel's
author. Three pieces:

1. The kernel crate declares route tags in its `Cargo.toml` (`[package.metadata.gust.routes]`), each optionally with the
   targets it allows and a default, and a module opts in with `#[gpu(route = "tag")]`.
2. The application chooses in a `gust.toml` beside the `Cargo.toml` of the package that `cargo gust` works on. The key
   is the Cargo package name and the tag, the value is `"gpu"`, `"cpu"` or a table keyed by build profile.
3. `cargo gust run` and `cargo gust test` read the `gust.toml` and give cargo one `GUST_ROUTE__<package>__<tag>`
   variable per tag; `cargo gust build` and `cargo gust check` do the same for the package they work on. Plain `cargo`
   sets none and builds the declared defaults. A mistake (an unknown tag, a target the tag does not allow) is an error
   with file, line and tag before cargo builds. Without a `gust.toml`, nothing changes.

Under `GUST_PROFILE=cpu` every tag goes to the CPU. Otherwise the order is: the nearest `gust.toml` entry, then the
nearest default in the declarations, then the GPU. The resolved table is written to `gust-artifacts/routes.json`, a
build output that changes with the profile: add it to your `.gitignore`.

G8, `routed`, is a two-stage pipeline: stage one saturates samples at 255 and is routed to the CPU, stage two blends
each sample with its neighbours and is routed to the GPU. The program holds one `gust::routed::RoutedDevice`, whose
calls run each kernel on the target its build resolved, so no line of the program names a target, and moving a stage is
an edit of `gust.toml` and a `cargo gust run`. Its manifest and `gust.toml` are:

```toml
# Cargo.toml of the kernel crate
[package.metadata.gust.routes]
tidy = { targets = ["gpu", "cpu"], default = "cpu" }
smooth = { targets = ["gpu", "cpu"], default = "gpu" }
```

```toml
# gust.toml, beside it
[routes]
"routed::tidy" = "cpu"
"routed::smooth" = "gpu"
```

This is the one program of chapters 4 to 6 that the doctests do not compile, so its block is `ignore`. A route tag must
be declared in the `Cargo.toml` of the crate that uses it, and the manifest of this crate declares none (the doctest
fails with GUST-E0748). The listing is compiled instead as the gallery entry `gallery/routed`, a real crate with those
tags, which `cargo xtask check-gallery` builds and runs, and the same command keeps the listing identical to it.

```rust,ignore
