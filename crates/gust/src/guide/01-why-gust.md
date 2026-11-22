# Why GUST

GUST (GPU <3 rUST) lets you write GPU compute kernels as Rust modules. A `#[gpu_module]` module is ordinary Rust syntax with
a deliberately small, GPU-safe vocabulary. rustc checks it, and the same source is translated to readable
[Slang](https://shader-slang.org/), which produces the shader for the target (WGSL and SPIR-V).

## What you get

- **Rust syntax and rustc checking.** The macro validates the accepted subset and rejects anything else with a
  diagnostic at the Rust boundary. A private Rust "shadow" of each function stays in the build, so rustc does the name
  resolution and the type checking, and the shadow also runs as the CPU twin in test builds.
- **Readable output.** The macro translates the original `syn` syntax tree directly to Slang. There is no custom
  intermediate representation and no handwritten SPIR-V emitter in the compiler core. The Slang, the WGSL and the
  SPIR-V stay inspectable files.
- **Slang does the shader work.** The Slang compiler provides the shader type system, legalisation, optimisation and
  code generation for the targets, so GUST does not re-implement them.
- **Release builds without Slang.** During development a kernel can be compiled at run time through Slang. For a
  release, `cargo gust build` compiles it ahead of time and commits the result in `gust-artifacts/`, and the macro
  embeds those files, so the release binary needs neither Slang nor the helper nor `cargo-gust`.

## What it is not

- It is **not** an arbitrary Rust-to-SPIR-V transpiler, and it does not promise ordinary Rust execution on a shader
  core. Unsupported constructs are rejected clearly until they have a defined lowering. An apparent syntax match is
  not proof of the same meaning.
- It is **not** a Rust interpreter. The CPU twin is the rustc-compiled shadow body, not an interpreter of the GPU
  module.
- It does **not** claim complete Slang language or resource coverage, and it does not hide its outputs.
- It is **not** (today) a game engine or an execution-graph compiler. Those are separately labelled plans in the
  repository's vision documents, not features of the released crates.

The released surface is compute only: kernels, typed buffers and calls, a headless wgpu backend, and a CPU twin used to
check kernels against independent host references.

## Where the sources of truth are

This chapter summarises `docs/VISION.md` and the "Decision" section of `docs/ARCHITECTURE.md` of the repository. If this
page and those files ever disagree, the repository files describe what the code does and this page should be fixed.

The next chapter installs the tools and runs a first kernel. The [index][`crate::guide`] lists the rest of the guide.
