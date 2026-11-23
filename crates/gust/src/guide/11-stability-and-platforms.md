# Stability, versions, platforms, and how to report a bug

GUST is an experimental compiler. This chapter says what a version number promises, what it does not, which platforms
the project claims, and how to report a problem. The README sections "Stability", "Platforms" and "Minimum Rust" are
the complete, current text
([README](https://github.com/nevermemo/gust/blob/main/README.md)); this chapter is a map of them.

## Versions

0.2.0 was the first public release. GUST follows SemVer for 0.x:

- a change that breaks existing code bumps the minor version (0.2 to 0.3);
- a patch release only adds or fixes. A construct may be added to the `#[gpu_module]` language in a patch, and is refused only
  in a minor;
- the four packages (`gust-compute`, `gust-compute-macros`, `gust-compute-wgpu` and `cargo-gust`) are released together
  at one version and depend on each other with exact (`=`) versions, so upgrade them together. A breaking change in one
  is a minor bump of all four.

Nothing here promises that the API stops changing between minors, only how a change is made and announced. Each release
is described in the
[CHANGELOG](https://github.com/nevermemo/gust/blob/main/CHANGELOG.md), with the migration for every break.

## What the promise covers

The promise covers the parts you write against:

- the public API of `gust-compute` and `gust-compute-wgpu`: every item that is not `#[doc(hidden)]`. Every public
  enum is `#[non_exhaustive]` except a few closed sets that the README names, so match the others with a wildcard
  arm;
- the `#[gpu_module]` language: the set of accepted constructs with their lowering contract (SEMANTICS, and
  [chapter 7][`crate::guide::semantics_and_the_twin`]);
- the diagnostic codes, `GUST-E....`: a code keeps its meaning and is never reused, and the message text may be
  reworded, so match on the code;
- the `cargo gust` commands and the lines they print, the artifact and record file formats (a newer release reads every
  older one), `gust.toml` and the `[package.metadata.gust]` table;
- two Cargo features by name and meaning, `slang-runtime` and `spirv-route`; no other feature is promised;
- the environment variables `GUST_PROFILE` (its grammar, including the `+capability` suffix), `GUST_SLANG_HOME`,
  `GUST_SLANG_REFLECT`, `GUST_ROUTE__*` and `GUST_CPU_THREADS`, and two that `cargo gust` sets for the crates it builds
  (`GUST_EXPORT_DIR` and `GUST_ROUTED_PROFILE`, which you should not set yourself).

It does not cover the support surface that is public only because generated code or `cargo-gust` needs it (the
`__gust_*` names, for example), the `GUST_*` variables that only tests read, the Slang text of a kernel, or the
`cargo xtask` commands and evidence tiers of the repository. `GUST_SPIRV_ROUTE_LOG` (a verification log read at run time)
is likewise not covered. The content key of a precompiled artifact hashes the Slang text, so it may change in a minor
release (a patch keeps committed artifacts valid); after such an upgrade, run `cargo gust build` again and commit
`gust-artifacts/`. The README's "Stability" section is the list to check when a detail matters.

**This guide is documentation only.** `gust::guide` and `gust_wgpu::guide` contain one empty module per chapter and no
other item, and nothing in them is part of the API. The set of chapters may change in any release, and a chapter may
move between the two crates; link to a page of the guide, not to a module path in code.

## The minimum Rust version

The minimum Rust version is 1.90, declared as `rust-version` in all four packages. The promise is that the packages
build on that toolchain with the dependency versions available at publish time. The repository checks it in a CI job
that runs `cargo check` for every workspace target on exactly 1.90.0 (it runs no tests), and by building the README's
first program on 1.90 with no lockfile, the way a user resolves it. A later release of a dependency that raises its own
minimum is outside the promise, and a patch release caps the range when the resolver cannot route around it. Raising
`rust-version` is a minor release, never a patch.

## The Slang pin

The version of Slang that GUST drives is pinned (2026.22.1 in the 0.2 line), and `cargo gust slang install` downloads
exactly that version and checks its SHA-256. A new pin is a minor release when it changes the key of a committed
artifact, and the CHANGELOG entry names the pinned version. In the 0.2 line, `cargo gust check` refuses artifacts
built with the previous pin (2026.20) until `cargo gust build` is run again.

## Platforms

- **Supported:** Vulkan on Windows and Linux, through the native backend, which asks wgpu for Vulkan only by default
  (`DeviceOptions::with_backends` changes that). In the browser, the backend uses WebGPU through
  `wasm32-unknown-unknown` (the project runs it in Chrome; see the browser chapter, [chapter 9][ch9]).
- **Not supported:** macOS and Metal (no Metal backend is compiled), and DX12 is not claimed. Software adapters such as
  Mesa lavapipe run the GPU tests in the repository's CI as software evidence; they are not a target.
- **CPU:** a `GUST_PROFILE=cpu` build contains no GPU runtime and runs the kernels as plain Rust.
- **Subgroups:** a `+subgroups` build needs a Vulkan adapter that offers subgroup operations. In the browser it is
  refused at build time, and a device opened with a capability there is `Error::MissingCapability`.
- **SPIR-V:** exported and validated when `spirv-val` is on `PATH`; the runtime executes the WGSL by default, and the
  opt-in `spirv-route` feature builds Vulkan pipelines from SPIR-V
  ([chapter 7][`crate::guide::semantics_and_the_twin`]).

Nothing else is claimed. The evidence for `pixel` kernels and textures is narrower than for the rest: they have run on
one hardware adapter (an NVIDIA RTX 5090 over Vulkan) and on the lavapipe software adapter, and another vendor's GPU
may show a difference the tests have not seen. The README says exactly what has run, and the NVIDIA array defect it
lists under "Platforms" is a reason to read it before you ship to one driver family.

## How to report a bug

Report it as an issue of the repository, <https://github.com/nevermemo/gust/issues>. These make a report easy to act on:

1. **The versions.** `cargo gust doctor` prints the `cargo-gust` and `gust-compute` versions, the `slangc` it found and
   what is still missing; paste its output, with the output of `rustc --version`.
2. **The build.** The value of `GUST_PROFILE` (or that it was unset), whether `slang-runtime` or `spirv-route` is on,
   and the operating system.
3. **The adapter**, for a GPU result: the name that `device.adapter_info().name` gives, and the driver version that
   its `driver_info` gives.
4. **The smallest kernel** that shows it, and the input values. A kernel with a run-time operand outside the defined
   domain is a documented limitation ([chapter 7][`crate::guide::semantics_and_the_twin`]), so check the ledger first;
   a difference that the ledger does not list is exactly the report that is wanted.
5. **Which targets disagree.** If you can run the CPU twin or a `GUST_PROFILE=cpu` build, say whether it agrees with a
   host reference written in plain Rust. A twin result that matches the reference while the GPU result does not is a
   different report from a kernel that is wrong everywhere, and neither says anything about a race the twin hides.
6. **The diagnostic code**, for a rejection: the `GUST-E....` code and the construct, and what you expected. A
   rejection that DIAGNOSTICS says is intended is not a bug, but a confusing message or a missing suggestion is worth
   an issue.

Two kinds of report are especially useful: a program that GUST accepts and that gives a GPU result outside the
contract, and a program that Rust accepts, the documentation says GUST supports, and that GUST refuses.
