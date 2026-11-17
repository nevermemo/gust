# Internals for contributors

This chapter is for someone who wants to change GUST: where a kernel goes between the Rust you write and the GPU, which
crate does which step, which commands check a change, and what adding a language construct involves. It is a map. The
details are in the repository, in
[ARCHITECTURE](https://github.com/nevermemo/gust/blob/main/docs/ARCHITECTURE.md),
[VALIDATION](https://github.com/nevermemo/gust/blob/main/docs/development/VALIDATION.md) and
[SEMANTICS](https://github.com/nevermemo/gust/blob/main/docs/SEMANTICS.md), and they are the source of truth when this
page and they disagree. None of the paths below is part of the published packages; read them in a checkout.

## The pipeline

GUST does not have a compiler IR, and that is a decision, not a gap. The `#[gpu]` macro does a small job:

1. **Parse and validate** a GPU-safe subset of Rust syntax, with a stable diagnostic code for every rejection.
2. **Keep the Rust body.** The body is kept as an ordinary private Rust function, so rustc provides name resolution
   and type checking, and the same body runs as the CPU twin in test and dual builds.
3. **Translate the `syn` syntax tree directly to readable Slang.** There is no intermediate instruction graph between
   the Rust syntax tree and the Slang text.
4. **Attach the Slang and the resource metadata to a host descriptor**, and generate the typed host calls
   (`call`, `submit`, `dispatch`) and the layout assertions that rustc evaluates.
5. **Let Slang do the rest.** At author time `cargo gust build` compiles the Slang to WGSL (and SPIR-V), checks the
   layouts against Slang's own reflection and commits the result in `gust-artifacts/`, and the macro embeds it. Only a
   `dev` build with the `slang-runtime` feature compiles at run time.

This keeps the ownership boundary simple: Rust provides the syntax, tooling and diagnostics of the surface language,
and Slang provides the shader type system, optimisation and code generation for the targets. A change that would add a
second representation of the program (a custom IR, a hand-written SPIR-V emitter) is a design change that needs a
decision of its own, not a detail of a feature.

## Where each step lives

- **`crates/gust-macros`**: the `#[gpu]` and `par_for!` macros. `validate/` holds the accepted subset, `slang/` the
  Slang emitter, `expand/` the host handles and maps, `profile/` what differs between the `grid` and `pixel` stages,
  and `diagnostics.rs` the codes.
- **`crates/gust`**: the library `gust` (package `gust-compute`): the host-side types, `gust::cpu` (the CPU twin and
  the production CPU target), the descriptor, artifacts, profiles and routes, the Slang bridge, and this guide.
- **`crates/gust-wgpu`**: the library `gust_wgpu` (package `gust-compute-wgpu`): `HeadlessDevice`, buffers, uniforms,
  pools and graphs, the WGSL and SPIR-V routes, and the browser surface.
- **`crates/cargo-gust`**: `cargo gust` with `build`, `check`, `run`, `test`, `doctor`, `slang install` and the rest.
- **`xtask`**: the repository's own commands (`cargo xtask ..`): the checks, the suites and the gallery runner. It is
  not published.
- **`tests/` and `gallery/`**: fixture workspaces (compile-fail cases in `tests/ui`, the differential tests, the CPU
  target and routing fixtures) and the gallery programs that this guide's listings come from.

The four packages that are to be published (none is yet) share one version. `gust-algo` (reduce, scan, compact) lives
in the repository and is not to be published.

## The gates

Every claim of evidence in the repository names a tier, and a tier is claimed only with a run record:

- **Unit**: validation, translation and xtask logic pass with no external tool, and the declared minimum Rust
  version compiles every target.
- **Cpu**: the production CPU target runs kernels against independent host references. Never GPU evidence.
- **Slangc**: the generated Slang compiles.
- **Spirv**: exported SPIR-V passes `spirv-val`.
- **Reflect**: Slang's reflection of a kernel matches the descriptor.
- **Gpu**: a read-back on a real adapter equals an independent host reference.

Each tier is one command, `cargo xtask test --suite <Tier>`. `cargo xtask check-fast` and `cargo xtask check-full` run
the staged sets, and `check-full` is the release-confidence run (it needs Slang, a GPU, a browser with WebGPU and
`wasm-bindgen`). Between them are the single checks that VALIDATION describes:
`check-feature <name>`, `check-ui` for the compile-fail fixtures, `check-cpu`, `check-difftest`, `check-gallery`,
`check-precompiled`, `check-web` and `check-release`. `cargo xtask lint` and `cargo xtask docs-check` guard the text of
the repository, among other things that a tracked file has no machine-specific path (the repository is public) and that
the links between its Markdown files resolve.

Continuous integration runs most of this on pushes to `main` and `release/*` and on every pull request: the `unit`,
`msrv`, `cpu-target`, `routing`, `difftest`, `slang`, `precompiled`, `release`, `semver` and `wasm` jobs of `ci.yml`,
and two further workflows, `gpu-software.yml` (the GPU tests on a software adapter) and `spirv-route.yml`. The runners
have no hardware GPU and no WebGPU adapter, so hardware evidence (the NVIDIA runs, a browser run) is produced on a
machine that has them and recorded in the repository. Read a green CI run for what it ran: a software adapter is `Gpu
(software)` evidence, and the `wasm` job compiles and lints the browser code without running it ([chapter 9][ch9]).

## How to add a construct

The order that works is contract first, then the checker, then the emitter, then the evidence:

1. **State the meaning.** Decide what the construct means on every target, where the GPU may differ from plain Rust,
   and whether the difference is outside the defined domain ([chapter 7][`crate::guide::semantics_and_the_twin`]).
   Write it in SEMANTICS with its label. A construct without a statement of meaning is not added.
2. **Validate it.** Accept it in `crates/gust-macros/src/validate/`. Every rejection goes through `diagnostics.rs` and
   has a code in the family of its area (the hundreds digit: item shape, types, expressions, control flow, `Option` and
   `Result`, atomics, macro entry, emitter-only, reserved names). A code keeps its meaning and is never reused; add its
   row to DIAGNOSTICS in the same change.
3. **Emit it.** Lower it in `crates/gust-macros/src/slang/` as a direct translation of the `syn` node. If a stage
   differs (the `pixel` profile refuses workgroup memory, for example), the difference belongs in the `StageProfile`
   in `profile/`, which answers questions and returns text and never walks a body.
4. **Run it on the twin.** The Rust body already runs on `gust::cpu::CpuDevice`. If the construct has no CPU form
   (workgroup memory has none), the kernel must say so with a diagnostic rather than fall silent.
5. **Test it at each layer.** A unit test of the validator and emitter in `crates/gust-macros/src/tests/`; the Slang
   golden for the emitted text (review the diff of `cargo xtask bless`, which rewrites a golden and fails by design);
   a compile-fail case in `tests/ui/<name>/` with the expected message for every new code, run by `cargo xtask
   check-ui`; and a GPU test in `crates/gust-wgpu/tests/` that compares the GPU result with the twin and with a host
   reference written without the kernel. If the construct has a result outside the defined domain, add a ledger row to
   SEMANTICS section 7 and the case to `crates/gust-wgpu/tests/mismatch.rs`; an id guard fails if only one is done.
6. **Document it.** The README's "Current subset", the CHANGELOG, and a guide chapter if a user will meet it. The
   guide's listings come from gallery programs and are kept byte-identical to them by `cargo xtask check-gallery`, so
   a new listing means an anchored region of a gallery file and an entry in the packaged-listing table of `xtask`.
7. **Run the gates the table names.** For validator, lowering and golden changes the first check is
   `cargo xtask check-feature macro`, then `cargo xtask check-workspace`, then `cargo xtask check-full` before a
   release. A change that needs a GPU says which adapter and driver ran it.

A pull request that does only some of these is not wrong to open; it is easier to review when it says which steps it
leaves open.
