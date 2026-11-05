# GUST — GPU <3 rUST

GUST is growing toward a heterogeneous Rust execution compiler and, eventually,
a GPU-centric ECS game engine. **Today it is a working experimental Rust → Slang
compute subsystem**. There is no ECS,
renderer, rustc semantic integration, or execution-graph compiler yet.

Rust supplies the higher-level language and checking environment; Slang owns GPU
code generation; GUST will connect intentional CPU and GPU execution. We preserve
the working subsystem while proving that larger model incrementally.

Start with [current status](docs/development/STATUS.md),
[next tasks](docs/development/NEXT_TASKS.md), and
[the handoff](docs/development/HANDOFF.md). Engineering agents should read
[AGENTS.md](AGENTS.md).
The [vision](docs/VISION.md), [roadmap](docs/ROADMAP.md), and
[Hurricane engine destination](docs/ENGINE_HURRICANE.md) distinguish implementation from plans.
Development conventions live in [repository structure](docs/development/REPO_STRUCTURE.md),
[validation strategy](docs/development/VALIDATION.md), and
[agent workflow](docs/development/AGENT_WORKFLOW.md).

The current executable path is:

```text
#[gpu] Rust module
        │
        ├─ rustc checks the shadow Rust types and kernel body
        │
        └─ proc macro translates the syn syntax tree directly to Slang
                                      │
                                      ├─ gust-slang-reflect (Slang API) ─> WGSL + layout ─> headless wgpu
                                      ├─ slangc -target wgsl ──> exported .wgsl
                                      └─ slangc -target spirv -> .spv / SPIRV-Tools
```

There is no custom shader IR and no hand-written SPIR-V emitter in this path. A
kernel descriptor stores generated Slang source plus macro-assigned binding metadata.
On its first pipeline-cache miss the wgpu backend runs the native reflection helper,
which compiles the kernel and reflects the *same linked program* through Slang's
layout API. The runtime refuses to create the pipeline unless the descriptor's
names, bindings, access modes, workgroup size, and every nested field offset, size,
alignment, and stride equal what the compiler reports; on success it caches the
shader module, bind-group layout, pipeline layout, and compute or render pipeline.

## Try it

Requirements (for authors and for this repository's checks; a crate that builds with committed `gust-artifacts/` under
a strict profile needs neither Slang nor the helper, D45):

- The stable Rust toolchain: 1.90 or newer (the `msrv` CI job type-checks the workspace on 1.90.0);
  this checkout's checks ran on stable 1.98.0 (see "Minimum Rust")
- Outside this checkout, once 0.1.0 is published, `cargo install cargo-gust --locked --version "~0.1"` installs the
  `cargo gust` tool; it works without `GUST_PROFILE` ("Install from crates.io" has the whole setup for a crate of your
  own)
- `slangc` on `PATH`: the Slang release that `cargo gust` pins, 2026.20 (`cargo gust slang install` installs it, and
  `cargo gust doctor` names a `slangc` of another version). `cargo gust` finds its managed install itself (in
  `GUST_SLANG_HOME` when set), but `cargo xtask` and the run-time bridge launch plain `slangc`, so put the install's
  `bin` directory first on `PATH`
- The native reflection helper, built once with
  `cargo xtask build-slang-reflect`. It needs the Slang
  SDK headers and import library (from `VULKAN_SDK` or `SLANG_SDK`) and a C++
  toolchain (MSVC on Windows). The runtime finds it in `target/slang-reflect/` or
  through `GUST_SLANG_REFLECT`; without it every pipeline creation fails with an
  explicit error rather than skipping the layout check
- A Vulkan compute adapter for the headless execution tests
- Optional: `spirv-val` from SPIRV-Tools for external SPIR-V validation

Run the smallest end-to-end example:

```sh
cargo run -p vector-add
```

It executes real code on the selected GPU and writes four local, ignored artifacts to
`generated-wgpu/`:

- `vector_add__add.slang` — direct Rust-to-Slang output
- `vector_add__add.wgsl` — Slang-generated WGSL used by wgpu
- `vector_add__add.spv` — Slang-generated SPIR-V
- `vector_add__add.rs` — readable descriptor-specialized wgpu host code

Run the normal development suite:

```sh
cargo test --workspace
```

Example crate tests are intentionally ignored by default. They remain available for
major changes and release validation, but the ordinary workspace loop prioritizes
compiler/runtime feedback over full example proofs.

Compiling a kernel at run time (launching `slangc` and the native reflection helper) is the
`slang-runtime` feature of `gust` and `gust-wgpu`, off by default (D44). The ten examples
enable it in their manifests, and `gust` and `gust-wgpu` enable it for their own tests through a
dev-dependency on themselves, so `cargo run -p vector-add`, `cargo test -p gust-compute-wgpu` and
`cargo test --workspace` need no flag. A crate of your own that runs kernels this way during
development has a feature of its own, `gust-dev = ["gust-compute-wgpu/slang-runtime"]` ("Install from crates.io"), or
passes `--features gust-compute-wgpu/slang-runtime`, so that its normal dependencies, and a release build, hold no
process-spawning compiler. A `[dev-dependencies]` entry with the feature serves `cargo test` and examples, but never
`cargo run` of a binary. Check that state with
`cargo check -p gust-compute -p gust-compute-wgpu --no-default-features --lib` (the Unit suite runs it).

Build profiles (D44, D51): the environment variable `GUST_PROFILE` is `dev`, `native`, `web` or `cpu`, and
`gust`'s build script reads it (any other value, including an empty one, fails the build). `native` and `web` are strict:
every `#[gpu]` kernel must have a precompiled artifact, or the build fails with a compile error at the kernel, and the
`slang-runtime` feature is refused, so the build holds no run-time Slang bridge (wgpu and the driver still compile
the shipped WGSL or SPIR-V when the pipeline is created). `dev` has neither rule. A release
build is strict even without the variable: with `GUST_PROFILE` unset, `cargo build --release` of a crate with
`#[gpu]` kernels is treated as `web` for `wasm32` and `native` for every other target, and prints
`warning: gust-compute@<version>: gust: release build without GUST_PROFILE treated as native` (or `web`;
cargo shows a build script's warning only for path and workspace packages, or with `-vv`). The one exception is the
internal `cargo-gust-bridge` feature (D108, D117): where `cargo-gust` is in the graph (`cargo install cargo-gust`, a
workspace that has it as a member, or a mistaken `cargo-gust` entry under `[dependencies]`), a release build without
the variable is `dev` and warns `gust: cargo-gust-bridge treats a release build without GUST_PROFILE as dev`, which
cargo shows only for path and workspace builds. A release build that
wants the run-time bridge, such as the benchmark examples below, sets `GUST_PROFILE=dev` explicitly
(`GUST_PROFILE=dev cargo run --release -p polynomial`; in PowerShell `$env:GUST_PROFILE = "dev"` first). The
repository's `cargo xtask` commands run their cargo children under `GUST_PROFILE=dev`, whatever the parent
environment holds (a different value is logged and overridden), which covers `check-full`'s release tests. A strict
build needs the artifacts of its kernels, committed in `gust-artifacts/` (written by `cargo gust build` and re-checked
by `cargo gust check`, D45); without them the build fails at the kernel by design.

The `cpu` profile (D21, D51) is the production CPU target. It is never inferred: a release build without the variable
stays `native`, so choosing the CPU is always explicit (`GUST_PROFILE=cpu cargo build --release`). The `#[gpu]` macro
then emits each kernel's `Run` impl for `gust::cpu::CpuDevice` in every build and uses no artifact, so a program can
call `dispatch` on the CPU and run the kernel as plain Rust. A build holds one target: `gust-wgpu` fails to compile
under `cpu` (a compile error that names the profile), and the `slang-runtime` feature is refused, so the binary
contains no Slang bridge, helper or GPU runtime. A kernel marked `#[kernel(cpu = "unsupported")]` has no CPU form, and a
`dispatch` on it does not compile. Kernels run on one thread, in ascending `global_id.x`, with one exception: the calls
of an element-wise map (`#[gpu]` on a function) run in parallel chunks of the output (see "Current subset").
`cargo xtask check-cpu-target` is the evidence (the `Cpu` tier).

For the full release-readiness check (including ignored example tests, all nine
examples, and mandatory external validation of the twenty SPIR-V files of the examples and of nine golden sets):

```sh
cargo xtask check-full
```

It also runs the format check, clippy and tests of the differential-test harness (`tests/difftest`, a workspace of
its own), the whole `check-precompiled` (the fixture `tests/precompiled`, with a GPU, Slang and the crates.io
registry), the whole `check-release` (packaging, docs and the crates.io recipe run on the GPU, with the registry), the
whole `check-web` (the browser WebGPU run: wasm32 target, `wasm-bindgen-cli`, Chrome with a WebGPU adapter) and, last,
`check-difftest` (generated kernels on an interpreter, the CPU twin and the GPU: all 8 seeds, 128 programs, and the
`short_circuit` GPU test). Nothing is skipped silently: a missing tool or adapter fails the run and names the cause,
and the tools of `check-web` are looked up first, so a missing one fails in seconds. On a machine without a browser,
`cargo xtask check-full --skip-web` leaves out `check-web` only, says so, and records `skipped_checks` in the file
below; `cargo xtask status` then shows `skipped=check-web`.

It writes `.ai/VALIDATION.json` with command exit codes, stdout, artifact hashes,
the skipped checks and explicit untested targets. This is local debug validation, not release benchmark
evidence or cross-platform certification.

Purpose-built check commands keep the inner loop honest:

| Command | Purpose | Runs |
| --- | --- | --- |
| `cargo xtask check-feature macro` | Macro/validator/emitter work | `cargo test -p gust-compute-macros` |
| `cargo xtask check-feature reflection` | Reflection/layout work | helper version check, core reflection tests, wgpu reflection tests |
| `cargo xtask check-feature loops` | Bounded-loop work | loop golden/rejection test plus GPU loop test |
| `cargo xtask check-feature gpu-smoke` | Cheapest wgpu runtime sanity | wgpu crate unit tests only |
| `cargo xtask check-feature gpu-semantics` | GPU semantic differential work | semantics, numeric, option, loops, struct-assignment tests |
| `cargo xtask check-feature gpu-runtime` | Runtime/reflection gate work | wgpu `reflection`, `limits`, `typed` and `graph` integration tests |
| `cargo xtask check-feature result` | Result lowering and match work | gust-macros `result` tests, then wgpu `result` |
| `cargo xtask check-feature vectors` | Vector and matrix types (StorageV2 buffers); needs slangc, `spirv-val`, the reflection helper and an adapter | gust-macros vector tests, `gust::math` unit tests, the `vectors` and `products` goldens, their SPIR-V made and validated with `spirv-val`, then wgpu `vectors` |
| `cargo xtask check-feature uniforms` | Uniform parameters (`Uniform<T>`, Std140); as for `vectors` | gust-macros uniform tests, `uniforms_twin`, the `uniforms` golden, its SPIR-V validated, then wgpu `uniforms` and the uniform cases of `typed` and `graph` |
| `cargo xtask check-feature map` | Element-wise maps, `GpuVec`, device options and the parallel CPU map | `gust` unit tests of `map::` and `cpu::parallel`, `gust` `tests/parallel.rs`, gust-macros map tests, wgpu `map`, `gpu_vec`, `device_options` and `limits` |
| `cargo xtask check-feature workgroup` | Workgroup memory, barriers, `LocalId` and `GroupId` | gust-macros and `gust` unit tests named `workgroup`, then wgpu `workgroup` (the block reduce, the derived ids and the barrier in the SPIR-V) |
| `cargo xtask check-feature algo` | The `gust-algo` library (reduce, scan, compact) | every `gust-algo` test: schedule, bound and golden tests and the CPU twins without an adapter, then the same cases on the first Vulkan adapter |
| `cargo xtask check-feature dispatch` | `par_for!` and multi-dimensional dispatch (`Extent`) | gust-macros tests `par_for_tests::` and `grid_tests::`, the `gust` library tests of `dispatch::` and of the CPU invocations of a multi-dimensional extent (each module or filter its own counted call), then wgpu `par_for`, `grid`, `concurrency` and `async_readback` |
| `cargo xtask check-changed` | Dirty-tree routing | chooses focused checks from changed paths: `macro`, `core`, `wgpu`, `reflection` and `examples` by crate or directory, and `loops`, `workgroup`, `algo`, `map` and `dispatch` by their fixtures, tests and sources (a build, CI or `xtask` change selects `check-fast`; no match, `check-workspace`) |
| `cargo xtask check-format` | Formatting only | `cargo fmt --all -- --check` |
| `cargo xtask check-lints` | Strict lints | workspace Clippy with warnings denied (`--locked`) |
| `cargo xtask lint` | Policy lint | no tracked shell or PowerShell scripts (D27) and no machine-specific paths in tracked text (Windows, Unix and WSL home paths; two test files may spell placeholder names only); scans tracked files only. Also the release rules, with package names read from `cargo metadata`: every package that can be published has `description`, `license`, `repository`, `documentation`, `readme`, `keywords` and `categories`, its own `README.md`, and byte-equal copies of `LICENSE-MIT` and `LICENSE-APACHE`; and every `path =` dependency on a package under `crates/` calls it by that package's name (or `package =`), so a manifest written before the crate rename fails here |
| `cargo xtask check-release [--check-only] [--work-dir DIR]` | Release gate (D113); `--check-only` needs no Slang and no adapter | packages every publishable crate in one `cargo package --locked` call with verification, judges the file lists, metadata, equal versions, exact internal pins and `publish` flags, builds the rustdoc of the library crates with warnings denied, checks each `.crate` size, builds the "Install from crates.io" recipe word for word from the unpacked `.crate` files (installing `cargo-gust` from its unpacked package too) and ends with `cargo publish --dry-run --locked`. The full mode also runs the recipe on a GPU and builds its release form; needs the crates.io registry. CI runs the `--check-only` form as the job `release`; `check-full` runs the full one. Steps for an actual release: [RELEASE](docs/development/RELEASE.md) |
| `cargo xtask docs-check` | Docs consistency | relative Markdown links in project `.md` files resolve to project files (anchors and external links are not checked), no merge-conflict markers, `docs/DECISIONS.md` ids unique and increasing with the right "Next id"; also sees untracked files that are not ignored |
| `cargo xtask check-arch` | Architecture guardrails (D02) | per-crate dependency allow-list read from `cargo metadata`, allow-listed external packages must resolve from crates.io in `Cargo.lock` (catches `[patch]`/`[replace]` redirects), no `ir` / `*_ir` module or file under `crates/` outside a `future/` path, no SPIR-V library in the compiler crates; also sees untracked files that are not ignored |
| `cargo xtask check-ui` | Diagnostic text for misuse (D27) | compiles each crate of the separate `tests/ui` workspace; a crate with `expected.txt` must fail with exactly one error (read from cargo's JSON) that satisfies every listed key (`code`, `file`, `line`, `message`, `label`; `code: none` matches a macro error without a rustc code and needs a `message:`), one without it must compile |
| `cargo xtask check-cpu` | CPU-twin evidence (no slangc, no GPU) | `--locked` tests of the examples with CPU twins (`vector-add`, `polynomial`, `signal-pipeline`, `particle-step`, `typed-pipeline`, `atomic-counter`, `staged-graph`, `component-pool`, `map-chain`, `texture-threshold`), whose non-ignored tests run each macro-emitted CPU twin against its host reference; part of the Unit suite. Each call fails when it reports no passed test (the test-count guard), and when a twin test or derivation-domain guard of the example is not reported `ok` by name |
| `cargo xtask check-difftest [--seeds <list\|range\|random:N>] [--cpu-only] [--report PATH]` | Differential test of generated scalar kernels (`tests/difftest`, `docs/development/DIFFTEST.md`); `--cpu-only` needs no slangc and no GPU | runs `difftest-run` and fails unless its report has the requested seeds and mode, `programs_executed` equal to 16 times the seeds, no mismatch (named by seed) and, without `--cpu-only`, an adapter; without `--cpu-only` it also runs the `short_circuit` GPU test. `--cpu-only --seeds 1-4` is the last step of the Unit suite; the Gpu suite and `check-full` run all 8 seeds |
| `cargo xtask check-dual [--check-only]` | Cross-crate dual-build evidence; `--check-only` needs no slangc and no GPU | refuses a preset `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS` or `CARGO_BUILD_RUSTFLAGS`; checks `tests/dual` with fmt and clippy, builds the workspace with `--cfg gust_dual` (through `CARGO_ENCODED_RUSTFLAGS`) in `<target>/dual`, runs clippy `-D warnings` on vector-add and polynomial with the cfg, then checks the `tests/dual` fixture: without the cfg a dependent's own `cfg(test)` (E1) and a production binary (E3) get no CPU twin of a library kernel (one E0277 each); with it the twin test runs and passes (E2). The full mode (GPU) also runs vector-add and polynomial with the cfg, which must print `CPU twin: matches (N elements, max ULP k)`, and vector-add without it, which must not. CI's `unit` job runs `--check-only`; `check-full` runs the full mode |
| `cargo xtask check-cpu-target` | Production CPU target evidence (`Cpu` tier; no slangc, no GPU) | builds the fixture `tests/cpu-target` in release under `GUST_PROFILE=cpu` and runs it against host references (E-a); scans the binary for `slangc`, `gust-slang-reflect`, `wgpu` and `vulkan` and reads `cargo tree -e features` (E-b); builds with `--cfg gust_dual` (E-c); checks four refusals (E-d to E-g: the missing twin without the profile, `gust-wgpu` under `cpu`, the bridge feature under `cpu`, an opted-out kernel); flips the profile six times in one target directory (E-h); runs `cargo test -p kernels` under `cpu` (E-i). Stops at the first failing step and names it. CI runs it as the job `cpu-target`, beside `unit` |
| `cargo xtask bless` | Update reviewed Slang goldens | reruns the golden tests with `BLESS_GOLDEN=1`, which rewrites changed `tests/fixtures/*.slang`, prints a line diff and fails the test; lists changed fixtures with `git diff --stat`. Exits non-zero whenever a fixture changed: review the diff, then run `cargo xtask check-feature macro` |
| `cargo xtask test --suite Unit\|Cpu\|Slangc\|Spirv\|Reflect\|Gpu\|Full [--skip-difftest]` | One evidence tier as one command; `--skip-difftest` (CI) leaves out the differential harness steps, which have their own job | Unit: macro and xtask tests, lint, docs-check, check-arch, check-ui, check-cpu, the `gust::cpu` unit tests, the format check, clippy and tests of `tests/difftest`, the CPU-only differential batch. Cpu: `check-cpu-target`. Slangc: Unit plus `gust` library tests. Spirv: `check-artifacts`. Reflect: `check-feature reflection`. Gpu: all `gust-wgpu` tests, all `gust-algo` tests (its `gpu::` tests required by name), then the differential batch of 8 seeds. Full: `check-full`. Each suite prints what it does not cover; a missing tool or adapter fails it |
| `cargo xtask check-fast` | Fast routine confidence | policy lint, docs-check, check-arch, check-ui, helper version check, fmt check, macro tests, core lib tests, wgpu tests (all `--locked`) |
| `cargo xtask check-workspace` | Default workspace confidence | `--locked` workspace tests with example packages excluded entirely |
| `cargo xtask check-examples` | Major example behavior confidence | every example test (`-- --include-ignored`, counted: no passed test or an ignored one fails it) plus example binaries |
| `cargo xtask check-artifacts` | Artifact/export confidence | example binaries plus external SPIR-V validation of their twenty files and of the files of nine golden sets (vectors and products, uniforms, map, grid, workgroup, `par_for`, reduce, scan, compact), with an `OpCopyLogical` count each |
| `cargo xtask check-full [--skip-web]` | Release confidence | full staged verification, then the whole `check-precompiled` and `check-web`, then `check-difftest`; `--skip-web` leaves out `check-web` only and records it |
| `cargo xtask status` | Resume context | git status, active claim, validation summary, suggested check |
| `cargo xtask doctor` | Environment readiness | required tools and reflection helper status; exits non-zero when cargo, rustc, slangc or the helper is missing |
| `cargo xtask doctor --smoke` | Toolchain smoke test | doctor, then a hand-written Slang kernel compiled to WGSL and SPIR-V in a fresh target directory and the SPIR-V validated; the Rust frontend and GPU execution are not run |
| `cargo xtask slangc <file.slang> --entry <name> --target wgsl\|spirv --out <path>` | One reproducible slangc compile | refuses an existing output, fails on a missing or empty artifact, records version, arguments and SHA-256 in `<path>.command.json` |
| `cargo xtask spirv-val <file.spv> --target-env <env>` | One SPIR-V validation | length and magic check, then `spirv-val`; prints the SHA-256 and validator version |
| `cargo xtask test-exact -p <package> --test <target> <name>` | One exact test, ignored or not | lists the tests, requires the name exactly once, runs it with `--include-ignored`, fails unless one test executed and passed |
| `cargo xtask list-tests` | Test inventory | summarizes test categories from Cargo's test list |
| `cargo xtask explain-check <area>` | Command intent | prints why/when to use a check |
| `cargo xtask measure-tests --json target/test-times.json` | Profiling before pruning | timed default test commands or one custom command |

`cargo xtask verify --mode fast|gpu|examples|artifacts|full` exposes the same staged
checks. Routine verify modes do not rewrite `.ai/VALIDATION.json` unless passed
`--record`; `check-full` and `verify --mode full` record by default. The native
reflection helper build is gated by content: a stamp beside the helper holds the hash of its C++ source, and
the helper's `--version` must give the Slang tag of the SDK, so repeated checks do not relink an unchanged helper
and a helper built against another Slang is rebuilt.

CI also runs the Gpu suite on Mesa lavapipe (`.github/workflows/gpu-software.yml`): `Gpu (software)` evidence, never hardware evidence and never a required check ([VALIDATION](docs/development/VALIDATION.md)).

The same checks are exposed as VS Code tasks: `GUST: check fast`, `GUST: check
feature`, `GUST: check changed`, `GUST: check examples`, `GUST: check artifacts`, and
`GUST: check full`.
Do not introduce shared `HeadlessDevice` fixtures until `cargo xtask measure-tests` shows
adapter/device setup is the bottleneck; cache-stat tests intentionally use isolated
devices.

## Install from crates.io

Not on crates.io yet: 0.1.0 is being prepared. Do not run the `cargo install` and `cargo add` commands of this
section until 0.1.0 is published. The two code blocks below are the recipe that `cargo xtask check-release` builds word
for word from the packaged crates.

GUST will be published as four packages that always share one version (0.1.0 is the first release):

| Package | Library name | What it is |
| --- | --- | --- |
| `gust-compute` | `gust` | The `#[gpu]` attribute, `par_for!`, typed buffers and calls, the CPU runtime |
| `gust-compute-wgpu` | `gust_wgpu` | The headless wgpu backend that runs the kernels |
| `gust-compute-macros` | `gust_macros` | The proc macros; `gust-compute` uses them and you never depend on them |
| `cargo-gust` | none (a binary) | The `cargo gust` tool: precompiled artifacts and the pinned Slang |

The package name and the library name differ, so you add one and write the other: `cargo add gust-compute
gust-compute-wgpu`, then `use gust::gpu;` and `use gust_wgpu::...` in the code. Every `use gust::` snippet in this
README assumes that `cargo add` line. Do not run `cargo add gust`: that name belongs to an unrelated crate.

A kernel runs in one of two ways (D44, D45). During development it is compiled at run time through Slang, which needs
the pinned Slang release and a native helper (set up below). In a release build it was compiled ahead of time by
`cargo gust build` and is embedded from the committed `gust-artifacts/` directory, so the release binary needs neither
Slang, nor the helper, nor `cargo-gust`.

### Set up the tools

```sh
cargo install cargo-gust --locked --version "~0.1"
cargo gust slang install        # downloads Slang 2026.20, checks its SHA-256, unpacks it into the cargo home
cargo gust slang build-helper   # compiles the native reflection helper against that Slang and prints its path
cargo gust doctor               # names whatever is still missing
```

`cargo install` needs no environment variable. Slang is downloaded from its own release page and never shipped inside
the crates ([licence notes](docs/development/LICENSES-SLANG.md)). `build-helper` compiles one C++ file, so it needs a
C++ compiler: MSVC on Windows, g++ or clang++ elsewhere. Only the development path and the author who runs
`cargo gust build` need the last three commands; a release build of your crate, and the people who run it, need none
of this.

The run-time compiler does not look into that cache by itself yet: it launches `slangc` from `PATH` and takes the
helper from `GUST_SLANG_REFLECT`. Give it both (`cargo gust` itself needs neither):

```sh
export GUST_SLANG_REFLECT=<the path that build-helper printed>
export PATH="<the folder of the slangc that `cargo gust slang locate` prints>:$PATH"
```

In PowerShell write `$env:GUST_SLANG_REFLECT = "<path>"` and `$env:PATH = "<folder>;$env:PATH"`.

### A first program

Make a binary crate (`cargo new hello-gust`) and give it this `Cargo.toml`. The `Cargo.toml` and `src/main.rs` blocks of
this section are the recipe that `cargo xtask check-release` builds from the packaged crates; its full mode also runs
them on a GPU.

```toml
[package]
name = "hello-gust"
version = "0.1.0"
edition = "2024"

[dependencies]
gust-compute = "0.1"
gust-compute-wgpu = "0.1"

[features]
# Run-time compile through the Slang toolchain, for development only (`cargo run --features gust-dev`).
gust-dev = ["gust-compute-wgpu/slang-runtime"]
```

`gust-dev` is a feature of your own crate that turns on the backend's run-time compiler, and it has to be your own
feature. A `slang-runtime` feature on the dependency line makes `cargo build --release` fail, because a release build
refuses the run-time compiler (D68). One on a `[dev-dependencies]` entry never reaches `cargo run`, which then stops
with "built without the `slang-runtime` feature". The dependencies say `0.1` and nothing more: the library crates pin
each other to one exact version, so a range cannot mix two releases.

And this `src/main.rs`:

```rust
use gust::gpu;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

#[gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}

fn main() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let ga = device.create_f32_buffer("a", &a, GpuBufferAccess::ReadOnly);
    let gb = device.create_f32_buffer("b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_f32_buffer("out", &[0.0; 4], GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(&device, 4, &ga, &gb, &out).expect("dispatch");
    let result = device.read_f32_buffer(&out).expect("read back");
    println!("hello-gust: {result:?}");
}
```

### Run it while you develop

```sh
cargo run --features gust-dev
```

The first call of the kernel compiles it through Slang, so this needs the tools and the two environment variables
above and a Vulkan compute adapter (see "Platforms"). It prints `hello-gust: [11.0, 22.0, 33.0, 44.0]`.

### Build it for release

```sh
cargo gust build -p hello-gust    # compiles the kernels with Slang, verifies them and writes gust-artifacts/
cargo build --release
./target/release/hello-gust       # target\release\hello-gust.exe on Windows
```

A release build with `GUST_PROFILE` unset is the strict `native` profile (`web` on wasm32): every `#[gpu]` kernel needs
its artifact, and the binary holds no Slang bridge (the build profiles are described under "Try it"; unless
`cargo-gust` is listed as a dependency, which cargo flags as invalid: that build is `dev`, D117). Commit
`gust-artifacts/`; `cargo gust check` fails when it no longer matches the sources. The first `cargo gust build` of a
crate runs `cargo clean -p` for it once and says so.

### Renamed dependencies and name clashes

The macro writes `::gust::` and `::gust_wgpu::` paths into your crate, and they resolve because the libraries are
called `gust` and `gust_wgpu`. Two things follow.

- If you rename a dependency in your manifest, for example `g = { package = "gust-compute", version = "0.1" }`, add
  `extern crate g as gust;` at the root of your crate. If `gust-compute-wgpu` is renamed too (say to `gw`), also add
  `extern crate gw as gust_wgpu;`. A doctest that uses `#[gpu]` needs the same lines inside it. The crate-path spike
  (D106) built and ran a consumer with one and with both dependencies renamed this way, on the GPU, and passed such a
  doctest.
- Two direct dependencies whose library is called `gust` cannot be used together: `gust-compute` and the unrelated
  crate `gust`. Cargo resolves them, but rustc stops at the first `use gust::...` with `error[E0464]: multiple
  candidates for rmeta dependency gust found`. The remedy is to rename the other dependency with `package =` in your
  manifest; the spike measured the error, not that remedy. With a stand-in crate named `gust` (the real one does not
  build on current stable for an unrelated reason), a transitive dependency on it built, and `cargo doc` of both only
  warned about an output filename collision.

## Platforms

- **Supported:** Vulkan on Windows and Linux. The native backend asks wgpu for Vulkan only, so
  `HeadlessDevice::new()` needs a Vulkan compute adapter. In the browser the backend uses WebGPU through
  `wasm32-unknown-unknown` (`examples/web-vector-add`, the strict `web` profile, `cargo xtask check-web`).
- **Not supported in 0.1.0:** macOS and Metal (no Metal backend is compiled), and DX12 is not claimed. Software
  adapters such as Mesa lavapipe run the GPU tests in CI as `Gpu (software)` evidence; they are not a target.
- **CPU:** a `GUST_PROFILE=cpu` build contains no GPU runtime and runs the kernels as plain Rust (see "Try it").
- **SPIR-V:** SPIR-V is exported (`cargo gust build --format spirv`) and validated with `spirv-val` when that is on
  `PATH`; the runtime executes the WGSL. Running the exported SPIR-V is planned for a 0.1.x release (D114).

## Stability

GUST is an experimental compiler and 0.1 is its first release. It follows SemVer for 0.x (D111): a change that breaks
existing code bumps the minor version (0.2.0), and a patch release only adds or fixes. The four packages are released
together at one version and depend on each other with exact versions, so upgrade them together. Each release is
described in [CHANGELOG](CHANGELOG.md).

The promise covers:

- the public API of `gust-compute` and `gust-compute-wgpu`: `#[gpu]` and `par_for!`, the typed call surface, the
  buffer, device and map types, the CPU runtime, the artifact types and the error enums (which may gain variants, so
  match them with a wildcard arm);
- the commands of `cargo-gust` and the lines they print, which scripts and CI read;
- the file formats: `gust-artifacts/manifest.json`, the record files and the reflection records; a newer release reads
  every older schema (D82, D89, D91);
- the diagnostic codes, `GUST-E....` ([DIAGNOSTICS](docs/DIAGNOSTICS.md)): a code keeps its meaning, the message text
  may be reworded.

It does not cover the support surface, which is public only because generated code or `cargo-gust` needs it:
`gust::profile`, `gust::spirv`, `gust::slang::probe`, `gust_wgpu::render_wgpu_source`, the `__gust_*` names that the
macro emits, the hidden `descriptor;` form of `par_for!`, the internal `cargo-gust-bridge` feature and
the toolchain support (the rules for finding Slang that `cargo-gust` shares with the run-time bridge, D109). The
`cargo xtask` commands and the evidence tiers belong to this repository and are not published.

`pixel` kernels, `Shape` and the texture and sampler types are in the published surface and under the promise above, so a
0.x release may change them like the rest. Their evidence is narrower than the rest: a GUST `pixel` or texture kernel
has run on one hardware adapter, the RTX 5090 (Vulkan), and on the lavapipe software adapter, and on no other hardware.
Headless Chrome on that same NVIDIA GPU has run a `pixel` kernel and no texture kernel. Another vendor's GPU, or a
phone, may show a difference the tests have not seen ([Shapes](#shapes-grid-and-pixel) lists what has run).

## Minimum Rust

The minimum Rust version is 1.90 (`rust-version` in all four packages, D30). The `msrv` CI job runs `cargo check
--workspace --all-targets --locked` on exactly 1.90.0, so it shows that the code type-checks there; it runs no tests.
The release checklist makes that job a required check on `main` ([RELEASE](docs/development/RELEASE.md)). The
repository's other checks run on current stable, 1.98.0 at the time of writing. A project of your own resolves its own
dependency versions, which can be newer than this repository's lockfile; if one of them needs a newer compiler, cargo
says so.

## Authoring model

The preferred source vocabulary deliberately resembles Slang (the snippet needs `cargo add gust-compute`; the library
is named `gust`):

```rust
use gust::gpu;

#[gpu]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}
```

`#[gpu]` automatically imports the shader prelude. The names `float`, `int`,
`uint`, `SV_DispatchThreadID`, `StructuredBuffer<T>`, and
`RWStructuredBuffer<T>` are lightweight Rust shadow types: they let rustc parse,
resolve, and type-check the source while the macro translates the original syntax
tree to Slang.

For compatibility with the first prototype, `f32`, `i32`, `u32`, `Invocation`,
`Storage<T>`, and `StorageMut<T>` are still accepted and map to the same Slang
types. New code should prefer the Slang-shaped spellings.

The macro turns each public kernel name into a zero-sized host handle:

```rust
let descriptor = vector_add::add::DESCRIPTOR;
println!("{}", descriptor.slang_source);
println!("entry point: {}", descriptor.entry_point);
```

The source function is retained as a private Rust shadow that rustc type-checks. It
is not a public CPU execution API (there is no `.cpu(...)` wrapper). In a test build
(`cfg(test)` of the crate that defines the kernel) or a dual build (`--cfg gust_dual`,
see "Development and next milestones") the macro also emits a CPU twin that runs this
shadow body on `gust::cpu::CpuDevice`; `#[kernel(cpu = "unsupported")]` leaves a kernel
without one. A production build contains no twin. CPU comparisons in the benchmark
examples are ordinary, independent Rust implementations outside the `#[gpu]` module, and
they stay the expected values in tests.

### Shapes: `grid` and `pixel`

A kernel has a shape, which says how its elements map onto the hardware (D61, D63; `gust::Shape`). `grid`, the
default, is the compute dispatch used so far. `pixel` runs the same kind of body in the fragment stage: the host draws a
full-screen triangle, and each fragment computes one element and stores its outputs, which stay ordinary storage
buffers, at its own index. It is computation, not a graphics API: GUST owns no blend, depth or raster state. A kernel
opts in with a pin:

```rust
#[kernel(shape = pixel)]
pub fn scale(id: SV_DispatchThreadID, input: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) {
    let i = id.x;
    out[i] = input[i] * 2.0f32;
}
```

The typed `call`, `submit` and `dispatch` are the same (without the `_extent` forms), the entry point is
`[shader("fragment")]`, and `id.x` is the element index (the fragment's `y * 8192 + x`; `id.y` and `id.z` are 0).
Nothing infers a shape at run time: a kernel is `grid` unless it pins `pixel`. The classifier
(`crates/gust-macros/src/validate/classify.rs`, run in the expansion for every kernel before the profile is chosen)
works out which shapes a kernel could take, lists them in
`KernelDescriptor::compatible_shapes` (information only) and checks a pin against them. Its rules are syntactic and
conservative, and whatever it does not recognise makes a kernel `grid`-only. A kernel is `grid`-only when it:

- has workgroup memory, a barrier, `LocalId` or `GroupId`, an atomic, a storage or write-only texture, or an explicit
  `workgroup_size`;
- writes an output at an index other than its own, reads an output, does not write every output on every path (an
  `if` without `else` counts only when its condition is `own < out.len()`), or has a `return`;
- has a `mut` invocation parameter, no output buffer, or an output element that is not a `u32`, `i32`, `f32` or a 2- or
  4-wide vector of them. The map generator's and `par_for!`'s kernels are `grid`-only by these rules.

A pin the kernel cannot take is a build error that names the construct (GUST-E0723 to E0728 in
[DIAGNOSTICS](docs/DIAGNOSTICS.md)); `shape = cpu` is E0722, because `cpu` is a build target (`GUST_PROFILE=cpu`), not
a shape. The CPU twin and the `cpu` target run `pixel` kernels too.

What a `pixel` call cannot do, each refused before any GPU work: it is one-dimensional (an extent call, an indirect
launch, the one-shot `dispatch_f32` and `dispatch_u32` and the generated host snippet are `Error::ShapeUnsupported`);
every output is bound `Shared` (an `Independent` read-write output is `Error::ShapeIndependentOutput`); and it covers
at most 2^26 elements (`gust::PIXEL_MAX_ELEMENTS`) on every device, which `grid` does not (`DeviceLimit` named
`pixel_max_elements`). It also needs an adapter whose fragment stage can write storage buffers
(`FRAGMENT_WRITABLE_STORAGE`); a native adapter without it is `ShapeUnsupported`. In the browser wgpu reports every
such capability, so GUST cannot refuse there; what an adapter without it does has not been observed.

Evidence is narrow. On the RTX 5090 (Vulkan), `pixel`, `grid` and an independent host replay gave identical words for
`u32`, `i32`, `f32`, `Vec2` and `Vec4` outputs at sizes from 0 up to about 1.7e7 elements (the row width and its
neighbours, `8192k + 1` for k up to 1000, and the squares up to 4097^2; `crates/gust-wgpu/tests/pixel.rs`); only `u32`
also ran at the cap of 2^26. In headless Chrome 156 on the same GPU, `examples/web-vector-add` ran the pixel form at
1000 and 8197 elements and matched the host and the `grid` form bit for bit; no other browser adapter has run it. That
is evidence for those bodies, not a proof for every kernel. The contract differs by type: integer outputs are
bit-identical between the shapes, while an `f32` output keeps only the float bounds of SEMANTICS section 4, because a
device may fuse or round differently in the compute and the fragment stage, so the shapes may differ in bits (D128; the
bodies of these tests did not differ; an earlier bench body did, by up to 4 ulp on the RTX 5090). The full contract is
in [SEMANTICS](docs/SEMANTICS.md) and the evidence labels in [VALIDATION](docs/development/VALIDATION.md).

### Textures and samplers

A kernel can take 2-D textures (D64, D90) through four parameter types:

| Parameter | Access | Methods |
| --- | --- | --- |
| `Texture2D<T>` | sampled; `T` is `f32`, `u32`, `i32` or a 2- or 4-wide vector of them | `load`, `sample` (float `T` only), `width`, `height` |
| `StorageTexture2D<F>` | read-write; `F` is `R32Float`, `R32Uint` or `R32Sint` | `load`, `store`, `width`, `height` |
| `WriteTexture2D<F>` | write-only; `F` is any of the nine formats | `store`, `width`, `height` |
| `Sampler` | none | the first argument of `sample` |

The nine formats are `R32`, `Rg32` and `Rgba32` in `Float`, `Uint` and `Sint` (`gust::texture::Format`). Texel
addresses are `u32` and `sample` takes normalised `f32` coordinates. Three-wide texels and converting formats (8-bit,
normalised, sRGB) are not admitted. On the host, `gust_wgpu::GpuTexture2d<F>` comes from `create_texture_2d` with a
`TextureUse` (`SAMPLED` or `STORAGE`), and `write_texture`, `read_texture` and `read_texture_async` move texels;
`GpuSampler` comes from `create_sampler(SamplerOptions)` (nearest or linear, clamp or repeat). The twin has
`CpuTexture2d<F>` and `CpuSampler`. A typed call takes textures and samplers in parameter order, and a buffer where a
texture belongs, or the reverse, is a compile error; use and format are checked at submit. This is the host side of
`examples/texture-threshold`:

```rust
let image: GpuTexture2d<R32Float> = device.create_texture_2d(WIDTH, HEIGHT, TextureUse::STORAGE)?;
device.write_texture(&image, texels)?;
let mask: GpuTexture2d<R32Uint> = device.create_texture_2d(WIDTH, HEIGHT, TextureUse::STORAGE)?;
let bright = device.create_typed_buffer("bright", &[0u32; 1], GpuBufferAccess::ReadWrite);
threshold::mark::dispatch_extent(device, Extent::new(WIDTH, HEIGHT, 1), &image, Independent(&bright), &mask)?;
let mask = device.read_texture(&mask)?;
```

`load` and `store` are inside the exact contract; `sample` is not, and its bound is in [SEMANTICS](docs/SEMANTICS.md).
A linear sample of an `f32` texture needs the device feature `float32-filterable`, which a native
`DeviceOptions::default()` requests when the adapter has it and `HeadlessDevice::new()` and the browser do not; without
it a linear sampler is `Error::TextureFilter` at submit. Binding one texture to two parameters of a call is refused, and
the staged graph refuses kernels with a texture or sampler parameter. A `pixel` kernel takes sampled textures and
samplers, not storage ones. The reflection gate checks each texture binding against the compiler (schema 3 of the
helper, see [architecture](docs/ARCHITECTURE.md)).

## Running a kernel: typed calls

Typed calls are the primary dispatch path. For every kernel handle the macro generates `call`, `submit` and
`dispatch`: `call(element_count, buffers...)`
builds a reusable `gust::Call`, `submit(device, element_count, buffers...)` starts
it and returns a job, and `dispatch(device, element_count, buffers...)` runs it and
waits. The resources follow the kernel's storage and uniform parameters, one reference
each, in parameter order: a buffer for a storage parameter, and a `GpuUniform<T>`
from `create_uniform` for a `Uniform<T>` or `ConstantBuffer<T>` parameter (a texture and a sampler go the same
way for the parameters of "Textures and samplers"). Passing
a buffer where a uniform is expected, or the reverse, is a compile error (E0271).
This is `add_on_gpu` from `examples/vector-add` for the kernel above:

```rust
let a = device.create_f32_buffer("vector add a", a, GpuBufferAccess::ReadOnly);
let b = device.create_f32_buffer("vector add b", b, GpuBufferAccess::ReadOnly);
let out = device.create_f32_buffer("vector add out", seed, GpuBufferAccess::ReadWrite);
vector_add::add::dispatch(device, seed.len() as u32, &a, &b, &out)?;
let result = device.read_f32_buffer(&out)?;
```

The generated functions check these things when the program is compiled: every
argument is a buffer of the element type its parameter declares, all buffers
belong to one device type, there is one argument per storage parameter, and that
device type can run the kernel (`gust::Run`). They do not check:

- access: a read-only buffer passed where the kernel writes compiles and fails
  with a run-time error (`Error::PersistentBufferAccess`) before any GPU work;
- argument order among buffers of the same element type (two read-only `f32`
  buffers of `signal-pipeline` can be swapped without a compile error);
- which device instance a buffer came from: a buffer of another `HeadlessDevice`
  is rejected at submit.

At submit the backend also checks every binding against the kernel descriptor
(access, element layout, device instance, length) and the launch against the
device limits, before anything is compiled or encoded. One buffer may be bound
twice only when both bindings are read-only; if either side writes, the call is
rejected at run time before it writes anything.

A buffer must hold exactly `element_count` elements unless the argument is
wrapped as `Independent(&buffer)` at the call site (a settings buffer, a
reduction's input or output); the kernel must then guard that buffer with its own
`.len()`, and an empty independent buffer is rejected when the dispatch has work.

Launch limits come from the device `HeadlessDevice` opened (m7.a0, D84).
`HeadlessDevice::new()` requests wgpu's default limits, the portable ones the
browser always gets. `HeadlessDevice::with_options` takes `DeviceOptions`, whose
`LimitsPolicy::Adapter` asks for the adapter's own `max_buffer_size`, storage
binding size and workgroups per dimension; that is the native default of
`DeviceOptions::default()` and so of the process's `default_device()` (the device
of the calls that take none, such as `GpuVec::from_slice` and `par_for!` without
`on(..)`). `HeadlessDevice::with_limits` can only lower the limits to emulate a
weaker target. A launch whose workgroup size on any axis, invocations per
workgroup, workgroups per dimension, storage buffer count, binding index or
binding size exceeds them is `Error::DeviceLimit`.

Dispatch can be one-, two- or three-dimensional (exists, m7.d, D102): the typed
`call_extent`, `submit_extent` and `dispatch_extent` take an `Extent { x, y, z }`,
and the one-dimensional forms mean `Extent { x: n, y: 1, z: 1 }`. A one-dimensional
launch covers at most `workgroup_size[0]` times
`gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION` (65 535) elements on
default limits. On both backends the invocation total, with each axis rounded up to
whole workgroups, is at most 2^32 (past it is `DeviceLimit` for
`global_invocation_id`); a zero axis dispatches nothing and returns `Ok(())`.
Staged graphs stay one-dimensional. Evidence (exists, m7.d): on the RTX 5090 a
launch of exactly 2^32 invocations is accepted and runs (the 2D launch reads back its last cell, the 3D one shows
acceptance and invocation 0), and one past it is refused; the 16x16 and 4x4x4 kernels equal a host reference on
the GPU and the twin.

`max_buffer_size` is checked before any wgpu call by the fallible allocations
(`try_create_typed_buffer`, `create_zeroed_buffer` and `GpuVec`), which return
`Error::DeviceLimit`, and an allocation the adapter cannot serve is
`Error::OutOfMemory` on native (exists, m7.a0, D84; the out-of-memory case has a unit
test only, not a hardware run). The infallible `create_typed_buffer` does not
check it.

The older slice paths (`dispatch_f32`, `dispatch_u32`, `F32Binding`,
`U32Binding`, `BufferDispatch` built by hand) still work and are not deprecated.

## Generated Slang

The kernel above becomes source in this shape:

```slang
[[vk::binding(0, 0)]] StructuredBuffer<float> a;
[[vk::binding(1, 0)]] StructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void gpu_vector_add_add(uint3 id : SV_DispatchThreadID)
{
    uint __gpu_len_out;
    uint __gpu_stride_out;
    out.GetDimensions(__gpu_len_out, __gpu_stride_out);
    var i = id.x;
    if (i < __gpu_len_out)
    {
        out[i] = a[i] + b[i];
    }
}
```

Resource bindings are assigned in parameter order. Entry points receive a
prefixed `gpu_<module>_<kernel>` name because source names such as `step`
can collide with Slang intrinsics. SPIR-V compilation uses
`-fvk-use-entrypoint-name` so the descriptor's entry point reaches the backend.
These prefixes are not a general hygienic name-resolution system.

Buffer `.len()` calls lower to Slang `GetDimensions` queries. The helper locals are
generated per resource, so bounds checks continue to describe the actual bound
buffer rather than a separately supplied element count.

## Runtime and artifacts

`gust::slang` is the compiler bridge:

```rust
let spirv_words = gust::slang::compile_spirv(&descriptor)?;
let wgsl = gust::slang::compile_wgsl(&descriptor)?;
```

Each invocation exclusively creates an isolated temporary directory, captures Slang
diagnostics, checks SPIR-V structure, and attempts cleanup on every return path. Missing
`slangc` and compiler failures are reported as normal Rust errors.

`gust::reflect` is the layout gate. `compile_reflected` runs the native helper
(`scripts/probes/slang-layout.cpp`, built by `cargo xtask build-slang-reflect`) to
compile one entry point and reflect the identical linked program; the result carries
the artifact, its hashes, the compiler build tag, and complete layouts (`StorageV1`
buffers; schema 2 adds vectors, matrices and constant buffers, matrices column-major; schema 3 adds 2D
textures, samplers and fragment entries, and the helper writes it only when a program has one of those, so every
compute kernel over buffers still gets its schema 1 or 2 document unchanged).
`Reflection::cross_check_kernel` compares that evidence against the descriptor and
fails on any difference. The SPIR-V it produces is byte-identical to `compile_spirv`,
and its WGSL differs from `compile_wgsl` only in line endings. The older
`reflect`/`Reflection::from_json` path reads `slangc -reflection-json`, which omits
aggregate size, alignment, and stride; it remains available as explicit partial
evidence and is not used by the runtime.

The headless backend currently feeds Slang-generated WGSL into wgpu. This keeps
whole-struct copies compatible with wgpu's shader frontend; Slang-generated SPIR-V
is still a first-class export and is validated by the integration tests with
SPIRV-Tools. A future backend can consume the SPIR-V artifact directly where the
driver/API path supports the emitted instruction set.

Pipeline compilation remains lazy and cached per device. Repeated and batched
dispatches reuse:

- the Slang-generated target shader;
- the wgpu shader module;
- the bind-group layout;
- the pipeline layout;
- the compute or render pipeline.

Persistent typed buffers, heterogeneous struct/scalar bindings, batched command
encoding, asynchronous jobs, and GPU timestamp queries remain available from
`gust-wgpu`.

`StagedGraph` is the first execution-graph slice: upload, dispatch, and readback
nodes, run as one ordered submission in insertion order. The typed methods
(`write`, `run` for a typed `Call`, `run_indirect`, `read`) derive each node's
edges from the buffers it touches and the access the kernel declares: for every
buffer, the latest earlier writer, and for a buffer the node writes, every earlier
node that touched it since that writer. The set is conservative, not a transitive
reduction. Edges are derived from typed calls' declared reads and writes, never
from kernel bodies; explicit edges remain available through `upload`, `dispatch`,
`dispatch_indirect` and `readback`, which take host-declared dependencies, and
the two styles can be mixed. Execution rejects a graph, derived or declared, whose
edges do not cover the buffer hazards its nodes create (a derived node never
repairs an invalid explicit one), and the report gives upload/readback bytes, the
bytes that stayed resident, dispatch, indirect dispatch and workgroup counts, and
how many edges were derived and how many declared. This is the staged-graph
example's graph:

```rust
let mut graph = StagedGraph::new();
// `transform` and `summarize` are typed `Call`s built before the graph.
graph.write(&settings, &new_settings);
graph.run(&transform);
graph.run(&summarize);
let read = graph.read(&summaries);
let output = device.execute_graph(&graph)?;
let summaries = output.readback::<Summary>(read)?;
```

`Independent(&buffer)` at the call site opts one argument out of the shared
element-count rule for settings buffers and reductions; the kernel must then guard
that buffer with its own `.len()`. The explicit
`BufferBinding::independent_length()` does the same for hand-built dispatches.

`GpuPool<T>` is the first growable resident collection: explicit capacity, a
host-tracked logical length mirrored into a one-element count buffer for kernels,
and host-driven geometric growth that copies the live prefix GPU-to-GPU (never a
readback) and retires the old allocation behind the copy's fence. A buffer from
`create_indirect_buffer` can drive `StagedGraph::run_indirect` (typed) or
`dispatch_indirect` (explicit), whose workgroup count the GPU derives. The host
never sees that count, so it is not checked: with wgpu's default instance flags
(`InstanceFlags::VALIDATION_INDIRECT_CALL`, which `HeadlessDevice` keeps) wgpu
silently zeroes a count above the real device's per-dimension limit on Vulkan
(wgpu-core also needs the adapter's `INDIRECT_EXECUTION` and `COMPUTE_SHADERS`
downlevel flags, which Vulkan adapters set); without that
flag the count reaches the GPU unvalidated. A limit lowered with `with_limits` does
not apply to it, and the kernel that writes the arguments must clamp. See the contract in
[docs/ENGINE_HURRICANE.md](docs/ENGINE_HURRICANE.md).

## Element-wise maps

`#[gpu]` on a free function of scalars (or vectors) generates the kernel, the buffers, the bounds handling and the
dispatch of an element-wise call (D60). The function stays an ordinary Rust function; its module of the same name
holds the calls (this snippet needs `cargo add gust-compute` too):

```rust
use gust::{gpu, map::Broadcast};

#[gpu]
pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { a * x + y }

let out: Vec<f32> = saxpy::map(Broadcast(2.0), &xs, &ys)?;          // host slices in, a Vec out
let on_device = saxpy::map_gpu(Broadcast(2.0), &gpu_xs, &gpu_ys)?;  // GpuVec in, GpuVec out, no read-back
```

`map_on` and `map_gpu_on` take a device explicitly, and `map_async` and `map_on_async` do not block (in the browser
the blocking `map` and `map_on` are not emitted). Mismatched lengths are an error, never a truncation. The example
`examples/map-chain` runs three such maps as a resident chain and as host-slice calls.

Map is for resident chains and compute-dense functions. For a cheap function (one multiply-add per element) a map over
host slices did not beat the CPU at any size measured, up to 4.2e6 elements; the planning break-even of the design study
was about 1e7 elements.

Measured 2026-11-04 on an NVIDIA GeForce RTX 5090 (Vulkan) with 24 logical cores, release build with
`-C target-cpu=native`: two runs of an out-of-repository probe (`plans/m7/probe/a2` of the work area) built against
`ea258f9`, in a quiet window (no other build, test or GPU job of this project ran; the GPU was at 0 % before each run).
3 warm-up and 11 timed rounds per cell (5 at 4.2e6), the backends interleaved per round, median of the timed rounds;
each result gives run 1 / run 2. Every map time includes the zero-fill of the map's output buffer, which the generated
call does before the kernel runs; "faster than the CPU" below means a time under 0.9 of the best CPU form (one thread,
or rayon on 24 threads):

| Measurement | n | Result (median, run 1 / run 2) |
|---|---|---|
| map over host slices (`a + b`) versus a hand-written typed dispatch | 1e3 to 4.2e6 (six sizes) | map / hand 0.93 / 0.90 at 1e3, 0.88 / 0.93 at 1e4, 0.84 / 0.89 at 6.6e4, 0.83 / 0.72 at 1e5, 0.64 / 0.64 at 1e6 and 0.61 / 0.62 at 4.2e6; at 4.2e6 map 13.5 / 13.1 ms and hand 22.3 / 21.0 ms (map reuses pooled buffers, the hand-written path allocates three per call) |
| resident three-step chain versus the same chain over host slices | 1e3 to 4.2e6 (six sizes) | host slices / resident 1.54 / 1.32 at 1e3, rising to 2.59 / 2.49 at 1e6 and 2.86 / 2.79 at 4.2e6 (resident 10.5 / 10.7 ms, host slices 30.1 / 29.9 ms) |
| per-call floor (whole call, host slice in, `Vec` out) | 1 / 256 | 0.33 / 0.36 and 0.30 / 0.33 ms (minimum 0.22 to 0.23) |
| break-even, 1 multiply-add per element | up to 4.2e6 | none: map takes 1.97 / 2.00 times the best CPU time at 1e6 and 2.92 / 2.99 at 4.2e6 |
| break-even, 64 multiply-adds per element | 1e3 to 4.2e6 | map is faster from 1e4 (0.79 / 0.58 of the best CPU time) to 1e6 (0.27 / 0.29) and slower at 1e3 (4.24 / 3.07); at 4.2e6 it is not faster (1.36 / 0.97): map takes 70.0 / 50.9 ms there against 4.0 / 4.4 ms at 1e6, three to four times linear scaling, also seen in the smoke run of 2026-11-03 and not explained |
| break-even, 256 multiply-adds per element | 1e3 to 4.2e6 | map is faster from 6.6e4 (0.37 / 0.15 of the best CPU time) to 4.2e6 (0.40 / 0.37); at 1e4 the runs disagree (1.24 / 0.60) and at 1e3 it is not faster (2.08 / 1.02) |

What the runs checked: for the rows that compare map with hand-written code and with the CPU (the first and the
break-even rows), one call per cell was compared with the plain Rust function on the CPU (the dense loops with a
tolerance of 1e-4 that is not derived), not every timed result. The resident-versus-host-slices row compared the two GPU
chains with each other only, and the per-call floor row checked nothing.

## `par_for!`

`gust::par_for!` runs a dialect body once for every index of a one-, two- or three-dimensional range that starts at 0
(exists, m7.p, D103). It is a function-like macro, not an attribute; the bindings are an explicit typed list:

```rust
gust::par_for!(on(device); (x, y) in (0..w, 0..h), [img: &[f32], out: &mut [f32], w: u32] {
    out[y * w + x] = img[y * w + x] * 0.5;
})?;
```

`i in 0..n`, `(x, y) in (0..w, 0..h)` and `(x, y, z) in (0..w, 0..h, 0..d)` give the arity. A binding is `&[T]`,
`&mut [T]`, `&GpuVec<T>`, `&mut GpuVec<T>` or a scalar (`u32`, `i32`, `f32`, `bool`). `on(device);` names the
device (without it the call opens the process's default device) and `workgroup_size(x, y, z);` overrides the
defaults (256x1x1, 16x16x1, 8x8x4). The call is blocking and returns `Result<(), gust::map::Error>`.

- The body is validated as a `#[gpu]` kernel (the same dialect and codes), and the macro adds a bounds guard on all
  three axes, so a workgroup larger than the loop runs nothing outside it. A zero extent returns `Ok(())` without a
  launch; an extent past `u32` or past the 2^32 invocation cap (D102) is an `Err` before any device work.
- Every buffer is an independent binding, and the iterations run in no defined order. A body whose iteration reads what
  another writes is a user error that nothing diagnoses ([SEMANTICS](docs/SEMANTICS.md)).
- Bodies that are equal after lowering share one generated kernel and one artifact. Under `GUST_PROFILE=cpu` the loop
  runs sequentially in ascending index order.
- Evidence (exists, m7.p): 26 tests (14 on the RTX 5090 and 12 on the CPU twin) run ten calls against independent
  host references (1D, 2D, 3D, extents off the workgroup, zero and over-cap extents, scalars, the default device,
  resident `GpuVec`); a Slang golden, byte goldens of the export records and a cpu-target case run without a GPU.
  `gust-wgpu/tests/par_for.rs` and `tests/fixtures/par_for.rs` hold them.
- Not yet: `cargo gust doctor` and `check` list only the first call site of a body, the naga layout gate skips
  `par_for!` kernels, and the over-cap error is `map::Error::Device` rather than a variant of its own.

## Workgroup memory and barriers

A kernel can declare workgroup memory and synchronise a workgroup (exists, m7.g1, D100 and D101):

```rust
#[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
pub fn block_sum(
    id: SV_DispatchThreadID,
    lid: LocalId,
    gid: GroupId,
    input: StructuredBuffer<f32>,
    mut partial: RWStructuredBuffer<f32>,
    mut scratch: Workgroup<[f32; 256]>,
) { /* write scratch[lid.x], workgroup_barrier(), a tree of rounds, partial[gid.x] = scratch[0] */ }
```

- `Workgroup<[T; N]>` is a kernel parameter (not a local, field or helper parameter) and becomes a `groupshared` array
  of the entry point; `N` is a nonzero literal, `T` a scalar or vector, and all of a kernel's arrays together are at
  most 16 KiB (E0718). Its contents start unspecified, so a kernel writes an element before it reads it.
- `LocalId` and `GroupId` (also spelled `SV_GroupThreadID` and `SV_GroupID`) are derived from the dispatch id and the
  workgroup size as `id % size` and `id / size` per axis.
- `workgroup_barrier()` is a statement, allowed only at the top level of the kernel body or of a top-level literal-range
  `for`, because every invocation of the workgroup must reach it (E0414). A kernel with a barrier has no `return`, no
  `break` or `continue` in a loop that holds a barrier, and no labelled jump or `?` (E0415, E0416).
  Helpers cannot take `Workgroup` (E0119). A barrier orders one workgroup only; nothing orders different workgroups.
- Such a kernel has no CPU twin: it must say `cpu = "unsupported"` (E0717), and `cpu = "lockstep"` is rejected (E0709).
  No CPU strategy for barrier kernels exists: a threaded lockstep twin measured about 2.9 ms per barrier round of 256
  threads, 102 to 104 s for a block reduce over 1e6 elements, so the lockstep twin was dropped (D118). Under
  `GUST_PROFILE=cpu` such a kernel has no `Run` implementation.
- Evidence (exists, m7.g1): on the RTX 5090 a 256-wide block reduce equals an integer host sum bit for bit at 1, 255,
  256, 257 and 65 537 elements, every invocation's ids equal `id % 256` and `id / 256`, and the barrier is an
  `OpControlBarrier` in the compiled SPIR-V (`gust-wgpu/tests/workgroup.rs`); 12 compile-fail fixtures and a passing
  control pin the new diagnostics. One adapter only; a read of workgroup memory the validator cannot see is
  outside the defined domain and no test can observe it ([SEMANTICS](docs/SEMANTICS.md)).

## Examples

The workspace contains ten runnable programs (and `web-vector-add`, a browser module run by
`cargo xtask check-web`, which also runs a `pixel` form of its kernel):

- `vector-add` uses the preferred Slang-shaped Rust vocabulary, runs through the
  generated typed `dispatch`, and demonstrates artifact export.
- `polynomial` evaluates a multi-step cubic expression and compares independent
  CPU and GPU implementations, including persistent, batched, timed, and async
  execution.
- `signal-pipeline` performs centering, gain, bias, normalization, energy, and a
  final adjustment across six buffers, with the same benchmark modes.
- `particle-step` exercises nested POD structs, integer metadata, whole-struct
  copies, field updates, heterogeneous typed buffers, and dependent batched
  kernels.
- `typed-pipeline` uses six distinct storage structs, three helper functions,
  three compute kernels, and one ordered `calibrate → classify → finalize` batch.
  Every stage consumes the previous stage's typed output and exports its own
  Slang, WGSL, SPIR-V, and readable wgpu host code.
- `staged-graph` runs one CPU settings write → `transform` → `summarize` → summary
  read as a `StagedGraph` whose edges derive from typed calls, keeping the samples
  and the intermediate resident and asserting the exact edge sets, transfer byte
  counts, ordering across settings updates, and rejection of explicit graphs that
  omit a dependency.
- `component-pool` grows a `GpuPool<Particle>` every frame while GPU-authored
  positions survive each GPU-to-GPU copy, and integrates through an indirect
  dispatch (`run_indirect`) whose active count and workgroup arguments a
  one-invocation kernel derives on the GPU, clamped to the allocation and a
  settings budget; the graph is rebuilt every frame, and its derived edges are
  printed and asserted by the tests.
- `atomic-counter` has 257 invocations increment one shared counter with
  `atomic_add` and checks the total against the host count.
- `map-chain` chains three `#[gpu] fn` maps two ways, as a GPU-resident chain
  (`GpuVec` in, `map_gpu_on` three times, one read-back) and as host-slice calls
  (`map_on` three times), checks both against an independent `f64` reference within a
  float bound derived in the example, asserts that the resident chain takes the device pool's hand-outs of
  one upload and one read-back (a read-back that bypasses the pool would not be counted), and prints both
  times with their conditions. See
  [Element-wise maps](#element-wise-maps).
- `texture-threshold` reads an `r32float` storage texture (37 by 21 texels, not a multiple of the workgroup),
  counts the texels brighter than one half into a buffer with `atomic_add`, and writes a 0 or 1 mask to a
  write-only `r32uint` texture. It checks the mask and the count against a host loop on the GPU and on the CPU
  twin, and exports its Slang, WGSL, SPIR-V and readable wgpu host code.

Release-mode benchmark examples use fixed sizes selected in each example's
`main` function, kept within the default limit of 65 535 workgroups per dispatch
dimension (`gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION`; a unit test
in each affected example checks it); there is currently no
environment-variable size override. The examples enable `slang-runtime`, which a release build refuses unless
`GUST_PROFILE=dev` is set (see the build profiles above):

```sh
GUST_PROFILE=dev cargo run --release -p polynomial
GUST_PROFILE=dev cargo run --release -p signal-pipeline
GUST_PROFILE=dev cargo run --release -p typed-pipeline
GUST_PROFILE=dev cargo run --release -p map-chain
```

CPU numbers in those examples measure separate host implementations. That makes
the comparison explicit: the shader frontend does not promise that one Rust function
has identical CPU and GPU operational semantics outside a defined domain. The CPU twin
of a kernel agrees with the GPU inside that domain and may differ outside it; the
[semantics contract](docs/SEMANTICS.md) states the domain, the float bounds and the
known-mismatch ledger.

## Current subset

The direct syntax translator currently supports:

- inline `#[gpu]` modules containing named-field structs and functions;
- compute kernels with one dispatch-thread parameter and structured-buffer
  resources;
- `float`/`f32`, `int`/`i32`, `uint`/`u32`, booleans, and nested POD structs;
- local bindings (including explicit type annotations), arithmetic and comparison operators, assignments, indexing,
  field access, `if`/`else`, returns, casts, and calls to helpers in the same module;
- implicit helper returns (including tail `if`/`else` and blocks), 32-bit numeric
  suffixes/radix literals, Rust expression grouping, and boolean/integer `!` and `^`;
- struct literals as a `let` initializer or the right side of an assignment (D07);
- `Option<T>` and `Result<T, T>` as local and helper values, including `if let` and
  a statement-only `match` with `Ok(x)`/`Err(e)` block arms (D14);
- the vector and matrix types of `gust::math` (`Vec2` to `Vec4`, `IVec2` to `IVec4`, `UVec2` to `UVec4`, `Mat2` to
  `Mat4`) in `#[gpu]` code, with swizzles, `dot`, `from_cols`, `from_rows`, `mul_vecN`, `mul_matN` and `transpose`,
  lowered to Slang vector and matrix operations, column-major (M6 m6.m1). In a module that names a matrix type, `*`
  and `*=` between matrices, vectors and scalars lower to one overloaded Slang helper, so `m * v` is the matrix times
  the vector as on the CPU side, and unary `-` works on matrices (m6.m2, m6.n; E0323 and E0324 in
  [DIAGNOSTICS](docs/DIAGNOSTICS.md)). Typed buffers of them (`Vec2`, `Vec4`, `Mat2`, `Mat4` and padding-free
  structs of these) are created, dispatched on the GPU and checked against the CPU twin (M6 m6.g1, StorageV2);
  3-wide vectors and `Mat3` are locals and helper values only. Limitations: the constants `Mat2::IDENTITY` and
  `Mat3::IDENTITY` crash the runner when their SPIR-V is handed to wgpu on slangc 2026.20 (the faulting layer is not
  isolated; the executed WGSL route is fine, and no M6 GPU case executes SPIR-V); see
  [SEMANTICS](docs/SEMANTICS.md) section 4;
- uniform parameters, `Uniform<T>` or its alias `ConstantBuffer<T>`, whose `T` is a struct of the module: read one
  field at a time and laid out `Std140V1` with explicit padding (M6 m6.u2; E0205 to E0209 and E0321 in
  [DIAGNOSTICS](docs/DIAGNOSTICS.md)). The host creates them with `create_uniform`, updates them with
  `write_uniform` (also as a `StagedGraph` node) and binds them in parameter order (m6.u1, m6.g2);
- a user name that collides with a Slang or WGSL keyword, a prelude name or a spelling GUST emits is renamed to
  `__gust_r_<name>` in the Slang text and in the descriptors (`ParameterDescriptor.name`, `StructFieldLayout.name`),
  which carry the emitted names; `rust_type` and your Rust source keep the spelling you wrote (m6.k, D17);
- statement-only atomics on `RWStructuredBuffer<u32/i32>` elements: `atomic_add`,
  `atomic_exchange`, and `atomic_compare_exchange`, plus unsigned-only
  `atomic_min`/`atomic_max` (T10);
- bounded `for i in start..end` loops whose bounds type-check as 32-bit integers,
  with `break`/`continue` (the end bound is evaluated once; see D18);
- `wrapping_add`, `wrapping_sub`, `wrapping_mul` and `wrapping_neg` on `u32` and `i32` (m7.w, D87): the plain Slang
  operator on the GPU and Rust's own `wrapping_*` on the CPU twin, so both wrap the same way for every input, and they
  are how a kernel asks for wrapping (a plain `+` that overflows is outside the defined domain). A method call with
  arguments that the dialect does not accept is E0325, whose message lists the accepted ones: `x.wrapping_div(y)`,
  `x.pow(2)`, `x.min(y)`, `x.checked_add(y)` and `x.saturating_sub(y)` are rejected. Related: an unannotated `let`
  whose initializer is only unsuffixed integer literals is E0326, which asks for a suffix or a type annotation (D96;
  both in [DIAGNOSTICS](docs/DIAGNOSTICS.md));
- one-, two- and three-dimensional runtime dispatch (`Extent`, m7.d, D102; see "Running a kernel") with explicit
  three-dimensional Slang workgroup sizes in descriptors;
- workgroup memory (`Workgroup<[T; N]>`), `LocalId`/`GroupId` and `workgroup_barrier()` (m7.g1, D100, D101), and
  `par_for!` loops (m7.p, D103): see their sections above;
- execution shapes (MX, D61, D63): `grid` by default and `pixel` by `#[kernel(shape = pixel)]`, with the classifier
  that decides which a kernel can take and the pin errors GUST-E0721 to E0728 (E0729 is the profile's own refusal); see "Shapes: `grid` and `pixel`" for
  what makes a kernel `grid`-only and for the limits of `pixel` (one-dimensional calls, `Shared` outputs, at most 2^26
  elements, fragment-stage storage writes);
- 2-D textures and samplers (MX, D64, D90): `Texture2D<T>`, `StorageTexture2D<F>` (the three `r32` formats),
  `WriteTexture2D<F>` (nine formats) and `Sampler` parameters with `load`, `store`, `sample`, `width` and `height`
  (GUST-E0731 to E0736); see "Textures and samplers". Three-wide texels, converting formats, 1-D, 3-D, cube, array and
  multisampled textures, comparison samplers and mip levels other than 0 are not in the dialect.

The validator intentionally rejects allocation, standard-library access, macros,
closures, async/await, references (other than an atomic's `&mut buffer[index]`
receiver), `while`/`loop`/inclusive ranges, general `match` and `?`,
`unwrap`/`expect`, unsafe code, arbitrary method calls, and external function calls
inside GPU modules. These restrictions describe the implemented translator, not
limits of Slang itself.

Struct literals in other expression positions, nested struct literals, and record
update syntax (`..base`) are rejected; whole-struct buffer copies and field updates
work. Option and Result values cannot be struct fields or buffer elements.
Booleans are expression values, not host-shareable `GpuPod` fields. A buffer element is a 32-bit scalar, a 2- or
4-wide vector, a 2x2 or 4x4 matrix or a struct of these. `StorageV2` describes the padding that natural alignment
implies, but a `GpuPod` type is the padding-free subset, so a struct with implicit padding is E0208 and no
uninitialised byte is uploaded. A uniform is a `Std140V1` struct whose padding is written as fields (E0209) and holds
no `Mat2` (E0205). 3-wide vectors and `Mat3` are not buffer elements (E0204), and neither are arrays: `[T; N]`
does not implement `GpuPod`, so rustc refuses one (no test pins this). Since m7.lv (D116) the macro also writes these
layouts into the export record, `storage_v2` for an element that holds a vector or a matrix and `std140_v1` for a
uniform, and `cargo gust build` and `check` compare each with the reflection (a layout the macro cannot describe, an
alias or a type from outside the module, is filled from the reflection and listed in `export_unchecked` instead);
before it, a `Vec4` or `Mat4` element failed `cargo gust build` and a uniform was not checked at export (D104,
retired by D116). See
[architecture](docs/ARCHITECTURE.md) for known semantic
gaps (and [SEMANTICS](docs/SEMANTICS.md) for limits); successful Rust checking does not prove equivalence to Slang inference.

Around the translator, the host side currently provides:

- generated typed `call`, `submit` and `dispatch` per kernel (a kernel with a
  uniform parameter takes one uniform resource per such parameter), implemented for the
  headless wgpu device only; compile-time checks of element type, device type and
  resource count, and run-time checks of access, layout, device instance,
  length and aliasing (see "Running a kernel");
- compile-time assertions emitted with `#[gpu]` that every buffer element type and
  struct field is a padding-free layout (StorageV1, or StorageV2 where a vector or matrix is involved) matching Rust's
  size and alignment (a failing one is rustc error E0080 that names the field or the
  parameter), and a run-time StorageV1 assertion when a typed buffer,
  pool buffer or indirect-arguments buffer is created for a hand-written `GpuPod`;
- launch-limit checks (`Error::DeviceLimit`) against the opened device's limits or
  lower ones from `with_limits`, before compilation and encoding;
- `StagedGraph` nodes with derived or declared edges, executed in insertion order
  and checked against the buffer hazards, with a report of transfers, residency and
  edge counts;
- `GpuPool<T>` and indirect dispatch whose workgroup count the GPU writes and the
  host does not check;
- `GpuVec<T>` and the buffer pool (m7.a0): a vector resident on one device, taken from that device's `BufferPool`
  (size classes, a 1 GiB cap on idle bytes that evicts the least recently returned first) and returned on
  drop. `from_slice` and `zeroed` use the default device, their `_on` forms a device you hold; `to_vec` and
  `to_vec_async` read it back; `&gpu_vec` goes where a typed call takes a buffer, and bindings show the logical length,
  not the pooled size. Resident arguments of two devices are `Error::MixedDevices`. Tests: `gust-wgpu/tests/gpu_vec.rs`;
- device options (m7.a0, D84): `HeadlessDevice::with_options(DeviceOptions)` chooses the adapter (`AdapterChoice`:
  `HighPerformance`, `LowPower`, `Index(n)` of `HeadlessDevice::enumerate`, or `NameContains(text)`) and the limits
  (`LimitsPolicy::Default`, wgpu's portable ones, or `Adapter`, the adapter's own buffer size, storage binding size
  and workgroups per dimension). `default_device()` opens `DeviceOptions::default()` once per process (native:
  Vulkan, high-performance, adapter limits), and `set_default_options` changes that before the first use (after it,
  `Error::DefaultDeviceOpened`). Tests: `gust-wgpu/tests/device_options.rs`;
- fallible buffer constructors (m7.a0, D84): `try_create_typed_buffer`, `create_zeroed_buffer` and `GpuVec` check
  `max_buffer_size` before any wgpu call (`Error::DeviceLimit`), and on native an allocation the adapter cannot serve is
  `Error::OutOfMemory` (a unit test only, no hardware run); `create_typed_buffer` stays infallible and unchecked;
- textures and the `pixel` runtime (MX): `GpuTexture2d<F>` and `GpuSampler` with the errors `TextureUse`, `TextureFormat`,
  `TextureFilter`, `TextureSize` and `SamplerOptions`, and bind-group layouts built per device with or without
  `float32-filterable`. A `pixel` call is one render pass of its own, in batch order, over an 8192-wide `r8unorm`
  attachment that nothing writes, with at most two scissored draws (the full rows, then the tail row); the kernel's
  fragments write the output buffers (`crates/gust-wgpu/src/render.rs`, D120, D127). Tests: `gust-wgpu/tests/textures.rs`,
  `pixel.rs` and `shape_refusal.rs`;
- `gust-algo` (m7.r1 to m7.r4; a workspace crate with `publish = false`, D110): device-resident algorithms over
  `GpuVec`, every one a GUST kernel. `reduce::sum`, `min` and `max` (`u32`, `i32`, `f32`) and `reduce::sum_wrapping`
  (`u32`) read one element back, `reduce::sum_on_device` keeps it on the device; `scan::inclusive` and `exclusive`
  (`u32`) return a new vector; `compact::count`, `indices` and `gather_u32`, `gather_i32` and `gather_f32` keep the count
  on the device. Each reduction and scan has a v0 form (block-strided, with CPU twins in test and dual builds) and a v1
  form (`Workgroup<[T; 256]>` and barriers, `cpu = "unsupported"`). The public functions run v1 for all six operations
  (`gust_algo::variant::shipped`; D121: v1 was faster than v0 at 1e6, 1e7 and 1e8 elements on the RTX 5090 and on the
  Radeon 890M) and fall back to v0 for an input or a device whose v1 launch does not fit; the `*_with` functions of
  `reduce` and `scan` ask for either (a v1 request falls back the same way). Integer sums and scans are exact where no
  partial sum (or prefix) overflows, `sum_wrapping` is defined modulo 2^32 for every input, and `min` and `max` are
  exact. A float sum is within the per-level bound of `gust_algo::bound` (`f32_sum_bound_v1` for v1, `f32_sum_bound`
  for v0) of the exact sum, not a bit pattern; the two variants may differ in the last bits, and the tests compare
  each with a reference, never with each other. The kernels compile at run time through the crate's `slang-runtime`
  feature; it ships no precompiled artifacts. Tests: `cargo xtask check-feature algo` (the adapter cases are pinned by
  name in the Gpu suite, see [VALIDATION](docs/development/VALIDATION.md)).

Its CPU backend, `gust::cpu`, runs the Rust body of each kernel that has a typed call, unless it opts out with
`#[kernel(cpu = "unsupported")]`: as a test twin (emitted under
`cfg(test)` or `--cfg gust_dual`) and, under `GUST_PROFILE=cpu`, as the production CPU target described above.
The twin is test evidence next to the GPU and the independent host
reference, never in place of them, and never GPU evidence; where it may differ from the
GPU, [SEMANTICS](docs/SEMANTICS.md) lists the limitation. It does not yet provide a compile-time portable limit floor
(roadmap M8) or an access mode in the buffer type. A kernel with workgroup memory or a barrier has no CPU twin and says
`cpu = "unsupported"` (E0717): no CPU strategy for barriers exists, and `cpu = "lockstep"` is rejected (E0709), because
the threaded lockstep twin measured far too slow (S7.4, D118).

Under `GUST_PROFILE=cpu` the calls of a generated map (`map`, `map_on` and `map_on_async`) run in parallel chunks of the
output on scoped `std` threads (m7.c, D119). Each chunk writes only its own elements, so the result equals a single
thread's bit for bit; if elements panic, the panic of the lowest index is raised and nothing is returned. The thread
count is `GUST_CPU_THREADS` (a positive integer; anything else is an error that names the value) or else the machine's
parallelism, then bounded by `min(threads, max(1, n / 2^18))`, so a short call stays on one thread; `wasm32` has one.
Everything else on the CPU stays sequential: `par_for!` bodies, user-written kernels and the dual twin. The source
comment of `gust::cpu::parallel::ELEMENTS_PER_THREAD` records the measurement behind the 2^18 (24 logical cores):
vector-add within 10 % of one thread from 2^16 to 1e6 elements, and the 64-step polynomial map 16 times faster than one
thread at 1e8. Tests: `gust/tests/parallel.rs` (equality with one thread, the lowest-index panic, the variable's
parsing through functions that take the value) and the map case of `cargo xtask check-cpu-target`, which does not set
`GUST_CPU_THREADS`.

## Workspace

```text
crates/
  gust/                 shadow types, descriptors, ABI metadata, slangc bridge, reflection
  gust-macros/          validation and direct syn AST -> Slang translation
  gust-wgpu/            cached headless wgpu execution, buffers, StagedGraph, GpuPool
  cargo-gust/           the `cargo gust` subcommand: init, build, check, doctor and the pinned-Slang manager
  gust-algo/            device-resident algorithms on GUST kernels over GpuVec (reduce: sum, min, max, sum_wrapping; scan: inclusive, exclusive; compact: count, indices, gather_u32/i32/f32); publish = false
examples/
  vector-add/
  polynomial/
  signal-pipeline/
  particle-step/
  typed-pipeline/
  staged-graph/
  component-pool/
  atomic-counter/
  map-chain/
  texture-threshold/
  web-vector-add/
tests/fixtures/         reviewed golden Slang and GPU semantics fixtures
tests/ui/               compile-fail fixtures for the stable diagnostics (check-ui)
tests/dual/             workspace of its own: the dual-build (CPU twin) check
tests/cpu-target/       workspace of its own: the production CPU target check
tests/precompiled/      workspace of its own: a consumer of precompiled kernels
tests/difftest/         workspace of its own: the differential-test harness
scripts/probes/         native Slang reflection helper source
xtask/                  project automation and validation commands
generated-wgpu/         ignored local artifacts emitted by examples
docs/
  ARCHITECTURE.md
  development/          status, task queue, contracts, handoff
```

## Development and next milestones

```sh
cargo xtask check-fast
cargo test --workspace
```

Use focused checks first. `cargo test --workspace` skips ignored example tests;
`cargo xtask check-full` runs them with `--include-ignored`, runs each example binary, and
validates exported SPIR-V artifacts. Per-feature compile-smoke tests use WGSL only;
SPIR-V structure/semantic validation is centralized in full verification.

Dual test builds contain the GPU pipeline and the CPU twins together, chosen at build
time and never through a Cargo feature (D26). `cfg(test)` gives a crate the twins of its
own kernels; `--cfg gust_dual` gives them to a dependent crate's tests and to binaries.
Do not set the flag by hand: `cargo xtask check-dual` sets it through
`CARGO_ENCODED_RUSTFLAGS`, builds into a separate target directory, and proves with the
`tests/dual` fixture that a dependent crate sees the twin only with the flag
(`--check-only` needs no GPU). `cargo xtask check-cpu` runs the twin tests of the
examples without a GPU. The workspace declares `gust_dual` to rustc's `unexpected_cfgs` lint.

Golden Slang and a real-GPU semantics fixture live in `tests/fixtures/`; update
expected output only after reviewing the change and testing both target compilers.
The [roadmap](docs/ROADMAP.md) prioritizes correctness, actionable diagnostics ([stable codes](docs/DIAGNOSTICS.md)),
ABI/capability evidence, and then a small explicit staged-execution proof.
Slang reflection now gates every pipeline for the StorageV1, StorageV2 and Std140V1 layouts (D17) and for 2-D textures and samplers (D91); broader
resource layouts need the same compiler evidence before they enter the runtime.
New ECS work starts with a small GPU-resident component pool, not a full engine rewrite.

See [the architecture note](docs/ARCHITECTURE.md) for the boundary decisions and
their rationale.

Licensed under MIT or Apache-2.0.
