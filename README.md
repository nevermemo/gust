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
shader module, bind-group layout, pipeline layout, and compute pipeline.

## Try it

Requirements:

- The stable Rust toolchain (this checkout was validated with Rust 1.98.0;
  the manifest's older minimum is not independently verified)
- `slangc` on `PATH` (tested with Slang 2026.13.1)
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
`slang-runtime` feature of `gust` and `gust-wgpu`, off by default (D44). The eight examples
enable it in their manifests, and `gust` and `gust-wgpu` enable it for their own tests through a
dev-dependency on themselves, so `cargo run -p vector-add`, `cargo test -p gust-wgpu` and
`cargo test --workspace` need no flag. A crate of your own that runs kernels this way during
development puts the feature on a `[dev-dependencies]` entry,
`gust-wgpu = { version = "...", features = ["slang-runtime"] }`, or passes
`--features gust-wgpu/slang-runtime`, so that its normal dependencies, and a release build, hold no
process-spawning compiler. Check that state with
`cargo check -p gust -p gust-wgpu --no-default-features --lib` (the Unit suite runs it).

Build profiles (D44): the environment variable `GUST_PROFILE` is `dev`, `native` or `web`, and `gust`'s build
script reads it (any other value, including an empty one, fails the build). `native` and `web` are strict: every
`#[gpu]` kernel must have a precompiled artifact, or the build fails with a compile error at the kernel, and the
`slang-runtime` feature is refused, so the build holds no run-time Slang compiler. `dev` has neither rule. A release
build is strict even without the variable: with `GUST_PROFILE` unset, `cargo build --release` of a crate with
`#[gpu]` kernels is treated as `web` for `wasm32` and `native` for every other target, and prints
`warning: gust@<version>: gust: release build without GUST_PROFILE treated as native` (or `web`;
cargo shows a build script's warning only for path and workspace packages, or with `-vv`). A release build that
wants the run-time bridge, such as the benchmark examples below, sets `GUST_PROFILE=dev` explicitly
(`GUST_PROFILE=dev cargo run --release -p polynomial`; in PowerShell `$env:GUST_PROFILE = "dev"` first). The
repository's `cargo xtask` commands run their cargo children under `GUST_PROFILE=dev`, whatever the parent
environment holds (a different value is logged and overridden), which covers `check-full`'s release tests. No
precompiled artifacts exist before slice m4.c, so until then a strict build of any kernel fails by design.

For the full release-readiness check (including ignored example tests, all eight
examples, and mandatory external validation of all thirteen SPIR-V exports):

```sh
cargo xtask check-full
```

It writes `.ai/VALIDATION.json` with command exit codes, stdout, artifact hashes,
and explicit untested targets. This is local debug validation, not release benchmark
evidence or cross-platform certification.

Purpose-built check commands keep the inner loop honest:

| Command | Purpose | Runs |
| --- | --- | --- |
| `cargo xtask check-feature macro` | Macro/validator/emitter work | `cargo test -p gust-macros` |
| `cargo xtask check-feature reflection` | Reflection/layout work | helper version check, core reflection tests, wgpu reflection tests |
| `cargo xtask check-feature loops` | Bounded-loop work | loop golden/rejection test plus GPU loop test |
| `cargo xtask check-feature gpu-smoke` | Cheapest wgpu runtime sanity | wgpu crate unit tests only |
| `cargo xtask check-feature gpu-semantics` | GPU semantic differential work | semantics, numeric, option, loops, struct-assignment tests |
| `cargo xtask check-feature gpu-runtime` | Runtime/reflection gate work | wgpu `reflection`, `limits`, `typed` and `graph` integration tests |
| `cargo xtask check-changed` | Dirty-tree routing | chooses a focused check from changed paths |
| `cargo xtask check-format` | Formatting only | `cargo fmt --all -- --check` |
| `cargo xtask check-lints` | Strict lints | workspace Clippy with warnings denied (`--locked`) |
| `cargo xtask lint` | Policy lint | no tracked shell or PowerShell scripts (D27) and no machine-specific paths in tracked text (Windows, Unix and WSL home paths; two test files may spell placeholder names only); scans tracked files only |
| `cargo xtask docs-check` | Docs consistency | relative Markdown links in project `.md` files resolve to project files (anchors and external links are not checked), no merge-conflict markers, `docs/DECISIONS.md` ids unique and increasing with the right "Next id"; also sees untracked files that are not ignored |
| `cargo xtask check-arch` | Architecture guardrails (D02) | per-crate dependency allow-list read from `cargo metadata`, allow-listed external packages must resolve from crates.io in `Cargo.lock` (catches `[patch]`/`[replace]` redirects), no `ir` / `*_ir` module or file under `crates/` outside a `future/` path, no SPIR-V library in the compiler crates; also sees untracked files that are not ignored |
| `cargo xtask check-ui` | Diagnostic text for misuse (D27) | compiles each crate of the separate `tests/ui` workspace; a crate with `expected.txt` must fail with exactly one error (read from cargo's JSON) that satisfies every listed key (`code`, `file`, `line`, `message`, `label`; `code: none` matches a macro error without a rustc code and needs a `message:`), one without it must compile |
| `cargo xtask check-cpu` | CPU-twin evidence (no slangc, no GPU) | `--locked` tests of the examples with CPU twins (`vector-add`, `polynomial`, `signal-pipeline`, `particle-step`, `typed-pipeline`, `atomic-counter`, `staged-graph`, `component-pool`), whose non-ignored tests run each macro-emitted CPU twin against its host reference; part of the Unit suite. Each call fails when it reports no passed test (the test-count guard), and when a twin test or derivation-domain guard of the example is not reported `ok` by name |
| `cargo xtask check-dual [--check-only]` | Cross-crate dual-build evidence; `--check-only` needs no slangc and no GPU | refuses a preset `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS` or `CARGO_BUILD_RUSTFLAGS`; checks `tests/dual` with fmt and clippy, builds the workspace with `--cfg gust_dual` (through `CARGO_ENCODED_RUSTFLAGS`) in `<target>/dual`, runs clippy `-D warnings` on vector-add and polynomial with the cfg, then checks the `tests/dual` fixture: without the cfg a dependent's own `cfg(test)` (E1) and a production binary (E3) get no CPU twin of a library kernel (one E0277 each); with it the twin test runs and passes (E2). The full mode (GPU) also runs vector-add and polynomial with the cfg, which must print `CPU twin: matches (N elements, max ULP k)`, and vector-add without it, which must not. CI's `unit` job runs `--check-only`; `check-full` runs the full mode |
| `cargo xtask bless` | Update reviewed Slang goldens | reruns the golden tests with `BLESS_GOLDEN=1`, which rewrites changed `tests/fixtures/*.slang`, prints a line diff and fails the test; lists changed fixtures with `git diff --stat`. Exits non-zero whenever a fixture changed: review the diff, then run `cargo xtask check-feature macro` |
| `cargo xtask test --suite Unit\|Slangc\|Spirv\|Reflect\|Gpu\|Full` | One evidence tier as one command | Unit: macro and xtask tests, lint, docs-check, check-arch, check-ui, check-cpu, the `gust::cpu` unit tests. Slangc: Unit plus `gust` library tests. Spirv: `check-artifacts`. Reflect: `check-feature reflection`. Gpu: all `gust-wgpu` tests. Full: `check-full`. Each suite prints what it does not cover; a missing tool or adapter fails it |
| `cargo xtask check-fast` | Fast routine confidence | policy lint, docs-check, check-arch, check-ui, helper version check, fmt check, macro tests, core lib tests, wgpu tests (all `--locked`) |
| `cargo xtask check-workspace` | Default workspace confidence | `--locked` workspace tests with example packages excluded entirely |
| `cargo xtask check-examples` | Major example behavior confidence | every example test (`-- --include-ignored`, counted: no passed test or an ignored one fails it) plus example binaries |
| `cargo xtask check-artifacts` | Artifact/export confidence | example binaries plus external SPIR-V validation |
| `cargo xtask check-full` | Release confidence | full staged verification |
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
reflection helper build is timestamp-gated and still runs `--version`, so repeated
checks do not relink unchanged C++ code.

CI also runs the Gpu suite on Mesa lavapipe (`.github/workflows/gpu-software.yml`): `Gpu (software)` evidence, never hardware evidence and never a required check ([VALIDATION](docs/development/VALIDATION.md)).

The same checks are exposed as VS Code tasks: `GUST: check fast`, `GUST: check
feature`, `GUST: check changed`, `GUST: check examples`, `GUST: check artifacts`, and
`GUST: check full`.
Do not introduce shared `HeadlessDevice` fixtures until `cargo xtask measure-tests` shows
adapter/device setup is the bottleneck; cache-stat tests intentionally use isolated
devices.

## Authoring model

The preferred source vocabulary deliberately resembles Slang:

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

## Running a kernel: typed calls

Typed calls are the primary dispatch path. For every kernel handle whose resource
parameters are all storage buffers the macro generates `call`, `submit` and
`dispatch`: `call(element_count, buffers...)`
builds a reusable `gust::Call`, `submit(device, element_count, buffers...)` starts
it and returns a job, and `dispatch(device, element_count, buffers...)` runs it and
waits. The buffers follow the kernel's storage parameters, one reference each, in
parameter order. The exception is a kernel with a uniform parameter
(`Uniform<T>` or `ConstantBuffer<T>`): its handle and descriptor exist and its Slang
is emitted, but it has no `call`, `submit` or `dispatch` (calling one is error
E0599), and the headless runtime rejects such a kernel on every launch path
(`Error::UnsupportedParameter`), so no kernel with a uniform parameter runs on it
today. This is `add_on_gpu` from `examples/vector-add` for the kernel above:

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

Launch limits come from the device `HeadlessDevice` opened (wgpu's default
limits, not the adapter's); `HeadlessDevice::with_limits` can only lower them to
emulate a weaker target. A launch whose workgroup width, invocations per
workgroup, workgroup count, storage buffer count, binding index or binding size
exceeds them is `Error::DeviceLimit`. Dispatch is one-dimensional, so one launch
covers at most `workgroup_size[0]` times
`gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION` (65 535) elements on
default limits. `max_buffer_size` is not checked by GUST.

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
buffers; schema 2 adds vectors, matrices and constant buffers, matrices column-major).
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
- the compute pipeline.

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

## Examples

The workspace contains eight runnable programs:

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
- statement-only atomics on `RWStructuredBuffer<u32/i32>` elements: `atomic_add`,
  `atomic_exchange`, and `atomic_compare_exchange`, plus unsigned-only
  `atomic_min`/`atomic_max` (T10);
- bounded `for i in start..end` loops whose bounds type-check as 32-bit integers,
  with `break`/`continue` (the end bound is evaluated once; see D18);
- one-dimensional runtime dispatch with explicit three-dimensional Slang
  workgroup sizes in descriptors.

The validator intentionally rejects allocation, standard-library access, macros,
closures, async/await, references (other than an atomic's `&mut buffer[index]`
receiver), `while`/`loop`/inclusive ranges, general `match` and `?`,
`unwrap`/`expect`, unsafe code, arbitrary method calls, and external function calls
inside GPU modules. These restrictions describe the implemented translator, not
limits of Slang itself.

Struct literals in other expression positions, nested struct literals, and record
update syntax (`..base`) are rejected; whole-struct buffer copies and field updates
work. Option and Result values cannot be struct fields or buffer elements.
Booleans are expression values, not host-shareable `GpuPod` fields. Only padding-free 32-bit scalar/nested-struct
storage layouts are executable today. Constant-buffer syntax is not yet supported
by the headless runtime. See [architecture](docs/ARCHITECTURE.md) for known semantic
gaps (and [SEMANTICS](docs/SEMANTICS.md) for limits); successful Rust checking does not prove equivalence to Slang inference.

Around the translator, the host side currently provides:

- generated typed `call`, `submit` and `dispatch` per kernel without a uniform
  parameter (a kernel with one has none), implemented for the
  headless wgpu device only; compile-time checks of element type, device type and
  storage-argument count, and run-time checks of access, layout, device instance,
  length and aliasing (see "Running a kernel");
- compile-time assertions emitted with `#[gpu]` that every buffer element type and
  struct field is a padding-free StorageV1 layout matching Rust's size and
  alignment (a failing one is rustc error E0080 that names the field or the
  parameter), and a run-time assertion with the same meaning when a typed buffer,
  pool buffer or indirect-arguments buffer is created for a hand-written `GpuPod`;
- launch-limit checks (`Error::DeviceLimit`) against the opened device's limits or
  lower ones from `with_limits`, before compilation and encoding;
- `StagedGraph` nodes with derived or declared edges, executed in insertion order
  and checked against the buffer hazards, with a report of transfers, residency and
  edge counts;
- `GpuPool<T>` and indirect dispatch whose workgroup count the GPU writes and the
  host does not check.

Its CPU backend, `gust::cpu`, runs the Rust body of each kernel that has a typed call, unless it opts out with
`#[kernel(cpu = "unsupported")]`, as a test-only twin (emitted under
`cfg(test)` or `--cfg gust_dual`, never in a production build; a production CPU target is
roadmap M5). The twin is test evidence next to the GPU and the independent host
reference, never in place of them, and never GPU evidence; where it may differ from the
GPU, [SEMANTICS](docs/SEMANTICS.md) lists the limitation. It does not yet provide multi-dimensional dispatch
(roadmap M7), a compile-time portable limit floor (roadmap M8), fallible buffer
creation against `max_buffer_size`, or an access mode in the buffer type.

## Workspace

```text
crates/
  gust/                 shadow types, descriptors, ABI metadata, slangc bridge, reflection
  gust-macros/          validation and direct syn AST -> Slang translation
  gust-wgpu/            cached headless wgpu execution, buffers, StagedGraph, GpuPool
examples/
  vector-add/
  polynomial/
  signal-pipeline/
  particle-step/
  typed-pipeline/
  staged-graph/
  component-pool/
  atomic-counter/
tests/fixtures/         reviewed golden Slang and GPU semantics fixtures
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
Slang reflection now gates every pipeline for the StorageV1 subset (D17); broader
resource layouts need the same compiler evidence before they enter the runtime.
New ECS work starts with a small GPU-resident component pool, not a full engine rewrite.

See [the architecture note](docs/ARCHITECTURE.md) for the boundary decisions and
their rationale.

Licensed under MIT or Apache-2.0.
