# Roadmap

This is the ordered plan the project follows. It was agreed with the owner on
2026-10-23. Milestones run in order; work inside a milestone may run in parallel
lanes. Decisions behind it are D21–D27 in [DECISIONS](DECISIONS.md). Claimable task
detail lives in [NEXT_TASKS](development/NEXT_TASKS.md) and task contracts in
[tasks/](development/tasks/README.md). Current state and evidence live in
[STATUS](development/STATUS.md).

## How to use this file

- Read **Now and next** first, including after a context compaction.
- Status words are facts: `planned`, `ready`, `in progress`, `done (<tier>, <commit>)`.
  A status never moves to `done` without a run record at the stated evidence tier
  (see [VALIDATION](development/VALIDATION.md)).
- Change the order or content of milestones only with an owner decision recorded in
  DECISIONS.
- No kill rules are attached for now (owner, 2026-10-23); scope is reviewed at
  milestone boundaries instead.

## Direction

GUST is a published, compute-only crate for writing GPU code in Rust: "Slang with Rust
syntax and rustc checking". It covers as much Rust syntax as can be lowered with a
written contract, and brings GPU concepts (vector types, SIMT dispatch, workgroup
memory, capability-specific code) into Rust. The target hardware and platform are
chosen at build time; GPU and CPU builds follow one semantics contract (D21, D22).
GUST is the foundation for two later projects: Breeze, a rendering and interaction
framework, and Hurricane, the game engine (today's
[ENGINE_NORTH_STAR](ENGINE_NORTH_STAR.md)). Their scope is defined when GUST is ready.

## Now and next

**Now: M3, dual test builds.** Plan adopted 2026-10-24 (owner-delegated); slices run one at a time. The M2 sanity check (D37) keeps M3's scope
and fixes the order inside it.

| Item | Work | Status |
| --- | --- | --- |
| M3 plan | Plan with pre-registered spikes; Fable plan, Codex review (adopt with changes), two revisions, Opus re-review in the Codex slot; overflow decision D38 | done (D38, confirmed by the owner in D39) |
| S1 | Spike: a minimal `gust::cpu` and a macro-emitted `impl Run` for vector-add, compared with the host reference and the GPU (kill: `unsafe`, a new dependency, or the downcast leaking into generated code) | done (decision A: no kill rule; twin == host reference == GPU exactly at counts 1, 63, 64, 65, 257; local spike commit, not merged; m3.b and m3.c build on it) |
| m3.a1 | Known-mismatch ledger: GPU edge-case probes (S2, incl. NaN, subnormal, out-of-bounds), fixture and ledger test keyed on the run-time overflow mode, release ledger pass in `check-full` | done (Gpu: ledger test debug and release, RTX 5090; Opus keep after one fix round; `96ec90b`) |
| m3.a2 | `docs/SEMANTICS.md`: the executable-semantics contract with spec-allowed, measured and proven labels, per-operation float bounds with input domains, memory preconditions | done (docs only; Opus keep after one fix round; `cc88b1a`) |
| m3.b | `gust::cpu`: `CpuDevice`, `CpuBuffer`, bindings, jobs, validation, `overflow_checks()` | done (Unit; Opus keep after two fix rounds; `a7dcb95`) |
| m3.c | Macro-emitted CPU twins under `cfg(any(test, gust_dual))`, `#[kernel(cpu = "unsupported")]`, vector-add and polynomial twin assertions with derived bounds | done (Unit + Gpu: check-ui exclusion fixture, vector-add exact, polynomial within 48u on RTX 5090; Opus keep after two fix rounds; `732c774`) |
| m3.c2 | `cargo xtask check-cpu` in the Unit suite; check-ui mode for macro errors without a rustc code, with E0709 and E0710 fixtures | done (Unit; Opus keep; `35eb2a5`) |
| m3.d | Shadow-type breadth (T01-T12): executable atomics, twin assertions in the fixture tests, ledger twin column | done (Gpu: RTX 5090, debug and release ledger; Opus keep after one fix round; `0f2ff27`) |
| m3.e1, m3.e2 | Remaining examples: per-kernel twins or explicit opt-outs (graph examples use read-back intermediates) | planned |
| m3.f1, m3.f2 | `--cfg gust_dual`: a cross-crate fixture proving the twin is absent from production builds, `check-dual`, check-cfg lints, example mains, CI step | planned |

**Done: M2, typed host API** (closed 2026-10-24, D37). Plan adopted after two Codex reviews (owner-delegated); slices
ran one at a time, and each landed slice is recorded in STATUS.

| Item | Work | Status |
| --- | --- | --- |
| S0 | Spike outside the repository: the typed call surface executes on a CPU-shaped device with safe Rust, and pins aliasing (read/read accepted, read/write and write/write rejected at run time) | done (decision A: the planned shape stands; `'r` bounds on generated signatures) |
| m2.b | Launch-limit checks: workgroup width, invocations, storage-buffer count, workgroup count and binding size checked against the opened device before any compile, returning `Error::DeviceLimit` instead of a wgpu panic; `HeadlessDevice::with_limits` lowers limits | done (Gpu: check-fast RTX 5090; Opus keep after fixes; `151ef56`) |
| m2.a | Typed call surface in `gust`: `TypedDevice` (extends `Device`), `Resource`, `Arg`, `Independent`, `Call`, the `Run<K, N>` hook; the backend contract checks bindings against the kernel at submit | done (Unit + check-fast; Opus keep after one fix round; `3060072`) |
| m2.c | `gust-wgpu` implements the typed surface: `HeadlessDevice` is a `TypedDevice` and runs `Call`s through the existing binding validation; one buffer bound twice is rejected when either binding writes; binding indices checked against the device limit | done (Gpu: check-fast and check-examples RTX 5090; Opus keep after one fix round; `119ece3`) |
| m2.d0 | `check-ui` on JSON diagnostics: a failing case has exactly one error, pinned by `code`, `file`, `line`, `message` and `label` keys on one primary span (D33, M1 risk 6) | done (Unit: check-ui and xtask tests; Opus keep after one fix round; `a8b074f`) |
| m2.d1 | Generated `call`, `submit`, `dispatch` on every storage-only kernel handle (uniform kernels get none), with a per-kernel const assert of the storage count; spike S3: a five-parameter kernel infers without turbofish | done (Unit: check-ui fixtures, expansion tests, bless unchanged; Opus keep after one fix round; `a83a96e`) |
| m2.d2 | Const layout assertions: per-field padding-free offsets, no tail padding, alignment 4 and `is_storage_v1` per struct; per-parameter element layout against its Rust size, alignment and stride; check-ui follows macro expansion to the user's line | done (Unit: check-ui fixtures `layout-outside-*`, expansion tests, bless unchanged; Opus keep after two fix rounds; `adeb7be`) |
| m2.e | `StagedGraph::{write, run, run_indirect, read}` derive dependencies from typed calls (RAW, WAW, WAR; conservative, validated, insertion order kept); `validate` runs the limit pass for every node before any compile and rejects a read-write binding of an indirect node's arguments buffer | done (Gpu: graph tests, check-examples RTX 5090; spike S4 matched the pre-registered sets; Opus keep; `229bfbf`) |
| m2.g | A hand-written `KernelDescriptor` that fails in slangc maps to its Rust line through `slang::compile`, both reflect paths and the native helper, independent of validator gaps (D33, M1 risk 5) | done (Slangc + Reflect; Opus keep after one fix round; `c6571d5`) |
| m2.f1 | Examples vector-add, atomic-counter, polynomial, signal-pipeline on generated typed calls; `gust-wgpu`'s typed tests use the generated call | done (Gpu: check-examples, release runs of polynomial and signal-pipeline, RTX 5090; Spirv: check-artifacts; Opus keep; `ef52d86`) |
| m2.f2 | Examples particle-step, typed-pipeline, staged-graph, component-pool on typed calls and derived graphs | done (Gpu: check-examples and four release runs, RTX 5090; Spirv: check-artifacts; derived sets as pre-registered; Opus keep; `2a4f3f6`) |
| close-out | Shared workgroup-limit constant, pool-buffer layout assert, README/ARCHITECTURE/EXECUTION_GRAPH/VALIDATION, D34-D36, `check-full`, M2 sanity check | done (Gpu: check-full RTX 5090 at `634a50e`; Opus keep after fixes; D34-D36; `009f86a`) |
| sanity | Two sanity reviews (Fable and Codex: proceed with fixes, D37) and their fixes: indirect launches skip only the workgroup-count check, `DeviceLimit` docs, the uniform-kernel exception named, the instance-flag condition on indirect-count zeroing | done (Gpu: graph test with mutations, check-fast RTX 5090; Opus keep after fixes; `2d1880b`) |

**Done: M1, diagnostics** (closed 2026-10-23, D33). Plan adopted 2026-10-23 (owner-delegated); slice details are in the
coordinator's plan file, and each landed slice is recorded in STATUS.

| Item | Work | Status |
| --- | --- | --- |
| M1.0 | Minimum Rust 1.90 (D29, D30) plus an `msrv` CI job; validation records stamped with the commit and dirty state; `check-arch` reads dependencies through `cargo metadata`; structured Full-step test | done (Unit: CI unit + msrv on Rust 1.90.0 green at 0e5df0a; Codex review keep after fixes) |
| M1.1 | Rust line table: statement-level source lines recorded by the emitter; sidecar schema v2 with Rust lines; one shared diagnostic annotation for both compile paths (CLI and reflection helper) | done (Slangc + Reflect: diagnostics 9, reflection 4 on RTX 5090; Codex reviews of 5cd2c85, adbd050, a4dabb1, 2197e6c, last verdict keep) |
| M1.2 | `#[diagnostic::on_unimplemented]` on the shadow-type traits; `cargo xtask check-ui` over a fixture workspace | done (Unit: check-ui, 122 xtask tests; Codex keep) |
| M1.3 | `#line` markers injected into the compile-time Slang copy only, so errors name the Rust file and line; exported `.slang` stays clean | done (Slangc + Reflect: 10 of 10 failures on the right Rust line; Codex keep; `8fcf944`, `7007827`) |
| M1.4 | Original tokens kept beside `compile_error!` so IDEs keep context | done (Unit; Fable keep after fixes, fixed; `e01b82c`, `f507ab4`) |
| M1.5 | Stable diagnostic codes (`[GUST-E0401]`), `docs/DIAGNOSTICS.md`, tests assert codes | done (Unit: 111 codes, every code tested or listed unreachable, three guards; Fable keep, fixes `f20d313`; `df07122`, `6e195bc`) |
| M1.6 | Sanity-check fixes (Opus and Fable, D33): clippy `disallowed-methods` replaces the uncoded-error scanner; the code guard counts assertions, not mentions; `resolver = "3"`; one `#line` path rule for macro and compiler; docs | done (Unit + Slangc + Reflect on the merged tree; Opus keep after two fix rounds; `1f961fc`) |

**Done: M0, close-out and housekeeping** (closed 2026-10-23, D28; M0.15 done 2026-11-01).

| Item | Work | Status |
| --- | --- | --- |
| M0.1 | DX19 close-out: `cargo xtask check-full` and an independent review | done (Gpu: check-full RTX 5090 at `71f8d71`; review fixes `027eef7` with check-fast; two Fable reviews) |
| M0.2 | Replace the six PowerShell skill scripts with xtask commands (`slangc`, `spirv-val`, `doctor --smoke`, `test-exact`; `--locked` on existing checks); delete `Test-VerticalSlice.ps1`; fix the stale `.ps1` reference in `crates/gust/src/reflect/mod.rs` | done (Unit, 3ae37e2; check-fast Gpu RTX 5090) |
| M0.3 | Lint that fails when a `.ps1`, `.bat`, `.cmd` or `.sh` file is tracked (D27) | done (Unit, 3ae37e2; check-fast Gpu RTX 5090) |
| M0.4 | `cargo xtask doctor` exits non-zero when a required tool is missing | done (Unit, 3ae37e2; check-fast Gpu RTX 5090) |
| M0.5 | Artifact check compares expected file names, not a count | done (Unit, 3ae37e2; check-fast Gpu RTX 5090) |
| M0.6 | `cargo xtask docs-check`: relative links, conflict markers, decision ids | done (Unit, efc4c2a; check-fast Gpu RTX 5090) |
| M0.7 | Machine-path lint for tracked files | done (Unit, 3ae37e2; check-fast Gpu RTX 5090) |
| M0.8 | Explicit `BLESS_GOLDEN=1` for reviewed goldens, never automatic | done (Unit, efc4c2a; check-fast Gpu RTX 5090) |
| M0.9 | `cargo xtask check-arch`: no custom shader IR (dependency allowlist, no IR modules outside a `future/` path) | done (Unit, efc4c2a; check-fast Gpu RTX 5090) |
| M0.10 | Suites named by evidence tier: `cargo xtask test --suite Unit\|Slangc\|Spirv\|Reflect\|Gpu\|Full` over existing commands | done (Unit, efc4c2a; check-fast Gpu RTX 5090) |
| M0.11 | Thin `CLAUDE.md`: read AGENTS.md first, role mapping, compaction instructions, skills list | done (docs) |
| M0.12 | Skill fixes: GUST does not use trybuild; "host reference" instead of "CPU oracle"; vertical-slice skill points at `check-artifacts`/`check-full` | done (Unit, 3ae37e2) |
| M0.13 | DECISIONS tidy: header with next id, trailing prose moved under a heading | done (docs) |
| M0.14 | Independent review of the 2026-10-23 `xtask` redaction and CI changes | done (Fable review: keep, three should-fixes landed in 3ae37e2) |
| M0.16 | Bounded retry for a transient Vulkan `NoAdapter` in `HeadlessDevice::new`; `NoAdapter` stays a failure, never a skip (seen twice in parallel atomics tests, 2026-10-23) | done (Gpu: check-fast RTX 5090; review keep) |
| M0.17 | DX19 scanner follow-ups: `else` prefix matching, balanced braces in conditions, `match` label numbering, contract note on the heuristic | done (Gpu: check-fast RTX 5090; review keep) |
| M0.18 | Slice B review fixes: check-arch reads root `[workspace.dependencies]` renames; `--locked` wording in VALIDATION.md or flags on Reflect/Spirv suites; xtask `rust-version`; nits | done (Unit, 5503e5a; Codex review: keep after fixes -> M0.20) |
| M0.19 | No silent GPU skips: examples fail on `NoAdapter`; `else if` header braces; `match` ordinal scope in the DX19 contract; retry visibility | done (Gpu: check-examples on RTX 5090, 3296885; Codex review keep) |
| M0.20 | check-arch: quoted TOML keys and basic-string escapes (Codex review of M0.18) | done (Gpu: check-full RTX 5090 2026-10-23T13:24Z at 8200431; Codex review keep) |
| M0.15 | Make the CI `unit` job a required check after a week of green runs (from 2026-10-30) | done (2026-11-01: branch protection on `main` requires `unit`; admins exempt, no required reviews; green since 2026-10-23) |

## Milestones

| Milestone | Content | Status |
| --- | --- | --- |
| M0 Close-out and housekeeping | See Now and next | done (D28; M0.15 done 2026-11-01) |
| M1 Diagnostics | `#line` markers so Slang errors point at Rust file and line (extends DX19); stable diagnostic codes; `#[diagnostic::on_unimplemented]` on shadow-type traits; original tokens kept beside `compile_error!` | done (D33; M1.0 to M1.6) |
| M2 Typed host API | Typed per-kernel dispatch, designed backend-neutral so the same call runs on the GPU or a CPU twin; const `offset_of!`/`size_of` layout assertions; StagedGraph deriving dependencies from typed reads and writes; launch-limit checks | done (D37) |
| M3 Dual test builds | Executable shadow types under the semantics contract (D22); CPU dispatch; CPU twins emitted under `cfg(test)`; `--cfg gust_dual` for cross-crate tests and benchmarks; per-kernel opt-out (`#[gpu(cpu = "unsupported")]`); existing GPU tests gain CPU-versus-GPU checks while semantics fixtures keep independent host references (D26). From M2 (D34): the CPU backend implements `TypedDevice` once and `Run<K, N>` per kernel, keeps the alias contract, and `#[gpu(cpu = "unsupported")]` is the absence of the `Run` impl | in progress |
| M4 Build-time targets and distribution | Build profiles (target plus capability set) chosen at build time; hash-keyed artifact manifest; D17 at export time for WGSL and SPIR-V; runtime `slangc` behind a development feature; a pinned-Slang manager for crate users; first browser WebGPU run | planned |
| M5 CPU target and evidence breadth | Production `cpu` build target and runtime; lavapipe CI job labelled `Gpu (software)`; differential program generator with shrinking; known-mismatch file; scalar `coretests` cases on the GPU | planned |
| M6 Language breadth I | Vectors and matrices (vec3 and mat3 rejected in buffers at first; `m * v` lowers to `mul`); collision-only identifier renaming; uniforms (`ConstantBuffer<T>`) | planned |
| M7 GPU concepts in Rust | `par_for!` SIMT dispatch on both pipelines; workgroup shared memory and barriers, including a CPU strategy; reduce, scan and compaction primitives (T13 lands here). Multi-dimensional dispatch lifts the `max_compute_workgroups_per_dimension` ceiling (D36) | planned |
| M8 Capability variants | Capability-keyed variants with platform aliases; exhaustiveness checked against the crate's declared target list, with CPU as a target; one rustc-checked signature; subgroup operations first (D23). Compile-time portable floor for workgroup size and storage counts; `with_limits` is the runtime half (D36) | planned |
| M9 Language breadth II | Methods in `impl` blocks; user enums with `match`; tuples; fixed-size arrays; non-capturing closures; then traits mapped to Slang interfaces and generics | planned |
| M10 Publication | Benchmarks against rayon, hand-written WGSL with wgpu, CubeCL and rust-gpu; a "Rust program to GPU" example gallery; docs and a book; API stability pass; crate naming (D25); crates.io release | planned |

A first public release may be cut after M7 so that publishing does not wait for M8
and M9; that cut is an owner decision when M7 is done.

## Parking lot

- GPU builds with a CPU fallback (D21 leaves this open as a future option).
- WebGL2 (out of scope for GUST compute, D24; revisit for Breeze).
- Slang automatic differentiation.
- An optional nightly-pinned THIR checker behind a cargo subcommand.
- Per-API SPIR-V, MSL and DXIL builds for native-only capabilities.
- Breeze and Hurricane scope.
- Validator rejections for recursion and for Rust identifiers that are Slang keywords (`new` is a Slang syntax error today; M6 collision-only renaming is the intended home). Found by the M1 plan.
- Release-size example runs in a check: xtask runs examples in debug only, which hid the 65 536-workgroup release bug fixed in `5c34e22`; M2 launch-limit checks (m2.b) make it a typed error.

## Completed foundation (before 2026-10-23)

- **S0, preserve and re-anchor (2026-09-07):** baseline verified, scope documented,
  repeatable full verification.
- **S1, translation and runtime stability:** golden translation and GPU semantics
  fixtures; silent translation errors fixed; entrypoint cache identity and compiler
  process lifecycle hardened.
- **S2, portable contract and language breadth (partial):** target probes,
  kernel-level diagnostics, bounded struct construction, `Option`/`Result`, bounded
  loops, atomics; reflection gates every StorageV1 pipeline (D17).
- **S3, explicit execution model (first proof):** `StagedGraph` with checked explicit
  dependencies and transfer reports; validated indirect dispatch.
- **S4, engine-shaped proofs:** `GpuPool<T>` with GPU-copy growth and GPU-derived
  indirect arguments.

Details are in [history/COMPLETED_TASKS](development/history/COMPLETED_TASKS.md).
