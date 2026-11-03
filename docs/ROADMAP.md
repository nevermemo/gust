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

GUST is a published compute crate for writing GPU code in Rust: "Slang with Rust
syntax and rustc checking". It covers as much Rust syntax as can be lowered with a
written contract, and brings GPU concepts (vector types, SIMT dispatch, workgroup
memory, capability-specific code) into Rust. The target hardware and platform are
chosen at build time; GPU and CPU builds follow one semantics contract (D21, D22).
Kernels may also run in non-compute GPU stages as computation (execution shapes,
D61); GUST owns no rendering state.
GUST is the foundation for two later projects: Breeze, a rendering and interaction
framework, and Hurricane, the game engine (destination in
[ENGINE_HURRICANE](ENGINE_HURRICANE.md)). Their scope is defined when GUST is ready.

## Now and next

**Now: M7, GPU concepts in Rust** (plan adopted 2026-11-02, owner-delegated). Its runtime prerequisites are done
(m7.a0, merge `bcb4da4`: device options, adapter limits, fallible buffers, the buffer pool, `GpuVec`, `MapHost`).
Done: m7.w (wrapping arithmetic and GUST-E0325, D87, merge `4797013`) and m7.r1 (reduce v0 in the new crate
`gust-algo`, merge `b5d2c68`), m6.z2 (GUST-E0326 for an unsuffixed literal `let`, D96, merge `a496968`) and
m7.rb (blocking readbacks wait for their own completion, D97, merge `893d959`), m7.t13 (T13 active count, merge
`b88bfb0`) and m7.r2 (scan v0, merge `09203ae`). Running: m7.a1 (the map API). Next: m7.r3 (gust-algo's GPU tests
without their mutex). The
wasm32 `Job` contract (D69) lands before m7.a's `map_async` reaches the browser. The MX plan is adopted (D88 to
D95).

**Done: M6, language breadth I** (closed 2026-11-02, D86). Plan adopted 2026-11-01 (owner-delegated) after an Opus
review, one revision and a re-check; decisions D77 to D85. Sanity check by Fable and by Opus in the Codex slot (owner
decision, Codex quota out; both "proceed with fixes"); the fixes are slice m6.z1 (merge `c8799b3`) and m6.z2, which
runs in M7 before m7.a1. Codex confirms when its quota returns.

| Item | Work | Status |
| --- | --- | --- |
| S6.3b, S6.1 | Spikes: the keyword sweep; vec3 and the matrix layout defaults (V1) | done (Slangc probes) |
| m6.a2 | Pure-Rust math types | done (Unit, `56b5ecd`) |
| m6.a1 | Layout model, `GpuPod`, schema-2 reader | done (Unit, Reflect, `b60fca2`) |
| m6.k | Collision-only renaming (D81) | done (Unit, Slangc, Reflect, `ce17eb2`) |
| m6.r | Reflection schema 2 in the helper (D82) | done (Reflect, byte-identical artifacts, `fe42f3f`) |
| m6.m1, m6.m2 | Matrix products and `__gust_mul` (D79) | done (Unit, Slangc, `6dc36ef`, `71ab804`) |
| m6.g1 | GPU evidence for vectors and matrices (D78) | done (Gpu RTX 5090, WGSL; Spirv validation, `5033b22`) |
| m6.n | Matrix negation | done (Gpu, `86f728c`) |
| m6.u1, m6.u2 | Uniforms: host side, macro side, typed calls and twins (D80, D85) | done (Unit, Slangc, `8d88b6d`, `1b592a1`) |
| m6.g2 | Uniform kernels on the GPU | done (Gpu, `475e2b1`) |
| m6.dv | `/` and `%` through `__gust_div`/`__gust_rem` (D77) | done (Gpu, Slangc, `94baab5`) |
| m6.d | Difftest vectors and matrices (`DIFFTEST_FORMAT` 6) | done (Gpu, `f452a37`) |
| m6.x | M6 wiring, goldens in `check-artifacts`, docs | done (Full, `8f069fb`) |
| m6.z1 | Sanity-check record fixes, D78 to D86 | done (docs, `c8799b3`) |
| m6.z2 | Literal-local typing (D86) | planned (in M7, after m7.w, before m7.a1) |

**Done: M5, CPU target and evidence breadth** (closed 2026-11-02, D76). Plan adopted 2026-11-01 (owner-delegated)
after a Codex review (adopt with changes) and one revision; decisions D51 and D52. Sanity check by Fable and Codex
(both "proceed with fixes"); the fixes are slice m5.f1, and the first green nightly `gpu-software` list is run
41442659698.

| Item | Work | Status |
| --- | --- | --- |
| S5.1 | Spike: the Gpu suite on Mesa lavapipe in WSL | done (decision A: 110/110 on llvmpipe 25.2.8, a WSL measurement) |
| m5.b0 | Probe: the ledger on lavapipe, debug and release | done (passes on lavapipe in both profiles; 20 of 132 rows differ from the RTX 5090) |
| m5.a | `gpu-software` CI job: Mesa pinned through the Ubuntu snapshot, adapter record asserted (D52) | done (`00ad885`) |
| m5.d1, m5.d2, m5.d3 | Scalar `coretests` cases: fixture and twin; on the GPU; golden and registration | m5.d1 done (`d2378ce`); d2 done (`ce17199`; 84 cases on the RTX 5090 and lavapipe); d3 done (`e249ae5`, merge `d879f33`) |
| m5.c1a, m5.c1a2, m5.c1b, m5.c1c | Differential generator in `tests/difftest/`: integers, floats v1 (interval bounds), harness, bounded shrinking | m5.c1a done (`3bcebaf`); c1a2 done (`f60f516`); c1b done (harness, S5.3 outcome A; Gpu: RTX 5090; `a3a7798`); c1c done (Gpu: RTX 5090; `a41ba93`) |
| S5.4 | Spike: expansion-time profile and cached rebuild transitions | done (decision A: both tracking paths suffice on their own) |
| m5.b | Ledger on two adapters (per-kind observation pins) | done (`236780b`; 18 software pins) |
| m5.e1, m5.e2 | Production `cpu` profile (D51); `check-cpu-target` and the `Cpu` tier | m5.e1 done (Unit, CPU only; Opus keep after one fix round; `f60c1b4`); m5.e2 done (Cpu: `check-cpu-target` E-a to E-i in the close gates and CI's `cpu-target` job; merge `f9fdd24`) |
| m5.c2, m5.a2 | `check-difftest` wiring; `gpu-software` on the Slang manager | m5.a2 done (Gpu (software): lavapipe; Opus merge after one fix round; `e098f2f`); m5.c2 done (`82c59d2`, fix rounds `f2a9410`, merge `68c6f40`; D74) |
| m5.f1 | Sanity-check fixes: the release ledger guarded by name in Full and in the new `check-ledger` (the nightly step), the Gpu pin in `test --suite Gpu`, `--locked` on three CI steps, apt retries for the Mesa snapshot | done (Gpu: RTX 5090; Opus merge after one fix round; merge `7dd9c3d`) |
| M5 close | `check-full` on the RTX 5090; sanity check by Fable and Codex; D76 | done (D76; Fable and Codex: proceed with fixes; `check-full` at `7dd9c3d` after m5.f1, clean tree) |

**Done: M4, build-time targets and distribution** (closed 2026-11-02, D73). Plan adopted 2026-11-01 (owner-delegated) after a Codex review
(rework), two Opus re-reviews and three revisions; decisions D44 to D47. S1, m4.0 and m4.h run in parallel on disjoint
files; everything after S2 runs on one lane. Stop rules: a kill rule stops M4, an undiagnosed read-back mismatch stops
its slice, and two fix rounds per slice.

| Item | Work | Status |
| --- | --- | --- |
| M4 plan | Plan with pre-registered spikes; Fable plan, Codex review (rework), Opus re-reviews in the Codex slot, three revisions | done (D44 to D47) |
| S1 | Spike: one precompiled, reflection-verified kernel (vector-add) runs without Slang installed and keeps D17's identity (kill: identity cannot be kept without shipping Slang, or the artifact path needs a custom IR) | done (decision A, Gpu: RTX 5090 with no Slang reachable; local spike commit, not merged) |
| m4.0 | Weak compare-exchange: uncontended success set-valued on the GPU, exact on the twin (D47) | done (Gpu: RTX 5090, debug and release; Opus keep after one fix round; `8d49412`) |
| m4.h | M3 follow-ups: `check-artifacts` clears stale generated output first; the adapter name in `VALIDATION.json` | done (Gpu: RTX 5090, `check-full` and `check-artifacts`; Opus keep after one fix round; `b26bccd`) |
| S2 | Spike: the `GUST_PROFILE` mechanism composes with `check-dual` and Cargo's rebuild tracking (D44) | done (decision A; local spike commit, not merged; `check-dual` must set `GUST_PROFILE=dev` for the dual fixture) |
| m4.a | `gust::artifact`: `Artifact`, `verify_artifact`, `content_key`, `KernelDescriptor::artifacts` | done (Unit; Opus keep after one fix round; `2dcb9e3`) |
| m4.b1, m4.b2 | `slang-runtime` feature, off by default; build profiles in `gust`'s build script, strict-profile refusals | m4.b1 done (Unit + Gpu; Opus keep after one fix round; `cbbf35a`); m4.b2 done (Unit + Gpu; Opus keep after one fix round; `41b6ff9`) |
| m4.c | Macro: content key, artifact and manifest embedding, export mode | done (`dba0e97`) |
| m4.d | `cargo-gust`: `init`, `build` (isolated collection, D17 at export for WGSL and SPIR-V), `check`, `doctor` (D46) | done (`e7dd73d`) |
| m4.e1, m4.e2 | `gust-wgpu` precompiled path and cache key; precompiled fixture, `check-precompiled`, packaged-consumer proof without Slang | m4.e1 done (Gpu: RTX 5090; Opus keep after one fix round; `894ab0e`); m4.e2a done (fixture, `check-precompiled` E1 to E4; Gpu: RTX 5090 with no Slang in the environment; Opus keep after one fix round; `29cfc58`); m4.e2b done (E5 to E8: strict plus dual, build-script refusals, re-expansion, `cargo package` allow-lists and a relocated registry-style consumer; Gpu: RTX 5090; CI `precompiled` job runs `--check-only` plus a NoAdapter run; Opus keep after one fix round; `3c95f8a`) |
| m4.f | Pinned-Slang manager with licence audit; CI `slang` job | licence audit done (all three platforms accepted, [LICENSES-SLANG](development/LICENSES-SLANG.md)); manager done (`cargo gust slang` install, locate and build-helper; Opus keep after one fix round; `9b42958`); CI `slang` job green on Linux (`9a5c529`; Reflect tier left to gpu-software) |
| m4.g1, m4.g2 | Completion model (async read-back); wasm32 and the first browser WebGPU run | m4.g1 done (Gpu: RTX 5090; moved ahead of m4.e1; Opus keep after one fix round; `671b3dd`); m4.g2 done (Gpu: WebGPU in Chrome 156 on the RTX 5090; Opus keep after two fix rounds; `4e096f5`) |
| m4.r | Slang pin bump to v2026.20 (fixes the WGSL template-list bug, shader-slang/slang 13566) and re-bless of every committed set (owner, 2026-11-01) | done (Gpu: RTX 5090; Slangc, Spirv, Reflect on 2026.20; Opus keep after one fix round; `12ab066`) |
| m4.z | Repository cleanup before the M4 sanity check: dead code, the `__gust_keep` workaround if m4.r makes it redundant, STATUS consolidation, stale docs (owner, 2026-11-01) | done (CPU gates; Opus merge after two review rounds; `6354747`, merge `c964222`) |
| M4 close | `check-full` on the RTX 5090; close rows D68 to D72; sanity check by Fable and Codex | done (D73; Fable and Codex: proceed with fixes; `check-full` at `27b342b` after m4.i) |
| m4.i | Sanity-check fixes: `CXX=cl` selects MSVC flags; the consumer tells NoAdapter from other errors; `check-full` runs `check-precompiled`, `check-web` and `check-artifacts` (plan §14 item 3); stale prose | done (Gpu: RTX 5090 and Gpu (WebGPU, Chrome 156.0.8078.74) through `check-full`; Opus merge after one fix round; merge `27b342b`) |

**Done: M3, dual test builds** (closed 2026-11-01, D43). Plan adopted 2026-10-24 (owner-delegated); slices ran one at a
time. The M2 sanity check (D37) kept M3's scope and fixed the order inside it.

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
| m3.e1, m3.e2 | Remaining examples: per-kernel twins or explicit opt-outs (graph examples use read-back intermediates) | m3.e1 done (Gpu: RTX 5090; Opus keep after one fix round; `41b4d0e`); m3.e2 done (Gpu: RTX 5090; Opus keep after one fix round; `00127a4`) |
| m3.f1, m3.f2 | `--cfg gust_dual`: a cross-crate fixture proving the twin is absent from production builds, `check-dual`, check-cfg lints, example mains, CI step | m3.f1 done (Unit, CPU only; Opus keep after one fix round; `5c5a9ad`); m3.f2 done (Unit + Gpu: RTX 5090; Opus keep after one fix round; `98acabc`) |
| m3.g | Gates that cannot pass by skipping (plan Revision 7): examples with `--include-ignored` in `check-full`, a test-count guard, release-only tests required by name, NaN-propagating error folds, pinned check-ui fixtures, `gust::cpu` unit tests in the Unit suite | done (Unit; Opus keep after two fix rounds; `95c5514`); `check-full` PASS at `96e4b9d` (clean tree, RTX 5090) |
| M3 sanity check | Fable and Codex, read-only, on `69b2bf3..1380913` | done (both "proceed with fixes"; D43) |
| m3.h | Sanity fixes: KM-11 store rows reach the store, cross-crate overflow proof in `check-dual`, `check-cpu` requires twin tests by name, documentation drift | done (Unit + Gpu: RTX 5090; Opus keep after one fix round; `a169bec`); `check-full` PASS at `5971c12` (clean tree, RTX 5090) |

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
| M3 Dual test builds | Executable shadow types under the semantics contract (D22); CPU dispatch; CPU twins emitted under `cfg(test)`; `--cfg gust_dual` for cross-crate tests and benchmarks; per-kernel opt-out (`#[kernel(cpu = "unsupported")]`); existing GPU tests gain CPU-versus-GPU checks while semantics fixtures keep independent host references (D26). From M2 (D34): the CPU backend implements `TypedDevice` once and `Run<K, N>` per kernel, keeps the alias contract, and `#[kernel(cpu = "unsupported")]` is the absence of the `Run` impl | done (D43) |
| M4 Build-time targets and distribution | Build profiles (target plus capability set) chosen at build time; hash-keyed artifact manifest; D17 at export time for WGSL and SPIR-V; runtime `slangc` behind a development feature; a pinned-Slang manager for crate users; first browser WebGPU run. Note: first spike: choose the build-profile mechanism (build.rs env, `--config`, or a cfg namespace) so it composes with check-dual's `CARGO_ENCODED_RUSTFLAGS` and its refusal; the artifact manifest hashes Slang output only (the twin never enters it or the published crate) | done (closed 2026-11-02, D73; plan D44 to D47; close rows D68 to D72). Capability sets are deferred to M8 (plan); D17 holds at export for describable layouts, and `layout: null` types are exported unchecked and checked at run time (D71) |
| M5 CPU target and evidence breadth | Production `cpu` build target and runtime; lavapipe CI job labelled `Gpu (software)`; differential program generator with shrinking; scalar `coretests` cases on the GPU. Note: the known-mismatch ledger and the semantics contract exist since M3 and are not new M5 work; M5 adds the production target, lavapipe and the generator | done (D76; closed 2026-11-02; sanity fixes m5.f1; `check-full` at `7dd9c3d`) |
| M6 Language breadth I | Vectors and matrices (vec3 and mat3 rejected in buffers at first; `m * v` lowers to `mul`); collision-only identifier renaming; uniforms (`ConstantBuffer<T>`). Note: uniform typed calls were decided in M3 (D40): such kernels have no typed call and no CPU twin until this milestone | done (D86; closed 2026-11-02) |
| M7 GPU concepts in Rust | First, m7.a: element-wise calls from a plain scalar `#[gpu] fn` (`name::map` over slices, GPU-resident `GpuVec` values, `map_async` for the browser), owner direction 2026-11-01, D60, spikes m7.s1 and m7.s2 first; `par_for!` reuses its kernel generator. Then `par_for!` SIMT dispatch on both pipelines; workgroup shared memory and barriers, including a CPU strategy; reduce, scan and compaction primitives (T13 lands here). Multi-dimensional dispatch lifts the `max_compute_workgroups_per_dimension` ceiling (D36). Note: `#[kernel(cpu = "unsupported")]` is the opt-out for barrier kernels until a CPU strategy exists; a validator rule rejects a construct with cross-invocation meaning unless the kernel carries `cpu = "unsupported"` or a declared twin strategy (D42) | now; plan adopted 2026-11-02 (owner-delegated): Codex review (rework), Revision 1, an Opus re-review in the Codex slot (adopt with changes), R1.1; spikes m7.s1, m7.s2 and m7.s3 (CUDA comparison, owner request) done; m7.a0 merged (`bcb4da4`). The wasm32 `Job` contract (D69) lands before m7.a's `map_async` reaches the browser (M4 sanity check) |
| MX Execution shapes and textures | After M7, before MR (D75). Shapes beyond `grid` (D61 to D63) and textures (D64), in the order of the planning notes: P0, the compute path behind the stage-profile interface of D63 (gate: byte-identical Slang for every fixture); textures as a GUST resource (storage textures for `grid`, render-target outputs for `pixel`, sampled textures with samplers; reflection, D17 and the twin's texture carrier); P1, the `pixel` shape (the same kernel as `grid` and `pixel`, bit-identical on the exact formats of D62, a cross-shape difftest, a per-shape performance test and the attachment bound in the validator; kill rule: `pixel` forces special cases into the emitter core beyond the profile hooks); the shape classifier and the shape pin of D63. Spike X-S1 (outcome A) measured the premise on the RTX 5090 and the 890M. Not in MX: `vertex` and `raster_fragment` (P2, with Breeze's first milestone) and shape inference with routing (P3, with M8's routing) | plan adopted 2026-11-03 (owner-delegated; Fable R1 and R1.1 after a Codex rework and an Opus re-review, adopt with changes; decisions D88 to D95; a Codex confirming pass when its quota returns). Spikes X-S2 and X-S3 and slice mx.t0a may run beside M7. Done beside M7: X-S2 (outcome A) and mx.t0a (helper schema 3, merge `754e439`); running: mx.n |
| MR First public release | After MX (D75), so publishing does not wait for M8 to M10. Crate names per D48 (`gust-compute`, `gust-compute-macros`, `gust-compute-wgpu`, `cargo-gust`; the macro resolves the imported crate path instead of `::gust::`); the published surface and its pre-1.0 stability promise; README, crate docs and a getting-started path that work from crates.io (the Slang pin and the reflection helper through `cargo-gust`); licence metadata; `cargo publish --dry-run` and a docs.rs-style build for every crate in dependency order, and a consumer built from the packaged crates. The coordinator prepares the release; the owner publishes to crates.io | planned |
| M8 Capability variants | Capability-keyed variants with platform aliases; exhaustiveness checked against the crate's declared target list, with CPU as a target; one rustc-checked signature; subgroup operations first (D23). Compile-time portable floor for workgroup size and storage counts; `with_limits` is the runtime half (D36). Build-time routing per function tag (D65 to D67) lands alongside, after m7.a. Precondition (M5 sanity check): decide how D65's CPU bodies in `dev`, `native` and `web` production builds meet D26 (the `typed-call-no-cpu-twin` fixture, check-dual E1 and E3) and the macro's one process-wide profile read | planned |
| M9 Language breadth II | Methods in `impl` blocks; user enums with `match`; tuples; fixed-size arrays; non-capturing closures; then traits mapped to Slang interfaces and generics | planned |
| M10 Publication | Benchmarks against rayon, hand-written WGSL with wgpu, CubeCL and rust-gpu; a "Rust program to GPU" example gallery; docs and a book; the API stability pass toward 1.0; the crates.io releases after MR's first | planned |

Order after M5 (owner, 2026-11-02, D75): M6, M7, MX (execution shapes and textures), MR (the first public
crates.io release), then M8, M9 and M10. MX and MR keep letters so that the decisions naming M8 to M10 stay
valid.

## Parking lot

- GPU builds with a CPU fallback (D21 leaves this open as a future option).
- WebGL2 (out of scope for GUST compute, D24; revisit for Breeze).
- Slang automatic differentiation.
- An optional nightly-pinned THIR checker behind a cargo subcommand.
- Per-API SPIR-V, MSL and DXIL builds for native-only capabilities.
- Breeze and Hurricane scope.
- Validator rejections for recursion. Found by the M1 plan. (Rust identifiers that are Slang keywords: done in M6 by m6.k's collision-only renaming, D81.)
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
