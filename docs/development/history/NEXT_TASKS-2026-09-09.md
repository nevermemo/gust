# Next tasks

Claim one bounded task in STATUS before editing. Verify repository state; this list
is not authority to contact external agents or change account settings.

## Repository structure pass — operational docs and examples — COMPLETE

Status: complete (2026-09-08). Human-readable operational markdown now lives under
`docs/development/`; `.ai/` keeps generated/machine-readable state only. Added
development docs for repository structure, validation strategy, and agent workflow.
All examples now use a thin `src/main.rs` plus `src/app.rs` implementation module.
Follow-up (not claimed): split large example `app.rs` files further into `gpu.rs`,
`cpu.rs`, `host.rs`, and `tests.rs`. Generated shader/host outputs now stay in the
ignored local `generated-wgpu/` directory.

## DX — test-speed split — COMPLETE

Status: complete (2026-09-08). Goal: make the inner loop purposeful and faster
without deleting coverage. Regular `cargo test --workspace` skips example crate tests
(they are marked ignored with the shared reason "example validation runs only in full
verification"). `scripts/verify.ps1 -Full` runs ignored example tests explicitly,
then runs each example binary and validates exported SPIR-V. Regular per-feature
compile-smoke tests use WGSL only; SPIR-V validation is centralized in full
verification. Verify routine work with the owning focused test plus normal workspace;
use `cargo xtask check-full` for major/release evidence. The original PowerShell
script mentioned here was superseded by the xtask slice below.

## DX — command surface and staged verification — COMPLETE

Status: complete (2026-09-08). Goal: remove command-choice ambiguity for humans and
agents. This was an intermediate PowerShell implementation and is now superseded by
the cross-platform `cargo xtask` slice below. The durable policy remains: do not
introduce shared `HeadlessDevice` fixtures until `cargo xtask measure-tests` shows GPU
adapter initialization, not shader compilation or GPU work, is the bottleneck. Verify
routine work with `cargo xtask check-feature <area>` or `cargo xtask check-fast`; use
`cargo xtask check-full` for major/release evidence.

## DX — cross-platform xtask command surface — COMPLETE

Status: complete (2026-09-08). Goal: remove PowerShell as a project requirement and
make validation work across Windows, Linux, and macOS through the Rust toolchain
already required by the project. Delivered `cargo xtask` commands for feature checks,
fast/full verification modes, examples/artifacts/export, native reflection helper
build, measurement, and hook formatting. Deleted `.ps1` command scripts; kept
`scripts/probes/*.cpp|*.slang` as source fixtures. Verify routine work with `cargo
xtask check-feature <area>` and `cargo xtask check-fast`; use `cargo xtask check-full`
for major/release evidence.

## DX — xtask automation batch — COMPLETE

Status: complete (2026-09-08). Delivered after the xtask conversion: `--record` /
`--no-record` verification policy, `check-workspace` with examples excluded,
`check-changed`, `status`, `doctor`, `check-format`, `check-lints`, `list-tests`,
`explain-check`, split GPU groups (`gpu-smoke`, `gpu-semantics`, `gpu-runtime`), and
`measure-tests --json`. Skipped by owner request: item 9 (`status --update`) and item
13 (`GUST_EXAMPLE_SIZE`). Shared `HeadlessDevice` fixtures remain evidence-gated;
measure first and keep cache-stat tests isolated.

## T01 — P0: Translation regression foundation

Status: first slice complete. Goal: preserve helper returns, literals, grouping, and
negation. Why: reproduced wrong/invalid Slang from accepted Rust. Files:
`crates/gust-macros/src/{slang,tests/regression,validate}.rs`,
`tests/fixtures/semantics.*`, `crates/gust-wgpu/tests/semantics.rs`.
Dependencies: installed Slang and Vulkan adapter for integration.
Done: failing-before tests, reviewed golden, both target compilers, real GPU results
at 1/63/64/65/257 elements. Verify: `cargo test --workspace`.

## T02 — P0: Runtime cache and compiler bridge hardening

Status: complete. Goal: correct entrypoint identity and owned temporary-file cleanup.
Why: inspection found cache key omits entrypoint; predictable directory reuse and
early error paths weaken compiler isolation. Files: wgpu `src/lib.rs`, core
`src/slang.rs`, targeted tests. Dependencies: none beyond baseline tools.
Done: regression proves two entries in one source cannot share the wrong pipeline;
exclusive temp creation, cleanup on failures, bounded artifact validation.
Verify: targeted core/wgpu tests, then `scripts/verify.ps1 -Full`.

## T03 — P0: Fail closed at unsupported frontend boundaries — COMPLETE

Status: complete (both slices, 2026-09-07). Goal: actionable errors for invalid
signatures, options, attributes, and resource helpers. Why: syntactic acceptance
currently exceeds proven lowering.
Files: macro validate/expand/slang and negative fixtures. Dependencies: T01.
Slice 1 (done): reject duplicate/missing invocation, non-unit kernels, zero/duplicate
workgroups, semantic attributes that are dropped, malformed builtin receivers/arguments;
compile-fail evidence.
Slice 2 (done this pass): broader name resolution and effect analysis.
`visit_expr_path` rejects multi-segment bare path values (associated constants / foreign
items like `f32::INFINITY`, `u32::MAX`, `core::f32::consts::PI`); `core` added to
`BANNED_NAMES`; `visit_expr_try` rejects `?`; `visit_expr_unsafe` rejects `unsafe`
blocks. `asm!` already covered (syn 2.0 parses it as `Expr::Macro`). Failing-first
tests: `unsupported_name_paths_are_rejected` (4 cases) and
`unsupported_effects_are_rejected` (3 cases).
Slice 3 (done 2026-09-07): type-name frontier. `visit_expr_cast` limits
`as` targets to `f32`/`float`, `i32`/`int`, `u32`/`uint`; `visit_type_path` rejects
Rust primitives outside the four-byte subset in every type position (`let v: f64`,
`fn helper(x: u64)`, `-> usize`, struct fields, `RWStructuredBuffer<i64>`);
`visit_expr_const` rejects inline `const { .. }` blocks by name. Failing-first tests:
`casts_to_unproven_targets_are_rejected`, `non_32_bit_primitive_types_are_rejected_before_slang`,
`const_blocks_are_rejected`; one `compile_fail` doctest in core `lib.rs`.
Remaining frontier (not claimed): std-prelude type names are not banned, so
`fn maybe() -> Option<uint> { None }` passes the validator and rustc and emits
`Option<uint> maybe()`, failing only in slangc (verified by probe 2026-09-07). A
fix needs an allowlist of type names (prelude + module structs) per the validator
contract, plus a test-order update for `unsupported_effects_are_rejected`.
Update (slice 4, 2026-09-07): the owner chose lowering over banning. `Option<T>` now
lowers to Slang `Optional<T>` (see D14 and ARCHITECTURE "Translation stability");
the probe above is now a supported program. Still open: other std-prelude names in
helper *signatures* (e.g. `fn f(r: Result<uint, uint>)`) pass the validator and
fail only in slangc; they cannot be constructed (`Ok`/`Err` calls are rejected), so
the leak is limited to signatures. A type-name allowlist remains the fix if wanted.
Verify: `cargo test -p gust-macros` and `cargo test --workspace`.

## T04 — P1: Resolved numeric semantics and struct construction — COMPLETE

Status: complete. Numeric semantics proven and a literal-zero div/rem diagnostic
added; field-aware struct construction implemented in bounded contexts.
Goal: explicit typed locals plus field-aware construction with preserved
evaluation order, or clear diagnostics. Why: syn inference is not rustc inference;
integer division/overflow/casts and effect order remain correctness risks.
Files: macro translator/shadows plus fixtures. Dependencies: T01/T03.
Done: signed/unsigned edge tests, inference counterexamples, reordered struct fields,
and side-effect-order checks; no public claim beyond proven cases.
Verify: golden, compile-fail, Slang WGSL/SPV, GPU readback tests.

Evidence (2026-09-07, slangc 2026.13.1-1-g84792eb15, Vulkan adapter present):
- `numeric` fixture golden (`tests/fixtures/numeric.slang`) locks the emitted Slang.
- `numeric` GPU test proves signed/unsigned div/mod, float→int casts, and struct
  field identity + source order against an independent host reference.
- Integer div/rem by a literal zero is rejected at the Rust boundary (Slang makes it
  a compile error; rustc treats it as a runtime panic).
- Struct literals lower to construct-then-assign by name in source order; supported
  as a `let` initializer and assignment RHS, rejected elsewhere.

## T05 — P1: Capability/ABI evidence and diagnostic source mapping — COMPLETE

Quality-review follow-up (2026-09-07): real compiler errors exposed the current
multiline diagnostic format, now covered alongside legacy line/column locations.
T04 follow-up builds struct RHS values in fresh temporaries before writing the
destination and explicitly rejects record updates (`..base`). GPU regressions
cover local/buffer swaps and shadowing at workgroup boundaries. Final independent
review is outstanding after worker allowance failures; see STATUS.

Status: complete (2026-09-07). Both slices done: capability probe records and
kernel-level Slang→Rust diagnostic mapping.
Goal: machine-readable probe records and useful Rust locations for Slang errors.
Why: target claims and layouts must follow evidence. Files:
`crates/gust/src/slang/mod.rs`, `crates/gust-macros/src/slang/mod.rs`,
`tests/fixtures/{numeric,semantics}.slang`, docs/PORTABLE_SLANG_CORE.md.
Dependencies: T02/T03.
Slice 1: `TargetProbe` record + `probe` in `crates/gust/src/slang/mod.rs`
compiles a known-good minimal kernel to a target and records `supported`/`detail`;
failing-first tests (WGSL + SPIR-V); PORTABLE_SLANG_CORE.md documents the record
format.
Slice 2: `emit_kernel` writes a `// @rust kernel: {module}::{kernel}` marker before
each kernel; on `slangc` failure the core bridge parses the reported line and
appends `originating Rust kernel: {module}::{kernel}` (nearest marker at or before
that line) to the `CompilationFailed` diagnostic. Kernel-level, not line-level:
stable-Rust `proc_macro` spans do not expose line numbers. Golden fixtures updated
with the marker; two tests cover mapping with a marker and `None` without.
Evidence (2026-09-07): `cargo test -p gust --lib` 12 passed (incl. both
mapping tests); `cargo test -p gust-macros` 20 passed (marker goldens);
`cargo test -p gust-wgpu --test semantics` 2 passed on RTX 5090.
Verify: repeat probes and intentional failing input; no silent skips.

## T06 — P1: Slang reflection / broader shadows — COMPLETE

Status: complete (2026-09-08, D17). Goal: compiler-authoritative layouts before
uniforms/textures/samplers/vectors expand the runtime ABI. Why: hand-classified
metadata was only a bootstrap contract.
Slice 1 (2026-09-07): core JSON parser, `reflect`, explicit `cross_check_pod` with
positive/negative comparisons. The CLI JSON omits aggregate size/alignment/stride, so
that path reports explicitly partial coverage and stays as inspectable evidence only.
Slice 2 (2026-09-08): native helper `gust-slang-reflect` (`scripts/probes/
slang-layout.cpp`, `scripts/build-slang-reflect.ps1`) compiles and reflects one
linked program through the Slang API and emits complete StorageV1 layouts or fails;
`reflect::compile_reflected` + `Reflection::cross_check_kernel` verify names, slots,
access, workgroup size and every nested offset/size/alignment/stride; the wgpu
runtime gates every pipeline-cache miss on that check and executes the helper's WGSL.
Tests: core 5→9, new GPU `crates/gust-wgpu/tests/reflection.rs` (4). Full
`verify.ps1 -Full` passed; it now builds the helper first.
Not done (by design): textures, samplers, uniforms, vectors/matrices, multiple
binding groups, specialization — each needs its own reflected evidence before it
enters the runtime contract. Follow-ups: exported `.rs` host snippets still show the
ungated `compile_wgsl` path; consider a content-hash `KernelCacheKey` if descriptors
ever stop being `&'static`.
Verify: `cargo test -p gust --test reflection` and
`cargo test -p gust-wgpu --test reflection`.

## T07 — P2: First explicit staged graph proof — COMPLETE

Status: complete (2026-09-07). Goal: CPU settings → two GPU stages → small summary,
with inspectable dependencies and transfer sizes. Why: bridge current batches toward
GUST execution.
Done: `gust_wgpu::StagedGraph` (`crates/gust-wgpu/src/graph.rs`) with
upload/dispatch/readback nodes, host-declared dependencies validated against actual
buffer hazards (`GraphDependencyOrder`, `GraphMissingDependency`), uploads encoded as
ordered copies, `GraphReport` with per-transfer bytes, dispatch/workgroup counts, and
resident bytes. `examples/staged-graph`: settings upload → `transform` → `summarize`
(8:1 reduction via a struct-returning helper) → summary readback; tests cover CPU
parity for N in {0,1,7,8,9,63,64,65,257,1000}, exact transfer/resident byte counts,
ordering across two settings updates, and seven rejection cases. Runtime addition:
`BufferBinding::independent_length()` (opt-in; `IndependentLengthEmpty` error).
`verify.ps1` now runs six examples and expects ten SPIR-V exports.
Not done (by design, see D15): dependency inference, reordering, transfer planning,
CPU nodes, continuations. Verify: `cargo test -p staged-graph` and the workspace.

## T08 — P2: Engine component pool / indirect workload proofs — COMPLETE

Status: complete (2026-09-07). Goal: GPU data survives host-driven capacity growth
without resize readback, then active-count-driven indirect dispatch. Why: concrete ECS
prerequisites. Contract specified first in `docs/ENGINE_HURRICANE.md` (D16).
Done: `gust_wgpu::GpuPool<T>` (`crates/gust-wgpu/src/pool.rs`):
capacity, host-tracked logical length mirrored to a one-element count buffer,
`pool_push`/`pool_reserve` growth by GPU→GPU copy of the live prefix with
`GrowthRecord` evidence, retirement list drained by `pool_reclaim`, `pool_truncate`,
`pool_read` (live prefix). `create_indirect_buffer` (three-`u32` layout, `INDIRECT`
usage, tagged) and `StagedGraph::dispatch_indirect` (args tag/layout checked, every
binding independent-length and non-empty, args counted as a read for hazards,
`GraphReport::indirect_dispatches`). `examples/component-pool`: `prepare_dispatch`
derives the active count and overflow-free workgroup args on the GPU clamped to the
allocation and a budget; `integrate` runs indirectly and guards on both. Tests: both
targets; GPU-authored state survives 4→8→70 growth (partial second workgroup
through the indirect path) and reclaim returns 2; budget clamp, truncate, zero
length, push after truncate; growth while a submission is in flight preserves its
writes; five rejection cases (plain buffer as args, two wrong layouts, strict
binding, undeclared prepare→integrate edge). `verify.ps1` runs seven examples and
expects twelve SPIR-V exports.
Not done (by design): freelists, compaction, entity IDs/generations, atomics, GPU-side
count generation beyond clamping (the dialect has no atomics or loops), culling,
rendering. Open follow-ups: a growth benchmark separating allocate/copy/submit, and
multi-buffer pools (SoA) once a second engine user needs them.

## T09 — P1: bounded loops in the dialect — COMPLETE

Status: complete (2026-09-08, D18). Owner pre-authorized (2026-09-08) as the first
of the ordered compiler extensions. Delivered: `for i in start..end` with the end
bound evaluated once into a reserved temporary, immutable fresh loop variables, `_`
counter naming, unlabeled valueless `break`/`continue`, and `return` from inside loop
bodies. Rejected: `while`, `loop`, `..=`, open ranges, non-range iterables,
tuple/`mut`/`ref` patterns, labels, value-position jumps, unsuffixed literal bounds,
and resource/dispatch-ID shadowing. Evidence: reviewed D18 contract,
`tests/fixtures/loops.{rs,slang}`, macro regression tests, external loop-specific
`spirv-val`, and a real-GPU differential test at 1/63/64/65/257. Full `verify.ps1
-Full` passed with no generated-wgpu drift.
Not done (by design): `while` with budgets, inclusive ranges, iterator adaptors,
runtime trip-count analysis, and validator-side proof of non-literal range operand
types (rustc shadows and Slang own those today).
Verify: `cargo test -p gust-macros loops` and `cargo test -p gust-wgpu
--test loops`.

## T10 — P2: atomics on `RWStructuredBuffer<u32|i32>` — COMPLETE

Status: complete (review fixes, 2026-09-09). Delivered: statement-only `atomic_add`,
`atomic_min`, `atomic_max`, `atomic_exchange`, and `atomic_compare_exchange` on
`RWStructuredBuffer<u32>`; `atomic_add`, `atomic_exchange`, and
`atomic_compare_exchange` on `RWStructuredBuffer<i32>`; signed min/max remain
rejected. The emitter discovers atomic buffers per emitted kernel, so sibling kernels
sharing a parameter name retain their own element type. Mixed ordinary indexed
read/write access to an atomic buffer is rejected until explicit atomic load/store
semantics are specified. Evidence: 17 macro regressions, 8 GPU tests including exact
257-thread add contention, deterministic CAS success/failure and signed execution;
the registered `atomic-counter` example dispatches one-thread workgroups through an
independent-length buffer and exports the 13th validated SPIR-V artifact. Full
verification, including reflection, all examples, and `spirv-val`, passed on the RTX
5090/Vulkan adapter. Verify: `cargo test -p gust-macros -- atomic`, `cargo test -p
gust-wgpu --test atomics`, and `cargo xtask check-full`.

## T12 — P2: statement `match Result<T, T>` lowering — COMPLETE

Status: complete (2026-09-09). An exhaustive, semicolon-terminated Result match
now lowers as one evaluated `__gust_match_N` temporary plus an `if/else` over its
tag. Exactly one `Ok(identifier)` block arm and one `Err(identifier)` block arm are
required; arm order is irrelevant. Match expressions, missing/duplicate arms, guards,
arm attributes, non-block arms, wildcard/nested/ref/mut patterns, Option/scalar
matches, and `?` remain rejected. Evidence: match-shape and evaluation-once macro
regressions, the reviewed Result golden, WGSL and SPIR-V compilation with external
`spirv-val`, and real GPU differential results at 1/63/64/65/257 elements. Verify:
`cargo test -p gust-macros -- match` and `cargo test -p gust-wgpu --test result`.

golden, GPU test); anything not lowered gets a type-name allowlist diagnostic.
## T11 — P2: standard-prelude lowering — COMPLETE

Status: complete (2026-09-09). `Result<T, T>` now lowers deterministically as a
local/helper-only `__GustResult<T>` tagged value. `Ok(value)`, `Err(value)`,
`.is_ok()`, `.is_err()`, `.unwrap_or(fallback)`, and `if let Ok(value) = result`
are supported. The single shared payload type is required because Slang cannot infer
the absent error type from `Ok(value)`/`Err(value)` constructors. Mixed-payload and
nested Results, struct/resource Results, tuples, slices, `unwrap`/`expect`,
combinators, `match`, and `?` remain rejected. Evidence: reviewed `result` Slang
golden, three macro regressions, WGSL compilation, and real GPU differential results
at 1/63/64/65/257 elements. Verify: `cargo test -p gust-macros -- result` and
`cargo test -p gust-wgpu --test result`.
