# Next tasks

This is the active queue only. Completed narrative is archived in
[history/NEXT_TASKS-2026-09-09.md](history/NEXT_TASKS-2026-09-09.md); feature
contracts live in [tasks/](tasks/README.md).

Claim one bounded task in [STATUS.md](STATUS.md) before editing. A task is not
active until its owner, scope, and focused check appear there.

The ordered plan is [ROADMAP.md](../ROADMAP.md). Its "Now and next" table lists the
claimable items of the current milestone; this file keeps deferred work and task detail
that does not fit there.

## Deferred

### T13: GPU active count to indirect dispatch

Deferred by owner. The T13 work in progress is not in the working tree: it is the
git stash `WIP T13 active count deferred`, based on `59a5c3c`. Its code changes
(`examples/component-pool/src/app.rs`, `xtask/src/{checks,verify}.rs`) applied
cleanly to `63769fb` in a 2026-10-22 dry run; its doc renames are superseded by
that commit and conflict. Do not apply, continue, or drop the stash unless the owner
explicitly requests it. Contract: [tasks/T13-active-count.md](tasks/T13-active-count.md).

## Claimable DX

### DX20: Internal ownership splits

Split the largest modules only where a named responsibility can move without
changing public APIs: macro validation by concern, macro regressions by feature,
or wgpu runtime internals by ownership. Use a no-behavior-change contract and
one focused check per move. Start with extracting the Result/Result-match
regressions from `crates/gust-macros/src/tests/regression.rs` into a feature-owned
test module; do not mix that mechanical move with behavior changes.

## Later engine proofs

- GPU active count to indirect dispatch (T13), when the owner reactivates it.
- Culling -> compacted visibility -> indirect arguments -> rendering.
- Entity identity/lifecycle and storage choices, guided by measured workloads.

## M2 follow-ups (filed at M2 close, D37)

- O5: `max_buffer_size` is not checked; buffer creation is infallible, so an oversized
  buffer is still a wgpu panic. Take the error-scope route (`push_error_scope` /
  `on_uncaptured_error`) rather than another host pre-check, with a test that pins
  today's panic first. M10 API pass at the latest.
- O8: deprecate the `F32Binding`/`U32Binding` slice paths once typed calls cover every
  example's use (M10 API pass).
- O4: access typestate on `GpuBuffer` (`GpuBuffer<T, A>`) so a read-only buffer passed
  as read-write fails at compile time; changes every `create_*` signature and the pool.
- O3: a `#[gpu]` parameter attribute for independent length, if D15's call-site
  `Independent(&buf)` is revisited when M7's primitives need it.
- Kernels with a `Uniform<T>`/`ConstantBuffer<T>` parameter have no generated typed call,
  and the headless runtime rejects them (`Error::UnsupportedParameter`); plain value
  parameters are rejected by the macro. Resolved at M3 close (D40): uniform kernels stay
  without a typed call and without a CPU twin until M6's uniform work, which decides
  both together.

## M3 follow-ups (filed at M3 close)

- Explicit `wrapping_add`, `wrapping_sub`, `wrapping_mul` and `wrapping_neg` methods in
  the dialect, lowered to the plain Slang operator, only when a kernel needs an
  intentional wrap. Not a fix for the overflow edge cases of the ledger: D39 drops the
  wrapping rewrite of twin bodies that D38 had deferred to M5.
- A debug write-set race checker in `gust::cpu`: the twin runs invocations one after
  another, so a kernel in which two invocations write one location gives a clean twin
  result and proves nothing about the GPU (SEMANTICS section 5). The checker would
  record the written indices of a call in debug builds and report a repeated one.
- M5, CPU limits: decide whether the production CPU target emulates the device limits
  (workgroup count per dimension, invocations per workgroup, storage binding size) or
  keeps documenting the difference; today the CPU backend rejects only a launch above
  2^32 invocations.
- M5, difftest shape and CI: the form of the differential test (the same kernel source
  on the CPU target and the GPU, generated inputs with shrinking, comparison under the
  SEMANTICS domain and tolerances) and a lavapipe CI job labelled `Gpu (software)`. The
  ledger and the contract exist since M3; M5 adds the production target, lavapipe and
  the generator.
- M7, parallel twin execution: run the invocations of a twin on several threads once
  barriers and shared memory exist. A CPU strategy for them is part of M7; until then a
  barrier kernel uses `#[kernel(cpu = "unsupported")]` (SEMANTICS section 5).
- M8, capability-gated tests that fail rather than skip: a test that needs a capability
  the adapter lacks is a failure of its tier, never a skipped or ignored pass.
- M10, benchmark fairness: bounds checking and panic behaviour must be equal on the
  compared implementations (the GUST CPU target, rayon, CubeCL, rust-gpu) before a
  benchmark is reported.
- Test pinning by name (known limitation of m3.g's test-count guard): a counted call
  fails when it reports no passed test, but deleting one test, or marking it
  `#[ignore]` outside an `--include-ignored` call, while its siblings still run passes.
  check-dual's E2, the release call's two release-only tests and, since m3.h, the twin
  and derivation-domain tests of each check-cpu example (`CPU_REQUIRED_TESTS`) are required
  by name; the example calls of `check-examples` and `check-full` and the filtered
  check-feature calls are still counted only.
- The module-wide `#[allow(unexpected_cfgs)]` that the `#[gpu]` expansion carries also
  silences a user's cfg typos inside the `#[gpu]` module; narrow it to the emitted twin
  items.
- `gust::cpu::invocations` iterates `workgroup_size` x only. `validate` rejects a y or z
  other than 1 (`UnsupportedWorkgroupSize`, as `gust-wgpu` does), so no typed call reaches the
  gap today; give `invocations` a guard of its own before M7 lifts the restriction.
- If kernels ever become generic, re-check that a defining crate's overflow checks survive instantiation in an
  unchecked crate (the defining-crate rule is measured for non-generic shadow bodies only).
- Optional: `cargo test -- --color never` in the counted calls would remove the ANSI
  stripping (`without_colors`) from the count guard.
- Record the adapter in `.ai/VALIDATION.json`: any Vulkan adapter satisfies the GPU
  tiers, a software one (lavapipe) included, and the record does not say which ran.
  Until then a GPU claim names its adapter in STATUS.
- Stale SPIR-V in the ignored `generated-wgpu` directory can satisfy `check-artifacts`'
  name check locally (CI starts clean); clear the directory before the export, or
  compare timestamps.

## Deferred language/runtime breadth

Vectors/matrices, uniforms, textures, samplers, additional binding groups, and
specialized layouts each require their own reflected ABI evidence. `while`,
inclusive ranges, general match expressions, `?`, and broader Result APIs need
explicit semantic contracts before implementation.
