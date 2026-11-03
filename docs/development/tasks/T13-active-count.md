# T13: GPU active count to indirect dispatch

## Goal

Prove GPU-authored active-count generation drives a validated indirect compute
dispatch without CPU readback. This is the third small engine proof from
`ENGINE_HURRICANE.md`.

## Contract

- A classification kernel reads resident component/state data and atomically
  increments a one-element `RWStructuredBuffer<uint>` count for every active item.
- A preparation kernel reads the GPU-authored count and writes indirect dispatch
  arguments, clamped to the data allocation and an explicit host-provided budget.
- An existing indirect consumer kernel dispatches from those arguments and guards
  `i < active && i < buffer.len()`.
- All count/args bindings use `independent_length()` and are non-empty.
- The graph declares explicit classification -> preparation -> indirect-consumer
  dependencies; it never infers or reorders them.
- No GPU-to-CPU readback is used to determine dispatch work.

## Out of scope

Freelists, entity IDs, compaction/scatter, culling output lists, rendering,
inter-workgroup synchronization, CPU fallback, and general atomic load/store
semantics.

## Owned files

- `examples/component-pool/`: runnable engine proof and independent CPU reference.
- `crates/gust-wgpu/src/{graph,pool,lib}.rs`: only if existing typed graph/pool APIs
  cannot express the bounded proof.
- `crates/gust-wgpu/tests/`: GPU readback, dependency, clamp, and invalid-binding
  regressions.
- `docs/ENGINE_HURRICANE.md`, `docs/DECISIONS.md`, and `docs/development/STATUS.md`:
  contract and evidence.

## Checks

```sh
cargo test -p component-pool -- --ignored
cargo xtask check-feature gpu-semantics
cargo xtask check-examples
cargo xtask check-full
```

## Completion

- GPU readback equals an independent CPU active-count reference at 0, 1, 63, 64,
  65, and 257 elements.
- Indirect arguments are derived on GPU and clamped to allocation and budget.
- The indirect consumer observes exactly the expected active subset without CPU
  count readback.
- Tests reject missing graph dependencies, invalid count/args layouts, zero-length
  independent bindings, and unclamped dispatch arguments.
- Full validation and independent review pass; record adapter and remaining
  cross-platform uncertainty in `STATUS.md`.

## Status and evidence

Status (slice m7.t13, 2026-11-03): implemented in `examples/component-pool`, in review. Compaction and retirement
are slice m7.t14; until then the active particles form a prefix of the pool (the emitter spawns in slot order and the
host reference asserts it). The regressions live in the example's tests, not in `crates/gust-wgpu/tests/`, because the
existing typed graph and pool APIs express the proof.

- Kernels: `classify` (atomic count of the live particles whose `spawn_frame <= settings.frame`),
  `prepare_dispatch` (reads the GPU-authored count; clamps to `particles.len()` and the budget) and `integrate`
  (the indirect consumer, guarded `i < active && i < particles.len()`). The graph declares the settings upload and
  the count reset, then classify, then prepare_dispatch, then the indirect integrate, then the readback; no edge is
  inferred. Only the active count is read back, for the demo and the tests.
- GPU tests (`cargo xtask check-examples`, RTX 5090):
  `gpu_active_count_matches_cpu_reference_at_boundaries` (0, 1, 63, 64, 65 and 257 live slots, a retired tail, four
  frames each), `indirect_arguments_are_clamped_to_budget_and_allocation`,
  `graph_rejects_undeclared_classification_edges` and `count_bindings_reject_empty_and_mismatched_buffers`. The twin
  of `classify` is checked without a GPU by `twin_classify_counts_the_spawned_live_slots` (`check-cpu`).
- Mutations, each caught: dropping the allocation clamp or the budget clamp in `prepare_dispatch`, dropping the
  `i < active` guard in `integrate`, and removing the classify to prepare_dispatch edge (the graph refuses the run
  with `GraphMissingDependency`).
- Exported kernels 13 to 14 (`pool__classify.spv`, `EXPECTED_SPIRV`). The example has no committed
  `gust-artifacts/`; its exports go to the ignored `generated-wgpu/`.
- Remaining uncertainty: one adapter (NVIDIA GeForce RTX 5090, Vulkan). The count is an integer atomic add, exact on
  any conforming adapter; other adapters and the browser pipeline are not exercised.
