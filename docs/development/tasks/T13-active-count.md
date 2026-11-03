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
cargo test -p component-pool -- --include-ignored
cargo xtask check-cpu
cargo xtask check-examples
cargo xtask check-full
```

`check-examples` and `check-full` run the example with `--include-ignored` and fail if any test stays ignored; the T13
GPU tests are `#[ignore]`d and run nowhere else. `check-feature gpu-semantics` runs no T13 test.

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

Status (slices m7.t13 and m7.t14, 2026-11-03): landed in m7.t13, in `examples/component-pool`; retirement and
compaction landed in m7.t14 (the last section below). On the T13 path (`run_frame_traced`, unchanged) the active
particles form a prefix of the pool. The emitter spawns in slot order and the host reference asserts it (`step_cpu`,
`the_emitter_reference_spawns_a_prefix`). On that path the invariant is still enforced on the host only: nothing on
the GPU checks it, so a push that released slots out of order would make `integrate` step the wrong particles
silently. The compacting path (`run_compacting_frame`) does not rely on the invariant, because it builds the dense
prefix itself every frame; it is the only path whose dispatch the gap does not affect. The regressions live in the
example's tests, not in `crates/gust-wgpu/tests/`, because the existing typed graph and pool APIs express the proof.

The stash's `active_count` entry in `check-feature gpu-semantics` is not added. It names
`crates/gust-wgpu/tests/active_count.rs`, which does not exist (the stash never had it, and gust-wgpu was out of this
slice), so the feature would run no T13 test. The real gates are `check-examples` and `check-full`.

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
  with `GraphMissingDependency`), dropping `i < live[0u32]` in `classify` (6 tests fail, among them
  `twin_classify_counts_the_spawned_live_slots` and `gpu_active_count_matches_cpu_reference_at_boundaries`), and
  dropping the count reset upload (5 tests fail, among them `gpu_active_count_matches_cpu_reference_at_boundaries`).
- Exported kernels 13 to 14 (`pool__classify.spv`, `EXPECTED_SPIRV`). The example has no committed
  `gust-artifacts/`; its exports go to the ignored `generated-wgpu/`.
- Remaining uncertainty: one adapter (NVIDIA GeForce RTX 5090, Vulkan). The count is an integer atomic add, exact on
  any conforming adapter; other adapters and the browser pipeline are not exercised.

### Retirement and compaction (slice m7.t14, 2026-11-03)

Status: landed in m7.t14, in `examples/component-pool` (a path dependency on `gust-algo`; `check-arch` accepts it).
The compacting path does not need the prefix invariant and is checked on the GPU. The gap is not closed beyond it:
the T13 path (`run_frame_traced`) and its tests are unchanged, so it is still valid only on emitter-ordered pools in
which no particle retires, with the prefix invariant enforced on the host only. The two paths must not share one
pool: `classify` ignores `lifetime`, so the T13 path on a pool that has retired particles would integrate them.

- Retirement rule: a particle retires once its `steps` counter (its age) reaches a lifetime, `steps >= lifetime`,
  with the lifetime in a one-element buffer (`Frame::lifetime`, written by every compacting frame). The T13 state
  has the counter but no lifetime, so the limit is new and the counter is the rule the example already had. A slot is
  judged on its own data, so particles retire in any slot order; a particle also takes part only when
  `spawn_frame <= settings.frame`.
- Per frame (`run_compacting_frame`): graph 1 uploads the settings and the lifetime, then `mark_live` writes a 0/1
  flag per slot; `gust_algo::compact::indices` turns the flags into the increasing slots of the live particles and
  their count (both on the device); graph 2 runs `gather_particles` into a dense, zero-initialised buffer, then
  `prepare_dispatch` (fed the compaction's count, clamped to the dense buffer and the budget), the indirect
  `integrate` over the dense prefix, `scatter_particles` back to each particle's own slot, and the readback of the
  active count. Retired and unspawned slots are never read by `integrate` and never written. No readback decides
  work (the host reads `active` for the demo and the tests). Explicit edges: graph 1 `mark_live` after both uploads;
  graph 2 `gather` (none), `prepare` after `gather`, `integrate` after `prepare`, `scatter` after `integrate`,
  readback after `integrate`. No graph edge orders the compaction before `prepare`: `compact::indices` runs outside
  both graphs (it submits its own batches, so the graph cannot span it), and that order rests on the blocking
  `dispatch_batch` and `execute_graph` waits of each call. A non-blocking submission there would remove it silently,
  and no test would catch it.
- An empty pool (length 0) returns `None` and submits nothing: the host owns the length, and an empty binding is
  refused.
- GPU tests (`cargo xtask check-examples`, RTX 5090):
  `compaction_keeps_the_prefix_and_never_integrates_retired_particles` (pool sizes 1, 63, 64, 65 and 257;
  retirement of none, the first, the last, a middle slot, all, alternate and scattered slots; everything released
  at once and an emitter at three per frame; budgets 1000 and 40; every frame until the particles retire),
  `compaction_at_one_past_65536_slots_matches_the_reference` (65 537 slots),
  `compaction_does_not_depend_on_the_order_particles_are_released` (the spawned particles are a suffix),
  `an_empty_pool_has_nothing_to_compact` (size 0) and `the_graph_refuses_a_compaction_pass_without_its_declared_edge`.
  Every frame reads the GPU's flags, count, indices and dense buffer back and asserts: the flags equal the rule, the
  count and the indices equal an independent host filter (increasing, zero after the count), the dense buffer holds
  exactly the live particles in slot order (the first `active` stepped) and is zero after the count, every particle
  of the dense prefix is live once its step is undone, retired and unspawned slots are byte-identical to the
  frame before, and the whole pool equals `step_retiring_cpu` (a filter, no compaction). The twins of the three new
  kernels are checked without a GPU by `twin_mark_live_flags_the_spawned_unretired_slots`,
  `twin_gather_and_scatter_follow_the_compaction_indices` and `twin_compacting_frames_match_host_reference`
  (`check-cpu`; the scan inside `compact::indices` is the library's and GPU-only here).
- Mutations, each caught: skipping the compaction (identity indices over the live count, the T13 prefix path) fails
  both large GPU tests on the first frame in which particles have retired out of order; gathering from
  `indices[j] + 1` fails the twin gather test and both GPU tests; dropping the `gather` to `prepare_dispatch` edge
  makes the graph refuse the frame with `GraphMissingDependency { node: 1, producer: 0 }`; dropping
  `steps < lifetime` from `mark_live` (retired particles integrated) fails both GPU tests. The dropped edge is a
  resource-hazard edge (`prepare_dispatch` reads `dense.len()`); it is not an edge between compaction and
  `prepare_dispatch`, which does not exist (see above).
- Exported kernels 14 to 17 (`pool__mark_live.spv`, `pool__gather_particles.spv`, `pool__scatter_particles.spv`;
  `EXPECTED_SPIRV`, `the_expected_list_is_seventeen_sorted_unique_spirv_names`). The example still has no committed
  `gust-artifacts/`, so there is nothing to regenerate; its exports go to the ignored `generated-wgpu/`
  (`cargo xtask check-artifacts` regenerates and validates them).
- Remaining: the pool's host length does not shrink when particles retire (the host never learns the count), so dead
  slots stay in the pool until the host truncates it, which needs a readback taken outside the frame. One adapter
  (NVIDIA GeForce RTX 5090, Vulkan), as above. Float results are compared with the same relative tolerance as the T13
  tests (`assert_close`), not bit-exact; `integrate` is unchanged.
