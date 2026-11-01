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
