# T10: Atomics

## Contract

Statement-only atomics on direct `&mut RWStructuredBuffer<T>[index]` receivers.

- `u32`: `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange`,
  `atomic_compare_exchange`.
- `i32`: `atomic_add`, `atomic_exchange`, `atomic_compare_exchange`.
- Atomic buffer typing is per emitted kernel.
- Ordinary indexed reads/writes of a buffer used atomically in that kernel reject.
- Relaxed/default memory ordering only.

## Rejected

Read-only/local/non-element receivers, float or unsupported elements, wrong arity,
signed min/max, expression-position atomics, and mixed atomic/plain access.

## Evidence

- Macro: `cargo test -p gust-compute-macros -- atomic`
- GPU: `cargo test -p gust-compute-wgpu --test atomics`
- Example: `cargo run -p atomic-counter`
- Release: `cargo xtask check-full`

Validated on NVIDIA GeForce RTX 5090 / Vulkan. Other backends/vendors remain
unverified.
