# mc-histogram

A sampled histogram smoothed by diffusion, in four stages, checked against host references written out in the test.

1. `deposit` (a `grid` kernel): each sample is a hash of a seed and its index (`mix` and `hash` are those of
   `examples/hash-rng`), turned into a cell of a `SIDE x SIDE` field with integer arithmetic only, and counted with
   `atomic_add` of `UNIT` (a fixed-point weight). Integer atomics are exact and order-free, so the histogram is equal
   bit for bit on the host reference, the CPU twin and every GPU.
2. `density` (a `#[gpu_fn] fn`, run with `map_gpu_on`): the fixed-point cells as `f32`, times `1 / UNIT`.
3. `diffuse` (a `grid` kernel): one step of a 5-point blend, wrapping at the edges with `rem_euclid`, the blend written
   with `mul_add`. Eight steps, ping-pong between two buffers.
4. The checks: the histogram exactly, its sum against `gust_algo::reduce::sum_wrapping`, the density exactly, the
   diffusion within a bound derived in the module documentation of `src/app.rs`.

`SIDE`, `UNIT`, `KEEP` and `SHARE` are module-level `const` items of the `#[gpu_module]` module (m11.g3); the host reads the
same items. `main` runs 2^24 + 1000 samples, which at workgroup 256 is more workgroups than the per-dimension limit of
most devices, so the host splits such a launch (m11.g4b); `main` prints the device's limit, the workgroup count and
whether the count is past the limit (it computes that from the two numbers; it does not observe the split).

Run with

```text
GUST_PROFILE=dev cargo run --release -p mc-histogram
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p mc-histogram`, and the
GPU test with `cargo test -p mc-histogram -- --include-ignored --nocapture`.
