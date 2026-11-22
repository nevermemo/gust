# walker-steer

Walkers on a periodic `SIDE x SIDE` grid that sample a `u32` field ahead, turn toward the strongest sample, move one
cell and deposit into the field, 24 steps, all in integers, so the host reference, the CPU twin and the GPU are equal
bit for bit after every step (field and walkers; on the GPU: Radeon 890M and RTX 5090).

- `walk` (one invocation per walker): read the field `dist` cells ahead at headings `h - 1`, `h` and `h + 1` (mod 8),
  take the largest (ties: straight, then `h - 1`, clockwise, then `h + 1`), with probability `prob / 2^16` replace the choice by a hash-chosen one
  (`hash(hash(seed, step), walker)`, the hash of `examples/hash-rng`), move, and `atomic_add` the amount into a deposit
  buffer at the new cell.
- `relax` (one invocation per cell): `field = field - (field >> shift) + deposit`, then the deposit is cleared.
- Walkers read only the previous step's field and the deposits are integer sums, so the order of walkers never matters.
- The parameters (seed, step, distance, probability, amount, shift) are one `Uniform<Params>` struct.
- No `u32` overflow: every cell stays at most `amount * walkers * 2^shift` (proof in the module documentation of
  `src/app.rs`); the tests assert it for their inputs.
- Cases cover fewer and more walkers than one workgroup, all walkers in one cell with one heading (every walker adds to
  one deposit cell), and `prob = 0` and `prob = 2^16`. A failure names the first differing walker or cell.

Run with

```text
GUST_PROFILE=dev cargo run --release -p walker-steer
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p walker-steer`, and the
GPU tests with `cargo test -p walker-steer -- --include-ignored --nocapture`.
