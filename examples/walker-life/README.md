# walker-life

Walkers on a periodic `SIDE x SIDE` grid that move, lose energy, retire and spawn children, 24 steps, all in integers,
so the host reference, the CPU twin and the GPU are equal bit for bit after every step (every field of every walker,
and the number of walkers). The live walkers stay dense in their buffers, so later kernels can launch over `0..n`.

- `act` (one invocation per slot): a live walker turns by a hash-drawn amount, moves one cell, loses one energy, and
  decides: it retires when its energy reaches 0 or a hash draw is below `retire / 2^16`; it spawns one child when its
  energy is above `threshold` and a second draw is below `spawn / 2^16`. It writes a `flags` word (bit 0: keep, bit 1:
  a child) and the count `keep + kid`. Slots at or past the live count write a count of 0. Radeon 890M only; RTX 5090
  pending.
- The counts are scanned with `gust_algo::scan::exclusive_into`; the first output slot of a walker is its start, and
  the next live count is the last start plus the last count, clamped to the capacity.
- `compact` (one invocation per live walker): a kept walker writes itself to `start`, a child (half the parent's
  energy, the parent keeps the rest, a hash-chosen heading) to `start + keep`, into the second set of buffers; slots at
  or past the capacity are dropped. The host swaps the buffer sets.
- No `atomic_add` assigns slots: an atomic append would make the order of the spawned walkers depend on scheduling.
  The scan makes it a pure function of the walkers.
- Cases cover fewer and more walkers than one workgroup, a population that hits the capacity, and one that dies out
  to 0 (every later step is a no-op). A failure names the first differing walker or the count.

Run with

```text
GUST_PROFILE=dev cargo run --release -p walker-life
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p walker-life`, and the
GPU tests with `cargo test -p walker-life -- --include-ignored --nocapture`.
