# walker-transport

One composite step of bins, walkers and a field on a periodic `SIDE x SIDE` grid, 24 steps, all in integers, so the host
reference, the CPU twin and the GPU are equal bit for bit after every step (field, walkers and the per-cell counts of
the bins; the walkers grouped by cell are equal per cell as sets, compared after sorting each cell's slice, because the
raw order inside a cell is unspecified on the GPU). On the GPU: Radeon 890M and RTX 5090. It runs the three
pieces of `examples/field-relax`, `examples/walker-steer` and `examples/cell-bins` together.

One step is four stages (the bins stage itself dispatches clear, count, scan and scatter):

- **bins** (`src/bins.rs`, the counting sort of `examples/cell-bins` copied and adapted to integer cells): the count of
  walkers in each cell, and the walkers grouped by cell.
- `walk` (one invocation per walker): as in `walker-steer` (sense `dist` cells ahead at headings `h - 1`, `h` and `h + 1`,
  ties: straight, then `h - 1`, then `h + 1`; with probability `prob / 2^16` a hash-chosen turn; move; `atomic_add` the
  amount into a deposit buffer), plus a **crowding rule**: a walker whose cell holds more than `crowd` walkers takes the
  hash-chosen turn whatever `prob` says (the same hash draw as the random turn).
- `spread` (one invocation per cell): with `g = field + deposit`, `next = g - 4 * (g >> 3) + sum of (g >> 3)` of the four
  wrapped neighbours (the exact `u32` channel of `field-relax`), written to a second field buffer; the host swaps the
  two buffers.
- `clear_deposit` (one invocation per cell): the deposit to zero (its own dispatch, because `spread` reads the
  deposits of the neighbours).

The parameters (seed, step, distance, probability, amount, crowd) are one `Uniform<Params>` struct created once and
written every step.

- No `u32` overflow: the spread conserves the sum, so every value stays at most `T0 + steps * amount * walkers` (proof
  in the module documentation of `src/app.rs`); the tests assert it for their inputs and the exact total after every
  step.
- Cases cover fewer and more walkers than one workgroup, all walkers in one cell (with and without one heading), every
  walker crowded (`crowd = 0`), and a crowd above the walker count, which must equal a host run without the rule. A
  failure names the first differing walker, field cell, bin count or bin slot.

Run with

```text
GUST_PROFILE=dev cargo run --release -p walker-transport
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p walker-transport`, and
the GPU tests with `cargo test -p walker-transport -- --include-ignored --nocapture`.
