# cell-bins

Binning elements by the cell of a `side x side` grid they lie in, as a piece a program can rebuild every step
(m11.bins). The `bins` module (`src/bins.rs`) owns the buffers: per-cell counts, the starts (the exclusive scan of the
counts, `gust_algo::scan::exclusive_into` into a buffer kept across steps), a cursor, and the element order.
`Bins::new(handle, side, capacity)` makes them once; `Bins::rebuild(&xs, &ys)` rebuilds the bins from the positions
(each in `[0, side)`, or past it: a position at or past `side - 1` goes to the last cell of its axis, also `+inf`; cell =
`floor(x) + side * floor(y)`) into the same buffers. The bins' own buffers are made once; the scan's temporaries come from
the device pool: the first rebuild may allocate them, and a second `exclusive_into` of the same size takes no new device
buffer (the pool's miss counter does not move; a test in `gust-algo` asserts this for `exclusive_into`
at one size, 70,001 elements, Radeon 890M only; nothing in
this crate measures it). `new` rejects `side == 0` and a grid of
more than `u32::MAX` cells.

Four device steps per rebuild: `clear` (counts and cursors to zero), `count` (an `atomic_add` per element on its cell),
the scan of the counts, `scatter` (each element takes the old value of an `atomic_add` on its cell's cursor as its
place and writes its index to `order[starts[cell] + place]`).

## Order contract

The **set** of elements in each cell is exact. Their **order** inside a cell is **unspecified** on the GPU: it is the
order of the atomic adds, which depends on the schedule. Compare per cell as sets (sort the cell's slice first), never
the raw order. The elements of cell `c` are `order[starts[c] .. starts[c] + counts[c]]`; the slots of `order` from the
element count on are not written. `Bins::read` returns the three arrays on the host, and `Snapshot::cell(c)` is a
cell's slice; a program reads the bins through these.

## Checks

`src/app.rs` checks the bins against a host counting sort written out there: counts and starts exactly, the total, and
every cell's set. Cases: no elements, all elements in one cell, elements only on the last row and column (positions
just below `side`), positions far past the grid (1e10, `f32::MAX`, `+inf`), 100,037 hash-spread elements (the `hash` of `examples/hash-rng`, copied as `examples/mc-histogram`
does), and a sequence of rebuilds with different positions and counts into the same bins (no stale data). The CPU twin
runs them in the `twin_*` tests; the GPU test runs the same cases on the device.

Run with

```text
GUST_PROFILE=dev cargo run --release -p cell-bins
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p cell-bins`, and the
GPU test with `cargo test -p cell-bins -- --include-ignored --nocapture`.
