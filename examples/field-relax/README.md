# field-relax

Three scalar fields and an exact integer channel relaxing on one periodic `SIDE x SIDE` grid, 16 steps, each step one
`grid` kernel (`relax`), checked against host references written out in the test.

- Each field spreads to its four neighbours at its own rate `d` and fades with its own keep factor `k`:
  `next = k * mul_add(d * 0.25, n + s + e + w, (1 - d) * c)`. The fields are one `f32` buffer (field-major), the
  integer channel one `u32` buffer, each with an `a` and a `b` copy that the steps alternate between.
- The integer channel keeps `c - 4 * (c >> 3)` and receives `x >> 3` from each neighbour. It is equal bit for bit on the
  host reference, the CPU twin and the GPU after every step, and its sum never changes.
- The six parameters (`d` and `k` of three fields) reach the kernel through one `Uniform<Params>` struct, with two
  explicit pad fields for the `Std140V1` layout (no parameter buffer, no launch per field).
- `SIDE` and the other sizes are module-level `const` items of the `#[gpu]` module (m11.g3). Because a constant has one
  value per module (`MASK` and `CELLS` are constant expressions of `SIDE`, m11.cexpr), a macro writes the module twice:
  `big` (512, `main`, the GPU test and a twin test) and `small` (32, the unit tests, the twin and a GPU test).
- The float bound is derived in the module documentation of `src/app.rs` ("The float bound"): a per-step bound and a
  bound after `S` steps, both asserted for every cell of every field; the worst observed error is printed.

Run with `cargo run --release -p field-relax`; the CPU tests run with `cargo test -p field-relax`, and the GPU tests
with `cargo test -p field-relax -- --include-ignored --nocapture`.
