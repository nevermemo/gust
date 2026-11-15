# reduce-scan (gallery G6)

A **sum** and an **inclusive scan** (prefix sums) of `u32` values on the GPU, built from kernels that share data through
**workgroup memory** and synchronise with **barriers**. Both are checked **exactly** against a host reference at eight
sizes.

What it shows:

- **Workgroup memory and barriers.** `tile_sum` and `tile_scan` take a `Workgroup<[u32; 64]>` parameter (one array per
  workgroup of 64 lanes, with a `LocalId` for the lane and a `GroupId` for the workgroup) and call `workgroup_barrier()`
  between a round of writes and the round of reads that depends on it. The barriers sit at the top level of the kernel body
  and of the top-level loop, where every lane reaches them (the validator checks the position). `tile_sum` is a tree
  (six rounds halve the active lanes); `tile_scan` is the Hillis and Steele scan (six rounds add the value `2^r` lanes to the
  left, with a barrier between the read and the write of each round).
- **`cpu = "unsupported"`.** A kernel with workgroup memory or a barrier has no CPU twin (SEMANTICS, "What the twin does
  not model"; D101): the twin runs one invocation at a time and would give a clean result that says nothing about the GPU.
  Those two kernels say so in their attribute, and the compiler (GUST-E0717) refuses them without it. Their result is
  checked **against the host reference on the GPU**: `main` asserts every size before it prints, and
  `cargo xtask check-gallery` runs `main` and compares the output line. `add_offsets`, the third kernel, uses neither, so it
  has a twin and a test.
- **Multi-level host code.** One tile is 64 elements, so a bigger input needs a pass per level: `reduce` loops (sums of
  tiles, then sums of those sums, until one tile is left) and `scan` recurses (scan the tiles, scan the tile totals with
  the same function, add each tile's offset with `add_offsets`). Buffers that hold exactly one value per
  element (the input, the scan output) are bound plainly, so the call checks their length; the per-tile ones (`sums`,
  `totals`, `offsets`) hold fewer values than the pass has invocations, so they are wrapped in `Independent`, the
  call-site statement that they are not one per invocation.

Sizes: 1, 63, 64, 65, 1000, 4096, 4097 and 100 000 values. 1, 63, 65, 1000, 4097 and 100 000 leave a **partial last
workgroup** (the lanes past the end hold 0), 64 and 4096 are whole tiles, 4097 has 65 tiles so the tile totals need a second
tile of their own, and 100 000 needs three levels.

The values are `(7 i + 3) % 10`, so no sum or prefix of these sizes leaves `u32` (the largest is 450 000) and every
result is an exact integer. There is no tolerance. Integer overflow is outside the defined domain of a sum (`+` wraps on a
GPU and panics in a debug twin, D39); this entry stays inside it.

Tests (`cargo test`): the host references against hand-computed values; `add_offsets` on the CPU twin completing a
host-made per-tile scan, against the host scan, at 1, 63, 64, 65, 129, 1000 and 4097 values (exact); and one test that
runs the GPU check for every size, marked `#[ignore]` because it needs an adapter (`cargo test -- --ignored`; it fails
loudly without one, it never skips silently). No gate runs that test (`cargo xtask check-gallery` runs `cargo test`
without `--ignored`), so the barrier kernels are not covered by a plain `cargo test`: their evidence is `cargo run
--release`, which the check does run. A GPU pass is evidence on that adapter, not a proof that the barriers are right: a
barrier that is missing can still pass where the 64 lanes happen to run in lockstep, so the result says nothing about
adapters whose lanes do not.

`gust-algo` (D110): the repository's `gust-algo` crate offers the general version of these operations (`reduce`, `scan`
and `compact` over `GpuVec`, with 256-lane tiles and a different schedule). This entry does **not** depend on it: its
kernels are compiled at run time through Slang and the crate ships no precompiled artifacts (D110: `gust-algo` includes
`tests/fixtures/*.rs` and has none), so a release build without Slang, which this gallery requires of every entry, could
not run it. The kernels of this entry are inline instead, committed under `gust-artifacts/` like the others.

Edge cases (D39): a workgroup's reads of another lane's cell are defined only after a barrier; workgroup memory starts as
zeros on every route (D174), which these kernels do not rely on (every lane writes its cell before the first barrier). A
sum that overflows is out of domain, as above.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the host references and the add_offsets twin; no GPU, no Slang
cargo test -- --ignored           # the GPU check of every size; needs an adapter
cargo run --features gust-dev     # development: compiles the kernels at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernels with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the kernels are embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints `reduce-scan: 100000 values in 8 sizes, sum 450000, scan ends at 450000 with checksum 22499850000` (the values are fixed, so the
numbers are too) and a line `adapter: <name>` with the GPU it ran on.
