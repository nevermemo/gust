# histogram (gallery G3)

A count of keys into bins, where every invocation adds one to the bin of its key with `atomic_add`. With 100 000 keys in
8 bins, between about six and thirty-five thousand invocations hit each bin, and the keys are skewed (the busiest bin
holds several times the least busy one), so the example is contention on purpose. A plain `bins[k] += 1` would lose
counts on a GPU; the atomic does not, and the result is checked **exactly** against a host count.

What it shows: atomics on a `RWStructuredBuffer<uint>`; a dispatch whose output is shorter than its element count
(`Independent(&bins)` is the call-site statement that the bins are not one per invocation); an exact check of an integer
result against an independent host count (`main` asserts it before it prints).

Tests (`cargo test`, CPU twin only): the kernel on `CpuDevice` against a plain Rust count over the keys, exact, at 1, 63,
64, 65, 1000 and 100 000 keys (the workgroup is 64, so the last one is partial in all but 64) and 1, 2, 8 and 256 bins;
every run also checks that the counts add up to the number of keys; one test checks hand-counted values; one guards the
skew of the keys.

What the twin does **not** show (D26, D39): the twin runs the invocations one after another, so its `atomic_add` is a
plain read-modify-write and cannot lose a count. Contention is real only on the GPU, and the evidence for it is `cargo
run --release` (and `cargo xtask check-gallery`, which runs it and compares the output line). The sum of an atomic
integer count does not depend on the order of the invocations, which is why both sides are exact whatever order the GPU
chooses. A kernel whose result depends on the order of the atomic operations (not an add) would differ between twin and
GPU, and that difference is documented, not chased.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the CPU twin against the host count; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernel at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernel with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the kernel is embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints the line `histogram: 100000 keys in 8 bins [..]` with the eight counts (the keys are fixed, so the counts are
too), then a line `adapter: <name>` with the GPU it ran on.
