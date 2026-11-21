# budget-kit

A GPU budget kit: how much compute work of a few common shapes fits in one 16.7 ms frame on this machine's GPU,
written with GUST's public API only.

| # | Workload | Shape |
| --- | --- | --- |
| W1 | Copy (reference) | read one array, write another |
| W2 | Field diffusion | 2D grid of four `f32` channels; each cell blends its four neighbours and decays |
| W3 | Particle step | position and velocity in `f32`; sample the W2 field at the particle's cell, move, wrap |
| W4 | Spatial binning | count particles per cell with atomics, then scan the counts |
| W5 | Reduction | `f32` sum with `gust-algo`, checked against an `f64` reference within the documented bound |
| W6 | Readback delay | submit a small kernel and wait until its result is on the CPU |

## Run it

```text
GUST_PROFILE=dev cargo run --release -p budget-kit -- --out PATH [--small]
```

`--out PATH` is required and receives the same text that is printed (there is no default path). `--small` runs only the
smallest size of each workload. `GUST_PROFILE=dev` is needed because a release build without it is strict and refuses
run-time Slang (see "Build profiles" in the top-level README).

The program prints the adapter, the backend, whether GPU timestamps are available and the relevant limits, then a
Markdown table with one row per workload and size: the median time of a step (30 steps after 5 warm-up steps), the
effective GB/s for the bytes the row states, and how many steps fit in 16.7 ms. The time comes from GPU timestamps where
the device has them and from the wall clock around a blocking submit otherwise; each row says which.

Every row is first checked against a CPU computation (a full comparison or a derived tolerance, stated in the source),
so a row measures real work; a row that fails its check prints FAILED instead of a time, and the exit code is 1.

## What this is not

The output is a one-machine design aid, not a benchmark. It asserts no time, it says nothing about another adapter, and
the numbers move with drivers, clocks, power and what else uses the GPU. Run it on the machine and in the conditions you
design for. Its output is not kept in the repository.
