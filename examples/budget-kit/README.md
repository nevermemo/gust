# budget-kit

A GPU budget kit: how much compute work of a few common shapes fits in one 16.7 ms frame on this machine's GPU,
written with GUST's public API only.

| # | Workload | Shape |
| --- | --- | --- |
| W1 | Copy (reference) | read one array, write another |
| W2 | Field diffusion | 2D grid of four `f32` channels; each cell blends its four neighbours and decays |
| W3 | Particle step | position and velocity in `f32`; sample the W2 field at the particle's cell, move, wrap |
| W4 | Spatial binning | count particles per cell with atomics, then scan the counts (the scatter is a gap: an atomic's old value cannot be used) |
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

W4 runs its own clear and count batch through `gust_algo::dispatch_batch_in_scope` inside the same `timed` scope as the
`gust-algo` scan, so the scope's batch count and GPU time cover both. Its row is therefore the summed GPU timestamps of
its own batch and of the scan's batches, or, on a device without timestamps, the wall clock around the whole step, as
for W5.

After the table the program runs a probe of the linear dispatch limit (gap G4), which is not a workload: a linear
kernel launch of 65 536 workgroups on x, one above the portable limit of 65 535. It prints one line starting `G4:` with
the outcome (a `DeviceLimit` error with the requested and maximum counts, which is gust-wgpu's own host-side check of the
launch against the device limits, made before anything is submitted; or success, with the results compared exactly with
the host), the device's `max_compute_workgroups_per_dimension` and the requested limits policy of the kit's device (a constant, not read back). Any error other than that `DeviceLimit` (the workgroup limit of x, 65 536 requested, the maximum the device reports) counts as FAILED. A
`G4 remedy:` line then shows a two-dimensional launch (`Extent::new(x, y, 1)`, `i = id.y * width + id.x`, and an
`if i >= n { return; }` guard in the kernel) over a count that is not a multiple of the row width, so the guard has
work to do: the values are checked exactly against the host and the buffer slots past the count must still hold their
sentinel. A probe or remedy that fails its check counts as a FAILED row.

## What this is not

The output is a one-machine design aid, not a benchmark. It asserts no time, it says nothing about another adapter, and
the numbers move with drivers, clocks, power and what else uses the GPU. Run it on the machine and in the conditions you
design for. Its output is not kept in the repository.
