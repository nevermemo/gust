# budget-kit

A GPU budget kit: how much compute work of a few common shapes fits in one 16.7 ms frame on this machine's GPU,
written with GUST's public API only.

| # | Workload | Shape |
| --- | --- | --- |
| W1 | Copy (reference) | read one array, write another |
| W2 | Field diffusion | 2D grid of four `f32` channels; each cell blends its four neighbours and decays |
| W3 | Particle step | position and velocity in `f32`; sample the W2 field at the particle's cell, move, wrap |
| W4 | Spatial binning | count particles per cell with atomics, then scan the counts; a second row adds the scatter of the particle indices (each takes its slot from the old value of an atomic add; the order inside a cell is unspecified) |
| W5 | Reduction | `f32` sum with `gust-algo`, checked against an `f64` reference within the documented bound |
| W6 | Readback delay | submit a small kernel and wait until its result is on the CPU |
| W7 | Composite step | one batch per step: diffuse a 1024 x 1024 field, step the particles, clear and count their cells; a sustained run reporting the GPU time's median, p95 and maximum and two host-side costs |
| W8 | Full step with lifecycle | one step on a 1024 x 1024 field: diffuse, move the particles over the field, bin them (clear, count, scan, scatter), deposit a fixed amount per particle into a `u32` grid with an atomic add and fold the grid into the field, then retire some particles by a hash, spawn a fixed number and keep the live set dense with a scan of alive flags and a compaction; a sustained run as W7, with the storage buffers each kernel binds in the note, the GPU time of each of its four batches and the median of each of its kernels run as one-kernel batches (a separate pass each) |
| W8a | W8 without bins | W8 with no count, no scan of the counts and no scatter |
| W8b | W8 with coarse bins | W8 with the bins on a 256 x 256 grid over the same 1024 x 1024 field |
| W8c | W8 with a fused particle kernel | W8 with the step, the count and the deposit in one particle kernel |
| W8d | W8 with a sparse lifecycle | W8 with the lifecycle (retire, flag scan, compaction, spawn) on every 4th step; the time is the amortised mean per step |

## Run it

```text
GUST_PROFILE=dev cargo run --release --manifest-path bench/Cargo.toml -p budget-kit -- --out PATH [--small]
```

`budget-kit` is a member of the `bench/` workspace (its own lock, release profile and check-cfg), run from the repository
root as above. `cargo xtask check-bench --budget-kit` runs it with `--small` and fails on a non-zero exit, a missing
row or probe line, a failed check or a missing `0 row(s) FAILED` summary.

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
for W5. The second W4 row ("count + scan + scatter") adds a third batch, the scatter, to the same scope; its check
compares, for every cell, the slots the scatter filled with the host's set of particle indices for that cell (each
range sorted first, since the order inside a cell is unspecified) and requires every index to appear once.

W7 is a sustained run, not a median of 30: 600 counted steps (120 with `--small`) after the warm-up, of one batch of
four kernels (`w2_diffuse` on 1024 x 1024, `w3_step`, `w4_clear`, `w4_count` into 1024 x 1024 cells), at 2^18 and 2^20
particles (2^18 only with `--small`). The time column is the median GPU time of the steps (the wall clock where the
device has no timestamps); the row's note gives its p95 and maximum, and the host side two ways: (a) `host_elapsed` of
the blocking `dispatch_batch_timed` (it includes the GPU work and the timestamp readback), and (b) the wall time for
`submit_batch` to return (encode and submit, no wait; the job is waited for after the measured span, so the queue is idle
at every call and (b) shows no back-pressure). The two come from two runs of the same length, because a step cannot be
both blocking and not. It is checked after the first step (field against the host's `cpu_diffuse`, particles against
`cpu_particle` over the GPU's own field, with W2's and W3's tolerances, counts exactly against the host's histogram of
the GPU's own positions) and again after the runs.

W8 is a sustained run of a whole step, with the same number of counted steps as W7, at a capacity of 2^17 and 2^20
particles (2^17 only with `--small`). A step is four batches in one `timed` scope: (1) `w2_diffuse` on 1024 x 1024,
`w3_step` over the diffused field, the clears of the counts, the cursors and the deposit grid, `w4_count`, `w8_deposit`
(an `atomic_add` of a fixed amount into a `u32` grid), `w8_fold` (the grid into one lane of the field) and `w8_retire`
(alive flags); (2) the `gust-algo` exclusive scan of the 1024 x 1024 counts; (3) the same scan of the alive flags;
(4) `w4_scatter`, `w8_compact` (survivors to the slots the scan gives) and `w8_spawn` (a fixed number of new particles
after the survivors, those past the capacity dropped). The population starts at half the capacity, about one particle in
sixteen retires per step by a hash of the step and the slot, and the live set stays dense. The live counts come from a
host model of that rule (an integer function of the step and the slot), not from a readback; a design that reads the
count back each step pays the W6 delay on top. The row's note gives the GPU time's median, p95 and maximum, three
host-side costs and the storage buffers each kernel binds, counted from the generated descriptors (the scan's four
kernels included; GUST opens devices with 8 storage buffers per stage). It is checked after steps 1 and 2 and after the
runs: the field against the host (diffusion of the GPU's previous field, then the deposit folded in, with W2's
tolerance), the moved particles as W3, the deposit grid, the counts and their scan exactly, the scatter as sets per cell
as W4, and the alive flags, their scan, the survivors, the live count and the live set exactly against the host's model
of the retire and spawn rule. The hash (`lb_mix` and `lb_hash`, used for both the retire draw and the spawn draw) is copied from
`examples/walker-life`, whose retire flags, scan and compaction pattern the step follows. The first step where the
population is clamped at the capacity is checked too, and the scatter's `order` is poisoned again before each checked
step.

Slice m11.cost asks where that step spends its time. After the sustained runs the W8 row's note gains two passes of 30
steps after the warm-up: each of the four batches in a `timed` scope of its own, and each kernel (the scans as one scope
each) as a one-kernel batch in a scope of its own, with the median of each and its share of the total; the sum of
one-kernel batches may differ from the fused batches. The rows W8a to W8d run the same sustained run with one change each
(no bins; the bins on a 256 x 256 grid; the step, the count and the deposit in one kernel; the lifecycle on every 4th
step, reported as the amortised mean per step over whole rounds of 4 steps) and are checked against the host as W8 is.

After the table the program runs a probe of the linear dispatch limit (gap G4), which is not a workload: a linear
kernel launch of 65 536 workgroups on x, one above the portable limit of 65 535. It prints one line starting `G4:` with
the outcome (since slice m11.g4b the host splits the launch into pieces and it succeeds, with the results compared
exactly with the host; a `DeviceLimit` error with the requested and maximum counts, gust-wgpu's own host-side check of
the launch against the device limits made before anything is submitted, is still reported if a device or build cannot
split it), the device's `max_compute_workgroups_per_dimension` and the requested limits policy of the kit's
device (a constant, not read back). Any error other than that `DeviceLimit` (the workgroup limit of x, 65 536
requested, the maximum the device reports) counts as FAILED. A
`G4 remedy:` line then shows a two-dimensional launch (`Extent::new(x, y, 1)`, `i = id.y * width + id.x`, and an
`if i >= n { return; }` guard in the kernel) over a count that is not a multiple of the row width, so the guard has
work to do: the values are checked exactly against the host and the buffer slots past the count must still hold their
sentinel. A probe or remedy that fails its check counts as a FAILED row.

## What this is not

The output is a one-machine design aid, not a benchmark. It asserts no time, it says nothing about another adapter, and
the numbers move with drivers, clocks, power and what else uses the GPU. Run it on the machine and in the conditions you
design for. Its output is not kept in the repository.
