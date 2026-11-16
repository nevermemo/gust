# GUST benchmark harness (M10 spike B1)

A custom harness in a nested, unpublished workspace. It measures four workload families:

- **W1 vector-add** and **W2 saxpy** (f32, memory-bandwidth bound), 4096 to 16 776 960 elements.
- **W3 Horner polynomial** (f32, `W3-d4`, `W3-d16`, `W3-d64`), 2^20 and 2^22 elements: arithmetic-bound, `2 d` flops per
  8 bytes. Correct when every element is inside a Higham error bound against an f64 reference (derived in
  `gust-bench/src/workloads/polynomial.rs`); a test shows the bound holds for a backend that fuses multiply-add and
  fails for a perturbed result.
- **W4 u32 atomic histogram** (`W4-256-uniform`, `W4-256-zipf`, `W4-65536-uniform`, `W4-65536-zipf`), 2^20 and 2^24
  keys, four keys per thread: atomic-contention-bound. Correct when every bin count equals the serial count exactly.

`--workloads` takes a workload id or a family (`W3` is the three degrees, `W4` the four histograms). Every workload
runs its own size ladder (`gust-bench list`); `--sizes` overrides it for all of them. On these backends:

| id | what it is |
|---|---|
| `gust-gpu` | the GUST kernel through `gust-wgpu`'s public API only (typed call, `dispatch_batch`, `dispatch_batch_timed`), WGSL route |
| `gust-gpu-spirv` | the same, in the `spirv-route` build (the SPIR-V route; see Groups and builds) |
| `wgsl-prebuilt` | hand-written WGSL on plain wgpu, host side built the careful way: bind group and read-back staging buffer built once, one submission per end-to-end sample |
| `wgsl-percall` | the same WGSL, host side shaped like `gust-wgpu` does it: a bind group per dispatch, a fresh staging buffer and `Vec` per read-back, two submissions with a wait after each |
| `gust-twin` | the GUST CPU twin (`--cfg gust_dual`), single thread, a semantic reference with per-call validation (not the production CPU target) |
| `gust-cpu` | the production CPU target (`GUST_PROFILE=cpu` build, no GPU runtime): the kernel's Rust body through `CpuDevice`, single thread, typed dispatch |
| `gust-cpu-par` | the same build's generated element-wise map (`#[gpu] fn`, chunked over scoped threads), W1 and W2 only (a map takes scalars, W3 needs a runtime coefficient array, W4 is a scatter) |
| `rayon-1`, `rayon-N` | `par_iter` over host slices in a pool of 1 thread and of all logical cores |
| `serial` | a plain loop (sanity baseline, not in the spike's brief) |

`wgsl-percall` has three **ablation switches** (plan 2.6), each one turned on alone as its own backend, so a gap to
`wgsl-prebuilt` can be attributed by measurement: `wgsl-percall+bind-group-reuse` (one bind group built once),
`wgsl-percall+staging-reuse` (one read-back staging buffer built once) and `wgsl-percall+one-submission` (dispatch and
copy in one submission, no wait between). They run only in the ablation group (see Groups and builds):
`--group ablation` runs them all, `--group ablation --ablation LIST` a chosen few. The fresh result `Vec` stays per
call in every ablation. A test counts bind groups, staging buffers and submissions per switch.

`--ablation all` also adds `wgsl-percall+all-on`: every switch on, the result `Vec` still fresh per call. The ablation
group then reads `wgsl-percall`, each switch alone, `+all-on` and `wgsl-prebuilt`; the step from `+all-on` to
`wgsl-prebuilt` is the result `Vec`, the step from `wgsl-percall` to `+all-on` is the three switches together.

## Host-side counts and wgpu traces

`gust-bench profile --out p.json [--workloads W1,W2] [--sizes N,N] [--backends a,b] [--samples CALLS]` counts, per call and
warm, the Rust heap allocations (a counting global allocator; the Vulkan driver's own allocations are not seen) and the
wgpu-core API calls (submissions, buffer and bind-group creations, polls, map requests; read from wgpu-core's own `log`
output, which covers GUST's device too) of `end_to_end`, `dispatch_host` and one 16-pass batch, for GUST and every
hand-WGSL backend. It needs `--features profile-counts` (the allocator is not in a timing build; the results file lists
the feature in `build.features`). The numbers are counts, never timings.

Each cell is measured in two passes. The heap numbers (`heap`) come from a pass with logging off, because at trace level
wgpu-core copies resource labels into `String`s to build its log arguments, which the allocator would count. The API
counts and `sequence_of_first_measured_call` come from a second pass with the log tap on; its heap is kept as
`heap_with_tap` for the difference only. The sequence is the whole wgpu-core log message of each call, and it is **not
command order**: wgpu-core logs a pass's contents (set_bind_group, dispatch) when the encoder finishes but a copy when it
is called, and logs no pipeline sets, timestamp writes, `resolve_query_set` or barriers. It cannot show the timestamp
span of a batch.

`gust-bench trace --out DIR [--workloads W1] [--sizes N,N] [--backends wgsl-prebuilt,wgsl-percall,gust-gpu] [--label batch16|end_to_end]`
writes wgpu's API trace of the hand-WGSL device, one directory per backend, workload, size and operation, in a
`--features wgpu-trace` build. GUST's device is opened inside `gust-wgpu` with the trace off and `HeadlessDevice` takes no
trace path, so the `gust-gpu` backend of `trace` works only against a `gust-wgpu` that reads a trace directory when it
opens its device (a `DeviceOptions::trace` hook, an api2b candidate; m10.b1c used a throwaway patch that is not part of
the crate, reading `GUST_WGPU_TRACE_DIR`). Without such a hook `trace --backends gust-gpu` fails with "no trace.ron",
rather than writing an empty directory.

`gust-bench dump-kernels --out DIR` (a `slang-runtime` build) writes each kernel's Slang source, the WGSL and the SPIR-V
that slangc makes of it, for diffing the two routes; the Slang source of a `spirv-route` build is the one that route emits.

## Groups and builds

A run is one **group**, chosen by `--group publishable|ablation` (default `publishable`) and written into the run file
(`group` at the top level and in every cell; `aggregate` refuses to mix groups):

- `publishable`: `gust-gpu`, `wgsl-prebuilt` and `wgsl-percall`, sampled together in seeded random order (below), plus
  the CPU backends run beside them (not in the rotation). This is the group a published table is made of.
- `ablation`: `wgsl-percall`, `wgsl-prebuilt` and the switch ablations, with no GUST backend. Its rows are compared
  only with the `wgsl-percall` and `wgsl-prebuilt` of the same run, never with numbers from a publishable run.
  `--ablation` without `--group ablation` is an error.

The route and the profile are properties of the build, not of a flag, so each build is its own result file, and a run
names its build in the JSON: `route` (`wgsl`, `spirv` or `cpu`), `build.features`, `build.gust_profile`,
`build.cfgs` and `config.shader_source`.

| build | command (from `bench/`) | route | GUST backend ids |
|---|---|---|---|
| default | `GUST_PROFILE=dev cargo build --release -p gust-bench` | `wgsl` | `gust-gpu` (run-time Slang compile) |
| `spirv-route` | the same with `--features spirv-route` | `spirv` | `gust-gpu-spirv` (the macro emits the SPIR-V route) |
| precompiled | `GUST_PROFILE=native cargo build --release -p gust-bench-pre` | `wgsl` | `gust-gpu` (embedded artifacts) |
| `cpu` | `GUST_PROFILE=cpu cargo build --release -p gust-bench --no-default-features` | `cpu` | `gust-cpu`, `gust-cpu-par`, no GPU rows |

Deviation from the plan's single group: the plan puts gust-gpu and gust-gpu on SPIR-V in one random-order group. The
SPIR-V route changes the code `gust-wgpu`'s macros emit (a Cargo feature), so one process holds one GUST route and
the two routes cannot share a rotation. Each build instead runs its own publishable group with its own
`wgsl-prebuilt` and `wgsl-percall`, and the routes are compared through their paired ratios to the `wgsl-prebuilt` of
their own run (the ratio of the two ratios), not through a direct pairing.

**Kill rule** (`group::kill_rule`, in the run file as `kill_rule` for the publishable group): a gust-gpu (or
gust-gpu-spirv) GPU-time (`gpu_single`, `gpu_batch_per_dispatch`) paired ratio to `wgsl-prebuilt` more than 10 % off 1.0
on any cell marks the run "C re-opened" (`kill_rule.c_reopened`, with the offending cells listed). It judges only a complete run: every GUST cell and quantity needs a ratio, and any
cell without one (a failed or dropped backend, a size with no GUST cell) counts in `kill_rule.unpaired`; `kill_rule.judged` is true only when `checked > 0` and `unpaired == 0`, and otherwise `c_reopened` is `null`, never `false`. `kill_rule.control` carries the same ratio for `wgsl-percall` (same kernel text as `wgsl-prebuilt`, so its distance from 1.0 is the noise floor of the method); it is a record beside the rule, not an input, and the rule's threshold and inputs are unchanged. The control is a noise floor for `gpu_single` only. For `gpu_batch_per_dispatch` it also differs from
`wgsl-prebuilt` in the per-pass bind groups (percall builds a fresh one each pass, as GUST does; prebuilt reuses one), so its
batch excess (1.08-1.13 on W1 1M in the m10.b1c run) may be a real effect of that and not noise. A control that is stream-identical for
the batch quantity would be a second `wgsl-prebuilt` instance, or `bind_group_reuse` on; until then "control out of tolerance" voids a
batch cell on a confounded control. A `publishable` run refuses a `--backends` list that leaves out `wgsl-prebuilt`, `wgsl-percall` or the build's GUST backend (use `--group ablation` for subsets). After every GUST cell of a `spirv-route` build the device's route events for that kernel are checked: a cell fails on any event of its own that is off the build's route, and otherwise needs the kernel's latest event over the device's life to exist and be on the route, naming only the build route (the no-event error names no event at all). That check is also written into the cell, as `route_evidence`
(build route; the number of route events for the kernel since this cell's `prepare`, how many of those are on and off the build's
route, and `events_cumulative`, the device's whole-life count; the last event). The events are cumulative on the device (the W3 cells
share `kernel::horner`), so the per-cell counts are the total minus what was there when the cell began. Events arise only when a
pipeline is compiled and the cache is keyed by kernel, so a cell that compiles nothing (for example every cell after a workload's first size) has `events` 0: it is
judged on the event of the pipeline it ran, the kernel's latest event, which an earlier cell's compile made. It is `null` in a build without `spirv-route`, where the runtime keeps no route events. (`cold_routes` stays `[]` unless the run takes cold starts.)

**Reproducibility clause (plan R1.3-4).** Each offender in `kill_rule.offenders` carries `control_median_ratio` and
`control_in_tolerance` for its own cell. An offender counts only when it recurs in every run of the build, with at least 3 runs of distinct
seeds (`run_seed`; two files, or one seed twice, never count), and the control in tolerance on that cell in each of those runs;
`aggregate` writes the count to `kill_rule_reproducibility` (with the seeds, once) and a list to the report. A hit that does not
meet this is listed, not counted. Per-point `per_run_p5_ms`, `per_run_p95_ms`, `per_run_outliers_gt_3mad` and
`two_state_suspect` are written beside the median for the bimodal cells (a p5 or p95 missing from an old run file is `null`, never the
median). `two_state_suspect` is true when either test fires, and both are written (`two_state_heavy_tail`, `two_state_wide_gap`):
the heavy-tail test (on every metric), a fifth of a run's samples beyond 3 MAD, and the gap test, (p95 - p5) / median above 0.10 in some run,
for `gpu_batch_per_dispatch` only. A near-even two-state split inflates the MAD and passes the first, which is why the second exists. The
flag asks for p5-p95 to be reported; the share of samples in each state is not computed.

Calibration of the gap test (plans/m10/probe/b1c/aggregate-w12.json, aggregate-w3.json; `flagcount.py` prints these): in the six m10.b1c
confirming run files the per-run batch gap is 0.004-0.086 on every point the test does not flag, and the points it flags have a run at
0.111-0.30. Those are, in the w12 file, 5 W1 1M points (gust-gpu, percall, prebuilt, gust-setup, gust-shader; gust-device is not flagged),
5 W2 1M points (every hand row) and 2 W2 4M points (gust-gpu, prebuilt); in the w3 file W3-d16 unchecked, W3-d4 prebuilt and W3-d4
unchecked. The W1 1M cell is two-state (about 6.1 and 7.9 us); whether the W2 and W3 cells are two-state is not established. A single dispatch's p5-p95 is 0.3-0.75 of its median from ordinary jitter, which is why
the test is not applied to `gpu_single`, `end_to_end` or `dispatch_host`. The 0.10 threshold sits in the gap between 0.086 and 0.111. It is
not fitted beyond these runs: a cell with a different two-state spacing, or a quieter or noisier machine, needs the data re-read, not the
threshold trusted. On those files the flag fires on 21 of 24 (w12) and 7 of 12 (w3) `gpu_batch_per_dispatch` points, 15 and 4
`gpu_single`, 9 and 2 `end_to_end`, 3 and 3 `dispatch_host`; the gap test by itself fires on 12 and 3 batch points and on no other metric; each of those also has a heavy tail, so the gap test
changes no flag on these files and the totals are the heavy-tail totals.

**Probe backends (m10.b1c).** `--with gust-setup,gust-shader,gust-device,unchecked` (publishable group only; list the ids in
`--backends` too when it is given) adds hand-WGSL rows to the rotation, each `wgsl-prebuilt` plus one named difference from GUST:
`wgsl-prebuilt+gust-device` (a separate wgpu device opened like GUST's, which requests `MEMORY_DECORATION_COHERENT`, with the
timestamp objects created before the data buffers: three differences at once, not separable by this probe), `wgsl-prebuilt+gust-shader`
(the shared hand device with GUST's own generated WGSL as the kernel), `wgsl-prebuilt+gust-setup` (both: the pre-registered
ablation) and `wgsl-prebuilt+unchecked` (the hand WGSL compiled without wgpu's runtime checks, bounds checks and loop bounding, on the
shared device). The kill rule ignores them; they exist to test a named difference between GUST and the hand path, so a ratio
between a probe and GUST is the test, and none of them is a result.

**Small-group run conditions.** A small group (one workload and size, e.g. a diagnostic run of W1 at 1 048 576) is run
with: the GPU idle (check `nvidia-smi` first and write down what else ran), one size, the run-time-compile build, a
fixed `--seed`, `--no-sampler` (the sampler's idle windows are for the full run), 30 samples per quantity, and the
paired seeded random-order rotation above. Such a run is labelled diagnostic in its `--label` and in whatever quotes it.
The report flags every GPU figure above the adapter's DRAM peak (`dram_peak`, `above_dram_peak` in the aggregate, and
`(>peak)` in the bandwidth table; the RTX 5090 peak is 1792 GB/s from NVIDIA's specification, other adapters are not
flagged): a number above it is not a DRAM bandwidth (the data sits in cache, or the timing is not the kernel alone).

The GPU backends of one workload and size are measured as a paired group: one idle window, one shared warm-up, then
every timed quantity in interleaved rounds. Each round visits every GPU backend once, in a seeded random order (a fresh
permutation per round from the run seed mixed with workload, size and quantity; `--seed N` fixes the run seed, which the
run file records as `run_seed`). The file records each quantity's seed, every round's order (`round_orders`) and each
sample's predecessor (`predecessor`), so a carry-over cost between neighbouring backends can be tested for. The
aggregate reports each backend's paired ratio to `wgsl-prebuilt` (the median of the per-round ratios, with its spread)
for all four timings.

The shared warm-up uses real load (end-to-end samples alternating with 1024-pass batches) until the clock query reports
P0 with the memory clock within 90 % and the SM clock within 50 % of their maxima, capped at 1 s (`--warmup-secs S`
changes the cap; the deadline is checked before every iteration, so it is overrun by at most one, and `warmup.actual_s`
records the real time). P0 is a record, not a goal: a group that hits the cap is kept and marked `p0_not_reached`; the
pairing, not P0, is what makes the backends comparable. The clocks at the end of the warm-up (`warmup.clocks`) and right
after the timed samples (`clocks_after_timing`), the PCIe link and the P0 counts are in the aggregate; every end-to-end
table carries the PCIe link. `--cold-only` skips the four timings and measures only the cold start (`--cold-repeats
N`). The precompiled package must pass `cargo gust check -p gust-bench-pre` (measure.sh --build-only runs it): a stale
artifact otherwise falls back to a run-time compile under dev.

Deviation from D179 (b0): the quiet-window sampler idles the GPU once per GPU group (record `"backend": "gpu-group"`),
not once before every cell; CPU backends still get one window each.

The first cold start of each kernel on a backend in a process is flagged (`cold_start_first_in_process`) and the aggregate lists it
apart from the rest: it also pays one-time costs. A hand-WGSL cold start is labelled "driver shader cache warm" (its
pipeline was already created once on the same device by `prepare`).

**Precompiled cold start.**

The `#[gpu]` macro embeds artifacts found in `gust-artifacts/` beside the Cargo.toml of the package that compiles it.
`gust-bench` has none, so it compiles Slang at run time. `gust-bench-pre` is the same harness sources (its binary
`gust-bench-pre` has `path = "../gust-bench/src/main.rs"`) in a package of its own, with the committed artifacts in
`gust-bench-pre/gust-artifacts/` (all four kernels: add, saxpy, horner, histogram); a build of it embeds them and its
results say `config.shader_source = "precompiled"`. It is built with `GUST_PROFILE=native`, which refuses
`slang-runtime` and requires an artifact for every kernel, so in that build the artifact path is the only possible one.
To regenerate them after a kernel changes: `cargo gust build --manifest-path bench/gust-bench-pre/Cargo.toml` (the
pinned Slang and its helper), check with `cargo gust check --manifest-path bench/gust-bench-pre/Cargo.toml`, then
commit the directory. Run it with
`gust-bench-pre run --workloads W1,W2 --sizes 4096 --backends gust-gpu,wgsl-prebuilt,wgsl-percall --cold-only --cold-repeats 10
--no-sampler --out FILE`.

**Route event per cold start.** Every cold start of a GUST backend records which route produced its shader, in the
cell's `cold_routes` (one record per cold-start sample): `path` is `artifact` or `run-time compile`, `shown` says
whether it is shown or inferred, and `evidence` names what showed it, in this order of strength: a route event from the
runtime (`route-event`, the `spirv-route` build), then the strict profile (`strict-profile-build`: under
`GUST_PROFILE=native` or `web` the build refuses any other path), then a kernel with no embedded artifact
(`no-embedded-artifact`: only a run-time compile is possible). Only a record built from the kernel's descriptor and
the policy alone is an inference (`shown: false`). A test (`a_cold_start_records_the_route_that_ran`, which runs in
`gust-bench-pre` as the precompiled-route test) checks that a kernel with embedded artifacts reports the artifact route
as shown.

The hand-WGSL device is a second `wgpu::Device` on the same adapter selection (Vulkan, high performance, default
limits, `TIMESTAMP_QUERY`), because `HeadlessDevice` does not expose its own device. The harness refuses to run if the
two devices land on different adapters, or if the adapter is a software one unless `--functional` is given.

## Build and run

The bench is its own workspace (`bench/Cargo.toml`), so run cargo from `bench/` or pass `--manifest-path bench/Cargo.toml`.
The GUST CPU twin exists only in a `--cfg gust_dual` build (D26: a cfg, never a Cargo feature); the recipe below is the
build line of `docs/BENCHMARKS.md`, without it (add `RUSTFLAGS="--cfg gust_dual"` to the build for a `gust-twin` row).

`gust-compute` refuses a release build that enables `slang-runtime` unless `GUST_PROFILE=dev` is set (the bench
compiles its kernels at run time), so every cargo command here runs with `GUST_PROFILE=dev`.

Build from a clean tree at the commit you will run, and write the outputs outside the work tree (`OUT` below): a file
written inside it makes the tree dirty and the run is then not publishable.

```sh
cd bench
GUST_PROFILE=dev cargo build --release --locked -p gust-bench
# the native Slang helper is found through GUST_SLANG_REFLECT, as for the main workspace
./target/release/gust-bench run --out "$OUT/run1.json" --label run1 --seed 1   # about 30 minutes (the large sizes poison and verify 64 MiB per sample); repeat for run2.json, run3.json with other seeds
./target/release/gust-bench aggregate --json "$OUT/results.json" --md "$OUT/results.md" "$OUT/run1.json" "$OUT/run2.json" "$OUT/run3.json"
./target/release/gust-bench dump-wgsl                 # the WGSL GUST generates for each kernel
./target/release/gust-bench list
```

`run` options: `--quick` (two small sizes, 5 samples, for smoke tests), `--workloads W1,W2`, `--sizes N,N`,
`--backends gust-gpu,rayon-N`, `--group publishable|ablation`, `--samples N`, `--cap-secs S`, `--label NAME`,
`--ablation LIST|all` (ablation group only), `--seed N`, `--settle-secs S`, `--window-secs S`, `--warmup-secs S`,
`--no-sampler`, `--functional` (correctness only, below). Every
path comes from an argument. `aggregate` also refuses to mix runs with and without the sampler (`sampler.enabled`). A build without `--cfg gust_dual` still works and simply has no `gust-twin` row. The builds
of the Groups and builds section (spirv-route, precompiled, cpu) take the same options; the `cpu` build runs the CPU
backends only; its sampler has no adapter to map, so every adapter's activity counts (the stricter fallback).

The build script of `gust-bench` (shared by `gust-bench-pre`) embeds the git HEAD at compile time, so build from the
commit you will run; the binary is `bench/target/release/gust-bench` (`bench/target/` is gitignored).

`--functional` is the CI-safe correctness run: quick sizes, no sampler, a software adapter (lavapipe) allowed, the
file marked `"functional": true` and never publishable. `cargo xtask check-bench --functional --quick` runs it.

## The quiet-window sampler and the publishable judgement

Before each cell the harness waits a 3 s settle interval and then samples a 2 s window at 4 Hz (8 samples): the
Windows performance counter `\GPU Engine(*)\Utilization Percentage` for every engine instance (the instance name
carries the owning pid and the adapter LUID) and the process table (pid, parent pid, name, CPU time, creation time)
at the window's start and end, read through ToolHelp and `GetProcessTimes` so that no helper process is spawned. The
harness's PID ancestry (itself, its shell-like parent chain up to the first process that is not cargo or a shell, and
the children of that chain) is computed and recorded. Everything goes to the sidecar `<results>.samples.json`; the
results file names it. Each window is judged (`judge.rs`): a foreign process (not the harness, its ancestry or its
descendants) above `T_gpu` on the adapter under test, or a named process (cargo, rustc, slangc, a model server, python,
node, `vmmem`) that used CPU or started or ended in the window, makes it not quiet; the cell is retried up to `N`
times and every attempt is in the sidecar. `T_cpu` is not set (no `vmmem` data), so a `vmmem` or `vmmemWSL` process in
the walk makes a window not quiet; a window recorded with the WSL service up (no running distribution confirmed) listed
neither, so whether the walk lists WSL's VM process by name is not established. The judge fails closed on a row it cannot read, an unparsed counter instance and an `invalid` sample
of a counter that matters. A run that changes the settle (`--settle-secs`), the window (`--window-secs`), the sample
cap (`--cap-secs`) or the warm-up (`--warmup-secs`), asks for fewer than 30 samples, or whose compiled-in git HEAD
differs from the run-time HEAD (or either is unknown), or whose build had uncommitted changes to tracked files (or
whose build state is unknown), gets a named blocker and is never publishable; a quantity with
fewer than 5 collected samples blocks its cell. Every cell carries `publishable` with a reason; the run file carries the run-level `publishable`, the
banner when it is false, its start and end times and the git state at both ends. `aggregate` judges the set (three
runs, one session, distinct seeds, every sidecar present) and writes the merged sidecar beside its JSON. The
thresholds and their data are in docs/BENCHMARKS.md.
The adapter under test is mapped to its LUID through DXGI (vendor and device id, then name); when that fails the
header says so and every adapter's activity is recorded. `gust-bench probe-adapters` verifies the mapping on a machine:
it keeps the adapter busy and reports on which LUID this process's own utilisation appears.

Every results file also carries `route` (`wgsl` or `spirv`, read from the build), `build` (release, profile, features,
cfgs, the commit and whether the tree was dirty) and `adapter` (name, backend, driver) at its top level. `aggregate`
refuses runs that disagree on route, build, adapter, group or config (quick against full, sizes, workloads, backends and
their ablations). The `nvidia-smi` cross-check of a window is taken after the settle sleep and before the window
starts.

Exit status is 0 only when every correctness check and every cache assertion passed.

## What is measured

Four quantities per cell (backend, workload, size), each sampled up to 30 times after a warm-up (3 end-to-end and 5
resident dispatches and one timestamped batch, unmeasured). A quantity stops early only when its 8 s wall cap runs out
(never below 5 samples); the sample count is in the JSON.

1. **`end_to_end`**: host inputs in, host output out, everything complete. GPU: write both inputs, dispatch, read
   back. The CPU backends have no transfer: this is the computation on host memory (the output vector is allocated
   outside the timed region). `gust-twin` writes its buffers, dispatches and copies the output out, like the GPU path.
2. **`dispatch_host`**: one dispatch on resident data, host wall clock from the call to completion. For `gust-gpu`
   this includes everything `dispatch_batch` does per call (validation, pipeline-cache lookup, bind-group creation,
   encode, submit, wait).
3. **`gpu_single`**: GPU timestamps around one compute pass (the pass-bracketing query that `dispatch_batch_timed`
   uses; `wgsl-*` use the same query writes on their own pass).
4. **`gpu_batch_per_dispatch`**: timestamps from the first to the last of 16 back-to-back passes in one submission,
   divided by 16. This is the like-for-like device figure for small sizes, where one pass is dominated by fixed cost.

"Steady state" means after the warm-up with the pipeline cache warm; the harness asserts that GUST's pipeline-cache
`misses` did not move across the timed samples, and the run fails if it did. The **cold start** is reported separately:
the pipeline state is dropped (`clear_pipeline_cache()` for GUST, a rebuilt pipeline for the hand paths) and the first
dispatch is timed, compile included. GUST's cold start includes launching the native Slang helper.

Correctness: before every sample the device output is overwritten with NaN (outside the timed region and flushed),
and after every sample the output is read back and compared with a serial reference. W1 must match bit for bit; W2 is
allowed two f32 epsilons of `|scale*x| + |y|`, because a GPU compiler may fuse the multiply-add; W3 must stay inside
its Higham bound; W4 must match the serial counts exactly (its bins start at zero, because the kernel adds into them, so a dispatch
that does not run reads back all zeros and fails). A NaN never passes.
A failed check or a moved cache counter marks the run as failed.

Statistics are median, min, p5, p95, MAD and the count of samples beyond 3 MAD (counted, never dropped). The aggregate
reports the median over the per-run medians of several runs and the min-max spread of those medians.

## Sizes

W1 and W2: `4096, 65536, 1048576, 4194304, 16776960` elements. W3: `1048576, 4194304`. W4: `1048576, 16777216` keys (four
keys per thread, so 2^24 keys are 16 384 workgroups). Workgroup size 256 everywhere. The default wgpu limits cap a
one-dimensional dispatch at 65535 workgroups: with 64-wide groups the ladder would stop at 4 194 240, just below
2^22, so 256-wide groups are used, and the largest launch is 65535 x 256 = 16 776 960 (2^24 itself needs 65536
workgroups and is rejected). Three f32 streams of the last size are 192 MiB, beyond the RTX 5090's 96 MB L2.

## Caveats

- Results from a shared machine are labelled "shared machine, not publishable"; other lanes run cargo and GPU tests.
  The JSON carries an `nvidia-smi` snapshot before and after every cell and process counts before and after a run.
- The harness never pins clocks and never stops other services.
- Results and the metadata record adapter, driver, OS, CPU and versions, and no paths or user names. Results go
  where the caller points `--out`, never into the repository; point it outside the work tree, because a run that
  leaves the tree dirty is not publishable.
- `gust-gpu` and the hand WGSL both use wgpu's checked shader path (`create_shader_module`), so robustness checks are
  on for both. rayon uses safe zipped slice iterators (no per-element bounds checks); overflow checks are off in the
  release profile for every backend.
- On a CPU the three streams of the largest sizes are memory bound; rayon-N did not beat one thread there on this
  machine (see the results), so the CPU rows say as much about DRAM bandwidth as about the schedulers.
- Not covered by this spike: workloads W5 to W9, other comparators (CubeCL, rust-gpu), multi-dimensional dispatch.
- W4's histogram accumulates across dispatches (a batch of passes keeps counting); the harness counts the passes since
  the bins were cleared and divides the counts by that number, and a count that is not a multiple of it fails the check.
  The bytes per element of W4 (4) count the key stream only, so its GB/s is the compulsory traffic, not the atomics'.
