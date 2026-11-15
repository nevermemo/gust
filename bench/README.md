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
warm, the heap allocations (a counting global allocator) and the wgpu-core API calls (submissions, buffer and bind-group
creations, polls, map requests; read from wgpu-core's own `log` output, which covers GUST's device too) of `end_to_end`,
`dispatch_host` and one 16-pass batch, for GUST and every hand-WGSL backend. It needs `--features profile-counts` (the
allocator is not in a timing build). The numbers are counts, never timings.

`gust-bench trace --out DIR [--workloads W1] [--sizes N,N] [--backends wgsl-prebuilt,wgsl-percall] [--label batch16|end_to_end]`
writes wgpu's API trace of the hand-WGSL device, one directory per backend, workload, size and operation, in a
`--features wgpu-trace` build. GUST's device is opened inside `gust-wgpu` with the trace off and `HeadlessDevice` takes no
trace path, so GUST has no trace from outside the crate.

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
cell without one (a failed or dropped backend, a size with no GUST cell) counts in `kill_rule.unpaired`; `kill_rule.judged` is true only when `checked > 0` and `unpaired == 0`, and otherwise `c_reopened` is `null`, never `false`. `kill_rule.control` carries the same ratio for `wgsl-percall` (same kernel text as `wgsl-prebuilt`, so its distance from 1.0 is the noise floor of the method); it is a record beside the rule, not an input, and the rule's threshold and inputs are unchanged. The judgement that
uses it is b1b's. A `publishable` run refuses a `--backends` list that leaves out `wgsl-prebuilt`, `wgsl-percall` or the build's GUST backend (use `--group ablation` for subsets). After every GUST cell of a `spirv-route` build the device's route events for that kernel are checked: any event off the build's route (or none at all) fails the cell, naming both routes.

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
two devices land on different adapters or if the adapter is a software one.

## Build and run

The bench is its own workspace (`bench/Cargo.toml`), so run cargo from `bench/` or pass `--manifest-path bench/Cargo.toml`.
The GUST CPU twin exists only in a `--cfg gust_dual` build (D26: a cfg, never a Cargo feature), so build with it:

`gust-compute` refuses a release build that enables `slang-runtime` unless `GUST_PROFILE=dev` is set (the bench
compiles its kernels at run time), so every cargo command here runs with `GUST_PROFILE=dev`.

```sh
cd bench
GUST_PROFILE=dev RUSTFLAGS="--cfg gust_dual" cargo build --release --locked
# the native Slang helper is found through GUST_SLANG_REFLECT, as for the main workspace
./target/release/gust-bench run --out run1.json       # about 30 minutes (the large sizes poison and verify 64 MiB per sample); repeat for run2.json, run3.json
./target/release/gust-bench aggregate --json results.json --md results.md run1.json run2.json run3.json
./target/release/gust-bench dump-wgsl                 # the WGSL GUST generates for each kernel
./target/release/gust-bench list
```

`run` options: `--quick` (two small sizes, 5 samples, for smoke tests), `--workloads W1,W2`, `--sizes N,N`,
`--backends gust-gpu,rayon-N`, `--group publishable|ablation`, `--samples N`, `--cap-secs S`, `--label NAME`,
`--ablation LIST|all` (ablation group only), `--seed N`, `--settle-secs S`, `--window-secs S`, `--no-sampler`. Every
path comes from an argument. `aggregate` also refuses to mix runs with and without the sampler (`sampler.enabled`). A build without `--cfg gust_dual` still works and simply has no `gust-twin` row. The builds
of the Groups and builds section (spirv-route, precompiled, cpu) take the same options; the `cpu` build runs the CPU
backends only and has no sampler.

## The quiet-window sampler (recording only)

Before each cell the harness waits a 3 s settle interval and then samples a 2 s window at 4 Hz (8 samples): the
Windows performance counter `\GPU Engine(*)\Utilization Percentage` for every engine instance (the instance name
carries the owning pid and the adapter LUID) and the process table (pid, parent pid, name, CPU time, creation time)
at the window's start and end, read through ToolHelp and `GetProcessTimes` so that no helper process is spawned. The
harness's PID ancestry (itself, its shell-like parent chain up to the first process that is not cargo or a shell, and
the children of that chain) is computed and recorded. Everything goes to the sidecar `<results>.samples.json`; the
results file names it and carries `"publishable": false` with a reason, because no judgement or threshold exists yet.
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
  where the caller points `--out`, never into the repository.
- `gust-gpu` and the hand WGSL both use wgpu's checked shader path (`create_shader_module`), so robustness checks are
  on for both. rayon uses safe zipped slice iterators (no per-element bounds checks); overflow checks are off in the
  release profile for every backend.
- On a CPU the three streams of the largest sizes are memory bound; rayon-N did not beat one thread there on this
  machine (see the results), so the CPU rows say as much about DRAM bandwidth as about the schedulers.
- Not covered by this spike: workloads W5 to W9, other comparators (CubeCL, rust-gpu), multi-dimensional dispatch.
- W4's histogram accumulates across dispatches (a batch of passes keeps counting); the harness counts the passes since
  the bins were cleared and divides the counts by that number, and a count that is not a multiple of it fails the check.
  The bytes per element of W4 (4) count the key stream only, so its GB/s is the compulsory traffic, not the atomics'.
