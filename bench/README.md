# GUST benchmark harness (M10 spike B1)

A custom harness in a nested, unpublished workspace. It measures six workload families:

- **W1 vector-add** and **W2 saxpy** (f32, memory-bandwidth bound), 4096 to 16 776 960 elements.
- **W3 Horner polynomial** (f32, `W3-d4`, `W3-d16`, `W3-d64`), 2^20 and 2^22 elements: arithmetic-bound, `2 d` flops per
  8 bytes. Correct when every element is inside a Higham error bound against an f64 reference (derived in
  `gust-bench/src/workloads/polynomial.rs`); a test shows the bound holds for a backend that fuses multiply-add and
  fails for a perturbed result.
- **W4 u32 atomic histogram** (`W4-256-uniform`, `W4-256-zipf`, `W4-65536-uniform`, `W4-65536-zipf`), 2^20 and 2^24
  keys, four keys per thread: atomic-contention-bound. Correct when every bin count equals the serial count exactly.
- **W7 sum** (`W7-u32` wrapping, `W7-f32`), 2^20 to 2^27 values, one output word. GUST runs `gust-algo`'s public sum (the
  `algo` feature, in `gust-bench` only: `gust-bench-pre` has no W7 GUST backend, since gust-algo ships no precompiled
  artifacts); the hand WGSL is a two-dispatch workgroup-tree reduction whose grid grows with `n`. `u32` is exact; `f32` is
  checked against a bound for the order of summation (GUST's library bound for its tree, the hand tree's own depth bound
  for the hand WGSL, the serial worst case for `serial` and rayon; derived in `gust-bench/src/workloads/sum.rs`, with the
  sizes where the serial worst case says nothing, `docs/BENCHMARKS.md`). The two largest sizes need the adapter's own
  storage-binding limit, which a run asks for only when it includes them (and then for every workload of that run). GUST
  and the hand WGSL run different algorithms on W7 and GUST's sum exposes no GPU timestamps, so W7 has no GUST/hand ratio
  in the report, and its cells are outside the kill rule and listed in `kill_rule.excluded`.
- **W5 particle step** (`W5`), 2^16 and 2^20 particles (a 28-byte std430 struct element: nested position and velocity pairs, a mass, a `u32` tag, an `i32` charge), 16 explicit-Euler steps per invocation. Backends: GUST (and the SPIR-V route), `gust-twin`, `gust-cpu`, rayon, serial and the hand WGSL (`shaders/particles.wgsl`, same layout, asserted against GUST's reflection). The integers and the mass are bit-exact; each float field is inside a bound derived from the expression tree (`gust-bench/src/workloads/particles.rs`, `docs/BENCHMARKS.md`). The size is the particle count.
- **W6 staged pipeline** (`W6-graph` and `W6-hand`), `signal-pipeline`'s six statements as six kernels in a chain, `f32`, 2^20 and 2^22 elements (quick: 2^20), five input streams, a buffer between stages, one output. `W6-graph` is GUST's `StagedGraph` (graph build, validation and execution inside every timed sample, no GPU timestamps); `W6-hand` is the same six calls as one `dispatch_batch`, no graph. Backends: `gust-gpu` (both forms, and the SPIR-V route build), `gust-twin` and `gust-cpu` (`W6-hand` only: they have no graph), rayon, serial and the hand WGSL (`shaders/signal_pipeline.wgsl`, six entry points, six passes, one submit; the baseline of both forms). The last stage is checked against a bound derived from the expression tree (`gust-bench/src/workloads/signal_pipeline.rs`, `docs/BENCHMARKS.md`), and a unit test checks every stage. The size is the element count.
- **W9 naive matrix product** (`W9-naive`), `N x N` f32 with `N` 256, 512, 1024, 2048, one output entry per invocation (a
  16 by 16 workgroup, a 2-D dispatch). Backends: GUST, hand WGSL, rayon, serial, and `gust-twin` in a `--cfg gust_dual`
  build (and `gust-gpu-spirv` in the `spirv-route` build). Correct when every entry is inside Higham's inner-product bound
  (derived in `gust-bench/src/workloads/matmul.rs`; `u = 2^-24` for the CPU backends, `2^-23` for every GPU backend). The
  size is the matrix side `N`, not an element count.

`--adapter "<name substring>"` opens both devices (GUST's and the hand wgpu one) on the one Vulkan adapter whose name contains the text, case-insensitively; no match or several matches is an error listing the adapters found, and without the flag the harness keeps its high-performance choice. The selection is recorded in the run file (`meta.adapter_selection`, and the hand device's own adapter in `meta.hand_wgsl_adapter_info`; the run fails if the devices, or the selection, are not on one adapter by name, vendor, device, backend and PCI bus id). Two adapters with identical names cannot be told apart by name, so `--adapter` reports them as ambiguous and cannot select one. `profile` and `trace` open their own devices on the default adapter and refuse the flag.

`--workloads` takes a workload id or a family (`W3` is the three degrees, `W4` the four histograms, `W5` the particle step, `W6` the two staged-pipeline forms, `W7` the two sums). Every workload
runs its own size ladder (`gust-bench list`); `--sizes` overrides it for all of them (for W9 the values are matrix sides,
and a value above 4096 is refused beside W9). On these backends:

| id | what it is |
|---|---|
| `gust-gpu` | the GUST kernel through `gust-wgpu`'s public API only (typed call, `dispatch_batch`, `dispatch_batch_timed`), WGSL route |
| `gust-gpu-spirv` | the same, in the `spirv-route` build (the SPIR-V route; judged in the `spirv` group, see Groups and builds) |
| `wgsl-prebuilt` | hand-written WGSL on plain wgpu, host side built the careful way: bind group and read-back staging buffer built once, one submission per end-to-end sample |
| `wgsl-percall` | the same WGSL, host side shaped like `gust-wgpu` does it: a bind group per dispatch, a fresh staging buffer and `Vec` per read-back, two submissions with a wait after each |
| `wgsl-prebuilt-unchecked`, `wgsl-percall-unchecked` | the `spirv` group's hand rows (D193, m10.b3d): `wgsl-prebuilt` and `wgsl-percall` with the shader module compiled without wgpu's runtime checks (bounds checks, loop bounding), as the SPIR-V route's shaders are; the first is the group's reference, the second its control. They are no `+` ids: they are table rows of their group, not probes |
| `rustgpu-wgpu` | the `rust-gpu` build only: the same four kernels (W1 to W4) written in Rust with rust-gpu 0.11.0, built on a pinned nightly to a committed SPIR-V module and loaded through wgpu's passthrough route, on the host side of `wgsl-prebuilt` (the rust-gpu comparator, below) |
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

A run is one **group**, chosen by `--group publishable|spirv|ablation` (default `publishable` in the WGSL build, `spirv` in the `spirv-route` build) and written into the run file
(`group` at the top level and in every cell; `aggregate` refuses to mix groups):

- `publishable`: `gust-gpu`, `wgsl-prebuilt` and `wgsl-percall`, sampled together in seeded random order (below), plus
  the CPU backends run beside them (not in the rotation). This is the group a published table is made of.
- `spirv` (D193, m10.b3d; the `spirv-route` build only): `gust-gpu-spirv`, `wgsl-prebuilt-unchecked` (the reference of its
  ratios) and `wgsl-percall-unchecked` (its control), sampled together in seeded random order, plus the CPU backends beside
  them. The SPIR-V route skips wgpu's runtime shader checks (its own guards, D174), so it is judged against hand WGSL that
  skips them too: **checks are off on both sides**. It has its own kill rule (below) and is never compared with the
  publishable group's rows. The `spirv-route` build refuses `--group publishable`, the WGSL build refuses `--group spirv`.
- `ablation`: `wgsl-percall`, `wgsl-prebuilt` and the switch ablations, with no GUST backend. Its rows are compared
  only with the `wgsl-percall` and `wgsl-prebuilt` of the same run, never with numbers from a publishable run.
  `--ablation` without `--group ablation` is an error.

The route and the profile are properties of the build, not of a flag, so each build is its own result file, and a run
names its build in the JSON: `route` (`wgsl`, `spirv` or `cpu`), `build.features`, `build.gust_profile`,
`build.cfgs` and `config.shader_source`.

| build | command (from `bench/`) | route | GUST backend ids |
|---|---|---|---|
| default | `GUST_PROFILE=dev cargo build --release -p gust-bench` | `wgsl` | `gust-gpu` (run-time Slang compile) |
| `spirv-route` | the same with `--features spirv-route` | `spirv` | `gust-gpu-spirv` (the macro emits the SPIR-V route); runs the `spirv` group |
| precompiled | `GUST_PROFILE=native cargo build --release -p gust-bench-pre` | `wgsl` | `gust-gpu` (embedded artifacts) |
| `cpu` | `GUST_PROFILE=cpu cargo build --release -p gust-bench --no-default-features` | `cpu` | `gust-cpu`, `gust-cpu-par`, no GPU rows |
| `rust-gpu` | the default build with `--features rust-gpu` | `wgsl` | `gust-gpu`, and `rustgpu-wgpu` beside the hand rows (not a table: the run is marked not publishable, see below) |

Deviation from the plan's single group: the plan put gust-gpu and gust-gpu on SPIR-V in one random-order group. The
SPIR-V route changes the code `gust-wgpu`'s macros emit (a Cargo feature), so one process holds one GUST route and
the two routes cannot share a rotation. Since D193 they also should not: the SPIR-V route skips wgpu's runtime checks,
so each build runs a group of its own with hand rows that have the same checks as its GUST route (`publishable` against
checked `wgsl-prebuilt`, `spirv` against `wgsl-prebuilt-unchecked`), and the two routes are not compared with each other.
A run file of the SPIR-V route in the `publishable` group, or a `publishable` file with a `gust-gpu-spirv` cell, is a
pre-D193 file: `aggregate` refuses it, and the xtask readers (`check-bench --render`) read it as legacy and label it.

**Kill rule** (`group::kill_rule(group, cells)`, in the run file as `kill_rule` for each table group, with `group` and
`reference` in the record): in `publishable`, a `gust-gpu` GPU-time (`gpu_single`, `gpu_batch_per_dispatch`) paired ratio to `wgsl-prebuilt` more than 10 % off 1.0
on any cell marks the run "C re-opened" (`kill_rule.c_reopened`, with the offending cells listed; each offender has a `direction`). In `spirv` (D193) the same rule reads `gust-gpu-spirv` against `wgsl-prebuilt-unchecked` with `wgsl-percall-unchecked` as the control, and the direction splits (plan R1.3-3): slower than 1.10 sets `c_reopened`, faster than 0.90 sets `reading_step` and does not re-open C; it asks for a reading of the cell's cause (as the W3 reading of b1q/b1c did). `gust-gpu-spirv` is not judged in `publishable`. A workload with no same-kernel
GUST/hand pairing (`Workload::same_kernel`, W7) is outside the rule and every such cell is listed in `kill_rule.excluded`
(`backend`, `workload`, `n`, `reason`), not silently absent. It judges only a complete run: every other GUST cell and quantity needs a ratio, and any
cell without one (a failed or dropped backend, a size with no GUST cell) counts in `kill_rule.unpaired`; `kill_rule.judged` is true only when `checked > 0` and `unpaired == 0`, and otherwise `c_reopened` is `null`, never `false`. `kill_rule.control` carries the same ratio for the group's control (`wgsl-percall`, or `wgsl-percall-unchecked`; same kernel text as the reference, so its distance from 1.0 is the noise floor of the method); it is a record beside the rule, not an input, and the rule's threshold and quantities are unchanged. The control is a noise floor for `gpu_single` only. For `gpu_batch_per_dispatch` it also differs from
`wgsl-prebuilt` in the per-pass bind groups (percall builds a fresh one each pass, as GUST does; prebuilt reuses one), so its
batch excess (1.08-1.13 on W1 1M in the m10.b1c run) may be a real effect of that and not noise. A control that is stream-identical for
the batch quantity would be a second `wgsl-prebuilt` instance, or `bind_group_reuse` on; until then "control out of tolerance" voids a
batch cell on a confounded control. A `publishable` run refuses a `--backends` list that leaves out `gust-gpu`, `wgsl-prebuilt` or `wgsl-percall`, and a `spirv` run one that leaves out `gust-gpu-spirv`, `wgsl-prebuilt-unchecked` or `wgsl-percall-unchecked` (use `--group ablation` for subsets). After every GUST cell of a `spirv-route` build the device's route events for that kernel are checked: a cell fails on any event of its own that is off the build's route, and otherwise needs the kernel's latest event over the device's life to exist and be on the route, naming only the build route (the no-event error names no event at all). That check is also written into the cell, as `route_evidence`
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

**Probe backends (m10.b1c).** `--with gust-setup,gust-shader,gust-device` (publishable group only; list the ids in
`--backends` too when it is given) adds hand-WGSL rows to the rotation, each `wgsl-prebuilt` plus one named difference from GUST:
`wgsl-prebuilt+gust-device` (a separate wgpu device opened like GUST's, which requests `MEMORY_DECORATION_COHERENT`, with the
timestamp objects created before the data buffers: three differences at once, not separable by this probe), `wgsl-prebuilt+gust-shader`
(the shared hand device with GUST's own generated WGSL as the kernel), `wgsl-prebuilt+gust-setup` (both: the pre-registered
ablation). The m10.b1c probe `unchecked` (the hand WGSL compiled without wgpu's runtime checks, bounds checks and loop bounding) is gone from `--with`: it is the `spirv` group's own `wgsl-prebuilt-unchecked` row now (m10.b3d, D193; `--with unchecked` is an error pointing at `--group spirv`), where it is the reference and not a probe. The kill rule ignores the others; they exist to test a named difference between GUST and the hand path, so a ratio between a probe and GUST is the test, and none of them is a result.

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
`gust-bench-pre/gust-artifacts/` (every kernel the build has: add, saxpy, horner, histogram, matmul and step, and the six W6 stage kernels: center, amplify, shift, rescale, energy and combine); a build of it embeds them and its
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

## The rust-gpu comparator (`rustgpu-wgpu`, `--features rust-gpu`)

Kernels written in Rust with rust-gpu 0.11.0 (`spirv-std` and `spirv-builder` `=0.11.0`; the project describes itself as
"not yet production-ready"), run on plain wgpu next to the hand WGSL. It is a comparator for W1 to W4 only; there is no
rust-gpu kernel for W5 to W9, and those workloads leave the backend out.

- `bench/rust-gpu-shaders/shader`: the `no_std` shader crate. Four kernels, each the algorithm, workgroup size (256) and
  storage-buffer binding order of the hand WGSL of its workload (`bench/shaders/*.wgsl`): `w1_f32`, `w2_saxpy`,
  `w3_horner`, `w4_histogram`, and `w1_u32`, which is built and validated with the module but run by no workload (the
  bench's W1 is `f32`). The doc comment of each kernel says how it differs from its hand WGSL (the W2 scalar is a
  one-element read-only storage buffer as in the hand WGSL, not a uniform; W4's add is `OpAtomicIAdd` on a `&mut [u32]`).
- `bench/rust-gpu-shaders/builder`: the builder package (`spirv-builder`, target `spirv-unknown-vulkan1.1`).
- Both are outside the bench workspace (a package and workspace root each, with their own `rust-toolchain.toml` and
  `Cargo.lock`), because they build with the nightly `nightly-2026-08-15` (components `rust-src`, `rustc-dev`,
  `llvm-tools`) while the bench stays on stable and `--locked`. Install it once with
  `rustup toolchain install nightly-2026-08-15 --profile minimal --component rust-src,rustc-dev,llvm-tools`; the rustup
  default toolchain is not changed.
- The built module is committed: `bench/rust-gpu-shaders/rustgpu.spv` and `rustgpu.manifest.json` (toolchain, crate
  versions, target, entry points, size and sha256). A normal bench build never runs the builder: `--features rust-gpu`
  embeds the committed bytes (`include_bytes!`), and a test checks them against the manifest.

Rebuild (from the repository root; the builder's inner cargo needs `RUSTUP_TOOLCHAIN` in the environment, a
`rust-toolchain.toml` is not enough when the builder is started from another directory):

```sh
RUSTUP_TOOLCHAIN=nightly-2026-08-15 cargo run --locked --release \
  --manifest-path bench/rust-gpu-shaders/builder/Cargo.toml -- bench/rust-gpu-shaders/shader OUT_DIR
# writes OUT_DIR/rustgpu.spv; use bench/rust-gpu-shaders as OUT_DIR to replace the committed module
cmp OUT_DIR/rustgpu.spv bench/rust-gpu-shaders/rustgpu.spv   # a rebuild reproduces the committed bytes
sha256sum OUT_DIR/rustgpu.spv   # if the module changed, update sha256 and bytes in rustgpu.manifest.json by hand
"$VULKAN_SDK/Bin/spirv-val" --target-env vulkan1.1 bench/rust-gpu-shaders/rustgpu.spv
git status --short bench/rust-gpu-shaders   # both Cargo.lock files must be unchanged (--locked enforces it)
```

The first build compiles `rustc_codegen_spirv` and about 170 crates (several minutes); later builds take under a
minute. A rebuild reproducing the committed bytes has been shown on one machine (the build machine); a test in the
default build checks the committed bytes against the manifest and the toolchain and crate pins against the two
`rust-toolchain.toml` and `Cargo.toml` files, but only a rebuild shows that the sources produce the bytes.

The backend opens its own wgpu device (the run's third, after GUST's and the hand-WGSL one) on the run's adapter
selection, only when the run selects the backend, with `Features::PASSTHROUGH_SHADERS`, and loads the module through
`create_shader_module_passthrough`, the call GUST's own SPIR-V route uses. Its host side is `wgsl-prebuilt`'s.

The module declares the Vulkan memory model (rust-gpu's Vulkan targets always do), and wgpu-hal 30.0.2 creates the
device with `vulkanMemoryModel` off unless cooperative matrices are requested. That breaks a Vulkan valid-usage rule
(VUID-VkShaderModuleCreateInfo-pCode-08740), so the behaviour is undefined by the spec; it ran correctly on the two
adapters tried. Under the Khronos validation layer (Vulkan SDK 1.4.357) a functional run reports it for this
module: VUID-VkShaderModuleCreateInfo-pCode-08740 and, for the queue-family scope of W4's atomic,
VUID-RuntimeSpirv-vulkanMemoryModel-06266. The backend therefore runs only on a physical device that reports
Vulkan 1.3 or later, where the feature must be supported, and is refused on any other. Making the device enable the
feature (an own Vulkan device creation, or an upstream wgpu change) is not done and is a condition of any publishable
rust-gpu row (`docs/BENCHMARKS.md`).

A refusal (a missing capability: the Vulkan 1.3 gate, or passthrough shaders, a defensive check because wgpu-hal sets
them on every Vulkan adapter) leaves the backend out of the run, and the run file lists it under
`config.backends_not_run` with the reason; naming it with `--backends` is then an error, and any other failure to open the
device fails the run. A run file records the module's identity under `versions.rust_gpu`.

The `rust-gpu` feature is listed in `build.features` and makes a run not publishable: a third device and one more
backend in the rotation have not been shown to leave the other rows unchanged. The mark is lifted by a run where the
rust-gpu device cannot disturb the others (the backend in a group or a process of its own, or a measured no-interference
result) together with a device that enables `vulkanMemoryModel`.

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
mkdir -p "$OUT"
./target/release/gust-bench run --out "$OUT/run1.json" --label run1 --seed 1   # long (the large sizes poison and verify 64 MiB per sample; W7 and W9 lengthen it, not re-measured); repeat for run2.json, run3.json with other seeds
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
and after every sample the output is read back and compared with a serial reference. W1 must match bit for bit; W2 must stay inside the derived
two-operation bound of `docs/BENCHMARKS.md` (a GPU compiler may fuse the multiply-add, which is inside it); W3 must stay inside
its Higham bound; W4 must match the serial counts exactly (its bins start at zero, because the kernel adds into them, so a dispatch
that does not run reads back all zeros and fails); W5 must keep `mass`, `tag` and `charge` bit for bit and keep each float field inside its per-field bound; W6 must keep its final stage inside the composed bound (and each stage inside its own, in a unit test); W7 `u32` must match the wrapping sum exactly and W7 `f32` stay inside the bound of the summation order it runs; W9 must keep every entry inside Higham's inner-product bound. A NaN never passes.
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
- Not covered by this spike: workloads W6 and W8 and other comparators (CubeCL, rust-gpu); the spike did not have multi-dimensional dispatch, which W9 now has.
- W4's histogram accumulates across dispatches (a batch of passes keeps counting); the harness counts the passes since
  the bins were cleared and divides the counts by that number, and a count that is not a multiple of it fails the check.
  The bytes per element of W4 (4) count the key stream only, so its GB/s is the compulsory traffic, not the atomics'.
