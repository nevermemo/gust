# Benchmarks

Status: **no result is published.** This page holds the method, the rules a result must pass before it may be shown,
and the known limits of the measurements. It has no headline number, no ranking and no speed claim; whether the project
makes one is an open decision. The tables near the end are generated from filed results by
`cargo xtask check-bench --render`, and are empty until a result passes the rule.

<!-- bench-header:begin -->
> Every comparator (the hand-written WGSL, the rayon CPU backend and the rest) runs inside GUST's harness with the settings stated here, not in its authors' recommended configuration; corrections are welcome as issues against this repository.
>
> A figure describes one adapter, one driver, one operating system and one commit, measured as the methods below say. It is not a ranking, and this page makes no speed claim.
<!-- bench-header:end -->

## Workloads

The harness lives in `bench/` (a workspace of its own; see `bench/README.md` for every command). Six families (W1 to W4, W7 and W9), each
checked against a reference on every sample; a sample that fails its check fails the cell.

| id | what it is | correct when |
|---|---|---|
| W1 vector-add | `out = x + y`, f32, memory-bandwidth bound | bit-exact |
| W2 saxpy | `out = a * x + y`, f32, memory-bandwidth bound | within two f32 epsilons of `|a x| + |y|` (room for a fused multiply-add) |
| W3 Horner polynomial | degree 4, 16 or 64, f32, arithmetic bound (`2 d` flops per 8 bytes) | within the derived bound below |
| W4 u32 atomic histogram | 256 or 65536 bins, uniform or Zipf keys, atomic-contention bound | every bin equals the serial count exactly |
| W7 sum | `u32` (wrapping) and `f32`, 2^20 to 2^27 values, one output word, a reduction; GUST runs `gust-algo`'s public sum | `u32`: the wrapping sum exactly; `f32`: within the bound of the summation order the backend runs (below) |
| W9 naive matrix product | `C = A B` of two `N x N` f32 matrices, `N` from 256 to 2048, one output entry per invocation (the tiled form is not in this set) | every entry within the derived bound below |

**The unit roundoff, per backend.** Every f32 add and multiply is modelled as `fl(a op b) = (a op b)(1 + delta) + eta`
with `|eta| <= 2^-126` and `|delta| <= u`, where `u` is the backend's: `u = 2^-24` for the CPU backends (`serial`, rayon,
`gust-twin`: host arithmetic, IEEE round to nearest) and `u = 2^-23` for every GPU backend (GUST, its SPIR-V route and
the hand WGSL). `docs/SEMANTICS.md` section 4 lets a GPU add or multiply round to either neighbour, within 1 ULP, which is
a relative error of up to `2^-23`; `gust_algo::bound::C_GPU` encodes the same constant. The reference carries both bounds
and the check picks the one of the backend's kind (`Rounding`). W3's bound was first written with `2^-24` for the GPU
backends too; it was corrected with W7 and W9 (no backend had failed it, so this widens nothing that failed, it makes the
stated model the one the semantics grant).

**The W3 bound.** Higham (Accuracy and Stability of Numerical Algorithms, 2nd ed., section 5.1)
bounds Horner's rule by `gamma_{2d} S` with `S = sum |c_k| |x|^k`. The reference is Horner's rule in f64, which obeys the
same theorem with the f64 unit roundoff, so the check is

```text
|p - p_f64| <= (gamma_{2d} + gamma64_{2d}) * S + 2d * 2^-126
```

The last term allows underflow. A backend that fuses multiply-add is inside the bound (a test shows it) and a perturbed
result is outside it. The bound is never widened to make a backend pass.

**The W7 checks.** The `u32` sum is exact and is the wrapping one (modulo 2^32): `gust_algo::reduce::sum_wrapping` is
defined modulo 2^32 for every input, WGSL integer addition wraps, and the CPU backends use `wrapping_add`; the inputs are
32-bit hash words, so the true sum passes 2^32 by far and a lost or doubled element changes the word. The `f32` sum has
no single order, so each backend is judged by the bound of the order it runs, all of the form `gamma_K S` with
`S = sum |x_i|` (Higham, section 4.2: every value meets at most `K` roundings, whatever the grouping), plus the f64
reference's own error `(n - 1) 2^-53 S` (`gust_algo::bound::f64_reference_error` has the same formula; a test pins them
equal, and another that the f64 reference sum is exact for these inputs, so the first-order form is no looser than the
rigorous `gamma64_{n-1} S`):

```text
GUST's tree (gust-gpu, gust-gpu-spirv):  |s - s_f64| <= gust_algo::bound::f32_sum_bound_v1(n, 2^-23, S) + (n - 1) 2^-53 S
hand WGSL tree (wgsl-prebuilt/-percall): |s - s_f64| <= gamma_K(2^-23) S + (n - 1) 2^-53 S,  K = chain_1 + 8 + chain_2 + 8
serial, rayon-1, rayon-N (u = 2^-24):    |s - s_f64| <= ((1 + u)^{n-1} - 1) S + (n - 1) 2^-126 (1 + factor) + (n - 1) 2^-53 S
```

*GUST's tree.* `c = 2^-23` is the library's constant (`bound::C_GPU`). Which GUST schedule ran is not reported by the
public API (`limits::admits` is private): the harness repeats its test from the public device limits (workgroup memory
for the v1 kernels, the input within the storage binding, the first pass within the workgroup limit) and takes
`f32_sum_bound_v1` where it holds; where it does not it takes the larger of `f32_sum_bound` (v0) and the v1 bound. That
repeated test is an assumption of the harness (a test shows the branch and the fallback).

*The hand tree* has a known order, so it has its own bound (the plan's serial worst case still bounds it; a test shows
the tree bound is inside it, and this is a tightening the coordinator recorded). Pass one is a grid-stride reduction:
`blocks = min(ceil(n / 2048), max_compute_workgroups_per_dimension)` workgroups of 256 lanes, so the grid grows with `n`
(512 workgroups at 2^20, 65535 at 2^27). A lane adds `chain_1 = ceil(n / (256 blocks))` values one after the other, then
the 256 lane values of the workgroup are added in a tree of 8 rounds; pass two is one workgroup in which a lane adds
`chain_2 = ceil(blocks / 256)` partials and then the same 8-round tree. `K = chain_1 + 8 + chain_2 + 8` counts the first
add of a chain (into a zero accumulator, exact on IEEE hardware) too, the conservative reading of "either neighbour"; a
compiler's reassociation (WGSL section 15.7.5) does not lengthen a chain. K is 26 at 2^20 and 281 at 2^27.

*The host backends* (`serial`, `rayon-1`, `rayon-N`, `u = 2^-24`). The order of `serial` is sequential and of rayon
unknown, so they are judged by the serial worst case, which covers every order: `((1 + u)^{n-1} - 1) S` is Higham's
product form (`|prod(1 + delta_j) - 1| <= (1 + u)^k - 1` for every `k`), used everywhere in place of `gamma_{n-1}`: it is
never larger than `gamma_{n-1}`, exists for every `n` (`gamma_k` needs `k u < 1`, which fails above `n = 2^24`) and grows
with `n`. No tighter bound is invented for them. **Where the `f32` check is vacuous.** On `serial`, `rayon-1` and `rayon-N`
the factor is 0.065 at 2^20, 0.28 at 2^22, 0.65 at 2^23 and above 1 from 2^24 (1.7 at 2^24, then larger): from `n = 2^24`
the bound exceeds `S >= |s|`, so a result of `0.0` passes whatever the data, and 2^22 to 2^23 is weak (over a quarter of
`S`). The check on those three backends is therefore meaningful at 2^20 and 2^21 only. Correctness of the `serial` and
rayon cells at 2^22 to 2^27 rests on the same code passing at the small sizes (the code does not branch on `n`) and on
W7-`u32` being exact on the same backends and sizes (a lost or doubled element changes the `u32` word). The `f32` check on
GUST's tree and the hand tree is meaningful at every size (the bound is a few times `1e-5 S` at most). `S` is itself
computed in f64, so the code scales it up by `1 / (1 - gamma64_{n-1})`.

The `f32` inputs are multiples of 2^-23 in `[-0.5, 1.5)`, so every addition rounds (a test shows a left-to-right sum
differs from the f64 one) and no partial sum overflows or goes subnormal. A backend that fails the bound fails the cell,
and a failure of the `f32` sum against its bound on any backend stops the slice that added it; the bound is never
widened to pass.

*The two algorithms differ.* GUST's v1 sum (`gust_algo::reduce`) gives a workgroup 8192 elements (256 lanes, each a chain
of up to 32 terms), tree-adds the lane values in 8 rounds into one partial per workgroup, and repeats over the partials:
one to three block-reduce dispatches (three at 2^27), with pooled partial buffers it zeroes on every call. The hand
reduction is two dispatches (above): one grid-stride pass whose grid grows with `n` and one workgroup. Remaining
differences: GUST's per-call pooling and zeroing, a different chain length and workgroup count, and (for `u32`) the
read-back inside the library call. They are different algorithms, so no GUST/hand ratio is a GUST overhead.

*What the timings of W7 are.* `gust-algo`'s sum exposes no GPU timestamps, so a W7 cell of GUST has no GPU time. Its
`dispatch_host` is `reduce::sum_on_device` for `f32` (the one-word result stays on the device, read back outside the timed
region, like the hand dispatch), but it still draws and zeroes its pooled partials per call, which the hand dispatch does
not. `u32` has no on-device sum, so W7-`u32`'s GUST `dispatch_host` is the public `sum_wrapping` call **including the
4-byte read-back**; the hand `dispatch_host` has none. The report shows no GUST/hand ratio for W7 and the kill rule leaves
W7 out (see "The kill rule").

*Devices.* Inputs above 2^25 words need a storage binding above the portable 128 MiB limit, so a run that includes them
(the default W7 ladder does) opens both GPU devices with the adapter's own limits (`LimitsPolicy::Adapter` for GUST, the
same three limits for the hand device), for every workload of that run. Both sides change the same way, so no pairing is
biased, but such a run is not directly comparable with an earlier run on the portable limits. Both grants are in the run
file's machine block (`wgpu_limits` and `wgpu_limits_hand_wgsl`).

**The W9 bound.** The entry `c_ij` is an inner product of length `N`: `N` multiplies and `N - 1` adds, in whatever order a
backend groups them. With the same rounding model as W3, Higham (section 3.1, eq. 3.5 and 3.13) gives `|C^ - C| <=
gamma_N |A| |B|` entrywise for the exact product `C` of the f32 inputs, and a fused multiply-add only has less error. The
reference is the product in f64 with `|A||B|` computed beside it (non-negative terms, relative error at most `gamma64_N`),
so the check, as W3's, is

```text
|c_ij - c_f64_ij| <= (gamma_N + gamma64_N) * (|A||B|)_ij * (1 + gamma64_N) + 2N * 2^-126
```

with `u = 2^-24` for the CPU backends (`serial`, rayon, `gust-twin`) and `u = 2^-23` for every GPU backend (GUST, its
SPIR-V route, the hand WGSL), as in "The unit roundoff, per backend" (the accumulation is a recursive sum of depth
`N - 1`, inside the theorem). The underflow term is first-order and moot here: the inputs are multiples of 2^-23, so the
smallest non-zero product is 2^-46 and nothing is subnormal. The matrices have full-mantissa entries in `[-1, 1)`, so the
products round. Tests show a plain result and a fused multiply-add result inside the bound (under both units), a result
between the two bounds passing for a GPU and failing for a CPU backend, and a result one f32 step past the bound outside;
the bound is never widened to pass. W9's size is the matrix side `N` (the matrices hold `N * N` words), and `--sizes` is
refused beside W9 above 4096 for that reason. Achieved bandwidth is not applicable to W9 (each matrix is read many times),
so the report shows none; a GFLOP/s column is not part of this slice.

## Backends and routes

| id | what it is |
|---|---|
| `gust-gpu` | the GUST kernel through `gust-wgpu`'s public API, WGSL route |
| `gust-gpu-spirv` | the same on the opt-in SPIR-V route (the `spirv-route` build) |
| `wgsl-prebuilt` | hand-written WGSL on plain wgpu, host side built the careful way: bind group and staging buffer made once |
| `wgsl-percall` | the same WGSL with a host side shaped like `gust-wgpu`'s: a bind group per dispatch, a fresh staging buffer, two submissions |
| `gust-twin`, `gust-cpu`, `gust-cpu-par` | GUST's CPU twin, the production CPU target and its element-wise parallel map |
| `rayon-1`, `rayon-N`, `serial` | `par_iter` on one thread and on all cores, and a plain loop |

A build holds one GUST route, so the two routes are separate result files, each with its own `wgsl-prebuilt` and
`wgsl-percall`, and are compared through their paired ratios to their own run's `wgsl-prebuilt`. `wgsl-percall`
also has three ablation switches (bind-group reuse, staging reuse, one submission); they run only in the ablation group,
whose rows are never part of a table.

## The four timings

Per cell (backend, workload, size), each sampled 30 times after a warm-up, under a cap of 8 s per quantity and never below 5 samples (a cell with fewer
than 30 collected is marked F in the tables):

1. `end_to_end`: host inputs in, host output out, everything complete.
2. `dispatch_host`: one dispatch on resident data, host wall clock from the call to completion.
3. `gpu_single`: GPU timestamps around one compute pass.
4. `gpu_batch_per_dispatch`: timestamps from the first to the last of 16 back-to-back passes in one submission, divided
   by 16.

The GPU backends of a cell are sampled together in interleaved rounds, each round in a seeded random order, and each
backend is paired against `wgsl-prebuilt` round by round. The warm-up runs real load until the clock query reports P0,
capped at one second; P0 is recorded, not required.

## The publishable rule

A result is shown only if all of these hold. The harness decides items 1 to 3 and marks every cell and every run; a run
that fails is written with the header "shared machine, not publishable" and the reasons. Item 5 is checked by the
renderer.

1. The commit is an ancestor of `origin/main`, the work tree is clean (untracked files included) at the start and at the
   end of the run, and the build is a release build with none of the tainting features (`profile-counts`, `wgpu-trace`).
   A run that is `--quick`, `--functional`, `--cold-only`, `--sizes`, asks for fewer than 30 samples, changes the settle,
   the window, the sample cap or the warm-up (`--settle-secs`, `--window-secs`, `--cap-secs`, `--warmup-secs`), is on a
   software adapter, is in the ablation group or has probe backends is never publishable, each with a named blocker. So
   is a run whose compiled-in git HEAD differs from its run-time HEAD, or where either is unknown (build from the commit
   you run). A quantity with fewer than 5 collected samples blocks its cell.
2. Three complete runs of one session (the first start to the last end within 12 hours, distinct seeds), all checks
   passed, no failures, one route, one build, one adapter.
3. A quiet window before every cell. The harness settles for 3 s, then samples a 2 s window at 4 Hz (8 samples) of the
   Windows GPU-engine counters (per process and per adapter) and the process table (CPU time, creation time). The
   harness itself, its parent chain up to the first process that is not a shell or cargo, and the harness's descendants
   are excluded by process ancestry. Anything else is foreign: a process above `T_gpu` on the adapter under test in any
   sample, a named process (cargo, rustc, rustdoc, slangc, a model server, python, node, `vmmem`) that used CPU or
   started or ended during the window. A window that cannot be judged (a counter error, a short sample list, an
   unreadable process table, a counter row that is not a number, an unparsed counter instance, or an `invalid` reading
   of a counter that matters) counts as not quiet. The quiet windows of a publishable session are taken with the
   monitoring scripts stopped. A cell that is not quiet is retried, up to `N` times; every attempt is
   kept in the sampler sidecar.
4. The thresholds below, which are set from recorded data.
5. An attestation line in the commit that files the results (`Quiet-window attestation: ...`), and the harness's own start
   and end times in the file. The renderer refuses a file whose sidecar is missing or whose commit has no such line.

### The thresholds

<!-- bench-thresholds:begin -->
| Constant | Value |
|---|---|
| Thresholds version | 1 |
| `T_gpu` (foreign GPU-engine use, percent of the engine, any sample) | 5.0 |
| `T_cpu` (`vmmem` CPU growth, cpu-seconds per second) | not set (no `vmmem` process in any recorded window; a `vmmem` or `vmmemWSL` process present makes a window not quiet) |
| `N` (settle-and-sample attempts per cell) | 5 |
| Runs per aggregate | 3 |
| Session span, hours | 12.0 |
| Samples per quantity | 30 under a cap of 8.0 s per quantity, never below 5 |
<!-- bench-thresholds:end -->

These come from the sampler sidecars recorded in M10 (the values above are read from the harness source by the
renderer, so this page cannot drift from the code).

**The data.** Two sets of sidecars, all on one machine (an RTX 5090 desktop under Windows 11):

- a *quiet* set: 9 sidecars (three full runs and six quiet-window runs taken with the machine otherwise idle), 726
  windows, 174 of them GPU-group windows;
- a *shared* set: 17 sidecars (quick dry runs and smoke runs taken while other builds and tests ran), 510 windows.

For each window the maximum over samples of the largest foreign process's GPU-engine use on the adapter:

| Windows | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| quiet set, all (726) | 0.44 | 2.94 | 5.25 | 15.07 | 17.11 |
| quiet set, GPU-group windows (174) | 0.44 | 3.53 | 6.68 | 14.67 | 17.11 |

The harness's own use in its idle windows is 0.00 percent in all 1236 windows of both sets, so the 3 s settle is long
enough for the harness's own work to drain. The ambient desktop does produce short bursts (the System process's copy
engine and the session manager's `csrss`): 40 of 726 quiet windows (5.5 percent) are above 5 percent, 2.9 percent above
10, 1.2 percent above 15, none above 20. The shared set never recorded a sustained foreign GPU load either (its maximum
is 15.51), because the other lanes' work on that machine was CPU work (cargo and rustc grew 0.1 to 3.8 CPU-seconds per
second there, and none in any quiet window).

**What this does and does not separate.** `T_gpu` is 5 percent: above the median and the 90th percentile of the quiet
distribution, and at about its 94th. The ambient bursts above it are not handled by raising the threshold but by
retrying. Under the whole rule as implemented (the GPU rule, the named-process rule and the fail-closed rows), 87 of the
726 quiet windows fail (12.0 percent), and a failing window is followed by another failing one in 22 of 86 cases (25.6
percent). So a window sequence (one per CPU cell, one per GPU group) fails all five attempts with probability about
5e-4. A full run takes 96 such sequences (120 with the CPU twin), so about 5 (6) runs in 100 end not publishable, and
about 14 (17) aggregates of three runs in 100 contain one. The recording itself held one sequence of five failing
windows in a row, where this model expects about 0.4. That is on a machine where the monitoring scripts of the
recording still ran; 53 of the 87 failures are named-process windows. The figures come from the M10 premise runs'
sidecars (not in the repository).

That is the honest limit of the data: the recorded windows separate the harness's own idle noise (zero) and the desktop's brief bursts from a sustained
load by their persistence, not by a measured distribution of foreign load, because no sustained foreign GPU load was
ever recorded. The one known sustained shape is the harness's own `probe-adapters` busy loop, which reads up to 42
percent. A foreign process that uses the GPU lightly and steadily below 5 percent is not detected.

**`T_cpu` is not set.** The only process the CPU rule needs a threshold for is `vmmem` (the WSL and Hyper-V memory
process). It is read like any named process, but with no threshold the judge fails closed: a `vmmem` or `vmmemWSL`
process present in a window makes it not quiet, whatever its CPU time does. No such process existed in any recorded
window, so there is no data to place a threshold on. **Limit:** one sampler window taken with the WSL service up, but
with no running distribution confirmed, listed neither `vmmem` nor `vmmemWSL` (only `wslservice` and `vmcompute`), so
whether the process walk lists WSL's VM process by name is not established, and WSL's CPU load may not be seen by the CPU
rule. Named processes need no threshold; any CPU growth in a window counts. (The monitoring scripts of
the recording grew up to 0.49 CPU-seconds per second in 32 of 726 quiet windows and are on the named list for that
reason.)

## The kill rule

For each cell of the publishable group the harness takes the paired ratio of the GUST backend's GPU time to
`wgsl-prebuilt`'s (`gpu_single` and `gpu_batch_per_dispatch`). A ratio more than 10 percent off 1.0 on any cell marks the
run "C re-opened" (an evidence flag that the GUST path adds cost the plan did not expect); the run file lists the
offending cells. The rule judges only a complete run: a cell without a ratio counts as unpaired and the verdict is then
`null`, never `false`.

**Cells outside the rule.** A workload whose GUST and hand paths are not the same kernel is outside the rule, on both
sides (`Workload::same_kernel`; today only W7): GUST runs `gust-algo`'s public sum, which exposes no GPU timestamps, so its
cell has no paired GPU-time ratio by construction, and the hand reduction is another algorithm. Such a cell is neither
counted as unpaired nor judged (it would otherwise keep every run that includes W7 from being judged, `c_reopened` always
null), and it is **listed**: the rule's record in the run file has `excluded: [{backend, workload, n, reason}]` for each
of them, so a reader sees W7 was present and left out on purpose. This is a deviation from the plan's "any cell of the
publishable group" (decision D190). It does not affect whether a run is publishable.

**Reproducibility clause.** A gap counts only if it recurs: the same cell must be more than 10 percent off in all three
seeds of a confirming run (at least three runs with distinct seeds), with the control (`wgsl-percall`) inside its tolerance
on that cell in each of those runs. A cell that recurs with the control out of tolerance, or in fewer seeds, is listed and
not counted.

**The control's limits for the batch quantity.** `wgsl-percall` runs the same kernel text as `wgsl-prebuilt`, so its
distance from 1.0 is a noise floor for `gpu_single`. For `gpu_batch_per_dispatch` it is not a pure noise floor: it
differs from `wgsl-prebuilt` in the per-pass bind groups (percall builds a fresh one in each of the 16 passes, as GUST does;
prebuilt reuses one), so its batch excess may be a real effect of that and not noise. A "control out of tolerance" verdict
therefore voids a batch cell on a confounded control. A control that is identical in command stream for the batch
quantity would be a second `wgsl-prebuilt` instance, or the bind-group-reuse switch turned on. The kill rule's
re-opening does not gate whether a run is publishable; it is a recorded finding, and this limit is part of how it is read.

## Known outcome: the batch path (outcome D)

Closed as "unexplained, not tuned".

- An earlier quiet-window run on an RTX 5090, which the harness marks not publishable, showed GUST's 16-pass batched
  GPU time at 1.12 to 1.21 times hand WGSL's (a ratio of the two GPU times) on W1 at 1M and 4M elements and W2 at 1M,
  with `gpu_single` at parity and the control at 0.99 to 1.00.
- The wgpu API traces of GUST's batch and the hand batch differ in one per-pass item: GUST binds a fresh bind group in
  each pass and `wgsl-prebuilt` reuses one. `wgsl-percall` matches GUST's stream.
- In the confirming run (seeds 11, 22, 33; on the same RTX 5090, in runs the harness marks not publishable) the gap
  recurred on W1 1M only (1.275, 1.221, 1.248). The control was inside its tolerance in seed 33 only (1.106, 1.125, 1.081), so by the clause above the cell is listed, not counted. W1 4M (1.00)
  and W2 1M (1.03 to 1.08) did not recur.
- Neither the per-pass bind groups (GUST over percall is about 1.09 to 1.15) nor the pre-registered ablation `gust-setup`
  (GUST over gust-setup is about 0.97 to 1.11) accounts for the gap in all three seeds.
- A probe that runs the hand kernel on a separate device opened like GUST's sits in the same slow state as GUST (1.24 to
  1.29). That is an observation, not a cause, because the probe changes three things at once.
- No further ablation was run (the kill rule).

**Limitation as measured.** On an RTX 5090, in runs the harness marks not publishable, the ratio of GUST's batched GPU
time (16 dispatches) to hand WGSL's on vector-add at 1M elements is 1.22 to 1.28 in the last measurement. Over earlier
quiet runs the ratio reached 1.12 to 1.21 on vector-add and saxpy at 1M to 4M. The ratio for single-dispatch GPU time is
0.99 to 1.03. The cell is bimodal for every backend (about 6.1 and 7.9
microseconds per dispatch on W1 1M), and the identical-kernel control moved by up to 1.13 times on the same cell. The
excess is unexplained and not tuned. It is not in the generated kernel: running GUST's WGSL in the hand path gives 1.00
times.

## An observation, beside the method: the SPIR-V route on W3

On an RTX 5090, in runs the harness marks not publishable, the GPU-time ratio of the opt-in SPIR-V route to hand WGSL
on the polynomial workload is 0.80 at degree 16 and 0.70 at degree 64, both inside the derived bound. Most of the difference is wgpu's shader runtime checks on the WGSL route (naga's
loop-bound counter, absent from slangc's SPIR-V): the same ratio for hand WGSL without those checks is 0.81 and 0.80. At degree
16 the checks account for nearly all of the gap; the remainder at degree 64 (about 0.10) is observed, and its cause is not
identified. The arithmetic is the same in both routes (no fused multiply-add, no unrolling). This is an observation about
the method, not a result of this page, and a ratio below 1 on a SPIR-V cell does not re-open the kill rule.

## Other limits of the measurements

- Cache residency: a cell whose working set fits in the GPU's L2 can read faster than DRAM; the harness flags every GPU
  figure above the adapter's DRAM peak (`dram_peak`, `above_dram_peak`). A figure above the peak is not a DRAM bandwidth.
- Two-state cells: some cells alternate between two speeds (W1 1M: about 6.1 and 7.9 microseconds). The report carries a
  gap test (`two_state_wide_gap`, 10 percent of the median between the 5th and 95th percentile of the batch quantity), which
  is calibrated on the m10 runs only and not fitted beyond them.
- The sampler idles the GPU once per GPU group, not before every backend; the CPU backends get a window each.
- P0 is not reached at 1M elements and below within the warm-up cap; the group is kept and marked.
- The sampler reads Windows counters; on other platforms a run cannot be judged and is never publishable.
- Comparators run inside this harness with the settings stated here (see the header).

## Results

<!-- bench-tables:begin -->
No results are filed. Nothing here is publishable until three complete runs of one session pass the rule above and are filed with the attestation line in their commit. The renderer refuses anything else.
<!-- bench-tables:end -->

## Reproducing

From `bench/` (the bench is a workspace of its own; every cargo command there runs with `GUST_PROFILE=dev`; its
`bench/target/` is gitignored, so the build leaves the tree clean). Build from a clean tree at the commit you will run:
the binary embeds its build commit and whether tracked files had uncommitted changes then, and a run whose build commit
differs from the checked-out one, or whose build was dirty (or whose build state is unknown), is not publishable.

```text
mkdir -p OUTSIDE_THE_WORK_TREE
GUST_PROFILE=dev cargo build --release --locked -p gust-bench
./target/release/gust-bench run --out OUTSIDE_THE_WORK_TREE/run1.json --label run1 --seed 1
```

Take the output path outside the work tree (a run that leaves the tree dirty is not publishable). The aggregate of three
runs is `gust-bench aggregate --json FILE --md FILE RUN.json...`. Filing is `bench/results/<adapter-slug>/<commit>-<route>.json` plus its
`.samples.json` (`<adapter-slug>` is the adapter's name in lower case with each run of characters other than letters and
digits made one hyphen, so "NVIDIA GeForce RTX 5090" is `nvidia-geforce-rtx-5090`; `<commit>` is the run's build commit,
at least its first 12 characters; `<route>` is the file's route), in a commit whose message carries the `Quiet-window attestation:` line. Then:

```text
cargo xtask check-bench --render          # writes the tables of this page
cargo xtask check-bench --render --check  # fails when the page is out of date (no workflow runs it; the xtask tests check the page against the renderer)
cargo xtask check-bench --functional --quick   # the correctness run on any adapter, no timing published
```
