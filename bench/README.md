# GUST benchmark harness (M10 spike B1)

A custom harness in a nested, unpublished workspace. It measures two workloads, **W1 vector-add** and **W2 saxpy**
(f32, memory-bandwidth bound), on these backends:

| id | what it is |
|---|---|
| `gust-gpu` | the GUST kernel through `gust-wgpu`'s public API only (typed call, `dispatch_batch`, `dispatch_batch_timed`) |
| `wgsl-prebuilt` | hand-written WGSL on plain wgpu, host side built the careful way: bind group and read-back staging buffer built once, one submission per end-to-end sample |
| `wgsl-percall` | the same WGSL, host side shaped like `gust-wgpu` does it: a bind group per dispatch, a fresh staging buffer and `Vec` per read-back, two submissions with a wait after each |
| `gust-twin` | the GUST CPU twin (`--cfg gust_dual`), single thread, a semantic reference with per-call validation (not the M5 production CPU target) |
| `rayon-1`, `rayon-N` | `par_iter` over host slices in a pool of 1 thread and of all logical cores |
| `serial` | a plain loop (sanity baseline, not in the spike's brief) |

`wgsl-percall` has three **ablation switches** (plan 2.6), each one turned on alone as its own backend, so a gap to
`wgsl-prebuilt` can be attributed by measurement: `wgsl-percall+bind-group-reuse` (one bind group built once),
`wgsl-percall+staging-reuse` (one read-back staging buffer built once) and `wgsl-percall+one-submission` (dispatch and
copy in one submission, no wait between). They are off by default; `--ablation LIST|all` adds them to a run. The fresh
result `Vec` stays per call in every ablation. A test counts bind groups, staging buffers and submissions per switch.

`--ablation all` also adds `wgsl-percall+all-on`: every switch on, the result `Vec` still fresh per call. The table
then reads `wgsl-percall`, each switch alone, `+all-on`, `wgsl-prebuilt` and `gust-gpu`; the step from `+all-on` to
`wgsl-prebuilt` is the result `Vec`, the step from `wgsl-percall` to `+all-on` is the three switches together.

Before the timed samples of a GPU cell the harness warms the cell up by time (0.75 s of its own dispatches, 0.5 s with
`--quick`, `--warmup-secs S` to change it), because the sampler idles the GPU for the settle and window before every
cell and the clocks would otherwise still be low. The SM clock, memory clock, pstate and PCIe link generation are read
while that warm-up is running and stored in the cell (`warmup.clocks`); the aggregate lists them. `--cold-only` skips
the four timings and measures only the cold start (`--cold-repeats N`).

The first cold start of each kernel on a backend in a process is flagged (`cold_start_first_in_process`) and the aggregate lists it
apart from the rest: it also pays one-time costs. A hand-WGSL cold start is labelled "driver shader cache warm" (its
pipeline was already created once on the same device by `prepare`).

**Precompiled cold start.**

The `#[gpu]` macro embeds artifacts found in `gust-artifacts/` beside the Cargo.toml of the package that compiles it.
`gust-bench` has none, so it compiles Slang at run time. `gust-bench-pre` is the same harness sources (its binary
`gust-bench-pre` has `path = "../gust-bench/src/main.rs"`) in a package of its own, with the committed artifacts in
`gust-bench-pre/gust-artifacts/`; a build of it embeds them and its results say `config.shader_source = "precompiled"`.
To regenerate them after a kernel changes: `cargo gust build --manifest-path bench/gust-bench-pre/Cargo.toml` (the
pinned Slang and its helper), then commit the directory. Run it with
`gust-bench-pre run --workloads W1,W2 --sizes 4096 --backends gust-gpu,wgsl-prebuilt --cold-only --cold-repeats 10
--no-sampler --out FILE`.

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
`--backends gust-gpu,rayon-N`, `--samples N`, `--cap-secs S`, `--label NAME`, `--ablation LIST|all`,
`--settle-secs S`, `--window-secs S`, `--no-sampler`. Every path comes from an argument. A build
without `--cfg gust_dual` still works and simply has no `gust-twin` row.

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
refuses runs that disagree on route, build, adapter or config (quick against full, sizes, workloads, backends and
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
allowed two f32 epsilons of `|scale*x| + |y|`, because a GPU compiler may fuse the multiply-add. A NaN never passes.
A failed check or a moved cache counter marks the run as failed.

Statistics are median, min, p5, p95, MAD and the count of samples beyond 3 MAD (counted, never dropped). The aggregate
reports the median over the per-run medians of several runs and the min-max spread of those medians.

## Sizes

`4096, 65536, 1048576, 4194304, 16776960` elements, workgroup size 256 everywhere. The default wgpu limits cap a
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
- Not covered by this spike: workloads W3 to W9, other comparators (CubeCL, rust-gpu), multi-dimensional dispatch.
