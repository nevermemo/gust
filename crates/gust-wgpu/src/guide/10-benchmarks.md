# Benchmarks: methods, definitions and fairness rules

This chapter describes how the repository measures GUST against other ways of running the same work. It states the
method and the rules a measurement must pass. It carries no figures, no ratios and no ranking: **no result is
published**, and whether the project makes a speed claim at all is an open decision. The canonical text is
[`docs/BENCHMARKS.md`](https://github.com/nevermemo/gust/blob/main/docs/BENCHMARKS.md) in the repository; this chapter
is a shorter reading of it, and the page wins if the two ever differ.

Every page that shows benchmark results carries the same two paragraphs. `cargo xtask check-bench --render` and the
tests of the `xtask` package keep this copy identical to the page's:

<!-- bench-header:begin -->
> Every comparator (the hand-written WGSL, the rayon CPU backend and the rest) runs inside GUST's harness with the settings stated here, not in its authors' recommended configuration; corrections are welcome as issues against this repository.
>
> A figure describes one adapter, one driver, one operating system and one commit, measured as the methods below say. It is not a ranking, and this page makes no speed claim.
<!-- bench-header:end -->

## The harness

The harness lives in `bench/` in the repository: a workspace of its own, unpublished, so the benchmark dependencies
(rayon and the rest) do not reach the crates you install. Its command line is `gust-bench`, with `run`, `aggregate`,
`list` and `dump-wgsl` among its subcommands; `bench/README.md` lists every option. The harness measures, and it judges
each cell and each run against the rule below; the renderer shows only what passed. A run records its build (commit,
features, profile), the adapter and its own start and end times in its results file, and the quiet windows in a sidecar
file beside it (`<results>.samples.json`).

## The workloads

Four families, each checked against a reference on every sample. A sample that fails its check fails its cell.

- **W1, vector add** (`out = x + y`, f32): memory-bandwidth bound. Correct when the output is bit-exact.
- **W2, saxpy** (`out = a * x + y`, f32): memory-bandwidth bound. Correct within two f32 epsilons of `|a x| + |y|`,
  which leaves room for a fused multiply-add.
- **W3, Horner polynomial** (degree 4, 16 or 64, f32): arithmetic bound. Correct when every element is inside a
  rounding-error bound derived from the standard bound for Horner's rule, against an f64 reference. A backend that
  fuses multiply-add is inside the bound and a perturbed result is outside it; the bound is never widened to make a
  backend pass. The derivation is in `docs/BENCHMARKS.md`.
- **W4, u32 atomic histogram** (256 or 65536 bins, uniform or Zipf keys): bound by atomic contention. Correct when
  every bin equals the serial count exactly.

The inputs are deterministic and not constant, so no compiler can fold a kernel away. Before every sample the device
output is overwritten with NaN for the f32 workloads and with zero for the W4 bins (the kernel adds into them, so a
dispatch that does not run reads back zeros and fails). After the sample the output is read back and compared with a
serial reference. A NaN never passes.

## Backends and routes

Which rows exist depends on the build. All of these exist on the main branch:

- `gust-gpu`: the GUST kernel through the public API of `gust-wgpu` only, on the WGSL route.
- `gust-gpu-spirv`: the same on the opt-in SPIR-V route, in a build with the `spirv-route` feature. A build holds one
  GUST route, so the two routes are separate result files. The SPIR-V route skips wgpu's runtime checks, so it runs in
  a group of its own (`spirv`) and is compared with hand WGSL that also skips them (`wgsl-prebuilt-unchecked`), never
  with the checked rows (D193).
- `wgsl-prebuilt`: hand-written WGSL on plain wgpu, with the host side built the careful way (bind group and read-back
  buffer made once).
- `wgsl-percall`: the same WGSL text with a host side shaped like `gust-wgpu`'s (a bind group per dispatch, a fresh
  read-back buffer, two submissions). It is the control: it runs the same kernel as `wgsl-prebuilt`, so its distance
  from `wgsl-prebuilt` is a noise floor for the single-dispatch GPU time. For the batch quantity it is not a pure noise
  floor, because it builds a fresh bind group in each pass, as GUST does; the page says how that limits the reading.
- `rayon-1`, `rayon-N`, `serial`: `par_iter` on one thread and on all logical cores, and a plain loop.
- `gust-twin`: GUST's CPU twin, present only in a GPU build made with `--cfg gust_dual`.
- `gust-cpu` and `gust-cpu-par`: the production CPU target and its element-wise parallel map (W1 and W2 only), present
  only in the `cpu` build (`GUST_PROFILE=cpu`, no GPU rows).

The default build has the GPU rows (`gust-gpu`, `wgsl-prebuilt`, `wgsl-percall`) and the rayon and serial rows. The
precompiled package `gust-bench-pre` builds the same GUST row from embedded artifacts. Ablation rows (`wgsl-percall`
with one host-side switch turned on, or all of them) and probe rows (`wgsl-prebuilt` with one named difference from
GUST's host side, for finding where a gap comes from) also exist; they are method evidence and are never rows of a
table.

Other comparators (CubeCL, rust-gpu) are not implemented and do not run.

## The four metrics

A cell is one backend on one workload at one size. Each GPU cell is sampled for four quantities:

1. `end_to_end`: host inputs in, host output out, everything complete.
2. `dispatch_host`: one dispatch on resident data, host wall clock from the call to completion.
3. `gpu_single`: GPU timestamps around one compute pass.
4. `gpu_batch_per_dispatch`: GPU timestamps from the first to the last of 16 back-to-back passes in one submission,
   divided by 16. For W7 each of the 16 repeats has its own pair and the figure is the sum of the per-repeat spans, on GUST
   and on the hand paths alike (the gaps between repeats are outside it), so W7's two GPU-time quantities are like-for-like;
   W6 is per six-pass run.

The CPU backends have no GPU clock, so their cells carry the first two only.

Each quantity is sampled 30 times after a warm-up, under a time cap per quantity and never fewer than 5 samples; a
cell with fewer than 30 is marked. The warm-up runs real load until the clock query reports the top performance state
or a cap is reached; that state is recorded, not required. The statistics are the median, the minimum, the 5th and 95th
percentiles and the median absolute deviation; outliers are counted and never dropped. The GPU backends of a cell are
sampled together in interleaved rounds, each round in a seeded random order, and each is paired with `wgsl-prebuilt`
round by round.

## Fairness rules

- Every comparator runs inside this harness with the settings the page states, not in its authors' recommended
  configuration. That is the first paragraph of the header above.
- Workgroup size is 256 for every GPU backend and workload, the same for the generated and the hand-written kernels.
- The harness is built in release mode, and a result from any other build is not publishable. Overflow checks are off
  in the release profile for every backend.
- `gust-gpu` and the hand-written WGSL both go through wgpu's checked shader path, so wgpu's runtime checks are on for
  both. `gust-gpu-spirv` is loaded through wgpu's SPIR-V passthrough, so those checks are not on for that route; it is
  compared with hand WGSL that also skips them (`wgsl-prebuilt-unchecked`, in the `spirv` group of its own, D193). The
  rayon rows use safe zipped slice iterators.
- The GPU backends are timed in interleaved, seeded random rounds and paired round by round, so slow drift of the
  machine is shared between them. The steady state is checked: GUST's pipeline-cache miss counter must not move during
  the timed samples, or the run fails.
- The harness never pins clocks and never stops other services; the quiet-window rule below is what keeps a foreign
  load out of a result.

## When a result may be shown

A result is shown only if the whole publishable rule holds. In plain words:

- The build is a release build, without the tainting features (`profile-counts`, `wgpu-trace`), of a commit that is an
  ancestor of `origin/main`, from a clean work tree at the start and the end of the run, and the commit compiled in
  matches the one that runs. Many other things make a run or a cell not publishable, among them a quick, functional,
  cold-only, partial-size or reduced-sample run, changed timing settings, the ablation group, probe backends, a
  software adapter, a run without the sampler, and a quantity with fewer than 5 samples. `judge.rs` lists them all,
  each with a named reason, and the page explains the main ones.
- At least three complete runs of one session, the first start to the last end within 12 hours, with distinct seeds,
  all checks passed, one route, one build and one adapter.
- A quiet window before every CPU cell and before every group of GPU backends: the harness settles, then samples the
  GPU-engine counters and the process table, and excludes itself, its parent chain and its descendants. A foreign
  process above a threshold of GPU-engine use on the adapter under test in any sample, or a named process (cargo,
  rustc, slangc, a model server, python, node and others) that used CPU or started or ended during the window, makes
  the window not quiet and the cell is retried. A window that cannot be judged counts as not quiet: the rule fails
  closed. A foreign process that uses the GPU lightly and steadily, below the threshold, is not detected; the page
  states this limit.
- The thresholds, which live in the harness source (`bench/gust-bench/src/judge.rs`) and are copied into the page by
  the renderer, so the page cannot drift from the code.
- An attestation line in the commit that files the results, and the harness's own start and end times in the file. The
  renderer refuses a file without its sampler record or whose commit has no such line.

The sampler reads Windows counters only. On any other system a window cannot be judged, so no run there is publishable.

The page also records the limits of the measurements (cache residency, two-state cells) and the kill rule. The kill
rule is two-sided: it marks a run when the GUST path's GPU time is further than a stated tolerance from the
hand-written kernel's, in either direction, and the outcome is a recorded finding. Both are kept on the page and not
repeated here. Today the page's results section states that no result is filed.

## Reproducing

The correctness run works on any Vulkan adapter, a software one included, publishes nothing and takes no timing for a
claim. Like the main workspace, it needs the Slang tools installed
([chapter 2][`gust::guide::install_and_first_program`] says how):

```sh
cargo xtask check-bench --functional --quick
```

The timing harness is built from `bench/` with `GUST_PROFILE=dev`, from a clean tree at the commit you will run. The
native Slang helper is found through `GUST_SLANG_REFLECT`, as for the main workspace. Outputs go outside the work
tree, because a file written inside it makes the tree dirty. Three runs with different seeds are needed before
`aggregate`:

```sh
OUT=/path/outside/the/work/tree
mkdir -p "$OUT"
cd bench
GUST_PROFILE=dev cargo build --release --locked -p gust-bench
./target/release/gust-bench run --out "$OUT/run1.json" --label run1 --seed 1
./target/release/gust-bench run --out "$OUT/run2.json" --label run2 --seed 2
./target/release/gust-bench run --out "$OUT/run3.json" --label run3 --seed 3
./target/release/gust-bench aggregate --json "$OUT/results.json" --md "$OUT/results.md" \
  "$OUT/run1.json" "$OUT/run2.json" "$OUT/run3.json"
./target/release/gust-bench list
```

`cargo xtask check-bench --render` writes the tables of the page from filed results, and `--render --check` fails when
the page is out of date. A run you make yourself is judged by the same rule, and its file says why it fails; it is
still a fair way to check the harness and your own adapter against the rules above.
