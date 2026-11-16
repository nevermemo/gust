# The GUST guide, running half

This guide is documentation only: its modules contain no items, and the set of chapters may change in any release. It
is the second half of the guide of `gust-compute`: the chapters that run kernels need this crate, and `gust-compute`
cannot depend on it. The first half, with the concept, language, semantics and stability chapters and the
installation of the tools, is [`gust::guide`].

## Chapters that live here

- Chapter 4, [running kernels][`crate::guide::running_kernels`]: typed calls, buffers, batches, `StagedGraph`, `GpuVec`
  and pools.
- Chapter 5, [the running half of GPU concepts][`crate::guide::gpu_concepts_running`]: textures, workgroup memory and
  subgroup variants on a device. The kernel half is [`gust::guide::gpu_concepts`].
- Chapter 6, [build profiles, artifacts and routing][`crate::guide::profiles_artifacts_routing`]: `cargo gust`,
  `GUST_PROFILE`, `gust.toml` and the CPU target.
- Chapter 9, [the browser][`crate::guide::browser`]: the gallery's browser page, `map_async`, building for `wasm32`,
  what the repository checks and what it cannot.
- Chapter 10, [benchmarks][`crate::guide::benchmarks`]: the harness, the workloads, the metrics, the fairness rules and
  the rule a result must pass before it may be shown. No result is published.

The first program of chapter 2 of the sibling guide is compiled by this crate's doctests, with the real backend (it is
not run: running it needs a GPU), and so is every program of chapters 4 and 5. Chapter 9's first block, the kernel and
the host reference of the browser page, is compiled by a doctest too (its `main` is empty); the page's browser code is
shown as text, because it exists on `wasm32` only. Chapter 6's one program, `routed`, is not a doctest either (a route
tag has to be declared in the manifest of the crate that uses it). The gallery entries those two texts come from are
built by `cargo xtask check-gallery`, which also keeps every listing identical to its entry; `web-hello` is run in
headless Chrome by the GPU part of that check, and `routed` is run natively.
