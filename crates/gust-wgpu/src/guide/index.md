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
- Chapters 9 (the browser) and 10 (benchmarks: the methods, definitions and fairness rules) will live here; they are not
  yet written, so there is no chapter page.

The first program of chapter 2 of the sibling guide is compiled by this crate's doctests, with the real backend (it is
not run: running it needs a GPU), and so is every program of chapters 4 and 5. Chapter 6's one program, `routed`, is not
a doctest (a route tag has to be declared in the manifest of the crate that uses it); the gallery entry it is copied
from is built and run by `cargo xtask check-gallery`.
