# The GUST guide, running half

This guide is documentation only: its modules contain no items, and the set of chapters may change in any release. It
is the second half of the guide of `gust-compute`: the chapters that run kernels need this crate, and `gust-compute`
cannot depend on it. The first half, with the concept, language, semantics and stability chapters and the
installation of the tools, is [`gust::guide`].

Chapters that will live here (none is written yet, so there is no chapter page):

- Chapter 4, running kernels: typed calls, buffers, jobs, read-back, `StagedGraph`, pools and `GpuVec`.
- Chapter 5, the running halves of the GPU concepts chapter.
- Chapter 6, build profiles, artifacts and routing: `cargo gust`, `GUST_PROFILE`, `gust.toml` and the CPU target.
- Chapter 9, the browser.
- Chapter 10, benchmarks: the methods, definitions and fairness rules.

The first program of chapter 2 of the sibling guide is compiled by this crate's doctests, with the real backend (it is
not run: running it needs a GPU).
