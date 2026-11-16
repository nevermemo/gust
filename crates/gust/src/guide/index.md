# The GUST guide

This guide is documentation only: its modules contain no items, and the set of chapters may change in any release. It
is split over two crates because the running chapters need the backend, and `gust-compute` cannot depend on
`gust-compute-wgpu` (the backend depends on it).

- This crate, `gust-compute` (library `gust`), holds the concept, language, semantics and stability chapters.
- The sibling guide, `gust_wgpu::guide` in `gust-compute-wgpu` (linked at the end of this page), holds the chapters
  about running kernels, routing, benchmarks and the browser. The doctests of that crate also compile the whole first
  program of chapter 2.

## Chapters

1. [Why GUST][`crate::guide::why_gust`]: what GUST is, and what it is not.
2. [Install and the first program][`crate::guide::install_and_first_program`]: the packages, the tools, and a vector
   add.
3. [The authoring model][`crate::guide::authoring_model`]: `#[gpu]` modules, kernels, the accepted subset and the
   diagnostic codes.
4. Running kernels: typed calls, buffers, batches, staged graphs and resident data. Written; it lives in
   `gust_wgpu::guide` (linked at the end of this page).
5. [GPU concepts in Rust][`crate::guide::gpu_concepts`]: shapes, textures, `par_for!`, workgroup memory, barriers and
   subgroup variants. Its running halves are in `gust_wgpu::guide`.
6. Build profiles, artifacts and routing. Written; it lives in `gust_wgpu::guide`.
7. [Semantics and the twin][`crate::guide::semantics_and_the_twin`]: the contract, the defined domain, the known
   limitations and the SPIR-V route.
8. [The newer language features][`crate::guide::m9_language`]: methods, enums, tuples, arrays, closures, traits and
   generics.
9. The browser: building for `wasm32`, WebGPU in the page and what the repository checks. Written; it lives in
   `gust_wgpu::guide` (linked at the end of this page).
10. Benchmarks: methods, definitions and fairness rules. Written; it lives in `gust_wgpu::guide` (linked at the end
    of this page).
11. [Stability, versions, platforms and how to report a bug][`crate::guide::stability_and_platforms`].
12. [Internals for contributors][`crate::guide::internals_for_contributors`]: the pipeline, the gates and how to add
    a construct.
