# The GUST guide

This guide is documentation only: its modules contain no items, and the set of chapters may change in any release. It
is split over two crates because the running chapters need the backend, and `gust-compute` cannot depend on
`gust-compute-wgpu` (the backend depends on it).

- This crate, `gust-compute` (library `gust`), holds the concept, language, semantics and stability chapters.
- The sibling guide, [`gust_wgpu::guide`](https://docs.rs/gust-compute-wgpu/latest/gust_wgpu/guide/index.html) in
  `gust-compute-wgpu`, holds the chapters about running kernels, routing, benchmarks and the browser, and the
  compile-checked version of the first program.

## Chapters

1. [Why GUST][`crate::guide::why_gust`]: what GUST is, and what it is not.
2. [Install and the first program][`crate::guide::install_and_first_program`]: the packages, the tools, and a vector add.
3. The authoring model: `#[gpu]` modules, kernels, the accepted subset and the diagnostic codes. Not yet written.
4. Running kernels: typed calls, buffers, jobs, read-back and staged graphs. Not yet written; it will live in
   `gust_wgpu::guide`.
5. GPU concepts in Rust: `par_for!`, workgroup memory, barriers, subgroups, shapes and textures. Not yet written.
6. Build profiles, artifacts and routing. Not yet written; it will live in `gust_wgpu::guide`.
7. Semantics and the twin: the contract, the defined domain and the known limitations. Not yet written.
8. The newer language features: methods, enums, tuples, arrays, closures, traits and generics. Not yet written.
9. The browser. Not yet written; it will live in `gust_wgpu::guide`.
10. Benchmarks: methods, definitions and fairness rules. Not yet written; it will live in `gust_wgpu::guide`.
11. Stability, versions, platforms and how to report a bug. Not yet written.
12. Internals for contributors. Not yet written.
