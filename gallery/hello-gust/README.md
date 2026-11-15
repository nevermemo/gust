# hello-gust (gallery G1)

The vector add of the repository README ("A first program"), as a crate shaped like yours: it depends on the library
crates by the package names `gust-compute` and `gust-compute-wgpu` (here by path), has a `gust-dev` feature of its own for
the run-time compile path, and commits the `gust-artifacts/` that `cargo gust build` wrote, so a release build needs no
Slang.

What it shows: the first-user path. One `#[gpu]` module with one kernel, a headless device, three buffers, the typed
`dispatch` and the read-back. The `tests` module runs the same kernel on the CPU twin and compares it with a plain Rust
host reference (not with the kernel's own output), including sizes that are not a multiple of the workgroup size.

Commands, from this directory (the tools are set up once with `cargo gust slang install`, `cargo gust slang
build-helper` and `cargo gust doctor`, see the repository README):

```sh
cargo test                        # the CPU twin against the host reference; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernel at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernels with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the kernel is embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints `hello-gust: [11.0, 22.0, 33.0, 44.0]` and a line `adapter: <name>` with the GPU it ran on.

`cargo xtask check-gallery` (from the repository root) runs all of this and compares the artifacts byte for byte.
