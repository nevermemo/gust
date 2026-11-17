# Install and the first program

**No usable version of GUST is published yet.** The four package names on crates.io hold only empty 0.0.0
placeholders, and the first release, 0.2.0, waits for the owner's review of the whole project. Until then, build from
the repository (its `gallery/` programs use path dependencies). The commands below are the recipe for the day 0.2.0 is
published; they resolve nothing before that.

GUST will be released as four packages that always share one version:

| Package | Library name | What it is |
| --- | --- | --- |
| `gust-compute` | `gust` | The `#[gpu]` attribute, `par_for!`, typed buffers and calls, the CPU runtime |
| `gust-compute-wgpu` | `gust_wgpu` | The headless wgpu backend that runs the kernels |
| `gust-compute-macros` | `gust_macros` | The proc macros; `gust-compute` uses them and you never depend on them |
| `cargo-gust` | none (a binary) | The `cargo gust` tool: precompiled artifacts and the pinned Slang |

The package name and the library name differ, so you add one and write the other: `cargo add gust-compute
gust-compute-wgpu`, then `use gust::gpu;` and `use gust_wgpu::...` in the code. Do not run `cargo add gust`: that name
belongs to an unrelated crate. Once 0.2.0 is on crates.io, a requirement such as `0.2` is enough. A pre-release (a
`-dev` version taken by git or path, say) is not matched by `0.2`, so where such a dependency line carries a `version` key, name the
pre-release in full; the library crates pin each other to one exact version.

A kernel runs in one of two ways. During development it is compiled at run time through Slang, which needs the pinned
Slang release and a native helper. In a release build it was compiled ahead of time by `cargo gust build` and is
embedded from the committed `gust-artifacts/` directory.

## Set up the tools

```sh
cargo install cargo-gust --locked --version <the version of gust-compute you depend on>
cargo gust slang install        # downloads the pinned Slang, checks its SHA-256, unpacks it into the cargo home
cargo gust slang build-helper   # compiles the native reflection helper against that Slang and prints its path
cargo gust doctor               # names whatever is still missing
```

Only the development path and the author who runs `cargo gust build` need the last three commands. `build-helper`
compiles one C++ file, so it needs a C++ compiler (MSVC on Windows, g++ or clang++ elsewhere). The README of the
repository has the details, including the cache location and the environment variables.

## The first program

Make a binary crate (`cargo new hello-gust`) with the two dependencies above and one feature of your own:

```toml
[features]
# Run-time compile through the Slang toolchain, for development only (`cargo run --features gust-dev`).
gust-dev = ["gust-compute-wgpu/slang-runtime"]
```

`gust-dev` has to be a feature of your own crate. A `slang-runtime` feature on the dependency line makes `cargo build
--release` fail, because a release build refuses the run-time compiler.

The program has one `#[gpu]` module with one kernel. The kernel adds two buffers element by element; the
`if i < out.len()` guard makes sizes that are not a multiple of the workgroup size safe. This block is compiled by
this crate's doctests (`no_run`: it needs no GPU to compile), with `use gust::gpu;` hidden:

```rust,no_run
# use gust::gpu;
