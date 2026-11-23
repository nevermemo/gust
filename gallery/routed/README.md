# routed (gallery G8)

A two-stage pipeline whose stages run on different targets, chosen at build time: stage one (`tidy::saturate`, saturate
every sample at 255) is routed to the **CPU**, and stage two (`smooth::blur`, the weighted sum 1, 2, 1 of a sample and
its two neighbours) is routed to the **GPU**. One `RoutedDevice` runs both: each call goes to the target its kernel was
built for, and the buffers move between the CPU and the GPU as the calls need. The source of the calls names no target.

The pieces:

- `Cargo.toml` declares the route tags (`[package.metadata.gust.routes]`): `tidy` with the default `cpu`, `smooth` with
  the default `gpu`.
- `gust.toml`, beside it, is the application's choice, and it is the file you edit to move a stage:

  ```toml
  [routes]
  "routed::tidy" = "cpu"
  "routed::smooth" = "gpu"
  ```

- `#[gpu_module(route = "tidy")]` and `#[gpu_module(route = "smooth")]` put each module under its tag.
- `main` opens the GPU, wraps it with `RoutedDevice::with_gpu`, and prints the target each kernel was built for
  (`DESCRIPTOR.target`).

The defaults and `gust.toml` say the same on purpose: plain `cargo` ignores `gust.toml` and builds the declared
defaults, so `cargo run --release` and `cargo gust run --release` build the same program, and the committed
`gust-artifacts/` (written by `cargo gust build`, which reads `gust.toml`) fit both. `cargo gust` is the way to run it
when you change the routes:

```sh
cargo gust run --release          # builds with the routes of gust.toml and runs; the summary line names each target
cargo gust test                   # the tests, with the same routes
```

## Both routes

The one run exercises both routes: the CPU-routed stage runs as plain Rust on `CpuDevice`, and the GPU-routed stage runs
on the adapter, and the result of the two together is compared exactly with a host reference before `main` prints.

To move a stage, change its value in `gust.toml` (`"gpu"` or `"cpu"`; a tag whose `targets` do not allow it is an error
that names the file and the line) and run `cargo gust run --release` again. `cargo gust build` exports the artifact of
every kernel, whichever route it has, so a flip needs no new artifacts and no Slang. All four combinations of the two
stages were run this way on the RTX 5090 (measured by the slice that wrote this entry, not repeated by `cargo xtask
check-gallery`, which runs the committed `gust.toml` only): CPU and GPU, GPU and CPU, both on the GPU and both on the
CPU, each printing the same checksum and the target of each stage (`tidy on Gpu, smooth on Cpu`, and so on). The program
still opens the GPU when both stages are on the CPU, because `main` does. `GUST_PROFILE=cpu` is the other switch: it
sends every tag to the CPU, but it is a different build, with no GPU runtime, so `main` (which names `HeadlessDevice`)
does not compile under it; the place that opens the GPU becomes `RoutedDevice::<NoGpu>::cpu_only()`.

`cargo gust build` and `cargo gust run` also write `gust-artifacts/routes.json` (the routes they resolved and the
profile); it is a driver output, ignored by git and by `cargo xtask check-gallery`, never committed. `cargo xtask
check-gallery` runs this entry through `cargo gust run --release`, so a `gust.toml` that drifts from the defaults of
`Cargo.toml` fails the check.

There is no fallback (D65): a GPU-routed kernel called on a handle that has no GPU is `Error::NoGpu`, never a silent CPU
run, and a test shows it.

## Tests

`cargo test` (CPU twin only, no GPU, no Slang): both stages on `CpuDevice` against the host reference, exact (every
value is an integer below 2^32), for 1, 2, 63, 64, 65 and 1000 samples (the workgroup is 64, so the last workgroup is
partial in all but 64); one hand-computed case; one test that the targets in this build are the routes of `gust.toml`;
one test that a `RoutedDevice::<NoGpu>::cpu_only()` runs the CPU stage and refuses the GPU stage with `Error::NoGpu`,
leaving the output untouched. The twin of a GPU-routed kernel exists in test builds only, as everywhere (D26); it is not
a fallback.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # no GPU, no Slang
cargo run --features gust-dev     # development: compiles the GPU kernel at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernels with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo build --release             # no Slang needed: the GPU kernel is embedded from gust-artifacts/
cargo run --release               # needs a Vulkan compute adapter
```

It prints `routed: 1000 samples, tidy on Cpu, smooth on Gpu, checksum ..` and a line `adapter: <name>` with the GPU it
ran on.