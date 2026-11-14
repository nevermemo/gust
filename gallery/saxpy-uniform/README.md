# saxpy-uniform (gallery G2)

`out = a * x + y`, where `a` is a **uniform**: one value in a constant buffer that every invocation reads, bound to a
`Uniform<Params>` parameter. It is shaped like `hello-gust` (same dependencies, the `gust-dev` feature, committed
`gust-artifacts/`, a release build without Slang).

What it shows:

- A uniform is a struct of the `#[gpu]` module with no implicit padding. `Params` writes its three `pad` fields out, so
  the 16 bytes of the constant buffer are the 16 bytes of the Rust value. `create_uniform` refuses a padded type with a
  message that names the fix, and the macro refuses a layout with a gap at compile time.
- The typed call, `saxpy::apply::dispatch(&device, n, &uniform, &x, &y, &out)`: the parameters are the kernel's own, in
  its order, and a buffer where the uniform goes (or the reverse) is a compile error, not a run-time one.
- The generated `call`, `saxpy::apply::call(n, ..)`: the same launch as a value, submitted with `dispatch_call`. The
  example builds it once, changes the uniform with `write_uniform`, and submits: the next dispatch sees the new value.

Tests (`cargo test`, CPU twin only, no GPU, no Slang): the kernel on `CpuDevice` against an independent host reference
(plain Rust in `f64`, rounded once to `f32`), at sizes 1, 63, 64, 65, 129 and 1000 (the workgroup is 64, so the last
workgroup is partial in most of them). Two data sets, each with its own assertion:

- small integers with `a` in 2, -0.5 and 0: every product and sum is exact in `f32`, so the comparison is exact;
- fractional data with `a = 1.7`: tolerance 2 ulp of `|a x| + |y|` (`2 * f32::EPSILON * (|a x| + |y|)`), because a device
  may fuse the multiply and the add (one rounding) where the twin rounds twice.

A third test writes the twin's uniform between two dispatches.

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the CPU twin against the host reference; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernel at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernel with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo run --release               # no Slang needed; needs a Vulkan compute adapter
```

`main` asserts both results before it prints (the values are small integers, exact in `f32`). It prints `saxpy-uniform: a=2 [12.0, 24.0, 36.0, 48.0], a=3 [13.0, 26.0, 39.0, 52.0]` and a line `adapter: <name>` with the GPU it ran on.

Edge case (D39): the kernel guards `i < out.len()`. A kernel without that guard panics on the twin where the GPU's robust
buffer access drops the write; that difference is documented in SEMANTICS, not chased here.
