```

`load` and `store` are exact; `sample` is not, and its bound is in SEMANTICS. A linear sample of an `f32` texture needs
the device feature `float32-filterable`, which a native default device requests when the adapter has it and a browser
device does not.

## `par_for!`

`gust::par_for!` is a loop that runs a body once for every index of a one-, two- or three-dimensional range starting at
0. It is a function-like macro that writes a kernel at the call site, with an explicit typed list of bindings
(`&[T]`, `&mut [T]`, a `GpuVec`, or a scalar), and it returns `Result<(), gust::map::Error>`. The iterations run in no
defined order. A complete program with a `par_for!` is in [chapter 4][ch4]; the macro is documented in the README
section of the same name.

## Workgroup memory and barriers

The invocations of one workgroup can share fast memory and wait for each other. `Workgroup<[T; N]>` is a kernel
parameter, never a local, field or helper parameter, that becomes a `groupshared` array of the entry point; `N` is a
nonzero literal and all of a kernel's arrays together are at most 16 KiB (GUST-E0718). Its contents start unspecified,
so a kernel writes an element before it reads it. `workgroup_barrier()` waits for the whole workgroup, and it orders one
workgroup only. It is allowed only at the top level of the kernel body or of a top-level `for` over two literals,
because every invocation must reach it (GUST-E0414). Helpers cannot take `Workgroup` (GUST-E0119).

A kernel with workgroup memory or a barrier has no CPU twin, so it says `cpu = "unsupported"` (GUST-E0717); no CPU
strategy for barriers exists, and a threaded lockstep twin was measured and dropped as far too slow (D118). Such a
kernel is checked against an independent host reference on the GPU instead, which is what G6, `reduce-scan`, does. The
first kernel sums a tile of 64 elements with a tree of six rounds; the second is an inclusive scan of a tile; the third
adds each tile's offset and has no workgroup memory, so it has a twin:

```rust,no_run
# use gust::gpu;
