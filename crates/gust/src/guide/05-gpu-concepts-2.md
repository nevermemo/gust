```

`load` and `store` are exact; `sample` is not, and its bound is in SEMANTICS. A linear sample of an `f32` texture needs
the device feature `float32-filterable`. `DeviceOptions::default()` (and so `default_device()`) requests it on native
when the adapter has it; `HeadlessDevice::new()` and a browser device do not, and a linear sampler on such a device is
`Error::TextureFilter` at submit.

## `par_for!`

`gust::par_for!` is a loop that runs a body once for every index of a one-, two- or three-dimensional range starting at
0. It is a function-like macro that writes a kernel at the call site, with an explicit typed list of bindings
(`&[T]`, `&mut [T]`, a `GpuVec`, or a scalar), and it returns `Result<(), gust::map::Error>`. The iterations run in no
defined order. A complete program with a `par_for!` is in [chapter 4][ch4]; the macro is documented in the README
section of the same name.

When the loop should produce a value instead of filling a buffer you own, `gust::gpu_task!` is the stage form of the
same kernel: `gpu_task!(n, [a: &[f32], k: f32], |i| -> f32 { a[i] * k })` is a `gust::task::Task` whose `submit()`
returns the read-back as a future and whose `dispatch()` waits for it, with a fresh `Vec<f32>` as the result. A
`(w, h)` shape gives a 2-D stage with a row-major result, and `gpu_task!(&mut data, [data: &mut [f32]], |i| { .. })`
updates a slice in place. On the GPU device a `submit()` is one queue submission, the dispatch and the read-back copy
together (counted on the Radeon 890M; the RTX 5090 run is pending). The example `examples/task-basics` runs all
three, and GUST-E1001, E1002, E1003, E1007 and E1008 are its diagnostics.

Several stages that belong together can be one sealed batch. `gust::gpu_batch!("name", on(&device); in_flight(1); |b| {
.. })` runs its block once: `b.input::<T>(shape)` declares a buffer that frames write, `b.param::<T>(value)` a scalar
that frames set, and each `gpu_task!` in the block is a stage that reads the handles and parameters declared by name
(no capture list for them; the macro reads each `let`'s type). The block returns the handles that frames use, and the
batch is sealed: the structure is frozen, the data stays writable. A frame is `batch.begin().await?`, then
`write`, `set` and `ticket`, then `submit()` (one queue submission on the GPU device, however many stages and tickets),
then `ticket.await`. A frame dropped before `submit` makes no submission and gives `Err(NotSubmitted)` to its tickets.
`in_flight(n)` (1 to 4; 1 when left out) lets `begin` return while up to n-1 earlier frames still run; outside 1 to 4 it
is GUST-E1005. The frames share one set of buffers, so the data persists from frame to frame: an input or parameter that a
frame does not write or set holds the value of the last frame that reached the queue, whatever n is. `examples/batch-basics` has a three-stage
batch (1-D, 2-D and in place) with two tickets, and GUST-E1005 and E1006 are the diagnostics of the form.

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
# use gust::gpu_module;
