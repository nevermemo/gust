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

A stage can be counted: `gpu_task!(count(n); capacity, |i| -> T { .. })` (or `gpu_task!(count(n); &mut target, |i| { .. })`)
keeps its ordinary extent as its capacity, and runs over the rows `0 .. min(n, capacity)`, where `n` is a one-element `u32`
handle of the same block that a stage before it wrote: a stage result, the count of `gust::compact` or `gust::count`, or an
in-place counter. A count above the capacity is clamped, not reported, and the rows from the count on are not written: they
keep what they held (zeros in a new buffer). Nothing reads the count back, so a count that changes every frame costs no
readback and no second submission. How the GPU launches the stage is the runtime's choice and gives the same result
everywhere (where the device can, it dispatches from a grid that a one-invocation kernel writes from the count, inside the
frame's one submission); the grid is two-dimensional, so a count is not limited to 65535 workgroups. A count of 0 runs no
row, and so does a count that a resize remakes: a resize makes the buffer new, and a new buffer holds zeros, so a count
the frame does not write again after the resize reads 0. The CPU twin reads `n` from its own buffer and loops over the
same rows. A counted stage is 1-D and its workgroup is along x (GUST-E1014); `count(..)` is a clause of a stage of a
batch (GUST-E1013) and names a handle of `u32`s of the block (GUST-E1015). `examples/batch-basics/tests/counted.rs` has the
checks.

A batch can report the GPU time of its frames. It is opt-in at the build: `b.frame_timing(FrameTiming::Frame)` (or
`FrameTiming::Passes`) in the block makes the timestamp objects once, when the batch is sealed, and only on a device
with timestamp queries; a batch built without it (the default, `Off`) makes none, and `batch.frame_timing()` says up
front what a batch can tell (`None`, or the level and the adapter's tick). A frame then asks with `f.time()`, before
`submit`, and reads the answer like a ticket, by polling, never by waiting: `FrameTimes { span, passes }`. Only the
frames that ask record timestamps, so a timed batch's other frames cost nothing. The **span** runs from the begin of
the frame's first pass to the end of its last pass, whatever lies between them; the frame's upload copies (before the
first pass) and its ticket copies (after the last) are **outside** it. It is the quantity the benchmarks call
`gpu_batch_per_dispatch` times the number of passes. At level `Passes` the result also holds one entry per stage that
has a pass: an entry is per **stage**, from the begin of the stage's first pass to the end of its last (a stage that
records several passes, or a launch split into pieces, which share one pass, is still one entry); a stage over zero
elements has none. The plan for fused passes is one entry per fused group, naming all its stages. The GPU device
records the plain `grid` dispatches of neighbouring stages in one compute pass when a frame is untimed or at `Frame`,
but at `Passes` every stage keeps a compute pass of its own, so a per-stage time includes the pass boundary and the
times can sum to more than the `Frame` span. A time is `Ok(None)`,
never an error, for a batch built `Off`, a device without timestamp queries (the CPU twin, an adopted device that did
not ask for them, `wasm32`), a frame with no pass, and a frame for which no read-back buffer was free (the batch
keeps a bounded set of them and replaces any that a dropped or unread frame cost, so this is rare, and a caller that
never reads its frames' times does not use the set up). A frame that was never submitted is `Err(NotSubmitted)`, and so
is a frame whose copy was recorded into a caller's encoder that never ran; only a frame that recorded no timestamp
(a batch built `Off`, no pass) gives `Ok(None)` there. `submit` and `encode` with `Encoded::submitted` both work; the
frame is still one submission. What a caller can tell from noise, measured on 2026-11-26 over 200 identical frames of
a four-stage batch: on a Radeon 890M the span of one frame lies between about 5 to 7 percent below and 10 to 13
percent above the median of the frames, on an RTX 5090 between about 2 percent below and 1 to 3 percent above it. A
timed frame costs a roughly fixed extra wall clock of about 0.4 ms (0.3 to 0.5 ms on the 890M, 0.39 ms on the 5090;
measured with a blocking wait per frame, not split into its causes), so the cost relative to an untimed batch depends
on the frame length, per adapter: wall clock 1.04 to 1.05 times that of untimed frames for frames of about 7 ms on the
890M, 1.20 to 1.24 times for frames of about 0.4 ms of span on the 5090 (1.22 to 1.23 for `Passes`). For short frames
it is above the 1.05 that GUST calls "free". Compare spans of the same adapter and the same size, never across
machines.

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
