# buffer-handoff

A GUST output buffer used by another wgpu compute pass on the same device, without a CPU round trip.

- The `#[gpu]` kernel `count` hashes each thread index into one of 64 bins and counts it with `atomic_add`. Integer
  atomics are exact, so the histogram does not depend on thread order.
- The CPU twin and a plain-Rust reference compute the same histogram; the reference is uploaded as the expected buffer.
- A compute pass the example owns (hand-written WGSL, its own `wgpu` pipeline) reads GUST's histogram buffer directly,
  next to the expected buffer, and adds one to a single `u32` for every bin that differs.
- Only that `u32` is read back. It is 0 for the right expected buffer and 1 when one bin is changed.
- No render pass, no texture, nothing drawn.

The example reaches the device and queue through `HeadlessDevice::wgpu_device()` and `HeadlessDevice::wgpu_queue()`, and
the buffer through `GpuBuffer::as_wgpu_buffer()`. Both submissions go to one queue, so the
compare pass sees the histogram without a barrier of its own or a readback in between: the queue behaves as if its
submissions ran in order and wgpu inserts the barriers (an ordering, not a promise that the earlier work has finished
before the later work starts). The tests cover a waited typed `dispatch` and an unwaited typed `submit`.

Run with `cargo run -p buffer-handoff`; the tests are `cargo test -p buffer-handoff -- --include-ignored --nocapture`.
