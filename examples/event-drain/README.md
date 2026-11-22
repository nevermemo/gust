# event-drain

A bounded GPU-to-host event queue that the host drains late, without blocking (m11.events). 1024 walkers move on a
periodic 64 x 64 grid for 24 steps; a walker that lands on a cell whose index is a multiple of 64 raises an event
`(step, walker, cell)`. The kernel appends it with `let slot = atomic_add(&mut counters[0], 1)`: with `slot < cap` it
writes the event at `slot`, else it counts it with `atomic_add(&mut counters[1], 1)`. So per step the queue holds the
head (the number of events), the overflow (`head - cap`, when positive) and up to `cap` events.

- Two buffers per queue: `counters` (word 0 the head, word 1 the overflow) and `cap` events of three words (a buffer
  that takes an atomic operation in a kernel refuses plain access there, so the events are apart). `clear` zeroes the
  counters of the queue a step is about to write; stale events stay and are ignored (only `min(head, cap)` count).
- The slots are assigned by the schedule, so the order of the events is unspecified. The checks compare per step the
  events as a set with the host's, the head with the number of events, and, when a queue overflows, the overflow with
  `events - cap` and the stored events as distinct members of the host's set.
- Cases: room for every event, a queue of 4 that overflows in most steps, and 7 walkers with steps that raise nothing.
- Host loop (see the module comment of `src/app.rs`): a ring of `LAG + 1` queues, step `s` in queue `s % QUEUES`. Per
  frame the host submits one step with `submit` (no wait) and starts the read-back of its queue at once
  (`GpuVec::to_vec_async`; its first poll submits the copy right behind the step's kernels). For each step that is
  `LAG` or more frames old it asks its job for `status` (one non-blocking poll) and, when complete, polls the
  read-back once with a no-op waker; a poll polls the device once and never waits. The loop never waits on the GPU
  (the first submit of a kernel may compile its Slang); the only blocking `wait` is at shutdown, for a step still
  undrained after `MAX_FRAMES` frames. It reports how many drains were late and by how many steps, how many
  completed only in the spins after the last frame, and how many steps drained at shutdown (the GPU test requires 0).
- The GPU run is on the Radeon 890M and the RTX 5090.
- No host API was added: `submit`, `WgpuJob::status`, `GpuVec::to_vec_async` and `write_uniform` are public.

Run with

```text
GUST_PROFILE=dev cargo run --release -p event-drain
```

(`GUST_PROFILE=dev` is needed because a plain release build is strict and refuses run-time Slang, which this crate
turns on; see "Build profiles" in the top-level README). The CPU tests run with `cargo test -p event-drain`, and the
GPU tests with `cargo test -p event-drain -- --include-ignored --nocapture`.
