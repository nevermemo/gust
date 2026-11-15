```

`dispatch` returns when the work is done. `submit` returns a value that implements the `gust::Job` trait instead, so a
program can keep several launches in flight or poll one; see `gust::Job` and `gust::JobStatus`.

## Several calls in one submission

A loop of dependent steps should not pay a submit-and-wait per step. `dispatch_batch` takes a slice of
`BufferDispatch` entries and runs them in one command buffer, one pass each, and every pass sees the writes of the one
before it. A typed `Call` converts into a `BufferDispatch`, so the call is built once and borrowed by every entry.

G4, `particles`, does that. Its kernel updates a struct with array fields (`Particle` has `pos: [f32; 2]` and
`vel: [f32; 2]`; the layout is Rust's own `repr(C)`, so the host fills the buffer with the same type), and `main` runs
sixteen steps in one batch and compares every particle with a host reference written in plain Rust. It also shows
`gust::par_for!`, a function-like macro that writes a kernel at the call site: `energies` computes a value per
particle with a loop body, naming the device with `on(device)`. A `par_for!` body is validated as a kernel with the same
dialect and codes, every binding is independent, and the iterations run in no defined order. It runs on the device you
name, or on the process's default device without `on(..)`; it needs a CPU twin only when it must also run in
`cargo test`, which is what the `Dev` alias in the listing is for (the GPU in the program, the CPU twin in tests).

```rust,no_run
