# particles (gallery G4)

Particles falling under gravity and bouncing on a floor, stepped sixteen times in one batch. A particle is a `#[gpu]`
struct with **array fields** (`pos: [f32; 2]`, `vel: [f32; 2]`, and a `bounces` counter), stored in a
`RWStructuredBuffer<Particle>` that the host fills with the same Rust type.

What it shows:

- A struct with array fields as a buffer element, copied into a local, changed field by field (`p.vel[1] = ..`) and
  written back whole.
- A **batch**: the call `particles::step::call(n, &buffer)` is built once and submitted sixteen times with
  `dispatch_batch`, one command buffer with one pass per step; each pass sees the writes of the one before.
- `par_for!`: the energy of every particle is a loop written at the call site over flat `&[f32]` arrays, with no kernel
  declared by hand.

Every constant of the kernel is a power of two (a step of 1/8, gravity 8, a bounce that halves the speed) and the host
starts from small integers, so every value is a small multiple of a power of two and exactly representable in `f32`.
That is why the float checks are **exact** and have no tolerance: the host reference, in `f64`, holds the same number as
the kernel at every step. A kernel with constants like 0.016 would need a tolerance, as `saxpy-uniform` has.

Tests (`cargo test`, CPU twin only): 1, 3 and 16 steps of the kernel on `CpuDevice` against a host simulation of its own,
with a particle type of its own, compared field by field, at 1, 63, 64, 65 and 1000 particles (partial last workgroup in
all but 64); a hand-computed first step; a guard that some particles do bounce; and the `par_for!` energy on the twin
against the host sum. The twin has no batch, so its tests loop over single dispatches; the batch itself is GPU evidence
(`cargo run --release`, which `cargo xtask check-gallery` runs and compares; `main` also asserts every particle against the host simulation before it prints).

Edge case (D39): `par_for!` runs its iterations in no defined order on a GPU and in ascending order on the twin. This body
reads and writes distinct elements, so the order does not matter; a body that reads what another iteration writes would
differ between the two, and the dialect does not diagnose it (see SEMANTICS).

Commands, from this directory (the tools are set up as for `hello-gust`):

```sh
cargo test                        # the CPU twin against the host simulation; no GPU, no Slang
cargo run --features gust-dev     # development: compiles the kernels at run time through Slang, needs an adapter
cargo gust build                  # compiles the kernels with Slang and writes gust-artifacts/ (commit them)
cargo gust check                  # fails when gust-artifacts/ no longer matches the sources
cargo run --release               # no Slang needed; needs a Vulkan compute adapter
```
