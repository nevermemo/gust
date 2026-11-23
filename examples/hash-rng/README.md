# hash-rng

A stateless random number generator for compute kernels: the result is a pure function of a seed and an index, so
it is designed to give the same numbers on every device (exact integer operations, exact conversion and a
power-of-two scale). It is measured on the Radeon 890M only.

- `mix(x) -> u32`: a multiply-xorshift finaliser built from `wrapping_mul`, `^` and `>>` with literal shift amounts
  below 32. The constants and shifts are "lowbias32" from Chris Wellons' hash-prospector
  (https://github.com/skeeto/hash-prospector, public domain / Unlicense).
- `hash(seed, index) -> u32`: `mix(mix(seed) ^ index)`. The seed is mixed first, so different seeds give unrelated
  streams, not index-shifted copies of one.
  `mix(0)` is 0, so `hash(0, 0)` is 0; a seed of 0 is valid but starts at zero.
- `unit(h) -> f32`: the top 24 bits of `h` mapped to `[0, 1)` by multiplying with 2^-24. Conversion and multiply are
  exact, so the float is designed to be bit-equal on every device (measured on the Radeon 890M only).
- The kernel `fill` writes `hash(seed, i)` and `unit(hash(seed, i))` into two buffers.
  The seed is a one-element buffer bound with `Independent`.

Notes:

- A shift of 32 or more is outside the domain (D39); every shift here is a literal below 32.
- Bit equality (plain Rust reference, CPU twin, GPU) is measured on the Radeon 890M only; other adapters are
  unmeasured.
- This is a hash for simulation, not for cryptography.
- A `#[gpu_module]` module accepts module-level `const` items since m11.g3 (the kernels read them and so does the host);
  this example still writes its constants as literals in place. The helpers must live in the module that uses them.

Run with `cargo run -p hash-rng`; the CPU tests run with `cargo test -p hash-rng`, and the GPU test with
`cargo test -p hash-rng -- --include-ignored --nocapture`.
