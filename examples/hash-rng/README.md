# hash-rng

A stateless random number generator for compute kernels: the result is a pure function of a seed and an index, so
every device produces the same numbers.

- `hash(seed, index) -> u32`: a multiply-xorshift integer hash built from `wrapping_mul`, `wrapping_add`, `^` and
  `>>` with literal shift amounts below 32.
- `unit(h) -> f32`: the top 24 bits of `h` mapped to `[0, 1)` by multiplying with 2^-24. Conversion and multiply are
  exact, so the float is bit-equal on every device.
- The kernel `fill` writes `hash(seed, i)` and `unit(hash(seed, i))` into two buffers.
  The dispatch requires every buffer to have one element per invocation, so the seed is passed as a buffer that
  holds the seed in every slot.

Notes:

- A shift of 32 or more is outside the domain (D39); every shift here is a literal below 32.
- Bit equality (plain Rust reference, CPU twin, GPU) is measured on the Radeon 890M only; other adapters are
  unmeasured.
- This is a hash for simulation, not for cryptography.
- A `#[gpu]` module refuses `use`, `const`, `static` and nested `mod`, so the helpers must live in the module that
  uses them and the constants are written as literals in place.

Run with `cargo run -p hash-rng`; the test is `cargo test -p hash-rng -- --nocapture`.
