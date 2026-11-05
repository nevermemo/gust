# T11: Result lowering

## Contract

Local/helper `Result<T, T>` lowers to `__GustResult<T>`.

- Supported: `Ok`, `Err`, `is_ok`, `is_err`, `unwrap_or`, `if let Ok`.
- Equal payload types are required because Slang cannot infer a missing generic
  error type from one-payload constructors.

## Rejected

Mixed or nested payloads, Result in structs/resources, tuples, slices, `unwrap`,
`expect`, combinators, `match` value expressions, and `?`.

## Evidence

- Macro: `cargo test -p gust-compute-macros -- result`
- GPU/targets: `cargo test -p gust-compute-wgpu --test result`
- Routine semantic group: `cargo xtask check-feature result`
- Release: `cargo xtask check-full`

The target test covers WGSL, SPIR-V plus `spirv-val`, Option/Result overload
resolution, and GPU readback at 1/63/64/65/257 elements.
