# T12: Statement Result match

## Contract

A direct, semicolon-terminated statement match over a known `Result<T, T>` value:

```rust
match result {
    Ok(value) => { /* statements */ }
    Err(error) => { /* statements */ }
};
```

Exactly one plain `Ok(identifier)` block arm and one plain `Err(identifier)` block
arm are required, in either source order. The scrutinee is evaluated once into a
reserved temporary before lowering to Slang `if/else`.

## Rejected

Value/semicolon-less matches, non-Result scrutinees, missing or duplicate arms,
guards, attributes, wildcard/nested/`mut`/`ref`/`@` patterns, non-block arms, and
bindings that shadow resource or dispatch parameters.

## Evidence

- Macro: `cargo test -p gust-compute-macros -- result_match`
- Result vertical slice: `cargo test -p gust-compute-wgpu --test result`
- Routine semantic group: `cargo xtask check-feature result`
- Release: `cargo xtask check-full`

The GPU differential executes both arms at 1/63/64/65/257 elements on NVIDIA
GeForce RTX 5090 / Vulkan.
