---
name: rust-gpu-ast-validation
description: "Implement or review the restricted Rust GPU dialect validator, module boundaries, syn traversal, rustc versus macro checks, source spans, or rejected-syntax diagnostics. Trigger when accepting or rejecting Rust constructs, not for ordinary host Rust code."
---

# Rust GPU AST validation

Validate the whole GPU module before direct syn AST -> Slang emission. Read current dialect tests and validator first; never infer supported Rust from what syn can parse.

1. Locate the public GPU boundary and write down accepted syntax, types, calls and intrinsics for this change.
2. Use an allowlist with explicit rejection of unknown variants, tokens and modifiers. Recurse into every child, including types, attributes, signatures, patterns and nested expressions.
3. Separate syntactic checks from name/type/alias checks. A proc macro does not receive rustc's typed AST. Ensure rustc checks the intended preserved/generated Rust, or implement the restricted semantic check explicitly.
4. Emit a spanned diagnostic, not a panic or silently omitted shader code. Add a positive fixture and a targeted rejection fixture.

Read [validation contract](references/validation-contract.md) for traversal and semantic pitfalls. For diagnostic fixtures load `compiler-testing`; do not load lowering/runtime references for a validator-only change.

Map functions (M7, m7.a1, D60): `#[gpu]` on a free function has one validator, `validate_module`. The generator checks the signature itself (GUST-E0714, GUST-E0715, and GUST-E0204 for a 3-wide parameter or result) and validates the user's function as a helper of a one-function module with its attributes cleared, so every body rule is the helper rule and the 3-wide types stay legal as locals. `#[gpu]` or `#[gust::gpu]` on a function inside a `#[gpu]` module is GUST-E0716 (`is_gpu_attribute`); a rejected module goes back beside the error without those attributes (`without_nested_gpu` in `lib.rs`), so the inner attribute does not expand by itself and add follow-on errors. Fixtures: `tests/ui/map-*`.

Workgroup memory and barriers (M7, m7.g1; `validate/barrier.rs`, run from `validate/mod.rs` only after the generic visitor found no error, so a construct rejected for its own sake gets no second error): `workgroup_barrier()` is a statement-only call, accepted as a statement of a kernel body or of a top-level `for a..b` whose two bounds are integer literals, and anything else is GUST-E0414 (a helper, an `if`, a nested `for`); a kernel with a `Workgroup<[T; N]>` parameter or a barrier call has no `return` (GUST-E0415) and no `break` or `continue` in a `for` loop that holds a barrier (GUST-E0416). `Workgroup` parameters are legal on kernels only, with an element type of `f32`, `i32`, `u32`, `bool` or a vector of them, and all arrays of a kernel together stay within 16384 bytes (GUST-E0718, checked at expansion). Such a kernel must say `cpu = "unsupported"` (GUST-E0717); `cpu = "lockstep"` is GUST-E0709. Fixtures: `tests/ui/barrier-*`, `workgroup-over-16kib`, `kernel-cpu-lockstep`, and the rejection tests in `crates/gust-macros/src/tests/workgroup.rs`.
