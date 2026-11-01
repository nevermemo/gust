---
name: rust-to-slang-lowering
description: "Implement or debug direct validated syn AST to Slang emission, including expressions, control flow, helper functions, intrinsics and binding metadata. Trigger on generated-code semantic mismatches or adding supported dialect constructs; do not introduce a custom IR."
---

# Rust to Slang lowering

Emit Slang directly from the validated syn AST. Retain the module boundary. Small scoped symbol/type/binding tables and an emitter are appropriate; a custom instruction IR is future/optional and requires an explicit architecture request.

1. Read the validator, emitter and closest accepted fixture. State the source-to-target rule before editing.
2. Preserve evaluation order, scope, signedness, width, precedence and control flow. Map only validated intrinsics/calls.
3. Keep shader names, resource bindings and runtime metadata deterministic and generated from one contract.
4. Add a focused source -> Slang assertion, compile affected targets, and execute a nontrivial semantic case when behavior changes.

Operators whose Rust meaning depends on the operand type go through `__gust_*` overload helpers, emitted only when the module uses them, because the macro sees syntax and not types: unary `!` is `__gust_not` (`!` on `bool`, `~` on integers), and `^` and `^=` are `__gust_xor` and `__gust_xor_assign` (`!=` on `bool`, `^` on integers), since WGSL defines `^` only on integers; the `_assign` form takes the place `inout`, so it is evaluated once. `&`, `|`, `&=` and `|=` on `bool` pass through unchanged.

Known gap: `f(a < b, c > d)` and `(a < b) ^ (c > d)` misparse as a WGSL template list until fx.wgsl.

Content key, embedding and export records (M4, D45, D46; `crates/gust-macros/src/expand/artifact.rs`): once a kernel's Slang text is final, and before the CPU twin is generated, the macro computes `content_key(entry_point, clean_slang)`: FNV-1a 64 over the entry point, one NUL byte and the clean text (no `#line` directives, no line-ending normalisation), as 16 hex digits. The function is a byte-for-byte copy of `gust::artifact::content_key`; both crates assert the same golden strings, so change neither alone. When `gust-artifacts/<key>.wgsl` and `<key>.wgsl.reflection.json` both exist under `CARGO_MANIFEST_DIR` the kernel's `artifacts` is the one-element slice built from `include_bytes!` and `include_str!` of `concat!(env!("CARGO_MANIFEST_DIR"), "/gust-artifacts/..")`, with `require_artifact(true)` (the module gets one `require_manifest(<manifest exists>)` for all its kernels, at the first kernel with a pair); with neither it is `&[]` and `require_artifact(false)`; with one of the two it is a coded `compile_error!` (E0711) at the kernel naming the missing file and the one to delete, never a silent `&[]`, except in export mode, where a half pair counts as absent so that `cargo gust build` can repair it (E0712 and E0713 are the export write and package-name errors). An existing `gust-artifacts/manifest.json` is included once per module so cargo tracks it. The macro never reads `GUST_PROFILE`: the strict checks live in `gust::profile`. With `GUST_EXPORT_DIR` set and non-empty, each kernel also writes `<dir>/<CARGO_PKG_NAME>/<key>.slang` (the clean text, exact bytes) and `<key>.kernel.json` (schema 1, sorted keys, no timestamp or path; layouts are the macro's StorageV1 claim, `null` for an element type it cannot describe); `write_records` refuses any record that holds generated CPU twin forms (`TWIN_TEXT`: the `cfg`, the `gust::cpu::` paths and the reserved prefixes; a user name such as `CpuDevice` or `gust_dual` is not twin text), and the export keeps `require_artifact(<present>)`. Test these with `expand_in` and an `Environment` of your own, never by changing the process environment; the `kernel.json` goldens live in `crates/gust-macros/tests/golden` and are blessed by `cargo xtask bless`.

Read [semantic mappings](references/semantic-mappings.md) for risky translations. Use `slang-language` only for target compilation issues and `gpu-vertical-slice-verification` for an end-to-end claim. Report frontend, shader and runtime evidence separately.
