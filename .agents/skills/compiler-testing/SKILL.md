---
name: compiler-testing
description: "Add or diagnose compiler regression tests, compile_fail doctests, span diagnostics, emitted Slang assertions, or independent host-reference/GPU differential tests. Use when validating compiler changes; not as a mandatory full test sweep for documentation edits."
---

# Compiler testing

Choose the smallest test that can disprove the change, then expand only across affected compiler boundaries. Keep the locked toolchain and dependency versions.

1. Validator: accepted fixture plus single-cause rejection test with a meaningful diagnostic, in `crates/gust-macros/src/tests`. A `compile_fail` doctest in `crates/gust/src/lib.rs` covers source that rustc itself must still reject.
2. Emitter: source -> expected Slang semantics against the reviewed goldens in `tests/fixtures/`, then actual slangc compilation for supported targets.
3. Runtime: expected values from an independent host reference against GPU readback, including partial workgroups.
4. Run focused tests before workspace checks. Never regenerate expected diagnostics or goldens automatically to make a test pass.

Read [test layers](references/test-layers.md) for regression selection. Pick the check with `cargo xtask check-feature <area>` or `cargo xtask check-changed`; `cargo xtask check-format`, `cargo xtask check-lints` (Clippy, warnings denied) and `cargo xtask check-workspace` run locked formatting, lint and workspace tests. They report cargo checks only; they do not certify a GPU run or that every intended fixture exists. Evidence tiers are defined in `docs/development/VALIDATION.md`.

CPU twin tests (M3): a twin assertion adds to the independent host-reference and GPU assertions and never replaces them; the twin is never the expected value (D26). A new CPU-versus-GPU difference follows the ledger procedure: add a `KM-nn` row to `docs/SEMANTICS.md` section 7 and its case to `crates/gust-wgpu/tests/mismatch.rs` (an id guard fails if only one is done), record the measured GPU outcome, pin the Rust and twin outcomes in the overflow mode of the test crate, and never loosen an assertion. Do not chase a match for an edge case (D39). The twin opt-out is `#[kernel(cpu = "unsupported")]` (GUST-E0709, GUST-E0710 fixtures under `tests/ui`); `cargo xtask check-cpu` and `cargo xtask check-dual --check-only` need no GPU.
