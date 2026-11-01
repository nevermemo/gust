---
name: gpu-vertical-slice-verification
description: "Verify a complete restricted Rust GPU kernel through AST validation, direct Slang emission, slangc WGSL/SPIR-V compilation, validation and headless wgpu numerical readback. Use for end-to-end acceptance or locating the first failing compiler boundary."
---

# GPU vertical slice verification

Require a traceable Rust source -> validated syn AST -> emitted Slang -> compiled shader -> wgpu readback chain. Do not introduce custom IR or accept a hand-authored shader or a host fallback as evidence of Rust compiler execution.

1. Pick one meaningful kernel (an example under `examples/`) and define expected outputs plus boundary cases from an independent host reference.
2. Run it: `cargo run -p <example>` executes on the selected adapter and exports `generated-wgpu/` (ignored, never committed).
3. Check the whole export set: `cargo xtask check-artifacts` runs every example binary and validates each exported SPIR-V with `spirv-val --target-env vulkan1.2`. For release evidence run `cargo xtask check-full`, which adds the ignored example tests and records `.ai/VALIDATION.json`.
4. Report each stage as PASS, FAIL or UNAVAILABLE with the command and result, and label the claim with the highest evidence tier it reached (Unit, Slangc, Spirv, Reflect, Gpu) as defined in `docs/development/VALIDATION.md`. A missing tool or adapter fails its tier; it is not a skip.

Read [evidence chain](references/evidence-chain.md) for what each command proves and for the single-stage commands that locate the first failing boundary. `cargo xtask doctor --smoke` is a standalone Slang/SPIR-V toolchain smoke test on a hand-written kernel; it is not vertical verification.

CPU twins (M3): in a test or dual build the same typed call also runs on `gust::cpu::CpuDevice`. A twin assertion is added to the host-reference and GPU assertions and never replaces either, and twin agreement is not GPU evidence: the Gpu tier still needs readback on a device equal to the independent host reference. When the twin and the GPU differ at an edge case, add a row to the known-mismatch ledger (`docs/SEMANTICS.md` section 7, pinned by `crates/gust-wgpu/tests/mismatch.rs`) with a measured GPU outcome, and never loosen an assertion to hide it. Edge-case differences are documented, not chased (D39). `cargo xtask check-dual` is the cross-crate proof that the twin exists only in dual builds.
