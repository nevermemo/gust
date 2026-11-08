---
name: spirv-validation
description: "Validate or inspect emitted SPIR-V modules, Vulkan target environments, capabilities, entry points and resource decorations using SPIRV-Tools. Use for .spv artifacts and backend validation errors; validation alone never proves kernel semantics."
---

# SPIR-V validation

Validate the exact fresh artifact produced by slangc, against the runtime's declared target environment.

1. Record input hash, slangc version/options and `spirv-val --version`.
2. Run `cargo xtask spirv-val <file.spv> --target-env <env>` with an explicit environment such as `vulkan1.2` only when that matches the runtime contract. `cargo xtask check-artifacts` validates every exported kernel this way.
3. If it fails, use `spirv-dis` to inspect entry point, execution mode, capabilities and decorations. Fix generation or capability negotiation rather than disabling validation.
4. Report structural validation separately from wgpu acceptance and numerical GPU execution. A valid no-op module is still a no-op.

Read [environment and diagnosis](references/environment-and-diagnosis.md) only for environment selection or validation failures. The command requires the validator and fails on missing, empty, malformed or invalid input.

Workgroup barriers (M7, m7.g1): `GroupMemoryBarrierWithGroupSync()` is `OpControlBarrier` (opcode 224) in the SPIR-V and `workgroupBarrier()` in the WGSL, and `groupshared` arrays are `Workgroup` storage-class variables (`var<workgroup>`). `crates/gust-wgpu/tests/workgroup.rs` counts the opcode by walking the module and cross-checks the disassembler's name for it, then runs `spirv-val --target-env vulkan1.2`. Validity of the barrier says nothing about whether every invocation reaches it; that is the validator's position rule (GUST-E0414 to E0416) and the GPU block-reduce test.

Fragment and texture modules (MX, D64, D120): a `pixel` kernel's entry point is a `Fragment` execution model, compiled with `-stage fragment` (`gust::slang` takes the stage from `Shape::slang_stage`; `cargo xtask slangc <file> --entry <name> --stage compute|fragment ...` defaults to compute, and a golden's per-entry stage is read from its `[shader("..")]` line by `shader_entries` in `xtask/src/shader.rs`). Validate it like any module (`spirv-val --target-env vulkan1.2`). The golden sets that `verify` and `check-artifacts` export and validate are `GOLDEN_SPIRV_SETS` in `xtask/src/checks.rs` (13 areas: M8 added `variant`, 2 files, and `subgroup`, 12 files, with one SPIR-V module per body of a kernel that has a subgroup variant; each exported module is validated with `vulkan1.2` like the others, and `spirv-val` accepting a subgroup module says nothing about whether wgpu or the driver runs it, which `tests/subgroup.rs` shows on an adapter); the MX ones are `textures` (8 files, `textures__<kernel>.spv`, from `tests/fixtures/textures.slang`) and `pixel` (3 files: `pixel__pack`, `pixel__row`, `pixel__shade`), and the example `texture-threshold` adds `threshold__mark.spv` to `EXPECTED_SPIRV`. A set gains or loses a file in the same commit as its golden. A valid fragment module says nothing about the rest: whether wgpu accepts the pipeline, whether the adapter can write storage buffers from a fragment stage (`FRAGMENT_WRITABLE_STORAGE`), which fragments run, or what they write. The runtime executes WGSL through a render pipeline (`crates/gust-wgpu/src/render.rs`), not these SPIR-V files, and SPIR-V execution of the goldens is not done (placed in MR, D95).
