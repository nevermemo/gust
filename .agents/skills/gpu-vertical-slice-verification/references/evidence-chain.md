# Evidence chain

The repository already contains the vertical slice: the `#[gpu]` macro, the Slang bridge, the native reflection gate (D17), the headless wgpu runtime, `cargo gust` (committed, content-keyed WGSL and SPIR-V artifacts under `gust-artifacts/`, D45) and nine examples that exercise them. There is no adapter-script layer; `cargo xtask` is the command surface (D27). Do not claim GPU acceptance before the Gpu tier ran on a device.

## What each command proves

| Stage | Command | Highest tier it reaches |
| --- | --- | --- |
| Rust -> Slang (validator, emitter, goldens) | `cargo xtask check-feature macro` | Unit |
| Slang -> WGSL | WGSL compile smoke tests in `cargo xtask check-fast` | Slangc |
| Slang -> SPIR-V, validated | `cargo xtask check-artifacts` | Spirv |
| Descriptor layout equals Slang's reflection | `cargo xtask check-feature reflection` | Reflect |
| Readback equals an independent host reference | `cargo xtask check-feature gpu-semantics`, `cargo xtask check-examples` | Gpu |
| Committed artifact runs with no Slang in the environment (D45) | `cargo xtask check-precompiled` | Gpu (the runs of E1, E3, E8) |
| Same kernel on a browser GPU | `cargo xtask check-web` (Chrome with WebGPU) | Gpu (WebGPU, Chrome version, adapter) |
| All of the above plus the example binaries | `cargo xtask check-full [--skip-web]` | Gpu (records `.ai/VALIDATION.json`; `skipped_checks` names a skipped check) |

Tiers, the label format and the software-adapter rule are in `docs/development/VALIDATION.md`. `check-artifacts` compares the exported `.spv` names with the expected list in `xtask/src/workspace.rs` (`EXPECTED_SPIRV`); a new example or kernel updates that list deliberately. Compile and validation success does not prove the kernel computes the Rust program: only the Gpu tier does, and only as far as the test's assertions go.

## Locating the first failing boundary

Work forward through the chain and stop at the first failure. The commands below take the Slang that an example exported to `generated-wgpu/` and write to fresh paths (they refuse an existing output).

```sh
cargo xtask check-feature macro
cargo xtask slangc generated-wgpu/vector_add__add.slang --entry gpu_vector_add_add --target wgsl --out target/probe/vector_add.wgsl
cargo xtask slangc generated-wgpu/vector_add__add.slang --entry gpu_vector_add_add --target spirv --out target/probe/vector_add.spv -- -fvk-use-entrypoint-name
cargo xtask spirv-val target/probe/vector_add.spv --target-env vulkan1.2
cargo xtask test-exact -p gust-wgpu --test semantics translated_semantics_execute_on_gpu
```

Each `slangc` output gets a `<out>.command.json` (compiler version, arguments, output SHA-256) and a `<out>.compile.log` beside it. `test-exact` runs one integration test by exact name with `--include-ignored` and fails unless libtest reports one executed, passing test; it certifies execution, so read the test for real GPU assertions.

`cargo xtask doctor --smoke` compiles `.agents/skills/slang-language/assets/vector-add.slang` to WGSL and SPIR-V in a fresh directory under the cargo target directory and validates the SPIR-V. It checks the toolchain only: the Rust frontend and GPU execution are not run.
