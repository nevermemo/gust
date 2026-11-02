---
name: slang-language
description: "Write or debug generated Slang compute shaders, resource bindings, entry points, target restrictions, or slangc WGSL/SPIR-V compilation in this Rust GPU compiler. Use for shader-language issues; use rust-to-slang-lowering for Rust emission rules."
---

# Slang language

Preserve restricted Rust GPU module -> validated syn AST -> direct Slang emission -> slangc -> WGSL/SPIR-V. Slang's internal compiler IR is outside this project's frontend architecture.

1. Inspect the failing emitted shader, binding metadata, command, and installed slangc version (the project pins one release: `cargo gust slang install` puts it in `GUST_SLANG_HOME` or the default cache, `cargo gust slang locate` prints it, `cargo gust doctor` names a `slangc` of another version; `cargo xtask` and the run-time bridge launch plain `slangc` from `PATH`). Reduce the shader before changing the Rust emitter.
2. Keep entry point, compute stage, workgroup dimensions, resource access and bindings explicit. Match the host ABI.
3. Compile each requested target independently. Success for SPIR-V does not establish WGSL compatibility or runtime correctness.
4. Return the smallest shader/emitter fix, exact command and target-specific result.

Read [target and ABI notes](references/targets-and-abi.md) only for language/target decisions. Run `cargo xtask slangc <file.slang> --entry <name> --target wgsl|spirv --out <path>` for repeatable compilation: it refuses an existing output path, fails on a missing or empty artifact, and writes `<out>.command.json` (slangc version, arguments, output SHA-256) and `<out>.compile.log` beside the output. Extra slangc arguments go after `--`. The [smoke shader](assets/vector-add.slang) tests tool installation, not Rust lowering; `cargo xtask doctor --smoke` compiles it to both targets.
