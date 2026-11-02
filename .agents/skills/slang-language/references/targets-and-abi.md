# Slang target and ABI notes

Use the repository's pinned slangc first (the version in `SLANG_VERSION` of `crates/cargo-gust/src/slang.rs`, installed by `cargo gust slang install`; `cargo gust doctor` names a different `slangc` first on `PATH`); record `slangc -version`. `cargo xtask slangc` specifies `-entry`, `-stage compute`, `-target` and `-o` explicitly and records them in `<out>.command.json`. Extra profile/capability/include arguments are caller supplied after `--`, not guessed; SPIR-V that must keep the descriptor's entry point name needs `-fvk-use-entrypoint-name`, as the Slang bridge passes. Compile WGSL and SPIR-V separately to expose target restrictions.

Project design guidance:
- Use an explicit `(set, binding, access, element layout)` contract shared by emitter and runtime. Slang `[[vk::binding(binding, set)]]` uses binding first; wgpu uses group=set.
- Keep module resources and helper names deterministic. Do not trust compiler auto-allocation to match independently handwritten host layouts.
- A compute entry uses `[shader("compute")]`, `[numthreads(x,y,z)]` and `uint3 ... : SV_DispatchThreadID`. Dispatch counts are workgroups, not invocations.
- Resource types such as `StructuredBuffer<float>` and `RWStructuredBuffer<float>` have different access requirements. Do not translate Rust references into arbitrary shader pointers.
- For portability start with scalar 32-bit storage elements. Rust struct layout alone is not proof of shader layout; verify offsets, alignment, stride and padding. Avoid host bool storage and vec3 packing assumptions.
- The bundled shader uses four storage bindings and a one-u32 length buffer for a simple explicit ABI. It is a toolchain fixture, not a mandated project ABI.
- Keep a bounds check around reads and writes when dispatch rounds up. Validate all input lengths on the host; checking only output length is insufficient.

WGSL has target-specific restrictions and binding behavior. Check the feature in the pinned Slang version and test the generated WGSL through wgpu validation. Do not work around errors by silently switching targets.

Official sources, checked 2026-09-07:
- [slangc options](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md)
- [WGSL target](https://shader-slang.org/slang/user-guide/wgsl-target-specific)
- [First compute shader](https://docs.shader-slang.org/en/stable/first-slang-shader.html)
