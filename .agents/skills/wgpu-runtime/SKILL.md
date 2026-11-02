---
name: wgpu-runtime
description: "Implement or debug headless wgpu compute execution, shader module validation, bind groups, buffer layout, dispatch, asynchronous readback or host-reference comparisons for this compiler. Use for actual runtime behavior, not solely Slang text generation."
---

# wgpu runtime

Consume compiled shader artifacts plus explicit binding/layout metadata; keep syn internals out of the runtime. Match the repository's pinned wgpu API and features.

1. Record adapter name, device type, backend, requested features/limits and shader hash. A missing adapter is unavailable execution, not a passing kernel.
2. Validate shader and pipeline, bind exactly the declared ABI, dispatch rounded-up workgroups, copy to staging, wait for mapping and compare readback.
3. Test partial groups, empty input and invalid host lengths. Propagate validation, device, mapping and comparison errors.
4. Do not substitute CPU execution or a different shader when reporting GPU success.
5. Native readback futures never wake themselves: wgpu runs the `map_async` callback inside a device poll, so `read_typed_buffer_async` completes only while something polls the device. The blocking wrappers (`read_typed_buffer`, the `_timed` and one-shot functions) therefore call `poll(Wait)` first and poll the finished future once; awaiting the future on a native executor that sleeps until woken hangs.
6. The `wasm32` surface (browser, `webgpu` feature enabled only for wasm32) has no blocking and no clocks: `new_async`, buffers, typed `dispatch`/`submit` and `read_typed_buffer_async` exist; the blocking readbacks, the `_timed` and one-shot `dispatch_f32`/`dispatch_u32` functions, the `graph` module and `pool_read` are `cfg(not(target_arch = "wasm32"))`, and `WgpuJob::wait`/`status` return at once there, so synchronise with an awaited readback. Gate new blocking or timed code the same way, and run `cargo xtask check-web` (needs Chrome) after touching it.

7. The shader a kernel runs is its embedded WGSL artifact (`KernelDescriptor::artifacts`), checked by `gust::artifact::verify_artifact` against the descriptor (D17) before a pipeline exists; only a kernel without one, in a build with the `slang-runtime` feature (`dev`), is compiled by Slang at run time, and a strict build without the artifact is `Error::NoArtifact`. wgpu and the driver still compile the WGSL. A missing Vulkan adapter is `Error::NoAdapter`; report any other initialisation failure as what it is (the precompiled consumer prints `GPU initialization failed:` for those), so a test or CI step that expects "no adapter" cannot be satisfied by another error.

Read [headless execution](references/headless-execution.md) for buffer lifecycle and test requirements. `cargo xtask test-exact -p <package> --test <target> <name>` runs one existing standard Rust integration test by exact name with `--include-ignored` and rejects zero, ignored or repeated tests. It certifies test execution, not the test's implementation; inspect the test for real GPU assertions.
