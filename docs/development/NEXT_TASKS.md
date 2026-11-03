# Next tasks

This is the active queue only. Completed narrative is archived in
[history/NEXT_TASKS-2026-09-09.md](history/NEXT_TASKS-2026-09-09.md); feature
contracts live in [tasks/](tasks/README.md).

Claim one bounded task in [STATUS.md](STATUS.md) before editing. A task is not
active until its owner, scope, and focused check appear there.

The ordered plan is [ROADMAP.md](../ROADMAP.md). Its "Now and next" table lists the
claimable items of the current milestone; this file keeps deferred work and task detail
that does not fit there.

## Deferred

### T13: GPU active count to indirect dispatch

Deferred by owner. The T13 work in progress is not in the working tree: it is the
git stash `WIP T13 active count deferred`, based on `59a5c3c`. Its code changes
(`examples/component-pool/src/app.rs`, `xtask/src/{checks,verify}.rs`) applied
cleanly to `63769fb` in a 2026-10-22 dry run; its doc renames are superseded by
that commit and conflict. Do not apply, continue, or drop the stash unless the owner
explicitly requests it. Contract: [tasks/T13-active-count.md](tasks/T13-active-count.md).

## Claimable DX

### DX20: Internal ownership splits

Split the largest modules only where a named responsibility can move without
changing public APIs: macro validation by concern, macro regressions by feature,
or wgpu runtime internals by ownership. Use a no-behavior-change contract and
one focused check per move. Start with extracting the Result/Result-match
regressions from `crates/gust-macros/src/tests/regression.rs` into a feature-owned
test module; do not mix that mechanical move with behavior changes.

## Later engine proofs

- GPU active count to indirect dispatch (T13), when the owner reactivates it.
- Culling -> compacted visibility -> indirect arguments -> rendering.
- Entity identity/lifecycle and storage choices, guided by measured workloads.

## M2 follow-ups (filed at M2 close, D37)

- O5: `max_buffer_size` is not checked; buffer creation is infallible, so an oversized
  buffer is still a wgpu panic. Take the error-scope route (`push_error_scope` /
  `on_uncaptured_error`) rather than another host pre-check, with a test that pins
  today's panic first. M10 API pass at the latest.
- O8: deprecate the `F32Binding`/`U32Binding` slice paths once typed calls cover every
  example's use (M10 API pass).
- O4: access typestate on `GpuBuffer` (`GpuBuffer<T, A>`) so a read-only buffer passed
  as read-write fails at compile time; changes every `create_*` signature and the pool.
- O3: a `#[gpu]` parameter attribute for independent length, if D15's call-site
  `Independent(&buf)` is revisited when M7's primitives need it.
- Kernels with a `Uniform<T>`/`ConstantBuffer<T>` parameter have no generated typed call,
  and the headless runtime rejects them (`Error::UnsupportedParameter`); plain value
  parameters are rejected by the macro. Resolved at M3 close (D40): uniform kernels stay
  without a typed call and without a CPU twin until M6's uniform work, which decides
  both together. Done in M6: m6.u1 (host side) and m6.u2 (merge `1b592a1`) give them typed calls and twins.

## M3 follow-ups (filed at M3 close)

- Done in M7 (m7.w, D87): explicit `wrapping_add`, `wrapping_sub`, `wrapping_mul` and
  `wrapping_neg` methods; D39 still drops any wrapping rewrite of twin bodies.
- A debug write-set race checker in `gust::cpu`: the twin runs invocations one after
  another, so a kernel in which two invocations write one location gives a clean twin
  result and proves nothing about the GPU (SEMANTICS section 5). The checker would
  record the written indices of a call in debug builds and report a repeated one.
- CPU limits (open after M5): decide whether the production CPU target emulates the device limits
  (workgroup count per dimension, invocations per workgroup, storage binding size) or
  keeps documenting the difference; today the CPU backend rejects only a launch above
  2^32 invocations.
- M7, parallel twin execution: run the invocations of a twin on several threads once
  barriers and shared memory exist. A CPU strategy for them is part of M7; until then a
  barrier kernel uses `#[kernel(cpu = "unsupported")]` (SEMANTICS section 5).
- M8, capability-gated tests that fail rather than skip: a test that needs a capability
  the adapter lacks is a failure of its tier, never a skipped or ignored pass.
- M10, benchmark fairness: bounds checking and panic behaviour must be equal on the
  compared implementations (the GUST CPU target, rayon, CubeCL, rust-gpu) before a
  benchmark is reported.
- Test pinning by name (known limitation of m3.g's test-count guard): a counted call
  fails when it reports no passed test, but deleting one test, or marking it
  `#[ignore]` outside an `--include-ignored` call, while its siblings still run passes.
  check-dual's E2, the release call's two release-only tests and, since m3.h, the twin
  and derivation-domain tests of each check-cpu example (`CPU_REQUIRED_TESTS`) are required
  by name; the example calls of `check-examples` and `check-full` and the filtered
  check-feature calls are still counted only.
- The module-wide `#[allow(unexpected_cfgs)]` that the `#[gpu]` expansion carries also
  silences a user's cfg typos inside the `#[gpu]` module; narrow it to the emitted twin
  items.
- `gust::cpu::invocations` iterates `workgroup_size` x only. `validate` rejects a y or z
  other than 1 (`UnsupportedWorkgroupSize`, as `gust-wgpu` does), so no typed call reaches the
  gap today; give `invocations` a guard of its own before M7 lifts the restriction.
- If kernels ever become generic, re-check that a defining crate's overflow checks survive instantiation in an
  unchecked crate (the defining-crate rule is measured for non-generic shadow bodies only).
- Optional: `cargo test -- --color never` in the counted calls would remove the ANSI
  stripping (`without_colors`) from the count guard.

## M4 follow-ups (filed at M4 close)

- `gust::slang` checks `slangc -version` against the pin (D72). Then remove the `__gust_keep` wrapper behind the
  naga comparison and a GPU read-back (m4.z measured that 2026.20 no longer needs it; 13.1 fails 13 of 30 shapes).
- `gust::Job` on wasm32 as D69 states: `status` from `on_submitted_work_done`, `wait` returns an error,
  `Device::dispatch` and `dispatch_call` gated. Until then `WgpuJob::wait` is a documented no-op.
- The wasm32 `disallowed-methods` ban (`std::thread::sleep`, `Instant::now`) for `gust` and `web-vector-add`, as
  `gust-wgpu` has it (m4.z).
- Browser automation in CI (S4, D69): measure a software WebGPU adapter on a Linux runner first.
- `layout: null` element types are exported unchecked (D71): describe them from the type, or refuse them.
- M10 publication: `cargo publish` strips self dev-dependencies (D68); docs.rs needs `package.metadata.docs.rs` and
  `doc(cfg)`; registry users see no inferred-profile warning.
- `xtask/src/bless.rs` hashes only `tests/fixtures/*.slang`, so blessing other goldens reports a false failure.

## M5 follow-ups (filed during M5)

- Lower integer `/` and `%` through `__gust_div`/`__gust_rem` helpers so slangc cannot fold a zero divisor (D74):
  done in m6.dv (merge `94baab5`, D77). Format 5 keeps no divisor-rule discard (the 13 of format 4 are gone), so
  no class needs naming.
- The StorageV1/StorageV2 wording mix in the struct layout asserts (pinned in xtask ui tests).
- gpu-software installs Mesa from one Ubuntu snapshot; m5.f1 added retries, but a long snapshot.ubuntu.com outage
  (three failures on 2026-11-02) still stops it. A tested fallback source (wgpu's CI uses `gfx-rs/ci-build`) would
  need its own identity assertions (D52).
- Before M8: reconcile D65 (CPU bodies in `dev`, `native` and `web` production builds) with D26 and its fixtures
  (`typed-call-no-cpu-twin`, check-dual E1 and E3) and with the macro's one process-wide profile read (M5 sanity
  check; in the M8 row).

## M6 follow-ups (filed during M6)

- `cargo xtask check-artifacts` does not export the `#[gpu]` modules of integration tests (for example
  `crates/gust-wgpu/tests/vectors.rs`); their kernels are checked only where the tests run them (m6.g1). Done
  in m6.x for the three golden modules (products, vectors, uniforms); other test modules are still unexported.
- A constant float2x2 or float3x3 (`Mat2::IDENTITY`, `Mat3::IDENTITY`) crashes the runner when its SPIR-V is
  handed to wgpu (exit 139). The faulting layer is **not isolated**: the M6 probe notes say naga's module creation
  passed and `create_compute_pipeline` crashed, so "naga's SPIR-V front end" is not established. Isolate the layer,
  then reduce it to an upstream report (naga, wgpu or the driver) or lower such constants another way (m6.g1;
  sanity-fable and sanity-opus, 2026-11-02, both asked for "not isolated").
- **Literal-local typing: fixed as a compile error by m6.z2 (GUST-E0326, D96); kept for the record.** An
  unannotated `let` whose initializer is an integer literal was emitted as Slang `var x = 7`, which is `int`, whereas
  rustc types the local from its later uses. Measured: `let mut d = 7; d /= y` with `y: u32` divides signed, so for
  `y >= 2^31` Rust gives 0 and the GPU gives `0xFFFFFFF9` (found in the m6.dv review; older than m6.dv). Signed
  comparisons (`d < y`), `>>` and casts of such a local very likely go wrong the same way; that part is inferred from
  the lowering (`slang/mod.rs` emits `var` for every unannotated local), **not measured**. It is a silent
  difference inside D22's domain, not a D39 edge case. The difftest emitter always annotates its `let`s
  (`tests/difftest/src/emit.rs:207`), so no gate sees this class. Decided approach: type the local from its uses
  when they agree, otherwise a compile error asking for a suffix or an annotation; a GPU regression case for
  `d /= y` with `y >= 2^31`, a signed comparison and `>>`; the difftest emitter writes some unannotated lets
  (sanity-opus, widened and made a blocker for m7.a; sanity-fable, scheduled before MR; 2026-11-02).
- shader-slang/slang#14198 (open, opened by the owner on 2026-11-02): WGSL emits `^` on `bool` and `vecN<bool>`
  operands, which WGSL does not define; the PR makes it `!=`. Once it is in a Slang release the pin adopts, the bool
  overloads of `__gust_xor` and `__gust_xor_assign` can go (a pin bump with the usual re-bless; D46). Until then the
  overloads stay. Recorded here because the repository did not mention it (sanity-fable and sanity-opus, 2026-11-02).
- SPIR-V is never executed by any M6 case: the Spirv tier of M6 is `spirv-val` of the exported modules, and
  `gust-wgpu` runs WGSL. The matrix layout decorations of the SPIR-V goldens and Std140 uniforms in SPIR-V have never
  been dispatched, yet MR ships SPIR-V to Vulkan users. Dispatch the three goldens (vectors and products, uniforms)
  through SPIR-V passthrough on the RTX 5090; this is input to the MX/MR planning, not an M6 task (sanity-opus,
  2026-11-02; ruling 4).
- Gate the layouts of the emitted WGSL with naga's validator and `proc::Layouter` (wgpu's default capabilities),
  which would replace m6.m2's by-hand naga run and catch a disagreement between reflection and the emitted WGSL for
  every kernel, not only the tested layouts. naga is already in the tree through wgpu, but `check-arch`'s D02
  allow-list would need a deliberate change, so it needs a decision first. Input to the MX/MR planning (sanity-opus
  and sanity-fable, 2026-11-02; ruling 4).
- `encase` or `crevice` (MIT OR Apache-2.0, licence not yet checked) as an independent Std140V1/StorageV2 oracle for
  the macro's computed offsets, and `bytemuck`'s `Pod` derive to replace the hand-written padding check behind
  `pod_slice_as_bytes` (the m6.g1 must-fix patched it by hand). Both dev-dependencies only, never in a production
  edge (D02), and a licence check first. Follow-ups, not M6 (sanity-fable and sanity-opus, 2026-11-02).
- "Default limits" has two meanings since m7.a0: `HeadlessDevice::new()` keeps wgpu's portable defaults
  (`LimitsPolicy::Default`), while `DeviceOptions::default()` and so `default_device()` take the adapter's own limits
  on native (`LimitsPolicy::Adapter`; `crates/gust-wgpu/src/device.rs:43-95`). A kernel validated through one entry
  point can exceed the other's limits, for example in a browser. Decide whether the two should agree or be named
  differently before MR, and keep it in M8's portable-floor work (sanity-fable and sanity-opus, 2026-11-02).
- The crate-path question before m7.p: the macro writes `::gust::` in about 90 places
  (`crates/gust-macros/src`, tests excluded), MR renames the crates (D48) and m7.p adds more generated paths.
  Decide between `[lib] name = "gust"` on the renamed crate and a macro that resolves the imported crate path (the ROADMAP's MR row
  says the latter), before m7.p writes more of them (sanity-opus, 2026-11-02).

## M7 follow-ups (filed during M7)

- Generated host code binds a buffer with `as_entire_binding()`, so a pooled `GpuVec` passed through
  `as_buffer().as_wgpu_buffer()` shows its whole size class to the kernel (`arrayLength`, the previous owner's
  tail): size those bindings in m7.p (m7.a0 review).
- `#[non_exhaustive]` on the public error enums (gust-wgpu `Error`, `gust::slang::Error`, `gust::reflect::Error`)
  before the first release (MR).
- `HeadlessDevice::enumerate(Backends::VULKAN)` returned no adapter once (the M6 close gates, `gpu-verify` at
  `f452a37`), then passed in `check-full` of the same run and 20 of 20 reruns. Seen again in m7.w's fix round
  (`an_adapter_that_does_not_exist_is_no_adapter` in check-fast, "index 0 is one past the adapters"; green on
  rerun). Find out whether an empty enumeration is transient on this driver under concurrent load, and whether
  `enumerate` or the test should report or retry it (m7.a1). (The `Error::Map` m7.r1 saw is a separate readback
  race, fixed in m7.rb.)
- Remove the mutex that serialises gust-algo's GPU tests (`crates/gust-algo/src/tests/gpu.rs`) once m7.rb's
  readback fix is on `main`, and run them in parallel once (m7.r2).
- `reduce` zero-clears its partials buffers although every lane writes its slot; skipping the clear needs a
  non-zeroing `GpuVec` constructor in gust-wgpu (m7.r1 review item 11; m7.r4 or m7.p).
- M8: naga 30 rejects an `i32` lane id in `subgroupShuffle`, `subgroupBroadcast` and `quadBroadcast`, which WGSL
  allows and Slang emits; filed by the owner as https://github.com/gfx-rs/wgpu/issues/10818. The M8-S1 workaround
  rewrites the id to `u32` in the exported WGSL; drop it when a fixed naga is pinned.
- Type inference for an unannotated literal `let` from its uses (M6 close ruling 3, D96): m6.z2 ships GUST-E0326
  alone because no kernel, fixture or example has such a let and inference needs a typer the lowering does not have
  (estimated 600 to 800 lines; plans m6/z2-measure, 2026-11-02). Inference would also close the float case D96
  documents (rustc's `f64` fallback for an unconstrained float literal against Slang's `float`).
- `crates/gust/tests/diagnostics.rs` `scripted_failing_helper` writes a shell script and runs it at once; on Linux a
  sibling test thread's `fork` can inherit the still-open write handle, and the exec fails with `ETXTBSY` ("Text file
  busy", `HelperUnavailable`). Seen once in CI on `0b0cfaf` (Slangc tier, green on rerun). Retry on `ETXTBSY` in the
  test, or write every scripted helper before any test spawns.

## Deferred language/runtime breadth

Vectors/matrices, uniforms, textures, samplers, additional binding groups, and
specialized layouts each require their own reflected ABI evidence. `while`,
inclusive ranges, general match expressions, `?`, and broader Result APIs need
explicit semantic contracts before implementation.
