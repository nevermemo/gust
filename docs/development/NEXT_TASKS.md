# Next tasks

This is the active queue only. Completed narrative is archived in
[history/NEXT_TASKS-2026-09-09.md](history/NEXT_TASKS-2026-09-09.md); feature
contracts live in [tasks/](tasks/README.md).

Claim one bounded task in [STATUS.md](STATUS.md) before editing. A task is not
active until its owner, scope, and focused check appear there.

The ordered plan is [ROADMAP.md](../ROADMAP.md). Its "Now and next" table lists the
claimable items of the current milestone; this file keeps deferred work and task detail
that does not fit there.

## Claimable DX

### DX20: Internal ownership splits

Split the largest modules only where a named responsibility can move without
changing public APIs: macro validation by concern, macro regressions by feature,
or wgpu runtime internals by ownership. Use a no-behavior-change contract and
one focused check per move. Start with extracting the Result/Result-match
regressions from `crates/gust-macros/src/tests/regression.rs` into a feature-owned
test module; do not mix that mechanical move with behavior changes.

## Later engine proofs

- GPU active count to indirect dispatch (T13): landed in m7.t13; retirement and compaction landed in m7.t14.
- component-pool after m7.t14: the pool's host length never shrinks and retired slots are never reused, so under a
  running emitter each frame's cost (mark, scan, gather, scatter) grows with every particle ever spawned. Options: a
  freelist, a double-buffered dense pool (the pool itself becomes the prefix), or a deferred host truncation from an
  async readback that decides no work in its own frame (m7.t14 review).
- m7.g1 follow-ups: m7.d keeps the derived ids exact (D100) if it folds a 1D launch into y; S7.4 ended C (D118), so
  there is no lockstep twin in M7 and `cpu = "unsupported"` stays mandatory for barrier kernels (D101); bool
  and vector `Workgroup` elements are accepted but never compiled through slangc or run; the SPIR-V test's storage
  class check is weak (`contains("Workgroup")`); one adapter only (m7.g1 review).
- gust-algo passes as graph nodes, so an order such as compaction before `prepare` is declared in the graph instead
  of resting on blocking calls; a non-blocking `dispatch_batch` would remove it silently today (m7.t14 review).
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
  Still open after M7: m7.c (D119) parallelises only generator-stamped map kernels, whose one write is the
  invocation's own index by construction; the same checker (or a per-kernel proof) is what a chunked `par_for!`
  or a parallel user kernel would need (see "m7.c follow-ups").
- CPU limits (open after M5): decide whether the production CPU target emulates the device limits
  (workgroup count per dimension, invocations per workgroup, storage binding size) or
  keeps documenting the difference; today the CPU backend rejects only a launch above
  2^32 invocations and a workgroup size with a zero axis (D102; D39 documents the rest).
- No CPU strategy for barrier kernels (D118, D101): a kernel with `Workgroup` memory or `workgroup_barrier()` has no
  CPU twin and uses `#[kernel(cpu = "unsupported")]` (E0717; `cpu = "lockstep"` is E0709; SEMANTICS section 5). Spike
  S7.4 measured a lockstep twin at about 2.9 ms per barrier round and D118 dropped it; v0 stays every CPU form of
  gust-algo. A thread-free twin is the unplanned idea "A lockstep twin without threads" below. The one parallel CPU
  path today is the element-wise map (`gust::cpu::parallel`, D119).
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
- `invocations` in `gust::cpu` (done in m7.d, D102): it walks the rounded grid in all three axes, caps its range at
  2^32 ids and debug-asserts that `validate` already refused a larger launch (it uses `launch_invocations`). The twin
  at 2^32 invocations has still not run; the cap tests assert the decision of `validate` only.
- If kernels ever become generic, re-check that a defining crate's overflow checks survive instantiation in an
  unchecked crate (the defining-crate rule is measured for non-generic shadow bodies only).
- Optional: `cargo test -- --color never` in the counted calls would remove the ANSI
  stripping (`without_colors`) from the count guard.

## M4 follow-ups (filed at M4 close)

- `gust::slang` checks `slangc -version` against the pin (D72). Then remove the `__gust_keep` wrapper behind the
  naga comparison and a GPU read-back (m4.z measured that 2026.20 no longer needs it; 13.1 fails 13 of 30 shapes).
- `gust::Job` on wasm32 as D69 states: `status` from `on_submitted_work_done`, `wait` returns an error,
  `Device::dispatch` and `dispatch_call` gated. Until then `WgpuJob::wait` is a documented no-op. It must land
  before MR publishes (see "Before MR publishes", M7 follow-ups).
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
- shader-slang/slang#14324 (open, opened by a Slang maintainer on 2026-11-04): WGSL emits `^` on `bool` and `vecN<bool>`
  operands, which WGSL does not define; the PR makes it `!=`. It carries the owner's fix from shader-slang/slang#14198
  unchanged, with its authorship kept, and replaces that PR's emitted-text test with a run-time test on every target;
  #14198 was closed in its favour. Once it is in a Slang release the pin adopts, the bool overloads of `__gust_xor` and
  `__gust_xor_assign` can go (a pin bump with the usual re-bless; D46). Until then the overloads stay. Recorded here
  because the repository did not mention it (sanity-fable and sanity-opus, 2026-11-02).
- Gate the layouts of the emitted WGSL with naga's validator and `proc::Layouter`: done in mx.n (merge `4d4dafe`,
  D98), through `wgpu::naga`, so the D02 allow-list is unchanged; it replaces m6.m2's by-hand run. Left: exported
  WGSL (`compile_reflected_clean`) is gated only through the one committed precompiled file.
- `encase` or `crevice` (MIT OR Apache-2.0, licence not yet checked) as an independent Std140V1/StorageV2 oracle for
  the macro's computed offsets, and `bytemuck`'s `Pod` derive to replace the hand-written padding check behind
  `pod_slice_as_bytes` (the m6.g1 must-fix patched it by hand). Both dev-dependencies only, never in a production
  edge (D02), and a licence check first. Follow-ups, not M6 (sanity-fable and sanity-opus, 2026-11-02).
- "Default limits" has two meanings since m7.a0: `HeadlessDevice::new()` keeps wgpu's portable defaults
  (`LimitsPolicy::Default`), while `DeviceOptions::default()` and so `default_device()` take the adapter's own limits on
  native (`LimitsPolicy::Adapter`; `LimitsPolicy` and `DeviceOptions` in `crates/gust-wgpu/src/device.rs`). A kernel
  validated through one entry point can exceed the other's limits, for example in a browser. Decide whether the two
  should agree or be named differently before MR, and keep it in M8's portable-floor work (sanity-fable and sanity-opus,
  2026-11-02).
- The crate-path question before m7.p: the macro writes `::gust::` in about 90 places
  (`crates/gust-macros/src`, tests excluded), MR renames the crates (D48) and m7.p adds more generated paths.
  Decide between `[lib] name = "gust"` on the renamed crate and a macro that resolves the imported crate path (the ROADMAP's MR row
  says the latter), before m7.p writes more of them (sanity-opus, 2026-11-02). Since m7.a1 the map code also names
  `::gust_wgpu` under GPU profiles (D99), so a user crate needs a direct `gust-wgpu` dependency; `cargo gust doctor`
  could name that for rustc's E0433.

## M7 follow-ups (filed during M7)

- Generated host code binds a buffer with `as_entire_binding()`, so a pooled `GpuVec` passed through
  `as_buffer().as_wgpu_buffer()` shows its whole size class to the kernel (`arrayLength`, the previous owner's
  tail). m7.p did not size them: `generated_host.rs` still emits `as_entire_binding()` (m7.a0 review). Open; the next
  slice that changes the generated host code takes it, with a test that reads `arrayLength` of a pooled `GpuVec`.
- Before MR publishes (M7 sanity, Fable/Opus): the wasm32 `Job` contract (D69; see the M4 follow-ups);
  `#[non_exhaustive]` on `DeviceOptions`, with builder methods (mx.t2 review; the public error enums have it since
  mr.b1); and a way to query which gust-algo variant ran, since v1 switches to v0 above about 2^28 elements without any
  sign, and the float bound changes with it (see the m7.r4 follow-ups).
- mr.t2 follow-ups (Opus review): xtask still starts the helper without the ETXTBSY retry (the two launches in
  `xtask/src/verify.rs`; xtask has no `gust` dependency); a third copy of `HELPER_VARIABLE` and `default_helper` is left
  in `xtask/src/release.rs`; `gust::slang` now checks the `slangc` version, so D72's `__gust_keep` follow-up can be
  revisited; a managed helper built before schema 3 is now picked first outside xtask, so rebuild it with
  `cargo gust build-helper`.
- `HeadlessDevice::enumerate(Backends::VULKAN)` returned no adapter once (the M6 close gates, `gpu-verify` at
  `f452a37`), then passed in `check-full` of the same run and 20 of 20 reruns. Seen again in m7.w's fix round
  (`an_adapter_that_does_not_exist_is_no_adapter` in check-fast, "index 0 is one past the adapters"; green on
  rerun). Find out whether an empty enumeration is transient on this driver under concurrent load, and whether
  `enumerate` or the test should report or retry it (m7.a1). (The `Error::Map` m7.r1 saw is a separate readback
  race, fixed in m7.rb, D97.) Third sighting: m7.rb's fix round (`an_adapter_is_chosen_by_index_and_by_name`, 12 of
  12 reruns passed); m7.rb's probe of about 360 concurrent enumerations found none empty.
- mx.t0b: drop schema 2 from xtask's `check_helper_pin` once every helper in use reports schema 3 (mx.t0a accepts
  both so lanes on older trees keep working).
- The mutex that serialised gust-algo's GPU tests is gone (m7.r3; heavy tests share one lock). The first
  gpu-software run after `8027169` is the first lavapipe run of the parallel set: if it flakes, widen the heavy lock,
  do not bring the full mutex back.
- A generic POD gather (`gather<T>` for a user struct) needs generic GPU functions (M9's traits or a macro-level
  monomorphisation); until then the user writes the block-strided gather kernel (compact.rs docs, m7.r3).
- `compact::indices` makes four submissions and reduces the mask twice; the scatter could write the total
  (`positions[n-1] + mask[n-1]`) itself (m7.r3 review N3; m7.r4).
- `reduce` zero-clears its partials buffers although every lane writes its slot; skipping the clear needs a
  non-zeroing `GpuVec` constructor in gust-wgpu (m7.r1 review item 11; m7.r4 or m7.p). The scan's `out` is
  zero-filled the same way (one clear of n elements per call, 2 GiB at 2^29; m7.r2 review), and so is the map's
  output (`map_on`, `map_gpu`; m7.a1 review N2: name it in m7.a2's timing conditions).
- Tidy follow-ups (m7.t1): the `element()` lookup is shared (m7.t2) and the ETXTBSY flake is fixed (m7.f1); the
  rest of the cleanup scout's list waits for the M7 close.
- gust-algo GPU lib tests: twice, on `1278b61` and on `88ac459`, the test binary died with STATUS_ACCESS_VIOLATION
  (0xc0000005) while `gpu_v1_float_sum_of_the_tie_ladder_is_within_the_bound` and `gpu_v1_runs_the_deepest_schedules`
  ran (over 60 s each, other lanes' builds on the machine, about 2.7 GB of video memory spilled to shared memory the
  second time). Both times the same tests passed later in the same gate run, and on `88ac459` the binary alone passed
  with one thread and twice with default threads. Not reproduced alone, so the faulting test is unknown. Next: a
  slice that runs the deepest GPU tests serially (one device, or a test lock), and a run under a second GPU process.
  Both runs were re-run green at their merges. This is the probe slice (M7 sanity, Fable/Opus): the deepest gust-algo
  GPU tests serially, and beside a second GPU process. It also times the dense64 cell (m7.a2 follow-ups).
- mx.p1b follow-ups: each pixel call creates its own attachment (64 MiB at n = 2^26), which wgpu zero-fills before the
  pass, so a batch of k cap-sized calls holds k times that until the submission ends; the cheap form is one attachment
  per batch, sized to its largest call (no pass writes it), and a cached one later (dropped, never `destroy`ed). The
  attachment is created outside an error scope, so running out of memory there is a wgpu panic, not an `Error`. The
  generated host refuses pixel kernels (`ShapeUnsupported`); MX ships no pixel snippet. Twice in this slice's runs, a
  Gpu-suite test failed once with `NoAdapter` (`limits`, `device_options`) and passed alone and on rerun: adapter
  enumeration can fail transiently while other GPU work runs.
- mx.p1c follow-ups: a mismatch-ledger row (DIFFTEST §7) for `pixel` against `grid` in `f32`, if a corpus program ever
  shows a bit difference (D128 records it as an observation). The grid-only rebuild path of `--shapes`
  (`execute_shapes`) is tested only in parts, because the corpus has no program the classifier refuses. The shape-bench
  smoke once showed K3L at 1024 elements at 0.42 ms as `pixel` against 0.023 ms as `grid`, unexplained; the quiet-GPU
  table decides whether it is noise.
- m7.av follow-ups (the access violation above; STATUS m7.av to m7.av4c): the crash is in the Vulkan validation layer of
  Vulkan SDK 1.4.357.0 (its bundled mimalloc 3.3.2), needs concurrent tests, and disappears with the layer disabled; it
  is a fault of the test environment, not of GUST, wgpu or the driver. Gates keep validation on, because the layer
  catches API misuse; a Gpu-step crash with rc 0xc0000005 is rerun alone, as before. Owner decisions: a Vulkan SDK whose
  validation layer bundles mimalloc 3.4.1 or later (an install), and an issue to KhronosGroup/Vulkan-ValidationLayers
  (public). Not needed now: the shared test-device helper per binary (m7.av2's change stays; fewer devices presumably
  start fewer of the layer's queue threads), and crash-dump capture. If a crash ever shows a stack outside the layer,
  reopen this. dense64 (m7.a2 follow-ups, below): the dispatch is tiny (0.015 ms); the whole call takes about 14 ms warm
  and 37 to 68 ms after about 3 s of GPU idle, whether the CPU sleeps or spins. A guess, not a finding: the idle PCIe
  link or power state slows the copies. Next: one run of the probe with no other GPU work, then fix the README cell text
  and add a warm-up to the README bench.
- m7.p follow-ups: `doctor` and `check` list only a `par_for!` kernel's origin site (`Record.sites` holds them all);
  the naga layout gate skips `par_for!` kernels (take their descriptors through the hidden `descriptor;` form, or
  delete it); the over-cap error is `map::Error::Device`, not an extent variant (needs `Extent` in
  `crates/gust/src/map.rs`); a `macro_rules!` that passes one body in from two sites gives two calls with one call
  line; the GPU tests of a taller workgroup cannot see a missing axis in the guard (only the twin can); xtask does not
  check the cpu-target `par_for` line.
- Gate honesty (M7 sanity, Fable/Opus): xtask does not assert the cpu-target `par_for` line (m7.p, above); the
  lavapipe schedule skip shows only on stderr, so put it in the gate summary.
- m7.d follow-ups: staged graphs are 1D only, and graph.rs validates with the x axis before `prepare_batch`, so a
  16x16 dispatch with a shared buffer fails with `BufferLength` and a zero y or z axis is refused rather than `Ok`
  (still open); `Call::element_count()` is the x axis only, a trap for buffer sizing; for m7.g2 the
  twin walks the grid x-fastest, so a 2D workgroup is not contiguous and a lockstep twin must group by workgroup;
  the twin at 2^32 invocations has not run (validate only).
- m7.r4 follow-ups: a pre-registered statistic for any re-run (the 890M half, D121, kept v1 on a reading chosen
  after the run); re-measure if the pool's zeroing or an unzeroed `GpuVec` constructor
  lands (the small-size win is per-call overhead); a public way to learn which variant (and so which f32 bound) ran
  (filed under "Before MR publishes"). Measure the cost of the map output's zero-fill: per r4's notes, v1's small-size
  win is mostly one zeroing fewer (M7 sanity, Fable/Opus).
- m7.c follow-ups: the `cpu` map copies its host slices once in `bind` before the chunked read; borrowing them
  would cut vector-add's serial part (m7.c review). `par_for!` and user kernels stay sequential on the `cpu` target
  (D119); a chunked `par_for!` would need its own write-set proof. Parallel CPU map (M7 sanity, Fable/Opus): the
  chunk floor F = 2^18 is counted in elements, not work, so the 64-step polynomial gets 3 threads at 1e6. Benchmark
  against rayon (the a2 bench has `cpun`), and consider a work hint.
- m7.a2 follow-ups: the README map table's 64-multiply-add cell at 4.2e6 takes three to four times linear
  scaling from 1e6 in all three runs (smoke and two published) while 1 and 256 multiply-adds scale linearly
  (not investigated). Probe slice (M7 sanity, Fable/Opus): time the dense64 cell at 4.2e6 with timestamp queries; in
  run 1 dense64 took 69.95 ms and dense256 took 69.88 ms, and across runs the cell took 43.6, 50.9 and 70.0 ms.
  Also: a `#[gpu] fn` from an artifact under `native` (the workspace profile pin requires
  `slang-runtime` on every
  example's gust-wgpu, and `native` refuses it; tests/precompiled has no `#[gpu] fn`); `map_async` runs in
  headless Chrome inside `check-web` (web-vector-add awaits `double_add::map_async` and fails unless the page logs
  `GUST_OK`), on one adapter and browser only, and it opens a second device through
  `default_device_async` beside the explicit one; the resident chain's count sees only pooled buffers, so a
  submission or read-back counter would close the gap (m7.a2 review).
- m7.a1 follow-ups: a `#[diagnostic::on_unimplemented]` on `MapArg` so a type mismatch
  reads in GUST's words (re-pin the two check-ui fixtures); `missing_docs` fires on generated constants of both
  `#[gpu] mod` and `#[gpu] fn` (older); `TooLong` is unit-tested only.
- m7.a0 follow-ups (filed at the M7 close): only `create_zeroed_buffer` has an `_async` sibling, and only on wasm32;
  `try_create_typed_buffer` has none, and the wasm32 synchronous forms apply the size pre-check without the
  out-of-memory scope. The out-of-memory path is unit-test only (no allocation was driven to a real device
  out-of-memory), and the branch that reports a non-out-of-memory scope error as `Error::Poll` was never observed on
  hardware (D123).
- Storage barriers (`storageBarrier`: a workgroup barrier over storage memory): M7 has `workgroup_barrier()` only
  (D100, D101); the storage form comes with M8 (M7 plan sections 7 and 9, D23). It orders the accesses of one
  workgroup and gives no ordering between workgroups of a dispatch.
- docs/ARCHITECTURE.md: the paragraphs "Shapes" (mx.k) and "Layouts in the record" (m7.lv) sit under `### gust-algo`
  and belong with `gust-macros` and `cargo gust`; move them (m7.x3 review, older text). STATUS tidy (M7 sanity,
  Fable/Opus): the "Latest DX evidence" section is dated 2026-10-23 and needs the same cleanup.
- Oracles to adopt (M7 sanity, Fable/Opus): naga's uniformity analysis, as a second rejecter for D101; the WebGPU CTS
  barrier and memory-model tests; Decoupled Fallback (Smith, Levien and Owens), whose bounded spin may fit a reworded
  no-forward-progress rule; cubecl-reduce, for speed only; wgpu's `util` helpers for readback staging.
- The a2 bench's log header template says "base ... plus the working tree" and "shared machine: yes ...
  provisional", but the published runs were built on a clean `ea258f9` in a quiet window; fix the template text
  (M7 sanity, Fable/Opus).
- A decoupled look-back scan: not built. It needs an inter-workgroup wait, which AGENTS.md rules out ("No GPU
  inter-workgroup spin waits"). `gust-algo`'s scan stays the multi-dispatch form of 2k + 1 dispatches (v0 and v1).
- The scan `_wrapping` form the M7 plan promises after m7.w: not built. The GPU scan already wraps (plain `+` on
  `u32` lowers to the same Slang as `wrapping_add`, D87); only the twin differs, and the debug twin panics on
  overflow where `reduce::sum_wrapping` wraps (`crates/gust-algo/src/cpu.rs:91`). Add `scan::inclusive_wrapping` and
  `exclusive_wrapping` (twins only) in its own slice or with M9.
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

## MX follow-ups (filed at the MX close)

- Phone and tile-based GPU measurement: the hypothesis that a fragment pass beats compute on a tiled GPU (X-S1 notes)
  is untested, and no document claims it (D95, plan F9). `examples/web-vector-add` is the browser vehicle (mobile
  Chrome on a WebGPU-capable phone); `examples/shape-bench` is native only (`slang-runtime`, no wasm32 build), so it
  would need a native Android/Vulkan build of shape-bench. Needs a device; until one is in the hardware set, no speed
  claim about `pixel` is made anywhere.
- User-owned render targets (Breeze, P2): `pixel` outputs are fragment storage writes (D120), so no call takes a
  `GpuTexture2d` as an output. `TextureUse::TARGET` exists as a flag and maps to a render attachment, but nothing
  consumes it. A user-provided target as a `pixel` output (and depth, stencil, blend, `discard`, interpolated inputs)
  waits for Breeze's `vertex` and `raster_fragment` shapes and the reserved `Body` tier.
- Converting texture formats: the dialect has the nine 32-bit formats (`R32/Rg32/Rgba32` x `Uint/Sint/Float`), and
  `StorageTexture2D` read-write only for the three `r32` ones (D90, D120). Unorm, snorm, 8- and 16-bit formats, sRGB,
  and read-write on wider formats (an adapter-specific wgpu feature) are not built; each needs a twin conversion rule,
  a golden and a ledger row. Texture arrays, 3-D and cube textures and mip levels are not built either.
- Pixel outputs as render targets: not a task. Storage-write `pixel` is adopted (D120, X-S3 outcome B: 0 differing
  words against `grid` and render targets on the RTX 5090 and the Radeon 890M, and cheaper per call). Render-target
  outputs return only with the user-owned targets above.
- 2-D `pixel` calls: a call is one-dimensional (`y` and `z` other than 1 are `ShapeUnsupported`, also in a graph node
  and on the twin), and the element is `y * 8192 + x`. `(x, y)` ids need an output width rule; one option is a buffer
  that knows its width (a texture target, so it would go with the user-owned targets), another a `Shared` output of
  `x * y` elements. The extent form of `pixel` is open.
- The preservation mode: a `pixel` kernel must write every output on every path (E0726), so a kernel that leaves an
  element unwritten, or reads its own output, is `grid`-only. Preserving the old value (copy the buffer in first, or
  rely on the in-place storage write) would widen `compatible_shapes`; relaxing the rule needs a DECISIONS row. The
  classifier's reach is the cost: of the ten examples that set a workgroup size, 4 of 21 kernels are `pixel`-eligible
  and 17 are not (DIFFTEST, "Shapes"). The reasons a preservation mode would move are E0726 (2) and some of E0728 (8:
  output reads and component writes); E0725's three scatters stay `grid`-only. A `return` lowering needs a second
  hook, which the profile's kill rule forbids today (D88).
- `GpuVec` textures: there is no texture in `GpuVec`, the buffer pool or the staged graph. A graph refuses a kernel
  with a texture or sampler parameter (`Error::UnsupportedParameter`), and a texture is created and written by hand
  (`create_texture_2d`, `write_texture`). A resident texture type with pooled storage, and graph nodes that carry
  textures, are open.
- SPIR-V execution of the goldens, for MR (D95; it replaces the M6 follow-up of the same gap, sanity-opus 2026-11-02,
  ruling 4). `gust-wgpu` executes WGSL only (D82). A passthrough needs `Features::PASSTHROUGH_SHADERS` (wgpu 30), a
  second `create_shader_module` path and a per-adapter record, all in `gust-wgpu/src/lib.rs`. MR is the milestone that
  ships SPIR-V to Vulkan users, and MX adds nothing that executes it: the SPIR-V of the golden sets (`textures` and
  `pixel` among the eleven) is validated with `spirv-val` and nothing more. The same gap leaves the matrix and uniform
  layout decorations unexecuted (SEMANTICS section 4).
- Classifier test gaps (mx.x review of part 1, item 14): five rejection reasons have no `check-ui` fixture. Four are
  pinned by `validate/classify/tests.rs` only (workgroup memory, `LocalId`/`GroupId`, a write to one component of an
  output, a `StorageMut` that is never written), and `OutputOther` (E0728 for an output rebound or passed to a helper)
  has no test at all. A small validator slice: one unit test for `OutputOther`, and a fixture each for the four where
  rustc adds no second error (add them to `tests/ui/shape-*`).
- Pixel-vs-grid `f32` in the ledger: D128 makes a bit difference between the shapes an observation. SEMANTICS section 7
  has no row for it (the record is in section 1 and DIFFTEST, "Shapes"; the K3 form that showed 4 ulp was not kept).
  File a row with a kernel of that shape (an unfused multiply-add chain after a division) and pin the observation per
  adapter kind, so a driver change shows up as a NOTE, as the other unenforced rows do.
- The bench table: `examples/shape-bench` prints X-S1's per-shape table, and the full run waits for a quiet GPU (to be
  recorded as `plans/shapes/probe/bench/table.md` with its machine conditions). The Radeon 890M row, and the 890M as a
  second adapter for the identity test, are run when the owner runs them; neither is a gate, and neither may stand in
  for the RTX 5090.
- The cross-shape batch on lavapipe: plan section 6 wanted `check-difftest --shapes grid,pixel --seeds 1-2` on the
  `gpu-software` pull-request list (else the nightly list), if its warm time fits the five-minute budget. Only
  `check-full` asks for `--shapes` (DIFFTEST, "Shapes"), so lavapipe has never run the cross-shape batch. Measure its
  warm time and add it to one of the two lists, or record why not.
- The lavapipe pixel record: STATUS has no `gpu-software` record for `tests/pixel.rs` and `tests/textures.rs`. The merge
  `ec6b79b` says "gpu-software green" in its message only, and the close-out (plan section 11, item 2 (ii)) wants the
  record in STATUS: the run id, the adapter and Mesa version, the sizes the identity tests skipped on the software
  adapter (above 1 000 003), and the label `Gpu (software: ...; shapes grid, pixel)`. The coordinator writes it.
- The bit-for-bit `f32` gates and D128: the `f32`, `Vec2` and `Vec4` identity tests of `tests/pixel.rs` and `check-web`
  compare bits, and their bodies (moves and one correctly rounded `f32` addition) leave the driver no freedom, so they
  agree with D128 today. Keep it so: a new `f32` body with a division or a multiply-add belongs in the difftest's
  interval judgement, not in a bit gate.
- Pixel gaps in the evidence (SEMANTICS section 1 states each): integer vector outputs (`IVec2`, `UVec4`, ...) are
  admitted by the classifier and have never run as `pixel` outputs; a linear `sample` in a fragment has not run (the
  nearest sampler has, against the twin); `bool`, struct and 3-wide outputs are `grid`-only "for now" (the list is
  mx.c's, kept after D120 removed its reason); the browser run of the `pixel` form covers one hardware adapter and not
  the software one (`check-web --chrome-arg` for SwiftShader); and an adapter that cannot write storage buffers from a
  fragment gives a wrong result in a browser, not `ShapeUnsupported`, because wgpu's WebGPU backend reports every
  downlevel flag. A canary dispatch that turns that wrong result into the named refusal is an idea, not a plan.
- Tidy: the `pixel` lowering emits a dead `_S1` line in the fragment WGSL (harmless; mx.w review nit 6). The per-call
  attachment, the missing pixel snippet in the generated host code and the out-of-memory panic of the attachment are
  already filed under "mx.p1b follow-ups". Two code comments contradict the code or the docs: `cpu/texture.rs:139`
  says "A nearest read has no tolerance: compare it exactly", while SEMANTICS and the textures test allow a 1/128 band
  next to a texel edge; and the usage text at `xtask/src/main.rs:136` lists `check-difftest` before `check-precompiled`,
  while `check-full` runs it last (`verify.rs:141`).

## Deferred language/runtime breadth

Additional binding groups and
specialized layouts each require their own reflected ABI evidence. `while`,
inclusive ranges, general match expressions, `?`, and broader Result APIs need
explicit semantic contracts before implementation.

### Idea (not planned): rounds of `par_for!` until a GPU-written condition

Owner, 2026-11-03: noted as a potential idea only. A `par_while!`-like construct (name open, for example
`par_for!(..).until(..)` or `par_rounds!`) would repeat a `par_for!` round until a flag or active count the
kernel writes says done: rounds in sequence, invocations parallel within a round, the twin running rounds in
order. It would need a mandatory round budget, ping-pong buffers instead of in-place stencils, a condition check
every k rounds (or GPU-driven through T13's indirect dispatch), and D39 wording for float convergence that takes
a different round count on CPU and GPU. It is the `while_gpu` candidate of EXECUTION_GRAPH.md. Measure first: a
host loop over resident `GpuVec` bindings with a one-word readback per round may already be enough.

### Idea (not planned): a lockstep twin without threads

S7.4 (D118) measured a threaded lockstep twin at about 2.9 ms per barrier round of 256 threads, far too slow. E0414
already allows a barrier only at the body's top level or at the top level of a top-level literal-range `for`, so
the macro could split the body at its barriers into phases and run each phase for every invocation of a workgroup
in a plain loop (the "work-item loop" of CPU OpenCL implementations), keeping the locals that live across a barrier
in per-invocation arrays: no condvar rounds, one thread per workgroup, workgroups in parallel. A spike would have to
settle locals across a split, loops around a barrier, panics and early returns, and the D22 contract for it.
