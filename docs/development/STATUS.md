# Current status

## Active: roadmap M7 (2026-11-03)

The roadmap agreed with the owner is in [ROADMAP.md](../ROADMAP.md). M0 closed on
2026-10-23 after a two-reviewer sanity check (D28): every M0 item is done; M0.15
(the CI `unit` job is a required check on `main`) followed on 2026-11-01. M1 (diagnostics) closed on
2026-10-23 (D33): M1.0 to M1.6 are done (per-item evidence in the roadmap), after a
sanity check by Fable and Opus (both "proceed with fixes") and the full validation
below. M2 (typed host API) closed on 2026-10-24 (D37) after a sanity check by Fable and
Codex (both "proceed with fixes") and `check-full` at `634a50e`. M3 (dual test builds) closed
on 2026-11-01 (D43) after a sanity check by Fable and Codex (both "proceed with fixes"), the
fixes in slice m3.h and `check-full` on a clean tree at `5971c12` (RTX 5090, 06:28-06:31Z).
M4 (build-time targets and distribution): plan adopted 2026-11-01 (owner-delegated) after a
Codex review (rework), two Opus re-reviews and three revisions; D44 (build profiles through
`GUST_PROFILE`), D45 (committed, re-verified artifacts), D46 (`cargo-gust` and the Slang pin)
and D47 (weak compare-exchange). M4 closed on 2026-11-02 (D73; close rows D68 to D72) after a sanity
check by Fable and Codex (both "proceed with fixes"), the fixes in slice m4.i and `check-full` at `27b342b`.
M5 (CPU target and evidence breadth): plan adopted early on 2026-11-01 (owner-delegated) after a
Codex review (adopt with changes) and one revision. D51 makes `cpu` a production profile with a
sequential contract, and D52 sets the rules for software-adapter evidence. M5 closed on
2026-11-02 (D76) after a sanity check by Fable and Codex (both "proceed with fixes"), the fixes in slice m5.f1 and
`check-full` at `7dd9c3d`. The M6 layout spike V1 (decision
C, a scratch probe on the RTX 5090) found:
- vec3 must be rejected in buffers;
- the matrix layout defaults differ between Slang's CLI and its API, so M6 pins column-major;
- `m * v` lowers to `mul`;
- uniforms take std140 padding.

M6 (language breadth I): plan adopted early on 2026-11-01 (owner-delegated) after an Opus review
(in the Codex slot), one revision and an Opus re-check. `m * v` ships in M6 through a
`__gust_mul` overload table, enabled only in modules that name a `Mat` type; the reflection
reader accepts schema 1 and 2, and the helper keeps schema 1 for scalar-only programs, so
committed artifacts stay byte-identical. The keyword and construct spikes and the pure-Rust math
types ran beside M4; M6 closed on 2026-11-02 (D86) after two sanity checks; its decisions are
D77 to D85, and slice m6.z2 (literal-local typing) follows m7.w.

- M4 m4.i (`b508e42`, fix round `08914bc`, merge `27b342b`): the M4 sanity-check fixes. A `CXX` whose file stem
  is `cl` gets MSVC arguments; the precompiled consumer prints "no GPU adapter" only for `Error::NoAdapter`, so
  CI's NoAdapter step discriminates, and CI runs the consumer's unit test. `check-full` now runs the whole
  `check-precompiled` and `check-web` after its old steps, checks check-web's tools first and fails without them
  unless `--skip-web`, which is printed, recorded in `skipped_checks` and shown by `cargo xtask status`; the record
  holds check-precompiled's summary. Prose fixed in VALIDATION, CI, ARCHITECTURE, HANDOFF, README, AGENTS,
  cargo-gust `--help` and three skills. Evidence: Gpu (RTX 5090) and Gpu (WebGPU, Chrome 156.0.8078.74, nvidia
  blackwell) through `check-full` at `27b342b`. Opus merge after one fix round.
- M4 m4.z (`fa3f9a1`; fix round 1 and the prose pass on top of the merge `d7344b6`): the cleanup slice, with no change
  to compiler output (bless: 0 changed goldens). Helper freshness is judged by content: a build stamp beside the helper
  holds the SHA-256 of the packaged probe source and the schema and Slang tag, and no file time plays a part, so a
  helper built against an older Slang is rebuilt (the m4.r S6 gap) and two cargo-gust versions with the same probe
  source leave each other's helper alone. `cargo gust doctor` names both versions and the fix when the `slangc` first
  on `PATH` is not the pin. A byte test compares all five committed copies of the precompiled artifact set with the
  kernels fixture.
  The `__gust_keep` wrapper stays (measured on 30 comparison-in-list shapes: 2026.20 validates all in naga without it,
  13.1 fails 13, and `gust::slang` runs any `slangc` on `PATH`). `crates/gust-wgpu/clippy.toml` bans
  `std::thread::sleep` and `Instant::now` for wasm32 (the CI wasm32 clippy command; native code is allowed by a
  crate-level `cfg_attr` allow in `lib.rs` and `tests/async_readback.rs`); the crates `gust` and `web-vector-add` have
  no such ban yet (a follow-up). The slice entries of M1 to M4 moved to `history/STATUS-M1-M4.md`; the unused
  `Ty::is_float` went. Known gaps: `gust::slang` does not check `slangc -version` against the pin (then
  `__gust_keep` can go behind a GPU read-back); the managed helper has no stamp until `cargo gust slang build-helper`
  runs once more. Evidence: Unit, Slangc, Spirv and Reflect on 2026.20, check-ui, check-dual `--check-only`,
  check-precompiled `--check-only`, check-cpu-target; CPU only. Opus merge after two review rounds.
- M5 m5.c2 (`82c59d2`, fix round `f2a9410`, merge `68c6f40`): difftest in the gates. `cargo xtask check-difftest
  [--seeds] [--cpu-only]` runs the harness, checks the report by field (`programs_executed == 16 x seeds`, the
  per-seed table against the seed list, no mismatches, the harness exit status) and, on the Gpu tier, the
  `short_circuit` GPU test. The Unit suite runs 4 seeds on the CPU plus the harness's own fmt, clippy and tests; a
  separate CI job `difftest` runs the same (the `unit` and `slang` jobs use `--skip-difftest`, which prints each
  skipped step); gpu-software runs 2 seeds per PR and 72 nightly on lavapipe and uploads the report; `check-full`
  runs 8 seeds on the hardware adapter and the harness checks. Seed 4 found a slangc limitation (D74, retired by m6.dv and D77): slangc
  folds integer `/` and `%` and refuses a divisor that folds to zero (E30002), even in a dead branch; DIAGNOSTICS
  lists what folds, the generator discards such programs as `static` (`DIFFTEST_FORMAT` 4), and a Slangc-tier test
  pins the reproducer. Evidence: 8 seeds, 128 of 128 programs, 0 mismatches on the RTX 5090; 19 gate steps on main
  at `68c6f40` (11:30-11:58Z), including `verify --mode gpu` and `check-full` with no skips (RTX 5090, driver
  610.88, the browser run on Chrome 156). Opus merge after fixes; two fix rounds.
- M5 m5.d3 (merge `d879f33`, feature `e249ae5`): the coretests golden `tests/fixtures/coretests.slang` (blessed) with
  its regression entry and a slangc test of each section; `coretests` joins `check-feature gpu-semantics` and
  `coretests_twin` the Unit suite. `gpu-semantics` passed on the RTX 5090. Opus keep.
- M5 m5.e2 (`ea386f3`, merge `f9fdd24`): gust-wgpu refuses the `cpu` profile. It evaluates
  `gust::profile::require_gpu_target()` in a const item, so a `GUST_PROFILE=cpu` build that links it fails with E0080
  (E-e). `cargo xtask check-cpu-target` runs E-a to E-i on `tests/cpu-target` and takes each binary from cargo's JSON
  output (the stale-binary fix); the Cpu suite is this command. CI runs it as the job `cpu-target` beside `unit`: 61 s
  on the runner, against 2 m 57 s for `unit`. Evidence: Unit, Cpu and Gpu smoke (RTX 5090). Opus keep after one fix
  round.
- M7 m7.w (`389d263`, fix round `70e09fd`, merge `4797013`): wrapping arithmetic and GUST-E0325 (D87).
  `wrapping_add/sub/mul/neg` on `u32`/`i32` lower to the plain operator (`wrapping_neg` as `(0 - a)`); the twin
  calls Rust's `wrapping_*`. Fixtures numeric (scramble, step) and mismatch (uwrap, iwrap: 15 kernels); a 7×7 edge
  grid per op and type inside the ledger test (GPU against host `wrapping_*` and an `i128` oracle; failures
  counted with the ledger's), green in the dev run and in `check-ledger`; 36 DOMAIN ledger rows; check-ui
  fixtures `wrapping-ok` and `wrapping-div-rejected` (pinned fixtures 22 to 24). Pin 216/25 unchanged (no new test
  fn). Mutations: `wrapping_mul` as `+` fails 195 ledger expectations and the WGSL text test; accepting
  `wrapping_div` fails check-ui; the old E0304 text fails `wrapping-div-rejected`. Opus merge after one fix round.
  Main: 22 gate steps green on `4797013` (from 22:11Z, RTX 5090), including `gpu-verify` and `check-difftest`.
  One more empty Vulkan adapter list in the fix round (`an_adapter_that_does_not_exist_is_no_adapter`, check-fast;
  green on rerun): the second sighting of the M6 enumerate flake (NEXT_TASKS).
- M7 m7.r1 (`978a65c`, fix round `874b7ed`, merge `b5d2c68`): reduce v0 in the new crate `gust-algo`.
  `reduce::{sum, min, max}` and `sum_on_device` for `u32`, `i32` and `f32` over a `GpuVec` (block-strided
  per-invocation partials, one pooled buffer per pass, at most 4 passes in one batch), twins on `CpuDevice`, an `f64`
  and `i128` reference, the per-level float bound asserted; f32 `min`/`max` padding lanes copy `input[0]`; integer
  sums wrap past the domain (tested). 18 kernels in `tests/fixtures/reduce.*`. S7.7 on the RTX 5090 and the Radeon
  890M: every case inside the per-level bound, including a 2^24 + 1 uniform sum on the twin and the GPU. gust-algo
  is not in the Unit suite (its CI job stays free of wgpu); its 12 GPU tests run in the Gpu suite and in
  `verify --mode gpu` with their own pin. Its GPU tests share one device behind a mutex until m7.rb (a readback
  race under concurrent polls, found in review) lands. Opus merge after one fix round. Main: 22 gate steps green on
  `b5d2c68` (from 23:42Z, RTX 5090).
- M6 m6.z2 (`71314f1`, fix round `dcd46a6`, merge `a496968`): GUST-E0326 for an unsuffixed literal `let` (D96).
  The rule covers literal-only integer initializers, `Some`/`Ok`/`Err` of one, `unwrap_or` of two and `if let`
  scrutinees; an unsuffixed float `let` is documented in SEMANTICS (rustc's `f64` fallback against Slang's
  `float`), not rejected. GPU case `typed_locals` (u32 `/=` by y >= 2^31, an unsigned compare of two locals,
  `>>` on u32 and i32) against the twin and host Rust; check-ui fixtures `literal-let-rejected` and
  `literal-let-ok` (pinned 24 to 26). Pin 216/25 to 217/25. Mutations: the check off fails check-ui; annotated
  u32 locals emitted `int` fail the GPU case at y = 0x80000000. Opus merge after one fix round. Main: 22 gate
  steps green on `a496968`.
- M7 m7.rb (`a0b791e`, fix round `9a656e1`, merge `893d959`): blocking readbacks wait for their own map completion
  (D97, amends D69). A concurrency test (8 threads on one device) gave 8 `Error::Map` in 2 903 iterations before the
  fix and 0 in 3 153 after (RTX 5090); a mutation restoring the single check fails it. Pin 222/26 in the lane; 223/26
  on `main` with m6.z2's test. The empty-enumeration flake (D86) showed once more in the fix round (12 of 12 reruns
  passed); a probe of about 360 concurrent `enumerate(VULKAN)` calls did not reproduce it. Opus merge after one fix
  round.
- MX mx.t0a (`83a6148`, fix round `be46e44`, merge `754e439`): reflection helper schema 3 (D91), C++ side. Written
  only when a 2D texture, a sampler or a non-compute stage is present, so every committed artifact set stays
  byte-identical (`cargo gust check --recompile`; old and new helpers equal on 18 compared documents). Texture
  bindings report shape, access, result type and format (a sampled texture always `read` and `unknown`; a storage
  texture its declared or inferred format; no byte layout for texture results); samplers; the stage (argv,
  default `compute`); a fragment's targets in SV_Target order. Comparison samplers, 1D/3D/cube/array/multisampled
  and combined textures are refused by name. X-S2 outcome A: the API's facts equal slangc `-reflection-json`, so the
  helper is the D17 witness and the CLI facts are an optional agreement check. 11 probe inputs with expected output
  run in the Reflect suite. xtask's helper pin accepts schema 2 or 3 (mx.t0b drops 2). Opus merge after one fix
  round.
  Main: 22 gate steps green on `893d959` (01:59 to 02:29Z, RTX 5090) with the schema-3 helper; pin 223/26.
- M7 m7.t13 (`40726ea`, fix round `4275104`, merge `b88bfb0`): T13, the GPU active count to indirect dispatch, in
  `examples/component-pool`, re-typed by hand from the `WIP T13` stash's design (never applied). `classify` counts
  live particles with an atomic add, `prepare` clamps the GPU-written count to the allocation and the budget into
  the indirect arguments, the consumer guards `i < active && i < len`; every binding is independent and non-empty,
  the graph's edges are declared, and no readback decides work. Every stash test is kept. 14 exported kernels
  (`pool__classify`). The plan's `active_count` feature is not added: the stash's entry named a gust-wgpu test file
  that does not exist; the T13 GPU tests are ignored tests that `check-examples` and `check-full` run with
  `--include-ignored` (16 of 16 on the RTX 5090, Vulkan). Mutations fail them: the clamp, either guard, the count
  reset; a dropped edge is refused (`GraphMissingDependency`). Uncertainty: only the RTX 5090 has run them so far;
  the nightly gpu-software list runs `check-examples` on lavapipe; the 890M, DX12 and Metal are untried. Active
  particles stay a prefix of the pool, checked on the host only, until m7.t14 adds compaction. Opus merge after one
  fix round.
- M7 m7.r2 (`900068e`, fix round `88793a0`, merge `09203ae`): scan v0 in `gust-algo`: `scan::inclusive` and
  `scan::exclusive` over `u32` (wrapping, as `reduce::sum_wrapping`, which this slice adds), `2k + 1` dispatches in
  one batch over `k` levels: 3 for n <= 65 536, 5 for n <= 2^24, 7 above. Deviation: the plan named 5 and 7 tiers
  only; a one-level input needs just 3. Golden `scan.slang` (6 entries) with its fixture; gust-algo GPU tests 12 to
  20 (`ALGO_GPU_TESTS`), including the 7-dispatch path at 2^25 on ones. Mutations fail a test: a skipped add-back,
  `exclusive` returning the inclusive sums, and plain `+` in `sum_wrapping` (by D87 the Slang is the same, so the
  twin and a source-text test catch it). Opus merge with no must-fix.
  Main: 22 gate steps green on `09203ae` (02:39 to 03:02Z, RTX 5090); pins 223/26 and gust-algo 20.
- MX mx.n (`cb528b2`, fix round `2d24624`, merge `4d4dafe`): the naga layout gate (M6 close ruling 4), a gust-wgpu
  test that needs no adapter. It parses and validates WGSL with the naga wgpu runs (`wgpu::naga` 30.0.1, the
  capabilities and flags of `create_shader_module`; a control test shows an f16 module that the all-capabilities
  default would pass) and compares the `proc::Layouter` size, alignment and offsets of every storage and uniform
  binding with the descriptor's `GpuPod::LAYOUT`, both ways. 120 modules: the 55 compute entries of the 13 goldens
  through the slangc CLI, the 65 fixture kernels through the reflection helper (the route `HeadlessDevice`
  compiles), and the committed precompiled WGSL. naga refuses none and no layout differs; nothing is allow-listed.
  It replaces m6.m2's by-hand naga run. It runs in the Slangc suite (CI job `slang`) and under the GPU pin (D98,
  amends D95): 223/26 to 232/27. Mutations: a shifted expected offset and the all-capabilities default each fail
  it. Exported WGSL (`compile_reflected_clean`) is gated only through the committed precompiled file. Opus merge
  after one fix round and a re-review.
  Main: 22 gate steps green on `4d4dafe` (03:45 to 04:14Z, RTX 5090); pin 232/27. CI's `msrv` job then failed on
  an `if let` match guard in the new test (not stable on Rust 1.90); fixed in `de289c5`, and the coordinator's main
  gates now run the same 1.90.0 check.
- M7 m7.r3 (`494bf32`, fix round `365722c`, merge `8027169`): compaction v0 in `gust-algo`. `compact::count` (a mask
  pass, then the exact on-device `reduce` sum; no readback, pinned by a source test), `compact::indices` (exclusive
  scan of the mask, then a scatter; `flags.len()` long, the first `count` entries are the positions, the tail is 0)
  and `gather_u32`, `gather_i32`, `gather_f32` (output `indices.len()` long, 0 from `count` on and wherever an index
  is past `values.len()`). Any non-zero flag counts as 1. The kernels are block-strided like reduce, because one
  invocation per element passes the 65 535-workgroup ceiling at 2^24 + 1. Golden `compact.slang` (5 entries; the
  naga gate now covers 130 modules). A user struct is gathered by the user's own block-strided kernel, shown in
  tests/struct_gather.rs; a generic POD gather is deferred to M9. gust-algo GPU tests 20 to 32. The mutex that
  serialised them is gone (D97): the light tests run in parallel and the 7 heavy ones (n >= 2^24 or at the binding
  limit) share one lock; 3 parallel runs passed 32 of 32, peaking about 2.5 GB above a 2.5 GB baseline (serial
  about 1.4 GB above, unlocked parallel 4 to 6 GB above). Mutations: a scatter to the element's own index and an
  off-by-one gather bound fail twin and GPU tests; a dropped gather guard fails the twin test only (the GPU test
  passes because out-of-range reads return 0 on this adapter). Opus merge after one fix round.
  Main: 23 gate steps (the 22 plus the Rust 1.90.0 check) green on `8027169` (05:16 to 05:42Z, RTX 5090); pins
  232/27 and gust-algo 32. CI's gpu-software ran the parallel gust-algo set on lavapipe (32 passed).
- M7 m7.a1 (`7ea07f4`, fix rounds `ffb5b76` and `d559e72`, merge `5609536`): the map API (D60, D99). `#[gpu] pub fn
  saxpy(a, x, y) -> f32` stays a plain function and gains `pub mod saxpy` with the generated kernel (grid-stride,
  workgroup 256, 16 776 960 invocations at most, a length-1 argument broadcast), `map`, `map_on`, `map_on_async`,
  `map_gpu`, `map_gpu_on` and `map_async`; under `GUST_PROFILE=cpu` only the CPU forms, with no `gust_wgpu`. Lengths
  are checked before any device work (`LengthMismatch`, `Empty`, `TooLong`). `map_async` exists on every target and
  the blocking calls are native only (D69, D97). GUST-E0714 (map signature), E0715 (name collision), E0716 (a nested
  `#[gpu]` function); 3-wide values reuse E0204. The dispatch id is a reserved name, so a parameter named `id`
  works. Map arguments are keyed on the host type, so a `bool` slice for a `u32` parameter fails to compile (two
  check-ui fixtures, 33 in all). Golden `map.slang` (5 entries; the naga gate covers 140 modules). check-cpu-target
  requires the map case's line; check-feature map runs the new tests. A wasm32 crate compiles `map_async` (dev
  profile; no browser run). Mutations: a dropped guard is caught only by the CPU twin (the GPU's robust bounds hide
  it); a skipped length check, a missing E0715 and a broadcast reading index i each fail tests. `TooLong` is
  unit-tested only (a real call needs more than 2^32 elements). Pin 232/27 to 244/28. Opus merge after two fix
  rounds and a re-review.
  Main: 23 gate steps green on `5609536` (06:06 to 06:42Z, RTX 5090); pins 244/28 and gust-algo 32.
- M7 m7.t14 (`c08e560`, fix round `dc7fc43`, merge `f96aff1`): retirement and compaction in component-pool. A
  particle retires when `steps >= lifetime` (a one-element device buffer), judged per slot, so in any slot order.
  Each frame: `mark_live`, then gust-algo `compact::indices`, then one graph that gathers the live particles into a
  dense prefix, runs `prepare` and the indirect `integrate` on it, and scatters them back. The pool stays sparse;
  retired and unspawned slots are never read or written. No readback decides work; the active-count readback only
  reports. The GPU result is checked every frame against an independent host filter: flags, count, indices, the
  dense prefix (and zeros after it), and every untouched slot byte for byte. Sizes 0 to 65 537, seven retirement
  patterns, two budgets, both release orders. This closes the T13 gap for the compacting path; the T13 path stays
  valid only on emitter-ordered pools where nothing retires (the two must not share a pool). The order from
  compaction to `prepare` rests on blocking waits (`dispatch_batch`, `execute_graph`), not on a graph edge: dropping
  the gather to `prepare` edge is refused as a resource hazard. gust-algo becomes a path dependency of the example
  (check-arch accepts it); exported kernels 14 to 17. Mutations: identity indices, a gather off by one, and a
  `mark_live` without the lifetime test each fail GPU tests. One adapter (RTX 5090, Vulkan). Opus merge after one
  fix round.
  Main: 23 gate steps green on `f96aff1` (07:09 to 07:40Z, RTX 5090); pins 244/28 and gust-algo 32.
- M6 close (D86, 2026-11-02): Fable and Opus in the Codex slot (owner decision; Codex quota out until
  2026-11-05T06:32, then a confirming pass) both said "proceed with fixes" (plans: sanity-fable, sanity-opus,
  close-rulings). No delivery gap; the gaps were in the record, plus the older literal-local typing bug (m6.z2).
  `gpu-verify` re-run green on `b5b87e3` at 20:35Z (216 passed over 25 lines).
- M6 m6.z1 (`cd707b7`, merge `c8799b3`): the record fixes. Descriptor names (README, this file's m6.k entry), no M6
  GPU case executes SPIR-V (VALIDATION, ARCHITECTURE, DIFFTEST), the two meanings of the default limits
  (ARCHITECTURE, D84), NEXT_TASKS follow-ups (shader-slang/slang#14198, SPIR-V execution, a naga layout gate,
  encase/bytemuck oracles, the crate path before m7.p), the nightly `gpu-software` difftest on seeds 1 to 72
  (seeds 1 to 8 took 45.5 s of `difftest-run` on the runner), and decision rows D78 to D86. docs-check and lint.
- M6 m6.x (`0d5fc4c`, fix round `1b878d9`, merge `8f069fb`): M6 wiring. `check-feature vectors` and `uniforms`;
  `vectors`, `uniforms` and `map` in `FEATURE_AREAS` with a test that reads the real accepted set; the golden
  modules (products, vectors, uniforms) are compiled and spirv-val'd by `check-artifacts`, the Spirv suite and
  `check-full`; every SPIR-V check prints an informational `OpCopyLogical` count (0 in all 16 modules); a test
  pins the matrix-layout flag in the two shared slangc command builders; `vector_ops::blend` in tests/cpu-target
  is a required `matches` line; README, ARCHITECTURE, VALIDATION and three skills updated, stale uniform text
  and the typed-call `on_unimplemented` label corrected. Opus merge after one fix round.
- M6 m6.d (`b5e8744`, fix round `e02be45`, merge `f452a37`): vectors and matrices in the difftest space
  (`DIFFTEST_FORMAT` 6). Each seed writes 10 float programs (format 5's, byte for byte) and 6 vector programs;
  dot, `m * v` and `m * m` are left-to-right chains under the existing bound; a `Mat2` needs a direct f32 input
  read. Seeds 1 to 8: 128 of 128 on the RTX 5090 and on `--cpu-only`, 0 mismatches; they write 34 of the 72
  aggregate forms (the 38 others are listed in DIFFTEST). Seeds 1 to 64 on the GPU: 1024 of 1024, 0 mismatches,
  71 forms written. A CPU test proves the generator writes every form within 72 seeds. With the float intervals
  narrowed to the reference point the GPU batch fails (561 of 8589 f32 elements differ), so the comparison has
  load on this device. Opus merge after one fix round.
- M6 close gates: 27 steps on main at `f452a37` (19:32 to 20:12Z, RTX 5090, driver 610.88,
  Slang v2026.20). 26 are green, including `check-full` (rc 0), the 8-seed difftest and `check-feature
  vectors|uniforms`. `gpu-verify` failed once in `device_options::enumerate_lists_the_vulkan_adapters` (the
  Vulkan adapter list came back empty); the test passed inside `check-full` of the same run and in 20 of 20
  reruns alone (NEXT_TASKS, M7 follow-ups). Pin 216/25 unchanged.
- M6 m6.dv (`1d5662c`, fix round `f7498c4`, merge `94baab5`): D74's cure (D77). Every `/` and `%` whose divisor is
  not a literal lowers to `__gust_div`/`__gust_rem` (crates/gust-macros/src/slang/div_table.rs: 19 division and
  3 scalar remainder overloads with `_assign` forms), emitted once per module that uses them; literal divisors
  stay plain, and E0317 now sees a literal zero through parentheses, macro groups, `/=` and `%=`. Evidence: the
  D74 reproducer compiles on WGSL and SPIR-V; a 9-shape Slangc table compiles every emitted overload; the GPU
  grid `crates/gust-wgpu/tests/division.rs` checks i32 and u32 `/`, `%`, `/=` and `%=` against the host and the
  KM-01 to KM-03 ledger rows; goldens coretests, mismatch and semantics re-blessed; no committed artifact
  changed. `DIFFTEST_FORMAT` 5 drops `first_iteration_zero_divisor`: seeds 1 to 8 discard 507 (was 512), the
  counts equal format 3's label for label, and `s4_a26` runs again. Opus merge after one fix round; 22 gate
  steps on main at `94baab5` (17:16 to 17:43Z), including the 8-seed difftest on the RTX 5090, pin 216/25.
- M7 m7.a0 (`e9f67f6`, fix round `3ccadca`, merge `bcb4da4`): runtime prerequisites in gust-wgpu.
  `DeviceOptions` (adapter choice and `LimitsPolicy`; `DeviceOptions::default()` asks for the adapter's limits,
  `HeadlessDevice::new()` keeps the WebGPU defaults), `enumerate`, `try_create_typed_buffer` and
  `create_zeroed_buffer` with a `max_buffer_size` pre-check and, on native, an out-of-memory error scope that
  resolves without a device wait (measured; wgpu 30's scopes are per thread); `Error::{OutOfMemory, MixedDevices,
  Extent, DefaultDeviceOpened}`; `BufferPool` size classes with a 1 GiB LRU, zeroed hand-outs cleared and
  logical-length bindings; `GpuVec<T>`, `DeviceHandle`, `default_device`; the `gust::map::MapHost` trait with
  `readback_async` on both targets and the blocking `readback` native only (D69). S7.6: the limits rows hold on
  the RTX 5090 (adapter policy 4 GiB-1, binding 2 GiB-4) and the Radeon 890M (2 GiB, 2 GiB-4), a 1 GiB zeroed
  buffer is created on both, and 64 GiB, `usize::MAX` and `u32::MAX` elements return `DeviceLimit` before any wgpu
  call; the out-of-memory row is not shown on hardware (unit test only), and on lavapipe CI's gpu-software run
  of `8befcf6` passed (214/24), so the adapter policy stays the native default. The raw buffer behind a
  pooled `GpuVec` is the whole size class (documented); sized bindings in generated host code are m7.p's.
  Opus merge after one fix round.
- M6 m6.g2 (`76c998a`, fix round `b71365c`, merge `475e2b1`): GPU evidence for uniforms.
  `crates/gust-wgpu/tests/uniforms.rs` dispatches, from `#[gpu]` modules on the RTX 5090, the `uniforms` golden,
  a scalar, a `Vec4`, a non-symmetric `Mat4` and a nested-struct uniform, one `GpuUniform` bound to two kernels,
  and a `StagedGraph` that calls `write_uniform` between dispatches; each result equals an independent host
  reference and the twin exactly (the inputs are exact in f32, and the reference asserts it). The reflected
  `constantBuffer` sizes equal `size_of` (D17). Opus merge after one fix round. m6.g2 and m7.a0 shared one main
  gate run: 21 steps (the 19 plus wasm32 clippy and `check-web`) at `bcb4da4`, 16:36 to 17:00Z, pin 214/24.
- M6 m6.u2 (`88cc120`, fix round `33ece37`, merge `1b592a1`): uniforms in the macro. A `Uniform<T>` (or the
  `ConstantBuffer<T>` alias) parameter needs a module struct `T` (E0206); a struct reachable from a uniform and a
  storage element is E0207; `Mat2` in a uniform-reachable struct is E0205; a uniform is read one field at a time,
  so a whole-value use, `u.0` and a `mut` uniform parameter are E0321. Uniform-reachable structs get `#[repr(C,
  align(16))]` and Std140V1 const asserts; any gap or tail in their Std140 layout is E0209 (E0208 first where a
  field's own alignment pads it), so the user writes explicit padding fields and no uninitialised byte is uploaded.
  Every generated bound names its `Kind` (`Storage` or `Uniform`), so a buffer where a uniform is expected and the
  reverse are compile errors (E0271, two ui cases), and the per-kernel arity assert counts both kinds; the twin
  builds `::gust::Uniform` from `read_uniform`, so uniform kernels have typed calls and CPU twins (D40's wait
  ends). Evidence: Unit (the macro tests; `uniforms_twin.rs` runs the golden's twin against a hand reference and
  pins `Frame` at 128 bytes, align 16) and Slangc (the `uniforms` golden on WGSL and SPIR-V); GPU dispatch is
  m6.g2. Opus merge after one fix round; 19 gate steps on main at `1b592a1`.
- M5 m5.f1 (`c988328`, fix round `e76f61a`, merge `7dd9c3d`): the M5 sanity-check fixes. The release-profile
  ledger call names its 6 tests (`RELEASE_LEDGER_TESTS`) and must run them all, with 0 failed and 0 ignored,
  in Full and in the new `cargo xtask check-ledger`, which the gpu-software nightly list now calls; it keeps
  `--nocapture`, so the step log shows the detected overflow mode. `cargo xtask test --suite Gpu` enforces
  `GPU_SUITE_PIN` as `verify --mode gpu` does. ci.yml's check-dual, check-cpu-target and check-web steps run
  `--locked`; the Mesa install retries (`Acquire::Retries=5`, timeouts, the install once more after 60 s) with the
  snapshot and the identity assertions unchanged (D52). Evidence: Gpu (RTX 5090) through the 21 close gates and
  `check-full` at `7dd9c3d`. Opus merge after one fix round.
- M6 m6.u1 (`ff6e188`, fix round `82ed0b5`, pin `896571d`, merge `8d88b6d`): uniforms, host side. `gust` gains
  `BindingKind`, the markers `kind::{Storage, Uniform}` and `Resource::Kind` (every implementer names its kind),
  `Call::new` counting storage plus uniform parameters, `CpuUniform<T>` and `cpu::read_uniform`. `gust-wgpu` gains
  `GpuUniform<T>` (`create_uniform`), `BufferBindingType::Uniform` entries, the two uniform launch limits, and
  `StagedGraph::write_uniform`, an encoded copy from a staging buffer, so two updates between two dispatches are
  observed in order (D15, D35); the immediate `HeadlessDevice::write_uniform` is a `Queue::write_buffer`, so every
  dispatch of one later submission sees the last value written. A uniform type must be Std140V1, padding-free and
  match Rust's size and alignment on both the GPU path and the twin, so no uninitialised byte is uploaded; users
  write explicit padding fields. A uniform passed where a buffer is expected becomes a compile error once m6.u2
  names `Kind` in the generated bounds; until then `BindingKind` refuses it at submit. Evidence: typed and graph
  tests on the RTX 5090 dispatch an `f32` uniform and a struct of a `Vec4` and four `f32` through `create_uniform`;
  a lone `Vec4` or `Mat4` uniform reached the GPU only in the raw-wgpu probe (m6.g2 dispatches them);
  `GPU_SUITE_PIN` is 180 passed over 21 lines. Opus merge after one
  fix round; 19 gate steps on main at `8d88b6d`.
- M6 m6.n (`f7a8e4f`, fix round `f88de7a`, pin `f8644ce`, merge `86f728c`): matrix negation. slangc emitted
  `matN(0) - m` for `-m`, which naga refuses on WGSL. In a module the m6.m2 gate marks, unary `-` on a non-literal
  lowers to `__gust_neg(x)` from one data table (`slang/neg_table.rs`, 10 rows): scalars and vectors as `-a`,
  matrices as `a * -1.0`, bit-identical to negation under IEEE (D39 notes where Vulkan or WGSL may differ for a
  zero or a subnormal). `gust::math` gains `Neg` for `IVec2` and `IVec4` (lane by lane, the crate's overflow mode),
  so the products golden holds only code rustc accepts. A new GPU test negates matrices and a vector over signed
  zeros, subnormals and the smallest normal: on the RTX 5090 every lane equals the twin's sign flip, debug and
  release. `GPU_SUITE_PIN` is 160 passed over 21 lines. Ungated goldens are byte-identical. Opus merge after one
  fix round; 19 gate steps on main at `86f728c`.
- M6 m6.g1 (`118037e`, fix round `a064378`, pin `6130183`, merge `5033b22`): GPU evidence for vectors and
  matrices. `gust-wgpu` accepts vector and matrix storage buffers: `buffer_resource_layout` gives StorageV1 for
  scalars and StorageV2 for vectors and matrices, `validate_kernel` refuses a parameter whose layout does not match
  its element type, and `assert_pod_matches_storage_layout` adds a `const fn` padding-free check, so a padded
  hand-written `GpuPod` struct is refused instead of uploading uninitialised bytes (CPU unit tests for each rule).
  `crates/gust-wgpu/tests/vectors.rs` holds seven GPU tests that compare the vector and matrix forms with the CPU
  twin on the RTX 5090, debug and release; `GPU_SUITE_PIN` is 159 passed over 21 result lines. Known limitations
  (SEMANTICS §4): a constant float2x2 or float3x3 `IDENTITY` crashes the runner when its SPIR-V
  is handed to wgpu (exit 139, layer not isolated); no M6 GPU case executes SPIR-V;
  `-m` on a matrix did not build on WGSL until m6.n. Opus merge after one fix round; 19 gate
  steps on main at `5033b22`, including `verify --mode gpu`.
- M6 m6.m2 (`49f00f9`, fix round `fec79e8`, merge `71ab804`): operator syntax for products. In a module whose
  unrawed tokens name `Mat2`, `Mat3` or `Mat4`, `*` lowers to `__gust_mul(a, b)` and `*=` on a pure place to
  `(p = __gust_mul(p, r))` (E0323 otherwise); one data table (`slang/mul_table.rs`, 39 rows) drives both the
  prepended Slang overloads and the hidden `gust_mul_impls!` that generates every `Mul`/`MulAssign` impl of
  `gust::math`. A module with any `dot` gets `__gust_dot`, a component sum for the integer vectors, so integer
  `dot` no longer needs `Capability DotProduct` (its SPIR-V is Shader-only and passes naga spv-in with empty
  capabilities; float `dot` SPIR-V is byte-identical). The golden `tests/fixtures/products.{rs,slang}` writes every
  table pair as `a * b` and slangc compiles it on WGSL and SPIR-V; the seven ungated goldens are byte-identical.
  Naga was run by hand (no naga dependency). Evidence: Unit, Slangc (2026.20), check-ui, check-dual. Opus merge
  after one fix round. Open: the StorageV1/StorageV2 wording mix in the struct asserts (pinned in xtask ui tests).
- M6 m6.m1 (`31e6e99`, merge `6dc36ef`): vector and matrix types in `#[gpu]` code, column-major. `from_cols` lowers to
  `transpose(floatNxN(..))`, `from_rows` to `floatNxN(..)`, `mul_vecN` and `mul_matN` to `mul(m, x)`, `transpose` to
  `transpose(m)`; the struct layout asserts use StorageV2. The golden `tests/fixtures/vectors.{rs,slang}` covers every
  accepted form and a slangc test compiles it. Known limitations: the integer-vector `dot` lowered to
  `OpSDot`/`OpUDot` (capability `DotProduct`), which naga 30 refuses on the SPIR-V route (fixed in m6.m2); a constant `Mat2::IDENTITY` or `Mat3::IDENTITY` hits the known SPIR-V crash for constant
  float2x2 and float3x3, and no M6 GPU case executes SPIR-V (m6.g1 runs on WGSL). Evidence: Unit, Slangc (2026.20),
  check-ui and check-dual; CI green on `6dc36ef`. Opus keep after one fix round.
- CI timing note: `gpu-software` took 1 m 25 s warm (on `8080320`) against 1 m 42 s cold.
- M5 S5.4 (spike, local commit only): decision A. A lane-local prototype in which the macro drops
  the twin's cfg gate under `GUST_PROFILE=cpu` ran a non-test binary on `CpuDevice` with the host
  result, built with `--cfg gust_dual` without a duplicate impl, and passed `cargo test` under
  `cpu`. In warm target dirs, flips in both directions (`cpu`, unset, `cpu`, ...) were each
  correct by behaviour. Each tracking path suffices alone: the macro's `option_env!` line
  (kernel crate and dependents rebuild) or `gust`'s `rerun-if-env-changed` (`gust` rebuilds
  too); with both removed, expansions went stale. m5.e1 keeps both and guards them.
- Fix fx.bxor (`7463405`, found by the m5.d1 corpus): WGSL has no `^` on bool, so bool `^`
  and `^=` now lower through `__gust_xor` and `__gust_xor_assign` overloads (bool `!=`,
  integer `^`; the place is evaluated once), emitted only in modules that use them. A GPU test
  runs every input pair of the bool operators and `^=` on buffer elements, struct fields and
  in helpers against the twin. Known gap, fixed next in fx.wgsl: slangc drops the parentheses
  in `f(a < b, c > d)`, which WGSL then reads as a template list. Opus keep after one fix
  round; gates on main at `9cfdfe4`.
- Fix fx.wgsl (`60d36c2`): naga reads `a < b, c > d` in a call, a `^` operand pair or an array
  literal as a template list, since a `<` after an identifier opens a candidate that any later
  `>` at its depth closes. The macro wraps every `<` comparison in a list element other than the
  last (also through the right operand of `&&` and `||`) in `__gust_keep(bool)`, an identity
  helper emitted only in modules that need it. Unsuffixed integer literals from 2^31 to
  `u32::MAX` are emitted as `uint(N)`. Emitter tests pin the wrapped and unwrapped shapes; two
  Gpu tests in `wgsl_spelling.rs` check both bugs against the twin; `gpu-software` pins 142
  passed over 19 result lines. Follow-up: the validator should reject literals outside 32 bits
  (E0812). Opus keep after one fix round; gates on main at `709a86d`.
- M5 m5.b (`236780b`): the mismatch ledger runs on two adapter kinds (D52, D39). The kind comes
  from the adapter's device type (`Cpu` is software). Unenforced rows carry one observation pin
  per kind; the software pins are m5.b0's lavapipe values for 18 rows (KM-07 NaN bits and row
  100, KM-08 denormals), and the set-valued KM-09 rows keep their set rule. A NOTE compares an
  observation with the running kind's pin and names driver, backend and `driver_info`; nothing
  unenforced fails, and an enforced row has no per-kind value. Each run writes
  `ledger-<kind>.txt` under `CARGO_TARGET_TMPDIR` and asserts its 132 lines. On lavapipe the
  NOTE count fell from 7 to 0. Opus keep after two fix rounds; gates on main at `bee2c5d`.
- M5 m5.d2 (`ce17199`): the 84 scalar coretests cases run on the GPU and the twin, each compared
  with its Rust literal under its R1-7 set (exact integers, 2.5 ULP for float `/`). Guards: every
  CORETESTS.md id is executed exactly once, the table's columns and rows are checked, and each
  slot is asserted once; a host unit test covers the ULP and neighbour branches, because no
  transcribed row reaches them (both `/` rows divide exactly). 84 of 84 pass on the RTX 5090 and
  on lavapipe (llvmpipe, Mesa 25.2.8). `gpu-software` pins 146 passed over 20 result lines. Opus
  keep after one fix round; gates on main at `bee2c5d`.
- M6 m6.a1 (`b60fca2`): the layout model gains vectors and matrices (`StorageV2`, with
  `StorageV1` as its scalar sub-case, and `Std140V1`; column-major pinned by `MATRIX_LAYOUT`).
  The buffer-eligible math types implement `GpuPod` with const asserts on size and alignment;
  3-wide types and `Mat3` stay out, and `GpuPod` covers only padding-free layouts. The
  reflection reader keeps schema 1 unchanged and adds schema 2 (vector, matrix,
  `constantBuffer`, mandatory `matrixLayout`); every committed artifact set still verifies.
  Until m6.g1, `create_typed_buffer::<Vec4>` compiled and panicked at run time in the StorageV1
  check (fixed there). Opus keep after one fix round; gates on main at `38ebf29`.
- M6 m6.k (`ce17eb2`): collision-only identifier renaming. A user identifier that collides with a
  Slang or WGSL keyword, a prelude name or a GUST-reserved spelling is emitted as `__gust_r_<name>`
  at every name site (locals, parameters, fields, structs, invocation and ConstantBuffer sites);
  `annotate_diagnostics` strips the prefix from slangc messages at a word boundary, and descriptors
  carry the emitted names (`ParameterDescriptor.name` and `StructFieldLayout.name` hold `__gust_r_<name>`, which the
  D17 cross-check compares; `rust_type` and the Rust source keep the Rust spelling; checked in the Reflect tier on
  `class`, `new`, `This` and `operator`; corrected in m6.z1, D81).
  A sweep of about 156 words in four positions found that a local
  named `true` or `false` silently rebinds the literal (`true ? 7u : 9u` stored 9); both are now
  renamed, and `var` (loud, E30060) joined the table. The `__gust` prefix is reserved (new UI
  case). Bless: 0 changed goldens. Evidence: Unit, Slangc and Reflect on main. Deferred to m6.m1:
  raw `r#` kernel and module names (`gpu_m_r#k`, loud). Opus keep after three fix rounds.
- M5 m5.c1c (`a41ba93`): bounded shrinking for difftest. `difftest-run --shrink` reduces a failing
  program with an AST reducer; one clock started after the original run bounds the search and
  each candidate's cargo steps, and no candidate starts with under 1 s left. Measured on the RTX
  5090: one literal fault shrank in 119.0 s over 17 candidates (mean 6.58 s), 0 mismatches in both
  batches; the no-fault check exits 1. An interrupted `--shrink` can leave cargo running on Unix
  (DIFFTEST.md). Evidence: Gpu (RTX 5090). Opus merge after three rounds.
- M5 m5.e1 (`f60c1b4`): the production `cpu` target (D51). `GUST_PROFILE=cpu` is a fourth profile:
  never inferred, not strict, refuses `slang-runtime`, does no artifact lookup (`IS_CPU`; `STRICT`
  stays `native` or `web`). Under `cpu` the macro emits one ungated `impl Run<K, N>` for
  `CpuDevice` and no `cfg(any(test, gust_dual))` copy, also with `cfg(test)` or `gust_dual`; it
  emits `option_env!("GUST_PROFILE")` beside gust's `rerun-if-env-changed`, and a guard test fails
  if both go. Only the exact value `cpu` selects it. Fixture `tests/cpu-target` and check-ui
  `typed-call-cpu-target`; steps E-a to E-d and E-f to E-h pass by hand (E-h: 10 alternating
  builds). Not yet enforced: a GPU runtime refusing `cpu` (E-e) needs gust-wgpu to call
  `require_gpu_target`, which is m5.e2. S5.2 outcome A: `CpuDevice` dispatch within 1.20 to 1.64
  times (vector-add) and 1.31 to 1.92 times (polynomial) of a hand loop, release, five runs.
  Evidence: Unit, CPU only; main gates plus Reflect, check-ui and check-cpu green on the merge.
  Opus merge after one fix round.
- M6 m6.r (`fe42f3f`): the reflection helper reports vectors (`scalarType`, `elementCount`), matrices
  (rows, columns, layout and the column stride, size divided by columns) and constant buffers (the
  element's Uniform layout plus the `descriptorTableSlot` binding), with matrices pinned
  column-major in the helper (`defaultMatrixLayoutMode`) and at both slangc call sites
  (`-matrix-layout-column-major`). Schema 2, with a top-level `matrixLayout`, appears only when a
  resource holds one of those kinds, so all 11 files of the 5 committed artifact sets stay
  byte-identical; cargo-gust accepts helper schemas 1 and 2. Storage buffers are unchanged: a padded
  `float3` is still refused by `is_storage_v1`/`v2`, the macro's layouts and gust-wgpu. The CLI and
  native paths give the same bytes for a matrix kernel, and real helper output passes the
  cross-check as StorageV2 and Std140V1. Evidence: Unit, Reflect, Slangc and Spirv on main; the
  machine's shared helper is now schema 2. D17's "fails on anything outside StorageV1" is amended at
  the M6 close. Opus merge after one fix round.
- M5 m5.a2 (`e098f2f`): `gpu-software` takes Slang from the cargo-gust manager's pin (version and
  Linux sha256 read from slang.rs, one line each or fail) through `cargo gust slang install` and
  `build-helper`, runs `cargo gust doctor`, and never restores a helper from the cache. The test
  counts moved from YAML into `cargo xtask verify --mode gpu` (`GPU_SUITE_PIN` in verify.rs: 146
  passed over 20 lines), which also refuses a named helper whose Slang tag differs from slangc's. A
  nightly list (03:00 UTC, or dispatch with `extended`) adds check-examples and the release ledger;
  both ledgers are uploaded. Evidence: Gpu (software), lavapipe on GitHub: PR list 2m31s and nightly
  list 4m21s, both without caches (the warm time comes from the next run). Opus merge after one fix
  round.
- M5 m5.c1b (`a3a7798`): the difftest harness `difftest-run`. It writes one run crate per
  batch, builds it with `cargo test --locked` under `GUST_PROFILE=dev`, requires exactly one
  complete result file per program, and writes `report.json` with exact counts. The GPU f32 is
  compared by value inside the interpreter's interval; the CPU twin and every integer are
  compared bit for bit. DIFFTEST_FORMAT 3 fixes dead and one-sided float conditions (both-sided
  `if`s: 2 of 55 before, 58 of 94 after); the integer programs are unchanged. A short-circuit
  test checks that `&&` and `||` lower to a guarded if/else (Gpu tier, run by m5.c2). Batches:
  seeds 1-2 on the RTX 5090, 32 of 32 with 0 mismatches; seeds 1-8 CPU-only, 128 of 128. S5.3
  outcome A: a batch of 32 takes about 26 s warm and 101 s cold. Opus keep after one fix round;
  gates on main at `63c8e25`.
- M5 m5.d1 (`d2378ce`): 84 scalar cases transcribed from Rust 1.98.0's coretests
  (`library/coretests/tests/num`) into `docs/development/CORETESTS.md`, with acceptance sets
  (exact integers in the domain, WGSL's 2.5 ULP for float `/`) and exclusions; five `#[gpu]`
  kernels in `tests/fixtures/coretests.rs`; a CPU twin test that parses the table. They add 50
  new (operator, type, class) triples to the 20 already tested. The fixture found a compiler gap:
  `bool ^` lowers to WGSL `^`, which naga rejects (CT-30), so the GPU half (m5.d2) waits for the
  fix. Opus keep after one fix round; gates on main at `3ec3094`.
- M5 m5.c1a (`3bcebaf`): the differential generator core in `tests/difftest/` (std-only). It
  generates `#[gpu]` programs over `u32`, `i32` and `bool`, interprets them as the oracle inside
  D38's domain, discards anything outside it (including a static fold of constant
  subexpressions in dead code), and emits them with full parentheses, checked by a re-parse
  test. `seeds.txt` pins seeds 1 to 8 with hashes of the code and the inputs. Generated crates
  take unique package names, because cargo gives two crates of one name in one target dir a
  single fingerprint (reproduced without GUST). Opus keep after one fix round; gates on main at
  `c2df942`.
- M5 m5.c1a2 (`f60f516`): floats v1 in the generator. f32 inputs are 0 or in 2^-20..2^20;
  `+ - * /`, negation, comparisons, casts to and from `i32` and `u32`, loop accumulators and
  short chains. Each f32 output carries an interval that encloses the twin and every GPU result
  SEMANTICS allows; a chain of n roundings widens by (n-1) ULPs of S' = S + 2(n-1) ULP(S), so
  binade edges stay sound (an Opus probe found the edge case; the new soundness test, with
  adversarial neighbours and three groupings, now finds it on its own). Programs that cannot be
  bounded are discarded by class; comparisons need disjoint intervals. `DIFFTEST_FORMAT` 2; the
  integer space keeps the c1a programs (hashes pinned). Opus keep after one fix round; gates on
  main at `34baf39`.
- M5 m5.a (`00ad885`): the `gpu-software` workflow runs the whole Gpu suite on Mesa lavapipe on
  a hosted runner (D52), installing only from one Ubuntu snapshot URI (stricter than the plan:
  a `Snapshot:` field on the live archive still read live indices). It asserts the adapter
  record (device type `Cpu`, llvmpipe, the full Mesa and LLVM `driver_info`) and the pinned
  passed count and result-line count, and `cargo xtask status` prints `tier=Gpu (software)`.
  Software evidence is never hardware evidence. On lavapipe the first readback poll blocks until
  the kernel ends, so one async test skips only its pending-path check there (VALIDATION.md).
  Runs: 1m52s cold, 1m27s warm. Any slice that adds Gpu tests bumps the two pins. Opus keep after
  two CI fix rounds; gates on main at `00ad885`.
- M6 m6.a2 (`56b5ecd`): `gust::math` vector and matrix types in pure Rust (§2.1 of the M6 plan),
  with swizzles, and dual kernels that use them in `tests/dual`. Integer vector operations are
  `#[inline]`, so in incremental dev and test builds they panic on overflow only when both `gust`
  and the compiling crate have checks on (measured over nine profile runs); SEMANTICS §3 records
  it as a limitation (D39). The struct-literal dual kernels emit Slang that nothing compiles yet;
  m6.m1 makes them compile. Opus keep after one fix round; gates on main at `76c83d5`.

### Done slices of M1 to M4 (summary); M4 closes after m4.z

The slice entries of M1 to M4 moved verbatim to [history/STATUS-M1-M4.md](history/STATUS-M1-M4.md) on 2026-11-02
(slice m4.z); ROADMAP.md and DECISIONS.md hold the decisions and the per-item evidence. This summary keeps what
landed, the evidence tier, the commits, the numbers later work cites and every gap that is still open.

**M1, diagnostics.** Closed 2026-10-23 (D33), after a Fable and Opus sanity check. Tier: Unit, Slangc, Reflect; full
validation at `a6cbb8f`.
- Landed: `#line` directives on the compile-time copy only, never in exported `.slang` (M1.3; 10 of 10 deliberate
  failures land on the right Rust line on WGSL and SPIR-V, bytes identical with and without directives for three
  fixtures; `8fcf944`, `7007827`); a rejected `#[gpu]` module returned beside `compile_error!` (M1.4; `e01b82c`,
  `f507ab4`); 111 stable codes `[GUST-E0401]`, one DIAGNOSTICS.md section each, every code asserted or listed as
  unreachable (M1.5, D31; `df07122`, `6e195bc`, `f20d313`); clippy `disallowed-methods` confined to `diagnostics.rs`
  (M1.6; `1f961fc`); carry-overs (`fb257c9`); the `typed-pipeline` and `staged-graph` release sizes became `1 << 21`
  because 65 536 workgroups exceed wgpu's default of 65 535 (`5c34e22`).
- Open: columns stay Slang columns and only the first error is attributed; the editor benefit of M1.4 is not measured
  (one rustc error before and after, the same message); seven codes are unreachable through `#[gpu]`.

**M2, typed host API.** Closed 2026-10-24 (D37), after a Fable and Codex sanity check. Tier: Gpu (RTX 5090);
`check-full` at `634a50e` (05:44Z). Decisions D34 to D36.
- Landed (m2.a to m2.g, close-out `009f86a`, sanity fixes `2d1880b`): the typed call surface in `gust` (`TypedDevice`,
  `Resource`, `Arg`, `Independent`, `LengthRule`, `Call`, `Run<K, N>`; `3060072`); launch-limit checks returning
  `Error::DeviceLimit` on every dispatch path (`151ef56`; measured before: workgroup size 512 panicked at pipeline
  creation and 65 536 workgroups at dispatch); `gust-wgpu` implements it and rejects one buffer bound twice when
  either binding is read-write, `Error::AliasedBinding` (`119ece3`); `check-ui` judged by the keys `code`, `file`,
  `line`, `message` and `label` (`a8b074f`); generated `call`, `submit` and `dispatch` per kernel, a wrong element
  type is E0271 (`a83a96e`); const layout assertions, so a `#[gpu]` struct with an outside `GpuPod` field of
  alignment 8 is E0080 (`adeb7be`); `StagedGraph` derives dependencies from typed calls (RAW, WAW, WAR; `229bfbf`),
  and staged-graph and component-pool derive exactly `[[], [0], [0, 1], [2]]` and `[[], [0], [0, 1], [1]]`; the
  examples dispatch through typed calls (`ef52d86`, `2a4f3f6`); a hand-written descriptor's slangc error lands on the
  marked Rust line (`c6571d5`); `DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION` is pinned to
  `wgpu::Limits::default()`, and one runtime StorageV1 assertion guards typed, pool and indirect-argument buffers.
- Open: the dispatch of `render_wgpu_source` and buffers above `max_buffer_size` are not limit-checked (D36);
  indirect counts are known only on the GPU, and wgpu zeroes over-limit counts silently (under its default instance
  flags only), so `with_limits` does not apply to them; a kernel with a `Uniform<T>` or `ConstantBuffer<T>` parameter
  has no typed call (`UnsupportedParameter` at run time); the `Into` and `Run` `on_unimplemented` notes still point at
  `#[gust::gpu]`; a non-GUST argument gives two E0277 and has no fixture; without `// @rust kernel:` markers
  `slang::compile` names no kernel, while `reflect` and the helper add `originating Rust kernel`; the "GPU end-to-end"
  figure of polynomial and signal-pipeline waits twice since m2.f1 and no before and after number was measured.

**M3, dual test builds.** Closed 2026-11-01 (D43), after a Fable and Codex sanity check (both "proceed with fixes"),
the fixes of m3.h (`a169bec`, merge `5971c12`) and `check-full` on a clean tree at `5971c12` (RTX 5090, 06:28 to
06:31Z; the run before it was `96e4b9d`, 05:36 to 05:43Z: 13 SPIR-V modules validated, `check-dual` in full mode,
every example test with `--include-ignored`, both release-only tests by name). Decisions D38 to D43; D39: edge-case
differences are documented limitations, not to be matched. Tier: Unit, plus Gpu (RTX 5090) for the twin-versus-GPU
checks.
- Landed: the known-mismatch ledger (`96ec90b`: 13 kernels and a blessed golden; spike S2: 33 in-domain controls agree;
  outcomes that WGSL defines are enforced, while NaN, infinities, subnormals, `NaN != NaN` and out-of-bounds access are
  recorded one-adapter observations; `verify --mode full` adds a `--release` pass); `docs/SEMANTICS.md` (`cc88b1a`);
  `gust::cpu` (`a7dcb95`, `#![forbid(unsafe_code)]`, no new dependencies); the macro-emitted CPU twin for every
  typed-call kernel (`732c774`: vector-add equals the host and the GPU exactly; polynomial equals `run_cpu` bit for
  bit and stays within the derived `48 * 2^-24` of the GPU, observed 2.38e-7, the worked bound being 46u); `check-cpu`
  and `code: none` in `check-ui` (`35eb2a5`); the five atomics on the twin in ascending invocation order (`0f2ff27`);
  `check-dual --check-only` and `tests/dual` (`5c5a9ad`: first dual build 76 s, no-change rerun 11 s, dual target
  directory about 8 GB); twin assertions for every typed-call kernel of eight examples (`41b4d0e`, `00127a4`), with
  bounds derived per kernel and written in each test (signal-pipeline `transform` 900u; particle-step `step` 3u to
  98u; typed-pipeline `calibrate` up to 172u, `classify` 63u, `finalize` 14u; staged-graph `transform` 193u,
  `summarize` 3886u; component-pool `integrate` 19u to 73u per field; u = 2^-24; every observed maximum inside its
  bound and none used as a tolerance); `cfg(gust_dual)` declared through `[workspace.lints.rust] unexpected_cfgs` and
  the full `check-dual` mode in `check-full` (`98acabc`; CI 82 s, inside the 5-minute budget); gates that cannot pass
  by skipping (`95c5514`: `--include-ignored`, a zero-test guard, NaN-propagating max-error folds, the 30 `gust::cpu`
  unit tests in the Unit suite); m3.h: the KM-11 store rows, a twin follows the crate that defines the kernel (D43),
  required test names in `check-dual` and `check-cpu`. Spike S1 (`dc2ad87`, a local branch, not merged) proved the
  twin at counts 1, 63, 64, 65 and 257.
- Cost numbers: `check-cpu` 7 to 8 s warm and 67 s cold locally; the CI Unit step went from 12 s to 41 s.
- Open: the test-count guard counts per call, not per test (NEXT_TASKS); the zero-storage twin is only
  expansion-tested; MSRV is checked by CI only; the root `fmt` and CI clippy did not reach `tests/ui` when m3.f1
  landed (m3.f2 added `tests/dual`).

**M4, build-time targets and distribution.** The code chain is merged through m4.r; the cleanup slice m4.z (entry above)
is merged, and M4 closed on 2026-11-02 (D73) after the sanity check and its fixes (m4.i, entry above).
Plan adopted 2026-11-01; D44 to D47 (profiles,
committed and re-verified artifacts, `cargo-gust` and the Slang pin, weak compare-exchange).
Tier: Gpu (RTX 5090) except where noted.
- Landed: spike S1 (local commit): with no Slang reachable, a precompiled vector-add verified through D17 and equal to
  the host reference on the RTX 5090; spike S2: the compile-time refusal (E0080) fires in all three native-setting
  cases, and a release build without a profile becomes native, with `GUST_PROFILE=dev` as the escape; m4.0 weak
  compare-exchange (`8d49412`); m4.h `check-artifacts` empties `generated-wgpu/` and every verify mode records the
  adapter (`b26bccd`); m4.a `gust::artifact` (`2dcb9e3`); m4.b1 the `slang-runtime` feature, off by default
  (`cbbf35a`); m4.b2 profiles through `GUST_PROFILE` and `require_artifact` (`41b6ff9`); m4.g1 the completion model in
  native form (`671b3dd`); m4.c content key and embedded artifacts (`dba0e97`); m4.d `cargo-gust` with `init`,
  `build`, `check` and `doctor` (`e7dd73d`; rebuild identical, about 55 s cold); m4.e1 gust-wgpu prefers the verified
  artifact (`894ab0e`); m4.e2a `tests/precompiled` and E1 to E4 (`29cfc58`); m4.f the pinned-Slang manager
  (`9b42958`); m4.e2b E5 to E8 (`3c95f8a`; full `check-precompiled` rc 0 on the merged tree, 640 s); m4.g2 wasm32 and
  the first browser WebGPU run (`4e096f5`; Chrome 156.0.8078.74, nvidia blackwell); m4.r Slang pinned at v2026.20
  with every committed set re-blessed (`12ab066`). `**/gust-artifacts/** -text` is required, because
  `core.autocrlf` would convert committed WGSL.
- Numbers later work cites: with 2026.20, particle-step SPIR-V read-backs equal those of 13.1 byte for byte on the
  RTX 5090; the CI `slang` job recompiles `tests/precompiled` in 36 s locally, and its recompile step has run on Linux
  (the job was green on `af0c2b6`, the step took 5 s); the `gpu-software` pins were 140 passed over 18 result lines at
  m4.e1 and live in `GPU_SUITE_PIN` of `xtask/src/verify.rs` since m5.a2.
- Slang 2026.20 fixes the WGSL template-list bug (shader-slang/slang 13566). The `__gust_keep` wrapper of Fix fx.wgsl
  is therefore redundant under 2026.20 (slice m4.z, 2026-11-02: 30 comparison-in-list shapes all validate in naga
  without it) but needed under 13.1 (13 of the 30 fail), and `gust::slang` runs whatever `slangc` is on `PATH`, so it
  stays until the library enforces the pin.
- Open (filed in NEXT_TASKS at the M4 close; D71 records the manifest field names): `gust::Job` needs the
  wasm-aware contract of D69 (`WgpuJob::wait` is a documented no-op on wasm32 and `status` reports
  `Complete` early); the wasm32 clippy ban on `std::thread::sleep` and `Instant::now` covers `gust-wgpu` only (m4.z),
  not `gust` or `web-vector-add`; a plain
  `cargo install cargo-gust` needs `GUST_PROFILE=dev`; CI's `precompiled` job runs `--check-only` plus a NoAdapter run
  of the release consumer, and E1, E3, E8 and E7b stay local; a GPU-side no-op compare-exchange is not detected by a
  GPU test and there is no signed ledger case (m4.0); S2 did not measure the wasm32-to-web inference (m4.g2 builds
  under an explicit `GUST_PROFILE=web`).
- Closed by m4.z: helper freshness judged by a source-hash stamp and the Slang tag, and a byte check of all five copies
  of the committed artifact set (the m4.r known gaps), a `cargo gust doctor` warning for a `PATH` `slangc` of another
  version, and the wasm32 clippy guard of `gust-wgpu`.

T13 remains deferred by owner; its work in progress is the git stash
`WIP T13 active count deferred`, not the working tree. Do not apply, continue,
validate, commit, or drop it without explicit direction.

```text
owner: Claude Code coordinator (monitored autonomy)
claim: roadmap M7 (m7.g1 and m7.a2 running) after the M6 close (D86); see the roadmap's Now
and next
next_focused_check: cargo xtask check-fast
full_check_needed_before_commit: no
```

## Current baseline

- Branch: `main`
- Last complete compiler feature: T12, statement-only `match Result<T, T>`
- Last full validation: `cargo xtask check-full` passed at `7dd9c3d` (M5 close after m5.f1, 2026-11-02, NVIDIA
  GeForce RTX 5090 / Vulkan, driver 610.88, Slang v2026.20, 13:49 to 14:01Z; clean tree, nothing untracked; it includes
  the whole `check-precompiled` and `check-web`, the 8-seed difftest and the release ledger guarded by name, and no
  check was skipped), through the
  shared cargo slot wrapper: workspace tests, the release-only tests required by name, every
  example test with `--include-ignored`, `check-dual` in full mode, and 13 SPIR-V artifacts
  through `spirv-val`. The runs before it were at `68c6f40` (m5.c2, 2026-11-02, one untracked log file), `27b342b` (M4 close after
  m4.i) and `c964222` (2026-11-02, before the M4 sanity
  fixes), then `5971c12` (M3 close, 2026-11-01, clean tree),
  then `96e4b9d` (2026-11-01, before the M3 sanity fixes), then `634a50e` (M2 close, 2026-10-24) and the one before that (M1) at `a6cbb8f`, together with the Slangc
  and Reflect suites. The local `.ai/VALIDATION.json` record stamps the commit and dirty
  state and contains no machine paths.
- DX19 is closed: full validation, two independent reviews, and fixes (`027eef7`,
  `08a4843`, `3296885`).
- CI (`unit` job: fmt, clippy, Unit suite; `msrv` job on Rust 1.90) passes on main;
  one push on 2026-10-23 (`10b2e06`) was red and was fixed in `3cdc607`.

## Current contracts

- Task acceptance criteria: [tasks/README.md](tasks/README.md)
- T10 atomics: [tasks/T10-atomics.md](tasks/T10-atomics.md)
- T11 Result lowering: [tasks/T11-result.md](tasks/T11-result.md)
- T12 Result match: [tasks/T12-result-match.md](tasks/T12-result-match.md)
- T13 active count (deferred): [tasks/T13-active-count.md](tasks/T13-active-count.md)
- DX19 sidecar maps: [tasks/DX19-sidecar-diagnostic-maps.md](tasks/DX19-sidecar-diagnostic-maps.md)

## Latest DX evidence

- 2026-10-23 Unit-tier CI (`.github/workflows/ci.yml`, Rust pinned to 1.98.0): fmt,
  workspace clippy with warnings denied, `cargo test -p gust-macros` (54 passed) and
  `cargo test -p xtask` (6 passed) passed locally before the first push. Evidence tiers
  (Unit, Slangc, Spirv, Reflect, Gpu) are defined in `VALIDATION.md`. The DX tasks
  were renamed DX19/DX20 so decision ids continue at D21. History was rewritten to
  replace machine paths and the repository is now `nevermemo/gust`.
- 2026-10-23 approval and privacy policy: AGENTS.md gained an Approvals section
  (all actions pre-authorized by the owner) and a parallel-lane branch rule;
  `.ai/VALIDATION.json` is now untracked and xtask redacts the workspace root and home
  directory when writing it. `cargo fmt --all -- --check`, `cargo test -p xtask`
  (6 tests), `cargo clippy -p xtask -- -D warnings`, and `cargo xtask check-fast`
  passed through the shared cargo slot wrapper.
- 2026-10-23 cleanup: the stale T13-derived `generated-wgpu/pool__classify.*` exports
  were moved out of the tree, so `generated-wgpu/` again holds the 13 `.spv` files
  `check-full` expects. The T13 stash is untouched.
- 2026-10-22 documentation sync, no code changes: README, ARCHITECTURE,
  ENGINE_HURRICANE, NEXT_TASKS, HANDOFF, and the Copilot builder agent profile now
  record DX19 as committed with release evidence pending, point T13 at its stash, and
  list atomics (T10), Option/Result (D14), and bounded struct literals (D07) as
  supported. An independent read-only review checked each added claim against the
  validator, emitter, regressions, fixtures, GPU tests, and xtask (PASS);
  `git diff --check` passed.
- `cargo xtask check-feature core` passed (19 core tests), including sidecar-map
  schema/line-range coverage and construct-enriched compiler diagnostics.
- `cargo test -p xtask` passed (3 routing tests).
- `cargo xtask check-feature result` passed (6 macro plus 4 GPU/target tests).
- `cargo fmt --all -- --check` and `git diff --check` passed.
- DX19 writes temporary `kernel.map.json` files with stable Slang line/construct
  segments and appends mapped kernel/construct labels to Slang failures. Focused
  `cargo test -p gust --lib slang` passed (14).
- Full validation was not refreshed after DX19 because T13 changes were then in the
  working tree. They are now stashed; the `generated-wgpu/` blocker above remains.

## Archive

Completed task reports and superseded ownership records are preserved in
[history/STATUS-2026-09-09.md](history/STATUS-2026-09-09.md). The slice entries of M1 to M4
are in [history/STATUS-M1-M4.md](history/STATUS-M1-M4.md).
