# Current status

## Active: roadmap M8 (2026-11-07)

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

M7 (GPU concepts in Rust) closed on 2026-11-04 (D126). MX (execution shapes and textures) closed on 2026-11-05 (D129)
after a sanity check by Fable and Opus in the Codex slot (both "proceed with fixes"), the fixes in slices mx.z and
mr.b3, and `check-full` at `27a461a`; a Codex confirming pass followed (proceed with fixes, nothing reopens MX; D132).
MR (the first public release): every slice is merged, the last mr.f, the fixes of the MR sanity check (Fable and
Codex, both "publish after fixes"; D135). The owner deferred the publish (2026-11-07, D148); `release/0.1` keeps
0.1.0 publishable. M8 (capability variants, 0.2.0; D136 to D147) is active: m8.d, m8.a, m8.w1, m8.b1, m8.w2, m8.f,
m8.b2, m8.s and m8.v are merged (merge `e11a3f0`; the 27 main gates on the merged tree on the RTX 5090, 16:29Z to
17:44Z; lavapipe green on every lane; main's CI green, ci 42397829562 and gpu-software 42397829548). The
`semver` job's first run reports B1 only (`KernelDescriptor` `#[non_exhaustive]`; nothing for gust-compute-wgpu).
Subgroup operations (m8.s): tests/subgroup.rs passed on the RTX 5090 at subgroup size 32 and on lavapipe at size 8
(run 42378261589); not run on the Radeon 890M. Local gates load a newer Vulkan validation layer (D149). The Slang
pin is 2026.21 (D150; spike SL21, slice m8.p, merge `b6d7c61`; 26 main gates green on the merge, 2026-11-08T01:07Z; lavapipe runs 42421531994 and 42427279802 green); `release/0.1` stays on
2026.20. The bump found test binaries that no suite or CI job runs (NEXT_TASKS, M8 follow-ups). Routing at
expansion is in (m8.r1, merge `b859af7`; 26 main gates green on the merge, 2026-11-08T06:13Z; lavapipe runs 42442568375 and 42448880812 green): the macro resolves each kernel's target from the `cpu`
profile, the route variable and the manifest's `routes`, and a CPU-routed kernel has no `GpuKernel` impl (D137,
amended). check-cpu-target E-j's flips now fail on a stale expansion: with the route's `option_env!` removed, the
gate failed. A release wasm32 build with no `GUST_PROFILE` and a CPU route builds (no E0752). Subgroup evidence
is in (m8.e, merge `e9150ad`; 26 main gates green on the merge, 2026-11-08T10:34Z; lavapipe runs 42461935960 and 42466622279 green). Measured first under a pre-registered rule (faster only when the
subgroup form's slowest of three interleaved runs beats v1's fastest), the subgroup sum was faster in none of six
cells (RTX 5090 and Radeon 890M, Vulkan, 1e6 to 1e8 elements), so it ships as `gust_algo::evidence`, outside the
variant set, and the shipped `sum` is unchanged. check-difftest now runs a `dev+subgroups` batch whose results must
report the profile's capabilities and whose records must equal the `dev` batch's byte for byte (RTX 5090 and
lavapipe); `examples/subgroup-reduce` is the strict `native+subgroups` proof. The routing
driver is in (m8.r2, merge `5238deb`, D151; 27 main gates green on the merge, 2026-11-08T15:17Z; lavapipe runs 42486455270 and 42487784359 green): `cargo gust build|check|run|test` read `gust.toml`,
validate it against `cargo metadata`'s declarations, set one route variable per declared tag and write
`routes.json`. check-routing measured the rebuild in both directions (with the route tracking removed, the
unset-to-set flip failed) and ran its GPU case on the RTX 5090; the CI `routing` job runs the GPU-free cases. Not
covered in the first round: its rc lines were deleted (the stop-rule logs remain), and check-dual ran
`--check-only`; the fix rounds ran check-dual in full. The tidy pass's code half is in (m8.x1, merge
`c245266`; 27 main gates green on the merge, 2026-11-08T20:07Z; pull request CI 42511073744 and lavapipe 42511073531 green on a630103): Unit runs gust-compute's `pixel_twin`, `textures_twin`, `parallel`, `toolchain`
and doc tests and cargo-gust's tests; Slangc runs the helper-backed cargo-gust tests and `testdata::variant`
under `dev+subgroups`, requiring each named test and the capability line; `--suite Gpu --profile` also runs
gust-algo; check-examples builds `subgroup-reduce` under plain `native` and runs its recompile check; `targets = []`
is E0753; the difftest result format is 7. Their first Linux run (a draft pull request, nevermemo/gust#4) failed
two cargo-gust tests that judged Windows paths; both were fixed in test code. The builder once moved round 0's rc
file aside, so lines of a superseded batch are lost; every final gate has its line. The prose half is in (m8.x2,
merge `4219bb5`; 27 main gates green on the merge, 2026-11-09T00:17Z; pull request CI 42532267792 and lavapipe 42532267814 green on a1f91ca): README, ARCHITECTURE, SEMANTICS, DIAGNOSTICS, AGENTS.md,
VALIDATION, DIFFTEST, RELEASE.md, NEXT_TASKS and the seven skills describe M8 as built; CHANGELOG 0.2.0 carries the
semver report (one break, B1, in two parts) and the compatibility list (113 manifests on `release/0.1`: 94 unchanged,
4 derived-only, 15 changed, each built by a main gate). The Opus review (0/3/13, re-check MERGE) found stale counts
and a NEXT_TASKS pointer to an item never filed; both fixed. D152 corrects D138's licence wording.

- M8 close records (M8 plan §4 row "close"; 2026-11-09; D153). (i) `cargo xtask check-full` passed at `2f96e69`
  (2026-11-09, 05:11 to 05:53Z, NVIDIA GeForce RTX 5090 / Vulkan, driver 610.88, Slang v2026.21; clean tree), with the
  `dev+subgroups` Gpu step m8.z added; `check-full` and `check-release --check-only` had also passed on the first
  closing candidate `c0c89c1` (00:21 to 01:12Z). (ii) The 27 main gates passed on the m8.z merge `2f96e69` (03:50 to
  05:10Z). (iii) Linux and lavapipe: main CI 42569921459 green on `2f96e69` (attempt 3; attempts 1 and 2 failed in the
  slang job on GitHub release-download 504s), pull request CI 42557064111 green on its rerun on `62ea3b4`, and
  gpu-software run 42569921556 green on `2f96e69`. (iv) The sanity checks, Fable (plans/m8/sanity-fable.md) and Opus in
  the Codex slot (plans/m8/sanity-opus.md; Codex was out of quota until 05:37Z), both said "proceed with fixes"; Codex's
  confirming pass on `2f96e69` (plans/m8/sanity-codex.md) said the same, found S1 to S4 closed by m8.z and one new item,
  filed in NEXT_TASKS: E0751's invocation product can overflow. Nothing reopened M8. Opus found what Fable missed:
  `cargo gust` inferred the profile from its command line only, so a target from cargo config or a custom profile could
  route a kernel for the wrong profile silently. (v) Slice m8.z (merge `2f96e69`; Opus review 0/3/6, re-check MERGE)
  fixed it with `GUST_ROUTED_PROFILE`, checked in the code the macro writes for a routed module after a measurement
  (with the check in `gust`'s build script a plain-cargo/driver switch recompiled three crates in the routing fixture,
  `gust` among them; in the macro's output two), added check-routing's `guard` case through the real driver, restored
  the flip's unset-to-set mutation proof, ran the two never-run subgroup broadcast forms on the RTX 5090 and lavapipe,
  put the `dev+subgroups` Gpu suite (gust-algo included) into `check-full`, and named the E0745 subgroup-name break in
  CHANGELOG. Next: the M9 plan.

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
  `scan::exclusive` over `u32` (plain `+`: the GPU wraps, the debug twin panics; `reduce::sum_wrapping`, which this
  slice adds, is the wrapping sum), `2k + 1` dispatches in
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
  Erratum (m7.a2): the map kernel named its dispatch id `__gust_id`, a reserved CPU-twin marker of the export
  guard, so from `5609536` until m7.a2 `cargo gust build` refused every `#[gpu] fn` (E0712). m7.a1's tests and
  its wasm32 build never ran export mode.
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
- M7 m7.g1 (`944b2c6`, fix round `25b08fe`, merge `1516c94`): workgroup shared memory and barriers in the dialect
  (D100, D101). `Workgroup<[T; N]>` kernel parameters become per-kernel `groupshared` arrays (16 KiB per kernel at
  most); `LocalId`/`GroupId` are derived from the dispatch id; `workgroup_barrier()` lowers to
  `GroupMemoryBarrierWithGroupSync()` (SPIR-V `OpControlBarrier`, checked with spirv-val and spirv-dis). The
  validator follows plan §4.2 as a whitelist of barrier positions, and the barrier pass itself rejects `return`,
  `?`, labelled jumps and `break`/`continue` in a barrier loop, so it does not lean on the dialect's general bans.
  New diagnostics E0119, E0210, E0327, E0414 to E0416, E0717, E0718; E0709 now rejects `cpu = "lockstep"` (m7.g2).
  13 check-ui fixtures (46 pinned in all); the `?` fixture keeps its `?` in a helper because rustc adds its own
  error beside the macro's for one in the kernel body, so direct pass tests cover that case. GPU (RTX 5090): the
  block reduce of plan §4.1 is bit-equal to an integer host sum at 1, 255, 256, 257 and 65 537 elements, and every
  invocation's ids equal `id % 256` and `id / 256`. Mutations: dropping either barrier fails the block reduce
  (both gave the same wrong partial, 10701 for 12120, on this GPU); a barrier allowed in `if` lets three fixtures
  compile; `GroupId` as `id % size` fails the id test; no 16 KiB check lets its fixture compile. check-changed now
  runs the macro suite whenever gust-macros changes. The CPU twin's id derivation writes a size-1 axis out (clippy
  `modulo_one` in user crates). One adapter. Pin 244/28 to 249/29. Opus merge after one fix round.
  Main: 23 gate steps green on `1516c94` (08:57 to 09:32Z, RTX 5090); pins 249/29 and gust-algo 32;
  check-examples green after the merge (the validator now runs on every example kernel).
- M7 m7.a2 (`6afbd36` and `126363c`, coordinator merges `218c06e` and `2285111`, fix round `ab7097e`, merge
  `44299e5`): map examples and evidence. `examples/map-chain` runs three `#[gpu] fn` maps as a resident chain
  (`GpuVec` and `map_gpu_on`, one upload and one read-back counted as the device pool's hand-outs; a read-back that
  bypasses the pool would not be counted) and as host-slice calls, both against an independent `f64` reference
  within a derived 9u bound; tests from 0 to 65 537 elements and one large size. web-vector-add gains a `map_async`
  path behind `default_device_async` (compiled for wasm32 and checked by check-web's tree guard; the full check-web
  also runs it, see the erratum below). Committed artifacts for both examples; the web byte-copy test now accepts a
  superset manifest and still
  compares the shared kernels, `schema` and `slang`. It fixes the m7.a1 map kernel id (renamed `__gust_gid`, with a
  regression test over the exported `.slang` and record). Exported kernels 20 (m7.t14 and m7.a2 three each). The
  README map table is provisional: a smoke run on the shared RTX 5090, labelled with its conditions and with what
  it checked; map over host slices is 0.49 to 1.05 of a hand-written dispatch, and a cheap function never beat the
  CPU at the sizes measured. The published run waits for a quiet window. Under `native` a `#[gpu] fn` cannot run
  from an artifact yet (the workspace profile pin requires `slang-runtime` on every example). Mutations: a
  read-back between steps, swapped steps, a deleted artifact and the old id name each fail. Opus merge after one
  fix round.
  Main: 23 gate steps green on `44299e5` (10:23 to 10:57Z, RTX 5090); pins 249/29 and gust-algo 32;
  check-examples green.
  Erratum (M7 close): the full check-web ran this path in headless Chrome at m7.a2's own gate; the page logs `GUST_OK`
  only after `double_add::map_async` matches the host reference (lib.rs:145-160). The sentence "not run in a browser"
  above was wrong. This does not prove the wasm32 `Job` contract (D69), which stays open (NEXT_TASKS).
- M7 m7.r4 (`0c5d456`, fix round `b945ab0`, merge `d6b425f`): gust-algo v1 with shared memory. Block reduce (sum,
  min, max over u32/i32/f32 and the wrapping u32 sum) and tile scan (u32 inclusive and exclusive) with
  `Workgroup<[T; 256]>` and barriers, `cpu = "unsupported"` (D101); v0 stays the CPU form and the fallback
  (`limits::admits` picks v0 where a v1 launch or its 3072 bytes of workgroup memory do not fit). Public signatures
  unchanged; f32 sums now differ bit-wise from v0 (another grouping) within the bound of the variant that ran, and a
  caller may assert the larger of the two bounds. Measured end to end on the shared RTX 5090 (interleaved, 5
  warm-ups, 3 to 5 runs): median v0/v1 1.16 to 1.38 for reduce and 1.52 to 3.76 for scan at 1e6, 1e7 and 1e8, so
  `variant::shipped` picks v1 for all six operations. The choice is provisional: the statistic (median of per-run
  medians) was chosen after the runs, the lowest single run was 1.085 (min f32, 1e8), the small-size win is mostly
  one dispatch and one pooled zeroing fewer, and the Radeon 890M half of plan §5.5 is owed. The builder's own VRAM
  reached 4.3 GB at 1e8, over the slice's 1 GB limit (no other work was disturbed). component-pool's ignored GPU
  tests pass on v1. Mutations: a dropped barrier (17 tests fail), a dropped scan guard (8 fail; the validator
  accepts it), a flipped shipped variant (the observer test fails). Pin gust-algo 32 to 49; naga_layout goldens 67
  to 82. Opus merge after one fix round.
  Erratum (public CI on `a6a1546`): `gpu_v1_runs_the_deepest_schedules` bound 2^26 + 1 `u32` (256 MiB) on the
  adapter's own limits, and lavapipe allows 128 MiB, so gpu-software failed with `DeviceLimit` before any dispatch.
  `282ad27` (merge `9dda10d`): below that limit the test prints that the three-pass reduction did not run, asserts
  the `DeviceLimit` of the smallest input past the limit, and runs the deepest reduction that fits (2^25 elements, two
  passes) and the deep scans; unchanged on the RTX 5090.
  Main: 23 gate steps green on `d6b425f` (12:15 to 12:46Z, RTX 5090); pins 249/29 and gust-algo 49.
- M7 m7.d (`dfead07`, fix round `4428dca`, coordinator merge `36d78c2`, merge `853aade`; D102): multi-dimensional
  dispatch. `Extent { x, y, z }` with `call_extent`, `submit_extent` and `dispatch_extent` beside the 1D forms;
  `LengthRule::Shared` compares the checked product. The cap rule is one function on both backends
  (`Extent::rounded_total`: whole workgroups per axis, then the product, at most 2^32); the GPU also checks
  workgroup size y and z and workgroups per axis; a zero axis dispatches nothing. The CPU twin walks z, y, x with a
  1D fast path (check-cpu-target 160 s to 72 s). `BufferDispatch` keeps `element_count` public and its y and z
  private, so the old `d.element_count = n` pattern dispatches n (regression test). Evidence on the RTX 5090: 16x16
  and 4x4x4 kernels on the GPU and the twin against a host reference (1x1, 15x17, 256x256, 3D, zero axes,
  `LocalId`/`GroupId` exact in 2D); exactly 2^32 invocations run in 2D (the last cell read back) and in 3D
  (acceptance and invocation 0), one past refused. On a CPU adapter the giant dispatch is not run and a line says
  so; that branch first runs in public CI (lavapipe). The twin at 2^32 is checked through `cpu::validate` only.
  Staged graphs stay 1D (SEMANTICS says how they refuse); the generated host takes an extent for 2D and 3D kernels.
  Error texts no longer call y and z unsupported or name M7. Mutations a to d each fail a test. Pin 249/29 to
  270/30 (17 GPU and 4 host tests). Opus merge after one fix round; the reviewer's clippy on the wip was green.
  Main: 22 of 23 gate steps green on `853aade` (12:50 to 13:27Z, RTX 5090; pins 270/30 and gust-algo 49) and
  check-examples green; clippy-wasm refused four native-only 1D launch helpers as dead code on wasm32, fixed in
  `154102a` with the file's `cfg_attr` allow. With the m7.r4 lavapipe fix merged (`9dda10d`), fmt, clippy,
  clippy-wasm, check-web, gust-algo and MSRV green (13:32 to 13:35Z).
- M7 m7.p (`e2612b6`, fix round `2095066`, merge `910ee1c`; D103): `par_for!`. The macro lowers the body once under a
  placeholder, names the inner `#[gpu]` module by the hash of that Slang, and expands to a typed launch through
  `dispatch_extent`; schema 2 export records without `source` and one sidecar per call site, read by cargo-gust
  beside schema 1 (unchanged byte for byte). Evidence on the RTX 5090: 26 GPU and twin tests against host references
  (1D, 2D, 3D, non-multiple and zero extents, over the cap, scalars, the default device, resident `GpuVec`, and a
  `workgroup_size` taller or deeper than the loop), the ten-call Slang golden, hygiene (spelling, line, file and
  profile give one key; two equal bodies give one artifact and two sidecars), byte goldens for the schema 2 record
  and the sidecar, check-ui fixtures for E0719 and E0720, and a cpu-target case. The Opus review found that the
  guard compared only the named axes, so a taller workgroup ran the body more than once (probe: `[4, 4, ..]` for a
  +1 body); fixed, and the twin tests fail on the old guard. On the GPU those tests do not catch the old guard,
  because the racing `+ 1` writes all store 1. Pin 270/30 to 296/31. Not yet: `doctor`/`check` list only the
  origin site, the naga layout gate skips `par_for!` kernels, the over-cap error is `map::Error::Device`, and
  `cargo gust build` cannot export a `par_for!` kernel until m6.u2f. Opus merge after one fix round.
  Main: 23 gate steps green on `910ee1c` (16:55 to 17:33Z, RTX 5090; pins 296/31 and gust-algo 49) and
  check-examples green.
- M6 m6.u2f (`9dc3169`, fix round `e596bb7`, merge `bee193a`; D104): `cargo gust build` on uniform kernels.
  Before it, every kernel with a `Uniform<T>` failed the export with "missing descriptor element layout" (found by
  the m7.p builder, reproduced on a plain `#[gpu]` kernel): the macro writes no uniform layout and cargo-gust filled
  only storage ones. cargo-gust now fills a uniform from the reflection's `constantBuffer` element as `Std140V1`
  (fill.rs, export.rs), for `build` and `check`. Evidence: stored helper output for a uniform kernel and a nested
  one with padding builds through the export path, and the filled layouts equal the macro's descriptor layouts; bad
  offset, size, nested alignment, field name, rules and resource kind are refused; a std140-valid rename passes the
  export and fails `verify_artifact`; a probe crate ran `build`, `check`, `check --recompile` and a SPIR-V build.
  Mutations a and b fail tests. Not yet: vector and matrix elements (`Vec4`, `Mat4`) still fail `cargo gust build`
  for storage and uniforms ("unsupported element type `vector`"), and uniforms are not listed in `export_unchecked`
  (D104). Opus merge after one fix round.
  Main: 23 gate steps green on `bee193a` (17:40 to 18:12Z, RTX 5090; pins 296/31 and gust-algo 49) and
  check-examples green.
- M7 m7.t1 (`e1c2410`, fix round `fd7a7b6`, merge `1278b61`): tidy pass 1, the cleanup scout's "now" items. xtask
  has one `exit_code` helper (process.rs) for its five copies, behaviour unchanged. The README's limits and
  dispatch claims now match m7.a0 (D84) and m7.d (D102), and short `par_for!` and workgroup-memory sections mark
  each claim as its honesty table does. HANDOFF drops the T13 stash note; `.gitignore` drops the dead
  `/artifacts/generated-wgpu/` entry and ignores `/.claude/`. The `element()` dedupe in gust-macros waits for m7.lv,
  which edits the same expand files. Opus merge after one README fix round.
  Main: 22 of 23 gate steps green on `1278b61` (19:40 to 20:22Z, RTX 5090); gpu-verify died once in the
  gust-algo GPU tests (STATUS_ACCESS_VIOLATION, NEXT_TASKS) and passed alone (20:25 to 20:32Z); check-examples
  green.
- M7 m7.f1 (`754f12f`, fix round `64100b2`, merge `96051c8`): the Linux CI flake "Text file busy" (ETXTBSY) on
  `6be06d4`. A test wrote a helper script and ran it while another thread's `fork` still held the write
  descriptor. The four places in `gust` that start the reflection helper or `slangc` (slang/mod.rs `compile_at`,
  reflect/mod.rs `compile_reflected_with_helper`, the clean-compile closure and `reflect`) now go through one
  bounded retry on `ExecutableFileBusy` only (5 tries, 10 to 80 ms apart; every other result returned at once), so
  real programs that write or download a helper and run it are covered too. Tests inject the spawn: the bound, the
  waits, the last error returned, and no retry for other errors. Opus approved; one test nit fixed.
- MR mr.t1 (`448c52b`, fix round `069db3b`, merge `76cfeb0`; D109): `gust::toolchain`, the one owner of the
  toolchain-location rules, as a std-only `#[doc(hidden)]` module that is not wired yet (mr.t2 adds the `mod` line
  and points cargo-gust and the bridge at it). The cache root, the Slang pin and validated-build rule, the `slangc`
  and helper paths in the order `GUST_SLANG_REFLECT`, managed cache, checkout, and the version check. Stricter
  than cargo-gust's copy: a relative `GUST_SLANG_REFLECT` is refused, no current-directory or bare-name lookup,
  and the checkout fallback needs the `xtask/Cargo.toml` marker and `crates/gust` resolving to the crate's own
  directory (a vendored copy or a nested work directory finds nothing). The tests copy cargo-gust's tables case by
  case (slang_tests.rs) and run on hand-built environments; three mutations fail tests. Opus "merge"; the four
  should-fixes and two nits were taken. Not yet: the `slangc -version` launch is outside m7.f1's retry (mr.t2).
  Main: 23 gate steps green on `96051c8` (20:50 to 21:23Z, RTX 5090) and check-examples green.
- M7 m7.lv (`3d5d0c8`, fix round `9d7165e`, merge `9d47f67`; D116): uniform and vector layouts in the
  export record. The macro now writes `std140_v1` layouts for uniforms and `storage_v2` layouts for storage
  elements holding `Vec2`/`Vec4`/`IVec*`/`UVec*`/`Mat2`/`Mat4` (new `vector` and `matrix` kinds), so those kernels
  build with `cargo gust build` and every layout the macro describes is checked against the reflection at export;
  a null uniform is listed in `export_unchecked` (D104 retired). Records with only scalars and every existing
  golden are unchanged; new golden uniform.kernel.json. Evidence: a probe crate with five kernels (nested uniforms,
  `Frame` with `Mat4`/`Vec4`/`Vec2`, vector storage, and storage `Mat4`/`Mat2` and a struct holding a `Mat4`) ran
  `build`, `check` and `check --recompile` with no `export_unchecked` entry; a test rebuilds the macro's own
  descriptors from the records; tampered records are refused, both by the rules and, with valid rules, by the
  reflection comparison; the run-time check (D17) still refuses a std140-valid rename; mutations a to c fail
  tests. Opus "merge after fixes" (a torn error message, the run-time pin, a rules-valid tamper); fixed, plus a
  storage-matrix end-to-end case.
  Main: 23 gate steps green on `88b46d0` (21:29 to 22:08Z, RTX 5090) and check-examples green.
- MR mr.a (`c520795`, fix round `f2125ec`, merge `88b46d0`; D107, D110): packaging metadata. The workspace
  gains `repository`, `keywords` and `categories`; the four publishable crates (D110) set `documentation` and
  docs.rs metadata, and the internal dependencies are exact `=0.1.0` pins (D107). `cargo-gust` is publishable now
  and excludes its test modules and testdata; gust-macros excludes its collision tests; every example, gust-algo
  and xtask say `publish = false`. Each crate has a README and the two licence files; the root `LICENSE-APACHE` is
  now the full Apache License 2.0 text (it was the short notice), byte-equal in every copy. `cargo xtask lint`
  checks the metadata, the licence copies and that every `path` dependency on a `crates/*` package names it (or a
  matching `package =`), from `cargo metadata` (offline). Evidence: the four package lists hold no test, fixture or
  machine-specific file; packaging with verification of all four passes; lint mutations a to d fail. Opus "merge
  after fixes"; the six items ruled were fixed. Not yet: the crate READMEs' `cargo install cargo-gust --locked`
  works only after mr.c (today it stops at the `gust` build script without `GUST_PROFILE=dev`), and the root
  README's lint line is mr.d's.
- M7 m7.t2 (`1142a09`, fix round `6f37387`, merge `6b0d59a`): one element-type lookup in gust-macros.
  `map` and `par_for!` held two copies of the type-to-element rules; they differed on purpose in diagnostic code
  (`MapSignature` against `ParForBinding`), wording and `bool` (a map accepts it, a `par_for!` buffer does not).
  Now `element_types` (expand/map.rs) is the one pure lookup, returning the element types or a reason
  (`ElementReject`), and each site keeps its own code, text and `bool` rule. Output is byte-identical: no golden,
  expected or UI file changed. The review found nothing pinned `par_for!`'s own rejections, so two check-ui
  fixtures now do (`par-for-bool-buffer`, E0720; `par-for-three-wide`, E0204), and dropping the `bool` guard fails
  the first. Opus merge; that should-fix and a doc nit taken. Main gates then found xtask's fixture-name pins
  (tests/ui.rs) still without the two new fixtures; `6c83aab` adds them.
  Main: 20 of 23 gate steps green on `6b0d59a` (22:21 to 22:52Z, RTX 5090); xtask-test, unit and slangc failed
  on those pins and passed with fmt and clippy on the fix (22:55 to 23:05Z), and check-examples green.
- MR mr.c (`9c19a08`, fix round `e421426`, merge `e85e1b5`; D108, D117): the `cargo-gust-bridge` feature.
  `gust` gains the internal feature `cargo-gust-bridge`, which `cargo-gust` enables on its `gust` dependency, so
  `cargo install cargo-gust --locked` builds with `GUST_PROFILE` unset: a release build with the bridge infers `dev`
  (not on `wasm32`) and prints one warning, and an explicit `native`, `web` or `cpu` is unchanged. The kill rule did
  not fire: from `tests/precompiled/app` the feature is not in the graph. check-precompiled gains E6g (`dev` and the
  warning), E6h (refused under `native`), E6i (this repository's `--workspace --release` check is `dev` with the
  warning) and E6j (a mistaken `cargo-gust` dependency, D117). Install proof: with `GUST_PROFILE` unset,
  `cargo install --locked` into a scratch root, then `--version` and `doctor`. The review's must-fix was the record
  (D117); its should-fixes (the evidence line, the stale sentences in README, lib.rs and VALIDATION, every gate on
  the final tree) and three nits were taken, and the ci.yml comment is mr.r's.
  Main: 23 gate steps green on `e85e1b5` (2026-11-03 23:34 to 2026-11-04 00:16Z, RTX 5090) and check-examples green.
- MX mx.p0 (`d10c871`, fix round `c5a4ac2`, merge `7ed260f`; implements D63, plan §2): P0, the
  stage-profile trait. `StageProfile` (crates/gust-macros/src/profile/mod.rs) has fourteen methods and one
  implementation, `grid` (profile/grid.rs); the emitter, the validator, the expansion and the twin harness now read
  every stage-specific rule through it: the stage attribute, the entry signature, prologue and epilogue, extra
  bindings, invocation names, binding slots (the emitter numbers and declares a resource by the same slot rule as the
  descriptor; a Probe test pins it), value-parameter refusal, `#[kernel]` options and the default workgroup, the
  dispatch-parameter sentences, the twin's iterator and the reflected workgroup size. The byte-identical gate held:
  no golden, artifact or UI expectation changed, and `cargo xtask export-artifacts --compare` (new, with
  `--hash-manifest`) matched a manifest of 80 exports (`.slang`, `.rs`, `.wgsl`, `.spv`) captured at `6b0d59a`
  before any source edit. Kill rule (§2.4) clear: fourteen methods, no hook sees a syn body, no profile call in
  `emit_statements` or `emit_expression`. Not yet read by anything: `shape`, `lower_output_write` (mx.c's
  classifier), `contract_tier`; a shape refusal from `validate_body` uses E0811 as a placeholder until mx.c gives it
  a code; an extra binding has no descriptor entry yet (mx.p1a). Opus merge after fixes; the must-fix (slot
  numbering), three should-fixes and four nits were taken.
  Main: 23 gate steps green on `7ed260f` (2026-11-04 01:15 to 01:54Z, RTX 5090), check-examples and the export compare green.
- MR mr.r (`dadf4f5`, fix round `b521215`, merge `9cc2f60`; D107, plan §3.3):
  `cargo xtask check-release [--check-only] [--work-dir DIR]`, the permanent release gate (xtask/src/release.rs). It
  finds the publishable members and their names in `cargo metadata`, so it passes unchanged after the D48 rename, and
  runs: R1 one `cargo package --locked` of all four with verification; R2 each package's file list against one
  allow-list (E8 now uses the same list and the same work directory, which reports a failed deletion); R3 metadata, one
  version, exact internal pins; R4 every other member unpublished; R5 `cargo doc` with every warning an error on stable
  and `+nightly --cfg docsrs` for the host target, skipped with a note when no nightly toolchain is installed
  (`KNOWN_DOC_DEBT` empty since mr.b2; docs.rs also builds the `wasm32` target, NEXT_TASKS, "MR follow-ups"); R6 each
  `.crate` under 2 MiB; R7 the README's "Install from crates.io" recipe word for word (its blocks equal the fixture
  `xtask/src/release/hello-gust/`, users depend on `"0.1"` per D107) built against the unpacked crates, `cargo install`
  of the unpacked `cargo-gust` without `GUST_PROFILE`, `cargo gust build`, the release build and both runs on the GPU;
  R8 `cargo publish --dry-run --locked` (it warns that an unrelated `gust` 0.1.0 exists, as expected before the rename).
  `--check-only` skips what needs Slang or a GPU and prints each skip. CI runs it in a new `release` job; `check-full`
  runs it in full. Opus merge after fixes; no must-fix, five should-fixes and five nits taken. Main: 23 gate steps green
  on `9cc2f60` (2026-11-04 02:08 to 02:39Z, RTX 5090), check-examples and the full check-release green.
- MR mr.d (`7a925dc`, fix round `b9357d2`, merge `7598a3c`): docs that work from crates.io, prose only.
  The root README gains the "Install from crates.io" path around check-release's code blocks (the dev run with a
  user feature, the release build, renamed dependencies and the two-direct-`gust` clash), with a note that the
  packages are not on crates.io yet: the prose uses the release names (D48) and the blocks keep today's names until
  mr.n. New "Platforms", "Stability" and "Minimum Rust" (1.90, the `msrv` CI job; gates on stable 1.98.0) sections;
  `CHANGELOG.md` (0.1.0) and `docs/development/RELEASE.md` (the steps, which are the owner's acts, and the
  post-publish check) are new; ARCHITECTURE names package and library names; the crate READMEs install with
  `--version "~0.1"`. Opus merge after fixes: the must-fix (the README said the crates were published), three
  should-fixes and ten nits taken.
  Main: docs-check, lint, xtask tests and check-release --check-only green on `7598a3c` (2026-11-04 03:38 to 03:42Z; Markdown-only slice).
- MX mx.k (`59c47c6`, fix round `f2fe244`, merge `289d958`; D89, plan §5.4): the shape in the
  descriptor, the record and the manifest. `gust::Shape` (`#[non_exhaustive]`: `Grid`, `Pixel`) and
  `KernelDescriptor::shape`, which the macro sets from the stage profile; new public `Shape::{name, from_name,
  slang_stage}` and `gust::slang::compile_source_as`. Record schema 3 is schema 2's grammar plus an optional `shape`,
  written only for a non-grid kernel, so every grid record keeps schema 1 or 2 byte for byte; readers accept 1 to 3,
  read a missing `shape` as `grid` and refuse `shape` under 1 or 2. Manifest schema 2 (a `shape` on every kernel) is
  written only when a non-grid kernel exists; `cargo gust check` compares each entry's shape with its record's.
  `gust::slang`, `gust::reflect`, the helper's argv and cargo-gust take the Slang stage from the shape;
  `KernelCacheKey` has `shape`; `GOLDEN_KEYS` pins one body under two shapes as two keys. The helper is now required
  at schema 3 (xtask and cargo-gust refuse an older one with a rebuild hint, and gust's reflection names it), since
  an older helper cannot take the stage argument. No pixel kernel exists yet: a hand-written pixel descriptor fails
  closed at the helper. Byte-identical gate held (80 exports, every artifact, the schema 1 and 2 goldens). Opus merge
  after fixes; no must-fix, two should-fixes and two nits taken.
  Main: 20 of 23 gate steps green on `289d958` (2026-11-04 03:55 to 04:28Z, RTX 5090); the Gpu suite and verify --mode
  gpu failed on the pin, fixed in `c9862eb` (GPU_SUITE_PIN 297/31), then green with fmt, xtask test and lint (04:36 to
  04:45Z); check-examples and the 80-export compare green; check-release --check-only green in a fresh target directory
  (04:47 to 04:52Z; in the shared one cargo reused a stale gust build from its temporary registry, an mr.p item).
- M7 S7.4 (spike, D118; plan §1a): the lockstep barrier twin premise ended in outcome C. A hand-written threaded
  twin of the block reduce (256 scoped threads per workgroup, a `Mutex` + `Condvar` phase barrier, std only, no
  `unsafe`) handled every injected panic, barrier mismatch and timeout cleanly, but took 102 to 104 s at 1e6
  elements (A needs 2 s, C starts at 20 s; five runs, median, quiet window 2026-11-04 04:58 to 08:08Z, both output
  carriers alike): about 2.9 ms per barrier round, linear in the workgroup count. m7.g2 is dropped; barrier kernels
  keep `cpu = "unsupported"` (D101); m7.c no longer waits on m7.g2. The scratch crate stays local, never merged.
- M7 S7.5 (spike, D119; plan §1a, §6): the chunked CPU map ended in outcome A. The generator's chunk body over
  `std::thread::scope` (safe Rust, std only) gave the k = 64 polynomial 14.2x at 1e8 elements on 24 threads and
  9.6x at 1e7; vector-add 2 to 9x; every result bit-exact with one thread; the lowest failing index re-raised (48
  cases, 0 failed). Spawning costs about 0.1 ms per thread, so m7.c also bounds the thread count by the input and
  measures that bound (D119). Quiet window 2026-11-04 08:09Z, alone. The scratch crate stays local, never merged.
- MR mr.r2 (`62dc2fa`, fix round `e67a1b1`, merge `b719a30`): check-release packages in a target of its own.
  R1's one-call `cargo package` and R8's `cargo publish --dry-run` verify each crate against cargo's temporary
  registry, and cargo fingerprints a registry crate by name and version, so a build left in the caller's target by an
  earlier run was reused although the source changed (seen on the mx.k merge: cargo-gust built against a `gust`
  without `Shape`). Both now build in `<work>/package-target`, new each run. Cargo also unpacks that registry into
  `$CARGO_HOME/registry/src/<hash>/`; check-release deletes only the entry that its own build's dep-info names (so a
  concurrent run's entry is never touched) and fails, naming the entry, when that entry existed before the run (a
  reused `--work-dir`). Reproduced first as pre-registered: a probe constant used by cargo-gust failed R1 on the old
  code (E0425) and passed on the new. The fail-closed path is covered by a unit test only. Opus merge after fixes:
  the must-fix (concurrent runs deleting each other's entry), two should-fixes and the nits taken.
  Main: 23 gate steps green on `b719a30` (2026-11-04 09:49 to 10:20Z, RTX 5090) and check-release --check-only green in
  the shared main target (10:20 to 10:24Z; registry/src clean after).
- MX mx.t1 (`e9c83ca`, fix round `a836136`, merge `00b72c7`; D64, MX plan §3.1, §3.2, §3.5): textures and samplers in
  `gust`. `gust::texture` has the dialect types `Texture2D<T>`, `StorageTexture2D<F>` (the three r32 formats),
  `WriteTexture2D<F>` (all nine), `Sampler`, `Format`, `TextureUse` and `SamplerOptions`, and no wgpu type. A texture is
  a `Resource` of kind `Texture`; a sampler is its own call argument (`SamplerArg`, a sealed `SamplerSet` on a defaulted
  parameter of `Call`), so a typed call refuses a texture, a sampler or a buffer in the wrong place at compile time
  (five `tests/ui` fixtures; check-ui pin 59). The CPU twin has `CpuTexture2d<F>` and `CpuSampler` with `load`, `store`
  and `sample` (nearest and bilinear, clamp and repeat, the error domain of a linear sample), each convention tested
  with hand-worked values; `cpu::validate` refuses a sampler from another device. Descriptors carry a storage stand-in
  for textures until mx.t0b's `texture_v1`; cargo-gust's record reader and the artifact writer refuse the texture and
  sampler kinds until mx.t3. SEMANTICS §1, §4 and §5 gain the twin's texture rules; the ledger rows KM-12 to KM-14 wait
  for mx.t4's measurement (the id guard keeps 11 rows). Opus merge after fixes: no must-fix, three should-fixes taken,
  the fourth a plan edge (mx.t2 after mx.t0b). Main: 23 of 23 gates green on `00b72c7` (2026-11-04 14:00 to 14:44Z, RTX
  5090).
- MX mx.t0b (`8ba6482`, fix round `55c5072`, merge `3bea3a3`; D17, D91, D120, MX plan §3.3): helper schema 3 on the Rust
  side. The reflection reader reads schema 3's texture and sampler bindings, the entry point's stage and a fragment's
  targets; every schema-1 and schema-2 fixture reads as before. `cross_check_kernel` checks a sampled `Texture2D<T>` by
  slot, dimension, sample class and result width, a storage or write-only texture by slot, access, result type and exact
  format, a sampler by slot, and the stage against the descriptor's shape; each mismatch names the field.
  `ResourceLayout::texture_v1` carries the texture layout, the descriptor coherence check compares a format's channel
  count with the result width, and cargo-gust's `fill` learns the layout for `export_unchecked`. X-S2 ended A, so the
  helper's facts are the D17 witness and no CLI-JSON gate is built. Tamper tests (stage, target, access, format, sampler
  slot, result width) run over six schema-3 fixtures that are byte copies of the helper's output. mx.t1's storage
  stand-in for textures in descriptors stays until mx.t3, which also teaches cargo-gust's record reader and the artifact
  writer the texture kinds. Opus merge after fixes: one must-fix (a public doc linking a private item, which
  check-release R5 refuses), three should-fixes taken, the fourth owned by mx.t3. Tests: gust lib 292, reflection 24,
  cargo-gust 210. Main: 24 of 24 gates green on `3bea3a3` (2026-11-04 16:51 to 17:40Z, RTX 5090).
- MX mx.c (`db21129`, fix round `b51e378`, post-merge fix `de4a641`, merge `c349a68`; D63, MX plan §5.1 to §5.4): the
  shape classifier and the pin. `validate/classify.rs` sorts each kernel into `grid` or `pixel` by §5.1's conservative
  syntactic rules; anything the rules do not recognise is `grid`-only with the span named, and the map generator's and
  `par_for!`'s kernels are `grid`-only by the same rules (a test each). The kernel attribute takes
  `shape = grid | pixel` (default `grid`, at most once; `shape = cpu` is refused with the `GUST_PROFILE=cpu` hint), and
  a pin outside the compatible set is an error at the pin with a note at the excluding construct. Codes GUST-E0721 to
  E0730 (E0729 replaces the profile refusal's borrowed E0811; E0730, a pinned shape not available yet, holds `pixel`
  until mx.p1a), each in DIAGNOSTICS; the E0707 message names `shape`. The descriptor gains `compatible_shapes`, which
  nothing reads at run time yet. An explicit `workgroup_size` makes a kernel `grid`-only (§5.1), so today's difftest and
  example corpus is `grid`-only; mx.p1c must generate pixel-eligible variants. 19 `shape-*` ui fixtures pin every
  reachable reason (check-ui pin 78). Opus merge after fixes: one must-fix (the missing ui reasons), two should-fixes
  taken, the third kept by ruling (the `workgroup_size` rule). Tests: gust-macros 414, gust lib 292. Main: 24 of 24
  gates green on `c349a68` (2026-11-04 17:42 to 18:34Z, RTX 5090).
- MR mr.t2 (`470530d`, fix round `a835c95`, merge `f6c12e9`; D72, D109, MR plan §2.4): the toolchain wiring.
  `gust::toolchain` is now a module of `gust` (under `slang-runtime`), and `gust::slang` and `gust::reflect` find
  `slangc` and the helper through it; the guarded checkout fallback replaces the `CARGO_MANIFEST_DIR` one. cargo-gust's
  `slang`, `toolchain`, `tools` and `doctor` use the same rules (one PATH rule, no relative entries), `SLANG_VERSION` is
  `toolchain::SLANG_PIN`, and `doctor` prints the `gust-compute` version and the install line. Every helper and `slangc`
  launch in `gust` and cargo-gust goes through the one std-only ETXTBSY retry (5 tries, 10 to 80 ms). A
  `slangc -version` that differs from the pin is the typed `slang::Error::SlangVersionMismatch` naming both; a bad
  `GUST_SLANG_HOME`, `GUST_SLANG_REFLECT` or version probe is `slang::Error::Toolchain`; `slang::Error` and
  `toolchain::VersionError` are `#[non_exhaustive]`. `toolchain::require` refuses helper schemas 1 and 2. xtask exports
  `GUST_SLANG_REFLECT` to its cargo children when the repository helper exists, and `PinRule`, the CI `gpu-software` pin
  step, VALIDATION and the slang-language skill read the pin from `crates/gust/src/toolchain.rs`. The managed helper
  under the cache root is now used whenever the cache root is nameable, not only beside a managed `slangc` (cargo-gust
  still checks its schema and its tag against the located `slangc`). `cargo gust doctor`'s binary-path lines are
  byte-identical before and after. Opus merge after fixes: one must-fix (the CI pin step would have read an empty
  version), five should-fixes taken. Tests: gust lib 297, cargo-gust 213, xtask 505, Slangc 1525. Main: 24 of 24 gates
  green on `f6c12e9` (2026-11-04 20:26 to 21:25Z, RTX 5090).
- MX mx.t2 (`2435a49`, fix round `873d0bf`, merge `db28395`; D64, D90, D91, D120, MX plan §3.4): textures in gust-wgpu.
  `GpuTexture2d<F>` and `GpuSampler`, with `create_texture_2d`, `write_texture`, `read_texture` (native) and
  `read_texture_async`, and `create_sampler`; mx.t1's nine formats map to `wgpu::TextureFormat` through wgpu's
  `block_copy_size` and `sample_type`, and readback strips the 256-byte row padding. Texture, storage-texture and
  sampler bind-group entries come in both layout variants: with `float32-filterable` an `f32` sampled texture is
  filterable and takes a filtering sampler; without it the texture is unfilterable and a linear sampler is
  `Error::TextureFilter`. A wide write-only format is always bound write-only, and read-write is only for the r32
  formats (D120). `DeviceOptions` gains `features: FeaturesPolicy` (`Adapter` on native, as D90 has it for limits;
  `Default` on `wasm32` and in `HeadlessDevice::new`). `check_kernel_limits` covers sampled and storage textures,
  samplers and the 2D dimension of the opened device, and a sampled texture whose format's channel count differs from
  the declared result is refused on the untyped path too. New errors `TextureUse`, `TextureFormat`, `TextureFilter`,
  `TextureSize` and `SamplerOptions`; a texture larger than `max_buffer_size` reads back as `DeviceLimit`. The staged
  graph refuses texture kernels, and each texture carries a 4-byte identity buffer for the aliasing check until mx.p1b.
  The generated host takes `float32_filterable` when a kernel samples. Tests: 31 in `tests/textures.rs` (the nine
  formats round-trip bit-exactly at width 67, nearest and linear sampling within §3.5's bound, read-write r32 and
  write-only kernels from hand-built descriptors, limits, aliasing), which CI's `gpu-software` (lavapipe) also runs; Gpu
  pin 333/32. `#[gpu]` texture kernels arrive with mx.t3. Opus merge after fixes: one must-fix (a separate features knob
  broke D90), four should-fixes taken, the identity buffer deferred to mx.p1b. Main: 23 of 24 gates green on `db28395`
  (2026-11-04 22:00 to 22:51Z, RTX 5090); the Gpu step hit the known gust-algo STATUS_ACCESS_VIOLATION (the third time
  at a merge) and passed alone at 22:52 to 22:57Z (pin 333/32, gust-algo 49).
- MX mx.t3 (`2b6c6b4`, fix round `7f615a7`, merge `c234b67`; D62, D64, D91, D120, MX plan §3.1 to §3.3): textures in the
  macro. A `#[gpu]` kernel takes `Texture2D<T>`, `StorageTexture2D<F>` (the r32 formats), `WriteTexture2D<F>` and
  `Sampler` parameters, with `load`, `store`, `sample`, `width` and `height` under the coordinate rules of §3.1; the
  emitter declares them with `[format(..)]` where D91 and D120 need it, write-only textures as `WTexture2D`, and D62's
  format table is checked per shape. Descriptors carry mx.t0b's `texture_v1` in place of mx.t1's storage stand-in, so
  D17 compares real texture facts: it passes on the eight kernels of the new fixture `tests/fixtures/textures` (all nine
  write formats; a tenth golden SPIR-V set under `spirv-val`). The artifact writer and cargo-gust's record reader know
  the texture and sampler kinds (a writer golden `texture.kernel.json` that cargo-gust parses). The classifier keeps
  storage textures out of `pixel`. Codes GUST-E0731 to E0736 (a type outside the dialect, a wrong coordinate or arity,
  `sample` without a sampler, `sample` on integer texels, a method the texture lacks, a texture or sampler used as a
  value), each in DIAGNOSTICS; E0732 and E0734 are pinned by unit tests only, since rustc adds its own error. Six ui
  fixtures (check-ui pin 84). The macro-generated CPU twin runs in `crates/gust/tests/textures_twin.rs` (storage load
  and store, a write-only store, nearest and linear samples, against hand-worked values). Opus merge after fixes: one
  must-fix (`(t).width()` passed validation and emitted invalid Slang), five should-fixes and the nits taken. Tests:
  gust-macros 428, gust lib 292, cargo-gust 213. GPU runs of `#[gpu]` texture kernels are mx.t4's. Main: 24 of 24 gates
  green on `c234b67` (2026-11-04 22:59 to 23:47Z, RTX 5090).
- MX mx.t4 (`fbc43f0`, fix round `300bfbc`, merge `fbe52e5`; D52, D62, D64, D91, MX plan §3.5, §6): texture evidence,
  the first GPU runs of `#[gpu]` texture kernels. `crates/gust-wgpu/tests/textures.rs` (39 tests): `load` and `store`
  against the twin, bit-exact on all nine formats and in place on the three r32 formats; the fixture kernels against
  host loops, `shift` included (made race-free in the fix round: it read a texel its neighbour wrote, against SEMANTICS
  §5); a twin out-of-range `store` panics; binding one texture twice is `cpu::Error::Aliased` on the twin and
  `Error::AliasedBinding` on the GPU (not `Borrowed`: the alias is found before any borrow); nearest sampling exact away
  from an edge, and linear sampling and `blur`/`gather` within §3.5's bound against the twin and an f64 host reference
  (worst 0.865 of the bound clamp, 0.977 repeat). The nearest band was probed at 2^-20, 2^-12, 2^-9, 2^-8 and 2^-7 texel
  from an edge: on the RTX 5090 the GPU picked the neighbour only at 2^-20 (6 of 42 clamp, 6 of 60 repeat samples), so
  the test's 1/128 band is a tolerance, not a measurement, and widens §3.5's "within 1 ulp" for nearest sampling. New
  example `examples/texture-threshold` (storage texture in, a buffer and a write-only texture out; twin test,
  check-examples, `EXPECTED_SPIRV`); a `tests/cpu-target` texture kernel whose output line xtask now requires. Ledger
  rows KM-12 to KM-14 (guard 11 to 14), pinned on the RTX 5090 only; KM-13 asserts the `TextureFilter` refusal on an
  adapter without `float32-filterable`. Lavapipe (gpu-software with the nightly list, runs 41955449332 on `fbc43f0` and
  41965659661 on `300bfbc`): textures 39 passed, the ledger passed, and KM-13 printed D52's NOTE on llvmpipe (observed
  `0x40753f7c`, no software pin). `export-artifacts --compare` lists the four new `threshold__mark` exports (80 to 84),
  with no existing hash changed. Gpu pin 341/32. Opus merge after fixes: two must-fixes (VALIDATION said `shift` ran on
  the twin when it ran nowhere; KM-14 called the 1/128 band measured) and four should-fixes taken. In the fix round's
  check-fast, the gust-wgpu `textures` binary died once with STATUS_ACCESS_VIOLATION and passed on rerun: the first
  sighting outside gust-algo (see the m7.av follow-ups). Main: 24 of 24 gates green on `fbe52e5` (2026-11-05 04:08 to
  04:59Z, RTX 5090).
- MX mx.p1a (`5dfa505`, fix round `a523511`, merge `6bb4dff`; D61 to D63, D93, D120, D127, MX plan §4.1, §4.2, §4.6):
  the `pixel` profile in the macro. `crates/gust-macros/src/profile/pixel.rs` emits the fragment entry: each output is a
  storage write at the element's own index (D120), and the element is `y * 8192 + x` (D127, Option S).
  `gust::PIXEL_ROW_WIDTH` and `gust::PIXEL_MAX_ELEMENTS` sit beside `Shape` in `crates/gust/src/descriptor.rs`, and a
  twin test checks that every fixture kernel's Slang carries the width. Until mx.p1b, gust-wgpu refuses a kernel whose
  shape is not `grid` with `Error::ShapeUnsupported` before any pipeline is created
  (`crates/gust-wgpu/tests/shape_refusal.rs`), so pixel kernels run on the CPU twin (`crates/gust/tests/pixel_twin.rs`)
  and the `cpu` target only; a `tests/cpu-target` pixel kernel prints a line xtask now requires (`pixel_shade::shade`,
  1000 elements, max ULP 0). No new diagnostic code; E0730 is retired (two UI cases removed, `shape-pixel-ok` added;
  check-ui pin 83). The kill rule held: `slang/mod.rs` and `validate/mod.rs` have no changed line, the grid goldens are
  unchanged, and `export-artifacts --compare` on the lane's base listed all 80 exports equal. One doc-only edit outside
  the brief's files: the `is_pixel_output` comment in `crates/gust/src/reflect/mod.rs` now says the function is dead for
  MX until Breeze, because the never-written attachment is a host object the shader does not name. Opus merge after
  fixes: no must-fix; three should-fixes taken (the constants and their twin test, a typed refusal in place of a panic,
  extra gates); two nits deferred to mx.p1b. Gpu pin 343/33. Main: 24 of 24 gates green on `a3a8362`, the merge of
  mx.p1a and m7.av2 (2026-11-05 05:39 to 06:42Z, RTX 5090).
- M7 follow-up m7.av, m7.av2 (probe in the work area; wip `84808e8`, merge `a3a8362`): the gust-algo
  STATUS_ACCESS_VIOLATION. Probe m7.av: whole-binary runs with default threads crashed 3 times in 14 (alone 1 in 5,
  beside a second GPU process 1 in 5, with a helper holding 25.5 GiB of video memory 1 in 4); 98 one-test processes, the
  single-thread runs and the narrowing runs (the 11 heavy tests alone, the last three, the 49 GPU tests alone) had none.
  Every crash came after `gpu_v1_dispatch_counts_follow_the_schedules` finished, with the tie-ladder or
  deepest-schedules test still running, and no Windows event was logged. Slice m7.av2: the gust-algo GPU tests share two
  devices, taken in turns (`crates/gust-algo/src/tests/gpu.rs`), with no assertion changed. Its pre-registered loop had
  0 crashes in 30 whole-binary runs, default threads, no added load. The matched baseline was 1 in 5 (Fisher exact p ~
  0.14), and 3 in 14 over all default-thread conditions (p ~ 0.03); the 95 % upper bound on the per-run rate is 9.5 %.
  Runs also got faster (71 to 107 s, against 94 to 112 s), so timing changed as well and the cause is not established: a
  mitigation in gust-algo only. Four `with_limits` tests still create and drop 5 devices per run, and the loop sampled
  no video-memory peak (the shared devices now outlive the tests). mx.t4 saw the fault once in gust-wgpu's `textures`
  binary. Opus: merge as is. Main: 24 of 24 gates green on `a3a8362`, the merge of mx.p1a and m7.av2 (2026-11-05 05:39
  to 06:42Z, RTX 5090).
- MX mx.p1b (`2eacd00`, fix round `617af97`, merge `61c31f6`; D61, D93, D120, D127, MX plan §4.3 to §4.5): the `pixel`
  runtime in gust-wgpu. `crates/gust-wgpu/src/render.rs` encodes one render pass per call, in batch order: a fresh
  `R8Unorm` attachment 8192 wide and ceil(n / 8192) rows, `Load` and `Discard`, an empty write mask, and at most two
  scissored draws of the committed full-screen vertex module (`src/shapes/fullscreen.wgsl`; the plan's `.slang` copy was
  dropped). Outputs are fragment storage writes at the element's own index. The pipeline cache holds compute or render
  pipelines (`cache.rs`), and the graph's resource identity is now a `Raw` handle, so mx.t2's 4-byte identity buffer per
  texture is gone. Refusals, each before any pipeline is built: `ShapeUnsupported { kernel, shape, reason }` for a y or
  z other than 1 (also in a graph node), an indirect launch, the `dispatch_f32`/`dispatch_u32` family, the generated
  host and an adapter without fragment-stage storage writes; `ShapeIndependentOutput` for an `Independent` output;
  `DeviceLimit` for n over 2^26 (`pixel_max_elements`, decided before the binding sizes) and for
  `max_texture_dimension_2d`. The twin and the `cpu` target refuse the same calls (`cpu::Error::PixelExtent`,
  `PixelIndependentOutput`, the cap). Evidence on the RTX 5090 (`crates/gust-wgpu/tests/pixel.rs`, `shape_refusal.rs`,
  render.rs tests): pixel equals grid equals an independent host replay for u32, i32, f32, `Vec2` and `Vec4` outputs at
  28 sizes each (the squares up to 4097^2, 8192k + 1 and 1 000 003), with 0 differing words; the cap of 2^26 ran, and
  2^26 + 1 is refused end to end on adapter limits. Because robustBufferAccess2 drops out-of-bounds stores on these
  adapters, a stray fragment cannot show up as a wrong word, so a hand-written fragment counts launches with an atomic:
  the count equalled n at 1, 2, 8191 to 8194, 16385, 16386, 57345, 57346, 8192001, 8192002 and 2^26. A pixel call feeds
  a grid call inside one batch and inside a graph; two threads submit pixel batches on one device (a smoke test of the
  queue and the first compile: Option S shares no uniform). The fix round also fixed an mx.p1a macro bug: a module with
  an explicit-`workgroup_size` grid kernel beside a `pixel` kernel failed with E0707, because the emitter parsed sibling
  kernels' options under the pixel profile (`crates/gust-macros/src/slang/mod.rs`; two regression tests; goldens
  unchanged, no `Shape` branch added). Opus merge after fixes: two must-fixes (the cap test had only reached its skip
  branch; the stray-store test could not fail), the should-fixes and nits taken. Gpu pin 377/34. Main: 24 of 24 gates
  green on `61c31f6` (2026-11-05 08:17 to 09:27Z, RTX 5090).
- M7 follow-up m7.av3, m7.av4a to m7.av4c (probes in the work area, no repository change): the access violation is in
  the Vulkan validation layer. Debug builds of wgpu ask for `VK_LAYER_KHRONOS_validation`, and on this machine the
  loader inserts it from Vulkan SDK 1.4.357.0 in every run. A plain-wgpu release-build reproducer (m7.av3: devices
  created and dropped on 8 threads while 8 others dispatch, 27 runs of 90 s) never crashed; a release build loads no
  validation layer. The unchanged gust-wgpu `textures` test binary did: 3 of 25 whole-binary runs (m7.av4a). With a
  vectored exception handler added in a throwaway copy (m7.av4b), both crashes in 40 runs faulted at the same offset of
  `VkLayer_khronos_validation.dll`, a null read in the layer's bundled mimalloc (`_mi_subproc` from `mi_thread_init`, on
  the first allocation of a new `vvl::Queue::ThreadFunc` thread), symbolised with the SDK's own .pdb; no frame of GUST,
  wgpu, naga or the driver was on the stack. With one test thread, 0 of 25 runs crashed. With the layer disabled for the
  test process (`VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`), 0 of 40 runs crashed (m7.av4c; about 4 % by
  luck at the rate seen). The layer's .pdb records mimalloc v3.3.2; a null dereference in `_mi_subproc` is reported
  fixed in mimalloc 3.4.1 upstream, which is a likely match but was not tested here. The gust-algo crashes had no stack,
  so their cause is presumed the same, not shown. Gates keep validation on.
- MX mx.w (`7d281c4`, fix round `d3066ba`, merge `7dc0dc1`; D61, D120, D127, MX plan §11, X-S4): the `pixel` form in the
  browser. `examples/web-vector-add` adds `add_pixel` (`#[kernel(shape = pixel)]`) to the module of its grid kernel and
  runs both forms at n = 1000 (one row, one draw) and n = 8197 (two draws, the second row), each pixel output first
  filled with -1e9. The page prints `GUST_OK` only when the pixel result equals the host reference and the grid result
  bit for bit, and a separate `GUST_FAIL` when no fragment wrote the output. In headless Chrome 156.0.8078.74 on the
  RTX 5090 (`nvidia blackwell`) both forms matched at both sizes: X-S4 ends in outcome A, and `pixel` runs in the
  browser. Only this hardware adapter has run it; the software adapter has not. wgpu 30's WebGPU backend reports every
  downlevel capability, so on WebGPU gust-wgpu cannot refuse a pixel kernel for a missing fragment-stage storage
  capability; such an adapter would fail as a wrong result. The example's manifest is now schema 2 with a `shape` on
  every entry, and the xtask test compares the fixture-shared entries exactly in their schema-2 form. A manual mutation
  (the pixel expectation shifted by one, not committed) made `check-web` print `GUST_FAIL`. Opus merge after fixes: one
  must-fix (a comment said browsers refuse with `ShapeUnsupported`), three should-fixes and the nits taken. Main: 24 of
  24 gates green on `7dc0dc1` (2026-11-05 09:35 to 10:22Z, RTX 5090).
- MX mx.p1c (`a704768`, fix round `5b5f1ac`, merge `eb009af`; D61, D62, D127, D128, MX plan §4.5): the cross-shape
  difftest and the shape benchmark. `cargo xtask check-difftest --shapes grid,pixel` runs every program that the
  classifier accepts a second time as a `pixel` kernel (`tests/difftest/src/shapes.rs`); a refusal E0723 to E0728 files
  the program as grid-only and rebuilds the module once without it. Only `check-full` asks for `pixel`, and a source pin
  in the xtask tests holds that; the default `check-difftest` and `k` are unchanged. Integer and `bool` outputs of the
  two shapes must be equal bit for bit; an `f32` output of each shape is judged against the interpreter's interval, and
  a bit difference is reported as an observation (D128). On the RTX 5090, seeds 1-8: 128 pixel programs, 0 grid-only, 0
  mismatches, and 0 of the 36 `f32` programs differed in a bit. Of the examples' 22 kernels (21 here before mr.b5
  counted again), the classifier accepts 5 and files 17 as grid-only (E0723 4, E0725 3, E0726 2, E0728 8).
  `examples/shape-bench` times X-S1's kernels K1 to K4 and K3L (K3 without `fma`, which the dialect lacks) as grid at
  widths 64, 128 and 256 against pixel; its smoke mode in check-examples asserts that every element was written and is
  finite, and bit equality for the integer kernel, with no time threshold. The smoke found the same bits in all 10
  cells; an earlier K3 form (an unfused multiply-add chain after a division, not kept) differed by up to 4 ulp with
  identical emitted code, which led to D128. The full bench table waits for a quiet GPU (MX close). Opus merge after
  fixes: three must-fixes (the `f32` comparison, the docs, an observed-maximum bound), seven should-fixes and the nits
  taken. Main: 24 of 24 gates green on `eb009af` (2026-11-05 11:47 to 12:43Z, RTX 5090).
- MX mx.x (part 1 `d4a5b38`, fix round `edd91bc`; part 2 `56096dc`, fix round `f5afd6e`; merge `e769077`; MX plan §11):
  the MX docs. Part 1: README (shapes, textures and the `pixel` call), ARCHITECTURE (the stage-profile trait, the
  `pixel` profile and runtime, textures from the type to the bind group, the classifier and the shape pin) and the
  skills rust-gpu-ast-validation, rust-to-slang-lowering, spirv-validation and wgpu-runtime. Part 2: SEMANTICS (the
  `pixel` contract: the element index, the refusals, D128 for `f32`; the texture contract: formats, access, `sample` and
  its bounds) and VALIDATION (the MX check labels, the golden sets, the naga layout gate's 19 of 20 goldens,
  check-full's order), and the MX follow-ups drafted for NEXT_TASKS (filed at the close). Opus merge after fixes on each
  part: one must-fix each (a wrong error variant in a skill; D128 read as universal while two gates still compare `f32`
  bits, now explained), and the should-fixes and nits taken. Docs only; the check-full of the close records below covers
  the merge.
- MX close records (MX plan §11; 2026-11-05). (i) `cargo xtask check-full` passed at `27a461a` (13:14 to 13:46Z,
  RTX 5090, Vulkan, driver 610.88, Slang v2026.20, clean tree): its difftest ran seeds 1 to 8 with `--shapes grid,pixel`
  (128 programs as `pixel`, 0 grid-only, 0 mismatches), and the run covered check-precompiled E1 to E8, check-artifacts,
  check-release and check-web (`GUST_OK nvidia blackwell`, grid and pixel at 1000 and 8197 elements). The run before it,
  at `e769077`, failed in the textures binary with the validation layer's access violation under concurrency
  (NEXT_TASKS, m7.av); `verify --mode gpu`, check-cpu-target and `check-feature reflection` passed there (12:49 to
  12:58Z). (ii) Gpu (software: llvmpipe, LLVM 20.1.2, Mesa 25.2.8; shapes grid, pixel): gpu-software run 42061990249,
  the extended list, on `6d7e89a`: the Gpu suite with the `pixel` and texture tests, check-examples (texture-threshold,
  the shape-bench smoke) and the ledger rows KM-12 to KM-14. On this `Cpu` adapter the identity tests skip sizes above
  1000003 and the 2^26 cap test dispatches nothing (VALIDATION). Corrected at the MX close: this record first cited run
  42033705296, a pull-request-list run, as the extended suite. Records (i) verify-gpu, (iii) and (vi) ran on `e769077`,
  which differs from `27a461a` only in tests/difftest. (iii) The shape benchmark in a quiet window (14:19 to 14:31Z on
  `27a461a`, release, 5 runs of 20 timed dispatches): in render-pass time (GPU timestamps of the pass, not the whole
  call), `pixel` takes 0.7 to 2 times the best `grid` width for the memory-bound K1, K2 and K4, 6.5 to 11 times for K3
  at 1024 and above and 8.5 to 21 times for K3L; all 25 cells had the same bits in both shapes; two cells varied widely
  between runs (K1 at 4096, 47 %; K2 at 8192, 77 %). This is X-S1's result on the RTX 5090, so `pixel` is not a speed
  path for ALU-bound kernels on this GPU; the table is in the work area (plans/shapes/probe/bench). The 890M row is
  owner-run only. (iv) In the same window, m7.av's dense64 timing: 13.1 ms warm at 4194304 elements; after an idle gap
  of 100 ms to 1 s about 15 ms (one 1 s row 37 ms), and after 3 s 68 ms whether the CPU slept or spun, with the GPU back
  at P8 and the PCIe link down-trained. So the slow call follows the GPU's idle state, not the CPU or GUST (the first
  row of that table, with no gap, also read 68 ms, unexplained; NEXT_TASKS, m7.av follow-ups). The sanity check and its
  fixes followed (D129, mx.z).
- MR mr.b1 (`7ad4aec`, fix round `0c57b76`, merge `440892b`; MR plan S2): `#[non_exhaustive]` on `gust::reflect::Error`
  and `gust_wgpu::Error` (`gust::slang::Error` had it since mr.t2), and a doc comment on every variant of the three,
  checked against the sites that return it. The dead `gust_wgpu::Error::Slang` is removed: nothing constructed it, and a
  run-time compile failure arrives as `Error::Reflection` (whose doc now names its three sources: the run-time compile,
  the descriptor cross-check and precompiled-artifact verification). No workspace `match` needed a new arm; with doc
  lines, the attribute and whitespace stripped, the three files are token-identical to their base apart from the removed
  variant. Variant-level `#[non_exhaustive]` on struct variants is left for mr.b2 or later (gust-algo builds
  `DeviceLimit`). Opus merge after fixes: one must-fix (the `Slang` doc described a path that does not exist), six
  should-fixes and the nits taken. Main: 24 of 24 gates green on `440892b` (2026-11-05 14:48 to 15:43Z, RTX 5090).
- MR mr.b2 (`3c0f22a`, fix round `633dfb2`, R5 text `81cb9ad`, merge `16d62b3`; MR plan §2.3, S3): the doc sweep and the
  support surface. `gust` and `gust-wgpu` carry `#![warn(missing_docs)]`, and every public item is documented (143 in
  gust, 135 in gust-wgpu). The support surface of §2.3 is `#[doc(hidden)]` and named in each crate root (`profile`,
  `spirv`, `slang::probe`, `TargetProbe`, `render_wgpu_source` and the macro's support items); each crate root has a
  stability paragraph with one wording for the support surface (no stability promise; a patch release may change it).
  The macro gives its generated `DESCRIPTOR` and `MODULE_DESCRIPTOR` a doc comment, so a user crate with
  `#![deny(missing_docs)]` compiles over `#[gpu]` modules and functions: check-ui fixtures `missing-docs-mod-ok`,
  `missing-docs-fn-ok` (cpu profile) and `missing-docs-fn-dev-ok` (dev profile, with `gust-wgpu`, which covers the
  generated map functions); without the generated docs all three fail (13 errors, a local negative run). check-release's
  R5 keeps its check with `KNOWN_DOC_DEBT` empty, so new rustdoc debt fails it. Opus merge after fixes: no must-fix,
  eight should-fixes (doc accuracy, the generated items, a dev-profile fixture) and the nits taken. Main: 24 of 24 gates
  green on `16d62b3` (2026-11-05 15:47 to 16:51Z, RTX 5090).
- MX mx.z (`6ef61cc`, fix round `fdcbddd`, merge `a586386`; D129): the MX close fixes. VALIDATION rows for
  `tests/pixel.rs`, `tests/shape_refusal.rs` and the `render.rs` fragment-count test, each with what a `Cpu` (software)
  adapter skips; NEXT_TASKS items the close records settled are closed, and new items filed (no typed refusal in a
  browser for missing fragment storage writes, which only a compatibility-mode device could lack; D92's 8-bit filter
  weights against Vulkan's guaranteed 4 `subTexelPrecisionBits`; the 890M run as an owner item); gpu-software runs
  `check-difftest --seeds 1-2 --shapes grid,pixel` on both lists (`if: !cancelled()`; 25 s on lavapipe, 32 programs as
  `pixel`, 0 mismatches), with the xtask pin renamed to `only_full_and_one_gpu_software_step_ask_for_pixel`; README
  Stability and CHANGELOG state the evidence of `pixel` and textures. Opus merge after fixes: no must-fix, four
  should-fixes and the nits taken. Main: 25 of 25 gates green on `4467dc4` (2026-11-05 18:13 to 19:15Z, RTX 5090).
- MR mr.b3 (`30628bc`, fix round `7b5efa9`, merge `4467dc4`; D129, close ruling 6): the pre-publish API items of the MX
  sanity check. A `pixel` call creates its attachment inside an out-of-memory error scope, as `create_texture_2d` does,
  so running out of memory there is `Error::OutOfMemory` with the label `GUST pixel attachment` and its size, not a wgpu
  panic (the scope's tail is `device::finish_popped_scope`, shared with the other allocations); the texture view and the
  bind group stay outside it, since neither allocates device memory of the attachment's size. A unit test runs the same
  scope with `ErrorFilter::Validation` over a texture one row taller than `max_texture_dimension_2d` and checks that the
  `Err` names the attachment and that no error reached the uncaptured-error handler; the out-of-memory filter itself is
  not forced. `DeviceOptions` and `SamplerOptions` are `#[non_exhaustive]`, built from `Default` with `with_*` methods
  (`with_backends`, `with_adapter`, `with_limits`, `with_features`; `with_filter`, `with_address`), and `FeaturesPolicy`
  is `#[non_exhaustive]`; the workspace tests and ARCHITECTURE use the builder form. Gpu pin 379/34 (`da62379`: the
  lane's gates had no `verify-gpu`, and lavapipe run 42078500116 caught the missing bump). Opus merge after fixes: no
  must-fix, three should-fixes (the test, the `Poll` text, the `OutOfMemory` doc for textures) and the nits taken but
  one (the view and the bind group, above). Main: 25 of 25 gates green on `4467dc4` (2026-11-05 18:13 to 19:15Z,
  RTX 5090).
- MR mr.b4 (`31eed30`, fix round `35fab2e`, merge `71a184a`; D131): the wasm32 `Job` contract. `WgpuJob::status` on
  `wasm32` reads a flag that `Queue::on_submitted_work_done` sets, registered in `submit_batch` right after
  `queue.submit`; `wait` and `wait_timeout` return the new `Error::WouldBlock` there (the work was submitted and runs;
  the payload names the refusing method). The trait methods and the macro's `dispatch` stay on every target (D69's
  "already gated" was wrong; the builder stopped at it and the coordinator ruled the run-time error); the four blocking
  `HeadlessDevice` helpers are `cfg(not(target_arch = "wasm32"))`. The review found that the generated `map_gpu` and
  `map_gpu_on` waited through `dispatch` and so always failed in the browser: on `wasm32` they now submit and drop the
  job (native expansion unchanged, no golden moved). The web example checks, strictly, `Pending` right after submit,
  `Complete` after the awaited readback, `WouldBlock` from `wait`, `wait_timeout` and the generated `dispatch`, and
  `map_gpu` read back with `to_vec_async`: `check-web` passed, `Gpu (WebGPU, Chrome 156.0.8078.74, nvidia blackwell)`.
  Native behaviour and the Gpu pin (379/34) are unchanged; lavapipe runs 42101881893 and 42110495808 green. Opus merge
  after fixes: one must-fix (`map_gpu` on `wasm32`), five should-fixes and the nits taken; a re-check of the fix round
  approved it with three nits, filed. Main: 25 of 25 gates green on `acdd53f` (2026-11-05 22:08 to 23:18Z, RTX 5090;
  `gpu-verify` passed when rerun alone, 23:20 to 23:26Z, after the known validation-layer access violation in the
  `textures` binary).
- MR mr.n (`b237924`, fix round `614e167`, merge `acdd53f`; D130): the rename to the D48 package names, `gust-compute`,
  `gust-compute-macros` and `gust-compute-wgpu`, with the library names kept. `package =` on 99 dependency lines (87
  `tests/ui` manifests and the cpu-target, dual, precompiled, difftest-template and hello-gust fixtures), every `-p` in
  xtask, its tests and CI, `CRATE_RULES`, the user-facing strings (`cargo tree -e features -i gust-compute`, the
  `gust-compute-macros` and `gust-compute-wgpu` hints), six hand-edited expected lines (the package name in a build
  error), and six lockfiles with names only. check-precompiled E8 failed on the rename: the packaged `kernels` lock
  resolved against crates.io, which had matched the unrelated `gust` crate before and found only the `gust-compute`
  0.0.0 placeholder after; it now patches the two packages onto this checkout through `--config` (no committed path; a
  root it cannot quote is refused; a unit test pins the arguments), with `tests/precompiled/Cargo.lock` unchanged and no
  unused-patch warning. The lane's own run of the main gate list passed 25 of 25 on `b237924`; the fix round ran on the
  merge with mr.b4. Opus merge after fixes: no must-fix, three should-fixes (the VALIDATION E8 and check-arch rows,
  package names in rustdoc and prose, the task files' `-p` lines) and the nits taken. Main: 25 of 25 gates green on
  `acdd53f` (2026-11-05 22:08 to 23:18Z, RTX 5090; `gpu-verify` passed when rerun alone, 23:20 to 23:26Z, after the
  known validation-layer access violation in the `textures` binary).
- MR mr.b5 (`6f7e876`, fix round `e391378`, merge `ce614ca`; D133): the two pre-publish fixes of the Codex confirming
  pass (D132). CPU textures: `CpuDevice::write_texture` and `read_texture` refuse a texture of another device with
  `WrongDevice { parameter: "texture" }`, as gust-compute-wgpu's do, `read_texture` returns `Result<Vec<_>, Error>`, and
  a borrowed texture is `Borrowed { parameter: "texture" }` from both; the CPU error texts say "resource". No caller
  outside the unit tests; binding a foreign texture to a call was already refused. The classifier: the ledger of D133,
  with probes for every `syn` expression, statement and pattern variant, every callee form and the two helper rules it
  relies on, each refused probe asserting its code. Classification is unchanged: the difftest, seeds 1-8 with both
  shapes, still 128 pixel, 0 grid-only, 0 mismatches; the examples' 22 kernels give 5 pixel-compatible and 17 grid-only
  with the same codes and none `Unrecognised` (a temporary probe test, deleted; the MX record's 21 and 4 missed one
  pixel-compatible kernel). Opus merge after fixes: one must-fix (free-function callees were outside the ledger), four
  should-fixes and the nits taken; a re-check of the fix round approved it with four nits, filed. Main: 25 of 25 gates
  green on `ce614ca` (2026-11-05 23:37Z to 2026-11-06 00:40Z, RTX 5090; `check-fast` passed when rerun alone, 00:41 to
  00:48Z, after the known validation-layer access violation in the `textures` binary).
- MR mr.p, the doc pass (`48c6526`, fix round `4737b0e`, re-check nits `f94c054`, merge `4c5bff9`; D134): the
  pre-publish documentation of D132 and the release checklist against the tree. The linear-filtering bound's assumption
  (D134) is stated once, in the same words, in README, SEMANTICS section 4, CHANGELOG's known limits and
  `gust::cpu::texture`. `HeadlessDevice::new`, `DeviceOptions::default` and `default_device` each name the other and the
  three limits that differ. The mr.b4 follow-ups: `WouldBlock`'s Display and docs say any submitted work will run and
  list `dispatch_extent`; the generated `map_gpu` and `map_gpu_on` docs say that on `wasm32` they return before the work
  finishes. RELEASE.md: step 1 uses `cargo xtask doctor` (a checkout has no installed cargo-gust) and states what it and
  the full check-release need (the pin's `slangc` first on `PATH`; the helper through `GUST_SLANG_REFLECT` or built into
  `target/slang-reflect`, which needs `SLANG_SDK` or `VULKAN_SDK` and a C++ compiler); step 5 sets no environment
  variable (the bridge of D109 finds the managed toolchain) and is the first run of that path, since R7 sets
  `GUST_SLANG_REFLECT`; README's setup text and `slang locate` follow. No behaviour change but the Display text, which
  no test pins; lavapipe runs 42133430222 and 42137303659 green. Opus merge after fixes: one must-fix (SEMANTICS still
  credited the 8-bit precision to WebGPU), four should-fixes and the nits taken; the re-check read both specifications
  and approved it with three nits, which the coordinator applied. Main: 25 of 25 gates green on `4c5bff9` (2026-11-06
  00:51 to 01:52Z, RTX 5090).
- MR mr.f, the fixes of the MR sanity check (`934676a`, fix rounds `cffa720` and `8af0647`, merge `1d8610b`; D135).
  CHANGELOG's heading is `## 0.1.0` and README reads as published. Every public enum but the closed sets is
  `#[non_exhaustive]` (twelve enums gained it: `AdapterChoice`, `LimitsPolicy`,
  `reflect::{Stage, ReflectedResourceKind, ReflectedTypeKind}`, `slang::Target`,
  `abi::{ScalarKind, TypeLayoutKind, TextureAccess, ShaderLayoutRules}`, `artifact::Format`, `JobStatus`); the macro's
  generated code only constructs them, and its one pattern is a `matches!`. Fallback arms in cargo-gust (an error where
  data flows in, `unreachable!` behind the `=0.1.0` pins) and gust-compute-wgpu (`padding_free` false, an unknown result
  shape refused, an in-memory cache key variant); `Format::ALL` and `ScalarKind::ALL` with tests in gust-compute and
  cargo-gust restore the completeness check the attribute removed. `cargo gust doctor`'s PATH warning names
  `cargo xtask` only as a tool of a GUST checkout. The 19 rustdoc links to native-only items resolve on both targets (an
  `_async` sibling, or a `cfg_attr` doc pair). Drift: README's environment-variable and examples lines, VALIDATION's
  SPIR-V line, SEMANTICS' browser refusal (expected, not observed) and `-p gust-compute`, the reflection recovery text
  (`cargo gust slang build-helper`), and old package names in rustdoc, log text and the generated-host header. Gates: 16
  local gates green on both rounds (including the wasm32 and native docs.rs-shaped builds at 0 gust-compute-wgpu
  warnings and MSRV); lavapipe runs 42157740335, 42164364077 and 42168641836 green. Opus merge, no must-fix; the
  should-fix (the doctor line) and seven nits taken, the eighth (an empty `[lints]` in gust-macros) filed, since a guard
  test bars it; the re-check: merge, its should-fix (`texture::Format::ALL` becomes a slice, so a new format changes no
  public type) and two wording nits taken in round 2. Main: 25 gate steps green on `1d8610b` (2026-11-06 05:44 to
  06:46Z, RTX 5090, no rerun).
- MR probe mr.e (no repository edits; D135): the first-user path and the docs.rs build rehearsed from the packaged
  crates at `4a4e8e1`. The README's crates.io blocks with fresh resolution (wgpu 30.0.1), no Slang variable and no
  `slangc` on `PATH`: `cargo gust slang install`, `build-helper`, `cargo run --features gust-dev`, `cargo gust build`,
  the release build and run and `cargo gust doctor`, all rc 0 on the RTX 5090 (the recipe prints no adapter name;
  filed). The managed cache went to `~/.cargo/gust` and was removed afterwards. docs.rs-shaped nightly builds
  (`DOCS_RS=1`, the metadata features, `--cfg docsrs`, offline): the three libraries on linux-gnu with 0 warnings;
  gust-compute-wgpu on `wasm32-unknown-unknown` with 19 unresolved-link warnings (fixed in mr.f); D68 does not fire; no
  source reads `docsrs`.
- M7 README map table, published run (replaces m7.a2's provisional smoke table). Two runs of the out-of-repository probe
  `plans/m7/probe/a2`, built against `ea258f9` in a detached worktree, in a quiet window (2026-11-04 13:48 to 13:50Z,
  55 s each, RTX 5090, Vulkan; 3 warm-up and 11 timed rounds, 5 at 4.2e6). Map over host slices is 0.61 to 0.93 of a
  hand-written dispatch; the resident chain is 1.3 to 2.9 times faster than host slices; a cheap function never beats
  the CPU up to 4.2e6; 64 multiply-adds win from 1e4 to 1e6 and 256 from 6.6e4 to 4.2e6. The 64-multiply-add cell at
  4.2e6 takes three to four times linear scaling from 1e6 in both runs and the smoke run (70.0 and 50.9 ms), so it is
  not faster there; unexplained, in NEXT_TASKS. The probe's header still prints the smoke run's base and "shared
  machine" lines; the README states the true conditions.
- M7 m7.x (the close-out, plan §11; three lanes). m7.x1 (`097c75f`, fix round `e61d237`, merge `3ebd368`): a
  feature area `dispatch` (the `par_for!` and multi-dimensional dispatch tests of gust-macros, gust and gust-wgpu,
  one counted call per filter so a filter that matches nothing fails); `map` also runs m7.c's tests; `check-changed`
  routes the gust-algo, map, dispatch and workgroup paths to their areas, and a changed fixture no rule claims keeps
  the `macro` suite; `check-artifacts` exports and `spirv-val`s the M7 goldens (map, grid, workgroup, par_for, reduce,
  scan, compact: 66 SPIR-V files) beside vectors and uniforms, splitting multi-kernel goldens at their marker lines
  (the earlier sets' bytes unchanged); `check-cpu-target` requires the `par_for!` line once; E0709 says GUST defines
  no CPU strategy for barrier kernels (D118). m7.x2 (`88abaeb`, rounds `eaa5647` and `66b682a`, merge `1e648b7`):
  README (the stale lockstep text, the check-feature table, the Current subset for M6 and M7), VALIDATION (command
  rows, the Cpu tier, the M7 runtime tests, the gust-algo Gpu step, the M7 golden sets) and REPO_STRUCTURE (cargo-gust
  and gust-algo, "publishable"). m7.x3 (`4811714`, round `b7b8bdd`, `f0551f3`, merge `07b851d`): a
  SEMANTICS map section and the M7 entries of "What is proven" (compaction's count and indices measured, as they run
  v1), ARCHITECTURE for map, the generator, `par_for!`, `GpuVec` and the pool, `cpu::parallel` and gust-algo,
  NEXT_TASKS (stale items, storage barriers and the decoupled look-back), three skills, and gust-algo's variant docs
  without "provisional" (D121). Opus merge after fixes on each; every must-fix taken. Main: 23 of 23 gates green on
  `1e648b7` (m7.x1 and m7.x2; 2026-11-04 13:01 to 13:41Z), and on the m7.x3 merge (docs and doc comments only) fmt,
  gust-algo lib and doc tests, docs-check, lint and clippy.
- M7 m7.c (`60e5aa3`, fix round `7f1b37f`, measured doc `3499303`, merge `88ac459`; D119): parallel CPU
  execution for generator-stamped map kernels. Under the `cpu` target the map generator emits a chunk body that
  `gust::cpu::parallel::run_map_chunked` runs over disjoint output chunks with `std::thread::scope` (safe Rust, std
  only); reads see the whole input, the only write is the invocation's own index, and the lowest failing index is
  re-raised. Threads: `GUST_CPU_THREADS`, else the machine's parallelism, bounded by
  `max(1, n / ELEMENTS_PER_THREAD)` with the factor 2^18 (one on wasm32); a bad value or a failed spawn is a
  `cpu::Error`. User kernels, `par_for!` and the test twin stay sequential, and under `cpu` `map_on` is concrete over
  `&CpuDevice`. Measured in a quiet window (2026-11-04 10:42 to 10:47Z, 24 logical cores, release): the k = 64
  polynomial 16.1x at 1e8 and 11.7x at 1e7 over one thread, vector-add 1.84x at 1e8; every result bit-exact with one
  thread. The first run's vector-add cell at 2^17 read 0.85 with one thread on both sides (the same code path), so
  the pre-registered re-run of the vector-add cells (21 runs) decided: 0.95 to 1.02 from 2^16 to 1e6, inside D119's
  10 %. Opus merge after fixes, no must-fix; five should-fixes and three nits taken. Main: 22 of 23 gates green on
  `88ac459` (2026-11-04 10:54 to 11:33Z); the Gpu suite step died with the gust-algo STATUS_ACCESS_VIOLATION a second
  time (NEXT_TASKS), while gpu-verify in the same run passed the same tests (297 pinned, gust-algo 49), and gust-algo's
  tests alone passed three times after (one thread, then default threads twice).
- MX X-S3 (spike, D120; MX plan §1a): the pixel output model ended in outcome B on both adapters. Grid, render
  targets with `LoadOp::Load` and two copies, and fragment storage writes gave bit-identical outputs on the RTX 5090
  and the Radeon 890M; the storage-write call took 0.12 to 0.76 of the render-target call on the 5090 and 0.45 to 0.81
  on the 890M (the copies are the difference). mx.p1b therefore writes pixel outputs as storage, with no transient
  textures. The validation cases came out as expected (a wider `read_write` format is refused, the layout variants
  hold, an overwritten uniform shows only its last value). Quiet window 2026-11-04 10:28 to 10:30Z. Hand-written WGSL
  in a scratch crate in the work area; nothing in the repository changed.
- M7 m7.r4, the Radeon 890M half (D121; plan §5.5): v0 against v1 on the integrated Radeon 890M (Vulkan,
  `AdapterChoice::LowPower`), 3 interleaved runs of 200 repetitions at 1e6, 1e7 and 1e8, release build of the m7.r4
  harness on `6d64fec` in a throwaway worktree. v1 was faster everywhere: the median over runs of v0 / v1 was 1.20
  to 1.26, 1.13 to 1.17 and 1.01 to 1.03 for the nine reductions at the three sizes (worst single run 1.003,
  `max_f32` at 1e8) and 9.8 to 67 for the two scans; every scan total matched the host sum for both variants. The
  reductions at 1e8 are memory-bound on shared system memory, so the brief's 10 % margin is not met there; D121 keeps
  v1 for all six operations. Quiet window 2026-11-04 10:30 to 10:42Z. The variant docs drop "provisional" in m7.x.
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

### Done slices of M7 (summary)

Map, residency, gust-algo, shared memory and barriers, multi-dimensional dispatch, `par_for!` and parallel CPU maps.
Tier: Gpu (RTX 5090, Vulkan) for the code slices (docs slices ran partial gates), lavapipe in CI, the Radeon 890M
for m7.r4 (D121) and the S7.6 limits.
Per-slice entries above carry the mutations and gate runs. Decisions D84, D87, D96, D97, D99 to D104, D116, D118,
D119, D121 to D125.
- Runtime: m7.a0 `bcb4da4` (D84; device options, pool, `GpuVec`, `default_device`, fallible buffers; S7.6 outcome A,
  limits measured on the 5090 and 890M, out-of-memory path unit-test only; D123). m7.rb `893d959` (D97; blocking
  readbacks wait for their own completion, 8 errors in 2 903 iterations before, 0 in 3 153 after).
- Dialect: m7.w `4797013` (D87; `wrapping_*`, GUST-E0325). m7.g1 `1516c94` (D100, D101; derived ids, `Workgroup`,
  `workgroup_barrier`). m7.d `853aade` (D102; `Extent`, multi-dimensional dispatch). m7.p `910ee1c` (D103;
  `par_for!`). m7.lv `9d47f67` (D116; uniform and vector layouts in the export record; retires D104).
  m6.z2 `a496968` (D96) and m6.u2f `bee193a` (D104) ran inside M7.
- Map and residency: m7.a1 `5609536` (D99, D122; `#[gpu]` on a scalar `fn` generates `map`, `map_on`, `map_gpu`,
  `map_gpu_on`, `map_async`, `map_on_async`). m7.a2 `44299e5` (examples `map-chain` and `web-vector-add`'s
  `map_async` path). README map table, published run on `ea258f9` (quiet window 2026-11-04 13:48 to 13:50Z): map
  over host slices 0.61 to 0.93 of a hand-written dispatch's time, resident chains 1.3 to 2.9x faster than host
  slices, no break-even for a cheap function up to 4.2e6 elements.
- gust-algo (D124): m7.r1 `b5d2c68` reduce v0. m7.r2 `09203ae` scan v0 and `sum_wrapping`. m7.r3 `8027169`
  compaction v0 (generic gather deferred to M9). m7.r4 `d6b425f` (lavapipe fix `9dda10d`) v1 with shared memory;
  890M half in D121 (v0/v1 1.01 to 1.26 for reductions, 9.8 to 67 for scans); v1 shipped for all six operations.
  Unpublished (D110).
- Component pool (D125): m7.t13 `b88bfb0` (GPU active count to indirect dispatch, re-typed from the stash design;
  no `active_count` check-feature entry) and m7.t14 `f96aff1` (retirement and compaction); the stash was dropped by
  the owner on 2026-11-03 (backup patches in the work area). Only the RTX 5090 has a recorded run of the T13 and T14
  GPU tests.
- Parallel CPU: m7.c `88ac459` (D119; chunked `run_map_chunked`; k = 64 polynomial 16.1x at 1e8 over one thread,
  vector-add 1.84x; bit-exact).
- Spikes: S7.4 lockstep twin, outcome C (D118; 102 to 104 s at 1e6, m7.g2 dropped, barrier kernels keep
  `cpu = "unsupported"`). S7.5 chunked CPU map, outcome A (D119). S7.6 limits (inside m7.a0). S7.7 (inside m7.r1).
  Pre-plan spikes m7.s1 to m7.s3 fed the plan.
- Tidy and close-out: m7.t1 `1278b61`, m7.f1 `96051c8` (Linux "Text file busy" retry), m7.t2 `6b0d59a`; m7.x1
  `3ebd368` (feature area `dispatch`, `check-artifacts` over the M7 goldens), m7.x2 `1e648b7` (README, VALIDATION,
  REPO_STRUCTURE), m7.x3 `07b851d` (SEMANTICS, ARCHITECTURE, NEXT_TASKS, skills, gust-algo docs).
- Records at the close (plan §11.2), on `dd74689`: (i) Gpu RTX 5090 `verify --mode gpu`, pin 297/31; (ii) Gpu
  (software): lavapipe `gpu-software`, GitHub Actions run 41871538616 on `db837b9`, green, llvmpipe (device type Cpu,
  Mesa 25.2.8); (iii) Cpu target `check-cpu-target`; (iv) web: `check-web` builds `web-vector-add` and runs it in
  headless Chrome, which includes the `map_async` path (`GUST_OK nvidia blackwell`, Chrome 156, passed); plan
  §11.2(iv) asked only for a compile-only web record, so the browser run goes beyond it; (v) precompiled:
  `check-precompiled` and `check-artifacts`. `check-cpu` is twin evidence. The CI push list does not run
  `check-examples`, so lavapipe never ran the T13 and T14 GPU tests; it also skips the deepest gust-algo schedules
  (128 MiB), and says so only on stderr.
- Open (also in NEXT_TASKS): the write-set race checker; storage barriers; decoupled look-back scan; generic gather
  (M9); the thread-free lockstep twin idea; `as_entire_binding()` in `generated_host.rs`; `gust-wgpu` `Error` is not
  `#[non_exhaustive]`; a `_wrapping` scan form; the 890M, DX12 and Metal are untried for T13 and T14; the browser run
  covers `map_async` on one adapter (RTX 5090, Chrome) only.

### Done slices of M1 to M4 (summary)

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
- Open (filed in NEXT_TASKS at the M4 close; D71 records the manifest field names): `gust::Job`'s wasm-aware contract of
  D69 (done in mr.b4, D131); the wasm32 clippy ban on `std::thread::sleep` and `Instant::now` covers `gust-wgpu` only
  (m4.z), not `gust` or `web-vector-add`; a plain `cargo install cargo-gust` needs `GUST_PROFILE=dev`; CI's
  `precompiled` job runs `--check-only` plus a NoAdapter run of the release consumer, and E1, E3, E8 and E7b stay local;
  a GPU-side no-op compare-exchange is not detected by a GPU test and there is no signed ledger case (m4.0); S2 did not
  measure the wasm32-to-web inference (m4.g2 builds under an explicit `GUST_PROFILE=web`).
- Closed by m4.z: helper freshness judged by a source-hash stamp and the Slang tag, and a byte check of all five copies
  of the committed artifact set (the m4.r known gaps), a `cargo gust doctor` warning for a `PATH` `slangc` of another
  version, and the wasm32 clippy guard of `gust-wgpu`.

T13 landed in M7 (m7.t13 and m7.t14, D125). The owner dropped the git stash `WIP T13 active count deferred` on
2026-11-03; backup patches are in the work area, outside the repository.

```text
owner: Claude Code coordinator (monitored autonomy)
claim: roadmap M7 (m7.p merged) after the M6 close (D86); see the roadmap's Now
and next
next_focused_check: cargo xtask check-fast
full_check_needed_before_commit: no
```

## Current baseline

- Branch: `main`
- Last complete compiler feature: MX (execution shapes and textures: the stage-profile trait, the `pixel` shape,
  textures and samplers, the classifier and the shape pin; D129); before it M7 (map generator, `GpuVec` residency,
  `par_for!`, shared memory and barriers, multi-dimensional dispatch, wrapping arithmetic, gust-algo; D126)
- Last full validation: `cargo xtask check-full` passed at `2f96e69` (M8 close records, 2026-11-09, 05:11 to 05:53Z,
  NVIDIA GeForce RTX 5090 / Vulkan, driver 610.88, Slang v2026.21; clean tree), through the shared cargo slot wrapper:
  workspace tests, the release-only tests required by name, every example test with `--include-ignored`, `check-dual` in
  full mode, the SPIR-V artifacts through `spirv-val`, and `check-web` with the browser run, and since mx.p1c the
  difftest with `--shapes grid,pixel`. The step output, including the `GUST_OK` line, lives in the local
  `.ai/VALIDATION.json` record of that run, which is not committed. The runs before it were at `dd74689` (M7 close,
  2026-11-04, 15:58 to 16:33Z), `f452a37` (M6 close, 2026-11-02, 19:32 to 20:12Z, driver 610.88, Slang v2026.20),
  `7dd9c3d` (M5 close after m5.f1, 2026-11-02, 13:49 to 14:01Z), `68c6f40` (m5.c2, 2026-11-02, one untracked log file),
  `27b342b` (M4 close after m4.i) and `c964222` (2026-11-02, before the M4 sanity fixes), then `5971c12` (M3 close,
  2026-11-01, clean tree), then `96e4b9d` (2026-11-01, before the M3 sanity fixes), then `634a50e` (M2 close,
  2026-10-24) and the one before that (M1) at `a6cbb8f`, together with the Slangc and Reflect suites. The local
  `.ai/VALIDATION.json` record stamps the commit and dirty state and contains no machine paths.
- DX19 is closed: full validation, two independent reviews, and fixes (`027eef7`,
  `08a4843`, `3296885`).
- CI (`unit` job: fmt, clippy, Unit suite; `msrv` job on Rust 1.90) passes on main;
  one push on 2026-10-23 (`10b2e06`) was red and was fixed in `3cdc607`.

## Current contracts

- Task acceptance criteria: [tasks/README.md](tasks/README.md)
- T10 atomics: [tasks/T10-atomics.md](tasks/T10-atomics.md)
- T11 Result lowering: [tasks/T11-result.md](tasks/T11-result.md)
- T12 Result match: [tasks/T12-result-match.md](tasks/T12-result-match.md)
- T13 active count (landed in M7, m7.t13 and m7.t14): [tasks/T13-active-count.md](tasks/T13-active-count.md)
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
