# Current status

## Active: roadmap M3 (2026-10-24)

The roadmap agreed with the owner is in [ROADMAP.md](../ROADMAP.md). M0 closed on
2026-10-23 after a two-reviewer sanity check (D28): every M0 item is done; M0.15
(the CI `unit` job is a required check on `main`) followed on 2026-11-01. M1 (diagnostics) closed on
2026-10-23 (D33): M1.0 to M1.6 are done (per-item evidence in the roadmap), after a
sanity check by Fable and Opus (both "proceed with fixes") and the full validation
below. M2 (typed host API) closed on 2026-10-24 (D37) after a sanity check by Fable and
Codex (both "proceed with fixes") and `check-full` at `634a50e`. M3 (dual test builds): plan
adopted 2026-10-24; D38 narrows D22's integer guarantee to a defined domain (the owner
confirmed it in D39: edge-case differences are documented limitations, not to be matched);
spike S1 passed (decision A); m3.a1 (ledger), m3.a2 (contract text), m3.b (`gust::cpu`),
m3.c (macro twins), m3.c2 (check-cpu), m3.d (atomics, fixture twins), m3.e1 (four example
twins), m3.e2 (graph example twins), m3.f1 (`check-dual --check-only`), m3.f2 (check-cfg
lints, dual mains, full `check-dual`, CI step) and m3.g (gates that cannot pass by skipping)
done; the close-out docs and D40, D41 are in; `check-full` on a clean tree and the M3 sanity
check next.

- M3 m3.g (`95c5514`): a read-only audit (plan Revision 7) found no gate that skips through
  an early return, a missing tool or a CI condition, but six holes, now closed: `check-full`
  and `verify --mode examples` ran the examples with `--ignored` only (now
  `--include-ignored`, so the twin-versus-host and guard tests run there too); a cargo test
  call that ran zero tests passed (a test-count guard now fails it, and also fails a failed or,
  under `--include-ignored`, an ignored test, for check-cpu, the examples, the release `--lib`
  call and the filtered check-feature calls); the release call now also runs the `gust::cpu`
  id-range test and requires both release-only tests by name; six examples' max-error folds
  turned a NaN into 0 (now propagated, with a NaN unit test each); the check-ui fixtures are
  pinned by name; and the 30 `gust::cpu` unit tests, which need neither slangc nor a GPU, are
  a Unit step, so CI runs them. Known limitation, in NEXT_TASKS: the guard counts per call,
  not per test. Evidence: fmt, clippy, xtask tests, check-cpu, Unit, check-ui, check-fast,
  check-examples (GPU, all eight examples with `--include-ignored`), docs-check, lint; the
  release call by hand (both release-only tests ok); five mutations killed.

- M3 m3.f2 (`98acabc`): the root manifest declares `cfg(gust_dual)` through
  `[workspace.lints.rust] unexpected_cfgs`, and every crate except the proc-macro crate
  gust-macros opts in with `[lints] workspace = true`. The gust-macros guard now allows exactly
  that root table and still rejects any lints table in its own manifest, however spaced or
  quoted. Under `--cfg gust_dual`, vector-add and polynomial run the twin on `main`'s inputs and
  print "CPU twin: matches (N elements, max ULP k)", asserting against the derived bound;
  normal builds print nothing new. `check-dual` gains fmt and clippy for `tests/dual`, clippy
  with the cfg on the two examples, and a full mode (GPU) that requires the line with the cfg
  and its absence without; `check-full` runs the full mode, and CI's `unit` job ends with
  `check-dual --check-only` (local: 9 s warm, 74 s for the first dual build; on the CI runner
  82 s, inside R5's 5-minute budget). Evidence: fmt,
  clippy, xtask and gust-macros tests, check-dual in both modes, check-ui, both examples' tests
  with and without `--ignored` (GPU), Unit suite, check-fast, docs-check, lint, and the same
  again on the merge plus check-cpu; four mutations killed.

- M3 m3.e2 (`00127a4`): every kernel of staged-graph and component-pool has CPU-twin
  assertions; none opts out. Each twin is fed the GPU's inputs to that kernel (read back where
  an earlier kernel produced them), and each test carries the label "kernel-level twin
  evidence; graph and indirect dispatch GPU-only": the graph, its dependency edges, the
  indirect launch and the pool growth run on the GPU alone. Integer and struct outputs are
  exact, including `prepare_dispatch`'s indirect arguments, which also match a host formula.
  Float bounds are derived per kernel (staged-graph `transform` 193u, `summarize` total 3886u;
  component-pool `integrate` 19u to 73u per field); every observed maximum on the RTX 5090 is
  inside its bound and none is used as a tolerance. A guard asserts that no host intermediate
  lies within the bound of staged-graph's f32 threshold. Both examples add non-ignored CPU-only
  tests to `check-cpu`. Evidence: fmt, clippy, both examples' tests with and without
  `--ignored` (GPU), check-cpu, check-examples, check-artifacts, xtask tests, Unit suite,
  check-fast, docs-check, lint, and the same test, check-cpu and Unit steps again on the merge;
  five mutations killed.

- M3 m3.e1 (`41b4d0e`): every kernel with a typed call in signal-pipeline, particle-step,
  typed-pipeline and atomic-counter has CPU-twin assertions; none opts out. The twin equals
  the host reference exactly. Against the GPU, integer and struct outputs are exact and float
  outputs stay within absolute bounds derived per kernel by the SEMANTICS.md §4 procedure and
  written in each test's doc comment (signal-pipeline `transform` 900u, u = 2^-24; particle-step
  `step` 3u to 98u per field; typed-pipeline `calibrate` up to 172u, `classify` 63u, `finalize`
  14u); every observed maximum on the RTX 5090 is inside its bound and none is used as a
  tolerance. typed-pipeline feeds each kernel's twin the GPU's intermediate buffers, and
  `classify`'s severity is compared after an asserted guard that no score lies within the
  bound of an f32 threshold. Each example adds non-ignored CPU-only tests to `check-cpu`.
  SEMANTICS.md gains a derived-kernels table and lists the exact example kernels as proven.
  Evidence: fmt, clippy, the four examples' tests with and without `--ignored` (GPU),
  check-cpu, check-examples, check-artifacts (spirv-val green, same `.spv` names), xtask
  tests, Unit suite, check-fast, docs-check, lint, and clippy, xtask tests, check-cpu,
  check-dual and Unit again on the merge; five mutations killed.

- M3 m3.f1 (`5c5a9ad`): `cargo xtask check-dual --check-only` and the `tests/dual` fixture
  workspace (`kernel-lib` with one `#[gust::gpu]` kernel; `consumer` with a `cfg(test)` twin
  test and a production binary). It refuses a preset `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS` or
  `CARGO_BUILD_RUSTFLAGS`, builds the workspace with `--cfg gust_dual` through
  `CARGO_ENCODED_RUSTFLAGS` in `<target>/dual`, and checks that without the cfg a dependent's
  own `cfg(test)` (E1) and a production binary (E3) get no twin of a library kernel (exactly one
  E0277 each, matched per target by `ui::judge`'s new target parameter), while with it the twin
  test compiles and passes (E2). E2 also shows `gust::cpu::overflow_checks!()` reports the
  calling crate's per-package overflow mode (Revision 5). S3: first dual build 76 s, no-change
  rerun 11 s; a no-change main build afterwards compiles nothing; the dual target directory is
  about 8 GB. Not yet in a suite or CI (m3.f2). Not covered: root fmt and CI clippy do not reach
  `tests/dual` (nor `tests/ui`). Evidence: fmt, clippy (workspace and `tests/dual` with and
  without the cfg), xtask tests (183), check-dual, check-ui, Unit suite, check-fast, docs-check,
  lint, and check-dual, xtask tests and Unit again on the merge with m3.d; four mutations killed.

- M3 m3.d (`0f2ff27`): the five `gust::runtime` atomics (u32 and i32) run for real on the CPU
  twin, sequentially in ascending invocation order (a valid ordering of the GPU semantics);
  `atomic_add` wraps in every build and never panics. The semantics, numeric, option, result,
  loops, struct_assignment and atomics fixture tests dispatch the twin with the same call as
  the GPU and assert twin == host reference and twin == GPU wherever the spec defines one
  result; for contended exchange and compare-exchange the GPU may land anywhere in the spec
  set (including 999 for the weak compare-exchange, KM-09) and the twin is pinned to its own
  value. The known-mismatch ledger has a twin column keyed on `gust::cpu::overflow_checks!()`
  (debug: Checked, release: Wrapping). A zero-storage kernel's twin runs and panics as plain
  Rust does. SEMANTICS.md gains the D39 limitation for arithmetic inside `gust` and a list of
  proven statements. Evidence: fmt, clippy, gust lib, gpu-semantics, the eight GPU fixture
  tests (RTX 5090), the ledger in release, Unit suite, check-fast, docs-check, lint; four
  mutations killed (one shows the ledger twin column fails on a reordered twin).
- M3 m3.c2 (`35eb2a5`): `cargo xtask check-cpu` runs the `--locked`, GPU-free tests of the
  vector-add and polynomial examples, so their macro-emitted CPU twins are checked against the
  host references; it is a step of the Unit suite and so of the CI `unit` job. check-ui gains
  `code: none` for macro errors without a rustc code (it needs a `message:` line and still
  pins file and line), with the fixtures `kernel-cpu-bad-value` (E0709) and
  `kernel-cpu-duplicate` (E0710). Cost: 7 to 8 s warm and 67 s cold locally; on the CI runner
  the Unit step went from 12 s to 41 s (plan R5 budget: five minutes). Evidence: fmt, clippy, xtask tests
  (161), check-ui, check-cpu, Unit suite, check-fast, docs-check, lint; three mutations
  killed. Opus keep, no must-fix.
- M3 m3.c (`732c774`): for every kernel with a typed call, the `#[gpu]` macro emits
  `#[cfg(any(test, gust_dual))] impl Run<K, N> for gust::cpu::CpuDevice`, which runs the
  shadow body as written. It matches m3.b's hand-written impl token for token. Uniform kernels
  and `#[kernel(cpu = "unsupported")]` get none; E0709 and E0710 reject bad `cpu` options.
  `gust::cpu::read`/`write` refuse a binding of the other access. The check-ui fixture
  `typed-call-no-cpu-twin` proves the twin is absent from a build without `cfg(test)`.
  vector-add's twin equals the host reference and the GPU exactly. polynomial's twin equals
  `run_cpu` bit for bit and stays within the derived bound `48 * 2^-24` of the GPU (S4,
  RTX 5090: max abs 2.38e-7 observed). CPU-only twin tests run without a GPU. Compile time
  is unchanged (`-p gust-wgpu --no-run`, warm, 18 s). Not covered: E0709/E0710 have unit
  tests but no check-ui fixture (macro errors carry no rustc code; m3.c2 added the mode and both fixtures), and
  the zero-storage twin is only expansion-tested. MSRV is checked by CI only. Evidence: fmt,
  clippy, gust-macros, gust lib and doctests, rustdoc `-D warnings`, check-ui, bless
  (unchanged), Unit suite, check-fast, docs-check, both examples with `--ignored`,
  check-examples; three mutations killed.

- M3 m3.b (`a7dcb95`): `gust::cpu`, the CPU backend the macro twin will use (m3.c, planned).
  It is `#![forbid(unsafe_code)]` with no new dependencies, and the erased-binding downcast is
  confined to `read`/`write`. Public items: `CpuDevice`, `CpuBuffer<T>`, `CpuBinding`,
  `CpuJob`, a non-exhaustive `Error`, and `ulps`. The `#[doc(hidden)]` macro support is
  `validate`, `read`, `write`, `invocations` and `FromGlobalId`. `validate` follows
  gust-wgpu's check order (aliasing last) and returns `DeviceLimit` above 2^32 invocations.
  Overflow mode: `overflow_checks()` reports `gust`'s own build, and `overflow_checks!()`
  reports the calling crate's (plan Revision 5; the cross-crate proof is due in m3.f1). The
  `dispatch.rs` tests run on `gust::cpu` instead of a mock. Evidence: fmt, clippy, `gust` lib
  and doctests (debug and release), check-feature core, check-arch, Unit suite, check-fast,
  docs-check, rustdoc `-D warnings`; 8 mutations killed.
- M3 m3.a2 (`cc88b1a`, docs only): `docs/SEMANTICS.md` is the executable-semantics
  contract. It states the scope, the labels (spec-allowed, measured, proven; nothing is
  proven yet), the defined domain with D22's narrowing (D38, D39), WGSL float rules quoted by
  section (15.7.2 to 15.7.6), a per-operation bound table with input domains, reassociation
  as a limitation, and the tolerance method. Worked polynomial bound: 46u, asserted at 48u in
  m3.c. Also covered: memory preconditions (a sequential twin hides races), the atomics rule
  (the weak compare-exchange may fail spuriously, so the atomics test is narrower than the
  spec; m3.d revisits it), and the ledger as limitations. The review corrected the plan's
  float bounds (correctly rounded allows either neighbour; 2.5 ULP is up to 5u). No test
  exercises float rounding yet, and none covers `f32 as u32` in range. Evidence: docs-check,
  lint, ledger test green.
- M3 m3.a1 (`96ec90b`): known-mismatch ledger. `tests/fixtures/mismatch.rs` (13 kernels,
  blessed golden) and `crates/gust-wgpu/tests/mismatch.rs` pin, per row, the GPU outcome and
  the plain-Rust outcome; the Rust column follows the run-time `overflow-checks` mode (debug
  run: checked; `--release` run: wrapping). GPU values that WGSL defines are enforced: division
  and remainder by zero, `MIN / -1`, shift-amount masking, wrapping, and finite out-of-range
  float-to-int casts (WGSL 15.7.6: 2147483520, 4294967040, `-1e20 -> i32::MIN`). NaN,
  infinities, subnormals (flushed on the RTX 5090), `NaN != NaN` (false on the GPU) and
  out-of-bounds access are recorded as unspecified one-adapter observations, not enforced
  (D39). `docs/SEMANTICS.md` lists them as limitations; an id guard keeps doc and test in step;
  `verify --mode full` adds a `--release` ledger pass. Spike S2 record: 33 in-domain controls
  agree; six cast outcomes differed from the body's pre-registration and are listed. Evidence:
  fmt, clippy, ledger debug and release, docs-check, lint, Unit suite, check-fast green.
- M3 spike S1 (pre-registered, decision A): a minimal `gust::cpu` (module-level
  `#![forbid(unsafe_code)]`, the erased-binding downcast confined to `gust::cpu::read`
  and `write`) plus a macro-emitted `#[cfg(any(test, gust_dual))] impl Run<K, N> for
  gust::cpu::CpuDevice` runs vector-add's shadow body on the CPU. The twin equals the host
  reference and the GPU exactly at counts 1, 63, 64, 65 and 257 (RTX 5090); no kill rule
  fired, no manifest changed, clippy and check-fast green. The spike stays on a local branch
  (`dc2ad87`, not merged); m3.b and m3.c rebuild it with tests.
- M2 sanity-check fixes (`2d1880b`, D37): indirect launches check everything except the
  workgroup count on both the validation and the execution path, so `with_limits` with a
  lower `max_compute_workgroups_per_dimension` no longer rejects them through a placeholder
  count (new graph test fails before the fix; three mutations caught); the `DeviceLimit`
  doc says `validate` checks every node before any compile; a kernel with a
  `Uniform<T>`/`ConstantBuffer<T>` parameter has no generated typed call and is rejected
  by the runtime (`UnsupportedParameter`), now said in every doc that claimed "every
  kernel"; wgpu zeroes over-limit indirect counts only under its default instance flags.
  Evidence: fmt, clippy, gust-wgpu tests, check-ui, docs-check, lint, Unit suite, check-fast
  green after two review rounds (Opus: keep after fixes).
- M2 close-out (code and user docs, `009f86a`): `gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION`
  is pinned to `wgpu::Limits::default()` by a unit test and to the opened device by a GPU test, and replaces
  the examples' seven copies of 65 535. One runtime StorageV1 and size assertion now guards typed, pool and
  indirect-argument buffers (unit tests per clause, a GPU test per allocation path, mutations caught).
  `StagedGraph::dependency_indices` replaces the examples' copies. README, ARCHITECTURE, EXECUTION_GRAPH and
  VALIDATION describe the M2 code; decisions D34 (typed call surface, alias contract), D35 (derived edges)
  and D36 (launch limits and what stays unchecked). Evidence: fmt, clippy, gust-wgpu tests, docs-check, lint,
  Unit suite and check-fast green after the review fix round (Opus: keep after fixes); `check-full`
  passed at `634a50e` (2026-10-24T05:44:34Z, RTX 5090).
- M2 m2.f2: particle-step and typed-pipeline batch typed calls; staged-graph and
  component-pool build every frame with `StagedGraph::write/run/run_indirect/read`, so all
  edges are derived, and they assert the derived sets (`[[], [0], [0, 1], [2]]` and
  `[[], [0], [0, 1], [1]]`; the pool readback depends on `prepare`, the last writer of
  `active`). Rejection tests keep the explicit API. Release runs of all four pass on the
  RTX 5090; swapping `calibration` and `readings` in typed-pipeline is E0271 (`2a4f3f6`).
- M2 m2.f1: vector-add, atomic-counter, polynomial and signal-pipeline dispatch through the
  generated typed calls (`<kernel>::dispatch`, `BufferDispatch::from(&call)` batches,
  `Independent(&buffer)`); host references, tolerances and export code are unchanged, and
  the release runs of polynomial and signal-pipeline pass on the RTX 5090. The one-shot
  "GPU end-to-end" figure in polynomial and signal-pipeline now waits twice (dispatch, then
  readback) instead of once; no before/after number was measured. Release workgroup counts
  are pinned by unit tests against a copied 65 535 (a shared constant follows in the M2
  close-out) (`ef52d86`).
- M2 m2.g: a hand-written `KernelDescriptor` whose Slang uses an undeclared identifier
  fails in slangc with the error on its marked Rust line, through `slang::compile`, both
  reflect paths and the native reflection helper (`#line`, not the table lookup); with an
  empty line table the Slang line is reported and no Rust source. The fixture never
  passes through `#[gpu]`, so later validator work cannot remove it (M1 risk 5, D33).
  Known difference: without `// @rust kernel:` markers, `slang::compile` names no kernel,
  while `reflect` and the helper add `originating Rust kernel` (`c6571d5`).
- M2 m2.e: `StagedGraph` derives dependencies from typed calls. `write`, `run`,
  `run_indirect` and `read` record the latest earlier writer of every buffer a node
  touches (RAW, WAW) and, for a writing node, every reader since that writer (WAR); the
  set is conservative and still validated, and execution stays in insertion order. Mixed
  graphs work. Measured (spike S4): staged-graph and component-pool derive exactly the
  pre-registered sets, with GPU readbacks equal to the declared graphs. `validate` now
  checks device limits for every node before anything compiles (closing the m2.b gap)
  and rejects an indirect node whose arguments buffer is also bound read-write (measured:
  a wgpu usage-conflict panic). Indirect counts are known only on the GPU: wgpu zeroes
  over-limit counts silently and checks the real device limits, so a limit lowered with
  `with_limits` does not apply to them. `GraphReport` counts derived and declared edges and
  is `#[non_exhaustive]` (`229bfbf`).
- M2 m2.d2: const layout assertions. Every `#[gpu]` struct asserts, in one const block,
  each field at its padding-free offset (the message names the field), no tail padding,
  alignment 4 and `LAYOUT.is_storage_v1()`; every kernel asserts per storage parameter
  that the element layout is StorageV1 and matches its Rust size, alignment and stride.
  `cargo check` now reports what only the runtime asserts caught. The plan's premise that
  no accepted struct could fail was false: a `#[gpu]` struct with a hand-written outside
  `GpuPod` field of alignment 8 passes the validator; it is now E0080 at compile time
  (fixtures `layout-outside-field`, `layout-outside-element`). check-ui follows a primary
  span's macro expansion to the outermost site inside the case (const-eval panics point
  into core). Open for m2.f2: pool buffers (`gust-wgpu/src/pool.rs`) skip the runtime
  layout assert that `create_typed_buffer` has (`adeb7be`).
- M2 m2.d1: `#[gpu]` generates `call`, `submit` and `dispatch` on every kernel handle:
  one argument per storage parameter in declaration order, element and device pinned by
  `Resource` bounds (each spanned on its parameter), access from the parameter. A
  per-kernel const assert pins the storage count against `DESCRIPTOR`, so `cargo check`
  catches a mismatch. Kernels with a `Uniform` parameter get no typed call; kernels with
  no storage parameters take the device type (`call::<D>(..)`). Measured: component-pool's
  five-parameter kernel infers without turbofish (spike S3); a wrong element type is
  E0271 on the user's line; a device without `Run` shows `Run`'s `on_unimplemented` text.
  A non-GUST argument gives two E0277 errors, so it has no fixture. Known limits: the
  `__Gust` prefix of generated generics is not reserved; the `Into` and `Run` notes still
  point at `#[gust::gpu]` (`a83a96e`).
- M2 m2.d0: `check-ui` reads `cargo check --message-format json` and judges only the
  case's own diagnostics; a failing case must produce exactly one error. `expected.txt`
  pins it by keys: `code`, `file` and `line` (exact, on one primary span), `message`
  (substring of the message or a child note) and `label` (substring of that span's label,
  where `on_unimplemented`'s label lands, measured on real runs). Editing the label text in
  `gust` now fails check-ui. Open for m2.d1: spans inside macro-generated code may need
  `expansion` followed, and one error per case may be tight for two-error fixtures
  (`a8b074f`).
- M2 m2.c: `gust-wgpu` implements the typed surface. `HeadlessDevice` is a
  `TypedDevice` with a `Run<K, N>` for every kernel, and `GpuBuffer<T>` is a `Resource`;
  every typed call goes through the existing persistent-binding validation (layout,
  device, access, current length) before any wgpu object. One buffer bound to two
  parameters is now rejected as `Error::AliasedBinding` when either binding is
  read-write, on typed, untyped and graph paths (measured before: read-only plus
  read-write panicked in wgpu, read-write twice ran with the last writer winning; no
  example binds a buffer twice, and check-examples passes). Binding indices are checked
  against `max_bindings_per_bind_group` on every dispatch path. 19 GPU tests in
  `tests/typed.rs`. The `_` arm on the `#[non_exhaustive]` `LengthRule` binds any
  rule this crate does not know with the strictest one (`Shared`), by design (decided at
  M2 close, D37); the indirect-arguments alias is checked since m2.e (`119ece3`).
- M2 m2.a: typed call surface in `gust` (`dispatch.rs`). `TypedDevice` extends
  `Device` (unchanged) with a `Binding<'r>` type; `Resource`, `Arg`, `Independent`,
  `LengthRule` and `Call` describe one typed launch, and `Run<K, N>` submits it.
  `Call::new` takes the descriptor from the kernel type and asserts the binding count
  at monomorphisation; order, access, element and device are checked by generated
  calls (types) and by each backend at submit (contract). Only test mocks implement it
  so far (a recording mock and a CPU-shaped `RefCell` mock); `gust-wgpu` follows in
  m2.c, the macro in m2.d1 (`3060072`).
- M2 m2.b: launch-limit checks. Every dispatch path checks the kernel and its bindings
  against the opened device's limits before any compile or cache lookup and returns
  `Error::DeviceLimit`; before, wgpu panicked (measured: workgroup size 512 at pipeline
  creation, 65 536 workgroups at dispatch). The opened RTX 5090 device carries wgpu's
  default limits, not the adapter's. `HeadlessDevice::with_limits` only lowers limits
  (five fields at the time; six since m2.c added the binding index). Known gaps: the
  dispatch emitted by `render_wgpu_source` and buffers above `max_buffer_size` are not
  checked (D36); the multi-node graph gap of this slice was closed by m2.e, whose
  `validate` runs the limit pass for every node before any compile (`151ef56`).

- M1.6: sanity-check fixes. Clippy `disallowed-methods`/`disallowed-macros` replace the
  hand-written uncoded-error scanner, and the exception is confined to `diagnostics.rs`
  by a test; a code counts as tested only when an assertion passed for it; `#line`
  paths follow the macro's path rule, including Unicode letters (slangc echoes UTF-8
  names); `resolver = "3"` with unchanged lockfiles. Unit, Slangc and Reflect passed on
  the merged tree (`1f961fc`).

- M1.3: the compile-time copy carries `#line` directives on `slang::compile*` and both
  reflection paths; exported `.slang` never does. The native helper honours them, and
  the sidecar records `line_directives`. 10 of 10 deliberate failures land on the right
  Rust line on WGSL and SPIR-V; WGSL/SPIR-V bytes are identical with and without
  directives for three fixtures. Columns stay Slang columns; only the first error is
  attributed (`8fcf944`, `7007827`).
- M1.4: a rejected `#[gpu]` module is returned beside `compile_error!` with `#[kernel]`
  removed, kernel fns renamed and private as on the success path, and the prelude
  imported; `mod m;` gets the error alone. Measured on `vector-add` with one rejection
  inside the referenced kernel (rustc 1.98): one error before and after, the same
  message; the editor benefit is not measured (`e01b82c`, `f507ab4`).
- M1.5: every `#[gpu]` rejection carries a stable code (`[GUST-E0401] message`, D31);
  111 codes, one `docs/DIAGNOSTICS.md` section each with its reachability. Every code is
  asserted by a test or listed as unreachable through `#[gpu]` (seven codes). Guards:
  unique codes, doc and enum agree, and a source scan for uncoded `syn` errors whose
  limits the doc states (`df07122`, `6e195bc`, `f20d313`).
- M1 carry-overs: check-ui matches each expectation within one rustc diagnostic; the
  `on_unimplemented` text no longer claims exclusivity; the vcvars capture refuses
  inherited multi-line values that could forge a variable (`fb257c9`).
- Bug fix: the `typed-pipeline` and `staged-graph` release runs dispatched 65 536
  workgroups against wgpu's default limit of 65 535 and panicked; release sizes are now
  `1 << 21` with a test per example, and both release runs complete (`5c34e22`).

T13 remains deferred by owner; its work in progress is the git stash
`WIP T13 active count deferred`, not the working tree. Do not apply, continue,
validate, commit, or drop it without explicit direction.

```text
owner: Claude Code coordinator (monitored autonomy)
claim: roadmap M3 planning (see the roadmap's Now and next)
next_focused_check: cargo xtask check-fast
full_check_needed_before_commit: no
```

## Current baseline

- Branch: `main`
- Last complete compiler feature: T12, statement-only `match Result<T, T>`
- Last full validation: `cargo xtask check-full` passed from 2026-10-24T05:40:27Z to
  05:44:34Z on NVIDIA GeForce RTX 5090 / Vulkan at `634a50e` (clean tree), through the
  shared cargo slot wrapper: workspace tests, release-mode slang tests, all 8 examples
  (debug builds), and 13 SPIR-V artifacts through `spirv-val`. Release-mode example runs
  are per-slice evidence (m2.f1, m2.f2 probe logs), not part of this record. The previous
  full run (M1) was at `a6cbb8f`, together with the Slangc and Reflect suites. The local `.ai/VALIDATION.json` record
  stamps the commit and dirty state and contains no machine paths.
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
  ENGINE_NORTH_STAR, NEXT_TASKS, HANDOFF, and the Copilot builder agent profile now
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
[history/STATUS-2026-09-09.md](history/STATUS-2026-09-09.md).
