# Changelog

All notable changes to the published packages `gust-compute`, `gust-compute-macros`, `gust-compute-wgpu` and
`cargo-gust` are listed here. The four packages always share one version. GUST follows SemVer for 0.x: a change that
breaks existing code bumps the minor version, and the promise is described in the
[README](README.md#stability).

## Unreleased

- A stage of a `gpu_batch!` refuses a `bool` element (m14.s2e part 1): a `Stage<bool>` it reads or writes, or a
  `bool` result type, is `GUST-E1004` with `u32` as the fix. `map`, `par_for!` and a stand-alone `gpu_task!` are
  unchanged.
- Tests: a `map` over a struct with `Vec4` and `Mat4` fields and over a struct that holds it now runs on the GPU and
  the CPU twin, every field compared (m14.mapw2; Radeon 890M, RTX 5090 pending). No behaviour change.
- `map` takes and returns user structs (m14.mapw1, D221): write `#[gpu_fn]` on a function inside the
  `#[gpu_module]` module that declares the struct and call `module::f::map(..)` (and `map_on`, `map_gpu`,
  `map_async`, ..) with slices, arrays, `Broadcast` values or resident buffers of the struct. The GUST-E0716 error
  for `#[gpu_fn]` inside a module is lifted.
- A sealed batch of stages with one frame in flight (m14.s2a2). New in `gust-compute`: `gpu_batch!("name", on(&device);
  in_flight(1); |b| { .. })` returns a `gust::Batch` and the handles of its block; a frame is `begin().await`, `write`,
  `set`, `ticket`, `submit()` (one queue submission on the GPU device) and `ticket.await`. New public names:
  `gust::{Batch, FrameRun, Submitted, Ticket}` and, in `gust::batch`, `Builder`, `Stage`, `Param`, `StageInfo`,
  `Report`, `D1`, `D2`, `Rank`, `IntoShape`, `ParamType`, `BoxFuture`, `Error` (with `NotInput` and `FrameLive`) and the
  native `Blocking` (`dispatch()`, whose every `read` is a submission of its own). `in_flight` above 1 is GUST-E1005;
  GUST-E1006 refuses a body or a place a batch stage does not take. Runs on `CpuDevice` and `HeadlessDevice`.

- Capture one GPU call and replay chosen invocations on the CPU twin (g2.cap). Every kernel with a twin gets a generated
  `capture(&gpu, &call)` (test builds and `--cfg gust_dual`) returning a `gust::capture::Captured` with
  `run_element(id)`, `run_until(id)` and `run_all()` (a call the device refuses is not captured);
  new `DeviceOptions::capture` / `with_capture` lets read-only buffers be copied.
  A replay reproduces the twin's order, not the GPU's schedule.

- A batch can be recorded into the caller's command encoder (m14.encb, D251). New in `gust-compute-wgpu`:
  `HeadlessDevice::encode_batch(&mut encoder, &dispatches)` records without submitting and
  `EncodedJob::submitted(&index)` returns the job after the caller's `queue.submit`; `submit_batch_read` and
  `encode_batch_read` run a batch and read one buffer back in one submission, with `uncleared_buffer` and `PendingRead`;
  `gust_submissions()` counts GUST's own submissions. Radeon 890M only; the RTX 5090 run is pending.

- A renderer can share GUST's device (m14.devshare.1, D247, D248). New in `gust-compute-wgpu`:
  `HeadlessDevice::adopt(&adapter, device, queue, capabilities)` wraps a device the caller opened in its own
  `wgpu::Instance`; `DeviceOptions::device_request(&adapter)` returns the `DeviceRequest` (features and limits) that
  GUST's own open asks for, to merge your own into; `set_default_device(handle)` makes a shared handle the default
  device; `gust_wgpu::wgpu` re-exports the wgpu behind the handles. New errors: `MissingFeatures`, `AdapterMismatch` and
  `DefaultDeviceCapability`. The text of `DefaultDeviceOpened` and of the launch refusal of a missing capability changed
  to name `set_default_device` and `adopt`. Radeon 890M only; the RTX 5090 run is pending.

- A `gpu_task!` submit is one queue submission on the GPU device (m14.taskone, D251): the dispatch, the clear of a
  recycled output and the read-back copy share one command buffer, through a new provided method of the `MapHost`
  device trait that `HeadlessDevice` overrides (an empty captured slice on a recycled pool buffer is still cleared
  in a submission of its own, in a task that is then refused). Radeon 890M only; the RTX 5090 run is pending.

- Added `gpu_task!`, a stage written at the call site and run as a free task: a 1-D, 2-D or in-place stage over host
  slices and scalars, returned as a `gust::task::Task` with `submit()` (an asynchronous read-back) and `dispatch()`.
  The example `task-basics` and the diagnostics GUST-E1001, E1002, E1003, E1007 and E1008 go with it (m14.s2a1).

- The pinned Slang is now 2026.22.1 (`cargo gust slang install` fetches it; run `cargo gust build` again for a committed
  artifact set). Two emitted differences: float and signed-integer vector negation is `-a` instead of `0 - a`, and a
  constant out-of-range float-to-integer cast folds at compile time to another value than before (m13.pin2221).
- Renamed the attribute: `#[gpu]` on an inline module is now `#[gpu_module]` and `#[gpu]` on a free function (the
  element-wise map) is now `#[gpu_fn]`, with the same arguments (`route`, `crate`) and behaviour. `gust::gpu` is no
  longer exported and there is no alias. `#[gpu_module]` on a function, or `#[gpu_fn]` on a module or any other item,
  is `GUST-E0702` and the message names the attribute that fits. The generated Slang is unchanged (m14.rename,
  D237, D243).
- Added `#[pad]` for a struct of a `#[gpu_module]` module that a uniform reaches: `#[gpu_module]` inserts the `f32`
  padding fields `_pad0`, `_pad1`, ... of its constant-buffer layout and generates `new(<your fields>)`; a misplaced
  `#[pad]` is the new `GUST-E0219`, and an unmarked struct keeps `GUST-E0209` (m14.pad, D239).

## 0.2.0

Nothing here has been published, and no release is planned yet: D226 abandoned the 0.1 and 0.2 plans, and releases
wait for maturity (the hold of D189 and D199 stays). Older sentences below that say 0.2.0 will be the first public
release, or wait for it to reach crates.io, record the plan of their time and are superseded by D226.
A 0.1.0 was prepared on the branch `release/0.1` and was never
published: it has no tag and is not on crates.io, so there is nothing to upgrade from, and a reader of a future first
release can skip "Changed, and may break code written for the unpublished 0.1.0 line" and "Compatibility with the
unpublished 0.1.0 line" below. Those two parts are kept for this repository's own fixtures and for anyone who built
from `release/0.1`. This list describes `main` as of the M10 close: what the milestones M8, M9 and M10 merged. No
release commit exists yet, and the semver report below will be rerun on the commit that is published.
Against `release/0.1` the tree has seven listed breaks, B1 to B7: B1 to B4 came with M8 and M9, and the API audit of
m10.api1 added B5 to B7 (below; they break no released version, because none exists). B1, below, in two
parts (the descriptor constructor and setters, and the `GpuKernel` bound), is the last of its kind: the descriptor is
`#[non_exhaustive]` from now on. B2 is a function name in a `#[gpu]` module that is now reserved (the 21 subgroup
operations). B3 is that recursion is refused (`GUST-E0120`). B4 is that more names are renamed in the emitted Slang,
which changes the content key of a precompiled artifact of a module that uses one (`Tuple`, `makeTuple`, `this`,
`interface`, `extension` and the 46 names of Slang's standard interfaces). The public API that M9 adds is additive
(D165 and the route types). The `semver` job of CI ran on `f613b32` (run 43535559023): cargo-semver-checks 0.51.0
against `release/0.1`, `--release-type patch`, the `dev` profile and the default features, 229 checks for each of
`gust-compute` and `gust-compute-wgpu` (31 skipped). Its one finding is the `#[non_exhaustive]` on `KernelDescriptor`
in `gust-compute` (B1, already listed); for `gust-compute-wgpu` it found nothing ("no semver update required"). After
m10.api2a.1 the same job (run 43660662918) lists, for `gust-compute`, `struct_marked_non_exhaustive` on
`KernelDescriptor` (B1), `ModuleDescriptor`, `Artifact` and `Verified` (B6); for `gust-compute-wgpu`,
`struct_marked_non_exhaustive` on the six record structs of B5, `inherent_method_now_doc_hidden` on
`HeadlessDevice::check_same_device` and `pub_module_level_const_now_doc_hidden` on `EXACT_CLASS_BYTES` (B7). It does
not judge the macro language, and it does not check items that were already `#[doc(hidden)]`, so the list is not a
proof.

### Known issues

- **An NVIDIA driver defect gives wrong values for one array pattern on the default WGSL route** (D168, D169; not a
  GUST bug: the generated Slang, slangc's WGSL and naga's SPIR-V are correct, and the AMD Radeon 890M computes Rust's
  values). Measured on one device only: an NVIDIA GeForce RTX 5090 through Vulkan, driver 610.88. In a kernel, a
  function-scope array has one or more elements read as 0 after an element at a dynamic index is loaded and written
  back with a value the driver can prove unchanged (`a[k] -= 0`, `a[k] ^= 0`, `a[k] <<= 0`, `a[k] &= !0`,
  `a[k] = a[k]`, or an operand that folds to 0 through `%` or a call). Arrays of length 3, 4, 5, 8, 16 and 17 whose
  elements all hold one value were measured to fail, and any array after an earlier dynamic store and a copy; length 2
  never failed, a zero read from buffer data never triggered it, and programs that write a real change are not
  affected. A stored array in a struct field is affected the same way; arrays in storage buffers were not measured.
  GUST does not refuse or rewrite the form (D39). It was not run on another NVIDIA architecture, so whether the defect
  appears on another NVIDIA GPU or driver is unknown. The GPU test `driver_sentinel`
  (`crates/gust-wgpu/tests/driver_sentinel.rs`) expects the defect's values on that device, so a driver change that
  fixes it fails the test, which is the cue to re-measure. Details: [SEMANTICS](docs/SEMANTICS.md), section 7.
  - **The opt-in cure is the `spirv-route` feature** (D172, listed under Added): with it on and the route taken, the
    defect did not reproduce on the RTX 5090. The route needs run-time compilation and Vulkan, so it is no cure for a
    shipped build: a precompiled artifact keeps its WGSL, and a build with the `native` or `web` profile (which has no
    run-time compiler) stays on the WGSL route and meets the defect where the pattern occurs. The route has its own
    documented result differences (SEMANTICS section 15).
- **An NVIDIA driver defect ends some float-compare loops at the wrong iteration** (m11.ld; observed on one device
  only: an NVIDIA GeForce RTX 5090 through Vulkan, driver 610.88). In a loop with a constant bound whose exit is a
  `break` on a float comparison, a `<` break due after iteration 128 is taken at 128, and an `==` break due at
  iteration 128 or later never runs. The SPIR-V route (`spirv-route`) shows the same values; the AMD Radeon 890M
  computes Rust's values. GUST does not refuse or rewrite the form (D39); a done flag or an integer exit avoids it,
  and GUST's own `rem_euclid` helper uses a flag since m11.fm. The GPU test `driver_sentinel` expects these values
  on that device, so a driver change fails it. Details: [SEMANTICS](docs/SEMANTICS.md), section 7.

### Added

- The old value of `atomic_add` in kernels (m11.fetch, D203): `let old = atomic_add(&mut c[i], v);` binds the value
  the cell held before this invocation's add (`u32`, `i32`; only as the whole initializer of a `let`). The other
  atomics, and `atomic_add` in any other position, stay `GUST-E0606`.
- `HeadlessDevice::wgpu_device` and `HeadlessDevice::wgpu_queue` (m11.handoff): the device and queue behind a
  `HeadlessDevice`, an advanced escape hatch like `GpuBuffer::as_wgpu_buffer`, so a caller's own wgpu pass can read a
  GUST buffer on the same queue without a readback (example `examples/buffer-handoff`).
- User names spelled like Slang's texture and sampler types (m11.tx, D204): a struct, global or kernel parameter
  named `RWTexture2D`, `WTexture2D` or `SamplerState`, and a global or kernel parameter named `Texture2D`, is
  renamed in the emitted Slang
  (`__gust_r_SamplerState`); before, slangc refused the module. A program that uses one of those names gets new
  emitted text and a new artifact content key. A struct named `Texture2D` or `Sampler` stays `GUST-E0736`.
- Module-level `const` items in a `#[gpu]` module (m11.g3, D205): `const NAME: T = <literal>;` with `T` one of `u32`,
  `i32`, `f32` and `bool` is shared by every kernel, helper, method and closure of the module and by the host; it
  lowers to a `static const` line and nothing is evaluated. Other forms, and a const whose name is also a type,
  method, field or prelude name of the module or is used as a binding pattern, are the new `GUST-E0127`.
- Constant expressions in module-level `const` items (m11.cexpr, D210): an initializer may combine literals and
  earlier consts of the module (`u32` and `i32` arithmetic, bit operators and shifts; `f32` `+ - * /` and negation;
  `bool` `! && ||`). rustc and slangc compile the same expression; nothing is evaluated by the macro. GPU
  checks: Radeon 890M and RTX 5090.
- Module-level array constants (m12.carr): `const NAME: [T; N] = [e0, e1, ..];` with `T` one of `u32`, `i32`, `f32`,
  `bool` and `N` a literal from 1 to 256 lowers to one `static const` array line per kernel and is read as `NAME[i]` or
  `NAME.len()`; a whole-array use, a store, a nested array and other element types are `GUST-E0127`.
- Differential test format 14 (m11.dt2): the generated programs gain scalar math methods and module constants,
  and a batch is 17 programs per seed (136 for the 8 default seeds). Under `--shapes grid,pixel` 104 of the 136
  also run as `pixel`; the other 32 stay `grid`-only for the reasons the classifier reports (an unrecognised construct,
  a write at another index, a use of an output).
- Launches past the per-dimension workgroup limit run (m11.g4b, D208): a `grid` launch whose workgroup count on an
  axis exceeds `max_compute_workgroups_per_dimension` is split by the host into pieces (at most
  `MAX_SPLIT_DISPATCHES`, 4096) in one compute pass, and every invocation runs once with the ids of an unsplit
  launch, so `id` stays the global index and kernels need no edit. Graphs, the deprecated `dispatch_f32`/`_u32`,
  pixel kernels, hand-built descriptors without a macro-made variant and builds without run-time compilation
  (precompiled-only, the browser) keep `DeviceLimit`; the 2^32 invocation cap is unchanged.
- Split launches in precompiled-only builds (m11.psplit-a, D209, D214): `cargo gust build` and `bless` also write a
  precompiled pair for the split form of each `grid` kernel, keyed on its own Slang text and named in the manifest
  by an optional `split_of` (manifest schema 4; kernel records schema 5). A launch past the per-dimension limit then
  runs that artifact after the usual identity check; with neither it nor a run-time compiler it is still refused
  with `DeviceLimit`. A crate with no `grid` GPU kernel keeps its manifest bytes; the committed artifact sets were
  re-blessed with split entries added and no existing artifact changed, and older schemas still load. The browser
  half is m11.web (below). The split launch test ran on the Radeon 890M and, in the full Gpu suite, on the RTX 5090;
  the precompiled half has no RTX 5090 record.
- budget-kit rows (m11.kit3): W4 binning with the scatter (count, scan, and placement by the old value of
  `atomic_add`, checked per cell against a host counting sort) and W7, a sustained composite step with its GPU
  median, p95 and maximum and two host-side costs (the blocking timed batch, the return of `submit_batch`).
- Example `mc-histogram` (m11.mc): hash-sampled points deposited into a fixed-point histogram with `atomic_add`
  (bit-equal on the host reference, the twin and the GPU), a density through the map API, and diffusion steps with
  `rem_euclid` and `mul_add` checked within a bound derived from the GPU model of SEMANTICS; its large run is a
  launch the host splits.
- Example `field-relax` (m11.field): three scalar fields relax on one wrapping grid, each with its own spread and
  keep factor from one `Uniform` struct, checked within a per-step bound derived from the GPU model of SEMANTICS
  (one step and sixteen steps), plus an exact `u32` channel that conserves its sum (bit-equal on the host
  reference, the twin and the GPU). Its grid constants are constant expressions of the side. Radeon 890M and RTX 5090.
- Example `cell-bins` (m11.bins): a reusable binning of elements by grid cell (count with `atomic_add`, an
  exclusive scan into a kept buffer with `gust_algo::scan::exclusive_into`, scatter by the old value of a cursor).
  The set of elements of each cell is exact; their order inside a cell is unspecified on the GPU. Checked per cell
  as sets against a host counting sort, on the twin and the GPU. Radeon 890M and RTX 5090.
- Example `walker-steer` (m11.steer): walkers sense a `u32` field ahead in three headings, turn toward the largest
  (a stated tie rule, and a hash-chosen turn with a set probability), move one wrapping cell and deposit with
  `atomic_add`; a cell kernel then relaxes the field by a shift. All integer: host reference, twin and GPU are
  bit-equal after every step, with a proven `u32` bound. Radeon 890M and RTX 5090.
- Example `walker-transport` (m11.transport): one composite step of the three pieces above on one grid (bins of
  walkers by cell, a walk step with a crowding rule that reads the bin counts, an exact spread of the field and a
  cleared deposit; four stages, the bins stage dispatching clear, count, scan and scatter), 24 steps, all integer:
  host reference, twin and GPU are bit-equal after every step (field, walkers and the per-cell bin counts; the walkers
  grouped by cell are equal per cell as sets after sorting), with a proven `u32` bound. Radeon 890M and RTX 5090.
- Example `walker-life` (m11.life): walkers that move, lose energy, retire and spawn children, kept dense in their
  buffers by an exclusive scan of per-walker counts and a compaction kernel (no atomic append, so the order of
  spawned walkers does not depend on scheduling; the population is clamped at the capacity by slot). Host
  reference, twin and GPU are bit-equal after every step. Radeon 890M and RTX 5090.
- The portable floor covers every kernel form (m11.floor): a crate with `limits = "portable"` now gets GUST-E0746
  for the kernel of a `#[gpu]` function and of a `par_for!` call too, not only for `#[kernel]` items of a `#[gpu]`
  module. Crates without the setting are unchanged.
- budget-kit row W8 (m11.kit4): a whole step (diffuse, step, binning with the scatter, a deposit folded into the
  field, retire, a scan of alive flags, compaction and spawn) as one timed scope, checked against a host model
  including a step where the capacity clamps; the note lists the storage buffers each kernel binds.
- budget-kit cost split (m11.cost): W8's note gives the GPU time per batch and per kernel, and rows W8a-W8d
  time the step without bins, with coarser bins, with one fused particle kernel and with the lifecycle every 4th
  step.
- One error for a refused `#[gpu]` module (m11.diag): when the macro refuses a module, the structs it hands back
  keep the `Clone`, `Copy` and `GpuPod` impls a successful expansion adds (where no impl was written by hand and every
  field is a known `Copy` type), so host code that uploads one of them no longer adds a misleading E0277 "not a GUST
  GPU POD type" next to the GUST error. `GUST-E0811` now shows a typed binding (`let a: [u32; 3] = [1, 2, 3];`).
- `if`/`else` as the whole initializer of a `let` (m12.ifx): `let w: u32 = if c { a } else if d { b } else { e };`
  (typed, or untyped when a branch tail shows the type) lowers to a local declared before the `if` and assigned in
  each branch, so the condition is evaluated once and exactly one branch runs. Scalar (`bool`, `i32`, `u32`, `f32`)
  and vector values only; an `if` without `else`, an `if let`, an `if` anywhere but the whole initializer of a `let`,
  and an array, struct or matrix value stay `GUST-E0811`.
- An untyped value `match` (m13.matchu): `let x = match e { 0 => 1u32, _ => 2u32 };` takes its type from an arm by the
  rule of an untyped value `if`, and lowers as the typed form does. When no arm shows a type, it stays `GUST-E0419`
  and the message names the typed form.
- Example event-drain (m11.events): a bounded GPU-to-host event queue that the host drains late, without
  blocking the frame.
- The browser page of web-vector-add (m11.web) runs a launch past a lowered workgroup limit from the split
  artifact, and reports the fallback as `DeviceLimit` with the per-dimension limit; `check-web` checks both.
- `KernelDescriptor::split` (m11.g4a, D208): every GPU `grid` kernel carries a split variant, compiled only when used,
  with one hidden launch-base uniform that its entry point adds to the invocation id. It is the ground for launches
  past the per-dimension workgroup limit (used by the host since m11.g4b). The kernel's own Slang,
  bindings, content key and artifacts do not change.
- `mul_add`, `hypot` and integer `pow` in compute kernels (m11.math3, D205): `a.mul_add(b, c)` lowers to `fma` (a
  driver may fuse it or not; judged by the bound of the unfused evaluation, D39), `x.hypot(y)` to a scaled helper
  (bound 22 ULP; infinite, NaN and subnormal operands are outside the domain) and `b.pow(e)` on `u32` and `i32` to a
  square-and-multiply helper (exact inside the no-overflow domain). The three names are now reserved as method names
  (`GUST-E0328`), so a struct method named `pow`, `hypot` or `mul_add` in a `#[gpu]` module is refused, and a user
  name `fma` is renamed in the emitted Slang (`__gust_r_fma`, a new artifact content key for a program that has one).
- More math methods for compute kernels (m11.math2): `round` (half-way cases away from zero), `signum` (`f32` and
  `i32`), `to_radians`, `to_degrees`, `asin`, `acos` and `rem_euclid` (`f32`, `i32`, `u32`). `round`, `signum` and
  `rem_euclid` lower to `__gust_`-prefixed helpers defined only in a module that calls them. The seven names are
  reserved as method names (`GUST-E0328`), and an unsuffixed integer literal operand of `rem_euclid` is `GUST-E0326`.
  A user helper, local or parameter named `asin` or `acos` is renamed in the emitted Slang (`__gust_r_asin`), so a
  program that uses one of those names gets new emitted text and a new artifact content key. `hypot`, `mul_add` and integer `pow` came in m11.math3. The `f32` `rem_euclid` helper ends its halving loop by a
  flag, not by a float-compare `break` (m11.fm): on the RTX 5090 with NVIDIA driver 610.88 that form ran a loop of
  more than 128 steps the wrong number of times, so remainders of large quotients were wrong.
- `gust_algo::dispatch_batch_in_scope` (m11.scope; in the unpublished `gust-algo`): dispatches a caller's own batch
  and counts it in the enclosing `timed` scope.
- `gust_algo::scan::exclusive_into` and `inclusive_into` (m11.fric1b; in the unpublished `gust-algo`): a scan into
  an output buffer the caller owns, so a caller that scans every step need not allocate the result per step (each call
  still builds its schedule, calls and host-side values, and a cold device pool can allocate the scan's temporaries);
  `exclusive` and `inclusive` now allocate and call them. A wrong output length is the new `Error::Length` (`Error` is
  `#[non_exhaustive]`), refused before any dispatch.
- `#[gpu(crate = <path>)]` and the `crate(<path>);` clause of `par_for!` (audit S6, D48): for a crate that reaches
  `gust` through another crate's `pub use gust;`, the macro rewrites the `::gust::` paths of its output to the given
  path (`::inner::gust`). `::gust_wgpu` stays literal, so the function form and `par_for!` still need a direct
  `gust-compute-wgpu` dependency. An addition, not a break; a malformed or repeated `crate = ..` is `GUST-E0701`.
- The SPIR-V route (m9.rt2, D172, D174): the opt-in Cargo feature `spirv-route` of `gust-compute-wgpu` and
  `gust-compute` builds Vulkan compute and pixel pipelines from slangc's SPIR-V (wgpu passthrough shaders) and falls
  back to WGSL, with a recorded reason (`HeadlessDevice::route_events`, `Route`, `RouteEvent`, `SPIRV_ROUTE`), when the
  device or the kernel cannot take it. With the feature on, the generated Slang clamps array indices, guards integer
  division and remainder, and zeroes workgroup memory; the WGSL text does not change with the feature off. The route
  needs run-time compilation (a build with `slang-runtime`, which the `native` and `web` profiles refuse): precompiled
  artifacts and those builds stay on WGSL. See SEMANTICS section 15.
- Capabilities in the build profile. `GUST_PROFILE` takes `<target>(+<capability>)*`, and `subgroups` is the first
  capability (`dev+subgroups` is the test profile, `native+subgroups` the strict one; `web+subgroups` and a capability
  on `cpu` are refused, `f16` and `atomics64` are reserved names). `gust::capability` (`Capability`, `enabled()`) and
  `cfg(gust_capability = "..")` read it.
- `[package.metadata.gust]` in a kernel crate's `Cargo.toml`: `profiles` (the build profiles the crate declares; the
  default is `["native", "web", "cpu"]`), `limits = "portable"` and `routes`. A malformed table is `GUST-E0753` with the
  line and column.
- Variants. `#[variant(of = kernel, requires(..))]` gives a kernel another body for devices that have a capability
  (the aliases `vulkan` and `webgpu` are the sets `subgroups` and none), and the macro checks that every declared
  profile has a body (`GUST-E0740` to `GUST-E0744`, `GUST-E0750`). `cargo gust` reads and writes artifact record
  schema 4 and manifest schema 3 for them; kernels without variants keep the bytes they had.
- Subgroup operations (21, as shadow functions such as `subgroup_add`, `subgroup_ballot`, `subgroup_shuffle`), accepted
  only in a body that requires `subgroups` and only where every invocation reaches them (`GUST-E0745`, `GUST-E0747`,
  `GUST-E0751`). The WGSL that Slang writes for them is repaired for the pinned `naga` by a text fix-up, in the
  run-time compiler, the reflected bridge and `cargo gust build`. They ran on an RTX 5090 (subgroup size 32) and on
  lavapipe in CI (size 8), the two m8.z forms of `subgroup_broadcast` (`i32` with a literal lane, `f32` with a computed
  lane) included (gpu-software run 42573264225, `dev+subgroups` step); no browser has run one, and the Radeon 890M has
  not run the fixtures.
- `gust-compute-wgpu` opens its devices with the capabilities of the build (`DeviceOptions::capabilities` and
  `with_capabilities`, `HeadlessDevice::capabilities()` and `subgroup_sizes()`), refuses a request the adapter cannot
  meet when the device opens, and refuses a kernel whose `requires` the device does not meet before launch
  (`Error::MissingCapability`, `Error::KernelTarget`, `Error::SubgroupSize`; the enum is `#[non_exhaustive]`).
- An opt-in portable floor: `limits = "portable"` checks every `#[kernel]` of a `#[gpu]` module against
  `gust::limits::PORTABLE`, WebGPU's default limits (`GUST-E0746`).
- Build-time routing. `#[gpu(route = "tag")]` on a module or a function sends its kernels to the CPU or the GPU for a
  whole build; the tags are declared in `[package.metadata.gust.routes]` (`GUST-E0748`, `GUST-E0749`) and chosen by a
  `gust.toml` beside the application's manifest. `cargo gust build`, `check`, `run` and `test` read it, set one
  `GUST_ROUTE__<package>__<tag>` variable per tag and write `gust-artifacts/routes.json`; `cargo gust run` and `test`
  are new. Without a `gust.toml` and without route tags a build is exactly what it was. The driver reads the profile
  from its command line and environment, so it also sets `GUST_ROUTED_PROFILE` to the profile it resolved the routes
  for, and the `#[gpu]` expansion of a routed module fails the compilation of its crate, naming both profiles, when
  cargo's own view differs (a target from `build.target` or `CARGO_BUILD_TARGET`, a custom profile that inherits
  `release`, a `GUST_PROFILE` set only in the config's `[env]`) instead of routing the kernels for another profile.
  Plain cargo sets nothing and is unchanged; a switch between plain cargo and `cargo gust` rebuilds the routed crates
  and the application, not `gust`. A `GUST_ROUTED_PROFILE` set by hand to anything but `dev`, `native`, `web` or `cpu`
  (an empty value included) is a compile error of its own that names the variable, the value and the four profiles.
- A hidden `gust::profile::is_profile` (the check the `#[gpu]` macro writes for a routed module; not public API).
- New public modules of `gust-compute`: `gust::capability`, `gust::limits`, `gust::target` (`Target`, and the hidden
  `GpuKernel` marker), `gust::routes` (`variable_name`, `VARIABLE_PREFIX`) and `gust::routed` (below).
- Methods (M9, slice a1). A `#[gpu]` module may hold one `impl S { .. }` block per struct, with `&self` methods that
  take value parameters and return a value; a call `recv.m(args)` works on a storage element (a read-write one
  included), a field of a uniform or a local (the method reads a copy); a whole uniform is `GUST-E0321`. In Slang the
  methods are members of the struct. Part 2 below adds the rest. In this first part (a1): `&mut self` and `self`
  methods are `GUST-E0116` in this part (part 2 below accepts them), a call of one on a storage, uniform or
  workgroup place is `GUST-E0333` (a texture receiver is `GUST-E0735`), an `impl` for a type the module does not
  define is `GUST-E0121` and a method named like a built-in method, a swizzle, a Slang member keyword or a field of
  its struct is `GUST-E0328`.
- A device handle that follows the route. `gust::routed::RoutedDevice` is one device type whose calls run each kernel on
  the target its build resolved, so a kernel call is the same source whether `gust.toml` routes the kernel to the CPU or
  to the GPU (`RoutedDevice::with_gpu(HeadlessDevice::new()?)`, or `RoutedDevice::<NoGpu>::cpu_only()`; buffers from
  `device.buffer(&[..])`, read with `buffer.read(&device)`). The `#[gpu]` macro writes one more line per kernel, an
  `impl gust::routed::Routed` that names the kernel's side (`OnGpu` or `OnCpu`); a GPU-routed kernel on a handle with no
  GPU is `Error::NoGpu { kernel }`, never a panic. It is additive: `CpuDevice`, `HeadlessDevice` and every
  explicit-device call are unchanged, and `gust` still does not depend on `gust-wgpu`. Limits: storage buffers only (no
  uniforms, textures or samplers), calls are synchronous, a call that changes side copies the whole buffer, and there is
  no GPU backend on wasm32 (`RoutedDevice<NoGpu>` works there).
- Methods, part 2 (M9, slice a2). `&mut self` methods (lowered as Slang `[mutating]`; callable on a `let mut` local, a
  field or an element of one, and on nothing else: a storage, uniform, workgroup or invocation-id place stays
  `GUST-E0333`, so do a `let mut c = out[i]; c.bump(); out[i] = c;`), `self` by value, associated functions without a
  receiver (`static` in Slang, called `S::make(..)` or `Self::make(..)`), `Self` as the name of the struct in types,
  literals and paths, several `impl` blocks for a struct, `impl self::S`, and calls between methods and helpers in both
  directions. `GUST-E0421` refuses a `&mut self` call whose order of evaluation on the GPU would differ from Rust's: in
  the bound of a `for` range (the Slang loop evaluates its end bound first), anywhere in the place of an assignment
  or compound assignment (Slang evaluates the call in the index first), in the value of `/=`, `%=` or `^=` whose
  place has an index (Slang lowers them to helpers that compute the place first), and in the value of `*=` in a
  module that names a matrix. In a `&mut self` method, `self` as a value (`let c = self;`) is `GUST-E0116`: Rust binds a
  reference there and Slang copies. A helper named like a method, or like a field of the struct whose method calls it,
  is `GUST-E0328`
  (Slang resolves an unqualified call to the member first). An attribute on an `impl` block is `GUST-E0115`.
- Fixed-size array layouts (m9.r2b, D159). `gust::TypeLayoutKind::Array { element, count }` and
  `TypeLayout::array(element, count)` describe `[T; N]` (size `element.stride * count`, the element's alignment). 39
  `GpuPod` impls, `[T; N]` for `N` in 1 to 32 and 64, 128, 256, 512, 1024, 2048 and 4096 (no blanket impl, so `[T; 0]`
  has none), when `T: GpuPod`, `T`'s std430 stride equals its Rust size and `T` is not an array; a hand-written element
  that breaks the rule fails when `LAYOUT` is used (E0080; a post-monomorphisation constant, so `cargo build` reports it
  and `cargo check` may not), `Vec3`, `IVec3`, `UVec3` and `Mat3` are not `GpuPod`, and the `on_unimplemented` note
  names the lengths. Arrays are `StorageV1` or `StorageV2` types (an array of vectors, matrices or structs of them) and
  never `Std140V1` (D159's scope: uniform arrays are refused for now, E0215). The reflection cross-check compares a host
  array with the schema-4 reflection (count, element stride, element, size), and `gust-compute-wgpu`'s binding checks
  accept the array kind. The enum is `#[non_exhaustive]` on `release/0.1` too, so the new variant is not a break (a
  `match` outside the crate already needs a wildcard arm). No `#[gpu]` module can name an array yet; that is the macro
  half of the slice (r2a).
- Enums and the statement `match` (M9, slice e1). A `#[gpu]` module may hold a field-less `enum`
  (`enum E { A, B = 3, C }`; the macro adds `Clone`, `Copy` and `#[repr(u32)]`, `#[repr(i32)]` is allowed, the user
  derives `PartialEq, Eq`). Values are locals, parameters and results of helpers, compared with `==` and `!=`, and cast
  with `as u32` (`as i32` for a signed enum) only. A statement `match` on an enum (`E::A => { .. }`,
  `E::A | E::B => { .. }`, `_ => { .. }`) becomes a Slang `switch`; a `return` in an arm is legal in a helper and stays
  `GUST-E0415` in a kernel with workgroup memory or a barrier. Refused: a data variant, a computed discriminant, a
  missing `PartialEq` or an `as` to another type (`GUST-E0122`), an enum in a struct field, a buffer, a uniform,
  workgroup memory, an `Option` or a `Result` (`GUST-E0211`), a `break` or `continue` in an arm aimed at the loop around
  the `match` (`GUST-E0417`), a guard (`GUST-E0418`), bindings and other patterns (`GUST-E0521`), other scrutinees
  (`GUST-E0515`) and methods on an enum (`GUST-E0121`). The value form of `match` is in slice e2 below. On an RTX 5090
  the fixture runs equal to a hand-written table, and the evaluation order of calls in a scrutinee, in `==` and `!=`,
  and in enum arguments is Rust's.
- The value form of `match` and integer patterns (M9, slice e2). `let x: T = match e { .. };` with a typed `let` is
  accepted for an enum, `u32` or `i32` scrutinee; an arm is an expression, or a block that ends in an expression or a
  `return`, and the result may be a struct. An integer `match` (statement or value) takes literal patterns (`0`,
  `1 | 2`, `3u32`, `-1i32`, within 32 bits) and needs a final `_`. It lowers to `T x;` and a `switch` whose arms assign
  `x`. A `match` that gives a value anywhere else, a function's tail included, is `GUST-E0419`; ranges, bindings and
  constants as patterns stay `GUST-E0521`. The scrutinee is read once and the operands of an arm are evaluated in the
  order of Rust (measured), so nothing is refused for order. On an RTX 5090 the fixture runs equal to hand-written
  tables on the GPU and the CPU twin.
- Tuples (M9, slice t). A tuple value `(a, b)` and type `(u32, f32)` of two to eight elements (scalars, vectors,
  matrices, structs, enums, or a tuple of tuples) is a local, a parameter, the result of a helper or a method and a
  method argument; `t.0`, `t.0 = e`, `f().1`, `let (x, y) = f();` and `let (x, y): (u32, f32) = f();` are accepted. It
  lowers to Slang's `Tuple<uint, float>`, `makeTuple(a, b)` and `._0`; `Tuple` and `makeTuple` join the renamed names.
  Refused: a tuple in a struct field, a buffer, a uniform or workgroup memory, or a resource as an element
  (`GUST-E0212`); more than eight elements, one element, an array, reference, `Option` or `Result` element, a comparison
  of a visible tuple (not supported yet), `Some((..))` and `.8` (`GUST-E0213`); a tuple struct (`GUST-E0104`); a nested
  pattern, `_` or `..` in a destructuring (`GUST-E0319`, whose text now allows the tuple pattern); a tuple as a `match`
  scrutinee (`GUST-E0515`). The elements are evaluated in the order of Rust (measured), so nothing is refused for order.
  On an RTX 5090 the fixture runs equal to hand-written tables on the GPU and the CPU twin.
- Arrays as values (M9, slice r1). `[T; N]` with a literal `N` above zero and a scalar, vector, matrix, struct or enum
  element is a typed local, a parameter, the result of a helper or a method, and an argument; literals `[a, b]`, repeats
  `[e; N]` (one temporary, then `{ t, t, .. }` up to 16 elements and a fill loop above), `a[i as usize]` and `a[2]`,
  `a[i] = e`, `a[i] += e` and `a.len()` are accepted. `usize` is accepted as the cast of an index only (`GUST-E0334`; it
  was `GUST-E0201` and `GUST-E0202`). Refused: a zero or non-literal length (`GUST-E0216`), a nested array or a tuple,
  `Option`, `Result` or reference element (`GUST-E0217`), a slice (`GUST-E0111`), `for x in a` (`GUST-E0409`), an array
  in a struct field, buffer, uniform or workgroup memory (`GUST-E0813`), an array in a tuple (`GUST-E0213`), and an
  assignment to an element whose index a `&mut self` call in the value changes (`GUST-E0421`: Rust runs the value first,
  Slang the index). The same `GUST-E0421` rule now covers a storage or workgroup element (`out[c.n] = c.next()`,
  `out[c.n] += c.next()`, `wg[w.n] = w.bump()`), which earlier slices accepted: that form gave another element than the
  CPU twin on the GPU with the pinned Slang (2026.21 reads the index first; 2026.20 called first), so this refuses a
  form that main accepted. On an RTX 5090 the fixture runs equal to hand-written tables on the GPU and the CPU twin.
- Arrays in structs and storage (M9, slice r2a, macro half). A struct field `[T; N]` and a buffer element `[T; N]`
  (`RWStructuredBuffer<[T; N]>`, `StructuredBuffer<[T; N]>`) are accepted, `N` in 1..=32 or 64..=4096 by powers of two
  and `T` a scalar, a 2- or 4-wide vector, `Mat2`, `Mat4` or a struct of the module. New: `GUST-E0214` (a `Vec3`,
  `UVec3` or `Mat3` element) and `GUST-E0215` (an array in a uniform); `GUST-E0216` also names the stored length set.
  `.len()` of a field or an element is its constant (a receiver with a call is `GUST-E0306`); `==` on them is
  `GUST-E0811`. A field of array type was `GUST-E0813` before. The `GUST-E0421` rule already covered every indexed
  place; field arrays of storage elements are covered by tests and a fixture.
- Non-capturing closures (M9, slice c, D160). `let f = |x: f32, y: f32| -> f32 { .. };` with typed parameters, an
  explicit result type and a block body, bound immutably and called as `f(a, b)` in a kernel, a helper or a method. Each
  closure is hoisted to a Slang function `__gust_cl_<owner>_<n>` with its parameters by value. A closure captures
  nothing and is not a value. New: `GUST-E0329` (a capture, or `move`; an earlier closure is a capture), `GUST-E0330` (a
  closure used as a value, `let mut f`, an annotation), `GUST-E0331` (a missing type), `GUST-E0332` (a closure in a
  closure); `GUST-E0314` now names only the forms with no lowering (`async`, `static`, `const`, `for<..>`, attributes);
  `GUST-E0328` also covers the name of a closure; `GUST-E0421` also covers a closure call in a `for` bound. A closure
  body is judged as a helper (no barrier, subgroup, atomic, `Workgroup` or resource). A closure was `GUST-E0314` before.
  On the RTX 5090 the fixture runs equal to hand-written tables on the GPU and the CPU twin, including the order of the
  arguments of a closure call.
- Added (m9.sen): the GPU test `driver_sentinel` for the NVIDIA array defect (D168, D169). It runs three `#[gpu]`
  kernels shaped like the investigations' programs and expects the defect's wrong values on the RTX 5090 (Vulkan) and
  Rust's values on other non-NVIDIA adapters, so a driver or toolchain change that makes these programs compute Rust's
  values fails the test, the cue to re-measure.
- Traits with static dispatch (M9, slice g1, D170).
  `trait T { fn m(&self) -> R; fn n(self) -> Self; fn k(&mut self, ..); }` with methods that have a receiver and no
  body, and `impl T for S` for a struct or a field-less enum of the module. A trait lowers to a Slang `interface` and
  each impl to an `extension S : T`; a call is the concrete method, by method syntax or by `T::m(&s, a)`, `T::m(s, a)`
  and `<S as T>::m(&s, a)` on a place (the path forms are rewritten to `s.m(a)` for validation and Slang only; the CPU
  twin keeps the call as written). New: `GUST-E0124` (a trait form outside the contract: a default body, an associated
  item, a supertrait, a generic trait or method, `&mut self` on an enum), `GUST-E0126` (a path call whose receiver is
  not a place) and `GUST-E0218` (`dyn`, `impl Trait`, a trait used as a type). `GUST-E0121` also covers a trait impl for
  a foreign type and a blanket impl; `GUST-E0328` also covers a trait method named like an inherent method of an
  implementing type, or declared by two traits one type implements. A trait impl was `GUST-E0102` or `GUST-E0121`
  before. The Slang names of the stdlib interfaces (`IFloat`, `IInteger`, ..) are renamed in a user trait.
- Generic helpers and methods with trait bounds, static specialisation (M9, slice g2, D170).
  `fn f<T: A + B>(t: T) -> T`, `fn f<T>(t: T) where T: A`, and `impl S { fn m<T: A>(&self, t: T) }` take type parameters
  bounded by traits of the module (`Copy` is accepted and dropped); the type arguments are inferred from the arguments,
  and `T` is written as a whole parameter, result or `let x: T` type. A struct, a unit enum or a struct with array
  fields can be the argument. Slang gets `T f<T : A & B>(T t)` and specialises it. New: `GUST-E0123` (a turbofish, or a
  type parameter no argument determines) and `GUST-E0125` (a bound that is not a trait of the module nor `Copy`: a std
  trait, `?Sized`, `for<..>`). `GUST-E0114` now means a generic kernel, closure, associated function without a receiver,
  const generic, lifetime or default, or `T` nested in an array, tuple or reference; `GUST-E0328` also covers a type
  parameter named like an item of the module, a dialect type or an id type. A generic trait method is `GUST-E0124`, and
  every type parameter needs a trait of the module as a bound (`Copy` alone, or none, is `GUST-E0125`). Calls through a
  bound are covered by the by-name call graph (`GUST-E0120`, D154), so every impl is validated as a helper. Generic
  structs and enums stay `GUST-E0103`.
- Fixed (m9.f1): a method call with a `<` comparison argument before another argument, such as `s.m(a < b, c)`, failed
  in slangc with an undefined `__gust_keep`, because the scan that defines that helper did not visit method calls.
- Fixed (m9.shift): a constant shift amount outside 0..32 (`x << 34u32`, `v >>= 32`) is refused with `GUST-E0335`;
  rustc refuses it when building, and naga used to reject the generated WGSL with an error that was not a GUST
  diagnostic. Known gap: an amount the macro does not see as constant (a literal sum, a local bound to a literal) is
  folded by Slang and still rejected by naga. A run-time amount is unchanged and masks to its low 5 bits (KM-04).
- Fixed (m9.pa): a kernel whose buffer element is or holds a fixed-size array (`RWStructuredBuffer<[u32; 4]>`, a struct
  with an array field, arrays of vectors and of structs) now builds through `cargo gust build` and runs from its
  precompiled artifact. The record has an `array` layout kind (`count` and a nested `element`), and the fill reads it
  from the reflection. An array of arrays is `GUST-E0217` in the macro (the tool refuses one too). The record
  schema stays 1: an older tool refuses the unknown kind by name.
- `gust::routed::Error::downcast_gpu::<E>()` (m10.api2a.1, G17) returns the GPU backend's own error when the error
  is `Error::Gpu`, as `gust::map::Error::downcast_ref` does for the map functions:
  `error.downcast_gpu::<gust_wgpu::Error>()`.
- `Debug` for the `gust-compute-wgpu` handles (m10.api2a.1, W2): `HeadlessDevice`, `GpuBuffer<T>`, `GpuUniform<T>`,
  `GpuPool<T>`, `StagedGraph`, `BufferDispatch` and `WgpuJob` print ids, lengths and counts, never a raw wgpu handle.
- `Artifact::with_format`, `with_key`, `with_bytes`, `with_reflection`, `Artifact::new` and `ModuleDescriptor::new`
  (m10.api2a.1): the constructors that the `#[non_exhaustive]` structs of B6 need (B5's record structs get none; only
  the crate builds them).
- `HeadlessDevice::execute_graph_timed` (m10.ts2, D194) runs a `StagedGraph` like `execute_graph` and also reports
  `GraphReport::gpu_elapsed`, the GPU time from the beginning of the first dispatch pass (compute, indirect or `pixel`)
  to the end of the last; copies between dispatches are inside, uploads before and read-backs after are not. `None`
  on an adapter without timestamp queries or for a graph that ran no dispatch pass, and always `None` from
  `execute_graph`, which takes no lock and is otherwise unchanged. Native only. An addition: `GraphReport` is
  `#[non_exhaustive]`. If reading the timestamps fails, the device's timestamp read-back buffer can stay mapped and a
  later timed call on that device can fail (also true of `dispatch_batch_timed`).
- The guide (m10.k0 to m10.k3, D182): the documentation-only modules `gust::guide` and `gust_wgpu::guide`, written as
  rustdoc chapters (why GUST, install and a first program, the authoring model, running kernels, GPU concepts,
  profiles and routing, semantics and the twin, the newer language features, the browser, benchmark methods,
  stability and platforms, and internals for contributors). Each listing is a doctest or a copy of a region of a
  gallery program, and every public type has an example (m10.k2). The modules hold nothing but empty chapter modules,
  and the chapter set carries no semver promise. The ten gallery programs (`gallery/`) belong to the repository, not to
  the packages.

### Deprecated

- The f32/u32 compatibility family (m10.api2a.2, audit W3): 13 functions and 6 types are `#[deprecated(since = "0.2.0")]`.
  They still work and are **removed at 1.0**. A call or a `use` of one warns, so a build with `-D warnings` needs the
  typed call. Each has a typed replacement that already exists:

  | Deprecated | Use instead |
  | --- | --- |
  | `HeadlessDevice::create_f32_buffer` | `create_typed_buffer` |
  | `write_f32_buffer`, `write_f32_buffer_timed` | `write_typed_buffer`, `write_typed_buffer_timed` |
  | `read_f32_buffer`, `read_f32_buffer_timed` | `read_typed_buffer`, `read_typed_buffer_timed` |
  | `submit_f32_buffers` | `submit_buffers` with `BufferBinding` |
  | `dispatch_f32_buffers`, `dispatch_f32_buffers_timed` | `dispatch_buffers`, `dispatch_buffers_timed` with `BufferBinding` |
  | `submit_f32_batch` | `submit_batch` with `BufferDispatch` |
  | `dispatch_f32_batch`, `dispatch_f32_batch_timed` | `dispatch_batch`, `dispatch_batch_timed` |
  | `dispatch_f32`, `dispatch_u32` | `create_typed_buffer`, then `dispatch_buffers` (or the kernel's typed `dispatch`), then `read_typed_buffer` |
  | `F32Binding`, `U32Binding`, `F32BufferBinding` | a `GpuBuffer<T>` and `BufferBinding` |
  | `F32BufferDispatch` | `BufferDispatch` |
  | `DispatchOutput`, `U32DispatchOutput` | `read_typed_buffer` on the buffers |

### Changed, and may break code written for the unpublished 0.1.0 line

These are the differences from `release/0.1`, the branch that was prepared as 0.1.0 and never published. They matter to
code built from that branch and to this repository's fixtures; code written for the registry starts at 0.2.0.

- **B1, part 1: `KernelDescriptor` is `#[non_exhaustive]`** and has new fields (`variant`, `requires`, `target`).
  A struct literal, or a struct update from a base value, no longer compiles outside `gust-compute`. Migration: build
  one with `KernelDescriptor::new(module, name, entry_point, workgroup_size, parameters, slang_source)`, which sets the
  grid shape, no artifacts, the portable variant, no capability and the GPU target, and change a field with its
  `with_*` setter (`with_shape`, `with_source`, `with_artifacts`, `with_variant`, `with_requires`, `with_target` and
  the others). Kernels written with `#[gpu]` need no change. `cargo semver-checks` lists this one.
- **B1, part 2: the `GpuKernel` bound.** The generic entry points of `gust-compute-wgpu` and the runs of `StagedGraph`
  require the `GpuKernel` marker. `#[gpu]` kernels get the marker from the macro, so they need no change. A
  hand-written `Kernel` impl that runs on `HeadlessDevice` adds `impl gust::target::GpuKernel for MyKernel {}`;
  without it the call is E0277 with a message that says so. The semver tool does not list this change (the marker is
  `#[doc(hidden)]`).
- **Subgroup names are reserved in a `#[gpu]` module** (`GUST-E0745`). A function of a `#[gpu]` module, a kernel or a
  helper, can no longer be named like one of the 21 subgroup operations (`subgroup_add`, `subgroup_size`,
  `subgroup_ballot` and the rest, listed in `docs/SEMANTICS.md`), because a call of the name is that operation. A 0.1.0
  module that has a helper with one of these names no longer compiles, with an error on the function. Migration:
  rename the function. It is not part of B1 (the semver tool cannot see it: it is a macro check); no example or fixture
  of this repository had such a name (the unit test that pins the error is the one place that writes one).
- **B3: recursion is refused in every `#[gpu]` module** (`GUST-E0120`). The validator builds a conservative call graph
  from names (`f(..)` to the helper, `x.m(..)` to every method named `m`, `S::f(..)` to the function of `S`) and refuses
  a cycle. A 0.1.0 module that routed a kernel to the CPU and kept a recursive helper compiled before, because a
  CPU-routed module is not translated, and so did a GPU-routed module whose recursive helper was never dispatched
  (slangc ran only at dispatch); with 0.2.0 the same source is `GUST-E0120`, because the check runs wherever the
  `#[gpu]` module is validated. Migration: rewrite the recursion as a `for` loop with a fixed trip count. No example of
  this repository had a recursive function; `crates/gust/tests/diagnostics.rs` had three recursive helpers (fixtures for
  the Slang line table), which were rewritten to put the recursive call into the Slang text. A call `x.m(..)` also
  counts as a call of every method named `m` of another struct, so rename a method that calls another struct's method of
  the same name.
- **B4: more names are renamed in the emitted Slang** (`Tuple`, `makeTuple`, `this`, `interface`, `extension`, and the
  46 interface names of Slang's standard module, `IFloat`, `IInteger`, `IComparable`, `__BuiltinType` and the rest,
  `STDLIB_INTERFACES` in `crates/gust-macros/src/slang/collisions.rs`). M9 uses `Tuple<..>` and `makeTuple` for tuples
  and `this` for method receivers, and emits a user `trait` as an `interface` and an impl as an `extension`, so a user
  name equal to one of these is emitted as `__gust_r_<name>`, as the 0.1.0 names of the collision table already were
  (D81). A 0.1.0 module with any user name of one of these spellings (a struct, field, local, helper or parameter; 0.1.0
  had no enums) gets other Slang text, hence another content key for its precompiled artifacts. Migration: run
  `cargo gust build` again and commit `gust-artifacts/`; the source needs no change, and a slangc message shows the
  Rust spelling again. Measured on slangc 2026.22 for `interface` and `extension`: as a local, field, helper,
  parameter or struct on its own the word compiles, and a user struct, helper or global of that name placed before an
  emitted `interface` or `extension` fails (E20001); the rename removes that. Not measured: how many 0.1.0 modules use
  such a name (no example of this repository does; the tests that use one are the renaming's own). The same table now
  also holds the 18 scalar math intrinsics the float math methods emit (`abs`, `atan`, `atan2`, `ceil`, `clamp`, `cos`,
  `exp`, `exp2`, `floor`, `log`, `log2`, `max`, `min`, `pow`, `sin`, `sqrt`, `tan`, `trunc`) and the other Slang
  intrinsics the emitter writes unqualified (`asint`, `asuint`, `countbits`, `GroupMemoryBarrierWithGroupSync` and
  `WaveActiveAllTrue`, `WaveActiveAnyTrue`, `WaveActiveBallot`, `WaveActiveBitAnd`, `WaveActiveBitOr`,
  `WaveActiveBitXor`, `WaveActiveMax`, `WaveActiveMin`, `WaveActiveProduct`, `WaveActiveSum`, `WaveBroadcastLaneAt`,
  `WaveGetLaneCount`, `WaveGetLaneIndex`, `WavePrefixProduct`, `WavePrefixSum`, `WaveReadLaneAt`,
  `WaveReadLaneFirst`, `WaveShuffle`), so a user local, parameter or helper of one of these names no longer captures
  the emitted call. A program with such a name now gets different emitted text and a different artifact content key
  (a program without one is unchanged); run `cargo gust build` again and commit `gust-artifacts/`.
- **B5: the record structs and `Route` of `gust-compute-wgpu` are `#[non_exhaustive]`** (m10.api2a.1, W8 and W13).
  `PipelineCacheStats`, `PoolStats`, `DispatchTiming`, `TransferTiming`, `GrowthRecord`, `TransferRecord`, `RouteEvent`
  and the enum `Route`. A struct literal or a struct update of one of the structs no longer compiles outside the crate,
  and a `match` on `Route` needs a wildcard arm (reading fields and `==` on a `Route` are unchanged). Migration: only
  the crate builds them, so a reader changes nothing; code that built one starts from `Default` where the type has it
  (`let mut t = TransferTiming::default(); t.bytes = ..;`, as `examples/signal-pipeline` now does) or compares the
  fields (as `examples/component-pool` now does for `GrowthRecord`, which has no `Default`). `cargo semver-checks`
  lists the six structs; `Route` and `RouteEvent` are new in 0.2.0 (not in the `release/0.1` baseline) and behind
  `spirv-route`, so they break no released version and the tool does not list them.
- **B6: `ModuleDescriptor`, `Artifact` and `Verified` are `#[non_exhaustive]`** (m10.api2a.1, G7a and G20), with
  constructors in the pattern of `KernelDescriptor`. Migration: `Artifact { format, key, bytes, reflection }` becomes
  `Artifact::new(format, key, bytes, reflection)`, `Artifact { bytes, ..BASE }` becomes `BASE.with_bytes(bytes)` (also
  `with_format`, `with_key`, `with_reflection`), and `ModuleDescriptor { name, kernels }` becomes
  `ModuleDescriptor::new(name, kernels)`. `#[gpu]` modules and `cargo gust` need no change (the macro and the tool
  emit the constructors); `Verified` is only produced by `verify_artifact`. `cargo semver-checks` lists these three.
- **B7: `HeadlessDevice::check_same_device` and `pool::EXACT_CLASS_BYTES` are `#[doc(hidden)]`** (m10.api2a.1, W1
  and W10). Both stay callable and keep their paths, so no program stops compiling; the semver tool lists a hide as a
  removal. Migration: none (the first is a support function of the tests, the second an allocator detail with no use
  outside the crate).
- `DeviceOptions::default()` and `HeadlessDevice::new()` now request `gust::capability::enabled()`. In a build with no
  capability that is the empty set, as before; in a `+subgroups` build they open a device with the feature or fail with
  `Error::MissingCapability`.
- Documented, not a break of 0.1.0 code: a route flip changes which device a kernel's call sites compile against. A
  kernel call names its device, so a kernel routed to the CPU has no `GpuKernel` impl (its `HeadlessDevice` calls are
  E0277) and has `CpuDevice` calls instead, and flipping the route back reverses both. Only the calls of a generated
  element-wise map (`name::map`) are the same on either target. No 0.1.0 kernel is routed. The device handle that
  follows the route is `gust::routed::RoutedDevice` (m9.h1, D165), listed under Added.
- The pinned Slang is 2026.22 (`cargo gust slang install` fetches it); the 0.1.0 line stays on 2026.20. Artifacts
  built with 2026.20 fail `cargo gust check` ("the manifest was built with Slang `2026.20` but the installed helper
  reports `2026.22`"); run `cargo gust build` again (or pass `--allow-slang-mismatch`). The emitted WGSL of one
  kernel kind changes visibly: a compare-exchange now retries after a spurious failure (Slang #13918, read from the
  emitted text). That stays inside the weak contract of `atomic_compare_exchange`, but it is a change in GPU behaviour.
- A known departure from Rust, documented rather than refused: in an assignment whose index and value are atomics on
  the same cell (`out[atomic_add(&mut c[i], 1u32) as usize] = atomic_add(&mut c[i], 10u32);`) the GPU runs the index
  first (slangc 2026.22, read from the emitted WGSL; not run on a GPU), where Rust and the CPU twin run the value
  first, so the two store different values (SEMANTICS, "Known departure from Rust: two atomics on one cell in an
  assignment"; D176). Whether Slang 2026.20, the 0.1.0 line, ran the value first was not measured.

### A new dependency

`gust-compute-macros` depends on `toml` 1 (no default features; `std` and `parse`) to read `[package.metadata.gust]`.
It brings four more crates, so a build of `gust-compute` compiles five that 0.1.0 did not: `toml`, `toml_parser`,
`toml_datetime` and `serde_spanned` (MIT or Apache-2.0) and `winnow` (MIT). Each declares a minimum Rust of 1.85 or
lower, under the workspace's 1.90. The reason is that `cargo publish` rewrites a manifest, so a reader of the source
form that was written by hand would refuse the form a published crate carries. This is not a code break.

### Compatibility with the unpublished 0.1.0 line

The `semver` job of CI compares `gust-compute` and `gust-compute-wgpu` with the branch `release/0.1` (the unpublished
0.1.0; it is the baseline until 0.2.0 is on crates.io) with `cargo-semver-checks` 0.51.0, `--release-type patch`. On
the commit `9895659` (`ci.yml` run 43719222853, job `semver`; the job treats its findings as expected and stays green;
the commit after it changed this record only, so the counts hold), the job found, as a record of that commit and not of
the commit that will be published (the semver job will be run again on that one):

- `gust-compute`: 229 checks, 228 passed, 31 skipped, 1 failed: `struct_marked_non_exhaustive` for `KernelDescriptor`
  (B1), `ModuleDescriptor`, `Artifact` and `Verified` (B6). The tool says a new major version is required (for 0.x,
  a minor bump).
- `gust-compute-wgpu`: 229 checks, 224 passed, 31 skipped, 5 failed: `struct_marked_non_exhaustive` for the six record
  structs of B5; `inherent_method_now_doc_hidden` for `HeadlessDevice::check_same_device` and
  `pub_module_level_const_now_doc_hidden` for `EXACT_CLASS_BYTES` (B7); `type_marked_deprecated` (six types) and
  `type_method_marked_deprecated` (13 methods) for the f32/u32 compatibility family, which is deprecated, not removed
  (a warning, not a break).
- What the job cannot see: the `GpuKernel` bound (hidden from the docs), the changed defaults above, the `toml`
  dependency, and `cargo-gust` (a binary).

The tree holds 113 package and workspace manifests of examples, fixtures and tests (`examples/`, `tests/` and
`crates/gust-algo`) that already exist on `release/0.1`. This is what `git diff release/0.1 main` says of each, and
which gate of `main` builds it (`check-examples` builds the examples but `web-vector-add`, which `check-web` builds;
`check-cpu` their twins; `check-ui` the `tests/ui` fixtures; `check-dual`, `check-cpu-target`, `check-precompiled` and
`check-difftest` the fixture workspaces of the same names; `check-feature algo` and the Gpu suite `gust-algo`):

- **88 unchanged**: no file in the crate's directory differs. (Method, recounted at the M10 close: for each of the 113
  manifests, `git diff --name-only release/0.1 <commit> -- <the manifest's directory>` is empty. A nested crate
  counts in its parent's directory too.)
- **4 changed only in derived files**, source untouched: `examples/map-chain` (its committed artifacts, written again
  for Slang 2026.21), `tests/difftest/template` (its `Cargo.lock`) and the `tests/ui` crates `profile-native-artifact`
  and `profile-native-no-manifest` (one artifact file each).
- **21 changed in a source file or a manifest.**
  - Four are the B1 migration (both parts), the only edits that existing code needed: the `tests/ui` crates
    `typed-call-buffer-for-sampler`, `typed-call-buffer-for-texture`, `typed-call-sampler-for-buffer` and
    `typed-call-texture-ok` wrote a `KernelDescriptor` literal and now call `KernelDescriptor::new(..)` and add `impl
    gust::target::GpuKernel for Blur {}` (three of the `expected.txt` files only change a line number).
  - One is a version literal: `tests/precompiled/kernels` named its `gust` path dependency `0.2.0-dev` (it is `0.2.0`
    in the release); its manifest and one reflection file are also written again by the new Slang.
  - Six are deprecation migrations of slice m10.api2a.2 (the W3 deprecations): `examples/component-pool`,
    `examples/particle-step`, `examples/polynomial`, `examples/signal-pipeline`, `examples/vector-add` and
    `tests/precompiled/app` moved their `create_f32_buffer`, `read_f32_buffer` and `_timed` calls to the typed
    ones. The old calls still compile (deprecated, not removed), so these edits were not needed for 0.1.0 code to build.
  - Ten only gained things (and `tests/difftest` a re-stamped seed file): `crates/gust-algo` (the `evidence` module,
    an example, and opt-in GPU timestamps: `timed`, `Timed` and `GpuTime` report the summed GPU time of the batches
    that a closure of gust-algo calls dispatches on its thread, D194; additive; not published), `examples/web-vector-add` (an added `check-web` assertion that a subgroup
    request is refused, and re-written artifacts), `tests/cpu-target` and its `kernels` (a `routed` crate and a
    `variant` module), `tests/dual` with its `consumer` and `kernel-lib` (a routed module and binary), `tests/difftest`
    (the format 7 harness and the profile flags; `seeds.txt` takes the new format stamp and all eight program hashes),
    `tests/precompiled` (an expectation file) and the `tests/ui` workspace (new members and fixture crates).

By this comparison the only source edits that 0.1.0 code needed are the four B1 migrations. That is a statement about
the diff, not about a run: whether each gate is green on a commit is recorded in STATUS, and the gates that 0.2.0 adds
(`check-routing`, the `dev+subgroups` batches) only add fixtures.

## 0.1.0 (never published)

History, kept as a record. 0.1.0 was prepared as the first release and was never published (D177, the owner's choice of
2026-11-14): the packages named 0.1.0 are not on crates.io, there is no `v0.1.0` tag, and `release/0.1` stays a branch
with no tag. 0.2.0 will be the first public release and will build on it, with the changes its own entry lists (the
pinned Slang below is 2026.20; 0.2.0 pins 2026.22). The entry describes the state of that branch, an experimental
compiler from a restricted subset of Rust to Slang for compute kernels.

### What you can do

- Write a kernel as a Rust module marked `#[gpu]`. Rustc checks it as ordinary Rust, and the macro translates the
  syntax tree directly to readable Slang. Constructs outside the supported subset are rejected with a stable
  diagnostic code, `GUST-E....`, listed in [DIAGNOSTICS](docs/DIAGNOSTICS.md).
- The supported subset covers scalars, nested plain structs, vectors and matrices, `Option` and `Result` values,
  bounded `for` loops, uniform parameters, atomics, workgroup memory with barriers, and one-, two- and
  three-dimensional dispatch. [README](README.md#current-subset) has the list.
- Call a kernel with a typed `call`, `submit` or `dispatch` that checks element types, resource counts, access,
  layouts, aliasing and device limits. `#[gpu]` on a free function of scalars or vectors generates an element-wise
  map, and `par_for!` writes a loop as a kernel.
- Run kernels on a Vulkan compute adapter, or on WebGPU in the browser from `wasm32-unknown-unknown`, with typed
  buffers, a pipeline cache, ordered batches (`StagedGraph`), device-resident pools and vectors (`GpuPool`, `GpuVec`)
  and asynchronous readback. Every pipeline is checked against the layout that Slang reports before it is created.
- Run the same kernels as plain Rust: a kernel has a CPU twin in test and dual builds unless it opts out, and a
  build with `GUST_PROFILE=cpu` is a production CPU target with no GPU runtime.
- Compile ahead of time. `cargo gust build` compiles the kernels with Slang, verifies their layouts and writes the
  `gust-artifacts/` directory, which you commit; `cargo gust check` fails when it is stale. A release build embeds the
  artifacts and needs neither Slang, nor the native helper, nor `cargo-gust`.
- Compile at run time during development, through a feature of your own crate that turns on `slang-runtime`.
- Manage the toolchain with `cargo-gust`: `init`, `build`, `check`, `doctor`, and `slang install`, `locate` and
  `build-helper` for the pinned Slang release (2026.20, downloaded and checked by SHA-256, never shipped in the
  crates).

### Execution shapes and textures

`pixel` kernels (`#[kernel(shape = pixel)]`: one fragment per element, outputs written as storage buffers), the `Shape`
type and the 2-D texture and sampler types are part of this release and of the 0.x promise, so a minor release may
change them. Their evidence is narrower than the rest: a GUST `pixel` or texture kernel has run on one hardware
adapter, the RTX 5090 (Vulkan), and on the lavapipe software adapter, and on no other hardware. Headless Chrome on that
NVIDIA GPU has run a `pixel` kernel and no texture kernel. README, "Stability", says the same.

### Platforms and requirements

- Vulkan on Windows and Linux, WebGPU in the browser. macOS and Metal are not supported, and DX12 is not claimed.
- Rust 1.90 or newer.
- A C++ compiler (MSVC on Windows, g++ or clang++ elsewhere) for `cargo gust slang build-helper`, needed only for the
  run-time compile path and for authors who export artifacts.

### Known limits

- The error bound of a linear texture `sample` assumes a sampler with 8-bit filter weights, rounded to the nearest
  step; no specification gives that (WebGPU states no number; Vulkan requires `subTexelPrecisionBits` of 4 in core and
  of 8 for Vulkan 1.4 or Roadmap 2022) and the evidence is one adapter, so the bound can be exceeded, and its test fail,
  on a device that reports fewer than 8 bits (SEMANTICS, section 4).
- In the browser (`wasm32`) nothing blocks: a job's `wait`, and so a kernel's `dispatch`, returns `Error::WouldBlock`
  (any work it submitted will run). Submit, then await a readback (`read_typed_buffer_async`, `GpuVec::to_vec_async`).
- SPIR-V is exported and validated with `spirv-val` when it is on `PATH`, but the runtime executes the WGSL.
  Running the exported SPIR-V was planned for a 0.1.x release, which will not exist; 0.2.0 has the opt-in
  `spirv-route` feature instead (listed under Added).
- A crate that depends directly on both `gust-compute` and the unrelated crate `gust` fails with rustc error
  E0464, because both libraries are called `gust`. Renaming a dependency needs an `extern crate ... as gust;` line.
- The packaged crates are not test-runnable: their tests read files of the repository. Run the tests from a checkout.
- Kernels that use workgroup memory or barriers have no CPU twin yet: they must say `cpu = "unsupported"`, and a
  `dispatch` on them does not compile under the `cpu` profile.
- The algorithms crate of the repository (`gust-algo`: reduce, scan, compact) and the examples are not published.
