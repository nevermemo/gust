# Changelog

All notable changes to the published packages `gust-compute`, `gust-compute-macros`, `gust-compute-wgpu` and
`cargo-gust` are listed here. The four packages always share one version. GUST follows SemVer for 0.x: a change that
breaks existing code bumps the minor version, and the promise is described in the
[README](README.md#stability).

## Unreleased (0.2.0)

The workspace version is `0.2.0-dev`. Work for 0.2.0 lands on `main`; the 0.1.0 line continues on the branch
`release/0.1`. This list is what the milestone (M8) has merged. 0.2.0 has three listed breaks. B1, below, in two parts (the
descriptor constructor and setters, and the `GpuKernel` bound), is the last of its kind: the descriptor is
`#[non_exhaustive]` from now on. The second is a function name in a `#[gpu]` module that is now reserved (the 21 subgroup
operations). The third, B3, is that recursion is refused (`GUST-E0120`).

### Added

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
  methods are members of the struct. Part 2 below adds the rest. Until then (a1): `&mut self` and `self`
  methods are `GUST-E0116` until the next slice, a call of one on a storage, uniform or workgroup place is
  `GUST-E0333` (a texture receiver is `GUST-E0735`), an `impl` for a type the module does not define is `GUST-E0121`
  and a method named like a built-in method, a swizzle, a Slang member keyword or a field of its struct is
  `GUST-E0328`.
- A device handle that follows the route. `gust::routed::RoutedDevice` is one device type whose calls run each kernel on
  the target its build resolved, so a kernel call is the same source whether `gust.toml` routes the kernel to the CPU
  or to the GPU (`RoutedDevice::with_gpu(HeadlessDevice::new()?)`, or `RoutedDevice::<NoGpu>::cpu_only()`; buffers from
  `device.buffer(&[..])`, read with `buffer.read(&device)`). The `#[gpu]` macro writes one more line per kernel, an
  `impl gust::routed::Routed` that names the kernel's side (`OnGpu` or `OnCpu`); a GPU-routed kernel on a handle with no
  GPU is `Error::NoGpu { kernel }`, never a panic. It is additive: `CpuDevice`, `HeadlessDevice` and every explicit-device
  call are unchanged, and `gust` still does not depend on `gust-wgpu`. Limits: storage buffers only (no uniforms,
  textures or samplers), calls are synchronous, a call that changes side copies the whole buffer, and there is no GPU
  backend on wasm32 (`RoutedDevice<NoGpu>` works there).
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

### Changed, and may break code written for 0.1.0

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
  from names (`f(..)` to the helper, `x.m(..)` to every method named `m`, `S::f(..)` to the function of `S`) and refuses a
  cycle. A 0.1.0 module that routed a kernel to the CPU and kept a recursive helper compiled before, because a
  CPU-routed module is not translated, and so did a GPU-routed module whose recursive helper was never dispatched
  (slangc ran only at dispatch); with 0.2.0 the same source is `GUST-E0120`, because the check runs wherever the
  `#[gpu]` module is validated. Migration: rewrite the recursion as a `for` loop with a fixed trip count. No example
  of this repository had a recursive function; `crates/gust/tests/diagnostics.rs` had three recursive helpers (fixtures
  for the Slang line table), which were rewritten to put the recursive call into the Slang text. A call `x.m(..)` also
  counts as a call of every method named `m` of another struct, so rename a method that calls another struct's
  method of the same name.
- `DeviceOptions::default()` and `HeadlessDevice::new()` now request `gust::capability::enabled()`. In a build with no
  capability that is the empty set, as before; in a `+subgroups` build they open a device with the feature or fail with
  `Error::MissingCapability`.
- Documented, not a break of 0.1.0 code: a route flip changes which device a kernel's call sites compile against. A
  kernel call names its device, so a kernel routed to the CPU has no `GpuKernel` impl (its `HeadlessDevice` calls are
  E0277) and has `CpuDevice` calls instead, and flipping the route back reverses both. Only the calls of a generated
  element-wise map (`name::map`) are the same on either target. No 0.1.0 kernel is routed. A device handle that
  follows the route is not built in 0.2.0.
- The pinned Slang is 2026.21 (`cargo gust slang install` fetches it); the 0.1.0 line stays on 2026.20. Artifacts
  built with 2026.20 fail `cargo gust check` ("the manifest was built with Slang `2026.20` but the installed helper
  reports `2026.21`"); run `cargo gust build` again (or pass `--allow-slang-mismatch`). The emitted WGSL of one
  kernel kind changes visibly: a compare-exchange now retries after a spurious failure (Slang #13918, read from the
  emitted text). That stays inside the weak contract of `atomic_compare_exchange`, but it is a change in GPU behaviour.

### A new dependency

`gust-compute-macros` depends on `toml` 1 (no default features; `std` and `parse`) to read `[package.metadata.gust]`.
It brings four more crates, so a build of `gust-compute` compiles five that 0.1.0 did not: `toml`, `toml_parser`,
`toml_datetime` and `serde_spanned` (MIT or Apache-2.0) and `winnow` (MIT). Each declares a minimum Rust of 1.85 or
lower, under the workspace's 1.90. The reason is that `cargo publish` rewrites a manifest, so a reader of the source
form that was written by hand would refuse the form a published crate carries. This is not a code break.

### Compatibility with 0.1.0

The `semver` job of CI compares `gust-compute` and `gust-compute-wgpu` with the branch `release/0.1` (0.1.0) with
`cargo-semver-checks` 0.51.0. On the merge of m8.x1 into `main` (`ci.yml` run 42518513308, job `semver`; the job treats
its findings as expected and stays green; run 42518612375 on `main` at `85d62fd` was green too, and `c245266..85d62fd`
changed documentation only, so the counts hold), the job found:

- `gust-compute`: 229 checks, 228 passed, 31 skipped, 1 failed. The failure is `struct_marked_non_exhaustive` for
  `KernelDescriptor` (`crates/gust/src/descriptor.rs`), which is B1, so the tool says a new major version is required
  (for 0.x, a minor bump).
- `gust-compute-wgpu`: 229 checks, all passed, 31 skipped, no version update required.
- What the job cannot see: the `GpuKernel` bound (hidden from the docs), the changed defaults above, the `toml`
  dependency, and `cargo-gust` (a binary).

The tree holds 113 package and workspace manifests of examples, fixtures and tests (`examples/`, `tests/` and
`crates/gust-algo`) that already exist on `release/0.1`. This is what `git diff release/0.1 main` says of each, and
which gate of `main` builds it (`check-examples` builds the examples but `web-vector-add`, which `check-web` builds;
`check-cpu` their twins; `check-ui` the `tests/ui` fixtures; `check-dual`, `check-cpu-target`, `check-precompiled` and
`check-difftest` the fixture workspaces of the same names; `check-feature algo` and the Gpu suite `gust-algo`):

- **94 unchanged**: no file in the crate's directory differs.
- **4 changed only in derived files**, source untouched: `examples/map-chain` (its committed artifacts, written again
  for Slang 2026.21), `tests/difftest/template` (its `Cargo.lock`) and the `tests/ui` crates `profile-native-artifact`
  and `profile-native-no-manifest` (one artifact file each).
- **15 changed in a source file or a manifest.**
  - Four are the B1 migration (both parts), the only edits that existing code needed: the `tests/ui` crates
    `typed-call-buffer-for-sampler`, `typed-call-buffer-for-texture`, `typed-call-sampler-for-buffer` and
    `typed-call-texture-ok` wrote a `KernelDescriptor` literal and now call `KernelDescriptor::new(..)` and add `impl
    gust::target::GpuKernel for Blur {}` (three of the `expected.txt` files only change a line number).
  - One is a version literal: `tests/precompiled/kernels` names its `gust` path dependency `0.2.0-dev` (RELEASE.md lists
    it for the release); its manifest and one reflection file are also written again by the new Slang.
  - Ten only gained things (and `tests/difftest` a re-stamped seed file): `crates/gust-algo` (the `evidence` module and
    an example, additive; not published), `examples/web-vector-add` (an added `check-web` assertion that a subgroup
    request is refused, and re-written artifacts), `tests/cpu-target` and its `kernels` (a `routed` crate and a
    `variant` module), `tests/dual` with its `consumer` and `kernel-lib` (a routed module and binary), `tests/difftest`
    (the format 7 harness and the profile flags; `seeds.txt` takes the new format stamp and all eight program hashes),
    `tests/precompiled` (an expectation file) and the `tests/ui` workspace (new members and fixture crates).

By this comparison the only source edits that 0.1.0 code needed are the four B1 migrations. That is a statement about
the diff, not about a run: whether each gate is green on a commit is recorded in STATUS, and the gates that 0.2.0 adds
(`check-routing`, the `dev+subgroups` batches) only add fixtures.

## 0.1.0 (not yet published)

The first public release of GUST, an experimental compiler from a restricted subset of Rust to Slang for compute
kernels. Nothing in this entry changes an earlier release: there is none. The release is deferred; the packages are not
on crates.io and there is no `v0.1.0` tag yet. The branch `release/0.1` holds the state that will be published.

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
  Running the exported SPIR-V is planned for a 0.1.x release.
- A crate that depends directly on both `gust-compute` and the unrelated crate `gust` fails with rustc error
  E0464, because both libraries are called `gust`. Renaming a dependency needs an `extern crate ... as gust;` line.
- The packaged crates are not test-runnable: their tests read files of the repository. Run the tests from a checkout.
- Kernels that use workgroup memory or barriers have no CPU twin yet: they must say `cpu = "unsupported"`, and a
  `dispatch` on them does not compile under the `cpu` profile.
- The algorithms crate of the repository (`gust-algo`: reduce, scan, compact) and the examples are not published.
