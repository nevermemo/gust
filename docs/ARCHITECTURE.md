# Current architecture: GUST

This file describes implemented mechanisms, not the future semantic frontend or
execution-graph compiler. See [vision](VISION.md), [execution graph](EXECUTION_GRAPH.md),
and [engine destination](ENGINE_HURRICANE.md) for those separately labeled plans.

## Decision

GUST no longer treats a custom IR or a handwritten SPIR-V emitter as its
compiler core. The macro's job is intentionally smaller:

1. Parse and validate a GPU-safe subset of Rust syntax.
2. Keep a private Rust shadow function so rustc provides name resolution and type
   checking (it also runs as the CPU twin in test and dual builds).
3. Translate the original `syn` syntax tree directly to readable Slang.
4. Attach that source and resource metadata to a host descriptor.
5. Let Slang produce backend artifacts: at author time `cargo gust build` compiles the Slang to WGSL (and SPIR-V),
   verifies the layouts and commits the result in `gust-artifacts/`, and the macro embeds it into the descriptor
   (D45, M4). Only a `dev` build with the `slang-runtime` feature compiles at run time.

This matches the desired ownership boundary. Rust provides familiar syntax,
tooling, modules, and diagnostics for the surface language. Slang provides the
shader type system, legalization, optimization, reflection direction, and target
code generation.

## Component responsibilities

Packages and library names (D48, D106). The directories keep their names. The first release publishes four
packages under the D48 names; until the rename slice (mr.n) lands after MX's last code merge, the workspace packages
still carry today's names. The library names do not change, which is why the macro's `::gust::` and `::gust_wgpu::`
paths keep resolving in a user's crate.

| Directory | Package today | Package when published | Library name | Published |
| --- | --- | --- | --- | --- |
| `crates/gust` | `gust` | `gust-compute` | `gust` | yes |
| `crates/gust-macros` | `gust-macros` | `gust-compute-macros` | `gust_macros` | yes |
| `crates/gust-wgpu` | `gust-wgpu` | `gust-compute-wgpu` | `gust_wgpu` | yes |
| `crates/cargo-gust` | `cargo-gust` | `cargo-gust` | none (binary only) | yes |
| `crates/gust-algo` | `gust-algo` | not published (D110) | `gust_algo` | no |

The four published packages share one version, bumped together, and depend on each other with exact (`=`) versions
(D107). `cargo-gust` has no library and so no docs.rs page; its documentation is its README. The examples, `xtask`,
the fixture workspaces and `gust-algo` are not published. A packaged crate is not test-runnable: its tests read files
of the repository, so tests run from a checkout. A release follows [RELEASE](development/RELEASE.md) and the gate
`cargo xtask check-release` (D113).

### `gust-macros`

The proc macro owns validation, syntax translation, binding assignment, and host
handle generation. Translation happens directly from `syn` nodes; there is no
intermediate instruction graph between the Rust AST and Slang text.

Host handle generation includes the typed `call`, `submit` and `dispatch` functions
of each kernel (below, under `gust`; since m6.u2 a kernel with a `Uniform<T>` or
`ConstantBuffer<T>` parameter has them too, and every bound names its parameter's `Kind`) and const assertions, evaluated by rustc, that
every buffer element type and struct field is a padding-free layout (StorageV1, or StorageV2 where a vector or matrix
is involved; Std140V1 for structs a uniform reaches, which get `#[repr(C, align(16))]`) matching Rust's size and
alignment; a failure is error E0080 that names the field
or the parameter. The macro still does not reflect: layouts and bindings are
assigned here and verified against Slang at run time (D17).

The current translator emits one self-contained Slang compilation unit per
kernel. It includes the module's struct declarations, non-kernel helper functions,
that kernel's resources, and one prefixed compute entry point. Helpers and user
identifiers are not fully hygienically renamed yet.

What differs between kinds of shader stage lives behind the `StageProfile` trait of
`crates/gust-macros/src/profile/` (D63). Its 14 methods answer, from facts and never from the
`syn` body, what a stage names its parameters (the invocation and the ids derived from it), which
resources get a binding slot and in what order, which `#[kernel(..)]` options it accepts, which
constructs it refuses, the text of the entry signature, prologue and epilogue, any extra binding,
how the CPU twin runs it, how it is reflected and how exact its CPU contract is. The emitter, the
validator, the expander and the option parser call it at those points and walk the body
themselves. `grid` (the compute grid of `#[kernel]`) is the only profile until P1, and its answers
are exactly the text the call sites wrote before the trait existed.

The original kernel function is renamed to a private `__gpu_typecheck_*` item in
the expanded Rust module. In a production build it is compiled only to make rustc
check the user's body. There is deliberately no public `.cpu(...)` wrapper.

In a test or dual build the same function is the CPU twin's body (roadmap M3). For
each kernel that has a typed call the macro emits, under `cfg(any(test, gust_dual))`,
an `impl Run<K, N>` for `gust::cpu::CpuDevice`: the CPU form of a kernel is that `Run`
impl, present only in test and dual builds. `cfg(test)` is evaluated in the crate that
defines the kernel; `--cfg gust_dual` makes a dependent crate's tests and a benchmark
see the twins of a library's kernels, and it is never a Cargo feature (D26, D21). The
emitted impl validates the call, acquires the bindings through the hidden support
functions of `gust::cpu`, and calls the shadow function as written once per
invocation, in ascending `global_id.x`. It copies no body, rewrites no operator and
emits no downcast or IR. A kernel opts out with `#[kernel(cpu = "unsupported")]`
(parsed with the other `#[kernel]` options; a wrong value is GUST-E0709 and a repeated
option GUST-E0710); `CpuDevice` then lacks the `Run` impl and the generated `dispatch`
on it does not compile, which is also what a production build shows. A kernel with a
uniform parameter has a typed call and a twin since m6.u2: the twin reads the uniform by copy
(`::gust::Uniform(read_uniform::<T>(..))`), so it cannot write through it.

Under `GUST_PROFILE=cpu` (the production CPU target, D21 and D51) the macro emits that
same `Run` impl in every build, without the `cfg(any(test, gust_dual))` gate, and embeds
no artifact. The profile is chosen at build time and is never inferred, so a binary holds
one target. `gust-wgpu` evaluates `const _: () = ::gust::profile::require_gpu_target();`
and so fails to compile (E0080) under `cpu`, and the build script of `gust` refuses the
`slang-runtime` feature there. A `cpu` build therefore links no Slang bridge, no helper and
no GPU runtime. Each kernel dispatch runs on one thread, in ascending `global_id.x`; the element-wise map is the one
exception (`gust::cpu::parallel`, below). `cargo xtask check-cpu-target` is the evidence (the `Cpu` tier of
docs/development/VALIDATION.md).

The map generator (D60, slice m7.a1, amended by D99; `crates/gust-macros/src/expand/map.rs`).
`#[gpu]` on a free scalar function (an `fn`, not a module) builds a synthetic module and hands it to the same
`expand_validated` path as a written module, so the body gets the validation and the lowering of any kernel. The module
holds one kernel, `map_kernel`, which is grid-stride at `workgroup_size(256, 1, 1)` and binds every buffer
`Independent`. The host launches `min(n, CEILING)` invocations, where `CEILING` is 65535 * 256 = 16,776,960
(`gust::map::CEILING`), and runs ceil(n / CEILING) rounds. The signature rules (GUST-E0714, E0715 and, for a 3-wide
type, E0204) ask for at least one parameter, each a plain identifier of a type that can be a buffer element, and a
result of such a type; a name that the generated module uses itself (`GENERATED_NAMES`) is rejected.

`par_for!` (slice m7.p, plan 3.1, D103) is a function-like macro of `gust-macros`, re-exported by `gust`, built on that
generator. It reuses the element-type lookup (`ElementTypes`, `element_types`) for its buffer elements and refuses a
`bool` buffer there; a `bool` is a scalar binding, a `u32` word in the extent record. A call
`par_for!(i in 0..n, [a: &[f32], out: &mut [f32]] { out[i] = a[i]; })` validates its body
with the same rules as a `#[gpu]` kernel (an error in the body carries the code and the
span of the offending statement, D33), lowers it once into a kernel `run` with an extent
uniform and a bounds guard, and expands to a block that holds a module named
`__gust_pf_<hash16>` and a launch of it. The hash is `content_key` of the entry point and
the clean Slang text, so two spellings of one body (other layout, other file, other
line) are one module and one kernel. The extent record is the three axis extents, the
scalar bindings and explicit padding to a multiple of 16
bytes; every buffer is `Independent`. The launch rounds each axis up to the workgroup,
returns `Ok(())` for a zero axis without touching the device, and a launch over the
invocation cap (`Extent::MAX_INVOCATIONS`) is the device's `DeviceLimit` error, checked before the
uniform is created or anything is uploaded. The twin (`CpuDevice`; every build under `GUST_PROFILE=cpu`, test and
dual builds otherwise) runs the loop sequentially.

Export records. A `#[gpu]` module or function writes a schema 1 `<key>.kernel.json`
whose `source` names its file and line table. A `par_for!` body can stand at any number
of call sites, so its record is schema 2 and has no `source`: equal bodies write equal
record bytes (`write_once` makes the second write a no-op). Each call site writes a
sidecar `<key>.<site16>.site.json` (`schema` 2, `key`, `file`, `lines`), where `site16`
is FNV-1a 64 over the relative file, a zero byte and the decimal line of the call.
`cargo gust` reads schemas 1, 2 and 3; the record keeps every call site, sorted by file and first line, and the
`SourceOrigin` of a schema 2 kernel is the first one's. A `.site.json` next to a schema 1 record, a schema 2
record without one, and a sidecar of another key are errors.

### `gust`

The core crate owns:

- Slang-shaped Rust shadow types;
- immutable module/kernel/parameter descriptors;
- the host-shareable POD layout contract used by the current buffer runtime;
- the `slangc` process bridge;
- small target-artifact inspection helpers;
- backend-neutral job/device traits;
- the typed call surface and the device contract behind it (`gust::dispatch`);
- the CPU twin backend (`gust::cpu`, roadmap M3): `CpuDevice` and `CpuBuffer`, which
  implement `TypedDevice` and the `Resource` contract in safe Rust (`forbid(unsafe_code)`),
  the hidden functions the macro-emitted twin calls (`validate`, `read`, `write`,
  `invocations`), and `overflow_checks()` with its macro `overflow_checks!()`. It is a
  testing backend in test and dual builds, and the production CPU target under
  `GUST_PROFILE=cpu` (D51). Outside that profile the macro emits the per-kernel `Run`
  impls only in test and dual builds. A call completes inside `submit` on the calling
  thread.

The typed call surface is what the macro generates `call`, `submit` and `dispatch`
against. `Device` is unchanged; the extension trait `TypedDevice: Device` names the
erased, `Copy` binding of one device, and `Run<K, N>: TypedDevice` is the executable
hook for one kernel `K` with `N` storage and uniform parameters. A GPU backend can implement `Run`
once for every kernel (one blanket impl; a kernel with a uniform parameter has the impl and,
since m6.u2, the generated `call`, `submit` and `dispatch` that reach it); a backend that cannot run a kernel simply
lacks the impl, and a generated `submit` or `dispatch` for it does not compile.
`Resource` is the trait of anything a call can bind (a buffer type of a backend);
its `Element` and `Device` associated types make a wrong element type or device type
a compile error in a generated call; `Call` is the reusable value
that holds the kernel, the element count and one binding per storage parameter, and
`Independent(&buffer)` is the call-site opt-in for a buffer whose length differs
from the element count. The device contract (documented in `gust::dispatch`): the
backend checks every binding against the kernel descriptor at submit and does not
trust the generated code, and one buffer bound twice is accepted when both
bindings are read-only and rejected, before anything is written, when either side
writes. Access is not part of the buffer type, so an access mismatch is a run-time
error, and argument order among buffers of one element type is not checked at all.
Generated names use the reserved `__gust_` prefix so that no user parameter name
can collide.

`gust::map` (D60, slice m7.a1, amended by D99) is the host side of the element-wise map.
`MapHost: TypedDevice` is what a device offers it (`upload`, `zeroed`, `readback`, `readback_async` and `map_error`);
`CpuDevice` implements it in `gust`, and `HeadlessDevice` in `gust-wgpu` (`map_host.rs`). `MapArg<H, D>` is what a call
argument may be (a slice, a `&Vec`, a `&[T; N]`, `Broadcast`, or a `&R` for a `Resident` buffer), and the sealed
`MapElement` ties a host type to its device element. The element count is checked before anything is bound or
uploaded (`Error::LengthMismatch`, `TooLong`, `Empty`); `MixedDevices` is a resident argument of another device and
`Device(Box)` a device failure (the enum is `#[non_exhaustive]`). A call over zero elements returns an empty `Vec` or
`GpuVec`. An input of length one is read at index 0 (`Broadcast`). The generated functions are `map` and `map_on`
(not emitted on `wasm32`), `map_gpu` and `map_gpu_on` (not emitted under `cpu`) and `map_async` and `map_on_async`
(`map_async` is not emitted under `cpu` either, and was compiled for `wasm32` but never run in a browser); `map` and
`map_gpu` open the default device first.

`gust::cpu::parallel` (slice m7.c, D119; `crates/gust/src/cpu/parallel.rs`) runs the map on the production CPU target
across threads. `map_threads` gives the runners their thread count, `min(configured, max(1, n / ELEMENTS_PER_THREAD))`
(`threads_from` and `bounded_threads`; `ELEMENTS_PER_THREAD` is 2^18), where `configured` is `GUST_CPU_THREADS`
(anything but a positive integer is `Error::InvalidThreads`) or else the machine's parallelism. `run_map_chunked` and
`run_map_into_vec` run the chunks with `std::thread::scope`, so `gust::cpu` stays `forbid(unsafe_code)`. Each chunk
runs under `catch_unwind`, and after the join the panic of the lowest element index is raised again. The macro emits
a CPU-only chunk function, `__gust_map_cpu_chunk`, for it. Under `cpu`, `map_on` is concrete over `&CpuDevice`.
`par_for!` and a kernel written by hand stay sequential.

`KernelDescriptor::slang_source` is the source of truth of executable shader artifacts; the precompiled ones
(`KernelDescriptor::artifacts`, embedded by the macro from the crate's `gust-artifacts/` when the pair for the
kernel's content key exists, m4.c, D45) are what a strict profile executes, and what `dev` prefers when present.
`KernelDescriptor::entry_point` is separate from the Rust-facing kernel name so
the generated symbol can avoid Slang intrinsic collisions and target-language
reserved identifiers.

### `gust-wgpu`

The wgpu crate validates host bindings against descriptors, manages typed GPU
buffers, and owns the per-device pipeline cache. On a cache miss it takes the kernel's
embedded WGSL artifact, verified against the descriptor (D17), or, for a kernel without
one in a build with the `slang-runtime` feature, asks Slang for WGSL, and creates the
shader module and pipeline (wgpu and the driver still compile that WGSL). On a cache hit
no compiler process or pipeline creation occurs. `HeadlessDevice::new` retries a missing Vulkan adapter
up to three times with a short backoff, then returns `NoAdapter` with the attempt
count; it never skips a test and GUST never falls back to a CPU path. Adapter
selection prefers high performance but does not exclude a software Vulkan driver;
the adapter name is part of every Gpu evidence record. A request that
succeeds after a failed attempt prints one line to stderr, so a transient loader failure
stays visible.

Blocking readbacks are safe to call from many threads on one device (m7.rb). wgpu runs a `map_async` callback inside
a device poll, and a poll runs the callbacks of every finished submission, whoever requested them, so another
thread's poll can collect this thread's callback and a `poll(Wait)` can return before it has run. A blocking
readback therefore waits for the mapping's own completion after the device wait (the callback wakes the waiting
thread, or sends on a channel), for at most 30 seconds, and a callback that does not arrive in that time is
`Error::Map`. A single check after one poll reported that error under load; `tests/concurrency.rs` is the regression
test (8 threads, 4 seconds), and the m7.rb run logs `m7rb-repro0.log` (8 `Error::Map` in 2903 iterations before the
change) and `m7rb-fix1.log` (0 in 3153 after) hold the counts. The `wasm32` readbacks are awaited and unaffected.

`HeadlessDevice` implements `TypedDevice` and, in one blanket impl, `Run<K, N>` for
every kernel type (a kernel with a `Uniform<T>` parameter runs through it too since m6.g2, with the uniform bound
in parameter order), and `GpuBuffer<T>` is a `Resource`. Every launch (typed call, slice dispatch,
persistent dispatch and graph dispatch node) is checked against the stored limits
before the pipeline cache is consulted, so an over-limit launch compiles nothing
and encodes nothing (an indirect dispatch node is checked for everything except
its workgroup count): workgroup width (`max_compute_workgroup_size_x`), invocations
per workgroup, workgroup count per dimension, storage buffers per shader stage,
binding index (`max_bindings_per_bind_group`) and storage binding size are
`Error::DeviceLimit`. The limits are those the device was opened with; `with_limits`
can only lower them, to emulate a weaker target. "Default" has two meanings since m7.a0:
`HeadlessDevice::new()` requests wgpu's portable defaults, while `DeviceOptions::default()`
(what `default_device()` opens) asks a native adapter for its own buffer, binding and dispatch
limits (`LimitsPolicy::Adapter`), so a kernel validated on one can exceed the other's limits.
`max_buffer_size` is checked only by the fallible buffer functions (`try_create_typed_buffer`,
`create_zeroed_buffer`, `GpuVec`); `create_typed_buffer` is infallible and never checks it. Indirect
workgroup counts are written by the GPU and cannot
be checked on the host: with wgpu's default instance flags
(`InstanceFlags::VALIDATION_INDIRECT_CALL`, which `HeadlessDevice` keeps) wgpu zeroes
a count above the real device's limit on Vulkan (wgpu-core also needs the adapter's
`INDIRECT_EXECUTION` and `COMPUTE_SHADERS` downlevel flags, which Vulkan adapters set),
and without that flag the count reaches the GPU
unvalidated. A lowered `with_limits` value does not apply to it. Typed buffer, pool buffer and
indirect-arguments buffer creation also assert, at run time, that a hand-written `GpuPod` layout is StorageV2
(StorageV1 is its scalar sub-case), has no padding and agrees with Rust's size and alignment; the panic message
keeps the older "StorageV1 representation" wording because tests pin it.

WGSL is the active wgpu target because wgpu's WGSL frontend accepts the nested
struct operations generated by Slang. Slang's direct SPIR-V may contain legal
instructions that the current wgpu SPIR-V frontend does not implement. SPIR-V is
therefore exported and validated independently rather than being forced through
that narrower frontend.

`GpuVec<T>` (slice m7.a0, D84; `crates/gust-wgpu/src/vec.rs`) is a typed vector that stays on a device. It is
built from a slice (`from_slice`, `from_slice_on`) or zeroed (`zeroed`, `zeroed_on`), reads back with `to_vec`, is a
`Resource` (so it binds to a generated call and to a map as a `Resident` argument), and carries the id of its device,
so `check_device` and a map refuse another device's vector with `Error::MixedDevices`. Its storage comes from
the device's `BufferPool` (`pool.rs`) and goes back to it on drop. The pool keeps idle buffers in size classes, exact
up to 64 KiB (`EXACT_CLASS_BYTES`) and a power of two above that, capped at the device's `max_buffer_size`. It drops
the least recently returned idle buffers past an idle cap of 1 GiB by default. A pooled buffer may be larger than the
vector, and every binding exposes the logical bytes only. A recycled buffer is cleared in a submission of its own
before it is handed out as zeroed, and a buffer with a mapping pending is never put back. `PoolStats` reports the
hand-outs, the returns and the idle bytes. `DeviceHandle` is the shared handle that `default_device()` returns, opened
with `DeviceOptions { backends, adapter, limits }`; `set_default_options` applies to the first `default_device()` and
is `Error::DefaultDeviceOpened` after that.

### `gust-algo`

The device-resident algorithms (roadmap M7). Its first member is `reduce` (`sum`, `min`, `max` of a `GpuVec<u32|i32|f32>`, and
`sum_wrapping` of a `GpuVec<u32>`, which is defined modulo 2^32 for every input), its second `scan` (`inclusive`
and `exclusive` prefix sums of a `GpuVec<u32>` in plain `+`, up to the device's binding limit), its third `compact`
(`count` of the non-zero flags of a `GpuVec<u32>` as a one-element vector that stays on the device, `indices` of the
set flags through the exclusive scan and a scatter, and `gather_u32`, `gather_i32` and `gather_f32` by those indices; no
readback anywhere, so the indices and the gathers are as long as the flags, zero from the count on; a generic gather
of structs waits for M9 and is a user kernel until then). The kernels are ordinary GUST kernels in
`tests/fixtures/reduce.rs`, `tests/fixtures/scan.rs` and `tests/fixtures/compact.rs` (one `#[gust::gpu]` module each,
included as `kernels`, `scan_kernels` and `compact_kernels` and compiled at run time through the `slang-runtime`
feature; the crate ships no precompiled artifact), and the host code only picks the schedule, takes one pooled
`GpuVec` per level and submits the dispatches in one `dispatch_batch`.

Two variants exist for the reduce and the scan (slice m7.r4, `variant.rs`). v0 is block-strided and has CPU twins
(`cpu.rs`, in test and dual builds): a reduction takes up to 4 dispatches and a scan 3, 5 or 7 dispatches by length. v1
keeps a tree in `Workgroup<[T; 256]>` and uses `workgroup_barrier()`, so its kernels are `cpu = "unsupported"` and have
no twin (D101): a workgroup owns 8192 elements of a reduce or a span of 4096 elements of a scan. `variant::shipped`
names `V1` for all six operations (D121), so the public functions run v1, and the `*_with` functions of `reduce` and
`scan` run either on request. `limits::admits` sends a call to v0 when the device has less workgroup storage than the
v1 kernels declare (3072 bytes), when the first dispatch is past the device's workgroup limit or when a binding does
not fit. The results agree: integers bit for bit, floats within the per-level bound of `bound.rs`, which has a form for
each variant (`f32_sum_bound` for v0 and `f32_sum_bound_v1` for v1). The measurement behind `shipped` is in D121:
the RTX 5090 half used a 10 % margin, and the Radeon 890M half put v1 ahead by 1.01 to 1.03 for the reductions at 1e8.

The crate depends on `gust` and `gust-wgpu` and on nothing else: `xtask/src/arch.rs` carries a `CrateRule` with exactly
that allow-list, and `cargo xtask check-arch` fails for a workspace crate without one. It is `publish = false` because
the kernel files live outside the package. Its twin tests run the v0 twins against independent references in its test
builds; the `gpu::` tests run on the adapter. Both are in the one `cargo test -p gust-algo` call that the Gpu suite and
`verify --mode gpu` make (`ALGO_GPU_CALL`, with 49 tests required by name); the Unit suite does not run the crate.

Shapes (slice mx.k, D89). `gust::Shape` (`Grid`, `Pixel`; `#[non_exhaustive]`) is a field of `KernelDescriptor`, set
by the macro from the stage profile, and it picks the `slangc -stage` (`compute` or `fragment`) in `gust::slang`,
`gust::reflect` (also the reflection helper's explicit last argument) and `cargo gust` (`build`, `check --recompile`).
It is not part of the content key: the stage is already in the clean Slang text, so one body under two shapes gives
two keys. Schema 3 is schema 2's grammar plus an optional `shape` (`grid` or `pixel`; absent means `grid`) and an
optional `source` (a record with one is read like schema 1, one without like schema 2). The macro writes it only for a
kernel whose shape is not `grid`: a `#[gpu]` grid kernel keeps schema 1 and a `par_for!` grid kernel keeps schema 2,
byte for byte. A reader of schemas 1 and 2 refuses a schema 3 record, and a `shape` in a record that says schema 1 or
2 is an error. The manifest (`gust-artifacts/manifest.json`) is schema 1 while every kernel is a grid kernel (no
`shape` key anywhere; `cargo gust init` writes it) and schema 2 once one is not: then every kernel entry carries
`shape`. A schema 1 manifest reads as all grid, and `check` compares each entry's shape with its record's.

Layouts in the record (slice m7.lv). A parameter's `layout` is `null` or `{rules, element_stride, element}`. `rules` is
`storage_v1` (a padding-free element of scalars and structs of them), `storage_v2` (a storage element that holds a
vector or a matrix) or `std140_v1` (the element of a `Uniform<T>`). The element's `kind` is `scalar`, `vector`
(`scalar`, `count`), `matrix` (`scalar`, `rows`, `cols`, `column_stride`) or `struct` (`fields` of `name`, `offset`
and a nested `type`). The macro writes these numbers from the same functions that produce its const assertions
(storage structs are padding-free; a uniform struct uses the std140 offsets, an alignment of at least 16 and a size
rounded up to 16, and its `element_stride` is its size), for every element it can describe from the module's own
types and the `gust::math` vectors and matrices (a 3-wide type is never a buffer element). Anything else stays `null`
(an alias or a type from outside the module): `cargo gust` then fills the layout from the helper's reflection and the
kernel's manifest entry lists the parameter in `export_unchecked`, uniforms included. A record with only `storage_v1`
layouts is byte for byte what it was, so no schema 1 record of a kernel without a uniform or a vector changed and the
schema stays 1. A `cargo-gust` from before this slice refuses a record with the new rules by path (an `unknown layout
rules` error that names `parameters[i].layout`) and needs the update.

## Descriptor flow

For a kernel with three storage buffers, the macro records:

```text
Rust parameter              Descriptor                    Slang declaration
SV_DispatchThreadID id      Invocation, no binding        uint3 : SV_DispatchThreadID
StructuredBuffer<float> a   Storage RO, group 0 binding 0 StructuredBuffer<float>
StructuredBuffer<float> b   Storage RO, group 0 binding 1 StructuredBuffer<float>
RWStructuredBuffer<float> o Storage RW, group 0 binding 2 RWStructuredBuffer<float>
LocalId lid                 Derived from id, no binding   uint3 lid = id % workgroup size
GroupId gid                 Derived from id, no binding   uint3 gid = id / workgroup size
Workgroup<[f32; 256]> s     Not in the descriptor         groupshared float s[256]; (module scope)
```

The current macro assigns bindings deterministically in resource-parameter order.
The macro still *assigns* this contract; since T06 the runtime *verifies* it against
the compiler before creating a pipeline (see "Struct ABI"). Runtime descriptors and
bind groups are not derived from reflection. Compiler-authoritative layout validation
now exists for the StorageV1 and StorageV2 buffers and the Std140V1 uniforms; textures, samplers, parameter blocks,
multiple groups, and specialization still need their own evidence before entering the
runtime contract. A `Uniform<T>` or `ConstantBuffer<T>` parameter takes the next binding in the same order
(`ConstantBuffer<T>` in Slang, `BufferBindingType::Uniform` in the bind group layout), and the compiler resource
it is checked against is a `constantBuffer` whose element has the `Std140V1` layout.

`KernelDescriptor::source` (a `SourceOrigin`, new in M1.1) is diagnostic metadata
only (the Rust file and a Slang-to-Rust line table) and never affects the emitted
Slang, the pipeline cache identity or the runtime ABI. Its file is never an absolute
path: both paths are first normalised lexically (repeated separators collapsed, `.`
dropped, `..` resolved), then a relative path is kept as its normalised components, an
absolute path under the expanding crate's `CARGO_MANIFEST_DIR` is stored relative to
it, and any other absolute path, or a relative path that climbs out of its root with
`..`, is reduced to its file name, always with `/` separators. The result is checked
against a whitelist: every component must be non-empty, not `.` or `..`, and made of
ASCII letters and digits, `_ . + -` and Unicode Alphabetic characters. A result that fails is
replaced by the file name, and if that fails too the origin is unknown (empty file,
no line table). Of the `\\?\` forms only `\\?\X:\` and `\\?\UNC\` are read as drive
and UNC paths; any other (GLOBALROOT, `Volume{GUID}`, device) is an absolute path
that is reduced to its file name. Adding the field is a
public API change: hand-written `KernelDescriptor` literals add
`source: SourceOrigin::UNKNOWN`, and the derived equality now includes the source
metadata. The same holds for `artifacts` (M4, D45): a hand-written literal adds it (`&[]` for none) and the
derived equality compares it. The macro fills it with the embedded artifact pair of each kernel whose
`gust-artifacts/` files exist (m4.c), and with `&[]` for a kernel that has none.

## Struct ABI

The runtime still needs a concrete agreement for uploading Rust values. Annotated
named-field structs therefore receive `repr(C)`, `Clone`, `Copy`, and a generated
`GpuPod` layout description. The accepted ABIs are deliberately narrow, three sets of rules in `gust::abi`:

- `StorageV1`: 32-bit scalar fields and recursively padding-free structs with four-byte alignment.
- `StorageV2` (M6): `StorageV1` plus the 2- and 4-wide vectors (`Vec2`/`Vec4` and the integer forms) and the 2x2 and
  4x4 matrices of `gust::math`, each with its natural alignment (8 or 16); a struct may carry the padding that
  alignment implies, but a `GpuPod` type is only the padding-free subset, so no uninitialised byte is uploaded.
- `Std140V1` (M6) for `Uniform<T>` and `ConstantBuffer<T>`: `StorageV2` with structs rounded up to a multiple of 16 and
  no 2x2 matrix. The macro marks a struct a uniform reaches `#[repr(C, align(16))]` and const-asserts the offsets;
  padding is explicit, written by the user as fields (E0208, E0209), and the same struct cannot also be a storage
  element (E0207).

3-wide vectors and `Mat3` are locals, helper parameters and results only: no buffer element, storage field or uniform
field may be one (E0204). Matrices are column-major (`abi::MATRIX_LAYOUT`): each column is contiguous with a stride of
8 or 16 bytes, so a Rust matrix of vector columns has the same bytes. That pin has three homes that change together:
`-matrix-layout-column-major` at both `slangc` call sites (`slang::slangc_command` and
`reflect::slangc_reflection_command`; `cargo xtask slangc` passes the same flag), `defaultMatrixLayoutMode` in the
reflection helper, and `matrixLayout` in the reflection JSON. The command line of `slangc` defaults to column-major
today and its native API to row-major (the M6 spike V1), so no path may rely on a default. Tests pin the argument
lists and compare the bytes the CLI path and the helper produce for a matrix kernel.

This host transfer ABI is not intended to reimplement Slang's full layout system.
It is a narrow executable bridge whose every claim is checked by the compiler.

`reflect::compile_reflected` runs the native helper `gust-slang-reflect`
(`scripts/probes/slang-layout.cpp`, built by `cargo xtask build-slang-reflect`
against the installed Slang SDK). The helper links one entry point for one target
through the Slang API, emits the artifact, and walks the *same linked program's*
`TypeLayoutReflection` to write a JSON record (schema 1; schema 2, with a top-level `matrixLayout`, only when a
resource holds a vector, matrix or constant buffer): compiler build tag, target,
entry point, source and artifact FNV-1a fingerprints, compute workgroup size, and for
every global structured buffer its name, descriptor slot and space, access, element
stride, and the element type with size, alignment, stride, scalar kind, and nested
fields (name, offset, size, type), and for a vector, matrix or constant buffer the scalar type, element count, rows,
columns and layout; matrices are column-major. Schema 3 (slice mx.t0a, D91; `--version` prints `schema=3`) is schema
2's grammar plus a top-level `stage` and is written only when the program holds a 2D texture or a sampler or the
entry's stage is not compute, so every document of a compute kernel over buffers is byte-identical to before. The stage
is the optional seventh argument of the helper (`compute` by default, or `fragment`): it is the caller's expectation of
the entry point and the helper refuses an entry of another stage. A texture is `{"kind":"texture","shape":"2d",
"access":"read"|"readWrite"|"write","result":<scalar or vector, no byte layout>,"format":"<name>"}` in one descriptor
slot; `format` is the storage format (`r32ui`, `rgba32f`, ...) of a read-write or write-only texture, declared with
`[format(..)]` or inferred by Slang (`RWTexture2D<uint>` is `r32ui`), and the fixed word `unknown` for a sampled texture
(access `read`), whatever the API reports for it (it says `r32ui` for a `Texture2D<uint>` and `unknown` for a
`Texture2D<float4>`, and the command line omits both format and access). A sampler is `{"kind":"sampler"}`. A fragment
entry has no `workgroupSize`; it has `targets`, `[{"index":n,"type":<scalar or vector>}]` in `SV_Target` order, and may
take `SV_Position`. Refused, each with a message that names it: a texture that is not a plain 2D one (1D, 3D, cube,
array, multisampled, combined `Sampler2D`), a comparison sampler (`SamplerComparisonState`), other fragment outputs, and
a storage texture with no declared or inferred format (`RWTexture2D<float4>`). The helper reports what Slang computes
and refuses what it cannot describe completely, with a message instead of a partial record: non-32-bit scalars, arrays
and other kinds of element or field, an empty struct, a field that does not sit at a plain byte offset, a counter,
entry-point resources and uniform bindings, binding arrays, and a global parameter that is neither an individual structured buffer nor an
individual constant buffer in one descriptor slot. What a storage buffer may hold is a separate rule, enforced by
`is_storage_v1`/`is_storage_v2`, the macro's layouts and `gust-wgpu`'s binding checks, so the helper reporting a padded
`float3` field does not make it acceptable. The helper is rebuilt only when it is stale, judged by content: a build
stamp beside it holds the SHA-256 of the C++ source, and its `--version` must give the Slang tag of the SDK (`cargo gust
slang build-helper`). `CompiledReflection::from_output` re-derives both fingerprints, rejects
identity or schema mismatches, decodes SPIR-V structurally, and requires every layout
field to be present.

`Reflection::cross_check_kernel` then compares the descriptor with that evidence:
no duplicate names or slots on either side, one compiler resource per storage or
uniform parameter and none left over, equal group/binding/access, and
`cross_check_pod`-style recursive equality of field identity, offset, size,
alignment, and stride down to scalars. The private `HeadlessDevice::compile_kernel`
(reached through `cached_kernel`) calls both on every pipeline-cache miss and refuses
to build the pipeline on any error; the WGSL it feeds to wgpu is the helper's
artifact, so the executed shader and the reflected layout come from one compilation.
The cache key is descriptor identity (`&'static` source and parameter slices plus
entry point), so a descriptor whose contract differs from a cached one produces a
different key, misses, and is checked before it can run (see
`crates/gust-wgpu/tests/reflection.rs`). The helper's SPIR-V is byte-identical
to `slang::compile_spirv`; its WGSL differs from `slang::compile_wgsl` only in line
endings.

`reflect::reflect` and `Reflection::from_json` still read `slangc -reflection-json`.
That CLI record omits aggregate size, alignment, and buffer stride in the tested
Slang 2026.20 (and in 2026.13.1 before the bump), so `cross_check_pod` on it returns an explicitly partial report
(`aggregate_size_verified`, `alignment_verified`, `element_stride_verified` false).
It is kept as inspectable evidence and is not consulted by the runtime. Missing
values are never inferred from the host layout. Nothing widens `GpuPod`.

## Compiler process boundary

The core bridge invokes the `slangc` executable rather than linking a specific
Slang library ABI. Each compilation:

- exclusively creates an owned temporary directory (never reuses an existing path)
  and writes a self-contained source unit;
- selects one entry point and target;
- preserves source entry-point names for SPIR-V;
- captures compiler stdout/stderr diagnostics on failure;
- reads the target artifact;
- checks SPIR-V structure at the compiler boundary;
- attempts cleanup of its owned directory on every return path, including errors.

Cleanup is best effort if the OS prevents deletion. Cached pipeline identity
includes the selected entrypoint, so descriptors sharing a compilation unit cannot
silently execute each other's kernels. Concurrent first misses can still duplicate
compilation work; the cache does not yet implement single-flight compilation.

This keeps the prototype small and lets the installed Slang version be upgraded
independently. Since M4 this bridge is a `dev` convenience behind the `slang-runtime` feature (D44); production
builds hold none of it and execute committed artifacts. Compiler-version discovery and the content-addressed store
exist in `cargo gust` (the pinned-Slang manager and the content key of D45), outside this crate: `gust::slang`
itself still runs whatever `slangc` is first on `PATH` and checks no version. Deterministic option records and
dependency tracking beyond the content key remain future work.

## CPU comparison policy

Independent CPU reference code lives outside `#[gpu]` modules. GPU authoring types are
Rust shadows for Slang concepts, and the shadow body is not the oracle for them (D04,
D26).

The polynomial, signal, particle, and typed-pipeline examples maintain independent
scalar CPU implementations, and the semantics fixtures compute independent host
references. Tests compare actual GPU execution against those, and those host
references stay the expected values. The typed pipeline additionally verifies
dependencies across three ordered kernels whose intermediate buffers all use different
struct types. This is useful for correctness and benchmarking.

Since M3 the shadow body also executes, as a CPU twin in test and dual builds, and the
twin is a third party next to the GPU and the host reference. A twin assertion is test
evidence of the shadow semantics. It is never GPU evidence (the Gpu tier needs readback
on a device) and never an oracle, and it is added to the host-reference and GPU
assertions, never in place of them (see "Evidence tiers" in
[VALIDATION](development/VALIDATION.md)). Where the twin and the GPU may legitimately
differ, the known-mismatch ledger in [SEMANTICS](SEMANTICS.md) lists the construct as a
documented limitation: the plain-Rust and twin outcomes are pinned, and the GPU outcome is measured
(enforced only where the WGSL specification fixes it, recorded as unspecified where it does not); such an edge-case
difference is recorded, not matched (D39).

The test devices in the `gust::dispatch` unit tests exist only to exercise the typed
call surface: the recording mock mirrors a GPU backend and never runs a kernel body,
and the CPU side is the real `gust::cpu::CpuDevice` (since m3.b), on which those tests
run hand-written test bodies through the same `Call`. The recording mock is not a CPU
runtime for `#[gpu]` shadow bodies. The macro-emitted CPU twin (the `Run<K, N>` impl for
`CpuDevice`) exists under `cfg(test)` of the defining crate, under `--cfg gust_dual`, and
in every build under `GUST_PROFILE=cpu`. A GPU production build (`native`, `web`) never
has it.

## Validation layers

The repository checks different failure classes at different layers. These are
not one sequential backend path: WGSL execution and SPIR-V validation are branches.

1. Macro validation rejects syntax that the direct translator does not support.
2. rustc checks shadow types, names, indexing, fields, and arithmetic.
3. `slangc` checks the generated shader and produces WGSL/SPIR-V.
4. The core SPIR-V helper checks binary structure.
5. `spirv-val --target-env vulkan1.2` performs external semantic validation of
   exported modules in artifact/full checks (the examples' files and the vectors, products and uniforms goldens, which
   those checks compile); a missing tool fails those checks. GUST's SPIR-V targets Vulkan drivers;
   feeding it to naga is unsupported, because Slang can emit `OpCopyLogical` (a copy between aggregates of different
   layouts), which naga's SPIR-V front end refuses. `check-artifacts`, the Spirv suite, `check-full`, `spirv-val` and the
   `check-feature` areas that validate SPIR-V print an informational `OpCopyLogical` count per module (never a failure): 0 in all three golden modules and all 13 example modules (2026-11-02).
6. wgpu validates the generated WGSL, bind-group layout, pipeline, and dispatch.
7. Integration tests compare readback values with independent CPU references. In test
   and dual builds the same typed call also runs on the CPU twin, and its result is
   asserted next to the GPU's and the reference's, never instead of them.

Slang diagnostics point to generated `.slang` lines. The bridge recognizes current
multiline and legacy locations, writes a temporary `kernel.map.json` beside its
owned `kernel.slang`, and appends the mapped kernel plus a stable generated construct
label when an error line is covered. Warnings, unrelated filenames, and out-of-range
locations do not establish attribution.

When the descriptor carries a line table (M1.1), the text handed to the compiler is a
compile-time copy with `#line <rust_line> "<file>"` directives (M1.3), so the
compiler's own error location is the Rust file and line, and the bridge appends
`originating Rust source: <file>:<line>`. The column stays a column of the generated
Slang. Lines without a Rust origin return to `kernel.slang` coordinates. The copy is
used by `slang::compile*` and by both reflection paths; the exported `.slang`
(`write_source`, `KernelDescriptor::slang_source`) never contains a directive. A
descriptor without a table compiles the clean text as before.

## Deliberate non-goals for this slice

- Recreating Slang IR or optimization in Rust.
- Treating generated WGSL or SPIR-V as compile-time constants in descriptors. D45 (M4) supersedes this for
  precompiled output only: `KernelDescriptor::artifacts` can hold compiler output plus the helper's raw reflection
  JSON, re-verified by `gust::artifact::verify_artifact`; the macro embeds the committed files (m4.c) and the Slang
  text stays the source of truth.
- Providing a CPU interpreter for GPU modules. There is none: the CPU twin is the
  rustc-compiled shadow body, emitted in test and dual builds and in every build under
  `GUST_PROFILE=cpu` (the production CPU target, D51).
- Supporting arbitrary Rust control flow or library calls.
- Claiming complete Slang language or resource coverage.
- Hiding target artifacts; readable Slang, WGSL, SPIR-V, and host code remain
  inspectable outputs.

## Near-term evolution

The architecture is ready for several incremental extensions without changing the
core boundary:

- extend the existing native reflection gate as new resource/layout contracts are proven;
- add vectors, matrices, textures, samplers, and constant buffers;
- map Slang errors back to Rust spans;
- enforce the Slang pin in `gust::slang` (the content-hash artifact set of D45 already caches compiler output for
  WGSL and SPIR-V; the follow-up is in NEXT_TASKS);
- select WGSL, SPIR-V, DXIL, or Metal targets per backend;
- expose controlled Slang escape hatches for features that do not map cleanly to
  Rust syntax.

## Translation stability and known limits

Tail values in non-void helpers become returns, including tail blocks and branches.
Explicit local type annotations are preserved as Slang declarations; they provide
a way to state signed/unsigned intent without pretending to consume rustc inference.
Binary expressions are parenthesized to retain Rust AST precedence; 32-bit literal
suffixes and integer radices are translated. Overloaded internal Slang functions
distinguish boolean negation from integer complement; `__gust_` and `__gpu_` are
reserved source prefixes. The shared semantics fixture tests exact Slang, target
compilation, and real GPU output at workgroup boundaries.

Struct literals are now lowered field-aware. Slang has no field-name or designated
initializers, so `Pair { first = 1 }` is not valid; the translator emits the portable
construct-then-assign form instead:

```slang
Pair __gust_struct_0;
__gust_struct_0.first = uint(1);
__gust_struct_0.second = uint(2);
Pair p = __gust_struct_0;
```

Fields are assigned by name in source order, which preserves both field identity
(`Pair { second, first }` still assigns `first` and `second` correctly) and
evaluation order. A fresh temporary holds all fields before the destination is
evaluated and assigned once. This preserves self-assignment swaps and reads of an
outer binding in a shadowing initializer. This is supported as a `let` initializer
and as the right side of an assignment to a struct lvalue. Functional record update
syntax (`Pair { first: 1u32, ..old }`) is rejected, not partially initialized. A
struct literal in any other expression position (a function argument, a nested
expression) is still rejected with a diagnostic, as is a struct literal nested inside
another struct literal's field value. Copying and updating an existing struct remains
executable.

Numeric semantics are proven, not assumed. For the supported 32-bit scalar subset the
GPU matches Rust: signed and unsigned division truncate toward zero, signed modulo
keeps the dividend's sign, unsigned modulo is non-negative, and in-range float-to-int
casts truncate toward zero. Integer overflow wraps on the GPU (matching Rust
`wrapping_*` / release semantics, not the debug panic; the CPU twin follows the overflow
mode of the build that compiles the kernel). These cases are locked in by
the `numeric` fixture's golden Slang, both target compilations, and a real-GPU
readback test against an independent host reference. Integer division or modulo by a
literal zero is rejected at the Rust boundary with a diagnostic, because Slang makes
it a compile error while rustc treats it as a runtime panic. Division or modulo by a
runtime-zero variable and float-to-int casts outside the representable range are
not guaranteed to preserve Rust semantics in this GPU subset. Rust itself defines
integer division/remainder by zero as a panic and float-to-integer `as` casts as
saturating (NaN maps to zero); do not describe those Rust operations as undefined.
See the [Rust operator reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html).
The defined domain, the float bounds and the known-mismatch ledger are in [SEMANTICS.md](SEMANTICS.md).

Primitive type names and cast targets are fail-closed at the macro boundary; tuples,
slices, and other standard-prelude names without a proven lowering remain rejected.
`as` casts accept only the proven
32-bit scalar targets (`f32`/`float`, `i32`/`int`, `u32`/`uint`); Rust primitives
outside the four-byte subset (`u8`, `i64`, `usize`, `f64`, `char`, ...) are rejected
in every type position, and inline `const { .. }` blocks are rejected by name. These
names are not shadowed by the prelude, so rustc accepts them in the shadow body and
the translator previously passed the Rust spelling through to a `slangc` failure at
pipeline creation. `bool` remains a supported type but not a cast target: rustc
already rejects numeric-to-`bool` casts, and `bool`-to-integer casts stay unproven.

Rust `Option<T>` is lowered deterministically to Slang's built-in `Optional<T>`,
which both targets legalize to a `{ value, hasValue }` struct. When a module uses
Option, the compilation unit gains two generic helpers, `__gust_some<T>` and
`__gust_unwrap_or<T>`, so payload types are inferred at each use without annotations.
`Some(x)`, `None`, `.is_some()`, `.is_none()`, and `.unwrap_or(default)` map
directly; `if let Some(x) = value { .. } else { .. }` evaluates the scrutinee once into
a reserved `__gust_opt_N` temporary and binds the payload inside the taken branch. The
payload must be a supported scalar or a module struct. Option is a local/helper value
only: it is rejected in struct fields and resource element types because it has no
proven storage layout, and `.unwrap()`/`.expect()` are rejected because Rust panics
there while GPU code cannot. `match` on Option, `?`, let chains, and other patterns
remain rejected. `Result<T, T>` is also lowered deterministically as a local/helper-only
tagged `__GustResult<T>`: `Ok`/`Err`, `is_ok`/`is_err`, `unwrap_or`, `if let Ok`, and
statement-only exhaustive `match` with `Ok(identifier)` / `Err(identifier)` block arms
are supported. A match evaluates its scrutinee once into a generated temporary before
branching on the tag. Equal payload types are required because Slang cannot infer an
absent type parameter from a single-payload constructor. Mixed/nested Results and
Results in structs or resources remain rejected. The `option` fixture locks the Slang, both targets compile, and a
real-GPU test matches an independent host `Option` reference at workgroup boundaries.

Loops are bounded by construction (D18). The only accepted form is
`for i in start..end { .. }`; after rustc shadow checking and Slang checking, the
bounds must be 32-bit integers. It lowers to

```slang
var __gust_end_3 = limit;
for (var i = uint(0); i < __gust_end_3; i++)
{
    ...
}
```

The end bound is evaluated exactly once before the loop, matching Rust's `Range`
construction, so mutating `limit` inside the body does not change the trip count. The
loop variable is `var`-typed from the start expression; an unsuffixed integer literal
start is rejected so that Slang cannot infer `int` where rustc inferred `u32`. `_`
patterns become a reserved `__gust_iter_N` counter. The temporary index is the
statement's position in its block, as for `__gust_struct_N`/`__gust_opt_N`; an inner
loop at the same position legally shadows the outer temporary in its nested scope.
`break` and `continue` (unlabeled, valueless) and `return` inside the body map
directly. `while`, `loop`, `..=`, labels, `mut`/tuple patterns, non-range iterables,
and `break`/`continue` outside a loop are rejected with diagnostics (each has a stable
code, see [DIAGNOSTICS.md](DIAGNOSTICS.md)); the reasons are
recorded in D18. The `loops` fixture locks the Slang, both targets compile, and a
real-GPU test matches an independent host reference (closed forms where possible) at
workgroup boundaries, including a body that shrinks its own end bound.

Atomics are statement-only calls on a direct `&mut buffer[index]` element of a
`RWStructuredBuffer<u32>` or `RWStructuredBuffer<i32>` (T10). `u32` accepts
`atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange`, and
`atomic_compare_exchange`; `i32` accepts all of them except min and max. A buffer used
atomically in a kernel is emitted as `RWStructuredBuffer<Atomic<T>>` for that kernel
only, and ordinary indexed reads or writes of it are rejected in that kernel. Only
relaxed (default) memory ordering is provided. Expression-position atomics, float
elements, and read-only, local, or non-element receivers are rejected. The `atomics`
fixture locks the Slang. Real-GPU tests check contended `atomic_add`, `atomic_min`,
`atomic_max`, and the failing compare-exchange exactly; contended
`atomic_exchange` and `atomic_compare_exchange` and the uncontended compare-exchange successes are checked only
for a valid result on the GPU (set-valued, D47) and exactly on the twin.

The syn translator still lacks rustc-resolved semantics. Inferred integer types,
overflow, casts outside their safe input domain, evaluation ordering for effectful
expressions, resource aliases, and comprehensive identifier hygiene need further
proofs or restrictions. A uniform parameter is an executable binding on wgpu (Std140V1; see "Struct ABI").
Do not infer arbitrary Rust support from successful parsing or shadow typechecking.
The ranked work list is in [next tasks](development/NEXT_TASKS.md).
